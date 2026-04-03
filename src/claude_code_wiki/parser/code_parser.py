import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tree_sitter import Language, Parser

from claude_code_wiki.graph.models import Entity, EntityType, KnowledgeGraph, Relation, RelationType
from claude_code_wiki.graph.lsp import DenoLSP

try:
    from tree_sitter_language_pack import get_language
except ImportError:
    get_language = None


def _get_parser_for_ext(ext: str) -> Parser | None:
    lang_name = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }.get(ext)
    if not lang_name:
        return None
    try:
        lang = get_language(lang_name)
        return Parser(lang)
    except Exception:
        return None


def _parse_file_raw(file_path_str: str, root_dir_str: str):
    file_path = Path(file_path_str)
    root_dir = Path(root_dir_str)
    ext = file_path.suffix

    parser = _get_parser_for_ext(ext)
    if not parser:
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = parser.parse(bytes(content, "utf-8"))
    except Exception:
        return None

    rel_path = str(file_path.relative_to(root_dir))
    module_name = str(file_path.stem)
    module_id = f"module:{rel_path}"
    module_entity = Entity(
        id=module_id,
        name=module_name,
        type=EntityType.MODULE,
        file_path=str(file_path),
        line_number=0,
    )

    entities = {module_id: module_entity}
    relations = []

    class SimpleVisitor:
        def __init__(self):
            self.current_class = None
            self.entities = {}
            self.relations = []

        def visit(self, node):
            method = getattr(self, f"visit_{node.type}", self.generic_visit)
            method(node)

        def generic_visit(self, node):
            for child in node.children:
                self.visit(child)

        def visit_class_declaration(self, node):
            name = self._get_name(node)
            entity_id = f"{file_path}:{name}"
            self.entities[entity_id] = Entity(
                id=entity_id,
                name=name,
                type=EntityType.CLASS,
                file_path=str(file_path),
                line_number=node.start_point[0] + 1,
            )
            old_class = self.current_class
            self.current_class = entity_id
            self.generic_visit(node)
            self.current_class = old_class

        def visit_function_declaration(self, node):
            name = self._get_name(node)
            entity_id = f"{file_path}:{name}"
            self.entities[entity_id] = Entity(
                id=entity_id,
                name=name,
                type=EntityType.FUNCTION,
                file_path=str(file_path),
                line_number=node.start_point[0] + 1,
            )
            if self.current_class:
                self.relations.append(Relation(
                    source_id=self.current_class,
                    target_id=entity_id,
                    type=RelationType.CONTAINS,
                    weight=2.0,
                ))
            self.generic_visit(node)

        def visit_method_definition(self, node):
            name = self._get_name(node)
            entity_id = f"{file_path}:{name}"
            self.entities[entity_id] = Entity(
                id=entity_id,
                name=name,
                type=EntityType.FUNCTION,
                file_path=str(file_path),
                line_number=node.start_point[0] + 1,
            )
            if self.current_class:
                self.relations.append(Relation(
                    source_id=self.current_class,
                    target_id=entity_id,
                    type=RelationType.CONTAINS,
                    weight=2.0,
                ))
            self.generic_visit(node)

        def visit_call_expression(self, node):
            func_name = self._get_call_name(node)
            if func_name:
                target_id = f"call:{func_name}"
                source_id = self.current_class or module_id
                self.relations.append(Relation(
                    source_id=source_id,
                    target_id=target_id,
                    type=RelationType.CALLS,
                    weight=1.0,
                ))
            self.generic_visit(node)

        def visit_import_statement(self, node):
            for child in node.children:
                if child.type == "module":
                    module_name = child.text.decode("utf-8")
                    target_id = f"module:{module_name}"
                    source_id = self.current_class or module_id
                    self.relations.append(Relation(
                        source_id=source_id,
                        target_id=target_id,
                        type=RelationType.IMPORTS,
                        weight=1.0,
                    ))
            self.generic_visit(node)

        def visit_import_from_statement(self, node):
            for child in node.children:
                if child.type == "module":
                    module_name = child.text.decode("utf-8")
                    target_id = f"module:{module_name}"
                    source_id = self.current_class or module_id
                    self.relations.append(Relation(
                        source_id=source_id,
                        target_id=target_id,
                        type=RelationType.IMPORTS,
                        weight=1.0,
                    ))
            self.generic_visit(node)

        def _get_name(self, node):
            for child in node.children:
                if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
                    return child.text.decode("utf-8")
            return ""

        def _get_call_name(self, node):
            if node.type == "identifier":
                return node.text.decode("utf-8")
            for child in node.children:
                if child.type in ("identifier", "attribute"):
                    if child.type == "attribute":
                        return child.children[-1].text.decode("utf-8") if child.children else ""
                    return child.text.decode("utf-8")
            return ""

    visitor = SimpleVisitor()
    visitor.visit(tree.root_node)

    entities.update(visitor.entities)
    relations.extend(visitor.relations)

    for entity in visitor.entities.values():
        if entity.type == EntityType.CLASS:
            relations.append(Relation(
                source_id=module_id,
                target_id=entity.id,
                type=RelationType.CONTAINS,
                weight=1.0,
            ))

    return {
        "entities": [(e.id, e) for e in entities.values()],
        "relations": visitor.relations,
    }


class CodeParser:
    EXT_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
    }

    def __init__(self, root_dir: str, use_lsp: bool = True):
        self.root_dir = Path(root_dir).resolve()
        self.graph = KnowledgeGraph()
        self._parsers = {}
        self._file_modules = {}
        self._lsp: DenoLSP | None = None
        self._use_lsp = use_lsp

    def _init_lsp(self):
        if self._lsp is None and self._use_lsp:
            try:
                self._lsp = DenoLSP(str(self.root_dir))
                self._lsp.start()
                self._lsp.initialize(f"file://{self.root_dir}")
            except Exception:
                self._lsp = None

    def _stop_lsp(self):
        if self._lsp:
            self._lsp.stop()
            self._lsp = None

    def _get_parser(self, ext: str) -> Parser:
        lang_name = self.EXT_MAP.get(ext)
        if not lang_name:
            return None

        if lang_name not in self._parsers:
            try:
                lang = get_language(lang_name)
                parser = Parser(lang)
                self._parsers[lang_name] = parser
            except Exception:
                return None
        return self._parsers.get(lang_name)

    def parse_file(self, file_path: Path) -> None:
        ext = file_path.suffix
        parser = self._get_parser(ext)
        if not parser:
            return

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = parser.parse(bytes(content, "utf-8"))
        except Exception:
            return

        rel_path = str(file_path.relative_to(self.root_dir))
        module_entity = self._get_or_create_module(rel_path, file_path)

        if ext in (".ts", ".tsx") and self._use_lsp:
            self._init_lsp()

        visitor = EntityVisitor(file_path, module_entity.id, self)
        visitor.visit(tree.root_node)
        self.graph.entities.update(visitor.entities)
        self.graph.relations.extend(visitor.relations)

        for entity in visitor.entities.values():
            if entity.type == EntityType.CLASS:
                self.graph.relations.append(
                    Relation(
                        source_id=module_entity.id,
                        target_id=entity.id,
                        type=RelationType.CONTAINS,
                        weight=1.0,
                    )
                )

    def _get_or_create_module(self, rel_path: str, file_path: Path) -> Entity:
        if rel_path in self._file_modules:
            return self._file_modules[rel_path]

        module_name = str(file_path.stem)
        entity_id = f"module:{rel_path}"
        entity = Entity(
            id=entity_id,
            name=module_name,
            type=EntityType.MODULE,
            file_path=str(file_path),
            line_number=0,
        )
        self.graph.entities[entity_id] = entity
        self._file_modules[rel_path] = entity
        return entity

    def resolve_call_target(
        self, func_name: str, from_file: Path, line: int = 0, char: int = 0
    ) -> str | None:
        rel_path = str(from_file.relative_to(self.root_dir))
        current_dir = from_file.parent

        for parent in [current_dir] + list(current_dir.parents):
            if parent == self.root_dir or not str(parent).startswith(str(self.root_dir)):
                break
            search_rel = str(parent.relative_to(self.root_dir))
            if search_rel in self._file_modules:
                module_entity = self._file_modules[search_rel]
                for eid, entity in self.graph.entities.items():
                    if entity.name == func_name and entity.file_path.startswith(str(parent)):
                        return eid

        if self._lsp and from_file.suffix in (".ts", ".tsx"):
            try:
                defs = self._lsp.get_definitions(str(from_file), line, char)
                for d in defs:
                    target_path = d.get("uri", "").replace("file://", "")
                    for eid, entity in self.graph.entities.items():
                        if entity.file_path == target_path:
                            return eid
            except Exception:
                pass

        return None


class EntityVisitor:
    def __init__(self, file_path: Path, module_id: str, parser: CodeParser):
        self.file_path = file_path
        self.module_id = module_id
        self.parser = parser
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._current_class: str | None = None
        self._known_functions: set[str] = set()

    def visit(self, node) -> None:
        method = getattr(self, f"visit_{node.type}", self.generic_visit)
        method(node)

    def generic_visit(self, node) -> None:
        for child in node.children:
            self.visit(child)

    def visit_class_declaration(self, node) -> None:
        name = self._get_name(node)
        entity_id = f"{self.file_path}:{name}"
        self.entities[entity_id] = Entity(
            id=entity_id,
            name=name,
            type=EntityType.CLASS,
            file_path=str(self.file_path),
            line_number=node.start_point[0] + 1,
            docstring=self._get_docstring(node),
            signature=self._get_class_signature(node),
        )
        old_class = self._current_class
        self._current_class = entity_id
        self.generic_visit(node)
        self._current_class = old_class

    def visit_function_declaration(self, node) -> None:
        name = self._get_name(node)
        entity_id = f"{self.file_path}:{name}"
        self.entities[entity_id] = Entity(
            id=entity_id,
            name=name,
            type=EntityType.FUNCTION,
            file_path=str(self.file_path),
            line_number=node.start_point[0] + 1,
            signature=self._get_func_signature(node),
        )
        self._known_functions.add(name)

        if self._current_class:
            self.relations.append(
                Relation(
                    source_id=self._current_class,
                    target_id=entity_id,
                    type=RelationType.CONTAINS,
                    weight=2.0,
                )
            )

        self.generic_visit(node)

    def visit_method_definition(self, node) -> None:
        name = self._get_name(node)
        entity_id = f"{self.file_path}:{name}"
        self.entities[entity_id] = Entity(
            id=entity_id,
            name=name,
            type=EntityType.FUNCTION,
            file_path=str(self.file_path),
            line_number=node.start_point[0] + 1,
            signature=self._get_func_signature(node),
        )
        self._known_functions.add(name)

        if self._current_class:
            self.relations.append(
                Relation(
                    source_id=self._current_class,
                    target_id=entity_id,
                    type=RelationType.CONTAINS,
                    weight=2.0,
                )
            )

        self.generic_visit(node)

    def visit_assignment(self, node) -> None:
        name = self._get_assignment_name(node)
        if name and name.isupper():
            entity_id = f"{self.file_path}:{name}"
            self.entities[entity_id] = Entity(
                id=entity_id,
                name=name,
                type=EntityType.VARIABLE,
                file_path=str(self.file_path),
                line_number=node.start_point[0] + 1,
            )
            if self._current_class:
                self.relations.append(
                    Relation(
                        source_id=self._current_class,
                        target_id=entity_id,
                        type=RelationType.CONTAINS,
                        weight=1.0,
                    )
                )
        self.generic_visit(node)

    def visit_call_expression(self, node) -> None:
        func_name = self._get_call_name(node)
        if not func_name:
            self.generic_visit(node)
            return

        line = node.start_point[0]
        char = node.start_point[1] if node.start_point else 0
        target_id = self.parser.resolve_call_target(func_name, self.file_path, line, char)

        if target_id:
            source_id = self._current_class if self._current_class else self.module_id
            self.relations.append(
                Relation(
                    source_id=source_id,
                    target_id=target_id,
                    type=RelationType.CALLS,
                    weight=3.0,
                )
            )
        else:
            target_id = f"call:{func_name}"
            self.relations.append(
                Relation(
                    source_id=self._current_class or self.module_id,
                    target_id=target_id,
                    type=RelationType.CALLS,
                    weight=1.0,
                )
            )

        self.generic_visit(node)

    def visit_import_statement(self, node) -> None:
        module_name = self._get_import_name(node)
        if module_name:
            self._add_import_relation(module_name)
        self.generic_visit(node)

    def visit_import_from_statement(self, node) -> None:
        module_name = self._get_import_from_name(node)
        if module_name:
            self._add_import_relation(module_name)
        self.generic_visit(node)

    def _add_import_relation(self, module_name: str) -> None:
        target_id = f"module:{module_name}"
        source_id = self._current_class or self.module_id

        if target_id not in self.graph.entities:
            self.graph.entities[target_id] = Entity(
                id=target_id,
                name=module_name.split(".")[-1],
                type=EntityType.MODULE,
                file_path=module_name,
                line_number=0,
            )

        self.relations.append(
            Relation(
                source_id=source_id,
                target_id=target_id,
                type=RelationType.IMPORTS,
                weight=1.0,
            )
        )

    def _get_name(self, node) -> str:
        for child in node.children:
            if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8")
        return ""

    def _get_call_name(self, node) -> str:
        if node.type == "identifier":
            return node.text.decode("utf-8")
        for child in node.children:
            if child.type in ("identifier", "attribute"):
                name = child.text.decode("utf-8")
                if child.type == "attribute":
                    return child.children[-1].text.decode("utf-8") if child.children else name
                return name
        return ""

    def _get_assignment_name(self, node) -> str:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return ""

    def _get_import_name(self, node) -> str:
        for child in node.children:
            if child.type == "module":
                return child.text.decode("utf-8")
        return ""

    def _get_import_from_name(self, node) -> str:
        for child in node.children:
            if child.type == "module":
                return child.text.decode("utf-8")
        return ""

    def _get_docstring(self, node) -> str | None:
        docstring = ""
        for child in node.children:
            if child.type == "expression_statement":
                for grandchild in child.children:
                    if grandchild.type == "string":
                        docstring = grandchild.text.decode("utf-8").strip('"').strip("'")
                        break
        return docstring if docstring else None

    def _get_class_signature(self, node) -> str:
        return f"class {self._get_name(node)}"

    def _get_func_signature(self, node) -> str:
        name = self._get_name(node)
        params = []
        for child in node.children:
            if child.type == "parameters":
                params.append(child.text.decode("utf-8"))
        return f"def {name}({', '.join(params)})"


def parse_project(root_dir: str, use_lsp: bool = True, progress_callback=None) -> KnowledgeGraph:
    root = Path(root_dir).resolve()
    graph = KnowledgeGraph()

    file_paths = []
    for ext in (".py", ".js", ".ts", ".jsx", ".tsx"):
        for file_path in root.rglob(f"*{ext}"):
            if "node_modules" in str(file_path) or "__pycache__" in str(file_path):
                continue
            file_paths.append(file_path)

    workers = min(32, os.cpu_count() or 4)
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_parse_file_raw, str(fp), str(root)): fp
            for fp in file_paths
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress_callback:
                progress_callback(len(results), len(file_paths))

    for result in results:
        if result is None:
            continue
        for eid, entity in result["entities"]:
            graph.entities[eid] = entity
        graph.relations.extend(result["relations"])

    return graph
