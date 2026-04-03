from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from claude_code_wiki.graph.models import Component, Entity, KnowledgeGraph


class WikiGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.templates_dir = Path(__file__).parent.parent.parent.parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(),
        )

    def generate(self, graph: KnowledgeGraph) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._generate_index(graph)
        self._generate_components(graph)
        self._generate_relationships(graph)
        self._generate_sidebar(graph)

    def _generate_index(self, graph: KnowledgeGraph) -> None:
        template = self.env.get_template("index.md.j2")
        content = template.render(
            components=list(graph.components.values()),
            entity_count=len(graph.entities),
            relation_count=len(graph.relations),
        )
        (self.output_dir / "index.md").write_text(content)

    def _generate_components(self, graph: KnowledgeGraph) -> None:
        comp_dir = self.output_dir / "components"
        comp_dir.mkdir(exist_ok=True)
        template = self.env.get_template("component.md.j2")

        used_names: dict[str, int] = {}
        entity_file_map: dict[str, tuple[str, str]] = {}

        for comp in graph.components.values():
            entities = [graph.entities[eid] for eid in comp.entities if eid in graph.entities]
            outgoing = [r for r in graph.relations if r.source_id in comp.entities]
            incoming = [r for r in graph.relations if r.target_id in comp.entities]

            base_name = comp.name.lower().replace(" ", "-").replace(".", "-")
            if base_name in used_names:
                used_names[base_name] += 1
                safe_name = f"{base_name}-{used_names[base_name]}"
            else:
                used_names[base_name] = 1
                safe_name = base_name

            subdir = comp_dir / safe_name
            subdir.mkdir(exist_ok=True)

            entity_file_map.clear()
            for entity in entities:
                safe_entity_name = self._safe_name(entity.name)
                entity_file = subdir / f"{safe_entity_name}.md"
                entity_content = self._render_entity(entity, graph)
                entity_file.write_text(entity_content)
                entity_file_map[entity.id] = (safe_name, safe_entity_name)

            content = template.render(
                component=comp,
                entities=entities,
                outgoing_relations=outgoing[:20],
                incoming_relations=incoming[:20],
                entity_file_map=entity_file_map,
            )
            (comp_dir / f"{safe_name}.md").write_text(content)

    def _safe_name(self, name: str) -> str:
        return name.replace(" ", "-").replace(":", "-").replace(".", "-")

    def _render_entity(self, entity: Entity, graph: KnowledgeGraph) -> str:
        lines = [f"# {entity.name}", ""]
        if entity.type.value:
            lines.append(f"**Type:** {entity.type.value}")
        if entity.signature:
            lines.append(f"```\n{entity.signature}\n```")
        if entity.docstring:
            lines.append(entity.docstring)
        if entity.file_path:
            lines.append(f"**File:** `{entity.file_path}`")
            lines.append(f"**Line:** {entity.line_number or 'unknown'}")
        return "\n".join(lines)

    def _generate_relationships(self, graph: KnowledgeGraph) -> None:
        template = self.env.get_template("relationships.md.j2")
        content = template.render(
            relations=graph.relations[:50],
            entities=graph.entities,
        )
        (self.output_dir / "relationships.md").write_text(content)

    def _generate_sidebar(self, graph: KnowledgeGraph) -> None:
        template = self.env.get_template("sidebar.md.j2")
        content = template.render(
            components=list(graph.components.values()),
        )
        (self.output_dir / "sidebar.md").write_text(content)
