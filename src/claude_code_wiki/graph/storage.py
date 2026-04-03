import sqlite3
from pathlib import Path
from typing import Optional

import networkx as nx

from .models import Component, Entity, KnowledgeGraph, Relation


class GraphStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                docstring TEXT,
                signature TEXT,
                layer INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS relations (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            );
            CREATE TABLE IF NOT EXISTS components (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent TEXT,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS component_entities (
                component_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (component_id, entity_id),
                FOREIGN KEY (component_id) REFERENCES components(id),
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
        """)
        conn.close()

    def save(self, graph: KnowledgeGraph) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for entity in graph.entities.values():
            cur.execute(
                """
                INSERT OR REPLACE INTO entities 
                (id, name, type, file_path, line_number, docstring, signature, layer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entity.id,
                    entity.name,
                    entity.type.value,
                    entity.file_path,
                    entity.line_number,
                    entity.docstring,
                    entity.signature,
                    entity.layer,
                ),
            )

        for rel in graph.relations:
            cur.execute(
                """
                INSERT OR REPLACE INTO relations (source_id, target_id, type, weight)
                VALUES (?, ?, ?, ?)
            """,
                (rel.source_id, rel.target_id, rel.type.value, rel.weight),
            )

        for comp in graph.components.values():
            cur.execute(
                """
                INSERT OR REPLACE INTO components (id, name, parent, description)
                VALUES (?, ?, ?, ?)
            """,
                (comp.id, comp.name, comp.parent, comp.description),
            )
            for entity_id in comp.entities:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO component_entities (component_id, entity_id)
                    VALUES (?, ?)
                """,
                    (comp.id, entity_id),
                )
        conn.commit()
        conn.close()

    def load(self) -> KnowledgeGraph:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        graph = KnowledgeGraph()
        for row in cur.execute("SELECT * FROM entities"):
            from .models import EntityType

            graph.entities[row["id"]] = Entity(
                id=row["id"],
                name=row["name"],
                type=EntityType(row["type"]),
                file_path=row["file_path"],
                line_number=row["line_number"],
                docstring=row["docstring"],
                signature=row["signature"],
                layer=row["layer"],
            )

        from .models import RelationType

        for row in cur.execute("SELECT * FROM relations"):
            graph.relations.append(
                Relation(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    type=RelationType(row["type"]),
                    weight=row["weight"],
                )
            )

        for row in cur.execute("SELECT * FROM components"):
            comp = Component(
                id=row["id"], name=row["name"], parent=row["parent"], description=row["description"]
            )
            for (entity_id,) in cur.execute(
                "SELECT entity_id FROM component_entities WHERE component_id = ?", (row["id"],)
            ):
                comp.entities.append(entity_id)
            graph.components[comp.id] = comp

        conn.close()
        return graph


class CommunityDetector:
    WEIGHT_MAP = {
        "calls": 3.0,
        "contains": 2.0,
        "imports": 1.0,
    }

    def detect_components(self, graph: KnowledgeGraph) -> list[set[str]]:
        entity_ids = list(graph.entities.keys())
        return self.detect_from_entities(entity_ids, graph.relations)

    def detect_from_entities(
        self, entity_ids: list[str], relations: list[Relation]
    ) -> list[set[str]]:
        nx_graph = nx.DiGraph()
        entity_set = set(entity_ids)
        for eid in entity_ids:
            nx_graph.add_node(eid)

        for rel in relations:
            if rel.source_id in entity_set and rel.target_id in entity_set:
                rel_type = rel.type.value
                if rel_type in self.WEIGHT_MAP:
                    weight = rel.weight * self.WEIGHT_MAP[rel_type]
                    nx_graph.add_edge(rel.source_id, rel.target_id, weight=weight)

        undirected = nx_graph.to_undirected()
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(undirected, weight="weight")
        except ImportError:
            partition = self._simple_community_detection(undirected)

        communities: dict[int, set[str]] = {}
        for entity_id, comm_id in partition.items():
            communities.setdefault(comm_id, set()).add(entity_id)
        return list(communities.values())

    def _simple_community_detection(self, graph) -> dict[str, int]:
        connected = list(nx.connected_components(graph))
        return {node: i for i, comp in enumerate(connected) for node in comp}


class DirectoryAwareDetector:
    """Groups entities by directory first, then refines with graph analysis."""

    # Min entities to consider a directory a separate component
    MIN_DIR_COMPONENT_SIZE = 3

    def detect_components(self, graph: KnowledgeGraph) -> list[set[str]]:
        # Step 1: Group entities by parent directory
        dir_groups: dict[str, set[str]] = {}
        module_to_dir: dict[str, str] = {}

        for eid, entity in graph.entities.items():
            if entity.type.value == "module":
                # module entities have id like "module:src/ink/output.ts"
                # file_path is the actual file path
                dir_path = str(Path(entity.file_path).parent)
                dir_name = Path(entity.file_path).parent.name
            else:
                # For classes/functions, find their parent module
                # Entity id is like "/path/to/file.ts:className"
                parts = eid.rsplit(":", 1)
                if len(parts) == 2 and ":" in eid:
                    file_part = parts[0]
                    dir_path = str(Path(file_part).parent)
                    dir_name = Path(file_part).parent.name
                else:
                    dir_path = str(Path(entity.file_path).parent)
                    dir_name = Path(entity.file_path).parent.name

            if dir_path not in dir_groups:
                dir_groups[dir_path] = set()
            dir_groups[dir_path].add(eid)

            if entity.type.value == "module":
                module_to_dir[eid] = dir_path

        # Step 2: Put all entities into directory-based components
        dir_components: dict[str, set[str]] = {}
        for eid, entity in graph.entities.items():
            if entity.type.value == "module":
                dir_path = module_to_dir.get(eid)
            else:
                parts = eid.rsplit(":", 1)
                file_part = parts[0] if len(parts) == 2 and ":" in eid else entity.file_path
                dir_path = str(Path(file_part).parent)

            if dir_path is None:
                continue
            if dir_path not in dir_components:
                dir_components[dir_path] = set()
            dir_components[dir_path].add(eid)

        # Step 3: Build a directory-level graph to find cross-directory relations
        dir_graph: dict[str, dict[str, float]] = {}
        for rel in graph.relations:
            src_dir = self._get_entity_dir(rel.source_id, graph.entities, module_to_dir)
            tgt_dir = self._get_entity_dir(rel.target_id, graph.entities, module_to_dir)
            if src_dir and tgt_dir and src_dir != tgt_dir:
                if src_dir not in dir_graph:
                    dir_graph[src_dir] = {}
                weight = rel.weight * (3.0 if rel.type.value == "calls" else 1.0)
                dir_graph[src_dir][tgt_dir] = dir_graph[src_dir].get(tgt_dir, 0) + weight

        # Step 4: Merge small directories that are heavily cross-connected
        merge_threshold = 5.0
        merged: dict[str, str] = {}

        for dir_path, entities in list(dir_components.items()):
            if len(entities) < self.MIN_DIR_COMPONENT_SIZE and dir_path in dir_graph:
                connections = dir_graph[dir_path]
                if connections:
                    best_target: str
                    best_weight: float
                    best_target, best_weight = max(connections.items(), key=lambda x: x[1])
                    if best_weight >= merge_threshold:
                        target_dir = merged.get(best_target, best_target)
                        if target_dir not in dir_components:
                            dir_components[target_dir] = set()
                        dir_components[target_dir].update(entities)
                        del dir_components[dir_path]
                        merged[dir_path] = target_dir

        # Step 5: Generate component names from directory names
        components = []
        for dir_path, entities in dir_components.items():
            if not entities:
                continue
            dir_name = Path(dir_path).name or Path(dir_path).parent.name
            components.append((dir_name, entities))

        return [entities for _, entities in components]

    def _get_entity_dir(
        self, eid: str, entities: dict[str, Entity], module_to_dir: dict[str, str]
    ) -> str | None:
        """Get the directory path for an entity."""
        if eid in module_to_dir:
            return module_to_dir[eid]

        if eid in entities:
            entity = entities[eid]
            return str(Path(entity.file_path).parent)

        # Try to extract file path from entity id
        parts = eid.rsplit(":", 1)
        if len(parts) == 2 and ":" in eid:
            return str(Path(parts[0]).parent)
        return None

