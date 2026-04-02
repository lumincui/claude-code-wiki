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

        for comp in graph.components.values():
            entities = [graph.entities[eid] for eid in comp.entities if eid in graph.entities]
            outgoing = [r for r in graph.relations if r.source_id in comp.entities]
            incoming = [r for r in graph.relations if r.target_id in comp.entities]

            content = template.render(
                component=comp,
                entities=entities,
                outgoing_relations=outgoing[:10],
                incoming_relations=incoming[:10],
            )
            safe_name = comp.name.lower().replace(" ", "-")
            (comp_dir / f"{safe_name}.md").write_text(content)

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
