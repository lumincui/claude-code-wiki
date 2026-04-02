from typing import Optional

from claude_code_wiki.graph.models import Component, Entity, KnowledgeGraph


class LLMClient:
    def __init__(self, config: Optional["LLMConfig"] = None):
        self.config = config
        self._agent = None

    @property
    def agent(self):
        if self._agent is None:
            from .agent import MiniMaxReActAgent

            self._agent = MiniMaxReActAgent(
                api_key=self.config.api_key if self.config else None,
                model=self.config.model if self.config else "MiniMax-M2.7",
            )
        return self._agent

    def generate_description(
        self, component: Component, entities: list[Entity], relations: list
    ) -> str:
        analysis = self.agent.analyze_component(
            component, KnowledgeGraph(entities={e.id: e for e in entities}, relations=relations)
        )
        intent = analysis.get("design_intent", "")
        decisions = analysis.get("key_decisions", [])
        return f"## Design Intent\n{intent}\n\n## Key Decisions\n" + "\n".join(
            f"- {d}" for d in decisions
        )

    def generate_component_name(self, entities: list[Entity]) -> str:
        return self.agent.name_component(entities)

    def should_merge(
        self, component_a: Component, component_b: Component, graph: KnowledgeGraph
    ) -> bool:
        return self.agent.should_merge(component_a, component_b, graph)


class ArchitectureAnalyzer:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def analyze_and_build_hierarchy(
        self, graph: KnowledgeGraph, workspace_root: Optional[str] = None
    ) -> KnowledgeGraph:
        from claude_code_wiki.graph.storage import CommunityDetector

        detector = CommunityDetector()
        communities = detector.detect_components(graph)

        all_components: dict[str, Component] = {}

        for i, community in enumerate(communities):
            entity_objs = [graph.entities[eid] for eid in community if eid in graph.entities]
            if not entity_objs:
                continue

            comp_id = f"comp_{i}"
            try:
                name = self.llm.generate_component_name(entity_objs)
            except Exception:
                name = self._fallback_name(entity_objs)

            all_components[comp_id] = Component(
                id=comp_id,
                name=name,
                entities=list(community),
                parent=None,
            )

        comp_list = list(all_components.values())
        to_merge = []
        for j, comp_a in enumerate(comp_list):
            for comp_b in comp_list[j + 1 :]:
                try:
                    if self.llm.should_merge(comp_a, comp_b, graph):
                        to_merge.append((comp_a.id, comp_b.id))
                except Exception:
                    pass

        for a_id, b_id in to_merge:
            if a_id in all_components and b_id in all_components:
                all_components[a_id].entities.extend(all_components[b_id].entities)
                del all_components[b_id]

        graph.components = all_components
        for comp in graph.components.values():
            comp.layer = 1
        return graph

    @property
    def agent(self):
        return self.llm.agent

    def _fallback_name(self, entities: list[Entity]) -> str:
        if not entities:
            return "UnnamedComponent"
        from pathlib import Path

        file_path = entities[0].file_path
        parts = Path(file_path).parts
        return parts[-2] if len(parts) > 1 else Path(file_path).stem


from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str = "minimax"
    model: str = "MiniMax-M2.7"
    api_key: Optional[str] = None
