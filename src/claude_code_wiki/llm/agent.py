import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from claude_code_wiki.graph.models import Component, Entity, KnowledgeGraph

load_dotenv()


@tool
def read_source_file(file_path: str, max_lines: int = 100) -> str:
    """Read the contents of a source code file.

    Args:
        file_path: Absolute path to the source file
        max_lines: Maximum number of lines to read (default 100)

    Returns:
        File contents as string
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        lines = path.read_text(encoding="utf-8").split("\n")[:max_lines]
        return f"=== {file_path} ===\n" + "\n".join(lines)
    except Exception as e:
        return f"Error reading {file_path}: {e}"


@tool
def read_multiple_files(file_paths: list[str], max_lines_each: int = 80) -> str:
    """Read multiple source files at once.

    Args:
        file_paths: List of file paths to read
        max_lines_each: Max lines per file

    Returns:
        Combined file contents
    """
    results = []
    for fp in file_paths[:5]:
        results.append(read_source_file.invoke({"file_path": fp, "max_lines": max_lines_each}))
    return "\n\n".join(results)


@tool
def search_files_by_name(root_dir: str, pattern: str) -> str:
    """Search for files matching a pattern in directory tree.

    Args:
        root_dir: Root directory to search
        pattern: Filename pattern (e.g., "*.py", "*.ts")

    Returns:
        List of matching file paths
    """
    try:
        root = Path(root_dir)
        if not root.exists():
            return "Directory not found"
        files = [str(p) for p in root.rglob(pattern)]
        return "\n".join(files[:50])
    except Exception as e:
        return f"Error: {e}"


@tool
def list_directory(path: str) -> str:
    """List contents of a directory.

    Args:
        path: Directory path

    Returns:
        Directory listing
    """
    try:
        p = Path(path)
        if not p.exists():
            return f"Directory not found: {path}"
        items = []
        for item in p.iterdir():
            items.append(f"{'d' if item.is_dir() else 'f'} {item.name}")
        return "\n".join(sorted(items)[:30])
    except Exception as e:
        return f"Error: {e}"


class MiniMaxReActAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "MiniMax-M2.7"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("MINIMAX_API_KEY")
        self.model = model
        self.api_base = os.getenv("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
        self._agent = None

    def _create_agent(self, workspace_root: str):
        from langgraph.prebuilt import create_react_agent

        base_url = (
            self.api_base.rstrip("/") if self.api_base else "https://api.minimaxi.com/anthropic"
        )

        llm = ChatAnthropic(
            model=self.model,
            anthropic_api_key=self.api_key,
            base_url=base_url,
            max_tokens_to_sample=4096,
        )

        tools = [read_source_file, read_multiple_files, search_files_by_name, list_directory]

        agent = create_react_agent(
            llm,
            tools,
            state_modifier="""
You are an expert architecture analyst with access to source code tools.
Always read relevant source files before providing your analysis.
""",
        )
        return agent

    def analyze_component(
        self, component: Component, graph: KnowledgeGraph, workspace_root: Optional[str] = None
    ) -> dict[str, Any]:
        entities = [graph.entities[eid] for eid in component.entities if eid in graph.entities]
        source_files = list(set(e.file_path for e in entities))

        workspace = workspace_root or (str(Path(source_files[0]).parent) if source_files else ".")

        prompt = f"""You are an expert architecture analyst. Analyze this component thoroughly.

Component: {component.name}
Entity count: {len(entities)}
Sample entities: {", ".join(e.name for e in entities[:10])}

Source files:
{chr(10).join(source_files[:20])}

Your task:
1. Read key source files to understand the implementation
2. Determine the DESIGN INTENT - what architectural problem does this solve?
3. Identify KEY ARCHITECTURAL DECISIONS
4. Extract CORE INTERFACES (public classes/functions with purposes)
5. Suggest a better COMPONENT NAME if needed
6. Determine the DOMAIN this belongs to

Work step by step using your tools.

Return JSON:
{{
  "design_intent": "description",
  "key_decisions": ["decision 1", "decision 2"],
  "core_interfaces": [{{"name": "ClassName", "signature": "...", "purpose": "..."}}],
  "domain": "Domain name",
  "suggested_name": "BetterName"
}}"""

        agent = self._create_agent(workspace)

        try:
            result = agent.invoke({"messages": [("user", prompt)]})
            messages = result.get("messages", [])
            response = ""
            for msg in messages:
                if hasattr(msg, "content"):
                    response = msg.content
            return self._parse_json_response(str(response))
        except Exception as e:
            return {
                "design_intent": f"Analysis failed: {e}",
                "key_decisions": [],
                "core_interfaces": [],
                "domain": "Unknown",
                "suggested_name": component.name,
            }

    def name_component(self, entities: list[Entity], workspace_root: Optional[str] = None) -> str:
        source_files = list(set(e.file_path for e in entities))
        workspace = workspace_root or (str(Path(source_files[0]).parent) if source_files else ".")

        prompt = f"""Analyze these code entities and suggest a component name.

Entities: {", ".join(e.name for e in entities)}

Files: {chr(10).join(source_files[:10])}

Read 2-3 key files to understand the purpose, then suggest a 1-2 word name.

Return ONLY the name."""

        agent = self._create_agent(workspace)

        try:
            result = agent.invoke({"messages": [("user", prompt)]})
            messages = result.get("messages", [])
            response = ""
            for msg in messages:
                if hasattr(msg, "content"):
                    response = msg.content
            return str(response).strip()
        except Exception:
            return self._fallback_name(entities)

    def should_merge(
        self,
        comp_a: Component,
        comp_b: Component,
        graph: KnowledgeGraph,
        workspace_root: Optional[str] = None,
    ) -> bool:
        entities_a = [graph.entities[eid] for eid in comp_a.entities if eid in graph.entities]
        entities_b = [graph.entities[eid] for eid in comp_b.entities if eid in graph.entities]

        source_files = list(set(e.file_path for e in entities_a + entities_b))
        workspace = workspace_root or (str(Path(source_files[0]).parent) if source_files else ".")

        prompt = f"""Two components - should they merge?

Component A ({comp_a.name}):
- {", ".join(e.name for e in entities_a[:8])}

Component B ({comp_b.name}):
- {", ".join(e.name for e in entities_b[:8])}

Read relevant files, then answer YES or NO."""

        agent = self._create_agent(workspace)

        try:
            result = agent.invoke({"messages": [("user", prompt)]})
            messages = result.get("messages", [])
            response = ""
            for msg in messages:
                if hasattr(msg, "content"):
                    response = msg.content
            return "YES" in str(response).upper()
        except Exception:
            return False

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {
            "design_intent": response[:300] if response else "No analysis",
            "key_decisions": [],
            "core_interfaces": [],
            "domain": "Unknown",
            "suggested_name": None,
        }

    def _fallback_name(self, entities: list[Entity]) -> str:
        if not entities:
            return "UnnamedComponent"
        file_path = entities[0].file_path
        parts = Path(file_path).parts
        return parts[-2] if len(parts) > 1 else Path(file_path).stem


class ArchitectureAgent:
    def __init__(self, agent: Optional[MiniMaxReActAgent] = None):
        self.agent = agent or MiniMaxReActAgent()

    def analyze_and_build_hierarchy(
        self, graph: KnowledgeGraph, workspace_root: Optional[str] = None
    ) -> KnowledgeGraph:
        from claude_code_wiki.graph.storage import CommunityDetector

        detector = CommunityDetector()
        communities = detector.detect_components(graph)

        components = {}
        for i, community in enumerate(communities):
            entity_objs = [graph.entities[eid] for eid in community if eid in graph.entities]
            if not entity_objs:
                continue

            comp_id = f"component_{i}"
            try:
                name = self.agent.name_component(entity_objs, workspace_root)
            except Exception:
                name = self._fallback_name(entity_objs)

            components[comp_id] = Component(
                id=comp_id,
                name=name,
                entities=list(community),
            )

        comp_list = list(components.values())
        to_merge = []
        for j, comp_a in enumerate(comp_list):
            for comp_b in comp_list[j + 1 :]:
                try:
                    if self.agent.should_merge(comp_a, comp_b, graph, workspace_root):
                        to_merge.append((comp_a.id, comp_b.id))
                except Exception:
                    pass

        for a_id, b_id in to_merge:
            if a_id in components and b_id in components:
                components[a_id].entities.extend(components[b_id].entities)
                del components[b_id]

        graph.components = components
        return graph

    def enrich_component(
        self, component: Component, graph: KnowledgeGraph, workspace_root: Optional[str] = None
    ) -> dict[str, Any]:
        try:
            return self.agent.analyze_component(component, graph, workspace_root)
        except Exception as e:
            return {
                "design_intent": f"Analysis failed: {e}",
                "key_decisions": [],
                "core_interfaces": [],
                "domain": "Unknown",
                "suggested_name": component.name,
            }

    def _fallback_name(self, entities: list[Entity]) -> str:
        if not entities:
            return "UnnamedComponent"
        file_path = entities[0].file_path
        parts = Path(file_path).parts
        return parts[-2] if len(parts) > 1 else Path(file_path).stem
