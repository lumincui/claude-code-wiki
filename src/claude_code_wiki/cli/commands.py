import os
from pathlib import Path

import click
from dotenv import load_dotenv

from claude_code_wiki.llm.agent import ArchitectureAgent, MiniMaxReActAgent
from claude_code_wiki.llm.client import LLMConfig
from claude_code_wiki.parser.code_parser import parse_project
from claude_code_wiki.wiki.generator import WikiGenerator

load_dotenv()


@click.group()
def cli():
    """Generate architecture-focused wiki from source code using ReAct Agent."""
    pass


@cli.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.option("--output", "-o", default="wiki-output", help="Output directory for wiki")
@click.option("--db", default="knowledge.db", help="SQLite graph database path")
@click.option("--model", default="MiniMax-M2.7")
@click.option("--api-key", default=None, help="API key (or set ANTHROPIC_API_KEY env)")
@click.option("--skip-agent", is_flag=True, help="Skip Agent analysis, use heuristics only")
@click.option("--lsp/--no-lsp", default=True, help="Use Deno LSP for TypeScript (default: enabled)")
def generate(
    source_dir: str, output: str, db: str, model: str, api_key: str, skip_agent: bool, lsp: bool
):
    """Parse source code and generate wiki using ReAct Agent."""
    click.echo(f"Parsing source code from: {source_dir}")

    graph = parse_project(source_dir, use_lsp=lsp)
    click.echo(f"Found {len(graph.entities)} entities, {len(graph.relations)} relations")

    from claude_code_wiki.graph.storage import GraphStorage

    storage = GraphStorage(db)
    storage.save(graph)
    click.echo(f"Saved graph to: {db}")

    if not skip_agent:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("MINIMAX_API_KEY")
        if not api_key:
            click.echo("Warning: No API key found, falling back to heuristics")
            skip_agent = True

    if skip_agent:
        click.echo("Using heuristic analysis (no LLM)...")
        from claude_code_wiki.graph.storage import CommunityDetector

        detector = CommunityDetector()
        communities = detector.detect_components(graph)
        from claude_code_wiki.graph.models import Component

        for i, community in enumerate(communities):
            entity_objs = [graph.entities[eid] for eid in community if eid in graph.entities]
            if not entity_objs:
                continue
            comp_id = f"comp_{i}"
            name = _fallback_name(entity_objs)
            graph.components[comp_id] = Component(id=comp_id, name=name, entities=list(community))
    else:
        click.echo("Running ReAct Agent analysis...")
        agent = ArchitectureAgent(MiniMaxReActAgent(api_key=api_key, model=model))

        click.echo("Building component hierarchy...")
        graph = agent.analyze_and_build_hierarchy(graph, workspace_root=source_dir)
        click.echo(f"Built {len(graph.components)} components")

        click.echo("Enriching components with deep analysis...")
        for comp in graph.components.values():
            click.echo(f"  Analyzing: {comp.name}")
            try:
                analysis = agent.enrich_component(comp, graph, workspace_root=source_dir)
                comp.description = analysis.get("design_intent", "")
                if analysis.get("suggested_name"):
                    comp.name = analysis["suggested_name"]
            except Exception as e:
                click.echo(f"    Warning: {e}")

        storage.save(graph)

    click.echo("Generating wiki...")
    wiki_gen = WikiGenerator(Path(output))
    wiki_gen.generate(graph)
    click.echo(f"Wrote wiki to: {output}")


@cli.command()
@click.argument("db", default="knowledge.db")
def stats(db: str):
    """Show graph statistics."""
    from claude_code_wiki.graph.storage import GraphStorage

    storage = GraphStorage(db)
    graph = storage.load()
    click.echo(f"Entities: {len(graph.entities)}")
    click.echo(f"Relations: {len(graph.relations)}")
    click.echo(f"Components: {len(graph.components)}")


def _fallback_name(entities: list) -> str:
    if not entities:
        return "UnnamedComponent"
    file_path = entities[0].file_path
    parts = Path(file_path).parts
    return parts[-2] if len(parts) > 1 else Path(file_path).stem


if __name__ == "__main__":
    cli()
