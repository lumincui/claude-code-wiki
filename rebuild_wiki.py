#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
from claude_code_wiki.graph.storage import GraphStorage, DirectoryAwareDetector
from claude_code_wiki.graph.models import Component
from claude_code_wiki.wiki.generator import WikiGenerator

storage = GraphStorage('knowledge.db')
graph = storage.load()
print(f'Loaded: {len(graph.entities)} entities')

detector = DirectoryAwareDetector()
communities = detector.detect_components(graph)
print(f'Communities: {len(communities)}')

graph.components.clear()
for i, community in enumerate(communities):
    entity_objs = [graph.entities[eid] for eid in community if eid in graph.entities]
    if not entity_objs:
        continue
    
    subdir_counts = Counter()
    for e in entity_objs:
        if e.file_path:
            parts = Path(e.file_path).parts
            if 'src' in parts:
                idx = parts.index('src')
                if idx + 1 < len(parts):
                    subdir_counts[parts[idx + 1]] += 1
    
    if subdir_counts:
        name = subdir_counts.most_common(1)[0][0]
    else:
        name = 'Component'
    
    comp_id = f'comp_{i}'
    graph.components[comp_id] = Component(id=comp_id, name=name, entities=list(community))

print(f'Built {len(graph.components)} components')

name_counts = Counter(comp.name for comp in graph.components.values())
print('Component name distribution:')
for name, count in name_counts.most_common(15):
    print(f'  {name}: {count}')

storage.save(graph)
print('Saved to DB')

wiki_gen = WikiGenerator(Path('wiki-output'))
wiki_gen.generate(graph)
print('Wiki generated!')