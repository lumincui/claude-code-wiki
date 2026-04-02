from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EntityType(Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    INTERFACE = "interface"


class RelationType(Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    CONTAINS = "contains"
    TYPE_OF = "type_of"
    DEPENDS_ON = "depends_on"


@dataclass
class Entity:
    id: str
    name: str
    type: EntityType
    file_path: str
    line_number: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    layer: int = 0  # 1=Class, 2=Component, 3=Domain, etc.


@dataclass
class Relation:
    source_id: str
    target_id: str
    type: RelationType
    weight: float = 1.0


@dataclass
class Component:
    id: str
    name: str
    entities: list[str] = field(default_factory=list)  # entity IDs
    parent: Optional[str] = None  # parent component ID
    description: Optional[str] = None


@dataclass
class KnowledgeGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    components: dict[str, Component] = field(default_factory=dict)
