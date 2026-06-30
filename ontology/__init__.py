"""
Ontology Module for PHTN.AI Sub-Agent Framework

Provides ontology and knowledge graph capabilities:
- Concept definitions (Agent, Capability, Intent, Resource, etc.)
- Relationship management
- Entity extraction
- Knowledge graph queries
- Terminology management
- Industry standards mapping
"""

from .types import (
    OntologyConcept,
    OntologyRelationship,
    OntologyEntity,
    OntologyProperty,
    ConceptType,
    RelationshipType,
)
from .knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphConfig,
    GraphNode,
    GraphEdge,
    GraphQuery,
    GraphQueryResult,
)
from .manager import (
    OntologyManager,
    OntologyConfig,
)
from .entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    EntityExtractionConfig,
)

__all__ = [
    "OntologyConcept",
    "OntologyRelationship",
    "OntologyEntity",
    "OntologyProperty",
    "ConceptType",
    "RelationshipType",
    "KnowledgeGraph",
    "KnowledgeGraphConfig",
    "GraphNode",
    "GraphEdge",
    "GraphQuery",
    "GraphQueryResult",
    "OntologyManager",
    "OntologyConfig",
    "EntityExtractor",
    "ExtractedEntity",
    "EntityExtractionConfig",
]
