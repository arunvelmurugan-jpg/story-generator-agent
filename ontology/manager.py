"""
Ontology Manager for PHTN.AI Sub-Agent Framework

Central manager for ontology operations:
- Concept management
- Entity management
- Knowledge graph operations
- Terminology management
- Industry standards mapping
- Export/Import (OWL, RDF, JSON-LD)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from .types import (
    ConceptType,
    RelationshipType,
    OntologyConcept,
    OntologyEntity,
    OntologyRelationship,
    OntologyProperty,
    CORE_CONCEPTS,
)
from .knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphConfig,
    GraphNode,
    GraphEdge,
    GraphQuery,
    GraphQueryResult,
)
from .entity_extractor import (
    EntityExtractor,
    EntityExtractionConfig,
    ExtractedEntity,
)

logger = logging.getLogger(__name__)


@dataclass
class TerminologyConfig:
    """Terminology configuration."""
    version: str = "1.0"
    standard: Optional[str] = None
    custom_terms: Dict[str, str] = field(default_factory=dict)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class OntologyConfig:
    """Configuration for ontology manager."""
    enabled: bool = True
    domain: Optional[str] = None
    sub_domain: Optional[str] = None
    terminology: Optional[TerminologyConfig] = None
    knowledge_base: Optional[str] = None
    industry_standards: List[str] = field(default_factory=list)
    enable_entity_extraction: bool = True
    enable_knowledge_graph: bool = True
    auto_populate_core_concepts: bool = True


class OntologyManager:
    """
    Central manager for ontology and knowledge graph operations.
    
    Features:
    - Concept CRUD operations
    - Entity management
    - Knowledge graph queries
    - Terminology management
    - Industry standards mapping
    - Export to OWL/RDF/JSON-LD
    
    Aligned with phtnai-frontend OntologyExplorerTab and OntologyDataContracts.
    """
    
    def __init__(self, config: OntologyConfig):
        """
        Initialize ontology manager.
        
        Args:
            config: Ontology configuration
        """
        self.config = config
        
        self._concepts: Dict[str, OntologyConcept] = {}
        self._entities: Dict[str, OntologyEntity] = {}
        
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        if config.enable_knowledge_graph:
            self._knowledge_graph = KnowledgeGraph()
        
        self._entity_extractor: Optional[EntityExtractor] = None
        if config.enable_entity_extraction:
            self._entity_extractor = EntityExtractor()
        
        self._terminology = config.terminology or TerminologyConfig()
        
        if config.auto_populate_core_concepts:
            self._load_core_concepts()
        
        logger.info(f"OntologyManager initialized for domain: {config.domain}")
    
    def _load_core_concepts(self):
        """Load core ontology concepts."""
        for concept_type, concept in CORE_CONCEPTS.items():
            self._concepts[concept.id] = concept
            
            if self._knowledge_graph:
                node = GraphNode(
                    id=concept.id,
                    label=concept.name,
                    node_type=concept.concept_type.value,
                    properties={
                        "description": concept.description,
                        "instances": concept.instances,
                    },
                )
                self._knowledge_graph.add_node(node)
        
        if self._knowledge_graph:
            for concept in CORE_CONCEPTS.values():
                for rel in concept.relationships:
                    source_id = f"concept-{rel.source_concept.lower()}"
                    target_id = f"concept-{rel.target_concept.lower()}"
                    
                    if source_id in self._concepts and target_id in self._concepts:
                        edge = GraphEdge(
                            id=f"rel-{source_id}-{rel.relationship_type.value}-{target_id}",
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type=rel.relationship_type.value,
                        )
                        self._knowledge_graph.add_edge(edge)
        
        logger.debug(f"Loaded {len(self._concepts)} core concepts")
    
    def add_concept(self, concept: OntologyConcept) -> bool:
        """
        Add a concept to the ontology.
        
        Args:
            concept: Concept to add
            
        Returns:
            True if added successfully
        """
        self._concepts[concept.id] = concept
        
        if self._knowledge_graph:
            node = GraphNode(
                id=concept.id,
                label=concept.name,
                node_type=concept.concept_type.value,
                properties={
                    "description": concept.description,
                },
            )
            self._knowledge_graph.add_node(node)
        
        logger.debug(f"Added concept: {concept.name}")
        return True
    
    def get_concept(self, concept_id: str) -> Optional[OntologyConcept]:
        """Get a concept by ID."""
        return self._concepts.get(concept_id)
    
    def get_concepts_by_type(self, concept_type: ConceptType) -> List[OntologyConcept]:
        """Get all concepts of a specific type."""
        return [
            c for c in self._concepts.values()
            if c.concept_type == concept_type
        ]
    
    def list_concepts(self) -> List[OntologyConcept]:
        """List all concepts."""
        return list(self._concepts.values())
    
    def add_entity(self, entity: OntologyEntity) -> bool:
        """
        Add an entity (instance of a concept).
        
        Args:
            entity: Entity to add
            
        Returns:
            True if added successfully
        """
        self._entities[entity.id] = entity
        
        if self._knowledge_graph:
            node = GraphNode(
                id=entity.id,
                label=entity.name,
                node_type=entity.concept_type.value,
                properties=entity.properties,
            )
            self._knowledge_graph.add_node(node)
            
            if entity.concept_id in self._concepts:
                edge = GraphEdge(
                    id=f"instance-{entity.id}-{entity.concept_id}",
                    source_id=entity.id,
                    target_id=entity.concept_id,
                    relationship_type="instanceOf",
                )
                self._knowledge_graph.add_edge(edge)
        
        logger.debug(f"Added entity: {entity.name}")
        return True
    
    def get_entity(self, entity_id: str) -> Optional[OntologyEntity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)
    
    def list_entities(
        self,
        concept_type: Optional[ConceptType] = None,
    ) -> List[OntologyEntity]:
        """List entities, optionally filtered by concept type."""
        entities = list(self._entities.values())
        if concept_type:
            entities = [e for e in entities if e.concept_type == concept_type]
        return entities
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a relationship between entities.
        
        Args:
            source_id: Source entity/concept ID
            target_id: Target entity/concept ID
            relationship_type: Type of relationship
            properties: Optional relationship properties
            
        Returns:
            True if added successfully
        """
        if not self._knowledge_graph:
            return False
        
        edge = GraphEdge(
            id=f"rel-{source_id}-{relationship_type.value}-{target_id}",
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type.value,
            properties=properties or {},
        )
        
        return self._knowledge_graph.add_edge(edge)
    
    def query_graph(self, query: GraphQuery) -> GraphQueryResult:
        """
        Query the knowledge graph.
        
        Args:
            query: Graph query
            
        Returns:
            Query result
        """
        if not self._knowledge_graph:
            return GraphQueryResult()
        
        return self._knowledge_graph.query(query)
    
    def find_related(
        self,
        entity_id: str,
        relationship_type: Optional[RelationshipType] = None,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity.
        
        Args:
            entity_id: Source entity ID
            relationship_type: Optional filter by relationship type
            max_depth: Maximum traversal depth
            
        Returns:
            List of related entities with relationship info
        """
        if not self._knowledge_graph:
            return []
        
        query = GraphQuery(
            query_type="subgraph",
            start_node=entity_id,
            max_depth=max_depth,
            relationship_types=[relationship_type.value] if relationship_type else [],
        )
        
        result = self._knowledge_graph.query(query)
        
        related = []
        for node in result.nodes:
            if node.id != entity_id:
                related.append({
                    "id": node.id,
                    "name": node.label,
                    "type": node.node_type,
                    "properties": node.properties,
                })
        
        return related
    
    def extract_entities_from_text(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not self._entity_extractor:
            return []
        
        return self._entity_extractor.extract_entities(text)
    
    def populate_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text and add to knowledge graph.
        
        Args:
            text: Input text
            
        Returns:
            Summary of extracted and added entities
        """
        if not self._entity_extractor:
            return {"entities_added": 0, "relationships_added": 0}
        
        extraction_result = self._entity_extractor.extract_all(text)
        
        entities_added = 0
        relationships_added = 0
        
        for entity_dict in extraction_result.get("entities", []):
            entity = OntologyEntity(
                id=f"entity-{entity_dict['text'].lower().replace(' ', '-')}",
                name=entity_dict["text"],
                concept_type=ConceptType(entity_dict["concept_type"]) if entity_dict.get("concept_type") else ConceptType.RESOURCE,
                concept_id=f"concept-{entity_dict.get('concept_type', 'resource').lower()}",
                properties=entity_dict.get("metadata", {}),
            )
            if self.add_entity(entity):
                entities_added += 1
        
        for rel_dict in extraction_result.get("relationships", []):
            rel_type_str = rel_dict.get("relationship_type")
            if rel_type_str:
                try:
                    rel_type = RelationshipType(rel_type_str)
                    subject_id = f"entity-{rel_dict['subject']['text'].lower().replace(' ', '-')}"
                    object_id = f"entity-{rel_dict['object']['text'].lower().replace(' ', '-')}"
                    
                    if self.add_relationship(subject_id, object_id, rel_type):
                        relationships_added += 1
                except ValueError:
                    pass
        
        return {
            "entities_added": entities_added,
            "relationships_added": relationships_added,
            "total_extracted": len(extraction_result.get("entities", [])),
        }
    
    def get_terminology(self, term: str) -> Optional[str]:
        """
        Get definition for a term.
        
        Args:
            term: Term to look up
            
        Returns:
            Definition or None
        """
        return self._terminology.custom_terms.get(term.lower())
    
    def add_terminology(self, term: str, definition: str):
        """Add a term definition."""
        self._terminology.custom_terms[term.lower()] = definition
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        return self._terminology.synonyms.get(term.lower(), [])
    
    def add_synonym(self, term: str, synonym: str):
        """Add a synonym for a term."""
        term_lower = term.lower()
        if term_lower not in self._terminology.synonyms:
            self._terminology.synonyms[term_lower] = []
        if synonym not in self._terminology.synonyms[term_lower]:
            self._terminology.synonyms[term_lower].append(synonym)
    
    def get_ontology_metadata(self) -> Dict[str, Any]:
        """
        Get ontology metadata for agent card.
        
        Returns format compatible with phtnai-frontend OntologyDataContracts.
        """
        return {
            "domain": self.config.domain,
            "sub_domain": self.config.sub_domain,
            "terminology": {
                "version": self._terminology.version,
                "standard": self._terminology.standard,
            },
            "knowledge_base": self.config.knowledge_base,
            "industry_standards": self.config.industry_standards,
            "concept_count": len(self._concepts),
            "entity_count": len(self._entities),
        }
    
    def export_json_ld(self) -> Dict[str, Any]:
        """Export ontology as JSON-LD."""
        if self._knowledge_graph:
            graph_export = self._knowledge_graph.export_json_ld()
        else:
            graph_export = {"@graph": []}
        
        return {
            "@context": {
                "@vocab": "https://phtn.ai/ontology/",
                "domain": self.config.domain,
                "subDomain": self.config.sub_domain,
            },
            "metadata": self.get_ontology_metadata(),
            **graph_export,
        }
    
    def export_cytoscape(self) -> Dict[str, Any]:
        """Export for Cytoscape.js visualization."""
        if self._knowledge_graph:
            return self._knowledge_graph.export_cytoscape()
        return {"elements": {"nodes": [], "edges": []}}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ontology statistics."""
        stats = {
            "domain": self.config.domain,
            "sub_domain": self.config.sub_domain,
            "concept_count": len(self._concepts),
            "entity_count": len(self._entities),
            "terminology_count": len(self._terminology.custom_terms),
            "industry_standards": self.config.industry_standards,
        }
        
        if self._knowledge_graph:
            stats["graph_stats"] = self._knowledge_graph.get_stats()
        
        return stats


def create_ontology_manager(
    config_dict: Optional[Dict[str, Any]] = None,
) -> OntologyManager:
    """Create ontology manager from config dict."""
    if config_dict is None:
        config_dict = {}
    
    terminology_dict = config_dict.get("terminology", {})
    terminology = TerminologyConfig(
        version=terminology_dict.get("version", "1.0"),
        standard=terminology_dict.get("standard"),
        custom_terms=terminology_dict.get("custom_terms", {}),
        synonyms=terminology_dict.get("synonyms", {}),
    )
    
    config = OntologyConfig(
        enabled=config_dict.get("enabled", True),
        domain=config_dict.get("domain"),
        sub_domain=config_dict.get("sub_domain"),
        terminology=terminology,
        knowledge_base=config_dict.get("knowledge_base"),
        industry_standards=config_dict.get("industry_standards", []),
        enable_entity_extraction=config_dict.get("enable_entity_extraction", True),
        enable_knowledge_graph=config_dict.get("enable_knowledge_graph", True),
        auto_populate_core_concepts=config_dict.get("auto_populate_core_concepts", True),
    )
    
    return OntologyManager(config)
