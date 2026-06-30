"""
Entity Extractor for PHTN.AI Sub-Agent Framework

Extracts entities from text for knowledge graph population:
- Named Entity Recognition (NER)
- Relationship extraction
- Concept mapping
- Entity linking
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .types import ConceptType, RelationshipType

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity types for extraction."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    AGENT = "AGENT"
    CAPABILITY = "CAPABILITY"
    INTENT = "INTENT"
    RESOURCE = "RESOURCE"


@dataclass
class ExtractedEntity:
    """An extracted entity from text."""
    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    normalized_value: Optional[str] = None
    concept_type: Optional[ConceptType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "normalized_value": self.normalized_value,
            "concept_type": self.concept_type.value if self.concept_type else None,
            "metadata": self.metadata,
        }


@dataclass
class ExtractedRelationship:
    """An extracted relationship between entities."""
    subject: ExtractedEntity
    predicate: str
    object_entity: ExtractedEntity
    confidence: float = 1.0
    relationship_type: Optional[RelationshipType] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "object": self.object_entity.to_dict(),
            "confidence": self.confidence,
            "relationship_type": self.relationship_type.value if self.relationship_type else None,
        }


@dataclass
class EntityExtractionConfig:
    """Configuration for entity extraction."""
    enabled: bool = True
    extract_persons: bool = True
    extract_organizations: bool = True
    extract_locations: bool = True
    extract_dates: bool = True
    extract_concepts: bool = True
    extract_relationships: bool = True
    min_confidence: float = 0.5
    use_llm: bool = False
    custom_patterns: Dict[str, str] = field(default_factory=dict)


class EntityExtractor:
    """
    Extracts entities and relationships from text.
    
    Uses pattern-based extraction with optional LLM enhancement.
    """
    
    PATTERNS = {
        EntityType.PERSON: [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        ],
        EntityType.ORGANIZATION: [
            r"\b(?:[A-Z][a-z]*\.?\s*)+(?:Inc|Corp|LLC|Ltd|Company|Co|Group|Foundation|Institute|University|Bank)\b",
            r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b",
        ],
        EntityType.LOCATION: [
            r"\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ],
        EntityType.DATE: [
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",
        ],
        EntityType.TIME: [
            r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b",
        ],
        EntityType.MONEY: [
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b",
            r"\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|EUR|GBP)\b",
        ],
        EntityType.PERCENT: [
            r"\b\d+(?:\.\d+)?%\b",
            r"\b\d+(?:\.\d+)?\s*percent\b",
        ],
        EntityType.PRODUCT: [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Pro|Plus|Max|Ultra|Enterprise|Premium)\b",
        ],
    }
    
    AGENT_PATTERNS = {
        EntityType.AGENT: [
            r"\b\w+\s+(?:Agent|Bot|Assistant)\b",
            r"\bagent[_-]?\w+\b",
        ],
        EntityType.CAPABILITY: [
            r"\b(?:can|able to|capability to)\s+(\w+(?:\s+\w+)*)\b",
        ],
        EntityType.INTENT: [
            r"\b(?:want to|need to|trying to|intend to)\s+(\w+(?:\s+\w+)*)\b",
        ],
    }
    
    RELATIONSHIP_PATTERNS = [
        (r"(\w+)\s+(?:is|are)\s+(?:a|an)\s+(\w+)", "is_a"),
        (r"(\w+)\s+(?:has|have)\s+(\w+)", "has"),
        (r"(\w+)\s+(?:uses?|utilizes?)\s+(\w+)", "uses"),
        (r"(\w+)\s+(?:belongs?\s+to|is\s+part\s+of)\s+(\w+)", "belongs_to"),
        (r"(\w+)\s+(?:creates?|produces?|generates?)\s+(\w+)", "produces"),
        (r"(\w+)\s+(?:requires?|needs?|depends?\s+on)\s+(\w+)", "requires"),
    ]
    
    def __init__(self, config: Optional[EntityExtractionConfig] = None):
        """
        Initialize entity extractor.
        
        Args:
            config: Extraction configuration
        """
        self.config = config or EntityExtractionConfig()
        self._compiled_patterns: Dict[EntityType, List[re.Pattern]] = {}
        self._compile_patterns()
        
        logger.debug("EntityExtractor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        for entity_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        for entity_type, patterns in self.AGENT_PATTERNS.items():
            self._compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        for pattern_str, custom_type in self.config.custom_patterns.items():
            try:
                entity_type = EntityType(custom_type)
                if entity_type not in self._compiled_patterns:
                    self._compiled_patterns[entity_type] = []
                self._compiled_patterns[entity_type].append(
                    re.compile(pattern_str, re.IGNORECASE)
                )
            except ValueError:
                logger.warning(f"Unknown entity type: {custom_type}")
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not self.config.enabled:
            return []
        
        entities = []
        seen_spans = set()
        
        for entity_type, patterns in self._compiled_patterns.items():
            if not self._should_extract_type(entity_type):
                continue
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())
                    
                    if self._overlaps_with_existing(span, seen_spans):
                        continue
                    
                    seen_spans.add(span)
                    
                    entity = ExtractedEntity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8,
                        concept_type=self._map_to_concept_type(entity_type),
                    )
                    entities.append(entity)
        
        entities.sort(key=lambda e: e.start_pos)
        
        return entities
    
    def extract_relationships(
        self,
        text: str,
        entities: Optional[List[ExtractedEntity]] = None,
    ) -> List[ExtractedRelationship]:
        """
        Extract relationships from text.
        
        Args:
            text: Input text
            entities: Optional pre-extracted entities
            
        Returns:
            List of extracted relationships
        """
        if not self.config.extract_relationships:
            return []
        
        if entities is None:
            entities = self.extract_entities(text)
        
        relationships = []
        
        for pattern_str, rel_type in self.RELATIONSHIP_PATTERNS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            
            for match in pattern.finditer(text):
                subject_text = match.group(1)
                object_text = match.group(2)
                
                subject_entity = self._find_or_create_entity(
                    subject_text, match.start(1), match.end(1), entities
                )
                object_entity = self._find_or_create_entity(
                    object_text, match.start(2), match.end(2), entities
                )
                
                relationship = ExtractedRelationship(
                    subject=subject_entity,
                    predicate=rel_type,
                    object_entity=object_entity,
                    confidence=0.7,
                    relationship_type=self._map_to_relationship_type(rel_type),
                )
                relationships.append(relationship)
        
        return relationships
    
    def extract_all(self, text: str) -> Dict[str, Any]:
        """
        Extract all entities and relationships.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entities and relationships
        """
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities)
        
        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "entity_count": len(entities),
            "relationship_count": len(relationships),
        }
    
    def _should_extract_type(self, entity_type: EntityType) -> bool:
        """Check if entity type should be extracted based on config."""
        type_config_map = {
            EntityType.PERSON: self.config.extract_persons,
            EntityType.ORGANIZATION: self.config.extract_organizations,
            EntityType.LOCATION: self.config.extract_locations,
            EntityType.DATE: self.config.extract_dates,
            EntityType.TIME: self.config.extract_dates,
            EntityType.CONCEPT: self.config.extract_concepts,
            EntityType.AGENT: self.config.extract_concepts,
            EntityType.CAPABILITY: self.config.extract_concepts,
            EntityType.INTENT: self.config.extract_concepts,
        }
        return type_config_map.get(entity_type, True)
    
    def _overlaps_with_existing(
        self,
        span: Tuple[int, int],
        existing: set,
    ) -> bool:
        """Check if span overlaps with existing spans."""
        start, end = span
        for ex_start, ex_end in existing:
            if not (end <= ex_start or start >= ex_end):
                return True
        return False
    
    def _map_to_concept_type(self, entity_type: EntityType) -> Optional[ConceptType]:
        """Map entity type to ontology concept type."""
        mapping = {
            EntityType.AGENT: ConceptType.AGENT,
            EntityType.CAPABILITY: ConceptType.CAPABILITY,
            EntityType.INTENT: ConceptType.INTENT,
            EntityType.RESOURCE: ConceptType.RESOURCE,
            EntityType.ORGANIZATION: ConceptType.DOMAIN,
        }
        return mapping.get(entity_type)
    
    def _map_to_relationship_type(self, rel_type: str) -> Optional[RelationshipType]:
        """Map extracted relationship to ontology relationship type."""
        mapping = {
            "is_a": RelationshipType.IS_SUBTYPE_OF,
            "has": RelationshipType.HAS_CAPABILITY,
            "uses": RelationshipType.USES_TOOL,
            "belongs_to": RelationshipType.BELONGS_TO_DOMAIN,
            "produces": RelationshipType.PRODUCES,
            "requires": RelationshipType.DEPENDS_ON,
        }
        return mapping.get(rel_type)
    
    def _find_or_create_entity(
        self,
        text: str,
        start: int,
        end: int,
        entities: List[ExtractedEntity],
    ) -> ExtractedEntity:
        """Find existing entity or create new one."""
        for entity in entities:
            if entity.start_pos <= start and entity.end_pos >= end:
                return entity
            if entity.text.lower() == text.lower():
                return entity
        
        return ExtractedEntity(
            text=text,
            entity_type=EntityType.CONCEPT,
            start_pos=start,
            end_pos=end,
            confidence=0.6,
        )


def create_entity_extractor(
    config_dict: Optional[Dict[str, Any]] = None,
) -> EntityExtractor:
    """Create entity extractor from config dict."""
    if config_dict is None:
        config_dict = {}
    
    config = EntityExtractionConfig(
        enabled=config_dict.get("enabled", True),
        extract_persons=config_dict.get("extract_persons", True),
        extract_organizations=config_dict.get("extract_organizations", True),
        extract_locations=config_dict.get("extract_locations", True),
        extract_dates=config_dict.get("extract_dates", True),
        extract_concepts=config_dict.get("extract_concepts", True),
        extract_relationships=config_dict.get("extract_relationships", True),
        min_confidence=config_dict.get("min_confidence", 0.5),
        use_llm=config_dict.get("use_llm", False),
        custom_patterns=config_dict.get("custom_patterns", {}),
    )
    
    return EntityExtractor(config)
