"""
Ontology Type Definitions for PHTN.AI Sub-Agent Framework

Defines core ontology concepts aligned with phtnai-frontend:
- Agent, Role, Capability, Intent, Resource, Channel, Context, Policy
- Relationships between concepts
- Properties and attributes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class ConceptType(str, Enum):
    """Core ontology concept types (aligned with frontend OntologyExplorerTab)."""
    AGENT = "Agent"
    ROLE = "Role"
    CAPABILITY = "Capability"
    INTENT = "Intent"
    RESOURCE = "Resource"
    CHANNEL = "Channel"
    CONTEXT = "Context"
    POLICY = "Policy"
    DOMAIN = "Domain"
    SKILL = "Skill"
    TOOL = "Tool"
    WORKFLOW = "Workflow"
    EVENT = "Event"
    ACTION = "Action"


class RelationshipType(str, Enum):
    """Ontology relationship types."""
    HAS_ROLE = "hasRole"
    HAS_CAPABILITY = "hasCapability"
    HAS_SKILL = "hasSkill"
    USES_TOOL = "usesTool"
    USES_CHANNEL = "usesChannel"
    CONSTRAINED_BY = "constrainedBy"
    BELONGS_TO_DOMAIN = "belongsToDomain"
    SUPPORTS_INTENT = "supportsIntent"
    OPERATES_ON = "operatesOn"
    IS_SUBTYPE_OF = "isSubtypeOf"
    RELATED_TO = "relatedTo"
    REQUIRES_CAPABILITY = "requiresCapability"
    DELEGATES_TO = "delegatesTo"
    HAS_ATTRIBUTE = "hasAttribute"
    TRIGGERS = "triggers"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    DEPENDS_ON = "dependsOn"


@dataclass
class OntologyProperty:
    """Property definition for an ontology concept."""
    name: str
    data_type: str
    description: Optional[str] = None
    required: bool = False
    default_value: Optional[Any] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description,
            "required": self.required,
            "default_value": self.default_value,
            "constraints": self.constraints,
        }


@dataclass
class OntologyRelationship:
    """Relationship between ontology concepts."""
    relationship_type: RelationshipType
    source_concept: str
    target_concept: str
    cardinality: str = "many-to-many"
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.relationship_type.value,
            "source": self.source_concept,
            "target": self.target_concept,
            "cardinality": self.cardinality,
            "description": self.description,
            "properties": self.properties,
        }


@dataclass
class OntologyConcept:
    """
    Ontology concept definition.
    
    Aligned with phtnai-frontend OntologyExplorerTab structure:
    - id, name, type, description
    - properties (key-value pairs)
    - relationships (type, target)
    - instances (example entities)
    """
    id: str
    name: str
    concept_type: ConceptType
    description: str
    properties: Dict[str, OntologyProperty] = field(default_factory=dict)
    relationships: List[OntologyRelationship] = field(default_factory=list)
    instances: List[str] = field(default_factory=list)
    parent_concept: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.concept_type.value,
            "description": self.description,
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "relationships": [r.to_dict() for r in self.relationships],
            "instances": self.instances,
            "parent_concept": self.parent_concept,
            "metadata": self.metadata,
        }
    
    def add_property(self, prop: OntologyProperty):
        """Add a property to the concept."""
        self.properties[prop.name] = prop
    
    def add_relationship(self, rel: OntologyRelationship):
        """Add a relationship to the concept."""
        self.relationships.append(rel)
    
    def add_instance(self, instance: str):
        """Add an instance to the concept."""
        if instance not in self.instances:
            self.instances.append(instance)


@dataclass
class OntologyEntity:
    """
    An instance of an ontology concept.
    
    Represents a concrete entity in the knowledge graph.
    """
    id: str
    name: str
    concept_type: ConceptType
    concept_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "concept_type": self.concept_type.value,
            "concept_id": self.concept_id,
            "properties": self.properties,
            "relationships": self.relationships,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


CORE_CONCEPTS = {
    ConceptType.AGENT: OntologyConcept(
        id="concept-agent",
        name="Agent",
        concept_type=ConceptType.AGENT,
        description="An autonomous system that performs tasks",
        properties={
            "agentId": OntologyProperty("agentId", "UUID", "Unique agent identifier", True),
            "agentName": OntologyProperty("agentName", "String", "Human-readable name", True),
            "agentType": OntologyProperty("agentType", "Enum", "Type: SuperAgent, SubAgent, ExternalAgent"),
            "platform": OntologyProperty("platform", "String", "Deployment platform"),
            "belongsToDomain": OntologyProperty("belongsToDomain", "Domain", "Associated domain"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.HAS_ROLE, "Agent", "Role"),
            OntologyRelationship(RelationshipType.HAS_CAPABILITY, "Agent", "Capability"),
            OntologyRelationship(RelationshipType.USES_CHANNEL, "Agent", "Channel"),
            OntologyRelationship(RelationshipType.CONSTRAINED_BY, "Agent", "Policy"),
        ],
        instances=["Lead Qualifier Agent", "Ticket Classifier Agent", "Campaign Optimizer Agent"],
    ),
    ConceptType.CAPABILITY: OntologyConcept(
        id="concept-capability",
        name="Capability",
        concept_type=ConceptType.CAPABILITY,
        description="What an agent can do (functional capability)",
        properties={
            "capabilityId": OntologyProperty("capabilityId", "UUID", "Unique capability identifier", True),
            "capabilityName": OntologyProperty("capabilityName", "String", "Capability name", True),
            "description": OntologyProperty("description", "String", "Capability description"),
            "domain": OntologyProperty("domain", "Domain", "Associated domain"),
            "complexity": OntologyProperty("complexity", "Enum", "Complexity: Simple, Medium, Complex"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.SUPPORTS_INTENT, "Capability", "Intent"),
            OntologyRelationship(RelationshipType.OPERATES_ON, "Capability", "Resource"),
            OntologyRelationship(RelationshipType.IS_SUBTYPE_OF, "Capability", "Capability"),
            OntologyRelationship(RelationshipType.RELATED_TO, "Capability", "Capability"),
        ],
        instances=["LeadScoring", "TicketClassification", "CampaignOptimization", "OrderProcessing"],
    ),
    ConceptType.INTENT: OntologyConcept(
        id="concept-intent",
        name="Intent",
        concept_type=ConceptType.INTENT,
        description="User or system goal that triggers agent action",
        properties={
            "intentId": OntologyProperty("intentId", "UUID", "Unique intent identifier", True),
            "intentName": OntologyProperty("intentName", "String", "Intent name", True),
            "description": OntologyProperty("description", "String", "Intent description"),
            "domain": OntologyProperty("domain", "Domain", "Associated domain"),
            "priority": OntologyProperty("priority", "Enum", "Priority: Low, Medium, High, Critical"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.REQUIRES_CAPABILITY, "Intent", "Capability"),
            OntologyRelationship(RelationshipType.OPERATES_ON, "Intent", "Resource"),
            OntologyRelationship(RelationshipType.DELEGATES_TO, "Intent", "Intent"),
        ],
        instances=["QualifyProspect", "OpenSupportTicket", "CreateCampaign", "ProcessOrder"],
    ),
    ConceptType.RESOURCE: OntologyConcept(
        id="concept-resource",
        name="Resource",
        concept_type=ConceptType.RESOURCE,
        description="Business entity that agents operate on",
        properties={
            "resourceId": OntologyProperty("resourceId", "UUID", "Unique resource identifier", True),
            "resourceType": OntologyProperty("resourceType", "String", "Resource type", True),
            "domain": OntologyProperty("domain", "Domain", "Associated domain"),
            "schema": OntologyProperty("schema", "JSON Schema", "Resource schema"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.BELONGS_TO_DOMAIN, "Resource", "Domain"),
            OntologyRelationship(RelationshipType.HAS_ATTRIBUTE, "Resource", "Attribute"),
            OntologyRelationship(RelationshipType.RELATED_TO, "Resource", "Resource"),
        ],
        instances=["Lead", "Ticket", "Campaign", "Order", "Product"],
    ),
    ConceptType.ROLE: OntologyConcept(
        id="concept-role",
        name="Role",
        concept_type=ConceptType.ROLE,
        description="Role that an agent can assume",
        properties={
            "roleId": OntologyProperty("roleId", "UUID", "Unique role identifier", True),
            "roleName": OntologyProperty("roleName", "String", "Role name", True),
            "permissions": OntologyProperty("permissions", "Array", "Role permissions"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.HAS_CAPABILITY, "Role", "Capability"),
            OntologyRelationship(RelationshipType.CONSTRAINED_BY, "Role", "Policy"),
        ],
        instances=["SupportAgent", "SalesAgent", "AdminAgent", "AnalystAgent"],
    ),
    ConceptType.CHANNEL: OntologyConcept(
        id="concept-channel",
        name="Channel",
        concept_type=ConceptType.CHANNEL,
        description="Communication channel for agent interaction",
        properties={
            "channelId": OntologyProperty("channelId", "UUID", "Unique channel identifier", True),
            "channelType": OntologyProperty("channelType", "String", "Channel type", True),
            "protocol": OntologyProperty("protocol", "String", "Communication protocol"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.SUPPORTS_INTENT, "Channel", "Intent"),
        ],
        instances=["WebChat", "Email", "API", "Voice", "SMS"],
    ),
    ConceptType.POLICY: OntologyConcept(
        id="concept-policy",
        name="Policy",
        concept_type=ConceptType.POLICY,
        description="Constraint or rule that governs agent behavior",
        properties={
            "policyId": OntologyProperty("policyId", "UUID", "Unique policy identifier", True),
            "policyName": OntologyProperty("policyName", "String", "Policy name", True),
            "policyType": OntologyProperty("policyType", "Enum", "Type: Security, Compliance, Business"),
            "rules": OntologyProperty("rules", "Array", "Policy rules"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.CONSTRAINED_BY, "Policy", "Policy"),
        ],
        instances=["GDPRPolicy", "RateLimitPolicy", "ContentFilterPolicy", "AccessControlPolicy"],
    ),
    ConceptType.DOMAIN: OntologyConcept(
        id="concept-domain",
        name="Domain",
        concept_type=ConceptType.DOMAIN,
        description="Business domain or area of expertise",
        properties={
            "domainId": OntologyProperty("domainId", "UUID", "Unique domain identifier", True),
            "domainName": OntologyProperty("domainName", "String", "Domain name", True),
            "industry": OntologyProperty("industry", "String", "Industry sector"),
            "subDomain": OntologyProperty("subDomain", "String", "Sub-domain"),
        },
        relationships=[
            OntologyRelationship(RelationshipType.IS_SUBTYPE_OF, "Domain", "Domain"),
        ],
        instances=["CustomerSupport", "Sales", "Marketing", "Finance", "Healthcare"],
    ),
}
