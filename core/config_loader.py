"""
Configuration Loader for PHTN.AI Sub-Agent Framework

Loads and validates agent configuration from .phtnai/PHTN-AGENT.json
Supports PHTN-AGENT-SCHEMA_v2 with all enterprise features.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, ConfigDict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionPattern(str, Enum):
    """Agent execution patterns."""
    SIMPLE = "SIMPLE"
    REACT = "REACT"
    COT = "COT"
    TOOL_USE = "TOOL_USE"
    RAG = "RAG"
    PLAN_EXECUTE = "PLAN_EXECUTE"
    CUSTOM = "CUSTOM"


class ContentType(str, Enum):
    """Standardized content types (32 types aligned with orchestration schema v8)."""
    TEXT = "TEXT"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    NULL = "NULL"
    JSON_OBJECT = "JSON_OBJECT"
    JSON_ARRAY = "JSON_ARRAY"
    TABLE = "TABLE"
    SCHEMA_INSTANCE = "SCHEMA_INSTANCE"
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    PDF = "PDF"
    EXCEL = "EXCEL"
    CHART = "CHART"
    CODE = "CODE"
    BINARY_BLOB = "BINARY_BLOB"
    FILE_REFERENCE = "FILE_REFERENCE"
    ARTIFACT = "ARTIFACT"
    ATTACHMENT = "ATTACHMENT"
    STREAM_CHUNK = "STREAM_CHUNK"
    SSE_EVENT = "SSE_EVENT"
    DELTA = "DELTA"
    ENTITY_REFERENCE = "ENTITY_REFERENCE"
    AGENT_REFERENCE = "AGENT_REFERENCE"
    RESOURCE_LOCATOR = "RESOURCE_LOCATOR"
    URI = "URI"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ERROR = "ERROR"
    METADATA = "METADATA"
    HUMAN_INPUT_REQUEST = "HUMAN_INPUT_REQUEST"
    HUMAN_APPROVAL_REQUEST = "HUMAN_APPROVAL_REQUEST"


class StatusStage(str, Enum):
    """Agent lifecycle stages."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentSchema(BaseModel):
    """Content schema definition."""
    contentType: ContentType = ContentType.JSON_OBJECT
    mimeType: str = "application/json"
    encoding: str = "utf-8"
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    maxSize: str = "10MB"
    multimodal: Optional[Dict[str, Any]] = None
    streaming: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True
        extra = "allow"


class CapabilitySkillRef(BaseModel):
    """Reference to a skill within a capability (simplified format for capabilities array)."""
    id: Optional[str] = Field(default=None, alias="skill_id")
    name: Optional[str] = None
    priority: int = 1
    required: bool = False
    
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    @model_validator(mode="before")
    @classmethod
    def handle_skill_id(cls, values):
        if isinstance(values, dict):
            if "skill_id" in values and "id" not in values:
                values["id"] = values["skill_id"]
            if "id" in values and "name" not in values:
                values["name"] = values.get("id", "")
        return values


class SkillDefinition(BaseModel):
    """Agent skill definition."""
    id: str = Field(alias="skill_id")
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    inputSchema: Optional[Dict[str, Any]] = None
    outputSchema: Optional[Dict[str, Any]] = None
    executionPattern: Optional[ExecutionPattern] = None
    tools: List[str] = Field(default_factory=list)
    promptTemplate: Optional[str] = None
    enabled: bool = True
    
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    @model_validator(mode="before")
    @classmethod
    def handle_skill_id(cls, values):
        if isinstance(values, dict):
            if "skill_id" in values and "id" not in values:
                values["id"] = values["skill_id"]
        return values


class CapabilityDefinition(BaseModel):
    """Agent capability definition."""
    id: str = Field(alias="capability_id")
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    skills: List[CapabilitySkillRef] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    inputSchema: Optional[Dict[str, Any]] = None
    outputSchema: Optional[Dict[str, Any]] = None
    routing_hints: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    enabled: bool = True
    accessControl: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    
    @model_validator(mode="before")
    @classmethod
    def handle_capability_id(cls, values):
        if isinstance(values, dict):
            if "capability_id" in values and "id" not in values:
                values["id"] = values["capability_id"]
        return values


class ModelParameters(BaseModel):
    """LLM model parameters."""
    temperature: float = 0.7
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop_sequences: List[str] = Field(default_factory=list)


class ModelRouting(BaseModel):
    """Model routing configuration."""
    strategy: str = "static"
    confidence_threshold: Optional[float] = None
    fallback_triggers: Dict[str, bool] = Field(default_factory=lambda: {
        "timeout": True,
        "error": True,
        "schema_failure": True,
        "low_confidence": True
    })
    max_retries: int = 3
    retry_backoff_ms: int = 1000
    traffic_split: Dict[str, float] = Field(default_factory=dict)
    circuit_breaker: Optional[Dict[str, Any]] = None


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration."""
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    api_version: str = "2024-02-15-preview"


class VisionConfig(BaseModel):
    """Vision/Image input configuration."""
    enabled: bool = False
    detail: str = "auto"
    max_images_per_request: int = 5
    supported_formats: List[str] = Field(default_factory=lambda: ["png", "jpeg", "jpg", "gif", "webp"])
    max_image_size_mb: float = 5.0


class AudioConfig(BaseModel):
    """Audio input/output configuration."""
    enabled: bool = False
    input_formats: List[str] = Field(default_factory=lambda: ["wav", "mp3"])
    output_format: str = "wav"
    voice: str = "alloy"
    max_audio_duration_seconds: int = 60


class StructuredOutputSchema(BaseModel):
    """Structured output schema definition."""
    name: str
    description: Optional[str] = None
    schema_: Dict[str, Any] = Field(alias="schema", default_factory=dict)
    strict: bool = True
    
    model_config = ConfigDict(populate_by_name=True)


class StructuredOutputConfig(BaseModel):
    """Structured output configuration."""
    enabled: bool = False
    strict: bool = True
    default_schema: Optional[Dict[str, Any]] = None
    schemas: Dict[str, StructuredOutputSchema] = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    """Embeddings API configuration."""
    enabled: bool = False
    model: str = "text-embedding-3-small"
    dimensions: Optional[int] = None
    batch_size: int = 100


class ModerationConfig(BaseModel):
    """OpenAI Moderation API configuration."""
    enabled: bool = False
    model: str = "text-moderation-latest"
    check_input: bool = True
    check_output: bool = False
    block_on_violation: bool = True
    categories_to_check: Optional[List[str]] = None
    threshold_overrides: Dict[str, float] = Field(default_factory=dict)


class StreamingConfig(BaseModel):
    """Streaming response configuration."""
    enabled: bool = True
    include_usage: bool = True
    chunk_size: int = 1


class ReasoningModelsConfig(BaseModel):
    """Configuration for o1/reasoning models."""
    enabled: bool = False
    max_completion_tokens: Optional[int] = None
    reasoning_effort: str = "medium"


class ModelConfig(BaseModel):
    """LLM model configuration."""
    primary_model: str
    provider: Optional[str] = None
    version: Optional[str] = None
    fallback_models: List[str] = Field(default_factory=list)
    parameters: ModelParameters = Field(default_factory=ModelParameters)
    routing: Optional[ModelRouting] = None
    provider_failover: Optional[Dict[str, Any]] = None
    determinism: Optional[Dict[str, Any]] = None
    sla: Optional[Dict[str, Any]] = None
    azure_config: Optional[AzureOpenAIConfig] = None
    vision: Optional[VisionConfig] = None
    audio: Optional[AudioConfig] = None
    structured_output: Optional[StructuredOutputConfig] = None
    embeddings: Optional[EmbeddingsConfig] = None
    moderation: Optional[ModerationConfig] = None
    streaming: Optional[StreamingConfig] = None
    reasoning_models: Optional[ReasoningModelsConfig] = None


class ToolDefinition(BaseModel):
    """Tool definition."""
    tool_id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    type: str = "BUILTIN"
    interface: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    mcp: Optional[Dict[str, Any]] = None
    http: Optional[Dict[str, Any]] = None
    access_control: Optional[Dict[str, Any]] = None
    audit: Optional[Dict[str, Any]] = None
    sandboxing: Optional[Dict[str, Any]] = None


class ContextCompressionConfig(BaseModel):
    """Context compression configuration."""
    enabled: bool = True
    strategy: str = "importance_weighted"
    max_tokens: int = 4000
    target_compression_ratio: float = 0.5
    preserve_recent_messages: int = 3
    preserve_system_prompt: bool = True
    importance_threshold: float = 0.3
    enable_semantic_dedup: bool = True
    dedup_similarity_threshold: float = 0.9


class RerankerConfig(BaseModel):
    """RAG reranker configuration."""
    enabled: bool = True
    provider: str = "bm25"
    model: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.0
    use_query_expansion: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    cohere_api_key: Optional[str] = None


class MemoryConfig(BaseModel):
    """Memory management configuration."""
    short_term: Optional[Dict[str, Any]] = None
    long_term: Optional[Dict[str, Any]] = None
    semantic: Optional[Dict[str, Any]] = None
    episodic: Optional[Dict[str, Any]] = None
    token_budget: Optional[int] = None
    compression: Optional[Dict[str, Any]] = None
    context_compression: Optional[Union[str, ContextCompressionConfig]] = None
    reranker: Optional[RerankerConfig] = None
    retention_policy: Optional[Dict[str, Any]] = None


class ContextConfig(BaseModel):
    """Context building configuration."""
    builder: str = "standard"
    compression: str = "none"
    token_budgeting: bool = True
    context_window_management: Optional[Dict[str, Any]] = None
    retrieval_augmentation: Optional[Dict[str, Any]] = None
    few_shot_learning: Optional[Dict[str, Any]] = None


class PromptConfig(BaseModel):
    """Prompt configuration."""
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    system_prompt: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    environment_overrides: Dict[str, Any] = Field(default_factory=dict)
    variables: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)


class InputGuardrails(BaseModel):
    """Input guardrails configuration."""
    pii_detection: Optional[Dict[str, Any]] = None
    phi_masking: Optional[Dict[str, Any]] = None
    prompt_injection_detection: Optional[Dict[str, Any]] = None
    jailbreak_detection: Optional[Dict[str, Any]] = None
    content_filtering: Optional[Dict[str, Any]] = None
    max_input_length: Optional[int] = None


class OutputGuardrails(BaseModel):
    """Output guardrails configuration."""
    toxicity_filtering: Optional[Dict[str, Any]] = None
    hallucination_detection: Optional[Dict[str, Any]] = None
    bias_detection: Optional[Dict[str, Any]] = None
    pii_leakage_prevention: Optional[Dict[str, Any]] = None
    schema_validation: Optional[Dict[str, Any]] = None
    required_disclaimers: List[str] = Field(default_factory=list)
    forbidden_phrases: List[str] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    """Guardrails configuration."""
    policies: List[str] = Field(default_factory=list)
    input_guardrails: Optional[InputGuardrails] = None
    output_guardrails: Optional[OutputGuardrails] = None
    tool_guardrails: Optional[Dict[str, Any]] = None
    custom_rules: List[Dict[str, Any]] = Field(default_factory=list)
    enforcement_mode: str = "enforce"
    fail_on_guardrail_error: bool = True


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""
    enabled: bool = True
    strategy: str = "token_bucket"
    requests_per_minute: int = 100
    burst_size: int = 20
    window_size_seconds: int = 60
    per_tenant: bool = True
    per_agent: bool = True
    per_endpoint: bool = False


class SecurityConfig(BaseModel):
    """Security configuration."""
    rbac: Optional[Dict[str, Any]] = None
    abac: Optional[Dict[str, Any]] = None
    authentication: Optional[Dict[str, Any]] = None
    rate_limiting: Optional[RateLimitingConfig] = None
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    secrets_management: Optional[Dict[str, Any]] = None
    tool_sandboxing: Optional[Dict[str, Any]] = None
    zero_trust: bool = True
    audit_logging: Optional[Dict[str, Any]] = None
    agent_interaction_policy: Optional[Dict[str, Any]] = None


class ObservabilityConfig(BaseModel):
    """Observability configuration."""
    tracing: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    phtn_agent_id: Optional[Dict[str, Any]] = None
    x_phtn_agent_id: Optional[Dict[str, Any]] = None
    lineage_tracking: Optional[Dict[str, Any]] = None
    evaluation_hooks: Optional[Union[bool, Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"


class CostConfig(BaseModel):
    """Cost governance configuration."""
    token_usage_tracking: bool = True
    per_request_budget: Optional[float] = None
    per_agent_cost_cap: Optional[float] = None
    per_tenant_budget: Optional[float] = None
    real_time_budget_breaker: bool = False
    model_downgrade_strategy: str = "none"
    pricing: Dict[str, Any] = Field(default_factory=dict)
    alerts: Optional[Dict[str, Any]] = None


class ExecutionConfig(BaseModel):
    """Execution engine configuration."""
    pattern: ExecutionPattern = ExecutionPattern.SIMPLE
    max_iterations: int = 10
    timeout_seconds: int = 300
    state_machine: Optional[Dict[str, Any]] = None
    react_config: Optional[Dict[str, Any]] = None
    cot_config: Optional[Dict[str, Any]] = None
    rag_config: Optional[Dict[str, Any]] = None
    plan_execute_config: Optional[Dict[str, Any]] = None
    custom_config: Optional[Dict[str, Any]] = None


class A2AConfig(BaseModel):
    """Agent-to-Agent protocol configuration."""
    enabled: bool = True
    protocol_version: str = "1.0"
    endpoint: Optional[Dict[str, Any]] = None
    capabilities_advertised: List[str] = Field(default_factory=list)
    skills_advertised: List[str] = Field(default_factory=list)
    discovery: Optional[Dict[str, Any]] = None
    message_format: str = "JSON"


class HealthCheckConfig(BaseModel):
    """Health check configuration."""
    enabled: bool = True
    endpoint: str = "/health"
    liveness: Optional[Dict[str, Any]] = None
    readiness: Optional[Dict[str, Any]] = None
    startup: Optional[Dict[str, Any]] = None
    dependencies: List[Dict[str, Any]] = Field(default_factory=list)


class MCPConfig(BaseModel):
    """Model Context Protocol configuration."""
    enabled: bool = False
    protocol_version: str = "1.0"
    role: str = "client"
    client: Optional[Dict[str, Any]] = None
    server: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None


class ResilienceConfig(BaseModel):
    """Resilience patterns configuration."""
    circuit_breaker: Optional[Dict[str, Any]] = None
    retry: Optional[Dict[str, Any]] = None
    rate_limiter: Optional[Dict[str, Any]] = None
    timeout: Optional[Dict[str, Any]] = None
    fallback: Optional[Dict[str, Any]] = None
    bulkhead: Optional[Dict[str, Any]] = None


class TerminologyConfig(BaseModel):
    """Terminology configuration for ontology."""
    version: str = "1.0"
    standard: Optional[str] = None
    custom_terms: Dict[str, str] = Field(default_factory=dict)
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)


class KnowledgeGraphConfig(BaseModel):
    """Knowledge graph configuration."""
    enabled: bool = True
    max_nodes: int = 10000
    max_edges: int = 50000
    enable_inference: bool = True
    export_formats: List[str] = Field(default_factory=lambda: ["json-ld"])


class EntityExtractionConfig(BaseModel):
    """Entity extraction configuration."""
    enabled: bool = True
    extract_persons: bool = True
    extract_organizations: bool = True
    extract_locations: bool = True
    extract_dates: bool = True
    extract_concepts: bool = True
    extract_relationships: bool = True
    min_confidence: float = 0.5
    use_llm: bool = False
    custom_patterns: Dict[str, str] = Field(default_factory=dict)


class OntologyConfig(BaseModel):
    """Ontology and knowledge graph configuration."""
    enabled: bool = True
    terminology: Optional[TerminologyConfig] = None
    knowledge_base: Optional[str] = None
    industry_standards: List[str] = Field(default_factory=list)
    knowledge_graph: Optional[KnowledgeGraphConfig] = None
    entity_extraction: Optional[EntityExtractionConfig] = None
    concepts: List[str] = Field(default_factory=lambda: ["Agent", "Capability", "Intent", "Resource"])
    auto_populate_core_concepts: bool = True


class DomainConfig(BaseModel):
    """Domain and industry context."""
    industry: Optional[str] = None
    sub_domain: Optional[str] = None
    use_cases: List[str] = Field(default_factory=list)
    compliance: List[str] = Field(default_factory=list)
    regulations: List[str] = Field(default_factory=list)
    terminology: Dict[str, str] = Field(default_factory=dict)
    constraints: Optional[Dict[str, Any]] = None
    target_audience: List[str] = Field(default_factory=list)
    ontology: Optional[OntologyConfig] = None


class DeploymentConfig(BaseModel):
    """Deployment configuration."""
    environment: Optional[str] = None
    region: Optional[str] = None
    deployment_strategy: str = "rolling"
    replicas: int = 1
    autoscaling: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    kubernetes: Optional[Dict[str, Any]] = None


class LifecycleConfig(BaseModel):
    """Lifecycle management configuration."""
    current_stage: Optional[StatusStage] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    rollback: Optional[Dict[str, Any]] = None
    versioning: Optional[Dict[str, Any]] = None


class RiskConfig(BaseModel):
    """Risk classification configuration."""
    level: Optional[RiskLevel] = None
    mitigation: Optional[str] = None
    assessment_history: List[Dict[str, Any]] = Field(default_factory=list)


class RegulatoryConfig(BaseModel):
    """Regulatory compliance configuration."""
    gdpr: bool = False
    eu_ai_act: bool = False
    hipaa: bool = False
    sox: bool = False
    dpdp_india: bool = False
    data_residency: Optional[str] = None
    consent_management: bool = False
    human_oversight: bool = False
    explainability: Optional[Union[bool, Dict[str, Any]]] = None
    explainability_config: Optional[Dict[str, Any]] = None
    data_privacy: Optional[Dict[str, Any]] = None
    audit_logging: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"


class EvaluationConfig(BaseModel):
    """Evaluation metrics configuration."""
    metrics: Optional[Dict[str, bool]] = None
    offline_evaluation: Optional[Dict[str, Any]] = None
    human_review: Optional[Dict[str, Any]] = None


class IntegrationsConfig(BaseModel):
    """Framework integrations configuration."""
    langchain: Optional[Dict[str, Any]] = None
    llamaindex: Optional[Dict[str, Any]] = None
    langgraph: Optional[Dict[str, Any]] = None
    autogen: Optional[Dict[str, Any]] = None
    crewai: Optional[Dict[str, Any]] = None


class AgentMetadata(BaseModel):
    """Agent metadata."""
    created_at: Optional[str] = None
    created_by: Optional[str] = None
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)


class AgentConfiguration(BaseModel):
    """Complete agent configuration (PHTN-AGENT.json)."""
    agent_id: str
    instance_id: Optional[str] = None
    phtn_agent_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    owner: Optional[str] = None
    tenant: Optional[str] = None
    status: Optional[StatusStage] = StatusStage.DRAFT
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[AgentMetadata] = None
    domain: Optional[DomainConfig] = None
    execution_config: Optional[ExecutionConfig] = None
    execution_profile: Optional[Dict[str, Any]] = None
    capabilities: List[CapabilityDefinition] = Field(default_factory=list)
    input_schema: Optional[ContentSchema] = None
    output_schema: Optional[ContentSchema] = None
    llm_config: Optional[ModelConfig] = Field(default=None, alias="model_config")
    tools: List[ToolDefinition] = Field(default_factory=list)
    skills: List[Dict[str, Any]] = Field(default_factory=list)
    memory_config: Optional[MemoryConfig] = None
    context_strategy: Optional[ContextConfig] = None
    prompt_config: Optional[PromptConfig] = None
    guardrails: Optional[GuardrailsConfig] = None
    security: Optional[SecurityConfig] = None
    cost_governance: Optional[CostConfig] = None
    observability_config: Optional[ObservabilityConfig] = None
    regulatory_profile: Optional[RegulatoryConfig] = None
    risk_classification: Optional[RiskConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    lifecycle: Optional[LifecycleConfig] = None
    deployment_metadata: Optional[DeploymentConfig] = None
    a2a_protocol: Optional[A2AConfig] = None
    health_check: Optional[HealthCheckConfig] = None
    mcp: Optional[MCPConfig] = None
    resilience_config: Optional[ResilienceConfig] = None
    integrations: Optional[IntegrationsConfig] = None
    runtime_limits: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"
        populate_by_name = True


class ConfigLoader:
    """
    Loads and validates PHTN.AI agent configuration.
    
    Supports PHTN-AGENT-SCHEMA_v2 with all enterprise features including:
    - Multiple execution patterns (SIMPLE, REACT, COT, TOOL_USE, RAG, PLAN_EXECUTE)
    - 32 standardized I/O content types
    - A2A protocol for agent communication
    - MCP compatibility
    - Multi-provider LLM routing
    - Comprehensive guardrails
    - RBAC/ABAC security
    - Full observability with X-PHTN-Agent-ID header
    """
    
    def __init__(self, phtnai_dir: Optional[Path] = None):
        """
        Initialize ConfigLoader.
        
        Args:
            phtnai_dir: Path to .phtnai directory (defaults to .phtnai in current dir)
        """
        if phtnai_dir is None:
            current_file = Path(__file__)
            framework_root = current_file.parent.parent
            phtnai_dir = framework_root / ".phtnai"
        
        self.phtnai_dir = Path(phtnai_dir)
        self.agent_config_path = self.phtnai_dir / "PHTN-AGENT.json"
        self.schema_path = self.phtnai_dir / "PHTN-AGENT-SCHEMA_v2.json"
        
        logger.info(f"ConfigLoader initialized with phtnai_dir: {self.phtnai_dir}")
    
    def load_agent_config(self, config_path: Optional[Path] = None) -> AgentConfiguration:
        """
        Load and validate agent configuration.
        
        Args:
            config_path: Optional path to agent config file (defaults to PHTN-AGENT.json)
            
        Returns:
            AgentConfiguration: Validated agent configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
        """
        path = config_path or self.agent_config_path
        
        if not path.exists():
            raise FileNotFoundError(
                f"Agent configuration not found at: {path}\n"
                f"Please ensure .phtnai/PHTN-AGENT.json exists"
            )
        
        logger.info(f"Loading agent configuration from: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            config = AgentConfiguration(**config_data)
            
            logger.info(f"✅ Agent configuration loaded successfully")
            logger.info(f"   Agent ID: {config.agent_id}")
            logger.info(f"   Name: {config.name}")
            logger.info(f"   Version: {config.version}")
            logger.info(f"   Execution Pattern: {config.execution_config.pattern.value}")
            logger.info(f"   Capabilities: {len(config.capabilities)}")
            logger.info(f"   Tools: {len(config.tools)}")
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in agent config: {e}")
            raise
        except ValidationError as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading agent configuration: {e}")
            raise
    
    def load_schema(self) -> Dict[str, Any]:
        """Load the PHTN-AGENT-SCHEMA_v2.json schema."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found at: {self.schema_path}")
        
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_capabilities(self) -> List[CapabilityDefinition]:
        """Get list of agent capabilities."""
        config = self.load_agent_config()
        return config.capabilities
    
    def get_capability_by_id(self, capability_id: str) -> Optional[CapabilityDefinition]:
        """Get capability by ID."""
        for cap in self.get_capabilities():
            if cap.id == capability_id:
                return cap
        return None
    
    def get_skill_by_id(self, capability_id: str, skill_id: str) -> Optional[SkillDefinition]:
        """Get skill by capability and skill ID."""
        capability = self.get_capability_by_id(capability_id)
        if capability:
            for skill in capability.skills:
                if skill.id == skill_id:
                    return skill
        return None
    
    def get_tools(self) -> List[ToolDefinition]:
        """Get list of configured tools."""
        config = self.load_agent_config()
        return config.tools
    
    def get_tool_by_id(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool by ID."""
        for tool in self.get_tools():
            if tool.tool_id == tool_id:
                return tool
        return None
    
    def get_model_config(self) -> Optional[ModelConfig]:
        """Get model configuration."""
        config = self.load_agent_config()
        return config.llm_config
    
    def get_execution_config(self) -> ExecutionConfig:
        """Get execution configuration."""
        config = self.load_agent_config()
        return config.execution_config
    
    def get_guardrails(self) -> GuardrailsConfig:
        """Get guardrails configuration."""
        config = self.load_agent_config()
        return config.guardrails
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        config = self.load_agent_config()
        return config.security
    
    def get_observability_config(self) -> ObservabilityConfig:
        """Get observability configuration."""
        config = self.load_agent_config()
        return config.observability_config
    
    def export_config_summary(self) -> Dict[str, Any]:
        """Export configuration summary for debugging/logging."""
        config = self.load_agent_config()
        
        return {
            "agent_id": config.agent_id,
            "name": config.name,
            "version": config.version,
            "status": config.status.value,
            "execution_pattern": config.execution_config.pattern.value,
            "primary_model": config.llm_config.primary_model if config.llm_config else None,
            "provider": config.llm_config.provider if config.llm_config else None,
            "capabilities": [
                {
                    "id": cap.id,
                    "name": cap.name,
                    "skills": [s.id for s in cap.skills]
                }
                for cap in config.capabilities
            ],
            "tools": [t.tool_id for t in config.tools],
            "guardrails_mode": config.guardrails.enforcement_mode,
            "loaded_at": datetime.utcnow().isoformat()
        }


_config_loader: Optional[ConfigLoader] = None


def get_config_loader(phtnai_dir: Optional[Path] = None) -> ConfigLoader:
    """Get global ConfigLoader instance (singleton pattern)."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(phtnai_dir)
    return _config_loader


def load_agent_config(phtnai_dir: Optional[Path] = None) -> AgentConfiguration:
    """Convenience function to load agent configuration."""
    loader = get_config_loader(phtnai_dir)
    return loader.load_agent_config()
