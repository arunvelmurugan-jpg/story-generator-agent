"""
Observability Module for PHTN.AI Sub-Agent Framework

Provides comprehensive observability compatible with phtnai-ops-metrics:
- OpenTelemetry tracing
- Metrics collection
- Structured logging with OTEL support
- X-PHTN-Agent-ID header management
- FinOps, TechOps, AIOps logging helpers
- Content type tracking (aligned with PHTNAI_ORCHESTRATION_SCHEMA_v8)

All logs include the 18 PHTN Agent ID fields required by phtnai-ops-metrics filters.
"""

from .tracer import Tracer
from .metrics import MetricsCollector
from .header import XPHTNAgentIDHeader
from .otel_logging import (
    # Configuration
    configure_otel_logging,
    get_logger,
    
    # Context management
    set_trace_context,
    set_trace_context_from_config,
    get_trace_context,
    
    # Core logging helpers
    log_with_context,
    log_llm_call,
    log_tool_call,
    log_agent_execution,
    
    # NEW: Feature-specific logging (n8n-comparable features)
    log_text_splitter,
    log_output_parser,
    log_vector_store,
    log_evaluation_metric,
    log_trigger_event,
    log_builtin_tool,
    log_llm_provider,
    log_memory_operation,
    log_mcp_operation,
    log_guardrail_check,
    log_human_loop,
    log_ontology_operation,
    log_rag_pipeline,
    
    # Data classes
    PHtnAgentIdParts,
    
    # Core context variables
    correlation_id_var,
    phtn_agent_id_var,
    trace_id_var,
    span_id_var,
    agent_id_var,
    tenant_id_var,
    request_id_var,
    
    # Extended context variables for phtnai-ops-metrics
    tenant_group_id_var,
    domain_id_var,
    team_id_var,
    project_id_var,
    user_id_var,
    agent_type_var,
    super_agent_id_var,
    sub_agent_id_var,
    super_agent_instance_id_var,
    sub_agent_instance_id_var,
    app_name_var,
    environment_var,
    capability_id_var,
    skill_id_var,
    agent_name_var,
)

from .otel_content_types import (
    # Enums
    ContentTypeCategory,
    CONTENT_TYPE_CATEGORIES,
    
    # Attribute creators
    create_content_type_attributes,
    create_artifact_attributes,
    create_multimodal_input_attributes,
    create_delta_update_attributes,
    create_input_output_schema_attributes,
    
    # Logging helpers
    log_content_type_event,
    log_artifact_event,
    log_multimodal_input,
    log_delta_update,
    log_streaming_event,
    log_content_type_cost,
    
    # Helper functions
    extract_content_type_from_output,
    extract_content_type_from_config,
)

__all__ = [
    # Core components
    "Tracer",
    "MetricsCollector", 
    "XPHTNAgentIDHeader",
    
    # Configuration
    "configure_otel_logging",
    "get_logger",
    
    # Context management
    "set_trace_context",
    "set_trace_context_from_config",
    "get_trace_context",
    
    # Core logging helpers for phtnai-ops-metrics
    "log_with_context",
    "log_llm_call",           # FinOps: LLM cost tracking
    "log_tool_call",          # TechOps: Tool performance
    "log_agent_execution",    # AIOps: Agent quality
    
    # NEW: Feature-specific logging (n8n-comparable features)
    "log_text_splitter",      # RAG: Text chunking operations
    "log_output_parser",      # LLM: Output parsing operations
    "log_vector_store",       # RAG: Vector database operations
    "log_evaluation_metric",  # AIOps: Evaluation metrics
    "log_trigger_event",      # TechOps: Trigger invocations
    "log_builtin_tool",       # TechOps: Built-in tool usage
    "log_llm_provider",       # FinOps: Extended LLM provider tracking
    "log_memory_operation",   # TechOps: Memory operations
    "log_mcp_operation",      # TechOps: MCP server operations
    "log_guardrail_check",    # AIOps: Safety guardrail checks
    "log_human_loop",         # AIOps: Human-in-the-loop events
    "log_ontology_operation", # AIOps: Knowledge graph operations
    "log_rag_pipeline",       # RAG: Full pipeline operations
    
    # Data classes
    "PHtnAgentIdParts",
    
    # Core context variables
    "correlation_id_var",
    "phtn_agent_id_var",
    "trace_id_var",
    "span_id_var",
    "agent_id_var",
    "tenant_id_var",
    "request_id_var",
    
    # Extended context variables for phtnai-ops-metrics
    "tenant_group_id_var",
    "domain_id_var",
    "team_id_var",
    "project_id_var",
    "user_id_var",
    "agent_type_var",
    "super_agent_id_var",
    "sub_agent_id_var",
    "super_agent_instance_id_var",
    "sub_agent_instance_id_var",
    "app_name_var",
    "environment_var",
    "capability_id_var",
    "skill_id_var",
    "agent_name_var",
    
    # Content type tracking (aligned with PHTNAI_ORCHESTRATION_SCHEMA_v8)
    "ContentTypeCategory",
    "CONTENT_TYPE_CATEGORIES",
    "create_content_type_attributes",
    "create_artifact_attributes",
    "create_multimodal_input_attributes",
    "create_delta_update_attributes",
    "create_input_output_schema_attributes",
    "log_content_type_event",
    "log_artifact_event",
    "log_multimodal_input",
    "log_delta_update",
    "log_streaming_event",
    "log_content_type_cost",
    "extract_content_type_from_output",
    "extract_content_type_from_config",
]
