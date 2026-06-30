"""
PHTN.AI Enterprise Sub-Agent Framework - Core Module

This module provides the core components for building enterprise-grade AI agents:
- Agent: Main agent class with execution patterns
- ConfigLoader: Load and validate PHTN-AGENT.json configurations
- SchemaValidator: Validate configurations against PHTN-AGENT-SCHEMA_v2
- ContentTypes: 34 standardized content types aligned with PHTNAI_ORCHESTRATION_SCHEMA_v8
- ContentProcessor: Full content type processing for input/output
"""

from .agent import Agent, AgentConfig
from .config_loader import ConfigLoader, load_agent_config
from .schema_validator import SchemaValidator
from .content_types import (
    # Enums
    ContentType,
    ContentEncoding,
    DeltaOperation,
    DeltaFormat,
    
    # Constants
    CONTENT_CATEGORIES,
    DEFAULT_MIME_TYPES,
    MAX_INLINE_SIZE,
    
    # Classes
    ContentEnvelope,
    Artifact,
    StreamChunk,
    DeltaUpdate,
    MultimodalPart,
    MultimodalInput,
    
    # Functions
    wrap_content,
    unwrap_content,
    is_multimodal_input,
    parse_multimodal_input,
    is_delta_update,
    apply_delta,
    deep_merge,
    get_content_category,
)
from .content_processor import (
    ContentProcessor,
    ContentProcessingResult,
    get_content_processor,
    initialize_content_processor,
)

__all__ = [
    # Core components
    "Agent",
    "AgentConfig",
    "ConfigLoader",
    "load_agent_config",
    "SchemaValidator",
    
    # Content type system (aligned with PHTNAI_ORCHESTRATION_SCHEMA_v8)
    "ContentType",
    "ContentEncoding",
    "DeltaOperation",
    "DeltaFormat",
    "CONTENT_CATEGORIES",
    "DEFAULT_MIME_TYPES",
    "MAX_INLINE_SIZE",
    "ContentEnvelope",
    "Artifact",
    "StreamChunk",
    "DeltaUpdate",
    "MultimodalPart",
    "MultimodalInput",
    "wrap_content",
    "unwrap_content",
    "is_multimodal_input",
    "parse_multimodal_input",
    "is_delta_update",
    "apply_delta",
    "deep_merge",
    "get_content_category",
    
    # Content processor
    "ContentProcessor",
    "ContentProcessingResult",
    "get_content_processor",
    "initialize_content_processor",
]

__version__ = "2.0.0"
