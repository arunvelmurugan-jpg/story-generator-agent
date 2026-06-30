"""
OpenTelemetry Content Type Span Attributes for PHTN.AI Sub-Agent Framework

This module provides OTEL span attributes and metrics specifically for content types,
artifacts, and streaming responses as defined in PHTNAI_ORCHESTRATION_SCHEMA_v8.json.

Features:
- Content type span attributes (gen_ai.content_type, gen_ai.mime_type)
- Artifact tracking (count, types, sizes)
- Streaming response metrics
- Content envelope metadata
- Multimodal input attributes
- Delta update tracking
- Cost tracking per content type

Aligned with super-agent framework for seamless orchestration and phtnai-ops-metrics.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Content Type Categories (from PHTNAI_ORCHESTRATION_SCHEMA_v8.json)
# =============================================================================

class ContentTypeCategory(str, Enum):
    """Content type categories for OTEL attributes."""
    PRIMITIVE = "PRIMITIVE"
    STRUCTURED = "STRUCTURED"
    RICH_CONTENT = "RICH_CONTENT"
    BINARY = "BINARY"
    STREAMING = "STREAMING"
    REFERENCE = "REFERENCE"
    SPECIAL = "SPECIAL"


CONTENT_TYPE_CATEGORIES = {
    # Primitive (4 types)
    "TEXT": ContentTypeCategory.PRIMITIVE,
    "NUMBER": ContentTypeCategory.PRIMITIVE,
    "BOOLEAN": ContentTypeCategory.PRIMITIVE,
    "NULL": ContentTypeCategory.PRIMITIVE,
    
    # Structured (4 types)
    "JSON_OBJECT": ContentTypeCategory.STRUCTURED,
    "JSON_ARRAY": ContentTypeCategory.STRUCTURED,
    "TABLE": ContentTypeCategory.STRUCTURED,
    "SCHEMA_INSTANCE": ContentTypeCategory.STRUCTURED,
    
    # Rich Content (9 types)
    "MARKDOWN": ContentTypeCategory.RICH_CONTENT,
    "HTML": ContentTypeCategory.RICH_CONTENT,
    "IMAGE": ContentTypeCategory.RICH_CONTENT,
    "AUDIO": ContentTypeCategory.RICH_CONTENT,
    "VIDEO": ContentTypeCategory.RICH_CONTENT,
    "PDF": ContentTypeCategory.RICH_CONTENT,
    "EXCEL": ContentTypeCategory.RICH_CONTENT,
    "CHART": ContentTypeCategory.RICH_CONTENT,
    "CODE": ContentTypeCategory.RICH_CONTENT,
    
    # Binary (4 types)
    "BINARY_BLOB": ContentTypeCategory.BINARY,
    "FILE_REFERENCE": ContentTypeCategory.BINARY,
    "ARTIFACT": ContentTypeCategory.BINARY,
    "ATTACHMENT": ContentTypeCategory.BINARY,
    
    # Streaming (3 types)
    "STREAM_CHUNK": ContentTypeCategory.STREAMING,
    "SSE_EVENT": ContentTypeCategory.STREAMING,
    "DELTA": ContentTypeCategory.STREAMING,
    
    # Reference (4 types)
    "ENTITY_REFERENCE": ContentTypeCategory.REFERENCE,
    "AGENT_REFERENCE": ContentTypeCategory.REFERENCE,
    "RESOURCE_LOCATOR": ContentTypeCategory.REFERENCE,
    "URI": ContentTypeCategory.REFERENCE,
    
    # Special (6 types)
    "TOOL_CALL": ContentTypeCategory.SPECIAL,
    "TOOL_RESULT": ContentTypeCategory.SPECIAL,
    "ERROR": ContentTypeCategory.SPECIAL,
    "METADATA": ContentTypeCategory.SPECIAL,
    "HUMAN_INPUT_REQUEST": ContentTypeCategory.SPECIAL,
    "HUMAN_APPROVAL_REQUEST": ContentTypeCategory.SPECIAL,
}


# =============================================================================
# Content Type OTEL Attributes
# =============================================================================

def create_content_type_attributes(
    content_type: Optional[str] = None,
    mime_type: Optional[str] = None,
    encoding: Optional[str] = None,
    content_size_bytes: Optional[int] = None,
    is_streaming: bool = False,
    stream_chunk_count: Optional[int] = None,
    stream_total_bytes: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create OTEL span attributes for content type information.
    
    Based on OpenTelemetry GenAI Semantic Conventions extended for
    multi-agent content types.
    """
    attrs = {}
    
    if content_type:
        attrs["gen_ai.content_type"] = content_type
        category = CONTENT_TYPE_CATEGORIES.get(content_type)
        if category:
            attrs["gen_ai.content_type.category"] = category.value
    
    if mime_type:
        attrs["gen_ai.mime_type"] = mime_type
    
    if encoding:
        attrs["gen_ai.content_encoding"] = encoding
    
    if content_size_bytes is not None:
        attrs["gen_ai.content_size_bytes"] = content_size_bytes
    
    attrs["gen_ai.streaming"] = is_streaming
    
    if is_streaming:
        if stream_chunk_count is not None:
            attrs["gen_ai.stream.chunk_count"] = stream_chunk_count
        if stream_total_bytes is not None:
            attrs["gen_ai.stream.total_bytes"] = stream_total_bytes
    
    return attrs


def create_artifact_attributes(
    artifact_count: int = 0,
    artifact_types: Optional[List[str]] = None,
    artifact_total_size_bytes: Optional[int] = None,
    artifact_names: Optional[List[str]] = None,
    artifact_storage_backend: Optional[str] = None
) -> Dict[str, Any]:
    """Create OTEL span attributes for artifact information."""
    attrs = {"gen_ai.artifact_count": artifact_count}
    
    if artifact_types:
        types_str = ",".join(artifact_types[:10])
        if len(artifact_types) > 10:
            types_str += f",... (+{len(artifact_types) - 10} more)"
        attrs["gen_ai.artifact_types"] = types_str
        
        category_counts = {}
        for art_type in artifact_types:
            category = CONTENT_TYPE_CATEGORIES.get(art_type, ContentTypeCategory.BINARY)
            category_counts[category.value] = category_counts.get(category.value, 0) + 1
        
        for cat, count in category_counts.items():
            attrs[f"gen_ai.artifact_category.{cat.lower()}_count"] = count
    
    if artifact_total_size_bytes is not None:
        attrs["gen_ai.artifact_total_size_bytes"] = artifact_total_size_bytes
        if artifact_total_size_bytes < 1024:
            size_human = f"{artifact_total_size_bytes}B"
        elif artifact_total_size_bytes < 1024 * 1024:
            size_human = f"{artifact_total_size_bytes / 1024:.1f}KB"
        elif artifact_total_size_bytes < 1024 * 1024 * 1024:
            size_human = f"{artifact_total_size_bytes / (1024 * 1024):.1f}MB"
        else:
            size_human = f"{artifact_total_size_bytes / (1024 * 1024 * 1024):.1f}GB"
        attrs["gen_ai.artifact_total_size_human"] = size_human
    
    if artifact_names:
        names_str = ",".join(artifact_names[:10])
        if len(artifact_names) > 10:
            names_str += f",... (+{len(artifact_names) - 10} more)"
        attrs["gen_ai.artifact_names"] = names_str
    
    if artifact_storage_backend:
        attrs["gen_ai.artifact_storage_backend"] = artifact_storage_backend
    
    return attrs


def create_multimodal_input_attributes(
    is_multimodal: bool = False,
    part_count: int = 0,
    content_types: Optional[List[str]] = None,
    primary_type: Optional[str] = None,
    total_size_bytes: Optional[int] = None,
    has_images: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
    has_text: bool = False,
    image_count: int = 0,
    audio_count: int = 0,
    video_count: int = 0,
    text_count: int = 0
) -> Dict[str, Any]:
    """Create OTEL span attributes for multimodal input information."""
    attrs = {"gen_ai.input.multimodal": is_multimodal}
    
    if is_multimodal:
        attrs["gen_ai.input.multimodal.part_count"] = part_count
        
        if content_types:
            attrs["gen_ai.input.multimodal.content_types"] = ",".join(content_types[:10])
        
        if primary_type:
            attrs["gen_ai.input.multimodal.primary_type"] = primary_type
        
        if total_size_bytes is not None:
            attrs["gen_ai.input.multimodal.total_size_bytes"] = total_size_bytes
        
        attrs["gen_ai.input.multimodal.has_images"] = has_images
        attrs["gen_ai.input.multimodal.has_audio"] = has_audio
        attrs["gen_ai.input.multimodal.has_video"] = has_video
        attrs["gen_ai.input.multimodal.has_text"] = has_text
        
        if image_count > 0:
            attrs["gen_ai.input.multimodal.image_count"] = image_count
        if audio_count > 0:
            attrs["gen_ai.input.multimodal.audio_count"] = audio_count
        if video_count > 0:
            attrs["gen_ai.input.multimodal.video_count"] = video_count
        if text_count > 0:
            attrs["gen_ai.input.multimodal.text_count"] = text_count
    
    return attrs


def create_delta_update_attributes(
    is_delta: bool = False,
    delta_format: Optional[str] = None,
    operation_count: int = 0,
    sequence: int = 0,
    base_version: Optional[str] = None,
    target_version: Optional[str] = None,
    is_text_delta: bool = False,
    text_delta_length: int = 0
) -> Dict[str, Any]:
    """Create OTEL span attributes for delta/incremental updates."""
    attrs = {"gen_ai.delta.is_delta": is_delta}
    
    if is_delta:
        if delta_format:
            attrs["gen_ai.delta.format"] = delta_format
        
        attrs["gen_ai.delta.sequence"] = sequence
        
        if operation_count > 0:
            attrs["gen_ai.delta.operation_count"] = operation_count
        
        if base_version:
            attrs["gen_ai.delta.base_version"] = base_version
        if target_version:
            attrs["gen_ai.delta.target_version"] = target_version
        
        if is_text_delta:
            attrs["gen_ai.delta.is_text_delta"] = True
            attrs["gen_ai.delta.text_length"] = text_delta_length
    
    return attrs


def create_input_output_schema_attributes(
    input_content_type: Optional[str] = None,
    input_mime_type: Optional[str] = None,
    output_content_type: Optional[str] = None,
    output_mime_type: Optional[str] = None,
    has_input_schema: bool = False,
    has_output_schema: bool = False,
    input_multimodal: bool = False,
    output_artifacts_enabled: bool = False,
    output_streaming_enabled: bool = False
) -> Dict[str, Any]:
    """Create OTEL span attributes for agent input/output schema information."""
    attrs = {}
    
    if input_content_type:
        attrs["gen_ai.input.content_type"] = input_content_type
    if input_mime_type:
        attrs["gen_ai.input.mime_type"] = input_mime_type
    attrs["gen_ai.input.has_schema"] = has_input_schema
    attrs["gen_ai.input.multimodal"] = input_multimodal
    
    if output_content_type:
        attrs["gen_ai.output.content_type"] = output_content_type
    if output_mime_type:
        attrs["gen_ai.output.mime_type"] = output_mime_type
    attrs["gen_ai.output.has_schema"] = has_output_schema
    attrs["gen_ai.output.artifacts_enabled"] = output_artifacts_enabled
    attrs["gen_ai.output.streaming_enabled"] = output_streaming_enabled
    
    return attrs


# =============================================================================
# Logging Helpers for Content Types
# =============================================================================

def log_content_type_event(
    logger_instance,
    event_name: str,
    content_type: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    agent_name: Optional[str] = None,
    **extra_data
):
    """Log a content type event with structured data."""
    log_data = {
        "event": event_name,
        "log_type": "sub_agent",
        "operation": "content_type_event",
    }
    
    if content_type:
        log_data["content_type"] = content_type
        category = CONTENT_TYPE_CATEGORIES.get(content_type)
        if category:
            log_data["content_type_category"] = category.value
    
    if mime_type:
        log_data["mime_type"] = mime_type
    
    if size_bytes is not None:
        log_data["size_bytes"] = size_bytes
    
    if agent_name:
        log_data["agent_name"] = agent_name
    
    log_data.update(extra_data)
    
    logger_instance.info(f"Content type event: {event_name}", extra={"extra_data": log_data})


def log_artifact_event(
    logger_instance,
    artifact_name: str,
    artifact_type: str,
    artifact_size_bytes: int,
    agent_name: Optional[str] = None,
    artifact_uri: Optional[str] = None,
    storage_backend: Optional[str] = None,
    **extra_data
):
    """Log an artifact creation event."""
    log_data = {
        "event": "artifact_created",
        "log_type": "sub_agent",
        "operation": "artifact_event",
        "artifact_name": artifact_name,
        "artifact_type": artifact_type,
        "artifact_size_bytes": artifact_size_bytes,
    }
    
    if agent_name:
        log_data["agent_name"] = agent_name
    
    if artifact_uri:
        if "://" in artifact_uri:
            log_data["artifact_uri_scheme"] = artifact_uri.split("://")[0]
    
    if storage_backend:
        log_data["storage_backend"] = storage_backend
    
    log_data.update(extra_data)
    
    logger_instance.info(f"Artifact created: {artifact_name}", extra={"extra_data": log_data})


def log_multimodal_input(
    logger_instance,
    part_count: int,
    content_types: List[str],
    total_size_bytes: int,
    agent_name: Optional[str] = None,
    has_images: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
    **extra_data
):
    """Log a multimodal input event."""
    log_data = {
        "event": "multimodal_input",
        "log_type": "sub_agent",
        "operation": "multimodal_event",
        "part_count": part_count,
        "content_types": ",".join(content_types),
        "total_size_bytes": total_size_bytes,
        "has_images": has_images,
        "has_audio": has_audio,
        "has_video": has_video,
    }
    
    if agent_name:
        log_data["agent_name"] = agent_name
    
    log_data.update(extra_data)
    
    logger_instance.info(f"Multimodal input: {part_count} parts", extra={"extra_data": log_data})


def log_delta_update(
    logger_instance,
    delta_format: str,
    sequence: int,
    operation_count: int = 0,
    agent_name: Optional[str] = None,
    applied_successfully: bool = True,
    error_message: Optional[str] = None,
    **extra_data
):
    """Log a delta update event."""
    log_data = {
        "event": "delta_update",
        "log_type": "sub_agent",
        "operation": "delta_event",
        "delta_format": delta_format,
        "sequence": sequence,
        "operation_count": operation_count,
        "applied_successfully": applied_successfully,
    }
    
    if agent_name:
        log_data["agent_name"] = agent_name
    
    if error_message:
        log_data["error"] = error_message
    
    log_data.update(extra_data)
    
    level = logging.INFO if applied_successfully else logging.WARNING
    logger_instance.log(level, f"Delta update: {delta_format}", extra={"extra_data": log_data})


def log_streaming_event(
    logger_instance,
    event_type: str,
    chunk_index: Optional[int] = None,
    chunk_size_bytes: Optional[int] = None,
    total_chunks: Optional[int] = None,
    total_bytes: Optional[int] = None,
    duration_seconds: Optional[float] = None,
    agent_name: Optional[str] = None,
    **extra_data
):
    """Log a streaming event."""
    log_data = {
        "event": f"streaming_{event_type}",
        "log_type": "sub_agent",
        "operation": "streaming_event",
        "stream_event_type": event_type,
    }
    
    if chunk_index is not None:
        log_data["chunk_index"] = chunk_index
    if chunk_size_bytes is not None:
        log_data["chunk_size_bytes"] = chunk_size_bytes
    if total_chunks is not None:
        log_data["total_chunks"] = total_chunks
    if total_bytes is not None:
        log_data["total_bytes"] = total_bytes
    if duration_seconds is not None:
        log_data["duration_seconds"] = duration_seconds
    if agent_name:
        log_data["agent_name"] = agent_name
    
    log_data.update(extra_data)
    
    logger_instance.info(f"Streaming {event_type}", extra={"extra_data": log_data})


# =============================================================================
# Content Type Cost Tracking
# =============================================================================

def log_content_type_cost(
    logger_instance,
    agent_name: str,
    content_type: str,
    total_cost_usd: float,
    input_cost_usd: float = 0.0,
    output_cost_usd: float = 0.0,
    processing_cost_usd: float = 0.0,
    storage_cost_usd: float = 0.0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    image_count: int = 0,
    model: str = None,
    **extra_data
):
    """Log content type cost for FinOps tracking."""
    log_data = {
        "event": "content_type_cost",
        "log_type": "llm_gateway",
        "operation": "cost_tracking",
        "agent_name": agent_name,
        "content_type": content_type,
        "cost": total_cost_usd,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "processing_cost_usd": processing_cost_usd,
        "storage_cost_usd": storage_cost_usd,
        "prompt_tokens": tokens_input,
        "completion_tokens": tokens_output,
        "total_tokens": tokens_input + tokens_output,
        "image_count": image_count,
    }
    
    if model:
        log_data["model"] = model
    
    category = CONTENT_TYPE_CATEGORIES.get(content_type)
    if category:
        log_data["content_type_category"] = category.value
    
    log_data.update(extra_data)
    
    logger_instance.info(f"Content type cost: ${total_cost_usd:.6f}", extra={"extra_data": log_data})


# =============================================================================
# Helper Functions for Workflow Integration
# =============================================================================

def extract_content_type_from_output(
    agent_output: Dict[str, Any],
    agent_config: Any = None
) -> Dict[str, Any]:
    """Extract content type information from agent output for OTEL attributes."""
    attrs = {}
    
    if isinstance(agent_output, dict):
        if "contentType" in agent_output:
            attrs.update(create_content_type_attributes(
                content_type=agent_output.get("contentType"),
                mime_type=agent_output.get("mimeType"),
                encoding=agent_output.get("encoding")
            ))
        
        artifacts = agent_output.get("artifacts", [])
        if artifacts:
            artifact_types = []
            artifact_names = []
            total_size = 0
            
            for artifact in artifacts:
                if isinstance(artifact, dict):
                    art_type = artifact.get("type") or artifact.get("contentType")
                    if art_type:
                        artifact_types.append(art_type)
                    art_name = artifact.get("name")
                    if art_name:
                        artifact_names.append(art_name)
                    art_size = artifact.get("size") or artifact.get("sizeBytes") or 0
                    total_size += art_size
            
            attrs.update(create_artifact_attributes(
                artifact_count=len(artifacts),
                artifact_types=artifact_types,
                artifact_total_size_bytes=total_size if total_size > 0 else None,
                artifact_names=artifact_names if artifact_names else None
            ))
        
        if agent_output.get("streaming") or agent_output.get("isStreaming"):
            attrs["gen_ai.streaming"] = True
    
    return attrs


def extract_content_type_from_config(agent_config: Any) -> Dict[str, Any]:
    """Extract content type schema information from agent configuration."""
    attrs = {}
    
    if not agent_config:
        return attrs
    
    input_schema = getattr(agent_config, 'input_schema', None) or getattr(agent_config, 'inputSchema', None)
    if input_schema:
        if hasattr(input_schema, 'contentType'):
            attrs["gen_ai.input.content_type"] = input_schema.contentType
        elif isinstance(input_schema, dict):
            if input_schema.get('contentType'):
                attrs["gen_ai.input.content_type"] = input_schema.get('contentType')
    
    output_schema = getattr(agent_config, 'output_schema', None) or getattr(agent_config, 'outputSchema', None)
    if output_schema:
        if hasattr(output_schema, 'contentType'):
            attrs["gen_ai.output.content_type"] = output_schema.contentType
        elif isinstance(output_schema, dict):
            if output_schema.get('contentType'):
                attrs["gen_ai.output.content_type"] = output_schema.get('contentType')
    
    return attrs


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "ContentTypeCategory",
    "CONTENT_TYPE_CATEGORIES",
    
    # Attribute creators
    "create_content_type_attributes",
    "create_artifact_attributes",
    "create_multimodal_input_attributes",
    "create_delta_update_attributes",
    "create_input_output_schema_attributes",
    
    # Logging helpers
    "log_content_type_event",
    "log_artifact_event",
    "log_multimodal_input",
    "log_delta_update",
    "log_streaming_event",
    "log_content_type_cost",
    
    # Helper functions
    "extract_content_type_from_output",
    "extract_content_type_from_config",
]
