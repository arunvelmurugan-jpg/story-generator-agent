"""
Content Type System for PHTN.AI Sub-Agent Framework

Implements 34 content types across 8 categories as defined in PHTNAI_ORCHESTRATION_SCHEMA_v8.json:
- Primitive: TEXT, NUMBER, BOOLEAN, NULL
- Structured: JSON_OBJECT, JSON_ARRAY, TABLE, SCHEMA_INSTANCE
- Rich Content: MARKDOWN, HTML, IMAGE, AUDIO, VIDEO, PDF, EXCEL, CHART, CODE
- Binary: BINARY_BLOB, FILE_REFERENCE, ARTIFACT, ATTACHMENT
- Streaming: STREAM_CHUNK, SSE_EVENT, DELTA
- Reference: ENTITY_REFERENCE, AGENT_REFERENCE, RESOURCE_LOCATOR, URI
- Special: TOOL_CALL, TOOL_RESULT, ERROR, METADATA, HUMAN_INPUT_REQUEST, HUMAN_APPROVAL_REQUEST

This module provides:
1. ContentType enum with all 34 types
2. ContentEncoding enum for encoding formats
3. ContentEnvelope for wrapping content with metadata
4. Utilities for wrapping/unwrapping content
5. MIME type mapping
6. Encoding/decoding utilities
7. Multimodal input support
8. Delta/incremental update support

Aligned with super-agent framework for seamless orchestration.
"""

import base64
import gzip
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Content Type Enum - 34 Types across 8 Categories (v8 schema)
# =============================================================================

class ContentType(str, Enum):
    """
    34 standardized content types for multi-agent I/O.
    
    Categories:
    - PRIMITIVE: Basic data types
    - STRUCTURED: Complex structured data
    - RICH_CONTENT: Formatted content (text, media)
    - BINARY: Binary/file data
    - STREAMING: Real-time streaming data
    - REFERENCE: Pointers/references to external resources
    - SPECIAL: Workflow-specific types
    """
    
    # =========================================================================
    # PRIMITIVE (4 types)
    # =========================================================================
    TEXT = "TEXT"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    NULL = "NULL"
    
    # =========================================================================
    # STRUCTURED (4 types)
    # =========================================================================
    JSON_OBJECT = "JSON_OBJECT"
    JSON_ARRAY = "JSON_ARRAY"
    TABLE = "TABLE"
    SCHEMA_INSTANCE = "SCHEMA_INSTANCE"
    
    # =========================================================================
    # RICH CONTENT (9 types)
    # =========================================================================
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    PDF = "PDF"
    EXCEL = "EXCEL"
    CHART = "CHART"
    CODE = "CODE"
    
    # =========================================================================
    # BINARY (4 types)
    # =========================================================================
    BINARY_BLOB = "BINARY_BLOB"
    FILE_REFERENCE = "FILE_REFERENCE"
    ARTIFACT = "ARTIFACT"
    ATTACHMENT = "ATTACHMENT"
    
    # =========================================================================
    # STREAMING (3 types)
    # =========================================================================
    STREAM_CHUNK = "STREAM_CHUNK"
    SSE_EVENT = "SSE_EVENT"
    DELTA = "DELTA"
    
    # =========================================================================
    # REFERENCE (4 types)
    # =========================================================================
    ENTITY_REFERENCE = "ENTITY_REFERENCE"
    AGENT_REFERENCE = "AGENT_REFERENCE"
    RESOURCE_LOCATOR = "RESOURCE_LOCATOR"
    URI = "URI"
    
    # =========================================================================
    # SPECIAL (6 types)
    # =========================================================================
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    ERROR = "ERROR"
    METADATA = "METADATA"
    HUMAN_INPUT_REQUEST = "HUMAN_INPUT_REQUEST"
    HUMAN_APPROVAL_REQUEST = "HUMAN_APPROVAL_REQUEST"


# =============================================================================
# Content Encoding Types
# =============================================================================

class ContentEncoding(str, Enum):
    """Supported content encodings."""
    UTF8 = "utf-8"
    UTF16 = "utf-16"
    ASCII = "ascii"
    BASE64 = "base64"
    GZIP = "gzip"
    GZIP_BASE64 = "gzip+base64"
    DEFLATE = "deflate"
    NONE = "none"


# =============================================================================
# Content Category Mapping
# =============================================================================

CONTENT_CATEGORIES = {
    "PRIMITIVE": [
        ContentType.TEXT, ContentType.NUMBER, ContentType.BOOLEAN, ContentType.NULL
    ],
    "STRUCTURED": [
        ContentType.JSON_OBJECT, ContentType.JSON_ARRAY, 
        ContentType.TABLE, ContentType.SCHEMA_INSTANCE
    ],
    "RICH_CONTENT": [
        ContentType.MARKDOWN, ContentType.HTML, ContentType.IMAGE,
        ContentType.AUDIO, ContentType.VIDEO, ContentType.PDF, 
        ContentType.EXCEL, ContentType.CHART, ContentType.CODE
    ],
    "BINARY": [
        ContentType.BINARY_BLOB, ContentType.FILE_REFERENCE,
        ContentType.ARTIFACT, ContentType.ATTACHMENT
    ],
    "STREAMING": [
        ContentType.STREAM_CHUNK, ContentType.SSE_EVENT, ContentType.DELTA
    ],
    "REFERENCE": [
        ContentType.ENTITY_REFERENCE, ContentType.AGENT_REFERENCE,
        ContentType.RESOURCE_LOCATOR, ContentType.URI
    ],
    "SPECIAL": [
        ContentType.TOOL_CALL, ContentType.TOOL_RESULT, ContentType.ERROR,
        ContentType.METADATA, ContentType.HUMAN_INPUT_REQUEST,
        ContentType.HUMAN_APPROVAL_REQUEST
    ]
}


# =============================================================================
# Default MIME Type Mapping
# =============================================================================

DEFAULT_MIME_TYPES: Dict[ContentType, str] = {
    # Primitive
    ContentType.TEXT: "text/plain",
    ContentType.NUMBER: "application/json",
    ContentType.BOOLEAN: "application/json",
    ContentType.NULL: "application/json",
    
    # Structured
    ContentType.JSON_OBJECT: "application/json",
    ContentType.JSON_ARRAY: "application/json",
    ContentType.TABLE: "application/json",
    ContentType.SCHEMA_INSTANCE: "application/json",
    
    # Rich Content
    ContentType.MARKDOWN: "text/markdown",
    ContentType.HTML: "text/html",
    ContentType.IMAGE: "image/png",
    ContentType.AUDIO: "audio/mpeg",
    ContentType.VIDEO: "video/mp4",
    ContentType.PDF: "application/pdf",
    ContentType.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ContentType.CHART: "application/json",
    ContentType.CODE: "text/plain",
    
    # Binary
    ContentType.BINARY_BLOB: "application/octet-stream",
    ContentType.FILE_REFERENCE: "application/x-uri",
    ContentType.ARTIFACT: "application/octet-stream",
    ContentType.ATTACHMENT: "application/octet-stream",
    
    # Streaming
    ContentType.STREAM_CHUNK: "application/octet-stream",
    ContentType.SSE_EVENT: "text/event-stream",
    ContentType.DELTA: "application/json",
    
    # Reference
    ContentType.ENTITY_REFERENCE: "application/json",
    ContentType.AGENT_REFERENCE: "application/json",
    ContentType.RESOURCE_LOCATOR: "application/x-uri",
    ContentType.URI: "application/x-uri",
    
    # Special
    ContentType.TOOL_CALL: "application/json",
    ContentType.TOOL_RESULT: "application/json",
    ContentType.ERROR: "application/json",
    ContentType.METADATA: "application/json",
    ContentType.HUMAN_INPUT_REQUEST: "application/json",
    ContentType.HUMAN_APPROVAL_REQUEST: "application/json",
}


# =============================================================================
# Content Envelope - Standard Wrapper
# =============================================================================

@dataclass
class ContentEnvelope:
    """
    Standard envelope for wrapping agent content with type metadata.
    
    Used to:
    1. Identify content type without parsing
    2. Handle encoding/decoding transparently
    3. Support binary data via base64/URI
    4. Enable schema validation
    """
    contentType: ContentType = ContentType.JSON_OBJECT
    mimeType: str = "application/json"
    encoding: str = "utf-8"
    data: Any = None
    uri: Optional[str] = None
    schema: Optional[Dict] = None
    size: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "contentType": self.contentType.value if isinstance(self.contentType, ContentType) else self.contentType,
            "mimeType": self.mimeType,
            "encoding": self.encoding,
        }
        
        if self.data is not None:
            result["data"] = self.data
        if self.uri:
            result["uri"] = self.uri
        if self.schema:
            result["schema"] = self.schema
        if self.size:
            result["size"] = self.size
        if self.checksum:
            result["checksum"] = self.checksum
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContentEnvelope":
        """Create from dictionary."""
        content_type_str = data.get("contentType", "JSON_OBJECT")
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            logger.warning(f"Unknown content type '{content_type_str}', defaulting to JSON_OBJECT")
            content_type = ContentType.JSON_OBJECT
        
        return cls(
            contentType=content_type,
            mimeType=data.get("mimeType", DEFAULT_MIME_TYPES.get(content_type, "application/json")),
            encoding=data.get("encoding", "utf-8"),
            data=data.get("data"),
            uri=data.get("uri"),
            schema=data.get("schema"),
            size=data.get("size"),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {})
        )
    
    def is_binary(self) -> bool:
        """Check if this content type is binary."""
        return self.contentType in CONTENT_CATEGORIES["BINARY"] or \
               self.contentType in [ContentType.IMAGE, ContentType.AUDIO, 
                                    ContentType.VIDEO, ContentType.PDF, ContentType.EXCEL]
    
    def is_streaming(self) -> bool:
        """Check if this content type is streaming."""
        return self.contentType in CONTENT_CATEGORIES["STREAMING"]
    
    def is_reference(self) -> bool:
        """Check if this content type is a reference."""
        return self.contentType in CONTENT_CATEGORIES["REFERENCE"] or \
               self.contentType == ContentType.FILE_REFERENCE
    
    def get_decoded_data(self) -> Any:
        """Get decoded data based on encoding."""
        if self.data is None:
            return None
        
        if self.encoding == ContentEncoding.BASE64.value or self.encoding == "base64":
            if isinstance(self.data, str):
                return base64.b64decode(self.data)
            return self.data
        
        if self.encoding == ContentEncoding.GZIP_BASE64.value or self.encoding == "gzip+base64":
            if isinstance(self.data, str):
                compressed = base64.b64decode(self.data)
                return gzip.decompress(compressed)
            return self.data
        
        return self.data


# =============================================================================
# Artifact - Named Output with Metadata
# =============================================================================

@dataclass
class Artifact:
    """Named artifact produced by an agent."""
    name: str
    type: ContentType = ContentType.ARTIFACT
    mimeType: str = "application/octet-stream"
    data: Optional[Any] = None
    uri: Optional[str] = None
    size: Optional[int] = None
    checksum: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    createdAt: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "type": self.type.value if isinstance(self.type, ContentType) else self.type,
            "mimeType": self.mimeType,
            "createdAt": self.createdAt,
        }
        
        if self.data is not None:
            result["data"] = self.data
        if self.uri:
            result["uri"] = self.uri
        if self.size:
            result["size"] = self.size
        if self.checksum:
            result["checksum"] = self.checksum
        if self.version:
            result["version"] = self.version
        if self.description:
            result["description"] = self.description
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Create from dictionary."""
        type_str = data.get("type", "ARTIFACT")
        try:
            artifact_type = ContentType(type_str)
        except ValueError:
            artifact_type = ContentType.ARTIFACT
        
        return cls(
            name=data.get("name", "unnamed"),
            type=artifact_type,
            mimeType=data.get("mimeType", "application/octet-stream"),
            data=data.get("data"),
            uri=data.get("uri"),
            size=data.get("size"),
            checksum=data.get("checksum"),
            version=data.get("version"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            createdAt=data.get("createdAt", datetime.utcnow().isoformat())
        )


# =============================================================================
# Stream Chunk - For Streaming Responses
# =============================================================================

@dataclass
class StreamChunk:
    """Chunk of streaming data for SSE/chunked responses."""
    sequence: int
    data: Any
    type: ContentType = ContentType.STREAM_CHUNK
    isDelta: bool = False
    isLast: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_sse(self) -> str:
        """Convert to SSE format."""
        event_data = {
            "sequence": self.sequence,
            "data": self.data,
            "isDelta": self.isDelta,
            "isLast": self.isLast,
            "timestamp": self.timestamp
        }
        return f"data: {json.dumps(event_data)}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "data": self.data,
            "type": self.type.value,
            "isDelta": self.isDelta,
            "isLast": self.isLast,
            "timestamp": self.timestamp
        }


# =============================================================================
# Delta/Incremental Updates
# =============================================================================

class DeltaOperation(str, Enum):
    """Operations for delta/incremental updates (RFC 6902)."""
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"
    APPEND = "append"
    INCREMENT = "increment"
    DECREMENT = "decrement"
    MERGE = "merge"


class DeltaFormat(str, Enum):
    """Supported delta formats."""
    JSON_PATCH = "JSON_PATCH"
    JSON_MERGE_PATCH = "JSON_MERGE_PATCH"
    CUSTOM = "CUSTOM"
    TEXT_DIFF = "TEXT_DIFF"


@dataclass
class DeltaUpdate:
    """Delta/incremental update for partial state changes."""
    format: DeltaFormat = DeltaFormat.JSON_MERGE_PATCH
    operations: List[Dict[str, Any]] = field(default_factory=list)
    patch: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    base_version: Optional[str] = None
    target_version: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    text_delta: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "isDelta": True,
            "format": self.format.value if isinstance(self.format, DeltaFormat) else self.format,
            "sequence": self.sequence,
            "timestamp": self.timestamp
        }
        
        if self.format == DeltaFormat.JSON_PATCH:
            result["operations"] = self.operations
        elif self.format == DeltaFormat.JSON_MERGE_PATCH:
            result["patch"] = self.patch
        elif self.format == DeltaFormat.TEXT_DIFF:
            result["textDelta"] = self.text_delta
        
        if self.base_version:
            result["baseVersion"] = self.base_version
        if self.target_version:
            result["targetVersion"] = self.target_version
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeltaUpdate":
        """Create from dictionary."""
        format_str = data.get("format", "JSON_MERGE_PATCH")
        try:
            delta_format = DeltaFormat(format_str)
        except ValueError:
            delta_format = DeltaFormat.JSON_MERGE_PATCH
        
        return cls(
            format=delta_format,
            operations=data.get("operations", []),
            patch=data.get("patch", {}),
            sequence=data.get("sequence", 0),
            base_version=data.get("baseVersion"),
            target_version=data.get("targetVersion"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            text_delta=data.get("textDelta")
        )
    
    @classmethod
    def create_merge_patch(cls, patch: Dict[str, Any], sequence: int = 0) -> "DeltaUpdate":
        """Create a JSON Merge Patch delta."""
        return cls(format=DeltaFormat.JSON_MERGE_PATCH, patch=patch, sequence=sequence)
    
    @classmethod
    def create_text_delta(cls, text: str, sequence: int = 0) -> "DeltaUpdate":
        """Create a text delta for streaming text."""
        return cls(format=DeltaFormat.TEXT_DIFF, text_delta=text, sequence=sequence)


# =============================================================================
# Multimodal Input Support
# =============================================================================

@dataclass
class MultimodalPart:
    """A single part of a multimodal input."""
    type: ContentType
    data: Any = None
    uri: Optional[str] = None
    mimeType: Optional[str] = None
    encoding: str = "utf-8"
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "type": self.type.value if isinstance(self.type, ContentType) else self.type,
        }
        
        if self.data is not None:
            result["data"] = self.data
        if self.uri:
            result["uri"] = self.uri
        if self.mimeType:
            result["mimeType"] = self.mimeType
        if self.encoding != "utf-8":
            result["encoding"] = self.encoding
        if self.name:
            result["name"] = self.name
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalPart":
        """Create from dictionary."""
        type_str = data.get("type", "TEXT")
        try:
            content_type = ContentType(type_str)
        except ValueError:
            content_type = ContentType.TEXT
        
        return cls(
            type=content_type,
            data=data.get("data"),
            uri=data.get("uri"),
            mimeType=data.get("mimeType"),
            encoding=data.get("encoding", "utf-8"),
            name=data.get("name"),
            metadata=data.get("metadata", {})
        )
    
    def is_binary(self) -> bool:
        """Check if this part contains binary data."""
        return self.type in [
            ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO,
            ContentType.PDF, ContentType.BINARY_BLOB, ContentType.ARTIFACT,
            ContentType.EXCEL
        ]


@dataclass
class MultimodalInput:
    """Container for multimodal input with multiple content types."""
    parts: List[MultimodalPart] = field(default_factory=list)
    primaryType: ContentType = ContentType.TEXT
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_text(self, text: str, name: str = None) -> "MultimodalInput":
        """Add a text part."""
        self.parts.append(MultimodalPart(
            type=ContentType.TEXT,
            data=text,
            mimeType="text/plain",
            name=name
        ))
        return self
    
    def add_image(
        self, 
        data: Union[bytes, str], 
        mime_type: str = "image/png",
        name: str = None,
        is_base64: bool = False
    ) -> "MultimodalInput":
        """Add an image part."""
        if isinstance(data, bytes) and not is_base64:
            encoded_data = base64.b64encode(data).decode("utf-8")
            encoding = "base64"
        else:
            encoded_data = data
            encoding = "base64" if is_base64 else "utf-8"
        
        self.parts.append(MultimodalPart(
            type=ContentType.IMAGE,
            data=encoded_data,
            mimeType=mime_type,
            encoding=encoding,
            name=name
        ))
        return self
    
    def add_audio(
        self, 
        data: Union[bytes, str], 
        mime_type: str = "audio/mpeg",
        name: str = None,
        is_base64: bool = False
    ) -> "MultimodalInput":
        """Add an audio part."""
        if isinstance(data, bytes) and not is_base64:
            encoded_data = base64.b64encode(data).decode("utf-8")
            encoding = "base64"
        else:
            encoded_data = data
            encoding = "base64" if is_base64 else "utf-8"
        
        self.parts.append(MultimodalPart(
            type=ContentType.AUDIO,
            data=encoded_data,
            mimeType=mime_type,
            encoding=encoding,
            name=name
        ))
        return self
    
    def add_json(self, data: Dict[str, Any], name: str = None) -> "MultimodalInput":
        """Add a JSON object part."""
        self.parts.append(MultimodalPart(
            type=ContentType.JSON_OBJECT,
            data=data,
            mimeType="application/json",
            name=name
        ))
        return self
    
    def get_text_parts(self) -> List[MultimodalPart]:
        """Get all text parts."""
        return [p for p in self.parts if p.type == ContentType.TEXT]
    
    def get_image_parts(self) -> List[MultimodalPart]:
        """Get all image parts."""
        return [p for p in self.parts if p.type == ContentType.IMAGE]
    
    def get_content_types(self) -> List[str]:
        """Get list of unique content types in this input."""
        return list(set(
            p.type.value if isinstance(p.type, ContentType) else p.type 
            for p in self.parts
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "multimodal": True,
            "primaryType": self.primaryType.value if isinstance(self.primaryType, ContentType) else self.primaryType,
            "parts": [p.to_dict() for p in self.parts],
            "partCount": len(self.parts),
            "contentTypes": self.get_content_types(),
            "context": self.context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalInput":
        """Create from dictionary."""
        parts = []
        for part_data in data.get("parts", []):
            parts.append(MultimodalPart.from_dict(part_data))
        
        primary_type_str = data.get("primaryType", "TEXT")
        try:
            primary_type = ContentType(primary_type_str)
        except ValueError:
            primary_type = ContentType.TEXT
        
        return cls(
            parts=parts,
            primaryType=primary_type,
            context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI Vision API format."""
        content = []
        for part in self.parts:
            if part.type == ContentType.TEXT:
                content.append({"type": "text", "text": part.data})
            elif part.type == ContentType.IMAGE:
                if part.uri:
                    content.append({"type": "image_url", "image_url": {"url": part.uri}})
                elif part.data:
                    mime = part.mimeType or "image/png"
                    data_url = f"data:{mime};base64,{part.data}"
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
        return content
    
    def to_anthropic_format(self) -> List[Dict[str, Any]]:
        """Convert to Anthropic Claude Vision format."""
        content = []
        for part in self.parts:
            if part.type == ContentType.TEXT:
                content.append({"type": "text", "text": part.data})
            elif part.type == ContentType.IMAGE:
                if part.uri:
                    content.append({"type": "image", "source": {"type": "url", "url": part.uri}})
                elif part.data:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.mimeType or "image/png",
                            "data": part.data
                        }
                    })
        return content


# =============================================================================
# Content Wrapping Utilities
# =============================================================================

MAX_INLINE_SIZE = 1 * 1024 * 1024  # 1MB


def wrap_content(
    data: Any,
    content_type: ContentType = None,
    mime_type: str = None,
    schema: Dict = None
) -> ContentEnvelope:
    """Wrap data in a ContentEnvelope with appropriate type detection."""
    if content_type is None:
        content_type = _detect_content_type(data)
    
    if mime_type is None:
        mime_type = DEFAULT_MIME_TYPES.get(content_type, "application/json")
    
    if _is_binary_type(content_type):
        if isinstance(data, bytes):
            return ContentEnvelope(
                contentType=content_type,
                mimeType=mime_type,
                encoding="base64",
                data=base64.b64encode(data).decode("ascii"),
                size=len(data),
                schema=schema
            )
    
    return ContentEnvelope(
        contentType=content_type,
        mimeType=mime_type,
        encoding="utf-8",
        data=data,
        schema=schema
    )


def unwrap_content(envelope: Union[ContentEnvelope, Dict[str, Any]]) -> Any:
    """Extract raw data from a ContentEnvelope."""
    if isinstance(envelope, dict):
        envelope = ContentEnvelope.from_dict(envelope)
    return envelope.get_decoded_data()


def _detect_content_type(data: Any) -> ContentType:
    """Auto-detect content type from data."""
    if data is None:
        return ContentType.NULL
    if isinstance(data, bool):
        return ContentType.BOOLEAN
    if isinstance(data, (int, float)):
        return ContentType.NUMBER
    if isinstance(data, str):
        if data.startswith("data:"):
            return ContentType.BINARY_BLOB
        if data.startswith(("http://", "https://", "s3://", "gs://", "azure://")):
            return ContentType.RESOURCE_LOCATOR
        if data.startswith("#") or "**" in data or data.startswith("-"):
            return ContentType.MARKDOWN
        if data.startswith("<") and ">" in data:
            return ContentType.HTML
        return ContentType.TEXT
    if isinstance(data, bytes):
        return ContentType.BINARY_BLOB
    if isinstance(data, list):
        return ContentType.JSON_ARRAY
    if isinstance(data, dict):
        if "error" in data and ("code" in data or "message" in data):
            return ContentType.ERROR
        if "tool" in data or "function" in data:
            return ContentType.TOOL_CALL
        return ContentType.JSON_OBJECT
    return ContentType.JSON_OBJECT


def _is_binary_type(content_type: ContentType) -> bool:
    """Check if content type represents binary data."""
    return content_type in [
        ContentType.BINARY_BLOB, ContentType.FILE_REFERENCE,
        ContentType.ARTIFACT, ContentType.ATTACHMENT,
        ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO,
        ContentType.PDF, ContentType.EXCEL
    ]


def is_multimodal_input(data: Any) -> bool:
    """Check if data is a multimodal input."""
    if isinstance(data, MultimodalInput):
        return True
    if isinstance(data, dict):
        if data.get("multimodal") is True:
            return True
        if "parts" in data and isinstance(data["parts"], list):
            return True
    return False


def parse_multimodal_input(data: Any) -> MultimodalInput:
    """Parse data into MultimodalInput."""
    if isinstance(data, MultimodalInput):
        return data
    if isinstance(data, dict):
        if data.get("multimodal") is True or "parts" in data:
            return MultimodalInput.from_dict(data)
        else:
            return MultimodalInput().add_json(data)
    if isinstance(data, str):
        return MultimodalInput().add_text(data)
    return MultimodalInput().add_json({"data": data})


def is_delta_update(data: Any) -> bool:
    """Check if data is a delta update."""
    if isinstance(data, DeltaUpdate):
        return True
    if isinstance(data, dict):
        if data.get("isDelta") is True:
            return True
        if "operations" in data and isinstance(data["operations"], list):
            return True
        if data.get("format") in ["JSON_PATCH", "JSON_MERGE_PATCH", "TEXT_DIFF"]:
            return True
    return False


def apply_delta(target: Any, delta: DeltaUpdate) -> Any:
    """Apply a delta update to target."""
    if delta.format == DeltaFormat.JSON_MERGE_PATCH:
        if isinstance(target, dict):
            return deep_merge(target, delta.patch)
        return target
    elif delta.format == DeltaFormat.TEXT_DIFF:
        if isinstance(target, str):
            return target + (delta.text_delta or "")
        return delta.text_delta or ""
    return target


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_content_category(content_type: Union[ContentType, str]) -> str:
    """Get the category for a content type."""
    if isinstance(content_type, str):
        try:
            content_type = ContentType(content_type)
        except ValueError:
            return "UNKNOWN"
    
    for category, types in CONTENT_CATEGORIES.items():
        if content_type in types:
            return category
    return "UNKNOWN"


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Enums
    "ContentType",
    "ContentEncoding",
    "DeltaOperation",
    "DeltaFormat",
    
    # Constants
    "CONTENT_CATEGORIES",
    "DEFAULT_MIME_TYPES",
    "MAX_INLINE_SIZE",
    
    # Classes
    "ContentEnvelope",
    "Artifact",
    "StreamChunk",
    "DeltaUpdate",
    "MultimodalPart",
    "MultimodalInput",
    
    # Functions
    "wrap_content",
    "unwrap_content",
    "is_multimodal_input",
    "parse_multimodal_input",
    "is_delta_update",
    "apply_delta",
    "deep_merge",
    "get_content_category",
]
