"""
Content Processor for PHTN.AI Sub-Agent Framework

Handles all 34 content types for input/output processing:
- Parsing and validation of incoming content
- Transformation between content types
- Multimodal content handling (images, audio, video)
- Artifact management
- Streaming content (SSE, delta updates)
- Content encoding/decoding (base64, gzip)

Aligned with PHTNAI_ORCHESTRATION_SCHEMA_v8.json for seamless super-agent integration.
"""

import base64
import gzip
import json
import hashlib
import logging
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

from .content_types import (
    ContentType,
    ContentEncoding,
    ContentEnvelope,
    Artifact,
    StreamChunk,
    DeltaUpdate,
    DeltaFormat,
    MultimodalInput,
    MultimodalPart,
    CONTENT_CATEGORIES,
    DEFAULT_MIME_TYPES,
    wrap_content,
    unwrap_content,
    is_multimodal_input,
    parse_multimodal_input,
    is_delta_update,
    get_content_category,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Content Processing Result
# =============================================================================

@dataclass
class ContentProcessingResult:
    """Result of content processing."""
    success: bool
    content_type: ContentType
    mime_type: str
    encoding: str
    data: Any
    original_data: Any = None
    artifacts: List[Artifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "contentType": self.content_type.value,
            "mimeType": self.mime_type,
            "encoding": self.encoding,
            "data": self.data,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# =============================================================================
# Content Processor
# =============================================================================

class ContentProcessor:
    """
    Processes all content types for agent I/O.
    
    Supports:
    - 34 content types across 8 categories
    - Multimodal inputs (text + images + audio + video)
    - Binary content with base64/gzip encoding
    - Artifacts with checksums and metadata
    - Streaming content (SSE, delta updates)
    - Content validation and transformation
    """
    
    def __init__(
        self,
        max_inline_size: int = 10 * 1024 * 1024,  # 10MB
        supported_image_formats: List[str] = None,
        supported_audio_formats: List[str] = None,
        supported_video_formats: List[str] = None,
        artifact_storage_path: Optional[Path] = None,
    ):
        """
        Initialize ContentProcessor.
        
        Args:
            max_inline_size: Maximum size for inline binary content
            supported_image_formats: Allowed image MIME types
            supported_audio_formats: Allowed audio MIME types
            supported_video_formats: Allowed video MIME types
            artifact_storage_path: Path for storing artifacts
        """
        self.max_inline_size = max_inline_size
        self.supported_image_formats = supported_image_formats or [
            "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"
        ]
        self.supported_audio_formats = supported_audio_formats or [
            "audio/mpeg", "audio/wav", "audio/ogg", "audio/webm"
        ]
        self.supported_video_formats = supported_video_formats or [
            "video/mp4", "video/webm", "video/ogg"
        ]
        self.artifact_storage_path = artifact_storage_path
        
        logger.debug("ContentProcessor initialized")
    
    # =========================================================================
    # Input Processing
    # =========================================================================
    
    def process_input(
        self,
        data: Any,
        declared_content_type: Optional[str] = None,
        declared_mime_type: Optional[str] = None,
        declared_encoding: Optional[str] = None,
    ) -> ContentProcessingResult:
        """
        Process incoming content and detect/validate its type.
        
        Args:
            data: Raw input data
            declared_content_type: Content type declared by sender
            declared_mime_type: MIME type declared by sender
            declared_encoding: Encoding declared by sender
            
        Returns:
            ContentProcessingResult with processed data
        """
        errors = []
        warnings = []
        artifacts = []
        
        # Detect content type if not declared
        detected_type = self._detect_content_type(data)
        
        if declared_content_type:
            try:
                declared_type = ContentType(declared_content_type)
                if declared_type != detected_type:
                    warnings.append(
                        f"Declared content type ({declared_content_type}) differs from "
                        f"detected type ({detected_type.value})"
                    )
                content_type = declared_type
            except ValueError:
                warnings.append(f"Unknown declared content type: {declared_content_type}")
                content_type = detected_type
        else:
            content_type = detected_type
        
        # Get MIME type
        mime_type = declared_mime_type or DEFAULT_MIME_TYPES.get(content_type, "application/octet-stream")
        
        # Get encoding
        encoding = declared_encoding or "utf-8"
        
        # Process based on content type
        processed_data = data
        
        try:
            if content_type in CONTENT_CATEGORIES["BINARY"]:
                processed_data, artifacts = self._process_binary_input(data, mime_type, encoding)
            
            elif content_type in [ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO, ContentType.PDF, ContentType.EXCEL]:
                processed_data, artifacts = self._process_media_input(data, content_type, mime_type, encoding)
            
            elif content_type in CONTENT_CATEGORIES["STREAMING"]:
                processed_data = self._process_streaming_input(data, content_type)
            
            elif is_multimodal_input(data):
                processed_data = self._process_multimodal_input(data)
            
            elif content_type in CONTENT_CATEGORIES["STRUCTURED"]:
                processed_data = self._process_structured_input(data, content_type)
            
            elif content_type in CONTENT_CATEGORIES["REFERENCE"]:
                processed_data = self._process_reference_input(data, content_type)
            
            elif content_type in CONTENT_CATEGORIES["SPECIAL"]:
                processed_data = self._process_special_input(data, content_type)
            
        except Exception as e:
            errors.append(f"Error processing content: {str(e)}")
            logger.error(f"Content processing error: {e}", exc_info=True)
        
        return ContentProcessingResult(
            success=len(errors) == 0,
            content_type=content_type,
            mime_type=mime_type,
            encoding=encoding,
            data=processed_data,
            original_data=data,
            artifacts=artifacts,
            metadata={
                "detected_type": detected_type.value,
                "category": get_content_category(content_type),
            },
            errors=errors,
            warnings=warnings,
        )
    
    def _detect_content_type(self, data: Any) -> ContentType:
        """Detect content type from data."""
        if data is None:
            return ContentType.NULL
        
        if isinstance(data, bool):
            return ContentType.BOOLEAN
        
        if isinstance(data, (int, float)):
            return ContentType.NUMBER
        
        if isinstance(data, str):
            # Check for special string patterns
            if data.startswith("data:"):
                return ContentType.BINARY_BLOB
            if data.startswith(("http://", "https://", "s3://", "gs://", "azure://")):
                return ContentType.URI
            if data.startswith("file://") or data.startswith("/"):
                return ContentType.FILE_REFERENCE
            if data.startswith("#") or "**" in data or data.startswith("- "):
                return ContentType.MARKDOWN
            if data.strip().startswith("<") and data.strip().endswith(">"):
                return ContentType.HTML
            if data.strip().startswith("```") or "def " in data or "function " in data:
                return ContentType.CODE
            return ContentType.TEXT
        
        if isinstance(data, bytes):
            return ContentType.BINARY_BLOB
        
        if isinstance(data, list):
            # Check if it's a table (list of dicts with same keys)
            if data and all(isinstance(item, dict) for item in data):
                keys = set(data[0].keys()) if data else set()
                if all(set(item.keys()) == keys for item in data):
                    return ContentType.TABLE
            return ContentType.JSON_ARRAY
        
        if isinstance(data, dict):
            # Check for special dict patterns
            if data.get("multimodal") is True or "parts" in data:
                return ContentType.JSON_OBJECT  # Will be handled as multimodal
            if "error" in data and ("code" in data or "message" in data):
                return ContentType.ERROR
            if "tool" in data or "function" in data or "tool_call_id" in data:
                return ContentType.TOOL_CALL
            if "result" in data and "tool_call_id" in data:
                return ContentType.TOOL_RESULT
            if data.get("isDelta") is True or "operations" in data:
                return ContentType.DELTA
            if data.get("requestType") in ["approval", "input"]:
                if data.get("requestType") == "approval":
                    return ContentType.HUMAN_APPROVAL_REQUEST
                return ContentType.HUMAN_INPUT_REQUEST
            if "contentType" in data and "data" in data:
                # Content envelope - extract inner type
                inner_type = data.get("contentType")
                try:
                    return ContentType(inner_type)
                except ValueError:
                    pass
            return ContentType.JSON_OBJECT
        
        return ContentType.JSON_OBJECT
    
    def _process_binary_input(
        self,
        data: Any,
        mime_type: str,
        encoding: str
    ) -> Tuple[Any, List[Artifact]]:
        """Process binary input data."""
        artifacts = []
        
        if isinstance(data, str):
            if encoding == "base64" or data.startswith("data:"):
                # Decode base64
                if data.startswith("data:"):
                    # Data URI format
                    header, encoded = data.split(",", 1)
                    mime_from_uri = header.split(":")[1].split(";")[0]
                    mime_type = mime_from_uri
                    decoded = base64.b64decode(encoded)
                else:
                    decoded = base64.b64decode(data)
                
                # Create artifact
                artifact = Artifact(
                    name=f"binary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    type=ContentType.BINARY_BLOB,
                    mimeType=mime_type,
                    data=data,  # Keep base64 encoded
                    size=len(decoded),
                    checksum=hashlib.sha256(decoded).hexdigest(),
                )
                artifacts.append(artifact)
                
                return {"encoded": data, "size": len(decoded), "mimeType": mime_type}, artifacts
        
        if isinstance(data, bytes):
            encoded = base64.b64encode(data).decode("ascii")
            artifact = Artifact(
                name=f"binary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                type=ContentType.BINARY_BLOB,
                mimeType=mime_type,
                data=encoded,
                size=len(data),
                checksum=hashlib.sha256(data).hexdigest(),
            )
            artifacts.append(artifact)
            return {"encoded": encoded, "size": len(data), "mimeType": mime_type}, artifacts
        
        return data, artifacts
    
    def _process_media_input(
        self,
        data: Any,
        content_type: ContentType,
        mime_type: str,
        encoding: str
    ) -> Tuple[Any, List[Artifact]]:
        """Process media input (image, audio, video, PDF, Excel)."""
        artifacts = []
        
        # Validate MIME type for media
        if content_type == ContentType.IMAGE:
            if mime_type not in self.supported_image_formats:
                logger.warning(f"Image format {mime_type} may not be supported")
        elif content_type == ContentType.AUDIO:
            if mime_type not in self.supported_audio_formats:
                logger.warning(f"Audio format {mime_type} may not be supported")
        elif content_type == ContentType.VIDEO:
            if mime_type not in self.supported_video_formats:
                logger.warning(f"Video format {mime_type} may not be supported")
        
        # Process as binary
        processed, artifacts = self._process_binary_input(data, mime_type, encoding)
        
        # Update artifact type
        for artifact in artifacts:
            artifact.type = content_type
        
        return processed, artifacts
    
    def _process_streaming_input(self, data: Any, content_type: ContentType) -> Any:
        """Process streaming input (SSE, delta, chunks)."""
        if content_type == ContentType.DELTA:
            if isinstance(data, dict):
                return DeltaUpdate.from_dict(data).to_dict()
        
        if content_type == ContentType.STREAM_CHUNK:
            if isinstance(data, dict):
                return StreamChunk(
                    sequence=data.get("sequence", 0),
                    data=data.get("data"),
                    isDelta=data.get("isDelta", False),
                    isLast=data.get("isLast", False),
                ).to_dict()
        
        return data
    
    def _process_multimodal_input(self, data: Any) -> Dict[str, Any]:
        """Process multimodal input with multiple content types."""
        mm_input = parse_multimodal_input(data)
        
        # Process each part
        processed_parts = []
        for part in mm_input.parts:
            processed_part = {
                "type": part.type.value if isinstance(part.type, ContentType) else part.type,
                "mimeType": part.mimeType,
                "encoding": part.encoding,
            }
            
            if part.data is not None:
                processed_part["data"] = part.data
            if part.uri:
                processed_part["uri"] = part.uri
            if part.name:
                processed_part["name"] = part.name
            
            processed_parts.append(processed_part)
        
        return {
            "multimodal": True,
            "primaryType": mm_input.primaryType.value,
            "parts": processed_parts,
            "partCount": len(processed_parts),
            "contentTypes": mm_input.get_content_types(),
        }
    
    def _process_structured_input(self, data: Any, content_type: ContentType) -> Any:
        """Process structured input (JSON, table, schema instance)."""
        if content_type == ContentType.TABLE:
            # Ensure it's a list of dicts
            if isinstance(data, list):
                return {"rows": data, "rowCount": len(data)}
            return data
        
        if content_type == ContentType.SCHEMA_INSTANCE:
            # Validate against schema if provided
            if isinstance(data, dict) and "$schema" in data:
                return {"instance": data, "schema": data.get("$schema")}
            return data
        
        return data
    
    def _process_reference_input(self, data: Any, content_type: ContentType) -> Any:
        """Process reference input (URI, file reference, entity reference)."""
        if isinstance(data, str):
            return {"uri": data, "type": content_type.value}
        
        if isinstance(data, dict):
            return {
                "uri": data.get("uri") or data.get("url") or data.get("reference"),
                "type": content_type.value,
                "metadata": data.get("metadata", {}),
            }
        
        return data
    
    def _process_special_input(self, data: Any, content_type: ContentType) -> Any:
        """Process special input types (tool calls, errors, HITL requests)."""
        if content_type == ContentType.TOOL_CALL:
            if isinstance(data, dict):
                return {
                    "toolCallId": data.get("tool_call_id") or data.get("id"),
                    "toolName": data.get("tool") or data.get("function", {}).get("name"),
                    "arguments": data.get("arguments") or data.get("function", {}).get("arguments"),
                }
        
        if content_type == ContentType.TOOL_RESULT:
            if isinstance(data, dict):
                return {
                    "toolCallId": data.get("tool_call_id"),
                    "result": data.get("result") or data.get("output"),
                    "success": data.get("success", True),
                }
        
        if content_type == ContentType.ERROR:
            if isinstance(data, dict):
                return {
                    "code": data.get("code") or data.get("errorCode"),
                    "message": data.get("message") or data.get("error"),
                    "details": data.get("details"),
                }
        
        if content_type in [ContentType.HUMAN_INPUT_REQUEST, ContentType.HUMAN_APPROVAL_REQUEST]:
            if isinstance(data, dict):
                return {
                    "requestId": data.get("requestId") or data.get("id"),
                    "requestType": "approval" if content_type == ContentType.HUMAN_APPROVAL_REQUEST else "input",
                    "prompt": data.get("prompt") or data.get("message"),
                    "options": data.get("options", []),
                    "timeout": data.get("timeout"),
                    "metadata": data.get("metadata", {}),
                }
        
        return data
    
    # =========================================================================
    # Output Processing
    # =========================================================================
    
    def process_output(
        self,
        data: Any,
        target_content_type: Optional[ContentType] = None,
        target_mime_type: Optional[str] = None,
        target_encoding: Optional[str] = None,
        include_envelope: bool = False,
    ) -> ContentProcessingResult:
        """
        Process output data for response.
        
        Args:
            data: Output data to process
            target_content_type: Desired output content type
            target_mime_type: Desired output MIME type
            target_encoding: Desired output encoding
            include_envelope: Whether to wrap in ContentEnvelope
            
        Returns:
            ContentProcessingResult with processed output
        """
        errors = []
        warnings = []
        artifacts = []
        
        # Detect current type
        current_type = self._detect_content_type(data)
        
        # Use target type or current type
        content_type = target_content_type or current_type
        mime_type = target_mime_type or DEFAULT_MIME_TYPES.get(content_type, "application/json")
        encoding = target_encoding or "utf-8"
        
        processed_data = data
        
        try:
            # Transform if needed
            if target_content_type and target_content_type != current_type:
                processed_data, transform_warnings = self._transform_content(
                    data, current_type, target_content_type
                )
                warnings.extend(transform_warnings)
            
            # Encode if needed
            if encoding == "base64" and isinstance(processed_data, (bytes, str)):
                if isinstance(processed_data, str):
                    processed_data = base64.b64encode(processed_data.encode()).decode("ascii")
                else:
                    processed_data = base64.b64encode(processed_data).decode("ascii")
            
            elif encoding == "gzip+base64":
                if isinstance(processed_data, str):
                    compressed = gzip.compress(processed_data.encode())
                elif isinstance(processed_data, bytes):
                    compressed = gzip.compress(processed_data)
                else:
                    compressed = gzip.compress(json.dumps(processed_data).encode())
                processed_data = base64.b64encode(compressed).decode("ascii")
            
            # Wrap in envelope if requested
            if include_envelope:
                envelope = ContentEnvelope(
                    contentType=content_type,
                    mimeType=mime_type,
                    encoding=encoding,
                    data=processed_data,
                )
                processed_data = envelope.to_dict()
            
        except Exception as e:
            errors.append(f"Error processing output: {str(e)}")
            logger.error(f"Output processing error: {e}", exc_info=True)
        
        return ContentProcessingResult(
            success=len(errors) == 0,
            content_type=content_type,
            mime_type=mime_type,
            encoding=encoding,
            data=processed_data,
            original_data=data,
            artifacts=artifacts,
            metadata={
                "original_type": current_type.value,
                "transformed": target_content_type is not None and target_content_type != current_type,
            },
            errors=errors,
            warnings=warnings,
        )
    
    def _transform_content(
        self,
        data: Any,
        from_type: ContentType,
        to_type: ContentType
    ) -> Tuple[Any, List[str]]:
        """Transform content from one type to another."""
        warnings = []
        
        # TEXT conversions
        if to_type == ContentType.TEXT:
            if isinstance(data, dict):
                return json.dumps(data, indent=2), warnings
            if isinstance(data, list):
                return json.dumps(data, indent=2), warnings
            return str(data), warnings
        
        # JSON conversions
        if to_type == ContentType.JSON_OBJECT:
            if isinstance(data, str):
                try:
                    return json.loads(data), warnings
                except json.JSONDecodeError:
                    return {"text": data}, warnings
            if isinstance(data, list):
                return {"items": data}, warnings
            return data, warnings
        
        if to_type == ContentType.JSON_ARRAY:
            if isinstance(data, dict):
                return [data], warnings
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, list):
                        return parsed, warnings
                    return [parsed], warnings
                except json.JSONDecodeError:
                    return [{"text": data}], warnings
            return [data], warnings
        
        # MARKDOWN conversion
        if to_type == ContentType.MARKDOWN:
            if isinstance(data, dict):
                md_lines = []
                for key, value in data.items():
                    md_lines.append(f"**{key}**: {value}")
                return "\n".join(md_lines), warnings
            if isinstance(data, list):
                md_lines = [f"- {item}" for item in data]
                return "\n".join(md_lines), warnings
            return str(data), warnings
        
        # HTML conversion
        if to_type == ContentType.HTML:
            if isinstance(data, dict):
                html_lines = ["<dl>"]
                for key, value in data.items():
                    html_lines.append(f"<dt>{key}</dt><dd>{value}</dd>")
                html_lines.append("</dl>")
                return "\n".join(html_lines), warnings
            if isinstance(data, list):
                html_lines = ["<ul>"]
                for item in data:
                    html_lines.append(f"<li>{item}</li>")
                html_lines.append("</ul>")
                return "\n".join(html_lines), warnings
            return f"<p>{data}</p>", warnings
        
        # TABLE conversion
        if to_type == ContentType.TABLE:
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return data, warnings
            if isinstance(data, dict):
                return [data], warnings
            warnings.append(f"Cannot convert {from_type.value} to TABLE")
            return data, warnings
        
        # No transformation available
        warnings.append(f"No transformation available from {from_type.value} to {to_type.value}")
        return data, warnings
    
    # =========================================================================
    # Artifact Management
    # =========================================================================
    
    def create_artifact(
        self,
        name: str,
        data: Union[bytes, str],
        content_type: ContentType = ContentType.ARTIFACT,
        mime_type: str = "application/octet-stream",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """Create an artifact from data."""
        if isinstance(data, str):
            # Assume base64 encoded
            try:
                decoded = base64.b64decode(data)
                size = len(decoded)
                checksum = hashlib.sha256(decoded).hexdigest()
            except Exception:
                size = len(data.encode())
                checksum = hashlib.sha256(data.encode()).hexdigest()
        else:
            size = len(data)
            checksum = hashlib.sha256(data).hexdigest()
            data = base64.b64encode(data).decode("ascii")
        
        return Artifact(
            name=name,
            type=content_type,
            mimeType=mime_type,
            data=data,
            size=size,
            checksum=checksum,
            description=description,
            metadata=metadata or {},
        )
    
    def extract_artifacts(self, data: Any) -> Tuple[Any, List[Artifact]]:
        """Extract artifacts from data."""
        artifacts = []
        
        if isinstance(data, dict):
            # Check for artifacts array
            if "artifacts" in data:
                for art_data in data.get("artifacts", []):
                    if isinstance(art_data, dict):
                        artifacts.append(Artifact.from_dict(art_data))
                # Remove artifacts from data
                data = {k: v for k, v in data.items() if k != "artifacts"}
        
        return data, artifacts
    
    # =========================================================================
    # Streaming Support
    # =========================================================================
    
    def create_stream_chunk(
        self,
        sequence: int,
        data: Any,
        is_delta: bool = False,
        is_last: bool = False,
    ) -> StreamChunk:
        """Create a stream chunk."""
        return StreamChunk(
            sequence=sequence,
            data=data,
            isDelta=is_delta,
            isLast=is_last,
        )
    
    def create_delta_update(
        self,
        patch: Dict[str, Any],
        sequence: int = 0,
        format: DeltaFormat = DeltaFormat.JSON_MERGE_PATCH,
    ) -> DeltaUpdate:
        """Create a delta update."""
        return DeltaUpdate(
            format=format,
            patch=patch,
            sequence=sequence,
        )
    
    def create_text_delta(self, text: str, sequence: int = 0) -> DeltaUpdate:
        """Create a text delta for streaming text."""
        return DeltaUpdate.create_text_delta(text, sequence)
    
    # =========================================================================
    # Multimodal Support
    # =========================================================================
    
    def create_multimodal_input(self) -> MultimodalInput:
        """Create a new multimodal input container."""
        return MultimodalInput()
    
    def parse_a2a_parts(self, parts: List[Dict[str, Any]]) -> MultimodalInput:
        """Parse A2A protocol message parts into MultimodalInput."""
        mm_input = MultimodalInput()
        
        for part in parts:
            kind = part.get("kind", "text")
            
            if kind == "text":
                mm_input.add_text(part.get("text", ""))
            
            elif kind == "image":
                data = part.get("data") or part.get("inlineData", {}).get("data")
                mime_type = part.get("mimeType") or part.get("inlineData", {}).get("mimeType", "image/png")
                if data:
                    mm_input.parts.append(MultimodalPart(
                        type=ContentType.IMAGE,
                        data=data,
                        mimeType=mime_type,
                        encoding="base64",
                    ))
                elif part.get("uri"):
                    mm_input.parts.append(MultimodalPart(
                        type=ContentType.IMAGE,
                        uri=part.get("uri"),
                        mimeType=mime_type,
                    ))
            
            elif kind == "audio":
                data = part.get("data") or part.get("inlineData", {}).get("data")
                mime_type = part.get("mimeType") or "audio/mpeg"
                if data:
                    mm_input.parts.append(MultimodalPart(
                        type=ContentType.AUDIO,
                        data=data,
                        mimeType=mime_type,
                        encoding="base64",
                    ))
            
            elif kind == "video":
                data = part.get("data") or part.get("inlineData", {}).get("data")
                mime_type = part.get("mimeType") or "video/mp4"
                if data:
                    mm_input.parts.append(MultimodalPart(
                        type=ContentType.VIDEO,
                        data=data,
                        mimeType=mime_type,
                        encoding="base64",
                    ))
            
            elif kind == "file":
                mm_input.parts.append(MultimodalPart(
                    type=ContentType.FILE_REFERENCE,
                    uri=part.get("uri"),
                    mimeType=part.get("mimeType"),
                    name=part.get("name"),
                ))
            
            elif kind == "data":
                mm_input.parts.append(MultimodalPart(
                    type=ContentType.JSON_OBJECT,
                    data=part.get("data"),
                    mimeType="application/json",
                ))
        
        return mm_input
    
    def to_a2a_parts(self, mm_input: MultimodalInput) -> List[Dict[str, Any]]:
        """Convert MultimodalInput to A2A protocol message parts."""
        parts = []
        
        for part in mm_input.parts:
            if part.type == ContentType.TEXT:
                parts.append({"kind": "text", "text": part.data})
            
            elif part.type == ContentType.IMAGE:
                if part.uri:
                    parts.append({"kind": "image", "uri": part.uri, "mimeType": part.mimeType})
                elif part.data:
                    parts.append({
                        "kind": "image",
                        "inlineData": {"mimeType": part.mimeType or "image/png", "data": part.data}
                    })
            
            elif part.type == ContentType.AUDIO:
                if part.data:
                    parts.append({
                        "kind": "audio",
                        "inlineData": {"mimeType": part.mimeType or "audio/mpeg", "data": part.data}
                    })
            
            elif part.type == ContentType.VIDEO:
                if part.data:
                    parts.append({
                        "kind": "video",
                        "inlineData": {"mimeType": part.mimeType or "video/mp4", "data": part.data}
                    })
            
            elif part.type == ContentType.FILE_REFERENCE:
                parts.append({
                    "kind": "file",
                    "uri": part.uri,
                    "mimeType": part.mimeType,
                    "name": part.name,
                })
            
            elif part.type == ContentType.JSON_OBJECT:
                parts.append({"kind": "data", "data": part.data, "mimeType": "application/json"})
            
            else:
                # Default to data
                parts.append({"kind": "data", "data": part.data, "mimeType": part.mimeType})
        
        return parts


# =============================================================================
# Global Content Processor Instance
# =============================================================================

_content_processor: Optional[ContentProcessor] = None


def get_content_processor() -> ContentProcessor:
    """Get or create the global content processor."""
    global _content_processor
    if _content_processor is None:
        _content_processor = ContentProcessor()
    return _content_processor


def initialize_content_processor(**kwargs) -> ContentProcessor:
    """Initialize the global content processor with custom settings."""
    global _content_processor
    _content_processor = ContentProcessor(**kwargs)
    return _content_processor


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "ContentProcessor",
    "ContentProcessingResult",
    "get_content_processor",
    "initialize_content_processor",
]
