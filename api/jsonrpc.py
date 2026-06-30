"""
JSON-RPC 2.0 Handler for A2A Protocol

Implements the A2A protocol JSON-RPC methods:
- message/send: Send a message to the agent
- message/stream: Stream a message response
- tasks/get: Get task status
- tasks/cancel: Cancel a task

Supports all 34 content types from PHTNAI_ORCHESTRATION_SCHEMA_v8:
- Multimodal inputs (text, images, audio, video)
- Artifacts (files, binary data)
- Streaming responses (SSE, delta updates)
- All encoding formats (utf-8, base64, gzip)

Includes OTEL-compatible logging with X-PHTN-Agent-ID support.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..observability.otel_logging import get_logger, get_trace_context
from ..core.content_types import (
    ContentType,
    ContentEncoding,
    ContentEnvelope,
    Artifact,
    MultimodalInput,
    MultimodalPart,
    DEFAULT_MIME_TYPES,
)
from ..core.content_processor import get_content_processor

logger = get_logger(__name__)

router = APIRouter()


@dataclass
class Part:
    """
    A2A Message Part with full content type support.
    
    Supports:
    - text: Plain text content
    - image: Image data (inline base64 or URI)
    - audio: Audio data (inline base64 or URI)
    - video: Video data (inline base64 or URI)
    - file: File reference
    - data: Structured JSON data
    """
    kind: str = "text"
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    mimeType: Optional[str] = None
    uri: Optional[str] = None
    inlineData: Optional[Dict[str, Any]] = None  # {mimeType, data}
    encoding: Optional[str] = None


@dataclass 
class Message:
    """A2A Message with multimodal support."""
    role: str
    parts: List[Dict[str, Any]]
    messageId: Optional[str] = None
    contentType: Optional[str] = None  # Overall content type
    
    def __post_init__(self):
        if self.messageId is None:
            self.messageId = str(uuid.uuid4())


@dataclass
class Task:
    """A2A Task with artifact support."""
    id: str
    status: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    createdAt: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updatedAt: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)


_tasks: Dict[str, Task] = {}


class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 Request."""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 Response."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


def create_error_response(
    id: Optional[Union[str, int]],
    code: int,
    message: str,
    data: Optional[Any] = None
) -> Dict[str, Any]:
    """Create a JSON-RPC error response."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "error": error,
        "id": id
    }


def create_success_response(
    id: Optional[Union[str, int]],
    result: Any
) -> Dict[str, Any]:
    """Create a JSON-RPC success response."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": id
    }


async def handle_message_send(params: Dict[str, Any], app_state: Any) -> Dict[str, Any]:
    """
    Handle message/send method with full content type support.
    
    Supports all 34 content types from PHTNAI_ORCHESTRATION_SCHEMA_v8:
    - Multimodal inputs (text + images + audio + video)
    - Binary content with base64/gzip encoding
    - Artifacts in request and response
    - All MIME types
    
    Returns A2A compliant response with Task containing contextId, Messages, and Artifacts.
    """
    trace_ctx = get_trace_context()
    content_processor = get_content_processor()
    
    message_data = params.get("message", {})
    context_id = params.get("contextId") or str(uuid.uuid4())
    
    # Extract parts from message
    parts = message_data.get("parts", [])
    if not parts:
        text_content = message_data.get("text", "")
        if text_content:
            parts = [{"kind": "text", "text": text_content}]
    
    # Parse multimodal input from A2A parts
    mm_input = content_processor.parse_a2a_parts(parts)
    
    # Extract text content for agent
    user_text = ""
    has_multimodal = False
    input_content_types = []
    
    for part in mm_input.parts:
        if part.type == ContentType.TEXT:
            user_text = part.data or ""
        else:
            has_multimodal = True
        input_content_types.append(part.type.value if isinstance(part.type, ContentType) else str(part.type))
    
    task_id = str(uuid.uuid4())
    
    logger.info(
        f"📨 message/send: task_id={task_id}, context_id={context_id}, "
        f"input_length={len(user_text)}, multimodal={has_multimodal}, "
        f"content_types={input_content_types}"
    )
    
    # Build user message with full content type info
    user_message = {
        "messageId": str(uuid.uuid4()),
        "role": "user",
        "parts": parts,
        "metadata": {
            "contentTypes": input_content_types,
            "multimodal": has_multimodal,
            "partCount": len(parts),
        }
    }
    
    agent = getattr(app_state, 'agent', None)
    config = getattr(app_state, 'config', None)
    phtn_agent_id = getattr(app_state, 'phtn_agent_id', trace_ctx.get('phtn_agent_id', ''))
    config_engine = getattr(app_state, 'config_engine', None)

    response_text = f"I received your message: '{user_text[:100]}{'...' if len(user_text) > 100 else ''}'. "
    response_parts = []
    response_artifacts = []
    response_content_type = ContentType.TEXT
    
    if agent and config:
        try:
            logger.info(f"🤖 Executing agent: {config.name} (agent_id={config.agent_id})")
            await agent.initialize()
            
            # Pass multimodal input if available
            if has_multimodal:
                # Create agent input with multimodal data
                from ..core.agent import AgentInput, AgentContext
                agent_input = AgentInput(
                    content=mm_input.to_dict(),
                    content_type=ContentType.JSON_OBJECT,
                    context=AgentContext(request_id=task_id),
                )
                result = await agent.execute(agent_input)
            else:
                result = await agent.execute(user_text)
            
            if result.success and result.content:
                # Process output content
                output_result = content_processor.process_output(result.content)
                response_content_type = output_result.content_type
                
                if output_result.content_type == ContentType.TEXT:
                    response_text = str(result.content)
                    response_parts = [{"kind": "text", "text": response_text}]
                elif output_result.content_type in [ContentType.JSON_OBJECT, ContentType.JSON_ARRAY]:
                    response_text = str(result.content) if isinstance(result.content, str) else ""
                    response_parts = [
                        {"kind": "data", "data": result.content, "mimeType": "application/json"}
                    ]
                    if response_text:
                        response_parts.insert(0, {"kind": "text", "text": response_text})
                elif output_result.content_type == ContentType.MARKDOWN:
                    response_text = str(result.content)
                    response_parts = [{"kind": "text", "text": response_text, "mimeType": "text/markdown"}]
                elif output_result.content_type == ContentType.IMAGE:
                    response_parts = [{
                        "kind": "image",
                        "inlineData": {
                            "mimeType": output_result.mime_type,
                            "data": output_result.data if isinstance(output_result.data, str) else ""
                        }
                    }]
                else:
                    response_text = str(result.content)
                    response_parts = [{"kind": "text", "text": response_text}]
                
                # Extract artifacts from result
                if hasattr(result, 'metadata') and result.metadata.get('artifacts'):
                    for art in result.metadata['artifacts']:
                        response_artifacts.append(art)
                
                logger.info(
                    f"✅ Agent execution successful: output_length={len(response_text)}, "
                    f"content_type={response_content_type.value}, artifacts={len(response_artifacts)}"
                )
            else:
                # Surface the real failure reason instead of a chat-style placeholder.
                err_payload = {
                    "error": "agent_execution_empty_or_failed",
                    "agent": getattr(config, "name", "unknown"),
                    "agent_id": getattr(config, "agent_id", None),
                    "result_error": result.error if hasattr(result, "error") else None,
                    "result_success": getattr(result, "success", None),
                    "trace_id": (trace_ctx or {}).get("trace_id"),
                }
                logger.warning(f"⚠️ Agent returned no content: {err_payload}")
                response_text = json.dumps(err_payload)
                response_parts = [{"kind": "data", "data": err_payload, "mimeType": "application/json"}]
                response_content_type = ContentType.JSON_OBJECT

        except Exception as e:
            import traceback as _tb
            tb_text = _tb.format_exc()
            logger.error(f"❌ Agent execution error: {e}", exc_info=True)
            err_payload = {
                "error": "agent_execution_exception",
                "agent": getattr(config, "name", "unknown") if config else "unknown",
                "agent_id": getattr(config, "agent_id", None) if config else None,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback": tb_text.splitlines()[-30:],
                "trace_id": (trace_ctx or {}).get("trace_id"),
            }
            response_text = json.dumps(err_payload)
            response_parts = [{"kind": "data", "data": err_payload, "mimeType": "application/json"}]
            response_content_type = ContentType.JSON_OBJECT
    else:
        logger.warning("⚠️ No agent or config available in app state")
        response_text += "How can I assist you today?"
        response_parts = [{"kind": "text", "text": response_text}]
    
    # Build agent message with content type metadata
    agent_message = {
        "messageId": str(uuid.uuid4()),
        "role": "agent", 
        "parts": response_parts,
        "metadata": {
            "contentType": response_content_type.value,
            "mimeType": DEFAULT_MIME_TYPES.get(response_content_type, "text/plain"),
        }
    }
    
    # Create task with artifacts
    task = Task(
        id=task_id,
        status="completed",
        messages=[user_message, agent_message],
        artifacts=response_artifacts,
        metadata={
            "inputContentTypes": input_content_types,
            "outputContentType": response_content_type.value,
            "multimodal": has_multimodal,
        }
    )
    
    _tasks[task_id] = task
    
    logger.info(
        f"📤 message/send completed: task_id={task_id}, status=completed, "
        f"output_type={response_content_type.value}, artifacts={len(response_artifacts)}"
    )
    
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "completed",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        },
        "messages": [agent_message],
        "artifacts": response_artifacts,
        "metadata": {
            "contentType": response_content_type.value,
            "inputContentTypes": input_content_types,
        }
    }


async def handle_tasks_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tasks/get method."""
    task_id = params.get("id")
    
    if not task_id:
        raise ValueError("Task ID is required")
    
    task = _tasks.get(task_id)
    
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    
    return {
        "id": task.id,
        "status": {
            "state": task.status,
            "timestamp": task.updatedAt
        },
        "messages": task.messages,
        "artifacts": task.artifacts
    }


async def handle_tasks_cancel(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tasks/cancel method."""
    task_id = params.get("id")
    
    if not task_id:
        raise ValueError("Task ID is required")
    
    task = _tasks.get(task_id)
    
    if not task:
        raise ValueError(f"Task not found: {task_id}")
    
    task.status = "canceled"
    task.updatedAt = datetime.utcnow().isoformat() + "Z"
    
    return {
        "id": task.id,
        "status": {
            "state": "canceled",
            "timestamp": task.updatedAt
        }
    }


async def handle_agent_card(app_state: Any, request: Request) -> Dict[str, Any]:
    """Handle agent/card method - returns the agent card."""
    from .app import generate_agent_card
    
    app = request.app
    return generate_agent_card(app, request)


@router.post("/")
async def jsonrpc_handler(request: Request):
    """
    Main JSON-RPC 2.0 endpoint for A2A protocol.
    
    Supports methods:
    - message/send: Send a message and get a response
    - message/stream: Stream a message response (SSE)
    - tasks/get: Get task by ID
    - tasks/cancel: Cancel a task
    - agent/card: Get agent card
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            content=create_error_response(None, -32700, f"Parse error: {e}"),
            status_code=200
        )
    
    if isinstance(body, list):
        responses = []
        for req in body:
            response = await process_single_request(req, request)
            if response is not None:
                responses.append(response)
        return JSONResponse(content=responses, status_code=200)
    else:
        response = await process_single_request(body, request)
        if response is None:
            return JSONResponse(content="", status_code=204)
        return JSONResponse(content=response, status_code=200)


async def process_single_request(body: Dict[str, Any], request: Request) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC request."""
    
    if body.get("jsonrpc") != "2.0":
        return create_error_response(
            body.get("id"),
            -32600,
            "Invalid Request: jsonrpc must be '2.0'"
        )
    
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")
    
    if not method:
        return create_error_response(request_id, -32600, "Invalid Request: method is required")
    
    logger.info(f"📥 JSON-RPC request: method={method}, id={request_id}")
    
    try:
        app_state = request.app.state
        
        if method == "message/send":
            result = await handle_message_send(params, app_state)
            return create_success_response(request_id, result)
        
        elif method == "tasks/get":
            result = await handle_tasks_get(params)
            return create_success_response(request_id, result)
        
        elif method == "tasks/cancel":
            result = await handle_tasks_cancel(params)
            return create_success_response(request_id, result)
        
        elif method == "agent/card":
            result = await handle_agent_card(app_state, request)
            return create_success_response(request_id, result)
        
        elif method == "tasks/list":
            tasks_list = [
                {
                    "id": t.id,
                    "status": {"state": t.status, "timestamp": t.updatedAt}
                }
                for t in _tasks.values()
            ]
            return create_success_response(request_id, {"tasks": tasks_list})
        
        else:
            return create_error_response(
                request_id,
                -32601,
                f"Method not found: {method}"
            )
    
    except ValueError as e:
        return create_error_response(request_id, -32602, f"Invalid params: {e}")
    
    except Exception as e:
        logger.exception(f"JSON-RPC error: {e}")
        return create_error_response(request_id, -32603, f"Internal error: {e}")
