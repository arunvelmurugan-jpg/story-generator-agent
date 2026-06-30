"""
Execution Stream - SSE endpoint for real-time execution status updates.

Provides Server-Sent Events (SSE) streaming for monitoring sub-agent execution phases:
- INPUT: User message received
- CONTEXT: Context building (memory, few-shot, token budgeting)
- GUARDRAILS_INPUT: Input guardrails (PII, prompt injection)
- REACT_LOOP: ReAct reasoning loop
- LLM_CALL: LLM inference calls
- TOOL_CALL: Tool executions
- GUARDRAILS_OUTPUT: Output guardrails (toxicity, validation)
- MEMORY: Memory updates
- OUTPUT: Final response
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from starlette.responses import FileResponse

from ..observability.otel_logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ExecutionPhase(str, Enum):
    """Execution phases within a sub-agent."""
    INPUT = "INPUT"
    CONTEXT = "CONTEXT"
    GUARDRAILS_INPUT = "GUARDRAILS_INPUT"
    REACT_LOOP = "REACT_LOOP"
    LLM_CALL = "LLM_CALL"
    TOOL_CALL = "TOOL_CALL"
    GUARDRAILS_OUTPUT = "GUARDRAILS_OUTPUT"
    MEMORY = "MEMORY"
    OUTPUT = "OUTPUT"


class PhaseStatus(str, Enum):
    """Status of an execution phase."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ExecutionEvent:
    """Represents an execution event for streaming."""
    phase: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    iteration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status,
            "details": self.details,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "iteration": self.iteration
        }


@dataclass
class ExecutionMetrics:
    """Metrics for the current execution."""
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    tool_calls: int = 0
    iterations: int = 0
    memory_items_retrieved: int = 0
    pii_detected: int = 0
    guardrails_passed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionStreamManager:
    """
    Manages execution event streaming to connected clients.
    
    Supports multiple concurrent clients and maintains execution state.
    """
    
    def __init__(self):
        self._clients: List[asyncio.Queue] = []
        self._current_execution: Optional[str] = None
        self._metrics = ExecutionMetrics()
        self._phase_states: Dict[str, str] = {phase.value: PhaseStatus.PENDING.value for phase in ExecutionPhase}
        self._react_trace: List[Dict[str, Any]] = []
        self._tool_trace: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def register_client(self) -> asyncio.Queue:
        """Register a new SSE client."""
        queue = asyncio.Queue()
        async with self._lock:
            self._clients.append(queue)
            logger.debug(f"SSE client registered. Total clients: {len(self._clients)}")
        return queue
    
    async def unregister_client(self, queue: asyncio.Queue):
        """Unregister an SSE client."""
        async with self._lock:
            if queue in self._clients:
                self._clients.remove(queue)
                logger.debug(f"SSE client unregistered. Total clients: {len(self._clients)}")
    
    async def emit_event(self, event: ExecutionEvent):
        """Emit an event to all connected clients."""
        async with self._lock:
            self._phase_states[event.phase] = event.status
            
            if event.phase == ExecutionPhase.LLM_CALL.value and event.status == PhaseStatus.COMPLETED.value:
                if "thought" in event.details:
                    self._react_trace.append({
                        "iteration": event.iteration or len(self._react_trace) + 1,
                        "thought": event.details.get("thought", ""),
                        "action": event.details.get("action", ""),
                        "observation": event.details.get("observation", ""),
                        "timestamp": event.timestamp
                    })
                if "tokens_input" in event.details:
                    self._metrics.input_tokens += event.details.get("tokens_input", 0)
                    self._metrics.output_tokens += event.details.get("tokens_output", 0)
                    self._metrics.total_tokens = self._metrics.input_tokens + self._metrics.output_tokens
            
            if event.phase == ExecutionPhase.TOOL_CALL.value:
                if event.status == PhaseStatus.COMPLETED.value:
                    self._tool_trace.append({
                        "tool_name": event.details.get("tool_name", "unknown"),
                        "input": event.details.get("tool_input", {}),
                        "output": event.details.get("output_preview", ""),
                        "success": event.details.get("success", True),
                        "latency_ms": event.details.get("latency_ms", 0),
                        "timestamp": event.timestamp
                    })
                    self._metrics.tool_calls += 1
            
            if event.phase == ExecutionPhase.REACT_LOOP.value and event.status == PhaseStatus.ACTIVE.value:
                self._metrics.iterations = event.details.get("iteration", self._metrics.iterations)
            
            event_data = {
                **event.to_dict(),
                "metrics": self._metrics.to_dict(),
                "phase_states": self._phase_states.copy(),
                "react_trace": self._react_trace[-5:],
                "tool_trace": self._tool_trace[-5:]
            }
            
            for client_queue in self._clients:
                try:
                    await client_queue.put(event_data)
                except Exception as e:
                    logger.warning(f"Failed to send event to client: {e}")
    
    def reset(self, request_id: Optional[str] = None):
        """Reset execution state for a new request."""
        self._current_execution = request_id
        self._metrics = ExecutionMetrics()
        self._phase_states = {phase.value: PhaseStatus.PENDING.value for phase in ExecutionPhase}
        self._react_trace = []
        self._tool_trace = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current execution state."""
        return {
            "request_id": self._current_execution,
            "metrics": self._metrics.to_dict(),
            "phase_states": self._phase_states.copy(),
            "react_trace": self._react_trace,
            "tool_trace": self._tool_trace
        }


execution_stream_manager = ExecutionStreamManager()


async def emit_execution_event(
    phase: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    iteration: Optional[int] = None
):
    """
    Emit an execution event to all connected SSE clients.
    
    Args:
        phase: Execution phase (INPUT, CONTEXT, LLM_CALL, etc.)
        status: Phase status (pending, active, completed, error)
        details: Additional details about the phase
        request_id: Optional request identifier
        iteration: Optional iteration number (for ReAct loop)
    """
    event = ExecutionEvent(
        phase=phase,
        status=status,
        details=details or {},
        request_id=request_id,
        iteration=iteration
    )
    await execution_stream_manager.emit_event(event)
    logger.debug(f"Execution event: {phase} -> {status}")


def reset_execution_state(request_id: Optional[str] = None):
    """Reset execution state for a new request."""
    execution_stream_manager.reset(request_id)


@router.get("/execution/stream")
async def execution_stream(request: Request):
    """
    SSE endpoint for real-time execution status updates.
    
    Streams execution events as Server-Sent Events (SSE).
    Connect to this endpoint to monitor agent execution in real-time.
    
    Event format:
    ```
    event: execution_status
    data: {"phase": "LLM_CALL", "status": "active", "details": {...}, "metrics": {...}}
    ```
    """
    client_queue = await execution_stream_manager.register_client()
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            initial_state = execution_stream_manager.get_state()
            yield f"event: initial_state\ndata: {json.dumps(initial_state)}\n\n"
            
            while True:
                if await request.is_disconnected():
                    break
                
                try:
                    event_data = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                    yield f"event: execution_status\ndata: {json.dumps(event_data)}\n\n"
                except asyncio.TimeoutError:
                    yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"
        finally:
            await execution_stream_manager.unregister_client(client_queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/execution/state")
async def get_execution_state():
    """Get current execution state (non-streaming)."""
    return execution_stream_manager.get_state()


@router.post("/execution/reset")
async def reset_execution(request_id: Optional[str] = None):
    """Reset execution state for a new request."""
    reset_execution_state(request_id)
    return {"status": "reset", "request_id": request_id}
