"""
Human-in-the-Loop Manager for PHTN.AI Sub-Agent Framework

Manages human approval workflows, input requests, and oversight.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Awaitable
from enum import Enum

from ..observability.otel_logging import get_logger

logger = get_logger(__name__)


class ApprovalStatus(str, Enum):
    """Approval request status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ApprovalType(str, Enum):
    """Types of approval requests."""
    TOOL_EXECUTION = "tool_execution"
    OUTPUT_REVIEW = "output_review"
    SENSITIVE_ACTION = "sensitive_action"
    ESCALATION = "escalation"
    DATA_ACCESS = "data_access"
    CUSTOM = "custom"


@dataclass
class ApprovalRequest:
    """Human approval request."""
    id: str
    type: ApprovalType
    title: str
    description: str
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    
    def __post_init__(self):
        if not self.expires_at:
            expires = datetime.utcnow() + timedelta(seconds=self.timeout_seconds)
            self.expires_at = expires.isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "metadata": self.metadata,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.value,
        }
    
    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return datetime.utcnow() > expires.replace(tzinfo=None)


@dataclass
class ApprovalResponse:
    """Human approval response."""
    request_id: str
    status: ApprovalStatus
    approved_by: Optional[str] = None
    reason: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    responded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "reason": self.reason,
            "modifications": self.modifications,
            "responded_at": self.responded_at,
        }


@dataclass
class InputRequest:
    """Human input request."""
    id: str
    prompt: str
    input_type: str = "text"
    options: Optional[List[str]] = None
    default_value: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "input_type": self.input_type,
            "options": self.options,
            "default_value": self.default_value,
            "validation": self.validation,
            "context": self.context,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "status": self.status,
        }


@dataclass
class InputResponse:
    """Human input response."""
    request_id: str
    value: Any
    provided_by: Optional[str] = None
    responded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "value": self.value,
            "provided_by": self.provided_by,
            "responded_at": self.responded_at,
        }


@dataclass
class HumanLoopConfig:
    """Human-in-the-loop configuration."""
    enabled: bool = False
    default_timeout_seconds: int = 300
    require_approval_for_tools: List[str] = field(default_factory=list)
    require_approval_for_actions: List[str] = field(default_factory=list)
    auto_approve_low_risk: bool = False
    escalation_enabled: bool = True
    escalation_timeout_seconds: int = 600
    notification_channels: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]
InputCallback = Callable[[InputRequest], Awaitable[InputResponse]]


class HumanLoopManager:
    """
    Manages human-in-the-loop workflows.
    
    Features:
    - Tool execution approval
    - Output review
    - Human input requests
    - Escalation workflows
    - Timeout handling
    - Webhook notifications
    """
    
    def __init__(self, config: Optional[HumanLoopConfig] = None):
        """
        Initialize HumanLoopManager.
        
        Args:
            config: Human loop configuration
        """
        self.config = config or HumanLoopConfig()
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        self._pending_inputs: Dict[str, InputRequest] = {}
        self._approval_callbacks: List[ApprovalCallback] = []
        self._input_callbacks: List[InputCallback] = []
        self._responses: Dict[str, Any] = {}
        self._response_events: Dict[str, asyncio.Event] = {}
        
        logger.info(f"HumanLoopManager initialized (enabled: {self.config.enabled})")
    
    def register_approval_callback(self, callback: ApprovalCallback):
        """Register a callback for approval requests."""
        self._approval_callbacks.append(callback)
    
    def register_input_callback(self, callback: InputCallback):
        """Register a callback for input requests."""
        self._input_callbacks.append(callback)
    
    async def request_approval(
        self,
        type: ApprovalType,
        title: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ApprovalResponse:
        """
        Request human approval.
        
        Args:
            type: Type of approval
            title: Request title
            description: Request description
            context: Additional context
            timeout_seconds: Timeout override
            
        Returns:
            Approval response
        """
        if not self.config.enabled:
            return ApprovalResponse(
                request_id="auto-approved",
                status=ApprovalStatus.APPROVED,
                reason="Human loop disabled - auto-approved",
            )
        
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            type=type,
            title=title,
            description=description,
            context=context or {},
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds,
        )
        
        self._pending_approvals[request.id] = request
        self._response_events[request.id] = asyncio.Event()
        
        logger.info(f"Approval requested: {request.id} - {title}")
        
        await self._notify_approval_request(request)
        
        try:
            await asyncio.wait_for(
                self._response_events[request.id].wait(),
                timeout=request.timeout_seconds,
            )
            
            response = self._responses.get(request.id)
            if response:
                return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Approval request timed out: {request.id}")
            request.status = ApprovalStatus.TIMEOUT
        
        finally:
            self._pending_approvals.pop(request.id, None)
            self._response_events.pop(request.id, None)
            self._responses.pop(request.id, None)
        
        return ApprovalResponse(
            request_id=request.id,
            status=ApprovalStatus.TIMEOUT,
            reason="Request timed out",
        )
    
    async def request_input(
        self,
        prompt: str,
        input_type: str = "text",
        options: Optional[List[str]] = None,
        default_value: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> InputResponse:
        """
        Request human input.
        
        Args:
            prompt: Input prompt
            input_type: Type of input (text, select, confirm, etc.)
            options: Options for select type
            default_value: Default value
            context: Additional context
            timeout_seconds: Timeout override
            
        Returns:
            Input response
        """
        if not self.config.enabled:
            return InputResponse(
                request_id="auto-input",
                value=default_value,
                provided_by="system",
            )
        
        request = InputRequest(
            id=str(uuid.uuid4()),
            prompt=prompt,
            input_type=input_type,
            options=options,
            default_value=default_value,
            context=context or {},
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds,
        )
        
        self._pending_inputs[request.id] = request
        self._response_events[request.id] = asyncio.Event()
        
        logger.info(f"Input requested: {request.id} - {prompt}")
        
        await self._notify_input_request(request)
        
        try:
            await asyncio.wait_for(
                self._response_events[request.id].wait(),
                timeout=request.timeout_seconds,
            )
            
            response = self._responses.get(request.id)
            if response:
                return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Input request timed out: {request.id}")
            request.status = "timeout"
        
        finally:
            self._pending_inputs.pop(request.id, None)
            self._response_events.pop(request.id, None)
            self._responses.pop(request.id, None)
        
        return InputResponse(
            request_id=request.id,
            value=default_value,
            provided_by="timeout",
        )
    
    async def submit_approval(
        self,
        request_id: str,
        approved: bool,
        approved_by: Optional[str] = None,
        reason: Optional[str] = None,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Submit approval response.
        
        Args:
            request_id: Request ID
            approved: Whether approved
            approved_by: Approver identifier
            reason: Approval/rejection reason
            modifications: Any modifications to the request
            
        Returns:
            True if submission successful
        """
        if request_id not in self._pending_approvals:
            logger.warning(f"Approval request not found: {request_id}")
            return False
        
        request = self._pending_approvals[request_id]
        
        response = ApprovalResponse(
            request_id=request_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
            approved_by=approved_by,
            reason=reason,
            modifications=modifications,
        )
        
        request.status = response.status
        self._responses[request_id] = response
        
        if request_id in self._response_events:
            self._response_events[request_id].set()
        
        logger.info(f"Approval submitted: {request_id} - {response.status.value}")
        return True
    
    async def submit_input(
        self,
        request_id: str,
        value: Any,
        provided_by: Optional[str] = None,
    ) -> bool:
        """
        Submit input response.
        
        Args:
            request_id: Request ID
            value: Input value
            provided_by: Provider identifier
            
        Returns:
            True if submission successful
        """
        if request_id not in self._pending_inputs:
            logger.warning(f"Input request not found: {request_id}")
            return False
        
        request = self._pending_inputs[request_id]
        
        response = InputResponse(
            request_id=request_id,
            value=value,
            provided_by=provided_by,
        )
        
        request.status = "completed"
        self._responses[request_id] = response
        
        if request_id in self._response_events:
            self._response_events[request_id].set()
        
        logger.info(f"Input submitted: {request_id}")
        return True
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            req for req in self._pending_approvals.values()
            if req.status == ApprovalStatus.PENDING and not req.is_expired
        ]
    
    def get_pending_inputs(self) -> List[InputRequest]:
        """Get all pending input requests."""
        return [
            req for req in self._pending_inputs.values()
            if req.status == "pending"
        ]
    
    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires approval."""
        return tool_name in self.config.require_approval_for_tools
    
    def requires_action_approval(self, action: str) -> bool:
        """Check if an action requires approval."""
        return action in self.config.require_approval_for_actions
    
    async def _notify_approval_request(self, request: ApprovalRequest):
        """Notify about approval request via callbacks and webhooks."""
        for callback in self._approval_callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Approval callback failed: {e}")
        
        if self.config.webhook_url:
            await self._send_webhook({
                "type": "approval_request",
                "request": request.to_dict(),
            })
    
    async def _notify_input_request(self, request: InputRequest):
        """Notify about input request via callbacks and webhooks."""
        for callback in self._input_callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Input callback failed: {e}")
        
        if self.config.webhook_url:
            await self._send_webhook({
                "type": "input_request",
                "request": request.to_dict(),
            })
    
    async def _send_webhook(self, payload: Dict[str, Any]):
        """Send webhook notification."""
        if not self.config.webhook_url:
            return
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Webhook failed: {resp.status}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")
