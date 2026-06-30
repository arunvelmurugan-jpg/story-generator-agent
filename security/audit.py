"""
Audit Logger for PHTN.AI Sub-Agent Framework

Provides comprehensive audit logging for security and compliance.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: str
    timestamp: str
    actor: str
    action: str
    resource: str
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditLogger:
    """
    Audit logging for security and compliance.
    
    Features:
    - Structured audit events
    - Multiple output destinations
    - Event filtering
    - Compliance reporting
    """
    
    EVENT_TYPES = {
        "AGENT_EXECUTE": "agent.execute",
        "TOOL_CALL": "tool.call",
        "ACCESS_GRANTED": "access.granted",
        "ACCESS_DENIED": "access.denied",
        "CONFIG_CHANGE": "config.change",
        "GUARDRAIL_TRIGGERED": "guardrail.triggered",
        "ERROR": "error",
    }
    
    def __init__(
        self,
        enabled: bool = True,
        log_level: str = "INFO",
        destinations: Optional[List[str]] = None,
    ):
        """
        Initialize AuditLogger.
        
        Args:
            enabled: Enable audit logging
            log_level: Log level
            destinations: Output destinations
        """
        self.enabled = enabled
        self.log_level = log_level
        self.destinations = destinations or ["log"]
        
        self._events: List[AuditEvent] = []
        self._max_buffer = 1000
        
        self._audit_logger = logging.getLogger("phtnai.audit")
        self._audit_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        logger.debug("AuditLogger initialized")
    
    def log(
        self,
        event_type: str,
        actor: str,
        action: str,
        resource: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditEvent]:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            actor: Who performed the action
            action: What action was performed
            resource: What resource was affected
            outcome: Result of the action
            details: Additional details
            metadata: Event metadata
            
        Returns:
            AuditEvent or None if disabled
        """
        if not self.enabled:
            return None
        
        import uuid
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            actor=actor,
            action=action,
            resource=resource,
            outcome=outcome,
            details=details or {},
            metadata=metadata or {},
        )
        
        self._emit(event)
        
        return event
    
    def log_agent_execution(
        self,
        agent_id: str,
        request_id: str,
        outcome: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditEvent]:
        """Log agent execution event."""
        return self.log(
            event_type=self.EVENT_TYPES["AGENT_EXECUTE"],
            actor=agent_id,
            action="execute",
            resource=f"request:{request_id}",
            outcome=outcome,
            details={
                "duration_ms": duration_ms,
                **(details or {}),
            },
        )
    
    def log_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditEvent]:
        """Log tool call event."""
        return self.log(
            event_type=self.EVENT_TYPES["TOOL_CALL"],
            actor=agent_id,
            action="call",
            resource=f"tool:{tool_name}",
            outcome=outcome,
            details=details,
        )
    
    def log_access_decision(
        self,
        subject: str,
        resource: str,
        action: str,
        allowed: bool,
        reason: str,
    ) -> Optional[AuditEvent]:
        """Log access control decision."""
        event_type = self.EVENT_TYPES["ACCESS_GRANTED"] if allowed else self.EVENT_TYPES["ACCESS_DENIED"]
        return self.log(
            event_type=event_type,
            actor=subject,
            action=action,
            resource=resource,
            outcome="allowed" if allowed else "denied",
            details={"reason": reason},
        )
    
    def log_guardrail_triggered(
        self,
        agent_id: str,
        rule_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[AuditEvent]:
        """Log guardrail trigger event."""
        return self.log(
            event_type=self.EVENT_TYPES["GUARDRAIL_TRIGGERED"],
            actor=agent_id,
            action=action,
            resource=f"guardrail:{rule_id}",
            outcome="triggered",
            details=details,
        )
    
    def _emit(self, event: AuditEvent):
        """Emit audit event to destinations."""
        if "log" in self.destinations:
            self._audit_logger.info(event.to_json())
        
        if "buffer" in self.destinations:
            self._events.append(event)
            while len(self._events) > self._max_buffer:
                self._events.pop(0)
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Get buffered audit events with optional filtering."""
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if actor:
            events = [e for e in events if e.actor == actor]
        
        if since:
            since_str = since.isoformat()
            events = [e for e in events if e.timestamp >= since_str]
        
        return events[-limit:]
