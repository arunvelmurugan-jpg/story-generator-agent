"""
X-PHTN-Agent-ID Header Management for PHTN.AI Sub-Agent Framework

Manages the 18-part distributed tracing header.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid

logger = logging.getLogger(__name__)


@dataclass
class XPHTNAgentIDHeader:
    """
    X-PHTN-Agent-ID header with 18 parts for distributed tracing.
    
    Format: PART1|PART2|...|PART18
    
    Parts:
    1. TRACE-ID: Unique trace identifier
    2. SPAN-ID: Current span identifier
    3. PARENT-SPAN-ID: Parent span identifier
    4. AGENT-TYPE: Type of agent (SUPER_AGENT, SUB_AGENT, TOOL_AGENT, etc.)
    5. AGENT-ID: Unique agent identifier
    6. AGENT-VERSION: Agent version
    7. EXECUTION-ID: Current execution identifier
    8. SESSION-ID: Session identifier
    9. USER-ID: User identifier
    10. TENANT-ID: Tenant identifier
    11. ENVIRONMENT: Deployment environment
    12. REGION: Deployment region
    13. TIMESTAMP: ISO timestamp
    14. SEQUENCE-NUM: Sequence number in trace
    15. HOP-COUNT: Number of hops in trace
    16. PRIORITY: Request priority
    17. FLAGS: Trace flags
    18. CUSTOM: Custom data (JSON encoded)
    """
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: str = ""
    agent_type: str = "SUB_AGENT"
    agent_id: str = ""
    agent_version: str = "1.0.0"
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    user_id: str = ""
    tenant_id: str = ""
    environment: str = "development"
    region: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    sequence_num: int = 1
    hop_count: int = 1
    priority: str = "NORMAL"
    flags: str = "00"
    custom: str = ""
    
    AGENT_TYPES = [
        "SUPER_AGENT",
        "SUB_AGENT",
        "TOOL_AGENT",
        "ORCHESTRATOR",
        "ROUTER",
        "EVALUATOR",
        "MEMORY_AGENT",
        "RETRIEVAL_AGENT",
    ]
    
    PRIORITIES = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
    
    def to_header(self) -> str:
        """Convert to header string."""
        parts = [
            self.trace_id,
            self.span_id,
            self.parent_span_id,
            self.agent_type,
            self.agent_id,
            self.agent_version,
            self.execution_id,
            self.session_id,
            self.user_id,
            self.tenant_id,
            self.environment,
            self.region,
            self.timestamp,
            str(self.sequence_num),
            str(self.hop_count),
            self.priority,
            self.flags,
            self.custom,
        ]
        return "|".join(parts)
    
    @classmethod
    def from_header(cls, header: str) -> "XPHTNAgentIDHeader":
        """Parse header string."""
        parts = header.split("|")
        
        if len(parts) < 18:
            parts.extend([""] * (18 - len(parts)))
        
        return cls(
            trace_id=parts[0] or str(uuid.uuid4()),
            span_id=parts[1] or str(uuid.uuid4())[:16],
            parent_span_id=parts[2],
            agent_type=parts[3] or "SUB_AGENT",
            agent_id=parts[4],
            agent_version=parts[5] or "1.0.0",
            execution_id=parts[6] or str(uuid.uuid4()),
            session_id=parts[7],
            user_id=parts[8],
            tenant_id=parts[9],
            environment=parts[10] or "development",
            region=parts[11],
            timestamp=parts[12] or datetime.utcnow().isoformat() + "Z",
            sequence_num=int(parts[13]) if parts[13].isdigit() else 1,
            hop_count=int(parts[14]) if parts[14].isdigit() else 1,
            priority=parts[15] or "NORMAL",
            flags=parts[16] or "00",
            custom=parts[17],
        )
    
    def create_child_span(self, agent_type: Optional[str] = None) -> "XPHTNAgentIDHeader":
        """Create child span header."""
        return XPHTNAgentIDHeader(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            agent_type=agent_type or self.agent_type,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            execution_id=self.execution_id,
            session_id=self.session_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            environment=self.environment,
            region=self.region,
            timestamp=datetime.utcnow().isoformat() + "Z",
            sequence_num=self.sequence_num + 1,
            hop_count=self.hop_count + 1,
            priority=self.priority,
            flags=self.flags,
            custom=self.custom,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "environment": self.environment,
            "region": self.region,
            "timestamp": self.timestamp,
            "sequence_num": self.sequence_num,
            "hop_count": self.hop_count,
            "priority": self.priority,
            "flags": self.flags,
            "custom": self.custom,
        }
    
    @classmethod
    def create(
        cls,
        agent_id: str,
        agent_type: str = "SUB_AGENT",
        session_id: str = "",
        user_id: str = "",
        tenant_id: str = "",
        environment: str = "development",
        parent_header: Optional["XPHTNAgentIDHeader"] = None,
    ) -> "XPHTNAgentIDHeader":
        """Create a new header."""
        if parent_header:
            return parent_header.create_child_span(agent_type)
        
        return cls(
            agent_type=agent_type,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            environment=environment,
        )
