"""
Tracer for PHTN.AI Sub-Agent Framework

OpenTelemetry-based distributed tracing.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from .header import XPHTNAgentIDHeader

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Trace span."""
    span_id: str
    name: str
    trace_id: str
    parent_span_id: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    end_time: Optional[str] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def end(self, status: str = "OK"):
        """End the span."""
        self.end_time = datetime.utcnow().isoformat() + "Z"
        self.status = status
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "attributes": attributes or {},
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """
    Distributed tracing with OpenTelemetry compatibility.
    
    Features:
    - Span creation and management
    - Context propagation
    - X-PHTN-Agent-ID integration
    - Export to various backends
    """
    
    def __init__(
        self,
        service_name: str,
        enabled: bool = True,
        exporter_type: str = "console",
        exporter_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Tracer.
        
        Args:
            service_name: Service name for traces
            enabled: Enable tracing
            exporter_type: Exporter type (console, otlp, jaeger)
            exporter_config: Exporter configuration
        """
        self.service_name = service_name
        self.enabled = enabled
        self.exporter_type = exporter_type
        self.exporter_config = exporter_config or {}
        
        self._spans: Dict[str, Span] = {}
        self._current_span: Optional[Span] = None
        
        self._otel_tracer = None
        if enabled:
            self._init_otel()
        
        logger.debug(f"Tracer initialized: {service_name}")
    
    def _init_otel(self):
        """Initialize OpenTelemetry tracer."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
            
            provider = TracerProvider()
            
            if self.exporter_type == "console":
                processor = BatchSpanProcessor(ConsoleSpanExporter())
                provider.add_span_processor(processor)
            elif self.exporter_type == "otlp":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                endpoint = self.exporter_config.get("endpoint", "localhost:4317")
                processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
                provider.add_span_processor(processor)
            
            trace.set_tracer_provider(provider)
            self._otel_tracer = trace.get_tracer(self.service_name)
            
        except ImportError:
            logger.warning("OpenTelemetry not installed, using fallback tracer")
            self._otel_tracer = None
    
    @contextmanager
    def start_span(
        self,
        name: str,
        parent_header: Optional[XPHTNAgentIDHeader] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """
        Start a new span.
        
        Args:
            name: Span name
            parent_header: Parent X-PHTN-Agent-ID header
            attributes: Span attributes
            
        Yields:
            Span
        """
        if not self.enabled:
            span = Span(
                span_id="disabled",
                name=name,
                trace_id="disabled",
            )
            yield span
            return
        
        import uuid
        
        trace_id = parent_header.trace_id if parent_header else str(uuid.uuid4())
        parent_span_id = parent_header.span_id if parent_header else None
        
        span = Span(
            span_id=str(uuid.uuid4())[:16],
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        
        self._spans[span.span_id] = span
        previous_span = self._current_span
        self._current_span = span
        
        try:
            yield span
            span.end("OK")
        except Exception as e:
            span.end("ERROR")
            span.set_attribute("error.message", str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise
        finally:
            self._current_span = previous_span
            self._export_span(span)
    
    def _export_span(self, span: Span):
        """Export span to backend."""
        if self.exporter_type == "console":
            logger.debug(f"Span: {span.name} [{span.status}] {span.span_id}")
    
    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self._current_span
    
    def create_header_from_span(
        self,
        span: Span,
        agent_id: str,
        agent_type: str = "SUB_AGENT",
    ) -> XPHTNAgentIDHeader:
        """Create X-PHTN-Agent-ID header from span."""
        return XPHTNAgentIDHeader(
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_span_id=span.parent_span_id or "",
            agent_type=agent_type,
            agent_id=agent_id,
        )
