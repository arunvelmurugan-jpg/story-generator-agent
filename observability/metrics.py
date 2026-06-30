"""
Metrics Collector for PHTN.AI Sub-Agent Framework

Collects and exports metrics for monitoring.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"


class MetricsCollector:
    """
    Metrics collection with OpenTelemetry compatibility.
    
    Features:
    - Counter, gauge, histogram metrics
    - Label support
    - Export to various backends
    - Aggregation
    """
    
    def __init__(
        self,
        service_name: str,
        enabled: bool = True,
        exporter_type: str = "console",
        exporter_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MetricsCollector.
        
        Args:
            service_name: Service name for metrics
            enabled: Enable metrics collection
            exporter_type: Exporter type
            exporter_config: Exporter configuration
        """
        self.service_name = service_name
        self.enabled = enabled
        self.exporter_type = exporter_type
        self.exporter_config = exporter_config or {}
        
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        self._otel_meter = None
        if enabled:
            self._init_otel()
        
        logger.debug(f"MetricsCollector initialized: {service_name}")
    
    def _init_otel(self):
        """Initialize OpenTelemetry meter."""
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            
            provider = MeterProvider()
            metrics.set_meter_provider(provider)
            self._otel_meter = metrics.get_meter(self.service_name)
            
        except ImportError:
            logger.warning("OpenTelemetry not installed, using fallback metrics")
            self._otel_meter = None
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a counter."""
        if not self.enabled:
            return
        
        key = self._make_key(name, labels)
        self._counters[key] += value
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Set a gauge value."""
        if not self.enabled:
            return
        
        key = self._make_key(name, labels)
        self._gauges[key] = value
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Record a histogram value."""
        if not self.enabled:
            return
        
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Create a timer context manager."""
        return Timer(self, name, labels)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create metric key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                    "avg": sum(v) / len(v) if v else 0,
                }
                for k, v in self._histograms.items()
            },
        }
    
    def record_execution(
        self,
        agent_id: str,
        duration_ms: float,
        success: bool,
        pattern: str,
    ):
        """Record agent execution metrics."""
        labels = {
            "agent_id": agent_id,
            "pattern": pattern,
            "success": str(success).lower(),
        }
        
        self.increment("agent_executions_total", labels=labels)
        self.histogram("agent_execution_duration_ms", duration_ms, labels={"agent_id": agent_id})
        
        if success:
            self.increment("agent_executions_success", labels={"agent_id": agent_id})
        else:
            self.increment("agent_executions_failed", labels={"agent_id": agent_id})
    
    def record_llm_call(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        duration_ms: float,
    ):
        """Record LLM call metrics."""
        labels = {"provider": provider, "model": model}
        
        self.increment("llm_calls_total", labels=labels)
        self.increment("llm_tokens_input", tokens_in, labels=labels)
        self.increment("llm_tokens_output", tokens_out, labels=labels)
        self.histogram("llm_call_duration_ms", duration_ms, labels=labels)
    
    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
    ):
        """Record tool call metrics."""
        labels = {"tool": tool_name, "success": str(success).lower()}
        
        self.increment("tool_calls_total", labels=labels)
        self.histogram("tool_call_duration_ms", duration_ms, labels={"tool": tool_name})


class Timer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.histogram(self.name, duration_ms, self.labels)
