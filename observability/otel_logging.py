"""
OTEL Logging Configuration for PHTN.AI Sub-Agent Framework

Provides OpenTelemetry-compatible structured logging with full X-PHTN-Agent-ID support.
Designed for seamless integration with phtnai-ops-metrics (FinOps, TechOps, AIOps).

X-PHTN-Agent-ID Format (18 parts):
  TenantGroup|Tenant|Domain|Team|Project|User|AgentType|SuperAgentId|AgentId|AgentInstanceId|
  AgentName|Environment|CorrelationId|TraceId|SpanId|HopCount|CapabilityId|SkillId

This module ensures all logs include fields required by phtnai-ops-metrics filters:
- phtn_tenant_group_id: Tenant group identifier
- phtn_tenant_id: Tenant identifier
- phtn_domain_id: Domain identifier
- phtn_team_id: Team identifier
- phtn_project_id: Project identifier
- phtn_user_id: User identifier
- phtn_agent_type: Agent type (SUB_AGENT, SUPER_AGENT, etc.)
- phtn_super_agent_id: Parent super agent ID
- phtn_sub_agent_id: Sub agent ID
- phtn_super_agent_instance_id: Super agent instance ID
- phtn_sub_agent_instance_id: Sub agent instance ID
- phtn_app_name: Application name
- phtn_environment: Deployment environment
- phtn_correlation_id: Request correlation ID
- phtn_trace_id: OpenTelemetry trace ID
- phtn_span_id: OpenTelemetry span ID
- phtn_agent_capability: Capability being executed
- phtn_agent_skill: Skill being used
- log_type: Log type (sub_agent, llm_gateway, api_gateway)
- model: LLM model name (for LLM calls)
- provider: LLM provider (for LLM calls)
- prompt_tokens, completion_tokens, total_tokens: Token usage
- cost: Cost in USD
- latency_ms: Request latency
"""

import logging
import json
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# Core trace context variables
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='no-correlation-id')
phtn_agent_id_var: ContextVar[str] = ContextVar('phtn_agent_id', default='no-phtn-agent-id')
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')
span_id_var: ContextVar[str] = ContextVar('span_id', default='')
agent_id_var: ContextVar[str] = ContextVar('agent_id', default='')
tenant_id_var: ContextVar[str] = ContextVar('tenant_id', default='')
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
default_model_var: ContextVar[str] = ContextVar('default_model', default='')
pricing_var: ContextVar[Dict[str, Dict[str, float]]] = ContextVar('pricing', default={})

# Extended context variables for phtnai-ops-metrics compatibility
tenant_group_id_var: ContextVar[str] = ContextVar('tenant_group_id', default='unknown')
domain_id_var: ContextVar[str] = ContextVar('domain_id', default='unknown')
team_id_var: ContextVar[str] = ContextVar('team_id', default='unknown')
project_id_var: ContextVar[str] = ContextVar('project_id', default='unknown')
user_id_var: ContextVar[str] = ContextVar('user_id', default='unknown')
agent_type_var: ContextVar[str] = ContextVar('agent_type', default='SUB_AGENT')
super_agent_id_var: ContextVar[str] = ContextVar('super_agent_id', default='unknown')
sub_agent_id_var: ContextVar[str] = ContextVar('sub_agent_id', default='unknown')
super_agent_instance_id_var: ContextVar[str] = ContextVar('super_agent_instance_id', default='unknown')
sub_agent_instance_id_var: ContextVar[str] = ContextVar('sub_agent_instance_id', default='unknown')
app_name_var: ContextVar[str] = ContextVar('app_name', default='phtnai-subagent')
environment_var: ContextVar[str] = ContextVar('environment', default='development')
capability_id_var: ContextVar[str] = ContextVar('capability_id', default='unknown')
skill_id_var: ContextVar[str] = ContextVar('skill_id', default='unknown')
agent_name_var: ContextVar[str] = ContextVar('agent_name', default='unknown')
request_method_var: ContextVar[str] = ContextVar("request_method", default="")
request_path_var: ContextVar[str] = ContextVar("request_path", default="")


DEFAULT_PRICING_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o-audio-preview": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o1-preview": {"input": 0.015, "output": 0.06},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "default": {"input": 0.001, "output": 0.002},
}


class _SkipBannerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True

        stripped = (message or "").strip()
        if len(stripped) >= 10 and set(stripped) == {"="}:
            return False

        return True


def _format_log_timestamp(record: logging.LogRecord) -> str:
    dt = datetime.fromtimestamp(record.created)
    return dt.strftime("%Y-%m-%d %H:%M:%S,") + f"{int(record.msecs):03d}"


def _normalize_pricing(pricing: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(pricing, dict):
        return {}

    normalized: Dict[str, Dict[str, float]] = {}
    for model, rates in pricing.items():
        if not isinstance(model, str) or not isinstance(rates, dict):
            continue
        input_rate = rates.get("input")
        output_rate = rates.get("output")
        if isinstance(input_rate, (int, float)) and isinstance(output_rate, (int, float)):
            normalized[model] = {"input": float(input_rate), "output": float(output_rate)}
    return normalized


def _calculate_token_cost_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    if not model:
        return 0.0

    merged_pricing = {
        **DEFAULT_PRICING_PER_1K_TOKENS,
        **(_normalize_pricing(pricing) if pricing else {}),
    }

    rates = merged_pricing.get(model) or merged_pricing.get(model.lower()) or merged_pricing["default"]
    input_rate = rates.get("input", merged_pricing["default"]["input"])
    output_rate = rates.get("output", merged_pricing["default"]["output"])

    input_cost = (max(prompt_tokens, 0) / 1000.0) * float(input_rate)
    output_cost = (max(completion_tokens, 0) / 1000.0) * float(output_rate)
    return float(input_cost + output_cost)


def _calculate_token_cost_components_usd(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    if not model:
        return {"input_cost": 0.0, "output_cost": 0.0}

    merged_pricing = {
        **DEFAULT_PRICING_PER_1K_TOKENS,
        **(_normalize_pricing(pricing) if pricing else {}),
    }

    rates = merged_pricing.get(model) or merged_pricing.get(model.lower()) or merged_pricing["default"]
    input_rate = rates.get("input", merged_pricing["default"]["input"])
    output_rate = rates.get("output", merged_pricing["default"]["output"])

    input_cost = (max(prompt_tokens, 0) / 1000.0) * float(input_rate)
    output_cost = (max(completion_tokens, 0) / 1000.0) * float(output_rate)
    return {"input_cost": float(input_cost), "output_cost": float(output_cost)}


@dataclass
class PHtnAgentIdParts:
    """Parsed X-PHTN-Agent-ID header parts."""
    tenant_group: str = ""
    tenant: str = ""
    domain: str = ""
    team: str = ""
    project: str = ""
    user: str = ""
    agent_type: str = "SUB_AGENT"
    super_agent_id: str = ""
    agent_id: str = ""
    agent_instance_id: str = ""
    agent_name: str = ""
    environment: str = "development"
    correlation_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    hop_count: str = "1"
    capability_id: str = ""
    skill_id: str = ""
    
    def to_header(self) -> str:
        """Convert to X-PHTN-Agent-ID header string."""
        return "|".join([
            self.tenant_group,
            self.tenant,
            self.domain,
            self.team,
            self.project,
            self.user,
            self.agent_type,
            self.super_agent_id,
            self.agent_id,
            self.agent_instance_id,
            self.agent_name,
            self.environment,
            self.correlation_id,
            self.trace_id,
            self.span_id,
            self.hop_count,
            self.capability_id,
            self.skill_id,
        ])
    
    @classmethod
    def from_header(cls, header: str) -> "PHtnAgentIdParts":
        """Parse X-PHTN-Agent-ID header string."""
        parts = header.split("|")
        if len(parts) < 18:
            parts.extend([""] * (18 - len(parts)))
        
        return cls(
            tenant_group=parts[0],
            tenant=parts[1],
            domain=parts[2],
            team=parts[3],
            project=parts[4],
            user=parts[5],
            agent_type=parts[6],
            super_agent_id=parts[7],
            agent_id=parts[8],
            agent_instance_id=parts[9],
            agent_name=parts[10],
            environment=parts[11],
            correlation_id=parts[12],
            trace_id=parts[13],
            span_id=parts[14],
            hop_count=parts[15],
            capability_id=parts[16],
            skill_id=parts[17],
        )
    
    @classmethod
    def from_config(cls, config: Any, correlation_id: str = "", trace_id: str = "", span_id: str = "") -> "PHtnAgentIdParts":
        """Create from AgentConfiguration."""
        return cls(
            tenant_group=config.tenant or "",
            tenant=config.tenant or "",
            domain=config.domain.industry if config.domain else "",
            team=config.metadata.labels.get("team", "") if config.metadata and config.metadata.labels else "",
            project="",
            user="",
            agent_type=config.observability_config.x_phtn_agent_id.get("agent_type", "SUB_AGENT") if config.observability_config and config.observability_config.x_phtn_agent_id else "SUB_AGENT",
            super_agent_id="",
            agent_id=config.agent_id,
            agent_instance_id=config.instance_id or "",
            agent_name=config.name,
            environment=config.deployment_metadata.environment if config.deployment_metadata else "development",
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            hop_count="1",
            capability_id="",
            skill_id="",
        )


class OTELJsonFormatter(logging.Formatter):
    """
    JSON formatter for OTEL-compatible structured logging.
    
    Outputs logs in a format compatible with:
    - OpenTelemetry collectors
    - phtnai-ops-metrics (FinOps, TechOps, AIOps filters)
    
    All phtn_* fields are included at the top level for direct filtering.
    """
    
    def __init__(self, agent_name: str = "phtnai-subagent", service_name: str = "phtnai-subagent-framework"):
        super().__init__()
        self.agent_name = agent_name
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        phtn_agent_id = getattr(record, "phtn_agent_id", phtn_agent_id_var.get())
        correlation_id = getattr(record, "correlation_id", correlation_id_var.get()) or "no-correlation-id"
        agent_id = getattr(record, "agent_id", agent_id_var.get()) or "unknown"
        agent_name = getattr(record, "agent_name", agent_name_var.get()) or self.agent_name

        x_phtn_id = phtn_agent_id
        if not x_phtn_id or x_phtn_id in {"no-phtn-agent-id", "no-x-phtn-id"}:
            x_phtn_id = "no-x-phtn-id"

        extra_data = getattr(record, "extra_data", {}) or {}
        log_message: Dict[str, Any] = {"message": record.getMessage()}

        model = extra_data.get("model") or default_model_var.get()
        prompt_tokens = extra_data.get("prompt_tokens", 0) or 0
        completion_tokens = extra_data.get("completion_tokens", 0) or 0
        total_tokens = extra_data.get("total_tokens")
        if isinstance(total_tokens, (int, float)) and float(total_tokens) <= 0 and (prompt_tokens or completion_tokens):
            total_tokens = None

        input_cost = extra_data.get("input_cost")
        output_cost = extra_data.get("output_cost")
        if model and ((input_cost is None) or (output_cost is None)):
            split_cost = _calculate_token_cost_components_usd(
                model=str(model),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                pricing=pricing_var.get(),
            )
            if input_cost is None:
                input_cost = split_cost["input_cost"]
            if output_cost is None:
                output_cost = split_cost["output_cost"]

        cost = extra_data.get("cost")
        if cost is None or (isinstance(cost, (int, float)) and float(cost) == 0.0):
            if isinstance(input_cost, (int, float)) or isinstance(output_cost, (int, float)):
                cost = float(input_cost or 0.0) + float(output_cost or 0.0)
            elif model and (prompt_tokens or completion_tokens):
                cost = _calculate_token_cost_usd(
                    model=str(model),
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    pricing=pricing_var.get(),
                )

        merged_extra = dict(extra_data)
        if "endpoint" in merged_extra and "path" not in merged_extra:
            merged_extra["path"] = merged_extra.pop("endpoint")

        merged_extra.setdefault("method", request_method_var.get() or "unknown")
        merged_extra.setdefault("path", request_path_var.get() or "unknown")
        if merged_extra.get("method") is None:
            merged_extra["method"] = request_method_var.get() or "unknown"
        if merged_extra.get("path") is None:
            merged_extra["path"] = request_path_var.get() or "unknown"

        merged_extra.setdefault("input", getattr(record, "input", None))
        merged_extra.setdefault("output", getattr(record, "output", None))
        if merged_extra.get("input") is None:
            merged_extra["input"] = extra_data.get("input")
        if merged_extra.get("output") is None:
            merged_extra["output"] = extra_data.get("output")

        # ── Prevent double-escaping: if input/output are JSON strings,
        #    parse them into dicts so the final json.dumps() serializes once. ──
        for _io_key in ("input", "output"):
            _io_val = merged_extra.get(_io_key)
            if isinstance(_io_val, str) and _io_val.strip().startswith(("{", "[")):
                try:
                    merged_extra[_io_key] = json.loads(_io_val)
                except (json.JSONDecodeError, ValueError):
                    pass  # keep as string

        merged_extra.setdefault("correlation_id", correlation_id)
        merged_extra.setdefault("agent_id", agent_id)
        if model:
            merged_extra.setdefault("model", model)
        merged_extra.setdefault("prompt_tokens", int(prompt_tokens))
        merged_extra.setdefault("completion_tokens", int(completion_tokens))
        if total_tokens is not None:
            merged_extra.setdefault("total_tokens", total_tokens)
        else:
            merged_extra.setdefault("total_tokens", int(prompt_tokens) + int(completion_tokens))
        merged_extra.setdefault("input_cost", float(input_cost) if input_cost is not None else 0.0)
        merged_extra.setdefault("output_cost", float(output_cost) if output_cost is not None else 0.0)
        merged_extra.setdefault("cost", float(cost) if cost is not None else 0.0)

        log_message.update(merged_extra)

        log_record = {
            "@timestamp": _format_log_timestamp(record),
            "level": record.levelname,
            "agent": agent_id or agent_name,
            "agent_id": agent_id,
            "x_phtn_id": x_phtn_id,
            "correlation_id": correlation_id,
            "log_message": log_message,
        }

        if record.exc_info:
            log_record["log_message"]["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info),
            }

        return json.dumps(log_record, ensure_ascii=False)


class OTELLogRecordFactory:
    """Custom LogRecord factory that adds all tracing fields for phtnai-ops-metrics."""
    
    _original_factory = None
    
    @classmethod
    def install(cls):
        """Install the custom factory."""
        if cls._original_factory is None:
            cls._original_factory = logging.getLogRecordFactory()
            logging.setLogRecordFactory(cls._create_record)
    
    @classmethod
    def _create_record(cls, *args, **kwargs):
        """Create a log record with all tracing fields required by phtnai-ops-metrics."""
        record = cls._original_factory(*args, **kwargs)
        
        # Core trace fields
        record.phtn_agent_id = phtn_agent_id_var.get()
        record.correlation_id = correlation_id_var.get()
        record.trace_id = trace_id_var.get()
        record.span_id = span_id_var.get()
        record.agent_id = agent_id_var.get()
        record.tenant_id = tenant_id_var.get()
        record.request_id = request_id_var.get()
        
        # Extended fields for phtnai-ops-metrics compatibility
        record.tenant_group_id = tenant_group_id_var.get()
        record.domain_id = domain_id_var.get()
        record.team_id = team_id_var.get()
        record.project_id = project_id_var.get()
        record.user_id = user_id_var.get()
        record.agent_type = agent_type_var.get()
        record.super_agent_id = super_agent_id_var.get()
        record.sub_agent_id = sub_agent_id_var.get()
        record.super_agent_instance_id = super_agent_instance_id_var.get()
        record.sub_agent_instance_id = sub_agent_instance_id_var.get()
        record.app_name = app_name_var.get()
        record.environment = environment_var.get()
        record.capability_id = capability_id_var.get()
        record.skill_id = skill_id_var.get()
        record.agent_name = agent_name_var.get()
        
        return record


_configured = False
_agent_name = "phtnai-subagent"
_service_name = "phtnai-subagent-framework"


def configure_otel_logging(
    agent_name: str = "phtnai-subagent",
    service_name: str = "phtnai-subagent-framework",
    log_level: str = "INFO",
    json_format: bool = True
):
    """
    Configure OTEL-compatible logging for the sub-agent framework.
    
    Args:
        agent_name: Name of the agent for logging
        service_name: Service name for OTEL
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format (True) or text format (False)
    """
    global _configured, _agent_name, _service_name
    
    if _configured:
        return
    
    _agent_name = agent_name
    _service_name = service_name
    
    OTELLogRecordFactory.install()
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    handler.addFilter(_SkipBannerFilter())
    
    if json_format:
        handler.setFormatter(OTELJsonFormatter(agent_name, service_name))
    else:
        text_format = (
            '%(asctime)s | %(levelname)-8s | %(name)s | '
            'agent=%(agent_id)s | correlation=%(correlation_id)s | '
            'phtn_id=%(phtn_agent_id)s | %(message)s'
        )
        handler.setFormatter(logging.Formatter(text_format))
    
    root_logger.addHandler(handler)
    
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with OTEL tracing support."""
    return logging.getLogger(name)


def set_trace_context(
    phtn_agent_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    request_id: Optional[str] = None,
    # Extended fields for phtnai-ops-metrics
    tenant_group_id: Optional[str] = None,
    domain_id: Optional[str] = None,
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    super_agent_id: Optional[str] = None,
    sub_agent_id: Optional[str] = None,
    super_agent_instance_id: Optional[str] = None,
    sub_agent_instance_id: Optional[str] = None,
    app_name: Optional[str] = None,
    environment: Optional[str] = None,
    capability_id: Optional[str] = None,
    skill_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    request_method: Optional[str] = None,
    request_path: Optional[str] = None,
):
    """
    Set trace context variables for the current async context.
    
    All fields are used by phtnai-ops-metrics for FinOps, TechOps, and AIOps filtering.
    """
    # Core trace fields
    if phtn_agent_id is not None:
        phtn_agent_id_var.set(phtn_agent_id)
    if correlation_id is not None:
        correlation_id_var.set(correlation_id)
    if trace_id is not None:
        trace_id_var.set(trace_id)
    if span_id is not None:
        span_id_var.set(span_id)
    if agent_id is not None:
        agent_id_var.set(agent_id)
    if tenant_id is not None:
        tenant_id_var.set(tenant_id)
    if request_id is not None:
        request_id_var.set(request_id)
    
    # Extended fields for phtnai-ops-metrics
    if tenant_group_id is not None:
        tenant_group_id_var.set(tenant_group_id)
    if domain_id is not None:
        domain_id_var.set(domain_id)
    if team_id is not None:
        team_id_var.set(team_id)
    if project_id is not None:
        project_id_var.set(project_id)
    if user_id is not None:
        user_id_var.set(user_id)
    if agent_type is not None:
        agent_type_var.set(agent_type)
    if super_agent_id is not None:
        super_agent_id_var.set(super_agent_id)
    if sub_agent_id is not None:
        sub_agent_id_var.set(sub_agent_id)
    if super_agent_instance_id is not None:
        super_agent_instance_id_var.set(super_agent_instance_id)
    if sub_agent_instance_id is not None:
        sub_agent_instance_id_var.set(sub_agent_instance_id)
    if app_name is not None:
        app_name_var.set(app_name)
    if environment is not None:
        environment_var.set(environment)
    if capability_id is not None:
        capability_id_var.set(capability_id)
    if skill_id is not None:
        skill_id_var.set(skill_id)
    if agent_name is not None:
        agent_name_var.set(agent_name)
    if request_method is not None:
        request_method_var.set(request_method)
    if request_path is not None:
        request_path_var.set(request_path)


def set_trace_context_from_config(config: Any):
    """
    Set trace context from AgentConfiguration.
    
    Extracts all fields needed by phtnai-ops-metrics from the agent config.
    """
    set_trace_context(
        agent_id=config.agent_id,
        agent_name=config.name,
        tenant_id=config.tenant,
        tenant_group_id=config.tenant,
        domain_id=config.domain.industry if config.domain else "unknown",
        team_id=config.metadata.labels.get("team", "unknown") if config.metadata and config.metadata.labels else "unknown",
        project_id=config.metadata.labels.get("project", "unknown") if config.metadata and config.metadata.labels else "unknown",
        agent_type=config.observability_config.x_phtn_agent_id.get("agent_type", "SUB_AGENT") if config.observability_config and config.observability_config.x_phtn_agent_id else "SUB_AGENT",
        sub_agent_id=config.agent_id,
        sub_agent_instance_id=config.instance_id or "unknown",
        app_name=config.name,
        environment=config.deployment_metadata.environment if config.deployment_metadata else "development",
    )

    if getattr(config, "llm_config", None) and getattr(config.llm_config, "primary_model", None):
        default_model_var.set(config.llm_config.primary_model)

    if getattr(config, "cost_governance", None) and getattr(config.cost_governance, "pricing", None):
        pricing_var.set(_normalize_pricing(config.cost_governance.pricing))


def get_trace_context() -> Dict[str, str]:
    """Get current trace context with all phtnai-ops-metrics fields."""
    return {
        # Core trace fields
        "phtn_agent_id": phtn_agent_id_var.get(),
        "correlation_id": correlation_id_var.get(),
        "trace_id": trace_id_var.get(),
        "span_id": span_id_var.get(),
        "agent_id": agent_id_var.get(),
        "tenant_id": tenant_id_var.get(),
        "request_id": request_id_var.get(),
        # Extended fields for phtnai-ops-metrics
        "phtn_tenant_group_id": tenant_group_id_var.get(),
        "phtn_tenant_id": tenant_id_var.get(),
        "phtn_domain_id": domain_id_var.get(),
        "phtn_team_id": team_id_var.get(),
        "phtn_project_id": project_id_var.get(),
        "phtn_user_id": user_id_var.get(),
        "phtn_agent_type": agent_type_var.get(),
        "phtn_super_agent_id": super_agent_id_var.get(),
        "phtn_sub_agent_id": sub_agent_id_var.get(),
        "phtn_super_agent_instance_id": super_agent_instance_id_var.get(),
        "phtn_sub_agent_instance_id": sub_agent_instance_id_var.get(),
        "phtn_app_name": app_name_var.get(),
        "phtn_environment": environment_var.get(),
        "phtn_agent_capability": capability_id_var.get(),
        "phtn_agent_skill": skill_id_var.get(),
        "agent_name": agent_name_var.get(),
    }


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    extra_data: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Log a message with additional context data."""
    extra = kwargs.get('extra', {})
    if extra_data:
        extra['extra_data'] = extra_data
    kwargs['extra'] = extra
    logger.log(level, message, **kwargs)


def log_llm_call(
    logger: logging.Logger,
    message: str,
    model: str,
    provider: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost: float = 0.0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    latency_ms: int = 0,
    operation: str = "chat_completion",
    status: int = 200,
    input: Optional[str] = None,
    output: Optional[str] = None,
    **kwargs
):
    """
    Log an LLM call with all fields required by phtnai-ops-metrics FinOps.
    
    This creates a log entry that can be processed by the FinOps calculator
    for cost tracking and token usage analysis.
    
    Args:
        logger: Logger instance
        message: Log message
        model: LLM model name (e.g., "gpt-4o")
        provider: LLM provider (e.g., "openai")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        total_tokens: Total tokens (prompt + completion)
        cost: Total cost (for backward compatibility)
        input_cost: Cost of input tokens
        output_cost: Cost of output tokens
        latency_ms: Request latency in milliseconds
        operation: Operation type (e.g., "chat_completion")
        status: HTTP status code
        input: Request payload/prompt
        output: Response content
        **kwargs: Additional fields
    """
    extra_data = {
        "log_type": "llm_gateway",
        "model": model,
        "provider": provider,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost or (input_cost + output_cost),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "latency_ms": latency_ms,
        "operation": operation,
        "status": status,
        **kwargs
    }
    
    # Add input/output if provided
    if input is not None:
        extra_data["input"] = input
    if output is not None:
        extra_data["output"] = output
    
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_tool_call(
    logger: logging.Logger,
    message: str,
    tool_name: str,
    tool_id: str,
    latency_ms: int = 0,
    status: int = 200,
    success: bool = True,
    **kwargs
):
    """
    Log a tool call with all fields required by phtnai-ops-metrics TechOps.
    
    This creates a log entry that can be processed by the TechOps calculator
    for tool performance and reliability analysis.
    """
    extra_data = {
        "log_type": "sub_agent",
        "tool_name": tool_name,
        "tool_id": tool_id,
        "latency_ms": latency_ms,
        "status": status,
        "success": success,
        "operation": "tool_call",
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_agent_execution(
    logger: logging.Logger,
    message: str,
    execution_pattern: str,
    latency_ms: int = 0,
    status: int = 200,
    iterations: int = 0,
    tool_calls: int = 0,
    **kwargs
):
    """
    Log an agent execution with all fields required by phtnai-ops-metrics AIOps.
    
    This creates a log entry that can be processed by the AIOps calculator
    for agent quality and orchestration analysis.
    """
    extra_data = {
        "log_type": "sub_agent",
        "execution_pattern": execution_pattern,
        "latency_ms": latency_ms,
        "status": status,
        "iterations": iterations,
        "tool_calls": tool_calls,
        "operation": "agent_execution",
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


# ============================================================================
# NEW FEATURE LOGGING FUNCTIONS (n8n-comparable features)
# ============================================================================

def log_text_splitter(
    logger: logging.Logger,
    message: str,
    splitter_type: str,
    chunk_size: int,
    chunk_overlap: int,
    input_length: int,
    output_chunks: int,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log text splitter operations for RAG pipeline monitoring.
    
    Tracks text chunking performance for document processing pipelines.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": "text_splitter",
        "splitter_type": splitter_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "input_length": input_length,
        "output_chunks": output_chunks,
        "latency_ms": latency_ms,
        "status": status,
        "avg_chunk_size": input_length // max(output_chunks, 1),
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_output_parser(
    logger: logging.Logger,
    message: str,
    parser_type: str,
    input_length: int,
    output_type: str,
    success: bool = True,
    retries: int = 0,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log output parser operations for LLM response processing.
    
    Tracks parsing success rates and auto-fixing attempts.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": "output_parser",
        "parser_type": parser_type,
        "input_length": input_length,
        "output_type": output_type,
        "success": success,
        "retries": retries,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_vector_store(
    logger: logging.Logger,
    message: str,
    provider: str,
    operation: str,
    collection_name: str = "",
    document_count: int = 0,
    embedding_dimension: int = 0,
    similarity_metric: str = "cosine",
    top_k: int = 0,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log vector store operations for RAG and semantic search.
    
    Tracks vector database performance for document retrieval.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": f"vector_store_{operation}",
        "vector_store_provider": provider,
        "collection_name": collection_name,
        "document_count": document_count,
        "embedding_dimension": embedding_dimension,
        "similarity_metric": similarity_metric,
        "top_k": top_k,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_evaluation_metric(
    logger: logging.Logger,
    message: str,
    metric_type: str,
    score: float,
    threshold: float,
    passed: bool,
    evaluation_id: str = "",
    latency_ms: int = 0,
    **kwargs
):
    """
    Log evaluation metric results for agent quality monitoring.
    
    Tracks relevance, faithfulness, hallucination, toxicity, etc.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": "evaluation_metric",
        "metric_type": metric_type,
        "score": score,
        "threshold": threshold,
        "passed": passed,
        "evaluation_id": evaluation_id,
        "latency_ms": latency_ms,
        "status": 200 if passed else 400,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_trigger_event(
    logger: logging.Logger,
    message: str,
    trigger_type: str,
    trigger_id: str = "",
    source: str = "",
    event_type: str = "",
    payload_size: int = 0,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log trigger events for agent invocation tracking.
    
    Tracks webhook, schedule, chat, and event-based triggers.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": "trigger_event",
        "trigger_type": trigger_type,
        "trigger_id": trigger_id,
        "trigger_source": source,
        "event_type": event_type,
        "payload_size": payload_size,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_builtin_tool(
    logger: logging.Logger,
    message: str,
    tool_name: str,
    tool_category: str,
    input_params: Dict[str, Any] = None,
    output_type: str = "",
    success: bool = True,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log built-in tool executions (Calculator, WebSearch, Wikipedia, etc.).
    
    Tracks performance and usage of framework-provided tools.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": "builtin_tool",
        "builtin_tool_name": tool_name,
        "tool_category": tool_category,
        "input_params": input_params or {},
        "output_type": output_type,
        "success": success,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_llm_provider(
    logger: logging.Logger,
    message: str,
    provider: str,
    model: str,
    operation: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost: float = 0.0,
    latency_ms: int = 0,
    status: int = 200,
    is_streaming: bool = False,
    is_multimodal: bool = False,
    **kwargs
):
    """
    Log LLM provider operations with extended provider support.
    
    Supports OpenAI, Anthropic, Azure, Bedrock, Gemini, Ollama, Groq, Mistral, Together, Cohere.
    """
    extra_data = {
        "log_type": "llm_gateway",
        "operation": operation,
        "llm_provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "latency_ms": latency_ms,
        "status": status,
        "is_streaming": is_streaming,
        "is_multimodal": is_multimodal,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_memory_operation(
    logger: logging.Logger,
    message: str,
    memory_type: str,
    operation: str,
    backend: str = "",
    item_count: int = 0,
    memory_size_bytes: int = 0,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log memory operations (short-term, long-term, semantic, episodic).
    
    Tracks memory usage and performance for conversation context.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": f"memory_{operation}",
        "memory_type": memory_type,
        "memory_backend": backend,
        "item_count": item_count,
        "memory_size_bytes": memory_size_bytes,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_mcp_operation(
    logger: logging.Logger,
    message: str,
    server_name: str,
    operation: str,
    provider: str = "custom",
    transport: str = "http",
    tool_name: str = "",
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log MCP (Model Context Protocol) operations.
    
    Tracks MCP server connections and tool invocations.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": f"mcp_{operation}",
        "mcp_server_name": server_name,
        "mcp_provider": provider,
        "mcp_transport": transport,
        "mcp_tool_name": tool_name,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_guardrail_check(
    logger: logging.Logger,
    message: str,
    guardrail_type: str,
    check_name: str,
    passed: bool,
    score: float = 0.0,
    threshold: float = 0.0,
    action_taken: str = "",
    latency_ms: int = 0,
    **kwargs
):
    """
    Log guardrail checks (PII, toxicity, hallucination, prompt injection).
    
    Tracks safety and compliance checks on agent inputs/outputs.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": "guardrail_check",
        "guardrail_type": guardrail_type,
        "check_name": check_name,
        "passed": passed,
        "score": score,
        "threshold": threshold,
        "action_taken": action_taken,
        "latency_ms": latency_ms,
        "status": 200 if passed else 403,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_human_loop(
    logger: logging.Logger,
    message: str,
    operation: str,
    request_type: str,
    request_id: str = "",
    timeout_seconds: int = 0,
    approved: bool = False,
    approver: str = "",
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log human-in-the-loop operations.
    
    Tracks approval requests and human intervention events.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": f"human_loop_{operation}",
        "hitl_request_type": request_type,
        "hitl_request_id": request_id,
        "timeout_seconds": timeout_seconds,
        "approved": approved,
        "approver": approver,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_ontology_operation(
    logger: logging.Logger,
    message: str,
    operation: str,
    entity_type: str = "",
    entity_id: str = "",
    relationship_type: str = "",
    concept_count: int = 0,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log ontology/knowledge graph operations.
    
    Tracks entity, concept, and relationship management.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": f"ontology_{operation}",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "relationship_type": relationship_type,
        "concept_count": concept_count,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


def log_rag_pipeline(
    logger: logging.Logger,
    message: str,
    operation: str,
    retriever_type: str = "",
    query_length: int = 0,
    documents_retrieved: int = 0,
    reranked: bool = False,
    context_compressed: bool = False,
    latency_ms: int = 0,
    status: int = 200,
    **kwargs
):
    """
    Log RAG pipeline operations.
    
    Tracks retrieval, reranking, and context compression.
    """
    extra_data = {
        "log_type": "sub_agent",
        "operation": f"rag_{operation}",
        "retriever_type": retriever_type,
        "query_length": query_length,
        "documents_retrieved": documents_retrieved,
        "reranked": reranked,
        "context_compressed": context_compressed,
        "latency_ms": latency_ms,
        "status": status,
        **kwargs
    }
    log_with_context(logger, logging.INFO, message, extra_data=extra_data)


__all__ = [
    # Configuration
    'configure_otel_logging',
    'get_logger',
    
    # Context management
    'set_trace_context',
    'set_trace_context_from_config',
    'get_trace_context',
    
    # Core logging helpers
    'log_with_context',
    'log_llm_call',
    'log_tool_call',
    'log_agent_execution',
    
    # NEW: Feature-specific logging (n8n-comparable features)
    'log_text_splitter',
    'log_output_parser',
    'log_vector_store',
    'log_evaluation_metric',
    'log_trigger_event',
    'log_builtin_tool',
    'log_llm_provider',
    'log_memory_operation',
    'log_mcp_operation',
    'log_guardrail_check',
    'log_human_loop',
    'log_ontology_operation',
    'log_rag_pipeline',
    
    # Data classes
    'PHtnAgentIdParts',
    
    # Core context variables
    'correlation_id_var',
    'phtn_agent_id_var',
    'trace_id_var',
    'span_id_var',
    'agent_id_var',
    'tenant_id_var',
    'request_id_var',
    
    # Extended context variables for phtnai-ops-metrics
    'tenant_group_id_var',
    'domain_id_var',
    'team_id_var',
    'project_id_var',
    'user_id_var',
    'agent_type_var',
    'super_agent_id_var',
    'sub_agent_id_var',
    'super_agent_instance_id_var',
    'sub_agent_instance_id_var',
    'app_name_var',
    'environment_var',
    'capability_id_var',
    'skill_id_var',
    'agent_name_var',
]
