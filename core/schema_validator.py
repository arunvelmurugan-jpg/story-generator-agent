"""
Schema Validator for PHTN.AI Sub-Agent Framework

Validates agent configurations against PHTN-AGENT-SCHEMA_v2.json
Provides detailed validation errors and suggestions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logger.warning("jsonschema not installed. Schema validation will be limited.")


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    def add_error(self, path: str, message: str, value: Any = None):
        """Add a validation error."""
        self.errors.append({
            "path": path,
            "message": message,
            "value": value
        })
        self.is_valid = False
    
    def add_warning(self, path: str, message: str, value: Any = None):
        """Add a validation warning."""
        self.warnings.append({
            "path": path,
            "message": message,
            "value": value
        })
    
    def add_suggestion(self, suggestion: str):
        """Add a suggestion for improvement."""
        self.suggestions.append(suggestion)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }


class SchemaValidator:
    """
    Validates agent configurations against PHTN-AGENT-SCHEMA_v2.json.
    
    Features:
    - JSON Schema validation (Draft-07)
    - Custom business rule validation
    - Security configuration validation
    - Guardrails completeness checks
    - Best practice suggestions
    """
    
    REQUIRED_FIELDS = [
        "agent_id", "name", "version"
    ]
    
    RECOMMENDED_FIELDS = [
        "owner", "tenant", "status", "execution_config", "model_config",
        "guardrails", "security", "observability_config"
    ]
    
    VALID_EXECUTION_PATTERNS = [
        "SIMPLE", "REACT", "COT", "TOOL_USE", "RAG", "PLAN_EXECUTE", "CUSTOM"
    ]
    
    VALID_CONTENT_TYPES = [
        "TEXT", "NUMBER", "BOOLEAN", "NULL", "JSON_OBJECT", "JSON_ARRAY",
        "TABLE", "SCHEMA_INSTANCE", "MARKDOWN", "HTML", "IMAGE", "AUDIO",
        "VIDEO", "PDF", "EXCEL", "CHART", "CODE", "BINARY_BLOB",
        "FILE_REFERENCE", "ARTIFACT", "ATTACHMENT", "STREAM_CHUNK",
        "SSE_EVENT", "DELTA", "ENTITY_REFERENCE", "AGENT_REFERENCE",
        "RESOURCE_LOCATOR", "URI", "TOOL_CALL", "TOOL_RESULT", "ERROR",
        "METADATA", "HUMAN_INPUT_REQUEST", "HUMAN_APPROVAL_REQUEST"
    ]
    
    VALID_STATUS_STAGES = [
        "draft", "review", "approved", "staging", "production", "deprecated", "archived"
    ]
    
    VALID_RISK_LEVELS = ["low", "medium", "high", "critical"]
    
    def __init__(self, schema_path: Optional[Path] = None):
        """
        Initialize SchemaValidator.
        
        Args:
            schema_path: Path to PHTN-AGENT-SCHEMA_v2.json
        """
        if schema_path is None:
            current_file = Path(__file__)
            framework_root = current_file.parent.parent
            schema_path = framework_root / ".phtnai" / "PHTN-AGENT-SCHEMA_v2.json"
        
        self.schema_path = Path(schema_path)
        self._schema: Optional[Dict[str, Any]] = None
        self._validator: Optional[Any] = None
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Load and cache the schema."""
        if self._schema is None:
            if not self.schema_path.exists():
                raise FileNotFoundError(f"Schema not found at: {self.schema_path}")
            
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                self._schema = json.load(f)
        
        return self._schema
    
    @property
    def validator(self) -> Any:
        """Get JSON Schema validator."""
        if self._validator is None and JSONSCHEMA_AVAILABLE:
            self._validator = Draft7Validator(self.schema)
        return self._validator
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate agent configuration against schema.
        
        Args:
            config: Agent configuration dictionary
            
        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        result = ValidationResult(is_valid=True)
        
        self._validate_required_fields(config, result)
        
        if JSONSCHEMA_AVAILABLE:
            self._validate_json_schema(config, result)
        
        self._validate_execution_config(config, result)
        self._validate_model_config(config, result)
        self._validate_capabilities(config, result)
        self._validate_guardrails(config, result)
        self._validate_security(config, result)
        self._validate_observability(config, result)
        self._validate_tools(config, result)
        
        self._add_best_practice_suggestions(config, result)
        
        return result
    
    def validate_file(self, config_path: Path) -> ValidationResult:
        """
        Validate agent configuration file.
        
        Args:
            config_path: Path to PHTN-AGENT.json file
            
        Returns:
            ValidationResult
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return self.validate(config)
        except json.JSONDecodeError as e:
            result = ValidationResult(is_valid=False)
            result.add_error("$", f"Invalid JSON: {e}")
            return result
        except FileNotFoundError:
            result = ValidationResult(is_valid=False)
            result.add_error("$", f"File not found: {config_path}")
            return result
    
    def _validate_required_fields(self, config: Dict[str, Any], result: ValidationResult):
        """Validate required fields are present."""
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                result.add_error(f"$.{field}", f"Required field '{field}' is missing")
        
        for field in self.RECOMMENDED_FIELDS:
            if field not in config:
                result.add_warning(f"$.{field}", f"Recommended field '{field}' is missing")
    
    def _validate_json_schema(self, config: Dict[str, Any], result: ValidationResult):
        """Validate against JSON Schema."""
        if not self.validator:
            return
        
        errors = list(self.validator.iter_errors(config))
        for error in errors:
            path = ".".join(str(p) for p in error.absolute_path) or "$"
            result.add_error(f"$.{path}", error.message, error.instance)
    
    def _validate_execution_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate execution configuration."""
        exec_config = config.get("execution_config", {})
        
        pattern = exec_config.get("pattern")
        if pattern and pattern not in self.VALID_EXECUTION_PATTERNS:
            result.add_error(
                "$.execution_config.pattern",
                f"Invalid execution pattern: {pattern}. Valid patterns: {self.VALID_EXECUTION_PATTERNS}",
                pattern
            )
        
        max_iterations = exec_config.get("max_iterations", 10)
        if max_iterations < 1:
            result.add_error(
                "$.execution_config.max_iterations",
                "max_iterations must be at least 1",
                max_iterations
            )
        elif max_iterations > 100:
            result.add_warning(
                "$.execution_config.max_iterations",
                f"max_iterations={max_iterations} is very high, consider reducing for safety",
                max_iterations
            )
        
        timeout = exec_config.get("timeout_seconds", 300)
        if timeout < 1:
            result.add_error(
                "$.execution_config.timeout_seconds",
                "timeout_seconds must be at least 1",
                timeout
            )
        
        if pattern == "REACT" and not exec_config.get("react_config"):
            result.add_warning(
                "$.execution_config.react_config",
                "REACT pattern selected but react_config not provided"
            )
        
        if pattern == "RAG" and not exec_config.get("rag_config"):
            result.add_warning(
                "$.execution_config.rag_config",
                "RAG pattern selected but rag_config not provided"
            )
    
    def _validate_model_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate model configuration."""
        model_config = config.get("model_config", {})
        
        if not model_config.get("primary_model"):
            result.add_error(
                "$.model_config.primary_model",
                "primary_model is required"
            )
        
        params = model_config.get("parameters", {})
        temp = params.get("temperature")
        if temp is not None and (temp < 0 or temp > 2):
            result.add_error(
                "$.model_config.parameters.temperature",
                f"temperature must be between 0 and 2, got {temp}",
                temp
            )
        
        top_p = params.get("top_p")
        if top_p is not None and (top_p < 0 or top_p > 1):
            result.add_error(
                "$.model_config.parameters.top_p",
                f"top_p must be between 0 and 1, got {top_p}",
                top_p
            )
        
        if not model_config.get("fallback_models"):
            result.add_warning(
                "$.model_config.fallback_models",
                "No fallback models configured. Consider adding fallbacks for resilience."
            )
    
    def _validate_capabilities(self, config: Dict[str, Any], result: ValidationResult):
        """Validate capabilities configuration."""
        capabilities = config.get("capabilities", [])
        skills = config.get("skills", [])
        
        if not capabilities and not skills:
            result.add_warning(
                "$.capabilities",
                "No capabilities or skills defined. Consider adding at least one."
            )
            return
        
        if not capabilities:
            return
        
        capability_ids = set()
        for i, cap in enumerate(capabilities):
            cap_id = cap.get("id")
            
            if not cap_id:
                result.add_error(
                    f"$.capabilities[{i}].id",
                    "Capability id is required"
                )
            elif cap_id in capability_ids:
                result.add_error(
                    f"$.capabilities[{i}].id",
                    f"Duplicate capability id: {cap_id}",
                    cap_id
                )
            else:
                capability_ids.add(cap_id)
            
            if not cap.get("name"):
                result.add_error(
                    f"$.capabilities[{i}].name",
                    "Capability name is required"
                )
            
            skills = cap.get("skills", [])
            skill_ids = set()
            for j, skill in enumerate(skills):
                skill_id = skill.get("id")
                if skill_id and skill_id in skill_ids:
                    result.add_error(
                        f"$.capabilities[{i}].skills[{j}].id",
                        f"Duplicate skill id within capability: {skill_id}",
                        skill_id
                    )
                else:
                    skill_ids.add(skill_id)
    
    def _validate_guardrails(self, config: Dict[str, Any], result: ValidationResult):
        """Validate guardrails configuration."""
        guardrails = config.get("guardrails", {})
        
        if not guardrails:
            result.add_warning(
                "$.guardrails",
                "Guardrails configuration is empty. Consider adding safety measures."
            )
            return
        
        input_gr = guardrails.get("input_guardrails", {})
        if not input_gr.get("prompt_injection_detection", {}).get("enabled", False):
            result.add_warning(
                "$.guardrails.input_guardrails.prompt_injection_detection",
                "Prompt injection detection is not enabled. Recommended for security."
            )
        
        if not input_gr.get("pii_detection", {}).get("enabled", False):
            result.add_warning(
                "$.guardrails.input_guardrails.pii_detection",
                "PII detection is not enabled. Consider enabling for privacy compliance."
            )
        
        output_gr = guardrails.get("output_guardrails", {})
        if not output_gr.get("toxicity_filtering", {}).get("enabled", False):
            result.add_warning(
                "$.guardrails.output_guardrails.toxicity_filtering",
                "Toxicity filtering is not enabled. Recommended for safe outputs."
            )
    
    def _validate_security(self, config: Dict[str, Any], result: ValidationResult):
        """Validate security configuration."""
        security = config.get("security", {})
        
        if not security:
            result.add_error(
                "$.security",
                "Security configuration is required"
            )
            return
        
        if not security.get("rbac", {}).get("enabled", False) and \
           not security.get("abac", {}).get("enabled", False):
            result.add_warning(
                "$.security",
                "Neither RBAC nor ABAC is enabled. Consider enabling access control."
            )
        
        if not security.get("audit_logging", {}).get("enabled", True):
            result.add_warning(
                "$.security.audit_logging",
                "Audit logging is disabled. Recommended for compliance."
            )
        
        if not security.get("encryption_at_rest", True):
            result.add_warning(
                "$.security.encryption_at_rest",
                "Encryption at rest is disabled. Recommended for data protection."
            )
    
    def _validate_observability(self, config: Dict[str, Any], result: ValidationResult):
        """Validate observability configuration."""
        obs_config = config.get("observability_config", {})
        
        if not obs_config:
            result.add_warning(
                "$.observability_config",
                "Observability configuration is empty. Consider adding tracing and metrics."
            )
            return
        
        if not obs_config.get("tracing", {}).get("enabled", False):
            result.add_warning(
                "$.observability_config.tracing",
                "Tracing is not enabled. Recommended for debugging and monitoring."
            )
        
        if not obs_config.get("metrics", {}).get("enabled", False):
            result.add_warning(
                "$.observability_config.metrics",
                "Metrics are not enabled. Recommended for monitoring."
            )
    
    def _validate_tools(self, config: Dict[str, Any], result: ValidationResult):
        """Validate tools configuration."""
        tools = config.get("tools", [])
        
        tool_ids = set()
        for i, tool in enumerate(tools):
            tool_id = tool.get("tool_id")
            
            if not tool_id:
                result.add_error(
                    f"$.tools[{i}].tool_id",
                    "Tool tool_id is required"
                )
            elif tool_id in tool_ids:
                result.add_error(
                    f"$.tools[{i}].tool_id",
                    f"Duplicate tool_id: {tool_id}",
                    tool_id
                )
            else:
                tool_ids.add(tool_id)
            
            if not tool.get("name"):
                result.add_error(
                    f"$.tools[{i}].name",
                    "Tool name is required"
                )
            
            sandboxing = tool.get("sandboxing", {})
            if not sandboxing.get("enabled", True):
                result.add_warning(
                    f"$.tools[{i}].sandboxing",
                    f"Tool '{tool_id}' has sandboxing disabled. Consider enabling for security."
                )
    
    def _add_best_practice_suggestions(self, config: Dict[str, Any], result: ValidationResult):
        """Add best practice suggestions."""
        if not config.get("description"):
            result.add_suggestion(
                "Add a description to help others understand the agent's purpose."
            )
        
        if not config.get("domain"):
            result.add_suggestion(
                "Consider adding domain configuration for industry-specific behavior."
            )
        
        if not config.get("resilience"):
            result.add_suggestion(
                "Consider adding resilience configuration (circuit breaker, retry, rate limiting)."
            )
        
        if not config.get("health_check"):
            result.add_suggestion(
                "Consider adding health check configuration for production deployments."
            )
        
        if not config.get("cost_governance"):
            result.add_suggestion(
                "Consider adding cost governance to track and limit token usage."
            )
        
        if not config.get("a2a_protocol"):
            result.add_suggestion(
                "Consider configuring A2A protocol for agent-to-agent communication."
            )
        
        exec_pattern = config.get("execution_config", {}).get("pattern")
        if exec_pattern == "SIMPLE":
            result.add_suggestion(
                "SIMPLE pattern is basic. Consider REACT or TOOL_USE for more complex tasks."
            )


def validate_agent_config(config: Dict[str, Any], schema_path: Optional[Path] = None) -> ValidationResult:
    """
    Convenience function to validate agent configuration.
    
    Args:
        config: Agent configuration dictionary
        schema_path: Optional path to schema file
        
    Returns:
        ValidationResult
    """
    validator = SchemaValidator(schema_path)
    return validator.validate(config)


def validate_agent_file(config_path: Path, schema_path: Optional[Path] = None) -> ValidationResult:
    """
    Convenience function to validate agent configuration file.
    
    Args:
        config_path: Path to PHTN-AGENT.json file
        schema_path: Optional path to schema file
        
    Returns:
        ValidationResult
    """
    validator = SchemaValidator(schema_path)
    return validator.validate_file(config_path)
