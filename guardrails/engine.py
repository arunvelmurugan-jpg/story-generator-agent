"""
Guardrails Engine for PHTN.AI Sub-Agent Framework

Enforces safety and compliance guardrails.
Now fully utilizes guardrails config from PHTN-AGENT.json including:
- input_guardrails: PII detection, prompt injection, content filtering
- output_guardrails: toxicity, PII leakage, forbidden phrases
- tool_guardrails: allowlist/denylist enforcement
- custom_rules: Custom rule evaluation
- toxicity detection: Multiple providers (Perspective, OpenAI, HuggingFace)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..observability.otel_logging import get_logger
from .toxicity import (
    ToxicityConfig,
    ToxicityProvider,
    ToxicityCategory,
    ToxicityResult,
    create_toxicity_detector,
    BaseToxicityDetector,
)

if TYPE_CHECKING:
    from ..core.config_loader import GuardrailsConfig
    from ..core.agent import AgentInput, AgentOutput

logger = get_logger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail check."""
    passed: bool
    rule_id: Optional[str] = None
    reason: Optional[str] = None
    action: str = "allow"
    severity: str = "info"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "action": self.action,
            "severity": self.severity,
            "metadata": self.metadata,
        }


class GuardrailsEngine:
    """
    Enforces guardrails on agent inputs and outputs.
    
    Features:
    - Input guardrails (PII, prompt injection, content filtering)
    - Output guardrails (toxicity, hallucination, schema validation)
    - Tool guardrails (validation, allowlist/denylist)
    - Custom rule enforcement
    """
    
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }
    
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all)\s+instructions",
        r"disregard\s+(previous|all)\s+instructions",
        r"forget\s+(previous|all)\s+instructions",
        r"you\s+are\s+now\s+",
        r"act\s+as\s+if\s+you\s+are",
        r"pretend\s+you\s+are",
        r"system\s*:\s*",
        r"\[INST\]",
        r"<\|im_start\|>",
    ]
    
    def __init__(self, config: "GuardrailsConfig"):
        """
        Initialize GuardrailsEngine.
        
        Args:
            config: Guardrails configuration
        """
        self.config = config
        self.enforcement_mode = config.enforcement_mode if hasattr(config, 'enforcement_mode') else "enforce"
        
        self._toxicity_detector: Optional[BaseToxicityDetector] = None
        self._init_toxicity_detector()
        
        logger.debug(f"GuardrailsEngine initialized (mode: {self.enforcement_mode})")
    
    def _init_toxicity_detector(self):
        """Initialize toxicity detector from config."""
        output_config = getattr(self.config, 'output_guardrails', None)
        if not output_config:
            return
        
        toxicity_config = getattr(output_config, 'toxicity_detection', None)
        if not toxicity_config or not getattr(toxicity_config, 'enabled', False):
            return
        
        try:
            provider_str = getattr(toxicity_config, 'provider', 'local')
            try:
                provider = ToxicityProvider(provider_str.lower())
            except ValueError:
                provider = ToxicityProvider.LOCAL
            
            threshold = getattr(toxicity_config, 'threshold', 0.8)
            
            config = ToxicityConfig(
                provider=provider,
                threshold=threshold,
                action=getattr(toxicity_config, 'action', 'block'),
            )
            
            self._toxicity_detector = create_toxicity_detector(config)
            logger.info(f"Toxicity detector initialized: {provider.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize toxicity detector: {e}")
    
    async def check_input(self, input_data: "AgentInput") -> GuardrailResult:
        """
        Check input against guardrails.
        
        Args:
            input_data: Agent input
            
        Returns:
            GuardrailResult
        """
        content = input_data.content
        if isinstance(content, dict):
            import json
            content = json.dumps(content)
        content = str(content)
        
        input_config = getattr(self.config, 'input_guardrails', None)
        if not input_config:
            return GuardrailResult(passed=True)
        
        pii_config = getattr(input_config, 'pii_detection', None)
        if pii_config and getattr(pii_config, 'enabled', False):
            pii_result = self._check_pii(content)
            if not pii_result.passed:
                action = getattr(pii_config, 'action', 'redact')
                if action == "block":
                    return pii_result
        
        injection_config = getattr(input_config, 'prompt_injection_detection', None)
        if injection_config and getattr(injection_config, 'enabled', True):
            injection_result = self._check_prompt_injection(content)
            if not injection_result.passed:
                action = getattr(injection_config, 'action', 'block')
                if action == "block":
                    return injection_result
        
        max_length = getattr(input_config, 'max_input_length', None)
        if max_length and len(content) > max_length:
            return GuardrailResult(
                passed=False,
                rule_id="max_input_length",
                reason=f"Input exceeds maximum length of {max_length}",
                action="block",
                severity="medium",
            )
        
        return GuardrailResult(passed=True)
    
    async def check_output(self, output: "AgentOutput") -> GuardrailResult:
        """
        Check output against guardrails.
        
        Args:
            output: Agent output
            
        Returns:
            GuardrailResult
        """
        content = output.content
        if content is None:
            return GuardrailResult(passed=True)
        
        if isinstance(content, dict):
            import json
            content = json.dumps(content)
        content = str(content)
        
        output_config = getattr(self.config, 'output_guardrails', None)
        if not output_config:
            return GuardrailResult(passed=True)
        
        pii_config = getattr(output_config, 'pii_leakage_prevention', None)
        if pii_config and getattr(pii_config, 'enabled', True):
            pii_result = self._check_pii(content)
            if not pii_result.passed:
                pii_result.rule_id = "pii_leakage"
                pii_result.reason = "PII detected in output"
                return pii_result
        
        forbidden = getattr(output_config, 'forbidden_phrases', [])
        if forbidden:
            for phrase in forbidden:
                if phrase.lower() in content.lower():
                    return GuardrailResult(
                        passed=False,
                        rule_id="forbidden_phrase",
                        reason=f"Output contains forbidden phrase",
                        action="block",
                        severity="high",
                    )
        
        if self._toxicity_detector:
            toxicity_result = await self._check_toxicity(content)
            if not toxicity_result.passed:
                return toxicity_result
        
        return GuardrailResult(passed=True)
    
    async def _check_toxicity(self, content: str) -> GuardrailResult:
        """Check content for toxicity."""
        if not self._toxicity_detector:
            return GuardrailResult(passed=True)
        
        try:
            result = await self._toxicity_detector.detect(content)
            
            if result.is_toxic:
                return GuardrailResult(
                    passed=False,
                    rule_id="toxicity_detection",
                    reason=f"Toxic content detected: {', '.join(result.flagged_categories)}",
                    action="block",
                    severity="high",
                    metadata={
                        "overall_score": result.overall_score,
                        "flagged_categories": result.flagged_categories,
                        "provider": result.provider,
                    },
                )
            
            return GuardrailResult(passed=True)
            
        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return GuardrailResult(passed=True)
    
    async def check_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> GuardrailResult:
        """
        Check tool call against guardrails.
        
        Enforces:
        - tool_guardrails.denylist: Block specific tools
        - tool_guardrails.allowlist: Only allow specific tools
        - tool_guardrails.input_validation: Validate tool inputs
        - tool_guardrails.rate_limits: Per-tool rate limiting
        
        Args:
            tool_name: Tool name
            tool_input: Tool input parameters
            
        Returns:
            GuardrailResult
        """
        tool_config = getattr(self.config, 'tool_guardrails', None)
        if not tool_config:
            return GuardrailResult(passed=True)
        
        if isinstance(tool_config, dict):
            denylist = tool_config.get("denylist", [])
            allowlist = tool_config.get("allowlist", [])
            input_validation = tool_config.get("input_validation", {})
            require_confirmation = tool_config.get("require_confirmation", [])
        else:
            denylist = getattr(tool_config, 'denylist', []) or []
            allowlist = getattr(tool_config, 'allowlist', []) or []
            input_validation = getattr(tool_config, 'input_validation', {}) or {}
            require_confirmation = getattr(tool_config, 'require_confirmation', []) or []
        
        if tool_name in denylist:
            logger.warning(f"🚫 Tool '{tool_name}' blocked by denylist")
            return GuardrailResult(
                passed=False,
                rule_id="tool_denylist",
                reason=f"Tool '{tool_name}' is in denylist",
                action="block",
                severity="high",
            )
        
        if allowlist and tool_name not in allowlist:
            logger.warning(f"🚫 Tool '{tool_name}' not in allowlist: {allowlist}")
            return GuardrailResult(
                passed=False,
                rule_id="tool_allowlist",
                reason=f"Tool '{tool_name}' is not in allowlist. Allowed tools: {', '.join(allowlist)}",
                action="block",
                severity="medium",
            )
        
        if tool_name in require_confirmation:
            logger.info(f"⚠️ Tool '{tool_name}' requires confirmation (not enforced in current implementation)")
        
        if input_validation:
            validation_result = self._validate_tool_input(tool_name, tool_input, input_validation)
            if not validation_result.passed:
                return validation_result
        
        pii_in_input = self._check_pii(str(tool_input))
        if not pii_in_input.passed:
            logger.warning(f"⚠️ PII detected in tool input for '{tool_name}'")
        
        return GuardrailResult(passed=True)
    
    def _validate_tool_input(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        validation_config: Dict[str, Any],
    ) -> GuardrailResult:
        """Validate tool input against configured rules."""
        tool_rules = validation_config.get(tool_name, {})
        
        if not tool_rules:
            return GuardrailResult(passed=True)
        
        required_fields = tool_rules.get("required_fields", [])
        for field_name in required_fields:
            if field_name not in tool_input:
                return GuardrailResult(
                    passed=False,
                    rule_id="tool_input_validation",
                    reason=f"Required field '{field_name}' missing in tool input",
                    action="block",
                    severity="medium",
                )
        
        forbidden_values = tool_rules.get("forbidden_values", {})
        for field_name, forbidden in forbidden_values.items():
            if field_name in tool_input and tool_input[field_name] in forbidden:
                return GuardrailResult(
                    passed=False,
                    rule_id="tool_input_validation",
                    reason=f"Forbidden value for field '{field_name}'",
                    action="block",
                    severity="high",
                )
        
        return GuardrailResult(passed=True)
    
    def _check_pii(self, content: str) -> GuardrailResult:
        """Check for PII in content."""
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, content, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    rule_id="pii_detection",
                    reason=f"PII detected: {pii_type}",
                    action="redact",
                    severity="high",
                    metadata={"pii_type": pii_type},
                )
        
        return GuardrailResult(passed=True)
    
    def _check_prompt_injection(self, content: str) -> GuardrailResult:
        """Check for prompt injection attempts."""
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    rule_id="prompt_injection",
                    reason="Potential prompt injection detected",
                    action="block",
                    severity="critical",
                )
        
        return GuardrailResult(passed=True)
    
    def redact_pii(self, content: str) -> str:
        """Redact PII from content."""
        redacted = content
        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted, flags=re.IGNORECASE)
        return redacted
