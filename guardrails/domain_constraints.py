"""
Domain Constraints Enforcer for PHTN.AI Sub-Agent Framework

Enforces domain-specific constraints including:
- Restricted content detection
- Compliance requirements (GDPR, SOC2, HIPAA, etc.)
- Required disclaimers
- Business rules
- Industry-specific regulations
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    GDPR = "GDPR"
    SOC2 = "SOC2"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"
    SOX = "SOX"
    CCPA = "CCPA"
    DPDP_INDIA = "DPDP_INDIA"
    EU_AI_ACT = "EU_AI_ACT"


class RestrictionAction(str, Enum):
    """Actions for restricted content."""
    BLOCK = "block"
    WARN = "warn"
    REDACT = "redact"
    DISCLAIMER = "disclaimer"


@dataclass
class DomainConstraintsConfig:
    """Configuration for domain constraints."""
    restricted_content: List[str] = field(default_factory=list)
    compliance_standards: List[str] = field(default_factory=list)
    required_disclaimers: List[str] = field(default_factory=list)
    business_rules: List[Dict[str, Any]] = field(default_factory=list)
    industry: Optional[str] = None
    sub_domain: Optional[str] = None
    action_on_violation: str = "block"
    enable_compliance_checks: bool = True


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint_type: str
    constraint_id: str
    message: str
    severity: str = "medium"
    action: str = "block"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "constraint_id": self.constraint_id,
            "message": self.message,
            "severity": self.severity,
            "action": self.action,
            "metadata": self.metadata,
        }


@dataclass
class ConstraintCheckResult:
    """Result of constraint check."""
    passed: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    disclaimers_required: List[str] = field(default_factory=list)
    modified_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "disclaimers_required": self.disclaimers_required,
            "has_modified_content": self.modified_content is not None,
        }


class DomainConstraintsEnforcer:
    """
    Enforces domain-specific constraints on agent inputs and outputs.
    
    Features:
    - Restricted content detection and blocking
    - Compliance standard enforcement
    - Required disclaimer injection
    - Business rule evaluation
    - Industry-specific regulations
    """
    
    RESTRICTED_CONTENT_PATTERNS = {
        "financial_advice": [
            r"\b(?:invest|buy|sell)\s+(?:stocks?|bonds?|securities|crypto)",
            r"\b(?:financial|investment)\s+advice\b",
            r"\b(?:guaranteed|certain)\s+(?:returns?|profit)",
            r"\bstock\s+(?:tips?|picks?|recommendations?)\b",
            r"\bretirement\s+(?:planning|advice)\b",
        ],
        "legal_advice": [
            r"\blegal\s+advice\b",
            r"\b(?:you\s+should|i\s+recommend)\s+(?:sue|file|litigate)",
            r"\blawsuit\s+(?:advice|recommendation)\b",
            r"\b(?:legal|attorney)\s+opinion\b",
            r"\bcontract\s+(?:advice|interpretation)\b",
        ],
        "medical_advice": [
            r"\bmedical\s+(?:advice|diagnosis|treatment)\b",
            r"\b(?:you\s+should|i\s+recommend)\s+(?:take|stop)\s+(?:medication|medicine)",
            r"\bprescription\s+(?:advice|recommendation)\b",
            r"\bdiagnos(?:e|is|ing)\b",
            r"\btreatment\s+(?:plan|recommendation)\b",
        ],
        "personal_data": [
            r"\bsocial\s+security\s+number\b",
            r"\bbank\s+account\s+(?:number|details)\b",
            r"\bcredit\s+card\s+(?:number|details)\b",
            r"\bpassword\b",
            r"\bpin\s+(?:number|code)\b",
        ],
    }
    
    COMPLIANCE_REQUIREMENTS = {
        ComplianceStandard.GDPR: {
            "data_minimization": True,
            "consent_required": True,
            "right_to_deletion": True,
            "data_portability": True,
            "breach_notification": True,
            "patterns_to_check": [
                r"\bpersonal\s+data\b",
                r"\bdata\s+subject\b",
                r"\bconsent\b",
            ],
        },
        ComplianceStandard.HIPAA: {
            "phi_protection": True,
            "minimum_necessary": True,
            "audit_trail": True,
            "patterns_to_check": [
                r"\bpatient\s+(?:name|id|record)\b",
                r"\bmedical\s+record\b",
                r"\bhealth\s+information\b",
                r"\bdiagnosis\b",
                r"\btreatment\b",
            ],
        },
        ComplianceStandard.PCI_DSS: {
            "card_data_protection": True,
            "encryption_required": True,
            "patterns_to_check": [
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                r"\bcvv\b",
                r"\bcard\s+(?:number|details)\b",
            ],
        },
        ComplianceStandard.SOC2: {
            "security_controls": True,
            "availability": True,
            "confidentiality": True,
            "patterns_to_check": [],
        },
    }
    
    INDUSTRY_DISCLAIMERS = {
        "finance": [
            "This information is for educational purposes only and should not be considered financial advice.",
            "Past performance is not indicative of future results.",
            "Please consult a licensed financial advisor before making investment decisions.",
        ],
        "healthcare": [
            "This information is not a substitute for professional medical advice.",
            "Please consult a healthcare provider for medical concerns.",
        ],
        "legal": [
            "This information is for general informational purposes only and does not constitute legal advice.",
            "Please consult a licensed attorney for legal matters.",
        ],
    }
    
    def __init__(self, config: DomainConstraintsConfig):
        """
        Initialize domain constraints enforcer.
        
        Args:
            config: Domain constraints configuration
        """
        self.config = config
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()
        
        logger.debug(f"DomainConstraintsEnforcer initialized for industry: {config.industry}")
    
    def _compile_patterns(self):
        """Compile regex patterns for performance."""
        for content_type in self.config.restricted_content:
            if content_type in self.RESTRICTED_CONTENT_PATTERNS:
                patterns = self.RESTRICTED_CONTENT_PATTERNS[content_type]
                self._compiled_patterns[content_type] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]
    
    async def check_input(self, content: str) -> ConstraintCheckResult:
        """
        Check input content against domain constraints.
        
        Args:
            content: Input content to check
            
        Returns:
            ConstraintCheckResult
        """
        violations = []
        
        for content_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    violations.append(ConstraintViolation(
                        constraint_type="restricted_content_request",
                        constraint_id=f"input_{content_type}",
                        message=f"Input requests {content_type.replace('_', ' ')}",
                        severity="medium",
                        action=self.config.action_on_violation,
                        metadata={"content_type": content_type},
                    ))
                    break
        
        return ConstraintCheckResult(
            passed=len(violations) == 0,
            violations=violations,
        )
    
    async def check_output(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ConstraintCheckResult:
        """
        Check output content against domain constraints.
        
        Args:
            content: Output content to check
            context: Optional context for compliance checks
            
        Returns:
            ConstraintCheckResult with violations and required disclaimers
        """
        violations = []
        disclaimers_required = []
        modified_content = content
        
        for content_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    violations.append(ConstraintViolation(
                        constraint_type="restricted_content",
                        constraint_id=f"output_{content_type}",
                        message=f"Output contains {content_type.replace('_', ' ')}",
                        severity="high",
                        action=self.config.action_on_violation,
                        metadata={"content_type": content_type},
                    ))
                    break
        
        if self.config.enable_compliance_checks:
            compliance_violations = await self._check_compliance(content, context)
            violations.extend(compliance_violations)
        
        if self.config.industry:
            industry_disclaimers = self.INDUSTRY_DISCLAIMERS.get(
                self.config.industry.lower(), []
            )
            
            for disclaimer in industry_disclaimers:
                if disclaimer not in content:
                    disclaimers_required.append(disclaimer)
        
        for disclaimer in self.config.required_disclaimers:
            if disclaimer not in content:
                disclaimers_required.append(disclaimer)
        
        business_violations = await self._check_business_rules(content, context)
        violations.extend(business_violations)
        
        action = self.config.action_on_violation
        if violations and action == RestrictionAction.REDACT.value:
            modified_content = self._redact_violations(content, violations)
        elif violations and action == RestrictionAction.DISCLAIMER.value:
            if disclaimers_required:
                disclaimer_text = "\n\n---\n" + "\n".join(disclaimers_required)
                modified_content = content + disclaimer_text
        
        passed = len(violations) == 0 or action in [
            RestrictionAction.WARN.value,
            RestrictionAction.DISCLAIMER.value,
        ]
        
        return ConstraintCheckResult(
            passed=passed,
            violations=violations,
            disclaimers_required=disclaimers_required,
            modified_content=modified_content if modified_content != content else None,
        )
    
    async def _check_compliance(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstraintViolation]:
        """Check content against compliance requirements."""
        violations = []
        
        for standard_str in self.config.compliance_standards:
            try:
                standard = ComplianceStandard(standard_str.upper())
            except ValueError:
                continue
            
            requirements = self.COMPLIANCE_REQUIREMENTS.get(standard, {})
            patterns = requirements.get("patterns_to_check", [])
            
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    if standard == ComplianceStandard.HIPAA:
                        violations.append(ConstraintViolation(
                            constraint_type="compliance",
                            constraint_id=f"hipaa_phi",
                            message="Potential PHI detected - HIPAA compliance required",
                            severity="critical",
                            action="block",
                            metadata={"standard": standard.value},
                        ))
                    elif standard == ComplianceStandard.PCI_DSS:
                        violations.append(ConstraintViolation(
                            constraint_type="compliance",
                            constraint_id=f"pci_card_data",
                            message="Card data detected - PCI DSS compliance required",
                            severity="critical",
                            action="block",
                            metadata={"standard": standard.value},
                        ))
        
        return violations
    
    async def _check_business_rules(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ConstraintViolation]:
        """Evaluate business rules against content."""
        violations = []
        
        for rule in self.config.business_rules:
            rule_id = rule.get("id", "unknown")
            rule_type = rule.get("type", "pattern")
            
            if rule_type == "pattern":
                pattern = rule.get("pattern")
                if pattern and re.search(pattern, content, re.IGNORECASE):
                    violations.append(ConstraintViolation(
                        constraint_type="business_rule",
                        constraint_id=rule_id,
                        message=rule.get("message", f"Business rule {rule_id} violated"),
                        severity=rule.get("severity", "medium"),
                        action=rule.get("action", "warn"),
                    ))
            
            elif rule_type == "forbidden_words":
                forbidden = rule.get("words", [])
                for word in forbidden:
                    if word.lower() in content.lower():
                        violations.append(ConstraintViolation(
                            constraint_type="business_rule",
                            constraint_id=rule_id,
                            message=f"Forbidden word detected: {word}",
                            severity=rule.get("severity", "medium"),
                            action=rule.get("action", "warn"),
                        ))
                        break
            
            elif rule_type == "required_elements":
                required = rule.get("elements", [])
                for element in required:
                    if element.lower() not in content.lower():
                        violations.append(ConstraintViolation(
                            constraint_type="business_rule",
                            constraint_id=rule_id,
                            message=f"Required element missing: {element}",
                            severity=rule.get("severity", "low"),
                            action=rule.get("action", "warn"),
                        ))
        
        return violations
    
    def _redact_violations(
        self,
        content: str,
        violations: List[ConstraintViolation],
    ) -> str:
        """Redact content that violates constraints."""
        redacted = content
        
        for violation in violations:
            if violation.constraint_type == "restricted_content":
                content_type = violation.metadata.get("content_type")
                if content_type and content_type in self._compiled_patterns:
                    for pattern in self._compiled_patterns[content_type]:
                        redacted = pattern.sub("[REDACTED]", redacted)
        
        return redacted
    
    def add_disclaimers(self, content: str, disclaimers: List[str]) -> str:
        """Add disclaimers to content."""
        if not disclaimers:
            return content
        
        disclaimer_section = "\n\n---\n**Disclaimers:**\n"
        for i, disclaimer in enumerate(disclaimers, 1):
            disclaimer_section += f"{i}. {disclaimer}\n"
        
        return content + disclaimer_section
    
    def get_required_disclaimers(self) -> List[str]:
        """Get all required disclaimers based on configuration."""
        disclaimers = list(self.config.required_disclaimers)
        
        if self.config.industry:
            industry_disclaimers = self.INDUSTRY_DISCLAIMERS.get(
                self.config.industry.lower(), []
            )
            for d in industry_disclaimers:
                if d not in disclaimers:
                    disclaimers.append(d)
        
        return disclaimers


def create_domain_constraints_enforcer(
    domain_config: Optional[Dict[str, Any]] = None,
) -> DomainConstraintsEnforcer:
    """Create domain constraints enforcer from config dict."""
    if domain_config is None:
        domain_config = {}
    
    config = DomainConstraintsConfig(
        restricted_content=domain_config.get("constraints", {}).get("restricted_content", []),
        compliance_standards=domain_config.get("compliance", []),
        required_disclaimers=domain_config.get("constraints", {}).get("required_disclaimers", []),
        business_rules=domain_config.get("business_rules", []),
        industry=domain_config.get("industry"),
        sub_domain=domain_config.get("sub_domain"),
        action_on_violation=domain_config.get("action_on_violation", "block"),
        enable_compliance_checks=domain_config.get("enable_compliance_checks", True),
    )
    
    return DomainConstraintsEnforcer(config)
