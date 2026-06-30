"""
Guardrails Engine for PHTN.AI Sub-Agent Framework

Provides comprehensive guardrails for:
- Input validation and filtering
- Output safety checks
- Tool call validation
- Custom rule enforcement
- Toxicity detection (Perspective, OpenAI, HuggingFace, Local)
- Hallucination detection
- Domain constraints enforcement
- Rate limiting
"""

from .engine import GuardrailsEngine, GuardrailResult
from .toxicity import (
    ToxicityConfig,
    ToxicityProvider,
    ToxicityCategory,
    ToxicityScore,
    ToxicityResult,
    BaseToxicityDetector,
    LocalToxicityDetector,
    PerspectiveToxicityDetector,
    OpenAIToxicityDetector,
    HuggingFaceToxicityDetector,
    create_toxicity_detector,
)
from .hallucination import (
    HallucinationConfig,
    HallucinationStrategy,
    HallucinationResult,
    HallucinationDetector,
    create_hallucination_detector,
)
from .domain_constraints import (
    DomainConstraintsConfig,
    DomainConstraintsEnforcer,
    ConstraintViolation,
    ConstraintCheckResult,
    ComplianceStandard,
    RestrictionAction,
    create_domain_constraints_enforcer,
)
from .rate_limiter import (
    RateLimitConfig,
    RateLimitStrategy,
    RateLimitResult,
    RateLimiter,
    TokenBucket,
    SlidingWindow,
    FixedWindow,
    get_rate_limiter,
    create_rate_limiter,
)

__all__ = [
    "GuardrailsEngine",
    "GuardrailResult",
    "ToxicityConfig",
    "ToxicityProvider",
    "ToxicityCategory",
    "ToxicityScore",
    "ToxicityResult",
    "BaseToxicityDetector",
    "LocalToxicityDetector",
    "PerspectiveToxicityDetector",
    "OpenAIToxicityDetector",
    "HuggingFaceToxicityDetector",
    "create_toxicity_detector",
    "HallucinationConfig",
    "HallucinationStrategy",
    "HallucinationResult",
    "HallucinationDetector",
    "create_hallucination_detector",
    "DomainConstraintsConfig",
    "DomainConstraintsEnforcer",
    "ConstraintViolation",
    "ConstraintCheckResult",
    "ComplianceStandard",
    "RestrictionAction",
    "create_domain_constraints_enforcer",
    "RateLimitConfig",
    "RateLimitStrategy",
    "RateLimitResult",
    "RateLimiter",
    "TokenBucket",
    "SlidingWindow",
    "FixedWindow",
    "get_rate_limiter",
    "create_rate_limiter",
]
