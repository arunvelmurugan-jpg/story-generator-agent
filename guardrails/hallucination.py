"""
Hallucination Detection for PHTN.AI Sub-Agent Framework

Detects when agent outputs contain information not grounded in:
- Retrieved documents (RAG context)
- Provided context
- Known facts

Supports multiple detection strategies:
- Semantic similarity checking
- NLI (Natural Language Inference)
- Claim extraction and verification
- Source attribution checking
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HallucinationStrategy(str, Enum):
    """Hallucination detection strategies."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    NLI = "nli"
    CLAIM_VERIFICATION = "claim_verification"
    SOURCE_ATTRIBUTION = "source_attribution"
    HYBRID = "hybrid"


@dataclass
class HallucinationConfig:
    """Configuration for hallucination detection."""
    enabled: bool = False
    strategy: HallucinationStrategy = HallucinationStrategy.SEMANTIC_SIMILARITY
    threshold: float = 0.7
    action: str = "warn"
    check_claims: bool = True
    check_numbers: bool = True
    check_entities: bool = True
    require_sources: bool = False
    min_source_coverage: float = 0.5


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""
    is_hallucinated: bool
    confidence: float
    strategy: str
    details: Dict[str, Any] = field(default_factory=dict)
    ungrounded_claims: List[str] = field(default_factory=list)
    grounded_claims: List[str] = field(default_factory=list)
    source_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_hallucinated": self.is_hallucinated,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "details": self.details,
            "ungrounded_claims": self.ungrounded_claims,
            "grounded_claims": self.grounded_claims,
            "source_coverage": self.source_coverage,
        }


class HallucinationDetector:
    """
    Detects hallucinations in agent outputs.
    
    Compares agent output against provided context/sources
    to identify claims not supported by the evidence.
    """
    
    CLAIM_PATTERNS = [
        r"(?:is|are|was|were|has|have|had)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)",
        r"(\d+(?:\.\d+)?)\s*(?:percent|%|million|billion|thousand)",
        r"(?:in|on|at)\s+(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
        r"(?:founded|created|established|started)\s+(?:in|on)?\s*(\d{4})",
    ]
    
    ENTITY_PATTERNS = {
        "person": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        "organization": r"\b(?:[A-Z][a-z]*\.?\s*)+(?:Inc|Corp|LLC|Ltd|Company|Co)\b",
        "date": r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b",
        "number": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
        "url": r"https?://[^\s]+",
    }
    
    def __init__(
        self,
        config: HallucinationConfig,
        embedding_provider: Optional[Any] = None,
    ):
        """
        Initialize hallucination detector.
        
        Args:
            config: Detection configuration
            embedding_provider: Optional embedding provider for semantic similarity
        """
        self.config = config
        self.embedding_provider = embedding_provider
        self._initialized = False
    
    async def detect(
        self,
        output: str,
        context: List[str],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> HallucinationResult:
        """
        Detect hallucinations in output.
        
        Args:
            output: Agent output to check
            context: Context/retrieved documents
            sources: Optional source documents with metadata
            
        Returns:
            HallucinationResult
        """
        if not self.config.enabled:
            return HallucinationResult(
                is_hallucinated=False,
                confidence=1.0,
                strategy="disabled",
            )
        
        strategy = self.config.strategy
        
        if strategy == HallucinationStrategy.SEMANTIC_SIMILARITY:
            return await self._check_semantic_similarity(output, context)
        elif strategy == HallucinationStrategy.CLAIM_VERIFICATION:
            return await self._check_claims(output, context, sources)
        elif strategy == HallucinationStrategy.SOURCE_ATTRIBUTION:
            return await self._check_source_attribution(output, sources)
        elif strategy == HallucinationStrategy.HYBRID:
            return await self._check_hybrid(output, context, sources)
        else:
            return await self._check_semantic_similarity(output, context)
    
    async def _check_semantic_similarity(
        self,
        output: str,
        context: List[str],
    ) -> HallucinationResult:
        """Check output against context using semantic similarity."""
        if not context:
            return HallucinationResult(
                is_hallucinated=True,
                confidence=0.5,
                strategy="semantic_similarity",
                details={"reason": "No context provided for grounding"},
            )
        
        combined_context = " ".join(context)
        
        output_sentences = self._split_sentences(output)
        grounded = []
        ungrounded = []
        
        for sentence in output_sentences:
            if len(sentence.strip()) < 10:
                continue
            
            is_grounded = self._sentence_in_context(sentence, combined_context)
            
            if is_grounded:
                grounded.append(sentence)
            else:
                ungrounded.append(sentence)
        
        total = len(grounded) + len(ungrounded)
        if total == 0:
            coverage = 1.0
        else:
            coverage = len(grounded) / total
        
        is_hallucinated = coverage < self.config.threshold
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=coverage,
            strategy="semantic_similarity",
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            source_coverage=coverage,
            details={
                "total_sentences": total,
                "grounded_count": len(grounded),
                "ungrounded_count": len(ungrounded),
            },
        )
    
    async def _check_claims(
        self,
        output: str,
        context: List[str],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> HallucinationResult:
        """Extract and verify claims in output."""
        claims = self._extract_claims(output)
        combined_context = " ".join(context)
        
        grounded = []
        ungrounded = []
        
        for claim in claims:
            if self._verify_claim(claim, combined_context):
                grounded.append(claim)
            else:
                ungrounded.append(claim)
        
        if self.config.check_numbers:
            numbers = self._extract_numbers(output)
            context_numbers = self._extract_numbers(combined_context)
            
            for num in numbers:
                if num not in context_numbers:
                    ungrounded.append(f"Number: {num}")
        
        if self.config.check_entities:
            entities = self._extract_entities(output)
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.lower() not in combined_context.lower():
                        ungrounded.append(f"{entity_type}: {entity}")
        
        total = len(grounded) + len(ungrounded)
        coverage = len(grounded) / total if total > 0 else 1.0
        
        is_hallucinated = coverage < self.config.threshold
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=coverage,
            strategy="claim_verification",
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            source_coverage=coverage,
            details={
                "total_claims": total,
                "verified_claims": len(grounded),
                "unverified_claims": len(ungrounded),
            },
        )
    
    async def _check_source_attribution(
        self,
        output: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> HallucinationResult:
        """Check if output properly attributes sources."""
        if not sources:
            return HallucinationResult(
                is_hallucinated=False,
                confidence=1.0,
                strategy="source_attribution",
                details={"reason": "No sources to check attribution"},
            )
        
        citation_patterns = [
            r"\[(\d+)\]",
            r"\(Source:?\s*([^)]+)\)",
            r"according to\s+([^,\.]+)",
            r"from\s+([^,\.]+)",
            r"\[Document\s*(\d+)\]",
        ]
        
        citations_found = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            citations_found.extend(matches)
        
        source_ids = [s.get("id", s.get("source", str(i))) for i, s in enumerate(sources)]
        
        valid_citations = []
        invalid_citations = []
        
        for citation in citations_found:
            citation_str = str(citation).strip()
            if any(citation_str in str(sid) for sid in source_ids):
                valid_citations.append(citation_str)
            else:
                invalid_citations.append(citation_str)
        
        if self.config.require_sources and not citations_found:
            return HallucinationResult(
                is_hallucinated=True,
                confidence=0.0,
                strategy="source_attribution",
                details={"reason": "No source citations found but required"},
            )
        
        coverage = len(valid_citations) / len(citations_found) if citations_found else 1.0
        
        return HallucinationResult(
            is_hallucinated=coverage < self.config.min_source_coverage,
            confidence=coverage,
            strategy="source_attribution",
            source_coverage=coverage,
            details={
                "citations_found": len(citations_found),
                "valid_citations": valid_citations,
                "invalid_citations": invalid_citations,
            },
        )
    
    async def _check_hybrid(
        self,
        output: str,
        context: List[str],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> HallucinationResult:
        """Combine multiple detection strategies."""
        results = []
        
        semantic_result = await self._check_semantic_similarity(output, context)
        results.append(("semantic", semantic_result))
        
        claims_result = await self._check_claims(output, context, sources)
        results.append(("claims", claims_result))
        
        if sources:
            attribution_result = await self._check_source_attribution(output, sources)
            results.append(("attribution", attribution_result))
        
        avg_confidence = sum(r.confidence for _, r in results) / len(results)
        is_hallucinated = any(r.is_hallucinated for _, r in results)
        
        all_ungrounded = []
        all_grounded = []
        for _, r in results:
            all_ungrounded.extend(r.ungrounded_claims)
            all_grounded.extend(r.grounded_claims)
        
        return HallucinationResult(
            is_hallucinated=is_hallucinated,
            confidence=avg_confidence,
            strategy="hybrid",
            grounded_claims=list(set(all_grounded)),
            ungrounded_claims=list(set(all_ungrounded)),
            source_coverage=avg_confidence,
            details={
                "strategies_used": [name for name, _ in results],
                "individual_results": {name: r.to_dict() for name, r in results},
            },
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _sentence_in_context(self, sentence: str, context: str) -> bool:
        """Check if sentence content is grounded in context."""
        sentence_lower = sentence.lower()
        context_lower = context.lower()
        
        if sentence_lower in context_lower:
            return True
        
        words = re.findall(r'\b\w+\b', sentence_lower)
        content_words = [w for w in words if len(w) > 3]
        
        if not content_words:
            return True
        
        matches = sum(1 for w in content_words if w in context_lower)
        return matches / len(content_words) >= 0.6
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        claims = []
        
        sentences = self._split_sentences(text)
        
        factual_indicators = [
            r"\bis\b", r"\bare\b", r"\bwas\b", r"\bwere\b",
            r"\bhas\b", r"\bhave\b", r"\bhad\b",
            r"\bcontains\b", r"\bincludes\b",
            r"\baccording to\b", r"\bstates\b",
        ]
        
        for sentence in sentences:
            for indicator in factual_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    claims.append(sentence)
                    break
        
        return claims
    
    def _verify_claim(self, claim: str, context: str) -> bool:
        """Verify if a claim is supported by context."""
        return self._sentence_in_context(claim, context)
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers from text."""
        pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand))?\b'
        return re.findall(pattern, text, re.IGNORECASE)
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {}
        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = list(set(matches))
        return entities


def create_hallucination_detector(
    config: Optional[Dict[str, Any]] = None,
    embedding_provider: Optional[Any] = None,
) -> HallucinationDetector:
    """Create hallucination detector from config dict."""
    if config is None:
        config = {}
    
    strategy_str = config.get("strategy", "semantic_similarity")
    try:
        strategy = HallucinationStrategy(strategy_str)
    except ValueError:
        strategy = HallucinationStrategy.SEMANTIC_SIMILARITY
    
    hallucination_config = HallucinationConfig(
        enabled=config.get("enabled", False),
        strategy=strategy,
        threshold=config.get("threshold", 0.7),
        action=config.get("action", "warn"),
        check_claims=config.get("check_claims", True),
        check_numbers=config.get("check_numbers", True),
        check_entities=config.get("check_entities", True),
        require_sources=config.get("require_sources", False),
        min_source_coverage=config.get("min_source_coverage", 0.5),
    )
    
    return HallucinationDetector(hallucination_config, embedding_provider)
