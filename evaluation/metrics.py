"""
Evaluation Metrics
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    RELEVANCE = "relevance"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    CUSTOM = "custom"


@dataclass
class MetricResult:
    """Result of a metric evaluation."""
    metric_name: str
    score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class EvaluationMetric(ABC):
    """Base class for evaluation metrics."""
    
    name: str = "base_metric"
    description: str = ""
    threshold: float = 0.5
    
    @abstractmethod
    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        expected: Optional[str] = None
    ) -> MetricResult:
        """Evaluate the response."""
        pass


class RelevanceMetric(EvaluationMetric):
    """Measures how relevant the response is to the query."""
    
    name = "relevance"
    description = "Measures response relevance to the query"
    
    def __init__(self, threshold: float = 0.7, use_llm: bool = False):
        self.threshold = threshold
        self.use_llm = use_llm
    
    async def evaluate(
        self, query: str, response: str,
        context: Optional[str] = None, expected: Optional[str] = None
    ) -> MetricResult:
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            score = 0.0
        else:
            overlap = len(query_words & response_words)
            score = min(1.0, overlap / len(query_words))
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            details={"query_words": len(query_words), "overlap": overlap if query_words else 0},
            explanation=f"Response contains {score*100:.1f}% of query terms"
        )


class FaithfulnessMetric(EvaluationMetric):
    """Measures if the response is faithful to the provided context."""
    
    name = "faithfulness"
    description = "Measures if response is grounded in context"
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    async def evaluate(
        self, query: str, response: str,
        context: Optional[str] = None, expected: Optional[str] = None
    ) -> MetricResult:
        if not context:
            return MetricResult(
                metric_name=self.name, score=1.0, passed=True,
                explanation="No context provided, skipping faithfulness check"
            )
        
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                       'into', 'through', 'during', 'before', 'after', 'above', 'below',
                       'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                       'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just'}
        
        response_content = response_words - common_words
        context_content = context_words - common_words
        
        if not response_content:
            score = 1.0
        else:
            grounded = len(response_content & context_content)
            score = grounded / len(response_content)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            details={"grounded_terms": grounded if response_content else 0},
            explanation=f"{score*100:.1f}% of response terms found in context"
        )


class HallucinationMetric(EvaluationMetric):
    """Detects potential hallucinations in the response."""
    
    name = "hallucination"
    description = "Detects ungrounded claims in response"
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    async def evaluate(
        self, query: str, response: str,
        context: Optional[str] = None, expected: Optional[str] = None
    ) -> MetricResult:
        if not context:
            return MetricResult(
                metric_name=self.name, score=0.0, passed=True,
                explanation="No context to check against"
            )
        
        faithfulness = FaithfulnessMetric()
        faith_result = await faithfulness.evaluate(query, response, context)
        
        hallucination_score = 1.0 - faith_result.score
        
        return MetricResult(
            metric_name=self.name,
            score=hallucination_score,
            passed=hallucination_score <= self.threshold,
            details={"faithfulness_score": faith_result.score},
            explanation=f"Potential hallucination score: {hallucination_score*100:.1f}%"
        )


class ToxicityMetric(EvaluationMetric):
    """Detects toxic content in the response."""
    
    name = "toxicity"
    description = "Detects toxic or harmful content"
    
    TOXIC_PATTERNS = [
        "hate", "kill", "die", "stupid", "idiot", "dumb",
        "racist", "sexist", "discriminat"
    ]
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
    
    async def evaluate(
        self, query: str, response: str,
        context: Optional[str] = None, expected: Optional[str] = None
    ) -> MetricResult:
        response_lower = response.lower()
        found_patterns = [p for p in self.TOXIC_PATTERNS if p in response_lower]
        
        score = min(1.0, len(found_patterns) * 0.2)
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score <= self.threshold,
            details={"found_patterns": found_patterns},
            explanation=f"Found {len(found_patterns)} potentially toxic patterns"
        )


class CoherenceMetric(EvaluationMetric):
    """Measures response coherence and readability."""
    
    name = "coherence"
    description = "Measures response coherence"
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    async def evaluate(
        self, query: str, response: str,
        context: Optional[str] = None, expected: Optional[str] = None
    ) -> MetricResult:
        sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        if len(sentences) == 0:
            score = 0.0
        elif len(sentences) == 1:
            score = 0.7
        else:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            length_score = min(1.0, avg_length / 15)
            structure_score = min(1.0, len(sentences) / 5)
            score = (length_score + structure_score) / 2
        
        return MetricResult(
            metric_name=self.name,
            score=score,
            passed=score >= self.threshold,
            details={"sentence_count": len(sentences)},
            explanation=f"Coherence score: {score*100:.1f}%"
        )


class CompletenessMetric(EvaluationMetric):
    """Measures if the response fully addresses the query."""
    
    name = "completeness"
    description = "Measures if response addresses the query"
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    async def evaluate(
        self, query: str, response: str,
        context: Optional[str] = None, expected: Optional[str] = None
    ) -> MetricResult:
        if expected:
            expected_words = set(expected.lower().split())
            response_words = set(response.lower().split())
            if expected_words:
                coverage = len(expected_words & response_words) / len(expected_words)
            else:
                coverage = 1.0
        else:
            min_length = max(20, len(query.split()) * 3)
            coverage = min(1.0, len(response.split()) / min_length)
        
        return MetricResult(
            metric_name=self.name,
            score=coverage,
            passed=coverage >= self.threshold,
            details={"response_length": len(response.split())},
            explanation=f"Completeness score: {coverage*100:.1f}%"
        )


__all__ = [
    "MetricType", "MetricResult", "EvaluationMetric",
    "RelevanceMetric", "FaithfulnessMetric", "HallucinationMetric",
    "ToxicityMetric", "CoherenceMetric", "CompletenessMetric"
]
