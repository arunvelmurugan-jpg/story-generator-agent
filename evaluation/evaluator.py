"""
Evaluator - Main evaluation orchestrator
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .metrics import (
    EvaluationMetric, MetricResult, RelevanceMetric, FaithfulnessMetric,
    HallucinationMetric, ToxicityMetric, CoherenceMetric, CompletenessMetric
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""
    metrics: List[str] = field(default_factory=lambda: ["relevance", "faithfulness", "coherence"])
    fail_fast: bool = False
    include_explanations: bool = True
    custom_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    query: str
    response: str
    metrics: List[MetricResult]
    overall_score: float
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEvaluationResult:
    """Result of batch evaluation."""
    results: List[EvaluationResult]
    summary: Dict[str, Any]
    total_examples: int
    passed_examples: int
    average_scores: Dict[str, float]


class Evaluator:
    """
    Main evaluator class for assessing agent responses.
    
    Features:
    - Multiple metric support
    - Batch evaluation
    - Custom metrics
    - Threshold configuration
    - Detailed reporting
    """
    
    METRIC_CLASSES = {
        "relevance": RelevanceMetric,
        "faithfulness": FaithfulnessMetric,
        "hallucination": HallucinationMetric,
        "toxicity": ToxicityMetric,
        "coherence": CoherenceMetric,
        "completeness": CompletenessMetric,
    }
    
    def __init__(self, config: Optional[EvaluatorConfig] = None):
        self.config = config or EvaluatorConfig()
        self._metrics: Dict[str, EvaluationMetric] = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize configured metrics."""
        for metric_name in self.config.metrics:
            if metric_name in self.METRIC_CLASSES:
                threshold = self.config.custom_thresholds.get(metric_name)
                if threshold:
                    self._metrics[metric_name] = self.METRIC_CLASSES[metric_name](threshold=threshold)
                else:
                    self._metrics[metric_name] = self.METRIC_CLASSES[metric_name]()
    
    def add_metric(self, metric: EvaluationMetric):
        """Add a custom metric."""
        self._metrics[metric.name] = metric
    
    async def evaluate(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        expected: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Evaluate a single response."""
        metric_results = []
        all_passed = True
        
        for metric_name, metric in self._metrics.items():
            try:
                result = await metric.evaluate(query, response, context, expected)
                metric_results.append(result)
                
                if not result.passed:
                    all_passed = False
                    if self.config.fail_fast:
                        break
            except Exception as e:
                logger.error(f"Metric {metric_name} failed: {e}")
                metric_results.append(MetricResult(
                    metric_name=metric_name, score=0.0, passed=False,
                    explanation=f"Error: {e}"
                ))
        
        overall_score = sum(r.score for r in metric_results) / len(metric_results) if metric_results else 0.0
        
        return EvaluationResult(
            query=query,
            response=response,
            metrics=metric_results,
            overall_score=overall_score,
            passed=all_passed,
            metadata=metadata or {}
        )
    
    async def evaluate_batch(
        self,
        examples: List[Dict[str, Any]]
    ) -> BatchEvaluationResult:
        """Evaluate a batch of examples."""
        results = []
        
        for example in examples:
            result = await self.evaluate(
                query=example.get("query", ""),
                response=example.get("response", ""),
                context=example.get("context"),
                expected=example.get("expected"),
                metadata=example.get("metadata")
            )
            results.append(result)
        
        passed_count = sum(1 for r in results if r.passed)
        
        avg_scores = {}
        for metric_name in self._metrics:
            scores = [r.metrics[i].score for r in results for i, m in enumerate(r.metrics) if m.metric_name == metric_name]
            avg_scores[metric_name] = sum(scores) / len(scores) if scores else 0.0
        
        return BatchEvaluationResult(
            results=results,
            summary={
                "total": len(results),
                "passed": passed_count,
                "failed": len(results) - passed_count,
                "pass_rate": passed_count / len(results) if results else 0.0
            },
            total_examples=len(results),
            passed_examples=passed_count,
            average_scores=avg_scores
        )
    
    def get_report(self, result: EvaluationResult) -> str:
        """Generate a human-readable report."""
        lines = [
            f"Evaluation Report",
            f"=" * 50,
            f"Query: {result.query[:100]}...",
            f"Response: {result.response[:100]}...",
            f"Overall Score: {result.overall_score:.2f}",
            f"Passed: {result.passed}",
            f"",
            "Metrics:",
            "-" * 30
        ]
        
        for metric in result.metrics:
            lines.append(f"  {metric.metric_name}: {metric.score:.2f} ({'PASS' if metric.passed else 'FAIL'})")
            if self.config.include_explanations and metric.explanation:
                lines.append(f"    {metric.explanation}")
        
        return "\n".join(lines)


def create_evaluator(
    metrics: Optional[List[str]] = None,
    **kwargs
) -> Evaluator:
    """Factory function to create an evaluator."""
    config = EvaluatorConfig(metrics=metrics or ["relevance", "coherence"], **kwargs)
    return Evaluator(config)


__all__ = ["Evaluator", "EvaluatorConfig", "EvaluationResult", "BatchEvaluationResult", "create_evaluator"]
