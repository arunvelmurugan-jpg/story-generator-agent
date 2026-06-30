"""
Evaluation Framework
"""

from .metrics import (
    EvaluationMetric, MetricResult, RelevanceMetric, FaithfulnessMetric,
    HallucinationMetric, ToxicityMetric, CoherenceMetric, CompletenessMetric
)
from .evaluator import Evaluator, EvaluatorConfig, EvaluationResult, BatchEvaluationResult
from .dataset import EvaluationDataset, EvaluationExample, DatasetLoader

__all__ = [
    "EvaluationMetric", "MetricResult", "RelevanceMetric", "FaithfulnessMetric",
    "HallucinationMetric", "ToxicityMetric", "CoherenceMetric", "CompletenessMetric",
    "Evaluator", "EvaluatorConfig", "EvaluationResult", "BatchEvaluationResult",
    "EvaluationDataset", "EvaluationExample", "DatasetLoader"
]
