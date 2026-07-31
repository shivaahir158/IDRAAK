"""Evaluation metrics and analysis."""

from idraak.evaluation.baselines import EmbeddingBaseline, ExactMatchBaseline, TokenOverlapBaseline
from idraak.evaluation.complexity import ComplexityScorer
from idraak.evaluation.error_analysis import ErrorAnalyzer
from idraak.evaluation.human_review import HumanReviewRouter
from idraak.evaluation.metrics import ClassificationMetrics
from idraak.evaluation.statistics import (
    BootstrapTest,
    ConfidenceInterval,
    EffectSize,
    McNemarTest,
    PairedTestResult,
    WilcoxonTest,
)

__all__ = [
    "BootstrapTest",
    "ClassificationMetrics",
    "ComplexityScorer",
    "ConfidenceInterval",
    "EmbeddingBaseline",
    "EffectSize",
    "ErrorAnalyzer",
    "ExactMatchBaseline",
    "HumanReviewRouter",
    "McNemarTest",
    "PairedTestResult",
    "TokenOverlapBaseline",
    "WilcoxonTest",
]
