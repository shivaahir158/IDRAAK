"""Classification and calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from idraak.utils.logging import get_logger

logger = get_logger("metrics")


@dataclass
class MetricResult:
    """Container for evaluation metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    macro_f1: float = 0.0
    mcc: float = 0.0
    auroc: float = 0.0
    average_precision: float = 0.0
    balanced_accuracy: float = 0.0
    confusion_matrix: list[list[int]] | None = None
    ece: float = 0.0
    brier_score: float = 0.0
    n_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "macro_f1": self.macro_f1,
            "mcc": self.mcc,
            "auroc": self.auroc,
            "average_precision": self.average_precision,
            "balanced_accuracy": self.balanced_accuracy,
            "ece": self.ece,
            "brier_score": self.brier_score,
            "n_samples": self.n_samples,
        }


class ClassificationMetrics:
    """Compute classification and calibration metrics."""

    @staticmethod
    def compute(
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        y_prob: list[float] | np.ndarray | None = None,
        n_bins: int = 15,
    ) -> MetricResult:
        """Compute all metrics from labels and predictions."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_true) == 0:
            return MetricResult()

        result = MetricResult(n_samples=len(y_true))

        result.accuracy = float(accuracy_score(y_true, y_pred))
        result.precision = float(precision_score(y_true, y_pred, zero_division=0))
        result.recall = float(recall_score(y_true, y_pred, zero_division=0))
        result.f1 = float(f1_score(y_true, y_pred, zero_division=0))
        result.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        result.mcc = float(matthews_corrcoef(y_true, y_pred))
        result.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        result.confusion_matrix = cm.tolist()

        if y_prob is not None:
            y_prob = np.array(y_prob)
            try:
                result.auroc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                result.auroc = 0.0
            try:
                result.average_precision = float(average_precision_score(y_true, y_prob))
            except ValueError:
                result.average_precision = 0.0

            # Brier score
            result.brier_score = float(np.mean((y_prob - y_true) ** 2))

            # ECE
            result.ece = ClassificationMetrics._expected_calibration_error(
                y_true, y_prob, n_bins
            )

        return result

    @staticmethod
    def _expected_calibration_error(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = mask | (y_prob == bin_boundaries[i + 1])
            n_bin = mask.sum()
            if n_bin > 0:
                avg_confidence = y_prob[mask].mean()
                avg_accuracy = y_true[mask].mean()
                ece += (n_bin / n) * abs(avg_accuracy - avg_confidence)

        return float(ece)

    @staticmethod
    def per_group_metrics(
        y_true: list[int],
        y_pred: list[int],
        groups: list[str],
        y_prob: list[float] | None = None,
    ) -> dict[str, MetricResult]:
        """Compute metrics for each group (language, domain, drift type, etc.)."""
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_prob_arr = np.array(y_prob) if y_prob else None
        groups_arr = np.array(groups)

        results = {}
        for group in sorted(set(groups)):
            mask = groups_arr == group
            group_prob = y_prob_arr[mask] if y_prob_arr is not None else None
            results[group] = ClassificationMetrics.compute(
                y_true_arr[mask], y_pred_arr[mask], group_prob
            )
        return results
