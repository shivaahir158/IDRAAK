"""Tests for evaluation metrics."""

import numpy as np
import pytest
from idraak.evaluation.metrics import ClassificationMetrics


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        result = ClassificationMetrics.compute(y_true, y_pred)
        assert result.accuracy == 1.0
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        result = ClassificationMetrics.compute(y_true, y_pred)
        assert result.accuracy == 0.0

    def test_with_probabilities(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9, 0.95]
        result = ClassificationMetrics.compute(y_true, y_pred, y_prob)
        assert result.auroc > 0.9
        assert result.brier_score < 0.1
        assert result.ece >= 0

    def test_empty_input(self):
        result = ClassificationMetrics.compute([], [])
        assert result.n_samples == 0

    def test_ece(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])
        ece = ClassificationMetrics._expected_calibration_error(y_true, y_prob, n_bins=5)
        assert 0 <= ece <= 1

    def test_per_group(self):
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]
        groups = ["en", "en", "hi", "hi", "ar", "ar"]
        results = ClassificationMetrics.per_group_metrics(y_true, y_pred, groups)
        assert "en" in results
        assert "hi" in results
        assert "ar" in results
