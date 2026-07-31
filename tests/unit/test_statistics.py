"""Tests for statistical significance testing."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from idraak.evaluation.statistics import (
    BootstrapTest,
    ConfidenceInterval,
    EffectSize,
    McNemarTest,
    PairedTestResult,
    WilcoxonTest,
)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


class TestBootstrapConfidenceInterval:
    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_pred = [1, 1, 0, 0, 1, 0, 1, 0]
        ci = BootstrapTest.confidence_interval(y_true, y_pred, _accuracy)
        assert ci.estimate == 1.0
        assert ci.ci_lower == 1.0
        assert ci.ci_upper == 1.0
        assert ci.confidence_level == 0.95

    def test_interval_contains_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=100).tolist()
        y_pred = rng.integers(0, 2, size=100).tolist()
        ci = BootstrapTest.confidence_interval(y_true, y_pred, _accuracy)
        assert ci.ci_lower <= ci.estimate <= ci.ci_upper

    def test_wider_at_lower_confidence(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=50).tolist()
        y_pred = rng.integers(0, 2, size=50).tolist()
        ci_90 = BootstrapTest.confidence_interval(
            y_true, y_pred, _accuracy, confidence_level=0.90
        )
        ci_99 = BootstrapTest.confidence_interval(
            y_true, y_pred, _accuracy, confidence_level=0.99
        )
        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 >= width_90

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BootstrapTest.confidence_interval([], [], _accuracy)

    def test_to_dict(self):
        ci = ConfidenceInterval(
            estimate=0.8, ci_lower=0.7, ci_upper=0.9,
            confidence_level=0.95, n_bootstrap=1000,
        )
        d = ci.to_dict()
        assert d["estimate"] == 0.8
        assert "ci_lower" in d
        assert "n_bootstrap" in d


class TestPairedBootstrap:
    def test_identical_methods_not_significant(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=100).tolist()
        y_pred = rng.integers(0, 2, size=100).tolist()
        result = BootstrapTest.paired_test(y_true, y_pred, y_pred, _accuracy)
        assert result.p_value == 1.0
        assert not result.significant
        assert result.method == "paired_bootstrap"

    def test_different_methods(self):
        y_true = [1] * 50 + [0] * 50
        y_pred_good = y_true[:]
        y_pred_bad = [0] * 100
        result = BootstrapTest.paired_test(
            y_true, y_pred_good, y_pred_bad, _accuracy
        )
        assert result.statistic > 0
        assert result.significant

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            BootstrapTest.paired_test([], [], [], _accuracy)


class TestMcNemarTest:
    def test_identical_classifiers(self):
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1]
        result = McNemarTest.test(y_true, y_pred, y_pred)
        assert result.p_value == 1.0
        assert not result.significant
        assert result.method == "mcnemar"

    def test_clearly_different_classifiers(self):
        n = 200
        y_true = np.array([1] * n + [0] * n)
        # Classifier A: mostly correct
        y_pred_a = y_true.copy()
        y_pred_a[:5] = 1 - y_pred_a[:5]
        # Classifier B: much worse
        y_pred_b = y_true.copy()
        y_pred_b[:80] = 1 - y_pred_b[:80]
        result = McNemarTest.test(y_true, y_pred_a, y_pred_b)
        assert result.significant
        assert result.p_value < 0.05

    def test_without_correction(self):
        y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        y_pred_a = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
        y_pred_b = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result = McNemarTest.test(y_true, y_pred_a, y_pred_b, correction=False)
        assert result.statistic >= 0

    def test_to_dict(self):
        result = PairedTestResult(
            statistic=3.5, p_value=0.06, method="mcnemar",
            significant=False, effect_size=None,
        )
        d = result.to_dict()
        assert d["method"] == "mcnemar"
        assert d["p_value"] == 0.06


class TestWilcoxonTest:
    def test_identical_scores(self):
        scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        result = WilcoxonTest.test(scores, scores)
        assert result.p_value == 1.0
        assert not result.significant

    def test_clearly_different_scores(self):
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.93]
        scores_b = [0.5, 0.45, 0.48, 0.52, 0.47, 0.51, 0.46, 0.53]
        result = WilcoxonTest.test(scores_a, scores_b)
        assert result.significant
        assert result.p_value < 0.05
        assert result.method == "wilcoxon"

    def test_one_sided(self):
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.93]
        scores_b = [0.5, 0.45, 0.48, 0.52, 0.47, 0.51, 0.46, 0.53]
        result = WilcoxonTest.test(scores_a, scores_b, alternative="greater")
        assert result.significant


class TestEffectSize:
    def test_identical_groups(self):
        scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        d = EffectSize.cohens_d(scores, scores)
        assert d == 0.0

    def test_large_effect(self):
        scores_a = [10.0, 11.0, 12.0, 10.5, 11.5]
        scores_b = [1.0, 2.0, 1.5, 2.5, 1.8]
        d = EffectSize.cohens_d(scores_a, scores_b)
        assert d > 0.8
        assert EffectSize.interpret(d) == "large"

    def test_negative_effect(self):
        scores_a = [1.0, 2.0, 1.5]
        scores_b = [10.0, 11.0, 10.5]
        d = EffectSize.cohens_d(scores_a, scores_b)
        assert d < 0

    def test_interpret_thresholds(self):
        assert EffectSize.interpret(0.0) == "negligible"
        assert EffectSize.interpret(0.1) == "negligible"
        assert EffectSize.interpret(0.3) == "small"
        assert EffectSize.interpret(0.6) == "medium"
        assert EffectSize.interpret(1.0) == "large"
        assert EffectSize.interpret(-0.9) == "large"

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 2"):
            EffectSize.cohens_d([1.0], [2.0])

    def test_zero_variance(self):
        d = EffectSize.cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert d == 0.0
