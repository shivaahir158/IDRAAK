"""Statistical significance testing for evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats

from idraak.utils.logging import get_logger

logger = get_logger("statistics")


@dataclass
class ConfidenceInterval:
    """Result of a bootstrap confidence interval estimation."""

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
        }


@dataclass
class PairedTestResult:
    """Result of a paired statistical test."""

    statistic: float
    p_value: float
    method: str
    significant: bool
    effect_size: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "method": self.method,
            "significant": self.significant,
            "effect_size": self.effect_size,
        }


class BootstrapTest:
    """Bootstrap-based confidence intervals and paired comparison tests."""

    @staticmethod
    def confidence_interval(
        y_true: list[int] | np.ndarray,
        y_pred: list[int] | np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 10_000,
        random_state: int | None = 42,
    ) -> ConfidenceInterval:
        """Compute a bootstrap confidence interval for any metric.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels or scores.
            metric_fn: Callable(y_true, y_pred) -> float.
            confidence_level: Confidence level (e.g. 0.95 for 95%).
            n_bootstrap: Number of bootstrap resamples.
            random_state: Seed for reproducibility.

        Returns:
            A ``ConfidenceInterval`` with point estimate and bounds.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)

        if n == 0:
            raise ValueError("Input arrays must not be empty.")

        rng = np.random.default_rng(random_state)
        point_estimate = float(metric_fn(y_true, y_pred))

        bootstrap_scores = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bootstrap_scores[i] = metric_fn(y_true[idx], y_pred[idx])

        alpha = 1.0 - confidence_level
        ci_lower = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        ci_upper = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

        logger.debug(
            "Bootstrap CI: estimate=%.4f, [%.4f, %.4f], n_bootstrap=%d",
            point_estimate,
            ci_lower,
            ci_upper,
            n_bootstrap,
        )

        return ConfidenceInterval(
            estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
        )

    @staticmethod
    def paired_test(
        y_true: list[int] | np.ndarray,
        y_pred_a: list[int] | np.ndarray,
        y_pred_b: list[int] | np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_bootstrap: int = 10_000,
        alpha: float = 0.05,
        random_state: int | None = 42,
    ) -> PairedTestResult:
        """Paired bootstrap test comparing two methods on the same data.

        Tests whether method A significantly outperforms method B by
        computing the fraction of bootstrap resamples where the performance
        difference is non-positive (one-sided p-value).

        Args:
            y_true: Ground-truth labels.
            y_pred_a: Predictions from method A.
            y_pred_b: Predictions from method B.
            metric_fn: Callable(y_true, y_pred) -> float.
            n_bootstrap: Number of bootstrap resamples.
            alpha: Significance threshold.
            random_state: Seed for reproducibility.

        Returns:
            A ``PairedTestResult`` with test statistic (mean delta) and p-value.
        """
        y_true = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred_a)
        y_pred_b = np.asarray(y_pred_b)
        n = len(y_true)

        if n == 0:
            raise ValueError("Input arrays must not be empty.")

        rng = np.random.default_rng(random_state)
        observed_diff = float(metric_fn(y_true, y_pred_a) - metric_fn(y_true, y_pred_b))

        deltas = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            score_a = metric_fn(y_true[idx], y_pred_a[idx])
            score_b = metric_fn(y_true[idx], y_pred_b[idx])
            deltas[i] = score_a - score_b

        # Two-sided p-value: fraction of resamples where sign differs
        p_value = float(np.mean(deltas * np.sign(observed_diff) <= 0)) if observed_diff != 0.0 else 1.0

        logger.debug(
            "Paired bootstrap: diff=%.4f, p=%.4f, significant=%s",
            observed_diff,
            p_value,
            p_value < alpha,
        )

        return PairedTestResult(
            statistic=observed_diff,
            p_value=p_value,
            method="paired_bootstrap",
            significant=p_value < alpha,
        )


class McNemarTest:
    """McNemar's test for comparing paired classifiers."""

    @staticmethod
    def test(
        y_true: list[int] | np.ndarray,
        y_pred_a: list[int] | np.ndarray,
        y_pred_b: list[int] | np.ndarray,
        alpha: float = 0.05,
        correction: bool = True,
    ) -> PairedTestResult:
        """Run McNemar's test on two sets of predictions.

        Compares the discordant pairs (samples where one classifier is correct
        and the other is wrong) using a chi-squared test.

        Args:
            y_true: Ground-truth labels.
            y_pred_a: Predictions from classifier A.
            y_pred_b: Predictions from classifier B.
            alpha: Significance threshold.
            correction: Whether to apply continuity correction.

        Returns:
            A ``PairedTestResult``.
        """
        y_true = np.asarray(y_true)
        y_pred_a = np.asarray(y_pred_a)
        y_pred_b = np.asarray(y_pred_b)

        correct_a = y_pred_a == y_true
        correct_b = y_pred_b == y_true

        # Discordant pairs
        b = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
        c = int(np.sum(~correct_a & correct_b))  # A wrong, B correct

        if b + c == 0:
            logger.info("McNemar's test: no discordant pairs, classifiers agree.")
            return PairedTestResult(
                statistic=0.0,
                p_value=1.0,
                method="mcnemar",
                significant=False,
            )

        if correction:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        else:
            chi2 = (b - c) ** 2 / (b + c)

        p_value = float(stats.chi2.sf(chi2, df=1))

        logger.debug(
            "McNemar's test: b=%d, c=%d, chi2=%.4f, p=%.4f",
            b,
            c,
            chi2,
            p_value,
        )

        return PairedTestResult(
            statistic=float(chi2),
            p_value=p_value,
            method="mcnemar",
            significant=p_value < alpha,
        )


class WilcoxonTest:
    """Wilcoxon signed-rank test for paired samples."""

    @staticmethod
    def test(
        scores_a: list[float] | np.ndarray,
        scores_b: list[float] | np.ndarray,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> PairedTestResult:
        """Run the Wilcoxon signed-rank test on paired scores.

        Args:
            scores_a: Per-sample scores from method A.
            scores_b: Per-sample scores from method B.
            alpha: Significance threshold.
            alternative: One of 'two-sided', 'greater', or 'less'.

        Returns:
            A ``PairedTestResult`` with the Wilcoxon statistic and p-value.
        """
        scores_a = np.asarray(scores_a, dtype=float)
        scores_b = np.asarray(scores_b, dtype=float)

        diffs = scores_a - scores_b
        if np.all(diffs == 0):
            logger.info("Wilcoxon test: all differences are zero.")
            return PairedTestResult(
                statistic=0.0,
                p_value=1.0,
                method="wilcoxon",
                significant=False,
            )

        stat, p_value = stats.wilcoxon(
            scores_a, scores_b, alternative=alternative, zero_method="wilcox"
        )

        logger.debug(
            "Wilcoxon test: statistic=%.4f, p=%.4f, significant=%s",
            stat,
            p_value,
            p_value < alpha,
        )

        return PairedTestResult(
            statistic=float(stat),
            p_value=float(p_value),
            method="wilcoxon",
            significant=p_value < alpha,
        )


class EffectSize:
    """Effect size calculations."""

    @staticmethod
    def cohens_d(
        scores_a: list[float] | np.ndarray,
        scores_b: list[float] | np.ndarray,
    ) -> float:
        """Compute Cohen's d for two sets of paired scores.

        Uses the pooled standard deviation as the denominator.

        Args:
            scores_a: Scores from method A.
            scores_b: Scores from method B.

        Returns:
            Cohen's d effect size (positive means A > B).
        """
        scores_a = np.asarray(scores_a, dtype=float)
        scores_b = np.asarray(scores_b, dtype=float)

        n_a = len(scores_a)
        n_b = len(scores_b)

        if n_a < 2 or n_b < 2:
            raise ValueError("Each group must have at least 2 observations.")

        mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
        var_a = float(np.var(scores_a, ddof=1))
        var_b = float(np.var(scores_b, ddof=1))

        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

        if pooled_std == 0.0:
            return 0.0

        d = mean_diff / pooled_std

        logger.debug("Cohen's d = %.4f", d)
        return float(d)

    @staticmethod
    def interpret(d: float) -> str:
        """Interpret a Cohen's d value using conventional thresholds.

        Args:
            d: Cohen's d effect size (absolute value is used).

        Returns:
            One of 'negligible', 'small', 'medium', or 'large'.
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
