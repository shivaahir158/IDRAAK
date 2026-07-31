"""Generate publication-quality plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from idraak.utils.logging import get_logger

logger = get_logger("plots")

# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
})


class PlotGenerator:
    """Generate publication-quality plots."""

    def __init__(self, output_dir: str | Path = "reports", formats: list[str] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats or ["png", "pdf"]

    def _save(self, fig: plt.Figure, name: str) -> None:
        for fmt in self.formats:
            path = self.output_dir / f"{name}.{fmt}"
            fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"Saved plot: {name}")

    def f1_by_language(self, results: dict[str, float], name: str = "f1_by_language") -> None:
        """Bar chart of F1 by language."""
        fig, ax = plt.subplots()
        langs = list(results.keys())
        f1s = list(results.values())
        bars = ax.bar(langs, f1s, color="steelblue", edgecolor="white")
        ax.set_xlabel("Language")
        ax.set_ylabel("F1 Score")
        ax.set_title("Drift Detection F1 by Language")
        ax.set_ylim(0, 1.05)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
        plt.xticks(rotation=45, ha="right")
        self._save(fig, name)

    def f1_by_drift_type(self, results: dict[str, float], name: str = "f1_by_drift_type") -> None:
        """Horizontal bar chart of F1 by drift type."""
        fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.5)))
        types = list(results.keys())
        f1s = list(results.values())
        bars = ax.barh(types, f1s, color="coral", edgecolor="white")
        ax.set_xlabel("F1 Score")
        ax.set_title("Drift Detection F1 by Drift Type")
        ax.set_xlim(0, 1.05)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)
        self._save(fig, name)

    def reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 15,
        name: str = "reliability_diagram",
    ) -> None:
        """Calibration reliability diagram."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accs = np.zeros(n_bins)
        bin_confs = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = mask | (y_prob == bin_boundaries[i + 1])
            n_bin = mask.sum()
            bin_counts[i] = n_bin
            if n_bin > 0:
                bin_accs[i] = y_true[mask].mean()
                bin_confs[i] = y_prob[mask].mean()

        # Main plot
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax1.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6,
                color="steelblue", edgecolor="white", label="Model")
        ax1.set_xlabel("Mean Predicted Confidence")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram")
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Histogram
        ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins,
                color="steelblue", edgecolor="white", alpha=0.6)
        ax2.set_xlabel("Mean Predicted Confidence")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        self._save(fig, name)

    def confusion_matrix_plot(
        self,
        cm: list[list[int]] | np.ndarray,
        labels: list[str] | None = None,
        name: str = "confusion_matrix",
    ) -> None:
        """Confusion matrix heatmap."""
        cm_arr = np.array(cm)
        labels = labels or ["No Drift", "Drift"]
        fig, ax = plt.subplots()
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                        color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")

        plt.colorbar(im)
        self._save(fig, name)

    def method_comparison(
        self,
        results: dict[str, dict[str, float]],
        metric: str = "f1",
        name: str = "method_comparison",
    ) -> None:
        """Grouped bar chart comparing methods across metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(results.keys())
        metrics = [m for m in ["accuracy", "precision", "recall", "f1", "mcc"] if m in next(iter(results.values()), {})]

        x = np.arange(len(methods))
        width = 0.15
        for i, m in enumerate(metrics):
            vals = [results[method].get(m, 0) for method in methods]
            ax.bar(x + i * width, vals, width, label=m.upper())

        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.set_title("Method Comparison")
        ax.set_xticks(x + width * len(metrics) / 2)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        self._save(fig, name)

    def confidence_distribution(
        self,
        correct_probs: np.ndarray,
        incorrect_probs: np.ndarray,
        name: str = "confidence_distribution",
    ) -> None:
        """Confidence distribution for correct and incorrect predictions."""
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 21)
        ax.hist(correct_probs, bins=bins, alpha=0.6, label="Correct", color="green", density=True)
        ax.hist(incorrect_probs, bins=bins, alpha=0.6, label="Incorrect", color="red", density=True)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.set_title("Confidence Distribution")
        ax.legend()
        self._save(fig, name)
