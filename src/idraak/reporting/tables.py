"""Generate publication-quality tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from idraak.utils.logging import get_logger

logger = get_logger("tables")


class TableGenerator:
    """Generate result tables in CSV, Markdown, and LaTeX."""

    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def metrics_table(
        self,
        results: dict[str, dict[str, float]],
        name: str = "main_results",
    ) -> pd.DataFrame:
        """Create a comparison table of methods and their metrics."""
        df = pd.DataFrame(results).T
        df.index.name = "Method"

        self._save(df, name)
        return df

    def per_language_table(
        self,
        results: dict[str, dict[str, float]],
        name: str = "per_language",
    ) -> pd.DataFrame:
        """Create per-language results table."""
        df = pd.DataFrame(results).T
        df.index.name = "Language"
        self._save(df, name)
        return df

    def dataset_stats_table(
        self,
        entries: list[dict[str, Any]],
        name: str = "dataset_stats",
    ) -> pd.DataFrame:
        """Create dataset statistics table."""
        df = pd.DataFrame(entries)
        stats = {
            "Total requirements": len(df),
        }
        if "domain" in df.columns:
            stats["Domains"] = df["domain"].nunique()
            domain_counts = df["domain"].value_counts().to_dict()
            stats.update({f"domain_{k}": v for k, v in domain_counts.items()})
        if "category" in df.columns:
            stats["Categories"] = df["category"].nunique()
        if "difficulty" in df.columns:
            diff_counts = df["difficulty"].value_counts().to_dict()
            stats.update({f"difficulty_{k}": v for k, v in diff_counts.items()})

        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ["Count"]
        self._save(stats_df, name)
        return stats_df

    def _save(self, df: pd.DataFrame, name: str) -> None:
        """Save table in multiple formats."""
        csv_path = self.output_dir / f"{name}.csv"
        df.to_csv(csv_path)
        logger.info(f"Saved CSV: {csv_path}")

        md_path = self.output_dir / f"{name}.md"
        md_path.write_text(df.to_markdown() + "\n")
        logger.info(f"Saved Markdown: {md_path}")

        try:
            latex_path = self.output_dir / f"{name}.tex"
            latex_path.write_text(df.to_latex())
            logger.info(f"Saved LaTeX: {latex_path}")
        except ImportError:
            logger.warning("LaTeX export requires jinja2>=3.1.5; skipping")
