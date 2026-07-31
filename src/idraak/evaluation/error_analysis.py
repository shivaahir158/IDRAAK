"""Automated error analysis report generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from idraak.evaluation.complexity import ComplexityScorer
from idraak.utils.logging import get_logger

logger = get_logger("error_analysis")


class ErrorAnalyzer:
    """Generate error analysis reports from evaluation results."""

    def __init__(self, output_dir: str | Path = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._scorer = ComplexityScorer()

    def analyze(
        self,
        results: list[dict[str, Any]],
        entries: dict[str, dict[str, Any]] | None = None,
    ) -> ErrorReport:
        """Analyze errors and generate a structured report."""
        errors = []
        for r in results:
            gold = r.get("gold_label", 0)
            pred = r.get("pred_label", 0)
            if gold != pred:
                error = self._build_error_record(r, entries)
                errors.append(error)

        report = ErrorReport(
            total_predictions=len(results),
            total_errors=len(errors),
            false_positives=[e for e in errors if e["error_type"] == "false_positive"],
            false_negatives=[e for e in errors if e["error_type"] == "false_negative"],
        )

        report.by_drift_type = self._group_errors(errors, "drift_type")
        report.by_severity = self._group_errors(errors, "severity")
        report.by_confidence_bin = self._bin_by_confidence(errors)

        return report

    def _build_error_record(
        self, result: dict, entries: dict | None
    ) -> dict[str, Any]:
        gold = result.get("gold_label", 0)
        pred = result.get("pred_label", 0)
        error_type = "false_positive" if pred == 1 and gold == 0 else "false_negative"

        original_text = result.get("original_text", "")
        if not original_text and entries:
            base_id = result.get("base_id", "")
            entry = entries.get(base_id, {})
            original_text = entry.get("original_text", "")

        complexity = self._scorer.score(original_text) if original_text else None

        return {
            "requirement_id": result.get("requirement_id", ""),
            "base_id": result.get("base_id", ""),
            "error_type": error_type,
            "gold_label": gold,
            "pred_label": pred,
            "confidence": result.get("confidence", 0),
            "drift_type": result.get("drift_type", "unknown"),
            "severity": result.get("severity", "unknown"),
            "n_diffs": result.get("n_diffs", 0),
            "original_text": original_text,
            "candidate_text": result.get("candidate_text", ""),
            "complexity": complexity.to_dict() if complexity else {},
            "explanation": result.get("explanation", ""),
        }

    def _group_errors(
        self, errors: list[dict], field: str
    ) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for e in errors:
            counter[str(e.get(field, "unknown"))] += 1
        return dict(counter.most_common())

    def _bin_by_confidence(self, errors: list[dict]) -> dict[str, int]:
        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for e in errors:
            c = e.get("confidence", 0)
            if c < 0.2:
                bins["0.0-0.2"] += 1
            elif c < 0.4:
                bins["0.2-0.4"] += 1
            elif c < 0.6:
                bins["0.4-0.6"] += 1
            elif c < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1
        return bins

    def save_report(self, report: "ErrorReport", name: str = "error_analysis") -> None:
        """Save error analysis as Markdown and JSONL."""
        # Markdown report
        md_lines = [
            f"# Error Analysis Report\n",
            f"**Total predictions:** {report.total_predictions}",
            f"**Total errors:** {report.total_errors}",
            f"**False positives:** {len(report.false_positives)}",
            f"**False negatives:** {len(report.false_negatives)}",
            f"**Error rate:** {report.total_errors / max(report.total_predictions, 1):.2%}\n",
            "## Errors by Drift Type\n",
            "| Drift Type | Count |",
            "|------------|-------|",
        ]
        for dt, count in report.by_drift_type.items():
            md_lines.append(f"| {dt} | {count} |")

        md_lines.extend([
            "\n## Errors by Confidence Bin\n",
            "| Confidence | Count |",
            "|------------|-------|",
        ])
        for bin_name, count in report.by_confidence_bin.items():
            md_lines.append(f"| {bin_name} | {count} |")

        md_lines.extend([
            "\n## False Positive Examples\n",
        ])
        for fp in report.false_positives[:10]:
            md_lines.append(f"- **{fp['requirement_id']}** (conf: {fp['confidence']:.2f})")
            if fp.get("original_text"):
                md_lines.append(f"  - Original: {fp['original_text'][:100]}...")

        md_lines.extend([
            "\n## False Negative Examples\n",
        ])
        for fn in report.false_negatives[:10]:
            md_lines.append(f"- **{fn['requirement_id']}** (conf: {fn['confidence']:.2f}, type: {fn['drift_type']})")
            if fn.get("original_text"):
                md_lines.append(f"  - Original: {fn['original_text'][:100]}...")

        md_path = self.output_dir / f"{name}.md"
        md_path.write_text("\n".join(md_lines) + "\n")
        logger.info(f"Saved error analysis: {md_path}")

        # JSONL of all errors
        jsonl_path = self.output_dir / f"{name}_errors.jsonl"
        with open(jsonl_path, "w") as f:
            for e in report.false_positives + report.false_negatives:
                f.write(json.dumps(e, default=str) + "\n")
        logger.info(f"Saved error details: {jsonl_path}")


class ErrorReport:
    """Container for error analysis results."""

    def __init__(
        self,
        total_predictions: int = 0,
        total_errors: int = 0,
        false_positives: list[dict] | None = None,
        false_negatives: list[dict] | None = None,
    ):
        self.total_predictions = total_predictions
        self.total_errors = total_errors
        self.false_positives = false_positives or []
        self.false_negatives = false_negatives or []
        self.by_drift_type: dict[str, int] = {}
        self.by_severity: dict[str, int] = {}
        self.by_confidence_bin: dict[str, int] = {}
