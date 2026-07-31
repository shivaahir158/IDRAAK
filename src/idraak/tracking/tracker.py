"""Lightweight experiment tracking with JSONL logs."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from idraak.utils.logging import get_logger

logger = get_logger("tracker")


class ExperimentTracker:
    """Track experiment runs with metadata and metrics."""

    def __init__(self, log_dir: str | Path = "artifacts/runs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self._log_file = self.log_dir / f"run_{self.run_id}.jsonl"
        self._metadata: dict[str, Any] = {}

    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration."""
        self._metadata["config"] = config
        self._metadata["run_id"] = self.run_id
        self._metadata["timestamp"] = self.start_time.isoformat()

        # Try to get git commit
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            self._metadata["git_commit"] = result.stdout.strip()
        except Exception:
            self._metadata["git_commit"] = "unknown"

    def log_metrics(self, metrics: dict[str, Any], step: str = "final") -> None:
        """Log metrics to JSONL file."""
        entry = {
            "run_id": self.run_id,
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info(f"Logged metrics for step '{step}'")

    def log_result(self, result: dict[str, Any]) -> None:
        """Log a single result entry."""
        with open(self._log_file, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

    def save_summary(self, metrics: dict[str, Any]) -> None:
        """Save final summary."""
        summary = {
            **self._metadata,
            "metrics": metrics,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
        summary_path = self.log_dir / f"summary_{self.run_id}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary: {summary_path}")
