"""Calibration Agent — estimates and calibrates final confidence."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent


class CalibrationAgent(BaseAgent):
    """Aggregates confidence from multiple sources and applies calibration."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="calibration", **kwargs)

    def run(
        self,
        extraction_confidence: float = 0.5,
        drift_confidence: float = 0.5,
        critic_confidence: float = 0.5,
        n_differences: int = 0,
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()

        # Weighted aggregation
        weights = {"extraction": 0.2, "drift": 0.5, "critic": 0.3}
        raw_confidence = (
            weights["extraction"] * extraction_confidence
            + weights["drift"] * drift_confidence
            + weights["critic"] * critic_confidence
        )

        # Adjust based on number of differences
        if n_differences == 0:
            raw_confidence = min(raw_confidence, 0.3)
        elif n_differences >= 3:
            raw_confidence = max(raw_confidence, 0.8)

        raw_confidence = max(0.0, min(1.0, raw_confidence))

        latency = time.time() - start
        return self._make_result(
            output={
                "raw_confidence": raw_confidence,
                "calibrated_confidence": raw_confidence,  # Post-hoc calibration applied later
                "components": {
                    "extraction": extraction_confidence,
                    "drift": drift_confidence,
                    "critic": critic_confidence,
                },
            },
            latency=latency,
        )
