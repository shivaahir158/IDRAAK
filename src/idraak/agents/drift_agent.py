"""Drift Detection Agent — detects field-level differences and classifies drift."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent
from idraak.drift.comparator import SRRComparator
from idraak.schemas.srr import SemanticRequirement


class DriftDetectionAgent(BaseAgent):
    """Detects drift between two SRRs using deterministic comparison."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="drift_detection", **kwargs)
        self._comparator = SRRComparator()

    def run(
        self,
        original_srr: dict[str, Any] | None = None,
        candidate_srr: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        original_srr = original_srr or {}
        candidate_srr = candidate_srr or {}

        try:
            orig = SemanticRequirement.model_validate(original_srr)
            cand = SemanticRequirement.model_validate(candidate_srr)
            diffs = self._comparator.compare(orig, cand)

            drift_detected = len(diffs) > 0
            drift_types = list({d.drift_type.value for d in diffs if d.drift_type})
            max_severity = "none"
            severity_order = ["none", "low", "medium", "high", "critical"]
            for d in diffs:
                if severity_order.index(d.severity.value) > severity_order.index(max_severity):
                    max_severity = d.severity.value

            latency = time.time() - start
            return self._make_result(
                output={
                    "drift_detected": drift_detected,
                    "drift_types": drift_types,
                    "severity": max_severity,
                    "field_differences": [d.model_dump() for d in diffs],
                    "n_differences": len(diffs),
                    "confidence": max((d.confidence for d in diffs), default=0.5) if diffs else 0.5,
                },
                latency=latency,
            )
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return AgentResult(
                output={"drift_detected": False, "error": str(e)},
                agent_name=self.name,
                success=False,
                error=str(e),
            )
