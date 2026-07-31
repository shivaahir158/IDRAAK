"""Workflow B: Structured single-agent — compare SRRs with one agent."""

from __future__ import annotations

import time
from typing import Any

from idraak.drift.comparator import SRRComparator
from idraak.extraction.hybrid import HybridExtractor
from idraak.providers.base import LLMProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity, DriftType
from idraak.schemas.srr import SemanticRequirement
from idraak.utils.logging import get_logger

logger = get_logger("workflow.structured_single")


class StructuredSingleWorkflow:
    """Extracts SRRs from both texts and compares deterministically."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._extractor = HybridExtractor(llm_provider=llm_provider)
        self._comparator = SRRComparator()

    def run(
        self,
        original_text: str,
        candidate_text: str,
        source_language: str = "en",
        target_language: str = "",
        requirement_id: str = "",
    ) -> DriftResult:
        start = time.time()

        # Extract SRRs
        orig_srr = self._extractor.extract(original_text, source_language, requirement_id)
        cand_srr = self._extractor.extract(candidate_text, target_language, f"{requirement_id}-cand")

        # Compare
        diffs = self._comparator.compare(orig_srr, cand_srr)

        drift_detected = len(diffs) > 0
        drift_types = list({d.drift_type for d in diffs if d.drift_type})
        max_severity = DriftSeverity.NONE
        severity_order = list(DriftSeverity)
        for d in diffs:
            if severity_order.index(d.severity) > severity_order.index(max_severity):
                max_severity = d.severity

        confidence = max((d.confidence for d in diffs), default=0.5) if diffs else 0.5

        return DriftResult(
            requirement_id=requirement_id,
            source_language=source_language,
            target_language=target_language,
            drift_detected=drift_detected,
            drift_label=DriftLabel.DRIFT if drift_detected else DriftLabel.NO_DRIFT,
            drift_types=drift_types,
            severity=max_severity,
            confidence=confidence,
            field_differences=diffs,
            explanation=f"Found {len(diffs)} field differences via structured comparison",
            workflow="structured_single",
            latency_seconds=time.time() - start,
        )
