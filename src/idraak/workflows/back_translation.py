"""Back-translation workflow: English -> Target -> English, then compare."""

from __future__ import annotations

import time
from typing import Any

from idraak.drift.comparator import SRRComparator
from idraak.extraction.hybrid import HybridExtractor
from idraak.providers.base import LLMProvider, TranslationProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity, DriftType
from idraak.utils.logging import get_logger

logger = get_logger("workflow.back_translation")


class BackTranslationWorkflow:
    """Back-translation workflow.

    Pipeline:
        English requirement
            -> Target-language translation
            -> English back-translation
        Compare:
            1. Original English SRR vs target-language SRR
            2. Original English SRR vs back-translated English SRR
            3. Target-language SRR vs back-translated SRR
    """

    def __init__(
        self,
        translation_provider: TranslationProvider,
        llm_provider: LLMProvider | None = None,
    ):
        self._translator = translation_provider
        self._extractor = HybridExtractor(llm_provider=llm_provider)
        self._comparator = SRRComparator()

    def run(
        self,
        original_text: str,
        target_language: str,
        source_language: str = "en",
        requirement_id: str = "",
    ) -> BackTranslationResult:
        """Run the back-translation workflow."""
        start = time.time()

        # Step 1: Translate to target language
        forward_result = self._translator.translate(
            original_text, source_language, target_language
        )
        translated_text = forward_result.translated_text

        # Step 2: Back-translate to source language
        back_result = self._translator.translate(
            translated_text, target_language, source_language
        )
        back_translated_text = back_result.translated_text

        # Step 3: Extract SRRs
        orig_srr = self._extractor.extract(original_text, source_language, requirement_id)
        trans_srr = self._extractor.extract(
            translated_text, target_language, f"{requirement_id}-trans"
        )
        back_srr = self._extractor.extract(
            back_translated_text, source_language, f"{requirement_id}-back"
        )

        # Step 4: Three-way comparison
        orig_vs_trans = self._comparator.compare(orig_srr, trans_srr)
        orig_vs_back = self._comparator.compare(orig_srr, back_srr)
        trans_vs_back = self._comparator.compare(trans_srr, back_srr)

        # Determine drift
        # Use orig_vs_back as primary (same language, most comparable)
        # Use orig_vs_trans as supporting evidence
        all_diffs = orig_vs_back
        drift_detected = len(all_diffs) > 0
        drift_types = list({d.drift_type for d in all_diffs if d.drift_type})

        severity_order = list(DriftSeverity)
        max_severity = DriftSeverity.NONE
        for d in all_diffs:
            if severity_order.index(d.severity) > severity_order.index(max_severity):
                max_severity = d.severity

        confidence = max((d.confidence for d in all_diffs), default=0.5) if all_diffs else 0.5

        latency = time.time() - start

        return BackTranslationResult(
            requirement_id=requirement_id,
            source_language=source_language,
            target_language=target_language,
            original_text=original_text,
            translated_text=translated_text,
            back_translated_text=back_translated_text,
            drift_result=DriftResult(
                requirement_id=requirement_id,
                source_language=source_language,
                target_language=target_language,
                drift_detected=drift_detected,
                drift_label=DriftLabel.DRIFT if drift_detected else DriftLabel.NO_DRIFT,
                drift_types=drift_types,
                severity=max_severity,
                confidence=confidence,
                field_differences=all_diffs,
                explanation=(
                    f"Back-translation comparison found {len(orig_vs_back)} differences "
                    f"(original vs back-translated). Forward translation had "
                    f"{len(orig_vs_trans)} differences."
                ),
                workflow="back_translation",
                latency_seconds=latency,
            ),
            n_forward_diffs=len(orig_vs_trans),
            n_back_diffs=len(orig_vs_back),
            n_trans_vs_back_diffs=len(trans_vs_back),
        )


class BackTranslationResult:
    """Result container for back-translation workflow."""

    def __init__(
        self,
        requirement_id: str,
        source_language: str,
        target_language: str,
        original_text: str,
        translated_text: str,
        back_translated_text: str,
        drift_result: DriftResult,
        n_forward_diffs: int,
        n_back_diffs: int,
        n_trans_vs_back_diffs: int,
    ):
        self.requirement_id = requirement_id
        self.source_language = source_language
        self.target_language = target_language
        self.original_text = original_text
        self.translated_text = translated_text
        self.back_translated_text = back_translated_text
        self.drift_result = drift_result
        self.n_forward_diffs = n_forward_diffs
        self.n_back_diffs = n_back_diffs
        self.n_trans_vs_back_diffs = n_trans_vs_back_diffs

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "back_translated_text": self.back_translated_text,
            "drift_detected": self.drift_result.drift_detected,
            "severity": self.drift_result.severity.value,
            "confidence": self.drift_result.confidence,
            "n_forward_diffs": self.n_forward_diffs,
            "n_back_diffs": self.n_back_diffs,
            "n_trans_vs_back_diffs": self.n_trans_vs_back_diffs,
        }
