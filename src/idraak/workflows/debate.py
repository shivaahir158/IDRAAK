"""Workflow D: Debate — pro-drift and no-drift agents argue, moderator decides."""

from __future__ import annotations

import time
from typing import Any

from idraak.drift.comparator import SRRComparator
from idraak.extraction.hybrid import HybridExtractor
from idraak.providers.base import LLMProvider
from idraak.schemas.drift import (
    DriftLabel,
    DriftResult,
    DriftSeverity,
    DriftType,
    FieldDifference,
)
from idraak.utils.logging import get_logger

logger = get_logger("workflow.debate")


class _DebateArgument:
    """One side's argument in the debate."""

    def __init__(self, position: str, reasoning: list[str], confidence: float):
        self.position = position
        self.reasoning = reasoning
        self.confidence = confidence


class DebateWorkflow:
    """Two-sided debate workflow for drift detection.

    A pro-drift advocate argues that drift exists, a no-drift advocate argues
    it doesn't, and a moderator synthesizes the final decision. This design
    surfaces arguments that a single-pass detector might miss.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        n_rounds: int = 2,
    ):
        self._llm = llm_provider
        self._n_rounds = max(1, n_rounds)
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

        # Step 1: Extract SRRs and find differences
        orig_srr = self._extractor.extract(original_text, source_language, requirement_id)
        cand_srr = self._extractor.extract(candidate_text, target_language, f"{requirement_id}-cand")
        field_diffs = self._comparator.compare(orig_srr, cand_srr)

        # Step 2: Run debate
        pro_drift = self._build_pro_drift_argument(original_text, candidate_text, field_diffs)
        no_drift = self._build_no_drift_argument(original_text, candidate_text, field_diffs)

        # Step 3: Moderator decides
        decision = self._moderate(pro_drift, no_drift, field_diffs)

        drift_detected = decision["drift_detected"]
        drift_types = list({d.drift_type for d in field_diffs if d.drift_type}) if drift_detected else []

        max_severity = DriftSeverity.NONE
        if drift_detected and field_diffs:
            severity_order = list(DriftSeverity)
            for d in field_diffs:
                if severity_order.index(d.severity) > severity_order.index(max_severity):
                    max_severity = d.severity

        return DriftResult(
            requirement_id=requirement_id,
            source_language=source_language,
            target_language=target_language,
            drift_detected=drift_detected,
            drift_label=DriftLabel.DRIFT if drift_detected else DriftLabel.NO_DRIFT,
            drift_types=drift_types,
            severity=max_severity,
            confidence=decision["confidence"],
            field_differences=field_diffs if drift_detected else [],
            evidence=decision.get("evidence", []),
            explanation=decision["explanation"],
            workflow="debate",
            model_metadata={
                "pro_drift_confidence": pro_drift.confidence,
                "no_drift_confidence": no_drift.confidence,
                "n_rounds": self._n_rounds,
            },
            latency_seconds=time.time() - start,
        )

    def _build_pro_drift_argument(
        self,
        original_text: str,
        candidate_text: str,
        diffs: list[FieldDifference],
    ) -> _DebateArgument:
        """Build the pro-drift advocate's argument."""
        reasoning = []
        confidence = 0.3  # Base confidence

        if diffs:
            confidence += min(0.4, len(diffs) * 0.1)
            for d in diffs:
                reasoning.append(
                    f"Field '{d.field}' changed: '{d.original}' -> '{d.candidate}' "
                    f"(type={d.drift_type.value if d.drift_type else 'unknown'}, "
                    f"severity={d.severity.value})"
                )

            # Check for critical/high severity
            critical = [d for d in diffs if d.severity in (DriftSeverity.CRITICAL, DriftSeverity.HIGH)]
            if critical:
                confidence += 0.2
                reasoning.append(
                    f"{len(critical)} critical/high severity difference(s) found"
                )

            # Check for safety-related drift
            safety_types = {DriftType.POLARITY, DriftType.MODALITY, DriftType.NUMERICAL}
            safety_drifts = [d for d in diffs if d.drift_type in safety_types]
            if safety_drifts:
                confidence += 0.1
                reasoning.append(
                    f"{len(safety_drifts)} safety-relevant drift(s) detected "
                    f"({', '.join(d.drift_type.value for d in safety_drifts if d.drift_type)})"
                )
        else:
            reasoning.append("No field-level differences detected by structured comparison")

        # Text-level check: if texts are very different, that's evidence of drift
        if original_text.strip().lower() != candidate_text.strip().lower():
            # Simple token overlap check
            orig_tokens = set(original_text.lower().split())
            cand_tokens = set(candidate_text.lower().split())
            if orig_tokens and cand_tokens:
                overlap = len(orig_tokens & cand_tokens) / len(orig_tokens | cand_tokens)
                if overlap < 0.5:
                    confidence += 0.1
                    reasoning.append(f"Low token overlap ({overlap:.2f}) suggests significant change")

        return _DebateArgument(
            position="drift",
            reasoning=reasoning,
            confidence=min(1.0, confidence),
        )

    def _build_no_drift_argument(
        self,
        original_text: str,
        candidate_text: str,
        diffs: list[FieldDifference],
    ) -> _DebateArgument:
        """Build the no-drift advocate's argument."""
        reasoning = []
        confidence = 0.5  # Base confidence for no drift

        if not diffs:
            confidence += 0.4
            reasoning.append("No structural differences found in SRR comparison")
        else:
            # Check how many diffs are low severity / low confidence
            low_severity = [d for d in diffs if d.severity in (DriftSeverity.NONE, DriftSeverity.LOW)]
            low_conf = [d for d in diffs if d.confidence < 0.7]

            if low_severity:
                frac = len(low_severity) / len(diffs)
                confidence += frac * 0.2
                reasoning.append(
                    f"{len(low_severity)}/{len(diffs)} differences are low severity "
                    f"(may be benign linguistic variation)"
                )

            if low_conf:
                frac = len(low_conf) / len(diffs)
                confidence += frac * 0.15
                reasoning.append(
                    f"{len(low_conf)}/{len(diffs)} differences have low confidence"
                )

            # Check for potential unit equivalences
            unit_diffs = [d for d in diffs if d.drift_type == DriftType.UNIT]
            if unit_diffs:
                reasoning.append(
                    f"{len(unit_diffs)} unit difference(s) may be equivalent conversions"
                )

            # Check for terminology that might be synonymous
            term_diffs = [d for d in diffs if d.drift_type == DriftType.TERMINOLOGY]
            if term_diffs:
                confidence += 0.05
                reasoning.append(
                    f"{len(term_diffs)} terminology difference(s) may be domain synonyms"
                )

        # If texts are very similar, that supports no drift
        if original_text.strip().lower() == candidate_text.strip().lower():
            confidence += 0.3
            reasoning.append("Texts are identical (case-insensitive)")

        return _DebateArgument(
            position="no_drift",
            reasoning=reasoning,
            confidence=min(1.0, confidence),
        )

    def _moderate(
        self,
        pro_drift: _DebateArgument,
        no_drift: _DebateArgument,
        diffs: list[FieldDifference],
    ) -> dict[str, Any]:
        """Moderator weighs both sides and makes final decision."""
        # Weighted scoring: structural evidence gets extra weight
        pro_score = pro_drift.confidence
        no_score = no_drift.confidence

        # Structural evidence bonus: real field differences are strong signals
        real_diffs = [
            d for d in diffs
            if d.severity not in (DriftSeverity.NONE,) and d.confidence >= 0.5
        ]
        if real_diffs:
            pro_score += 0.15 * min(len(real_diffs), 3)

        # Critical diffs are very strong evidence
        critical_diffs = [d for d in diffs if d.severity in (DriftSeverity.CRITICAL, DriftSeverity.HIGH)]
        if critical_diffs:
            pro_score += 0.2

        drift_detected = pro_score > no_score

        # Build explanation
        explanation_parts = [
            f"Debate result: pro-drift score={pro_score:.2f}, no-drift score={no_score:.2f}.",
        ]
        if drift_detected:
            explanation_parts.append("Pro-drift arguments prevailed:")
            explanation_parts.extend(f"  - {r}" for r in pro_drift.reasoning[:3])
        else:
            explanation_parts.append("No-drift arguments prevailed:")
            explanation_parts.extend(f"  - {r}" for r in no_drift.reasoning[:3])

        # Confidence reflects how decisive the debate was
        score_gap = abs(pro_score - no_score)
        confidence = 0.5 + min(0.5, score_gap)

        return {
            "drift_detected": drift_detected,
            "confidence": confidence,
            "explanation": "\n".join(explanation_parts),
            "evidence": (pro_drift.reasoning if drift_detected else no_drift.reasoning),
        }
