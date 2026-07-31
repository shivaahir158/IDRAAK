"""Workflow D: Ensemble — combines deterministic SRR comparison with LLM judge."""

from __future__ import annotations

import json
import time
from typing import Any

from idraak.drift.comparator import SRRComparator
from idraak.extraction.deterministic import DeterministicExtractor
from idraak.providers.base import LLMProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity, DriftType
from idraak.utils.logging import get_logger

logger = get_logger("workflow.ensemble")

_ENSEMBLE_SYSTEM = """You are an expert semantic equivalence judge. You receive two texts plus structured analysis from a deterministic comparison system.

IMPORTANT calibration rules:
- Paraphrases and stylistic rewording are NOT drift. "shall" and "must" both mean mandatory. Sentence restructuring that preserves meaning is NOT drift.
- Synonym substitution preserving meaning is NOT drift.
- Changes to numerical values, units, operators, thresholds ARE drift.
- Changes to modality (shall→should, must→may) ARE drift.
- Polarity inversions (enable→disable, shall→shall not) ARE drift.
- Entity/role swaps (A tributary of B → B tributary of A) ARE drift.
- Adding/removing conditions, exceptions, or critical information ARE drift.

When the structured analysis finds 0 differences but you see a real semantic change, flag it as drift.
When the structured analysis finds differences but they are just rewording/paraphrases, say NO drift.
Your judgment overrides the structured analysis."""

_ENSEMBLE_PROMPT = """Original: {original_text}
Candidate: {candidate_text}

Structured analysis found {n_diffs} difference(s):
{diff_summary}

Determine if semantic drift occurred. Respond with JSON only:
{{"drift_detected": true/false, "drift_types": [...], "severity": "none/low/medium/high/critical", "confidence": 0.0-1.0, "explanation": "..."}}"""


class EnsembleWorkflow:
    """Combines deterministic SRR comparison with LLM judge.

    The deterministic system provides structured evidence (field differences),
    and the LLM makes the final call informed by both the evidence and its
    own understanding of the texts.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._extractor = DeterministicExtractor()
        self._comparator = SRRComparator()
        self._llm = llm_provider

    def run(
        self,
        original_text: str,
        candidate_text: str,
        source_language: str = "en",
        target_language: str = "",
        requirement_id: str = "",
    ) -> DriftResult:
        start = time.time()

        # Step 1: Deterministic extraction and comparison
        orig_srr = self._extractor.extract(original_text, source_language, requirement_id)
        cand_srr = self._extractor.extract(candidate_text, target_language, f"{requirement_id}-cand")
        diffs = self._comparator.compare(orig_srr, cand_srr)

        # Build diff summary for the LLM
        if diffs:
            diff_lines = []
            for d in diffs[:8]:
                field = d.field if hasattr(d, 'field') else str(d)
                original = d.original_value if hasattr(d, 'original_value') else ''
                changed = d.changed_value if hasattr(d, 'changed_value') else ''
                severity = d.severity.value if hasattr(d, 'severity') else 'unknown'
                diff_lines.append(f"- {field}: '{original}' → '{changed}' (severity: {severity})")
            diff_summary = "\n".join(diff_lines)
        else:
            diff_summary = "No structured differences found."

        # Step 2: If no LLM, fall back to deterministic result
        if self._llm is None:
            drift_detected = len(diffs) > 0
            severity_order = list(DriftSeverity)
            max_severity = DriftSeverity.NONE
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
                severity=max_severity,
                confidence=confidence,
                field_differences=diffs,
                explanation=f"Deterministic: {len(diffs)} differences found",
                workflow="ensemble",
                latency_seconds=time.time() - start,
            )

        # Step 3: LLM makes final judgment informed by structured evidence
        prompt = _ENSEMBLE_PROMPT.format(
            original_text=original_text,
            candidate_text=candidate_text,
            n_diffs=len(diffs),
            diff_summary=diff_summary,
        )

        try:
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_ENSEMBLE_SYSTEM,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response["content"])
            latency = time.time() - start

            return DriftResult(
                requirement_id=requirement_id,
                source_language=source_language,
                target_language=target_language,
                drift_detected=data.get("drift_detected", False),
                drift_label=DriftLabel.DRIFT if data.get("drift_detected") else DriftLabel.NO_DRIFT,
                severity=DriftSeverity(data.get("severity", "none")),
                confidence=data.get("confidence", 0.5),
                field_differences=diffs,
                explanation=data.get("explanation", ""),
                workflow="ensemble",
                model_metadata={"model": response.get("model", ""), "usage": response.get("usage", {})},
                latency_seconds=latency,
            )
        except Exception as e:
            logger.error(f"Ensemble LLM judge failed: {e}")
            # Fall back to deterministic
            drift_detected = len(diffs) > 0
            return DriftResult(
                requirement_id=requirement_id,
                drift_detected=drift_detected,
                confidence=0.5 if not diffs else max(d.confidence for d in diffs),
                field_differences=diffs,
                explanation=f"LLM failed ({e}), deterministic: {len(diffs)} diffs",
                workflow="ensemble",
                latency_seconds=time.time() - start,
            )
