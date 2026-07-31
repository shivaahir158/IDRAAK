"""Workflow A: Direct LLM judge — single model evaluates original vs translated."""

from __future__ import annotations

import json
import time
from typing import Any

from idraak.providers.base import LLMProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity
from idraak.utils.logging import get_logger

logger = get_logger("workflow.direct_judge")

_JUDGE_SYSTEM = """You are an expert semantic equivalence judge. Your job is to determine whether two texts preserve the same meaning, or whether semantic drift has occurred.

Semantic drift means the core meaning has changed — not just surface-level rewording. Be precise:
- Paraphrases and stylistic rewording are NOT drift
- Synonym substitution that preserves meaning is NOT drift
- Changes to numerical values, units, operators, thresholds ARE drift
- Changes to modality (shall→should, must→may) ARE drift
- Polarity inversion (shall→shall not) ARE drift
- Adding/removing conditions or exceptions ARE drift
- Swapping entities or roles (A depends on B → B depends on A) ARE drift
- Omitting or adding critical information ARE drift"""

_JUDGE_FEW_SHOT = """Here are examples to calibrate your judgment:

Example 1 — NO DRIFT (paraphrase):
Original: "The system shall respond within 100ms when valid input is asserted."
Candidate: "When valid input is asserted, the system must respond in under 100 milliseconds."
Answer: {{"drift_detected": false, "drift_types": [], "severity": "none", "confidence": 0.95, "explanation": "Semantically equivalent: 'shall' and 'must' both express mandatory modality, '100ms' equals '100 milliseconds', sentence restructuring preserves meaning."}}

Example 2 — DRIFT (numerical change):
Original: "The system shall respond within 100ms when valid input is asserted."
Candidate: "The system shall respond within 200ms when valid input is asserted."
Answer: {{"drift_detected": true, "drift_types": ["numerical_drift"], "severity": "high", "confidence": 0.99, "explanation": "Response time threshold changed from 100ms to 200ms — a 2x relaxation of the timing constraint."}}

Example 3 — NO DRIFT (cross-lingual equivalent):
Original: "The device shall not exceed 5W power consumption during idle mode."
Candidate: "Das Gerät darf im Leerlaufmodus einen Stromverbrauch von 5W nicht überschreiten."
Answer: {{"drift_detected": false, "drift_types": [], "severity": "none", "confidence": 0.92, "explanation": "German translation preserves all technical details: 5W limit, idle mode condition, prohibition modality."}}

Example 4 — DRIFT (polarity inversion):
Original: "The module shall disable output when fault is detected."
Candidate: "The module shall enable output when fault is detected."
Answer: {{"drift_detected": true, "drift_types": ["polarity_drift"], "severity": "critical", "confidence": 0.99, "explanation": "Action inverted from 'disable' to 'enable' — opposite behavior on fault condition, safety-critical change."}}

Example 5 — DRIFT (entity swap):
Original: "The Tabaci River is a tributary of the River Leurda in Romania."
Candidate: "The Leurda River is a tributary of the River Tabaci in Romania."
Answer: {{"drift_detected": true, "drift_types": ["entity_drift"], "severity": "high", "confidence": 0.98, "explanation": "River roles swapped — Tabaci is tributary of Leurda in original, but Leurda is tributary of Tabaci in candidate. Factual meaning reversed."}}

Example 6 — NO DRIFT (stylistic variation):
Original: "He moved to New York in 2010 and stayed there until 2015."
Candidate: "From 2010 to 2015, he lived in New York."
Answer: {{"drift_detected": false, "drift_types": [], "severity": "none", "confidence": 0.90, "explanation": "Same factual content: person in New York during 2010-2015. 'moved to and stayed' vs 'lived' is a stylistic variation."}}"""

_JUDGE_PROMPT = """Now analyze this pair:

Original: {original_text}
Candidate: {candidate_text}

Respond with JSON only:
{{"drift_detected": true/false, "drift_types": [...], "severity": "none/low/medium/high/critical", "confidence": 0.0-1.0, "explanation": "..."}}"""


class DirectJudgeWorkflow:
    """Single LLM judges whether two texts have the same meaning."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self._llm = llm_provider

    def run(
        self,
        original_text: str,
        candidate_text: str,
        target_language: str = "",
        requirement_id: str = "",
    ) -> DriftResult:
        start = time.time()

        if self._llm is None:
            # Mock fallback
            return DriftResult(
                requirement_id=requirement_id,
                target_language=target_language,
                drift_detected=False,
                confidence=0.5,
                explanation="No LLM provider — mock result",
                workflow="direct_judge",
                latency_seconds=time.time() - start,
            )

        prompt = _JUDGE_FEW_SHOT + "\n\n" + _JUDGE_PROMPT.format(
            original_text=original_text,
            candidate_text=candidate_text,
        )

        try:
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_JUDGE_SYSTEM,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response["content"])
            latency = time.time() - start

            return DriftResult(
                requirement_id=requirement_id,
                target_language=target_language,
                drift_detected=data.get("drift_detected", False),
                drift_label=DriftLabel.DRIFT if data.get("drift_detected") else DriftLabel.NO_DRIFT,
                drift_types=[],
                severity=DriftSeverity(data.get("severity", "none")),
                confidence=data.get("confidence", 0.5),
                explanation=data.get("explanation", ""),
                workflow="direct_judge",
                model_metadata={"model": response.get("model", ""), "usage": response.get("usage", {})},
                latency_seconds=latency,
            )
        except Exception as e:
            logger.error(f"Direct judge failed: {e}")
            return DriftResult(
                requirement_id=requirement_id,
                drift_detected=False,
                confidence=0.0,
                explanation=f"Error: {e}",
                workflow="direct_judge",
                latency_seconds=time.time() - start,
            )
