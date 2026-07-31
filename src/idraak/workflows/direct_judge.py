"""Workflow A: Direct LLM judge — single model evaluates original vs translated."""

from __future__ import annotations

import json
import time
from typing import Any

from idraak.providers.base import LLMProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity
from idraak.utils.logging import get_logger

logger = get_logger("workflow.direct_judge")

_JUDGE_PROMPT = """Compare these two technical requirements and determine if the meaning has changed.

Original (English): {original_text}
Translated ({target_language}): {candidate_text}

Analyze whether any semantic drift occurred. Consider:
- Numerical values, units, operators
- Modality (shall/must/should/may)
- Polarity (positive/negative)
- Conditions and exceptions
- Temporal relations
- Scope and entities

Respond with JSON:
{{
  "drift_detected": true/false,
  "drift_types": ["list of drift types if any"],
  "severity": "none/low/medium/high/critical",
  "confidence": 0.0-1.0,
  "explanation": "brief explanation"
}}"""


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

        prompt = _JUDGE_PROMPT.format(
            original_text=original_text,
            candidate_text=candidate_text,
            target_language=target_language,
        )

        try:
            response = self._llm.complete(
                prompt=prompt,
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
