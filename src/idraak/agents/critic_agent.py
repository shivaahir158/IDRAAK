"""Critic Agent — challenges drift decisions to reduce false positives."""

from __future__ import annotations

import json
import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent

_CRITIC_PROMPT = """You are a critic reviewing a semantic drift detection decision.

Original requirement: {original_text}
Translated/candidate requirement: {candidate_text}

The drift detection system found these differences:
{differences}

Current drift decision: {drift_decision}

Your job is to challenge this decision:
1. Could the differences be benign linguistic variation?
2. Are there false positives?
3. Is the wording different but semantically equivalent?
4. Could unit conversions explain numerical differences?

Respond with JSON:
{{
  "agrees_with_decision": true/false,
  "false_positive_fields": ["list of fields that may be false positives"],
  "assessment": "brief explanation",
  "revised_confidence": 0.0-1.0
}}"""


class CriticAgent(BaseAgent):
    """Challenges drift-detection decisions to identify false positives."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="critic", **kwargs)

    def run(
        self,
        original_text: str = "",
        candidate_text: str = "",
        field_differences: list[dict[str, Any]] | None = None,
        drift_decision: bool = False,
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        field_differences = field_differences or []

        if self.llm:
            return self._llm_critique(
                original_text, candidate_text, field_differences, drift_decision, start
            )

        # Deterministic critique: check for obvious false positives
        return self._deterministic_critique(field_differences, drift_decision, start)

    def _llm_critique(
        self,
        original_text: str,
        candidate_text: str,
        diffs: list[dict],
        drift_decision: bool,
        start: float,
    ) -> AgentResult:
        prompt = _CRITIC_PROMPT.format(
            original_text=original_text,
            candidate_text=candidate_text,
            differences=json.dumps(diffs[:5], indent=2, default=str),
            drift_decision="DRIFT DETECTED" if drift_decision else "NO DRIFT",
        )
        try:
            result = self._timed_llm_call(prompt, response_format={"type": "json_object"})
            data = json.loads(result["content"])
            latency = time.time() - start
            return self._make_result(
                output=data,
                latency=latency,
                usage=result.get("usage", {}),
            )
        except Exception as e:
            self.logger.warning(f"LLM critic failed: {e}")
            return self._deterministic_critique(diffs, drift_decision, start)

    def _deterministic_critique(
        self, diffs: list[dict], drift_decision: bool, start: float
    ) -> AgentResult:
        false_positives = []
        for d in diffs:
            severity = d.get("severity", "none")
            confidence = d.get("confidence", 0)
            if severity in ("none", "low") and confidence < 0.7:
                false_positives.append(d.get("field", ""))

        latency = time.time() - start
        real_diffs = len(diffs) - len(false_positives)
        return self._make_result(
            output={
                "agrees_with_decision": drift_decision and real_diffs > 0,
                "false_positive_fields": false_positives,
                "assessment": f"Found {len(false_positives)} potential false positives out of {len(diffs)} differences",
                "revised_confidence": max(0.3, 1.0 - len(false_positives) / max(len(diffs), 1)),
            },
            latency=latency,
        )
