"""Judge Agent — synthesizes all agent outputs into a final decision."""

from __future__ import annotations

import json
import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent

_JUDGE_PROMPT = """You are the final judge in a semantic drift detection system.

Review the following evidence and make a final decision.

Original requirement: {original_text}
Candidate requirement: {candidate_text}

Field differences found: {n_differences}
Drift types: {drift_types}
Maximum severity: {severity}

Critic assessment: {critic_assessment}
Confidence: {confidence}

Based on all evidence, provide your final judgment as JSON:
{{
  "drift_detected": true/false,
  "drift_types": ["list of confirmed drift types"],
  "severity": "none/low/medium/high/critical",
  "confidence": 0.0-1.0,
  "explanation": "brief explanation of your decision",
  "requires_human_review": true/false,
  "review_reasons": ["reasons if human review needed"]
}}"""

# Confidence threshold below which human review is recommended
HUMAN_REVIEW_THRESHOLD = 0.6


class JudgeAgent(BaseAgent):
    """Final judge that synthesizes all agent outputs."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="judge", **kwargs)

    def run(
        self,
        original_text: str = "",
        candidate_text: str = "",
        drift_output: dict[str, Any] | None = None,
        critic_output: dict[str, Any] | None = None,
        calibration_output: dict[str, Any] | None = None,
        evidence_output: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        drift_output = drift_output or {}
        critic_output = critic_output or {}
        calibration_output = calibration_output or {}

        if self.llm:
            return self._llm_judge(
                original_text, candidate_text,
                drift_output, critic_output, calibration_output, start,
            )

        return self._deterministic_judge(
            drift_output, critic_output, calibration_output, start,
        )

    def _llm_judge(
        self,
        original_text: str,
        candidate_text: str,
        drift_out: dict,
        critic_out: dict,
        cal_out: dict,
        start: float,
    ) -> AgentResult:
        prompt = _JUDGE_PROMPT.format(
            original_text=original_text,
            candidate_text=candidate_text,
            n_differences=drift_out.get("n_differences", 0),
            drift_types=drift_out.get("drift_types", []),
            severity=drift_out.get("severity", "none"),
            critic_assessment=critic_out.get("assessment", "N/A"),
            confidence=cal_out.get("calibrated_confidence", 0.5),
        )
        try:
            result = self._timed_llm_call(prompt, response_format={"type": "json_object"})
            data = json.loads(result["content"])
            latency = time.time() - start
            return self._make_result(output=data, latency=latency, usage=result.get("usage", {}))
        except Exception as e:
            self.logger.warning(f"LLM judge failed: {e}")
            return self._deterministic_judge(drift_out, critic_out, cal_out, start)

    def _deterministic_judge(
        self, drift_out: dict, critic_out: dict, cal_out: dict, start: float
    ) -> AgentResult:
        confidence = cal_out.get("calibrated_confidence", cal_out.get("raw_confidence", 0.5))
        drift_detected = drift_out.get("drift_detected", False)
        drift_types = drift_out.get("drift_types", [])
        severity = drift_out.get("severity", "none")

        # Apply critic feedback
        if not critic_out.get("agrees_with_decision", True):
            confidence *= 0.7
            if confidence < 0.4:
                drift_detected = False

        review_reasons = []
        requires_review = False
        if confidence < HUMAN_REVIEW_THRESHOLD:
            requires_review = True
            review_reasons.append("Low final confidence")
        if not critic_out.get("agrees_with_decision", True):
            requires_review = True
            review_reasons.append("Critic disagrees with drift detection")

        n_diffs = drift_out.get("n_differences", 0)
        explanation = (
            f"{'Drift' if drift_detected else 'No drift'} detected with "
            f"{n_diffs} field difference(s). "
            f"Severity: {severity}. Confidence: {confidence:.2f}."
        )

        latency = time.time() - start
        return self._make_result(
            output={
                "drift_detected": drift_detected,
                "drift_types": drift_types,
                "severity": severity,
                "confidence": confidence,
                "explanation": explanation,
                "requires_human_review": requires_review,
                "review_reasons": review_reasons,
            },
            latency=latency,
        )
