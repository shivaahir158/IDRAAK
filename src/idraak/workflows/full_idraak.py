"""Workflow C: Full IDRAAK multi-agent pipeline."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.alignment_agent import AlignmentAgent
from idraak.agents.calibration_agent import CalibrationAgent
from idraak.agents.critic_agent import CriticAgent
from idraak.agents.drift_agent import DriftDetectionAgent
from idraak.agents.evidence_agent import EvidenceAgent
from idraak.agents.extraction_agent import ExtractionAgent
from idraak.agents.judge_agent import JudgeAgent
from idraak.agents.translation_agent import TranslationAgent
from idraak.providers.base import LLMProvider, TranslationProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity, DriftType
from idraak.utils.logging import get_logger

logger = get_logger("workflow.full_idraak")


class FullIDRAAKWorkflow:
    """Complete multi-agent IDRAAK workflow.

    Pipeline:
        Translation -> Extraction -> Alignment -> Drift Detection ->
        Evidence -> Critic -> Calibration -> Judge
    """

    def __init__(
        self,
        translation_provider: TranslationProvider | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        self._translation_provider = translation_provider
        self._llm = llm_provider

        # Initialize agents
        self._extraction = ExtractionAgent(llm_provider=llm_provider)
        self._alignment = AlignmentAgent()
        self._drift = DriftDetectionAgent()
        self._evidence = EvidenceAgent()
        self._critic = CriticAgent(llm_provider=llm_provider)
        self._calibration = CalibrationAgent()
        self._judge = JudgeAgent(llm_provider=llm_provider)

        if translation_provider:
            self._translation = TranslationAgent(
                translation_provider=translation_provider,
                llm_provider=llm_provider,
            )
        else:
            self._translation = None

    def run(
        self,
        original_text: str,
        candidate_text: str | None = None,
        source_language: str = "en",
        target_language: str = "",
        requirement_id: str = "",
    ) -> DriftResult:
        """Run the full multi-agent pipeline."""
        start = time.time()
        agent_outputs: dict[str, Any] = {}

        # Step 1: Translation (if no candidate text provided)
        if candidate_text is None and self._translation and target_language:
            trans_result = self._translation.run(
                text=original_text,
                source_language=source_language,
                target_language=target_language,
            )
            candidate_text = trans_result.output.get("translated_text", "")
            agent_outputs["translation"] = trans_result.output
        elif candidate_text is None:
            candidate_text = original_text

        # Step 2: Extraction
        orig_result = self._extraction.run(
            text=original_text, language=source_language, requirement_id=requirement_id
        )
        cand_result = self._extraction.run(
            text=candidate_text, language=target_language or source_language,
            requirement_id=f"{requirement_id}-cand",
        )
        agent_outputs["extraction_original"] = orig_result.output
        agent_outputs["extraction_candidate"] = cand_result.output

        # Step 3: Alignment
        align_result = self._alignment.run(
            original_srr=orig_result.output,
            candidate_srr=cand_result.output,
        )
        agent_outputs["alignment"] = align_result.output

        # Step 4: Drift Detection
        drift_result = self._drift.run(
            original_srr=orig_result.output,
            candidate_srr=cand_result.output,
        )
        agent_outputs["drift"] = drift_result.output

        # Step 5: Evidence
        evidence_result = self._evidence.run(
            original_text=original_text,
            candidate_text=candidate_text,
            field_differences=drift_result.output.get("field_differences", []),
        )
        agent_outputs["evidence"] = evidence_result.output

        # Step 6: Critic
        critic_result = self._critic.run(
            original_text=original_text,
            candidate_text=candidate_text,
            field_differences=drift_result.output.get("field_differences", []),
            drift_decision=drift_result.output.get("drift_detected", False),
        )
        agent_outputs["critic"] = critic_result.output

        # Step 7: Calibration
        cal_result = self._calibration.run(
            extraction_confidence=orig_result.output.get("extraction_confidence", 0.5),
            drift_confidence=drift_result.output.get("confidence", 0.5),
            critic_confidence=critic_result.output.get("revised_confidence", 0.5),
            n_differences=drift_result.output.get("n_differences", 0),
        )
        agent_outputs["calibration"] = cal_result.output

        # Step 8: Judge
        judge_result = self._judge.run(
            original_text=original_text,
            candidate_text=candidate_text,
            drift_output=drift_result.output,
            critic_output=critic_result.output,
            calibration_output=cal_result.output,
            evidence_output=evidence_result.output,
        )
        agent_outputs["judge"] = judge_result.output

        # Build final result
        total_latency = time.time() - start
        judge_out = judge_result.output

        drift_types_raw = judge_out.get("drift_types", [])
        drift_types = []
        for dt in drift_types_raw:
            try:
                drift_types.append(DriftType(dt))
            except ValueError:
                pass

        return DriftResult(
            requirement_id=requirement_id,
            source_language=source_language,
            target_language=target_language,
            drift_detected=judge_out.get("drift_detected", False),
            drift_label=DriftLabel.DRIFT if judge_out.get("drift_detected") else DriftLabel.NO_DRIFT,
            drift_types=drift_types,
            severity=DriftSeverity(judge_out.get("severity", "none")),
            confidence=judge_out.get("confidence", 0.5),
            calibrated_confidence=cal_result.output.get("calibrated_confidence"),
            field_differences=drift_result.output.get("field_differences", []),
            evidence=[e.get("explanation", "") for e in evidence_result.output.get("evidence", [])],
            explanation=judge_out.get("explanation", ""),
            critic_assessment=critic_result.output.get("assessment", ""),
            requires_human_review=judge_out.get("requires_human_review", False),
            review_reasons=judge_out.get("review_reasons", []),
            workflow="full_idraak",
            model_metadata={"agents": list(agent_outputs.keys())},
            latency_seconds=total_latency,
        )
