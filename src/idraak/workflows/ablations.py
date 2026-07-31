"""Ablation study framework — systematically disable components."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from idraak.agents.calibration_agent import CalibrationAgent
from idraak.agents.critic_agent import CriticAgent
from idraak.agents.drift_agent import DriftDetectionAgent
from idraak.agents.evidence_agent import EvidenceAgent
from idraak.agents.extraction_agent import ExtractionAgent
from idraak.agents.judge_agent import JudgeAgent
from idraak.providers.base import LLMProvider, TranslationProvider
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity
from idraak.utils.logging import get_logger

logger = get_logger("ablation")


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str
    disable_critic: bool = False
    disable_evidence: bool = False
    disable_calibration: bool = False
    disable_deterministic_comparison: bool = False
    disable_back_translation: bool = False
    disable_field_confidence: bool = False
    use_single_extraction: bool = False
    use_only_embeddings: bool = False
    use_only_direct_judge: bool = False
    use_only_srr_comparison: bool = False
    use_single_judge: bool = False
    disable_glossary: bool = False


# Predefined ablation configurations
ABLATION_CONFIGS = {
    "no_critic": AblationConfig(
        name="no_critic",
        description="Remove Critic Agent",
        disable_critic=True,
    ),
    "no_evidence": AblationConfig(
        name="no_evidence",
        description="Remove Evidence Agent",
        disable_evidence=True,
    ),
    "no_calibration": AblationConfig(
        name="no_calibration",
        description="Remove calibration",
        disable_calibration=True,
    ),
    "no_deterministic": AblationConfig(
        name="no_deterministic",
        description="Remove deterministic SRR comparison",
        disable_deterministic_comparison=True,
    ),
    "single_extraction": AblationConfig(
        name="single_extraction",
        description="Use one extraction agent instead of language-specific",
        use_single_extraction=True,
    ),
    "only_embeddings": AblationConfig(
        name="only_embeddings",
        description="Use only embeddings",
        use_only_embeddings=True,
    ),
    "only_direct_judge": AblationConfig(
        name="only_direct_judge",
        description="Use only direct LLM judge",
        use_only_direct_judge=True,
    ),
    "only_srr": AblationConfig(
        name="only_srr",
        description="Use only SRR comparison",
        use_only_srr_comparison=True,
    ),
    "single_judge": AblationConfig(
        name="single_judge",
        description="Replace multi-agent with single judge",
        use_single_judge=True,
    ),
    "no_glossary": AblationConfig(
        name="no_glossary",
        description="Disable domain terminology glossary",
        disable_glossary=True,
    ),
}


class AblationWorkflow:
    """Runs the IDRAAK workflow with specific components disabled."""

    def __init__(
        self,
        ablation_config: AblationConfig,
        llm_provider: LLMProvider | None = None,
        translation_provider: TranslationProvider | None = None,
    ):
        self.config = ablation_config
        self._llm = llm_provider
        self._translator = translation_provider

    def run(
        self,
        original_text: str,
        candidate_text: str,
        requirement_id: str = "",
        source_language: str = "en",
        target_language: str = "",
    ) -> DriftResult:
        """Run ablated workflow."""
        start = time.time()

        # Short-circuit for pure baseline ablations
        if self.config.use_only_embeddings:
            return self._embeddings_only(original_text, candidate_text, requirement_id, start)

        if self.config.use_only_direct_judge:
            from idraak.workflows.direct_judge import DirectJudgeWorkflow
            wf = DirectJudgeWorkflow(llm_provider=self._llm)
            return wf.run(original_text, candidate_text, target_language, requirement_id)

        if self.config.use_only_srr_comparison:
            from idraak.workflows.structured_single import StructuredSingleWorkflow
            wf = StructuredSingleWorkflow(llm_provider=None)
            return wf.run(original_text, candidate_text, source_language, target_language, requirement_id)

        # Modified multi-agent workflow
        extraction = ExtractionAgent(llm_provider=self._llm)
        drift_agent = DriftDetectionAgent()

        # Extract
        orig_result = extraction.run(text=original_text, language=source_language, requirement_id=requirement_id)
        cand_result = extraction.run(text=candidate_text, language=target_language or source_language)

        # Drift detection
        drift_result = drift_agent.run(original_srr=orig_result.output, candidate_srr=cand_result.output)
        drift_out = drift_result.output

        # Evidence (optional)
        evidence_out: dict[str, Any] = {}
        if not self.config.disable_evidence:
            evidence_agent = EvidenceAgent()
            ev = evidence_agent.run(
                original_text=original_text,
                candidate_text=candidate_text,
                field_differences=drift_out.get("field_differences", []),
            )
            evidence_out = ev.output

        # Critic (optional)
        critic_out: dict[str, Any] = {"agrees_with_decision": True, "revised_confidence": drift_out.get("confidence", 0.5)}
        if not self.config.disable_critic:
            critic = CriticAgent(llm_provider=self._llm)
            cr = critic.run(
                original_text=original_text,
                candidate_text=candidate_text,
                field_differences=drift_out.get("field_differences", []),
                drift_decision=drift_out.get("drift_detected", False),
            )
            critic_out = cr.output

        # Calibration (optional)
        confidence = drift_out.get("confidence", 0.5)
        if not self.config.disable_calibration:
            cal = CalibrationAgent()
            cal_result = cal.run(
                extraction_confidence=orig_result.output.get("extraction_confidence", 0.5),
                drift_confidence=confidence,
                critic_confidence=critic_out.get("revised_confidence", 0.5),
                n_differences=drift_out.get("n_differences", 0),
            )
            confidence = cal_result.output.get("calibrated_confidence", confidence)

        # Judge
        if self.config.use_single_judge:
            # Skip multi-agent synthesis, use drift detection directly
            pass
        else:
            judge = JudgeAgent(llm_provider=self._llm)
            judge_result = judge.run(
                original_text=original_text,
                candidate_text=candidate_text,
                drift_output=drift_out,
                critic_output=critic_out,
                calibration_output={"calibrated_confidence": confidence},
            )
            drift_out = {**drift_out, **judge_result.output}
            confidence = judge_result.output.get("confidence", confidence)

        latency = time.time() - start
        return DriftResult(
            requirement_id=requirement_id,
            source_language=source_language,
            target_language=target_language,
            drift_detected=drift_out.get("drift_detected", False),
            drift_label=DriftLabel.DRIFT if drift_out.get("drift_detected") else DriftLabel.NO_DRIFT,
            severity=DriftSeverity(drift_out.get("severity", "none")),
            confidence=confidence,
            field_differences=drift_out.get("field_differences", []),
            explanation=drift_out.get("explanation", ""),
            critic_assessment=critic_out.get("assessment", ""),
            workflow=f"ablation_{self.config.name}",
            latency_seconds=latency,
        )

    def _embeddings_only(
        self, orig: str, cand: str, req_id: str, start: float
    ) -> DriftResult:
        from idraak.evaluation.baselines import EmbeddingBaseline
        baseline = EmbeddingBaseline()
        result = baseline.predict(orig, cand)
        return DriftResult(
            requirement_id=req_id,
            drift_detected=result["drift_detected"],
            drift_label=DriftLabel.DRIFT if result["drift_detected"] else DriftLabel.NO_DRIFT,
            confidence=result["confidence"],
            explanation=f"Embedding similarity: {result.get('similarity', 0):.3f}",
            workflow="ablation_only_embeddings",
            latency_seconds=time.time() - start,
        )


def get_all_ablation_configs() -> dict[str, AblationConfig]:
    """Return all predefined ablation configurations."""
    return ABLATION_CONFIGS.copy()
