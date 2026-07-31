"""Human-review routing — identifies cases requiring expert review."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from idraak.utils.logging import get_logger

logger = get_logger("human_review")


class HumanReviewDecision(BaseModel):
    """Decision about whether human review is needed."""

    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    priority: str = "low"  # low, medium, high, critical
    suggested_reviewer: str = ""  # domain expert, linguist, etc.


# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_HIGH_SEVERITY_THRESHOLD = 2
DEFAULT_AGENT_DISAGREEMENT_THRESHOLD = 0.3


class HumanReviewRouter:
    """Routes uncertain or conflicting cases to human review."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        high_severity_threshold: int = DEFAULT_HIGH_SEVERITY_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.high_severity_threshold = high_severity_threshold

    def evaluate(
        self,
        confidence: float = 0.5,
        drift_detected: bool = False,
        severity: str = "none",
        n_differences: int = 0,
        critic_agrees: bool = True,
        critic_confidence: float = 1.0,
        extraction_confidence: float = 1.0,
        drift_types: list[str] | None = None,
        has_unsupported_terminology: bool = False,
        source_ambiguous: bool = False,
    ) -> HumanReviewDecision:
        """Determine if a case requires human review."""
        reasons: list[str] = []
        priority = "low"

        # Low confidence
        if confidence < self.confidence_threshold:
            reasons.append(f"Low final confidence ({confidence:.2f})")
            priority = "medium"

        # Very low confidence
        if confidence < 0.3:
            reasons.append(f"Very low confidence ({confidence:.2f})")
            priority = "high"

        # Critic disagrees
        if not critic_agrees:
            reasons.append("Critic disagrees with drift detection")
            priority = max(priority, "medium", key=["low", "medium", "high", "critical"].index)

        # Large confidence gap between critic and detector
        if abs(confidence - critic_confidence) > DEFAULT_AGENT_DISAGREEMENT_THRESHOLD:
            reasons.append(
                f"Agent disagreement (detector: {confidence:.2f}, critic: {critic_confidence:.2f})"
            )
            priority = "medium"

        # Multiple high-severity differences
        if severity in ("high", "critical") and n_differences >= self.high_severity_threshold:
            reasons.append(f"Multiple high-severity differences ({n_differences})")
            priority = "high"

        # Critical severity
        if severity == "critical":
            reasons.append("Critical severity drift detected")
            priority = "critical"

        # Low extraction confidence
        if extraction_confidence < 0.4:
            reasons.append(f"Low extraction confidence ({extraction_confidence:.2f})")
            priority = max(priority, "medium", key=["low", "medium", "high", "critical"].index)

        # Unsupported terminology
        if has_unsupported_terminology:
            reasons.append("Unsupported domain terminology detected")

        # Ambiguous source
        if source_ambiguous:
            reasons.append("Source requirement is ambiguous")

        # Safety or security drift requires review
        drift_types = drift_types or []
        safety_types = {"polarity_drift", "modality_drift", "exception_drift"}
        if any(dt in safety_types for dt in drift_types) and severity in ("high", "critical"):
            reasons.append("Safety-relevant drift type with high severity")
            priority = max(priority, "high", key=["low", "medium", "high", "critical"].index)

        requires_review = len(reasons) > 0

        # Determine suggested reviewer
        reviewer = ""
        if has_unsupported_terminology:
            reviewer = "domain_expert"
        elif any(dt in safety_types for dt in drift_types):
            reviewer = "safety_engineer"
        elif reasons:
            reviewer = "linguist"

        return HumanReviewDecision(
            requires_human_review=requires_review,
            review_reasons=reasons,
            priority=priority,
            suggested_reviewer=reviewer,
        )
