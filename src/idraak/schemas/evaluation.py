"""Evaluation output schema."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvaluationOutput(BaseModel):
    """Final normalized output for an evaluated requirement pair."""

    requirement_id: str = ""
    source_language: str = "en"
    target_language: str = ""
    original_text: str = ""
    translated_text: str = ""
    back_translation: str = ""
    drift_detected: bool = False
    drift_type: list[str] = Field(default_factory=list)
    severity: str = "none"
    confidence: float = 0.0
    calibrated_confidence: Optional[float] = None
    field_differences: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    explanation: str = ""
    critic_assessment: str = ""
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    workflow: str = ""
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    gold_label: Optional[int] = None
    gold_drift_type: Optional[str] = None
    gold_severity: Optional[str] = None
