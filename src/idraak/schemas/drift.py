"""Drift detection schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DriftType(str, Enum):
    NUMERICAL = "numerical_drift"
    UNIT = "unit_drift"
    POLARITY = "polarity_drift"
    MODALITY = "modality_drift"
    CONDITION = "condition_drift"
    TEMPORAL = "temporal_drift"
    THRESHOLD = "threshold_drift"
    ENTITY = "entity_drift"
    RELATION = "relation_drift"
    EXCEPTION = "exception_drift"
    OMISSION = "omission_drift"
    ADDITION = "addition_drift"
    TERMINOLOGY = "terminology_drift"
    SCOPE = "scope_drift"
    REFERENCE = "reference_drift"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftLabel(int, Enum):
    NO_DRIFT = 0
    DRIFT = 1


class FieldDifference(BaseModel):
    """A single field-level difference between two SRRs."""

    field: str = Field(description="JSON path to the differing field")
    original: Any = None
    candidate: Any = None
    difference_type: str = ""
    drift_type: Optional[DriftType] = None
    severity: DriftSeverity = DriftSeverity.NONE
    confidence: float = 1.0
    explanation: str = ""
    evidence_original: str = ""
    evidence_candidate: str = ""


class DriftResult(BaseModel):
    """Complete drift detection result for a requirement pair."""

    requirement_id: str = ""
    source_language: str = "en"
    target_language: str = ""
    drift_detected: bool = False
    drift_label: DriftLabel = DriftLabel.NO_DRIFT
    drift_types: list[DriftType] = Field(default_factory=list)
    severity: DriftSeverity = DriftSeverity.NONE
    confidence: float = 0.0
    calibrated_confidence: Optional[float] = None
    field_differences: list[FieldDifference] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    explanation: str = ""
    critic_assessment: str = ""
    requires_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    workflow: str = ""
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
