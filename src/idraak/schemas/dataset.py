"""Dataset and perturbation schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetEntry(BaseModel):
    """A single technical requirement in the benchmark dataset."""

    requirement_id: str
    domain: str
    category: str
    original_text: str
    structured_ground_truth: dict[str, Any] = Field(default_factory=dict)
    critical_attributes: list[str] = Field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    source: str = "synthetic"
    complexity_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PerturbationRecord(BaseModel):
    """A perturbation applied to a base requirement."""

    requirement_id: str
    base_requirement_id: str
    original_text: str = ""
    perturbed_text: str
    drift_label: int  # 0 = no drift, 1 = drift
    drift_type: Optional[str] = None
    severity: Optional[str] = None
    changed_fields: list[str] = Field(default_factory=list)
    original_value: Any = None
    modified_value: Any = None
    perturbation_method: str = ""
    description: str = ""
    source_language: str = "en"
    target_language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
