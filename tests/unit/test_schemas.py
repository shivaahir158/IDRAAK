"""Tests for Pydantic schemas."""

import pytest
from idraak.schemas.srr import (
    ComparisonOperator,
    Condition,
    Modality,
    NumericalConstraint,
    Polarity,
    SemanticRequirement,
    TemporalConstraint,
    TemporalRelation,
)
from idraak.schemas.drift import DriftLabel, DriftResult, DriftSeverity, DriftType, FieldDifference
from idraak.schemas.dataset import DatasetEntry, PerturbationRecord


class TestSemanticRequirement:
    def test_create_minimal(self):
        srr = SemanticRequirement()
        assert srr.requirement_id == ""
        assert srr.modality is None
        assert srr.conditions == []

    def test_create_full(self):
        srr = SemanticRequirement(
            requirement_id="REQ-0001",
            domain="digital_hardware",
            actor="the controller",
            action="assert",
            object="ready",
            modality=Modality.MANDATORY,
            polarity=Polarity.POSITIVE,
            conditions=[Condition(condition_type="if", trigger="valid is asserted")],
            temporal_constraints=[
                TemporalConstraint(
                    relation=TemporalRelation.WITHIN,
                    value=3,
                    unit="clock_cycle",
                )
            ],
            numerical_constraints=[
                NumericalConstraint(
                    parameter="timing",
                    operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                    value=10,
                    unit="ms",
                )
            ],
            source_language="en",
            extraction_confidence=0.9,
        )
        assert srr.requirement_id == "REQ-0001"
        assert srr.modality == Modality.MANDATORY
        assert len(srr.conditions) == 1
        assert srr.temporal_constraints[0].value == 3

    def test_serialization(self):
        srr = SemanticRequirement(requirement_id="REQ-0001", actor="test")
        data = srr.model_dump()
        assert data["requirement_id"] == "REQ-0001"
        restored = SemanticRequirement.model_validate(data)
        assert restored.actor == "test"

    def test_json_round_trip(self):
        srr = SemanticRequirement(
            requirement_id="REQ-0001",
            modality=Modality.FORBIDDEN,
            polarity=Polarity.NEGATIVE,
        )
        json_str = srr.model_dump_json()
        restored = SemanticRequirement.model_validate_json(json_str)
        assert restored.modality == Modality.FORBIDDEN


class TestDriftSchemas:
    def test_field_difference(self):
        fd = FieldDifference(
            field="modality",
            original="mandatory",
            candidate="recommended",
            drift_type=DriftType.MODALITY,
            severity=DriftSeverity.HIGH,
        )
        assert fd.drift_type == DriftType.MODALITY

    def test_drift_result(self):
        result = DriftResult(
            requirement_id="REQ-0001",
            drift_detected=True,
            drift_label=DriftLabel.DRIFT,
            severity=DriftSeverity.HIGH,
            confidence=0.95,
        )
        assert result.drift_label == DriftLabel.DRIFT
        assert result.confidence == 0.95


class TestDatasetSchemas:
    def test_dataset_entry(self):
        entry = DatasetEntry(
            requirement_id="REQ-0001",
            domain="digital_hardware",
            category="timing_constraint",
            original_text="The controller shall assert ready within 3 clock cycles.",
        )
        assert entry.domain == "digital_hardware"

    def test_perturbation_record(self):
        record = PerturbationRecord(
            requirement_id="REQ-0001-PERT-NUM",
            base_requirement_id="REQ-0001",
            perturbed_text="The controller shall assert ready within 10 clock cycles.",
            drift_label=1,
            drift_type="numerical_drift",
            severity="high",
        )
        assert record.drift_label == 1
