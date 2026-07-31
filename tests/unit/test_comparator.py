"""Tests for SRR comparison engine."""

import pytest
from idraak.drift.comparator import SRRComparator
from idraak.drift.unit_converter import UnitConverter
from idraak.schemas.srr import (
    ComparisonOperator,
    Modality,
    NumericalConstraint,
    Polarity,
    SemanticRequirement,
    TemporalConstraint,
    TemporalRelation,
)
from idraak.schemas.drift import DriftSeverity, DriftType


class TestUnitConverter:
    def test_same_unit(self):
        assert UnitConverter.are_equivalent(10, "ms", 10, "ms")

    def test_ms_to_s(self):
        assert UnitConverter.are_equivalent(1000, "ms", 1, "s")

    def test_mhz_to_ghz(self):
        assert UnitConverter.are_equivalent(1000, "MHz", 1, "GHz")

    def test_bytes_to_bits(self):
        assert UnitConverter.are_equivalent(1, "byte", 8, "bit")

    def test_different_values(self):
        assert not UnitConverter.are_equivalent(10, "ms", 10, "s")

    def test_convert(self):
        result = UnitConverter.convert(1000, "ms", "s")
        assert result is not None
        assert abs(result - 1.0) < 1e-6


class TestSRRComparator:
    def setup_method(self):
        self.comp = SRRComparator()

    def test_identical(self):
        srr = SemanticRequirement(
            modality=Modality.MANDATORY,
            polarity=Polarity.POSITIVE,
            actor="the controller",
        )
        diffs = self.comp.compare(srr, srr)
        assert len(diffs) == 0

    def test_modality_drift(self):
        orig = SemanticRequirement(modality=Modality.MANDATORY)
        cand = SemanticRequirement(modality=Modality.RECOMMENDED)
        diffs = self.comp.compare(orig, cand)
        assert len(diffs) == 1
        assert diffs[0].drift_type == DriftType.MODALITY
        assert diffs[0].severity == DriftSeverity.HIGH

    def test_modality_critical(self):
        orig = SemanticRequirement(modality=Modality.MANDATORY)
        cand = SemanticRequirement(modality=Modality.FORBIDDEN)
        diffs = self.comp.compare(orig, cand)
        assert any(d.severity == DriftSeverity.CRITICAL for d in diffs)

    def test_polarity_drift(self):
        orig = SemanticRequirement(polarity=Polarity.POSITIVE)
        cand = SemanticRequirement(polarity=Polarity.NEGATIVE)
        diffs = self.comp.compare(orig, cand)
        assert any(d.drift_type == DriftType.POLARITY for d in diffs)
        assert any(d.severity == DriftSeverity.CRITICAL for d in diffs)

    def test_numerical_drift(self):
        orig = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(parameter="timing", value=10, unit="ms")
            ]
        )
        cand = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(parameter="timing", value=100, unit="ms")
            ]
        )
        diffs = self.comp.compare(orig, cand)
        assert any(d.drift_type == DriftType.NUMERICAL for d in diffs)

    def test_unit_equivalence(self):
        """1000 ms == 1 s should NOT be drift."""
        orig = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(parameter="timing", value=1000, unit="ms")
            ]
        )
        cand = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(parameter="timing", value=1, unit="s")
            ]
        )
        diffs = self.comp.compare(orig, cand)
        assert len(diffs) == 0

    def test_operator_drift(self):
        orig = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(
                    parameter="v", operator=ComparisonOperator.GREATER_THAN_OR_EQUAL, value=10
                )
            ]
        )
        cand = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(
                    parameter="v", operator=ComparisonOperator.LESS_THAN_OR_EQUAL, value=10
                )
            ]
        )
        diffs = self.comp.compare(orig, cand)
        assert any(d.drift_type == DriftType.THRESHOLD for d in diffs)

    def test_temporal_drift(self):
        orig = SemanticRequirement(
            temporal_constraints=[
                TemporalConstraint(relation=TemporalRelation.BEFORE, reference_event="clock")
            ]
        )
        cand = SemanticRequirement(
            temporal_constraints=[
                TemporalConstraint(relation=TemporalRelation.AFTER, reference_event="clock")
            ]
        )
        diffs = self.comp.compare(orig, cand)
        assert any(d.drift_type == DriftType.TEMPORAL for d in diffs)

    def test_missing_constraint(self):
        orig = SemanticRequirement(
            numerical_constraints=[
                NumericalConstraint(parameter="v", value=10, unit="ms")
            ]
        )
        cand = SemanticRequirement(numerical_constraints=[])
        diffs = self.comp.compare(orig, cand)
        assert any(d.drift_type == DriftType.OMISSION for d in diffs)

    def test_synonym_tolerance(self):
        orig = SemanticRequirement(actor="the controller")
        cand = SemanticRequirement(actor="the control module")
        comp = SRRComparator(synonym_tolerance=True)
        diffs = comp.compare(orig, cand)
        # Should be tolerant of synonyms
        actor_diffs = [d for d in diffs if d.field == "actor"]
        assert len(actor_diffs) == 0
