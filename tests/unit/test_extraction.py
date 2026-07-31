"""Tests for deterministic SRR extraction."""

import pytest
from idraak.extraction.deterministic import DeterministicExtractor, normalize_unit, parse_number
from idraak.schemas.srr import ComparisonOperator, Modality, Polarity, TemporalRelation


class TestParseNumber:
    def test_integer(self):
        assert parse_number("42") == 42.0

    def test_float(self):
        assert parse_number("3.14") == 3.14

    def test_word_number(self):
        assert parse_number("three") == 3.0
        assert parse_number("hundred") == 100.0

    def test_invalid(self):
        assert parse_number("abc") is None


class TestNormalizeUnit:
    def test_known_aliases(self):
        assert normalize_unit("milliseconds") == "ms"
        assert normalize_unit("seconds") == "s"
        assert normalize_unit("megahertz") == "MHz"
        assert normalize_unit("kilobytes") == "KB"

    def test_unknown(self):
        assert normalize_unit("foobar") == "foobar"


class TestDeterministicExtractor:
    def setup_method(self):
        self.ext = DeterministicExtractor()

    def test_modality_shall(self):
        srr = self.ext.extract("The system shall process the request.")
        assert srr.modality == Modality.MANDATORY

    def test_modality_should(self):
        srr = self.ext.extract("The system should log all errors.")
        assert srr.modality == Modality.RECOMMENDED

    def test_modality_may(self):
        srr = self.ext.extract("The system may retry the operation.")
        assert srr.modality == Modality.PERMITTED

    def test_modality_shall_not(self):
        srr = self.ext.extract("The system shall not discard data.")
        assert srr.modality == Modality.FORBIDDEN

    def test_polarity_negative(self):
        srr = self.ext.extract("The system shall not modify the register.")
        assert srr.polarity == Polarity.NEGATIVE

    def test_polarity_positive(self):
        srr = self.ext.extract("The system shall process the data.")
        assert srr.polarity == Polarity.POSITIVE

    def test_actor_extraction(self):
        srr = self.ext.extract("The controller shall assert ready.")
        assert srr.actor is not None
        assert "controller" in srr.actor.lower()

    def test_temporal_within(self):
        srr = self.ext.extract("The controller shall respond within 3 clock cycles of the request.")
        assert len(srr.temporal_constraints) >= 1
        tc = srr.temporal_constraints[0]
        assert tc.relation == TemporalRelation.WITHIN
        assert tc.value == 3.0

    def test_temporal_before(self):
        srr = self.ext.extract("Signal A shall be stable before signal B transitions.")
        assert any(tc.relation == TemporalRelation.BEFORE for tc in srr.temporal_constraints)

    def test_numerical_at_least(self):
        srr = self.ext.extract("The system shall support at least 100 connections.")
        assert len(srr.numerical_constraints) >= 1
        nc = srr.numerical_constraints[0]
        assert nc.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL
        assert nc.value == 100.0

    def test_numerical_not_exceed(self):
        srr = self.ext.extract("Latency shall not exceed 10 ms.")
        assert len(srr.numerical_constraints) >= 1
        nc = srr.numerical_constraints[0]
        assert nc.operator == ComparisonOperator.LESS_THAN_OR_EQUAL
        assert nc.value == 10.0

    def test_numerical_between(self):
        srr = self.ext.extract("The voltage shall be between 3 and 5 volts.")
        assert any(nc.operator == ComparisonOperator.BETWEEN for nc in srr.numerical_constraints)

    def test_condition_if(self):
        srr = self.ext.extract("If the system is idle, the controller shall enter sleep mode.")
        assert len(srr.conditions) >= 1
        assert srr.conditions[0].condition_type == "if"

    def test_exception_unless(self):
        srr = self.ext.extract("The system shall respond unless a fault is detected.")
        assert len(srr.exceptions) >= 1
        assert "fault" in srr.exceptions[0].condition.lower()

    def test_ordering(self):
        srr = self.ext.extract("The reset signal shall be asserted before the clock starts.")
        assert len(srr.ordering_constraints) >= 1

    def test_empty_text(self):
        srr = self.ext.extract("")
        assert srr.modality is None
        assert srr.conditions == []

    def test_units_extraction(self):
        srr = self.ext.extract("The latency shall be at most 100 milliseconds.")
        assert "ms" in srr.units
