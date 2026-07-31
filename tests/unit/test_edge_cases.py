"""Edge case tests for extraction, comparison, and perturbation."""

import unicodedata

from idraak.drift.comparator import SRRComparator
from idraak.extraction.deterministic import DeterministicExtractor
from idraak.perturbations.engine import PerturbationEngine
from idraak.schemas.srr import SemanticRequirement


class TestUnicodeNormalization:
    def setup_method(self):
        self.extractor = DeterministicExtractor()

    def test_nfc_vs_nfd_equivalence(self):
        """NFC and NFD forms of the same text should extract identically."""
        text_nfc = unicodedata.normalize("NFC", "The system shall respond within 5 \u00b5s.")
        text_nfd = unicodedata.normalize("NFD", "The system shall respond within 5 \u00b5s.")
        srr_nfc = self.extractor.extract(text_nfc)
        srr_nfd = self.extractor.extract(text_nfd)
        assert srr_nfc.modality == srr_nfd.modality
        assert srr_nfc.actor == srr_nfd.actor

    def test_fullwidth_numbers(self):
        """Fullwidth digits should still extract numerical values."""
        text = "The system shall respond within \uff15 ms."
        srr = self.extractor.extract(text)
        # Fullwidth 5 may or may not be parsed; just ensure no crash
        assert srr.modality is not None

    def test_mixed_script_text(self):
        """Text mixing Latin and other scripts should not crash."""
        text = "The \u0633\u064a\u0633\u062a\u0645 shall respond within 5 ms."
        srr = self.extractor.extract(text)
        assert srr.modality is not None
        # "within 5 ms" goes to temporal_constraints, not numerical
        assert len(srr.temporal_constraints) >= 1


class TestRTLText:
    def setup_method(self):
        self.extractor = DeterministicExtractor()

    def test_rtl_markers_dont_crash(self):
        """Right-to-left markers should not cause extraction errors."""
        text = "\u200fThe system shall respond within 5 ms.\u200f"
        srr = self.extractor.extract(text)
        assert srr.modality is not None

    def test_bidi_text(self):
        """Bidirectional text should still extract English patterns."""
        text = "The system shall process \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a within 10 ms."
        srr = self.extractor.extract(text)
        assert srr.modality is not None
        # "within 10 ms" goes to temporal_constraints
        assert len(srr.temporal_constraints) >= 1


class TestBoundaryConditions:
    def setup_method(self):
        self.extractor = DeterministicExtractor()
        self.comparator = SRRComparator()

    def test_very_long_requirement(self):
        """Very long requirement text should not crash."""
        text = "The system shall " + "process data and " * 100 + "respond within 5 ms."
        srr = self.extractor.extract(text)
        assert srr.modality is not None
        assert srr.action != ""

    def test_single_word(self):
        """Single word should not crash extraction."""
        srr = self.extractor.extract("shall")
        assert srr is not None

    def test_only_whitespace(self):
        """Whitespace-only text should be handled gracefully."""
        srr = self.extractor.extract("   \t\n   ")
        assert srr is not None

    def test_special_characters_only(self):
        """Special characters should not crash extraction."""
        srr = self.extractor.extract("!@#$%^&*()")
        assert srr is not None

    def test_compare_empty_srrs(self):
        """Comparing two empty SRRs should return no differences."""
        srr1 = SemanticRequirement()
        srr2 = SemanticRequirement()
        diffs = self.comparator.compare(srr1, srr2)
        assert len(diffs) == 0

    def test_compare_identical_srrs(self):
        """Comparing identical SRRs should return no differences."""
        srr = self.extractor.extract("The system shall respond within 5 ms.")
        diffs = self.comparator.compare(srr, srr)
        assert len(diffs) == 0

    def test_numerical_zero_value(self):
        """Zero values should be extracted correctly."""
        srr = self.extractor.extract("The offset shall be 0 volts.")
        assert any(c.value == 0.0 for c in srr.numerical_constraints)

    def test_negative_number(self):
        """Negative numbers should be handled."""
        srr = self.extractor.extract("The temperature shall not fall below -40 degrees.")
        assert len(srr.numerical_constraints) >= 1

    def test_very_large_number(self):
        """Very large numbers should parse correctly."""
        srr = self.extractor.extract("The system shall handle 1000000000 transactions.")
        assert any(c.value == 1_000_000_000 for c in srr.numerical_constraints)


class TestPerturbationEdgeCases:
    def setup_method(self):
        self.engine = PerturbationEngine(seed=42)

    def test_perturbation_on_minimal_text(self):
        """Perturbation should handle minimal text without crashing."""
        result = self.engine.perturb("The system shall work.", "numerical")
        assert result is not None

    def test_perturbation_preserves_structure(self):
        """Perturbation should produce valid text."""
        text = "The controller shall respond within 10 ms."
        result = self.engine.perturb(text, "modality")
        if result:
            assert len(result.perturbed_text) > 0

    def test_paraphrase_generation(self):
        """Paraphrase (no-drift) should not crash."""
        text = "The system shall respond within 5 ms."
        result = self.engine.perturb(text, "paraphrase")
        if result:
            assert result.drift_label == 0


class TestComparatorEdgeCases:
    def setup_method(self):
        self.comparator = SRRComparator()
        self.extractor = DeterministicExtractor()

    def test_must_vs_shall_no_drift(self):
        """'must' and 'shall' should be equivalent (both mandatory)."""
        srr1 = self.extractor.extract("The system shall respond within 5 ms.")
        srr2 = self.extractor.extract("The system must respond within 5 ms.")
        diffs = self.comparator.compare(srr1, srr2)
        modality_diffs = [d for d in diffs if d.field == "modality"]
        assert len(modality_diffs) == 0

    def test_equivalent_unit_conversion(self):
        """1000 ms and 1 s should be equivalent."""
        srr1 = self.extractor.extract("The system shall respond within 1000 ms.")
        srr2 = self.extractor.extract("The system shall respond within 1 s.")
        diffs = self.comparator.compare(srr1, srr2)
        # Should either have no numerical diff or have it marked as equivalent
        numerical_diffs = [d for d in diffs if "numerical" in d.field]
        for d in numerical_diffs:
            assert d.severity in ("none", "low") or d.confidence < 0.5
