"""Tests for perturbation engine."""

import pytest
from idraak.datasets.generator import DatasetGenerator
from idraak.perturbations.engine import PerturbationEngine


class TestPerturbationEngine:
    def setup_method(self):
        self.engine = PerturbationEngine(seed=42)
        gen = DatasetGenerator(seed=42)
        self.entries = gen.generate(20)

    def test_generates_perturbations(self):
        perts = self.engine.generate_all(self.entries[0])
        assert len(perts) >= 1

    def test_has_drift_labels(self):
        for entry in self.entries[:10]:
            perts = self.engine.generate_all(entry)
            for p in perts:
                assert p.drift_label in (0, 1)
                if p.drift_label == 1:
                    assert p.drift_type is not None
                    assert p.severity is not None

    def test_paraphrase_no_drift(self):
        perts = self.engine.generate_all(self.entries[0])
        paraphrases = [p for p in perts if p.perturbation_method == "paraphrase"]
        for p in paraphrases:
            assert p.drift_label == 0

    def test_numerical_drift(self):
        # Find an entry with numbers
        for entry in self.entries:
            import re
            if re.search(r"\d+", entry.original_text):
                perts = self.engine.generate_all(entry)
                num_perts = [p for p in perts if p.drift_type == "numerical_drift"]
                if num_perts:
                    p = num_perts[0]
                    assert p.drift_label == 1
                    assert p.perturbed_text != entry.original_text
                    return
        # If no entries with numbers, skip
        pytest.skip("No entries with numbers found")

    def test_modality_drift(self):
        for entry in self.entries:
            if "shall" in entry.original_text.lower():
                perts = self.engine.generate_all(entry)
                mod_perts = [p for p in perts if p.drift_type == "modality_drift"]
                if mod_perts:
                    assert mod_perts[0].drift_label == 1
                    return

    def test_deterministic_with_seed(self):
        eng1 = PerturbationEngine(seed=123)
        eng2 = PerturbationEngine(seed=123)
        perts1 = eng1.generate_all(self.entries[0])
        perts2 = eng2.generate_all(self.entries[0])
        assert len(perts1) == len(perts2)
        for p1, p2 in zip(perts1, perts2):
            assert p1.perturbed_text == p2.perturbed_text
