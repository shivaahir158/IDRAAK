"""Tests for complexity scoring."""

from idraak.evaluation.complexity import ComplexityScorer


class TestComplexityScorer:
    def setup_method(self):
        self.scorer = ComplexityScorer()

    def test_simple_requirement(self):
        score = self.scorer.score("The system shall process data.")
        assert score.overall_score < 0.4
        assert score.complexity_level in ("easy", "medium")

    def test_complex_requirement(self):
        text = (
            "If the watchdog timer expires and the interrupt handler has not "
            "responded within 10 clock cycles, the controller shall enter "
            "safe state before asserting the reset signal, unless the system "
            "is in debug mode or calibration is active."
        )
        score = self.scorer.score(text)
        assert score.condition_count >= 1
        assert score.temporal_relation_count >= 1
        assert score.token_count > 20
        assert score.overall_score > 0.3

    def test_numerical_density(self):
        text = "The value shall be between 3.0 and 5.5 volts at 100 MHz."
        score = self.scorer.score(text)
        assert score.numerical_count >= 3

    def test_domain_term_density(self):
        text = "The buffer register overflow interrupt shall trigger the watchdog."
        score = self.scorer.score(text)
        assert score.domain_term_density > 0

    def test_empty_text(self):
        score = self.scorer.score("")
        assert score.overall_score < 0.1
        assert score.token_count == 0

    def test_to_dict(self):
        score = self.scorer.score("The system shall respond.")
        d = score.to_dict()
        assert "overall_score" in d
        assert "complexity_level" in d
