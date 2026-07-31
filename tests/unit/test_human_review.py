"""Tests for human review routing."""

from idraak.evaluation.human_review import HumanReviewRouter


class TestHumanReviewRouter:
    def setup_method(self):
        self.router = HumanReviewRouter()

    def test_confident_no_review(self):
        decision = self.router.evaluate(
            confidence=0.95, drift_detected=True, severity="medium",
            n_differences=1, critic_agrees=True,
        )
        assert not decision.requires_human_review

    def test_low_confidence_triggers_review(self):
        decision = self.router.evaluate(confidence=0.3)
        assert decision.requires_human_review
        assert any("confidence" in r.lower() for r in decision.review_reasons)

    def test_critic_disagrees_triggers_review(self):
        decision = self.router.evaluate(
            confidence=0.8, critic_agrees=False,
        )
        assert decision.requires_human_review
        assert any("critic" in r.lower() for r in decision.review_reasons)

    def test_critical_severity_triggers_review(self):
        decision = self.router.evaluate(
            confidence=0.9, severity="critical", n_differences=1,
        )
        assert decision.requires_human_review
        assert decision.priority == "critical"

    def test_safety_drift_high_priority(self):
        decision = self.router.evaluate(
            confidence=0.7, severity="high",
            drift_types=["polarity_drift"],
        )
        assert decision.requires_human_review
        assert decision.priority in ("high", "critical")

    def test_agent_disagreement(self):
        decision = self.router.evaluate(
            confidence=0.9, critic_confidence=0.3,
        )
        assert decision.requires_human_review
        assert any("disagreement" in r.lower() for r in decision.review_reasons)

    def test_unsupported_terminology(self):
        decision = self.router.evaluate(
            confidence=0.8, has_unsupported_terminology=True,
        )
        assert decision.requires_human_review
        assert decision.suggested_reviewer == "domain_expert"
