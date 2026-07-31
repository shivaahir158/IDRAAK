"""Tests for debate workflow."""

from idraak.workflows.debate import DebateWorkflow


class TestDebateWorkflow:
    def test_identical_texts_no_drift(self):
        wf = DebateWorkflow()
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            candidate_text="The system shall respond within 5 ms.",
            requirement_id="REQ-D001",
        )
        assert not result.drift_detected
        assert result.workflow == "debate"
        assert result.confidence >= 0.5

    def test_numerical_drift_detected(self):
        wf = DebateWorkflow()
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            candidate_text="The system shall respond within 50 ms.",
            requirement_id="REQ-D002",
        )
        assert result.drift_detected
        assert result.workflow == "debate"
        assert len(result.field_differences) > 0

    def test_modality_drift_detected(self):
        wf = DebateWorkflow()
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            candidate_text="The system should respond within 5 ms.",
            requirement_id="REQ-D003",
        )
        assert result.drift_detected
        assert result.workflow == "debate"

    def test_polarity_drift_detected(self):
        wf = DebateWorkflow()
        result = wf.run(
            original_text="The system shall not exceed 100 watts.",
            candidate_text="The system shall exceed 100 watts.",
            requirement_id="REQ-D004",
        )
        assert result.drift_detected
        assert result.confidence > 0.6

    def test_debate_metadata(self):
        wf = DebateWorkflow(n_rounds=3)
        result = wf.run(
            original_text="The controller shall assert ready.",
            candidate_text="The controller should assert ready.",
            requirement_id="REQ-D005",
        )
        assert "pro_drift_confidence" in result.model_metadata
        assert "no_drift_confidence" in result.model_metadata
        assert result.model_metadata["n_rounds"] == 3

    def test_explanation_contains_reasoning(self):
        wf = DebateWorkflow()
        result = wf.run(
            original_text="The buffer shall hold 64 KB.",
            candidate_text="The buffer shall hold 64 MB.",
            requirement_id="REQ-D006",
        )
        assert "score=" in result.explanation
        assert result.explanation != ""
