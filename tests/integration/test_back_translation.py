"""Tests for back-translation workflow."""

from idraak.providers.mock import MockTranslationProvider
from idraak.workflows.back_translation import BackTranslationWorkflow


class TestBackTranslationWorkflow:
    def test_basic_run(self):
        provider = MockTranslationProvider()
        wf = BackTranslationWorkflow(translation_provider=provider)
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            target_language="hi",
            requirement_id="REQ-0001",
        )
        assert result.requirement_id == "REQ-0001"
        assert result.translated_text != ""
        assert result.back_translated_text != ""
        assert result.drift_result.workflow == "back_translation"

    def test_to_dict(self):
        provider = MockTranslationProvider()
        wf = BackTranslationWorkflow(translation_provider=provider)
        result = wf.run(
            original_text="The controller shall assert ready.",
            target_language="ar",
            requirement_id="REQ-0002",
        )
        d = result.to_dict()
        assert "original_text" in d
        assert "translated_text" in d
        assert "back_translated_text" in d
        assert "drift_detected" in d
