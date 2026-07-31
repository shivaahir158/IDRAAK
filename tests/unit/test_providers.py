"""Tests for mock providers."""

from idraak.providers.mock import MockLLMProvider, MockTranslationProvider


class TestMockTranslationProvider:
    def test_translate(self):
        provider = MockTranslationProvider()
        result = provider.translate("Hello world", "en", "hi")
        assert "[HI]" in result.translated_text
        assert result.source_language == "en"
        assert result.target_language == "hi"
        assert result.provider == "mock"

    def test_same_language(self):
        provider = MockTranslationProvider()
        result = provider.translate("Hello world", "en", "en")
        assert result.translated_text == "Hello world"

    def test_provider_name(self):
        assert MockTranslationProvider().provider_name == "mock"


class TestMockLLMProvider:
    def test_complete(self):
        provider = MockLLMProvider()
        result = provider.complete("test prompt")
        assert "content" in result
        assert result["model"] == "mock-llm-v1"

    def test_json_response(self):
        provider = MockLLMProvider()
        result = provider.complete("extract json from text", response_format={"type": "json_object"})
        import json
        data = json.loads(result["content"])
        assert "extraction_confidence" in data

    def test_provider_name(self):
        assert MockLLMProvider().provider_name == "mock"
        assert MockLLMProvider().model_name == "mock-llm-v1"
