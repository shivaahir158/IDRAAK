"""Mock providers for testing without API access."""

from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Any

from idraak.schemas.translation import TranslationResult


# Simple mock translations — just wraps text with language marker
_MOCK_PREFIXES = {
    "hi": "[HI]", "ur": "[UR]", "ar": "[AR]", "zh": "[ZH]",
    "ja": "[JA]", "es": "[ES]", "fr": "[FR]", "de": "[DE]",
    "pt": "[PT]", "tr": "[TR]", "bn": "[BN]", "en": "",
}


class MockTranslationProvider:
    """Mock translation provider that wraps text with language markers.

    Useful for testing pipeline logic without API calls.
    For more realistic testing, use cached real translations.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        **kwargs: Any,
    ) -> TranslationResult:
        prefix = _MOCK_PREFIXES.get(target_language, f"[{target_language.upper()}]")
        if target_language == source_language:
            translated = text
        else:
            translated = f"{prefix} {text}"

        return TranslationResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            source_text=text,
            provider="mock",
            model="mock-v1",
            temperature=0.0,
            timestamp=datetime.now(),
            estimated_confidence=0.5,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0},
            cached=False,
        )


class MockLLMProvider:
    """Mock LLM provider that returns structured dummy responses."""

    def __init__(self, default_response: dict[str, Any] | None = None):
        self._default = default_response or {}

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-llm-v1"

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Return a basic SRR-like extraction if JSON is requested
        if response_format or "json" in prompt.lower():
            return {
                "content": json.dumps({
                    "actor": "the system",
                    "action": "process",
                    "object": "the request",
                    "modality": "mandatory",
                    "polarity": "positive",
                    "conditions": [],
                    "temporal_constraints": [],
                    "numerical_constraints": [],
                    "extraction_confidence": 0.7,
                }),
                "model": "mock-llm-v1",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "finish_reason": "stop",
            }

        # For judge/critic prompts, return a basic assessment
        if "drift" in prompt.lower() or "judge" in prompt.lower():
            return {
                "content": json.dumps({
                    "drift_detected": random.random() > 0.5,
                    "confidence": round(random.uniform(0.5, 1.0), 2),
                    "explanation": "Mock assessment — no real analysis performed.",
                }),
                "model": "mock-llm-v1",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "finish_reason": "stop",
            }

        return {
            "content": "Mock response",
            "model": "mock-llm-v1",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "finish_reason": "stop",
            **self._default,
        }
