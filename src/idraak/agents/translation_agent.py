"""Translation Agent — translates requirements preserving technical semantics."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent
from idraak.providers.base import TranslationProvider
from idraak.utils.logging import get_logger


class TranslationAgent(BaseAgent):
    """Translates technical requirements while preserving critical attributes."""

    def __init__(
        self,
        translation_provider: TranslationProvider,
        **kwargs: Any,
    ):
        super().__init__(name="translation", **kwargs)
        self._translator = translation_provider

    def run(
        self,
        text: str = "",
        source_language: str = "en",
        target_language: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        try:
            result = self._translator.translate(text, source_language, target_language)
            latency = time.time() - start
            return self._make_result(
                output={
                    "translated_text": result.translated_text,
                    "source_language": source_language,
                    "target_language": target_language,
                    "provider": result.provider,
                    "model": result.model,
                    "confidence": result.estimated_confidence,
                },
                latency=latency,
                usage=result.token_usage,
            )
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            return AgentResult(
                output={"translated_text": "", "error": str(e)},
                agent_name=self.name,
                success=False,
                error=str(e),
            )
