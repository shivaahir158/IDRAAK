"""Extraction Agent — converts text to SRR."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent
from idraak.extraction.hybrid import HybridExtractor
from idraak.providers.base import LLMProvider


class ExtractionAgent(BaseAgent):
    """Converts text into a Semantic Requirement Representation."""

    def __init__(self, llm_provider: LLMProvider | None = None, **kwargs: Any):
        super().__init__(name="extraction", llm_provider=llm_provider, **kwargs)
        self._extractor = HybridExtractor(llm_provider=llm_provider)

    def run(
        self,
        text: str = "",
        language: str = "en",
        requirement_id: str = "",
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        try:
            srr = self._extractor.extract(text, language, requirement_id)
            latency = time.time() - start
            return self._make_result(
                output=srr.model_dump(),
                latency=latency,
            )
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return AgentResult(
                output={}, agent_name=self.name, success=False, error=str(e)
            )
