"""Base agent class with common functionality."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from idraak.providers.base import LLMProvider
from idraak.utils.logging import get_logger

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class AgentResult(BaseModel):
    """Wrapper for agent output with metadata."""

    output: dict[str, Any]
    agent_name: str = ""
    model: str = ""
    latency_seconds: float = 0.0
    token_usage: dict[str, Any] = {}
    estimated_cost_usd: float = 0.0
    success: bool = True
    error: str = ""


class BaseAgent(ABC):
    """Base class for all IDRAAK agents.

    Each agent has a name, optional LLM provider, and produces
    structured Pydantic output with metadata tracking.
    """

    def __init__(
        self,
        name: str,
        llm_provider: LLMProvider | None = None,
        temperature: float = 0.0,
    ):
        self.name = name
        self.llm = llm_provider
        self.temperature = temperature
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    def run(self, **kwargs: Any) -> AgentResult:
        """Execute the agent's task and return structured output."""
        ...

    def _timed_llm_call(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> dict[str, Any]:
        """Make an LLM call with timing and error handling."""
        if self.llm is None:
            return {"content": "", "usage": {}, "model": "none"}

        start = time.time()
        result = self.llm.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.temperature,
            **kwargs,
        )
        result["latency"] = time.time() - start
        return result

    def _make_result(
        self, output: dict[str, Any], latency: float = 0.0, usage: dict = {}, **kwargs: Any
    ) -> AgentResult:
        return AgentResult(
            output=output,
            agent_name=self.name,
            model=self.llm.model_name if self.llm else "deterministic",
            latency_seconds=latency,
            token_usage=usage,
            **kwargs,
        )
