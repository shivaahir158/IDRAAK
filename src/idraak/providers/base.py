"""Base protocol definitions for providers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from idraak.schemas.translation import TranslationResult


@runtime_checkable
class TranslationProvider(Protocol):
    """Protocol for translation providers."""

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        **kwargs: Any,
    ) -> TranslationResult: ...

    @property
    def provider_name(self) -> str: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM inference providers."""

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...
