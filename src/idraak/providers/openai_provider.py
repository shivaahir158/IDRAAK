"""OpenAI-compatible API provider."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from idraak.schemas.translation import TranslationResult
from idraak.utils.cache import (
    get_cached,
    set_cached,
    translation_cache_key,
)
from idraak.utils.logging import get_logger

logger = get_logger("openai_provider")


class OpenAITranslationProvider:
    """Translation via OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        cache_enabled: bool = True,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self._model = model
        self._temperature = temperature
        self._cache_enabled = cache_enabled

        if not self._api_key:
            logger.warning("No OPENAI_API_KEY set — API calls will fail")

    @property
    def provider_name(self) -> str:
        return "openai"

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        **kwargs: Any,
    ) -> TranslationResult:
        if self._cache_enabled:
            ck = translation_cache_key(
                text, source_language, target_language,
                "openai", self._model, self._temperature,
            )
            cached = get_cached(ck, namespace="translations")
            if cached:
                return TranslationResult(**cached, cached=True)

        prompt = (
            f"Translate the following technical requirement from {source_language} "
            f"to {target_language}. Preserve all technical terms, numerical values, "
            f"units, operators, and logical structure exactly.\n\n"
            f"Text: {text}\n\n"
            f"Translation:"
        )

        response = self._chat_complete(prompt, temperature=self._temperature)
        translated = response["content"].strip()

        result = TranslationResult(
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            source_text=text,
            provider="openai",
            model=self._model,
            temperature=self._temperature,
            timestamp=datetime.now(),
            token_usage=response.get("usage", {}),
        )

        if self._cache_enabled:
            set_cached(ck, result.model_dump(), namespace="translations")

        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def _chat_complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", self._model),
            "usage": data.get("usage", {}),
            "finish_reason": data["choices"][0].get("finish_reason", "stop"),
        }


class OpenAILLMProvider:
    """General LLM provider via OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self._model = model

        if not self._api_key:
            logger.warning("No OPENAI_API_KEY set — API calls will fail")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", self._model),
            "usage": data.get("usage", {}),
            "finish_reason": data["choices"][0].get("finish_reason", "stop"),
        }
