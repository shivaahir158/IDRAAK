"""Deterministic caching for API calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from idraak.utils.logging import get_logger

logger = get_logger("cache")

_CACHE_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "cache"


def cache_key(*parts: Any) -> str:
    """Create a deterministic cache key from parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(key: str, namespace: str = "default") -> Optional[dict]:
    """Retrieve a cached result."""
    path = _CACHE_DIR / namespace / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def set_cached(key: str, value: dict, namespace: str = "default") -> None:
    """Store a result in the cache."""
    path = _CACHE_DIR / namespace / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2, default=str)


def translation_cache_key(
    source_text: str,
    source_lang: str,
    target_lang: str,
    provider: str,
    model: str,
    temperature: float = 0.0,
    prompt_version: str = "v1",
) -> str:
    """Build a deterministic cache key for translations."""
    return cache_key(
        source_text, source_lang, target_lang, provider, model, temperature, prompt_version
    )
