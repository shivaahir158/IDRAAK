"""YAML-based configuration system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_config: dict[str, Any] = {}

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts; override wins on conflicts."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file(s), merging with defaults."""
    global _config

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            _config = yaml.safe_load(f) or {}

    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            with open(p) as f:
                override = yaml.safe_load(f) or {}
            _config = _deep_merge(_config, override)

    # Apply env var overrides
    if os.getenv("IDRAAK_ALLOW_EXTERNAL_API", "").lower() == "true":
        _config.setdefault("privacy", {})["allow_external_api"] = True

    return _config


def get_config() -> dict[str, Any]:
    """Return the current config, loading defaults if needed."""
    if not _config:
        load_config()
    return _config
