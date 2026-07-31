"""Translation result schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TranslationResult(BaseModel):
    """Result of a translation operation."""

    translated_text: str
    source_language: str
    target_language: str
    source_text: str = ""
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    estimated_confidence: Optional[float] = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: Optional[float] = None
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
