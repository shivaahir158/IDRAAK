"""Base extractor protocol."""

from __future__ import annotations

from typing import Protocol

from idraak.schemas.srr import SemanticRequirement


class RequirementExtractor(Protocol):
    """Protocol for SRR extraction from text."""

    def extract(
        self, text: str, language: str, requirement_id: str = ""
    ) -> SemanticRequirement: ...
