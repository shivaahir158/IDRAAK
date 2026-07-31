"""Evidence Agent — locates textual evidence for detected differences."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent


class EvidenceAgent(BaseAgent):
    """Locates supporting evidence for drift decisions."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="evidence", **kwargs)

    def run(
        self,
        original_text: str = "",
        candidate_text: str = "",
        field_differences: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        field_differences = field_differences or []
        evidence_items = []

        for diff in field_differences:
            field = diff.get("field", "")
            original_val = diff.get("original")
            candidate_val = diff.get("candidate")

            # Find relevant spans in texts
            evidence = {
                "field": field,
                "original_span": self._find_span(original_text, original_val),
                "candidate_span": self._find_span(candidate_text, candidate_val),
                "explanation": diff.get("explanation", f"Field '{field}' differs between original and candidate"),
            }
            evidence_items.append(evidence)

        latency = time.time() - start
        return self._make_result(
            output={
                "evidence": evidence_items,
                "n_evidence": len(evidence_items),
            },
            latency=latency,
        )

    def _find_span(self, text: str, value: Any) -> str:
        """Find a short span in text that matches the value."""
        if value is None or not text:
            return ""
        val_str = str(value).lower()
        text_lower = text.lower()
        idx = text_lower.find(val_str)
        if idx >= 0:
            start = max(0, idx - 20)
            end = min(len(text), idx + len(val_str) + 20)
            return text[start:end]
        return ""
