"""Alignment Agent — aligns fields across two SRRs."""

from __future__ import annotations

import time
from typing import Any

from idraak.agents.base import AgentResult, BaseAgent


class AlignmentAgent(BaseAgent):
    """Aligns fields and entities across two SRRs."""

    def __init__(self, **kwargs: Any):
        super().__init__(name="alignment", **kwargs)

    def run(
        self,
        original_srr: dict[str, Any] | None = None,
        candidate_srr: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        start = time.time()
        original_srr = original_srr or {}
        candidate_srr = candidate_srr or {}

        aligned_fields = []
        unmatched_original = []
        unmatched_candidate = []

        # Field-level alignment
        for field in [
            "actor", "action", "object", "modality", "polarity",
            "conditions", "temporal_constraints", "numerical_constraints",
            "ordering_constraints", "exceptions", "safety_constraints",
            "security_constraints", "units",
        ]:
            orig_val = original_srr.get(field)
            cand_val = candidate_srr.get(field)

            if orig_val is not None and cand_val is not None:
                aligned_fields.append({
                    "field": field,
                    "original": orig_val,
                    "candidate": cand_val,
                    "aligned": True,
                })
            elif orig_val is not None:
                unmatched_original.append(field)
            elif cand_val is not None:
                unmatched_candidate.append(field)

        latency = time.time() - start
        return self._make_result(
            output={
                "aligned_fields": aligned_fields,
                "unmatched_original": unmatched_original,
                "unmatched_candidate": unmatched_candidate,
                "alignment_score": len(aligned_fields) / max(
                    len(aligned_fields) + len(unmatched_original) + len(unmatched_candidate), 1
                ),
            },
            latency=latency,
        )
