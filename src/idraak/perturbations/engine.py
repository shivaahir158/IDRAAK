"""Controlled perturbation engine for generating labeled drift examples."""

from __future__ import annotations

import random
import re
from typing import Any

from idraak.schemas.dataset import DatasetEntry, PerturbationRecord
from idraak.utils.logging import get_logger

logger = get_logger("perturbation")


class PerturbationEngine:
    """Generates controlled perturbations with known drift types and labels."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_all(
        self, entry: DatasetEntry, max_perturbations: int = 5
    ) -> list[PerturbationRecord]:
        """Generate multiple perturbations for a single requirement."""
        perturbations: list[PerturbationRecord] = []
        text = entry.original_text

        # Always try to generate a semantically equivalent paraphrase
        para = self._paraphrase(entry)
        if para:
            perturbations.append(para)

        # Generate drift perturbations
        methods = [
            self._numerical_drift,
            self._unit_drift,
            self._polarity_drift,
            self._modality_drift,
            self._condition_drift,
            self._temporal_drift,
            self._threshold_drift,
            self._omission_drift,
            self._scope_drift,
            self._entity_drift,
            self._terminology_drift,
        ]

        self.rng.shuffle(methods)
        for method in methods:
            if len(perturbations) >= max_perturbations:
                break
            result = method(entry)
            if result and result.perturbed_text != text:
                perturbations.append(result)

        return perturbations

    def _make_id(self, base_id: str, suffix: str) -> str:
        return f"{base_id}-PERT-{suffix}"

    # ── Semantically equivalent paraphrase ───────────────────────────────────

    def _paraphrase(self, entry: DatasetEntry) -> PerturbationRecord | None:
        """Generate a semantically equivalent paraphrase (no drift)."""
        text = entry.original_text
        paraphrased = text

        # Simple voice/structure changes
        transforms = [
            (r"\bshall\b", "must"),
            (r"\bwithin\b", "in under"),
            (r"\bat least\b", "no less than"),
            (r"\bno more than\b", "at most"),
        ]
        chosen = self.rng.choice(transforms)
        paraphrased = re.sub(chosen[0], chosen[1], paraphrased, count=1)

        if paraphrased == text:
            # Try clause reordering
            parts = text.split(",", 1)
            if len(parts) == 2:
                paraphrased = parts[1].strip().rstrip(".") + ", " + parts[0].strip().lower() + "."

        if paraphrased == text:
            return None

        return PerturbationRecord(
            requirement_id=self._make_id(entry.requirement_id, "PARA"),
            base_requirement_id=entry.requirement_id,
            perturbed_text=paraphrased,
            drift_label=0,
            drift_type=None,
            severity=None,
            perturbation_method="paraphrase",
            description="Semantically equivalent paraphrase",
        )

    # ── Numerical drift ──────────────────────────────────────────────────────

    def _numerical_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
        if not numbers:
            return None

        target = self.rng.choice(numbers)
        original_val = float(target)
        # Perturb by a factor
        factor = self.rng.choice([2, 5, 10, 0.5, 0.1])
        new_val = original_val * factor
        if new_val == int(new_val):
            new_str = str(int(new_val))
        else:
            new_str = f"{new_val:.1f}"

        perturbed = text.replace(target, new_str, 1)
        if perturbed == text:
            return None

        severity = "high" if factor >= 5 or factor <= 0.2 else "medium"

        return PerturbationRecord(
            requirement_id=self._make_id(entry.requirement_id, "NUM"),
            base_requirement_id=entry.requirement_id,
            perturbed_text=perturbed,
            drift_label=1,
            drift_type="numerical_drift",
            severity=severity,
            changed_fields=["numerical_constraints.value"],
            original_value=original_val,
            modified_value=new_val,
            perturbation_method="numerical_drift",
            description=f"Changed {original_val} to {new_val}",
        )

    # ── Unit drift ────────────────────────────────────────────────────────────

    def _unit_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        unit_swaps = {
            "ms": "seconds", "seconds": "ms",
            "MHz": "GHz", "GHz": "MHz",
            "ns": "us", "us": "ns",
            "KB": "MB", "MB": "KB",
            "Mbps": "Gbps", "Gbps": "Mbps",
            "mA": "A", "mW": "W",
            "bytes": "bits", "bits": "bytes",
        }
        for old, new in unit_swaps.items():
            if old in text:
                perturbed = text.replace(old, new, 1)
                return PerturbationRecord(
                    requirement_id=self._make_id(entry.requirement_id, "UNIT"),
                    base_requirement_id=entry.requirement_id,
                    perturbed_text=perturbed,
                    drift_label=1,
                    drift_type="unit_drift",
                    severity="high",
                    changed_fields=["units"],
                    original_value=old,
                    modified_value=new,
                    perturbation_method="unit_drift",
                    description=f"Changed unit from {old} to {new}",
                )
        return None

    # ── Polarity drift ───────────────────────────────────────────────────────

    def _polarity_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            (r"\bshall not\b", "shall"),
            (r"\bshall\b", "shall not"),
            (r"\bmust not\b", "must"),
            (r"\benabled\b", "disabled"),
            (r"\bdisabled\b", "enabled"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, text):
                perturbed = re.sub(pattern, replacement, text, count=1)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "POL"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="polarity_drift",
                        severity="critical",
                        changed_fields=["polarity"],
                        original_value=re.search(pattern, text).group(),  # type: ignore
                        modified_value=replacement,
                        perturbation_method="polarity_drift",
                        description=f"Flipped polarity",
                    )
        return None

    # ── Modality drift ───────────────────────────────────────────────────────

    def _modality_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            (r"\bshall\b", "should"),
            (r"\bmust\b", "may"),
            (r"\bshould\b", "may"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, text, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, text, count=1)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "MOD"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="modality_drift",
                        severity="high",
                        changed_fields=["modality"],
                        original_value=re.search(pattern, text, re.IGNORECASE).group(),  # type: ignore
                        modified_value=replacement,
                        perturbation_method="modality_drift",
                        description=f"Weakened modality to '{replacement}'",
                    )
        return None

    # ── Condition drift ──────────────────────────────────────────────────────

    def _condition_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            (r"\band\b", "or"),
            (r"\bonly if\b", "if"),
            (r"\bif and only if\b", "if"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, text, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "COND"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="condition_drift",
                        severity="high",
                        changed_fields=["conditions"],
                        original_value=re.search(pattern, text, re.IGNORECASE).group(),  # type: ignore
                        modified_value=replacement,
                        perturbation_method="condition_drift",
                        description=f"Changed logical condition",
                    )
        return None

    # ── Temporal drift ───────────────────────────────────────────────────────

    def _temporal_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            (r"\bbefore\b", "after"),
            (r"\bafter\b", "before"),
            (r"\bwithin\b", "after"),
            (r"\bprecede\b", "follow"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, text, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "TEMP"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="temporal_drift",
                        severity="high",
                        changed_fields=["temporal_constraints"],
                        original_value=re.search(pattern, text, re.IGNORECASE).group(),  # type: ignore
                        modified_value=replacement,
                        perturbation_method="temporal_drift",
                        description=f"Changed temporal relation",
                    )
        return None

    # ── Threshold drift ──────────────────────────────────────────────────────

    def _threshold_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            (r"\bat least\b", "at most"),
            (r"\bat most\b", "at least"),
            (r"\bno more than\b", "no less than"),
            (r"\bno less than\b", "no more than"),
            (r"\bexceed\b", "fall below"),
            (r"\bbelow\b", "above"),
            (r"\bless than\b", "greater than"),
            (r"\bgreater than\b", "less than"),
            (r"\bmaximum\b", "minimum"),
            (r"\bminimum\b", "maximum"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, text, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "THR"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="threshold_drift",
                        severity="high",
                        changed_fields=["numerical_constraints.operator"],
                        original_value=re.search(pattern, text, re.IGNORECASE).group(),  # type: ignore
                        modified_value=replacement,
                        perturbation_method="threshold_drift",
                        description=f"Reversed threshold direction",
                    )
        return None

    # ── Omission drift ───────────────────────────────────────────────────────

    def _omission_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        # Remove a clause after comma or semicolon
        parts = [p.strip() for p in text.split(",")]
        if len(parts) >= 2:
            removed = parts.pop(self.rng.randint(1, len(parts) - 1))
            perturbed = ", ".join(parts)
            if not perturbed.endswith("."):
                perturbed = perturbed.rstrip(",;") + "."
            return PerturbationRecord(
                requirement_id=self._make_id(entry.requirement_id, "OMIT"),
                base_requirement_id=entry.requirement_id,
                perturbed_text=perturbed,
                drift_label=1,
                drift_type="omission_drift",
                severity="medium",
                changed_fields=["clauses"],
                original_value=removed,
                modified_value=None,
                perturbation_method="omission_drift",
                description=f"Removed clause: '{removed}'",
            )
        return None

    # ── Scope drift ──────────────────────────────────────────────────────────

    def _scope_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            (r"\ball\b", "some"),
            (r"\bevery\b", "some"),
            (r"\beach\b", "a"),
            (r"\balways\b", "sometimes"),
        ]
        for pattern, replacement in swaps:
            if re.search(pattern, text, re.IGNORECASE):
                perturbed = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "SCOPE"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="scope_drift",
                        severity="high",
                        changed_fields=["qualifiers"],
                        original_value=re.search(pattern, text, re.IGNORECASE).group(),  # type: ignore
                        modified_value=replacement,
                        perturbation_method="scope_drift",
                        description=f"Narrowed scope from '{pattern}' to '{replacement}'",
                    )
        return None

    # ── Entity drift ─────────────────────────────────────────────────────────

    def _entity_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            ("transmit", "receive"), ("receive", "transmit"),
            ("input", "output"), ("output", "input"),
            ("read", "write"), ("write", "read"),
            ("source", "destination"), ("destination", "source"),
            ("master", "slave"), ("request", "response"),
        ]
        for old, new in swaps:
            if old in text.lower():
                perturbed = re.sub(rf"\b{old}\b", new, text, count=1, flags=re.IGNORECASE)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "ENT"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="entity_drift",
                        severity="high",
                        changed_fields=["entities"],
                        original_value=old,
                        modified_value=new,
                        perturbation_method="entity_drift",
                        description=f"Swapped entity '{old}' with '{new}'",
                    )
        return None

    # ── Terminology drift ────────────────────────────────────────────────────

    def _terminology_drift(self, entry: DatasetEntry) -> PerturbationRecord | None:
        text = entry.original_text
        swaps = [
            ("latency", "bandwidth"), ("throughput", "latency"),
            ("buffer", "cache"), ("register", "variable"),
            ("interrupt", "exception"), ("assertion", "assignment"),
            ("handshake", "polling"), ("overflow", "underflow"),
        ]
        for old, new in swaps:
            if old in text.lower():
                perturbed = re.sub(rf"\b{old}\b", new, text, count=1, flags=re.IGNORECASE)
                if perturbed != text:
                    return PerturbationRecord(
                        requirement_id=self._make_id(entry.requirement_id, "TERM"),
                        base_requirement_id=entry.requirement_id,
                        perturbed_text=perturbed,
                        drift_label=1,
                        drift_type="terminology_drift",
                        severity="medium",
                        changed_fields=["terminology"],
                        original_value=old,
                        modified_value=new,
                        perturbation_method="terminology_drift",
                        description=f"Replaced term '{old}' with misleading '{new}'",
                    )
        return None
