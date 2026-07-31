"""Deterministic SRR field-level comparison engine."""

from __future__ import annotations

from idraak.drift.unit_converter import UnitConverter
from idraak.schemas.drift import DriftSeverity, DriftType, FieldDifference
from idraak.schemas.srr import (
    ComparisonOperator,
    Modality,
    NumericalConstraint,
    Polarity,
    SemanticRequirement,
    TemporalConstraint,
)
from idraak.utils.logging import get_logger

logger = get_logger("comparator")

# Severity mappings for modality changes
_MODALITY_SEVERITY = {
    (Modality.MANDATORY, Modality.RECOMMENDED): DriftSeverity.HIGH,
    (Modality.MANDATORY, Modality.PERMITTED): DriftSeverity.CRITICAL,
    (Modality.MANDATORY, Modality.OPTIONAL): DriftSeverity.CRITICAL,
    (Modality.MANDATORY, Modality.FORBIDDEN): DriftSeverity.CRITICAL,
    (Modality.RECOMMENDED, Modality.PERMITTED): DriftSeverity.MEDIUM,
    (Modality.RECOMMENDED, Modality.MANDATORY): DriftSeverity.LOW,
    (Modality.FORBIDDEN, Modality.MANDATORY): DriftSeverity.CRITICAL,
    (Modality.FORBIDDEN, Modality.RECOMMENDED): DriftSeverity.CRITICAL,
    (Modality.FORBIDDEN, Modality.PERMITTED): DriftSeverity.CRITICAL,
}


class SRRComparator:
    """Deterministic field-level SRR comparison.

    Compares two SemanticRequirement objects and returns a list of
    FieldDifference objects with drift type, severity, and confidence.
    """

    def __init__(self, unit_conversion: bool = True, synonym_tolerance: bool = True):
        self._unit_conv = unit_conversion
        self._synonym_tol = synonym_tolerance
        self._converter = UnitConverter()

    def compare(
        self, original: SemanticRequirement, candidate: SemanticRequirement
    ) -> list[FieldDifference]:
        """Compare two SRRs and return field-level differences."""
        diffs: list[FieldDifference] = []

        diffs.extend(self._compare_modality(original, candidate))
        diffs.extend(self._compare_polarity(original, candidate))
        diffs.extend(self._compare_scalar("actor", original.actor, candidate.actor))
        diffs.extend(self._compare_scalar("action", original.action, candidate.action))
        diffs.extend(self._compare_scalar("object", original.object, candidate.object))
        diffs.extend(self._compare_numerical(original, candidate))
        diffs.extend(self._compare_temporal(original, candidate))
        diffs.extend(self._compare_conditions(original, candidate))
        diffs.extend(self._compare_exceptions(original, candidate))
        diffs.extend(self._compare_ordering(original, candidate))
        diffs.extend(self._compare_list("safety_constraints", original.safety_constraints, candidate.safety_constraints))
        diffs.extend(self._compare_list("security_constraints", original.security_constraints, candidate.security_constraints))

        return diffs

    def _compare_modality(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        if orig.modality == cand.modality:
            return []
        if orig.modality is None or cand.modality is None:
            sev = DriftSeverity.MEDIUM
        else:
            sev = _MODALITY_SEVERITY.get(
                (orig.modality, cand.modality), DriftSeverity.HIGH
            )
        return [FieldDifference(
            field="modality",
            original=orig.modality.value if orig.modality else None,
            candidate=cand.modality.value if cand.modality else None,
            difference_type="modality_change",
            drift_type=DriftType.MODALITY,
            severity=sev,
            confidence=0.95,
            explanation=f"Modality changed from '{orig.modality}' to '{cand.modality}'",
        )]

    def _compare_polarity(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        if orig.polarity == cand.polarity:
            return []
        return [FieldDifference(
            field="polarity",
            original=orig.polarity.value if orig.polarity else None,
            candidate=cand.polarity.value if cand.polarity else None,
            difference_type="polarity_change",
            drift_type=DriftType.POLARITY,
            severity=DriftSeverity.CRITICAL,
            confidence=0.98,
            explanation=f"Polarity changed from '{orig.polarity}' to '{cand.polarity}'",
        )]

    def _compare_scalar(
        self, field: str, orig: str | None, cand: str | None
    ) -> list[FieldDifference]:
        if orig is None and cand is None:
            return []
        if orig is None or cand is None:
            return [FieldDifference(
                field=field,
                original=orig,
                candidate=cand,
                difference_type="missing_field",
                severity=DriftSeverity.MEDIUM,
                confidence=0.7,
            )]
        if orig.lower().strip() == cand.lower().strip():
            return []
        # Check synonym tolerance
        if self._synonym_tol and self._are_synonyms(orig, cand):
            return []
        return [FieldDifference(
            field=field,
            original=orig,
            candidate=cand,
            difference_type="value_change",
            drift_type=DriftType.ENTITY,
            severity=DriftSeverity.MEDIUM,
            confidence=0.6,
        )]

    def _compare_numerical(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        diffs = []
        # Match constraints by parameter or position
        for i, oc in enumerate(orig.numerical_constraints):
            if i < len(cand.numerical_constraints):
                cc = cand.numerical_constraints[i]
                diffs.extend(self._compare_one_numerical(f"numerical_constraints[{i}]", oc, cc))
            else:
                diffs.append(FieldDifference(
                    field=f"numerical_constraints[{i}]",
                    original=oc.model_dump(),
                    candidate=None,
                    difference_type="missing_constraint",
                    drift_type=DriftType.OMISSION,
                    severity=DriftSeverity.HIGH,
                    confidence=0.9,
                ))
        # Extra constraints in candidate
        for i in range(len(orig.numerical_constraints), len(cand.numerical_constraints)):
            cc = cand.numerical_constraints[i]
            diffs.append(FieldDifference(
                field=f"numerical_constraints[{i}]",
                original=None,
                candidate=cc.model_dump(),
                difference_type="added_constraint",
                drift_type=DriftType.ADDITION,
                severity=DriftSeverity.MEDIUM,
                confidence=0.8,
            ))
        return diffs

    def _compare_one_numerical(
        self, prefix: str, oc: NumericalConstraint, cc: NumericalConstraint
    ) -> list[FieldDifference]:
        diffs = []

        # Compare operator
        if oc.operator != cc.operator:
            diffs.append(FieldDifference(
                field=f"{prefix}.operator",
                original=oc.operator.value,
                candidate=cc.operator.value,
                difference_type="operator_change",
                drift_type=DriftType.THRESHOLD,
                severity=DriftSeverity.HIGH,
                confidence=0.95,
            ))

        # Compare value with unit conversion awareness
        values_equal = False
        if oc.unit and cc.unit and self._unit_conv:
            values_equal = self._converter.are_equivalent(
                oc.value, oc.unit, cc.value, cc.unit
            )
        else:
            values_equal = abs(oc.value - cc.value) < 1e-6

        if not values_equal:
            # Determine if it's a unit drift or numerical drift
            if oc.unit != cc.unit and oc.value == cc.value:
                dtype = DriftType.UNIT
            else:
                dtype = DriftType.NUMERICAL
            diffs.append(FieldDifference(
                field=f"{prefix}.value",
                original=oc.value,
                candidate=cc.value,
                difference_type="value_change",
                drift_type=dtype,
                severity=DriftSeverity.HIGH,
                confidence=0.95,
            ))

        # Compare units if values weren't equivalent through conversion
        if oc.unit != cc.unit and not values_equal:
            diffs.append(FieldDifference(
                field=f"{prefix}.unit",
                original=oc.unit,
                candidate=cc.unit,
                difference_type="unit_change",
                drift_type=DriftType.UNIT,
                severity=DriftSeverity.HIGH,
                confidence=0.95,
            ))

        return diffs

    def _compare_temporal(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        diffs = []
        for i, ot in enumerate(orig.temporal_constraints):
            if i < len(cand.temporal_constraints):
                ct = cand.temporal_constraints[i]
                if ot.relation != ct.relation:
                    diffs.append(FieldDifference(
                        field=f"temporal_constraints[{i}].relation",
                        original=ot.relation.value,
                        candidate=ct.relation.value,
                        difference_type="temporal_change",
                        drift_type=DriftType.TEMPORAL,
                        severity=DriftSeverity.HIGH,
                        confidence=0.9,
                    ))
                if ot.value is not None and ct.value is not None:
                    if ot.unit and ct.unit and self._unit_conv:
                        if not self._converter.are_equivalent(ot.value, ot.unit, ct.value, ct.unit):
                            diffs.append(FieldDifference(
                                field=f"temporal_constraints[{i}].value",
                                original=ot.value,
                                candidate=ct.value,
                                difference_type="value_change",
                                drift_type=DriftType.NUMERICAL,
                                severity=DriftSeverity.HIGH,
                                confidence=0.9,
                            ))
                    elif abs(ot.value - ct.value) > 1e-6:
                        diffs.append(FieldDifference(
                            field=f"temporal_constraints[{i}].value",
                            original=ot.value,
                            candidate=ct.value,
                            difference_type="value_change",
                            drift_type=DriftType.NUMERICAL,
                            severity=DriftSeverity.HIGH,
                            confidence=0.9,
                        ))
            else:
                diffs.append(FieldDifference(
                    field=f"temporal_constraints[{i}]",
                    original=ot.model_dump(),
                    candidate=None,
                    difference_type="missing_constraint",
                    drift_type=DriftType.OMISSION,
                    severity=DriftSeverity.HIGH,
                    confidence=0.85,
                ))
        return diffs

    def _compare_conditions(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        diffs = []
        for i, oc in enumerate(orig.conditions):
            if i < len(cand.conditions):
                cc = cand.conditions[i]
                if oc.condition_type != cc.condition_type:
                    diffs.append(FieldDifference(
                        field=f"conditions[{i}].condition_type",
                        original=oc.condition_type,
                        candidate=cc.condition_type,
                        difference_type="condition_type_change",
                        drift_type=DriftType.CONDITION,
                        severity=DriftSeverity.HIGH,
                        confidence=0.85,
                    ))
                if oc.negated != cc.negated:
                    diffs.append(FieldDifference(
                        field=f"conditions[{i}].negated",
                        original=oc.negated,
                        candidate=cc.negated,
                        difference_type="negation_change",
                        drift_type=DriftType.POLARITY,
                        severity=DriftSeverity.CRITICAL,
                        confidence=0.9,
                    ))
            else:
                diffs.append(FieldDifference(
                    field=f"conditions[{i}]",
                    original=oc.model_dump(),
                    candidate=None,
                    difference_type="missing_condition",
                    drift_type=DriftType.OMISSION,
                    severity=DriftSeverity.HIGH,
                    confidence=0.85,
                ))
        return diffs

    def _compare_exceptions(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        diffs = []
        for i, oe in enumerate(orig.exceptions):
            if i < len(cand.exceptions):
                ce = cand.exceptions[i]
                if oe.condition.lower() != ce.condition.lower():
                    diffs.append(FieldDifference(
                        field=f"exceptions[{i}].condition",
                        original=oe.condition,
                        candidate=ce.condition,
                        difference_type="exception_change",
                        drift_type=DriftType.EXCEPTION,
                        severity=DriftSeverity.HIGH,
                        confidence=0.85,
                    ))
            else:
                diffs.append(FieldDifference(
                    field=f"exceptions[{i}]",
                    original=oe.model_dump(),
                    candidate=None,
                    difference_type="missing_exception",
                    drift_type=DriftType.EXCEPTION,
                    severity=DriftSeverity.HIGH,
                    confidence=0.85,
                ))
        return diffs

    def _compare_ordering(
        self, orig: SemanticRequirement, cand: SemanticRequirement
    ) -> list[FieldDifference]:
        diffs = []
        for i, oo in enumerate(orig.ordering_constraints):
            if i < len(cand.ordering_constraints):
                co = cand.ordering_constraints[i]
                if oo.first.lower() != co.first.lower() or oo.second.lower() != co.second.lower():
                    # Check if order is simply swapped
                    if oo.first.lower() == co.second.lower() and oo.second.lower() == co.first.lower():
                        diffs.append(FieldDifference(
                            field=f"ordering_constraints[{i}]",
                            original=f"{oo.first} before {oo.second}",
                            candidate=f"{co.first} before {co.second}",
                            difference_type="order_reversal",
                            drift_type=DriftType.RELATION,
                            severity=DriftSeverity.HIGH,
                            confidence=0.9,
                        ))
            else:
                diffs.append(FieldDifference(
                    field=f"ordering_constraints[{i}]",
                    original=oo.model_dump(),
                    candidate=None,
                    difference_type="missing_ordering",
                    drift_type=DriftType.OMISSION,
                    severity=DriftSeverity.MEDIUM,
                    confidence=0.8,
                ))
        return diffs

    def _compare_list(
        self, field: str, orig: list[str], cand: list[str]
    ) -> list[FieldDifference]:
        orig_set = {s.lower().strip() for s in orig}
        cand_set = {s.lower().strip() for s in cand}
        diffs = []
        missing = orig_set - cand_set
        added = cand_set - orig_set
        if missing:
            diffs.append(FieldDifference(
                field=field,
                original=list(missing),
                candidate=None,
                difference_type="missing_items",
                drift_type=DriftType.OMISSION,
                severity=DriftSeverity.HIGH if "safety" in field else DriftSeverity.MEDIUM,
                confidence=0.8,
            ))
        if added:
            diffs.append(FieldDifference(
                field=field,
                original=None,
                candidate=list(added),
                difference_type="added_items",
                drift_type=DriftType.ADDITION,
                severity=DriftSeverity.LOW,
                confidence=0.7,
            ))
        return diffs

    def _are_synonyms(self, a: str, b: str) -> bool:
        """Basic synonym check with article stripping."""
        synonyms = [
            {"assert", "set", "activate", "enable"},
            {"deassert", "clear", "deactivate", "disable"},
            {"controller", "control unit", "control module"},
            {"buffer", "queue", "fifo"},
            {"register", "reg"},
            {"initialize", "init", "start up"},
            {"process", "handle", "execute"},
        ]
        a_low = a.lower().strip()
        b_low = b.lower().strip()
        # Strip common articles
        for prefix in ["the ", "a ", "an "]:
            if a_low.startswith(prefix):
                a_low = a_low[len(prefix):]
            if b_low.startswith(prefix):
                b_low = b_low[len(prefix):]
        for group in synonyms:
            if a_low in group and b_low in group:
                return True
        return False
