"""Deterministic (regex-based) SRR extraction for reliable fields."""

from __future__ import annotations

import re
from typing import Optional

from idraak.schemas.srr import (
    ComparisonOperator,
    Condition,
    ExceptionClause,
    Modality,
    NumericalConstraint,
    OrderingConstraint,
    Polarity,
    SemanticRequirement,
    TemporalConstraint,
    TemporalRelation,
)
from idraak.utils.logging import get_logger

logger = get_logger("deterministic_extractor")

# ── Number word mapping ───────────────────────────────────────────────────────

_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "hundred": 100, "thousand": 1000,
}

# ── Unit normalization ────────────────────────────────────────────────────────

_UNIT_ALIASES = {
    "milliseconds": "ms", "millisecond": "ms", "msec": "ms",
    "microseconds": "us", "microsecond": "us", "usec": "us",
    "nanoseconds": "ns", "nanosecond": "ns",
    "picoseconds": "ps", "picosecond": "ps",
    "seconds": "s", "second": "s", "sec": "s",
    "minutes": "min", "minute": "min",
    "hours": "h", "hour": "h",
    "clock cycles": "clock_cycle", "clock cycle": "clock_cycle",
    "cycles": "cycle", "cycle": "cycle",
    "megahertz": "MHz", "gigahertz": "GHz", "kilohertz": "kHz", "hertz": "Hz",
    "kilobytes": "KB", "megabytes": "MB", "gigabytes": "GB", "terabytes": "TB",
    "bytes": "byte", "byte": "byte", "bits": "bit", "bit": "bit",
    "milliamps": "mA", "milliamp": "mA", "amps": "A",
    "milliwatts": "mW", "watts": "W",
    "percent": "%", "percentage": "%",
}


def normalize_unit(raw: str) -> str:
    """Normalize a unit string to canonical form."""
    return _UNIT_ALIASES.get(raw.lower().strip(), raw.strip())


def parse_number(s: str) -> Optional[float]:
    """Parse a number from a string, including word forms."""
    s = s.strip().lower().replace(",", "")
    if s in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[s])
    try:
        return float(s)
    except ValueError:
        return None


class DeterministicExtractor:
    """Regex-based extraction for high-confidence fields.

    Extracts modality, polarity, numerical constraints, temporal relations,
    conditions, and exceptions using pattern matching.
    """

    def extract(
        self, text: str, language: str = "en", requirement_id: str = ""
    ) -> SemanticRequirement:
        text_lower = text.lower()

        return SemanticRequirement(
            requirement_id=requirement_id,
            modality=self._extract_modality(text_lower),
            polarity=self._extract_polarity(text_lower),
            actor=self._extract_actor(text),
            action=self._extract_action(text),
            conditions=self._extract_conditions(text),
            temporal_constraints=self._extract_temporal(text),
            numerical_constraints=self._extract_numerical(text),
            ordering_constraints=self._extract_ordering(text),
            exceptions=self._extract_exceptions(text),
            units=self._extract_units(text),
            source_language=language,
            normalized_text=text.strip(),
            raw_text=text,
            extraction_confidence=0.6,
            extraction_method="deterministic",
        )

    def _extract_modality(self, text: str) -> Optional[Modality]:
        if "shall not" in text or "must not" in text:
            return Modality.FORBIDDEN
        if "shall" in text or "must" in text:
            return Modality.MANDATORY
        if "should" in text:
            return Modality.RECOMMENDED
        if "may" in text or "can" in text:
            return Modality.PERMITTED
        return None

    def _extract_polarity(self, text: str) -> Optional[Polarity]:
        neg_patterns = [
            r"\bshall not\b", r"\bmust not\b", r"\bshall never\b",
            r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bnor\b",
            r"\bunder no circumstances\b",
        ]
        for pat in neg_patterns:
            if re.search(pat, text):
                return Polarity.NEGATIVE
        return Polarity.POSITIVE

    def _extract_actor(self, text: str) -> Optional[str]:
        # "The X shall ..." pattern
        m = re.match(r"^(The\s+[\w\s]+?)\s+(?:shall|must|should|may|will)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # "When ..., the X shall ..." pattern
        m = re.search(r",\s*(the\s+[\w\s]+?)\s+(?:shall|must|should|may)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _extract_action(self, text: str) -> Optional[str]:
        m = re.search(
            r"(?:shall|must|should|may|will)\s+(?:not\s+)?([\w\s]+?)(?:\s+the\b|\s+within\b|\s+before\b|\s+after\b|\s+when\b|\s+if\b|\s+unless\b|\s*[,.])",
            text, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return None

    def _extract_conditions(self, text: str) -> list[Condition]:
        conditions = []
        # "If X" patterns
        for m in re.finditer(r"\b(?:if|when|provided that)\s+(.+?)(?:,|\bthen\b)", text, re.IGNORECASE):
            conditions.append(Condition(
                condition_type="if",
                trigger=m.group(1).strip(),
                raw_text=m.group(0).strip(),
            ))
        # "only if X"
        for m in re.finditer(r"\bonly if\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE):
            conditions.append(Condition(
                condition_type="only_if",
                trigger=m.group(1).strip(),
                raw_text=m.group(0).strip(),
            ))
        # "if and only if"
        for m in re.finditer(r"\bif and only if\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE):
            conditions.append(Condition(
                condition_type="if_and_only_if",
                trigger=m.group(1).strip(),
                raw_text=m.group(0).strip(),
            ))
        return conditions

    def _extract_temporal(self, text: str) -> list[TemporalConstraint]:
        constraints = []
        # "within N units"
        for m in re.finditer(r"\bwithin\s+(\w+)\s+([\w\s]+?)(?:\s+(?:of|after|before|from)\b|\.|,|$)", text, re.IGNORECASE):
            val = parse_number(m.group(1))
            if val is not None:
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.WITHIN,
                    value=val,
                    unit=normalize_unit(m.group(2).strip()),
                    raw_text=m.group(0).strip(),
                ))
        # "before/after X"
        for m in re.finditer(r"\b(before|after)\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE):
            rel = TemporalRelation.BEFORE if m.group(1).lower() == "before" else TemporalRelation.AFTER
            constraints.append(TemporalConstraint(
                relation=rel,
                reference_event=m.group(2).strip(),
                raw_text=m.group(0).strip(),
            ))
        # "for at least N units"
        for m in re.finditer(r"\bfor at least\s+(\w+)\s+([\w\s]+?)(?:\s+before\b|\.|,|$)", text, re.IGNORECASE):
            val = parse_number(m.group(1))
            if val is not None:
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.WITHIN,
                    value=val,
                    unit=normalize_unit(m.group(2).strip()),
                    raw_text=m.group(0).strip(),
                ))
        return constraints

    def _extract_numerical(self, text: str) -> list[NumericalConstraint]:
        constraints = []
        # "at least N units"
        for m in re.finditer(r"\bat least\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                value=float(m.group(1)),
                unit=normalize_unit(m.group(2)),
                raw_text=m.group(0),
            ))
        # "no more than / at most N units"
        for m in re.finditer(r"\b(?:no more than|at most)\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                value=float(m.group(1)),
                unit=normalize_unit(m.group(2)),
                raw_text=m.group(0),
            ))
        # "not exceed N units"
        for m in re.finditer(r"\bnot exceed\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.LESS_THAN_OR_EQUAL,
                value=float(m.group(1)),
                unit=normalize_unit(m.group(2)),
                raw_text=m.group(0),
            ))
        # "exactly N units"
        for m in re.finditer(r"\bexactly\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.EQUAL,
                value=float(m.group(1)),
                unit=normalize_unit(m.group(2)),
                raw_text=m.group(0),
            ))
        # "between N and M units"
        for m in re.finditer(r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.BETWEEN,
                value=float(m.group(1)),
                value_upper=float(m.group(2)),
                unit=normalize_unit(m.group(3)),
                raw_text=m.group(0),
            ))
        # "less than N units"
        for m in re.finditer(r"\bless than\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.LESS_THAN,
                value=float(m.group(1)),
                unit=normalize_unit(m.group(2)),
                raw_text=m.group(0),
            ))
        # "greater than N units"
        for m in re.finditer(r"\bgreater than\s+(\d+(?:\.\d+)?)\s*([\w/%]+)", text, re.IGNORECASE):
            constraints.append(NumericalConstraint(
                parameter="value",
                operator=ComparisonOperator.GREATER_THAN,
                value=float(m.group(1)),
                unit=normalize_unit(m.group(2)),
                raw_text=m.group(0),
            ))
        # Generic "N units" with preceding value word
        if not constraints:
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s+([\w]+(?:\s+[\w]+)?)\b", text):
                val = float(m.group(1))
                raw_unit = m.group(2).strip()
                normalized = normalize_unit(raw_unit)
                if normalized != raw_unit or raw_unit.lower() in _UNIT_ALIASES:
                    constraints.append(NumericalConstraint(
                        parameter="value",
                        operator=ComparisonOperator.EQUAL,
                        value=val,
                        unit=normalized,
                        raw_text=m.group(0),
                    ))
        return constraints

    def _extract_ordering(self, text: str) -> list[OrderingConstraint]:
        constraints = []
        # "X before Y"
        for m in re.finditer(r"(\w+(?:\s+\w+)?)\s+before\s+(\w+(?:\s+\w+)?)", text, re.IGNORECASE):
            constraints.append(OrderingConstraint(
                first=m.group(1).strip(),
                second=m.group(2).strip(),
                ordering_type="before",
                raw_text=m.group(0),
            ))
        # "first X then Y"
        for m in re.finditer(r"\bfirst\s+(\w+)\s+.*?\bthen\s+(\w+)", text, re.IGNORECASE):
            constraints.append(OrderingConstraint(
                first=m.group(1).strip(),
                second=m.group(2).strip(),
                ordering_type="before",
                raw_text=m.group(0),
            ))
        return constraints

    def _extract_exceptions(self, text: str) -> list[ExceptionClause]:
        exceptions = []
        for m in re.finditer(r"\b(unless|except when|except)\s+(.+?)(?:\.|,|$)", text, re.IGNORECASE):
            exceptions.append(ExceptionClause(
                exception_type="unless" if "unless" in m.group(1).lower() else "except",
                condition=m.group(2).strip(),
                raw_text=m.group(0).strip(),
            ))
        return exceptions

    def _extract_units(self, text: str) -> list[str]:
        found = []
        for alias in _UNIT_ALIASES:
            if alias.lower() in text.lower():
                normalized = _UNIT_ALIASES[alias]
                if normalized not in found:
                    found.append(normalized)
        return found
