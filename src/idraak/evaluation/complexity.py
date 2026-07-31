"""Requirement complexity scoring for stratified analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

from idraak.utils.logging import get_logger

logger = get_logger("complexity")


@dataclass
class ComplexityScore:
    """Detailed complexity breakdown for a requirement."""

    token_count: int = 0
    clause_count: int = 0
    condition_count: int = 0
    numerical_count: int = 0
    nesting_depth: int = 0
    entity_count: int = 0
    logical_operator_count: int = 0
    temporal_relation_count: int = 0
    domain_term_density: float = 0.0
    overall_score: float = 0.0
    complexity_level: str = "medium"  # easy, medium, hard

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "token_count": self.token_count,
            "clause_count": self.clause_count,
            "condition_count": self.condition_count,
            "numerical_count": self.numerical_count,
            "nesting_depth": self.nesting_depth,
            "entity_count": self.entity_count,
            "logical_operator_count": self.logical_operator_count,
            "temporal_relation_count": self.temporal_relation_count,
            "domain_term_density": self.domain_term_density,
            "overall_score": self.overall_score,
            "complexity_level": self.complexity_level,
        }


# Domain-specific terms for density calculation
_DOMAIN_TERMS = {
    "clock", "cycle", "register", "buffer", "interrupt", "assert", "deassert",
    "handshake", "latency", "throughput", "FIFO", "FSM", "pipeline", "arbiter",
    "timeout", "watchdog", "scheduler", "mutex", "semaphore", "deadlock",
    "overflow", "underflow", "checksum", "CRC", "parity", "redundancy",
    "failsafe", "interlock", "heartbeat", "acknowledge", "request", "grant",
    "encryption", "authentication", "certificate", "token", "session",
    "calibration", "threshold", "hysteresis", "ramp", "dwell", "setpoint",
    "protocol", "interface", "bus", "port", "endpoint", "socket", "packet",
    "firmware", "bootloader", "RTOS", "driver", "kernel", "daemon",
    "transaction", "settlement", "reconciliation", "ledger", "audit",
    "infusion", "dosage", "ventilator", "sensor", "actuator", "PLC", "SCADA",
}


class ComplexityScorer:
    """Compute complexity scores for requirements."""

    def score(self, text: str) -> ComplexityScore:
        """Compute a multi-dimensional complexity score."""
        tokens = text.split()
        token_count = len(tokens)
        clause_count = text.count(",") + text.count(";") + text.count(" and ") + 1

        condition_keywords = ["if", "when", "unless", "only if", "provided that",
                              "except", "in case", "whenever"]
        condition_count = sum(
            1 for kw in condition_keywords if kw in text.lower()
        )

        numerical_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))

        # Nesting depth — count nested parenthetical/conditional structures
        nesting_depth = self._compute_nesting(text)

        # Entities — capitalized multi-word terms or technical signal names
        entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        entity_count = len(re.findall(entity_pattern, text))
        # Also count signal-style names (snake_case, camelCase)
        entity_count += len(re.findall(r"\b\w+_\w+\b", text))

        logical_operators = ["and", "or", "not", "nor", "xor", "if and only if"]
        logical_operator_count = sum(
            len(re.findall(rf"\b{op}\b", text, re.IGNORECASE))
            for op in logical_operators
        )

        temporal_keywords = ["before", "after", "within", "during", "until",
                             "simultaneously", "precede", "follow", "concurrent"]
        temporal_relation_count = sum(
            1 for kw in temporal_keywords if kw in text.lower()
        )

        # Domain term density
        tokens_lower = {t.lower().strip(".,;:()") for t in tokens}
        domain_hits = len(tokens_lower & _DOMAIN_TERMS)
        domain_term_density = domain_hits / max(token_count, 1)

        # Overall score (0-1 normalized)
        raw_score = (
            min(token_count / 50, 1.0) * 0.15
            + min(clause_count / 5, 1.0) * 0.15
            + min(condition_count / 3, 1.0) * 0.15
            + min(numerical_count / 4, 1.0) * 0.10
            + min(nesting_depth / 3, 1.0) * 0.10
            + min(entity_count / 5, 1.0) * 0.05
            + min(logical_operator_count / 4, 1.0) * 0.10
            + min(temporal_relation_count / 3, 1.0) * 0.10
            + domain_term_density * 0.10
        )
        overall_score = min(1.0, raw_score)

        if overall_score >= 0.6:
            level = "hard"
        elif overall_score >= 0.3:
            level = "medium"
        else:
            level = "easy"

        return ComplexityScore(
            token_count=token_count,
            clause_count=clause_count,
            condition_count=condition_count,
            numerical_count=numerical_count,
            nesting_depth=nesting_depth,
            entity_count=entity_count,
            logical_operator_count=logical_operator_count,
            temporal_relation_count=temporal_relation_count,
            domain_term_density=round(domain_term_density, 4),
            overall_score=round(overall_score, 4),
            complexity_level=level,
        )

    def _compute_nesting(self, text: str) -> int:
        """Estimate nesting depth from structural markers."""
        depth = 0
        max_depth = 0
        nesting_markers = [
            (r"\b(?:if|when|unless)\b", 1),
            (r"\b(?:then|otherwise)\b", -1),
        ]
        # Simple heuristic: count condition openers
        for marker, delta in nesting_markers:
            for _ in re.finditer(marker, text, re.IGNORECASE):
                depth += delta
                max_depth = max(max_depth, depth)

        # Also count parentheses
        paren_depth = 0
        for ch in text:
            if ch == "(":
                paren_depth += 1
                max_depth = max(max_depth, paren_depth)
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)

        return max_depth
