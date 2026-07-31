"""Unit conversion and equivalence checking using Pint."""

from __future__ import annotations

from typing import Optional

from idraak.utils.logging import get_logger

logger = get_logger("unit_converter")

try:
    import pint
    _ureg = pint.UnitRegistry()
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False
    _ureg = None

# Manual equivalences for domain-specific units not in Pint
_CUSTOM_EQUIVALENCES = {
    ("ms", "s"): 0.001,
    ("s", "ms"): 1000,
    ("us", "ms"): 0.001,
    ("ms", "us"): 1000,
    ("ns", "us"): 0.001,
    ("us", "ns"): 1000,
    ("MHz", "GHz"): 0.001,
    ("GHz", "MHz"): 1000,
    ("kHz", "MHz"): 0.001,
    ("MHz", "kHz"): 1000,
    ("KB", "MB"): 1 / 1024,
    ("MB", "KB"): 1024,
    ("MB", "GB"): 1 / 1024,
    ("GB", "MB"): 1024,
    ("GB", "TB"): 1 / 1024,
    ("TB", "GB"): 1024,
    ("byte", "bit"): 8,
    ("bit", "byte"): 1 / 8,
    ("mA", "A"): 0.001,
    ("A", "mA"): 1000,
    ("mW", "W"): 0.001,
    ("W", "mW"): 1000,
    ("Mbps", "Gbps"): 0.001,
    ("Gbps", "Mbps"): 1000,
}


class UnitConverter:
    """Check whether two value+unit pairs are physically equivalent."""

    @staticmethod
    def are_equivalent(
        value1: float, unit1: str,
        value2: float, unit2: str,
        tolerance: float = 1e-6,
    ) -> bool:
        """Check if value1 in unit1 equals value2 in unit2."""
        if unit1 == unit2:
            return abs(value1 - value2) <= tolerance

        # Try custom equivalences
        key = (unit1, unit2)
        if key in _CUSTOM_EQUIVALENCES:
            converted = value1 * _CUSTOM_EQUIVALENCES[key]
            return abs(converted - value2) <= tolerance * max(abs(converted), abs(value2), 1)

        # Try Pint
        if _PINT_AVAILABLE and _ureg is not None:
            try:
                q1 = _ureg.Quantity(value1, unit1)
                q2 = _ureg.Quantity(value2, unit2)
                return abs(q1.to(unit2).magnitude - value2) <= tolerance * max(abs(value2), 1)
            except Exception:
                pass

        return False

    @staticmethod
    def convert(value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert a value from one unit to another."""
        if from_unit == to_unit:
            return value

        key = (from_unit, to_unit)
        if key in _CUSTOM_EQUIVALENCES:
            return value * _CUSTOM_EQUIVALENCES[key]

        if _PINT_AVAILABLE and _ureg is not None:
            try:
                return _ureg.Quantity(value, from_unit).to(to_unit).magnitude
            except Exception:
                pass

        return None
