"""Hybrid SRR extractor combining deterministic and LLM-based extraction."""

from __future__ import annotations

import json
from typing import Any, Optional

from idraak.extraction.deterministic import DeterministicExtractor
from idraak.providers.base import LLMProvider
from idraak.schemas.srr import SemanticRequirement
from idraak.utils.logging import get_logger

logger = get_logger("hybrid_extractor")

_EXTRACTION_PROMPT = """Extract a structured semantic representation from the following technical requirement.

Return a JSON object with these fields:
- actor: the primary actor/subject (string or null)
- action: the main action/verb (string or null)
- object: the target/object of the action (string or null)
- modality: one of "mandatory", "recommended", "permitted", "optional", "forbidden" (string or null)
- polarity: "positive" or "negative" (string or null)
- conditions: list of {{condition_type, trigger, negated, raw_text}}
- temporal_constraints: list of {{relation, value, unit, reference_event, raw_text}}
- numerical_constraints: list of {{parameter, operator, value, value_upper, unit, raw_text}}
- ordering_constraints: list of {{first, second, ordering_type, strict, raw_text}}
- exceptions: list of {{exception_type, condition, raw_text}}
- entities: list of {{name, entity_type, role}}
- safety_constraints: list of strings
- security_constraints: list of strings
- interface_entities: list of strings
- units: list of normalized unit strings
- qualifiers: list of strings (e.g., "all", "every", "at least")
- extraction_confidence: float 0-1

Requirement text:
{text}

Language: {language}

Respond with valid JSON only."""


class HybridExtractor:
    """Combines deterministic regex extraction with LLM for complex fields.

    The deterministic extractor handles high-confidence fields (modality, polarity,
    numerical values, units). The LLM handles actor/action/object, entities, and
    relations. Deterministic results override LLM results for fields where regex
    is more reliable.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self._deterministic = DeterministicExtractor()
        self._llm = llm_provider

    def extract(
        self, text: str, language: str = "en", requirement_id: str = ""
    ) -> SemanticRequirement:
        # Always run deterministic extraction
        det_srr = self._deterministic.extract(text, language, requirement_id)

        if self._llm is None:
            return det_srr

        # Run LLM extraction
        try:
            llm_srr = self._llm_extract(text, language, requirement_id)
        except Exception as e:
            logger.warning(f"LLM extraction failed for {requirement_id}: {e}")
            return det_srr

        # Merge: deterministic wins for high-confidence fields
        return self._merge(det_srr, llm_srr)

    def _llm_extract(self, text: str, language: str, requirement_id: str) -> SemanticRequirement:
        assert self._llm is not None
        prompt = _EXTRACTION_PROMPT.format(text=text, language=language)
        response = self._llm.complete(
            prompt=prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from LLM for {requirement_id}")
            return SemanticRequirement(
                requirement_id=requirement_id,
                raw_text=text,
                source_language=language,
                extraction_method="llm_failed",
            )

        # Build SRR from LLM output, with safe parsing
        from idraak.schemas.srr import (
            Condition, TemporalConstraint, NumericalConstraint,
            OrderingConstraint, ExceptionClause, Entity,
        )

        def _safe_parse(model_cls, items):
            """Parse list of dicts into Pydantic models, skipping failures."""
            result = []
            for item in (items or []):
                if isinstance(item, dict):
                    try:
                        result.append(model_cls.model_validate(item))
                    except Exception:
                        pass
            return result

        return SemanticRequirement(
            requirement_id=requirement_id,
            actor=data.get("actor"),
            action=data.get("action"),
            object=data.get("object"),
            modality=data.get("modality"),
            polarity=data.get("polarity"),
            conditions=_safe_parse(Condition, data.get("conditions")),
            temporal_constraints=_safe_parse(TemporalConstraint, data.get("temporal_constraints")),
            numerical_constraints=_safe_parse(NumericalConstraint, data.get("numerical_constraints")),
            ordering_constraints=_safe_parse(OrderingConstraint, data.get("ordering_constraints")),
            exceptions=_safe_parse(ExceptionClause, data.get("exceptions")),
            entities=_safe_parse(Entity, data.get("entities")),
            safety_constraints=data.get("safety_constraints") or [],
            security_constraints=data.get("security_constraints") or [],
            interface_entities=data.get("interface_entities") or [],
            units=data.get("units") or [],
            qualifiers=data.get("qualifiers") or [],
            source_language=language,
            normalized_text=text.strip(),
            raw_text=text,
            extraction_confidence=data.get("extraction_confidence", 0.5),
            extraction_method="llm",
            metadata={"llm_model": self._llm.model_name},
        )

    def _merge(self, det: SemanticRequirement, llm: SemanticRequirement) -> SemanticRequirement:
        """Merge deterministic and LLM extractions conservatively.

        Strategy: Deterministic ALWAYS wins for comparison-critical fields
        (modality, polarity, numerical, temporal, units, conditions, exceptions,
        ordering, qualifiers). These are what the drift comparator checks.

        LLM only fills in fields that deterministic cannot extract well
        (actor, action, object, entities, relations, interface_entities)
        and only when deterministic found nothing.
        """
        return SemanticRequirement(
            requirement_id=det.requirement_id,
            domain=det.domain or llm.domain,
            # LLM fills semantic fields only when deterministic has nothing
            actor=det.actor or llm.actor,
            action=det.action or llm.action,
            object=det.object or llm.object,
            # Deterministic ALWAYS wins for comparison-critical fields
            modality=det.modality,
            polarity=det.polarity,
            conditions=det.conditions,
            temporal_constraints=det.temporal_constraints,
            numerical_constraints=det.numerical_constraints,
            ordering_constraints=det.ordering_constraints,
            exceptions=det.exceptions,
            units=det.units,
            qualifiers=det.qualifiers,
            safety_constraints=det.safety_constraints,
            security_constraints=det.security_constraints,
            # LLM fills entity/relation fields when deterministic has nothing
            entities=det.entities if det.entities else llm.entities,
            relations=det.relations if det.relations else llm.relations,
            interface_entities=det.interface_entities if det.interface_entities else llm.interface_entities,
            source_language=det.source_language,
            normalized_text=det.normalized_text,
            raw_text=det.raw_text,
            extraction_confidence=det.extraction_confidence,
            extraction_method="hybrid",
        )
