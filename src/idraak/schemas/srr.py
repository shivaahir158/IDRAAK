"""Semantic Requirement Representation (SRR) — the language-independent intermediate form."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ComparisonOperator(str, Enum):
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    BETWEEN = "between"
    APPROXIMATELY = "approximately"


class Modality(str, Enum):
    MANDATORY = "mandatory"  # shall, must
    RECOMMENDED = "recommended"  # should
    PERMITTED = "permitted"  # may, can
    OPTIONAL = "optional"  # optionally
    FORBIDDEN = "forbidden"  # shall not, must not


class Polarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class TemporalRelation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    WITHIN = "within"
    DURING = "during"
    UNTIL = "until"
    AT = "at"
    EVENTUALLY = "eventually"
    IMMEDIATELY = "immediately"
    SIMULTANEOUSLY = "simultaneously"


class Condition(BaseModel):
    """A conditional clause (if/when/unless/only if)."""

    condition_type: str = Field(description="Type: if, when, unless, only_if, if_and_only_if")
    trigger: str = Field(description="The trigger expression")
    negated: bool = Field(default=False)
    nested_conditions: list[Condition] = Field(default_factory=list)
    raw_text: str = Field(default="")

    model_config = {"json_schema_extra": {"examples": [{"condition_type": "if", "trigger": "valid is asserted", "negated": False, "raw_text": "if valid is asserted"}]}}


class TemporalConstraint(BaseModel):
    """A temporal/timing constraint."""

    relation: TemporalRelation
    value: Optional[float] = None
    unit: Optional[str] = None
    reference_event: Optional[str] = None
    raw_text: str = Field(default="")


class NumericalConstraint(BaseModel):
    """A numerical value with operator and unit."""

    parameter: str = Field(description="What is being constrained")
    operator: ComparisonOperator = ComparisonOperator.EQUAL
    value: float = 0.0
    value_upper: Optional[float] = None  # for BETWEEN
    unit: Optional[str] = None
    raw_text: str = Field(default="")


class OrderingConstraint(BaseModel):
    """An ordering/sequence constraint between events or actions."""

    first: str
    second: str
    ordering_type: str = Field(default="before", description="before, after, concurrent")
    strict: bool = Field(default=True)
    raw_text: str = Field(default="")


class ExceptionClause(BaseModel):
    """An exception or exclusion clause."""

    exception_type: str = Field(default="unless", description="unless, except, provided_that")
    condition: str = Field(default="")
    raw_text: str = Field(default="")


class Entity(BaseModel):
    """A named entity or signal in the requirement."""

    name: str
    entity_type: Optional[str] = None  # signal, module, register, buffer, etc.
    role: Optional[str] = None  # subject, object, instrument


class Relation(BaseModel):
    """A relation between two entities."""

    subject: str
    predicate: str
    object: str
    raw_text: str = Field(default="")


class SemanticRequirement(BaseModel):
    """The complete Semantic Requirement Representation (SRR).

    This is the language-independent structured form that enables
    deterministic comparison across translations.
    """

    requirement_id: str = ""
    domain: str = ""
    actor: Optional[str] = None
    action: Optional[str] = None
    object: Optional[str] = None
    modality: Optional[Modality] = None
    polarity: Optional[Polarity] = None
    conditions: list[Condition] = Field(default_factory=list)
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list)
    numerical_constraints: list[NumericalConstraint] = Field(default_factory=list)
    ordering_constraints: list[OrderingConstraint] = Field(default_factory=list)
    exceptions: list[ExceptionClause] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    safety_constraints: list[str] = Field(default_factory=list)
    security_constraints: list[str] = Field(default_factory=list)
    interface_entities: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    source_language: str = "en"
    normalized_text: str = ""
    raw_text: str = ""
    extraction_confidence: float = 0.0
    extraction_method: str = ""
    field_confidences: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
