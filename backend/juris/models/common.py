"""Base model, enums, ID types and ID allocation shared by all domain models (IDEA_final §8)."""

import re
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints


class JurisModel(BaseModel):
    """Base for every domain model.

    Unknown fields are rejected so malformed LLM output fails loudly, and instances are
    immutable: the Case Record replaces objects (``model_copy(update=...)``) instead of
    mutating them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


# ---- IDs -------------------------------------------------------------------------------

EvidenceId = Annotated[str, StringConstraints(pattern=r"^E-\d{3,}$")]
IssueId = Annotated[str, StringConstraints(pattern=r"^I-\d+$")]
PositionId = Annotated[str, StringConstraints(pattern=r"^I-\d+\.[A-C]$")]
ClaimId = Annotated[str, StringConstraints(pattern=r"^C-\d{3,}$")]
ArgumentId = Annotated[str, StringConstraints(pattern=r"^A-\d{3,}$")]
CounterargumentId = Annotated[str, StringConstraints(pattern=r"^R-\d{3,}$")]
ObjectionId = Annotated[str, StringConstraints(pattern=r"^O-\d{3,}$")]
JurorId = Annotated[str, StringConstraints(pattern=r"^J-\d+$")]
# Corpus IDs are assigned at ingestion (PLAN 3.4) and stay stable across runs.
DocumentId = Annotated[str, StringConstraints(min_length=1, max_length=200)]
ChunkId = Annotated[str, StringConstraints(min_length=1, max_length=240)]

_PREFIXED = {"E": 3, "C": 3, "A": 3, "R": 3, "O": 3}


class IdAllocator:
    """Hands out stable per-run IDs: ``E-001``, ``C-001``, ``I-1``, ``I-1.A``, ``J-1``.

    Deterministic: the same sequence of requests always yields the same IDs, so replayed
    runs match. ``E``/``C``/``A``/``R``/``O`` are zero-padded to 3 digits (4+ digits beyond 999).
    """

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def _next(self, prefix: str) -> int:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return self._counters[prefix]

    def next(self, prefix: str) -> str:
        if prefix not in _PREFIXED:
            raise ValueError(f"unknown ID prefix {prefix!r}; use issue()/position()/juror()")
        return f"{prefix}-{self._next(prefix):0{_PREFIXED[prefix]}d}"

    def issue(self) -> str:
        return f"I-{self._next('I')}"

    @staticmethod
    def position(issue_id: str, index: int) -> str:
        """Position ``index`` (0-based: A, B, C) of an issue."""
        if not re.fullmatch(r"I-\d+", issue_id) or not 0 <= index <= 2:
            raise ValueError(f"bad issue ID or position index: {issue_id!r}, {index}")
        return f"{issue_id}.{'ABC'[index]}"

    def juror(self) -> str:
        return f"J-{self._next('J')}"


# ---- Stages and provenance ---------------------------------------------------------------


class Stage(StrEnum):
    """Pipeline stages S0-S10 (IDEA_final §6)."""

    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"
    S6 = "S6"
    S7 = "S7"
    S8 = "S8"
    S9 = "S9"
    S10 = "S10"


class CreatedBy(JurisModel):
    """Which stage and agent produced an object (e.g. ``counsel_a``, ``case_record``)."""

    stage: Stage
    agent: Annotated[str, StringConstraints(min_length=1)]


# ---- Enums -------------------------------------------------------------------------------


class DocumentKind(StrEnum):
    JUDGMENT = "judgment"
    STATUTE = "statute"
    OTHER = "other"


class CourtLevel(StrEnum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    TRIBUNAL = "tribunal"
    SUBORDINATE = "subordinate"


class ClaimStatus(StrEnum):
    """proposed → verified | weak | unsupported → contested → survives | falls."""

    PROPOSED = "proposed"
    VERIFIED = "verified"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"
    CONTESTED = "contested"
    SURVIVES = "survives"
    FALLS = "falls"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    WEAK = "weak"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class VerificationCheck(StrEnum):
    """The verifier's checks, cheapest first; ``check`` records the one that decided."""

    EXISTENCE = "existence"
    QUOTE = "quote"
    ENTAILMENT = "entailment"


class EntailmentLabel(StrEnum):
    SUPPORTS = "supports"
    PARTIALLY_SUPPORTS = "partially_supports"
    DOES_NOT_SUPPORT = "does_not_support"
    CONTRADICTS = "contradicts"


class ObjectionType(StrEnum):
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CITATION_MISMATCH = "citation_mismatch"
    DISTINGUISHABLE_PRECEDENT = "distinguishable_precedent"
    OVERRULED_OR_DOUBTED = "overruled_or_doubted"
    WRONG_STATUTE_OR_VERSION = "wrong_statute_or_version"
    LOGICAL_GAP = "logical_gap"
    MISSING_EVIDENCE = "missing_evidence"
    FACT_ASSUMPTION = "fact_assumption"


class PrecedentFit(StrEnum):
    DIRECTLY_SUPPORTS = "directly_supports"
    SUPPORTS_BY_ANALOGY = "supports_by_analogy"
    DISTINGUISHABLE = "distinguishable"
    DOES_NOT_SUPPORT = "does_not_support"
    CONTRARY = "contrary"


class Treatment(StrEnum):
    """How a later judgment treated a precedent (cues from PLAN 3.9)."""

    FOLLOWED = "followed"
    AFFIRMED = "affirmed"
    DISTINGUISHED = "distinguished"
    DOUBTED = "doubted"
    OVERRULED = "overruled"
    PER_INCURIAM = "per_incuriam"


class Leaning(StrEnum):
    STRONGLY_A = "strongly_A"
    LEANING_A = "leaning_A"
    BALANCED = "balanced"
    LEANING_B = "leaning_B"
    STRONGLY_B = "strongly_B"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Confidence(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
