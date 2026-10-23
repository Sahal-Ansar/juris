"""Request and response bodies of the REST API (PLAN 9.1) that are not domain models.

The run itself is served as the domain types (``CaseView``, ``CaseAnalysis``,
``EvidenceGraph``); these are the API's own envelopes around them. They are exported with
the rest of the UI contract (``scripts/export_schemas.py``).
"""

import datetime as dt
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from juris.config import ProfileKind
from juris.models import Chunk, Document, Evidence, JurisModel, VerificationResult
from juris.models.common import ArgumentId, ClaimId

Text = Annotated[str, StringConstraints(min_length=1)]
RunStatus = Literal["queued", "running", "completed", "failed"]


class CaseRequest(JurisModel):
    """``POST /cases``: a legal question or fact scenario, and the configuration to run."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10)]
    profile: Text = Field(default="juris_full", description="A name from GET /configs")


class CaseCreated(JurisModel):
    case_id: Text
    run_id: Text


class RunInfo(JurisModel):
    run_id: Text
    profile: Text
    status: RunStatus
    created_at: dt.datetime


class CaseInfo(JurisModel):
    """``GET /cases/{id}``: the question and every run of it, oldest first."""

    case_id: Text
    question: Text
    created_at: dt.datetime
    runs: list[RunInfo]


class ConfigInfo(JurisModel):
    """``GET /configs``: a profile in ``configs/pipeline/`` that ``POST /cases`` accepts."""

    name: Text
    kind: ProfileKind
    description: str
    stages: dict[str, str] = Field(description="Stage -> on, off or passthrough")


class DocumentParagraph(JurisModel):
    """One paragraph of a judgment, or one section of a statute."""

    seq: int = Field(ge=0, description="Order in the document; a chunk's para_start/para_end")
    no: Text = Field(description='"12", "12-13", "p-4" (inferred), or a statute section "74"')
    part: Literal["front", "headnote", "body", "section"]
    title: str | None = Field(default=None, description="A statute section's heading")
    text: Text
    page_start: int | None = None
    page_end: int | None = None


class DocumentDetail(JurisModel):
    """``GET /documents/{id}``: a corpus document with its full text, paragraph by paragraph."""

    document: Document
    headnote: str | None = None
    paragraphs: list[DocumentParagraph]


class EvidenceCitation(JurisModel):
    """A claim that cites the evidence, with the latest verification of that pair."""

    claim_id: ClaimId
    argument_id: ArgumentId
    pinpoint: Text
    quote: Text
    verification: VerificationResult | None = None


class EvidenceDetail(JurisModel):
    """``GET /evidence/{run_id}/{evidence_id}``: the evidence, its source and who cites it."""

    run_id: Text
    evidence: Evidence
    document: Document
    chunk: Chunk
    citations: list[EvidenceCitation]
