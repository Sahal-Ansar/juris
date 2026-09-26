"""Case Record objects produced during a run (IDEA_final §6, §8)."""

import re
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from juris.models.common import (
    ArgumentId,
    ChunkId,
    ClaimId,
    ClaimStatus,
    CounterargumentId,
    CreatedBy,
    DocumentId,
    EntailmentLabel,
    EvidenceId,
    IssueId,
    JurisModel,
    JurorId,
    ObjectionId,
    ObjectionType,
    PositionId,
    PrecedentFit,
    Stage,
    Treatment,
    VerificationCheck,
    VerificationStatus,
)

Text = Annotated[str, StringConstraints(min_length=1)]


class Evidence(JurisModel):
    """A retrieved chunk registered for this run. Only registered evidence may be cited."""

    id: EvidenceId
    chunk_id: ChunkId
    document_id: DocumentId
    issue_ids: list[IssueId] = Field(min_length=1)
    retrieval_score: float | None = None
    rerank_score: float | None = None
    retrieved_by_query: Text
    registered_at_stage: Stage
    created_by: CreatedBy


class Position(JurisModel):
    """One side of an issue, phrased neutrally."""

    id: PositionId
    statement: Text


class Issue(JurisModel):
    """A precise legal question with 2 (at most 3) positions."""

    id: IssueId
    question: Text
    positions: list[Position] = Field(min_length=2, max_length=3)
    created_by: CreatedBy

    @model_validator(mode="after")
    def _positions_belong(self) -> Self:
        expected = [f"{self.id}.{letter}" for letter in "ABC"[: len(self.positions)]]
        if [p.id for p in self.positions] != expected:
            raise ValueError(f"positions of {self.id} must be {expected}")
        return self


class VerificationResult(JurisModel):
    """The Citation Verifier's verdict on one citation (S4/S7/S10)."""

    status: VerificationStatus
    check: VerificationCheck = Field(description="The check that decided the status")
    label: EntailmentLabel | None = Field(default=None, description="Set by the entailment check")
    justification: Text
    verifier_model: str | None = Field(default=None, description="None for code-only checks")
    created_by: CreatedBy

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.check is VerificationCheck.ENTAILMENT and self.label is None:
            raise ValueError("an entailment check needs a label")
        if self.check is not VerificationCheck.ENTAILMENT and self.status is not (
            VerificationStatus.INVALID
        ):
            raise ValueError("existence/quote checks can only fail (status 'invalid')")
        return self


class Citation(JurisModel):
    """A claim's link to evidence: pinpoint plus a verbatim quote from the chunk."""

    claim_id: ClaimId
    evidence_id: EvidenceId
    pinpoint: Text = Field(description="Paragraph or page, e.g. 'para 12' or 'p. 358'")
    quote: Text = Field(description="Must be an exact normalised substring of the chunk")
    verification: VerificationResult | None = None


class Claim(JurisModel):
    """An atomic legal proposition put forward for a position."""

    id: ClaimId
    text: Text
    position_id: PositionId
    author_agent: Text
    status: ClaimStatus = ClaimStatus.PROPOSED
    created_by: CreatedBy


class Argument(JurisModel):
    """Links claims to evidence with short, structured reasoning."""

    id: ArgumentId
    claim_ids: list[ClaimId] = Field(min_length=1)
    reasoning: Text
    citations: list[Citation] = Field(default_factory=list)
    created_by: CreatedBy


class Counterargument(JurisModel):
    """A rebuttal (S5) aimed at a specific opposing claim."""

    id: CounterargumentId
    targets: ClaimId
    argument_id: ArgumentId
    created_by: CreatedBy


class TreatmentSignal(JurisModel):
    """A treatment cue in the corpus: a later judgment that followed, overruled, ... this one."""

    kind: Treatment
    by_document_id: DocumentId
    cue_text: Text


class PrecedentCard(JurisModel):
    """The Precedent Analyst's assessment of one cited judgment (S3).

    Unknown treatment is ``treatment=[]`` with ``treatment_verified=False``: never good law
    by default (IDEA_final §9, D-013).
    """

    document_id: DocumentId
    principle: Text = Field(description="The ratio as the model reads it")
    principle_source: Literal["model_extracted"] = "model_extracted"
    material_facts: Text
    distinguishing_factors: list[str] = Field(default_factory=list)
    treatment: list[TreatmentSignal] = Field(default_factory=list)
    treatment_verified: bool = False
    fit_by_argument: dict[ArgumentId, PrecedentFit] = Field(default_factory=dict)
    created_by: CreatedBy


_ANY_RECORD_ID = re.compile(r"^(?:[ECARO]-\d{3,}|I-\d+(?:\.[A-C])?)$")


class Objection(JurisModel):
    """A Cross-Examiner objection (S6) against a claim, citation, argument, ... ID."""

    id: ObjectionId
    type: ObjectionType
    target_id: Text
    raised_by: Text
    text: Text
    response: str | None = Field(default=None, description="The targeted counsel's one reply")
    resolved: bool = False
    created_by: CreatedBy

    @model_validator(mode="after")
    def _target_is_record_id(self) -> Self:
        if not _ANY_RECORD_ID.match(self.target_id):
            raise ValueError(f"target_id {self.target_id!r} is not a Case Record ID")
        return self


Score = Annotated[int, Field(ge=1, le=5)]


class RubricScores(JurisModel):
    """The jury rubric for one position (IDEA_final §6.1), each scored 1-5."""

    evidence_strength: Score
    precedent_fit: Score
    counterargument_handling: Score
    unresolved_risk: Score


class Justification(JurisModel):
    text: Text
    cited_ids: list[str] = Field(min_length=1, description="Evidence/claim/objection IDs")


class JuryBallot(JurisModel):
    """One juror's independent scores (S8). Jurors cannot add evidence."""

    juror_id: JurorId
    per_position_scores: dict[PositionId, RubricScores] = Field(min_length=2)
    justifications: list[Justification] = Field(min_length=1)
    created_by: CreatedBy
