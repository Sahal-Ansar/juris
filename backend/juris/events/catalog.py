"""Event catalog: one typed payload per event type, as a union discriminated on ``type``.

The initial catalog is IDEA_final §12, plus ``claim_status_changed`` (D-016): domain models
are immutable, so status changes need their own event.
"""

from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, TypeAdapter

from juris.config import Role
from juris.events.envelope import EnvelopeBase
from juris.models import (
    Argument,
    CaseAnalysis,
    Chunk,
    Claim,
    ClaimStatus,
    Counterargument,
    Document,
    Evidence,
    Issue,
    JurisModel,
    JuryBallot,
    Objection,
    PrecedentCard,
    RunTotals,
    VerificationResult,
)
from juris.models.common import ClaimId, EvidenceId, IssueId, ObjectionId, PositionId

Text = Annotated[str, StringConstraints(min_length=1)]


# ---- Payloads ----------------------------------------------------------------------------


class CaseCreatedPayload(JurisModel):
    question: Text
    profile: Text
    corpus_snapshot_id: Text


class StageStartedPayload(JurisModel):
    mode: Literal["on", "passthrough"] = "on"


class StageCompletedPayload(JurisModel):
    outcome: Literal["completed", "skipped", "passthrough"] = "completed"
    reason: str | None = Field(default=None, description="Why a stage was skipped")


class AgentStartedPayload(JurisModel):
    role: Role
    model: str | None = None


class AgentCompletedPayload(JurisModel):
    role: Role
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = None
    cached: bool = False


class IssueFramedPayload(JurisModel):
    issue: Issue
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class QueryIssuedPayload(JurisModel):
    query: Text
    issue_ids: list[IssueId] = Field(default_factory=list)


class EvidenceRegisteredPayload(JurisModel):
    """The evidence plus its document and chunk, so the UI never needs a second lookup."""

    evidence: Evidence
    document: Document
    chunk: Chunk


class ClaimCreatedPayload(JurisModel):
    claim: Claim


class ClaimStatusChangedPayload(JurisModel):
    claim_id: ClaimId
    status: ClaimStatus
    reason: Text


class ArgumentCreatedPayload(JurisModel):
    argument: Argument


class CounterargumentCreatedPayload(JurisModel):
    counterargument: Counterargument


class PrecedentCardCreatedPayload(JurisModel):
    card: PrecedentCard


class VerificationResultPayload(JurisModel):
    claim_id: ClaimId
    evidence_id: EvidenceId
    result: VerificationResult


class ObjectionRaisedPayload(JurisModel):
    objection: Objection


class ObjectionAnsweredPayload(JurisModel):
    objection_id: ObjectionId
    response: Text
    resolved: bool


class UnresolvedQuestionAddedPayload(JurisModel):
    question: Text
    issue_id: IssueId | None = None
    from_objection_id: ObjectionId | None = None


class JuryBallotCastPayload(JurisModel):
    ballot: JuryBallot


class ScoreStats(JurisModel):
    mean: float = Field(ge=1, le=5)
    spread: float = Field(ge=0, description="Max minus min across jurors")


class JuryAggregatedPayload(JurisModel):
    """Deterministic aggregate of all ballots: per position, per rubric criterion."""

    per_position: dict[PositionId, dict[str, ScoreStats]]
    contested_issue_ids: list[IssueId] = Field(
        default_factory=list, description="High spread: flagged for the Judge"
    )


class AnalysisReadyPayload(JurisModel):
    analysis: CaseAnalysis


class ValidationFailedPayload(JurisModel):
    errors: list[str] = Field(min_length=1)
    dropped: str | None = Field(default=None, description="What was dropped after the repair")


class BudgetWarningPayload(JurisModel):
    tokens_used: int = Field(ge=0)
    tokens_budget: int | None = None
    usd_used: float = Field(ge=0)
    usd_budget: float | None = None
    action: Text = Field(description="What the orchestrator skipped or reduced")


class RunFailedPayload(JurisModel):
    reason: Text


class RunCompletedPayload(JurisModel):
    totals: RunTotals


# ---- Events ------------------------------------------------------------------------------


class CaseCreated(EnvelopeBase):
    type: Literal["case_created"] = "case_created"
    payload: CaseCreatedPayload


class StageStarted(EnvelopeBase):
    type: Literal["stage_started"] = "stage_started"
    payload: StageStartedPayload


class StageCompleted(EnvelopeBase):
    type: Literal["stage_completed"] = "stage_completed"
    payload: StageCompletedPayload


class AgentStarted(EnvelopeBase):
    type: Literal["agent_started"] = "agent_started"
    payload: AgentStartedPayload


class AgentCompleted(EnvelopeBase):
    type: Literal["agent_completed"] = "agent_completed"
    payload: AgentCompletedPayload


class IssueFramed(EnvelopeBase):
    type: Literal["issue_framed"] = "issue_framed"
    payload: IssueFramedPayload


class QueryIssued(EnvelopeBase):
    type: Literal["query_issued"] = "query_issued"
    payload: QueryIssuedPayload


class EvidenceRegistered(EnvelopeBase):
    type: Literal["evidence_registered"] = "evidence_registered"
    payload: EvidenceRegisteredPayload


class ClaimCreated(EnvelopeBase):
    type: Literal["claim_created"] = "claim_created"
    payload: ClaimCreatedPayload


class ClaimStatusChanged(EnvelopeBase):
    type: Literal["claim_status_changed"] = "claim_status_changed"
    payload: ClaimStatusChangedPayload


class ArgumentCreated(EnvelopeBase):
    type: Literal["argument_created"] = "argument_created"
    payload: ArgumentCreatedPayload


class CounterargumentCreated(EnvelopeBase):
    type: Literal["counterargument_created"] = "counterargument_created"
    payload: CounterargumentCreatedPayload


class PrecedentCardCreated(EnvelopeBase):
    type: Literal["precedent_card_created"] = "precedent_card_created"
    payload: PrecedentCardCreatedPayload


class VerificationResultRecorded(EnvelopeBase):
    type: Literal["verification_result"] = "verification_result"
    payload: VerificationResultPayload


class ObjectionRaised(EnvelopeBase):
    type: Literal["objection_raised"] = "objection_raised"
    payload: ObjectionRaisedPayload


class ObjectionAnswered(EnvelopeBase):
    type: Literal["objection_answered"] = "objection_answered"
    payload: ObjectionAnsweredPayload


class UnresolvedQuestionAdded(EnvelopeBase):
    type: Literal["unresolved_question_added"] = "unresolved_question_added"
    payload: UnresolvedQuestionAddedPayload


class JuryBallotCast(EnvelopeBase):
    type: Literal["jury_ballot_cast"] = "jury_ballot_cast"
    payload: JuryBallotCastPayload


class JuryAggregated(EnvelopeBase):
    type: Literal["jury_aggregated"] = "jury_aggregated"
    payload: JuryAggregatedPayload


class AnalysisReady(EnvelopeBase):
    type: Literal["analysis_ready"] = "analysis_ready"
    payload: AnalysisReadyPayload


class ValidationFailed(EnvelopeBase):
    type: Literal["validation_failed"] = "validation_failed"
    payload: ValidationFailedPayload


class BudgetWarning(EnvelopeBase):
    type: Literal["budget_warning"] = "budget_warning"
    payload: BudgetWarningPayload


class RunFailed(EnvelopeBase):
    type: Literal["run_failed"] = "run_failed"
    payload: RunFailedPayload


class RunCompleted(EnvelopeBase):
    type: Literal["run_completed"] = "run_completed"
    payload: RunCompletedPayload


Event = Annotated[
    CaseCreated
    | StageStarted
    | StageCompleted
    | AgentStarted
    | AgentCompleted
    | IssueFramed
    | QueryIssued
    | EvidenceRegistered
    | ClaimCreated
    | ClaimStatusChanged
    | ArgumentCreated
    | CounterargumentCreated
    | PrecedentCardCreated
    | VerificationResultRecorded
    | ObjectionRaised
    | ObjectionAnswered
    | UnresolvedQuestionAdded
    | JuryBallotCast
    | JuryAggregated
    | AnalysisReady
    | ValidationFailed
    | BudgetWarning
    | RunFailed
    | RunCompleted,
    Field(discriminator="type"),
]

EVENT_ADAPTER: TypeAdapter[Event] = TypeAdapter(Event)

EVENT_TYPES: dict[str, type[EnvelopeBase]] = {
    cls.model_fields["type"].default: cls
    for cls in EnvelopeBase.__subclasses__()
    if "type" in cls.model_fields
}


def parse_event(data: dict[str, Any] | str | bytes) -> Event:
    """Validate a raw event (dict or JSON). Unknown ``type`` values are rejected."""
    if isinstance(data, dict):
        return EVENT_ADAPTER.validate_python(data)
    return EVENT_ADAPTER.validate_json(data)
