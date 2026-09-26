"""``fold(events) -> CaseView``: the read model the UI consumes (IDEA_final §12).

Pure and deterministic: the same events always give the same view, and the input view is
never mutated. Re-delivered events (same ``seq`` and ``event_id``, e.g. after an SSE
reconnect) are skipped, so folding is idempotent. Anything else out of order is an error.
"""

from collections.abc import Iterable
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from juris.config import Role
from juris.events.catalog import (
    AgentCompleted,
    AgentStarted,
    AnalysisReady,
    ArgumentCreated,
    BudgetWarning,
    BudgetWarningPayload,
    CaseCreated,
    ClaimCreated,
    ClaimStatusChanged,
    CounterargumentCreated,
    Event,
    EvidenceRegistered,
    IssueFramed,
    JuryAggregated,
    JuryAggregatedPayload,
    JuryBallotCast,
    ObjectionAnswered,
    ObjectionRaised,
    PrecedentCardCreated,
    QueryIssued,
    RunCompleted,
    RunFailed,
    StageCompleted,
    StageStarted,
    UnresolvedQuestionAdded,
    ValidationFailed,
    VerificationResultRecorded,
)
from juris.models import (
    Argument,
    CaseAnalysis,
    Chunk,
    Claim,
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
from juris.models.common import Stage


class FoldError(ValueError):
    """The event stream is inconsistent (wrong run, out of order, unknown reference)."""


class StageState(JurisModel):
    status: Literal["running", "completed", "skipped", "passthrough"]
    reason: str | None = None


class AgentState(JurisModel):
    role: Role
    status: Literal["running", "done"]
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None


class QueryRecord(JurisModel):
    query: str
    issue_ids: list[str]
    stage: Stage | None
    agent: str | None


class VerificationRecord(JurisModel):
    claim_id: str
    evidence_id: str
    result: VerificationResult
    seq: int


class UnresolvedQuestion(JurisModel):
    question: str
    issue_id: str | None
    from_objection_id: str | None


class ValidationFailure(JurisModel):
    stage: Stage | None
    agent: str | None
    errors: list[str]
    dropped: str | None


class TimelineEntry(JurisModel):
    seq: int
    event_id: str
    ts: AwareDatetime
    type: str
    stage: Stage | None
    agent: str | None


class CaseView(BaseModel):
    """Everything the UI shows for one run. Dicts keep insertion (= event) order."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    question: str
    profile: str
    corpus_snapshot_id: str
    status: Literal["running", "completed", "failed"] = "running"
    error: str | None = None
    current_stage: Stage | None = None
    stages: dict[Stage, StageState] = Field(default_factory=dict)
    agents: dict[str, AgentState] = Field(default_factory=dict)
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    issues: dict[str, Issue] = Field(default_factory=dict)
    queries: list[QueryRecord] = Field(default_factory=list)
    documents: dict[str, Document] = Field(default_factory=dict)
    chunks: dict[str, Chunk] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    claims: dict[str, Claim] = Field(default_factory=dict)
    arguments: dict[str, Argument] = Field(default_factory=dict)
    counterarguments: dict[str, Counterargument] = Field(default_factory=dict)
    precedent_cards: dict[str, PrecedentCard] = Field(default_factory=dict)
    objections: dict[str, Objection] = Field(default_factory=dict)
    unresolved_questions: list[UnresolvedQuestion] = Field(default_factory=list)
    verifications: list[VerificationRecord] = Field(default_factory=list)
    jury_ballots: dict[str, JuryBallot] = Field(default_factory=dict)
    jury_aggregate: JuryAggregatedPayload | None = None
    validation_failures: list[ValidationFailure] = Field(default_factory=list)
    budget_warnings: list[BudgetWarningPayload] = Field(default_factory=list)
    analysis: CaseAnalysis | None = None
    totals: RunTotals | None = None
    timeline: list[TimelineEntry] = Field(default_factory=list)


def fold(events: Iterable[Event], start: CaseView | None = None) -> CaseView:
    """Fold events into a ``CaseView``, optionally continuing from an earlier view."""
    view = start.model_copy(deep=True) if start is not None else None
    for event in events:
        if view is None:
            if not isinstance(event, CaseCreated):
                raise FoldError(f"the first event must be case_created, got {event.type}")
            view = CaseView(
                run_id=event.run_id,
                case_id=event.case_id,
                question=event.payload.question,
                profile=event.payload.profile,
                corpus_snapshot_id=event.payload.corpus_snapshot_id,
            )
        elif _already_applied(view, event):
            continue
        else:
            _check_order(view, event)
            _apply(view, event)
        view.timeline.append(
            TimelineEntry(
                seq=event.seq,
                event_id=event.event_id,
                ts=event.ts,
                type=event.type,
                stage=event.stage,
                agent=event.agent,
            )
        )
    if view is None:
        raise FoldError("no events to fold")
    return view


def _already_applied(view: CaseView, event: Event) -> bool:
    for entry in view.timeline:
        if entry.seq == event.seq:
            if entry.event_id != event.event_id:
                raise FoldError(f"seq {event.seq} already used by {entry.event_id}")
            return True
    return False


def _check_order(view: CaseView, event: Event) -> None:
    if event.run_id != view.run_id or event.case_id != view.case_id:
        raise FoldError(f"event {event.event_id} belongs to another run or case")
    if view.timeline and event.seq <= view.timeline[-1].seq:
        raise FoldError(f"event seq {event.seq} is out of order")
    if view.status != "running":
        raise FoldError(f"event {event.event_id} arrived after the run {view.status}")
    if isinstance(event, CaseCreated):
        raise FoldError("case_created can only be the first event")


def _need(mapping: dict[str, object], key: str, what: str) -> None:
    if key not in mapping:
        raise FoldError(f"unknown {what} {key!r}")


def _apply(view: CaseView, event: Event) -> None:
    match event:
        case StageStarted():
            stage = _stage(event.stage, event.type)
            status: Literal["running", "passthrough"] = (
                "passthrough" if event.payload.mode == "passthrough" else "running"
            )
            view.stages[stage] = StageState(status=status)
            view.current_stage = stage
        case StageCompleted():
            stage = _stage(event.stage, event.type)
            view.stages[stage] = StageState(
                status=event.payload.outcome, reason=event.payload.reason
            )
        case AgentStarted():
            agent = _agent(event.agent, event.type)
            view.agents[agent] = AgentState(
                role=event.payload.role, status="running", model=event.payload.model
            )
        case AgentCompleted():
            agent = _agent(event.agent, event.type)
            previous = view.agents.get(agent)
            view.agents[agent] = AgentState(
                role=event.payload.role,
                status="done",
                model=previous.model if previous else None,
                input_tokens=(previous.input_tokens if previous else 0)
                + event.payload.input_tokens,
                output_tokens=(previous.output_tokens if previous else 0)
                + event.payload.output_tokens,
                cost_usd=event.payload.cost_usd,
            )
        case IssueFramed():
            view.issues[event.payload.issue.id] = event.payload.issue
            view.facts += [f for f in event.payload.facts if f not in view.facts]
            view.assumptions += [a for a in event.payload.assumptions if a not in view.assumptions]
        case QueryIssued():
            view.queries.append(
                QueryRecord(
                    query=event.payload.query,
                    issue_ids=list(event.payload.issue_ids),
                    stage=event.stage,
                    agent=event.agent,
                )
            )
        case EvidenceRegistered():
            payload = event.payload
            view.documents[payload.document.id] = payload.document
            view.chunks[payload.chunk.id] = payload.chunk
            view.evidence[payload.evidence.id] = payload.evidence
        case ClaimCreated():
            view.claims[event.payload.claim.id] = event.payload.claim
        case ClaimStatusChanged():
            claim_id = event.payload.claim_id
            _need(view.claims, claim_id, "claim")  # type: ignore[arg-type]
            view.claims[claim_id] = view.claims[claim_id].model_copy(
                update={"status": event.payload.status}
            )
        case ArgumentCreated():
            view.arguments[event.payload.argument.id] = event.payload.argument
        case CounterargumentCreated():
            view.counterarguments[event.payload.counterargument.id] = event.payload.counterargument
        case PrecedentCardCreated():
            view.precedent_cards[event.payload.card.document_id] = event.payload.card
        case VerificationResultRecorded():
            view.verifications.append(
                VerificationRecord(
                    claim_id=event.payload.claim_id,
                    evidence_id=event.payload.evidence_id,
                    result=event.payload.result,
                    seq=event.seq,
                )
            )
        case ObjectionRaised():
            view.objections[event.payload.objection.id] = event.payload.objection
        case ObjectionAnswered():
            objection_id = event.payload.objection_id
            _need(view.objections, objection_id, "objection")  # type: ignore[arg-type]
            view.objections[objection_id] = view.objections[objection_id].model_copy(
                update={"response": event.payload.response, "resolved": event.payload.resolved}
            )
        case UnresolvedQuestionAdded():
            view.unresolved_questions.append(
                UnresolvedQuestion(
                    question=event.payload.question,
                    issue_id=event.payload.issue_id,
                    from_objection_id=event.payload.from_objection_id,
                )
            )
        case JuryBallotCast():
            view.jury_ballots[event.payload.ballot.juror_id] = event.payload.ballot
        case JuryAggregated():
            view.jury_aggregate = event.payload
        case AnalysisReady():
            view.analysis = event.payload.analysis
        case ValidationFailed():
            view.validation_failures.append(
                ValidationFailure(
                    stage=event.stage,
                    agent=event.agent,
                    errors=list(event.payload.errors),
                    dropped=event.payload.dropped,
                )
            )
        case BudgetWarning():
            view.budget_warnings.append(event.payload)
        case RunFailed():
            view.status = "failed"
            view.error = event.payload.reason
        case RunCompleted():
            view.status = "completed"
            view.totals = event.payload.totals
        case _:
            raise FoldError(f"no fold rule for {event.type}")


def _stage(stage: Stage | None, event_type: str) -> Stage:
    if stage is None:
        raise FoldError(f"{event_type} needs a stage")
    return stage


def _agent(agent: str | None, event_type: str) -> str:
    if agent is None:
        raise FoldError(f"{event_type} needs an agent")
    return agent
