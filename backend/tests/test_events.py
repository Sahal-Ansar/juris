from datetime import UTC, datetime, timedelta
from itertools import count
from typing import Any

import pytest
from pydantic import ValidationError

from juris.events import (
    EVENT_TYPES,
    CaseView,
    Event,
    EventEmitter,
    EventOrderError,
    FoldError,
    InMemoryEventStore,
    fold,
    parse_event,
)
from juris.events import catalog as c
from juris.models import (
    Argument,
    CaseAnalysis,
    Chunk,
    Claim,
    Document,
    Evidence,
    Issue,
    JuryBallot,
    Objection,
    RunTotals,
    VerificationResult,
)
from juris.models.common import ClaimStatus, Stage
from tests.test_models import EXAMPLES

SPEC_CATALOG = {
    "case_created", "stage_started", "stage_completed", "agent_started", "agent_completed",
    "issue_framed", "query_issued", "evidence_registered", "claim_created", "argument_created",
    "counterargument_created", "precedent_card_created", "verification_result",
    "objection_raised", "objection_answered", "unresolved_question_added", "jury_ballot_cast",
    "jury_aggregated", "analysis_ready", "validation_failed", "budget_warning", "run_failed",
    "run_completed",
}  # fmt: skip

T0 = datetime(2026, 9, 26, 10, 0, tzinfo=UTC)


def m(model: Any) -> Any:
    return model.model_validate(EXAMPLES[model])


def recorded_run() -> tuple[InMemoryEventStore, list[Event]]:
    """A small but complete run, written by hand the way the orchestrator will emit it."""
    store = InMemoryEventStore()
    ticks = count()
    emitter = EventEmitter(
        store,
        run_id="run_abc",
        case_id="CASE-001",
        clock=lambda: T0 + timedelta(seconds=next(ticks)),
    )
    e = emitter.emit
    objection = m(Objection).model_copy(update={"response": None, "resolved": False})
    e(
        c.CaseCreatedPayload(
            question="Can the seller forfeit the earnest money?",
            profile="juris_full",
            corpus_snapshot_id="snap-2026-09",
        )
    )
    e(c.StageStartedPayload(), stage=Stage.S0)
    e(
        c.AgentStartedPayload(role="issue_framer", model="claude-opus-5-5"),
        stage=Stage.S0,
        agent="issue_framer",
    )
    e(
        c.IssueFramedPayload(
            issue=m(Issue), facts=["Buyer defaulted."], assumptions=["No loss pleaded."]
        ),
        stage=Stage.S0,
        agent="issue_framer",
    )
    e(
        c.AgentCompletedPayload(
            role="issue_framer", input_tokens=900, output_tokens=300, cost_usd=0.0096
        ),
        stage=Stage.S0,
        agent="issue_framer",
    )
    e(c.StageCompletedPayload(), stage=Stage.S0)
    e(c.StageStartedPayload(), stage=Stage.S1)
    e(
        c.QueryIssuedPayload(query="section 74 earnest money forfeiture", issue_ids=["I-1"]),
        stage=Stage.S1,
        agent="researcher",
    )
    e(
        c.EvidenceRegisteredPayload(evidence=m(Evidence), document=m(Document), chunk=m(Chunk)),
        stage=Stage.S1,
        agent="researcher",
    )
    e(c.StageCompletedPayload(), stage=Stage.S1)
    e(c.StageStartedPayload(), stage=Stage.S2)
    e(
        c.ClaimCreatedPayload(claim=m(Claim).model_copy(update={"status": ClaimStatus.PROPOSED})),
        stage=Stage.S2,
        agent="counsel_a",
    )
    e(c.ArgumentCreatedPayload(argument=m(Argument)), stage=Stage.S2, agent="counsel_a")
    e(c.StageCompletedPayload(), stage=Stage.S2)
    e(
        c.VerificationResultPayload(
            claim_id="C-001", evidence_id="E-001", result=m(VerificationResult)
        ),
        stage=Stage.S4,
        agent="verifier",
    )
    e(
        c.ClaimStatusChangedPayload(
            claim_id="C-001", status=ClaimStatus.VERIFIED, reason="Entailment: supports"
        ),
        stage=Stage.S4,
    )
    e(
        c.BudgetWarningPayload(
            tokens_used=180_000, tokens_budget=200_000, usd_used=1.9, action="skip rebuttal round 2"
        ),
        stage=Stage.S5,
    )
    e(c.StageCompletedPayload(outcome="skipped", reason="budget"), stage=Stage.S5)
    e(c.ObjectionRaisedPayload(objection=objection), stage=Stage.S6, agent="cross_examiner")
    e(
        c.ObjectionAnsweredPayload(
            objection_id="O-001", response="The ratio on s.74 is general.", resolved=True
        ),
        stage=Stage.S6,
        agent="counsel_a",
    )
    e(
        c.UnresolvedQuestionAddedPayload(
            question="Was actual loss pleaded?", issue_id="I-1", from_objection_id="O-001"
        ),
        stage=Stage.S6,
    )
    e(c.JuryBallotCastPayload(ballot=m(JuryBallot)), stage=Stage.S8, agent="juror_1")
    e(
        c.JuryAggregatedPayload(
            per_position={"I-1.A": {"evidence_strength": c.ScoreStats(mean=4, spread=0)}}
        ),
        stage=Stage.S8,
    )
    e(c.AnalysisReadyPayload(analysis=m(CaseAnalysis)), stage=Stage.S9, agent="judge")
    e(
        c.RunCompletedPayload(
            totals=RunTotals(llm_calls=12, input_tokens=180_000, output_tokens=20_000, usd=1.9)
        )
    )
    return store, store.read("run_abc")


def test_catalog_covers_the_spec() -> None:
    assert set(EVENT_TYPES) == SPEC_CATALOG | {"claim_status_changed"}


def test_recorded_run_folds_to_expected_view() -> None:
    _, events = recorded_run()
    view = fold(events)

    assert (view.run_id, view.case_id, view.profile) == ("run_abc", "CASE-001", "juris_full")
    assert view.status == "completed" and view.error is None
    assert view.current_stage is Stage.S2  # last stage that *started*
    assert {s: st.status for s, st in view.stages.items()} == {
        Stage.S0: "completed",
        Stage.S1: "completed",
        Stage.S2: "completed",
        Stage.S5: "skipped",
    }
    assert view.agents["issue_framer"].status == "done"
    assert view.agents["issue_framer"].model == "claude-opus-5-5"
    assert view.agents["issue_framer"].input_tokens == 900
    assert list(view.issues) == ["I-1"]
    assert view.facts == ["Buyer defaulted."] and view.assumptions == ["No loss pleaded."]
    assert [q.query for q in view.queries] == ["section 74 earnest money forfeiture"]
    assert list(view.evidence) == ["E-001"]
    assert list(view.documents) == ["SC-2023-INSC-1043"] and len(view.chunks) == 1
    assert view.claims["C-001"].status is ClaimStatus.VERIFIED
    assert list(view.arguments) == ["A-001"]
    assert view.verifications[0].result.status.value == "verified"
    assert view.objections["O-001"].resolved is True
    assert view.objections["O-001"].response == "The ratio on s.74 is general."
    assert view.unresolved_questions[0].from_objection_id == "O-001"
    assert view.budget_warnings[0].action == "skip rebuttal round 2"
    assert list(view.jury_ballots) == ["J-1"]
    assert view.jury_aggregate is not None
    assert view.analysis is not None and view.analysis.issues[0].leaning.value == "leaning_A"
    assert view.totals is not None and view.totals.llm_calls == 12
    assert [t.seq for t in view.timeline] == list(range(1, len(events) + 1))
    assert view.timeline[-1].type == "run_completed"


def test_fold_is_deterministic_and_idempotent() -> None:
    _, events = recorded_run()
    once = fold(events)
    assert fold(events) == once
    # Re-delivered events (e.g. an SSE reconnect) change nothing.
    assert fold(events + events[5:10]) == once
    # Folding in two parts gives the same view, and doesn't mutate the first part.
    head = fold(events[:10])
    head_before = head.model_copy(deep=True)
    assert fold(events[10:], start=head) == once
    assert fold(events[5:], start=head) == once  # overlap is skipped
    assert head == head_before


def test_view_serialises_for_the_ui() -> None:
    _, events = recorded_run()
    view = fold(events)
    assert CaseView.model_validate_json(view.model_dump_json()) == view


def test_events_round_trip_through_json() -> None:
    _, events = recorded_run()
    for event in events:
        assert parse_event(event.model_dump_json()) == event
        assert parse_event(event.model_dump(mode="json")) == event


def envelope(**overrides: Any) -> dict[str, Any]:
    return {
        "event_id": "evt_000001",
        "run_id": "run_abc",
        "case_id": "CASE-001",
        "seq": 1,
        "ts": "2026-09-26T10:00:00Z",
        "type": "case_created",
        "payload": {"question": "q", "profile": "juris_full", "corpus_snapshot_id": "s"},
        **overrides,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "made_up_event"},  # unknown type
        {"type": "claim_created"},  # payload doesn't match the type
        {"payload": {"question": "q"}},  # payload missing fields
        {"extra": 1},  # unknown envelope field
        {"ts": "2026-09-26T10:00:00"},  # timestamp without timezone
        {"event_id": "123"},  # bad event ID
        {"seq": 0},  # seq starts at 1
        {"schema_version": "2.0"},  # unsupported major version
    ],
)
def test_invalid_events_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_event(envelope(**overrides))


def test_fold_rejects_inconsistent_streams() -> None:
    _, events = recorded_run()
    with pytest.raises(FoldError, match="first event must be case_created"):
        fold(events[1:])
    with pytest.raises(FoldError, match="out of order"):
        fold([events[0], events[2], events[1]])  # seq 3 arrives before seq 2
    with pytest.raises(FoldError, match="already used"):
        fold([events[0], events[1], events[2].model_copy(update={"seq": 2})])
    with pytest.raises(FoldError, match="another run"):
        fold([events[0], events[1].model_copy(update={"run_id": "run_other"})])
    with pytest.raises(FoldError, match="after the run completed"):
        late = events[1].model_copy(update={"seq": 99, "event_id": "evt_000099"})
        fold([*events, late])
    unknown_claim = c.ClaimStatusChanged(
        event_id="evt_000002",
        run_id="run_abc",
        case_id="CASE-001",
        seq=2,
        ts=T0,
        payload=c.ClaimStatusChangedPayload(claim_id="C-042", status=ClaimStatus.FALLS, reason="x"),
    )
    with pytest.raises(FoldError, match="unknown claim"):
        fold([events[0], unknown_claim])
    with pytest.raises(FoldError, match="no events"):
        fold([])


def test_store_enforces_order() -> None:
    store, events = recorded_run()
    assert store.last_seq("run_abc") == len(events)
    assert [e.seq for e in store.read("run_abc", after_seq=20)] == list(range(21, len(events) + 1))
    with pytest.raises(EventOrderError, match="not after"):
        store.append(events[3])
    with pytest.raises(EventOrderError, match="belongs to case"):
        store.append(events[-1].model_copy(update={"seq": 999, "case_id": "CASE-999"}))
    with pytest.raises(EventOrderError, match="must start with case_created"):
        InMemoryEventStore().append(events[1])
    assert store.read("no_such_run") == [] and store.last_seq("no_such_run") == 0


def test_emitter_derives_ids_and_rejects_non_payloads() -> None:
    store, events = recorded_run()
    assert [e.event_id for e in events[:3]] == ["evt_000001", "evt_000002", "evt_000003"]
    assert all(e.ts.tzinfo is not None for e in events)
    emitter = EventEmitter(store, run_id="run_abc", case_id="CASE-001")
    with pytest.raises(TypeError, match="not an event payload"):
        emitter.emit(m(Claim))
