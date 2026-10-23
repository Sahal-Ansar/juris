"""The Case Record (PLAN 7.1): IDEA_final §8's invariants and the claim lifecycle. A rejected
write raises ``InvariantViolation`` listing every problem, and appends nothing."""

from collections.abc import Callable
from typing import Any

import pytest

from juris.config import DeliberationParams
from juris.events import EventEmitter, InMemoryEventStore, parse_event
from juris.events.catalog import (
    CaseCreatedPayload,
    JuryAggregatedPayload,
    ScoreStats,
    VerificationResultPayload,
)
from juris.models import (
    CaseAnalysis,
    Citation,
    CreatedBy,
    IssueAnalysis,
    JuryBallot,
    PositionAnalysis,
    PrecedentCard,
    VerificationResult,
)
from juris.models.analysis import EvidenceRef, SourceRef
from juris.models.common import (
    ClaimStatus,
    Confidence,
    EntailmentLabel,
    Leaning,
    ObjectionType,
    PrecedentFit,
    Stage,
    VerificationCheck,
    VerificationStatus,
)
from juris.models.record import Justification, RubricScores
from juris.pipeline.case_record import CaseRecord, InvariantViolation
from tests.test_b1 import FATEH, HITS, SATYABRATA_QUOTE, STATUTE_QUOTE, FakeService

S2, S4 = Stage.S2, Stage.S4
FATEH_QUOTE = "The party must prove actual loss in the case of forfeiture of earnest money."


def new_record(params: DeliberationParams | None = None) -> CaseRecord:
    emitter = EventEmitter(InMemoryEventStore(), run_id="JE-001-s0", case_id="JE-001")
    emitter.emit(CaseCreatedPayload(question="Q?", profile="test", corpus_snapshot_id="snap"))
    return CaseRecord(emitter, FakeService(), params=params)


def framed() -> CaseRecord:
    """One issue I-1 (A/B) and three evidence items: E-001 statute, E-002/E-003 judgments."""
    record = new_record()
    record.frame_issue("Can the seller forfeit?", ["No.", "Yes."], stage=Stage.S0, agent="framer")
    for hit in HITS:
        record.register_evidence(hit, ["I-1"], stage=Stage.S1, agent="researcher")
    return record


def cite(claim_id: str, evidence_id: str, quote: str) -> Citation:
    return Citation(claim_id=claim_id, evidence_id=evidence_id, pinpoint="para 1", quote=quote)


def argued() -> CaseRecord:
    """framed() plus C-001 (I-1.A, A-001 citing E-001 and E-002) and C-002 (I-1.B, A-002
    citing E-003)."""
    record = framed()
    record.add_claim("Compensation is capped.", "I-1.A", stage=S2, agent="counsel_a")
    record.add_argument(
        ["C-001"],
        "s. 74 caps it.",
        [cite("C-001", "E-001", STATUTE_QUOTE), cite("C-001", "E-002", FATEH_QUOTE)],
        stage=S2,
        agent="counsel_a",
    )
    record.add_claim("Earnest money may be kept.", "I-1.B", stage=S2, agent="counsel_b")
    record.add_argument(
        ["C-002"], "Satyabrata.", [cite("C-002", "E-003", SATYABRATA_QUOTE)], stage=S2,
        agent="counsel_b",
    )  # fmt: skip
    return record


def verdict(status: VerificationStatus = VerificationStatus.VERIFIED) -> VerificationResult:
    label = {
        VerificationStatus.VERIFIED: EntailmentLabel.SUPPORTS,
        VerificationStatus.WEAK: EntailmentLabel.PARTIALLY_SUPPORTS,
        VerificationStatus.UNSUPPORTED: EntailmentLabel.DOES_NOT_SUPPORT,
    }[status]
    return VerificationResult(
        status=status,
        check=VerificationCheck.ENTAILMENT,
        label=label,
        justification="x",
        verifier_model="m",
        created_by=CreatedBy(stage=S4, agent="citation_verifier"),
    )


def rejected(record: CaseRecord, write: Callable[[], Any], *expected: str) -> list[str]:
    """The write is refused with errors containing each ``expected`` text; nothing is added."""
    before = len(record.emitter.events())
    with pytest.raises(InvariantViolation) as caught:
        write()
    assert len(record.emitter.events()) == before
    for text in expected:
        assert any(text in e for e in caught.value.errors), (text, caught.value.errors)
    return caught.value.errors


# ---- S0-S1 ----------------------------------------------------------------------------------


def test_frame_issue_allocates_ids_and_emits() -> None:
    record = new_record()
    issue = record.frame_issue(
        " Is time of the essence? ", ["Yes.", "No."], stage=Stage.S0, agent="framer",
        facts=["A house sale."], assumptions=["No waiver."],
    )  # fmt: skip
    assert issue.id == "I-1" and [p.id for p in issue.positions] == ["I-1.A", "I-1.B"]
    assert issue.question == "Is time of the essence?"
    view = record.view
    assert view.issues["I-1"] == issue and view.facts == ["A house sale."]
    events = record.emitter.events()
    assert all(parse_event(e.model_dump(mode="json")) == e for e in events)


def test_frame_issue_limits() -> None:
    record = new_record(DeliberationParams(max_issues=1))
    rejected(record, lambda: record.frame_issue("Q", ["A"], stage=Stage.S0, agent="f"), "2 to 2")
    rejected(
        record, lambda: record.frame_issue(" ", ["A", " "], stage=Stage.S0, agent="f"),
        "question is empty", "position statement is empty",
    )  # fmt: skip
    record.frame_issue("Q", ["A", "B"], stage=Stage.S0, agent="f")
    rejected(record, lambda: record.frame_issue("Q2", ["A", "B"], stage=Stage.S0, agent="f"),
             "at most 1 issues")  # fmt: skip
    three = new_record(DeliberationParams(positions_per_issue=3))
    issue = three.frame_issue("Q", ["A", "B", "C"], stage=Stage.S0, agent="f")
    assert issue.positions[2].id == "I-1.C"


def test_evidence_and_queries_need_known_issues() -> None:
    record = framed()
    assert [e.id for e in record.view.evidence.values()] == ["E-001", "E-002", "E-003"]
    rejected(record, lambda: record.register_evidence(HITS[0], ["I-9"], stage=Stage.S1,
             agent="r"), "unknown issue 'I-9'")  # fmt: skip
    rejected(record, lambda: record.register_evidence(HITS[0], [], stage=Stage.S1, agent="r"),
             "at least one issue")  # fmt: skip
    rejected(record, lambda: record.add_query("q", ["I-2"], stage=Stage.S1, agent="r"), "I-2")
    record.add_query("q", ["I-1"], stage=Stage.S1, agent="r")
    assert record.view.queries[-1].query == "q"


# ---- S2: claims and arguments (invariants 1-2) ---------------------------------------------


def test_claims_need_a_known_position() -> None:
    record = framed()
    rejected(record, lambda: record.add_claim("x", "I-1.C", stage=S2, agent="c"), "I-1.C")
    rejected(record, lambda: record.add_claim(" ", "I-1.A", stage=S2, agent="c"), "empty")
    claim = record.add_claim(" x ", "I-1.A", stage=S2, agent="counsel_a")
    assert (claim.id, claim.text, claim.author_agent) == ("C-001", "x", "counsel_a")
    assert claim.status is ClaimStatus.PROPOSED


def test_arguments_enforce_invariants_1_and_2() -> None:
    record = framed()
    record.add_claim("x", "I-1.A", stage=S2, agent="c")
    errors = rejected(
        record,
        lambda: record.add_argument(
            ["C-001", "C-009"],
            " ",
            [
                cite("C-001", "E-007", "anything"),  # invariant 1: not registered
                cite("C-001", "E-002", "the party need not prove loss"),  # invariant 2
                cite("C-002", "E-001", STATUTE_QUOTE),  # not one of this argument's claims
            ],
            stage=S2,
            agent="c",
        ),
        "unknown claim 'C-009'",
        "reasoning is empty",
        "citation 1: E-007",
        "existence check",
        "citation 2: E-002",
        "quote check",
        "citation 3 is for C-002",
    )
    assert len(errors) == 5
    argument = record.add_argument(
        ["C-001"], "ok", [cite("C-001", "E-001", "  reasonable  compensation ")], stage=S2,
        agent="c",
    )  # fmt: skip
    assert argument.id == "A-001"  # IDs are only used by accepted writes
    assert record.view.arguments["A-001"].citations[0].evidence_id == "E-001"


def test_counterarguments_answer_the_other_side_of_the_same_issue() -> None:
    record = argued()
    record.frame_issue("Another?", ["Yes.", "No."], stage=Stage.S0, agent="f")
    record.add_claim("Other issue.", "I-2.A", stage=S2, agent="c")
    record.add_argument(["C-003"], "r", [cite("C-003", "E-001", STATUTE_QUOTE)], stage=S2,
                        agent="c")  # fmt: skip
    own = lambda: record.add_counterargument("C-001", "A-001", stage=Stage.S5, agent="c")  # noqa: E731
    rejected(record, own, "own position")
    other = lambda: record.add_counterargument("C-001", "A-003", stage=Stage.S5, agent="c")  # noqa: E731
    rejected(record, other, "not about C-001's issue (I-1)")
    unknown = lambda: record.add_counterargument("C-099", "A-099", stage=Stage.S5, agent="c")  # noqa: E731
    rejected(record, unknown, "unknown claim", "unknown argument")
    counter = record.add_counterargument("C-001", "A-002", stage=Stage.S5, agent="counsel_b")
    assert (counter.id, counter.targets) == ("R-001", "C-001")


def test_claim_status_lifecycle() -> None:
    record = argued()
    set_ = record.set_claim_status
    rejected(record, lambda: set_("C-001", ClaimStatus.FALLS, "r", stage=S4, agent="v"),
             "from proposed to falls")  # fmt: skip
    assert set_("C-001", ClaimStatus.VERIFIED, "best citation verified", stage=S4, agent="v")
    assert not set_("C-001", ClaimStatus.VERIFIED, "again", stage=S4, agent="v")  # no-op
    assert set_("C-001", ClaimStatus.WEAK, "S7 re-check", stage=Stage.S7, agent="v")
    assert set_("C-001", ClaimStatus.CONTESTED, "objection O-001", stage=Stage.S6, agent="s")
    rejected(record, lambda: set_("C-001", ClaimStatus.VERIFIED, "r", stage=S4, agent="v"),
             "from contested to verified")  # fmt: skip
    assert set_("C-001", ClaimStatus.SURVIVES, "answered", stage=Stage.S9, agent="judge")
    rejected(record, lambda: set_("C-001", ClaimStatus.FALLS, "r", stage=Stage.S9, agent="j"),
             "from survives to falls")  # fmt: skip
    rejected(record, lambda: set_("C-404", ClaimStatus.WEAK, "r", stage=S4, agent="v"), "C-404")
    assert record.view.claims["C-001"].status is ClaimStatus.SURVIVES
    # a claim from S5 can be challenged in S6 before its first gate (S7)
    assert set_("C-002", ClaimStatus.CONTESTED, "objection O-002", stage=Stage.S6, agent="s")


# ---- S3, S4 ---------------------------------------------------------------------------------


def card(document_id: str, fits: dict[str, PrecedentFit] | None = None) -> PrecedentCard:
    return PrecedentCard(
        document_id=document_id,
        principle="p",
        material_facts="f",
        fit_by_argument=fits or {},
        created_by=CreatedBy(stage=Stage.S3, agent="precedent_analyst"),
    )


def test_precedent_cards() -> None:
    record = argued()
    add = record.add_precedent_card
    rejected(record, lambda: add(card("ACT-contract_act"), stage=Stage.S3, agent="p"),
             "is a statute, not a judgment")  # fmt: skip
    rejected(
        record, lambda: add(card("SC-unknown"), stage=Stage.S3, agent="p"), "not in the record"
    )
    rejected(
        record,
        lambda: add(card(FATEH, {"A-002": PrecedentFit.CONTRARY, "A-009": PrecedentFit.CONTRARY}),
                    stage=Stage.S3, agent="p"),
        "A-002 does not cite", "unknown argument 'A-009'",
    )  # fmt: skip
    add(card(FATEH, {"A-001": PrecedentFit.DIRECTLY_SUPPORTS}), stage=Stage.S3, agent="p")
    rejected(record, lambda: add(card(FATEH), stage=Stage.S3, agent="p"), "already has")
    assert record.view.precedent_cards[FATEH].fit_by_argument == {"A-001": "directly_supports"}


def test_verification_results_need_the_citation() -> None:
    record = argued()
    add = record.add_verification
    rejected(record, lambda: add("C-001", "E-003", verdict(), stage=S4, agent="v"),
             "no argument cites E-003 for C-001")  # fmt: skip
    rejected(record, lambda: add("C-009", "E-099", verdict(), stage=S4, agent="v"),
             "unknown claim", "unknown evidence")  # fmt: skip
    assert record.best_verification("E-001") is None
    add("C-001", "E-001", verdict(VerificationStatus.UNSUPPORTED), stage=S4, agent="v")
    add("C-001", "E-001", verdict(VerificationStatus.WEAK), stage=Stage.S7, agent="v")
    assert record.best_verification("E-001") is VerificationStatus.WEAK


def test_the_view_includes_events_other_code_emitted() -> None:
    record = argued()
    _ = record.view  # fold once, then write past the record (as the Citation Verifier does)
    record.emitter.emit(
        VerificationResultPayload(claim_id="C-002", evidence_id="E-003", result=verdict()),
        stage=S4,
        agent="citation_verifier",
    )
    assert record.best_verification("E-003") is VerificationStatus.VERIFIED


# ---- S6 -------------------------------------------------------------------------------------


def test_objections_target_existing_ids() -> None:
    record = argued()
    raise_ = record.raise_objection
    for missing in ("C-404", "E-404", "I-1.C", "O-001"):
        rejected(record, lambda m=missing: raise_(ObjectionType.LOGICAL_GAP, m, "t",  # type: ignore[misc]
                 stage=Stage.S6, agent="x"), "not in the record")  # fmt: skip
    targets = ["C-001", "E-002", "I-1.A", "A-002", "I-1"]
    for target in targets:
        raise_(ObjectionType.UNSUPPORTED_CLAIM, target, "t", stage=Stage.S6, agent="examiner")
    raise_(ObjectionType.LOGICAL_GAP, "O-001", "on an objection", stage=Stage.S6, agent="x")
    assert [o.target_id for o in record.view.objections.values()] == [*targets, "O-001"]
    assert record.view.objections["O-001"].raised_by == "examiner"


def test_one_reply_per_objection_and_unresolved_questions() -> None:
    record = argued()
    record.raise_objection(ObjectionType.MISSING_EVIDENCE, "C-002", "t", stage=Stage.S6, agent="x")
    answer = record.answer_objection
    answer("O-001", "It is in E-003.", resolved=False, stage=Stage.S6, agent="counsel_b")
    rejected(record, lambda: answer("O-001", "again", resolved=True, stage=Stage.S6, agent="c"),
             "already been answered")  # fmt: skip
    rejected(record, lambda: answer("O-002", "r", resolved=True, stage=Stage.S6, agent="c"),
             "unknown objection")  # fmt: skip
    add = record.add_unresolved_question
    rejected(record, lambda: add("q", stage=Stage.S6, agent="x", issue_id="I-2",
             from_objection_id="O-002"), "unknown issue", "unknown objection")  # fmt: skip
    add("Was the sum reasonable?", stage=Stage.S6, agent="x", issue_id="I-1",
        from_objection_id="O-001")  # fmt: skip
    view = record.view
    assert view.objections["O-001"].response == "It is in E-003."
    assert view.unresolved_questions[0].from_objection_id == "O-001"


# ---- S8-S9 ----------------------------------------------------------------------------------

SCORES = RubricScores(
    evidence_strength=4, precedent_fit=3, counterargument_handling=3, unresolved_risk=2
)


def ballot(juror: str, positions: tuple[str, ...] = ("I-1.A", "I-1.B"),
           cited: tuple[str, ...] = ("E-001",)) -> JuryBallot:  # fmt: skip
    return JuryBallot(
        juror_id=juror,
        per_position_scores=dict.fromkeys(positions, SCORES),
        justifications=[Justification(text="t", cited_ids=list(cited))],
        created_by=CreatedBy(stage=Stage.S8, agent=f"juror_{juror}"),
    )


def test_ballots_and_the_aggregate() -> None:
    record = argued()
    stats = {"evidence_strength": ScoreStats(mean=4, spread=0)}
    aggregate = JuryAggregatedPayload(per_position={"I-1.A": stats, "I-1.B": stats})
    rejected(record, lambda: record.set_jury_aggregate(aggregate, stage=Stage.S8, agent="jury"),
             "no ballots")  # fmt: skip
    cast = record.cast_ballot
    rejected(record, lambda: cast(ballot("J-1", ("I-1.A", "I-2.B"), ("E-001", "C-404")),
             stage=Stage.S8, agent="j"), "unknown position 'I-2.B'", "['C-404']")  # fmt: skip
    cast(ballot("J-1"), stage=Stage.S8, agent="juror_1")
    rejected(record, lambda: cast(ballot("J-1"), stage=Stage.S8, agent="j"), "already voted")
    bad = JuryAggregatedPayload(per_position={"I-3.A": stats}, contested_issue_ids=["I-3"])
    rejected(record, lambda: record.set_jury_aggregate(bad, stage=Stage.S8, agent="jury"),
             "unknown position 'I-3.A'", "unknown issue 'I-3'")  # fmt: skip
    record.set_jury_aggregate(aggregate, stage=Stage.S8, agent="jury")
    assert record.view.jury_aggregate == aggregate


def analysis(
    *,
    issues: tuple[str, ...] = ("I-1",),
    positions: tuple[str, ...] = ("A", "B"),
    evidence: str = "E-001",
    authorities: tuple[str, ...] = (),
    sources: tuple[str, ...] = ("ACT-contract_act",),
) -> CaseAnalysis:
    return CaseAnalysis(
        question="Q?",
        issues=[
            IssueAnalysis(
                issue_id=issue,
                issue_text="t",
                positions=[
                    PositionAnalysis(
                        position_id=f"{issue}.{side}",
                        summary="s",
                        supporting_authorities=list(authorities) if side == "A" else [],
                        key_evidence=[EvidenceRef(evidence_id=evidence, pinpoint="s. 74")]
                        if side == "A"
                        else [],
                    )
                    for side in positions
                ],
                leaning=Leaning.LEANING_A,
                confidence=Confidence.MODERATE,
                confidence_reason="r",
            )
            for issue in issues
        ],
        overall_summary="Summary [E-001].",
        sources=[SourceRef(document_id=d) for d in sources],
        created_by=CreatedBy(stage=Stage.S9, agent="judge"),
    )


def test_the_analysis_enforces_invariants_3_and_5() -> None:
    record = argued()
    set_ = record.set_analysis
    S9 = Stage.S9
    rejected(record, lambda: set_(analysis(), stage=S9, agent="judge"), "E-001 (never verified)")
    record.add_verification("C-001", "E-001", verdict(), stage=S4, agent="v")
    record.add_verification("C-001", "E-002", verdict(VerificationStatus.UNSUPPORTED),
                            stage=S4, agent="v")  # fmt: skip
    rejected(record, lambda: set_(analysis(evidence="E-002"), stage=S9, agent="j"),
             "I-1.A cites E-002 (unsupported)")  # fmt: skip
    rejected(record, lambda: set_(analysis(issues=("I-2",)), stage=S9, agent="j"),
             "does not cover I-1", "I-2 is not an S0 issue")  # fmt: skip
    rejected(record, lambda: set_(analysis(positions=("A", "B", "C")), stage=S9, agent="j"),
             "must analyse positions ['I-1.A', 'I-1.B']")  # fmt: skip
    rejected(record, lambda: set_(analysis(authorities=(FATEH,)), stage=S9, agent="j"),
             f"relies on {FATEH}")  # fmt: skip
    rejected(record, lambda: set_(analysis(sources=("SC-elsewhere",)), stage=S9, agent="j"),
             "source SC-elsewhere")  # fmt: skip
    record.add_verification("C-001", "E-002", verdict(VerificationStatus.WEAK), stage=Stage.S7,
                            agent="v")  # fmt: skip
    set_(analysis(authorities=(FATEH,)), stage=S9, agent="judge")  # weak is citable
    assert record.view.analysis is not None


def test_reject_logs_validation_failed() -> None:
    record = framed()
    record.reject(["E-009: unknown"], stage=S2, agent="counsel_a", dropped="citation 2")
    failure = record.view.validation_failures[0]
    assert (failure.errors, failure.dropped, failure.agent) == (["E-009: unknown"], "citation 2",
                                                               "counsel_a")  # fmt: skip


def test_the_view_needs_case_created() -> None:
    emitter = EventEmitter(InMemoryEventStore(), run_id="r", case_id="c")
    with pytest.raises(RuntimeError, match="case_created"):
        _ = CaseRecord(emitter, FakeService()).view
