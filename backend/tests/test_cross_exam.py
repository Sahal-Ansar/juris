"""The Cross-Examiner, counsel's replies and S6 (PLAN 7.8) on scripted outputs: objections
target valid IDs, the most severe are kept, replies are recorded once, and what stays open
becomes an unresolved question. No model, no database."""

from typing import Any

from juris.models import Citation
from juris.models.common import Stage
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.cross_exam import CrossExamStage
from tests.test_b1 import STATUTE_QUOTE
from tests.test_rebuttal import Script, at, context, verified

CE, REPLY = "cross_examiner", "counsel_reply"


def objection(kind: str, target: str, text: str = "A flaw.", severity: str = "medium") -> dict[
    str, Any
]:  # fmt: skip
    return {"type": kind, "target_id": target, "text": text, "severity": severity}


def reply(objection_id: str, stance: str, response: str = "Because.") -> dict[str, Any]:
    return {"objection_id": objection_id, "stance": stance, "response": response}


async def setup(replies: dict[tuple[str, str], list[Any]]) -> tuple[StageContext, Script]:
    """S0-S5 done: C-001 (A, verified), C-002 (B, weak), C-003 (B, verified; A-003 cites a
    judgment found distinguishable), and A's S5 counterargument C-004 answering C-002."""
    ctx, provider, _ = context(replies)
    await verified(ctx)
    record, S5 = ctx.record, Stage.S5
    claim = record.add_claim("Earnest money is part of the price.", "I-1.A", stage=S5, agent="a")
    cite = Citation(claim_id=claim.id, evidence_id="E-001", pinpoint="s. 74", quote=STATUTE_QUOTE)
    argument = record.add_argument([claim.id], "It is capped.", [cite], stage=S5, agent="a")
    record.add_counterargument("C-002", argument.id, stage=S5, agent="a")
    return at(ctx, Stage.S6), provider


OBJECTIONS = {
    "objections": [
        objection("unsupported_claim", "C-2", "E-003 only partly supports it.", "high"),
        objection("distinguishable_precedent", "A-003", "Fateh Chand differs."),
        objection("logical_gap", "C-004", "Capping is not the price.", "low"),
    ]
}


async def test_objections_replies_and_what_stays_open() -> None:
    replies: dict[tuple[str, str], list[Any]] = {
        ("cross_examiner", CE): [OBJECTIONS],
        ("counsel_b", REPLY): [
            {"replies": [reply("O-1", "concede"), reply("O-002", "maintain", "It applies.")]}
        ],
        ("counsel_a", REPLY): [{"replies": []}],
    }
    ctx, provider = await setup(replies)
    await CrossExamStage().run(ctx)
    view = ctx.record.view
    raised = [(o.id, o.type.value, o.target_id) for o in view.objections.values()]
    assert raised == [
        ("O-001", "unsupported_claim", "C-002"),
        ("O-002", "distinguishable_precedent", "A-003"),
        ("O-003", "logical_gap", "C-004"),
    ]
    # the replies: B conceded O-001 and maintained O-002; A did not answer O-003
    o = view.objections
    assert (o["O-001"].response, o["O-001"].resolved) == ("Because.", True)
    assert (o["O-002"].response, o["O-002"].resolved) == ("It applies.", False)
    assert o["O-003"].response is None
    # C-002 was contested (by O-001 and R-001) and fell; C-003 is contested through A-003;
    # C-004, from S5 and not verified yet, is contested too; C-001 was not challenged
    statuses = {c.id: c.status.value for c in view.claims.values()}
    assert statuses == {
        "C-001": "verified",
        "C-002": "falls",
        "C-003": "contested",
        "C-004": "contested",
    }
    changes = [e.payload for e in ctx.run.emitter.events() if e.type == "claim_status_changed"]
    assert [(c.claim_id, c.reason) for c in changes[-4:]] == [
        ("C-002", "challenged by O-001, R-001"),
        ("C-003", "challenged by O-002"),
        ("C-004", "challenged by O-003"),
        ("C-002", "counsel conceded O-001"),
    ]
    questions = [(q.from_objection_id, q.issue_id) for q in view.unresolved_questions]
    assert questions == [("O-002", "I-1"), ("O-003", "I-1")]
    assert "disputed; counsel replied: It applies." in view.unresolved_questions[0].question
    assert "(not answered)" in view.unresolved_questions[1].question
    assert ctx.run.values == ctx.run.values | {
        "s6.objections_proposed": 3.0,
        "s6.objections": 3.0,
        "s6.replies": 2.0,
        "s6.conceded": 1.0,
        "s6.maintained": 1.0,
        "s6.unanswered": 1.0,
        "s6.unresolved_questions": 2.0,
        "s6.contested": 3.0,
        "s6.falls": 1.0,
    }
    # what each side saw
    examiner = provider.prompts(CE)[0].messages[0].content
    assert "[C-002] for I-1.B, weak: A reasonable sum may be kept." in examiner
    assert "[C-004] for I-1.A, proposed: Earnest money is part of the price. (answers C-002)" in (
        examiner
    )
    assert "[verification verified; precedent fit distinguishable]" in examiner
    assert "Fateh Chand v. Balkishan Dass: principle (model-extracted): p" in examiner
    b = next(c for c in provider.prompts(REPLY) if c.tags["agent"] == "counsel_b")
    body = b.messages[0].content
    assert "[O-001] unsupported_claim, against C-002: E-003 only partly supports it." in body
    assert "your argument A-003: r" in body and "[O-003]" not in body


async def test_bad_objections_go_back_once_then_are_dropped() -> None:
    bad = {
        "objections": [
            objection("logical_gap", "C-009"),  # not in the record
            objection("logical_gap", "I-1"),  # not a claim or argument
            objection("unsupported_claim", "C-001"),
            objection("unsupported_claim", "C-001"),  # the same type and target twice
            objection("missing_evidence", "A-001", " "),  # empty
        ]
    }
    replies: dict[tuple[str, str], list[Any]] = {
        ("cross_examiner", CE): [bad, bad],
        ("counsel_a", REPLY): [{"replies": [reply("O-001", "maintain")]}],
    }
    ctx, provider = await setup(replies)
    await CrossExamStage(per_issue=5).run(ctx)
    repair = next(r for r in provider.calls if r.tags.get("repair") == "record")
    feedback = repair.messages[2].content
    assert "objections[0]: 'C-009' is not a claim or argument of this issue" in feedback
    assert "objections[1]: 'I-1' is not a claim or argument of this issue" in feedback
    assert "objections[3]: a second unsupported_claim objection against C-001" in feedback
    assert "objections[4]: the objection is empty" in feedback
    view = ctx.record.view
    assert [(o.id, o.target_id) for o in view.objections.values()] == [("O-001", "C-001")]
    assert view.validation_failures[0].agent == "cross_examiner"


async def test_only_the_most_severe_objections_are_kept() -> None:
    many = {
        "objections": [
            objection("logical_gap", "C-001", severity="low"),
            objection("unsupported_claim", "C-002", severity="high"),
            objection("fact_assumption", "C-003", severity="medium"),
        ]
    }
    replies: dict[tuple[str, str], list[Any]] = {
        ("cross_examiner", CE): [many],
        ("counsel_b", REPLY): [{"replies": [reply("O-001", "maintain"), reply("O-2", "maintain")]}],
    }
    ctx, _ = await setup(replies)
    await CrossExamStage(max_objections=2).run(ctx)
    view = ctx.record.view
    assert [o.target_id for o in view.objections.values()] == ["C-002", "C-003"]  # output order
    assert ctx.run.values["s6.objections_proposed"] == 3
    assert view.claims["C-001"].status.value == "verified"


async def test_bad_replies_are_dropped_and_the_objection_stays_open() -> None:
    bad = {"replies": [reply("O-009", "concede"), reply("O-001", "maintain", " ")]}
    replies: dict[tuple[str, str], list[Any]] = {
        ("cross_examiner", CE): [{"objections": [objection("logical_gap", "C-001")]}],
        ("counsel_a", REPLY): [bad, bad],
    }
    ctx, provider = await setup(replies)
    await CrossExamStage().run(ctx)
    repair = next(r for r in provider.calls if r.tags.get("repair") == "record")
    feedback = repair.messages[2].content
    assert "replies[0]: 'O-009' is not one of the objections to answer" in feedback
    assert "replies[1]: the reply is empty" in feedback
    view = ctx.record.view
    assert view.objections["O-001"].response is None
    assert [q.from_objection_id for q in view.unresolved_questions] == ["O-001"]
    assert view.claims["C-001"].status.value == "contested"


async def test_no_objections_and_passthrough() -> None:
    ctx, provider = await setup({("cross_examiner", CE): [{"objections": []}]})
    await CrossExamStage().passthrough(ctx)
    assert provider.prompts(CE) == []
    await CrossExamStage().run(ctx)
    view = ctx.record.view
    assert view.objections == {} and view.unresolved_questions == []
    assert provider.prompts(REPLY) == []
    # the counterargument still contests its target
    assert view.claims["C-002"].status.value == "contested"
