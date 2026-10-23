"""Jurors and S8 (PLAN 7.10) on scripted ballots: every juror sees the same frozen record,
ballots validate against it, and the aggregate is deterministic. No model, no database."""

import json
from typing import Any

import pytest

from juris.models import CreatedBy, Issue, JuryBallot, Justification, Position, RubricScores
from juris.models.common import ObjectionType, Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.jury import JuryStage, aggregate
from tests.test_rebuttal import Script, at, context, verified

JUROR = "juror"


def scores(position: str, e: int, p: int, c: int, u: int, *ids: str) -> dict[str, Any]:
    return {
        "position_id": position,
        "evidence_strength": e,
        "precedent_fit": p,
        "counterargument_handling": c,
        "unresolved_risk": u,
        "justification": "Because.",
        "cited_ids": list(ids) or ["C-001"],
    }


Four = tuple[int, int, int, int]


def ballot(a: Four, b: Four) -> dict[str, Any]:
    return {"scores": [scores("I-1.A", *a), scores("I-1.B", *b, "C-2", "O-001")]}


async def setup(replies: dict[tuple[str, str], list[Any]]) -> tuple[StageContext, Script]:
    """S0-S7 done: C-001 (A, verified), C-002 (B, weak), C-003 (B, verified); an objection
    O-001 against C-002 that B maintained, and the unresolved question it left."""
    ctx, provider, _ = context(replies)
    await verified(ctx)
    record, S6 = ctx.record, Stage.S6
    record.raise_objection(ObjectionType.UNSUPPORTED_CLAIM, "C-002", "Only partly.", stage=S6,
                           agent="cross_examiner")  # fmt: skip
    record.answer_objection("O-001", "It holds.", resolved=False, stage=S6, agent="counsel_b")
    record.add_unresolved_question("Does O-001 hold?", stage=S6, agent="cross_examiner",
                                   issue_id="I-1", from_objection_id="O-001")  # fmt: skip
    return at(ctx, Stage.S8), provider


def jurors(*ballots: Any) -> dict[tuple[str, str], list[Any]]:
    return {(f"juror_{k}", JUROR): [b] for k, b in enumerate(ballots, 1)}


async def test_three_jurors_score_the_same_record_and_are_aggregated() -> None:
    replies = jurors(
        ballot((4, 3, 3, 4), (2, 2, 3, 2)),
        ballot((4, 4, 3, 4), (3, 2, 3, 2)),
        ballot((5, 3, 4, 4), (2, 3, 5, 5)),
    )
    ctx, provider = await setup(replies)
    await JuryStage().run(ctx)
    view = ctx.record.view
    assert list(view.jury_ballots) == ["J-1", "J-2", "J-3"]
    j1 = view.jury_ballots["J-1"]
    assert j1.per_position_scores["I-1.B"].evidence_strength == 2
    assert j1.justifications[1].text == "I-1.B: Because."
    assert j1.justifications[1].cited_ids == ["C-002", "O-001"]  # "C-2" read as C-002
    stats = view.jury_aggregate
    assert stats is not None
    a = stats.per_position["I-1.A"]
    assert (a["evidence_strength"].mean, a["evidence_strength"].spread) == (4.3333, 1.0)
    assert (a["overall"].mean, a["overall"].spread) == (3.75, 0.5)
    assert stats.per_position["I-1.B"]["overall"].spread == 1.5  # 2.25 to 3.75
    assert stats.contested_issue_ids == ["I-1"]
    values = ctx.run.values
    assert values["s8.ballots"] == 3 and values["s8.contested_issues"] == 1
    # the same prompt for every juror; only the seed differs
    calls = provider.prompts(JUROR)
    assert len({c.messages[0].content for c in calls}) == 1
    assert sorted(c.seed for c in calls if c.seed is not None) == [1, 2, 3]
    body = calls[0].messages[0].content
    assert "[C-002] for I-1.B, weak: A reasonable sum may be kept." in body
    assert (
        "[O-001] unsupported_claim against C-002: Only partly.\n  reply maintained: It holds."
        in (body)
    )
    assert "Unresolved questions:\n- Does O-001 hold?" in body
    assert "retrieval" not in body and "score" not in body.split("<record>")[1].split("Rubric")[0]


async def test_a_ballot_is_all_or_nothing() -> None:
    one_position = {"scores": [scores("I-1.A", 3, 3, 3, 3)]}
    unknown_id = ballot((3, 3, 3, 3), (3, 3, 3, 3))
    unknown_id["scores"][0]["cited_ids"] = ["C-001", "E-404"]
    replies = {
        ("juror_1", JUROR): [one_position, one_position],
        ("juror_2", JUROR): [unknown_id, unknown_id],
        ("juror_3", JUROR): [ballot((3, 3, 3, 3), (3, 3, 3, 3))],
    }
    ctx, provider = await setup(replies)
    await JuryStage().run(ctx)
    view = ctx.record.view
    assert list(view.jury_ballots) == ["J-2", "J-3"]  # J-1 never scored I-1.B
    assert view.jury_ballots["J-2"].justifications[0].cited_ids == ["C-001"]  # E-404 dropped
    repair = next(
        r for r in provider.calls if r.tags.get("agent") == "juror_1" and r.tags.get("repair")
    )
    assert "no scores for I-1.B" in repair.messages[2].content
    failures = {f.agent: f.dropped for f in view.validation_failures}
    assert failures == {"juror_1": "the whole output", "juror_2": "scores[0].cited_ids[1]"}


async def test_only_one_juror_once_the_budget_is_spent() -> None:
    ctx, provider = await setup(jurors(ballot((3, 3, 3, 3), (2, 2, 2, 2))))
    spent = BudgetManager(ctx.run.config.budget.model_copy(update={"max_tokens_per_case": 10}),
                          lambda: Spend(tokens=10))  # fmt: skip
    await JuryStage().run(StageContext(ctx.run, ctx.record, spent, Stage.S8, "on"))
    assert len(provider.prompts(JUROR)) == 1
    warnings = [e.payload.action for e in ctx.run.emitter.events() if e.type == "budget_warning"]
    assert warnings == ["ran 1 juror of 3"]
    assert ctx.record.view.jury_aggregate is not None


async def test_no_ballot_and_passthrough_write_no_aggregate() -> None:
    bad: dict[str, list[Any]] = {"scores": []}
    ctx, _ = await setup({(f"juror_{k}", JUROR): [bad, bad] for k in (1, 2, 3)})
    await JuryStage().passthrough(ctx)
    await JuryStage().run(ctx)
    view = ctx.record.view
    assert view.jury_ballots == {} and view.jury_aggregate is None
    assert ctx.run.values["s8.ballots"] == 0


def _ballot(juror: str, a: int, b: int, c: int) -> JuryBallot:
    def rubric(n: int) -> RubricScores:
        return RubricScores(
            evidence_strength=n, precedent_fit=n, counterargument_handling=n, unresolved_risk=n
        )

    return JuryBallot(
        juror_id=juror,
        per_position_scores={
            "I-1.A": rubric(a),
            "I-1.B": rubric(b),
            "I-2.A": rubric(c),
            "I-2.B": rubric(c),
        },
        justifications=[Justification(text="t", cited_ids=["C-001"])],
        created_by=CreatedBy(stage=Stage.S8, agent=juror),
    )


@pytest.mark.parametrize("threshold, contested", [(1.0, ["I-1"]), (2.0, [])])
def test_the_aggregate_is_deterministic_and_order_free(
    threshold: float, contested: list[str]
) -> None:
    by = CreatedBy(stage=Stage.S0, agent="f")
    issues = {
        i: Issue(id=i, question="q", created_by=by,
                 positions=[Position(id=f"{i}.A", statement="a"),
                            Position(id=f"{i}.B", statement="b")])
        for i in ("I-1", "I-2")
    }  # fmt: skip
    ballots = [_ballot("J-1", 5, 2, 3), _ballot("J-2", 3, 2, 3), _ballot("J-3", 4, 2, 4)]
    first = aggregate(ballots, issues, contested_spread=threshold)
    again = aggregate(list(reversed(ballots)), issues, contested_spread=threshold)
    assert first == again
    assert json.loads(first.model_dump_json()) == json.loads(again.model_dump_json())
    assert first.per_position["I-1.A"]["overall"].mean == 4.0
    assert first.per_position["I-1.A"]["overall"].spread == 2.0
    assert first.contested_issue_ids == contested
