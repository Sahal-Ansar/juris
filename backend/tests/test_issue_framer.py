"""The Issue Framer and S0 (PLAN 7.3) on canned outputs: what reaches the Case Record, the
seeded A/B labels, and what the checks drop. No model, no database."""

import copy
import json
import random
from typing import Any

import pytest

from juris.agents.issue_framer import IssueFramer
from juris.baselines.base import RunContext
from juris.config import (
    DeliberationParams,
    ModelsConfig,
    ModelSpec,
    PipelineProfile,
    Settings,
    resolve_run_config,
)
from juris.eval.juris_eval import load_split
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.framing import FramingFailed, IssueFramingStage
from tests.test_b1 import FakeService

PRESUMPTION = "Does a near relationship alone raise a presumption of undue influence?"
OUTPUT: dict[str, Any] = {
    "facts": ["An old man gifted land to his grandson.", " "],
    "assumptions": ["The deed was registered."],
    "issues": [
        {
            "question": PRESUMPTION,
            "positions": ["Yes, it raises a presumption.", "No, domination must be shown."],
            "queries": ["presumption undue influence near relations", "dominate the will"],
            "statute_refs": ["s. 16 of the Indian Contract Act"],
        },
        {
            "question": "On whom does the burden of proof lie?",
            "positions": ["On the donee.", "On the plaintiffs."],
            "queries": ["burden of proof undue influence", "s. 16(3) burden shifts"],
            "statute_refs": [],
        },
    ],
}


def context(*responses: Any, seed: int = 0, params: DeliberationParams | None = None) -> tuple[
    StageContext, FakeProvider
]:  # fmt: skip
    provider = FakeProvider(
        {"issue_framer": [r if isinstance(r, str) else json.dumps(r) for r in responses]}
    )
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    profile = PipelineProfile(name="t", kind="juris", params=params or DeliberationParams())
    config = resolve_run_config(settings, profile)
    item = load_split("dev")[0]
    emitter = EventEmitter(InMemoryEventStore(), run_id=f"{item.id}-s0", case_id=item.id)
    emitter.emit(CaseCreatedPayload(question=item.question, profile="t", corpus_snapshot_id="s"))
    record = CaseRecord(emitter, FakeService(), params=profile.params)
    run = RunContext(item, config, LLMGateway({"fake": provider}, prices={}), emitter, seed)
    return StageContext(run, record, BudgetManager(config.budget, Spend), Stage.S0, "on"), provider


def swaps(seed: int, issues: int) -> list[bool]:
    return [random.Random(f"{seed}:{n}").random() < 0.5 for n in range(issues)]


def seed_with(pattern: list[bool]) -> int:
    return next(s for s in range(1000) if swaps(s, len(pattern)) == pattern)


async def test_the_framing_reaches_the_record() -> None:
    seed = seed_with([False, False])
    ctx, provider = context(OUTPUT, seed=seed)
    await IssueFramingStage().run(ctx)
    view = ctx.record.view
    assert list(view.issues) == ["I-1", "I-2"]
    assert view.issues["I-1"].question == PRESUMPTION
    assert [p.statement for p in view.issues["I-1"].positions] == OUTPUT["issues"][0]["positions"]
    assert view.facts == ["An old man gifted land to his grandson."]
    assert view.assumptions == ["The deed was registered."]
    queries = [(q.query, q.issue_ids, q.kind, q.stage, q.agent) for q in view.queries]
    assert queries == [
        ("presumption undue influence near relations", ["I-1"], "search", Stage.S0, "issue_framer"),
        ("dominate the will", ["I-1"], "search", Stage.S0, "issue_framer"),
        ("s. 16 of the Indian Contract Act", ["I-1"], "provision", Stage.S0, "issue_framer"),
        ("burden of proof undue influence", ["I-2"], "search", Stage.S0, "issue_framer"),
        ("s. 16(3) burden shifts", ["I-2"], "search", Stage.S0, "issue_framer"),
    ]
    assert view.validation_failures == []
    assert ctx.run.values == {"s0.issues": 2.0, "s0.positions_swapped": 0.0}
    body = provider.calls[0].messages[0].content
    assert ctx.run.item.question in body and ctx.run.item.facts in body
    assert "the case turns on, up to 4" in body and "2 to 5 short search queries" in body
    assert provider.calls[0].system is not None
    assert provider.calls[0].system.startswith("You are the issue framer in Juris")


async def test_which_position_is_a_comes_from_the_seed() -> None:
    seed = seed_with([True, False])
    ctx, _ = context(OUTPUT, seed=seed)
    await IssueFramingStage().run(ctx)
    positions = {i: [p.statement for p in v.positions] for i, v in ctx.record.view.issues.items()}
    assert positions["I-1"] == ["No, domination must be shown.", "Yes, it raises a presumption."]
    assert positions["I-2"] == ["On the donee.", "On the plaintiffs."]
    assert ctx.run.details["s0.swapped"] == ["I-1"]
    again, _ = context(OUTPUT, seed=seed)
    await IssueFramingStage().run(again)
    assert again.record.view.issues == ctx.record.view.issues


def messy() -> dict[str, Any]:
    out = copy.deepcopy(OUTPUT)
    first, second = out["issues"]
    first["positions"].append("It depends on the facts.")  # a third position
    first["queries"] += ["Dominate  the will", ""]  # a repeat and an empty query
    first["statute_refs"].append("s. 300 of the Indian Penal Code")  # not in the corpus
    second["queries"] = ["burden of proof undue influence"]  # only one query: issue dropped
    extra = {"question": "Extra?", "positions": ["a", "b"], "queries": ["q1", "q2"],
             "statute_refs": []}  # fmt: skip
    out["issues"] += [copy.deepcopy(extra) for _ in range(3)]  # five issues in all
    return out


async def test_what_fails_the_checks_is_dropped_after_one_repair() -> None:
    ctx, provider = context(messy(), messy(), seed=seed_with([False] * 4))
    await IssueFramingStage().run(ctx)
    assert len(provider.calls) == 2  # the call and the repair
    feedback = provider.calls[1].messages[2].content
    assert "issues[0].positions[2]: only 2 are allowed" in feedback
    assert "issues[4]: at most 4 issues" in feedback
    view = ctx.record.view
    assert [i.question for i in view.issues.values()] == [PRESUMPTION, "Extra?", "Extra?"]
    assert len(view.issues["I-1"].positions) == 2
    assert [q.query for q in view.queries if q.issue_ids == ["I-1"]] == [
        "presumption undue influence near relations",
        "dominate the will",
        "s. 16 of the Indian Contract Act",
    ]
    failure = view.validation_failures[0]
    assert failure.agent == "issue_framer" and failure.stage is Stage.S0
    assert failure.dropped == (
        "issues[4], issues[1], issues[0].statute_refs[1], issues[0].queries[3], "
        "issues[0].queries[2], issues[0].positions[2]"
    )
    assert any("Indian Penal Code" in e for e in failure.errors)
    assert any(e == "issues[1]: an issue needs at least two search queries" for e in failure.errors)


async def test_a_repair_can_fix_the_framing() -> None:
    bad = copy.deepcopy(OUTPUT)
    bad["issues"][0]["statute_refs"] = ["s. 16 of the Contract Act, the undue influence one"]
    bad["issues"][1]["positions"] = ["On the donee.", "on the  donee."]
    ctx, provider = context(bad, OUTPUT)
    await IssueFramingStage().run(ctx)
    feedback = provider.calls[1].messages[2].content
    assert "issues[1].positions[1]: the position repeats an earlier one" in feedback
    assert "issues[1]: an issue needs two distinct positions" in feedback
    assert ctx.record.view.validation_failures == []
    assert len(ctx.record.view.issues) == 2


async def test_the_issue_limit_follows_the_profile() -> None:
    ctx, _ = context(OUTPUT, OUTPUT, params=DeliberationParams(max_issues=1))
    await IssueFramingStage().run(ctx)
    assert list(ctx.record.view.issues) == ["I-1"]


async def test_no_usable_issue_fails_the_stage() -> None:
    empty: dict[str, list[Any]] = {"facts": [], "assumptions": [], "issues": []}
    ctx, _ = context(empty, empty)
    with pytest.raises(FramingFailed, match="no usable issue"):
        await IssueFramingStage().run(ctx)
    assert ctx.record.view.validation_failures[0].dropped == "the whole output"


async def test_passthrough_frames_the_question_itself() -> None:
    ctx, provider = context()
    await IssueFramingStage().passthrough(ctx)
    view = ctx.record.view
    assert provider.calls == []
    assert view.issues["I-1"].question == ctx.run.item.question
    assert [p.statement for p in view.issues["I-1"].positions] == ["Yes.", "No."]
    assert [(q.query, q.kind) for q in view.queries] == [(ctx.run.item.question, "search")]


def test_the_framer_sees_only_the_question_and_facts() -> None:
    framer = IssueFramer("  ", order_seed=0)
    ctx, _ = context()
    fields = framer.input_view(ctx.record.view)
    assert fields["question"] == ctx.run.item.question
    assert fields["facts"] == "(none given beyond the question)"
