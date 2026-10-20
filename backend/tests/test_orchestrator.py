"""The orchestrator and the budget (PLAN 7.1): a stub pipeline S0-S10 through the experiment
runner, stage modes, budget skipping and failure handling. No model, no database."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from juris.config import (
    BudgetConfig,
    ModelsConfig,
    ModelSpec,
    PipelineProfile,
    Settings,
    StagesConfig,
    TokenPrice,
    resolve_run_config,
)
from juris.eval.juris_eval import load_split
from juris.eval.runner import ExperimentSpec, GatewayFactory, run_experiment
from juris.events import CaseView, parse_event
from juris.llm.cost import CallRecord, CostLedger
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend, ledger_spend
from juris.pipeline.case_record import InvariantViolation
from juris.pipeline.orchestrator import STAGES, Orchestrator, PipelineConfigError, StageContext
from tests.test_b1 import HITS, STATUTE_QUOTE, FakeService, Scripted
from tests.test_case_record import analysis, cite, verdict

Writer = Callable[[StageContext], Awaitable[None]]


async def frame(ctx: StageContext) -> None:
    ctx.record.frame_issue("Can the seller forfeit?", ["No.", "Yes."], stage=ctx.stage, agent="f")


async def research(ctx: StageContext) -> None:
    for hit in HITS:
        ctx.record.register_evidence(hit, ["I-1"], stage=ctx.stage, agent="researcher")


async def open_(ctx: StageContext) -> None:
    claim = ctx.record.add_claim("Capped.", "I-1.A", stage=ctx.stage, agent="counsel_a")
    ctx.record.add_argument(
        [claim.id], "s. 74", [cite(claim.id, "E-001", STATUTE_QUOTE)], stage=ctx.stage,
        agent="counsel_a",
    )  # fmt: skip


async def verify(ctx: StageContext) -> None:
    ctx.record.add_verification("C-001", "E-001", verdict(), stage=ctx.stage, agent="verifier")


async def judge(ctx: StageContext) -> None:
    ctx.record.set_analysis(analysis(), stage=ctx.stage, agent="judge")


WRITERS: dict[Stage, Writer] = {
    Stage.S0: frame,
    Stage.S1: research,
    Stage.S2: open_,
    Stage.S4: verify,
    Stage.S9: judge,
}


@dataclass
class Stub:
    """A stage that writes the minimum later stages need, and can spend tokens or fail."""

    stage: Stage
    optional: bool = False
    cost: Spend | None = None  # the estimate
    spend: int = 0  # tokens it records in the ledger, as a model call would
    fail: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def estimate(self, ctx: StageContext) -> Spend | None:
        return self.cost

    async def run(self, ctx: StageContext) -> None:
        self.calls.append("run")
        if self.spend:
            ctx.run.gateway.ledger.add(
                CallRecord(provider="fake", model="m", input_tokens=self.spend, output_tokens=0,
                           cost_usd=None, cached=False)
            )  # fmt: skip
        if self.fail is not None:
            raise self.fail
        if self.stage in WRITERS:
            await WRITERS[self.stage](ctx)

    async def passthrough(self, ctx: StageContext) -> None:
        self.calls.append("passthrough")


def stubs(**overrides: Stub) -> dict[Stage, Stub]:
    out = {stage: Stub(stage) for stage, _, _ in STAGES}
    for name, stub in overrides.items():
        out[Stage(name.upper())] = stub
    return out


@dataclass
class Result:
    status: str
    view: CaseView
    events: list[dict[str, Any]]


async def run(
    tmp_path: Path,
    runners: dict[Stage, Stub],
    *,
    stages: StagesConfig | None = None,
    budget: BudgetConfig | None = None,
) -> Result:
    item = load_split("dev")[0]
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    profile = PipelineProfile(
        name="stub", kind="juris", stages=stages or StagesConfig(), budget=budget
    )
    system = Orchestrator(
        list(runners.values()),
        lambda: FakeService(),
        corpus_snapshot_id="snap",
        embedding_model="none",
    )
    result = await run_experiment(
        ExperimentSpec("orch", "stub", "juris", "dev", (item.id,), (0,)),
        [item],
        resolve_run_config(settings, profile),
        system=system,
        gateways=GatewayFactory({"fake": Scripted({})}, prices={}),
        results_dir=tmp_path,
    )
    directory = tmp_path / "orch" / "runs" / f"{item.id}-s0"
    view = CaseView.model_validate_json((directory / "caseview.json").read_text("utf-8"))
    lines = (directory / "events.jsonl").read_text("utf-8").splitlines()
    return Result(result.outcomes[0].status, view, [json.loads(line) for line in lines])


def stage_events(events: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    return [
        (e["type"], e["stage"], e["payload"].get("mode") or e["payload"].get("outcome"))
        for e in events
        if e["type"] in ("stage_started", "stage_completed")
    ]


async def test_a_stub_pipeline_runs_s0_to_s10(tmp_path: Path) -> None:
    runners = stubs()
    r = await run(tmp_path, runners)
    assert r.status == "completed" and r.view.analysis is not None
    for e in r.events:
        parse_event(e)  # a valid event stream
    expected = []
    for stage, _, _ in STAGES:
        expected += [("stage_started", stage.value, "on"), ("stage_completed", stage.value,
                                                            "completed")]  # fmt: skip
    assert stage_events(r.events) == expected
    assert all(s.status == "completed" for s in r.view.stages.values())
    assert all(stub.calls == ["run"] for stub in runners.values())
    assert [e["type"] for e in r.events][-2:] == ["stage_completed", "run_completed"]


async def test_stages_can_be_off_or_passthrough(tmp_path: Path) -> None:
    runners = stubs()
    r = await run(
        tmp_path, runners, stages=StagesConfig(s6_cross_exam="off", s8_jury="passthrough")
    )
    assert r.status == "completed"
    assert runners[Stage.S6].calls == [] and runners[Stage.S8].calls == ["passthrough"]
    events = stage_events(r.events)
    assert ("stage_completed", "S6", "skipped") in events
    assert not any(t == "stage_started" and s == "S6" for t, s, _ in events)
    assert ("stage_started", "S8", "passthrough") in events
    assert ("stage_completed", "S8", "passthrough") in events
    assert r.view.stages[Stage.S6].reason == "off in the profile"
    assert r.view.stages[Stage.S8].status == "passthrough"


async def test_an_unimplemented_stage_fails_before_anything_runs(tmp_path: Path) -> None:
    runners = stubs()
    del runners[Stage.S3]
    r = await run(tmp_path, runners)
    assert r.status == "failed"
    assert r.view.error == "PipelineConfigError: no implementation for S3 (precedent analysis)"
    assert stage_events(r.events) == [] and runners[Stage.S0].calls == []
    off = await run(tmp_path / "off", stubs(), stages=StagesConfig(s3_precedent="off"))
    assert off.status == "completed"


def test_two_implementations_of_a_stage_are_refused() -> None:
    with pytest.raises(PipelineConfigError, match="two implementations of S2"):
        Orchestrator([Stub(Stage.S2), Stub(Stage.S2)], FakeService, corpus_snapshot_id="s",
                     embedding_model="e")  # fmt: skip


async def test_a_failing_stage_fails_the_run_and_keeps_the_record(tmp_path: Path) -> None:
    runners = stubs(s3=Stub(Stage.S3, fail=ValueError("boom")))
    r = await run(tmp_path, runners)
    assert r.status == "failed"
    assert r.view.error == "StageFailed: S3 precedent analysis: ValueError: boom"
    assert r.view.current_stage is Stage.S3 and r.view.stages[Stage.S3].status == "running"
    assert r.view.stages[Stage.S2].status == "completed" and "C-001" in r.view.claims
    assert runners[Stage.S4].calls == []


async def test_an_invariant_violation_names_the_stage(tmp_path: Path) -> None:
    error = InvariantViolation(["unknown claim 'C-404'"])
    r = await run(tmp_path, stubs(s5=Stub(Stage.S5, fail=error)))
    assert r.view.error == "StageFailed: S5 rebuttal: InvariantViolation: unknown claim 'C-404'"


async def test_a_pipeline_without_an_analysis_fails(tmp_path: Path) -> None:
    r = await run(tmp_path, stubs(), stages=StagesConfig(s9_judge="passthrough"))
    assert r.status == "failed" and "without an analysis" in (r.view.error or "")


async def test_the_budget_skips_optional_stages_only(tmp_path: Path) -> None:
    runners = stubs(
        s2=Stub(Stage.S2, spend=900),
        s5=Stub(Stage.S5, optional=True, cost=Spend(tokens=500)),  # does not fit the 100 left
        s6=Stub(Stage.S6, optional=True, cost=Spend(tokens=50)),  # fits
        s8=Stub(Stage.S8, optional=True),  # unknown estimate: runs while anything is left
    )
    r = await run(tmp_path, runners, budget=BudgetConfig(max_tokens_per_case=1000))
    assert r.status == "completed"
    assert runners[Stage.S5].calls == []
    assert runners[Stage.S6].calls == ["run"] and runners[Stage.S8].calls == ["run"]
    assert ("stage_completed", "S5", "skipped") in stage_events(r.events)
    assert r.view.stages[Stage.S5].reason == "budget"
    warning = r.view.budget_warnings[0]
    assert (warning.tokens_used, warning.tokens_budget, warning.action) == (
        900,
        1000,
        "skipped S5 (rebuttal)",
    )

    spent = stubs(
        s2=Stub(Stage.S2, spend=1000),
        s8=Stub(Stage.S8, optional=True),
        s9=Stub(Stage.S9, cost=Spend(tokens=10**6)),  # required: runs even over budget
    )
    r = await run(tmp_path / "spent", spent, budget=BudgetConfig(max_tokens_per_case=1000))
    assert spent[Stage.S8].calls == [] and spent[Stage.S9].calls == ["run"]
    assert r.status == "completed"


async def test_stages_warn_when_they_cut_their_own_steps(tmp_path: Path) -> None:
    class Rebuttal(Stub):
        async def run(self, ctx: StageContext) -> None:
            ctx.warn_budget("skipped rebuttal round 2")

    r = await run(tmp_path, stubs(s5=Rebuttal(Stage.S5)))
    events = [e for e in r.events if e["type"] == "budget_warning"]
    assert [(e["stage"], e["payload"]["action"]) for e in events] == [
        ("S5", "skipped rebuttal round 2")
    ]
    assert events[0]["payload"]["tokens_budget"] is None  # no limit set


# ---- the budget -----------------------------------------------------------------------------


def test_spend_counts_cached_calls_at_their_original_cost() -> None:
    ledger = CostLedger()
    for cached in (False, True):
        ledger.add(CallRecord(provider="p", model="m", input_tokens=1000, output_tokens=500,
                              cost_usd=None if cached else 0.02, cached=cached))  # fmt: skip
    ledger.add(CallRecord(provider="p", model="free", input_tokens=10, output_tokens=0,
                          cost_usd=None, cached=False))  # fmt: skip
    spend = ledger_spend(ledger, {"m": TokenPrice(input=10.0, output=20.0)})()
    assert spend.tokens == 3010
    assert spend.usd == pytest.approx(2 * (0.01 + 0.01))  # both calls priced; unpriced = 0
    assert ledger.total_tokens() == (1010, 500)  # the gateway's own totals: live calls only


def test_budget_manager() -> None:
    used = Spend()
    manager = BudgetManager(BudgetConfig(), lambda: used)
    assert not manager.limited and manager.remaining() == (None, None)
    assert manager.allows(Spend(10**9, 10**6)) and manager.allows(None)

    limited = BudgetManager(
        BudgetConfig(max_tokens_per_case=100, max_usd_per_case=1.0), lambda: Spend(60, 0.9)
    )
    assert limited.limited
    tokens, usd = limited.remaining()
    assert tokens == 40 and usd == pytest.approx(0.1)
    assert limited.allows(Spend(40, 0.05)) and limited.allows(None)
    assert not limited.allows(Spend(41, 0.0)) and not limited.allows(Spend(1, 0.2))
    over = BudgetManager(BudgetConfig(max_usd_per_case=1.0), lambda: Spend(0, 1.5))
    assert over.remaining() == (None, 0.0) and over.exhausted() and not over.allows(None)


# ---- parallel calls -------------------------------------------------------------------------


def context() -> StageContext:
    return StageContext(None, None, None, Stage.S2, "on")  # type: ignore[arg-type]


async def test_gather_keeps_order_and_runs_concurrently() -> None:
    started: list[int] = []

    async def call(n: int) -> int:
        started.append(n)
        await asyncio.sleep(0.01 * (3 - n))
        return n * 10

    assert await context().gather(call(1), call(2), call(3)) == [10, 20, 30]
    assert started == [1, 2, 3]


async def test_gather_cancels_the_rest_when_one_fails() -> None:
    cancelled = asyncio.Event()

    async def slow() -> int:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return 1

    async def broken() -> int:
        raise ValueError("counsel B failed")

    with pytest.raises(ValueError, match="counsel B failed"):
        await context().gather(slow(), broken())
    assert cancelled.is_set()
