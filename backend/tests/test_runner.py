"""Experiment runner and comparison (PLAN 5.5): the dummy config over 3 Juris-Eval dev items."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from juris.baselines.dummy import DummySystem, EchoProvider
from juris.config import ModelSpec, RunConfig, Settings, load_profile, resolve_run_config
from juris.eval.compare import (
    Experiment,
    comparison_table,
    load_experiment,
    paired,
    paired_table,
    sign_flip_p,
)
from juris.eval.juris_eval import JurisEvalItem, load_split
from juris.eval.metrics import Grader
from juris.eval.runner import (
    SYSTEMS,
    ExperimentMismatch,
    ExperimentSpec,
    GatewayFactory,
    run_experiment,
)
from juris.events.catalog import EVENT_TYPES, Event
from juris.events.fold import CaseView, fold
from juris.llm.cache import ResponseCache
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.llm.types import ProviderRequest, ProviderResponse


class CountingEcho(EchoProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        return await super().complete(request)


@pytest.fixture
def items() -> list[JurisEvalItem]:
    return load_split("dev")[:3]


@pytest.fixture
def config() -> RunConfig:
    return resolve_run_config(Settings(_env_file=None), load_profile("dummy"))  # type: ignore[call-arg]


def spec_for(items: list[JurisEvalItem], name: str = "dummy-dev", seeds: int = 1) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=name,
        profile="dummy",
        kind="dummy",
        split="dev",
        item_ids=tuple(i.id for i in items),
        seeds=tuple(range(seeds)),
    )


def factory(tmp_path: Path, echo: EchoProvider) -> GatewayFactory:
    return GatewayFactory(
        {"dummy": echo}, prices={}, cache=ResponseCache(tmp_path / "cache.sqlite")
    )


async def test_dummy_experiment_end_to_end(
    tmp_path: Path, items: list[JurisEvalItem], config: RunConfig
) -> None:
    echo = CountingEcho()
    spec = spec_for(items)
    result = await run_experiment(
        spec,
        items,
        config,
        system=DummySystem(),
        gateways=factory(tmp_path, echo),
        results_dir=tmp_path,
    )
    assert [o.status for o in result.outcomes] == ["completed"] * 3
    assert echo.calls == 3

    run = tmp_path / "dummy-dev" / "runs" / f"{items[0].id}-s0"
    assert {p.name for p in run.iterdir()} == {
        "manifest.json",
        "events.jsonl",
        "caseview.json",
        "calls.json",
        "scores.json",
    }
    manifest = json.loads((run / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "completed" and manifest["profile"] == "dummy"
    assert manifest["totals"]["llm_calls"] == 1 and manifest["seeds"]["item"] == 0
    assert any(k.startswith("grader_key_points@v") for k in manifest["prompt_hashes"])

    adapter: TypeAdapter[Event] = TypeAdapter(Event)
    events = [
        adapter.validate_json(line)
        for line in (run / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert events[0].type == "case_created" and events[-1].type == "run_completed"
    assert {e.type for e in events} <= set(EVENT_TYPES)
    view = CaseView.model_validate_json((run / "caseview.json").read_text("utf-8"))
    assert fold(events) == view and view.status == "completed" and view.analysis is not None

    scores = json.loads((run / "scores.json").read_text("utf-8"))
    values = scores["values"]
    assert scores["status"] == "completed"
    assert values["citation_validity"] == 1.0 and values["citation_faithfulness"] == 1.0
    assert values["authority_recall.supporting"] == 1.0
    # The dummy cites evidence only for I-1.A; every other position summary is uncited.
    positions = [f"I-{n}.{s}" for n in range(1, len(items[0].gold_issues[:4]) + 1) for s in "AB"]
    unsupported = scores["details"]["unsupported_claim_rate"]["unsupported"]
    assert [u["where"] for u in unsupported] == positions[1:]
    assert values["key_point_coverage"] is None  # no grader
    assert values["cost_latency.llm_calls"] == 1

    summary = result.summary
    assert summary["runs"] == 3 and summary["failed_runs"] == 0
    assert summary["aggregate"]["answered"]["mean"] == 1.0
    assert summary["aggregate"]["answered"]["n"] == 3
    assert summary["aggregate"]["quality_rubric"] is None
    assert "| citation_validity | 1.000 | 1.000-1.000 | 3 |" in (
        tmp_path / "dummy-dev" / "summary.md"
    ).read_text("utf-8")
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1 and rows[0]["experiment_id"] == "dummy-dev"
    assert rows[0]["runs"] == "3" and rows[0]["answered"] == "1"


async def test_resume_skips_finished_runs(
    tmp_path: Path, items: list[JurisEvalItem], config: RunConfig
) -> None:
    echo = CountingEcho()
    spec = spec_for(items[:2])
    await run_experiment(
        spec,
        items[:2],
        config,
        system=DummySystem(),
        gateways=factory(tmp_path, echo),
        results_dir=tmp_path,
    )
    # A crash mid-run leaves a run without scores.json: only that one runs again.
    scores_path = tmp_path / "dummy-dev" / "runs" / f"{items[1].id}-s0" / "scores.json"
    before = json.loads(scores_path.read_text("utf-8"))["values"]
    scores_path.unlink()
    again = await run_experiment(
        spec,
        items[:2],
        config,
        system=DummySystem(),
        gateways=factory(tmp_path, echo),
        results_dir=tmp_path,
    )
    assert [o.status for o in again.outcomes] == ["skipped", "completed"]
    assert echo.calls == 2  # the re-run call was answered from the cache
    after = json.loads(scores_path.read_text("utf-8"))["values"]
    assert after["cost_latency.cached_calls"] == 1 and before["cost_latency.cached_calls"] == 0
    assert after["cost_latency.total_tokens"] == before["cost_latency.total_tokens"] > 0
    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 1  # upserted, not appended

    with pytest.raises(ExperimentMismatch):
        await run_experiment(
            spec_for(items[:2], seeds=2),
            items[:2],
            config,
            system=DummySystem(),
            gateways=factory(tmp_path, echo),
            results_dir=tmp_path,
        )


async def test_failed_runs_are_recorded_and_retried(
    tmp_path: Path, items: list[JurisEvalItem], config: RunConfig
) -> None:
    echo = CountingEcho()
    spec = spec_for(items)
    failing = DummySystem(fail_items={items[1].id})
    result = await run_experiment(
        spec,
        items,
        config,
        system=failing,
        gateways=factory(tmp_path, echo),
        results_dir=tmp_path,
    )
    assert [o.status for o in result.outcomes] == ["completed", "failed", "completed"]
    run = tmp_path / "dummy-dev" / "runs" / f"{items[1].id}-s0"
    manifest = json.loads((run / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "failed" and "dummy failure" in manifest["error"]
    view = CaseView.model_validate_json((run / "caseview.json").read_text("utf-8"))
    assert view.status == "failed" and view.analysis is None
    assert result.summary["failed_runs"] == 1
    assert result.summary["aggregate"]["answered"]["mean"] == pytest.approx(2 / 3)

    retried = await run_experiment(
        spec,
        items,
        config,
        system=DummySystem(),
        gateways=factory(tmp_path, echo),
        results_dir=tmp_path,
        retry_failed=True,
    )
    assert [o.status for o in retried.outcomes] == ["skipped", "completed", "skipped"]
    assert retried.summary["failed_runs"] == 0


async def test_seeds_are_averaged_per_item(
    tmp_path: Path, items: list[JurisEvalItem], config: RunConfig
) -> None:
    spec = spec_for(items[:1], seeds=2)
    result = await run_experiment(
        spec,
        items[:1],
        config,
        system=DummySystem(),
        gateways=factory(tmp_path, EchoProvider()),
        results_dir=tmp_path,
    )
    assert result.summary["runs"] == 2
    assert list(result.summary["per_item"]) == [items[0].id]
    assert result.summary["aggregate"]["answered"]["n"] == 1


async def test_grader_values_are_stored(
    tmp_path: Path, items: list[JurisEvalItem], config: RunConfig
) -> None:
    item = items[0]
    key_points = json.dumps(
        {
            "judgements": [
                {"index": n, "verdict": "covered", "reason": "Yes."}
                for n in range(1, len(item.key_points) + 1)
            ]
        }
    )
    rubric = json.dumps(
        {
            c: {"score": 3, "reason": "Fair."}
            for c in (
                "correctness",
                "completeness",
                "reasoning_consistency",
                "uncertainty_calibration",
            )
        }
    )
    fake = FakeProvider({"grader_key_points": [key_points], "grader_quality_rubric": [rubric]})
    grader = Grader(
        LLMGateway({"fake": fake}, prices={}, cache=None, cache_mode="off"),
        ModelSpec(provider="fake", name="grader", effort=None),
    )
    result = await run_experiment(
        spec_for([item]),
        [item],
        config,
        system=DummySystem(),
        gateways=factory(tmp_path, EchoProvider()),
        grader=grader,
        results_dir=tmp_path,
    )
    values = result.summary["per_item"][item.id]
    assert values["key_point_coverage"] == 1.0 and values["quality_rubric"] == 3.0
    assert len(fake.calls) == 2  # every summary sentence cites evidence: no assertion call


def test_unbuilt_systems_say_when_they_arrive() -> None:
    with pytest.raises(NotImplementedError, match=r"PLAN 6\.3"):
        SYSTEMS["b1"]()


# ---- comparison ----------------------------------------------------------------------------


def test_sign_flip_p() -> None:
    assert sign_flip_p([1.0, 1.0, 1.0]) == pytest.approx(2 / 8)  # all + or all -
    assert sign_flip_p([0.0, 0.0]) == 1.0
    assert sign_flip_p([0.5, -0.5, 0.1, -0.1]) == 1.0
    many = [0.2] * 30
    assert sign_flip_p(many) == pytest.approx(1 / 20_001)  # the random path, seeded
    assert sign_flip_p([0.2, -0.1] * 15) == sign_flip_p([0.2, -0.1] * 15)


def test_paired_comparison() -> None:
    a = Experiment("a", {}, {"x": {"m": 0.5}, "y": {"m": 0.5}, "z": {"m": None}})
    b = Experiment("b", {}, {"x": {"m": 1.0}, "y": {"m": 0.75}, "z": {"m": 1.0}, "w": {"m": 0}})
    r = paired(a, b, "m")
    assert r is not None
    assert (r.n, r.mean_a, r.mean_b, r.diff) == (2, 0.5, 0.875, 0.375)
    assert r.p == pytest.approx(2 / 4)
    assert paired(a, b, "missing") is None
    table = paired_table(a, b, ["m", "missing"])
    assert "| m | 2 | 0.500 | 0.875 | +0.375 |" in table
    assert "| missing | 0 | n/a |" in table


async def test_compare_two_experiments(
    tmp_path: Path, items: list[JurisEvalItem], config: RunConfig
) -> None:
    for name, system in (("good", DummySystem()), ("bad", DummySystem(fail_items={items[0].id}))):
        await run_experiment(
            spec_for(items, name),
            items,
            config,
            system=system,
            gateways=factory(tmp_path, EchoProvider()),
            results_dir=tmp_path,
        )
    good, bad = load_experiment(tmp_path / "good"), load_experiment(tmp_path / "bad")
    table = comparison_table([good, bad], ["answered", "citation_validity"])
    assert table.splitlines()[0] == "| Experiment | Items | answered | citation_validity |"
    assert "| good | 3 | 1.000 (1.000-1.000) | 1.000 (1.000-1.000) |" in table
    assert table.splitlines()[3].startswith("| bad | 3 | 0.667 (")
    paired_md = paired_table(good, bad, ["answered"])
    assert "| answered | 3 | 1.000 | 0.667 | -0.333 |" in paired_md
