"""The experiment runner (PLAN 5.5): one config over one Juris-Eval split, scored and stored.

    uv run scripts/run_experiment.py --config b1 --split dev --limit 20 --seeds 1

Layout under ``eval/results/`` (``results_dir``)::

    <experiment_id>/experiment.json            the spec: profile, split, items, seeds, grader
    <experiment_id>/runs/<item>-s<seed>/       one run per item and seed:
        manifest.json  events.jsonl  caseview.json  calls.json  scores.json
    <experiment_id>/summary.json, summary.md   per-item values and means with 95% CIs
    results.csv                                one summary row per experiment

Runs are independent and limited to ``concurrency`` at a time. Each run gets its own gateway
(its own cost ledger, the shared response cache). ``scores.json`` is written last, so a run
without it is unfinished and a re-run of the same experiment does only those (resumable).
"""

import asyncio
import csv
import json
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from juris.baselines.b0 import B0System
from juris.baselines.base import RunContext, System
from juris.baselines.dummy import DummySystem, EchoProvider
from juris.config import REPO_ROOT, ModelSpec, RunConfig, Settings, TokenPrice
from juris.eval.juris_eval import JurisEvalItem
from juris.eval.metrics import Answer, Grader, MetricResult
from juris.eval.metrics.suite import score_answer
from juris.eval.stats import aggregate
from juris.events.catalog import CaseCreatedPayload, RunCompletedPayload, RunFailedPayload
from juris.events.fold import fold
from juris.events.store import EventEmitter, InMemoryEventStore
from juris.llm.cache import ResponseCache
from juris.llm.gateway import LLMGateway
from juris.llm.providers.anthropic import AnthropicProvider
from juris.llm.providers.base import Provider
from juris.llm.providers.openai_compat import OpenAICompatProvider
from juris.runs import RunRecorder, git_state

RESULTS_DIR = REPO_ROOT / "eval" / "results"
RESULTS_TABLE = "results.csv"

# Values shown in summaries, results.csv and comparisons (all others stay in summary.json).
HEADLINE = (
    "answered",
    "citation_validity",
    "citation_faithfulness",
    "unsupported_claim_rate",
    "authority_recall",
    "key_point_coverage",
    "quality_rubric",
    "cost_latency.total_tokens",
    "cost_latency.usd",
    "cost_latency.wall_seconds",
)


def _not_yet(step: str) -> Callable[[], System]:
    def make() -> System:
        raise NotImplementedError(f"this config's system arrives in PLAN {step}")

    return make


SYSTEMS: dict[str, Callable[[], System]] = {
    "dummy": DummySystem,
    "b0": B0System,
    "b1": _not_yet("6.3"),
    "b2": _not_yet("6.4"),
    "juris": _not_yet("7.12"),
}


class ExperimentMismatch(ValueError):
    """The experiment ID exists with a different spec; use a new ID."""


@dataclass
class GatewayFactory:
    """Makes one gateway per run: shared providers and cache, a fresh cost ledger."""

    providers: Mapping[str, Provider]
    prices: Mapping[str, TokenPrice]
    cache: ResponseCache | None = None
    cache_mode: str = "read_write"
    max_concurrency: int = 4

    @classmethod
    def from_settings(cls, settings: Settings) -> "GatewayFactory":
        return cls(
            providers={
                "anthropic": AnthropicProvider(settings.anthropic_api_key),
                "openai": OpenAICompatProvider(settings.openai_api_key, settings.openai_base_url),
                "dummy": EchoProvider(),
            },
            prices=settings.llm_prices,
            cache=None if settings.llm_cache == "off" else ResponseCache(settings.llm_cache_path),
            cache_mode=settings.llm_cache,
            max_concurrency=settings.llm_max_concurrency,
        )

    def __call__(self) -> LLMGateway:
        return LLMGateway(
            self.providers,
            prices=self.prices,
            cache=self.cache,
            cache_mode=self.cache_mode,  # type: ignore[arg-type]
            max_concurrency=self.max_concurrency,
        )


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    profile: str
    kind: str
    split: str
    item_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    grader: ModelSpec | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "profile": self.profile,
            "kind": self.kind,
            "split": self.split,
            "item_ids": list(self.item_ids),
            "seeds": list(self.seeds),
            "grader": self.grader.model_dump() if self.grader else None,
        }


def default_experiment_id(profile: str, split: str, limit: int | None, n_seeds: int) -> str:
    return f"{profile}-{split}-n{limit if limit else 'all'}-s{n_seeds}"


@dataclass
class RunOutcome:
    item_id: str
    seed: int
    status: str  # "completed", "failed" or "skipped" (finished earlier)


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    directory: Path
    outcomes: list[RunOutcome] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, data: Any) -> None:
    """Written to a temporary file first, so a crash never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def run_dir(experiment_dir: Path, item_id: str, seed: int) -> Path:
    return experiment_dir / "runs" / f"{item_id}-s{seed}"


async def run_one(
    system: System,
    item: JurisEvalItem,
    seed: int,
    *,
    config: RunConfig,
    experiment_dir: Path,
    gateways: Callable[[], LLMGateway],
    grader: Grader | None,
    prices: Mapping[str, TokenPrice] | None = None,
) -> str:
    """Run, store and score one item with one seed; returns the run's status."""
    directory = run_dir(experiment_dir, item.id, seed)
    gateway = gateways()
    store = InMemoryEventStore()
    run_id = directory.name
    emitter = EventEmitter(store, run_id=run_id, case_id=item.id)
    recorder = RunRecorder(
        config,
        corpus_snapshot_id=system.corpus_snapshot_id,
        embedding_model=system.embedding_model,
        ledger=gateway.ledger,
        runs_dir=directory.parent,
        seeds={"item": seed},
        run_id=run_id,
    )
    ctx = RunContext(item, config, gateway, emitter, seed)
    try:
        with recorder:
            emitter.emit(
                CaseCreatedPayload(
                    question=item.question,
                    profile=config.profile.name,
                    corpus_snapshot_id=system.corpus_snapshot_id,
                )
            )
            await system.run(ctx)
    except Exception as exc:  # a failed run is recorded and scored, not raised
        emitter.emit(RunFailedPayload(reason=f"{type(exc).__name__}: {exc}"))
    else:
        emitter.emit(RunCompletedPayload(totals=recorder.manifest.totals))

    events = store.read(run_id)
    view = fold(events)
    (directory / "events.jsonl").write_text(
        "".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8"
    )
    _write_json(directory / "caseview.json", view.model_dump(mode="json"))
    _write_json(
        directory / "calls.json", [r.model_dump(mode="json") for r in gateway.ledger.records]
    )
    answer = Answer(item, view, calls=gateway.ledger.records, prices=prices)
    scores = await score_answer(answer, grader=grader)
    if ctx.values or ctx.details:  # values only the system can compute (base.RunContext)
        scores.results["system"] = MetricResult("system", dict(ctx.values), dict(ctx.details))
    _write_json(directory / "scores.json", {"status": view.status, "seed": seed} | scores.to_json())
    return view.status


def load_scores(experiment_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((experiment_dir / "runs").glob("*/scores.json"))
    ]


def per_item_values(scores: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Each item's values averaged over its seeds (``None`` where no seed has a value)."""
    by_item: dict[str, list[dict[str, float | None]]] = {}
    for s in scores:
        by_item.setdefault(s["item_id"], []).append(s["values"])
    out: dict[str, dict[str, float | None]] = {}
    for item_id, runs in sorted(by_item.items()):
        names = dict.fromkeys(k for r in runs for k in r)
        out[item_id] = {}
        for name in names:
            present = [v for r in runs if (v := r.get(name)) is not None]
            out[item_id][name] = statistics.mean(present) if present else None
    return out


def summarise(spec: ExperimentSpec, experiment_dir: Path) -> dict[str, Any]:
    scores = load_scores(experiment_dir)
    items = per_item_values(scores)
    names = list(dict.fromkeys(k for values in items.values() for k in values))
    aggregates = {}
    for name in names:
        agg = aggregate([values.get(name) for values in items.values()])
        aggregates[name] = None if agg is None else vars(agg)
    return {
        "spec": spec.to_json(),
        "runs": len(scores),
        "failed_runs": sum(1 for s in scores if s["status"] != "completed"),
        "aggregate": aggregates,
        "per_item": items,
    }


def summary_table(summary: Mapping[str, Any], names: Sequence[str] = HEADLINE) -> str:
    rows = ["| Metric | Mean | 95% CI | n |", "|---|---|---|---|"]
    for name in names:
        agg = summary["aggregate"].get(name)
        if agg is None:
            rows.append(f"| {name} | n/a | n/a | 0 |")
        else:
            rows.append(
                f"| {name} | {agg['mean']:.3f} | {agg['lo']:.3f}-{agg['hi']:.3f} | {agg['n']} |"
            )
    return "\n".join(rows)


def update_results_table(results_dir: Path, summary: Mapping[str, Any]) -> None:
    """Upsert this experiment's row in ``results.csv`` (keyed by experiment ID)."""
    path = results_dir / RESULTS_TABLE
    spec = summary["spec"]
    git = git_state()
    row: dict[str, Any] = {
        "experiment_id": spec["experiment_id"],
        "profile": spec["profile"],
        "kind": spec["kind"],
        "split": spec["split"],
        "items": len(spec["item_ids"]),
        "seeds": len(spec["seeds"]),
        "runs": summary["runs"],
        "failed_runs": summary["failed_runs"],
        "grader": spec["grader"]["name"] if spec["grader"] else "",
        "git_commit": git.commit[:12],
        "git_dirty": git.dirty,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    for name in HEADLINE:
        agg = summary["aggregate"].get(name)
        row[name] = "" if agg is None else f"{agg['mean']:.6g}"
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["experiment_id"] != row["experiment_id"]]
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


async def run_experiment(
    spec: ExperimentSpec,
    items: Sequence[JurisEvalItem],
    config: RunConfig,
    *,
    system: System,
    gateways: Callable[[], LLMGateway],
    grader: Grader | None = None,
    results_dir: Path = RESULTS_DIR,
    concurrency: int = 2,
    retry_failed: bool = False,
    prices: Mapping[str, TokenPrice] | None = None,
) -> ExperimentResult:
    """``prices`` price every call at list price, cache hits included (``cost_latency``)."""
    by_id = {i.id: i for i in items}
    if [i.id for i in items] != list(spec.item_ids):
        raise ValueError("items must match the spec's item IDs, in order")
    directory = results_dir / spec.experiment_id
    spec_path = directory / "experiment.json"
    if spec_path.exists():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        if existing != spec.to_json():
            raise ExperimentMismatch(f"{spec.experiment_id} exists with another spec")
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(spec_path, spec.to_json())

    result = ExperimentResult(spec, directory)
    limit = asyncio.Semaphore(concurrency)

    async def one(item_id: str, seed: int) -> RunOutcome:
        done = run_dir(directory, item_id, seed) / "scores.json"
        if done.exists():
            status = json.loads(done.read_text(encoding="utf-8"))["status"]
            if not (retry_failed and status == "failed"):
                return RunOutcome(item_id, seed, "skipped")
        async with limit:
            status = await run_one(
                system,
                by_id[item_id],
                seed,
                config=config,
                experiment_dir=directory,
                prices=prices,
                gateways=gateways,
                grader=grader,
            )
        return RunOutcome(item_id, seed, status)

    result.outcomes = list(
        await asyncio.gather(*(one(i, s) for i in spec.item_ids for s in spec.seeds))
    )
    result.summary = summarise(spec, directory)
    _write_json(directory / "summary.json", result.summary)
    (directory / "summary.md").write_text(
        f"# {spec.experiment_id}\n\n{summary_table(result.summary)}\n", encoding="utf-8"
    )
    update_results_table(results_dir, result.summary)
    return result
