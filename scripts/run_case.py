"""Run one question through the Juris pipeline and print the analysis (PLAN 7.12).

    uv run scripts/run_case.py "A buyer paid earnest money ... Can the seller forfeit it?"
    uv run scripts/run_case.py --item JE-001 [--config no_jury]

Options:
    --config NAME    a profile in configs/pipeline/ (default juris_full; an ablation works too)
    --item ID        a Juris-Eval item's question instead of free text
    --seed N         default 0
    --out DIR        default <runs_dir>/cases/<timestamp>

Writes to the output directory: ``manifest.json`` (config, models, git state, call totals),
``events.jsonl`` (the event log), ``caseview.json`` (the folded record), ``analysis.json`` (the
``CaseAnalysis``) and ``calls.json`` (every model call). Prints the issues with their leanings
and confidence, the overall summary, the limitations, and the tokens and time it took. Nothing
is scored: free questions have no gold answer.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from juris.baselines.base import RunContext
from juris.config import get_settings, load_profile, resolve_run_config
from juris.eval.juris_eval import JurisEvalItem, free_item, load_split
from juris.eval.runner import GatewayFactory
from juris.events.catalog import CaseCreatedPayload, RunCompletedPayload, RunFailedPayload
from juris.events.fold import fold
from juris.events.store import EventEmitter, InMemoryEventStore
from juris.pipeline.system import JurisSystem
from juris.runs import RunRecorder


def find_item(item_id: str) -> JurisEvalItem:
    for split in ("dev", "test"):
        for item in load_split(split):
            if item.id == item_id:
                return item
    raise SystemExit(f"no Juris-Eval item {item_id}")


async def run(item: JurisEvalItem, config_name: str, seed: int, out: Path) -> int:
    settings = get_settings()
    profile = load_profile(config_name)
    if profile.kind != "juris":
        raise SystemExit(f"{config_name} is a {profile.kind} profile, not a Juris one")
    config = resolve_run_config(settings, profile)
    system = JurisSystem()
    gateway = GatewayFactory.from_settings(settings)()
    store = InMemoryEventStore()
    emitter = EventEmitter(store, run_id=out.name, case_id=item.id)
    out.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder(
        config,
        corpus_snapshot_id=system.corpus_snapshot_id,
        embedding_model=system.embedding_model,
        ledger=gateway.ledger,
        runs_dir=out.parent,
        seeds={"item": seed},
        run_id=out.name,
    )
    ctx = RunContext(item, config, gateway, emitter, seed)
    started = time.perf_counter()
    failure = None
    try:
        with recorder:
            emitter.emit(
                CaseCreatedPayload(
                    question=item.question,
                    profile=profile.name,
                    corpus_snapshot_id=system.corpus_snapshot_id,
                )
            )
            await system.run(ctx)
    except Exception as exc:  # recorded, then reported
        failure = f"{type(exc).__name__}: {exc}"
        emitter.emit(RunFailedPayload(reason=failure))
    else:
        emitter.emit(RunCompletedPayload(totals=recorder.manifest.totals))
    seconds = time.perf_counter() - started
    events = store.read(out.name)
    view = fold(events)
    (out / "events.jsonl").write_text(
        "".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8"
    )
    (out / "caseview.json").write_text(view.model_dump_json(indent=1), encoding="utf-8")
    (out / "calls.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in gateway.ledger.records], indent=1),
        encoding="utf-8",
    )
    analysis = view.analysis
    if analysis is not None:
        (out / "analysis.json").write_text(analysis.model_dump_json(indent=1), encoding="utf-8")
    tokens = sum(r.input_tokens + r.output_tokens for r in gateway.ledger.records)
    cached = sum(r.cached for r in gateway.ledger.records)
    print(f"Question: {item.question}\n")
    if analysis is None:
        print(f"No analysis: {failure or 'the pipeline ended without one'}")
    else:
        for issue in analysis.issues:
            print(f"{issue.issue_id}: {issue.issue_text}")
            print(f"  leaning {issue.leaning.value}, confidence {issue.confidence.value}: "
                  f"{issue.confidence_reason}")  # fmt: skip
            for p in issue.positions:
                print(f"  {p.position_id}: {p.summary}")
        print(f"\n{analysis.overall_summary}\n")
        print("Limitations:")
        for line in analysis.limitations:
            print(f"- {line}")
        print(f"\n{analysis.disclaimer}")
    print(f"\n{len(gateway.ledger.records)} model calls ({cached} cached), {tokens:,} tokens, "
          f"{seconds:.0f} s. Written to {out}")  # fmt: skip
    return 0 if analysis is not None else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--item")
    ap.add_argument("--config", default="juris_full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if bool(args.question) == bool(args.item):
        ap.error("give a question or --item, not both")
    item = find_item(args.item) if args.item else free_item(args.question)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or get_settings().runs_dir / "cases" / f"{item.id}-{stamp}"
    return asyncio.run(run(item, args.config, args.seed, out))


if __name__ == "__main__":
    sys.exit(main())
