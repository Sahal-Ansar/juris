"""Measure the S1 evidence pool's recall of gold authorities (PLAN 7.4), against B1's.

    uv run scripts/eval_researcher.py run [--split dev] [--limit N]
    uv run scripts/eval_researcher.py report

``run`` frames each item with the Issue Framer (S0; the configured model, cached) and builds
its pool with the Researcher (S1; the local hybrid retriever on the CPU). For every item it
records which gold documents and sections the pool holds. Results go to
``eval/results/researcher/s1-<split>.json``.

``report`` compares the pools as recall per item, averaged over items:

- supporting: gold supporting judgments whose document is in the pool;
- all gold: supporting and contrary judgments;
- sections: gold statute sections with a chunk in the pool.

B1's pool is the 20 chunks of each run of ``b1-v2-dev-nall-s1`` (6.3).
"""

import argparse
import asyncio
import json
import sys
from statistics import mean
from typing import Any

from juris.baselines.b1 import B1System
from juris.baselines.base import RunContext
from juris.config import REPO_ROOT, PipelineProfile, get_settings, resolve_run_config
from juris.eval.juris_eval import JurisEvalItem, load_split
from juris.eval.runner import GatewayFactory
from juris.events import CaseView, EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.ingest.statutes import load_acts, resolve_act
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.research import ResearchStage

RESULTS = REPO_ROOT / "eval" / "results"
OUT = RESULTS / "researcher"
B1_EXPERIMENT = "b1-v2-dev-nall-s1"


def pool(view: CaseView) -> tuple[set[str], set[str]]:
    """The documents and the statute section IDs a run's evidence holds."""
    acts = load_acts()
    documents = {e.document_id for e in view.evidence.values()}
    sections: set[str] = set()
    for e in view.evidence.values():
        statute = view.chunks[e.chunk_id].statute
        act_id = resolve_act(statute.act, acts) if statute else None
        if statute and act_id:
            sections.add(f"{act_id}:{statute.section}")
    return documents, sections


def recall(item: JurisEvalItem, documents: set[str], sections: set[str]) -> dict[str, Any]:
    supporting = [a.doc_id for a in item.gold_supporting_authorities]
    every = supporting + [a.doc_id for a in item.gold_contrary_authorities]
    return {
        "supporting": sum(d in documents for d in supporting) / len(supporting),
        "all_gold": sum(d in documents for d in every) / len(every),
        "sections": sum(s in sections for s in item.gold_sections) / len(item.gold_sections),
        "found": sorted(d for d in every if d in documents),
        "sections_found": sorted(s for s in item.gold_sections if s in sections),
    }


async def run(split: str, limit: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s1-eval", kind="juris")
    config = resolve_run_config(settings, profile)
    gateways = GatewayFactory.from_settings(settings)
    service = RetrievalService.local(device="cpu")  # the GPU belongs to the local LLM
    research = ResearchStage(lambda: service)
    items = load_split(split)[:limit]
    rows = []
    for item in items:
        gateway = gateways()
        emitter = EventEmitter(InMemoryEventStore(), run_id=f"{item.id}-s0", case_id=item.id)
        emitter.emit(
            CaseCreatedPayload(
                question=item.question,
                profile=profile.name,
                corpus_snapshot_id=B1System.corpus_snapshot_id,
            )
        )
        record = CaseRecord(emitter, service, params=profile.params)
        run_ctx = RunContext(item, config, gateway, emitter, 0)
        budget = BudgetManager(config.budget, Spend)
        await IssueFramingStage().run(StageContext(run_ctx, record, budget, Stage.S0, "on"))
        await research.run(StageContext(run_ctx, record, budget, Stage.S1, "on"))
        view = record.view
        documents, sections = pool(view)
        rows.append(
            {
                "item": item.id,
                "issues": len(view.issues),
                "evidence": len(view.evidence),
                "documents": len(documents),
                "values": run_ctx.values,
                "pools": run_ctx.details["s1.pools"],
                **recall(item, documents, sections),
            }
        )
        print(item.id, rows[-1]["evidence"], rows[-1]["supporting"], rows[-1]["sections"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s1-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def b1_rows(split: str) -> list[dict[str, Any]]:
    items = {i.id: i for i in load_split(split)}
    rows = []
    for path in sorted((RESULTS / B1_EXPERIMENT / "runs").glob("*/caseview.json")):
        view = CaseView.model_validate_json(path.read_text(encoding="utf-8"))
        documents, sections = pool(view)
        rows.append(
            {
                "item": view.case_id,
                "evidence": len(view.evidence),
                "documents": len(documents),
                **recall(items[view.case_id], documents, sections),
            }
        )
    return rows


def report(split: str) -> int:
    s1 = json.loads((OUT / f"s1-{split}.json").read_text(encoding="utf-8"))
    b1 = {r["item"]: r for r in b1_rows(split)}
    both = [r for r in s1 if r["item"] in b1]
    print(f"{len(both)} items in both\n")
    print("| Pool | evidence | documents | supporting recall | all-gold recall | section recall |")
    print("|---|---|---|---|---|---|")
    for name, rows in (("B1 (20 chunks)", [b1[r["item"]] for r in both]), ("S1 Researcher", both)):
        cells = [mean(r[k] for r in rows) for k in ("evidence", "documents")]
        rates = [mean(r[k] for r in rows) for k in ("supporting", "all_gold", "sections")]
        print(
            f"| {name} | {cells[0]:.1f} | {cells[1]:.1f} | "
            + " | ".join(f"{x:.3f}" for x in rates)
            + " |"
        )
    print("\n| Item | B1 supporting | S1 supporting | B1 sections | S1 sections |")
    print("|---|---|---|---|---|")
    for r in both:
        o = b1[r["item"]]
        print(f"| {r['item']} | {o['supporting']:.2f} | {r['supporting']:.2f} | "
              f"{o['sections']:.2f} | {r['sections']:.2f} |")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    if args.command == "run":
        return asyncio.run(run(args.split, args.limit))
    return report(args.split)


if __name__ == "__main__":
    sys.exit(main())
