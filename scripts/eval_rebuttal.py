"""Check S5's rebuttal round on dev (PLAN 7.7: each counterargument targets a valid claim; new
evidence from targeted searches is registered and verified at S7).

    uv run scripts/eval_rebuttal.py run [--split dev] [--limit N] [--max-evidence-chars 28000]
    uv run scripts/eval_rebuttal.py report [--split dev]
    uv run scripts/eval_rebuttal.py sample [--n 10] [--seed 0]

``run`` takes each item through S0-S5 and the S7 gate with the configured model (the cache
makes S0-S4 free after the earlier scripts, with the same evidence budget). Per item it keeps
S5's counts (``s5.*``, see ``pipeline/stages/rebuttal.py``), S7's, and every counterargument:
its target, its claim and citations, and each citation's S7 result. Results go to
``eval/results/rebuttal/s5-<split>.json``.

``report`` pools the counts. ``sample`` prints a seeded sample of counterarguments for a manual
check.
"""

import argparse
import asyncio
import json
import random
import sys
from typing import Any

from juris.baselines.b1 import B1System
from juris.baselines.base import RunContext
from juris.config import REPO_ROOT, PipelineProfile, get_settings, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.eval.runner import GatewayFactory
from juris.events import CaseView, EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext, StageRunner
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.precedent import PrecedentStage
from juris.pipeline.stages.rebuttal import RebuttalStage
from juris.pipeline.stages.research import ResearchStage
from juris.pipeline.stages.verify import VerificationGate

OUT = REPO_ROOT / "eval" / "results" / "rebuttal"


def counters(view: CaseView) -> list[dict[str, Any]]:
    """Every counterargument with its target and its citations' latest results."""
    results = {(v.claim_id, v.evidence_id): v.result for v in view.verifications}
    out = []
    for r in view.counterarguments.values():
        argument = view.arguments[r.argument_id]
        claim = view.claims[argument.claim_ids[0]]
        target = view.claims[r.targets]
        out.append(
            {
                "id": r.id,
                "by": claim.position_id,
                "target": {"id": target.id, "status": target.status.value, "text": target.text},
                "claim": {"id": claim.id, "status": claim.status.value, "text": claim.text},
                "reasoning": argument.reasoning,
                "citations": [
                    {
                        "evidence": c.evidence_id,
                        "found_in_s5": view.evidence[c.evidence_id].created_by.stage is Stage.S5,
                        "quote": c.quote,
                        "s7": (
                            results[(c.claim_id, c.evidence_id)].status.value
                            if (c.claim_id, c.evidence_id) in results
                            else None
                        ),
                    }
                    for c in argument.citations
                ],
            }
        )
    return out


async def run(split: str, limit: int | None, max_evidence_chars: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s5-eval", kind="juris")
    config = resolve_run_config(settings, profile)
    gateways = GatewayFactory.from_settings(settings)
    service = RetrievalService.local(device="cpu")  # the GPU belongs to the local LLM
    research = ResearchStage(lambda: service)
    stages: list[tuple[Stage, StageRunner]] = [
        (Stage.S0, IssueFramingStage()),
        (Stage.S1, research),
        (Stage.S2, OpeningStage(max_evidence_chars=max_evidence_chars)),
        (Stage.S3, PrecedentStage(lambda: service)),
        (Stage.S4, VerificationGate(Stage.S4)),
        (Stage.S5, RebuttalStage(lambda: research.researcher,
                                 max_evidence_chars=max_evidence_chars)),
        (Stage.S7, VerificationGate(Stage.S7)),
    ]  # fmt: skip
    rows: list[dict[str, Any]] = []
    for item in load_split(split)[:limit]:
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
        for stage, runner in stages:
            await runner.run(StageContext(run_ctx, record, budget, stage, "on"))
        view = record.view
        rows.append(
            {
                "item": item.id,
                "issues": len(view.issues),
                "values": {k: v for k, v in run_ctx.values.items() if k.startswith(("s5.", "s7."))},
                "s5_tokens": sum(
                    r.input_tokens + r.output_tokens
                    for r in gateway.ledger.records
                    if r.tags.get("stage") == Stage.S5.value
                ),
                "queries": [q.query for q in view.queries if q.stage is Stage.S5],
                "counterarguments": counters(view),
                "dropped": [
                    {"agent": f.agent, "errors": f.errors, "dropped": f.dropped}
                    for f in view.validation_failures
                    if f.stage is Stage.S5
                ],
            }
        )
        v = rows[-1]["values"]
        print(item.id, v.get("s5.counterarguments"), v.get("s5.evidence_added"),
              v.get("s7.checked"), file=sys.stderr)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s5-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def report(split: str) -> int:
    rows = json.loads((OUT / f"s5-{split}.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(r["values"].get(key) or 0 for r in rows)

    counters_ = [c for r in rows for c in r["counterarguments"]]
    citations = [c for x in counters_ for c in x["citations"]]
    new = [c for c in citations if c["found_in_s5"]]
    after, failing = total("s5.citations_after_repair"), total("s5.citation_failures_after_repair")
    print(f"{len(rows)} items, {sum(r['issues'] for r in rows)} issues")
    print(f"searches {total('s5.searches'):.0f}, evidence added {total('s5.evidence_added'):.0f}")
    print(f"counterarguments {len(counters_)} (items with none: "
          f"{sum(not r['counterarguments'] for r in rows)})")  # fmt: skip
    print(f"citations after repair {after:.0f}, failing {failing:.0f}: validity "
          f"{1 - failing / after if after else float('nan'):.3f}; repairs "
          f"{total('s5.repairs'):.0f}")  # fmt: skip
    print(f"in the record: {len(citations)} citations, {len(new)} to evidence found in S5")
    checked = [c for c in citations if c["s7"] is not None]
    print(f"S7 results: {len(checked)}/{len(citations)} citations "
          f"({sum(c['s7'] is not None for c in new)}/{len(new)} of the new evidence)")  # fmt: skip
    statuses: dict[str, int] = {}
    for c in checked:
        statuses[c["s7"]] = statuses.get(c["s7"], 0) + 1
    print("S7 statuses:", dict(sorted(statuses.items())))
    targets: dict[str, int] = {}
    for x in counters_:
        targets[x["target"]["status"]] = targets.get(x["target"]["status"], 0) + 1
    print("targets by status:", dict(sorted(targets.items())))
    tokens = [r["s5_tokens"] for r in rows]
    print(f"S5 tokens per item: mean {sum(tokens) / len(tokens):.0f}, max {max(tokens)}")
    dropped = [d for r in rows for d in r["dropped"]]
    print(f"validation failures {len(dropped)}")
    return 0


def sample(split: str, n: int, seed: int) -> int:
    rows = json.loads((OUT / f"s5-{split}.json").read_text(encoding="utf-8"))
    pool = [(r["item"], c) for r in rows for c in r["counterarguments"]]
    for item, c in random.Random(seed).sample(pool, min(n, len(pool))):
        print(f"--- {item} {c['id']} ({c['by']}) answers {c['target']['id']} "
              f"[{c['target']['status']}]: {c['target']['text']}")  # fmt: skip
        print(f"  claim [{c['claim']['status']}]: {c['claim']['text']}")
        print(f"  reasoning: {c['reasoning']}")
        for x in c["citations"]:
            new = " (found in S5)" if x["found_in_s5"] else ""
            print(f"  {x['evidence']}{new} [{x['s7']}]: {x['quote']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["run", "report", "sample"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-evidence-chars", type=int, default=28000)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.command == "run":
        return asyncio.run(run(args.split, args.limit, args.max_evidence_chars))
    if args.command == "sample":
        return sample(args.split, args.n, args.seed)
    return report(args.split)


if __name__ == "__main__":
    sys.exit(main())
