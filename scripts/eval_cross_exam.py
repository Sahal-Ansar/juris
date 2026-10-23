"""Check S6's cross-examination on dev (PLAN 7.8: objections always target valid IDs; the
replies are recorded; unresolved questions appear in the record).

    uv run scripts/eval_cross_exam.py run [--split dev] [--limit N] [--max-evidence-chars 28000]
    uv run scripts/eval_cross_exam.py report [--split dev]
    uv run scripts/eval_cross_exam.py sample [--n 10] [--seed 0]

``run`` takes each item through S0-S6 with the configured model (the cache makes S0-S5 free
after ``eval_rebuttal.py``, with the same evidence budget). Per item it keeps S6's counts
(``s6.*``, see ``pipeline/stages/cross_exam.py``), every objection (type, target and its claim,
text, the reply and whether it was conceded; the record keeps no severity) and the unresolved
questions. Results go to ``eval/results/cross_exam/s6-<split>.json``.

``report`` pools the counts. ``sample`` prints a seeded sample of objections for a manual check.
"""

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
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
from juris.pipeline.stages.cross_exam import CrossExamStage
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.precedent import PrecedentStage
from juris.pipeline.stages.rebuttal import RebuttalStage
from juris.pipeline.stages.research import ResearchStage
from juris.pipeline.stages.verify import VerificationGate

OUT = REPO_ROOT / "eval" / "results" / "cross_exam"


def objections(view: CaseView) -> list[dict[str, Any]]:
    """Every objection with its target's text and the reply."""
    out = []
    for o in view.objections.values():
        if o.target_id in view.claims:
            claim = view.claims[o.target_id]
        else:
            claim = view.claims[view.arguments[o.target_id].claim_ids[0]]
        out.append(
            {
                "id": o.id,
                "type": o.type.value,
                "target": o.target_id,
                "claim": {
                    "id": claim.id,
                    "position": claim.position_id,
                    "status": claim.status.value,
                    "text": claim.text,
                },
                "text": o.text,
                "response": o.response,
                "conceded": o.resolved,
            }
        )
    return out


async def run(split: str, limit: int | None, max_evidence_chars: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s6-eval", kind="juris")
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
        (Stage.S6, CrossExamStage()),
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
                "claims": len(view.claims),
                "values": {k: v for k, v in run_ctx.values.items() if k.startswith("s6.")},
                "s6_tokens": sum(
                    r.input_tokens + r.output_tokens
                    for r in gateway.ledger.records
                    if r.tags.get("stage") == Stage.S6.value
                ),
                "objections": objections(view),
                "unresolved": [q.question for q in view.unresolved_questions],
                "dropped": [
                    {"agent": f.agent, "errors": f.errors, "dropped": f.dropped}
                    for f in view.validation_failures
                    if f.stage is Stage.S6
                ],
            }
        )
        v = rows[-1]["values"]
        print(item.id, v.get("s6.objections"), v.get("s6.conceded"),
              v.get("s6.unresolved_questions"), file=sys.stderr)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s6-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def report(split: str) -> int:
    rows = json.loads((OUT / f"s6-{split}.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(r["values"].get(key) or 0 for r in rows)

    found = [o for r in rows for o in r["objections"]]
    print(f"{len(rows)} items, {sum(r['issues'] for r in rows)} issues, "
          f"{sum(r['claims'] for r in rows)} claims")  # fmt: skip
    print(f"objections proposed {total('s6.objections_proposed'):.0f}, raised {len(found)} "
          f"(items with none: {sum(not r['objections'] for r in rows)})")  # fmt: skip
    print("types:", dict(Counter(o["type"] for o in found).most_common()))
    print("against:", dict(Counter(o["claim"]["position"][-1] for o in found)),
          "| target kinds:", dict(Counter(o["target"][0] for o in found)))  # fmt: skip
    print(f"replies {total('s6.replies'):.0f}: conceded {total('s6.conceded'):.0f}, "
          f"maintained {total('s6.maintained'):.0f}; "
          f"unanswered {total('s6.unanswered'):.0f}")  # fmt: skip
    print(f"unresolved questions {total('s6.unresolved_questions'):.0f}; contested "
          f"{total('s6.contested'):.0f}; fell {total('s6.falls'):.0f}")  # fmt: skip
    tokens = [r["s6_tokens"] for r in rows]
    print(f"S6 tokens per item: mean {sum(tokens) / len(tokens):.0f}, max {max(tokens)}")
    errors = Counter(e.split(": ", 1)[-1][:60] for r in rows for d in r["dropped"]
                     for e in d["errors"])  # fmt: skip
    print(
        f"validation failures {sum(len(r['dropped']) for r in rows)}:", dict(errors.most_common(6))
    )
    return 0


def sample(split: str, n: int, seed: int) -> int:
    rows = json.loads((OUT / f"s6-{split}.json").read_text(encoding="utf-8"))
    pool = [(r["item"], o) for r in rows for o in r["objections"]]
    for item, o in random.Random(seed).sample(pool, min(n, len(pool))):
        c = o["claim"]
        print(f"--- {item} {o['id']} {o['type']} against {o['target']} "
              f"({c['id']} for {c['position']}, now {c['status']}): {c['text']}")  # fmt: skip
        print(f"  objection: {o['text']}")
        stance = "conceded" if o["conceded"] else "maintained" if o["response"] else "no reply"
        print(f"  reply ({stance}): {o['response']}")
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
