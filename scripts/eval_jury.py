"""Check S8's jury on dev (PLAN 7.10: ballots validate; the aggregate is deterministic given the
ballots; jurors cannot add evidence).

    uv run scripts/eval_jury.py run [--split dev] [--limit N] [--max-evidence-chars 28000]
    uv run scripts/eval_jury.py report [--split dev]

``run`` takes each item through S0-S8 with the configured model (the cache makes S0-S6 free
after ``eval_cross_exam.py``). Per item it keeps S8's counts (``s8.*``), the ballots, the
aggregate, a re-computed aggregate from the ballots in reverse order (determinism), and per
position the share of its claims still standing (not fallen, not unsupported). Results go to
``eval/results/jury/s8-<split>.json``.

``report`` pools them, including how often the jury's higher ``overall`` mean goes to the
position with more standing claims.
"""

import argparse
import asyncio
import json
import sys
from typing import Any

from juris.baselines.b1 import B1System
from juris.baselines.base import RunContext
from juris.config import REPO_ROOT, PipelineProfile, get_settings, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.eval.runner import GatewayFactory
from juris.events import CaseView, EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.models.common import ClaimStatus, Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext, StageRunner
from juris.pipeline.stages.cross_exam import CrossExamStage
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.jury import OVERALL, JuryStage, aggregate
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.precedent import PrecedentStage
from juris.pipeline.stages.rebuttal import RebuttalStage
from juris.pipeline.stages.research import ResearchStage
from juris.pipeline.stages.verify import VerificationGate

OUT = REPO_ROOT / "eval" / "results" / "jury"
STANDING = {ClaimStatus.VERIFIED, ClaimStatus.WEAK, ClaimStatus.CONTESTED, ClaimStatus.SURVIVES}


def standing(view: CaseView) -> dict[str, dict[str, int]]:
    """Per position: its claims, and how many still stand."""
    out: dict[str, dict[str, int]] = {}
    for issue in view.issues.values():
        for position in issue.positions:
            mine = [c for c in view.claims.values() if c.position_id == position.id]
            out[position.id] = {
                "claims": len(mine),
                "standing": sum(c.status in STANDING for c in mine),
            }
    return out


async def run(split: str, limit: int | None, max_evidence_chars: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s8-eval", kind="juris")
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
        (Stage.S7, VerificationGate(Stage.S7)),
        (Stage.S8, JuryStage()),
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
        ballots = list(view.jury_ballots.values())
        again = aggregate(reversed(ballots), view.issues, contested_spread=1.0) if ballots else None
        rows.append(
            {
                "item": item.id,
                "issues": len(view.issues),
                "values": {k: v for k, v in run_ctx.values.items() if k.startswith("s8.")},
                "s8_tokens": sum(
                    r.input_tokens + r.output_tokens
                    for r in gateway.ledger.records
                    if r.tags.get("stage") == Stage.S8.value
                ),
                "ballots": [b.model_dump(mode="json") for b in ballots],
                "aggregate": view.jury_aggregate.model_dump(mode="json")
                if view.jury_aggregate
                else None,
                "deterministic": again == view.jury_aggregate,
                "standing": standing(view),
                "positions": {i.id: [p.id for p in i.positions] for i in view.issues.values()},
                "dropped": [
                    {"agent": f.agent, "errors": f.errors, "dropped": f.dropped}
                    for f in view.validation_failures
                    if f.stage is Stage.S8
                ],
            }
        )
        v = rows[-1]["values"]
        print(item.id, v.get("s8.ballots"), v.get("s8.contested_issues"), file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s8-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def report(split: str) -> int:
    rows = json.loads((OUT / f"s8-{split}.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(r["values"].get(key) or 0 for r in rows)

    issues = sum(r["issues"] for r in rows)
    print(f"{len(rows)} items, {issues} issues")
    print(f"ballots cast {total('s8.ballots'):.0f} of {3 * len(rows)}; items without an "
          f"aggregate: {sum(r['aggregate'] is None for r in rows)}")  # fmt: skip
    print(f"aggregate reproduced from reversed ballots: "
          f"{sum(r['deterministic'] for r in rows if r['aggregate'])}/"
          f"{sum(r['aggregate'] is not None for r in rows)}")  # fmt: skip
    print(f"contested issues {total('s8.contested_issues'):.0f}/{issues}")
    spreads = [r["values"]["s8.overall_spread"] for r in rows if "s8.overall_spread" in r["values"]]
    print(f"mean overall spread per position {sum(spreads) / len(spreads):.2f}")
    agree = ties = split_ = 0
    for r in rows:
        if r["aggregate"] is None:
            continue
        for positions in r["positions"].values():
            means = {p: r["aggregate"]["per_position"][p][OVERALL]["mean"] for p in positions}
            shares = {
                p: r["standing"][p]["standing"] / max(1, r["standing"][p]["claims"])
                for p in positions
            }
            if len(set(shares.values())) == 1:
                ties += 1
                continue
            split_ += 1
            agree += max(means, key=means.__getitem__) == max(shares, key=shares.__getitem__)
    print(f"jury's top position = the one with more standing claims: {agree}/{split_} "
          f"({ties} issues tied on standing claims)")  # fmt: skip
    tokens = [r["s8_tokens"] for r in rows]
    print(f"S8 tokens per item: mean {sum(tokens) / len(tokens):.0f}, max {max(tokens)}")
    errors = [e for r in rows for d in r["dropped"] for e in d["errors"]]
    print(f"validation failures {sum(len(r['dropped']) for r in rows)}; errors: {errors[:6]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-evidence-chars", type=int, default=28000)
    args = ap.parse_args(argv)
    if args.command == "run":
        return asyncio.run(run(args.split, args.limit, args.max_evidence_chars))
    return report(args.split)


if __name__ == "__main__":
    sys.exit(main())
