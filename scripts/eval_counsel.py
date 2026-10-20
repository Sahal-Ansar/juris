"""Measure Counsel's citation validity after repair (PLAN 7.5: at least 0.95 on dev).

    uv run scripts/eval_counsel.py run [--split dev] [--limit N] [--max-evidence-chars 28000]
    uv run scripts/eval_counsel.py report [--split dev]

``run`` takes each item through S0 (Issue Framer), S1 (Researcher) and S2 (Counsel A and B per
issue) with the configured model; the response cache is on as the settings say. Per item it
keeps S2's counts (``s2.*``, see ``pipeline/stages/opening.py``) and the claims and arguments
that reached the record, in ``eval/results/counsel/s2-<split>.json``.

``report`` pools the counts over items:

- validity after repair = 1 - citations failing existence or quote checks / citations, over
  the counsel outputs after the one repair attempt, before anything is dropped;
- validity first = the same over the first outputs.
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
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.research import ResearchStage

OUT = REPO_ROOT / "eval" / "results" / "counsel"


async def run(split: str, limit: int | None, max_evidence_chars: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s2-eval", kind="juris")
    config = resolve_run_config(settings, profile)
    gateways = GatewayFactory.from_settings(settings)
    service = RetrievalService.local(device="cpu")  # the GPU belongs to the local LLM
    research = ResearchStage(lambda: service)
    opening = OpeningStage(max_evidence_chars=max_evidence_chars)
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
        for stage, runner in ((Stage.S0, IssueFramingStage()), (Stage.S1, research),
                              (Stage.S2, opening)):  # fmt: skip
            await runner.run(StageContext(run_ctx, record, budget, stage, "on"))
        view = record.view
        rows.append(
            {
                "item": item.id,
                "issues": len(view.issues),
                "evidence": len(view.evidence),
                "values": {k: v for k, v in run_ctx.values.items() if k.startswith("s2.")},
                "tokens": sum(r.input_tokens + r.output_tokens for r in gateway.ledger.records),
                "claims": [
                    {"id": c.id, "position": c.position_id, "text": c.text}
                    for c in view.claims.values()
                ],
                "citations": [
                    {"claim": c.claim_id, "evidence": c.evidence_id, "quote": c.quote}
                    for a in view.arguments.values()
                    for c in a.citations
                ],
                "dropped": [f.dropped for f in view.validation_failures],
            }
        )
        v = rows[-1]["values"]
        print(
            item.id, v.get("s2.citations_after_repair"), v.get("s2.citation_validity_after_repair")
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s2-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def report(split: str) -> int:
    rows = json.loads((OUT / f"s2-{split}.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(r["values"].get(key) or 0 for r in rows)

    proposed, first = total("s2.citations_proposed"), total("s2.citation_failures_first")
    after, failing = total("s2.citations_after_repair"), total("s2.citation_failures_after_repair")
    print(f"{len(rows)} items, {sum(r['issues'] for r in rows)} issues")
    print(f"citations proposed {proposed:.0f}, failing {first:.0f}: "
          f"validity first {1 - first / proposed if proposed else float('nan'):.3f}")  # fmt: skip
    validity = 1 - failing / after if after else float("nan")
    print(f"after repair {after:.0f}, failing {failing:.0f}: validity after repair {validity:.3f}")
    print(f"in the record: {total('s2.claims'):.0f} claims, {total('s2.citations'):.0f} citations; "
          f"repairs {total('s2.repairs'):.0f}")  # fmt: skip
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
