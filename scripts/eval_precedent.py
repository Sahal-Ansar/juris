"""Check S3's precedent cards on dev (PLAN 7.6: a card for every cited judgment; fit labels
spot-checked on 10 cases).

    uv run scripts/eval_precedent.py run [--split dev] [--limit N] [--max-evidence-chars 28000]
    uv run scripts/eval_precedent.py report [--split dev]
    uv run scripts/eval_precedent.py sample [--n 10] [--seed 0]

``run`` takes each item through S0-S3 with the configured model (the cache makes S0-S2 free
after ``eval_counsel.py``, with the same evidence budget). Per item it keeps S3's counts and
every card, with its fits: the claim, the quote and the analyst's reason. Results go to
``eval/results/precedent/s3-<split>.json``.

``report`` pools the counts. ``sample`` prints a seeded sample of fits for a manual check.
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
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext, StageRunner
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.precedent import PrecedentStage
from juris.pipeline.stages.research import ResearchStage

OUT = REPO_ROOT / "eval" / "results" / "precedent"


async def run(split: str, limit: int | None, max_evidence_chars: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s3-eval", kind="juris")
    config = resolve_run_config(settings, profile)
    gateways = GatewayFactory.from_settings(settings)
    service = RetrievalService.local(device="cpu")  # the GPU belongs to the local LLM
    stages: list[tuple[Stage, StageRunner]] = [
        (Stage.S0, IssueFramingStage()),
        (Stage.S1, ResearchStage(lambda: service)),
        (Stage.S2, OpeningStage(max_evidence_chars=max_evidence_chars)),
        (Stage.S3, PrecedentStage(lambda: service)),
    ]
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
        reasons = {
            (f["document"], f["argument"]): f["reason"] for f in run_ctx.details.get("s3.fits", [])
        }
        cards = []
        for card in view.precedent_cards.values():
            fits = []
            for argument_id, fit in card.fit_by_argument.items():
                argument = view.arguments[argument_id]
                claims = [view.claims[c] for c in argument.claim_ids]
                quotes = [
                    c.quote
                    for c in argument.citations
                    if view.evidence[c.evidence_id].document_id == card.document_id
                ]
                fits.append(
                    {
                        "argument": argument_id,
                        "position": claims[0].position_id,
                        "claim": " ".join(c.text for c in claims),
                        "quotes": quotes,
                        "fit": fit.value,
                        "reason": reasons.get((card.document_id, argument_id)),
                    }
                )
            cards.append(
                {
                    "document": card.document_id,
                    "title": view.documents[card.document_id].title,
                    "principle": card.principle,
                    "material_facts": card.material_facts,
                    "distinguishing_factors": card.distinguishing_factors,
                    "treatment": [s.model_dump(mode="json") for s in card.treatment],
                    "fits": fits,
                }
            )
        rows.append(
            {
                "item": item.id,
                "question": item.question,
                "issues": {i.id: i.question for i in view.issues.values()},
                "values": {k: v for k, v in run_ctx.values.items() if k.startswith("s3.")},
                "cards": cards,
            }
        )
        print(item.id, rows[-1]["values"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s3-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def report(split: str) -> int:
    rows = json.loads((OUT / f"s3-{split}.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(r["values"].get(key) or 0 for r in rows)

    judgments, cards = total("s3.judgments"), total("s3.cards")
    print(f"{len(rows)} items: {judgments:.0f} cited judgments, {cards:.0f} cards "
          f"({total('s3.fallback_cards'):.0f} fallback)")  # fmt: skip
    print(f"fits {total('s3.fits'):.0f}, missing {total('s3.fits_missing'):.0f}; "
          f"treatment signals {total('s3.treatment_signals'):.0f}")  # fmt: skip
    labels: dict[str, int] = {}
    for r in rows:
        for c in r["cards"]:
            for f in c["fits"]:
                labels[f["fit"]] = labels.get(f["fit"], 0) + 1
    print("fit labels:", dict(sorted(labels.items(), key=lambda kv: -kv[1])))
    return 0


def sample(split: str, n: int, seed: int) -> int:
    rows = json.loads((OUT / f"s3-{split}.json").read_text(encoding="utf-8"))
    fits = [(r, c, f) for r in rows for c in r["cards"] for f in c["fits"]]
    for r, c, f in random.Random(seed).sample(fits, min(n, len(fits))):
        print(f"== {r['item']} {f['argument']} ({f['position']}) -> {c['title']}")
        print(f"   question: {r['question']}")
        print(f"   claim: {f['claim']}")
        print(f"   quote: {f['quotes']}")
        print(f"   principle: {c['principle']}")
        print(f"   fit: {f['fit']} - {f['reason']}\n")
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
    if args.command == "report":
        return report(args.split)
    return sample(args.split, args.n, args.seed)


if __name__ == "__main__":
    sys.exit(main())
