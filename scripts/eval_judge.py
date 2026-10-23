"""Check S9-S10 on dev (PLAN 7.11: the output always validates; zero invalid citations in the
final output; a leaning and a confidence for every issue).

    uv run scripts/eval_judge.py run [--split dev] [--limit N] [--max-evidence-chars 28000]
    uv run scripts/eval_judge.py report [--split dev]

``run`` takes each item through the whole pipeline, S0-S10, with the configured model (the
cache makes S0-S8 free after ``eval_jury.py``). Per item it keeps S9's and S10's counts, the
final ``CaseAnalysis``, and every evidence ID the final analysis cites (key evidence and inline)
with its best verification. Results go to ``eval/results/judge/s9-<split>.json``.

``report`` pools them: analyses written, citations in the final output that are not verified
or weak (should be 0), issues with a leaning and confidence, leanings against the jury's
higher-scored position, and what S10 removed.
"""

import argparse
import asyncio
import json
import sys
from collections import Counter
from typing import Any

from juris.agents.judge import inline_refs
from juris.baselines.b1 import B1System
from juris.baselines.base import RunContext
from juris.config import REPO_ROOT, PipelineProfile, get_settings, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.eval.runner import GatewayFactory
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CITABLE, CaseRecord
from juris.pipeline.orchestrator import StageContext, StageRunner
from juris.pipeline.stages.cross_exam import CrossExamStage
from juris.pipeline.stages.final_verify import FinalVerificationStage
from juris.pipeline.stages.framing import IssueFramingStage
from juris.pipeline.stages.judge import JudgeStage
from juris.pipeline.stages.jury import OVERALL, JuryStage
from juris.pipeline.stages.opening import OpeningStage
from juris.pipeline.stages.precedent import PrecedentStage
from juris.pipeline.stages.rebuttal import RebuttalStage
from juris.pipeline.stages.research import ResearchStage
from juris.pipeline.stages.verify import VerificationGate

OUT = REPO_ROOT / "eval" / "results" / "judge"


async def run(split: str, limit: int | None, max_evidence_chars: int | None) -> int:
    from juris.retrieval.service import RetrievalService

    settings = get_settings()
    profile = PipelineProfile(name="s9-eval", kind="juris")
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
        (Stage.S9, JudgeStage()),
        (Stage.S10, FinalVerificationStage()),
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
        analysis = view.analysis
        cited: dict[str, str | None] = {}
        jury: dict[str, str] = {}
        if analysis is not None:
            for issue in analysis.issues:
                for p in issue.positions:
                    for e in [r.evidence_id for r in p.key_evidence] + inline_refs(p.summary):
                        best = record.best_verification(e)
                        cited[e] = best.value if best else None
            for e in inline_refs(analysis.overall_summary):
                best = record.best_verification(e)
                cited[e] = best.value if best else None
            if view.jury_aggregate is not None:
                for issue in analysis.issues:
                    means = {
                        p.position_id[-1]: view.jury_aggregate.per_position[p.position_id][
                            OVERALL
                        ].mean
                        for p in issue.positions
                    }
                    top = sorted(means.values())
                    jury[issue.issue_id] = (
                        "tie" if top[-1] == top[-2] else max(means, key=means.__getitem__)
                    )
        rows.append(
            {
                "item": item.id,
                "values": {
                    k: v for k, v in run_ctx.values.items() if k.startswith(("s9.", "s10."))
                },
                "tokens": {
                    stage: sum(
                        r.input_tokens + r.output_tokens
                        for r in gateway.ledger.records
                        if r.tags.get("stage") == stage
                    )
                    for stage in ("S9",)
                },
                "total_tokens": sum(
                    r.input_tokens + r.output_tokens for r in gateway.ledger.records
                ),
                "analysis": analysis.model_dump(mode="json") if analysis else None,
                "cited": cited,
                "jury_top": jury,
                "removed": run_ctx.details.get("s10.removed", []),
            }
        )
        v = rows[-1]["values"]
        print(item.id, v.get("s9.analysis"), v.get("s10.judge_citations"),
              v.get("s10.judge_unsupported"), file=sys.stderr)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"s9-{split}.json"
    path.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def report(split: str) -> int:
    rows = json.loads((OUT / f"s9-{split}.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(r["values"].get(key) or 0 for r in rows)

    written = [r for r in rows if r["analysis"]]
    issues = [i for r in written for i in r["analysis"]["issues"]]
    print(f"{len(rows)} items: analyses written {len(written)}")
    statuses = Counter(str(s) for r in written for s in r["cited"].values())
    bad = sum(n for s, n in statuses.items() if s not in {x.value for x in CITABLE})
    print(f"evidence cited in the final analyses {sum(statuses.values())}: {dict(statuses)}; "
          f"not verified or weak: {bad}")  # fmt: skip
    complete = sum(bool(i["leaning"]) and bool(i["confidence"]) for i in issues)
    print(f"issues {len(issues)}, with leaning and confidence {complete}")
    print("leanings:", dict(Counter(i["leaning"] for i in issues).most_common()))
    print("confidence:", dict(Counter(i["confidence"] for i in issues).most_common()))
    agree = disagree = other = 0
    for r in written:
        for i in r["analysis"]["issues"]:
            top = r["jury_top"].get(i["issue_id"])
            side = i["leaning"][-1] if i["leaning"][-2:] in {"_A", "_B"} else None
            if top in {"A", "B"} and side:
                agree += side == top
                disagree += side != top
            else:
                other += 1
    print(f"leaning toward the jury's higher-scored position: {agree} agree, {disagree} not, "
          f"{other} balanced/insufficient or jury tie")  # fmt: skip
    print(f"S10: Judge citations {total('s10.judge_citations'):.0f}, not citable "
          f"{total('s10.judge_not_citable'):.0f}, unsupported "
          f"{total('s10.judge_unsupported'):.0f}; sentences flagged "
          f"{total('s10.sentences_flagged'):.0f}")  # fmt: skip
    unresolved = sum(len(i["unresolved_questions"]) for i in issues)
    print(f"unresolved questions answered {unresolved}; key conflicts "
          f"{sum(len(i['key_conflicts']) for i in issues)}; sources "
          f"{sum(len(r['analysis']['sources']) for r in written)}")  # fmt: skip
    s9 = [r["tokens"]["S9"] for r in rows]
    whole = [r["total_tokens"] for r in rows]
    print(f"tokens per item: S9 {sum(s9) / len(s9):.0f}; "
          f"whole pipeline {sum(whole) / len(whole):.0f} (max {max(whole)})")  # fmt: skip
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
