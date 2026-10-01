"""Score the verifier's entailment check on labelled pairs (PLAN 6.1).

    uv run scripts/eval_verifier.py run --set juris
    uv run scripts/eval_verifier.py run --set coliee4 --n 100 --model openai:juris-qwen2.5-7b
    uv run scripts/eval_verifier.py report

Sets:
    juris     the 50 hand-labelled pairs of eval/verifier/juris_pairs.yaml (4 labels)
    coliee4   COLIEE statute entailment, a seeded label-balanced sample (binary)
    coliee2   COLIEE case entailment: half positives, negatives from the same cases (binary)

``run`` options:
    --n N                  COLIEE sample size (default 100)
    --seed S               sampling seed, also the judge's seed (default 0)
    --batch-size B         pairs per model call (default 1)
    --model PROVIDER:NAME  default: the verifier role's model from the settings
    --max-passage-chars N  cut longer passages to N characters around the quote
    --results-dir DIR      default eval/results/verifier

Results: <set>-<model>-p<prompt version>-b<batch>.json; the response cache is on as the
settings say, so a repeated run costs nothing. ``report`` prints them as a Markdown table.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from juris.config import REPO_ROOT, ModelSpec, Role, get_settings
from juris.eval.datasets.coliee import find_task2, find_task4, load_task2, load_task4
from juris.eval.runner import GatewayFactory
from juris.eval.verifier_eval import (
    EvalPair,
    coliee_metrics,
    coliee_pairs,
    juris_metrics,
    juris_pairs,
    sample_balanced,
    sample_case_entailment,
)
from juris.prompts import load_prompt
from juris.runs import git_state
from juris.verify.entailment import ENTAILMENT_PROMPT, LLMEntailmentJudge

SETS = ("juris", "coliee4", "coliee2")
PAIRS_FILE = REPO_ROOT / "eval" / "verifier" / "juris_pairs.yaml"
RESULTS_DIR = REPO_ROOT / "eval" / "results" / "verifier"


def load_set(name: str, n: int, seed: int) -> list[EvalPair]:
    if name == "juris":
        return juris_pairs(yaml.safe_load(PAIRS_FILE.read_text(encoding="utf-8"))["pairs"])
    if name == "coliee4":
        return coliee_pairs(sample_balanced(load_task4(find_task4()).examples, n, seed))
    cases, labels = find_task2()
    return coliee_pairs(sample_case_entailment(load_task2(cases, labels).examples, n, seed))


def result_name(set_name: str, model: ModelSpec, prompt_version: int, batch_size: int) -> str:
    model_part = re.sub(r"[^A-Za-z0-9.]+", "-", model.name).strip("-")
    return f"{set_name}-{model_part}-p{prompt_version}-b{batch_size}.json"


def score(set_name: str, pairs: Sequence[EvalPair], labels: Sequence[str]) -> dict[str, Any]:
    if set_name == "juris":
        return juris_metrics([str(p.gold) for p in pairs], labels, [p.construction for p in pairs])
    return coliee_metrics([bool(p.gold) for p in pairs], labels)


def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    model = args.model or settings.models.for_role(Role.VERIFIER)
    pairs = load_set(args.set, args.n, args.seed)
    prompt = load_prompt(ENTAILMENT_PROMPT)
    gateway = GatewayFactory.from_settings(settings)()
    judge = LLMEntailmentJudge(
        gateway,
        model,
        batch_size=args.batch_size,
        seed=args.seed,
        max_passage_chars=args.max_passage_chars,
    )
    started = time.monotonic()
    judgements = asyncio.run(judge.judge([p.pair for p in pairs]))
    wall = time.monotonic() - started
    labels = [j.label.value for j in judgements]
    in_tokens, out_tokens = gateway.ledger.total_tokens()
    calls = gateway.ledger.records
    git = git_state()
    result = {
        "spec": {
            "set": args.set,
            "n": len(pairs),
            "seed": args.seed,
            "model": model.model_dump(mode="json"),
            "prompt": {"id": prompt.id, "version": prompt.version, "sha256": prompt.sha256},
            "batch_size": args.batch_size,
            "max_passage_chars": args.max_passage_chars,
            "git_commit": git.commit,
            "git_dirty": git.dirty,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "rows": [
            {
                "id": p.pair.claim_id,
                "gold": p.gold,
                "predicted": j.label.value,
                "justification": j.justification,
                **({"construction": p.construction} if p.construction else {}),
            }
            for p, j in zip(pairs, judgements, strict=True)
        ],
        "metrics": score(args.set, pairs, labels),
        "cost": {
            "calls": len(calls),
            "cached_calls": sum(r.cached for r in calls),
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "usd": gateway.ledger.total_usd(),
        },
        "wall_seconds": wall,
    }
    path = args.results_dir / result_name(args.set, model, prompt.version, args.batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    live = sum(not r.cached for r in calls)
    print(f"{args.set}: {len(pairs)} pairs, {len(calls)} calls ({live} live), {wall:.1f}s")
    print(f"-> {path}")
    print(json.dumps(result["metrics"], indent=2))
    return 0


def _cells(set_name: str, m: dict[str, Any]) -> list[str]:
    """Headline, secondary, F1 and kappa cells (kappa only for the 4-label set)."""
    if set_name == "juris":
        lo, hi = m["status_agreement_ci"]
        return [
            f"{m['status_agreement']:.3f} [{lo:.3f}, {hi:.3f}]",
            f"{m['label_accuracy']:.3f}",
            f"{m['supports_f1']:.3f}",
            f"{m['status_kappa']:.3f}",
        ]
    strict, lenient = m["strict"], m["lenient"]
    return [f"{strict['accuracy']:.3f}", f"{lenient['accuracy']:.3f}", f"{strict['f1']:.3f}", "-"]


def report(args: argparse.Namespace) -> int:
    files = args.files or sorted(args.results_dir.glob("*.json"))
    rows = [
        "| set | model | prompt | batch | n | headline | secondary | F1 | kappa | seconds |",
        "|" + "---|" * 10,
    ]
    for f in files:
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        spec = r["spec"]
        cells = _cells(spec["set"], r["metrics"])
        rows.append(
            f"| {spec['set']} | {spec['model']['name']} | "
            f"{spec['prompt']['id']}@v{spec['prompt']['version']} | {spec['batch_size']} | "
            f"{spec['n']} | {' | '.join(cells)} | {r['wall_seconds']:.0f} |"
        )
    print("\n".join(rows))
    print(
        "\njuris: headline = status agreement [95% CI], secondary = 4-way label accuracy, "
        "F1 of supports.\ncoliee: headline = strict accuracy, secondary = lenient accuracy, "
        "F1 strict."
    )
    return 0


def parse_model(value: str) -> ModelSpec:
    provider, _, name = value.partition(":")
    if not name:
        raise argparse.ArgumentTypeError("must be PROVIDER:MODEL")
    return ModelSpec(provider=provider, name=name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="judge a set and store the result")
    p_run.add_argument("--set", choices=SETS, required=True)
    p_run.add_argument("--n", type=int, default=100)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--batch-size", type=int, default=1)
    p_run.add_argument("--model", type=parse_model, help="PROVIDER:MODEL")
    p_run.add_argument("--max-passage-chars", type=int, default=None)
    p_run.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p_rep = sub.add_parser("report", help="a Markdown table of result files")
    p_rep.add_argument("files", nargs="*", type=Path)
    p_rep.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    return run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    sys.exit(main())
