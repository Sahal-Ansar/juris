"""Run one config over a Juris-Eval split, score it and store the results (PLAN 5.5).

    uv run scripts/run_experiment.py --config b1 --split dev --limit 20 --seeds 1
    uv run scripts/run_experiment.py --config dummy --split dev --limit 3

Options:
    --experiment-id ID    default <config>-<split>-n<limit|all>-s<seeds>; re-running the same
                          ID resumes it (finished runs are skipped)
    --grader PROVIDER:MODEL
                          grade key points, the rubric and summary sentences with this model
                          (off by default: those values stay empty)
    --concurrency N       runs in flight at once (default 2)
    --retry-failed        run again the runs that failed
    --results-dir DIR     default eval/results

Results: eval/results/<experiment_id>/ and a row in eval/results/results.csv.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from juris.config import ModelSpec, get_settings, load_profile, resolve_run_config
from juris.eval.juris_eval import SPLITS, load_split
from juris.eval.metrics import Grader
from juris.eval.runner import (
    RESULTS_DIR,
    SYSTEMS,
    ExperimentSpec,
    GatewayFactory,
    default_experiment_id,
    run_experiment,
    summary_table,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="profile name in configs/pipeline/")
    ap.add_argument("--split", choices=SPLITS, default="dev")
    ap.add_argument("--limit", type=int, default=None, help="first N items by ID")
    ap.add_argument("--seeds", type=int, default=1, help="runs per item (seeds 0..N-1 + profile)")
    ap.add_argument("--experiment-id")
    ap.add_argument("--grader", help="PROVIDER:MODEL")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)

    settings = get_settings()
    profile = load_profile(args.config)
    config = resolve_run_config(settings, profile)
    items = load_split(args.split)[: args.limit] if args.limit else load_split(args.split)
    grader_model = None
    if args.grader:
        provider, _, name = args.grader.partition(":")
        if not name:
            ap.error("--grader must be PROVIDER:MODEL")
        grader_model = ModelSpec(provider=provider, name=name)
    spec = ExperimentSpec(
        experiment_id=args.experiment_id
        or default_experiment_id(profile.name, args.split, args.limit, args.seeds),
        profile=profile.name,
        kind=profile.kind,
        split=args.split,
        item_ids=tuple(i.id for i in items),
        seeds=tuple(profile.seed + n for n in range(args.seeds)),
        grader=grader_model,
    )
    gateways = GatewayFactory.from_settings(settings)
    grader = Grader(gateways(), grader_model) if grader_model else None
    result = asyncio.run(
        run_experiment(
            spec,
            items,
            config,
            system=SYSTEMS[profile.kind](),
            gateways=gateways,
            grader=grader,
            results_dir=args.results_dir,
            concurrency=args.concurrency,
            retry_failed=args.retry_failed,
            prices=settings.llm_prices,
        )
    )
    counts: dict[str, int] = {}
    for o in result.outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    print(f"{spec.experiment_id}: {counts} -> {result.directory}")
    print(summary_table(result.summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
