"""Compare experiments (PLAN 5.5, IDEA_final §11.3): a table of means with 95% bootstrap CIs,
and paired tests between two configs on the items both ran.

Paired test: per item, the difference B - A of the item's value (averaged over seeds).
The interval is a percentile bootstrap of the mean difference; the p-value is a two-sided
sign-flip randomisation test (exact up to 16 items, else 20,000 random flips with a fixed
seed). No distribution is assumed, which suits small, bounded, skewed item scores.
"""

import itertools
import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from juris.eval.runner import HEADLINE, load_scores, per_item_values
from juris.eval.stats import aggregate, bootstrap_ci

EXACT_UP_TO = 16
RANDOM_FLIPS = 20_000


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    spec: dict[str, Any]
    per_item: dict[str, dict[str, float | None]]


def load_experiment(directory: Path) -> Experiment:
    spec = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    return Experiment(spec["experiment_id"], spec, per_item_values(load_scores(directory)))


def _cell(values: Sequence[float | None]) -> str:
    agg = aggregate(values)
    return "n/a" if agg is None else f"{agg.mean:.3f} ({agg.lo:.3f}-{agg.hi:.3f})"


def comparison_table(experiments: Sequence[Experiment], metrics: Sequence[str] = HEADLINE) -> str:
    """One row per experiment: each metric's mean over items with its 95% CI."""
    rows = [
        "| Experiment | Items | " + " | ".join(metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for e in experiments:
        cells = [_cell([v.get(m) for v in e.per_item.values()]) for m in metrics]
        rows.append(f"| {e.experiment_id} | {len(e.per_item)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


@dataclass(frozen=True)
class Paired:
    metric: str
    n: int
    mean_a: float
    mean_b: float
    diff: float  # B - A
    lo: float
    hi: float
    p: float


def sign_flip_p(diffs: Sequence[float], seed: int = 0) -> float:
    """Two-sided p-value of the mean paired difference under random sign flips."""
    observed = abs(sum(diffs)) - 1e-12  # sums, not means: same order, fewer divisions
    if len(diffs) <= EXACT_UP_TO:
        flips = list(itertools.product((1, -1), repeat=len(diffs)))
        extreme = sum(
            abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed
            for signs in flips
        )
        return extreme / len(flips)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(RANDOM_FLIPS):
        extreme += abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed
    return (extreme + 1) / (RANDOM_FLIPS + 1)


def paired(a: Experiment, b: Experiment, metric: str) -> Paired | None:
    """None if no item has the metric in both experiments."""
    pairs = [
        (va, vb)
        for item_id in sorted(a.per_item.keys() & b.per_item.keys())
        if (va := a.per_item[item_id].get(metric)) is not None
        and (vb := b.per_item[item_id].get(metric)) is not None
    ]
    if not pairs:
        return None
    diffs = [vb - va for va, vb in pairs]
    lo, hi = bootstrap_ci(diffs)
    return Paired(
        metric,
        len(pairs),
        statistics.mean(va for va, _ in pairs),
        statistics.mean(vb for _, vb in pairs),
        statistics.mean(diffs),
        lo,
        hi,
        sign_flip_p(diffs),
    )


def paired_table(a: Experiment, b: Experiment, metrics: Sequence[str] = HEADLINE) -> str:
    rows = [
        f"A = {a.experiment_id}, B = {b.experiment_id}",
        "",
        "| Metric | n | A | B | B - A | 95% CI | p |",
        "|---|---|---|---|---|---|---|",
    ]
    for metric in metrics:
        r = paired(a, b, metric)
        if r is None:
            rows.append(f"| {metric} | 0 | n/a | n/a | n/a | n/a | n/a |")
        else:
            rows.append(
                f"| {metric} | {r.n} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.diff:+.3f} | "
                f"{r.lo:+.3f} to {r.hi:+.3f} | {r.p:.3f} |"
            )
    return "\n".join(rows)
