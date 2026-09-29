"""Summary statistics shared by the evaluation modules (IDEA_final §11.3)."""

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


def bootstrap_ci(
    values: Sequence[float], resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean (items resampled with replacement)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


@dataclass(frozen=True)
class Aggregate:
    """Mean over the items that have a value, with its 95% bootstrap interval."""

    mean: float
    lo: float
    hi: float
    n: int


def aggregate(values: Sequence[float | None]) -> Aggregate | None:
    """``None`` values (metric not applicable to an item) are left out; all ``None`` -> None."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    lo, hi = bootstrap_ci(present)
    return Aggregate(statistics.mean(present), lo, hi, len(present))
