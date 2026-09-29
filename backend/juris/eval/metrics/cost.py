"""Cost and latency per case (IDEA_final §11.2): LLM calls, tokens, dollars, wall-clock."""

from collections.abc import Sequence
from typing import Protocol

from juris.eval.metrics.common import MetricResult
from juris.events.fold import CaseView


class CallLike(Protocol):
    """One LLM call: a gateway ``CallRecord`` or an ``llm_calls`` row (``db.models.LlmCall``)."""

    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cost_usd(self) -> float | None: ...
    @property
    def cached(self) -> bool: ...


def cost_latency(
    view: CaseView, calls: Sequence[CallLike] | None = None, wall_seconds: float | None = None
) -> MetricResult:
    """From the run's calls when given, else from the run's ``run_completed`` totals.

    Tokens and dollars count live calls only (a cache hit costs nothing). Wall-clock is
    ``wall_seconds`` when given, else the span from the first to the last event.
    """
    if calls is not None:
        live = [c for c in calls if not c.cached]
        n_calls: float | None = len(calls)
        tokens_in: float | None = sum(c.input_tokens for c in live)
        tokens_out: float | None = sum(c.output_tokens for c in live)
        usd: float | None = sum(c.cost_usd or 0.0 for c in live)
        unpriced = sum(1 for c in live if c.cost_usd is None)
    elif view.totals is not None:
        n_calls = view.totals.llm_calls
        tokens_in, tokens_out = view.totals.input_tokens, view.totals.output_tokens
        usd, unpriced = view.totals.usd, view.totals.unpriced_calls
    else:
        n_calls = tokens_in = tokens_out = usd = None
        unpriced = 0
    if wall_seconds is None and len(view.timeline) > 1:
        wall_seconds = (view.timeline[-1].ts - view.timeline[0].ts).total_seconds()
    total = None if tokens_in is None or tokens_out is None else tokens_in + tokens_out
    return MetricResult(
        "cost_latency",
        {
            "cost_latency.llm_calls": n_calls,
            "cost_latency.input_tokens": tokens_in,
            "cost_latency.output_tokens": tokens_out,
            "cost_latency.total_tokens": total,
            "cost_latency.usd": usd,
            "cost_latency.wall_seconds": wall_seconds,
        },
        {"unpriced_calls": unpriced, "source": "calls" if calls is not None else "totals"},
    )
