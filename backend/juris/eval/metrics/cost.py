"""Cost and latency per case (IDEA_final §11.2): LLM calls, tokens, dollars, wall-clock."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from juris.config import TokenPrice
from juris.eval.metrics.common import MetricResult
from juris.events.fold import CaseView
from juris.llm.cost import cost_usd


class CallLike(Protocol):
    """One LLM call: a gateway ``CallRecord`` or an ``llm_calls`` row (``db.models.LlmCall``)."""

    @property
    def model(self) -> str: ...
    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cost_usd(self) -> float | None: ...
    @property
    def cached(self) -> bool: ...


def cost_latency(
    view: CaseView,
    calls: Sequence[CallLike] | None = None,
    wall_seconds: float | None = None,
    prices: Mapping[str, TokenPrice] | None = None,
) -> MetricResult:
    """From the run's calls when given, else from the run's ``run_completed`` totals.

    With calls, tokens count every call, cache hits included: they measure what the config
    consumes, which is what compute matching (B2, IDEA_final R1) compares, and a warm cache
    must not make a config look cheaper. Dollars are the list price of every call when
    ``prices`` are given, else the live spend (a cache hit costs nothing). ``.cached_calls``
    says how many were hits. The totals fallback counts live calls only. Wall-clock is
    ``wall_seconds`` when given, else the span from the first to the last event; cache hits
    shorten it.
    """
    cached: float | None = None
    if calls is not None:
        n_calls: float | None = len(calls)
        cached = sum(1 for c in calls if c.cached)
        tokens_in: float | None = sum(c.input_tokens for c in calls)
        tokens_out: float | None = sum(c.output_tokens for c in calls)
        if prices is not None:
            listed = [cost_usd(c.model, c.input_tokens, c.output_tokens, prices) for c in calls]
            usd: float | None = sum(u or 0.0 for u in listed)
            unpriced = sum(1 for u in listed if u is None)
        else:
            live = [c for c in calls if not c.cached]
            usd = sum(c.cost_usd or 0.0 for c in live)
            unpriced = sum(1 for c in live if c.cost_usd is None)
    elif view.totals is not None:
        n_calls = view.totals.llm_calls
        cached = view.totals.cached_calls
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
            "cost_latency.cached_calls": cached,
            "cost_latency.input_tokens": tokens_in,
            "cost_latency.output_tokens": tokens_out,
            "cost_latency.total_tokens": total,
            "cost_latency.usd": usd,
            "cost_latency.wall_seconds": wall_seconds,
        },
        {
            "unpriced_calls": unpriced,
            "source": "calls" if calls is not None else "totals",
            "usd": "list price" if calls is not None and prices is not None else "live spend",
        },
    )
