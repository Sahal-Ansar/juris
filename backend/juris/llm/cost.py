"""Token and dollar accounting per provider call."""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from juris.config import TokenPrice

log = logging.getLogger("juris.llm")


def cost_usd(
    model_name: str, input_tokens: int, output_tokens: int, prices: Mapping[str, TokenPrice]
) -> float | None:
    """Dollar cost of one call, or None if the model has no price entry."""
    price = prices.get(model_name)
    if price is None:
        return None
    return (input_tokens * price.input + output_tokens * price.output) / 1_000_000


class CallRecord(BaseModel):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    cached: bool
    tags: dict[str, str] = Field(default_factory=dict)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CostLedger:
    """In-memory log of every call; the run manifest (PLAN 0.5) persists it."""

    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def add(self, record: CallRecord) -> None:
        self.records.append(record)
        log.info(
            "llm call provider=%s model=%s in=%d out=%d usd=%s cached=%s tags=%s",
            record.provider,
            record.model,
            record.input_tokens,
            record.output_tokens,
            "?" if record.cost_usd is None else f"{record.cost_usd:.6f}",
            record.cached,
            record.tags,
        )

    def total_tokens(self) -> tuple[int, int]:
        live = [r for r in self.records if not r.cached]
        return sum(r.input_tokens for r in live), sum(r.output_tokens for r in live)

    def total_usd(self) -> float:
        """Spend on live calls. Calls to unpriced models count as 0; see ``unpriced_calls``."""
        return sum(r.cost_usd or 0.0 for r in self.records if not r.cached)

    def unpriced_calls(self) -> int:
        return sum(1 for r in self.records if not r.cached and r.cost_usd is None)
