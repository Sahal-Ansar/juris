"""The per-case budget (IDEA_final §6.2, §16; PLAN 7.1): when what is left will not cover an
optional stage or step, the orchestrator skips it and logs ``budget_warning``.

Spend is read from the gateway's call ledger and counts **every** call, cached ones at the
tokens they cost when first made, priced at list prices. A cached re-run therefore makes the
same skip decisions as the run it replays. Counting only live calls would make it think it had
spent nothing, and it would run stages the original skipped. The one gap: when the gateway's
structured-output repair ran, only the repaired response is cached, so a re-run counts that
call once.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from juris.config import BudgetConfig, TokenPrice
from juris.llm.cost import CostLedger, cost_usd


@dataclass(frozen=True)
class Spend:
    tokens: int = 0
    usd: float = 0.0

    def __add__(self, other: "Spend") -> "Spend":
        return Spend(self.tokens + other.tokens, self.usd + other.usd)


def ledger_spend(ledger: CostLedger, prices: Mapping[str, TokenPrice]) -> Callable[[], Spend]:
    """The run's spend so far, cached calls included (see the module docstring)."""

    def spend() -> Spend:
        tokens = usd = 0.0
        for r in ledger.records:
            tokens += r.input_tokens + r.output_tokens
            usd += cost_usd(r.model, r.input_tokens, r.output_tokens, prices) or 0.0
        return Spend(int(tokens), usd)

    return spend


class BudgetManager:
    """Answers "can the case still afford this?" against ``BudgetConfig``'s limits."""

    def __init__(self, budget: BudgetConfig, spend: Callable[[], Spend]) -> None:
        self.budget = budget
        self._spend = spend

    @property
    def used(self) -> Spend:
        return self._spend()

    @property
    def limited(self) -> bool:
        return (
            self.budget.max_tokens_per_case is not None or self.budget.max_usd_per_case is not None
        )

    def remaining(self) -> tuple[int | None, float | None]:
        """Tokens and dollars left (None where there is no limit; never below 0)."""
        used = self.used
        tokens = self.budget.max_tokens_per_case
        usd = self.budget.max_usd_per_case
        return (
            None if tokens is None else max(0, tokens - used.tokens),
            None if usd is None else max(0.0, usd - used.usd),
        )

    def exhausted(self) -> bool:
        tokens, usd = self.remaining()
        return tokens == 0 or usd == 0.0

    def allows(self, estimate: Spend | None) -> bool:
        """True if the estimate fits what is left. An unknown estimate (None) fits unless
        the budget is already spent."""
        if estimate is None:
            return not self.exhausted()
        tokens, usd = self.remaining()
        return (tokens is None or estimate.tokens <= tokens) and (
            usd is None or estimate.usd <= usd
        )
