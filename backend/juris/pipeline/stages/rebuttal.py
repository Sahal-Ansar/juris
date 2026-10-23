"""S5, the rebuttal round (IDEA_final §6, PLAN 7.7): every counsel answers the opposing claims
of its issue (``agents/rebuttal.py``). Counsel A and B work at once and are written in a fixed
order, as in S2.

Each round, for every counsel that has claims to answer:

1. **Searches** (if a Researcher is given and ``extra_searches_per_counsel`` > 0): all
   ``CounselSearch`` calls at once, then each counsel's searches run in order, registering new
   evidence for its issue.
2. **Counterarguments**: all ``RebuttalCounsel`` calls at once, then written in order.

``rebuttal_rounds`` rounds (default 1, at most 2; 0, or ``single_counsel``, turns the stage into
a no-op). The second
round answers what the first round added, and is optional: it is skipped with a
``budget_warning`` when what is left will not cover what the first round spent. The searches
are skipped the same way once the budget is spent.

Recorded (``s5.*``), pooled over rounds: rounds run, searches, evidence added, counterarguments
written, and the same citation counts as S2 (first outputs, after the one repair).

Passthrough: nothing is written (S5 adds no state later stages need).
"""

from collections.abc import Callable

from juris.agents.rebuttal import (
    MAX_COUNTERS,
    SEARCH_K,
    CounselSearch,
    RebuttalCounsel,
    counter_counts,
)
from juris.agents.researcher import Researcher
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext


class RebuttalStage:
    stage = Stage.S5
    optional = False

    def __init__(
        self,
        researcher: Callable[[], Researcher] | None = None,
        *,
        max_counters: int = MAX_COUNTERS,
        max_evidence_chars: int | None = None,
        per_query: int = SEARCH_K,
    ) -> None:
        self._researcher = researcher  # S1's (``ResearchStage.researcher``); None: no searches
        self.max_counters = max_counters
        self.max_evidence_chars = max_evidence_chars
        self.per_query = per_query

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped (its second round is)

    async def run(self, ctx: StageContext) -> None:
        counsel: list[RebuttalCounsel] = []
        searches: list[CounselSearch] = []
        rounds = 0
        last = Spend()
        planned = 0 if ctx.params.single_counsel else ctx.params.rebuttal_rounds  # no opponent
        for round_no in range(1, planned + 1):
            if round_no > 1 and not ctx.budget.allows(last):
                ctx.warn_budget(f"skipped rebuttal round {round_no}")
                break
            before = ctx.budget.used
            done = await self._round(ctx, round_no)
            counsel += done[0]
            searches += done[1]
            rounds += 1
            after = ctx.budget.used
            last = Spend(after.tokens - before.tokens, after.usd - before.usd)
        self._record(ctx, rounds, counsel, searches)

    async def passthrough(self, ctx: StageContext) -> None:
        return None

    async def _round(
        self, ctx: StageContext, round_no: int
    ) -> tuple[list[RebuttalCounsel], list[CounselSearch]]:
        view = ctx.record.view
        counsel = [
            RebuttalCounsel(
                issue.id,
                position.id,
                max_counters=self.max_counters,
                max_evidence_chars=self.max_evidence_chars,
                round_no=round_no,
            )
            for issue in view.issues.values()
            for position in issue.positions
        ]
        counsel = [c for c in counsel if c.targets(view)]
        searches: list[CounselSearch] = []
        most = ctx.params.extra_searches_per_counsel
        if counsel and self._researcher is not None and most > 0:
            if ctx.budget.allows(None):
                researcher = self._researcher()
                searches = [
                    CounselSearch(c, researcher, most=most, per_query=self.per_query)
                    for c in counsel
                ]
                queries = await ctx.gather(*(s.produce(ctx) for s in searches))
                for s, asked in zip(searches, queries, strict=True):
                    if asked is not None:
                        s.apply(asked, ctx.record, stage=self.stage)
            else:
                ctx.warn_budget(f"skipped the targeted searches of rebuttal round {round_no}")
        outputs = await ctx.gather(*(c.produce(ctx) for c in counsel))
        for c, output in zip(counsel, outputs, strict=True):
            if output is not None:
                c.apply(output, ctx.record, stage=self.stage)
        return counsel, searches

    def _record(
        self,
        ctx: StageContext,
        rounds: int,
        counsel: list[RebuttalCounsel],
        searches: list[CounselSearch],
    ) -> None:
        first = [counter_counts(*c.attempts[0]) for c in counsel if c.attempts]
        last = [counter_counts(*c.attempts[-1]) for c in counsel if c.attempts]
        after = sum(t for t, _ in last)
        failing = sum(f for _, f in last)
        view = ctx.record.view
        values = ctx.run.values
        values["s5.rounds"] = float(rounds)
        values["s5.searches"] = float(sum(q.stage is Stage.S5 for q in view.queries))
        values["s5.evidence_added"] = float(sum(len(c.found) for c in counsel))
        values["s5.counterarguments"] = float(len(view.counterarguments))
        values["s5.citations_proposed"] = float(sum(t for t, _ in first))
        values["s5.citation_failures_first"] = float(sum(f for _, f in first))
        values["s5.citations_after_repair"] = float(after)
        values["s5.citation_failures_after_repair"] = float(failing)
        values["s5.citation_validity_after_repair"] = 1 - failing / after if after else None
        repairs = [len(c.attempts) > 1 for c in counsel] + [len(s.attempts) > 1 for s in searches]
        values["s5.repairs"] = float(sum(repairs))
