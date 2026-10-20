"""S2, opening arguments (PLAN 7.5): one Counsel per position of every issue, all calls at once
(Counsel A and B in parallel, IDEA_final §6.1). The outputs are written issue by issue and
position by position, so claim and argument IDs do not depend on which call finished first.

Recorded with the run (``s2.*``), pooled over every counsel:

- ``citations_proposed`` and ``citation_failures_first``: the first outputs;
- ``citations_after_repair``, ``citation_failures_after_repair`` and
  ``citation_validity_after_repair``: the outputs after the one repair, before anything is
  dropped. This is 7.5's measure (target at least 0.95);
- ``claims`` and ``citations``: what reached the record, all of which pass invariants 1-2.

Passthrough: no counsel runs and nothing is written (for testing later stages).
"""

from juris.agents.counsel import MAX_CLAIMS, Counsel, citation_counts
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext


class OpeningStage:
    stage = Stage.S2
    optional = False

    def __init__(
        self, *, max_claims: int = MAX_CLAIMS, max_evidence_chars: int | None = None
    ) -> None:
        self.max_claims = max_claims
        self.max_evidence_chars = max_evidence_chars

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped

    async def run(self, ctx: StageContext) -> None:
        view = ctx.record.view
        counsel = [
            Counsel(
                issue.id,
                position.id,
                max_claims=self.max_claims,
                max_evidence_chars=self.max_evidence_chars,
            )
            for issue in view.issues.values()
            for position in issue.positions
        ]
        outputs = await ctx.gather(*(c.produce(ctx) for c in counsel))
        for c, output in zip(counsel, outputs, strict=True):
            if output is not None:
                c.apply(output, ctx.record, stage=self.stage)
        self._record(ctx, counsel)

    async def passthrough(self, ctx: StageContext) -> None:
        return None

    @staticmethod
    def _record(ctx: StageContext, counsel: list[Counsel]) -> None:
        first = [citation_counts(*c.attempts[0]) for c in counsel if c.attempts]
        last = [citation_counts(*c.attempts[-1]) for c in counsel if c.attempts]
        after = sum(t for t, _ in last)
        failing = sum(f for _, f in last)
        view = ctx.record.view
        values = ctx.run.values
        values["s2.claims"] = float(len(view.claims))
        values["s2.citations"] = float(sum(len(a.citations) for a in view.arguments.values()))
        values["s2.citations_proposed"] = float(sum(t for t, _ in first))
        values["s2.citation_failures_first"] = float(sum(f for _, f in first))
        values["s2.citations_after_repair"] = float(after)
        values["s2.citation_failures_after_repair"] = float(failing)
        values["s2.citation_validity_after_repair"] = 1 - failing / after if after else None
        values["s2.repairs"] = float(sum(len(c.attempts) > 1 for c in counsel))
