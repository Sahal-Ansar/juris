"""S9, the Judge (IDEA_final §6.1, PLAN 7.11): one call writes the ``CaseAnalysis`` through the
Case Record (``set_analysis`` checks invariants 3 and 5). S10 then checks its citations.

Recorded (``s9.*``): whether an analysis was written, its issues, and the inline citations and
key evidence it gives.

Passthrough runs the Judge too: every pipeline needs an analysis.
"""

from juris.agents.judge import Judge, inline_refs
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext


class JudgeStage:
    stage = Stage.S9
    optional = False

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped

    async def run(self, ctx: StageContext) -> None:
        await Judge().run(ctx)
        analysis = ctx.record.view.analysis
        values = ctx.run.values
        values["s9.analysis"] = float(analysis is not None)
        if analysis is None:
            return
        positions = [p for i in analysis.issues for p in i.positions]
        values["s9.issues"] = float(len(analysis.issues))
        values["s9.inline_citations"] = float(
            sum(len(inline_refs(p.summary)) for p in positions)
            + len(inline_refs(analysis.overall_summary))
        )
        values["s9.key_evidence"] = float(sum(len(p.key_evidence) for p in positions))

    async def passthrough(self, ctx: StageContext) -> None:
        await self.run(ctx)
