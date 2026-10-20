"""S0, issue framing (PLAN 7.3): the Issue Framer writes the issues, positions, queries and
statute references to the Case Record. Required: the case cannot go on without an issue.

Passthrough (no framer, e.g. to test later stages): one issue, the question itself, with the
positions "Yes." and "No." and the question as its one search query.
"""

from juris.agents.issue_framer import AGENT, IssueFramer
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext


class FramingFailed(RuntimeError):
    """The Issue Framer left no issue in the record."""


class IssueFramingStage:
    stage = Stage.S0
    optional = False

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped, so never estimated

    async def run(self, ctx: StageContext) -> None:
        params = ctx.params
        framer = IssueFramer(
            ctx.run.item.facts,
            order_seed=ctx.run.seed,
            max_issues=params.max_issues,
            positions_per_issue=params.positions_per_issue,
        )
        await framer.run(ctx)
        view = ctx.record.view
        if not view.issues:
            raise FramingFailed("the Issue Framer framed no usable issue")
        ctx.run.values["s0.issues"] = float(len(view.issues))
        ctx.run.values["s0.positions_swapped"] = float(len(framer.swapped))
        ctx.run.details["s0.swapped"] = framer.swapped

    async def passthrough(self, ctx: StageContext) -> None:
        record = ctx.record
        question = record.view.question
        issue = record.frame_issue(question, ["Yes.", "No."], stage=self.stage, agent=AGENT)
        record.add_query(question, [issue.id], stage=self.stage, agent=AGENT)
