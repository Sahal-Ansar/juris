"""S8, the jury (IDEA_final §6.1, PLAN 7.10): ``jurors`` independent Jurors, all at once, then a
deterministic aggregate.

Ballots are written in juror order (``J-1``, ``J-2``, ...), so the record does not depend on
which call finished first. A juror whose ballot fails its checks after the repair casts none.

**Aggregate** (``aggregate``, pure code): per position and per rubric criterion, the mean and
the spread (max minus min) over the ballots, plus ``overall``, each ballot's mean of the four
criteria. An issue is **contested** when the ``overall`` spread of any of its positions is more
than ``contested_spread`` (default 1.0: jurors more than a point apart on average). A single
criterion is too noisy: on dev, every issue had some criterion two points apart (D-058).
Nothing is written if no ballot was cast.

Budget: the first juror always runs; the others are skipped with a ``budget_warning`` once the
budget is spent (IDEA_final §6.2: some jurors are optional).

Recorded (``s8.*``): ballots cast, issues contested, and the mean spread of ``overall``.

Passthrough: nothing is written (the ``no_jury`` ablation: the Judge reads the record).
"""

from collections.abc import Iterable, Mapping

from juris.agents.juror import Juror
from juris.events.catalog import JuryAggregatedPayload, ScoreStats
from juris.models import Issue, JuryBallot, RubricScores
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext

AGENT = "jury"
CRITERIA = tuple(RubricScores.model_fields)
OVERALL = "overall"


def aggregate(
    ballots: Iterable[JuryBallot], issues: Mapping[str, Issue], *, contested_spread: float
) -> JuryAggregatedPayload:
    """Mean and spread per position and criterion (and ``overall``); issues where a position's
    ``overall`` spread exceeds ``contested_spread`` are contested. Order-independent."""
    cast = list(ballots)
    per_position: dict[str, dict[str, ScoreStats]] = {}
    for issue in issues.values():
        for position in issue.positions:
            scores = [b.per_position_scores[position.id] for b in cast]
            columns = {c: [float(getattr(s, c)) for s in scores] for c in CRITERIA}
            columns[OVERALL] = [
                sum(getattr(s, c) for c in CRITERIA) / len(CRITERIA) for s in scores
            ]
            per_position[position.id] = {
                name: ScoreStats(
                    mean=round(sum(values) / len(values), 4),
                    spread=round(max(values) - min(values), 4),
                )
                for name, values in columns.items()
            }
    contested = [
        issue.id
        for issue in issues.values()
        if any(per_position[p.id][OVERALL].spread > contested_spread for p in issue.positions)
    ]
    return JuryAggregatedPayload(per_position=per_position, contested_issue_ids=contested)


class JuryStage:
    stage = Stage.S8
    optional = False

    def __init__(self, *, contested_spread: float = 1.0) -> None:
        self.contested_spread = contested_spread

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped (the no_jury ablation turns it off)

    async def run(self, ctx: StageContext) -> None:
        n = ctx.params.jurors
        if n > 1 and not ctx.budget.allows(None):
            ctx.warn_budget(f"ran 1 juror of {n}")
            n = 1
        jurors = [Juror(k) for k in range(1, n + 1)]
        outputs = await ctx.gather(*(j.produce(ctx) for j in jurors))
        for juror, output in zip(jurors, outputs, strict=True):
            if output is not None:
                juror.apply(output, ctx.record, stage=self.stage)
        view = ctx.record.view
        values = ctx.run.values
        values["s8.ballots"] = float(len(view.jury_ballots))
        if not view.jury_ballots:
            values["s8.contested_issues"] = 0.0
            return
        result = aggregate(
            view.jury_ballots.values(), view.issues, contested_spread=self.contested_spread
        )
        ctx.record.set_jury_aggregate(result, stage=self.stage, agent=AGENT)
        values["s8.contested_issues"] = float(len(result.contested_issue_ids))
        spreads = [stats[OVERALL].spread for stats in result.per_position.values()]
        values["s8.overall_spread"] = sum(spreads) / len(spreads)

    async def passthrough(self, ctx: StageContext) -> None:
        return None
