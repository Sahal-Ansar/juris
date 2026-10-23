"""S6, cross-examination (IDEA_final §6, §8; PLAN 7.8).

1. **Objections.** One Cross-Examiner per issue, all at once. Across issues the most severe
   are kept (high, medium, low; then issue and output order), at most ``max_objections`` per
   case, and raised issue by issue.
2. **Contested.** A claim that an objection targets, directly or through its argument, or that
   a counterargument answers, becomes contested (IDEA_final §8). That includes a claim from S5,
   still proposed until S7 (D-057): S7 records its citations' results but leaves its status.
3. **Replies.** One ``CounselReply`` per counsel with objections, all at once, written in
   order: one reply per objection.
4. **Outcome.** A conceded objection is resolved, and the contested claims it targets fall. An
   objection that counsel maintained, or did not answer, becomes an unresolved question for the
   Jury and the Judge.

Recorded (``s6.*``): objections proposed and raised (and how many are high), replies, conceded,
maintained, unanswered, unresolved questions, claims contested and claims that fell.

Passthrough: nothing is written.
"""

from collections import defaultdict

from juris.agents.counsel_reply import CounselReply
from juris.agents.cross_examiner import (
    AGENT,
    MAX_PER_ISSUE,
    SEVERITY,
    CrossExaminer,
    CrossExamOutput,
    targets,
)
from juris.events import CaseView
from juris.models import Objection
from juris.models.common import ClaimStatus, Stage
from juris.pipeline.budget import Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext

MAX_OBJECTIONS = 8
_OPEN = frozenset(
    {ClaimStatus.PROPOSED, ClaimStatus.VERIFIED, ClaimStatus.WEAK, ClaimStatus.UNSUPPORTED}
)


def targeted_claims(view: CaseView, target_id: str) -> list[str]:
    """The claims an objection is about: the claim itself, or the argument's claims."""
    if target_id in view.claims:
        return [target_id]
    if target_id in view.arguments:
        return list(view.arguments[target_id].claim_ids)
    return []


class CrossExamStage:
    stage = Stage.S6
    optional = False

    def __init__(
        self, *, max_objections: int = MAX_OBJECTIONS, per_issue: int = MAX_PER_ISSUE
    ) -> None:
        self.max_objections = max_objections
        self.per_issue = per_issue

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped (the no_cross_exam ablation turns it off)

    async def run(self, ctx: StageContext) -> None:
        record = ctx.record
        before = set(record.view.objections)
        examiners = [
            CrossExaminer(issue.id, max_objections=self.per_issue)
            for issue in record.view.issues.values()
        ]
        outputs = await ctx.gather(*(e.produce(ctx) for e in examiners))
        proposed = sum(len(o.objections) for o in outputs if o is not None)
        for examiner, output in zip(examiners, self._keep(outputs), strict=True):
            if output is not None:
                examiner.apply(output, record, stage=self.stage)
        raised = [o for i, o in record.view.objections.items() if i not in before]
        contested = self._contest(record, raised)
        replies = self._repliers(record.view, raised)
        answers = await ctx.gather(*(r.produce(ctx) for r in replies))
        for counsel, answer in zip(replies, answers, strict=True):
            if answer is not None:
                counsel.apply(answer, record, stage=self.stage)
        fell, unresolved = self._settle(record, [o.id for o in raised])
        view = record.view
        final = [view.objections[o.id] for o in raised]
        values = ctx.run.values
        values["s6.objections_proposed"] = float(proposed)
        values["s6.objections"] = float(len(final))
        values["s6.replies"] = float(sum(o.response is not None for o in final))
        values["s6.conceded"] = float(sum(o.resolved for o in final))
        values["s6.maintained"] = float(
            sum(o.response is not None and not o.resolved for o in final)
        )
        values["s6.unanswered"] = float(sum(o.response is None for o in final))
        values["s6.unresolved_questions"] = float(unresolved)
        values["s6.contested"] = float(contested)
        values["s6.falls"] = float(fell)
        ctx.run.details["s6.objection_types"] = [o.type.value for o in final]

    async def passthrough(self, ctx: StageContext) -> None:
        return None

    def _keep(self, outputs: list[CrossExamOutput | None]) -> list[CrossExamOutput | None]:
        """The most severe objections across issues, at most ``max_objections``."""
        ranked = sorted(
            (SEVERITY[o.severity], k, n)
            for k, output in enumerate(outputs)
            if output is not None
            for n, o in enumerate(output.objections)
        )
        kept = {(k, n) for _, k, n in ranked[: self.max_objections]}
        return [
            None
            if output is None
            else output.model_copy(
                update={
                    "objections": [o for n, o in enumerate(output.objections) if (k, n) in kept]
                }
            )
            for k, output in enumerate(outputs)
        ]

    def _contest(self, record: CaseRecord, raised: list[Objection]) -> int:
        """Mark the open claims that an objection or a counterargument targets."""
        view = record.view
        reasons: dict[str, list[str]] = defaultdict(list)
        for objection in raised:
            for claim_id in targeted_claims(view, objection.target_id):
                reasons[claim_id].append(objection.id)
        for counter in view.counterarguments.values():
            reasons[counter.targets].append(counter.id)
        changed = 0
        for claim_id, by in sorted(reasons.items()):
            if view.claims[claim_id].status in _OPEN:
                reason = f"challenged by {', '.join(dict.fromkeys(by))}"
                changed += record.set_claim_status(
                    claim_id, ClaimStatus.CONTESTED, reason, stage=self.stage, agent=AGENT
                )
        return changed

    @staticmethod
    def _repliers(view: CaseView, raised: list[Objection]) -> list[CounselReply]:
        """One counsel per position with objections against it, in position order."""
        by_position: dict[str, list[str]] = defaultdict(list)
        for objection in raised:
            issue_id = next(i for i in view.issues if objection.target_id in targets(view, i))
            by_position[targets(view, issue_id)[objection.target_id]].append(objection.id)
        return [
            CounselReply(position.rsplit(".", 1)[0], position, ids)
            for position, ids in sorted(by_position.items())
        ]

    def _settle(self, record: CaseRecord, objection_ids: list[str]) -> tuple[int, int]:
        """Conceded: the contested claims fall. Otherwise: an unresolved question."""
        fell = unresolved = 0
        for objection_id in objection_ids:
            view = record.view
            objection = view.objections[objection_id]
            if objection.resolved:
                for claim_id in targeted_claims(view, objection.target_id):
                    if view.claims[claim_id].status is ClaimStatus.CONTESTED:
                        fell += record.set_claim_status(
                            claim_id,
                            ClaimStatus.FALLS,
                            f"counsel conceded {objection.id}",
                            stage=self.stage,
                            agent=AGENT,
                        )
                continue
            claim_id = targeted_claims(view, objection.target_id)[0]
            issue_id = view.claims[claim_id].position_id.rsplit(".", 1)[0]
            state = (
                "not answered"
                if objection.response is None
                else f"disputed; counsel replied: {objection.response}"
            )
            record.add_unresolved_question(
                f"Does objection {objection.id} ({objection.type.value.replace('_', ' ')}) "
                f"against {objection.target_id} hold? {objection.text} ({state})",
                stage=self.stage,
                agent=AGENT,
                issue_id=issue_id,
                from_objection_id=objection.id,
            )
            unresolved += 1
        return fell, unresolved
