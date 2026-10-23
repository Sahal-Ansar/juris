"""Counsel's reply to objections (S6; IDEA_final §6.1, PLAN 7.8): the targeted counsel gets one
reply per objection.

One call per counsel (issue and position) that has objections against its claims or arguments.
It sees the issue, the positions, S0's facts, and each objection with what it targets: the
claim, or the argument with its reasoning, quotes and their S4 results and S3 fits. It does not
see the evidence pool: a reply answers the objection, it does not add citations.

Output: per objection, a stance and one or two sentences. ``concede`` resolves the objection
(the flaw is admitted); ``maintain`` answers it but leaves it open for the Jury and the Judge.
An objection with no reply stays unanswered.

Checks (``validate``): the objection is one shown (written loosely is fine: "O-2" is "O-002"),
one reply per objection, the reply is not empty.
"""

from typing import Literal

from pydantic import BaseModel, Field

from juris.agents.base import Problem
from juris.agents.counsel import CounselBase
from juris.agents.rebuttal import assessment
from juris.events import CaseView
from juris.models.common import Stage
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.citing import clean_record_id


class Reply(BaseModel):
    objection_id: str = Field(description='The objection you answer, e.g. "O-002"')
    stance: Literal["maintain", "concede"] = Field(
        description='"maintain": your claim stands despite it; "concede": the objection is right'
    )
    response: str = Field(description="One or two sentences: your answer to the objection")


class ReplyOutput(BaseModel):
    replies: list[Reply] = Field(description="One reply per objection")


class CounselReply(CounselBase[ReplyOutput]):
    prompt = "counsel_reply"
    duty = (
        "Counsel answers each objection raised against its own claims and arguments, once, "
        "either maintaining the point or conceding it."
    )
    output_model = ReplyOutput
    max_tokens = 1500

    def __init__(self, issue_id: str, position_id: str, objection_ids: list[str]) -> None:
        super().__init__(issue_id, position_id)
        self.objection_ids = objection_ids

    def describe_objections(self, view: CaseView) -> str:
        blocks = []
        for objection_id in self.objection_ids:
            o = view.objections[objection_id]
            lines = [f"[{o.id}] {o.type.value}, against {o.target_id}: {o.text}"]
            if o.target_id in view.claims:
                claim = view.claims[o.target_id]
                lines.append(f"  your claim {claim.id} ({claim.status.value}): {claim.text}")
            elif o.target_id in view.arguments:
                argument = view.arguments[o.target_id]
                claim = view.claims[argument.claim_ids[0]]
                lines.append(f"  your claim {claim.id} ({claim.status.value}): {claim.text}")
                lines.append(f"  your argument {argument.id}: {argument.reasoning}")
                lines += [
                    f'    {c.evidence_id}: "{c.quote}" [{assessment(view, c, argument.id)}]'
                    for c in argument.citations
                ]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def input_view(self, view: CaseView) -> dict[str, str]:
        issue = view.issues[self.issue_id]
        own = next(p for p in issue.positions if p.id == self.position_id)
        others = [p for p in issue.positions if p.id != self.position_id]
        facts = [*view.facts, *(f"(assumed) {a}" for a in view.assumptions)]
        return {
            "issue_id": issue.id,
            "issue": issue.question,
            "position_id": own.id,
            "position": own.statement,
            "other_positions": "\n".join(f"{p.id}: {p.statement}" for p in others),
            "facts": "\n".join(f"- {f}" for f in facts) or "(none stated)",
            "objections": self.describe_objections(view),
        }

    def validate(self, output: ReplyOutput, record: CaseRecord) -> list[Problem]:
        answered: set[str] = set()
        problems: list[Problem] = []
        for n, reply in enumerate(output.replies):
            at = ("replies", n)
            objection_id = clean_record_id(reply.objection_id, "O")
            if objection_id not in self.objection_ids:
                problems.append(
                    Problem(f"{reply.objection_id!r} is not one of the objections to answer", at)
                )
            elif objection_id in answered:
                problems.append(Problem(f"{objection_id} is already answered; one reply each", at))
            answered.add(objection_id)
            if not reply.response.strip():
                problems.append(Problem("the reply is empty", at))
        return problems

    def apply(self, output: ReplyOutput, record: CaseRecord, *, stage: Stage) -> None:
        for reply in output.replies:
            record.answer_objection(
                clean_record_id(reply.objection_id, "O"),
                reply.response.strip(),
                resolved=reply.stance == "concede",
                stage=stage,
                agent=self.name,
            )
