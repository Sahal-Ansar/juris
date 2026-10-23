"""The Cross-Examiner (S6; IDEA_final §6.1, PLAN 7.8): adversarial quality control over both
sides of one issue. It raises typed objections; it does not decide them.

One call per issue. It sees the issue, its positions and S0's facts, and every claim of the
issue from both sides with its arguments: the reasoning, each citation's quote, its S4 result
and its S3 precedent fit, and which claim a counterargument answers. Below them, the precedent
cards of the judgments cited (principle and distinguishing factors; treatment is never
verified, D-013).

Output: up to ``max_objections`` objections, each a type (IDEA_final §6.1's list), a target (a
claim or an argument of the issue), one or two sentences, and a severity. The stage keeps the
most severe across issues (PLAN 7.8: at most 8 per case).

Checks (``validate``): the target is a claim or argument shown, the text is not empty, one
objection per (type, target), at most ``max_objections``.
"""

from typing import Literal

from pydantic import BaseModel, Field

from juris.agents.base import Agent, Problem
from juris.agents.rebuttal import assessment
from juris.config import Role
from juris.events import CaseView
from juris.models.common import ObjectionType, Stage
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.citing import clean_record_id

MAX_PER_ISSUE = 4
AGENT = "cross_examiner"
SEVERITY = {"high": 0, "medium": 1, "low": 2}  # sort key: most severe first
_PRINCIPLE_CHARS = 400


class RaisedObjection(BaseModel):
    type: ObjectionType = Field(description="The kind of flaw")
    target_id: str = Field(description='The claim or argument it is about, e.g. "C-004", "A-002"')
    text: str = Field(description="One or two sentences: the flaw, naming the evidence if any")
    severity: Literal["high", "medium", "low"] = Field(
        description="How much the flaw matters to the issue"
    )


class CrossExamOutput(BaseModel):
    objections: list[RaisedObjection] = Field(description="Your objections; may be empty")


def targets(view: CaseView, issue_id: str) -> dict[str, str]:
    """The IDs an objection on ``issue_id`` may target, each with the position it is
    against: the issue's claims and the arguments for them."""
    out: dict[str, str] = {}
    for claim in view.claims.values():
        if claim.position_id.startswith(f"{issue_id}."):
            out[claim.id] = claim.position_id
    for argument in view.arguments.values():
        if argument.claim_ids and argument.claim_ids[0] in out:
            out[argument.id] = out[argument.claim_ids[0]]
    return out


def describe_claims(view: CaseView, issue_id: str) -> str:
    """Every claim of the issue with its arguments and citations, by position then ID."""
    answers = {
        view.arguments[r.argument_id].claim_ids[0]: r.targets
        for r in view.counterarguments.values()
    }
    issue = view.issues[issue_id]
    blocks = []
    for position in issue.positions:
        for claim in view.claims.values():
            if claim.position_id != position.id:
                continue
            head = f"[{claim.id}] for {position.id}, {claim.status.value}: {claim.text}"
            if claim.id in answers:
                head += f" (answers {answers[claim.id]})"
            lines = [head]
            for argument in view.arguments.values():
                if claim.id not in argument.claim_ids:
                    continue
                lines.append(f"  [{argument.id}] reasoning: {argument.reasoning}")
                for c in argument.citations:
                    title = view.documents[view.evidence[c.evidence_id].document_id].title
                    lines.append(
                        f'    {c.evidence_id} ({title}, {c.pinpoint}): "{c.quote}" '
                        f"[{assessment(view, c, argument.id)}]"
                    )
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "(no claims)"


def describe_precedents(view: CaseView, issue_id: str) -> str:
    """The cards of the judgments the issue's arguments cite."""
    shown = targets(view, issue_id)
    documents = dict.fromkeys(
        view.evidence[c.evidence_id].document_id
        for a in view.arguments.values()
        if a.id in shown
        for c in a.citations
    )
    blocks = []
    for document_id in documents:
        card = view.precedent_cards.get(document_id)
        if card is None:
            continue
        title = view.documents[document_id].title
        principle = card.principle[:_PRINCIPLE_CHARS]
        factors = "; ".join(card.distinguishing_factors) or "none noted"
        treatment = ", ".join(t.kind.value for t in card.treatment) or "none found"
        blocks.append(
            f"{title}: principle (model-extracted): {principle}\n"
            f"  distinguishing factors: {factors}\n"
            f"  treatment cues (unverified): {treatment}"
        )
    return "\n\n".join(blocks) or "(none)"


class CrossExaminer(Agent[CrossExamOutput]):
    role = Role.CROSS_EXAMINER
    prompt = "cross_examiner"
    duty = (
        "The Cross-Examiner raises typed objections against specific claims and arguments of "
        "both sides, without deciding the issue."
    )
    output_model = CrossExamOutput
    max_tokens = 2000

    def __init__(self, issue_id: str, *, max_objections: int = MAX_PER_ISSUE) -> None:
        super().__init__(AGENT)
        self.issue_id = issue_id
        self.max_objections = max_objections

    @property
    def call_tags(self) -> dict[str, str]:
        return {"issue": self.issue_id}

    def input_view(self, view: CaseView) -> dict[str, str]:
        issue = view.issues[self.issue_id]
        facts = [*view.facts, *(f"(assumed) {a}" for a in view.assumptions)]
        return {
            "issue_id": issue.id,
            "issue": issue.question,
            "positions": "\n".join(f"{p.id}: {p.statement}" for p in issue.positions),
            "facts": "\n".join(f"- {f}" for f in facts) or "(none stated)",
            "claims": describe_claims(view, self.issue_id),
            "precedents": describe_precedents(view, self.issue_id),
            "max_objections": str(self.max_objections),
        }

    def validate(self, output: CrossExamOutput, record: CaseRecord) -> list[Problem]:
        allowed = targets(record.view, self.issue_id)
        seen: set[tuple[str, str]] = set()
        problems: list[Problem] = []
        for n, objection in enumerate(output.objections):
            at = ("objections", n)
            if n >= self.max_objections:
                problems.append(Problem(f"at most {self.max_objections} objections", at))
                continue
            target = _clean_target(objection.target_id)
            key = (objection.type.value, target)
            if target not in allowed:
                problems.append(
                    Problem(f"{objection.target_id!r} is not a claim or argument of this issue", at)
                )
            elif key in seen:
                problems.append(Problem(f"a second {key[0]} objection against {target}", at))
            seen.add(key)
            if not objection.text.strip():
                problems.append(Problem("the objection is empty", at))
        return problems

    def apply(self, output: CrossExamOutput, record: CaseRecord, *, stage: Stage) -> None:
        for objection in output.objections:
            record.raise_objection(
                objection.type,
                _clean_target(objection.target_id),
                objection.text.strip(),
                stage=stage,
                agent=self.name,
            )


def _clean_target(raw: str) -> str:
    text = raw.strip().strip("[]()").strip()
    prefix = text[:1].upper()
    return clean_record_id(text, prefix) if prefix in {"C", "A"} else text
