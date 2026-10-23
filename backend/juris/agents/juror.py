"""A Juror (S8; IDEA_final §6.1, PLAN 7.10): an independent, rubric-based assessment of every
position. Jurors differ only by sampling seed (``seed_offset``); they cannot add evidence.

Every juror sees the same frozen record, issue by issue: the positions; every claim with its
status, its arguments, each citation's quote with its S4/S7 result and S3 fit (no retrieval
scores, no evidence pool); the precedent cards; the objections with counsel's replies; and the
unresolved questions.

Output: per position, the four rubric scores (1-5, higher is better for the position) and a
short justification citing record IDs. In the record it is one ``JuryBallot``.

Checks (``validate``): every position of the case scored exactly once (a ballot is all or
nothing), a justification, and cited IDs that exist (written loosely is fine: "C-4" is
"C-004"). A citation of an unknown ID is dropped; a justification left with none drops its
position, and then the whole ballot.
"""

from pydantic import BaseModel, Field

from juris.agents.base import Agent, Problem
from juris.agents.cross_examiner import describe_claims, describe_precedents
from juris.config import Role
from juris.events import CaseView
from juris.models import CreatedBy, JuryBallot, Justification, RubricScores
from juris.models.common import Stage
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.citing import clean_record_id

_PREFIXES = {"E", "C", "A", "R", "O"}


class PositionScores(BaseModel):
    position_id: str = Field(description='The position scored, e.g. "I-1.A"')
    evidence_strength: int = Field(ge=1, le=5, description="Verified support and its weight")
    precedent_fit: int = Field(ge=1, le=5, description="How well the precedents relied on fit")
    counterargument_handling: int = Field(
        ge=1, le=5, description="How well it answers the other side and the objections"
    )
    unresolved_risk: int = Field(
        ge=1, le=5, description="5: little left open against it; 1: much left open"
    )
    justification: str = Field(description="One or two sentences on these scores")
    cited_ids: list[str] = Field(description="Record IDs the justification relies on")


class JurorOutput(BaseModel):
    scores: list[PositionScores] = Field(description="One entry per position of every issue")


def clean_id(raw: str) -> str:
    text = raw.strip().strip("[]()").strip()
    prefix = text[:1].upper()
    return clean_record_id(text, prefix) if prefix in _PREFIXES else text


def describe_objections(view: CaseView, issue_id: str) -> str:
    claims = {c.id for c in view.claims.values() if c.position_id.startswith(f"{issue_id}.")}
    arguments = {a.id for a in view.arguments.values() if a.claim_ids[0] in claims}
    lines = []
    for o in view.objections.values():
        if o.target_id not in claims | arguments:
            continue
        if o.response is None:
            outcome = "no reply"
        else:
            outcome = f"{'conceded' if o.resolved else 'maintained'}: {o.response}"
        lines.append(f"[{o.id}] {o.type.value} against {o.target_id}: {o.text}\n  reply {outcome}")
    return "\n".join(lines) or "(none)"


def describe_case(view: CaseView) -> str:
    """The frozen record every juror sees, issue by issue."""
    blocks = []
    for issue in view.issues.values():
        positions = "\n".join(f"{p.id}: {p.statement}" for p in issue.positions)
        questions = [q.question for q in view.unresolved_questions if q.issue_id == issue.id]
        blocks.append(
            f'<issue id="{issue.id}">\n{issue.question}\n\nPositions:\n{positions}\n\n'
            f"Claims and arguments:\n{describe_claims(view, issue.id)}\n\n"
            f"Precedents cited:\n{describe_precedents(view, issue.id)}\n\n"
            f"Objections:\n{describe_objections(view, issue.id)}\n\n"
            f"Unresolved questions:\n" + ("\n".join(f"- {q}" for q in questions) or "(none)")
            + "\n</issue>"
        )  # fmt: skip
    return "\n\n".join(blocks)


class Juror(Agent[JurorOutput]):
    role = Role.JUROR
    prompt = "juror"
    duty = (
        "A Juror scores every position on a fixed rubric from the frozen record alone, "
        "independently of the other jurors, without adding evidence."
    )
    output_model = JurorOutput
    max_tokens = 2500

    def __init__(self, number: int) -> None:
        super().__init__(f"juror_{number}", seed_offset=number)
        self.juror_id = f"J-{number}"

    def input_view(self, view: CaseView) -> dict[str, str]:
        facts = [*view.facts, *(f"(assumed) {a}" for a in view.assumptions)]
        return {
            "facts": "\n".join(f"- {f}" for f in facts) or "(none stated)",
            "case": describe_case(view),
            "positions": ", ".join(p.id for i in view.issues.values() for p in i.positions),
        }

    def validate(self, output: JurorOutput, record: CaseRecord) -> list[Problem]:
        view = record.view
        positions = [p.id for i in view.issues.values() for p in i.positions]
        known = (
            set(view.evidence)
            | set(view.claims)
            | set(view.arguments)
            | set(view.counterarguments)
            | set(view.objections)
        )
        problems: list[Problem] = []
        scored = [s.position_id.strip() for s in output.scores]
        missing = [p for p in positions if p not in scored]
        if missing:
            problems.append(Problem(f"no scores for {', '.join(missing)}"))
        seen: set[str] = set()
        for n, entry in enumerate(output.scores):
            at = ("scores", n)
            position = entry.position_id.strip()
            if position not in positions:
                problems.append(Problem(f"{entry.position_id!r} is not a position of this case"))
            elif position in seen:
                problems.append(Problem(f"{position} is scored twice"))
            seen.add(position)
            if not entry.justification.strip():
                problems.append(Problem("the justification is empty", at))
            if not entry.cited_ids:
                problems.append(Problem("cite at least one record ID", at))
            for k, raw in enumerate(entry.cited_ids):
                if clean_id(raw) not in known:
                    problems.append(
                        Problem(f"{raw!r} is not an ID in the record", (*at, "cited_ids", k))
                    )
        return problems

    def apply(self, output: JurorOutput, record: CaseRecord, *, stage: Stage) -> None:
        record.cast_ballot(
            JuryBallot(
                juror_id=self.juror_id,
                per_position_scores={
                    s.position_id.strip(): RubricScores(
                        evidence_strength=s.evidence_strength,
                        precedent_fit=s.precedent_fit,
                        counterargument_handling=s.counterargument_handling,
                        unresolved_risk=s.unresolved_risk,
                    )
                    for s in output.scores
                },
                justifications=[
                    Justification(
                        text=f"{s.position_id.strip()}: {s.justification.strip()}",
                        cited_ids=list(dict.fromkeys(clean_id(i) for i in s.cited_ids)),
                    )
                    for s in output.scores
                ],
                created_by=CreatedBy(stage=stage, agent=self.name),
            ),
            stage=stage,
            agent=self.name,
        )
