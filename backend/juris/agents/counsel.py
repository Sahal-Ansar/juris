"""Counsel (S2; IDEA_final §6.1, PLAN 7.5): one class, one instance per position and issue
(Counsel A argues ``I-n.A``, Counsel B ``I-n.B``). The prompt is the same for every instance
except the assigned position, and all of them see the same evidence pool.

Output: up to ``max_claims`` claims, each with one sentence of reasoning and its citations (an
evidence ID and a verbatim quote). In the record each claim becomes a ``Claim`` plus an
``Argument`` holding its citations. The pinpoint is taken from the chunk, not from the model.

What counsel sees: the issue, its own position, the other positions, S0's facts and assumptions,
and the evidence registered for the issue, in ID order. ``max_evidence_chars`` cuts that list
for a small-context model. Counsel may cite only the items it was shown.

Checks (``validate``), each pointing at what to drop after the one repair:

- a claim needs text, reasoning and at least one citation, and at most ``max_claims`` count;
- a citation's ID must be one of the items shown (written loosely is fine: "E-3" is "E-003");
- its quote must be in that item (invariant 2, ``CaseRecord.check_citation``).

``CounselBase`` holds what every counsel mode shares (the evidence shown, the citation checks,
writing a claim with its argument); S5's rebuttal mode is in ``agents/rebuttal.py``.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field, ValidationError

from juris.agents.base import Agent, Problem
from juris.config import Role
from juris.events import CaseView
from juris.models import Argument, Citation
from juris.models.common import Stage
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.citing import clean_evidence_id, pinpoint

MAX_CLAIMS = 6


class CounselCitation(BaseModel):
    evidence_id: str = Field(description='An ID from the evidence list, e.g. "E-003"')
    quote: str = Field(description="Copied word for word from that evidence item")


class CounselClaim(BaseModel):
    text: str = Field(description="One atomic legal proposition for your position")
    reasoning: str = Field(description="One sentence: how the quoted evidence supports it")
    citations: list[CounselCitation] = Field(description="At least one")


class CounselOutput(BaseModel):
    claims: list[CounselClaim] = Field(description="Your claims; may be empty")


class CounselBase[Out: BaseModel](Agent[Out]):
    """One counsel: an issue and the position it argues."""

    role = Role.COUNSEL

    def __init__(
        self,
        issue_id: str,
        position_id: str,
        *,
        max_evidence_chars: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name or f"counsel_{position_id[-1].lower()}")
        self.issue_id = issue_id
        self.position_id = position_id
        self.max_evidence_chars = max_evidence_chars

    def candidates(self, view: CaseView) -> list[str]:
        """The evidence counsel may see, in the order it is shown: the issue's, by ID."""
        return [e.id for e in view.evidence.values() if self.issue_id in e.issue_ids]

    def shown(self, view: CaseView) -> list[str]:
        """The evidence IDs counsel sees: ``candidates`` within the budget."""
        ids: list[str] = []
        used = 0
        for evidence_id in self.candidates(view):
            evidence = view.evidence[evidence_id]
            size = len(view.chunks[evidence.chunk_id].text)
            if (
                self.max_evidence_chars is not None
                and ids
                and used + size > self.max_evidence_chars
            ):
                break
            ids.append(evidence_id)
            used += size
        return ids

    def position_fields(self, view: CaseView) -> dict[str, str]:
        """The template fields every counsel prompt has: issue, positions, facts, evidence."""
        issue = view.issues[self.issue_id]
        own = next(p for p in issue.positions if p.id == self.position_id)
        others = [p for p in issue.positions if p.id != self.position_id]
        facts = [*view.facts, *(f"(assumed) {a}" for a in view.assumptions)]
        blocks = []
        for evidence_id in self.shown(view):
            evidence = view.evidence[evidence_id]
            chunk = view.chunks[evidence.chunk_id]
            title = view.documents[evidence.document_id].title
            blocks.append(f"[{evidence_id}] {title}, {pinpoint(chunk, None)}\n{chunk.text}")
        return {
            "issue_id": issue.id,
            "issue": issue.question,
            "position_id": own.id,
            "position": own.statement,
            "other_positions": "\n".join(f"{p.id}: {p.statement}" for p in others),
            "facts": "\n".join(f"- {f}" for f in facts) or "(none stated)",
            "evidence": "\n\n".join(blocks) or "(no evidence was found for this issue)",
        }

    def item_problems(
        self,
        at: tuple[str, int],
        what: str,
        text: str,
        reasoning: str,
        citations: Sequence[CounselCitation],
        shown: set[str],
        record: CaseRecord,
    ) -> list[Problem]:
        """The checks on one claim-like item: text, reasoning, and each citation."""
        problems: list[Problem] = []
        if not text.strip():
            problems.append(Problem(f"the {what} is empty", at))
        if not reasoning.strip():
            problems.append(Problem(f"the {what} has no reasoning", at))
        if not citations:
            problems.append(Problem(f"every {what} needs at least one citation", at))
        for k, c in enumerate(citations):
            problem = self._citation_problem(c, shown, record)
            if problem is not None:
                problems.append(Problem(problem, (*at, "citations", k)))
        return problems

    def write_claim(
        self,
        text: str,
        reasoning: str,
        cites: Sequence[CounselCitation],
        record: CaseRecord,
        *,
        stage: Stage,
    ) -> Argument:
        """A claim for counsel's position and the argument holding its citations (a repeated
        citation once)."""
        claim = record.add_claim(text, self.position_id, stage=stage, agent=self.name)
        citations: dict[tuple[str, str], Citation] = {}
        for x in cites:
            evidence_id = clean_evidence_id(x.evidence_id)
            citations.setdefault(
                (evidence_id, x.quote.strip()),
                Citation(
                    claim_id=claim.id,
                    evidence_id=evidence_id,
                    pinpoint=pinpoint(record.registry.chunk(evidence_id), None),
                    quote=x.quote.strip(),
                ),
            )
        return record.add_argument(
            [claim.id], reasoning, list(citations.values()), stage=stage, agent=self.name
        )

    @staticmethod
    def _citation_problem(
        citation: CounselCitation, shown: set[str], record: CaseRecord
    ) -> str | None:
        evidence_id = clean_evidence_id(citation.evidence_id)
        if evidence_id not in shown:
            return f"{citation.evidence_id!r} is not one of the evidence items you were given"
        try:
            checked = Citation(
                claim_id="C-001", evidence_id=evidence_id, pinpoint="-", quote=citation.quote
            )
        except ValidationError:
            return "the quote is empty"
        return record.check_citation(checked)


class Counsel(CounselBase[CounselOutput]):
    prompt = "counsel"
    duty = (
        "Counsel argues the strongest evidence-backed case for one assigned position on one "
        "issue, citing only registered evidence with verbatim quotes."
    )
    output_model = CounselOutput
    max_tokens = 3000

    def __init__(
        self,
        issue_id: str,
        position_id: str,
        *,
        max_claims: int = MAX_CLAIMS,
        max_evidence_chars: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(issue_id, position_id, max_evidence_chars=max_evidence_chars, name=name)
        self.max_claims = max_claims

    def input_view(self, view: CaseView) -> dict[str, str]:
        return self.position_fields(view) | {"max_claims": str(self.max_claims)}

    def validate(self, output: CounselOutput, record: CaseRecord) -> list[Problem]:
        shown = set(self.shown(record.view))
        problems: list[Problem] = []
        for n, claim in enumerate(output.claims):
            at = ("claims", n)
            if n >= self.max_claims:
                problems.append(Problem(f"at most {self.max_claims} claims", at))
                continue
            problems += self.item_problems(
                at, "claim", claim.text, claim.reasoning, claim.citations, shown, record
            )
        return problems

    def apply(self, output: CounselOutput, record: CaseRecord, *, stage: Stage) -> None:
        for c in output.claims:
            self.write_claim(c.text, c.reasoning, c.citations, record, stage=stage)


def citation_counts(output: CounselOutput, problems: list[Problem]) -> tuple[int, int]:
    """(citations, citations with a problem) in one attempt."""
    return count_citations([c.citations for c in output.claims], problems)


def count_citations(
    citations: Sequence[Sequence[CounselCitation]], problems: list[Problem]
) -> tuple[int, int]:
    """(citations, citations with a problem), given each output item's citations."""
    total = sum(len(c) for c in citations)
    failing = {p.path for p in problems if "citations" in p.path}
    return total, len(failing)
