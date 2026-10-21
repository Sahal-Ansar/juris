"""Counsel in rebuttal mode (S5; IDEA_final §6.1, PLAN 7.7): each side answers the other's
strongest claims on the same issue.

Two agents per counsel (issue and position) and round, both counsel underneath (``CounselBase``,
same name and model as in S2):

1. ``CounselSearch`` asks for up to K targeted searches (``extra_searches_per_counsel``). Each
   goes through the Researcher (``targeted_search``), which registers what it finds for the
   issue before counsel sees it.
2. ``RebuttalCounsel`` writes counterarguments. Each one targets an opposing claim of the
   issue and is a claim for counsel's own position with reasoning and citations, like an S2
   claim. In the record it is a ``Claim`` plus an ``Argument`` plus a ``Counterargument`` that
   points the argument at its target. The S7 gate verifies its citations.

What counsel sees: the issue and positions, S0's facts and the **targets**, the opposing claims
not yet answered from its position, strongest first (verified, weak, proposed, unsupported; a
claim that fell is no target). Each target shows its reasoning and, per citation, the
verifier's result (S4) and the precedent fit (S3). The evidence comes in this order: what
counsel's own searches just found, the evidence the targets cite, then the rest of the issue's
pool, within ``max_evidence_chars``.

Checks (``validate``) on top of S2's: the target is one of those shown (written loosely is fine:
"C-4" is "C-004"), one counterargument per target, at most ``max_counters``, and each one cites
evidence.
"""

from pydantic import BaseModel, Field

from juris.agents.base import Problem
from juris.agents.counsel import CounselBase, CounselCitation, count_citations
from juris.agents.researcher import Researcher
from juris.events import CaseView
from juris.models import Citation, Claim
from juris.models.common import ClaimStatus, Stage
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.citing import clean_record_id

MAX_COUNTERS = 3
SEARCH_K = 3  # new chunks per targeted search
# strongest first: a verified claim is the one most worth answering
_STRENGTH = {
    ClaimStatus.VERIFIED: 0,
    ClaimStatus.SURVIVES: 0,
    ClaimStatus.WEAK: 1,
    ClaimStatus.CONTESTED: 1,
    ClaimStatus.PROPOSED: 2,
    ClaimStatus.UNSUPPORTED: 3,
}


class Counterpoint(BaseModel):
    target_claim_id: str = Field(description='The opposing claim you answer, e.g. "C-004"')
    text: str = Field(description="One atomic legal proposition for your position that answers it")
    reasoning: str = Field(description="One sentence: how the quoted evidence answers the target")
    citations: list[CounselCitation] = Field(description="At least one")


class RebuttalOutput(BaseModel):
    counterarguments: list[Counterpoint] = Field(description="Your counterarguments; may be empty")


class SearchOutput(BaseModel):
    queries: list[str] = Field(description="Searches of the case-law corpus; may be empty")


class RebuttalCounsel(CounselBase[RebuttalOutput]):
    prompt = "counsel_rebuttal"
    duty = (
        "Counsel answers the opposing side's strongest claims on one issue with "
        "counterarguments for its own position, citing only registered evidence with verbatim "
        "quotes."
    )
    output_model = RebuttalOutput
    max_tokens = 2500

    def __init__(
        self,
        issue_id: str,
        position_id: str,
        *,
        max_counters: int = MAX_COUNTERS,
        max_evidence_chars: int | None = None,
        round_no: int = 1,
    ) -> None:
        super().__init__(issue_id, position_id, max_evidence_chars=max_evidence_chars)
        self.max_counters = max_counters
        self.round_no = round_no
        self.found: list[str] = []  # E- IDs this counsel's searches registered this round

    @property
    def call_tags(self) -> dict[str, str]:
        return {"round": str(self.round_no)}

    def targets(self, view: CaseView) -> list[Claim]:
        """The opposing claims of the issue not yet answered from this position, strongest
        first."""
        answered = {
            r.targets
            for r in view.counterarguments.values()
            if any(
                view.claims[c].position_id == self.position_id
                for c in view.arguments[r.argument_id].claim_ids
            )
        }
        claims = [
            c
            for c in view.claims.values()
            if c.position_id.startswith(f"{self.issue_id}.")
            and c.position_id != self.position_id
            and c.status is not ClaimStatus.FALLS
            and c.id not in answered
        ]
        return sorted(claims, key=lambda c: (_STRENGTH[c.status], c.id))

    def candidates(self, view: CaseView) -> list[str]:
        pool = super().candidates(view)
        targets = {c.id for c in self.targets(view)}
        cited = [
            c.evidence_id
            for a in view.arguments.values()
            for c in a.citations
            if c.claim_id in targets
        ]
        held = set(pool)
        return [e for e in dict.fromkeys([*self.found, *cited, *pool]) if e in held]

    def describe_targets(self, view: CaseView) -> str:
        blocks = []
        for claim in self.targets(view):
            lines = [f"[{claim.id}] {claim.position_id}, {claim.status.value}: {claim.text}"]
            for argument in view.arguments.values():
                if claim.id not in argument.claim_ids:
                    continue
                lines.append(f"  reasoning: {argument.reasoning}")
                lines += [
                    f"  cites {c.evidence_id}: {_assessment(view, c, argument.id)}"
                    for c in argument.citations
                ]
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) or "(none)"

    def input_view(self, view: CaseView) -> dict[str, str]:
        return self.position_fields(view) | {
            "targets": self.describe_targets(view),
            "max_counters": str(self.max_counters),
        }

    def validate(self, output: RebuttalOutput, record: CaseRecord) -> list[Problem]:
        view = record.view
        targets = {c.id for c in self.targets(view)}
        shown = set(self.shown(view))
        answered: set[str] = set()
        problems: list[Problem] = []
        for n, counter in enumerate(output.counterarguments):
            at = ("counterarguments", n)
            if n >= self.max_counters:
                problems.append(Problem(f"at most {self.max_counters} counterarguments", at))
                continue
            target = clean_record_id(counter.target_claim_id, "C")
            if target not in targets:
                problems.append(
                    Problem(f"{counter.target_claim_id!r} is not one of the claims to answer", at)
                )
            elif target in answered:
                problems.append(Problem(f"{target} is already answered; one per claim", at))
            answered.add(target)
            problems += self.item_problems(
                at,
                "counterargument",
                counter.text,
                counter.reasoning,
                counter.citations,
                shown,
                record,
            )
        return problems

    def apply(self, output: RebuttalOutput, record: CaseRecord, *, stage: Stage) -> None:
        for c in output.counterarguments:
            argument = self.write_claim(c.text, c.reasoning, c.citations, record, stage=stage)
            target = clean_record_id(c.target_claim_id, "C")
            record.add_counterargument(target, argument.id, stage=stage, agent=self.name)


class CounselSearch(CounselBase[SearchOutput]):
    """Before the rebuttal: counsel's targeted searches, run through the Researcher."""

    prompt = "counsel_search"
    duty = (
        "Counsel asks for targeted searches of the case-law corpus to find evidence that "
        "answers the opposing side's claims on one issue."
    )
    output_model = SearchOutput
    max_tokens = 600

    def __init__(
        self,
        counsel: RebuttalCounsel,
        researcher: Researcher,
        *,
        most: int,
        per_query: int = SEARCH_K,
    ) -> None:
        super().__init__(counsel.issue_id, counsel.position_id, name=counsel.name)
        self.counsel = counsel
        self.researcher = researcher
        self.most = most
        self.per_query = per_query

    @property
    def call_tags(self) -> dict[str, str]:
        return self.counsel.call_tags

    def input_view(self, view: CaseView) -> dict[str, str]:
        issue = view.issues[self.issue_id]
        own = next(p for p in issue.positions if p.id == self.position_id)
        held = dict.fromkeys(
            view.documents[e.document_id].title
            for e in view.evidence.values()
            if self.issue_id in e.issue_ids
        )
        return {
            "issue_id": issue.id,
            "issue": issue.question,
            "position_id": own.id,
            "position": own.statement,
            "targets": self.counsel.describe_targets(view),
            "held": "\n".join(f"- {t}" for t in held) or "(none)",
            "most": str(self.most),
        }

    def validate(self, output: SearchOutput, record: CaseRecord) -> list[Problem]:
        seen: set[str] = set()
        problems: list[Problem] = []
        for n, query in enumerate(output.queries):
            at = ("queries", n)
            key = " ".join(query.lower().split())
            if n >= self.most:
                problems.append(Problem(f"at most {self.most} searches", at))
            elif not key:
                problems.append(Problem("the search is empty", at))
            elif key in seen:
                problems.append(Problem("the same search twice", at))
            seen.add(key)
        return problems

    def apply(self, output: SearchOutput, record: CaseRecord, *, stage: Stage) -> None:
        for query in output.queries:
            found = self.researcher.targeted_search(
                record, query, [self.issue_id], stage=stage, agent=self.name, k=self.per_query
            )
            self.counsel.found += [e.id for e in found]


def counter_counts(output: RebuttalOutput, problems: list[Problem]) -> tuple[int, int]:
    """(citations, citations with a problem) in one attempt."""
    return count_citations([c.citations for c in output.counterarguments], problems)


def _assessment(view: CaseView, citation: Citation, argument_id: str) -> str:
    """What S3 and S4 found about one citation of a target."""
    results = [
        v.result
        for v in view.verifications
        if v.claim_id == citation.claim_id and v.evidence_id == citation.evidence_id
    ]
    parts = [f"verification {results[-1].status.value}" if results else "not verified yet"]
    evidence = view.evidence.get(citation.evidence_id)
    card = view.precedent_cards.get(evidence.document_id) if evidence else None
    fit = card.fit_by_argument.get(argument_id) if card else None
    if fit is not None:
        parts.append(f"precedent fit {fit.value}")
    return "; ".join(parts)
