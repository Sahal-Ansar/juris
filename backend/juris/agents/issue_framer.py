"""The Issue Framer (S0; IDEA_final §6.1, PLAN 7.3): the question becomes facts, assumptions,
1-4 issues with two neutral positions each, and per issue the search queries and statute
sections the Researcher will run (S1). It does not answer the question.

Checks (``validate``), each dropping only what it points at:

- at most ``max_issues`` issues, each with a question;
- at most ``positions_per_issue`` positions, non-empty and not repeated; an issue left with
  fewer than two is dropped;
- 2 to ``MAX_QUERIES`` non-empty queries per issue (an issue left with fewer is dropped);
- statute references the alias table can read (``retrieval.lookup.parse_provision``).

Label bias: the model tends to put the answer it favours first. So which position becomes
``A`` is decided by a seeded coin per issue (``order_seed``), not by the model's order. Same
seed, same labels.
"""

import random
from collections.abc import Sequence

from pydantic import BaseModel, Field

from juris.agents.base import Agent, Problem
from juris.config import Role
from juris.events import CaseView
from juris.models.common import Stage
from juris.pipeline.case_record import CaseRecord
from juris.retrieval.lookup import parse_provision

AGENT = "issue_framer"
MAX_QUERIES = 5


class FramedIssue(BaseModel):
    question: str = Field(description="One precise legal question, as a court would frame it")
    positions: list[str] = Field(description="Exactly two neutral, mutually exclusive answers")
    queries: list[str] = Field(description="2 to 5 short search queries in a judgment's words")
    statute_refs: list[str] = Field(
        description='Sections it turns on, e.g. "s. 16 of the Indian Contract Act"'
    )


class FramingOutput(BaseModel):
    facts: list[str] = Field(description="The facts as stated")
    assumptions: list[str] = Field(description="What had to be assumed")
    issues: list[FramedIssue] = Field(description="1 to 4 issues")


class IssueFramer(Agent[FramingOutput]):
    role = Role.ISSUE_FRAMER
    prompt = "issue_framer"
    duty = (
        "the Issue Framer turns a question into the precise legal issues it raises, with two "
        "neutral positions on each and the searches that will find the evidence."
    )
    output_model = FramingOutput
    max_tokens = 3000

    def __init__(
        self,
        facts: str,
        *,
        order_seed: int,
        max_issues: int = 4,
        positions_per_issue: int = 2,
        name: str = AGENT,
    ) -> None:
        super().__init__(name)
        self.facts = facts
        self.order_seed = order_seed
        self.max_issues = max_issues
        self.positions_per_issue = positions_per_issue
        self.swapped: list[str] = []  # issues whose positions were relabelled B -> A

    def input_view(self, view: CaseView) -> dict[str, str]:
        return {
            "question": view.question,
            "facts": self.facts.strip() or "(none given beyond the question)",
            "max_issues": str(self.max_issues),
            "max_queries": str(MAX_QUERIES),
        }

    def validate(self, output: FramingOutput, record: CaseRecord) -> list[Problem]:
        problems: list[Problem] = []
        if not output.issues:
            return [Problem("frame at least one issue")]
        for n, issue in enumerate(output.issues):
            at = ("issues", n)
            if n >= self.max_issues:
                problems.append(Problem(f"at most {self.max_issues} issues", at))
                continue
            if not issue.question.strip():
                problems.append(Problem("the issue's question is empty", at))
            problems += _list_problems(
                issue.positions, (*at, "positions"), "position", self.positions_per_issue
            )
            if len(_kept(issue.positions, self.positions_per_issue)) < 2:
                problems.append(Problem("an issue needs two distinct positions", at))
            problems += _list_problems(issue.queries, (*at, "queries"), "query", MAX_QUERIES)
            if len(_kept(issue.queries, MAX_QUERIES)) < 2:
                problems.append(Problem("an issue needs at least two search queries", at))
            for k, ref in enumerate(issue.statute_refs):
                if parse_provision(ref) is None:
                    problems.append(
                        Problem(
                            f"{ref!r} is not a section of the Indian Contract Act, the Specific "
                            "Relief Act or the Sale of Goods Act",
                            (*at, "statute_refs", k),
                        )
                    )
        return problems

    def apply(self, output: FramingOutput, record: CaseRecord, *, stage: Stage) -> None:
        facts = [f.strip() for f in output.facts if f.strip()]
        assumptions = [a.strip() for a in output.assumptions if a.strip()]
        for n, framed in enumerate(output.issues):
            positions = [p.strip() for p in framed.positions]
            if random.Random(f"{self.order_seed}:{n}").random() < 0.5:
                positions.reverse()
                swap = True
            else:
                swap = False
            issue = record.frame_issue(
                framed.question,
                positions,
                stage=stage,
                agent=self.name,
                facts=facts if n == 0 else (),  # the record keeps one list for the case
                assumptions=assumptions if n == 0 else (),
            )
            if swap:
                self.swapped.append(issue.id)
            for query in framed.queries:
                record.add_query(query, [issue.id], stage=stage, agent=self.name)
            for ref in framed.statute_refs:
                record.add_query(ref, [issue.id], stage=stage, agent=self.name, kind="provision")


def _kept(items: Sequence[str], most: int) -> list[str]:
    """What survives the per-item checks: non-empty, first occurrences, up to ``most``."""
    seen: list[str] = []
    for item in items:
        key = " ".join(item.split()).casefold()
        if key and key not in seen:
            seen.append(key)
    return seen[:most]


def _list_problems(
    items: Sequence[str], at: tuple[str | int, ...], what: str, most: int
) -> list[Problem]:
    """Empty, repeated and surplus items, each pointing at itself."""
    problems: list[Problem] = []
    seen: set[str] = set()
    for k, item in enumerate(items):
        key = " ".join(item.split()).casefold()
        if not key:
            problems.append(Problem(f"the {what} is empty", (*at, k)))
        elif key in seen:
            problems.append(Problem(f"the {what} repeats an earlier one", (*at, k)))
        elif len(seen) >= most:
            problems.append(
                Problem(f"only {most} are allowed; this {what} is one too many", (*at, k))
            )
        else:
            seen.add(key)
    return problems
