"""Score answers with every metric; per-item values, and means with bootstrap 95% CIs."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from juris.eval.juris_eval import JurisEvalItem
from juris.eval.metrics.authorities import authority_recall
from juris.eval.metrics.citations import EntailmentJudge, citation_faithfulness, citation_validity
from juris.eval.metrics.claims import uncited_sentences, unsupported_claim_rate
from juris.eval.metrics.common import MetricResult
from juris.eval.metrics.cost import CallLike, cost_latency
from juris.eval.metrics.graders import Grader, key_point_coverage, quality_rubric
from juris.eval.stats import Aggregate, aggregate
from juris.events.fold import CaseView

METRICS = (
    "answered",
    "citation_validity",
    "citation_faithfulness",
    "unsupported_claim_rate",
    "authority_recall",
    "key_point_coverage",
    "quality_rubric",
    "cost_latency",
)


@dataclass
class Answer:
    """One config's run on one Juris-Eval item."""

    item: JurisEvalItem
    view: CaseView
    calls: Sequence[CallLike] | None = None
    wall_seconds: float | None = None


@dataclass
class ItemScores:
    item_id: str
    results: dict[str, MetricResult]

    @property
    def values(self) -> dict[str, float | None]:
        return {k: v for r in self.results.values() for k, v in r.values.items()}

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "values": self.values,
            "details": {name: r.details for name, r in self.results.items()},
        }


@dataclass
class SuiteResult:
    items: list[ItemScores] = field(default_factory=list)

    def aggregate(self) -> dict[str, Aggregate | None]:
        """Every value name, in metric order, with its mean over the items that have one."""
        names = list(dict.fromkeys(k for i in self.items for k in i.values))
        return {name: aggregate([i.values.get(name) for i in self.items]) for name in names}

    def table(self) -> str:
        rows = ["| Metric | Mean | 95% CI | n |", "|---|---|---|---|"]
        for name, agg in self.aggregate().items():
            if agg is None:
                rows.append(f"| {name} | n/a | n/a | 0 |")
            else:
                rows.append(f"| {name} | {agg.mean:.3f} | {agg.lo:.3f}-{agg.hi:.3f} | {agg.n} |")
        return "\n".join(rows)


async def score_answer(
    answer: Answer, *, grader: Grader | None = None, judge: EntailmentJudge | None = None
) -> ItemScores:
    """All metrics for one answer. Without ``grader`` the LLM-graded values are None and
    uncited summary sentences all count as assertions; without ``judge`` pairs the run did
    not verify are left out of faithfulness."""
    item, view = answer.item, answer.view
    answered = view.analysis is not None
    sentences = [s.text for s in uncited_sentences(view.analysis)] if view.analysis else []
    faithfulness_task = citation_faithfulness(view, judge)
    if grader is not None and answered:
        faithfulness, (key_points, rubric, legal) = await asyncio.gather(
            faithfulness_task, grader.grade(item, view, sentences)
        )
    else:
        faithfulness = await faithfulness_task
        key_points, rubric, legal = None, None, None
    results = [
        MetricResult("answered", {"answered": 1.0 if answered else 0.0}),
        citation_validity(view),
        faithfulness,
        unsupported_claim_rate(view, legal),
        authority_recall(view, item),
        key_point_coverage(item, key_points, answered=answered),
        quality_rubric(rubric),
        cost_latency(view, answer.calls, answer.wall_seconds),
    ]
    return ItemScores(item.id, {r.name: r for r in results})


async def score_answers(
    answers: Sequence[Answer],
    *,
    grader: Grader | None = None,
    judge: EntailmentJudge | None = None,
) -> SuiteResult:
    """Items are scored concurrently; the gateway's semaphore limits the grader calls."""
    scored = await asyncio.gather(*(score_answer(a, grader=grader, judge=judge) for a in answers))
    return SuiteResult(list(scored))
