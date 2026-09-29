"""LLM-graded metrics: key-point coverage and the quality rubric (IDEA_final §11.2).

``Grader`` makes the calls (one per item per grader, through the gateway, so they are cached
and costed); the metric functions turn its structured grades into numbers. The prompts are
versioned templates in ``juris/prompts/`` (``grader_*.md``, ``string.Template`` fields).

The grader model is a parameter. It should differ from the generator's model where possible
(IDEA_final §11.3); choosing one is the experiment's decision (5.5), not this module's.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from juris.config import ModelSpec
from juris.eval.juris_eval import Authority, JurisEvalItem
from juris.eval.metrics.common import MetricResult, ratio
from juris.events.fold import CaseView
from juris.llm.gateway import LLMGateway
from juris.llm.types import ChatMessage
from juris.prompts import load_prompt

KEY_POINTS_PROMPT = "grader_key_points"
RUBRIC_PROMPT = "grader_quality_rubric"
ASSERTIONS_PROMPT = "grader_assertions"

RUBRIC_CRITERIA = (
    "correctness",
    "completeness",
    "reasoning_consistency",
    "uncertainty_calibration",
)


class _Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeyPointJudgement(_Grade):
    index: int = Field(description="The key point's number, from 1")
    verdict: Literal["covered", "partially", "missing"]
    reason: str = Field(description="One sentence")


class KeyPointGrade(_Grade):
    judgements: list[KeyPointJudgement]


class CriterionScore(_Grade):
    score: Annotated[int, Field(ge=1, le=5)]
    reason: str = Field(description="One or two sentences")


class RubricGrade(_Grade):
    correctness: CriterionScore
    completeness: CriterionScore
    reasoning_consistency: CriterionScore
    uncertainty_calibration: CriterionScore


class AssertionJudgement(_Grade):
    index: int = Field(description="The sentence's number, from 1")
    legal_assertion: bool


class AssertionGrade(_Grade):
    sentences: list[AssertionJudgement]


class GraderOutputError(ValueError):
    """The grade is well-formed but doesn't match the item (wrong or missing numbers)."""


def _check_indices(indices: Sequence[int], expected: int, what: str) -> None:
    if sorted(indices) != list(range(1, expected + 1)):
        raise GraderOutputError(f"expected {what} 1-{expected}, got {sorted(indices)}")


# ---- rendering -----------------------------------------------------------------------------


def _bullets(lines: Sequence[str]) -> str:
    return "\n".join(f"- {line}" for line in lines) or "(none)"


def _numbered(lines: Sequence[str]) -> str:
    return "\n".join(f"{n}. {line}" for n, line in enumerate(lines, 1))


def _document_label(view: CaseView, doc_id: str) -> str:
    doc = view.documents.get(doc_id)
    if doc is None:
        return doc_id
    return f"{doc.title} [{doc.citations[0]}]" if doc.citations else doc.title


def render_analysis(view: CaseView) -> str:
    """The analysis as a grader reads it: plain text, authorities named by title."""
    a = view.analysis
    if a is None:
        return "(no answer)"
    lines = [f"Question: {a.question}", "Facts:", _bullets(a.facts)]
    lines += ["Assumptions:", _bullets(a.assumptions)]
    for issue in a.issues:
        lines += ["", f"Issue {issue.issue_id}: {issue.issue_text}"]
        for p in issue.positions:
            lines.append(f"  Position {p.position_id}: {p.summary}")
            if p.supporting_authorities:
                named = "; ".join(_document_label(view, d) for d in p.supporting_authorities)
                lines.append(f"    Authorities: {named}")
            if p.key_evidence:
                refs = ", ".join(f"{r.evidence_id} ({r.pinpoint})" for r in p.key_evidence)
                lines.append(f"    Key evidence: {refs}")
        for c in issue.key_conflicts:
            weightier = _document_label(view, c.weightier) if c.weightier else "neither"
            lines.append(
                f"  Conflict: {_document_label(view, c.authority_a)} vs "
                f"{_document_label(view, c.authority_b)}. {c.distinction} "
                f"Weightier: {weightier}. {c.reason}"
            )
        lines.append(
            f"  Leaning: {issue.leaning} (confidence {issue.confidence}: {issue.confidence_reason})"
        )
        if issue.unresolved_questions:
            lines += ["  Unresolved questions:", _bullets(issue.unresolved_questions)]
    lines += ["", f"Overall summary: {a.overall_summary}", "Limitations:", _bullets(a.limitations)]
    return "\n".join(lines)


def _authority_line(a: Authority) -> str:
    return f"{a.title} [{a.citation}]: {a.proposition}"


def render_reference(item: JurisEvalItem) -> str:
    """The benchmark's gold for the rubric grader: issues, authorities, sections, key points."""
    return "\n".join(
        [
            "Issues:",
            _bullets(item.gold_issues),
            "Supporting authorities:",
            _bullets([_authority_line(a) for a in item.gold_supporting_authorities]),
            "Contrary or limiting authorities:",
            _bullets([_authority_line(a) for a in item.gold_contrary_authorities]),
            "Statute sections:",
            _bullets(item.gold_sections),
            "Key points:",
            _numbered(item.key_points),
        ]
    )


# ---- the grader ----------------------------------------------------------------------------


@dataclass
class Grader:
    gateway: LLMGateway
    model: ModelSpec
    seed: int = 0

    async def _ask[T: BaseModel](
        self, prompt: str, fields: dict[str, str], response_model: type[T], item_id: str
    ) -> T:
        template = load_prompt(prompt)
        result = await self.gateway.complete(
            [ChatMessage(role="user", content=Template(template.body).substitute(fields))],
            model=self.model,
            response_model=response_model,
            temperature=0.0,
            seed=self.seed,
            tags={
                "agent": prompt,
                "prompt": f"{template.id}@v{template.version}",
                "item": item_id,
            },
        )
        if result.parsed is None:
            raise GraderOutputError(f"{prompt}: no structured output")
        return result.parsed

    async def key_points(self, item: JurisEvalItem, view: CaseView) -> KeyPointGrade:
        grade = await self._ask(
            KEY_POINTS_PROMPT,
            {
                "question": item.question,
                "facts": item.facts,
                "key_points": _numbered(item.key_points),
                "answer": render_analysis(view),
            },
            KeyPointGrade,
            item.id,
        )
        _check_indices([j.index for j in grade.judgements], len(item.key_points), "key points")
        return grade

    async def rubric(self, item: JurisEvalItem, view: CaseView) -> RubricGrade:
        return await self._ask(
            RUBRIC_PROMPT,
            {
                "question": item.question,
                "facts": item.facts,
                "reference": render_reference(item),
                "answer": render_analysis(view),
            },
            RubricGrade,
            item.id,
        )

    async def assertions(
        self, item: JurisEvalItem, view: CaseView, sentences: Sequence[str]
    ) -> list[bool]:
        """For each sentence, whether it makes a legal assertion that needs an authority."""
        if not sentences:
            return []
        grade = await self._ask(
            ASSERTIONS_PROMPT,
            {"question": item.question, "sentences": _numbered(sentences)},
            AssertionGrade,
            item.id,
        )
        _check_indices([s.index for s in grade.sentences], len(sentences), "sentences")
        by_index = {s.index: s.legal_assertion for s in grade.sentences}
        return [by_index[n] for n in range(1, len(sentences) + 1)]

    async def grade(
        self, item: JurisEvalItem, view: CaseView, sentences: Sequence[str]
    ) -> tuple[KeyPointGrade, RubricGrade, list[bool]]:
        """All three grader calls for one item, concurrently."""
        return await asyncio.gather(
            self.key_points(item, view),
            self.rubric(item, view),
            self.assertions(item, view, sentences),
        )


# ---- metrics from grades -------------------------------------------------------------------


_CREDIT = {"covered": 1.0, "partially": 0.5, "missing": 0.0}


def key_point_coverage(
    item: JurisEvalItem, grade: KeyPointGrade | None, *, answered: bool = True
) -> MetricResult:
    """Mean credit over the key points: covered 1, partially 0.5, missing 0. ``.full`` is the
    share fully covered. No answer -> 0; no grade (grader off) -> None."""
    n = len(item.key_points)
    if not answered:
        return MetricResult(
            "key_point_coverage",
            {"key_point_coverage": 0.0, "key_point_coverage.full": 0.0},
            {"note": "no analysis"},
        )
    if grade is None:
        return MetricResult(
            "key_point_coverage",
            {"key_point_coverage": None, "key_point_coverage.full": None},
            {"note": "not graded"},
        )
    verdicts = {j.index: j for j in grade.judgements}
    credit = sum(_CREDIT[j.verdict] for j in grade.judgements)
    return MetricResult(
        "key_point_coverage",
        {
            "key_point_coverage": credit / n,
            "key_point_coverage.full": ratio(
                sum(1 for j in grade.judgements if j.verdict == "covered"), n
            ),
        },
        {
            "judgements": [
                {"key_point": item.key_points[i - 1], "verdict": j.verdict, "reason": j.reason}
                for i, j in sorted(verdicts.items())
            ]
        },
    )


def quality_rubric(grade: RubricGrade | None) -> MetricResult:
    """Mean of the four 1-5 criteria, and each criterion. None when not graded or no answer."""
    if grade is None:
        empty: dict[str, float | None] = {"quality_rubric": None}
        empty |= {f"quality_rubric.{c}": None for c in RUBRIC_CRITERIA}
        return MetricResult("quality_rubric", empty, {"note": "not graded"})
    scores = {c: getattr(grade, c) for c in RUBRIC_CRITERIA}
    return MetricResult(
        "quality_rubric",
        {"quality_rubric": sum(s.score for s in scores.values()) / len(scores)}
        | {f"quality_rubric.{c}": float(s.score) for c, s in scores.items()},
        {"reasons": {c: s.reason for c, s in scores.items()}},
    )
