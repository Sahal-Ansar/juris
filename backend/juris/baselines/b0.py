"""Baseline B0, LLM only (PLAN 6.2, IDEA_final §5): question -> one model call -> answer.

No retrieval. The model returns a ``CaseAnalysis``-shaped answer (``B0Answer``) that names
its authorities in free text. A post-processor (``authorities.AuthorityResolver``) looks each
one up in the corpus, so the share that cannot be found (the hallucinated-authority rate, an
upper bound) is recorded with the run's scores (``system`` metric, ``b0.*`` values).

The ``CaseAnalysis`` names only authorities found in the corpus: judgments by document ID,
statute sections as sources whose pinpoint is the section ID. The rest are listed in its
limitations. B0 registers no evidence, so it has no quoted citations to validate.
"""

from collections.abc import Callable
from string import Template
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import create_engine

from juris.baselines.authorities import AuthorityResolver, Resolution, summarise
from juris.baselines.base import RunContext
from juris.config import Role, get_settings
from juris.events.catalog import (
    AgentCompletedPayload,
    AgentStartedPayload,
    AnalysisReadyPayload,
    IssueFramedPayload,
)
from juris.llm.types import ChatMessage
from juris.models import (
    CaseAnalysis,
    CreatedBy,
    Issue,
    IssueAnalysis,
    Position,
    PositionAnalysis,
    SourceRef,
)
from juris.models.analysis import SUMMARY_MAX_WORDS
from juris.models.common import Confidence, Leaning, Stage
from juris.prompts import load_prompt
from juris.retrieval.lookup import Lookup

AGENT = "b0"
PROMPT = "baseline_b0"
# Limitation text for authorities the analysis cannot rely on, by resolution status.
UNCHECKED = {
    "unverified": "Cited from the model's memory and not found in the corpus or its citations",
    "mismatch": "The citation given names a different judgment",
    "outside_corpus": "A real judgment outside the corpus slice, not checked",
    "unknown_act": "A statute outside the corpus, not checked",
}


# ---- the model's answer --------------------------------------------------------------------


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class B0Authority(_Out):
    number: int = Field(description="1, 2, 3 ... as referred to from the positions")
    kind: Literal["judgment", "statute"]
    name: str = Field(description="Case name, or the Act and section")
    citation: str | None = Field(default=None, description="Reporter citation of a judgment")
    proposition: str


class B0Position(_Out):
    statement: str = Field(description="The position, stated neutrally")
    summary: str = Field(description="The strongest argument for it")
    authorities: list[int] = Field(default_factory=list, description="Authority numbers")


class B0Issue(_Out):
    question: str
    position_a: B0Position
    position_b: B0Position
    leaning: Leaning
    confidence: Confidence
    confidence_reason: str
    unresolved_questions: list[str] = Field(default_factory=list)


class B0Answer(_Out):
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    issues: Annotated[list[B0Issue], Field(min_length=1, max_length=4)]
    authorities: list[B0Authority] = Field(default_factory=list)
    overall_summary: str
    limitations: list[str] = Field(default_factory=list)


# ---- the system ----------------------------------------------------------------------------


def _local_resolver() -> AuthorityResolver:
    return AuthorityResolver(Lookup(create_engine(get_settings().database_url())))


class B0System:
    corpus_snapshot_id = "mvp_contract-1f53c208a8"  # the slice the lookups search (D-019)
    embedding_model = "none"  # no retrieval

    def __init__(self, resolver: Callable[[], AuthorityResolver] = _local_resolver) -> None:
        self._make_resolver = resolver
        self._resolver: AuthorityResolver | None = None

    @property
    def resolver(self) -> AuthorityResolver:
        if self._resolver is None:
            self._resolver = self._make_resolver()
        return self._resolver

    async def run(self, ctx: RunContext) -> None:
        item, emit = ctx.item, ctx.emitter.emit
        model = ctx.config.model_for(Role.BASELINE)
        template = load_prompt(PROMPT)
        emit(AgentStartedPayload(role=Role.BASELINE, model=model.name), agent=AGENT)
        result = await ctx.gateway.complete(
            [
                ChatMessage(
                    role="user",
                    content=Template(template.body).substitute(
                        question=item.question, facts=item.facts
                    ),
                )
            ],
            model=model,
            response_model=B0Answer,
            seed=ctx.seed,
            max_tokens=6000,
            tags={
                "agent": AGENT,
                "prompt": f"{template.id}@v{template.version}",
                "item": item.id,
            },
        )
        emit(
            AgentCompletedPayload(
                role=Role.BASELINE,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cost_usd=result.usage.cost_usd,
                cached=result.cached,
            ),
            agent=AGENT,
        )
        answer = result.parsed
        assert answer is not None  # the gateway raises if the output never validates

        resolutions = {a.number: self._resolve(a) for a in answer.authorities}
        ctx.values.update(summarise(list(resolutions.values()), AGENT))
        ctx.details["authorities"] = [
            {"number": n, **r.to_json()} for n, r in sorted(resolutions.items())
        ]

        framed = CreatedBy(stage=Stage.S0, agent=AGENT)
        issues = [
            Issue(
                id=f"I-{n}",
                question=i.question,
                positions=[
                    Position(id=f"I-{n}.A", statement=i.position_a.statement),
                    Position(id=f"I-{n}.B", statement=i.position_b.statement),
                ],
                created_by=framed,
            )
            for n, i in enumerate(answer.issues, 1)
        ]
        for issue in issues:
            emit(
                IssueFramedPayload(issue=issue, facts=answer.facts, assumptions=answer.assumptions),
                stage=Stage.S0,
                agent=AGENT,
            )
        emit(
            AnalysisReadyPayload(
                analysis=self._analysis(item.question, answer, issues, resolutions)
            ),
            stage=Stage.S9,
            agent=AGENT,
        )

    def _resolve(self, a: B0Authority) -> Resolution:
        if a.kind == "statute":
            return self.resolver.statute(a.name, a.citation)
        return self.resolver.judgment(a.name, a.citation)

    @staticmethod
    def _analysis(
        question: str,
        answer: B0Answer,
        issues: list[Issue],
        resolutions: dict[int, Resolution],
    ) -> CaseAnalysis:
        def judgments(numbers: list[int]) -> list[str]:
            found: dict[str, None] = {}
            for n in numbers:
                r = resolutions.get(n)
                if r and r.kind == "judgment" and r.status == "corpus" and r.doc_id:
                    found.setdefault(r.doc_id, None)
            return list(found)

        analyses = [
            IssueAnalysis(
                issue_id=issue.id,
                issue_text=b0.question,
                positions=[
                    PositionAnalysis(
                        position_id=f"{issue.id}.{side}",
                        summary=p.summary or p.statement,
                        supporting_authorities=judgments(p.authorities),
                    )
                    for side, p in (("A", b0.position_a), ("B", b0.position_b))
                ],
                leaning=b0.leaning,
                confidence=b0.confidence,
                confidence_reason=b0.confidence_reason,
                unresolved_questions=b0.unresolved_questions,
            )
            for issue, b0 in zip(issues, answer.issues, strict=True)
        ]
        sources: list[SourceRef] = []
        for r in resolutions.values():
            if r.status != "corpus" or r.doc_id is None:
                continue
            if r.kind == "statute" and r.section_id is not None:
                sources.append(SourceRef(document_id=r.doc_id, pinpoints_used=[r.section_id]))
            elif r.kind == "judgment":
                sources.append(SourceRef(document_id=r.doc_id, citation=r.citation))
        missing = [
            f"{UNCHECKED[r.status]}: {r.name}" + (f", {r.citation}" if r.citation else "")
            for r in resolutions.values()
            if r.status in UNCHECKED
        ]
        words = answer.overall_summary.split()
        limitations = list(answer.limitations) + missing
        if len(words) > SUMMARY_MAX_WORDS:
            limitations.append(f"The model's summary was cut to {SUMMARY_MAX_WORDS} words.")
        summary = " ".join(words[:SUMMARY_MAX_WORDS]) or "No summary was given."
        return CaseAnalysis(
            question=question,
            facts=answer.facts,
            assumptions=answer.assumptions,
            issues=analyses,
            overall_summary=summary,
            limitations=limitations,
            sources=sources,
            created_by=CreatedBy(stage=Stage.S9, agent=AGENT),
        )
