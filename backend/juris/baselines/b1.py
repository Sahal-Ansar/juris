"""Baseline B1, single-agent RAG (PLAN 6.3, IDEA_final §5): question -> hybrid retrieval ->
one model call -> cited answer.

Same retriever, model, citation format and verifier as every config, so the comparison with
Juris isolates the multi-agent structure. Steps, with the stage each is recorded under:

- S0: the question becomes search queries (one structured call, or the question itself) and
  the answer's issues are framed;
- S1: the hybrid search's chunks are registered as evidence ``E-001``, ``E-002`` ... in the
  order the model saw them, so the model's evidence IDs are the registry's;
- S9: one structured call writes the answer (``B1Answer``): per issue two positions, each with
  claims and their quoted citations. Optionally one repair call fixes quotes that are not in
  their chunk (IDEA_final §8's single repair attempt);
- S10: the Citation Verifier (6.1) checks every citation, then the analysis is built from the
  verified and weak ones only (§8 invariant 3).

Claims, arguments and verification results are recorded as events like Juris's, so the answer
metrics read B1 in the same way.
"""

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from string import Template
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from juris.baselines.base import RunContext
from juris.config import Role
from juris.events.catalog import (
    AgentCompletedPayload,
    AgentStartedPayload,
    AnalysisReadyPayload,
    ArgumentCreatedPayload,
    ClaimCreatedPayload,
    IssueFramedPayload,
    QueryIssuedPayload,
)
from juris.llm.errors import StructuredOutputError
from juris.llm.types import ChatMessage, LLMResult
from juris.models import (
    Argument,
    CaseAnalysis,
    Chunk,
    Citation,
    Claim,
    CreatedBy,
    Document,
    EvidenceRef,
    Issue,
    IssueAnalysis,
    Position,
    PositionAnalysis,
    SourceRef,
    VerificationResult,
)
from juris.models.analysis import SUMMARY_MAX_WORDS
from juris.models.common import (
    Confidence,
    DocumentKind,
    Leaning,
    Stage,
    VerificationStatus,
)
from juris.pipeline.citing import clean_evidence_id, pinpoint
from juris.pipeline.evidence_registry import EvidenceRegistry
from juris.prompts import load_prompt
from juris.retrieval.service import Hit, RetrievalService
from juris.verify import CitationVerifier, LLMEntailmentJudge, check_citation

AGENT = "b1"
QUERIES_PROMPT = "baseline_b1_queries"
ANSWER_PROMPT = "baseline_b1"
REPAIR_PROMPT = "baseline_b1_repair"
MIN_QUERIES = 3
SUPPORTED = (VerificationStatus.VERIFIED, VerificationStatus.WEAK)
NO_REASONING = "The position's claim, with the evidence quoted for it."


# ---- the model's outputs -------------------------------------------------------------------


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class B1Queries(_Out):
    queries: list[str] = Field(min_length=1, description="Short search queries")


class B1Citation(_Out):
    evidence_id: str = Field(description="An ID from the evidence list, e.g. E-003")
    quote: str = Field(description="Copied word for word from that evidence item")


class B1Claim(_Out):
    text: str = Field(description="One atomic legal proposition")
    citations: list[B1Citation] = Field(min_length=1, description="At least one")


class B1Position(_Out):
    statement: str = Field(description="The position, stated neutrally")
    summary: str = Field(description="The strongest argument for it")
    # Required, not defaulted: a local model leaves an optional field out (the first smoke
    # run of B1 had no claims at all). The list may still be empty.
    claims: list[B1Claim] = Field(description="The position's claims, each with its citations")


class B1Issue(_Out):
    question: str
    position_a: B1Position
    position_b: B1Position
    leaning: Leaning
    confidence: Confidence
    confidence_reason: str
    unresolved_questions: list[str] = Field(default_factory=list)


class B1Answer(_Out):
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    issues: Annotated[list[B1Issue], Field(min_length=1, max_length=4)]
    overall_summary: str = Field(description="At most 200 words; cites evidence as [E-001]")
    limitations: list[str] = Field(default_factory=list)


class B1Fix(_Out):
    number: int = Field(description="The item's number, from 1")
    quote: str | None = Field(description="The corrected quote, or null to drop the citation")


class B1Repair(_Out):
    fixes: list[B1Fix] = Field(default_factory=list)


# ---- retrieval -----------------------------------------------------------------------------


class SearchService(Protocol):
    """What B1 needs from retrieval (``RetrievalService`` has it; tests pass a fake)."""

    def search(self, queries: Sequence[str], *, k: int = 10, rerank: bool = True) -> list[Hit]: ...

    def records(self, chunk_id: str) -> tuple[Document, Chunk]: ...


def _local_service(rerank: bool) -> SearchService:
    """The hybrid retriever on the CPU: the GPU belongs to the local LLM (Ollama), and a few
    queries per case do not need it. Weights are loaded once per process."""
    return RetrievalService.local(rerank=rerank, device="cpu")


@dataclass(frozen=True)
class _Presented:
    """The evidence as the model saw it, by ID: the verifier's rule applied before registry."""

    texts: dict[str, str]

    def __contains__(self, evidence_id: str, /) -> bool:
        return evidence_id in self.texts

    def chunk_text(self, evidence_id: str) -> str | None:
        return self.texts.get(evidence_id)


def evidence_block(hits: Sequence[Hit], ids: Sequence[str] | None = None) -> str:
    """``[E-001] <header>`` and the chunk text, one block per hit, in hit order. ``ids``
    labels the hits when they are not ``E-001``, ``E-002``... (B2 shows a subset)."""
    labels = ids if ids is not None else [f"E-{n:03d}" for n in range(1, len(hits) + 1)]
    return "\n\n".join(
        f"[{i}] {h.context_header or h.title}\n{h.text}" for i, h in zip(labels, hits, strict=True)
    )


@dataclass(frozen=True)
class _Cited:
    """One citation with where it sits in the answer."""

    issue: int  # 0-based index into the answer's issues
    side: str  # "A" or "B"
    citation: Citation


class B1System:
    corpus_snapshot_id = "mvp_contract-1f53c208a8"  # the slice the search covers (D-019)
    embedding_model = "BAAI/bge-m3 + intfloat/multilingual-e5-large + lexical (RRF)"
    agent = AGENT  # names the events, the call tags and the recorded values (B2 subclasses)

    def __init__(self, service: Callable[[bool], SearchService] = _local_service) -> None:
        self._make_service = service
        self._service: SearchService | None = None

    def _search_service(self, rerank: bool) -> SearchService:
        if self._service is None:
            self._service = self._make_service(rerank)
        return self._service

    async def run(self, ctx: RunContext) -> None:
        params = ctx.config.profile.baseline
        item, emit = ctx.item, ctx.emitter.emit
        model = ctx.config.model_for(Role.BASELINE)
        emit(AgentStartedPayload(role=Role.BASELINE, model=model.name), agent=self.agent)

        # S0: the search queries
        if params.queries == "llm":
            queries = await self._queries(ctx, params.max_queries)
        else:
            queries = [item.question]
        ctx.details["queries"] = queries
        ctx.values[f"{self.agent}.queries"] = float(len(queries))
        for query in queries:
            emit(QueryIssuedPayload(query=query), stage=Stage.S0, agent=self.agent)

        # S1: retrieve (hits are numbered E-001... in this order, once per chunk)
        service = self._search_service(params.rerank)
        found = service.search(queries, k=params.evidence_k, rerank=params.rerank)
        hits = list({h.chunk_id: h for h in found}.values())

        # S9: the answer (B2 may search again here and add hits)
        answer, hits, uncited = await self._compose(ctx, service, hits)
        ctx.values[f"{self.agent}.evidence"] = float(len(hits))
        presented = _Presented({f"E-{n:03d}": h.text for n, h in enumerate(hits, 1)})
        malformed, normalised = self._normalise_ids(answer)
        ctx.values[f"{self.agent}.malformed_citations_dropped"] = float(malformed)
        ctx.values[f"{self.agent}.ids_normalised"] = float(normalised)
        if params.repair_quotes:
            await self._repair(ctx, answer, presented)
        # every claim cites (§10): drop those left without a citation
        uncited += self._drop_uncited(answer)
        ctx.values[f"{self.agent}.uncited_claims_dropped"] = float(uncited)

        framed = CreatedBy(stage=Stage.S0, agent=self.agent)
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
                agent=self.agent,
            )

        registry = self._register(ctx, service, hits, answer, issues)
        headers = {f"E-{n:03d}": h.context_header or h.title for n, h in enumerate(hits, 1)}
        claims, cited = self._record_claims(ctx, registry, answer, issues, headers)
        citations = [c.citation for c in cited]
        ctx.values[f"{self.agent}.claims"] = float(len(claims))
        ctx.values[f"{self.agent}.citations"] = float(len(citations))

        # S10: verify, then analyse from what held up
        verifier = CitationVerifier(
            evidence=registry,
            judge=LLMEntailmentJudge(
                ctx.gateway,
                ctx.config.model_for(Role.VERIFIER),
                batch_size=1,
                seed=ctx.seed,
            ),
            emitter=ctx.emitter,
            verifier_model=ctx.config.model_for(Role.VERIFIER).name,
        )
        results = await verifier.verify(claims, citations, stage=Stage.S10)
        self._record_outcomes(ctx, cited, results)
        emit(
            AnalysisReadyPayload(
                analysis=self._analysis(item.question, answer, issues, cited, results, registry)
            ),
            stage=Stage.S9,
            agent=self.agent,
        )

    async def _compose(
        self, ctx: RunContext, service: SearchService, hits: list[Hit]
    ) -> tuple[B1Answer, list[Hit], int]:
        """The answer, the hits it may cite (``E-001``... in this order) and how many claims
        were dropped for having no citation. B1: one call on the first search's hits."""
        answer, uncited = await self._answer(ctx, hits)
        return answer, hits, uncited

    # ---- model calls -----------------------------------------------------------------------

    async def _call[T: BaseModel](
        self,
        ctx: RunContext,
        prompt: str,
        response_model: type[T],
        max_tokens: int,
        **fields: str,
    ) -> T:
        _, result = await self._complete(ctx, prompt, response_model, max_tokens, **fields)
        assert result.parsed is not None  # the gateway raises if the output never validates
        return result.parsed

    async def _complete[T: BaseModel](
        self,
        ctx: RunContext,
        prompt: str,
        response_model: type[T],
        max_tokens: int,
        **fields: str,
    ) -> tuple[str, LLMResult[T]]:
        """One structured call on the BASELINE model: the prompt as sent, and the result."""
        template = load_prompt(prompt)
        model = ctx.config.model_for(Role.BASELINE)
        content = Template(template.body).substitute(**fields)
        result = await ctx.gateway.complete(
            [ChatMessage(role="user", content=content)],
            model=model,
            response_model=response_model,
            seed=ctx.seed,
            max_tokens=max_tokens,
            tags={
                "agent": self.agent,
                "prompt": f"{template.id}@v{template.version}",
                "item": ctx.item.id,
            },
        )
        ctx.emitter.emit(
            AgentCompletedPayload(
                role=Role.BASELINE,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                cost_usd=result.usage.cost_usd,
                cached=result.cached,
            ),
            agent=self.agent,
        )
        return content, result

    async def _queries(self, ctx: RunContext, max_queries: int) -> list[str]:
        out = await self._call(
            ctx,
            QUERIES_PROMPT,
            B1Queries,
            1000,
            question=ctx.item.question,
            facts=ctx.item.facts,
            max_queries=str(max_queries),
        )
        queries = list(dict.fromkeys(q.strip() for q in out.queries if q.strip()))[:max_queries]
        return queries or [ctx.item.question]  # an empty answer falls back to the question

    async def _answer(self, ctx: RunContext, hits: Sequence[Hit]) -> tuple[B1Answer, int]:
        return await self._written_answer(
            ctx,
            ANSWER_PROMPT,
            question=ctx.item.question,
            facts=ctx.item.facts,
            evidence=evidence_block(hits) or "(the search found no evidence)",
        )

    async def _written_answer(
        self, ctx: RunContext, prompt: str, **fields: str
    ) -> tuple[B1Answer, int]:
        """An answer call and how many claims were dropped for having no citation. The schema
        requires one, so the gateway's repair retry usually fixes a bare claim; if the output
        is still invalid, the raw text is read once more with the bare claims removed."""
        try:
            answer = await self._call(ctx, prompt, B1Answer, 6000, **fields)
        except StructuredOutputError as exc:
            try:
                raw = json.loads(exc.raw_text)
            except ValueError:
                raise exc from None
            dropped = 0
            for issue in raw.get("issues", []) if isinstance(raw, dict) else []:
                for side in ("position_a", "position_b"):
                    claims = issue.get(side, {}).get("claims") if isinstance(issue, dict) else None
                    if isinstance(claims, list):
                        kept = [
                            c
                            for c in claims
                            if not (isinstance(c, dict) and not c.get("citations"))
                        ]
                        dropped += len(claims) - len(kept)
                        issue[side]["claims"] = kept
            if not dropped:
                raise exc from None
            try:
                return B1Answer.model_validate(raw), dropped
            except ValidationError:
                raise exc from None
        return answer, 0

    @staticmethod
    def _drop_uncited(answer: B1Answer) -> int:
        dropped = 0
        for issue in answer.issues:
            for position in (issue.position_a, issue.position_b):
                kept = [c for c in position.claims if c.citations]
                dropped += len(position.claims) - len(kept)
                position.claims = kept
        return dropped

    @staticmethod
    def _normalise_ids(answer: B1Answer) -> tuple[int, int]:
        """Put evidence IDs in the prompt's form, then drop citations whose ID or quote still
        cannot be a ``Citation``. Returns (dropped, normalised). An ID that is well formed
        but was never presented stays: it fails the verifier's existence check."""
        dropped = normalised = 0
        for _, _, claim in _claims(answer):
            kept = []
            for c in claim.citations:
                cleaned = clean_evidence_id(c.evidence_id)
                normalised += cleaned != c.evidence_id and cleaned.startswith("E-")
                c.evidence_id = cleaned
                try:
                    Citation(
                        claim_id="C-001", evidence_id=c.evidence_id, pinpoint="x", quote=c.quote
                    )
                except ValidationError:
                    dropped += 1
                else:
                    kept.append(c)
            claim.citations = kept
        return dropped, normalised

    async def _repair(self, ctx: RunContext, answer: B1Answer, presented: _Presented) -> None:
        """One call asking for a corrected quote for each citation whose quote is not in its
        chunk. Fixes are applied as given, so a quote that is still wrong stays in (the
        verifier marks it invalid); ``null`` drops the citation."""

        def failing(c: B1Citation) -> bool:
            return (
                check_citation(
                    presented,
                    Citation(
                        claim_id="C-001", evidence_id=c.evidence_id, pinpoint="x", quote=c.quote
                    ),
                )
                is not None
            )

        before = [
            (claim, c) for _, _, claim in _claims(answer) for c in claim.citations if failing(c)
        ]
        ctx.values[f"{self.agent}.citation_failures_before_repair"] = float(len(before))
        todo = [(claim, c) for claim, c in before if c.evidence_id in presented]
        dropped = 0
        if todo:
            blocks = [
                f'<item number="{n}">\n<claim>\n{claim.text}\n</claim>\n'
                f'<evidence id="{c.evidence_id}">\n{presented.texts[c.evidence_id]}\n</evidence>\n'
                f"<bad_quote>\n{c.quote}\n</bad_quote>\n"
                "<error>the quote is not word for word in the evidence</error>\n</item>"
                for n, (claim, c) in enumerate(todo, 1)
            ]
            out = await self._call(ctx, REPAIR_PROMPT, B1Repair, 2000, items="\n\n".join(blocks))
            for fix in out.fixes:
                if not 1 <= fix.number <= len(todo):
                    continue
                claim, c = todo[fix.number - 1]
                if fix.quote is None:
                    claim.citations = [x for x in claim.citations if x is not c]
                    dropped += 1
                elif fix.quote.strip():
                    c.quote = fix.quote.strip()
        ctx.values[f"{self.agent}.citation_failures_after_repair"] = float(
            sum(1 for _, _, claim in _claims(answer) for c in claim.citations if failing(c))
        )
        ctx.values[f"{self.agent}.citations_dropped_by_repair"] = float(dropped)

    # ---- the record ------------------------------------------------------------------------

    def _register(
        self,
        ctx: RunContext,
        service: SearchService,
        hits: Sequence[Hit],
        answer: B1Answer,
        issues: Sequence[Issue],
    ) -> EvidenceRegistry:
        """Register every hit in the order the model saw them. An evidence item belongs to
        the issues whose claims cite it, else to all of them."""
        cited_in: defaultdict[str, list[str]] = defaultdict(list)
        for n, _, claim in _claims(answer):
            for c in claim.citations:
                cited_in[c.evidence_id].append(f"I-{n + 1}")
        registry = EvidenceRegistry(service, ctx.emitter)
        every = [i.id for i in issues]
        for n, hit in enumerate(hits, 1):
            evidence = registry.register(
                hit, cited_in.get(f"E-{n:03d}") or every, Stage.S1, agent=self.agent
            )
            assert evidence.id == f"E-{n:03d}"  # the prompt's IDs are the registry's
        return registry

    def _record_claims(
        self,
        ctx: RunContext,
        registry: EvidenceRegistry,
        answer: B1Answer,
        issues: Sequence[Issue],
        headers: dict[str, str],
    ) -> tuple[dict[str, Claim], list[_Cited]]:
        by = CreatedBy(stage=Stage.S9, agent=self.agent)
        claims: dict[str, Claim] = {}
        cited: list[_Cited] = []
        for n, side, claim in _claims(answer):
            if not claim.text.strip():
                continue
            position = _position(answer.issues[n], side)
            created = Claim(
                id=registry.ids.next("C"),
                text=claim.text.strip(),
                position_id=f"{issues[n].id}.{side}",
                author_agent=self.agent,
                created_by=by,
            )
            claims[created.id] = created
            ctx.emitter.emit(ClaimCreatedPayload(claim=created), stage=Stage.S9, agent=self.agent)
            seen: set[tuple[str, str]] = set()
            citations: list[Citation] = []
            for c in claim.citations:
                if (c.evidence_id, c.quote) in seen:
                    continue
                seen.add((c.evidence_id, c.quote))
                where = (
                    pinpoint(registry.chunk(c.evidence_id), headers.get(c.evidence_id))
                    if c.evidence_id in registry
                    else "unknown"
                )
                citations.append(
                    Citation(
                        claim_id=created.id,
                        evidence_id=c.evidence_id,
                        pinpoint=where,
                        quote=c.quote,
                    )
                )
            cited += [_Cited(n, side, c) for c in citations]
            ctx.emitter.emit(
                ArgumentCreatedPayload(
                    argument=Argument(
                        id=registry.ids.next("A"),
                        claim_ids=[created.id],
                        reasoning=position.summary.strip() or NO_REASONING,
                        citations=citations,
                        created_by=by,
                    )
                ),
                stage=Stage.S9,
                agent=self.agent,
            )
        return claims, cited

    def _record_outcomes(
        self, ctx: RunContext, cited: Sequence[_Cited], results: Sequence[VerificationResult]
    ) -> None:
        counts = {status: 0 for status in VerificationStatus}
        for r in results:
            counts[r.status] += 1
        total = len(results)
        for status in VerificationStatus:
            ctx.values[f"{self.agent}.share_{status.value}"] = (
                counts[status] / total if total else None
            )
        ctx.values[f"{self.agent}.invalid_citations"] = float(counts[VerificationStatus.INVALID])
        ctx.details["citations"] = [
            {
                "claim_id": c.citation.claim_id,
                "evidence_id": c.citation.evidence_id,
                "quote": c.citation.quote,
                "status": r.status.value,
                "check": r.check.value,
                "label": r.label.value if r.label else None,
                "justification": r.justification,
            }
            for c, r in zip(cited, results, strict=True)
        ]

    def _analysis(
        self,
        question: str,
        answer: B1Answer,
        issues: Sequence[Issue],
        cited: Sequence[_Cited],
        results: Sequence[VerificationResult],
        registry: EvidenceRegistry,
    ) -> CaseAnalysis:
        """Key evidence is the verified and weak citations only (§8 invariant 3)."""
        kept: defaultdict[tuple[int, str], list[Citation]] = defaultdict(list)
        failed: defaultdict[int, int] = defaultdict(int)
        total: defaultdict[int, int] = defaultdict(int)
        for c, r in zip(cited, results, strict=True):
            total[c.issue] += 1
            if r.status in SUPPORTED:
                kept[(c.issue, c.side)].append(c.citation)
            else:
                failed[c.issue] += 1

        used: dict[str, tuple[Document, dict[str, None]]] = {}  # document -> its pinpoints used
        analyses: list[IssueAnalysis] = []
        for n, (issue, b1) in enumerate(zip(issues, answer.issues, strict=True)):
            positions: list[PositionAnalysis] = []
            for side in "AB":
                refs: dict[str, EvidenceRef] = {}
                for cit in kept[(n, side)]:
                    refs.setdefault(
                        cit.evidence_id,
                        EvidenceRef(evidence_id=cit.evidence_id, pinpoint=cit.pinpoint),
                    )
                    document = registry.document(cit.evidence_id)
                    used.setdefault(document.id, (document, {}))[1].setdefault(cit.pinpoint)
                judgments = list(
                    dict.fromkeys(
                        d.id
                        for ref in refs.values()
                        if (d := registry.document(ref.evidence_id)).kind is DocumentKind.JUDGMENT
                    )
                )
                p = _position(b1, side)
                positions.append(
                    PositionAnalysis(
                        position_id=f"{issue.id}.{side}",
                        summary=p.summary.strip() or p.statement,
                        supporting_authorities=judgments,
                        key_evidence=list(refs.values()),
                    )
                )
            analyses.append(
                IssueAnalysis(
                    issue_id=issue.id,
                    issue_text=b1.question,
                    positions=positions,
                    leaning=b1.leaning,
                    confidence=b1.confidence,
                    confidence_reason=b1.confidence_reason,
                    unresolved_questions=b1.unresolved_questions,
                )
            )

        sources: list[SourceRef] = []
        for document, where in used.values():
            sources.append(
                SourceRef(
                    document_id=document.id,
                    court=document.court,
                    date=document.decision_date,
                    citation=document.citations[0] if document.citations else None,
                    link=document.source_url,
                    pinpoints_used=list(where),
                )
            )
        limitations = list(answer.limitations)
        for n, issue in enumerate(issues):
            if failed[n]:
                limitations.append(
                    f"{issue.id}: {failed[n]} of {total[n]} citations failed verification "
                    "and were left out of the key evidence."
                )
        words = answer.overall_summary.split()
        if len(words) > SUMMARY_MAX_WORDS:
            limitations.append(f"The model's summary was cut to {SUMMARY_MAX_WORDS} words.")
        return CaseAnalysis(
            question=question,
            facts=answer.facts,
            assumptions=answer.assumptions,
            issues=analyses,
            overall_summary=" ".join(words[:SUMMARY_MAX_WORDS]) or "No summary was given.",
            limitations=limitations,
            sources=sources,
            created_by=CreatedBy(stage=Stage.S9, agent=self.agent),
        )


def _position(issue: B1Issue, side: str) -> B1Position:
    return issue.position_a if side == "A" else issue.position_b


def _claims(answer: B1Answer) -> list[tuple[int, str, B1Claim]]:
    """Every claim with its issue index (0-based) and side, in answer order."""
    return [
        (n, side, claim)
        for n, issue in enumerate(answer.issues)
        for side in "AB"
        for claim in _position(issue, side).claims
    ]
