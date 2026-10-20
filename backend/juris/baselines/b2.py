"""Baseline B2, the compute-matched single agent (PLAN 6.4, IDEA_final §5 and R1): B1 given
roughly Juris's token budget, so a gain of Juris over B2 is not just "more tokens".

B2 is B1 with a longer S9. The draft is B1's answer (same prompt, same evidence). Then up to
``max_rounds`` rounds of:

- **critique**: one call lists the draft's problems. The quote errors are found in code (the
  verifier's exact-quote rule) and handed to it, as Juris's agents get them from the Case Record.
  The first critique also writes follow-up queries;
- **second search** (first round only): those queries retrieve up to ``follow_up_k`` new chunks,
  numbered after the first search's (``E-021``...), so every ID stays the registry's;
- **revise**: one call rewrites the whole answer from the draft, the critique and the evidence.

Critique and revise see at most ``revise_evidence_k`` chunks: the ones the draft cites, then
the new ones, then the rest by rank. With a draft in the prompt, all of them would not fit a
local model's context. A round starts only while the generation spend is under
``target_tokens``. Spend is estimated from characters (4 per token), not from the gateway's
usage: a cache hit reports zero tokens, and a budget read from usage would make a cached
re-run take more rounds than the original. After the rounds, B1's repair, record and verifier
steps run unchanged. The budget is recalibrated against ``juris_full`` in PLAN 7.12.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from juris.baselines.b1 import (
    B1Answer,
    B1System,
    SearchService,
    _claims,
    _Out,
    _Presented,
    clean_evidence_id,
    evidence_block,
)
from juris.baselines.base import RunContext
from juris.config import BaselineParams
from juris.events.catalog import QueryIssuedPayload
from juris.llm.errors import StructuredOutputError
from juris.llm.types import LLMResult
from juris.models import Citation
from juris.models.common import Stage
from juris.retrieval.service import Hit
from juris.verify import check_citation

AGENT = "b2"
CRITIQUE_PROMPT = "baseline_b2_critique"
REVISE_PROMPT = "baseline_b2_revise"
CHARS_PER_TOKEN = 4
NO_QUOTE_ERRORS = "(none: every quote is word for word in its evidence)"


class B2Critique(_Out):
    problems: list[str] = Field(description="One sentence per problem; empty if there are none")
    queries: list[str] = Field(description="Searches for missing evidence; may be empty")


class B2System(B1System):
    agent = AGENT

    async def _complete[T: BaseModel](
        self,
        ctx: RunContext,
        prompt: str,
        response_model: type[T],
        max_tokens: int,
        **fields: str,
    ) -> tuple[str, LLMResult[T]]:
        content, result = await super()._complete(ctx, prompt, response_model, max_tokens, **fields)
        key = f"{self.agent}.tokens_estimate"
        spent = ctx.values.get(key) or 0.0
        ctx.values[key] = spent + (len(content) + len(result.text)) / CHARS_PER_TOKEN
        return content, result

    async def _compose(
        self, ctx: RunContext, service: SearchService, hits: list[Hit]
    ) -> tuple[B1Answer, list[Hit], int]:
        params = ctx.config.profile.baseline
        answer, uncited = await self._answer(ctx, hits)  # the draft: B1's answer
        hits = list(hits)
        first = len(hits)
        rounds = revisions = failures = 0
        critiques: list[list[str]] = []
        follow_up: list[str] = []
        while rounds < params.max_rounds and not self._spent(ctx, params):
            rounds += 1
            shown = _shown(answer, len(hits), first, params.revise_evidence_k)
            critique = await self._call(
                ctx,
                CRITIQUE_PROMPT,
                B2Critique,
                2000,
                question=ctx.item.question,
                facts=ctx.item.facts,
                evidence=_block(hits, shown),
                draft=answer.model_dump_json(),
                quote_errors=_quote_errors(answer, hits) or NO_QUOTE_ERRORS,
                max_queries=str(params.follow_up_queries),
            )
            problems = [p.strip() for p in critique.problems if p.strip()]
            critiques.append(problems)
            added = 0
            if rounds == 1 and params.follow_up_queries and params.follow_up_k:
                follow_up = list(dict.fromkeys(q.strip() for q in critique.queries if q.strip()))
                follow_up = follow_up[: params.follow_up_queries]
                added = self._search_again(ctx, service, hits, follow_up, params)
            if not problems and not added:
                break  # nothing to revise
            shown = _shown(answer, len(hits), first, params.revise_evidence_k)
            review = "\n".join(f"- {p}" for p in problems) or "- No problems were found."
            try:
                answer, uncited = await self._written_answer(
                    ctx,
                    REVISE_PROMPT,
                    question=ctx.item.question,
                    facts=ctx.item.facts,
                    evidence=_block(hits, shown),
                    draft=answer.model_dump_json(),
                    review=review,
                )
                revisions += 1
            except StructuredOutputError:
                failures += 1  # keep the last valid answer
                break

        ctx.values[f"{self.agent}.critiques"] = float(rounds)
        ctx.values[f"{self.agent}.revisions"] = float(revisions)
        ctx.values[f"{self.agent}.revise_failures"] = float(failures)
        ctx.values[f"{self.agent}.follow_up_queries"] = float(len(follow_up))
        ctx.values[f"{self.agent}.follow_up_evidence"] = float(len(hits) - first)
        ctx.details["critiques"] = critiques
        ctx.details["follow_up_queries"] = follow_up
        return answer, hits, uncited

    def _spent(self, ctx: RunContext, params: BaselineParams) -> bool:
        spent = ctx.values.get(f"{self.agent}.tokens_estimate") or 0.0
        return params.target_tokens is not None and spent >= params.target_tokens

    def _search_again(
        self,
        ctx: RunContext,
        service: SearchService,
        hits: list[Hit],
        queries: Sequence[str],
        params: BaselineParams,
    ) -> int:
        """The second search: up to ``follow_up_k`` chunks not already found, appended to
        ``hits``. Returns how many were added."""
        if not queries:
            return 0
        for query in queries:
            ctx.emitter.emit(QueryIssuedPayload(query=query), stage=Stage.S1, agent=self.agent)
        found = service.search(queries, k=params.follow_up_k + len(hits), rerank=params.rerank)
        seen = {h.chunk_id for h in hits}
        new: list[Hit] = []
        for h in found:
            if h.chunk_id not in seen and len(new) < params.follow_up_k:
                seen.add(h.chunk_id)
                new.append(h)
        hits.extend(new)
        return len(new)


def _cited(answer: B1Answer, total: int) -> list[int]:
    """The 0-based hit indices the answer cites, in ID order (IDs it could not mean ignored)."""
    out: set[int] = set()
    for _, _, claim in _claims(answer):
        for c in claim.citations:
            cleaned = clean_evidence_id(c.evidence_id)
            if cleaned.startswith("E-") and cleaned[2:].isdigit():
                n = int(cleaned[2:])
                if 1 <= n <= total:
                    out.add(n - 1)
    return sorted(out)


def _shown(answer: B1Answer, total: int, first: int, cap: int) -> list[int]:
    """What critique and revise see: the cited hits, then the second search's, then the rest
    in rank order, at most ``cap``, listed in ID order."""
    cited = _cited(answer, total)
    order = cited + [i for i in range(first, total) if i not in cited]
    order += [i for i in range(first) if i not in cited]
    return sorted(order[:cap])


def _block(hits: Sequence[Hit], shown: Sequence[int]) -> str:
    return (
        evidence_block([hits[i] for i in shown], [f"E-{i + 1:03d}" for i in shown])
        or "(the search found no evidence)"
    )


def _quote_errors(answer: B1Answer, hits: Sequence[Hit]) -> str:
    """One line per citation that fails the existence or exact-quote check."""
    presented = _Presented({f"E-{n:03d}": h.text for n, h in enumerate(hits, 1)})
    lines: list[str] = []
    for _, _, claim in _claims(answer):
        for c in claim.citations:
            evidence_id = clean_evidence_id(c.evidence_id)
            if evidence_id not in presented:
                lines.append(f'"{evidence_id}" is not an evidence ID (claim: "{claim.text}").')
                continue
            citation = Citation(
                claim_id="C-001",
                evidence_id=evidence_id,
                pinpoint="x",
                quote=c.quote.strip() or "x",
            )
            if check_citation(presented, citation) is not None:
                lines.append(f'{evidence_id}: "{c.quote}" is not word for word in {evidence_id}.')
    return "\n".join(lines)
