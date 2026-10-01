"""The entailment check (IDEA_final §6.1 check 3): does the cited passage support the claim?

``LLMEntailmentJudge`` asks a model to label each (claim, quote, passage) pair
``supports`` / ``partially_supports`` / ``does_not_support`` / ``contradicts`` with a
one-sentence justification. It is the verifier's third and costliest check, so the verifier
only sends it citations that passed existence and the exact quote.

One pair per call by default. Batching several pairs per call is allowed (``batch_size``) but
off, because a small local model mixed up pairs within a batch (PLAN 6.1 says batch "where
it's safe"); re-measure before raising it for another model.

The prompt is a versioned template (``verifier_entailment.md``, ``string.Template`` fields).
The model is a parameter; choosing one is the experiment's decision, not this module's.
It also satisfies ``EntailmentJudge``, so the answer metrics (5.4) can judge pairs the run
never verified with the same code.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from string import Template
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from juris.config import ModelSpec
from juris.llm.errors import StructuredOutputError
from juris.llm.gateway import LLMGateway
from juris.llm.types import ChatMessage
from juris.models.common import EntailmentLabel
from juris.prompts import load_prompt

ENTAILMENT_PROMPT = "verifier_entailment"
AGENT = "citation_verifier"


@dataclass(frozen=True)
class ClaimEvidencePair:
    """What the entailment check reads: does ``passage`` (quoted) support ``claim``?"""

    claim_id: str
    claim: str
    evidence_id: str
    quote: str
    passage: str


class EntailmentJudge(Protocol):
    """The verifier's entailment check (PLAN 6.1), for pairs the run did not verify."""

    async def __call__(self, pairs: Sequence[ClaimEvidencePair]) -> list[EntailmentLabel]: ...


@dataclass(frozen=True)
class Judgement:
    label: EntailmentLabel
    justification: str


class PairJudge(Protocol):
    """What ``CitationVerifier`` needs: a label and a justification per pair, in order."""

    async def judge(self, pairs: Sequence[ClaimEvidencePair]) -> list[Judgement]: ...


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairJudgement(_Out):
    index: int = Field(description="The pair's number, from 1")
    label: EntailmentLabel
    justification: str = Field(description="One sentence")


class BatchJudgement(_Out):
    judgements: list[PairJudgement] = Field(min_length=1)


class SingleJudgement(_Out):
    """The answer for a lone pair: no index, no list (a small model returned an empty list)."""

    label: EntailmentLabel
    justification: str = Field(description="One sentence")


class VerifierOutputError(ValueError):
    """The model's judgement of a single pair is missing or malformed, even asked alone."""


def _dedup_key(pair: ClaimEvidencePair) -> tuple[str, str, str, str]:
    return (pair.claim, pair.evidence_id, pair.quote, pair.passage)


@dataclass
class LLMEntailmentJudge:
    gateway: LLMGateway
    model: ModelSpec
    batch_size: int = 1
    seed: int = 0
    max_passage_chars: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

    def _passage(self, pair: ClaimEvidencePair) -> str:
        """The passage, cut to ``max_passage_chars`` around the quote when it is longer."""
        limit = self.max_passage_chars
        text = pair.passage
        if limit is None or len(text) <= limit:
            return text
        at = text.find(pair.quote)
        start = 0 if at < 0 else max(0, min(at - (limit - len(pair.quote)) // 2, len(text) - limit))
        cut = text[start : start + limit]
        return ("[...] " if start else "") + cut + (" [...]" if start + limit < len(text) else "")

    def render(self, pairs: Sequence[ClaimEvidencePair]) -> str:
        """The numbered pairs as the prompt shows them."""
        blocks = [
            f'<pair number="{n}">\n<claim>\n{p.claim}\n</claim>\n'
            f"<quote>\n{p.quote}\n</quote>\n<passage>\n{self._passage(p)}\n</passage>\n</pair>"
            for n, p in enumerate(pairs, 1)
        ]
        return "\n\n".join(blocks)

    async def _ask(self, pairs: Sequence[ClaimEvidencePair]) -> list[Judgement] | None:
        """One call; None if the output doesn't match the pairs (missing, wrong or repeated
        numbers, or no structured output)."""
        template = load_prompt(ENTAILMENT_PROMPT)
        single = len(pairs) == 1
        response_model: type[SingleJudgement] | type[BatchJudgement] = (
            SingleJudgement if single else BatchJudgement
        )
        try:
            result = await self.gateway.complete(
                [
                    ChatMessage(
                        role="user",
                        content=Template(template.body).substitute(pairs=self.render(pairs)),
                    )
                ],
                model=self.model,
                response_model=response_model,
                temperature=0.0,
                seed=self.seed,
                tags={
                    "agent": AGENT,
                    "prompt": f"{template.id}@v{template.version}",
                    "pairs": str(len(pairs)),
                },
            )
        except StructuredOutputError:
            return None
        if result.parsed is None:
            return None
        parsed = result.parsed
        if isinstance(parsed, SingleJudgement):
            return [Judgement(parsed.label, parsed.justification)] if single else None
        if single:
            return None
        by_index = {j.index: j for j in parsed.judgements}
        if len(by_index) != len(parsed.judgements) or sorted(by_index) != list(
            range(1, len(pairs) + 1)
        ):
            return None
        return [
            Judgement(by_index[n].label, by_index[n].justification)
            for n in range(1, len(pairs) + 1)
        ]

    async def _batch(self, pairs: Sequence[ClaimEvidencePair]) -> list[Judgement]:
        judged = await self._ask(pairs)
        if judged is not None:
            return judged
        out: list[Judgement] = []
        for pair in pairs:  # a bad batch: ask each pair alone
            single = await self._ask([pair])
            if single is None:
                raise VerifierOutputError(
                    f"no usable judgement for {pair.claim_id} / {pair.evidence_id}"
                )
            out.append(single[0])
        return out

    async def judge(self, pairs: Sequence[ClaimEvidencePair]) -> list[Judgement]:
        """A judgement per pair, in order. Identical pairs are judged once."""
        unique: dict[tuple[str, str, str, str], ClaimEvidencePair] = {}
        for pair in pairs:
            unique.setdefault(_dedup_key(pair), pair)
        todo = list(unique.values())
        size = self.batch_size
        batches = await asyncio.gather(
            *(self._batch(todo[i : i + size]) for i in range(0, len(todo), size))
        )
        judged = dict(zip(unique, (j for batch in batches for j in batch), strict=True))
        return [judged[_dedup_key(p)] for p in pairs]

    async def __call__(self, pairs: Sequence[ClaimEvidencePair]) -> list[EntailmentLabel]:
        return [j.label for j in await self.judge(pairs)]
