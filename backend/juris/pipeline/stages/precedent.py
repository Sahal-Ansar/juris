"""S3, precedent analysis (PLAN 7.6): a ``PrecedentCard`` for every judgment cited in S2.

The cited judgments are taken in order of first citation. For each, code gathers what the
Precedent Analyst reads:

- **Passages**, within ``max_passage_chars``: the cited chunks first, then the judgment's
  headnote (up to ``headnote_chunks``), then each cited chunk's neighbours. They are shown in
  reading order, each labelled.
- **Treatment cues** from the citation graph (``treatment_signals``).

The analysts run at once and their cards are written in that order. A judgment whose analyst
gave nothing usable gets a ``fallback_card``, so every cited judgment has a card (7.6's bar).
Recorded (``s3.*``): judgments, cards, fallback cards, fits given and missing, treatment
signals. The fits, with their reasons, are in the details for spot checks.

Passthrough: fallback cards from code facts only, no model.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

from juris.agents.precedent_analyst import (
    AGENT,
    PrecedentAnalyst,
    PrecedentOutput,
    fallback_card,
    treatment_signals,
)
from juris.events import CaseView
from juris.models import Chunk, Document
from juris.models.common import DocumentKind, Stage
from juris.pipeline.budget import Spend
from juris.pipeline.citing import pinpoint
from juris.pipeline.orchestrator import StageContext
from juris.retrieval.lookup import LinkedCase


class PrecedentService(Protocol):
    """What S3 needs from retrieval (``RetrievalService``; tests pass a fake)."""

    def records(self, chunk_id: str) -> tuple[Document, Chunk]: ...

    def chunk_ids(self, doc_id: str, section: str | None = None) -> list[str]: ...

    def citing_cases(self, doc_id: str) -> list[LinkedCase]: ...


def cited_judgments(view: CaseView) -> dict[str, dict[str, list[str]]]:
    """Judgment -> {cited chunk -> its evidence IDs}, in order of first citation."""
    out: dict[str, dict[str, list[str]]] = {}
    for argument in view.arguments.values():
        for citation in argument.citations:
            evidence = view.evidence.get(citation.evidence_id)
            if evidence is None:
                continue
            if view.documents[evidence.document_id].kind is not DocumentKind.JUDGMENT:
                continue
            chunks = out.setdefault(evidence.document_id, {})
            ids = chunks.setdefault(evidence.chunk_id, [])
            if evidence.id not in ids:
                ids.append(evidence.id)
    return out


class PrecedentStage:
    stage = Stage.S3
    optional = False

    def __init__(
        self,
        service: Callable[[], PrecedentService],
        *,
        max_passage_chars: int = 12000,
        headnote_chunks: int = 3,
        neighbours: int = 1,
    ) -> None:
        self._make_service = service
        self._service: PrecedentService | None = None
        self.max_passage_chars = max_passage_chars
        self.headnote_chunks = headnote_chunks
        self.neighbours = neighbours

    @property
    def service(self) -> PrecedentService:
        if self._service is None:
            self._service = self._make_service()
        return self._service

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped

    def passages(self, view: CaseView, document_id: str, cited: dict[str, list[str]]) -> str:
        order = self.service.chunk_ids(document_id)
        place = {chunk_id: n for n, chunk_id in enumerate(order)}
        headnote = self.service.chunk_ids(document_id, "headnote")[: self.headnote_chunks]
        around = [
            order[n + d]
            for chunk_id in cited
            if (n := place.get(chunk_id)) is not None
            for d in range(-self.neighbours, self.neighbours + 1)
            if d and 0 <= n + d < len(order)
        ]
        chosen: list[tuple[str, Chunk]] = []
        used = 0
        for chunk_id in dict.fromkeys([*cited, *headnote, *around]):
            chunk = view.chunks.get(chunk_id) or self.service.records(chunk_id)[1]
            if chosen and used + len(chunk.text) > self.max_passage_chars:
                continue
            chosen.append((chunk_id, chunk))
            used += len(chunk.text)
        chosen.sort(key=lambda c: place.get(c[0], len(order)))
        blocks = []
        for chunk_id, chunk in chosen:
            if chunk_id in cited:
                label = f"cited as {', '.join(cited[chunk_id])}"
            elif chunk_id in headnote:
                label = "headnote"
            else:
                label = "context"
            blocks.append(f"[{label}; {pinpoint(chunk, None)}]\n{chunk.text}")
        return "\n\n".join(blocks)

    async def run(self, ctx: StageContext) -> None:
        view = ctx.record.view
        judgments = cited_judgments(view)
        analysts = [
            PrecedentAnalyst(
                document_id,
                passages=self.passages(view, document_id, cited),
                signals=treatment_signals(self.service.citing_cases(document_id)),
            )
            for document_id, cited in judgments.items()
        ]
        outputs = await ctx.gather(*(a.produce(ctx) for a in analysts))
        fallbacks = 0
        for analyst, output in zip(analysts, outputs, strict=True):
            if output is not None:
                analyst.apply(output, ctx.record, stage=self.stage)
            else:
                fallbacks += 1
                ctx.record.add_precedent_card(
                    fallback_card(
                        analyst.document_id,
                        analyst.signals,
                        "the Precedent Analyst gave no valid output",
                        stage=self.stage,
                    ),
                    stage=self.stage,
                    agent=AGENT,
                )
        self._record(ctx, analysts, outputs, fallbacks)

    async def passthrough(self, ctx: StageContext) -> None:
        for document_id in cited_judgments(ctx.record.view):
            signals = treatment_signals(self.service.citing_cases(document_id))
            card = fallback_card(document_id, signals, "S3 ran as a passthrough", stage=self.stage)
            ctx.record.add_precedent_card(card, stage=self.stage, agent=AGENT)

    @staticmethod
    def _record(
        ctx: StageContext,
        analysts: Sequence[PrecedentAnalyst],
        outputs: Sequence[PrecedentOutput | None],
        fallbacks: int,
    ) -> None:
        view = ctx.record.view
        citing = sum(len(a.citing_arguments(view)) for a in analysts)
        fits = sum(len(c.fit_by_argument) for c in view.precedent_cards.values())
        values = ctx.run.values
        values["s3.judgments"] = float(len(analysts))
        values["s3.cards"] = float(len(view.precedent_cards))
        values["s3.fallback_cards"] = float(fallbacks)
        values["s3.fits"] = float(fits)
        values["s3.fits_missing"] = float(citing - fits)
        values["s3.treatment_signals"] = float(sum(len(a.signals) for a in analysts))
        ctx.run.details["s3.fits"] = [
            {
                "document": a.document_id,
                "argument": f.argument_id,
                "fit": f.fit.value,
                "reason": f.reason,
            }
            for a, output in zip(analysts, outputs, strict=True)
            if output is not None
            for f in output.fits
        ]
