"""The retrieval service: the one interface agents and baselines use (PLAN 4.6, D-033).

Agents never touch the database. They call:

- ``search(queries, filters, k, rerank=True)``: hybrid search (lexical + dense, RRF), reranked
  by default, returning ``Hit`` objects that carry the chunk's text and where it comes from;
- ``section(reference)``, ``case_by_citation``, ``case_by_title``, ``citing_cases``,
  ``cited_cases``, ``cases_citing_section``: the structured lookups (``retrieval.lookup``);
- ``section_hits(reference)``: a named provision's chunks as ``Hit``s, so they can be
  registered as evidence like search results;
- ``records(chunk_id)``: the ``Document`` and ``Chunk`` models, for evidence events.

A baseline needs only this::

    service = RetrievalService.local()
    hits = service.search("Can earnest money be forfeited without proof of loss?", k=10)
    context = "\\n\\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from juris.config import get_settings
from juris.models import Chunk, Document, StatuteMeta
from juris.retrieval.dense import DenseRetriever
from juris.retrieval.filters import SearchFilters
from juris.retrieval.hybrid import HybridHit, HybridRetriever
from juris.retrieval.lexical import LexicalRetriever
from juris.retrieval.lookup import CaseMatch, LinkedCase, Lookup, Section, TitleMatch
from juris.retrieval.rerank import Reranker


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk with its text and provenance."""

    chunk_id: str
    doc_id: str
    rank: int  # 1-based
    text: str
    context_header: str | None  # "Case title - Court - Year - para 12" / "Act - s. 74 - heading"
    title: str  # the document's title
    query: str  # the query that ranked this chunk highest (Evidence.retrieved_by_query)
    retrieval_score: float | None = None  # RRF score; None for a lookup
    rerank_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)  # rank per retriever and query


class RetrievalService:
    def __init__(
        self,
        engine: Engine,
        hybrid: HybridRetriever,
        lookup: Lookup | None = None,
    ) -> None:
        self.engine = engine
        self.hybrid = hybrid
        self.lookups = lookup or Lookup(engine)

    @classmethod
    def local(cls, engine: Engine | None = None, rerank: bool = True) -> "RetrievalService":
        """The service over the configured database with the local models (bge-m3, and
        bge-reranker-v2-m3 if ``rerank``). Needs ``uv sync --group embed`` and the weights."""
        from juris.ingest.tokens import REVISION, tokenizer_dir
        from juris.retrieval.embed import BgeM3Encoder
        from juris.retrieval.rerank import BgeReranker, reranker_dir

        engine = engine or create_engine(get_settings().database_url())
        dense = DenseRetriever(engine, BgeM3Encoder(path=tokenizer_dir(), revision=REVISION))
        reranker = Reranker(engine, BgeReranker(reranker_dir())) if rerank else None
        return cls(engine, HybridRetriever(LexicalRetriever(engine), dense, reranker=reranker))

    # ---- search --------------------------------------------------------------------------

    def search(
        self,
        queries: str | Sequence[str],
        filters: SearchFilters | None = None,
        k: int = 10,
        rerank: bool = True,
    ) -> list[Hit]:
        """Hybrid search over one or more queries; reranked when a reranker is configured."""
        qs = [queries] if isinstance(queries, str) else list(queries)
        qs = list(dict.fromkeys(q.strip() for q in qs if q.strip()))  # as HybridRetriever does
        use_rerank = rerank and self.hybrid.reranker is not None
        fused = self.hybrid.search(qs, filters, k=k, rerank=use_rerank)
        return self._hits(fused, qs)

    def _hits(self, fused: list[HybridHit], queries: list[str]) -> list[Hit]:
        if not fused:
            return []
        texts = self._texts([h.chunk_id for h in fused])
        out = []
        for h in fused:
            best = min(h.ranks, key=lambda name: (h.ranks[name], name))
            body, header, title = texts[h.chunk_id]
            out.append(
                Hit(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    rank=h.rank,
                    text=body,
                    context_header=header,
                    title=title,
                    query=queries[int(best.split(":")[1])],
                    retrieval_score=h.score,
                    rerank_score=h.rerank_score,
                    ranks=dict(h.ranks),
                )
            )
        return out

    def _texts(self, chunk_ids: list[str]) -> dict[str, tuple[str, str | None, str]]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT c.chunk_id, c.text, c.context_header, d.title FROM chunks c "
                    "JOIN documents d ON d.doc_id = c.doc_id WHERE c.chunk_id = ANY(:ids)"
                ),
                {"ids": chunk_ids},
            )
            return {r[0]: (r[1], r[2], r[3]) for r in rows}

    # ---- lookups -------------------------------------------------------------------------

    def section(self, reference: str) -> Section | None:
        """A provision named in free text: "s. 74 ICA", "Section 10 of the SRA" ..."""
        return self.lookups.lookup_provision(reference)

    def section_hits(self, reference: str) -> list[Hit]:
        """The chunks of a named provision, as hits (no retrieval score) to register."""
        found = self.section(reference)
        if found is None:
            return []
        texts = self._texts(found.chunk_ids)
        return [
            Hit(
                chunk_id=cid,
                doc_id=f"ACT-{found.act_id}",
                rank=i,
                text=texts[cid][0],
                context_header=texts[cid][1],
                title=texts[cid][2],
                query=reference,
            )
            for i, cid in enumerate(found.chunk_ids, 1)
        ]

    def case_by_citation(self, citation: str) -> CaseMatch | None:
        return self.lookups.get_case_by_citation(citation)

    def case_by_title(self, title: str, limit: int = 5) -> list[TitleMatch]:
        return self.lookups.get_case_by_title(title, limit=limit)

    def citing_cases(self, doc_id: str) -> list[LinkedCase]:
        return self.lookups.get_citing_cases(doc_id)

    def cited_cases(self, doc_id: str) -> list[LinkedCase]:
        return self.lookups.get_cited_cases(doc_id)

    def cases_citing_section(self, act: str, section: str) -> list[LinkedCase]:
        return self.lookups.get_cases_citing_section(act, section)

    # ---- records for evidence events -----------------------------------------------------

    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        """The domain models of a chunk and its document (``evidence_registered`` payload)."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT c.chunk_id, c.doc_id, c.text, c.para_start, c.para_end, "
                    "c.page_start, c.page_end, c.char_start, c.char_end, c.section AS heading, "
                    "d.kind, d.title, d.court, d.court_level, d.bench_strength, d.judges, "
                    "d.decision_date, d.cnr, d.citations, d.source_url, d.licence, "
                    "d.snapshot_id, s.act, s.section AS s_section, s.title AS s_title, "
                    "s.in_force_from, s.in_force_to, "
                    "s.amended_by, s.amendments_curated "
                    "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
                    "LEFT JOIN statute_sections s ON s.section_id = c.statute_section_id "
                    "WHERE c.chunk_id = :c"
                ),
                {"c": chunk_id},
            ).one_or_none()
        if row is None:
            raise KeyError(f"unknown chunk: {chunk_id}")
        document = Document(
            id=row.doc_id,
            kind=row.kind,
            title=row.title,
            court=row.court,
            court_level=row.court_level,
            bench_strength=row.bench_strength,
            judges=row.judges or [],
            decision_date=row.decision_date,
            cnr=row.cnr,
            citations=row.citations or [],
            source_url=row.source_url,
            licence=row.licence,
            corpus_snapshot=row.snapshot_id,
        )
        statute = None
        if row.act is not None:
            statute = StatuteMeta(
                act=row.act,
                section=row.s_section,
                title=row.s_title,
                in_force_from=row.in_force_from,
                in_force_to=row.in_force_to,
                amended_by=row.amended_by or [],
                amendments_curated=bool(row.amendments_curated),
            )
        chunk = Chunk(
            id=row.chunk_id,
            document_id=row.doc_id,
            text=row.text,
            para_start=row.para_start,
            para_end=row.para_end,
            page_start=row.page_start,
            page_end=row.page_end,
            char_start=row.char_start,
            char_end=row.char_end,
            section=row.heading,
            statute=statute,
        )
        return document, chunk
