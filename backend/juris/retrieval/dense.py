"""Dense retrieval over chunk embeddings with pgvector HNSW (PLAN 4.2, D-029).

The query is encoded by the model that embedded the chunks (``juris.retrieval.embed``).
bge-m3's dense retrieval takes the query as is; a model trained with a query instruction
(multilingual-e5's ``"query: "``) gets it from ``query_prefix`` in ``configs/embeddings.yaml``.

Two search paths, chosen per query:

- Without filters, or when the filters still allow many chunks: the model's partial HNSW
  index (``juris.db.embeddings``), ordering by the indexed ``(embedding::vector(d)) <=> q``.
  Filters apply while the index is walked. pgvector's iterative scan
  (``hnsw.iterative_scan = strict_order``) keeps going until ``k`` rows pass, so a filter
  doesn't cut the result short. ``ef_search`` is at least ``k``.
- When the filters allow at most ``exact_below`` chunks (a few documents, one Act, a narrow
  date range): an exact scan of those chunks' vectors. It is faster there than walking the
  graph past most of its nodes, and its recall is exact.

All filters are per document (``SearchFilters``), so the allowed documents are resolved first
(2,298 rows) and their chunk count read from a per-process cache.

Scores are cosine similarities (1 - cosine distance), in [-1, 1].
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from juris.db.embeddings import EMBEDDINGS_CONFIG
from juris.retrieval.embed import Encoder, _literal
from juris.retrieval.filters import SearchFilters, filter_sql


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    doc_id: str
    score: float
    rank: int  # 1-based


def query_prefix(model: str, path: Path | None = None) -> str:
    """The instruction a model expects before a query ("" if none, or the model is unknown)."""
    data = yaml.safe_load((path or EMBEDDINGS_CONFIG).read_text(encoding="utf-8"))
    return str(data["models"].get(model, {}).get("query_prefix") or "")


class DenseRetriever:
    """Ranked chunk IDs for a query by embedding similarity, with filters."""

    def __init__(
        self,
        engine: Engine,
        encoder: Encoder,
        ef_search: int = 400,
        max_scan_tuples: int = 20_000,
        exact_below: int = 5_000,
        prefix: str | None = None,
    ) -> None:
        self.engine = engine
        self.encoder = encoder
        self.ef_search = ef_search
        self.max_scan_tuples = max_scan_tuples  # iterative scan's limit per query
        self.exact_below = exact_below  # allowed chunks at or below which to scan exactly
        self.prefix = query_prefix(encoder.model) if prefix is None else prefix
        self._chunks_per_doc: dict[str, int] | None = None
        self.last_path: str | None = None  # "hnsw" or "exact", for tests and benchmarks

    def encode(self, queries: Sequence[str]) -> list[list[float]]:
        return self.encoder.encode([self.prefix + q for q in queries])

    def search(
        self, query: str, filters: SearchFilters | None = None, k: int = 50
    ) -> list[DenseHit]:
        return self.search_vector(self.encode([query])[0], filters, k)

    def chunks_per_doc(self, conn: Connection) -> dict[str, int]:
        if self._chunks_per_doc is None:
            rows = conn.execute(text("SELECT doc_id, count(*) FROM chunks GROUP BY doc_id"))
            self._chunks_per_doc = {r[0]: r[1] for r in rows}
        return self._chunks_per_doc

    def search_vector(
        self, vector: Sequence[float], filters: SearchFilters | None = None, k: int = 50
    ) -> list[DenseHit]:
        if len(vector) != self.encoder.dim:
            raise ValueError(f"query vector has {len(vector)} dimensions, not {self.encoder.dim}")
        dim = self.encoder.dim
        # the indexed expression (juris.db.embeddings.vector_expr) on the aliased table
        distance = f"(e.embedding::vector({dim})) <=> CAST(:q AS vector({dim}))"
        params: dict[str, Any] = {
            "q": _literal(vector),
            "model": self.encoder.model,
            "k": k,
        }
        with self.engine.connect() as conn:  # one transaction, so the settings stay local
            docs: list[str] | None = None
            if filters is not None:
                where, fparams = filter_sql(filters)
                docs = list(
                    conn.execute(text(f"SELECT d.doc_id FROM documents d WHERE {where}"), fparams)
                    .scalars()
                    .all()
                )
                if not docs:
                    return []
            counts = self.chunks_per_doc(conn)
            if docs is not None and sum(counts.get(d, 0) for d in docs) <= self.exact_below:
                self.last_path = "exact"
                # ordering by the score, not the distance, keeps the planner off the index
                sql = f"""
                    SELECT c.chunk_id, c.doc_id, 1 - ({distance}) AS score
                      FROM chunks c
                      JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id AND e.model = :model
                     WHERE c.doc_id = ANY(:docs)
                     ORDER BY score DESC, c.chunk_id
                     LIMIT :k
                """
                params["docs"] = docs
            else:
                self.last_path = "hnsw"
                conn.execute(
                    text(
                        "SELECT set_config('hnsw.ef_search', :ef, true), "
                        "set_config('hnsw.iterative_scan', 'strict_order', true), "
                        "set_config('hnsw.max_scan_tuples', :mst, true)"
                    ),
                    {"ef": str(max(self.ef_search, k)), "mst": str(self.max_scan_tuples)},
                )
                allowed = ""
                if docs is not None:
                    allowed = "AND c.doc_id = ANY(:docs)"
                    params["docs"] = docs
                sql = f"""
                    SELECT c.chunk_id, c.doc_id, 1 - ({distance}) AS score
                      FROM chunk_embeddings e
                      JOIN chunks c ON c.chunk_id = e.chunk_id
                     WHERE e.model = :model {allowed}
                     ORDER BY {distance}
                     LIMIT :k
                """
            rows = conn.execute(text(sql), params).all()
        return [DenseHit(r[0], r[1], float(r[2]), i) for i, r in enumerate(rows, 1)]
