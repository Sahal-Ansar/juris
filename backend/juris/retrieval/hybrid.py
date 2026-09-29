"""Hybrid retrieval: Reciprocal Rank Fusion of lexical and dense results (PLAN 4.3, D-030).

For each query, the lexical and dense retrievers each return their top ``depth`` chunks. Every
ranked list is fused with RRF (Cormack et al., 2009):

    score(chunk) = sum over lists L containing it of  weight(L) / (rrf_k + rank_L(chunk))

with 1-based ranks and ``rrf_k = 60``. Several queries (multi-query: the issue, its
rephrasings, the provisions it names) simply add more lists, so a chunk found by several
queries or both retrievers rises, and each chunk appears once in the result.

Only ranks are used, so the retrievers' scores (IDF coverage, cosine similarity) never need
to be put on one scale. Ties are broken by the chunk's best rank in any list, then its ID.

With a ``Reranker`` (PLAN 4.4), the fused top ``top_n`` are re-scored by a cross-encoder
against the first (primary) query and reordered; the rest follow in fused order. Reranking is
on by default when a reranker is given and can be switched off per search.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from juris.retrieval.filters import SearchFilters
from juris.retrieval.lexical import Mode
from juris.retrieval.rerank import Reranker

RRF_K = 60


class RankedChunk(Protocol):
    @property
    def chunk_id(self) -> str: ...
    @property
    def doc_id(self) -> str: ...
    @property
    def rank(self) -> int: ...


class LexicalSearch(Protocol):
    def search(
        self, query: str, filters: SearchFilters | None = ..., k: int = ..., mode: Mode = ...
    ) -> Sequence[RankedChunk]: ...


class DenseSearch(Protocol):
    def encode(self, queries: Sequence[str]) -> list[list[float]]: ...
    def search_vector(
        self, vector: Sequence[float], filters: SearchFilters | None = ..., k: int = ...
    ) -> Sequence[RankedChunk]: ...


@dataclass(frozen=True)
class HybridHit:
    chunk_id: str
    doc_id: str
    score: float  # the RRF score
    rank: int  # 1-based, after reranking when there was one
    # the chunk's rank in each list that found it: {"lexical:0": 3, "dense:1": 7}
    ranks: Mapping[str, int] = field(default_factory=dict)
    rerank_score: float | None = None  # the cross-encoder's score, if reranked


def rrf(
    lists: Mapping[str, Sequence[RankedChunk]],
    k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[HybridHit]:
    """Fuse ranked lists by Reciprocal Rank Fusion; each chunk appears once."""
    if k < 0:
        raise ValueError("rrf k must be >= 0")
    scores: dict[str, float] = {}
    docs: dict[str, str] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name, hits in lists.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        seen: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen:  # a list counts a chunk once, at its best rank
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + hit.rank)
            docs[hit.chunk_id] = hit.doc_id
            ranks.setdefault(hit.chunk_id, {})[name] = hit.rank
    order = sorted(scores, key=lambda c: (-scores[c], min(ranks[c].values()), c))
    return [HybridHit(c, docs[c], scores[c], i, dict(ranks[c])) for i, c in enumerate(order, 1)]


class HybridRetriever:
    """Lexical + dense search over one or more queries, fused with RRF."""

    def __init__(
        self,
        lexical: LexicalSearch | None,
        dense: DenseSearch | None,
        depth: int = 100,
        rrf_k: int = RRF_K,
        weights: Mapping[str, float] | None = None,
        lexical_mode: Mode = "any",
        reranker: Reranker | None = None,
        extra_dense: Sequence[DenseSearch] = (),
    ) -> None:
        if lexical is None and dense is None:
            raise ValueError("need at least one retriever")
        self.lexical = lexical
        self.dense = dense
        # more dense retrievers (other embedding models), each adding its own lists, named
        # "dense2:<i>", "dense3:<i>" ...; weights key them as "dense2", "dense3" (5.3)
        self.extra_dense = list(extra_dense)
        self.depth = depth  # how many hits each retriever returns per query
        self.rrf_k = rrf_k
        self.weights = weights  # per retriever: {"lexical": 1.0, "dense": 1.0}
        self.lexical_mode = lexical_mode
        self.reranker = reranker

    def ranked_lists(
        self, queries: Sequence[str], filters: SearchFilters | None = None
    ) -> dict[str, Sequence[RankedChunk]]:
        """Every retriever's ranked list for every query, named ``"<retriever>:<query index>"``."""
        lists: dict[str, Sequence[RankedChunk]] = {}
        if self.lexical is not None:
            for i, q in enumerate(queries):
                lists[f"lexical:{i}"] = self.lexical.search(
                    q, filters, k=self.depth, mode=self.lexical_mode
                )
        denses = ([self.dense] if self.dense is not None else []) + self.extra_dense
        for n, dense in enumerate(denses, 1):
            name = "dense" if n == 1 else f"dense{n}"
            for i, vector in enumerate(dense.encode(list(queries))):  # one batch
                lists[f"{name}:{i}"] = dense.search_vector(vector, filters, k=self.depth)
        return lists

    def search(
        self,
        queries: str | Sequence[str],
        filters: SearchFilters | None = None,
        k: int = 50,
        rerank: bool | None = None,
    ) -> list[HybridHit]:
        """Fused (and, by default, reranked) hits; ``rerank=False`` skips the reranker."""
        if rerank and self.reranker is None:
            raise ValueError("rerank requested but no reranker was given")
        queries = [queries] if isinstance(queries, str) else list(queries)
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))  # drop repeats
        if not queries:
            return []
        lists = self.ranked_lists(queries, filters)
        weights = None
        if self.weights:
            weights = {name: self.weights.get(name.split(":")[0], 1.0) for name in lists}
        fused = rrf(lists, self.rrf_k, weights)
        if self.reranker is None or rerank is False:
            return fused[:k]
        reranked = self.reranker.rerank(queries[0], fused)
        by_id = {h.chunk_id: h for h in fused}
        head = [replace(by_id[r.chunk_id], rerank_score=r.score) for r in reranked]
        tail = fused[len(head) :]
        return [replace(h, rank=i) for i, h in enumerate([*head, *tail][:k], 1)]
