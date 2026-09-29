"""Hybrid fusion (PLAN 4.3): the RRF arithmetic, and multi-query fusion that dedupes by chunk.

Stub retrievers return fixed rankings, so every expected score can be written out by hand. One
``db`` test runs the real lexical and dense retrievers (fake encoder) together.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from sqlalchemy.engine import Engine

from juris.db.load import refresh_lexeme_stats
from juris.retrieval.dense import DenseRetriever
from juris.retrieval.filters import SearchFilters
from juris.retrieval.hybrid import HybridRetriever, rrf
from juris.retrieval.lexical import LexicalRetriever, Mode
from tests.test_dense import SC_OLD, corpus  # noqa: F401 (fixture)
from tests.test_embed import FakeEncoder


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    rank: int


def ranked(*chunk_ids: str) -> list[Hit]:
    return [Hit(c, c.split("#")[0], i) for i, c in enumerate(chunk_ids, 1)]


def test_rrf_scores_are_reciprocal_ranks_summed() -> None:
    fused = rrf({"a": ranked("x", "y", "z"), "b": ranked("y", "w")}, k=60)
    scores = {h.chunk_id: h.score for h in fused}
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["w"] == pytest.approx(1 / 62)
    assert scores["z"] == pytest.approx(1 / 63)
    assert [h.chunk_id for h in fused] == ["y", "x", "w", "z"]
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    assert fused[0].ranks == {"a": 2, "b": 1}


def test_rrf_k_and_weights() -> None:
    fused = rrf({"a": ranked("x", "y"), "b": ranked("y")}, k=0)
    assert {h.chunk_id: h.score for h in fused} == {
        "x": pytest.approx(1.0),
        "y": pytest.approx(1 / 2 + 1),
    }
    weighted = rrf({"a": ranked("x"), "b": ranked("y")}, k=60, weights={"a": 2.0})
    assert [h.chunk_id for h in weighted] == ["x", "y"]
    assert weighted[0].score == pytest.approx(2 / 61)
    with pytest.raises(ValueError):
        rrf({"a": ranked("x")}, k=-1)


def test_rrf_ties_break_on_best_rank_then_id() -> None:
    # "p" and "q" both score 1/61 + 1/63, just above "s" at 1/62 + 1/62
    fused = rrf({"a": ranked("p", "r", "q"), "b": ranked("q", "s", "p"), "c": ranked("x", "s")})
    assert [h.chunk_id for h in fused] == ["p", "q", "s", "x", "r"]
    assert fused[0].score == pytest.approx(fused[1].score)  # a tie, broken by ID
    # equal scores, different best ranks: the chunk ranked higher somewhere comes first
    # with k = 0: "z" is 1/1 in one list, "a" is 1/2 in two lists: both score 1.0
    second = [Hit("a", "d", 2)]
    tied = rrf({"x": ranked("z"), "y": second, "w": second}, k=0)
    assert [h.chunk_id for h in tied] == ["z", "a"]
    assert tied[0].score == pytest.approx(tied[1].score) == pytest.approx(1.0)


def test_a_chunk_repeated_within_one_list_counts_once() -> None:
    fused = rrf({"a": [Hit("x", "d", 1), Hit("x", "d", 3)]})
    assert len(fused) == 1 and fused[0].score == pytest.approx(1 / 61)


@dataclass
class StubLexical:
    by_query: dict[str, list[Hit]]
    calls: list[tuple[str, int, str]] = field(default_factory=list)

    def search(
        self, query: str, filters: SearchFilters | None = None, k: int = 50, mode: Mode = "any"
    ) -> Sequence[Hit]:
        self.calls.append((query, k, mode))
        return self.by_query.get(query, [])[:k]


@dataclass
class StubDense:
    by_query: dict[str, list[Hit]]
    batches: list[list[str]] = field(default_factory=list)

    def encode(self, queries: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(queries))
        return [[float(i)] for i in range(len(queries))]

    def search_vector(
        self, vector: Sequence[float], filters: SearchFilters | None = None, k: int = 50
    ) -> Sequence[Hit]:
        query = self.batches[-1][int(vector[0])]
        return self.by_query.get(query, [])[:k]


def test_multi_query_fusion_dedupes_by_chunk() -> None:
    lexical = StubLexical({"q1": ranked("D#1", "D#2"), "q2": ranked("D#2", "E#1")})
    dense = StubDense({"q1": ranked("E#1", "D#1"), "q2": ranked("D#2")})
    hits = HybridRetriever(lexical, dense, depth=10).search(["q1", "q2", "q1 ", ""])
    assert [h.chunk_id for h in hits].count("D#2") == 1
    assert {h.chunk_id for h in hits} == {"D#1", "D#2", "E#1"}
    d2 = next(h for h in hits if h.chunk_id == "D#2")
    assert d2.ranks == {"lexical:0": 2, "lexical:1": 1, "dense:1": 1}
    assert d2.score == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert hits[0].chunk_id == "D#2"
    # repeated and blank queries are dropped; dense encodes all queries in one batch
    assert [c[0] for c in lexical.calls] == ["q1", "q2"]
    assert dense.batches == [["q1", "q2"]]
    assert all(c[1] == 10 for c in lexical.calls)


def test_extra_dense_retrievers_add_their_own_lists() -> None:
    lexical = StubLexical({"q": ranked("A#1", "B#1")})
    dense = StubDense({"q": ranked("B#1", "C#1")})
    other = StubDense({"q": ranked("C#1", "D#1")})
    hybrid = HybridRetriever(lexical, dense, extra_dense=[other], weights={"dense2": 2.0})
    hits = hybrid.search("q")
    assert {h.chunk_id for h in hits} == {"A#1", "B#1", "C#1", "D#1"}
    c1 = next(h for h in hits if h.chunk_id == "C#1")
    assert c1.ranks == {"dense:0": 2, "dense2:0": 1}
    assert c1.score == pytest.approx(1 / 62 + 2 / 61) and hits[0].chunk_id == "C#1"


def test_one_retriever_weights_and_k() -> None:
    lexical = StubLexical({"q": ranked("A#1", "B#1", "C#1")})
    dense = StubDense({"q": ranked("C#1", "B#1", "A#1")})
    assert [h.chunk_id for h in HybridRetriever(lexical, None).search("q")] == [
        "A#1",
        "B#1",
        "C#1",
    ]
    favour_dense = HybridRetriever(lexical, dense, weights={"dense": 2.0}).search("q")
    assert favour_dense[0].chunk_id == "C#1"
    assert len(HybridRetriever(lexical, dense).search("q", k=2)) == 2
    assert HybridRetriever(lexical, dense).search(["", "  "]) == []
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever(None, None)


@pytest.mark.db
def test_real_retrievers_fuse(corpus: Engine) -> None:  # noqa: F811
    with corpus.begin() as conn:
        refresh_lexeme_stats(conn)
    hybrid = HybridRetriever(LexicalRetriever(corpus), DenseRetriever(corpus, FakeEncoder()))
    hits = hybrid.search(["penalty compensation", "bail custody"], k=10)
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))
    assert f"{SC_OLD}#c0002" in ids  # only the second query finds the bail chunk
    both = [h for h in hits if any(n.startswith("lexical") for n in h.ranks)]
    assert both and all(h.score > 0 for h in hits)
