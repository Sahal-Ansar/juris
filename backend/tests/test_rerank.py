"""Reranking (PLAN 4.4): the pinned download, reordering, top_n, and the hybrid toggle.

A fake cross-encoder (word overlap with the query) stands in for bge-reranker-v2-m3; the real
model is checked only when PyTorch and its weights are present (``uv sync --group embed`` and
``scripts/fetch_reranker.py``).
"""

import hashlib
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine

from juris.retrieval import rerank as rr
from juris.retrieval import weights
from juris.retrieval.hybrid import HybridRetriever
from juris.retrieval.rerank import Reranker, reranker_dir
from tests.test_hybrid import StubDense, StubLexical, ranked
from tests.test_lexical import ACT, HC, SC_NEW, SC_OLD, corpus  # noqa: F401 (fixture)


@dataclass
class OverlapEncoder:
    """Share of the query's words found in the passage."""

    model: str = "test/overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        words = set(query.lower().split())
        return [len(words & set(p.lower().split())) / len(words) for p in passages]


@dataclass(frozen=True)
class Cand:
    chunk_id: str
    doc_id: str
    score: float
    rank: int


class MemoryReranker(Reranker):
    """Passages from a dict instead of the database."""

    def __init__(self, texts: dict[str, str], top_n: int = 50) -> None:
        super().__init__(None, OverlapEncoder(), top_n)  # type: ignore[arg-type]
        self.texts = texts

    def passages(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        return {c: self.texts[c] for c in chunk_ids if c in self.texts}


# ---- no database needed ------------------------------------------------------------------


def test_fetch_checks_hashes_and_skips_good_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"a.json": b"{}", "w.bin": b"weights"}
    monkeypatch.setattr(
        rr, "FILES", {"a.json": None, "w.bin": hashlib.sha256(b"weights").hexdigest()}
    )
    calls: list[str] = []

    def fake_retrieve(url: str, path: Path) -> None:
        name = url.rsplit("/", 1)[1]
        calls.append(name)
        Path(path).write_bytes(payload[name])

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake_retrieve)
    target = rr.fetch_reranker(tmp_path)
    assert target == reranker_dir(tmp_path)
    assert (target / "w.bin").read_bytes() == b"weights" and calls == ["a.json", "w.bin"]
    rr.fetch_reranker(tmp_path)  # everything present and valid: nothing downloaded
    assert calls == ["a.json", "w.bin"]
    (target / "w.bin").write_bytes(b"corrupt")  # a bad file is fetched again
    rr.fetch_reranker(tmp_path)
    assert calls[-1] == "w.bin" and (target / "w.bin").read_bytes() == b"weights"
    payload["w.bin"] = b"tampered"
    (target / "w.bin").unlink()
    with pytest.raises(ValueError, match="SHA-256"):
        rr.fetch_reranker(tmp_path)
    assert not (target / "w.bin").exists() and not (target / "w.bin.part").exists()


def test_rerank_reorders_the_top_n_and_keeps_retrieval_ranks() -> None:
    texts = {"a": "nothing relevant", "b": "penalty clause", "c": "penalty clause reasonable"}
    cands = [Cand("a", "D", 0.9, 1), Cand("b", "D", 0.8, 2), Cand("c", "E", 0.7, 3)]
    hits = MemoryReranker(texts).rerank("reasonable penalty clause", cands)
    assert [h.chunk_id for h in hits] == ["c", "b", "a"]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert (hits[0].retrieval_rank, hits[0].retrieval_score) == (3, 0.7)
    assert hits[0].score == pytest.approx(1.0)
    # only the top_n candidates are scored; ties keep the retrieval order
    top2 = MemoryReranker(texts, top_n=2).rerank("reasonable penalty clause", cands)
    assert [h.chunk_id for h in top2] == ["b", "a"]
    assert [h.chunk_id for h in MemoryReranker(texts).rerank("zzz", cands)] == ["a", "b", "c"]
    assert len(MemoryReranker(texts).rerank("penalty", cands, k=1)) == 1
    assert MemoryReranker(texts).rerank("penalty", []) == []
    with pytest.raises(KeyError):
        MemoryReranker({}).rerank("penalty", cands)


def test_hybrid_rerank_is_toggleable() -> None:
    q = "penalty clause"
    lexical = StubLexical({q: ranked("A#1", "B#1", "C#1", "D#1")})
    dense = StubDense({q: ranked("A#1", "B#1", "C#1", "D#1")})
    texts = {"A#1": "x", "B#1": "x", "C#1": "penalty clause", "D#1": "x"}
    hybrid = HybridRetriever(lexical, dense, reranker=MemoryReranker(texts, top_n=3))
    on = hybrid.search("penalty clause")
    assert [h.chunk_id for h in on] == ["C#1", "A#1", "B#1", "D#1"]  # D#1 was beyond top_n
    assert on[0].rerank_score == pytest.approx(1.0) and on[3].rerank_score is None
    assert [h.rank for h in on] == [1, 2, 3, 4]
    off = hybrid.search("penalty clause", rerank=False)
    assert [h.chunk_id for h in off] == ["A#1", "B#1", "C#1", "D#1"]
    assert all(h.rerank_score is None for h in off)
    assert len(hybrid.search("penalty clause", k=2)) == 2
    with pytest.raises(ValueError, match="no reranker"):
        HybridRetriever(lexical, dense).search("q", rerank=True)


# ---- database ----------------------------------------------------------------------------


@pytest.mark.db
def test_reranker_reads_header_and_text_from_the_database(corpus: Engine) -> None:  # noqa: F811
    cands = [
        Cand(f"{HC}#c0001", HC, 0.5, 1),
        Cand(f"{SC_NEW}#c0002", SC_NEW, 0.4, 2),
        Cand(f"{ACT}#s74", ACT, 0.3, 3),
    ]
    reranker = Reranker(corpus, OverlapEncoder())
    passages = reranker.passages([c.chunk_id for c in cands])
    assert passages[f"{ACT}#s74"].startswith("Indian Contract Act, 1872 — s. 74")
    hits = reranker.rerank("Indian Contract Act penalty", cands)
    assert hits[0].chunk_id == f"{ACT}#s74"  # the header supplies "Indian Contract Act"
    assert SC_OLD not in {h.doc_id for h in hits}


# ---- the real model (optional) -----------------------------------------------------------

HAVE_MODEL = (
    importlib.util.find_spec("torch") is not None
    and (reranker_dir() / "model.safetensors").exists()
)


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the embed group and the reranker weights")
def test_bge_reranker_prefers_the_relevant_passage() -> None:
    model = rr.BgeReranker(reranker_dir())
    scores = model.score(
        "Can earnest money be forfeited without proof of loss?",
        [
            "The High Court granted bail to the accused pending trial.",
            "Forfeiture of earnest money is permissible where the contract so provides, "
            "even without proof of actual loss.",
        ],
    )
    assert scores[1] > scores[0] and all(0 < s < 1 for s in scores)


@pytest.mark.skipif(not HAVE_MODEL, reason="needs the embed group and the reranker weights")
def test_bge_reranker_accepts_a_query_longer_than_its_window() -> None:
    long_query = "The appellant sued for breach of contract. " * 150  # ~1,000 tokens
    scores = rr.BgeReranker(reranker_dir()).score(long_query, ["A short passage."])
    assert len(scores) == 1 and 0 < scores[0] < 1
