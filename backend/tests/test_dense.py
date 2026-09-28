"""Dense retrieval (PLAN 4.2): ranking, filters, both search paths, query prefixes.

The fake encoder from ``test_embed`` (hashed bag-of-words vectors, 4 dimensions) stands in for
bge-m3, so the tests run without PyTorch. Each ``db`` test gets a fresh database.
"""

import datetime as dt
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert, text
from sqlalchemy.engine import Engine

from juris.db import models as m
from juris.db.embeddings import hnsw_params, index_name, prewarm
from juris.models import CourtLevel, DocumentKind
from juris.retrieval.dense import DenseRetriever, query_prefix
from juris.retrieval.embed import embed_missing
from juris.retrieval.filters import SearchFilters
from tests.conftest import migrate
from tests.test_embed import FakeEncoder

# ---- no database needed ------------------------------------------------------------------


def test_query_prefix_comes_from_the_model_config(tmp_path: Path) -> None:
    assert query_prefix("BAAI/bge-m3") == ""
    assert query_prefix("intfloat/multilingual-e5-large") == "query: "
    assert query_prefix("unknown/model") == ""
    cfg = tmp_path / "e.yaml"
    cfg.write_text("models:\n  x/y:\n    dim: 4\n    query_prefix: 'q: '\n", encoding="utf-8")
    assert query_prefix("x/y", cfg) == "q: "


class RecordingEncoder(FakeEncoder):
    seen: list[str]

    def encode(self, texts: Any) -> list[list[float]]:
        self.seen = list(texts)
        return super().encode(texts)


def test_encode_adds_the_prefix() -> None:
    enc = RecordingEncoder()
    DenseRetriever(None, enc, prefix="query: ").encode(["a b"])  # type: ignore[arg-type]
    assert enc.seen == ["query: a b"]


def test_wrong_dimension_is_rejected() -> None:
    r = DenseRetriever(None, FakeEncoder(), prefix="")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimensions"):
        r.search_vector([1.0, 0.0])


# ---- database ----------------------------------------------------------------------------

SC_OLD, SC_NEW, HC, ACT = "SC-1990_1_1_10", "SC-2015_2_1_20", "HC-DLHC0001_1_2019", "ACT-x"
TEXTS = {
    # chunk_id: (doc_id, text) - the fake encoder hashes words into 4 buckets
    f"{SC_OLD}#c0001": (SC_OLD, "penalty compensation breach"),
    f"{SC_OLD}#c0002": (SC_OLD, "bail custody accused"),
    f"{SC_NEW}#c0001": (SC_NEW, "penalty compensation breach"),
    f"{SC_NEW}#c0002": (SC_NEW, "arbitration award seat"),
    f"{HC}#c0001": (HC, "penalty compensation breach damages"),
    f"{ACT}#s74": (ACT, "penalty compensation breach"),
}


@pytest.fixture
def corpus(engine: Engine) -> Engine:
    docs: list[dict[str, Any]] = [
        {"doc_id": SC_OLD, "kind": "judgment", "court_level": "supreme_court",
         "decision_date": dt.date(1990, 3, 1), "title": "A v. B"},
        {"doc_id": SC_NEW, "kind": "judgment", "court_level": "supreme_court",
         "decision_date": dt.date(2015, 6, 1), "title": "C v. D"},
        {"doc_id": HC, "kind": "judgment", "court_level": "high_court",
         "decision_date": dt.date(2019, 1, 1), "title": "E v. F"},
        {"doc_id": ACT, "kind": "statute", "court_level": None, "decision_date": None,
         "title": "Indian Contract Act, 1872"},
    ]  # fmt: skip
    with engine.begin() as conn:
        migrate(conn)
        conn.execute(insert(m.Snapshot).values(snapshot_id="s", slice_name="t"))
        conn.execute(
            insert(m.Document),
            [{**d, "snapshot_id": "s", "source": "test", "licence": "test"} for d in docs],
        )
        conn.execute(
            insert(m.StatuteSection).values(
                section_id="contract_act:74", doc_id=ACT, act_id="contract_act",
                act="Indian Contract Act", act_year=1872, section="74", title="t", text="t",
                char_start=0, char_end=1, page_start=1, page_end=1,
            )
        )  # fmt: skip
        conn.execute(
            insert(m.Chunk),
            [
                {"chunk_id": cid, "doc_id": doc, "text": body, "char_start": 0,
                 "char_end": len(body)}
                for cid, (doc, body) in TEXTS.items()
            ],
        )  # fmt: skip
        conn.execute(
            insert(m.CitationEdge).values(
                source_doc_id=SC_NEW, raw="s. 74", char_start=0, char_end=5, kind="statute",
                target_section_id="contract_act:74",
            )
        )  # fmt: skip
    embed_missing(engine, FakeEncoder())
    return engine


def ids(hits: list[Any]) -> list[str]:
    return [h.chunk_id for h in hits]


@pytest.mark.db
def test_nearest_chunks_come_first_with_cosine_scores(corpus: Engine) -> None:
    r = DenseRetriever(corpus, FakeEncoder())
    hits = r.search("penalty compensation breach")
    assert r.last_path == "hnsw"
    assert set(ids(hits[:3])) == {f"{SC_OLD}#c0001", f"{SC_NEW}#c0001", f"{ACT}#s74"}
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in pairwise(hits))
    assert len(r.search("penalty", k=2)) == 2


@pytest.mark.db
@pytest.mark.parametrize("exact_below", [0, 1_000])  # force HNSW, then the exact scan
def test_filters_on_both_paths(corpus: Engine, exact_below: int) -> None:
    r = DenseRetriever(corpus, FakeEncoder(), exact_below=exact_below)

    def docs(f: SearchFilters) -> set[str]:
        hits = r.search("penalty compensation breach", f, k=10)
        assert r.last_path == ("exact" if exact_below else "hnsw")
        return {h.doc_id for h in hits}

    assert docs(SearchFilters(court_levels=(CourtLevel.HIGH_COURT,))) == {HC, ACT}
    assert docs(SearchFilters(date_from=dt.date(2000, 1, 1))) == {SC_NEW, HC, ACT}
    assert docs(SearchFilters(doc_kinds=(DocumentKind.STATUTE,))) == {ACT}
    assert docs(SearchFilters(acts=("ICA",))) == {SC_NEW, ACT}
    assert docs(SearchFilters(doc_ids=(SC_OLD,))) == {SC_OLD}
    # a filter leaves the nearest chunks out, and k is still filled from further away
    hits = r.search("penalty compensation breach", SearchFilters(doc_ids=(SC_OLD,)), k=2)
    assert ids(hits) == [f"{SC_OLD}#c0001", f"{SC_OLD}#c0002"]


@pytest.mark.db
def test_filters_matching_nothing_return_nothing(corpus: Engine) -> None:
    r = DenseRetriever(corpus, FakeEncoder())
    assert r.search("penalty", SearchFilters(acts=("SRA",))) == []


@pytest.mark.db
def test_prewarm_reads_the_index_and_vectors(corpus: Engine) -> None:
    with corpus.begin() as conn:
        assert prewarm(conn, FakeEncoder().model) > 0


@pytest.mark.db
def test_indexes_are_built_with_the_configured_hnsw_parameters(corpus: Engine) -> None:
    params = hnsw_params()
    with corpus.connect() as conn:
        indexdef: str = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
            {"n": index_name(FakeEncoder().model)},
        ).scalar_one()
    assert f"m='{params['m']}'" in indexdef
    assert f"ef_construction='{params['ef_construction']}'" in indexdef
