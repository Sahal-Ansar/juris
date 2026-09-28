"""Embedding and vector indexing (PLAN 3.8).

A fake encoder with tiny deterministic vectors stands in for bge-m3, so the resumable loop,
the index handling and the neighbour query run without PyTorch. The ``db`` tests use a fresh
database per test (``engine`` fixture in conftest.py).
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from juris.db.embeddings import index_name
from juris.retrieval.embed import (
    _literal,
    count_missing,
    embed_missing,
    embedding_input,
    nearest,
    record_in_snapshot,
)
from tests.conftest import migrate

DIM = 4


@dataclass
class FakeEncoder:
    """Vectors from a hash of the words: texts sharing words point the same way."""

    model: str = "test/fake"
    revision: str = "abc123"
    dim: int = DIM
    calls: list[int] = field(default_factory=list)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for word in t.lower().split():
                h = hashlib.sha256(word.encode()).digest()
                v[h[0] % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def test_embedding_input_prepends_the_header() -> None:
    assert embedding_input("A v. B - 2015", "Text.") == "A v. B - 2015\nText."
    assert embedding_input(None, "Text.") == "Text."


def test_vector_literal() -> None:
    assert _literal([0.5, -1.0]) == "[0.500000,-1.000000]"


def seed(engine: Engine, n: int) -> None:
    with engine.begin() as conn:
        migrate(conn)
        conn.execute(
            text(
                "INSERT INTO snapshots (snapshot_id, slice_name, counts, notes) "
                "VALUES ('s', 'x', '{}', '[]')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO documents (doc_id, snapshot_id, kind, source, licence, title, "
                "judges, citations, meta_sources, meta_conflicts) VALUES ('D1', 's', 'judgment',"
                " 'aws_sc', 'CC-BY-4.0', 'A v. B', '{}', '{}', '{}', '[]')"
            )
        )
        words = ["contract breach damages", "bail accused custody", "contract damages award"]
        for i in range(n):
            conn.execute(
                text(
                    "INSERT INTO chunks (chunk_id, doc_id, text, context_header, char_start, "
                    "char_end) VALUES (:c, 'D1', :t, 'A v. B', 0, 1)"
                ),
                {"c": f"D1#c{i:04d}", "t": words[i % 3]},
            )


@pytest.mark.db
def test_embed_missing_is_resumable_and_rebuilds_the_index(engine: Engine) -> None:
    seed(engine, 7)
    enc = FakeEncoder()
    report = embed_missing(engine, enc, fetch=3)
    assert report.embedded == 7 and report.index_rebuilt
    assert enc.calls == [3, 3, 1]
    with engine.connect() as conn:
        assert count_missing(conn, enc.model) == (7, 0)
        idx = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
            {"n": index_name(enc.model)},
        ).scalar()
        assert idx is not None and f"vector({DIM})" in idx
        rev = conn.execute(
            text("SELECT revision FROM embedding_models WHERE model = :m"), {"m": enc.model}
        ).scalar()
        assert rev == "abc123"

    # nothing left: a second run embeds nothing and keeps the index
    again = embed_missing(engine, FakeEncoder(), fetch=3)
    assert again.embedded == 0 and not again.index_rebuilt

    # a new chunk is picked up on the next run
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO chunks (chunk_id, doc_id, text, char_start, char_end) "
                "VALUES ('D1#c9999', 'D1', 'contract breach', 0, 1)"
            )
        )
    assert embed_missing(engine, FakeEncoder(), fetch=3).embedded == 1


@pytest.mark.db
def test_nearest_neighbours_and_snapshot_record(engine: Engine) -> None:
    seed(engine, 6)
    enc = FakeEncoder()
    embed_missing(engine, enc)
    with engine.begin() as conn:
        hits = nearest(conn, enc.model, DIM, "D1#c0000", k=5)
        assert [h[0] for h in hits][:1] == ["D1#c0003"]  # same words, distance 0
        assert hits[0][2] == pytest.approx(0.0, abs=1e-6)
        assert "D1#c0000" not in [h[0] for h in hits]
        entry = record_in_snapshot(conn, "s", enc)
        assert entry["chunks"] == entry["of"] == 6 and entry["revision"] == "abc123"
        stored = conn.execute(
            text(
                "SELECT embeddings -> 'test/fake' ->> 'dim' FROM snapshots WHERE snapshot_id = 's'"
            )
        ).scalar()
        assert stored == str(DIM)
