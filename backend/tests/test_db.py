"""Database schema, migrations and loaders (PLAN 3.6).

Tests marked ``db`` need the Postgres from docker-compose.yml; each gets a fresh, empty
database that is dropped afterwards, so they never touch the working ``juris`` database.
"""

import datetime as dt
import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from juris.db import models as m
from juris.db.embeddings import index_name, model_dims, register_model, vector_expr
from juris.db.load import CorpusLoader, row_counts
from tests.conftest import migrate

# ---- no database needed ------------------------------------------------------------------


def test_candidate_embedding_models_have_dimensions() -> None:
    dims = model_dims()
    assert dims["BAAI/bge-m3"] == 1024
    assert all(d > 0 for d in dims.values())


def test_index_names_are_valid_identifiers() -> None:
    assert index_name("BAAI/bge-m3") == "ix_chunk_embeddings_hnsw_baai_bge_m3"
    assert len(index_name("x" * 200)) <= 63
    assert vector_expr(1024) == "(embedding::vector(1024))"


def test_upsert_compiles_to_on_conflict_update() -> None:
    stmt = insert(m.Snapshot).values([{"snapshot_id": "s", "slice_name": "x"}])
    stmt = stmt.on_conflict_do_update(
        index_elements=["snapshot_id"], set_={"slice_name": stmt.excluded.slice_name}
    )
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (snapshot_id) DO UPDATE" in sql


def tables(conn: Connection) -> set[str]:
    rows = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
    return {r[0] for r in rows}


@pytest.mark.db
def test_upgrade_head_on_an_empty_database_and_back(engine: Engine) -> None:
    with engine.begin() as conn:
        migrate(conn)
        assert {
            "snapshots", "documents", "paragraphs", "chunks", "chunk_embeddings",
            "embedding_models", "statute_sections", "citation_edges", "runs", "events",
            "llm_calls",
        } <= tables(conn)  # fmt: skip
        # the models and the migration describe the same schema
        diff = compare_metadata(MigrationContext.configure(conn), m.Base.metadata)
        assert diff == []
    with engine.begin() as conn:
        migrate(conn, "base", down=True)
        assert tables(conn) == {"alembic_version"}
    with engine.begin() as conn:
        migrate(conn)
        assert "documents" in tables(conn)


@pytest.mark.db
def test_chunks_get_a_tsvector_and_vectors_are_searchable_per_model(engine: Engine) -> None:
    with engine.begin() as conn:
        migrate(conn)
        conn.execute(text("INSERT INTO snapshots (snapshot_id, slice_name, counts, notes) "
                          "VALUES ('s', 'x', '{}', '[]')"))  # fmt: skip
        conn.execute(
            text(
                "INSERT INTO documents (doc_id, snapshot_id, kind, source, licence, title, judges,"
                " citations, meta_sources, meta_conflicts) VALUES ('D1', 's', 'judgment', 'aws_sc',"
                " 'CC-BY-4.0', 'A v. B', '{}', '{}', '{}', '[]')"
            )
        )
        for cid, body in [("D1#1", "specific performance of a contract"), ("D1#2", "bail")]:
            conn.execute(
                text(
                    "INSERT INTO chunks (chunk_id, doc_id, text, char_start, char_end) "
                    "VALUES (:c, 'D1', :t, 0, 10)"
                ),
                {"c": cid, "t": body},
            )
        query = text("SELECT chunk_id FROM chunks WHERE tsv @@ plainto_tsquery('english', :q)")
        hits: list[str] = list(conn.execute(query, {"q": "performed"}).scalars())
        assert hits == ["D1#1"]  # stemmed: "performed" matches "performance"

        register_model(conn, "test/tiny", 3)
        register_model(conn, "test/tiny", 3)  # idempotent
        with pytest.raises(ValueError, match="dim 3"):
            register_model(conn, "test/tiny", 4)
        for cid, vec in [("D1#1", "[1,0,0]"), ("D1#2", "[0,1,0]")]:
            conn.execute(
                text(
                    "INSERT INTO chunk_embeddings (chunk_id, model, embedding) "
                    "VALUES (:c, 'test/tiny', :v)"
                ),
                {"c": cid, "v": vec},
            )
        nearest = conn.execute(
            text(
                f"SELECT chunk_id FROM chunk_embeddings WHERE model = 'test/tiny' "
                f"ORDER BY {vector_expr(3)} <=> '[0.9,0.1,0]'::vector(3) LIMIT 1"
            )
        ).scalar()
        assert nearest == "D1#1"
        idx = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"),
            {"n": index_name("test/tiny")},
        ).scalar()
        assert idx is not None and "hnsw" in idx and "vector(3)" in idx


# ---- loader on a tiny processed slice ----------------------------------------------------


def write_slice(root: Path, paragraphs: int = 3) -> None:
    snap = "snap-t"
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, Any]] = [
        {"doc_id": "SC-2015_3_243_286", "kind": "judgment", "reason": "core", "source": "aws_sc",
         "source_url": "https://example.org/a.pdf", "sha256": "ab"},
        {"doc_id": "ACT-contract_act", "kind": "statute", "reason": "statute",
         "source": "hf_legal_docs", "source_url": None, "sha256": "cd"},
    ]  # fmt: skip
    (root / "snapshots" / f"{snap}.json").write_text(
        json.dumps(
            {
                "snapshot_id": snap,
                "slice_name": "t",
                "slice_config_sha256": "00",
                "created_at": "2026-09-27T01:30:45Z",
                "counts": {"total": 2},
                "notes": ["test"],
                "documents": docs,
            }
        )
    )
    parsed = root / "interim" / "parsed" / snap
    parsed.mkdir(parents=True, exist_ok=True)
    (parsed / "_summary.json").write_text(
        json.dumps([{"doc_id": d["doc_id"], "pages": 3, "language": "en"} for d in docs])
    )
    seg = root / "interim" / "segmented" / snap
    seg.mkdir(parents=True, exist_ok=True)
    (seg / "_summary.json").write_text(
        json.dumps([{"doc_id": "SC-2015_3_243_286", "numbering": "explicit"}])
    )
    paras = [
        {"no": str(i + 1), "part": 1, "section": "body", "numbering": "explicit",
         "contested": False, "char_start": i * 10, "char_end": i * 10 + 9, "page_start": 1,
         "page_end": 1, "text": f"{i + 1}. Paragraph."}
        for i in range(paragraphs)
    ]  # fmt: skip
    with gzip.open(seg / "SC-2015_3_243_286.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"doc_id": "SC-2015_3_243_286", "paragraphs": paras}, fh)
    meta = root / "interim" / "metadata" / snap
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "metadata.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "SC-2015_3_243_286",
                "title": "K.P. MANU versus CHAIRMAN",
                "court": "Supreme Court of India",
                "court_level": "supreme_court",
                "decision_date": "2015-02-26",
                "judges": ["DIPAK MISRA", "V. GOPALA GOWDA"],
                "bench_strength": 2,
                "citations": ["[2015] 3 SCR 243", "2015 INSC 163"],
                "sources": {"judges": "aws_sc"},
                "conflicts": [],
            }
        )
        + "\n"
    )
    statutes = root / "processed" / "statutes"
    statutes.mkdir(parents=True, exist_ok=True)
    section: dict[str, Any] = {
        "act": "Indian Contract Act", "act_year": 1872, "title": "t", "text": "x",
        "chapter": None, "repealed": False, "sub_sections": [], "amendment_notes": [],
        "amended_by": [], "in_force_from": "1872-09-01", "in_force_to": None,
        "amendments_curated": False, "curation_source": None, "char_start": 0, "char_end": 1,
        "page_start": 1, "page_end": 1,
    }  # fmt: skip
    (statutes / "contract_act.jsonl").write_text(
        "\n".join(json.dumps(dict(section, section=no)) for no in ("1", "2", "19A")) + "\n"
    )
    chunks = root / "processed" / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    records = [
        {"chunk_id": f"SC-2015_3_243_286#c{i:04d}", "doc_id": "SC-2015_3_243_286",
         "text": f"{i}. Paragraph.", "context_header": "A v. B", "char_start": 0, "char_end": 9,
         "tokens": 4, "embed_tokens": 9, "kind": "paragraphs", "section": "body",
         "para_start": i - 1, "para_end": i - 1, "page_start": 1, "page_end": 1,
         "statute_section_id": None, "part": 1}
        for i in range(1, paragraphs + 1)
    ] + [
        {"chunk_id": "ACT-contract_act#s19A", "doc_id": "ACT-contract_act", "text": "x",
         "context_header": "ICA", "char_start": 0, "char_end": 1, "tokens": 1,
         "embed_tokens": 4, "kind": "section", "section": "s. 19A", "para_start": None,
         "para_end": None, "page_start": 1, "page_end": 1,
         "statute_section_id": "contract_act:19A", "part": None}
    ]  # fmt: skip
    with gzip.open(chunks / f"{snap}.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in records))


@pytest.mark.db
def test_loading_twice_leaves_the_same_rows(engine: Engine, tmp_path: Path) -> None:
    write_slice(tmp_path)
    with engine.begin() as conn:
        migrate(conn)
    counts = []
    for _ in range(2):
        with engine.begin() as conn:
            CorpusLoader(tmp_path, "snap-t").load_all(conn)
            counts.append(row_counts(conn))
    assert counts[0] == counts[1]
    assert counts[0]["documents"] == 2 and counts[0]["paragraphs"] == 3
    assert counts[0]["statute_sections"] == 3 and counts[0]["embedding_models"] >= 2
    assert counts[0]["chunks"] == 4

    with engine.begin() as conn:
        doc = conn.execute(
            text(
                "SELECT court_level, decision_date, judges, citations, para_numbering, licence "
                "FROM documents WHERE doc_id = 'SC-2015_3_243_286'"
            )
        ).one()
        assert doc == (
            "supreme_court",
            dt.date(2015, 2, 26),
            ["DIPAK MISRA", "V. GOPALA GOWDA"],
            ["[2015] 3 SCR 243", "2015 INSC 163"],
            "explicit",
            "CC-BY-4.0",
        )
        act = conn.execute(
            text("SELECT title, licence FROM documents WHERE doc_id = 'ACT-contract_act'")
        ).one()
        assert act[0] == "Indian Contract Act, 1872" and act[1].startswith("Apache-2.0")

    # re-segmenting to fewer paragraphs removes the stale paragraph and chunk rows
    write_slice(tmp_path, paragraphs=2)
    with engine.begin() as conn:
        CorpusLoader(tmp_path, "snap-t").load_all(conn)
        after = row_counts(conn)
        assert after["paragraphs"] == 2 and after["chunks"] == 3
        stored = conn.execute(text("SELECT tokens, kind FROM chunks WHERE chunk_id LIKE 'ACT%'"))
        assert stored.one() == (1, "section")
