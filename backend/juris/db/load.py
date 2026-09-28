"""Idempotent loaders for processed corpus data (PLAN 3.6).

Every row is keyed by a stable ingestion ID and written with ``INSERT ... ON CONFLICT DO
UPDATE``, so loading the same snapshot twice leaves the same rows. Paragraphs are replaced per
document (upsert by ``(doc_id, seq)``, then rows past the new count are deleted), so a
re-segmented document doesn't keep stale tail paragraphs.
"""

import datetime as dt
import gzip
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from juris.db import models as m
from juris.db.embeddings import model_dims, register_model
from juris.ingest.statutes import load_acts

LICENCES = {
    "aws_sc": "CC-BY-4.0",
    "aws_hc": "CC-BY-4.0",
    "hf_legal_docs": "Apache-2.0 (research use; text not redistributed)",
}
BATCH = 1000


def _batches(rows: Iterable[dict[str, Any]], size: int = BATCH) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def upsert(conn: Connection, table: Any, rows: Iterable[dict[str, Any]], keys: list[str]) -> int:
    """Insert rows, updating every non-key column on conflict; returns rows written."""
    n = 0
    for batch in _batches(rows):
        stmt = insert(table).values(batch)
        cols = {c: stmt.excluded[c] for c in batch[0] if c not in keys}
        stmt = stmt.on_conflict_do_update(index_elements=keys, set_=cols) if cols else stmt
        conn.execute(stmt if cols else stmt.on_conflict_do_nothing(index_elements=keys))
        n += len(batch)
    return n


def _date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


def _read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class LoadStats:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, table: str, n: int) -> None:
        self.counts[table] = self.counts.get(table, 0) + n


class CorpusLoader:
    """Loads one snapshot's processed outputs from ``data_dir`` into the database."""

    def __init__(self, data_dir: Path, snapshot_id: str) -> None:
        self.data_dir = data_dir
        self.snapshot_id = snapshot_id
        self.snapshot = json.loads(
            (data_dir / "snapshots" / f"{snapshot_id}.json").read_text(encoding="utf-8")
        )
        self.stats = LoadStats()

    def _interim(self, stage: str) -> Path:
        return self.data_dir / "interim" / stage / self.snapshot_id

    # -- snapshots and documents ---------------------------------------------------------

    def load_snapshot(self, conn: Connection) -> None:
        s = self.snapshot
        row = {
            "snapshot_id": s["snapshot_id"],
            "slice_name": s["slice_name"],
            "slice_config_sha256": s.get("slice_config_sha256"),
            "created_at": dt.datetime.fromisoformat(s["created_at"].replace("Z", "+00:00")),
            "counts": s.get("counts", {}),
            "notes": s.get("notes", []),
        }
        self.stats.add("snapshots", upsert(conn, m.Snapshot, [row], ["snapshot_id"]))

    def document_rows(self) -> list[dict[str, Any]]:
        parsed = {
            r["doc_id"]: r
            for r in json.loads((self._interim("parsed") / "_summary.json").read_text())
        }
        seg_path = self._interim("segmented") / "_summary.json"
        segmented = (
            {r["doc_id"]: r for r in json.loads(seg_path.read_text())} if seg_path.exists() else {}
        )
        meta_path = self._interim("metadata") / "metadata.jsonl"
        metadata: dict[str, dict[str, Any]] = {}
        if meta_path.exists():
            with meta_path.open(encoding="utf-8") as fh:
                metadata = {(r := json.loads(line))["doc_id"]: r for line in fh}
        acts = load_acts()
        rows = []
        for doc in self.snapshot["documents"]:
            doc_id = doc["doc_id"]
            meta = metadata.get(doc_id, {})
            p = parsed.get(doc_id, {})
            act = acts.get(doc_id.removeprefix("ACT-")) if doc["kind"] == "statute" else None
            title = meta.get("title") or (f"{act.title}, {act.year}" if act else doc_id)
            rows.append(
                {
                    "doc_id": doc_id,
                    "snapshot_id": self.snapshot_id,
                    "kind": doc["kind"],
                    "reason": doc.get("reason"),
                    "source": doc["source"],
                    "source_url": doc.get("source_url"),
                    "licence": LICENCES.get(doc["source"], "unknown"),
                    "sha256": doc.get("sha256"),
                    "title": title,
                    "court": meta.get("court"),
                    "court_level": meta.get("court_level"),
                    "decision_date": _date(meta.get("decision_date")),
                    "cnr": meta.get("cnr"),
                    "case_number": meta.get("case_number"),
                    "petitioner": meta.get("petitioner"),
                    "respondent": meta.get("respondent"),
                    "judges": meta.get("judges") or [],
                    "author": meta.get("author"),
                    "bench_strength": meta.get("bench_strength"),
                    "citations": meta.get("citations") or [],
                    "neutral_citation": meta.get("neutral_citation"),
                    "disposition": meta.get("disposition"),
                    "headnote": meta.get("headnote"),
                    "pages": p.get("pages"),
                    "language": p.get("language"),
                    "para_numbering": segmented.get(doc_id, {}).get("numbering"),
                    "meta_sources": meta.get("sources") or {},
                    "meta_conflicts": meta.get("conflicts") or [],
                    "updated_at": func.now(),
                }
            )
        return rows

    def load_documents(self, conn: Connection) -> None:
        self.stats.add("documents", upsert(conn, m.Document, self.document_rows(), ["doc_id"]))

    # -- paragraphs and statute sections ---------------------------------------------------

    def load_paragraphs(self, conn: Connection) -> None:
        seg_dir = self._interim("segmented")
        for path in sorted(seg_dir.glob("*.json.gz")):
            doc = _read_gz(path)
            rows = [
                {
                    "doc_id": doc["doc_id"],
                    "seq": seq,
                    "no": p["no"],
                    "part": p["part"],
                    "section": p["section"],
                    "numbering": p["numbering"],
                    "contested": p.get("contested", False),
                    "char_start": p["char_start"],
                    "char_end": p["char_end"],
                    "page_start": p["page_start"],
                    "page_end": p["page_end"],
                    "text": p["text"],
                }
                for seq, p in enumerate(doc["paragraphs"])
            ]
            self.stats.add("paragraphs", upsert(conn, m.Paragraph, rows, ["doc_id", "seq"]))
            conn.execute(
                delete(m.Paragraph).where(
                    m.Paragraph.doc_id == doc["doc_id"], m.Paragraph.seq >= len(rows)
                )
            )

    def load_statutes(self, conn: Connection) -> None:
        out_dir = self.data_dir / "processed" / "statutes"
        known = {d["doc_id"] for d in self.snapshot["documents"]}
        for path in sorted(out_dir.glob("*.jsonl")):
            act_id = path.stem
            if f"ACT-{act_id}" not in known:
                continue
            with path.open(encoding="utf-8") as fh:
                sections = [json.loads(line) for line in fh]
            rows = [
                {
                    "section_id": f"{act_id}:{s['section']}",
                    "doc_id": f"ACT-{act_id}",
                    "act_id": act_id,
                    "act": s["act"],
                    "act_year": s["act_year"],
                    "section": s["section"],
                    "title": s["title"],
                    "text": s["text"],
                    "chapter": s["chapter"],
                    "repealed": s["repealed"],
                    "sub_sections": s["sub_sections"],
                    "amendment_notes": s["amendment_notes"],
                    "amended_by": s["amended_by"],
                    "in_force_from": _date(s["in_force_from"]),
                    "in_force_to": _date(s["in_force_to"]),
                    "amendments_curated": s["amendments_curated"],
                    "curation_source": s["curation_source"],
                    "char_start": s["char_start"],
                    "char_end": s["char_end"],
                    "page_start": s["page_start"],
                    "page_end": s["page_end"],
                }
                for s in sections
            ]
            self.stats.add("statute_sections", upsert(conn, m.StatuteSection, rows, ["section_id"]))

    def load_chunks(self, conn: Connection) -> None:
        """Chunks from 3.7; chunks of these documents that are no longer produced are removed."""
        path = self.data_dir / "processed" / "chunks" / f"{self.snapshot_id}.jsonl.gz"
        if not path.exists():
            return
        columns = {c.name for c in m.Chunk.__table__.columns} - {"tsv"}
        ids: set[str] = set()
        docs: set[str] = set()

        def rows() -> Iterator[dict[str, Any]]:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    record = json.loads(line)
                    ids.add(record["chunk_id"])
                    docs.add(record["doc_id"])
                    yield {k: v for k, v in record.items() if k in columns}

        self.stats.add("chunks", upsert(conn, m.Chunk, rows(), ["chunk_id"]))
        stale = [
            cid
            for cid in conn.execute(
                select(m.Chunk.chunk_id).where(m.Chunk.doc_id.in_(docs))
            ).scalars()
            if cid not in ids
        ]
        for batch in range(0, len(stale), BATCH):
            conn.execute(delete(m.Chunk).where(m.Chunk.chunk_id.in_(stale[batch : batch + BATCH])))

    def load_citations(self, conn: Connection) -> None:
        """Citation edges and aliases from 3.9; edges no longer produced are removed."""
        base = self.data_dir / "processed" / "citations" / self.snapshot_id
        edges_path, aliases_path = Path(f"{base}.edges.jsonl.gz"), Path(f"{base}.aliases.jsonl.gz")
        if not edges_path.exists():
            return
        with gzip.open(aliases_path, "rt", encoding="utf-8") as fh:
            aliases = [json.loads(line) for line in fh]
        self.stats.add("citation_aliases", upsert(conn, m.CitationAlias, aliases, ["alias"]))
        keep = {a["alias"] for a in aliases}
        stale_aliases = [
            a for a in conn.execute(select(m.CitationAlias.alias)).scalars() if a not in keep
        ]
        for i in range(0, len(stale_aliases), BATCH):
            conn.execute(
                delete(m.CitationAlias).where(
                    m.CitationAlias.alias.in_(stale_aliases[i : i + BATCH])
                )
            )

        keys: set[tuple[str, int, str]] = set()
        docs: set[str] = set()

        def rows() -> Iterator[dict[str, Any]]:
            with gzip.open(edges_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    edge = json.loads(line)
                    keys.add((edge["source_doc_id"], edge["char_start"], edge["raw"]))
                    docs.add(edge["source_doc_id"])
                    yield edge

        key_cols = ["source_doc_id", "char_start", "raw"]
        self.stats.add("citation_edges", upsert(conn, m.CitationEdge, rows(), key_cols))
        current = conn.execute(
            select(
                m.CitationEdge.edge_id,
                m.CitationEdge.source_doc_id,
                m.CitationEdge.char_start,
                m.CitationEdge.raw,
            ).where(m.CitationEdge.source_doc_id.in_(docs))
        )
        stale = [r[0] for r in current if (r[1], r[2], r[3]) not in keys]
        for i in range(0, len(stale), BATCH):
            conn.execute(
                delete(m.CitationEdge).where(m.CitationEdge.edge_id.in_(stale[i : i + BATCH]))
            )

    def register_embedding_models(self, conn: Connection) -> None:
        for model, dim in model_dims().items():
            register_model(conn, model, dim)
            self.stats.add("embedding_models", 1)

    def load_all(self, conn: Connection) -> LoadStats:
        self.load_snapshot(conn)
        self.load_documents(conn)
        self.load_paragraphs(conn)
        self.load_statutes(conn)
        self.load_chunks(conn)
        self.load_citations(conn)
        self.register_embedding_models(conn)
        return self.stats


TABLES = [
    m.Snapshot,
    m.Document,
    m.Paragraph,
    m.StatuteSection,
    m.Chunk,
    m.EmbeddingModel,
    m.ChunkEmbedding,
    m.CitationEdge,
    m.CitationAlias,
]


def row_counts(conn: Connection) -> dict[str, int]:
    return {
        t.__tablename__: conn.execute(select(func.count()).select_from(t)).scalar_one()
        for t in TABLES
    }
