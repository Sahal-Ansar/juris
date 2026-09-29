"""Index a benchmark's own document pool with the Juris pipeline (PLAN 5.3, D-011).

External benchmarks share no document IDs with our corpus, so their pools are indexed
separately, each in its own Postgres database (``juris_pool_<name>``, same schema through
Alembic). The same code paths as the corpus are used: ``segment`` (3.3), ``chunk_judgment``
(3.7), the ``chunks`` table with its generated ``tsv`` and ``lexeme_stats`` (4.1), and
``embed_missing`` (3.8). The retrievers then run on the pool's engine unchanged.

Indexing is resumable: documents already loaded are skipped, and embedding only fills
missing vectors.
"""

import dataclasses
import re
from collections.abc import Callable, Iterable

import psycopg
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from tokenizers import Tokenizer

from juris.config import REPO_ROOT, get_settings
from juris.db import models as m
from juris.db.load import refresh_lexeme_stats
from juris.eval.datasets.common import Document
from juris.ingest.chunk import chunk_judgment
from juris.ingest.segment import segment
from juris.ingest.tokens import load_tokenizer

SNAPSHOT = "benchmark-pool"
BATCH = 500


def pool_db_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"juris_pool_{slug}"[:63]


def pool_engine(name: str) -> Engine:
    """An engine on the pool's database, created and migrated to head if needed."""
    settings = get_settings()
    db = pool_db_name(name)
    with psycopg.connect(settings.database_url(driver=None), autocommit=True) as admin:
        exists = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,)).fetchone()
        if not exists:
            admin.execute(f"CREATE DATABASE {db}")
    engine = create_engine(settings.database_url().rsplit("/", 1)[0] + f"/{db}")
    with engine.begin() as conn:
        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
    return engine


def chunk_document(doc: Document, tokenizer: Tokenizer | None = None) -> list[dict[str, object]]:
    """Chunks of one pool document, exactly as a corpus judgment is chunked."""
    paragraphs = [dataclasses.asdict(p) for p in segment(doc.text, [0]).paragraphs]
    meta = {"title": doc.title or doc.id}
    columns = {c.name for c in m.Chunk.__table__.columns} - {"tsv"}
    chunks = chunk_judgment(doc.id, doc.text, paragraphs, meta, tokenizer or load_tokenizer())
    return [{k: v for k, v in c.to_json().items() if k in columns} for c in chunks]


def index_pool(
    engine: Engine,
    documents: Iterable[Document],
    kind: str,
    licence: str,
    progress: Callable[[int, int], None] | None = None,
    tokenizer: Tokenizer | None = None,  # bge-m3's by default (3.7); tests pass a small one
) -> dict[str, int]:
    """Load the pool's documents and chunks (skipping loaded ones) and refresh lexeme stats."""
    docs = list(documents)
    with engine.begin() as conn:
        conn.execute(
            insert(m.Snapshot)
            .values(snapshot_id=SNAPSHOT, slice_name="benchmark pool")
            .on_conflict_do_nothing()
        )
        loaded = set(conn.execute(select(m.Document.doc_id)).scalars())
    todo = [d for d in docs if d.id not in loaded]
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        rows = [c for d in batch for c in chunk_document(d, tokenizer)]
        with engine.begin() as conn:
            conn.execute(
                insert(m.Document),
                [
                    {"doc_id": d.id, "snapshot_id": SNAPSHOT, "kind": kind, "source": "benchmark",
                     "licence": licence, "title": d.title or d.id}
                    for d in batch
                ],
            )  # fmt: skip
            for i in range(0, len(rows), BATCH):
                conn.execute(insert(m.Chunk), rows[i : i + BATCH])
        if progress:
            progress(start + len(batch), len(todo))
    with engine.begin() as conn:
        if todo:
            refresh_lexeme_stats(conn)
        return {
            "documents": conn.execute(select(func.count()).select_from(m.Document)).scalar_one(),
            "chunks": conn.execute(select(func.count()).select_from(m.Chunk)).scalar_one(),
            "new_documents": len(todo),
            "lexemes": conn.execute(text("SELECT count(*) FROM lexeme_stats")).scalar_one(),
        }
