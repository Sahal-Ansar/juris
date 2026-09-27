"""SQLAlchemy 2 models for the corpus, runs and events (PLAN 3.6). Alembic owns the schema.

Stable text IDs from ingestion are the primary keys (``SC-2015_3_243_286``,
``contract_act:10``), so loading the same data twice upserts instead of duplicating.

Vectors live in ``chunk_embeddings``, one row per (chunk, embedding model), in an unsized
``vector`` column: models of different dimensions coexist during the bake-off, and each model
gets its own partial HNSW index cast to its dimension (``juris.db.embeddings``).
"""

import datetime as dt
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB, list[str]: ARRAY(Text)}


def _now() -> Any:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Snapshot(Base):
    __tablename__ = "snapshots"

    snapshot_id: Mapped[str] = mapped_column(Text, primary_key=True)
    slice_name: Mapped[str] = mapped_column(Text)
    slice_config_sha256: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    counts: Mapped[dict[str, Any]] = mapped_column(default=dict)
    notes: Mapped[Any] = mapped_column(JSONB, default=list)
    loaded_at: Mapped[dt.datetime] = _now()


class Document(Base):
    """A judgment or statute (one row per snapshot document) with its normalised metadata."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("snapshots.snapshot_id"), index=True)
    kind: Mapped[str] = mapped_column(Text)  # judgment | statute
    reason: Mapped[str | None] = mapped_column(Text)  # core | one_hop | distractor | ...
    source: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    licence: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    court: Mapped[str | None] = mapped_column(Text)
    court_level: Mapped[str | None] = mapped_column(Text, index=True)
    decision_date: Mapped[dt.date | None] = mapped_column(Date, index=True)
    cnr: Mapped[str | None] = mapped_column(Text, index=True)
    case_number: Mapped[str | None] = mapped_column(Text)
    petitioner: Mapped[str | None] = mapped_column(Text)
    respondent: Mapped[str | None] = mapped_column(Text)
    judges: Mapped[list[str]] = mapped_column(default=list)
    author: Mapped[str | None] = mapped_column(Text)
    bench_strength: Mapped[int | None] = mapped_column(Integer)
    citations: Mapped[list[str]] = mapped_column(default=list)
    neutral_citation: Mapped[str | None] = mapped_column(Text, index=True)
    disposition: Mapped[str | None] = mapped_column(Text)
    headnote: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[int | None] = mapped_column(Integer)
    language: Mapped[str | None] = mapped_column(Text)
    para_numbering: Mapped[str | None] = mapped_column(Text)  # explicit | inferred
    meta_sources: Mapped[dict[str, Any]] = mapped_column(default=dict)
    meta_conflicts: Mapped[Any] = mapped_column(JSONB, default=list)  # list of strings
    updated_at: Mapped[dt.datetime] = _now()

    __table_args__ = (Index("ix_documents_citations", "citations", postgresql_using="gin"),)


class Paragraph(Base):
    __tablename__ = "paragraphs"

    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), primary_key=True
    )
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)  # order within the document
    no: Mapped[str] = mapped_column(Text)  # "12", "12-13", "p-4", "f-1", "h-2"
    part: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text)  # front | headnote | body
    numbering: Mapped[str] = mapped_column(Text)  # explicit | inferred
    contested: Mapped[bool] = mapped_column(Boolean, default=False)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class StatuteSection(Base):
    __tablename__ = "statute_sections"

    section_id: Mapped[str] = mapped_column(Text, primary_key=True)  # "contract_act:10"
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"))
    act_id: Mapped[str] = mapped_column(Text, index=True)
    act: Mapped[str] = mapped_column(Text)
    act_year: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    chapter: Mapped[str | None] = mapped_column(Text)
    repealed: Mapped[bool] = mapped_column(Boolean, default=False)
    sub_sections: Mapped[list[str]] = mapped_column(default=list)
    amendment_notes: Mapped[list[str]] = mapped_column(default=list)
    amended_by: Mapped[list[str]] = mapped_column(default=list)
    in_force_from: Mapped[dt.date | None] = mapped_column(Date)
    in_force_to: Mapped[dt.date | None] = mapped_column(Date)
    amendments_curated: Mapped[bool] = mapped_column(Boolean, default=False)
    curation_source: Mapped[str | None] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)

    __table_args__ = (UniqueConstraint("act_id", "section"),)


class Chunk(Base):
    """A retrievable span (PLAN 3.7). ``tsv`` is generated from ``text`` for lexical search."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(240), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    context_header: Mapped[str | None] = mapped_column(Text)  # prepended for embedding only
    para_start: Mapped[int | None] = mapped_column(Integer)
    para_end: Mapped[int | None] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(Text)
    statute_section_id: Mapped[str | None] = mapped_column(
        ForeignKey("statute_sections.section_id", ondelete="CASCADE")
    )
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    __table_args__ = (Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),)


class EmbeddingModel(Base):
    __tablename__ = "embedding_models"

    model: Mapped[str] = mapped_column(Text, primary_key=True)  # e.g. "BAAI/bge-m3"
    dim: Mapped[int] = mapped_column(Integer)
    index_name: Mapped[str] = mapped_column(Text)
    registered_at: Mapped[dt.datetime] = _now()


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(ForeignKey("embedding_models.model"), primary_key=True)
    embedding: Mapped[Any] = mapped_column(Vector())  # dimension fixed per model, see above
    created_at: Mapped[dt.datetime] = _now()


class CitationEdge(Base):
    """A citation found in a judgment (PLAN 3.9): resolved to a corpus document when possible."""

    __tablename__ = "citation_edges"

    edge_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source_doc_id: Mapped[str] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True
    )
    target_doc_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.doc_id", ondelete="SET NULL"), index=True
    )
    raw: Mapped[str] = mapped_column(Text)
    canonical: Mapped[str | None] = mapped_column(Text, index=True)
    reporter: Mapped[str | None] = mapped_column(Text)
    source_para: Mapped[str | None] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    treatment: Mapped[str | None] = mapped_column(Text)  # followed | distinguished | ... (D-013)

    __table_args__ = (UniqueConstraint("source_doc_id", "char_start", "raw"),)


class RunRow(Base):
    """Run manifests (PLAN 0.5, D-009); the table predates Alembic and is adopted by it."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    profile: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    manifest: Mapped[dict[str, Any]] = mapped_column()


class EventRow(Base):
    """The append-only event log (IDEA_final §12): one row per event, ordered by ``seq``."""

    __tablename__ = "events"

    run_id: Mapped[str] = mapped_column(Text, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(Text)
    case_id: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, index=True)
    stage: Mapped[str | None] = mapped_column(Text)
    agent: Mapped[str | None] = mapped_column(Text)
    schema_version: Mapped[str] = mapped_column(Text)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column()


class LlmCall(Base):
    __tablename__ = "llm_calls"

    call_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(Text, index=True)
    provider: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)
    cached: Mapped[bool] = mapped_column(Boolean)
    tags: Mapped[dict[str, Any]] = mapped_column(default=dict)
    at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
