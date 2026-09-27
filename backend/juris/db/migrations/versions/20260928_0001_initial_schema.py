"""Initial schema: corpus, vectors, citations, runs, events, LLM calls (PLAN 3.6).

Revision ID: 0001
Revises:
Create Date: 2026-09-28 02:44:30.013229
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embedding_models",
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("index_name", sa.Text(), nullable=False),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("model"),
    )
    op.create_table(
        "events",
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("case_id", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=True),
        sa.Column("agent", sa.Text(), nullable=True),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "seq"),
    )
    op.create_index(op.f("ix_events_type"), "events", ["type"], unique=False)
    op.create_table(
        "llm_calls",
        sa.Column("call_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("call_id"),
    )
    op.create_index(op.f("ix_llm_calls_run_id"), "llm_calls", ["run_id"], unique=False)
    # `runs` predates Alembic (PLAN 0.5 created it with IF NOT EXISTS): adopt it if present.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id      text PRIMARY KEY,
            profile     text NOT NULL,
            status      text NOT NULL,
            started_at  timestamptz NOT NULL,
            ended_at    timestamptz,
            manifest    jsonb NOT NULL
        )
        """
    )
    op.create_table(
        "snapshots",
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("slice_name", sa.Text(), nullable=False),
        sa.Column("slice_config_sha256", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("counts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("notes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "loaded_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.PrimaryKeyConstraint("snapshot_id"),
    )
    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(length=200), nullable=False),
        sa.Column("snapshot_id", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("licence", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("court", sa.Text(), nullable=True),
        sa.Column("court_level", sa.Text(), nullable=True),
        sa.Column("decision_date", sa.Date(), nullable=True),
        sa.Column("cnr", sa.Text(), nullable=True),
        sa.Column("case_number", sa.Text(), nullable=True),
        sa.Column("petitioner", sa.Text(), nullable=True),
        sa.Column("respondent", sa.Text(), nullable=True),
        sa.Column("judges", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("bench_strength", sa.Integer(), nullable=True),
        sa.Column("citations", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("neutral_citation", sa.Text(), nullable=True),
        sa.Column("disposition", sa.Text(), nullable=True),
        sa.Column("headnote", sa.Text(), nullable=True),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("para_numbering", sa.Text(), nullable=True),
        sa.Column("meta_sources", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("meta_conflicts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["snapshots.snapshot_id"],
        ),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index(
        "ix_documents_citations", "documents", ["citations"], unique=False, postgresql_using="gin"
    )
    op.create_index(op.f("ix_documents_cnr"), "documents", ["cnr"], unique=False)
    op.create_index(op.f("ix_documents_court_level"), "documents", ["court_level"], unique=False)
    op.create_index(
        op.f("ix_documents_decision_date"), "documents", ["decision_date"], unique=False
    )
    op.create_index(
        op.f("ix_documents_neutral_citation"), "documents", ["neutral_citation"], unique=False
    )
    op.create_index(op.f("ix_documents_snapshot_id"), "documents", ["snapshot_id"], unique=False)
    op.create_table(
        "citation_edges",
        sa.Column("edge_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source_doc_id", sa.String(length=200), nullable=False),
        sa.Column("target_doc_id", sa.String(length=200), nullable=True),
        sa.Column("raw", sa.Text(), nullable=False),
        sa.Column("canonical", sa.Text(), nullable=True),
        sa.Column("reporter", sa.Text(), nullable=True),
        sa.Column("source_para", sa.Text(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("treatment", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["source_doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_doc_id"], ["documents.doc_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("edge_id"),
        sa.UniqueConstraint("source_doc_id", "char_start", "raw"),
    )
    op.create_index(
        op.f("ix_citation_edges_canonical"), "citation_edges", ["canonical"], unique=False
    )
    op.create_index(
        op.f("ix_citation_edges_source_doc_id"), "citation_edges", ["source_doc_id"], unique=False
    )
    op.create_index(
        op.f("ix_citation_edges_target_doc_id"), "citation_edges", ["target_doc_id"], unique=False
    )
    op.create_table(
        "paragraphs",
        sa.Column("doc_id", sa.String(length=200), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("no", sa.Text(), nullable=False),
        sa.Column("part", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("numbering", sa.Text(), nullable=False),
        sa.Column("contested", sa.Boolean(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("doc_id", "seq"),
    )
    op.create_table(
        "statute_sections",
        sa.Column("section_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.String(length=200), nullable=False),
        sa.Column("act_id", sa.Text(), nullable=False),
        sa.Column("act", sa.Text(), nullable=False),
        sa.Column("act_year", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("chapter", sa.Text(), nullable=True),
        sa.Column("repealed", sa.Boolean(), nullable=False),
        sa.Column("sub_sections", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("amendment_notes", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("amended_by", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("in_force_from", sa.Date(), nullable=True),
        sa.Column("in_force_to", sa.Date(), nullable=True),
        sa.Column("amendments_curated", sa.Boolean(), nullable=False),
        sa.Column("curation_source", sa.Text(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("section_id"),
        sa.UniqueConstraint("act_id", "section"),
    )
    op.create_index(
        op.f("ix_statute_sections_act_id"), "statute_sections", ["act_id"], unique=False
    )
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=240), nullable=False),
        sa.Column("doc_id", sa.String(length=200), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("context_header", sa.Text(), nullable=True),
        sa.Column("para_start", sa.Integer(), nullable=True),
        sa.Column("para_end", sa.Integer(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("statute_section_id", sa.Text(), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.doc_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["statute_section_id"], ["statute_sections.section_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index(op.f("ix_chunks_doc_id"), "chunks", ["doc_id"], unique=False)
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], unique=False, postgresql_using="gin")
    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.String(length=240), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.chunk_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["model"],
            ["embedding_models.model"],
        ),
        sa.PrimaryKeyConstraint("chunk_id", "model"),
    )


def downgrade() -> None:
    op.drop_table("chunk_embeddings")
    op.drop_index("ix_chunks_tsv", table_name="chunks", postgresql_using="gin")
    op.drop_index(op.f("ix_chunks_doc_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_statute_sections_act_id"), table_name="statute_sections")
    op.drop_table("statute_sections")
    op.drop_table("paragraphs")
    op.drop_index(op.f("ix_citation_edges_target_doc_id"), table_name="citation_edges")
    op.drop_index(op.f("ix_citation_edges_source_doc_id"), table_name="citation_edges")
    op.drop_index(op.f("ix_citation_edges_canonical"), table_name="citation_edges")
    op.drop_table("citation_edges")
    op.drop_index(op.f("ix_documents_snapshot_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_neutral_citation"), table_name="documents")
    op.drop_index(op.f("ix_documents_decision_date"), table_name="documents")
    op.drop_index(op.f("ix_documents_court_level"), table_name="documents")
    op.drop_index(op.f("ix_documents_cnr"), table_name="documents")
    op.drop_index("ix_documents_citations", table_name="documents", postgresql_using="gin")
    op.drop_table("documents")
    op.drop_table("snapshots")
    op.execute("DROP TABLE IF EXISTS runs")
    op.drop_index(op.f("ix_llm_calls_run_id"), table_name="llm_calls")
    op.drop_table("llm_calls")
    op.drop_index(op.f("ix_events_type"), table_name="events")
    op.drop_table("events")
    op.drop_table("embedding_models")
