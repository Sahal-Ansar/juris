"""Legal-text normalisation for full-text search, and lexeme statistics (PLAN 4.1).

``chunks.tsv`` is rebuilt from ``juris_chunk_tsv(context_header, text)``:

- ``juris_legal_text`` rewrites citation shorthand before tokenising, so "s. 74", "Sec. 74",
  "u/s 74" and "Section 74" index the same way ("s" alone is an English stop word), and number
  ranges ("73-74") don't become a negative number ("-74").
- The case title (or the Act, section and heading) from ``context_header`` is indexed at
  weight B, the chunk text at weight A. The paragraph label ("¶ 12", "headnote") is dropped.
- The column is stored inline (``STORAGE MAIN``), so ranking reads heap pages instead of
  fetching every matching row's vector from TOAST.

``lexeme_stats`` holds each lexeme's chunk frequency (``ts_stat``) for IDF weights; the loader
refreshes it after loading chunks. The query side calls the same ``juris_legal_text``.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-28 12:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Plain groups only: "(?:" would read as a bind parameter in text().
LEGAL_TEXT = r"""
CREATE OR REPLACE FUNCTION juris_legal_text(t text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
RETURN regexp_replace(regexp_replace(regexp_replace(regexp_replace(t,
    '\mu/ss?\.?\s*(?=\d)', 'under section ', 'gi'),
    '\m(ss?\.|secs?\.?)\s*(?=\d)', 'section ', 'gi'),
    '\marts?\.\s*(?=\d)', 'article ', 'gi'),
    '(\d)-(?=\d)', '\1 ', 'g')
"""

CHUNK_TSV = r"""
CREATE OR REPLACE FUNCTION juris_chunk_tsv(header text, body text) RETURNS tsvector
LANGUAGE sql IMMUTABLE PARALLEL SAFE
RETURN setweight(to_tsvector('english'::regconfig, juris_legal_text(coalesce(
           regexp_replace(header, ' — (¶|paras |front matter|headnote)[^—]*$', ''), ''))), 'B')
       || setweight(to_tsvector('english'::regconfig, juris_legal_text(body)), 'A')
"""

REFRESH_STATS = (
    "INSERT INTO lexeme_stats (lexeme, ndoc) "
    "SELECT word, ndoc FROM ts_stat('SELECT tsv FROM chunks')"
)


def upgrade() -> None:
    op.execute(LEGAL_TEXT)
    op.execute(CHUNK_TSV)
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_column("chunks", "tsv")
    # one statement, so the rewrite that fills the column already uses STORAGE MAIN
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (juris_chunk_tsv(context_header, text)) STORED, "
        "ALTER COLUMN tsv SET STORAGE MAIN"
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_table(
        "lexeme_stats",
        sa.Column("lexeme", sa.Text(), nullable=False),
        sa.Column("ndoc", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("lexeme"),
    )
    op.execute(REFRESH_STATS)


def downgrade() -> None:
    op.drop_table("lexeme_stats")
    op.drop_index("ix_chunks_tsv", table_name="chunks")
    op.drop_column("chunks", "tsv")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english'::regconfig, text)) STORED"
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.execute("DROP FUNCTION juris_chunk_tsv(text, text)")
    op.execute("DROP FUNCTION juris_legal_text(text)")
