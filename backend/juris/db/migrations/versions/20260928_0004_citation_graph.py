"""Citation graph columns and parallel-citation aliases (PLAN 3.9).

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-28 07:37:22.823174
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FK_NAME = "citation_edges_target_section_id_fkey"  # Postgres's default name


def upgrade() -> None:
    op.create_table(
        "citation_aliases",
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("target_ref", sa.Text(), nullable=False),
        sa.Column("evidence", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("alias"),
    )
    op.create_index(
        op.f("ix_citation_aliases_target_ref"), "citation_aliases", ["target_ref"], unique=False
    )
    op.add_column("citation_edges", sa.Column("kind", sa.Text(), nullable=True))
    op.add_column("citation_edges", sa.Column("target_ref", sa.Text(), nullable=True))
    op.add_column("citation_edges", sa.Column("target_section_id", sa.Text(), nullable=True))
    op.add_column("citation_edges", sa.Column("resolution", sa.Text(), nullable=True))
    op.add_column("citation_edges", sa.Column("source_para_seq", sa.Integer(), nullable=True))
    op.add_column("citation_edges", sa.Column("treatment_cue", sa.Text(), nullable=True))
    op.add_column("citation_edges", sa.Column("context", sa.Text(), nullable=True))
    op.create_index(
        op.f("ix_citation_edges_target_ref"), "citation_edges", ["target_ref"], unique=False
    )
    op.create_index(
        op.f("ix_citation_edges_target_section_id"),
        "citation_edges",
        ["target_section_id"],
        unique=False,
    )
    op.create_foreign_key(
        FK_NAME,
        "citation_edges",
        "statute_sections",
        ["target_section_id"],
        ["section_id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(FK_NAME, "citation_edges", type_="foreignkey")
    op.drop_index(op.f("ix_citation_edges_target_section_id"), table_name="citation_edges")
    op.drop_index(op.f("ix_citation_edges_target_ref"), table_name="citation_edges")
    op.drop_column("citation_edges", "context")
    op.drop_column("citation_edges", "treatment_cue")
    op.drop_column("citation_edges", "source_para_seq")
    op.drop_column("citation_edges", "resolution")
    op.drop_column("citation_edges", "target_section_id")
    op.drop_column("citation_edges", "target_ref")
    op.drop_column("citation_edges", "kind")
    op.drop_index(op.f("ix_citation_aliases_target_ref"), table_name="citation_aliases")
    op.drop_table("citation_aliases")
