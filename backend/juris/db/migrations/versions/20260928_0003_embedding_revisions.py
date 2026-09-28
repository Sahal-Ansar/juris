"""Embedding model revision and per-snapshot embedding record (PLAN 3.8).

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-28 05:28:58.787253
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("embedding_models", sa.Column("revision", sa.Text(), nullable=True))
    op.add_column(
        "snapshots",
        sa.Column(
            "embeddings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("snapshots", "embeddings")
    op.drop_column("embedding_models", "revision")
