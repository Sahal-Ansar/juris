"""The pg_prewarm extension, to load a model's HNSW index and vectors into memory (PLAN 4.2).

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-28 15:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_prewarm")
