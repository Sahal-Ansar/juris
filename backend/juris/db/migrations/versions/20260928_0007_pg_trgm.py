"""The pg_trgm extension, for fuzzy case-title lookup (PLAN 4.5).

No index: the documents table is small (2,298 rows on the MVP slice), so a scan is fast.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-28 21:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
