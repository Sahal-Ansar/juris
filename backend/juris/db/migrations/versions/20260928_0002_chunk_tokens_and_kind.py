"""Chunk token count and kind (PLAN 3.7).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28 04:28:43.198731
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("tokens", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("kind", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "kind")
    op.drop_column("chunks", "tokens")
