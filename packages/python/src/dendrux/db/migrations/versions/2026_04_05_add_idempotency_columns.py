"""Add idempotency_key and idempotency_fingerprint columns to agent_runs.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-05
"""

from __future__ import annotations

from typing import Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, tuple[str, ...], None] = None
depends_on: Union[str, tuple[str, ...], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("idempotency_key", sa.String(255), nullable=True),
    )
    op.add_column(
        "agent_runs",
        sa.Column("idempotency_fingerprint", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_agent_runs_idempotency_key",
        "agent_runs",
        ["idempotency_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_idempotency_key", table_name="agent_runs")
    op.drop_column("agent_runs", "idempotency_fingerprint")
    op.drop_column("agent_runs", "idempotency_key")
