"""Add fear and need to RAG entry category CHECK constraint.

Revision ID: 003
Revises: 002
Create Date: 2026-03-27
"""
from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add CHECK constraint for RAG category (8 values including fear + need)
    op.create_check_constraint(
        "ck_rag_category", "rag_entries",
        "category IS NULL OR category IN ('position','expertise','opinion','preference','fact','anecdote','fear','need')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_rag_category", "rag_entries", type_="check")
