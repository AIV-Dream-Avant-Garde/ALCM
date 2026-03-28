"""Add CHECK constraints for enum-like string columns and fix index direction.

Revision ID: 002
Revises: 001
Create Date: 2026-03-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # twin_profiles constraints
    op.create_check_constraint(
        "ck_twin_status", "twin_profiles",
        "status IN ('INITIALIZING','BUILDING','ACTIVE','PROTECTED_HOLD','LOCKED','ARCHIVED')",
    )
    op.create_check_constraint(
        "ck_twin_health_status", "twin_profiles",
        "health_status IN ('BUILDING','HEALTHY','ATTENTION_NEEDED','ACTION_REQUIRED')",
    )
    op.create_check_constraint(
        "ck_twin_clone_type", "twin_profiles",
        "clone_type IN ('PUBLIC_FIGURE','FICTIONAL_CHARACTER')",
    )

    # psychographic_data constraints
    op.create_check_constraint(
        "ck_psycho_category", "psychographic_data",
        "category IN ('MIND','HEART','SPIRIT','PHYSICALITY','EXPERIENCES',"
        "'RELATIONSHIPS','SURROUNDINGS','WORK','ETHICS','FUTURE','INTERESTS_TASTES')",
    )
    op.create_check_constraint(
        "ck_psycho_modality", "psychographic_data",
        "modality IN ('TEXT','AUDIO','VIDEO','URL','STRUCTURED_DATA')",
    )
    op.create_check_constraint(
        "ck_psycho_source_type", "psychographic_data",
        "source_type IN ('SCRAPED_PUBLIC','PROFESSIONAL_UPLOAD','TRAINING_AREA',"
        "'IN_PLATFORM','CROSS_PLATFORM','FEEDBACK')",
    )
    op.create_check_constraint(
        "ck_psycho_approval_status", "psychographic_data",
        "approval_status IN ('AUTO_APPROVED','PENDING','APPROVED','REJECTED')",
    )

    # dimensional_scores constraints
    op.create_check_constraint(
        "ck_dims_dimension", "dimensional_scores",
        "dimension IN ('COGNITIVE','EMOTIONAL','SOCIAL','EVOLUTIONARY','VISUAL')",
    )
    op.create_check_constraint(
        "ck_dims_distribution_type", "dimensional_scores",
        "distribution_type IN ('CONTINUOUS','CATEGORICAL')",
    )

    # context_modulation constraint
    op.create_check_constraint(
        "ck_ctx_context_type", "context_modulation",
        "context_type IN ('PROFESSIONAL','CASUAL','INTIMATE','FORMAL','CONFLICT','CREATIVE','PUBLIC')",
    )

    # episodic_memories constraint
    op.create_check_constraint(
        "ck_episodic_valence", "episodic_memories",
        "emotional_valence IN ('POSITIVE','NEUTRAL','NEGATIVE')",
    )

    # processing_jobs constraint
    op.create_check_constraint(
        "ck_jobs_status", "processing_jobs",
        "status IN ('QUEUED','PROCESSING','COMPLETED','FAILED','RETRY')",
    )

    # Fix episodic memory index direction (ASC → DESC on interaction_at)
    op.drop_index("idx_episodic_twin_time", table_name="episodic_memories")
    op.create_index(
        "idx_episodic_twin_time", "episodic_memories",
        ["twin_id", sa.text("interaction_at DESC")],
    )


def downgrade() -> None:
    # Revert index
    op.drop_index("idx_episodic_twin_time", table_name="episodic_memories")
    op.create_index("idx_episodic_twin_time", "episodic_memories", ["twin_id", "interaction_at"])

    # Drop all CHECK constraints
    op.drop_constraint("ck_jobs_status", "processing_jobs", type_="check")
    op.drop_constraint("ck_episodic_valence", "episodic_memories", type_="check")
    op.drop_constraint("ck_ctx_context_type", "context_modulation", type_="check")
    op.drop_constraint("ck_dims_distribution_type", "dimensional_scores", type_="check")
    op.drop_constraint("ck_dims_dimension", "dimensional_scores", type_="check")
    op.drop_constraint("ck_psycho_approval_status", "psychographic_data", type_="check")
    op.drop_constraint("ck_psycho_source_type", "psychographic_data", type_="check")
    op.drop_constraint("ck_psycho_modality", "psychographic_data", type_="check")
    op.drop_constraint("ck_psycho_category", "psychographic_data", type_="check")
    op.drop_constraint("ck_twin_clone_type", "twin_profiles", type_="check")
    op.drop_constraint("ck_twin_health_status", "twin_profiles", type_="check")
    op.drop_constraint("ck_twin_status", "twin_profiles", type_="check")
