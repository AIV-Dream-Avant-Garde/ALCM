"""Initial schema — all 12 ALCM tables.

Revision ID: 001
Revises: None
Create Date: 2026-03-27
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. twin_profiles
    op.create_table(
        "twin_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("identity_category", sa.String(50), nullable=False, server_default="ENTERTAINMENT"),
        sa.Column("clone_type", sa.String(50), nullable=False, server_default="PUBLIC_FIGURE"),
        sa.Column("big_five", postgresql.JSONB, server_default="{}"),
        sa.Column("mbti", postgresql.JSONB, server_default="{}"),
        sa.Column("cognitive_complexity", postgresql.JSONB, server_default="{}"),
        sa.Column("personality_core_confidence", sa.Float, server_default="0.0"),
        sa.Column("psychographic_coverage", postgresql.JSONB, server_default="{}"),
        sa.Column("overall_coverage", sa.Float, server_default="0.0"),
        sa.Column("cfs", sa.Float, server_default="0.0"),
        sa.Column("cfs_last_computed", sa.DateTime(timezone=True)),
        sa.Column("health_status", sa.String(50), server_default="BUILDING"),
        sa.Column("guardrail_config", postgresql.JSONB, server_default="{}"),
        sa.Column("guardrail_version", sa.Integer, server_default="0"),
        sa.Column("baseline_snapshot_ref", sa.String(255)),
        sa.Column("baseline_set_at", sa.DateTime(timezone=True)),
        sa.Column("drift_score", sa.Float, server_default="0.0"),
        sa.Column("drift_last_checked", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(50), nullable=False, server_default="INITIALIZING"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_twin_profiles_status", "twin_profiles", ["status"])

    # 2. psychographic_data
    op.create_table(
        "psychographic_data",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("sub_category", sa.String(100)),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64)),
        sa.Column("modality", sa.String(50), nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("source_reliability", sa.Float, nullable=False, server_default="0.6"),
        sa.Column("classification_confidence", sa.Float, nullable=False),
        sa.Column("contributor_id", postgresql.UUID(as_uuid=True)),
        sa.Column("contributor_type", sa.String(50)),
        sa.Column("approval_status", sa.String(50), server_default="AUTO_APPROVED"),
        sa.Column("processed", sa.Boolean, server_default="false"),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_psycho_twin", "psychographic_data", ["twin_id"])
    op.create_index("idx_psycho_category", "psychographic_data", ["twin_id", "category"])
    op.create_index("idx_psycho_unprocessed", "psychographic_data", ["twin_id"], postgresql_where=sa.text("processed = false"))

    # 3. dimensional_scores
    op.create_table(
        "dimensional_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dimension", sa.String(50), nullable=False),
        sa.Column("sub_component", sa.String(100), nullable=False),
        sa.Column("value", sa.Float, nullable=False, server_default="50.0"),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("std_error", sa.Float, nullable=False, server_default="30.0"),
        sa.Column("prior_mean", sa.Float, server_default="50.0"),
        sa.Column("prior_variance", sa.Float, server_default="900.0"),
        sa.Column("observation_count", sa.Integer, server_default="0"),
        sa.Column("distribution_type", sa.String(20), server_default="CONTINUOUS"),
        sa.Column("categorical_dist", postgresql.JSONB),
        sa.Column("context_overrides", postgresql.JSONB, server_default="{}"),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("twin_id", "dimension", "sub_component", name="uq_dims_twin_dim_sub"),
    )
    op.create_index("idx_dims_twin", "dimensional_scores", ["twin_id"])
    op.create_index("idx_dims_twin_dim", "dimensional_scores", ["twin_id", "dimension"])

    # 4. personality_cores
    op.create_table(
        "personality_cores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("big_five", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("mbti", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("cognitive_complexity", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("derivation_confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("derived_from_observations", sa.Integer, nullable=False, server_default="0"),
        sa.Column("is_current", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("twin_id", "version", name="uq_pc_twin_version"),
    )
    op.create_index("idx_pc_twin_current", "personality_cores", ["twin_id"], postgresql_where=sa.text("is_current = true"))

    # 5. context_modulation
    op.create_table(
        "context_modulation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("context_type", sa.String(50), nullable=False),
        sa.Column("sub_component", sa.String(100), nullable=False),
        sa.Column("override_value", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.0"),
        sa.Column("observation_count", sa.Integer, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("twin_id", "context_type", "sub_component", name="uq_ctx_twin_type_sub"),
    )
    op.create_index("idx_ctx_twin", "context_modulation", ["twin_id"])
    op.create_index("idx_ctx_twin_type", "context_modulation", ["twin_id", "context_type"])

    # 6. score_to_gen_instructions
    op.create_table(
        "score_to_gen_instructions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("layer", sa.Integer, nullable=False),
        sa.Column("context_type", sa.String(50), server_default="DEFAULT"),
        sa.Column("instruction_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("generated_from_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("is_current", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("twin_id", "layer", "context_type", name="uq_stg_twin_layer_ctx"),
    )
    op.create_index("idx_stg_twin_current", "score_to_gen_instructions", ["twin_id"], postgresql_where=sa.text("is_current = true"))

    # 7. rag_entries
    op.create_table(
        "rag_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", postgresql.JSONB),
        sa.Column("topic", sa.String(255)),
        sa.Column("category", sa.String(50)),
        sa.Column("source_type", sa.String(50)),
        sa.Column("source_ref", sa.Text),
        sa.Column("conviction", sa.Float, server_default="0.5"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_rag_twin", "rag_entries", ["twin_id"], postgresql_where=sa.text("is_active = true"))

    # 8. voice_profiles
    op.create_table(
        "voice_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("speech_rate", sa.Float),
        sa.Column("pitch_range", postgresql.JSONB),
        sa.Column("accent_markers", postgresql.ARRAY(sa.String)),
        sa.Column("filler_patterns", postgresql.JSONB),
        sa.Column("prosodic_data", postgresql.JSONB),
        sa.Column("tts_provider", sa.String(50), server_default="ELEVENLABS"),
        sa.Column("tts_voice_id", sa.String(255)),
        sa.Column("tts_model_id", sa.String(255)),
        sa.Column("source_audio_urls", postgresql.ARRAY(sa.String)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # 9. visual_profiles
    op.create_table(
        "visual_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("appearance", postgresql.JSONB),
        sa.Column("expression_baselines", postgresql.JSONB),
        sa.Column("gesture_patterns", postgresql.JSONB),
        sa.Column("mannerisms", postgresql.ARRAY(sa.String)),
        sa.Column("likeness_hash", sa.String(128)),
        sa.Column("source_media_urls", postgresql.ARRAY(sa.String)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )

    # 10. relationship_graphs
    op.create_table(
        "relationship_graphs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("contact_id", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(255)),
        sa.Column("relationship_type", sa.String(50)),
        sa.Column("inferred_traits", postgresql.JSONB, server_default="{}"),
        sa.Column("behavioral_overrides", postgresql.JSONB, server_default="{}"),
        sa.Column("interaction_count", sa.Integer, server_default="0"),
        sa.Column("last_interaction", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("twin_id", "contact_id", name="uq_rel_twin_contact"),
    )
    op.create_index("idx_rel_twin", "relationship_graphs", ["twin_id"])

    # 11. episodic_memories
    op.create_table(
        "episodic_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("topics", postgresql.ARRAY(sa.String), server_default="{}"),
        sa.Column("contact_id", sa.String(255)),
        sa.Column("emotional_valence", sa.String(20), server_default="NEUTRAL"),
        sa.Column("outcome", sa.Text),
        sa.Column("deployment_scope", sa.String(100), server_default="TRAINING_AREA"),
        sa.Column("retrieval_weight", sa.Float, server_default="1.0"),
        sa.Column("last_referenced", sa.DateTime(timezone=True)),
        sa.Column("interaction_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_seconds", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_episodic_twin", "episodic_memories", ["twin_id"])
    op.create_index("idx_episodic_twin_scope", "episodic_memories", ["twin_id", "deployment_scope"])
    op.create_index("idx_episodic_twin_time", "episodic_memories", ["twin_id", "interaction_at"])

    # 12. processing_jobs
    op.create_table(
        "processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("twin_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("input_data", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.String(50), server_default="QUEUED"),
        sa.Column("result", postgresql.JSONB),
        sa.Column("error", sa.Text),
        sa.Column("priority", sa.Integer, server_default="5"),
        sa.Column("attempts", sa.Integer, server_default="0"),
        sa.Column("max_attempts", sa.Integer, server_default="3"),
        sa.Column("queued_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_jobs_status", "processing_jobs", ["status"], postgresql_where=sa.text("status IN ('QUEUED', 'RETRY')"))
    op.create_index("idx_jobs_twin", "processing_jobs", ["twin_id"])


def downgrade() -> None:
    op.drop_table("processing_jobs")
    op.drop_table("episodic_memories")
    op.drop_table("relationship_graphs")
    op.drop_table("visual_profiles")
    op.drop_table("voice_profiles")
    op.drop_table("rag_entries")
    op.drop_table("score_to_gen_instructions")
    op.drop_table("context_modulation")
    op.drop_table("personality_cores")
    op.drop_table("dimensional_scores")
    op.drop_table("psychographic_data")
    op.drop_table("twin_profiles")
