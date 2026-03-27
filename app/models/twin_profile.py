"""Central identity record in the ALCM API — twin_profiles table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class TwinProfile(Base):
    __tablename__ = "twin_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic metadata (received from platform at creation)
    identity_category = Column(String(50), nullable=False, default="ENTERTAINMENT")
    clone_type = Column(String(50), nullable=False, default="PUBLIC_FIGURE")

    # Personality Core (derived, not input) — stored as JSONB snapshots
    big_five = Column(JSONB, default=dict)
    # {"openness": {"score": 72.0, "confidence": 0.68, "std_error": 4.2}, ...}
    mbti = Column(JSONB, default=dict)
    # {"type": "ENTJ", "dichotomies": {...}, "derivation_confidence": 0.71}
    cognitive_complexity = Column(JSONB, default=dict)
    # {"vocabulary_range": 82, "explanation_depth": 74, ...}

    personality_core_confidence = Column(Float, default=0.0)

    # Coverage tracking — JSONB for per-category scores
    psychographic_coverage = Column(JSONB, default=dict)
    # {"mind": 62, "heart": 45, "spirit": 38, ...} (0-100 per category)
    overall_coverage = Column(Float, default=0.0)

    # Fidelity
    cfs = Column(Float, default=0.0)
    cfs_last_computed = Column(DateTime(timezone=True))
    health_status = Column(String(50), default="BUILDING")

    # Guardrails (pushed from platform)
    guardrail_config = Column(JSONB, default=dict)
    guardrail_version = Column(Integer, default=0)

    # Drift baseline
    baseline_snapshot_ref = Column(String(255))
    baseline_set_at = Column(DateTime(timezone=True))
    drift_score = Column(Float, default=0.0)
    drift_last_checked = Column(DateTime(timezone=True))

    # Status lifecycle: INITIALIZING | BUILDING | ACTIVE | PROTECTED_HOLD | LOCKED | ARCHIVED
    status = Column(String(50), nullable=False, default="INITIALIZING")

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships (cascade delete all child records)
    psychographic_data = relationship("PsychographicData", back_populates="twin", cascade="all, delete-orphan")
    dimensional_scores = relationship("DimensionalScore", back_populates="twin", cascade="all, delete-orphan")
    personality_cores = relationship("PersonalityCore", back_populates="twin", cascade="all, delete-orphan")
    context_modulations = relationship("ContextModulation", back_populates="twin", cascade="all, delete-orphan")
    score_to_gen_instructions = relationship("ScoreToGenInstruction", back_populates="twin", cascade="all, delete-orphan")
    rag_entries = relationship("RagEntry", back_populates="twin", cascade="all, delete-orphan")
    voice_profile = relationship("VoiceProfile", back_populates="twin", uselist=False, cascade="all, delete-orphan")
    visual_profile = relationship("VisualProfile", back_populates="twin", uselist=False, cascade="all, delete-orphan")
    relationship_graphs = relationship("RelationshipGraph", back_populates="twin", cascade="all, delete-orphan")
    episodic_memories = relationship("EpisodicMemory", back_populates="twin", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="twin", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_twin_profiles_status", "status"),
    )
