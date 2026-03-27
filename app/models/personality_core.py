"""Personality Core derivation history — personality_cores table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, UniqueConstraint, Index, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class PersonalityCore(Base):
    __tablename__ = "personality_cores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)
    version = Column(Integer, nullable=False, default=1)

    big_five = Column(JSONB, nullable=False, default=dict)
    # {"openness": {"score": 72, "confidence": 0.68, "std_error": 4.2}, ...}
    mbti = Column(JSONB, nullable=False, default=dict)
    # {"type": "ENTJ", "dichotomies": {...}, "derivation_confidence": 0.71}
    cognitive_complexity = Column(JSONB, nullable=False, default=dict)
    # {"vocabulary_range": 82, "explanation_depth": 74, ...}

    derivation_confidence = Column(Float, nullable=False, default=0.0)
    derived_from_observations = Column(Integer, nullable=False, default=0)

    is_current = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="personality_cores")

    __table_args__ = (
        UniqueConstraint("twin_id", "version", name="uq_pc_twin_version"),
        Index("idx_pc_twin_current", "twin_id", postgresql_where=text("is_current = true")),
    )
