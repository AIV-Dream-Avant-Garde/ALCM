"""Temporal interaction history — episodic_memories table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, ForeignKey, Index, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from ..database import Base


class EpisodicMemory(Base):
    __tablename__ = "episodic_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    summary = Column(Text, nullable=False)
    topics = Column(ARRAY(String), default=list)
    contact_id = Column(String(255))
    emotional_valence = Column(String(20), default="NEUTRAL")  # POSITIVE | NEUTRAL | NEGATIVE
    outcome = Column(Text)

    # Scoping: TRAINING_AREA | deal UUID for per-deployment scoping
    deployment_scope = Column(String(100), default="TRAINING_AREA")

    # Retrieval: decays over time, reinforced on reference
    retrieval_weight = Column(Float, default=1.0)
    last_referenced = Column(DateTime(timezone=True))

    interaction_at = Column(DateTime(timezone=True), nullable=False)
    duration_seconds = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="episodic_memories")

    __table_args__ = (
        Index("idx_episodic_twin", "twin_id"),
        Index("idx_episodic_twin_scope", "twin_id", "deployment_scope"),
        Index("idx_episodic_twin_time", "twin_id", "interaction_at"),
    )
