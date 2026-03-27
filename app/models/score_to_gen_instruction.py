"""Cached prompt layers for Score-to-Generation — score_to_gen_instructions table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, UniqueConstraint, Index, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from ..database import Base


class ScoreToGenInstruction(Base):
    __tablename__ = "score_to_gen_instructions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    # Layer: 1 (System Identity) | 2 (Behavioral Parameters) | 3 (Mood/State) | 5 (Discourse)
    # Layers 4 (RAG), 6 (History), 7 (Input) are assembled dynamically
    layer = Column(Integer, nullable=False)
    context_type = Column(String(50), default="DEFAULT")

    instruction_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)

    generated_from_version = Column(Integer, nullable=False, default=1)
    is_current = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="score_to_gen_instructions")

    __table_args__ = (
        UniqueConstraint("twin_id", "layer", "context_type", name="uq_stg_twin_layer_ctx"),
        Index("idx_stg_twin_current", "twin_id", postgresql_where=text("is_current = true")),
    )
