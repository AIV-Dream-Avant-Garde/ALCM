"""Context-specific value overrides — context_modulation table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from ..database import Base


class ContextModulation(Base):
    __tablename__ = "context_modulation"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    # Context type: PROFESSIONAL | CASUAL | INTIMATE | FORMAL | CONFLICT | CREATIVE | PUBLIC
    context_type = Column(String(50), nullable=False)
    sub_component = Column(String(100), nullable=False)

    override_value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False, default=0.0)
    observation_count = Column(Integer, default=0)

    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="context_modulations")

    __table_args__ = (
        UniqueConstraint("twin_id", "context_type", "sub_component", name="uq_ctx_twin_type_sub"),
        Index("idx_ctx_twin", "twin_id"),
        Index("idx_ctx_twin_type", "twin_id", "context_type"),
    )
