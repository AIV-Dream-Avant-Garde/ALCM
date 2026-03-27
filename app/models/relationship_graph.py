"""Per-contact behavioral adaptation — relationship_graphs table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class RelationshipGraph(Base):
    __tablename__ = "relationship_graphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)
    contact_id = Column(String(255), nullable=False)

    display_name = Column(String(255))
    # Relationship type: PROFESSIONAL | PERSONAL | FAMILY | PUBLIC | UNKNOWN
    relationship_type = Column(String(50))

    inferred_traits = Column(JSONB, default=dict)
    behavioral_overrides = Column(JSONB, default=dict)
    interaction_count = Column(Integer, default=0)
    last_interaction = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="relationship_graphs")

    __table_args__ = (
        UniqueConstraint("twin_id", "contact_id", name="uq_rel_twin_contact"),
        Index("idx_rel_twin", "twin_id"),
    )
