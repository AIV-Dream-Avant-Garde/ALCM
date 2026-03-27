"""Visual identity data — visual_profiles table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class VisualProfile(Base):
    __tablename__ = "visual_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(
        UUID(as_uuid=True),
        ForeignKey("twin_profiles.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    appearance = Column(JSONB)
    # {"hair": {...}, "build": {...}, "distinguishing_features": [...]}
    expression_baselines = Column(JSONB)
    # {"neutral": {...}, "smile": {...}, "focused": {...}}
    gesture_patterns = Column(JSONB)
    mannerisms = Column(ARRAY(String))
    likeness_hash = Column(String(128))

    source_media_urls = Column(ARRAY(String))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="visual_profile")
