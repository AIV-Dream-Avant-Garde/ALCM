"""Classified psychographic data — psychographic_data table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Boolean, DateTime, ForeignKey, Index, text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class PsychographicData(Base):
    __tablename__ = "psychographic_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    # Category: MIND | HEART | SPIRIT | PHYSICALITY | EXPERIENCES
    #   | RELATIONSHIPS | SURROUNDINGS | WORK | ETHICS | FUTURE | INTERESTS_TASTES
    category = Column(String(50), nullable=False)
    sub_category = Column(String(100))

    content = Column(Text, nullable=False)
    content_hash = Column(String(64))
    modality = Column(String(50), nullable=False)  # TEXT | AUDIO | VIDEO | URL | STRUCTURED_DATA
    source_type = Column(String(50), nullable=False)  # SCRAPED_PUBLIC | PROFESSIONAL_UPLOAD | TRAINING_AREA | IN_PLATFORM | CROSS_PLATFORM | FEEDBACK
    source_reliability = Column(Float, nullable=False, default=0.6)

    classification_confidence = Column(Float, nullable=False)

    contributor_id = Column(UUID(as_uuid=True))
    contributor_type = Column(String(50))  # TALENT | TEAM_MEMBER | AIV_INTERNAL

    # Approval: AUTO_APPROVED | PENDING | APPROVED | REJECTED
    approval_status = Column(String(50), default="AUTO_APPROVED")

    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime(timezone=True))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="psychographic_data")

    __table_args__ = (
        Index("idx_psycho_twin", "twin_id"),
        Index("idx_psycho_category", "twin_id", "category"),
        Index("idx_psycho_unprocessed", "twin_id", postgresql_where=text("processed = false")),
        CheckConstraint(
            "category IN ('MIND','HEART','SPIRIT','PHYSICALITY','EXPERIENCES',"
            "'RELATIONSHIPS','SURROUNDINGS','WORK','ETHICS','FUTURE','INTERESTS_TASTES')",
            name="ck_psycho_category",
        ),
        CheckConstraint(
            "modality IN ('TEXT','AUDIO','VIDEO','URL','STRUCTURED_DATA')",
            name="ck_psycho_modality",
        ),
        CheckConstraint(
            "source_type IN ('SCRAPED_PUBLIC','PROFESSIONAL_UPLOAD','TRAINING_AREA',"
            "'IN_PLATFORM','CROSS_PLATFORM','FEEDBACK')",
            name="ck_psycho_source_type",
        ),
        CheckConstraint(
            "approval_status IN ('AUTO_APPROVED','PENDING','APPROVED','REJECTED')",
            name="ck_psycho_approval_status",
        ),
    )
