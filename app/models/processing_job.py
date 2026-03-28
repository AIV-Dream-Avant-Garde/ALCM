"""Async processing job queue — processing_jobs table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Index, text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    # Job type: CLASSIFY | ATTRIBUTE | DERIVE_PERSONALITY | COMPUTE_CFS
    #   | GENERATE_INSTRUCTIONS | ANALYZE_MEDIA | SCRAPE_CONTENT
    #   | COMPUTE_DRIFT | SUMMARIZE_EPISODE
    job_type = Column(String(50), nullable=False)
    input_data = Column(JSONB, nullable=False, default=dict)

    # Status: QUEUED | PROCESSING | COMPLETED | FAILED | RETRY
    status = Column(String(50), default="QUEUED")
    result = Column(JSONB)
    error = Column(Text)

    priority = Column(Integer, default=5)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    queued_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    # Relationship
    twin = relationship("TwinProfile", back_populates="processing_jobs")

    __table_args__ = (
        Index("idx_jobs_status", "status", postgresql_where=text("status IN ('QUEUED', 'RETRY')")),
        Index("idx_jobs_twin", "twin_id"),
        CheckConstraint(
            "status IN ('QUEUED','PROCESSING','COMPLETED','FAILED','RETRY')",
            name="ck_jobs_status",
        ),
    )
