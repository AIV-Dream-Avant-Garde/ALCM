"""RAG Knowledge Base entries — rag_entries table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Boolean, DateTime, ForeignKey, Index, text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base

# Note: pgvector Vector type used only when pgvector extension is available.
# Embedding column is currently JSONB (list of floats).
# Migration to pgvector Vector(1536) for similarity search is a future enhancement.


class RagEntry(Base):
    __tablename__ = "rag_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)
    embedding = Column(JSONB)  # JSONB list of floats; pgvector Vector(1536) is a future enhancement

    topic = Column(String(255))
    category = Column(String(50))  # position | expertise | opinion | preference | fact | anecdote | fear | need
    source_type = Column(String(50))
    source_ref = Column(Text)

    conviction = Column(Float, default=0.5)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="rag_entries")

    __table_args__ = (
        Index("idx_rag_twin", "twin_id", postgresql_where=text("is_active = true")),
        CheckConstraint(
            "category IS NULL OR category IN ('position','expertise','opinion','preference','fact','anecdote','fear','need')",
            name="ck_rag_category",
        ),
    )
