"""Voice identity data — voice_profiles table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(
        UUID(as_uuid=True),
        ForeignKey("twin_profiles.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Voice characteristics (derived from professional audio uploads)
    speech_rate = Column(Float)  # words per minute
    pitch_range = Column(JSONB)  # {"low": 85, "high": 255}
    accent_markers = Column(ARRAY(String))
    filler_patterns = Column(JSONB)  # {"um": 0.02, "like": 0.01}
    prosodic_data = Column(JSONB)

    # TTS provider reference
    tts_provider = Column(String(50), default="ELEVENLABS")
    tts_voice_id = Column(String(255))
    tts_model_id = Column(String(255))

    # Source files
    source_audio_urls = Column(ARRAY(String))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="voice_profile")
