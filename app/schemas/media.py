"""Media analysis and speech schemas."""
from pydantic import BaseModel, Field


class AnalyzeMediaRequest(BaseModel):
    twin_id: str
    media_url: str = Field(min_length=1)
    media_type: str = Field(pattern="^(VIDEO|AUDIO|IMAGE)$")


class AnalyzeMediaResponse(BaseModel):
    processing_id: str
    status: str = "QUEUED"
    estimated_duration_seconds: int = 120


class SpeechRequest(BaseModel):
    twin_id: str
    text: str = Field(min_length=1, max_length=5000)
