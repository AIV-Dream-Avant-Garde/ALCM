"""Shared schemas used across multiple endpoints."""
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class GuardrailPush(BaseModel):
    blocked_topics: List[str] = []
    restricted_topics: dict = {}
    language_restrictions: List[str] = []
    min_formality: int = Field(default=0, ge=0, le=100)
    max_controversy: int = Field(default=100, ge=0, le=100)
    humor_permitted: bool = True
    humor_blacklist: List[str] = []
    require_ai_disclosure: bool = True
    disclosure_text: str = "This is an AI-generated response."


class GuardrailResponse(BaseModel):
    confirmation: bool
    guardrail_version: int
    propagation_status: str


class ValidateRequest(BaseModel):
    twin_id: str
    sample_content: str = Field(min_length=1, max_length=50000)
    sample_context: str = ""


class ValidateResponse(BaseModel):
    consistency_score: Optional[float] = None
    passed: bool = False
    details: str = ""
    divergent_traits: List[str] = []
    recommendation: str = ""


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


class DriftResponse(BaseModel):
    drift_score: float
    threshold: float = 0.25
    threshold_status: str = "WITHIN_BOUNDS"
    per_dimension_drift: dict = {}
    baseline_set_at: Optional[str] = None
    last_checked: Optional[str] = None


class PackageResponse(BaseModel):
    twin_id: str
    version: int = 1
    seal_hash: Optional[str] = None
    generated_at: str
    modules: Dict[str, Any] = {}


class SnapshotResponse(BaseModel):
    snapshot_ref: str
    seal_hash: str
    version_number: int
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
