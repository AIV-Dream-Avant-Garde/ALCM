"""Twin lifecycle schemas."""
from pydantic import BaseModel, Field
from typing import Optional


class TwinCreateRequest(BaseModel):
    identity_category: str = Field(default="ENTERTAINMENT", pattern="^(ENTERTAINMENT|SPORTS|CORPORATE|EDUCATION|CREATOR_ECONOMY|BRAND_PERSONA|LEGACY_ESTATE|GAMING_VIRTUAL)$")
    clone_type: str = Field(default="PUBLIC_FIGURE", pattern="^(PUBLIC_FIGURE|FICTIONAL_CHARACTER)$")


class TwinCreateResponse(BaseModel):
    alcm_twin_id: str
    status: str
    created_at: str


class TwinDeleteResponse(BaseModel):
    deleted: bool
    twin_id: str


class PersonalityCoreResponse(BaseModel):
    big_five: dict = {}
    mbti: dict = {}
    ccp: dict = {}
    overall_confidence: float = 0.0


class TwinHealthResponse(BaseModel):
    twin_id: str
    cfs: float
    health_status: str
    personality_core: PersonalityCoreResponse = PersonalityCoreResponse()
    per_dimension_fidelity: dict = {}
    coverage: dict = {}
    last_training_activity: Optional[str] = None
    last_cfs_computation: Optional[str] = None
