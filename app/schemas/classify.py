"""Classification schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class ClassifyRequest(BaseModel):
    twin_id: str
    content: str = Field(min_length=1, max_length=50000)
    modality: str = Field(default="TEXT", pattern="^(TEXT|AUDIO|VIDEO|URL|STRUCTURED_DATA)$")
    source_reliability: float = Field(default=0.6, ge=0.0, le=1.0)
    contributor_id: Optional[str] = None
    contributor_type: Optional[str] = Field(default=None, pattern="^(TALENT|TEAM_MEMBER|AIV_INTERNAL)$")


class CategoryAffected(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    processing_id: str
    categories_affected: List[CategoryAffected] = []
    sub_categories: List[str] = []
    psychographic_data_id: str
