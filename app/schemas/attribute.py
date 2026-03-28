"""Attribution schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class ClassifiedDataInput(BaseModel):
    psychographic_data_id: Optional[str] = None
    category: str = ""
    content_summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.6, ge=0.0, le=1.0)


class AttributeRequest(BaseModel):
    twin_id: str
    classified_data: ClassifiedDataInput


class SubComponentUpdate(BaseModel):
    dimension: str
    sub_component: str
    old_value: float
    new_value: float
    confidence_delta: float


class AttributeResponse(BaseModel):
    sub_components_updated: List[SubComponentUpdate] = []
    personality_core_updated: bool = False
    personality_core_confidence: float = 0.0
