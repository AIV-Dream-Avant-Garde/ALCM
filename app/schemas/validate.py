"""Personality validation schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List


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
