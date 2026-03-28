"""Guardrail configuration schemas."""
from pydantic import BaseModel, Field
from typing import List


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
