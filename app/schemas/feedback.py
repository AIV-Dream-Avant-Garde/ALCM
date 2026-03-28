"""Feedback schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class FeedbackSignal(BaseModel):
    original_response: Optional[str] = None
    corrected_response: Optional[str] = None
    context: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback_type: str = Field(pattern="^(USER_CORRECTION|RATING|IMPLICIT_ACCEPT|REFINEMENT)$")
    signal: FeedbackSignal


class FeedbackResponse(BaseModel):
    processed: bool
    learning_applied: bool
    sub_components_affected: List[str] = []
    confidence_deltas: dict = {}
