"""Feedback endpoint — receive interaction feedback and apply learning.

POST /twin/{id}/feedback — stores feedback, creates episodic memory,
applies basic Bayesian update for corrections.
"""
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.episodic_memory import EpisodicMemory
from ..models.dimensional_score import DimensionalScore
from ..core.bayesian import update_sub_component
from ..services.llm_provider import get_llm_provider
from ..utils import parse_uuid

router = APIRouter(tags=["feedback"])
logger = logging.getLogger(__name__)

CREDIT_ASSIGNMENT_PROMPT = """You are a credit assignment engine. Given feedback about a twin's response,
identify which personality sub-components were involved and need adjustment.

Sub-components to consider:
- formality (0-100): how formal vs casual
- verbosity (0-100): how detailed vs concise
- directness (0-100): how direct vs indirect
- expressiveness (0-100): how emotionally expressive
- humor_frequency (0-100): how often humor is used
- risk_tolerance (0-100): how bold vs cautious

Return JSON:
{
  "affected_components": [
    {"dimension": "COGNITIVE", "sub_component": "formality", "suggested_value": 45, "confidence": 0.5}
  ]
}
Only return valid JSON."""


class FeedbackSignal(BaseModel):
    original_response: Optional[str] = None
    corrected_response: Optional[str] = None
    context: Optional[str] = None
    rating: Optional[int] = None  # 1-5


class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback_type: str  # USER_CORRECTION | RATING | IMPLICIT_ACCEPT | REFINEMENT
    signal: FeedbackSignal


class FeedbackResponse(BaseModel):
    processed: bool
    learning_applied: bool
    sub_components_affected: List[str] = []
    confidence_deltas: dict = {}


@router.post("/twin/{twin_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    twin_id: str, req: FeedbackRequest, db: AsyncSession = Depends(get_db),
):
    """Process interaction feedback for adaptive learning."""
    twin_uuid = parse_uuid(twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Create episodic memory from the interaction
    summary = req.signal.context or f"Feedback ({req.feedback_type}) on interaction {req.interaction_id}"
    if req.signal.corrected_response:
        summary = f"Correction: {req.signal.corrected_response[:200]}"

    valence = "NEUTRAL"
    if req.feedback_type == "USER_CORRECTION":
        valence = "NEGATIVE"
    elif req.signal.rating and req.signal.rating >= 4:
        valence = "POSITIVE"
    elif req.signal.rating and req.signal.rating <= 2:
        valence = "NEGATIVE"

    memory = EpisodicMemory(
        twin_id=twin_uuid,
        summary=summary,
        topics=[],
        emotional_valence=valence,
        outcome=req.feedback_type,
        deployment_scope="TRAINING_AREA",
        interaction_at=datetime.now(timezone.utc),
    )
    db.add(memory)

    sub_components_affected = []
    confidence_deltas = {}
    learning_applied = False

    # For USER_CORRECTION: identify affected sub-components and apply Bayesian update
    if req.feedback_type == "USER_CORRECTION" and req.signal.corrected_response:
        provider = get_llm_provider()
        credit_input = (
            f"Original: {req.signal.original_response or 'N/A'}\n"
            f"Correction: {req.signal.corrected_response}\n"
            f"Context: {req.signal.context or 'N/A'}"
        )
        credit_result = await provider.classify(credit_input, CREDIT_ASSIGNMENT_PROMPT)
        affected = credit_result.get("affected_components", [])

        for comp in affected:
            dimension = comp.get("dimension", "").upper()
            sub_component = comp.get("sub_component", "")
            suggested_value = comp.get("suggested_value")
            inf_confidence = comp.get("confidence", 0.4)

            if not dimension or not sub_component or suggested_value is None:
                continue

            score_result = await db.execute(
                select(DimensionalScore).where(
                    DimensionalScore.twin_id == twin_uuid,
                    DimensionalScore.dimension == dimension,
                    DimensionalScore.sub_component == sub_component,
                )
            )
            score = score_result.scalar_one_or_none()

            if score:
                old_conf = score.confidence
                new_value, new_std_error, new_confidence = update_sub_component(
                    current_value=score.value,
                    current_std_error=score.std_error,
                    observation=float(suggested_value),
                    inference_confidence=inf_confidence,
                    source_reliability=1.0,  # Direct user feedback = highest reliability
                )
                score.value = round(new_value, 2)
                score.std_error = round(new_std_error, 4)
                score.confidence = round(new_confidence, 4)
                score.observation_count += 1
                score.last_updated = datetime.now(timezone.utc)

                sub_components_affected.append(sub_component)
                confidence_deltas[sub_component] = round(new_confidence - old_conf, 4)
                learning_applied = True

    # For RATING: boost or decay confidence
    elif req.feedback_type == "RATING" and req.signal.rating:
        # Positive ratings increase overall confidence slightly
        if req.signal.rating >= 4:
            profile.personality_core_confidence = min(
                1.0, (profile.personality_core_confidence or 0.0) + 0.01
            )
            learning_applied = True

    await db.flush()

    logger.info(
        f"Feedback processed for twin {twin_id}: type={req.feedback_type}, "
        f"affected={sub_components_affected}, learning={learning_applied}"
    )

    return FeedbackResponse(
        processed=True,
        learning_applied=learning_applied,
        sub_components_affected=sub_components_affected,
        confidence_deltas=confidence_deltas,
    )
