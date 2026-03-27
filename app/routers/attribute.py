"""Attribution endpoint — apply classified data to dimensional scores.

POST /attribute — Bayesian update of dimensional sub-component values.
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
from ..models.psychographic_data import PsychographicData
from ..models.dimensional_score import DimensionalScore
from ..core.bayesian import update_sub_component, INITIAL_STD_ERROR
from ..services.llm_provider import get_llm_provider
from ..utils import parse_uuid

router = APIRouter(tags=["attribute"])
logger = logging.getLogger(__name__)

ATTRIBUTION_PROMPT = """You are a dimensional attribution engine. Given a piece of classified behavioral data,
infer what personality sub-component values are implied.

The 5 dimensions and their key sub-components:
- COGNITIVE: formality (0-100), verbosity (0-100), humor_frequency (0-100), reasoning_mode (enum: analytical/intuitive/mixed)
- EMOTIONAL: expressiveness (0-100), baseline_valence (-100 to +100), stress_threshold (0-100)
- SOCIAL: directness (0-100), trust_speed (0-100), conflict_style (enum: competing/collaborating/compromising/avoiding/accommodating)
- EVOLUTIONARY: learning_rate (0-1), change_resistance (0-100)
- VISUAL: speech_rate_wpm (int), vocabulary_level (0-100)

Return a JSON object with:
{
  "inferences": [
    {"dimension": "COGNITIVE", "sub_component": "formality", "value": 75, "confidence": 0.6},
    ...
  ]
}

Be conservative — a single data point should produce moderate confidence (0.3-0.6).
Only return valid JSON."""


class ClassifiedDataInput(BaseModel):
    psychographic_data_id: Optional[str] = None
    category: str = ""
    content_summary: str = ""
    confidence: float = 0.5
    source_reliability: float = 0.6


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


@router.post("/attribute", response_model=AttributeResponse)
async def attribute_data(req: AttributeRequest, db: AsyncSession = Depends(get_db)):
    """Apply classified data to dimensional scores via Bayesian update."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Load classified content
    content_summary = req.classified_data.content_summary
    if req.classified_data.psychographic_data_id:
        pd_uuid = parse_uuid(req.classified_data.psychographic_data_id, "psychographic_data_id")
        pd_result = await db.execute(
            select(PsychographicData).where(PsychographicData.id == pd_uuid)
        )
        pd = pd_result.scalar_one_or_none()
        if pd:
            content_summary = content_summary or pd.content[:500]
            # Mark as processed
            pd.processed = True
            pd.processed_at = datetime.now(timezone.utc)

    if not content_summary:
        raise HTTPException(status_code=422, detail="No content to attribute.")

    # Use LLM to infer dimensional sub-component values
    provider = get_llm_provider()
    inference_prompt = f"Category: {req.classified_data.category}\nContent: {content_summary}"
    inferences_raw = await provider.classify(inference_prompt, ATTRIBUTION_PROMPT)
    inferences = inferences_raw.get("inferences", [])

    updates = []
    source_reliability = req.classified_data.source_reliability

    for inf in inferences:
        dimension = inf.get("dimension", "").upper()
        sub_component = inf.get("sub_component", "")
        observed_value = inf.get("value")
        inference_confidence = inf.get("confidence", 0.4)

        if not dimension or not sub_component or observed_value is None:
            continue

        # Load or create dimensional score row
        score_result = await db.execute(
            select(DimensionalScore).where(
                DimensionalScore.twin_id == twin_uuid,
                DimensionalScore.dimension == dimension,
                DimensionalScore.sub_component == sub_component,
            )
        )
        score = score_result.scalar_one_or_none()

        if score:
            old_value = score.value
            old_confidence = score.confidence

            # Bayesian update
            new_value, new_std_error, new_confidence = update_sub_component(
                current_value=score.value,
                current_std_error=score.std_error,
                observation=float(observed_value),
                inference_confidence=inference_confidence,
                source_reliability=source_reliability,
            )
            score.value = round(new_value, 2)
            score.std_error = round(new_std_error, 4)
            score.confidence = round(new_confidence, 4)
            score.observation_count += 1
            score.last_updated = datetime.now(timezone.utc)
        else:
            old_value = 50.0
            old_confidence = 0.0
            new_value = float(observed_value)
            new_confidence = min(1.0, inference_confidence * source_reliability)

            score = DimensionalScore(
                twin_id=twin_uuid,
                dimension=dimension,
                sub_component=sub_component,
                value=round(new_value, 2),
                confidence=round(new_confidence, 4),
                std_error=round(INITIAL_STD_ERROR * (1 - new_confidence), 4),
                observation_count=1,
            )
            db.add(score)

        updates.append(SubComponentUpdate(
            dimension=dimension,
            sub_component=sub_component,
            old_value=round(old_value, 2),
            new_value=round(score.value, 2),
            confidence_delta=round(score.confidence - old_confidence, 4),
        ))

    # Update personality confidence on profile
    if updates:
        profile.personality_core_confidence = min(
            1.0, (profile.personality_core_confidence or 0.0) + 0.02 * len(updates)
        )

    await db.flush()

    return AttributeResponse(
        sub_components_updated=updates,
        personality_core_updated=False,  # Phase 2: trigger re-derivation
        personality_core_confidence=profile.personality_core_confidence or 0.0,
    )
