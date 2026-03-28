"""Dimensional attribution service.

Converts classified psychographic data into dimensional score updates
via LLM inference + Bayesian Normal-Normal updating.
"""
import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.twin_profile import TwinProfile
from app.models.psychographic_data import PsychographicData
from app.models.dimensional_score import DimensionalScore
from app.core.bayesian import update_sub_component, INITIAL_STD_ERROR
from app.core.errors import TwinNotFound, InsufficientData
from app.services.llm_provider import get_llm_provider
from app.schemas.attribute import AttributeRequest, AttributeResponse, SubComponentUpdate

logger = logging.getLogger(__name__)

ATTRIBUTION_PROMPT = """You are a dimensional attribution engine. Given classified behavioral data,
infer personality sub-component values.

Dimensions and key sub-components:
- COGNITIVE: formality (0-100), verbosity (0-100), humor_frequency (0-100), reasoning_mode (enum: analytical/intuitive/mixed)
- EMOTIONAL: expressiveness (0-100), baseline_valence (-100 to +100), stress_threshold (0-100)
- SOCIAL: directness (0-100), trust_speed (0-100), conflict_style (enum: competing/collaborating/compromising/avoiding/accommodating)
- EVOLUTIONARY: learning_rate (0-1), change_resistance (0-100)
- VISUAL: speech_rate_wpm (int), vocabulary_level (0-100)

Return JSON: {"inferences": [{"dimension": "COGNITIVE", "sub_component": "formality", "value": 75, "confidence": 0.6}]}
Be conservative — single data points produce moderate confidence (0.3-0.6). Only return valid JSON."""


async def attribute_data(req: AttributeRequest, db: AsyncSession) -> AttributeResponse:
    """Apply classified data to dimensional scores via Bayesian update."""
    twin_uuid = UUID(req.twin_id)

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(req.twin_id)

    # Load classified content
    content_summary = req.classified_data.content_summary
    if req.classified_data.psychographic_data_id:
        pd_uuid = UUID(req.classified_data.psychographic_data_id)
        pd_result = await db.execute(select(PsychographicData).where(PsychographicData.id == pd_uuid))
        pd = pd_result.scalar_one_or_none()
        if pd:
            content_summary = content_summary or pd.content[:500]
            pd.processed = True
            pd.processed_at = datetime.now(timezone.utc)

    if not content_summary:
        raise InsufficientData("No content to attribute.")

    # LLM inference
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
            new_value, new_std_error, new_confidence = update_sub_component(
                score.value, score.std_error, float(observed_value),
                inference_confidence, source_reliability,
            )
            score.value = round(new_value, 2)
            score.std_error = round(new_std_error, 4)
            score.confidence = round(new_confidence, 4)
            score.observation_count += 1
            score.last_updated = datetime.now(timezone.utc)
        else:
            old_value = 50.0
            old_confidence = 0.0
            new_confidence = min(1.0, inference_confidence * source_reliability)
            score = DimensionalScore(
                twin_id=twin_uuid, dimension=dimension, sub_component=sub_component,
                value=round(float(observed_value), 2), confidence=round(new_confidence, 4),
                std_error=round(INITIAL_STD_ERROR * (1 - new_confidence), 4), observation_count=1,
            )
            db.add(score)

        updates.append(SubComponentUpdate(
            dimension=dimension, sub_component=sub_component,
            old_value=round(old_value, 2), new_value=round(score.value, 2),
            confidence_delta=round(score.confidence - old_confidence, 4),
        ))

    # Re-derive Personality Core + recompute CFS
    personality_core_updated = False
    if updates:
        from app.services.personality_service import derive_personality_core
        from app.services.fidelity_service import compute_cfs

        core = await derive_personality_core(twin_uuid, db)
        personality_core_updated = True
        profile.personality_core_confidence = core.derivation_confidence
        profile.big_five = core.big_five
        profile.mbti = core.mbti
        profile.cognitive_complexity = core.cognitive_complexity
        profile.cfs = await compute_cfs(twin_uuid, db)
        profile.cfs_last_computed = datetime.now(timezone.utc)

    await db.flush()

    return AttributeResponse(
        sub_components_updated=updates,
        personality_core_updated=personality_core_updated,
        personality_core_confidence=profile.personality_core_confidence or 0.0,
    )
