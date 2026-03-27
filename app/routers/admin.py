"""Admin endpoints — developer/admin access to internal data.

Phase 1: Read-only profile access for debugging.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.personality_core import PersonalityCore
from ..models.dimensional_score import DimensionalScore
from ..models.psychographic_data import PsychographicData
from ..utils import parse_uuid

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/twin/{twin_id}/profile")
async def get_full_profile(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Full clone profile including all internal metrics. Admin only."""
    tid = parse_uuid(twin_id, "twin_id")

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == tid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Personality Core
    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == tid,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()

    # Dimensional Scores
    dim_result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == tid)
    )
    scores = dim_result.scalars().all()

    # Psychographic data count
    psycho_result = await db.execute(
        select(PsychographicData).where(PsychographicData.twin_id == tid)
    )
    psycho_data = psycho_result.scalars().all()

    return {
        "twin_id": str(profile.id),
        "status": profile.status,
        "identity_category": profile.identity_category,
        "clone_type": profile.clone_type,
        "personality_core": {
            "big_five": core.big_five if core else {},
            "mbti": core.mbti if core else {},
            "cognitive_complexity": core.cognitive_complexity if core else {},
            "derivation_confidence": core.derivation_confidence if core else 0.0,
            "version": core.version if core else 0,
        },
        "dimensional_scores": [
            {
                "dimension": s.dimension,
                "sub_component": s.sub_component,
                "value": s.value,
                "confidence": s.confidence,
                "std_error": s.std_error,
                "observation_count": s.observation_count,
            }
            for s in scores
        ],
        "psychographic_data_count": len(psycho_data),
        "psychographic_coverage": profile.psychographic_coverage,
        "overall_coverage": profile.overall_coverage,
        "cfs": profile.cfs,
        "health_status": profile.health_status,
        "guardrail_config": profile.guardrail_config,
        "guardrail_version": profile.guardrail_version,
        "drift_score": profile.drift_score,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }
