"""Admin endpoints — developer/admin access to internal data per spec Section 16.3."""
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.personality_core import PersonalityCore
from ..models.dimensional_score import DimensionalScore
from ..models.psychographic_data import PsychographicData
from ..core.errors import TwinNotFound
from ..utils import parse_uuid

router = APIRouter(prefix="/admin", tags=["admin"])


async def _load_twin(twin_id: str, db: AsyncSession) -> TwinProfile:
    tid = parse_uuid(twin_id, "twin_id")
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == tid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(twin_id)
    return profile


@router.get("/twin/{twin_id}/profile")
async def get_full_profile(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Full clone profile including all internal metrics."""
    profile = await _load_twin(twin_id, db)
    tid = profile.id

    core_result = await db.execute(
        select(PersonalityCore).where(PersonalityCore.twin_id == tid, PersonalityCore.is_current == True)
    )
    core = core_result.scalar_one_or_none()

    dim_result = await db.execute(select(DimensionalScore).where(DimensionalScore.twin_id == tid))
    scores = dim_result.scalars().all()

    psycho_count = await db.execute(
        select(func.count(PsychographicData.id)).where(PsychographicData.twin_id == tid)
    )

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
            {"dimension": s.dimension, "sub_component": s.sub_component, "value": s.value,
             "confidence": s.confidence, "std_error": s.std_error, "observation_count": s.observation_count}
            for s in scores
        ],
        "psychographic_data_count": psycho_count.scalar() or 0,
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


@router.get("/twin/{twin_id}/personality-core")
async def get_personality_core(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Direct access to derived Big Five, MBTI, CCP, and derivation confidence."""
    profile = await _load_twin(twin_id, db)
    core_result = await db.execute(
        select(PersonalityCore).where(PersonalityCore.twin_id == profile.id, PersonalityCore.is_current == True)
    )
    core = core_result.scalar_one_or_none()
    if not core:
        return {"twin_id": twin_id, "personality_core": None, "message": "No personality core derived yet."}

    return {
        "twin_id": twin_id,
        "version": core.version,
        "big_five": core.big_five,
        "mbti": core.mbti,
        "cognitive_complexity": core.cognitive_complexity,
        "derivation_confidence": core.derivation_confidence,
        "derived_from_observations": core.derived_from_observations,
        "is_current": core.is_current,
        "created_at": core.created_at.isoformat() if core.created_at else None,
    }


@router.get("/twin/{twin_id}/psychographics")
async def get_psychographics(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Category-level coverage, data points, and gaps."""
    profile = await _load_twin(twin_id, db)
    tid = profile.id

    # Count per category
    psycho_result = await db.execute(
        select(PsychographicData.category, func.count(PsychographicData.id))
        .where(PsychographicData.twin_id == tid)
        .group_by(PsychographicData.category)
    )
    category_counts = {row[0]: row[1] for row in psycho_result.fetchall()}

    all_categories = [
        "MIND", "HEART", "SPIRIT", "PHYSICALITY", "EXPERIENCES",
        "RELATIONSHIPS", "SURROUNDINGS", "WORK", "ETHICS", "FUTURE", "INTERESTS_TASTES",
    ]
    coverage = profile.psychographic_coverage or {}

    categories = []
    for cat in all_categories:
        categories.append({
            "category": cat,
            "data_points": category_counts.get(cat, 0),
            "coverage_score": coverage.get(cat.lower(), 0),
            "status": "covered" if coverage.get(cat.lower(), 0) > 30 else "gap",
        })

    return {
        "twin_id": twin_id,
        "overall_coverage": profile.overall_coverage or 0.0,
        "categories": categories,
        "gaps": [c["category"] for c in categories if c["status"] == "gap"],
    }


@router.get("/twin/{twin_id}/dimensions/raw")
async def get_raw_dimensions(twin_id: str, db: AsyncSession = Depends(get_db)):
    """All dimensional sub-components with full Bayesian state."""
    profile = await _load_twin(twin_id, db)
    dim_result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == profile.id)
    )
    scores = dim_result.scalars().all()

    return {
        "twin_id": twin_id,
        "total_sub_components": len(scores),
        "dimensions": [
            {
                "dimension": s.dimension,
                "sub_component": s.sub_component,
                "value": s.value,
                "confidence": s.confidence,
                "std_error": s.std_error,
                "prior_mean": s.prior_mean,
                "prior_variance": s.prior_variance,
                "observation_count": s.observation_count,
                "distribution_type": s.distribution_type,
                "categorical_dist": s.categorical_dist,
                "context_overrides": s.context_overrides,
                "last_updated": s.last_updated.isoformat() if s.last_updated else None,
            }
            for s in scores
        ],
    }
