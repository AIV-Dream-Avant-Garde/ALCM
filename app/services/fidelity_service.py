"""Fidelity measurement service — CFS computation and health indicators.

See Developer Guide Section 18 (Fidelity Measurement Framework).
Phase 1: Simplified CFS from dimensional score confidences.
"""
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dimensional_score import DimensionalScore
from app.models.twin_profile import TwinProfile

# MVP dimension weights (spec Section 18.1)
DIMENSION_WEIGHTS = {
    "COGNITIVE": 0.30,
    "EMOTIONAL": 0.25,
    "SOCIAL": 0.25,
    "EVOLUTIONARY": 0.10,
    "VISUAL": 0.10,
}


async def compute_cfs(twin_id: UUID, db: AsyncSession) -> float:
    """Compute Composite Fidelity Score for a twin.

    Phase 1: Weighted average of per-dimension mean confidence.
    Phase 2: Full formula with user_rating and consistency scores.
    """
    result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == twin_id)
    )
    scores = result.scalars().all()

    if not scores:
        return 0.0

    # Group by dimension
    dim_confidences = {}
    for s in scores:
        dim_confidences.setdefault(s.dimension, []).append(s.confidence)

    # Compute weighted CFS
    cfs = 0.0
    for dim, weight in DIMENSION_WEIGHTS.items():
        confs = dim_confidences.get(dim, [])
        if confs:
            avg_confidence = sum(confs) / len(confs)
            cfs += weight * avg_confidence * 100  # Scale to 0-100
        # If no scores for this dimension, it contributes 0

    return round(min(100.0, cfs), 2)


async def compute_health_status(twin_id: UUID, db: AsyncSession) -> str:
    """Compute health status from CFS and coverage.

    Returns: BUILDING | HEALTHY | ATTENTION_NEEDED | ACTION_REQUIRED
    """
    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        return "BUILDING"

    cfs = await compute_cfs(twin_id, db)

    if cfs >= 65 and (profile.overall_coverage or 0) >= 50:
        return "HEALTHY"
    elif cfs >= 55:
        return "ATTENTION_NEEDED"
    elif cfs > 0:
        return "ATTENTION_NEEDED"
    else:
        return "BUILDING"
