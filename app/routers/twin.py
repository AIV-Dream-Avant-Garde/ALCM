"""Twin lifecycle endpoints — POST /twin, DELETE /twin/{id}, GET /twin/{id}/health."""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.personality_core import PersonalityCore
from ..models.dimensional_score import DimensionalScore
from ..core.errors import TwinNotFound
from ..schemas.twin import (
    TwinCreateRequest, TwinCreateResponse, TwinDeleteResponse,
    TwinHealthResponse, PersonalityCoreResponse,
)
from ..utils import parse_uuid

router = APIRouter(prefix="/twin", tags=["twin"])


@router.post("", response_model=TwinCreateResponse, status_code=201)
async def create_twin(
    req: TwinCreateRequest = TwinCreateRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Create a new identity record in the ALCM."""
    profile = TwinProfile(
        identity_category=req.identity_category,
        clone_type=req.clone_type,
        status="INITIALIZING",
    )
    db.add(profile)
    await db.flush()

    core = PersonalityCore(
        twin_id=profile.id, version=1, big_five={}, mbti={},
        cognitive_complexity={}, derivation_confidence=0.0,
        derived_from_observations=0, is_current=True,
    )
    db.add(core)
    await db.flush()

    return TwinCreateResponse(
        alcm_twin_id=str(profile.id),
        status=profile.status,
        created_at=profile.created_at.isoformat(),
    )


@router.delete("/{twin_id}", response_model=TwinDeleteResponse)
async def delete_twin(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an identity record and all associated data."""
    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == parse_uuid(twin_id, "twin_id"))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(twin_id)
    await db.delete(profile)
    return TwinDeleteResponse(deleted=True, twin_id=twin_id)


@router.get("/{twin_id}/health", response_model=TwinHealthResponse)
async def get_twin_health(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Get health and fidelity indicators for a twin."""
    tid = parse_uuid(twin_id, "twin_id")
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == tid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(twin_id)

    core_result = await db.execute(
        select(PersonalityCore).where(PersonalityCore.twin_id == tid, PersonalityCore.is_current == True)
    )
    core = core_result.scalar_one_or_none()

    dim_result = await db.execute(select(DimensionalScore).where(DimensionalScore.twin_id == tid))
    scores = dim_result.scalars().all()

    per_dim = {}
    for s in scores:
        per_dim.setdefault(s.dimension.lower(), []).append(s.confidence)
    per_dim_fidelity = {d: round(sum(c) / len(c), 2) for d, c in per_dim.items() if c}

    return TwinHealthResponse(
        twin_id=twin_id,
        cfs=profile.cfs or 0.0,
        health_status=profile.health_status or "BUILDING",
        personality_core=PersonalityCoreResponse(
            big_five=core.big_five if core else {},
            mbti=core.mbti if core else {},
            ccp=core.cognitive_complexity if core else {},
            overall_confidence=core.derivation_confidence if core else 0.0,
        ),
        per_dimension_fidelity=per_dim_fidelity,
        coverage={"overall": profile.overall_coverage or 0.0, "per_category": profile.psychographic_coverage or {}},
        last_cfs_computation=profile.cfs_last_computed.isoformat() if profile.cfs_last_computed else None,
    )
