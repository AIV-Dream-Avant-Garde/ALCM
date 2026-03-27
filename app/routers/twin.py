"""Twin lifecycle endpoints — POST /twin, DELETE /twin/{id}, GET /twin/{id}/health."""
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.personality_core import PersonalityCore
from ..models.dimensional_score import DimensionalScore
from ..utils import parse_uuid

router = APIRouter(prefix="/twin", tags=["twin"])


# --- Request/Response schemas ---

class TwinCreateRequest(BaseModel):
    identity_category: str = "ENTERTAINMENT"
    clone_type: str = "PUBLIC_FIGURE"


class TwinCreateResponse(BaseModel):
    alcm_twin_id: str
    status: str
    created_at: str


class TwinDeleteResponse(BaseModel):
    deleted: bool
    twin_id: str


class PersonalityCoreResponse(BaseModel):
    big_five: dict = {}
    mbti: dict = {}
    ccp: dict = {}
    overall_confidence: float = 0.0


class TwinHealthResponse(BaseModel):
    twin_id: str
    cfs: float
    health_status: str
    personality_core: PersonalityCoreResponse
    per_dimension_fidelity: dict = {}
    coverage: dict = {}
    last_training_activity: Optional[str] = None
    last_cfs_computation: Optional[str] = None


# --- Endpoints ---

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

    # Create initial empty personality core (v1)
    core = PersonalityCore(
        twin_id=profile.id,
        version=1,
        big_five={},
        mbti={},
        cognitive_complexity={},
        derivation_confidence=0.0,
        derived_from_observations=0,
        is_current=True,
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
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")
    await db.delete(profile)
    return TwinDeleteResponse(deleted=True, twin_id=twin_id)


@router.get("/{twin_id}/health", response_model=TwinHealthResponse)
async def get_twin_health(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Get health and fidelity indicators for a twin."""
    tid = parse_uuid(twin_id, "twin_id")
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == tid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Load current personality core
    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == tid,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()

    # Compute per-dimension fidelity from dimensional scores
    dim_result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == tid)
    )
    scores = dim_result.scalars().all()

    per_dim_fidelity = {}
    dim_groups = {}
    for s in scores:
        dim_groups.setdefault(s.dimension, []).append(s.confidence)
    for dim, confidences in dim_groups.items():
        per_dim_fidelity[dim.lower()] = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    personality_core_resp = PersonalityCoreResponse()
    if core:
        personality_core_resp = PersonalityCoreResponse(
            big_five=core.big_five or {},
            mbti=core.mbti or {},
            ccp=core.cognitive_complexity or {},
            overall_confidence=core.derivation_confidence,
        )

    return TwinHealthResponse(
        twin_id=twin_id,
        cfs=profile.cfs or 0.0,
        health_status=profile.health_status or "BUILDING",
        personality_core=personality_core_resp,
        per_dimension_fidelity=per_dim_fidelity,
        coverage={
            "overall": profile.overall_coverage or 0.0,
            "per_category": profile.psychographic_coverage or {},
        },
        last_training_activity=None,
        last_cfs_computation=profile.cfs_last_computed.isoformat() if profile.cfs_last_computed else None,
    )
