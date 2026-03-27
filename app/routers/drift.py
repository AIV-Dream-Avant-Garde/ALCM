"""Drift detection endpoint — personality drift monitoring.

GET /twin/{id}/drift — returns drift score and threshold status.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..utils import parse_uuid

router = APIRouter(tags=["drift"])


class DriftResponse(BaseModel):
    drift_score: float
    threshold: float = 0.25
    threshold_status: str = "WITHIN_BOUNDS"
    per_dimension_drift: dict = {}
    baseline_set_at: Optional[str] = None
    last_checked: Optional[str] = None


@router.get("/twin/{twin_id}/drift", response_model=DriftResponse)
async def check_drift(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Check personality drift for a twin.

    Phase 1: Returns stored drift_score from twin profile.
    Phase 2: Computes drift from dimensional score history.
    """
    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == parse_uuid(twin_id, "twin_id"))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    return DriftResponse(
        drift_score=profile.drift_score or 0.0,
        threshold=0.25,
        threshold_status="WITHIN_BOUNDS" if (profile.drift_score or 0) < 0.25 else "EXCEEDED",
        per_dimension_drift={},
        baseline_set_at=profile.baseline_set_at.isoformat() if profile.baseline_set_at else None,
        last_checked=profile.drift_last_checked.isoformat() if profile.drift_last_checked else None,
    )
