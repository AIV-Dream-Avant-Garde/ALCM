"""Drift detection endpoint — personality drift monitoring.

GET /twin/{id}/drift — computes drift from baseline and returns status.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..services.drift_service import compute_drift
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
    """Compute personality drift from baseline for a twin."""
    tid = parse_uuid(twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == tid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Compute live drift from dimensional scores vs baseline
    drift_data = await compute_drift(tid, db)

    # Persist computed drift score on profile
    profile.drift_score = drift_data["drift_score"]
    profile.drift_last_checked = datetime.now(timezone.utc)
    await db.flush()

    return DriftResponse(
        drift_score=drift_data["drift_score"],
        threshold=0.25,
        threshold_status=drift_data["threshold_status"],
        per_dimension_drift=drift_data["per_dimension_drift"],
        baseline_set_at=profile.baseline_set_at.isoformat() if profile.baseline_set_at else None,
        last_checked=profile.drift_last_checked.isoformat() if profile.drift_last_checked else None,
    )
