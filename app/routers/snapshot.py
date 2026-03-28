"""Snapshot endpoint — create versioned identity snapshots.

POST /twin/{id}/snapshot — creates a sealed snapshot of current identity state.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..core.errors import TwinNotFound
from ..models.personality_core import PersonalityCore
from ..models.dimensional_score import DimensionalScore
from ..schemas.snapshot import SnapshotResponse
from ..utils import parse_uuid

router = APIRouter(tags=["snapshot"])


@router.post("/twin/{twin_id}/snapshot", response_model=SnapshotResponse, status_code=201)
async def create_snapshot(twin_id: str, db: AsyncSession = Depends(get_db)):
    """Create a versioned snapshot of the current identity state."""
    tid = parse_uuid(twin_id, "twin_id")

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == tid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(twin_id)

    # Load current personality core
    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == tid,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()

    # Load dimensional scores
    dim_result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == tid)
    )
    scores = dim_result.scalars().all()

    # Build snapshot data
    snapshot_data = {
        "personality_core": {
            "big_five": core.big_five if core else {},
            "mbti": core.mbti if core else {},
            "cognitive_complexity": core.cognitive_complexity if core else {},
        },
        "dimensional_scores": {
            f"{s.dimension}.{s.sub_component}": {"value": s.value, "confidence": s.confidence}
            for s in scores
        },
        "guardrail_config": profile.guardrail_config or {},
        "cfs": profile.cfs,
        "coverage": profile.overall_coverage,
    }

    data_str = json.dumps(snapshot_data, sort_keys=True, separators=(",", ":"), default=str)
    seal_hash = f"sha256:{hashlib.sha256(data_str.encode()).hexdigest()}"

    # Store snapshot reference on profile
    snapshot_ref = str(uuid.uuid4())
    profile.baseline_snapshot_ref = snapshot_ref
    profile.baseline_set_at = datetime.now(timezone.utc)
    await db.flush()

    return SnapshotResponse(
        snapshot_ref=snapshot_ref,
        seal_hash=seal_hash,
        version_number=core.version if core else 1,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
