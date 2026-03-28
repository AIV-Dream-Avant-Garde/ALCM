"""Guardrails endpoint — POST /twin/{id}/guardrails."""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..core.errors import TwinNotFound
from ..schemas.guardrails import GuardrailPush, GuardrailResponse
from ..utils import parse_uuid

router = APIRouter(tags=["guardrails"])


@router.post("/twin/{twin_id}/guardrails", response_model=GuardrailResponse)
async def push_guardrails(
    twin_id: str, config: GuardrailPush, db: AsyncSession = Depends(get_db),
):
    """Store updated guardrail config and increment version."""
    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == parse_uuid(twin_id, "twin_id"))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(twin_id)

    profile.guardrail_config = config.model_dump()
    profile.guardrail_version = (profile.guardrail_version or 0) + 1
    await db.flush()

    return GuardrailResponse(
        confirmation=True,
        guardrail_version=profile.guardrail_version,
        propagation_status="APPLIED",
    )
