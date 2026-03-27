"""Guardrails endpoint — receive guardrail config from the platform.

POST /twin/{id}/guardrails — stores config and increments version.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..utils import parse_uuid

router = APIRouter(tags=["guardrails"])


class GuardrailPush(BaseModel):
    blocked_topics: List[str] = []
    restricted_topics: dict = {}
    language_restrictions: List[str] = []
    min_formality: int = 0
    max_controversy: int = 100
    humor_permitted: bool = True
    humor_blacklist: List[str] = []
    require_ai_disclosure: bool = True
    disclosure_text: str = "This is an AI-generated response."


class GuardrailResponse(BaseModel):
    confirmation: bool
    guardrail_version: int
    propagation_status: str


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
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    profile.guardrail_config = config.model_dump()
    profile.guardrail_version = (profile.guardrail_version or 0) + 1
    await db.flush()

    return GuardrailResponse(
        confirmation=True,
        guardrail_version=profile.guardrail_version,
        propagation_status="APPLIED",
    )
