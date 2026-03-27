"""Validation endpoint — personality consistency checking.

POST /validate — check if sample content is consistent with twin's personality.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.personality_core import PersonalityCore
from ..models.dimensional_score import DimensionalScore
from ..services.llm_provider import get_llm_provider
from ..utils import parse_uuid

router = APIRouter(tags=["validate"])


class ValidateRequest(BaseModel):
    twin_id: str
    sample_content: str
    sample_context: str = ""


class ValidateResponse(BaseModel):
    consistency_score: Optional[float] = None
    passed: bool = False
    details: str = ""
    divergent_traits: List[str] = []
    recommendation: str = ""


@router.post("/validate", response_model=ValidateResponse)
async def validate_content(req: ValidateRequest, db: AsyncSession = Depends(get_db)):
    """Check if sample content is consistent with the twin's personality profile."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Load personality data from normalized tables
    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == twin_uuid,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()

    dim_result = await db.execute(
        select(DimensionalScore).where(
            DimensionalScore.twin_id == twin_uuid,
            DimensionalScore.confidence > 0.3,
        )
    )
    scores = dim_result.scalars().all()

    # Build personality context for validation
    personality_desc = "No personality data available."
    if core and core.big_five:
        traits = []
        for trait, data in core.big_five.items():
            if isinstance(data, dict):
                score = data.get("score", 50)
                traits.append(f"{trait}: {score}/100")
        personality_desc = f"Big Five: {', '.join(traits)}"

    behavioral = []
    for s in scores:
        behavioral.append(f"{s.sub_component}: {s.value}")
    if behavioral:
        personality_desc += f"\nBehavioral: {', '.join(behavioral[:10])}"

    validation_prompt = f"""You are a personality consistency validator. Given a twin's personality profile
and a sample piece of content, rate how consistent the content is with the personality on a scale of 0.0 to 1.0.

Personality Profile:
{personality_desc}

Sample Content: "{req.sample_content}"
Context: "{req.sample_context}"

Return JSON:
{{"consistency_score": 0.0-1.0, "passed": true/false, "details": "explanation", "divergent_traits": [], "recommendation": ""}}
Only return valid JSON."""

    provider = get_llm_provider()
    validation = await provider.classify(req.sample_content, validation_prompt)

    consistency_score = validation.get("consistency_score")
    if consistency_score is not None:
        consistency_score = float(consistency_score)
        passed = consistency_score >= 0.6
    else:
        consistency_score = None
        passed = False

    return ValidateResponse(
        consistency_score=consistency_score,
        passed=validation.get("passed", passed),
        details=validation.get("details", ""),
        divergent_traits=validation.get("divergent_traits", []),
        recommendation=validation.get("recommendation", ""),
    )
