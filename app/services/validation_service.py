"""Personality consistency validation service.

Validates content against a twin's personality profile using LLM-based assessment.
"""
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.twin_profile import TwinProfile
from app.models.personality_core import PersonalityCore
from app.models.dimensional_score import DimensionalScore
from app.core.errors import TwinNotFound
from app.services.llm import get_llm_provider
from app.schemas.validate import ValidateRequest, ValidateResponse

logger = logging.getLogger(__name__)


async def validate_consistency(req: ValidateRequest, db: AsyncSession) -> ValidateResponse:
    """Check if sample content is consistent with the twin's personality profile."""
    twin_uuid = UUID(req.twin_id)

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(req.twin_id)

    # Load personality data
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

    # Build personality summary
    personality_desc = "No personality data available."
    if core and core.big_five:
        traits = [f"{t}: {d.get('score', 50)}/100" for t, d in core.big_five.items() if isinstance(d, dict)]
        personality_desc = f"Big Five: {', '.join(traits)}"

    behavioral = [f"{s.sub_component}: {s.value}" for s in scores[:10]]
    if behavioral:
        personality_desc += f"\nBehavioral: {', '.join(behavioral)}"

    validation_prompt = f"""You are a personality consistency validator. Rate how consistent the content is with the personality on a scale of 0.0 to 1.0.

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
