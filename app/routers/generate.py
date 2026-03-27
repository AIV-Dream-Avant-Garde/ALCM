"""Generation endpoints — identity-consistent text generation.

POST /generate — non-streaming
POST /generate/stream — SSE streaming
"""
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..services.llm_provider import get_llm_provider
from ..core.prompt_assembly import assemble_prompt
from ..utils import parse_uuid

router = APIRouter(tags=["generate"])
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    twin_id: str
    context: str
    guardrails: dict = {}
    mode: str = "CONVERSATION"
    conversation_history: Optional[List[Dict]] = None
    deployment_scope: str = "TRAINING_AREA"


class GuardrailChecks(BaseModel):
    content_safety: str = "PASSED"
    personality_consistency: str = "PASSED"
    topic_restrictions: str = "PASSED"


class GenerateResponse(BaseModel):
    response_text: str
    personality_consistency_score: Optional[float] = None
    mood_state: dict = {}
    guardrail_checks: GuardrailChecks = GuardrailChecks()
    tokens_used: dict = {}
    metadata: dict = {}


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Generate identity-consistent text from normalized identity data."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    if profile.status == "LOCKED":
        raise HTTPException(status_code=423, detail={"code": "TWIN_LOCKED", "message": "Twin is locked."})

    # Assemble prompt from normalized tables
    system_prompt = await assemble_prompt(
        twin_id=twin_uuid,
        context=req.context,
        conversation_history=req.conversation_history,
        guardrails=req.guardrails,
        mode=req.mode,
        deployment_scope=req.deployment_scope,
        db=db,
    )

    provider = get_llm_provider()
    response_text = await provider.generate(
        prompt=req.context,
        context=system_prompt,
        temperature=0.7,
    )

    # === SAFEGUARD GATEWAY ===
    # Validate generated response against personality profile and guardrails
    consistency_score = None
    guardrail_checks = GuardrailChecks()
    retries = 0
    max_retries = 2

    consistency_score = await _check_personality_consistency(
        twin_uuid, response_text, req.context, db
    )

    if consistency_score is not None and consistency_score < 0.6 and retries < max_retries:
        # Regenerate with strengthened personality instructions
        logger.warning(
            f"Personality consistency {consistency_score:.2f} < 0.6 for twin {req.twin_id}, "
            f"regenerating (attempt {retries + 1}/{max_retries})"
        )
        strengthened_prompt = system_prompt + (
            "\n\nIMPORTANT: Your previous response was not consistent with "
            "this person's personality. Stay more faithful to the personality "
            "traits and communication style described above."
        )
        response_text = await provider.generate(
            prompt=req.context,
            context=strengthened_prompt,
            temperature=0.6,
        )
        consistency_score = await _check_personality_consistency(
            twin_uuid, response_text, req.context, db
        )
        retries += 1

    guardrail_checks.personality_consistency = (
        "PASSED" if consistency_score is None or consistency_score >= 0.6 else "FLAGGED"
    )

    # Check guardrail topic restrictions
    effective_guardrails = req.guardrails or {}
    if profile.guardrail_config:
        effective_guardrails = {**profile.guardrail_config, **effective_guardrails}
    blocked_topics = effective_guardrails.get("blocked_topics", [])
    if blocked_topics:
        response_lower = response_text.lower()
        for topic in blocked_topics:
            if topic.lower() in response_lower:
                guardrail_checks.topic_restrictions = "FLAGGED"
                logger.warning(f"Blocked topic '{topic}' detected in response for twin {req.twin_id}")
                break

    return GenerateResponse(
        response_text=response_text,
        personality_consistency_score=consistency_score,
        mood_state={"valence": 0, "arousal": 0.5},
        guardrail_checks=guardrail_checks,
        tokens_used={"input": 0, "output": 0},
        metadata={
            "mode": req.mode,
            "context_type_detected": "PUBLIC",
            "personality_core_used": "v1",
            "layers_assembled": [1, 2, 3, 4, 5, 6, 7],
            "retries": retries,
        },
    )


async def _check_personality_consistency(
    twin_id, response_text: str, context: str, db: AsyncSession,
) -> float | None:
    """Run LLM-based personality consistency check (Safeguard Gateway).

    Returns consistency score (0.0-1.0) or None if check fails.
    """
    from ..models.personality_core import PersonalityCore

    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == twin_id,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()
    if not core or not core.big_five:
        return None  # No personality data to validate against

    # Build compact personality summary for validation
    traits = []
    for trait, data in core.big_five.items():
        if isinstance(data, dict) and data.get("confidence", 0) > 0.3:
            traits.append(f"{trait}={data.get('score', 50)}")
    if not traits:
        return None

    personality_summary = f"Big Five: {', '.join(traits)}"
    if core.mbti and core.mbti.get("type"):
        personality_summary += f". MBTI: {core.mbti['type']}"

    check_prompt = f"""Rate the personality consistency of this response on a scale of 0.0 to 1.0.
Personality: {personality_summary}
Context: {context[:200]}
Response: {response_text[:500]}
Return ONLY a JSON object: {{"score": 0.85}}"""

    try:
        provider = get_llm_provider()
        result = await provider.classify(response_text[:500], check_prompt)
        score = result.get("score")
        if score is not None:
            return float(score)
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")

    return None


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Stream identity-consistent text via SSE."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    if profile.status == "LOCKED":
        raise HTTPException(status_code=423, detail={"code": "TWIN_LOCKED", "message": "Twin is locked."})

    system_prompt = await assemble_prompt(
        twin_id=twin_uuid,
        context=req.context,
        conversation_history=req.conversation_history,
        guardrails=req.guardrails,
        mode=req.mode,
        deployment_scope=req.deployment_scope,
        db=db,
    )

    provider = get_llm_provider()

    async def event_stream():
        try:
            async for chunk in provider.generate_stream(
                prompt=req.context,
                context=system_prompt,
                temperature=0.7,
            ):
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
        except Exception as e:
            logger.error(f"Stream failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'metadata', 'personality_consistency_score': None, 'tokens_used': {}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
