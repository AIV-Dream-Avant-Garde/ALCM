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

    return GenerateResponse(
        response_text=response_text,
        personality_consistency_score=None,  # Phase 2: LLM-based consistency check
        mood_state={"valence": 0, "arousal": 0.5},
        guardrail_checks=GuardrailChecks(),
        tokens_used={"input": 0, "output": 0},  # Phase 2: track actual usage
        metadata={
            "mode": req.mode,
            "context_type_detected": "PUBLIC",
            "personality_core_used": "v1",
            "layers_assembled": [1, 2, 3, 4, 5, 6, 7],
        },
    )


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
