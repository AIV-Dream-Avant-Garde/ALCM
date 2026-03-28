"""Generation endpoints — POST /generate, POST /generate/stream."""
import json
import logging
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..services.generation_service import generate_response
from ..services.llm_provider import get_llm_provider
from ..core.prompt_assembly import assemble_prompt
from ..core.errors import TwinNotFound, TwinLocked, SuccessorHold
from ..schemas.generate import GenerateRequest, GenerateResponse
from ..utils import parse_uuid

router = APIRouter(tags=["generate"])
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Generate identity-consistent text. All logic delegated to generation_service."""
    return await generate_response(req, db)


@router.post("/generate/stream")
async def generate_stream(req: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Stream identity-consistent text via SSE."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(req.twin_id)
    if profile.status == "LOCKED":
        raise TwinLocked()
    if profile.status == "PROTECTED_HOLD":
        raise SuccessorHold()

    system_prompt = await assemble_prompt(
        twin_id=twin_uuid, context=req.context,
        conversation_history=req.conversation_history,
        guardrails=req.guardrails, mode=req.mode,
        deployment_scope=req.deployment_scope, db=db,
    )
    provider = get_llm_provider()

    async def event_stream():
        try:
            async for chunk in provider.generate_stream(
                prompt=req.context, context=system_prompt, temperature=0.7,
            ):
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
        except Exception as e:
            logger.error(f"Stream failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'metadata', 'personality_consistency_score': None})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
