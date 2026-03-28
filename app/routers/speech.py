"""Speech generation endpoint — POST /generate-speech."""
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.voice_profile import VoiceProfile
from ..services.tts_service import get_tts_service
from ..core.errors import TwinNotFound, InsufficientData, ServiceUnavailable
from ..schemas.common import SpeechRequest
from ..utils import parse_uuid

router = APIRouter(tags=["speech"])


@router.post("/generate-speech")
async def generate_speech(req: SpeechRequest, db: AsyncSession = Depends(get_db)):
    """Generate speech from text using the twin's cloned voice."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    if not result.scalar_one_or_none():
        raise TwinNotFound(req.twin_id)

    vp_result = await db.execute(select(VoiceProfile).where(VoiceProfile.twin_id == twin_uuid))
    vp = vp_result.scalar_one_or_none()
    if not vp or not vp.tts_voice_id:
        raise InsufficientData("No voice profile configured for this twin.")

    tts = get_tts_service()
    audio_bytes = await tts.text_to_speech(vp.tts_voice_id, req.text)
    if not audio_bytes:
        raise ServiceUnavailable("Speech generation failed.")

    return Response(content=audio_bytes, media_type="audio/mpeg")
