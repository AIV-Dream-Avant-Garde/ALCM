"""Health check endpoint — verifies all critical dependencies."""
import shutil
from fastapi import APIRouter
from sqlalchemy import text

from ..database import async_session_maker
from ..services.llm_provider import get_llm_provider
from ..services.tts_service import get_tts_service

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Comprehensive health check. Verifies DB, LLM, TTS, FFmpeg."""
    checks = {}

    # Database
    try:
        async with async_session_maker() as db:
            await db.execute(text("SELECT 1"))
            checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)[:100]}"

    # LLM provider
    provider = get_llm_provider()
    checks["llm_provider"] = type(provider).__name__
    checks["llm_configured"] = provider.is_configured

    # TTS
    tts = get_tts_service()
    checks["tts_configured"] = tts.is_configured

    # FFmpeg
    checks["ffmpeg_installed"] = shutil.which("ffmpeg") is not None

    overall = "ok" if checks["database"] == "ok" else "degraded"

    return {
        "status": overall,
        "service": "alcm-api",
        "version": "0.1.0",
        "checks": checks,
    }
