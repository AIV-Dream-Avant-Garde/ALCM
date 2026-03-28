"""Health check endpoint — GET /health per spec Section 3.9."""
import time
from fastapi import APIRouter
from sqlalchemy import text

from ..database import async_session_maker
from ..services.llm import get_llm_provider
from ..services.tts_service import get_tts_service

router = APIRouter(tags=["health"])

_startup_time = time.time()


@router.get("/health")
async def health():
    """Service health check matching spec Section 3.9 response shape."""
    db_status = "disconnected"
    try:
        async with async_session_maker() as db:
            await db.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception:
        db_status = "disconnected"

    provider = get_llm_provider()
    llm_status = "connected" if provider.is_configured else "not_configured"

    tts = get_tts_service()
    tts_status = "connected" if tts.is_configured else "not_configured"

    status = "ok" if db_status == "connected" else "degraded"

    return {
        "status": status,
        "version": "2.0.0",
        "uptime_seconds": int(time.time() - _startup_time),
        "database": db_status,
        "llm_provider": llm_status,
        "tts_provider": tts_status,
    }
