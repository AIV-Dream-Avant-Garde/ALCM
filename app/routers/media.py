"""Media analysis endpoint — process uploaded audio/video/images.

POST /analyze-media — queues a processing job for media analysis.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.processing_job import ProcessingJob
from ..utils import parse_uuid

router = APIRouter(tags=["media"])
logger = logging.getLogger(__name__)


class AnalyzeMediaRequest(BaseModel):
    twin_id: str
    media_url: str
    media_type: str  # VIDEO | AUDIO | IMAGE


class AnalyzeMediaResponse(BaseModel):
    processing_id: str
    status: str = "QUEUED"
    estimated_duration_seconds: int = 120


@router.post("/analyze-media", response_model=AnalyzeMediaResponse, status_code=202)
async def analyze_media(req: AnalyzeMediaRequest, db: AsyncSession = Depends(get_db)):
    """Queue media analysis for voice/visual profile extraction.

    Phase 1: Creates a ProcessingJob. Actual processing is Phase 2.
    """
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    job = ProcessingJob(
        twin_id=twin_uuid,
        job_type="ANALYZE_MEDIA",
        input_data={"media_url": req.media_url, "media_type": req.media_type},
        status="QUEUED",
        priority=5,
    )
    db.add(job)
    await db.flush()

    logger.info(f"Media analysis queued: job={job.id}, twin={req.twin_id}, type={req.media_type}")

    return AnalyzeMediaResponse(
        processing_id=str(job.id),
        status="QUEUED",
        estimated_duration_seconds=120,
    )
