"""Media analysis endpoint — POST /analyze-media."""
import logging
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.processing_job import ProcessingJob
from ..core.errors import TwinNotFound
from ..schemas.common import AnalyzeMediaRequest, AnalyzeMediaResponse
from ..utils import parse_uuid

router = APIRouter(tags=["media"])
logger = logging.getLogger(__name__)


@router.post("/analyze-media", response_model=AnalyzeMediaResponse, status_code=202)
async def analyze_media(req: AnalyzeMediaRequest, db: AsyncSession = Depends(get_db)):
    """Queue media analysis for voice/visual profile extraction."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    if not result.scalar_one_or_none():
        raise TwinNotFound(req.twin_id)

    job = ProcessingJob(
        twin_id=twin_uuid,
        job_type="ANALYZE_MEDIA",
        input_data={"media_url": req.media_url, "media_type": req.media_type},
        status="QUEUED",
    )
    db.add(job)
    await db.flush()

    return AnalyzeMediaResponse(processing_id=str(job.id))
