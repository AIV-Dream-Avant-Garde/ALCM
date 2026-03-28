"""Processing job status endpoint — GET /jobs/{id}."""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.processing_job import ProcessingJob
from ..core.errors import ALCMError
from ..schemas.jobs import JobStatusResponse
from ..utils import parse_uuid

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Check the status of an async processing job."""
    result = await db.execute(
        select(ProcessingJob).where(ProcessingJob.id == parse_uuid(job_id, "job_id"))
    )
    job = result.scalar_one_or_none()
    if not job:
        raise ALCMError(code="INVALID_REQUEST", message="Job not found.", status=404)

    return JobStatusResponse(
        job_id=str(job.id),
        job_type=job.job_type,
        status=job.status,
        result=job.result,
        error=job.error,
        queued_at=job.queued_at.isoformat() if job.queued_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )
