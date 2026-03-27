"""Processing job queue worker for async operations.

Processes jobs from the processing_jobs table in priority order.
Handles: ANALYZE_MEDIA, CLASSIFY, ATTRIBUTE, DERIVE_PERSONALITY,
COMPUTE_CFS, COMPUTE_DRIFT, SUMMARIZE_EPISODE.

Phase 3: Basic sequential worker. Production: concurrent workers with locking.
"""
import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_maker
from app.models.processing_job import ProcessingJob

logger = logging.getLogger(__name__)


async def process_next_job() -> bool:
    """Process the next queued job. Returns True if a job was processed."""
    async with async_session_maker() as db:
        # Fetch next job by priority (lower = higher priority)
        result = await db.execute(
            select(ProcessingJob)
            .where(ProcessingJob.status.in_(["QUEUED", "RETRY"]))
            .order_by(ProcessingJob.priority.asc(), ProcessingJob.queued_at.asc())
            .limit(1)
        )
        job = result.scalar_one_or_none()
        if not job:
            return False

        # Mark as processing
        job.status = "PROCESSING"
        job.started_at = datetime.now(timezone.utc)
        job.attempts += 1
        await db.commit()

        logger.info(f"Processing job {job.id}: type={job.job_type}, attempt={job.attempts}")

        try:
            result_data = await _execute_job(job, db)
            job.status = "COMPLETED"
            job.result = result_data
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info(f"Job {job.id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)
            job.error = str(e)
            if job.attempts < job.max_attempts:
                job.status = "RETRY"
            else:
                job.status = "FAILED"
                job.completed_at = datetime.now(timezone.utc)
            await db.commit()

        return True


async def _execute_job(job: ProcessingJob, db: AsyncSession) -> dict:
    """Execute a job based on its type."""
    input_data = job.input_data or {}

    if job.job_type == "ANALYZE_MEDIA":
        return await _handle_analyze_media(job.twin_id, input_data, db)
    elif job.job_type == "DERIVE_PERSONALITY":
        return await _handle_derive_personality(job.twin_id, db)
    elif job.job_type == "COMPUTE_CFS":
        return await _handle_compute_cfs(job.twin_id, db)
    elif job.job_type == "COMPUTE_DRIFT":
        return await _handle_compute_drift(job.twin_id, db)
    elif job.job_type == "SUMMARIZE_EPISODE":
        return await _handle_summarize_episode(job.twin_id, input_data, db)
    else:
        return {"status": "unsupported_job_type", "job_type": job.job_type}


async def _handle_analyze_media(twin_id, input_data: dict, db: AsyncSession) -> dict:
    """Process media analysis job.

    Phase 3: Basic metadata extraction. Production: full voice/visual analysis.
    """
    media_url = input_data.get("media_url", "")
    media_type = input_data.get("media_type", "")

    # Phase 3: Acknowledge processing, return placeholder
    # Production: call tts_service for voice, CV models for visual
    return {
        "status": "processed",
        "media_url": media_url,
        "media_type": media_type,
        "voice_profile": {},
        "visual_descriptors": {},
        "note": "Full media analysis pipeline is Phase 3+",
    }


async def _handle_derive_personality(twin_id, db: AsyncSession) -> dict:
    """Derive personality core from dimensional scores."""
    from app.services.personality_service import derive_personality_core
    core = await derive_personality_core(twin_id, db)
    await db.commit()
    return {
        "status": "derived",
        "big_five": core.big_five,
        "mbti": core.mbti,
        "confidence": core.derivation_confidence,
    }


async def _handle_compute_cfs(twin_id, db: AsyncSession) -> dict:
    """Compute and persist CFS."""
    from app.services.fidelity_service import compute_cfs
    from app.models.twin_profile import TwinProfile

    cfs = await compute_cfs(twin_id, db)
    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_id)
    )
    profile = result.scalar_one_or_none()
    if profile:
        profile.cfs = cfs
        profile.cfs_last_computed = datetime.now(timezone.utc)
    await db.commit()
    return {"status": "computed", "cfs": cfs}


async def _handle_compute_drift(twin_id, db: AsyncSession) -> dict:
    """Compute and persist drift score."""
    from app.services.drift_service import compute_drift
    drift_data = await compute_drift(twin_id, db)
    await db.commit()
    return {"status": "computed", **drift_data}


async def _handle_summarize_episode(twin_id, input_data: dict, db: AsyncSession) -> dict:
    """Summarize an interaction into an episodic memory."""
    from app.services.memory_service import create_episode
    episode = await create_episode(
        twin_id=twin_id,
        summary=input_data.get("summary", "Interaction occurred"),
        topics=input_data.get("topics", []),
        emotional_valence=input_data.get("emotional_valence", "NEUTRAL"),
        deployment_scope=input_data.get("deployment_scope", "TRAINING_AREA"),
        db=db,
    )
    await db.commit()
    return {"status": "summarized", "episode_id": str(episode.id)}


async def run_worker(poll_interval: float = 2.0):
    """Run the job worker loop. Call this from a background task."""
    logger.info("Job worker started")
    while True:
        try:
            processed = await process_next_job()
            if not processed:
                await asyncio.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Job worker error: {e}", exc_info=True)
            await asyncio.sleep(poll_interval)
