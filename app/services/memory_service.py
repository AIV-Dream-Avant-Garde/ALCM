"""Episodic memory service — creation, retrieval, and decay.

See Developer Guide Section 6.1.2 (Episodic Memory).
Supports creation and recency-based retrieval scoped to training area.
Vector-based retrieval, per-deployment scoping, and decay mechanics are future enhancements.
"""
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episodic_memory import EpisodicMemory


async def create_episode(
    twin_id: UUID,
    summary: str,
    topics: List[str] = None,
    contact_id: Optional[str] = None,
    emotional_valence: str = "NEUTRAL",
    outcome: Optional[str] = None,
    deployment_scope: str = "TRAINING_AREA",
    db: AsyncSession = None,
) -> EpisodicMemory:
    """Create a new episodic memory entry."""
    episode = EpisodicMemory(
        twin_id=twin_id,
        summary=summary,
        topics=topics or [],
        contact_id=contact_id,
        emotional_valence=emotional_valence,
        outcome=outcome,
        deployment_scope=deployment_scope,
        interaction_at=datetime.now(timezone.utc),
    )
    db.add(episode)
    await db.flush()
    return episode


async def get_relevant_episodes(
    twin_id: UUID,
    deployment_scope: str = "TRAINING_AREA",
    limit: int = 3,
    db: AsyncSession = None,
) -> List[EpisodicMemory]:
    """Retrieve most relevant recent episodes for prompt context.

    Currently uses recency-based retrieval.
    Topic similarity + recency weighting is a future enhancement.
    """
    result = await db.execute(
        select(EpisodicMemory)
        .where(
            EpisodicMemory.twin_id == twin_id,
            EpisodicMemory.deployment_scope == deployment_scope,
            EpisodicMemory.retrieval_weight > 0.1,
        )
        .order_by(EpisodicMemory.interaction_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
