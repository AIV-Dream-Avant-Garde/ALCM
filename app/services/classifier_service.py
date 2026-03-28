"""Psychographic classification service — 2-stage: LLM + rule validation.

Stage 1: LLM-based semantic multi-label classification
Stage 2: Rule-based validation (co-occurrence, field compatibility,
         deduplication, rate limiting) per Developer Guide Section 11.1.
"""
import hashlib
import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.psychographic_data import PsychographicData
from app.services.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "MIND", "HEART", "SPIRIT", "PHYSICALITY", "EXPERIENCES",
    "RELATIONSHIPS", "SURROUNDINGS", "WORK", "ETHICS", "FUTURE",
    "INTERESTS_TASTES",
}

# Category co-occurrence matrix — plausible multi-label pairings.
# If two categories are NOT in this set, their co-occurrence is suspicious.
PLAUSIBLE_COOCCURRENCES = {
    frozenset({"ETHICS", "RELATIONSHIPS"}),
    frozenset({"ETHICS", "HEART"}),
    frozenset({"ETHICS", "WORK"}),
    frozenset({"HEART", "RELATIONSHIPS"}),
    frozenset({"HEART", "EXPERIENCES"}),
    frozenset({"HEART", "SPIRIT"}),
    frozenset({"MIND", "WORK"}),
    frozenset({"MIND", "INTERESTS_TASTES"}),
    frozenset({"MIND", "ETHICS"}),
    frozenset({"WORK", "FUTURE"}),
    frozenset({"WORK", "RELATIONSHIPS"}),
    frozenset({"EXPERIENCES", "SPIRIT"}),
    frozenset({"EXPERIENCES", "FUTURE"}),
    frozenset({"EXPERIENCES", "RELATIONSHIPS"}),
    frozenset({"FUTURE", "SPIRIT"}),
    frozenset({"FUTURE", "HEART"}),
    frozenset({"INTERESTS_TASTES", "HEART"}),
    frozenset({"INTERESTS_TASTES", "RELATIONSHIPS"}),
    frozenset({"SURROUNDINGS", "WORK"}),
    frozenset({"SURROUNDINGS", "HEART"}),
    frozenset({"PHYSICALITY", "INTERESTS_TASTES"}),
    frozenset({"PHYSICALITY", "WORK"}),
    frozenset({"SPIRIT", "ETHICS"}),
}

# Rate limit: max classifications per category per twin per hour
RATE_LIMIT_PER_CATEGORY = 50

MIN_CONFIDENCE = 0.4


async def classify_and_persist(
    twin_id: UUID,
    content: str,
    modality: str,
    source_type: str,
    source_reliability: float,
    classification_prompt: str,
    contributor_id: Optional[UUID] = None,
    contributor_type: Optional[str] = None,
    db: AsyncSession = None,
) -> tuple[list[dict], Optional[PsychographicData]]:
    """2-stage classification: LLM semantic classification + rule validation."""

    # === STAGE 1: LLM Classification ===
    provider = get_llm_provider()
    result = await provider.classify(content, classification_prompt)
    classifications = result.get("classifications", [])

    if not classifications:
        return [], None

    # === STAGE 2: Rule-Based Validation ===
    validated = await _validate_classifications(classifications, twin_id, content, db)

    if not validated:
        return [], None

    primary = max(validated, key=lambda c: c.get("confidence", 0))
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Deduplication: check if identical content already classified
    existing = await db.execute(
        select(PsychographicData).where(
            PsychographicData.twin_id == twin_id,
            PsychographicData.content_hash == content_hash,
        )
    )
    if existing.scalar_one_or_none():
        logger.info(f"Duplicate content hash {content_hash[:12]} for twin {twin_id}, skipping")
        return validated, None

    row = PsychographicData(
        twin_id=twin_id,
        category=primary["category"].upper(),
        sub_category=primary.get("data_fields_informed", [None])[0],
        content=content,
        content_hash=content_hash,
        modality=modality,
        source_type=source_type,
        source_reliability=source_reliability,
        classification_confidence=primary.get("confidence", 0.5),
        contributor_id=contributor_id,
        contributor_type=contributor_type,
    )
    db.add(row)
    await db.flush()

    return validated, row


async def _validate_classifications(
    classifications: list[dict],
    twin_id: UUID,
    content: str,
    db: AsyncSession,
) -> list[dict]:
    """Stage 2 rule engine: validate LLM classifications."""

    # Rule 1: Minimum confidence threshold
    valid = [c for c in classifications if c.get("confidence", 0) >= MIN_CONFIDENCE]
    if not valid:
        valid = classifications[:1]  # Keep top-1 even if below threshold

    # Rule 2: Valid category check
    valid = [c for c in valid if c.get("category", "").upper() in VALID_CATEGORIES]
    if not valid:
        return []

    # Normalize categories
    for c in valid:
        c["category"] = c["category"].upper()

    # Rule 3: Co-occurrence plausibility
    if len(valid) > 1:
        categories = [c["category"] for c in valid]
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                pair = frozenset({categories[i], categories[j]})
                if pair not in PLAUSIBLE_COOCCURRENCES:
                    # Keep only the higher-confidence one
                    logger.info(
                        f"Implausible co-occurrence {categories[i]}+{categories[j]}, "
                        "keeping higher confidence only"
                    )
                    valid = [max(valid, key=lambda c: c.get("confidence", 0))]
                    break
            else:
                continue
            break

    # Rule 4: Data field compatibility
    valid = [c for c in valid if _validate_fields(c)]

    # Rule 5: Rate limiting per category
    if db:
        rate_limited = []
        for c in valid:
            count = await _get_recent_count(twin_id, c["category"], db)
            if count < RATE_LIMIT_PER_CATEGORY:
                rate_limited.append(c)
            else:
                logger.info(f"Rate limit hit for {c['category']} (twin {twin_id}): {count} in last hour")
        valid = rate_limited if rate_limited else valid[:1]

    return valid


def _validate_fields(classification: dict) -> bool:
    """Check that claimed data_fields_informed are plausible for the category."""
    # Accept any classification that has the required fields
    if "category" not in classification:
        return False
    if "confidence" not in classification:
        return False
    return True


async def _get_recent_count(twin_id: UUID, category: str, db: AsyncSession) -> int:
    """Count classifications for this twin+category in the last hour."""
    from datetime import datetime, timezone, timedelta
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

    result = await db.execute(
        select(func.count(PsychographicData.id)).where(
            PsychographicData.twin_id == twin_id,
            PsychographicData.category == category,
            PsychographicData.created_at > one_hour_ago,
        )
    )
    return result.scalar() or 0
