"""Psychographic classification service.

Wraps the LLM classification call and psychographic_data persistence.
Called directly by the classify router.
Batch classification and rate limiting per category are future enhancements.
"""
import hashlib
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.psychographic_data import PsychographicData
from app.services.llm_provider import get_llm_provider


VALID_CATEGORIES = {
    "MIND", "HEART", "SPIRIT", "PHYSICALITY", "EXPERIENCES",
    "RELATIONSHIPS", "SURROUNDINGS", "WORK", "ETHICS", "FUTURE",
    "INTERESTS_TASTES",
}


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
) -> tuple[list[dict], PsychographicData]:
    """Classify content via LLM and persist the primary classification.

    Returns:
        (valid_classifications, psychographic_data_row)
    """
    provider = get_llm_provider()
    result = await provider.classify(content, classification_prompt)

    classifications = result.get("classifications", [])

    # Filter by minimum confidence (0.4) and valid categories
    valid = [
        c for c in classifications
        if c.get("confidence", 0) >= 0.4
        and c.get("category", "").upper() in VALID_CATEGORIES
    ]
    if not valid and classifications:
        valid = classifications[:1]

    if not valid:
        return [], None

    primary = max(valid, key=lambda c: c.get("confidence", 0))
    content_hash = hashlib.sha256(content.encode()).hexdigest()

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

    return valid, row
