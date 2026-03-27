"""Classification endpoint — psychographic classification via LLM.

Classifies content into the 11 psychographic categories and persists
the result in the psychographic_data table.
"""
import hashlib
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.psychographic_data import PsychographicData
from ..services.llm_provider import get_llm_provider
from ..utils import parse_uuid

router = APIRouter(tags=["classify"])

CLASSIFICATION_PROMPT = """You are a psychographic classifier for the ALCM identity engine.
Analyze the following content and classify it into one or more of the 11 psychographic categories.
A single piece of content may belong to multiple categories.

The 11 categories are:
- MIND: thinking patterns, reasoning, humor, intellectual curiosity
- HEART: core values, emotional expression, stress responses, empathy
- SPIRIT: purpose, meaning, philosophical orientation, connection to causes
- PHYSICALITY: energy patterns, physical comfort, health approach
- EXPERIENCES: formative memories, life events, lessons from failure
- RELATIONSHIPS: social roles, boundaries, conflict resolution, attachment
- SURROUNDINGS: environment preferences, routines, financial psychology
- WORK: success definition, leadership style, pressure response
- ETHICS: moral framework, non-negotiable principles, justice orientation
- FUTURE: idealized self, aspirations, fears, vision for the world
- INTERESTS_TASTES: cultural preferences, hobbies, expertise, opinions

Return a JSON object with:
{
  "classifications": [
    {
      "category": "CATEGORY_NAME",
      "confidence": 0.0-1.0,
      "data_fields_informed": ["field1", "field2"],
      "evidence": "Brief explanation"
    }
  ]
}

Only return valid JSON, no other text."""


class ClassifyRequest(BaseModel):
    twin_id: str
    content: str
    modality: str = "TEXT"
    source_reliability: float = 0.6
    contributor_id: Optional[str] = None
    contributor_type: Optional[str] = None  # TALENT | TEAM_MEMBER | AIV_INTERNAL


class CategoryAffected(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    processing_id: str
    categories_affected: List[CategoryAffected] = []
    sub_categories: List[str] = []
    psychographic_data_id: str


@router.post("/classify", response_model=ClassifyResponse)
async def classify_content(req: ClassifyRequest, db: AsyncSession = Depends(get_db)):
    """Classify content into psychographic categories and persist."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")

    result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_uuid)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Twin not found in ALCM")

    # Call LLM for classification
    provider = get_llm_provider()
    classification = await provider.classify(req.content, CLASSIFICATION_PROMPT)

    classifications = classification.get("classifications", [])
    if not classifications:
        raise HTTPException(
            status_code=422,
            detail={"code": "CLASSIFICATION_FAILED", "message": "Content could not be classified into any psychographic category."},
        )

    # Filter by minimum confidence threshold (0.4 per spec Section 11.1)
    valid_classifications = [c for c in classifications if c.get("confidence", 0) >= 0.4]
    if not valid_classifications:
        valid_classifications = classifications[:1]  # Keep at least one

    # Use the highest-confidence category as primary
    primary = max(valid_classifications, key=lambda c: c.get("confidence", 0))
    content_hash = hashlib.sha256(req.content.encode()).hexdigest()

    # Persist to psychographic_data table
    psycho_data = PsychographicData(
        twin_id=twin_uuid,
        category=primary["category"],
        sub_category=primary.get("data_fields_informed", [None])[0],
        content=req.content,
        content_hash=content_hash,
        modality=req.modality,
        source_type="TRAINING_AREA",  # Default; platform should specify
        source_reliability=req.source_reliability,
        classification_confidence=primary.get("confidence", 0.5),
        contributor_id=uuid.UUID(req.contributor_id) if req.contributor_id else None,
        contributor_type=req.contributor_type,
    )
    db.add(psycho_data)
    await db.flush()

    # Update psychographic coverage on the twin profile
    coverage = dict(profile.psychographic_coverage or {})
    for c in valid_classifications:
        cat = c["category"].lower()
        current = coverage.get(cat, 0)
        # Increment coverage (simplified — full formula in spec Section 11.3)
        coverage[cat] = min(100, current + max(1, int(c.get("confidence", 0.5) * 5)))
    profile.psychographic_coverage = coverage

    # Recompute overall coverage
    if coverage:
        profile.overall_coverage = sum(coverage.values()) / (11 * 100) * 100
    await db.flush()

    # Build response
    categories_affected = [
        CategoryAffected(category=c["category"], confidence=c.get("confidence", 0.5))
        for c in valid_classifications
    ]
    sub_categories = []
    for c in valid_classifications:
        sub_categories.extend(c.get("data_fields_informed", []))

    return ClassifyResponse(
        processing_id=str(uuid.uuid4()),
        categories_affected=categories_affected,
        sub_categories=list(set(sub_categories)),
        psychographic_data_id=str(psycho_data.id),
    )
