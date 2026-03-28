"""Classification endpoint — POST /classify."""
import hashlib
import uuid
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models.twin_profile import TwinProfile
from ..models.psychographic_data import PsychographicData
from ..services.llm import get_llm_provider
from ..core.errors import TwinNotFound, ClassificationFailed
from ..schemas.classify import ClassifyRequest, ClassifyResponse, CategoryAffected
from ..utils import parse_uuid

router = APIRouter(tags=["classify"])

CLASSIFICATION_PROMPT = """You are a psychographic classifier for the ALCM identity engine.
Analyze the following content and classify it into one or more of the 11 psychographic categories.

The 11 categories: MIND, HEART, SPIRIT, PHYSICALITY, EXPERIENCES, RELATIONSHIPS, SURROUNDINGS, WORK, ETHICS, FUTURE, INTERESTS_TASTES

Return JSON: {"classifications": [{"category": "CATEGORY", "confidence": 0.0-1.0, "data_fields_informed": ["field"], "evidence": "why"}]}
Only return valid JSON."""


@router.post("/classify", response_model=ClassifyResponse)
async def classify_content(req: ClassifyRequest, db: AsyncSession = Depends(get_db)):
    """Classify content into psychographic categories and persist."""
    twin_uuid = parse_uuid(req.twin_id, "twin_id")
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    if not result.scalar_one_or_none():
        raise TwinNotFound(req.twin_id)

    # Re-fetch for update (needed after scalar check)
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    profile = result.scalar_one_or_none()

    provider = get_llm_provider()
    classification = await provider.classify(req.content, CLASSIFICATION_PROMPT)
    classifications = classification.get("classifications", [])

    valid = [c for c in classifications if c.get("confidence", 0) >= 0.4]
    if not valid:
        valid = classifications[:1] if classifications else []
    if not valid:
        raise ClassificationFailed()

    primary = max(valid, key=lambda c: c.get("confidence", 0))

    psycho_data = PsychographicData(
        twin_id=twin_uuid,
        category=primary["category"].upper(),
        sub_category=primary.get("data_fields_informed", [None])[0],
        content=req.content,
        content_hash=hashlib.sha256(req.content.encode()).hexdigest(),
        modality=req.modality,
        source_type="TRAINING_AREA",
        source_reliability=req.source_reliability,
        classification_confidence=primary.get("confidence", 0.5),
        contributor_id=uuid.UUID(req.contributor_id) if req.contributor_id else None,
        contributor_type=req.contributor_type,
    )
    db.add(psycho_data)

    coverage = dict(profile.psychographic_coverage or {})
    for c in valid:
        cat = c["category"].lower()
        coverage[cat] = min(100, coverage.get(cat, 0) + max(1, int(c.get("confidence", 0.5) * 5)))
    profile.psychographic_coverage = coverage
    if coverage:
        profile.overall_coverage = sum(coverage.values()) / (11 * 100) * 100

    # Detect fear/need signals and create RAG entries
    from ..services.classifier_service import detect_fear_need_signals, create_fear_need_rag_entries
    fear_need = detect_fear_need_signals(req.content)
    if fear_need:
        await create_fear_need_rag_entries(twin_uuid, req.content, fear_need, db)

    await db.flush()

    return ClassifyResponse(
        processing_id=str(uuid.uuid4()),
        categories_affected=[CategoryAffected(category=c["category"], confidence=c.get("confidence", 0.5)) for c in valid],
        sub_categories=list({f for c in valid for f in c.get("data_fields_informed", [])}),
        psychographic_data_id=str(psycho_data.id),
    )
