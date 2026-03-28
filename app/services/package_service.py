"""Identity Package assembly and scoped delivery.

Assembles scoped delivery modules from normalized tables. Each module
is independently queryable per deal terms. See spec Section 16.0.
"""
import hashlib
import json
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.twin_profile import TwinProfile
from app.models.personality_core import PersonalityCore
from app.models.dimensional_score import DimensionalScore
from app.models.context_modulation import ContextModulation
from app.models.rag_entry import RagEntry
from app.models.voice_profile import VoiceProfile
from app.models.visual_profile import VisualProfile
from app.core.errors import TwinNotFound, InvalidRequest
from app.schemas.package import PackageResponse

VALID_MODULES = {"identity_profile", "knowledge_base", "voice_identity", "visual_identity"}


async def assemble_package(twin_id_str: str, scope: str, db: AsyncSession) -> PackageResponse:
    """Assemble a scoped identity package from normalized tables."""
    tid = UUID(twin_id_str)

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == tid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(twin_id_str)

    requested = {s.strip() for s in scope.split(",")} & VALID_MODULES
    if not requested:
        raise InvalidRequest(f"Invalid scope. Valid modules: {sorted(VALID_MODULES)}")

    modules = {}
    if "identity_profile" in requested:
        modules["identity_profile"] = await _build_identity_profile(tid, profile, db)
    if "knowledge_base" in requested:
        modules["knowledge_base"] = await _build_knowledge_base(tid, db)
    if "voice_identity" in requested:
        modules["voice_identity"] = await _build_voice_identity(tid, db)
    if "visual_identity" in requested:
        modules["visual_identity"] = await _build_visual_identity(tid, db)

    data_str = json.dumps(modules, sort_keys=True, separators=(",", ":"), default=str)
    seal_hash = f"sha256:{hashlib.sha256(data_str.encode()).hexdigest()}"

    return PackageResponse(
        twin_id=twin_id_str,
        version=1,
        seal_hash=seal_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
        modules=modules,
    )


async def _build_identity_profile(tid, profile: TwinProfile, db: AsyncSession) -> dict:
    core_result = await db.execute(
        select(PersonalityCore).where(PersonalityCore.twin_id == tid, PersonalityCore.is_current == True)
    )
    core = core_result.scalar_one_or_none()
    personality_core = {"big_five": core.big_five, "mbti": core.mbti, "ccp": core.cognitive_complexity} if core else {}

    dim_result = await db.execute(select(DimensionalScore).where(DimensionalScore.twin_id == tid))
    scores = dim_result.scalars().all()
    behavioral = {}
    for s in scores:
        behavioral.setdefault(s.dimension.lower(), {})[s.sub_component] = {"value": s.value, "confidence": s.confidence}

    ctx_result = await db.execute(select(ContextModulation).where(ContextModulation.twin_id == tid))
    ctx_mods = ctx_result.scalars().all()
    context_mod = {}
    for cm in ctx_mods:
        context_mod.setdefault(cm.context_type, {})[cm.sub_component] = {"override_value": cm.override_value, "confidence": cm.confidence}

    return {"personality_core": personality_core, "behavioral_instructions": behavioral, "context_modulation": context_mod, "guardrail_config": profile.guardrail_config or {}}


async def _build_knowledge_base(tid, db: AsyncSession) -> dict:
    rag_result = await db.execute(select(RagEntry).where(RagEntry.twin_id == tid, RagEntry.is_active == True).order_by(RagEntry.conviction.desc()))
    entries = rag_result.scalars().all()
    return {"entries": [{"content": e.content, "category": e.category, "topic": e.topic, "conviction": e.conviction} for e in entries], "total_entries": len(entries)}


async def _build_voice_identity(tid, db: AsyncSession) -> dict:
    vp_result = await db.execute(select(VoiceProfile).where(VoiceProfile.twin_id == tid))
    vp = vp_result.scalar_one_or_none()
    if not vp:
        return {}
    return {"speech_rate": vp.speech_rate, "pitch_range": vp.pitch_range, "accent_markers": vp.accent_markers, "filler_patterns": vp.filler_patterns, "prosodic_data": vp.prosodic_data, "tts_provider": vp.tts_provider, "tts_voice_id": vp.tts_voice_id, "tts_model_id": vp.tts_model_id}


async def _build_visual_identity(tid, db: AsyncSession) -> dict:
    vis_result = await db.execute(select(VisualProfile).where(VisualProfile.twin_id == tid))
    vis = vis_result.scalar_one_or_none()
    if not vis:
        return {}
    return {"appearance": vis.appearance, "expression_baselines": vis.expression_baselines, "gesture_patterns": vis.gesture_patterns, "mannerisms": vis.mannerisms, "likeness_hash": vis.likeness_hash}
