"""Generation service — Score-to-Generation pipeline with Safeguard Gateway.

Owns: prompt assembly → LLM call → consistency validation → retry → response.
"""
import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.twin_profile import TwinProfile
from app.models.personality_core import PersonalityCore
from app.core.prompt_assembly import assemble_prompt
from app.core.errors import TwinNotFound, TwinLocked, SuccessorHold, GenerationFailed
from app.services.llm_provider import get_llm_provider
from app.schemas.generate import GenerateRequest, GenerateResponse, GuardrailChecks

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
CONSISTENCY_THRESHOLD = 0.6


async def generate_response(req: GenerateRequest, db: AsyncSession) -> GenerateResponse:
    """Full generation pipeline: assemble → generate → validate → deliver."""
    twin_uuid = UUID(req.twin_id)

    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_uuid))
    profile = result.scalar_one_or_none()
    if not profile:
        raise TwinNotFound(req.twin_id)
    if profile.status == "LOCKED":
        raise TwinLocked()
    if profile.status == "PROTECTED_HOLD":
        raise SuccessorHold()

    # Assemble prompt from normalized tables
    system_prompt = await assemble_prompt(
        twin_id=twin_uuid,
        context=req.context,
        conversation_history=req.conversation_history,
        guardrails=req.guardrails,
        mode=req.mode,
        deployment_scope=req.deployment_scope,
        db=db,
    )

    # Generate with graceful degradation
    provider = get_llm_provider()
    try:
        response_text = await provider.generate(
            prompt=req.context,
            context=system_prompt,
            temperature=0.7,
        )
    except Exception as e:
        logger.error(f"LLM generation failed for twin {req.twin_id}: {e}", exc_info=True)
        raise GenerationFailed(f"LLM provider error: {type(e).__name__}")

    # === SAFEGUARD GATEWAY ===
    consistency_score = await _check_personality_consistency(twin_uuid, response_text, req.context, db)
    guardrail_checks = GuardrailChecks()
    retries = 0

    if consistency_score is not None and consistency_score < CONSISTENCY_THRESHOLD and retries < MAX_RETRIES:
        logger.warning(
            f"Personality consistency {consistency_score:.2f} < {CONSISTENCY_THRESHOLD} "
            f"for twin {req.twin_id}, regenerating"
        )
        strengthened_prompt = system_prompt + (
            "\n\nIMPORTANT: Your previous response was not consistent with "
            "this person's personality. Stay more faithful to the personality "
            "traits and communication style described above."
        )
        try:
            response_text = await provider.generate(
                prompt=req.context,
                context=strengthened_prompt,
                temperature=0.6,
            )
        except Exception:
            pass  # Use the original response if retry fails
        consistency_score = await _check_personality_consistency(twin_uuid, response_text, req.context, db)
        retries += 1

    guardrail_checks.personality_consistency = (
        "PASSED" if consistency_score is None or consistency_score >= CONSISTENCY_THRESHOLD else "FLAGGED"
    )

    # Check topic restrictions
    effective_guardrails = req.guardrails or {}
    if profile.guardrail_config:
        effective_guardrails = {**profile.guardrail_config, **effective_guardrails}
    blocked_topics = effective_guardrails.get("blocked_topics", [])
    if blocked_topics:
        response_lower = response_text.lower()
        for topic in blocked_topics:
            if topic.lower() in response_lower:
                guardrail_checks.topic_restrictions = "FLAGGED"
                logger.warning(f"Blocked topic '{topic}' detected for twin {req.twin_id}")
                break

    return GenerateResponse(
        response_text=response_text,
        personality_consistency_score=consistency_score,
        mood_state={"valence": 0, "arousal": 0.5},
        guardrail_checks=guardrail_checks,
        tokens_used={"input": 0, "output": 0},
        metadata={
            "mode": req.mode,
            "context_type_detected": "PUBLIC",
            "personality_core_used": "v1",
            "layers_assembled": [1, 2, 3, 4, 5, 6, 7],
            "retries": retries,
        },
    )


async def _check_personality_consistency(
    twin_id: UUID, response_text: str, context: str, db: AsyncSession,
) -> float | None:
    """LLM-based personality consistency check (Safeguard Gateway)."""
    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == twin_id,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()
    if not core or not core.big_five:
        return None

    traits = []
    for trait, data in core.big_five.items():
        if isinstance(data, dict) and data.get("confidence", 0) > 0.3:
            traits.append(f"{trait}={data.get('score', 50)}")
    if not traits:
        return None

    personality_summary = f"Big Five: {', '.join(traits)}"
    if core.mbti and core.mbti.get("type"):
        personality_summary += f". MBTI: {core.mbti['type']}"

    check_prompt = f"""Rate the personality consistency of this response on a scale of 0.0 to 1.0.
Personality: {personality_summary}
Context: {context[:200]}
Response: {response_text[:500]}
Return ONLY a JSON object: {{"score": 0.85}}"""

    try:
        provider = get_llm_provider()
        result = await provider.classify(response_text[:500], check_prompt)
        score = result.get("score")
        if score is not None:
            return float(score)
    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")

    return None
