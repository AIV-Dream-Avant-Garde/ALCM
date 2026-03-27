"""7-layer prompt assembly for Score-to-Generation.

Replaces the inline _build_identity_prompt in generate.py.
See Developer Guide Section 13 (Score-to-Generation Architecture).

Layer 1: System Identity (static per clone)        — max 4,000 tokens
Layer 2: Behavioral Parameters (dynamic per context) — max 2,000 tokens
Layer 3: Mood & State Modifiers (dynamic)           — max 500 tokens
Layer 4: RAG — User's Own Words (dynamic per query)  — max 3,000 tokens
Layer 5: Discourse Instructions (semi-static)        — max 1,500 tokens
Layer 6: Conversation History (sliding window)       — max 8,000 tokens
Layer 7: Current Input                               — max 4,000 tokens
"""
import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.mood_state import get_mood_state

logger = logging.getLogger(__name__)


async def assemble_prompt(
    twin_id: UUID,
    context: str,
    conversation_history: Optional[list] = None,
    guardrails: Optional[dict] = None,
    mode: str = "CONVERSATION",
    deployment_scope: str = "TRAINING_AREA",
    db: AsyncSession = None,
) -> str:
    """Assemble the full system prompt from normalized tables.

    Phase 1: Loads personality core, dimensional scores, RAG entries,
    and guardrails. Mood defaults to neutral. Context modulation basic.
    """
    from app.models.twin_profile import TwinProfile
    from app.models.personality_core import PersonalityCore
    from app.models.dimensional_score import DimensionalScore
    from app.models.rag_entry import RagEntry
    from app.models.score_to_gen_instruction import ScoreToGenInstruction

    layers = []

    # --- Load twin profile ---
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return "You are a helpful assistant."

    # === LAYER 1: System Identity ===
    layer_1 = await _build_layer_1_system_identity(twin_id, profile, db)
    layers.append(layer_1)

    # === LAYER 2: Behavioral Parameters ===
    layer_2 = await _build_layer_2_behavioral(twin_id, db)
    if layer_2:
        layers.append(layer_2)

    # === LAYER 3: Mood & State ===
    mood = await get_mood_state(twin_id, db)
    layer_3 = _build_layer_3_mood(mood)
    if layer_3:
        layers.append(layer_3)

    # === LAYER 4: RAG — User's Own Words ===
    layer_4 = await _build_layer_4_rag(twin_id, context, db)
    if layer_4:
        layers.append(layer_4)

    # === LAYER 5: Discourse Instructions ===
    layer_5 = await _build_layer_5_discourse(twin_id, db)
    if layer_5:
        layers.append(layer_5)

    # === GUARDRAILS (applied on top) ===
    effective_guardrails = guardrails or {}
    if profile.guardrail_config:
        # Merge: request guardrails override stored config
        merged = {**profile.guardrail_config, **effective_guardrails}
        effective_guardrails = merged

    guardrail_text = _build_guardrail_instructions(effective_guardrails)
    if guardrail_text:
        layers.append(guardrail_text)

    return "\n\n".join(layers)


async def _build_layer_1_system_identity(
    twin_id: UUID, profile, db: AsyncSession
) -> str:
    """Layer 1: Natural-language personality description from Personality Core."""
    from app.models.personality_core import PersonalityCore

    result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == twin_id,
            PersonalityCore.is_current == True,
        )
    )
    core = result.scalar_one_or_none()

    parts = [
        "You are embodying a specific identity. Stay faithful to this person's "
        "personality, knowledge, values, and communication style."
    ]

    if core and core.big_five:
        big_five = core.big_five
        traits = []
        for trait, data in big_five.items():
            if isinstance(data, dict) and data.get("confidence", 0) > 0.3:
                score = data.get("score", 50)
                if trait == "openness":
                    traits.append(f"{'highly open to new experiences' if score > 65 else 'practical and conventional' if score < 35 else 'balanced in openness'}")
                elif trait == "conscientiousness":
                    traits.append(f"{'organized and disciplined' if score > 65 else 'flexible and spontaneous' if score < 35 else 'moderately structured'}")
                elif trait == "extraversion":
                    traits.append(f"{'outgoing and energetic' if score > 65 else 'reserved and reflective' if score < 35 else 'balanced between social and private'}")
                elif trait == "agreeableness":
                    traits.append(f"{'compassionate and cooperative' if score > 65 else 'direct and challenging' if score < 35 else 'balanced in agreeableness'}")
                elif trait == "neuroticism":
                    traits.append(f"{'emotionally sensitive' if score > 65 else 'emotionally stable and calm' if score < 35 else 'moderate emotional reactivity'}")
        if traits:
            parts.append(f"Personality traits: {', '.join(traits)}.")

    if core and core.mbti:
        mbti_type = core.mbti.get("type")
        if mbti_type:
            parts.append(f"Cognitive style aligns with {mbti_type} patterns.")

    if core and core.cognitive_complexity:
        ccp = core.cognitive_complexity
        vocab = ccp.get("vocabulary_range", 50)
        if vocab > 70:
            parts.append("Uses rich, varied vocabulary with domain-specific terminology.")
        elif vocab < 30:
            parts.append("Uses simple, accessible language.")

    return "\n".join(parts)


async def _build_layer_2_behavioral(twin_id: UUID, db: AsyncSession) -> str:
    """Layer 2: Behavioral parameters from dimensional scores."""
    from app.models.dimensional_score import DimensionalScore

    result = await db.execute(
        select(DimensionalScore).where(
            DimensionalScore.twin_id == twin_id,
            DimensionalScore.confidence > 0.3,
        )
    )
    scores = result.scalars().all()
    if not scores:
        return ""

    instructions = []
    for s in scores:
        instruction = _score_to_instruction(s.dimension, s.sub_component, s.value)
        if instruction:
            instructions.append(instruction)

    if not instructions:
        return ""

    return "Behavioral guidelines:\n" + "\n".join(f"- {i}" for i in instructions)


def _build_layer_3_mood(mood: dict) -> str:
    """Layer 3: Current mood state modifiers."""
    valence = mood.get("current_valence", 0)
    energy = mood.get("current_energy", 50)

    # Phase 1: Only apply if mood deviates significantly from neutral
    if abs(valence) < 10 and abs(energy - 50) < 15:
        return ""

    parts = []
    if valence > 20:
        parts.append("Current mood is positive and upbeat.")
    elif valence < -20:
        parts.append("Current mood is more reserved and serious.")
    if energy > 70:
        parts.append("Energy level is high — more animated and verbose.")
    elif energy < 30:
        parts.append("Energy level is low — more concise and measured.")

    return "Current state: " + " ".join(parts) if parts else ""


async def _build_layer_4_rag(twin_id: UUID, context: str, db: AsyncSession) -> str:
    """Layer 4: Relevant RAG entries (user's own words, positions, expertise).

    Phase 1: Simple keyword matching. Phase 2: Vector similarity search.
    """
    from app.models.rag_entry import RagEntry

    # Phase 1: Load most recent active RAG entries (limited to 5)
    result = await db.execute(
        select(RagEntry)
        .where(RagEntry.twin_id == twin_id, RagEntry.is_active == True)
        .order_by(RagEntry.conviction.desc())
        .limit(5)
    )
    entries = result.scalars().all()
    if not entries:
        return ""

    rag_texts = []
    for entry in entries:
        label = entry.category or "reference"
        rag_texts.append(f"[{label}] {entry.content}")

    return "Relevant knowledge and positions:\n" + "\n".join(rag_texts)


async def _build_layer_5_discourse(twin_id: UUID, db: AsyncSession) -> str:
    """Layer 5: Discourse instructions from cached score-to-gen instructions."""
    from app.models.score_to_gen_instruction import ScoreToGenInstruction

    result = await db.execute(
        select(ScoreToGenInstruction).where(
            ScoreToGenInstruction.twin_id == twin_id,
            ScoreToGenInstruction.layer == 5,
            ScoreToGenInstruction.is_current == True,
        )
    )
    instruction = result.scalar_one_or_none()
    return instruction.instruction_text if instruction else ""


def _build_guardrail_instructions(guardrails: dict) -> str:
    """Build guardrail enforcement text."""
    if not guardrails:
        return ""

    parts = []
    blocked = guardrails.get("blocked_topics", [])
    if blocked:
        parts.append(f"NEVER discuss these topics: {', '.join(blocked)}.")

    restricted = guardrails.get("restricted_topics", {})
    for topic, rule in restricted.items():
        parts.append(f"Topic '{topic}': {rule}.")

    if guardrails.get("require_ai_disclosure", False):
        disclosure = guardrails.get(
            "disclosure_text",
            "This is an AI-generated response."
        )
        parts.append(f"If asked whether you are AI, acknowledge: {disclosure}")

    max_controversy = guardrails.get("max_controversy")
    if max_controversy is not None and max_controversy < 30:
        parts.append("Avoid controversial statements. Stay neutral on divisive topics.")

    return "Guardrails:\n" + "\n".join(f"- {p}" for p in parts) if parts else ""


def _score_to_instruction(dimension: str, sub_component: str, value: float) -> str:
    """Translate a dimensional score to a natural-language instruction.

    Phase 1: Key sub-components only. Phase 2: Full translation tables.
    """
    # Cognitive dimension
    if sub_component == "formality":
        if value > 70:
            return "Use formal, professional language."
        elif value < 30:
            return "Use casual, conversational language."
    elif sub_component == "verbosity":
        if value > 70:
            return "Provide detailed, thorough responses."
        elif value < 30:
            return "Keep responses brief and concise."
    elif sub_component == "humor_frequency":
        if value > 60:
            return "Include humor naturally in responses."
        elif value < 20:
            return "Maintain a serious tone; avoid humor."

    # Emotional dimension
    elif sub_component == "expressiveness":
        if value > 70:
            return "Express emotions openly and warmly."
        elif value < 30:
            return "Maintain emotional restraint."

    # Social dimension
    elif sub_component == "directness":
        if value > 70:
            return "Be direct and straightforward."
        elif value < 30:
            return "Use diplomatic, indirect communication."
    elif sub_component == "listen_speak_ratio":
        if value > 0.7:
            return "Take the lead in conversations."
        elif value < 0.3:
            return "Ask questions and listen more than speak."

    return ""
