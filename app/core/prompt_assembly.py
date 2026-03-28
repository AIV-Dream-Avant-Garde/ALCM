"""7-layer prompt assembly for Score-to-Generation.

See Developer Guide Section 13 (Score-to-Generation Architecture).

Layer 1: System Identity (static per clone)        — max 4,000 tokens
Layer 2: Behavioral Parameters (dynamic per context) — max 2,000 tokens
Layer 3: Mood & State Modifiers (dynamic)           — max 500 tokens
Layer 4: RAG — User's Own Words (dynamic per query)  — max 3,000 tokens
Layer 5: Discourse Instructions (semi-static)        — max 1,500 tokens
Layer 6: Conversation History (sliding window)       — max 8,000 tokens
Layer 7: Current Input                               — max 4,000 tokens
Total context budget: max 27,000 tokens input
"""
import logging
from typing import Optional, List, Dict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.mood_state import get_mood_state

logger = logging.getLogger(__name__)

# Token budget per layer (approximate: 1 token ≈ 4 chars)
CHARS_PER_TOKEN = 4
LAYER_BUDGETS = {
    1: 4000 * CHARS_PER_TOKEN,   # 16,000 chars
    2: 2000 * CHARS_PER_TOKEN,   # 8,000 chars
    3: 500 * CHARS_PER_TOKEN,    # 2,000 chars
    4: 3000 * CHARS_PER_TOKEN,   # 12,000 chars
    5: 1500 * CHARS_PER_TOKEN,   # 6,000 chars
    6: 8000 * CHARS_PER_TOKEN,   # 32,000 chars
    "guardrails": 1000 * CHARS_PER_TOKEN,
}
MAX_TOTAL_CHARS = 27000 * CHARS_PER_TOKEN  # 108,000 chars

# Context types for detection
CONTEXT_KEYWORDS = {
    "PROFESSIONAL": ["meeting", "business", "quarterly", "revenue", "client", "deadline", "strategy"],
    "CASUAL": ["hey", "lol", "haha", "chill", "hang out", "weekend", "fun"],
    "FORMAL": ["dear", "sincerely", "respectfully", "hereby", "pursuant"],
    "CONFLICT": ["disagree", "wrong", "frustrated", "disappointed", "unacceptable", "complaint"],
    "CREATIVE": ["brainstorm", "imagine", "what if", "idea", "create", "design", "concept"],
    "INTIMATE": ["love", "miss you", "feeling", "worried about us", "relationship"],
}


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, preserving complete lines."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.7:
        return truncated[:last_newline]
    return truncated


def _detect_context_type(context: str, conversation_history: Optional[list] = None) -> str:
    """Detect situational context type from input signals.

    Returns: PROFESSIONAL | CASUAL | INTIMATE | FORMAL | CONFLICT | CREATIVE | PUBLIC
    """
    text = context.lower()
    if conversation_history:
        for msg in conversation_history[-3:]:
            text += " " + (msg.get("content", "") or "").lower()

    scores = {}
    for ctx_type, keywords in CONTEXT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[ctx_type] = score

    if scores:
        return max(scores, key=scores.get)
    return "PUBLIC"


async def assemble_prompt(
    twin_id: UUID,
    context: str,
    conversation_history: Optional[List[Dict]] = None,
    guardrails: Optional[dict] = None,
    mode: str = "CONVERSATION",
    deployment_scope: str = "TRAINING_AREA",
    db: AsyncSession = None,
) -> str:
    """Assemble the full system prompt from normalized tables with token budgets."""
    from app.models.twin_profile import TwinProfile

    layers = []
    total_chars = 0

    # --- Load twin profile ---
    result = await db.execute(select(TwinProfile).where(TwinProfile.id == twin_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return "You are a helpful assistant."

    # Detect context type
    context_type = _detect_context_type(context, conversation_history)

    # === LAYER 1: System Identity (max 4,000 tokens) ===
    layer_1 = await _build_layer_1_system_identity(twin_id, profile, db)
    layer_1 = _truncate(layer_1, LAYER_BUDGETS[1])
    layers.append(layer_1)
    total_chars += len(layer_1)

    # === LAYER 2: Behavioral Parameters with context modulation (max 2,000 tokens) ===
    layer_2 = await _build_layer_2_behavioral(twin_id, context_type, db)
    if layer_2:
        layer_2 = _truncate(layer_2, LAYER_BUDGETS[2])
        layers.append(layer_2)
        total_chars += len(layer_2)

    # === LAYER 3: Mood & State (max 500 tokens) ===
    mood = await get_mood_state(twin_id, db)
    layer_3 = _build_layer_3_mood(mood)
    if layer_3:
        layer_3 = _truncate(layer_3, LAYER_BUDGETS[3])
        layers.append(layer_3)
        total_chars += len(layer_3)

    # === LAYER 4: RAG + Episodic Memory (max 3,000 tokens) ===
    layer_4 = await _build_layer_4_rag_and_memory(twin_id, context, deployment_scope, db)
    if layer_4:
        layer_4 = _truncate(layer_4, LAYER_BUDGETS[4])
        layers.append(layer_4)
        total_chars += len(layer_4)

    # === LAYER 5: Discourse Instructions (max 1,500 tokens) ===
    layer_5 = await _build_layer_5_discourse(twin_id, db)
    if layer_5:
        layer_5 = _truncate(layer_5, LAYER_BUDGETS[5])
        layers.append(layer_5)
        total_chars += len(layer_5)

    # === LAYER 6: Conversation History (max 8,000 tokens) ===
    if conversation_history:
        layer_6 = _build_layer_6_conversation_history(conversation_history)
        if layer_6:
            layer_6 = _truncate(layer_6, LAYER_BUDGETS[6])
            layers.append(layer_6)
            total_chars += len(layer_6)

    # === GUARDRAILS (applied on top) ===
    effective_guardrails = guardrails or {}
    if profile.guardrail_config:
        merged = {**profile.guardrail_config, **effective_guardrails}
        effective_guardrails = merged
    guardrail_text = _build_guardrail_instructions(effective_guardrails)
    if guardrail_text:
        guardrail_text = _truncate(guardrail_text, LAYER_BUDGETS["guardrails"])
        layers.append(guardrail_text)
        total_chars += len(guardrail_text)

    # Final total budget check — compress if needed
    assembled = "\n\n".join(layers)
    if len(assembled) > MAX_TOTAL_CHARS:
        logger.warning(
            f"Prompt exceeds budget ({len(assembled)} chars > {MAX_TOTAL_CHARS}). "
            "Compressing conversation history and RAG."
        )
        assembled = _truncate(assembled, MAX_TOTAL_CHARS)

    return assembled


# ===== Layer Builders =====

async def _build_layer_1_system_identity(
    twin_id: UUID, profile, db: AsyncSession,
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
                    traits.append("highly open to new experiences" if score > 65 else "practical and conventional" if score < 35 else "balanced in openness")
                elif trait == "conscientiousness":
                    traits.append("organized and disciplined" if score > 65 else "flexible and spontaneous" if score < 35 else "moderately structured")
                elif trait == "extraversion":
                    traits.append("outgoing and energetic" if score > 65 else "reserved and reflective" if score < 35 else "balanced between social and private")
                elif trait == "agreeableness":
                    traits.append("compassionate and cooperative" if score > 65 else "direct and challenging" if score < 35 else "balanced in agreeableness")
                elif trait == "neuroticism":
                    traits.append("emotionally sensitive" if score > 65 else "emotionally stable and calm" if score < 35 else "moderate emotional reactivity")
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


async def _build_layer_2_behavioral(
    twin_id: UUID, context_type: str, db: AsyncSession,
) -> str:
    """Layer 2: Behavioral parameters with context-modulated values."""
    from app.models.dimensional_score import DimensionalScore
    from app.core.context_modulation import get_all_overrides

    result = await db.execute(
        select(DimensionalScore).where(
            DimensionalScore.twin_id == twin_id,
            DimensionalScore.confidence > 0.3,
        )
    )
    scores = result.scalars().all()
    if not scores:
        return ""

    # Load context overrides
    overrides = await get_all_overrides(twin_id, context_type, db)

    instructions = []
    for s in scores:
        # Apply context override if available
        value = overrides.get(s.sub_component, s.value)
        instruction = _score_to_instruction(s.dimension, s.sub_component, value)
        if instruction:
            instructions.append(instruction)

    if not instructions:
        return ""

    header = f"Behavioral guidelines (context: {context_type.lower()}):"
    return header + "\n" + "\n".join(f"- {i}" for i in instructions)


def _build_layer_3_mood(mood: dict) -> str:
    """Layer 3: Current mood state modifiers."""
    valence = mood.get("current_valence", 0)
    energy = mood.get("current_energy", 50)

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


async def _build_layer_4_rag_and_memory(
    twin_id: UUID, context: str, deployment_scope: str, db: AsyncSession,
) -> str:
    """Layer 4: RAG entries + episodic memory for contextual grounding."""
    from app.services.rag_service import retrieve_relevant_entries
    from app.services.memory_service import get_relevant_episodes

    parts = []

    # RAG entries — uses vector similarity search when embeddings available,
    # falls back to conviction-based retrieval
    entries = await retrieve_relevant_entries(twin_id, context, limit=5, db=db)
    if entries:
        rag_lines = []
        for entry in entries:
            label = entry.category or "reference"
            rag_lines.append(f"[{label}] {entry.content}")
        parts.append("Relevant knowledge and positions:\n" + "\n".join(rag_lines))

    # Episodic memory (top 3 recent episodes)
    episodes = await get_relevant_episodes(twin_id, deployment_scope, limit=3, db=db)
    if episodes:
        ep_lines = []
        for ep in episodes:
            date_str = ep.interaction_at.strftime("%Y-%m-%d") if ep.interaction_at else "unknown"
            ep_lines.append(f"[{date_str}] {ep.summary}")
        parts.append("Recent relevant interactions:\n" + "\n".join(ep_lines))

    return "\n\n".join(parts) if parts else ""


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


def _build_layer_6_conversation_history(conversation_history: list) -> str:
    """Layer 6: Sliding window of conversation history (FIFO trimming)."""
    if not conversation_history:
        return ""

    lines = []
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")

    if not lines:
        return ""

    return "Conversation history:\n" + "\n".join(lines)


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
    """Translate a dimensional score to a natural-language instruction."""
    # Cognitive dimension
    if sub_component == "formality":
        if value > 70: return "Use formal, professional language."
        elif value < 30: return "Use casual, conversational language."
    elif sub_component == "verbosity":
        if value > 70: return "Provide detailed, thorough responses."
        elif value < 30: return "Keep responses brief and concise."
    elif sub_component == "humor_frequency":
        if value > 60: return "Include humor naturally in responses."
        elif value < 20: return "Maintain a serious tone; avoid humor."
    elif sub_component == "structure_preference":
        if value > 70: return "Structure responses with clear organization."
        elif value < 30: return "Keep responses free-flowing and conversational."
    elif sub_component == "abstraction_level":
        if value > 70: return "Comfortable with abstract concepts and frameworks."
        elif value < 30: return "Stay concrete and practical in explanations."
    # Emotional dimension
    elif sub_component == "expressiveness":
        if value > 70: return "Express emotions openly and warmly."
        elif value < 30: return "Maintain emotional restraint."
    elif sub_component == "baseline_valence":
        if value > 30: return "Default to an upbeat, positive tone."
        elif value < -30: return "Default to a more serious, measured tone."
    elif sub_component == "vulnerability_comfort":
        if value > 70: return "Comfortable showing vulnerability and personal feelings."
        elif value < 30: return "Keep personal feelings private; stay composed."
    # Social dimension
    elif sub_component == "directness":
        if value > 70: return "Be direct and straightforward."
        elif value < 30: return "Use diplomatic, indirect communication."
    elif sub_component == "listen_speak_ratio":
        if value > 0.7: return "Take the lead in conversations."
        elif value < 0.3: return "Ask questions and listen more than speak."
    elif sub_component == "trust_speed":
        if value > 70: return "Be open and trusting in communication."
        elif value < 30: return "Be measured and cautious with new contacts."
    elif sub_component == "conflict_style":
        pass  # Categorical — handled differently
    # Evolutionary dimension
    elif sub_component == "risk_tolerance":
        if value > 70: return "Comfortable with bold, risk-taking positions."
        elif value < 30: return "Prefer cautious, well-considered approaches."
    elif sub_component == "change_resistance":
        if value > 70: return "Prefer stability and proven approaches."
        elif value < 30: return "Embrace change and new ways of doing things."

    return ""
