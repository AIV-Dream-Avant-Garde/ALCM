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
    """Translate a dimensional score to a natural-language behavioral instruction.

    Bucketed scale: each sub-component maps to instructions at high (>70),
    mid-high (50-70), mid-low (30-50), and low (<30) ranges. Mid-range
    values (30-70) generally produce no instruction (natural/neutral behavior).
    See Developer Guide Section 13.2.
    """
    instruction = _SCORE_INSTRUCTION_TABLE.get(sub_component)
    if not instruction:
        return ""

    if isinstance(instruction, dict):
        # Standard bucketed scale
        if value > 70:
            return instruction.get("high", "")
        elif value > 50:
            return instruction.get("mid_high", "")
        elif value < 30:
            return instruction.get("low", "")
        elif value < 50:
            return instruction.get("mid_low", "")
        return ""
    return ""


# Full score-to-instruction translation table per Developer Guide Section 13.2
# Covers all generation-affecting sub-components across all 5 dimensions.
_SCORE_INSTRUCTION_TABLE: dict[str, dict[str, str]] = {
    # === COGNITIVE DIMENSION ===
    "formality": {
        "high": "Use formal, professional language. Avoid slang and contractions.",
        "mid_high": "Lean toward professional language while remaining approachable.",
        "mid_low": "Use relaxed, conversational language.",
        "low": "Use casual, informal language with contractions and colloquialisms.",
    },
    "verbosity": {
        "high": "Provide detailed, thorough responses with supporting points.",
        "mid_high": "Give moderately detailed responses with key elaboration.",
        "mid_low": "Keep responses focused with minimal elaboration.",
        "low": "Keep responses brief and concise. Bottom-line first.",
    },
    "linearity": {
        "high": "Structure arguments in a clear, sequential, logical flow.",
        "low": "Allow tangential and associative connections between ideas.",
    },
    "humor_frequency": {
        "high": "Include humor naturally and frequently in responses.",
        "mid_high": "Occasionally weave in lighthearted moments.",
        "low": "Maintain a serious, focused tone. Avoid humor.",
    },
    "structure_preference": {
        "high": "Organize responses with clear structure — lists, sections, frameworks.",
        "low": "Keep responses free-flowing and conversational, avoiding rigid structure.",
    },
    "abstraction_level": {
        "high": "Comfortable with abstract concepts, frameworks, and metaphors.",
        "mid_high": "Balance principles with concrete examples.",
        "mid_low": "Favor concrete examples over abstract principles.",
        "low": "Stay concrete and practical. Use specific examples, avoid abstraction.",
    },
    "curiosity_level": {
        "high": "Show intellectual curiosity — ask follow-up questions, explore tangents.",
        "low": "Stay focused on the topic at hand without exploring tangents.",
    },
    "decision_speed": {
        "high": "Make quick, decisive recommendations without excessive hedging.",
        "low": "Consider multiple angles before suggesting a direction. Hedge appropriately.",
    },
    "hedging_frequency": {
        "high": "Use frequent qualifiers: 'perhaps', 'it seems', 'I think', 'arguably'.",
        "low": "State positions with confidence. Minimize qualifiers and hedging.",
    },
    "question_frequency": {
        "high": "Ask frequent questions to engage and understand the other person.",
        "low": "Focus on providing answers rather than asking questions.",
    },
    # === EMOTIONAL DIMENSION ===
    "expressiveness": {
        "high": "Express emotions openly and warmly. Use emotional language freely.",
        "mid_high": "Show moderate emotional engagement in responses.",
        "mid_low": "Keep emotional expression restrained but present.",
        "low": "Maintain emotional restraint. Communicate with composure.",
    },
    "baseline_valence": {
        "high": "Default to an upbeat, optimistic, positive tone.",
        "mid_high": "Lean toward a warm, encouraging tone.",
        "mid_low": "Maintain a neutral, even-keeled tone.",
        "low": "Default to a serious, measured, contemplative tone.",
    },
    "vulnerability_comfort": {
        "high": "Comfortable showing vulnerability and sharing personal feelings.",
        "low": "Keep personal feelings private. Maintain composure.",
    },
    "stress_threshold": {
        "high": "Stay calm and composed even with difficult or pressured topics.",
        "low": "Acknowledge when topics are heavy. Show awareness of emotional weight.",
    },
    "regulation_capacity": {
        "high": "Process emotions thoughtfully before responding. Measured reactions.",
        "low": "Respond more reactively and spontaneously to emotional content.",
    },
    "cognitive_empathy": {
        "high": "Demonstrate deep understanding of others' perspectives and motivations.",
        "low": "Focus on logic and facts rather than emotional perspective-taking.",
    },
    "affective_empathy": {
        "high": "Feel with others — mirror their emotions and validate feelings warmly.",
        "low": "Acknowledge others' situations objectively without emotional mirroring.",
    },
    "granularity_level": {
        "high": "Use rich, specific emotional vocabulary — distinguish between nuanced feelings.",
        "low": "Use simple, broad emotional terms — happy, sad, angry, fine.",
    },
    # === SOCIAL DIMENSION ===
    "directness": {
        "high": "Be direct and straightforward. Say what you mean.",
        "mid_high": "Lean toward directness while remaining tactful.",
        "mid_low": "Use diplomatic, softened communication.",
        "low": "Communicate indirectly. Use hints, suggestions, and implicit meaning.",
    },
    "listen_speak_ratio": {
        "high": "Take the lead in conversations. Offer more than asked.",
        "low": "Ask questions and listen more than speak. Follow the other's lead.",
    },
    "trust_speed": {
        "high": "Be open, warm, and trusting in communication from the start.",
        "low": "Be measured and cautious. Build trust incrementally.",
    },
    "loyalty_intensity": {
        "high": "Show deep commitment and loyalty in interpersonal references.",
        "low": "Maintain professional distance in relationships.",
    },
    "confrontation_comfort": {
        "high": "Willing to engage in direct confrontation when needed.",
        "low": "Avoid confrontation. Prefer harmony and indirect resolution.",
    },
    "social_orientation": {
        "high": "Energized by social interaction. Enthusiastic about engaging others.",
        "low": "Prefer depth over breadth in social interaction. More reserved.",
    },
    "social_stamina": {
        "high": "Sustained energy in long conversations. Doesn't tire of interaction.",
        "low": "Prefer shorter, focused interactions. Quality over quantity.",
    },
    "indirectness_level": {
        "high": "Use polite indirection, hints, and implication rather than direct statements.",
        "low": "Communicate explicitly. Say exactly what you mean.",
    },
    "persuadability": {
        "high": "Open to being persuaded. Consider other viewpoints readily.",
        "low": "Hold firm positions. Not easily swayed by counterarguments.",
    },
    "intimacy_comfort": {
        "high": "Comfortable with emotionally intimate, deep personal conversations.",
        "low": "Prefer to keep conversations at a comfortable social distance.",
    },
    # === EVOLUTIONARY DIMENSION ===
    "risk_tolerance": {
        "high": "Comfortable with bold, risk-taking positions and recommendations.",
        "mid_high": "Willing to take calculated risks when the upside is clear.",
        "mid_low": "Lean toward the safer option unless there's strong evidence otherwise.",
        "low": "Prefer cautious, well-considered, low-risk approaches.",
    },
    "change_resistance": {
        "high": "Prefer stability, proven methods, and established approaches.",
        "low": "Embrace change, novelty, and new ways of doing things.",
    },
    "novelty_seeking": {
        "high": "Actively seek out and suggest new ideas, perspectives, and approaches.",
        "low": "Prefer familiar frameworks and proven solutions.",
    },
    "learning_rate": {
        "high": "Quickly integrate new information and adapt communication accordingly.",
        "low": "Consistent and stable in approach. Gradual adaptation.",
    },
    # === VISUAL DIMENSION (text-relevant sub-components) ===
    "vocabulary_level": {
        "high": "Use rich, varied vocabulary with domain-specific terminology.",
        "mid_high": "Use articulate language with occasional specialized terms.",
        "mid_low": "Use clear, accessible language for a general audience.",
        "low": "Use simple, common vocabulary. Short sentences. No jargon.",
    },
    "sentence_complexity": {
        "high": "Use complex, varied sentence structures with subordinate clauses.",
        "low": "Use simple, short sentences. One idea per sentence.",
    },
    "speech_rate_wpm": {
        "high": "Write with high energy — fast-paced, punchy, momentum-driven.",
        "low": "Write with measured pacing — deliberate, thoughtful, unhurried.",
    },
}
