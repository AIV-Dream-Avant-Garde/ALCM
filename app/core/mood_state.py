"""Dynamic Mood State system.

Implements mood shifts, decay, emotional contagion, and mood-to-behavior
modulation per Developer Guide Section 4.2. Mood is a transient state
(not a trait) that influences all dimensional sub-components.
"""
import time
import logging
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Mood-to-behavior modulation table (Developer Guide Section 4.2)
# Format: (mood_axis, condition, sub_component, sensitivity, direction)
MOOD_MODULATION_TABLE = [
    # Valence (negative)
    ("valence", "negative", "formality", 8, -1),
    ("valence", "negative", "humor_frequency", 12, 1),  # decreases humor
    ("valence", "negative", "expressiveness", 10, 1),  # decreases
    ("valence", "negative", "generation_confidence", 0.08, -1),
    ("valence", "negative", "risk_tolerance", 8, 1),  # decreases
    # Valence (positive)
    ("valence", "positive", "humor_frequency", 8, 1),
    ("valence", "positive", "expressiveness", 8, 1),
    ("valence", "positive", "risk_tolerance", 6, 1),
    # Energy (high/low — symmetric via signed deviation)
    ("energy", "any", "verbosity", 10, 1),
    ("energy", "any", "directness", 6, 1),
    ("energy", "any", "speech_rate_wpm", 15, 1),
    # Energy (low only)
    ("energy", "low", "listen_speak_ratio", 0.1, 1),
]

# Max mood shift per session (spec Section 20.6)
MAX_SESSION_MOOD_SHIFT = 30

# Modifier caps (spec Section 4.2)
MODIFIER_CAP_STANDARD = 20  # for 0-100 scale sub-components
MODIFIER_CAP_RATIO = 0.20  # for 0.0-1.0 scale sub-components

RATIO_SCALE_COMPONENTS = {"generation_confidence", "listen_speak_ratio", "learning_rate"}


# In-memory mood cache per twin per session (production: Redis)
_mood_cache = {}


async def get_mood_state(twin_id: UUID, db=None) -> dict:
    """Get current mood state for a twin. Returns cached session mood or baseline."""
    key = str(twin_id)
    if key in _mood_cache:
        mood = _mood_cache[key]
        # Apply decay based on time elapsed
        _apply_decay(mood)
        return mood

    # Load baseline from dimensional scores if available
    baseline = await _load_baseline(twin_id, db)
    mood = {
        "baseline_valence": baseline.get("baseline_valence", 0),
        "baseline_energy": baseline.get("baseline_energy", 50),
        "current_valence": baseline.get("baseline_valence", 0),
        "current_energy": baseline.get("baseline_energy", 50),
        "mood_triggers_active": [],
        "last_mood_update": time.time(),
        "session_start_valence": baseline.get("baseline_valence", 0),
        "recovery_speed": baseline.get("recovery_speed", 50),
        "regulation_capacity": baseline.get("regulation_capacity", 50),
        "affective_empathy": baseline.get("affective_empathy", 50),
    }
    _mood_cache[key] = mood
    return mood


def shift_mood(
    twin_id: UUID,
    valence_delta: float = 0,
    energy_delta: float = 0,
    trigger: Optional[str] = None,
    trigger_intensity: float = 50,
    is_contagion: bool = False,
) -> dict:
    """Apply a mood shift from a trigger or emotional contagion.

    Mood shifts are dampened by regulation_capacity.
    Contagion shifts are weighted by affective_empathy.
    Total session shift is bounded by MAX_SESSION_MOOD_SHIFT.
    """
    key = str(twin_id)
    mood = _mood_cache.get(key)
    if not mood:
        return {"current_valence": 0, "current_energy": 50}

    regulation = mood.get("regulation_capacity", 50) / 100
    empathy = mood.get("affective_empathy", 50) / 100

    # Dampen by regulation capacity
    effective_valence = valence_delta * (1 - regulation) * (trigger_intensity / 100)
    effective_energy = energy_delta * (1 - regulation) * (trigger_intensity / 100)

    # Weight by empathy for contagion
    if is_contagion:
        effective_valence *= empathy
        effective_energy *= empathy

    # Bound total session shift
    session_start = mood.get("session_start_valence", 0)
    proposed_valence = mood["current_valence"] + effective_valence
    if abs(proposed_valence - session_start) > MAX_SESSION_MOOD_SHIFT:
        if proposed_valence > session_start:
            proposed_valence = session_start + MAX_SESSION_MOOD_SHIFT
        else:
            proposed_valence = session_start - MAX_SESSION_MOOD_SHIFT

    mood["current_valence"] = max(-100, min(100, proposed_valence))
    mood["current_energy"] = max(0, min(100, mood["current_energy"] + effective_energy))
    mood["last_mood_update"] = time.time()

    if trigger:
        mood["mood_triggers_active"].append({
            "trigger": trigger,
            "timestamp": time.time(),
            "intensity": trigger_intensity,
        })

    return mood


def compute_mood_modifiers(mood: dict) -> dict:
    """Compute transient mood modifiers for all affected sub-components.

    Returns dict of sub_component -> modifier value. Applied on top of
    context-modulated values during prompt assembly.
    """
    baseline_valence = mood.get("baseline_valence", 0)
    baseline_energy = mood.get("baseline_energy", 50)
    current_valence = mood.get("current_valence", 0)
    current_energy = mood.get("current_energy", 50)

    valence_deviation = current_valence - baseline_valence
    energy_deviation = current_energy - baseline_energy

    modifiers = {}

    for axis, condition, sub_component, sensitivity, direction in MOOD_MODULATION_TABLE:
        if axis == "valence":
            deviation = valence_deviation
            if condition == "negative" and deviation >= 0:
                continue
            if condition == "positive" and deviation <= 0:
                continue
        elif axis == "energy":
            deviation = energy_deviation
            if condition == "low" and deviation >= 0:
                continue
            # "any" applies to both directions
        else:
            continue

        modifier = (deviation / 100) * sensitivity * direction

        # Apply caps
        if sub_component in RATIO_SCALE_COMPONENTS:
            modifier = max(-MODIFIER_CAP_RATIO, min(MODIFIER_CAP_RATIO, modifier))
        else:
            modifier = max(-MODIFIER_CAP_STANDARD, min(MODIFIER_CAP_STANDARD, modifier))

        modifiers[sub_component] = round(modifier, 3)

    return modifiers


def _apply_decay(mood: dict):
    """Decay mood back toward baseline based on elapsed time and recovery_speed."""
    last_update = mood.get("last_mood_update", 0)
    elapsed = time.time() - last_update
    if elapsed < 1:
        return

    recovery_speed = mood.get("recovery_speed", 50) / 100
    decay_factor = min(1.0, elapsed * recovery_speed / 300)  # Full decay in ~5 min at speed=100

    baseline_v = mood.get("baseline_valence", 0)
    baseline_e = mood.get("baseline_energy", 50)

    mood["current_valence"] += (baseline_v - mood["current_valence"]) * decay_factor
    mood["current_energy"] += (baseline_e - mood["current_energy"]) * decay_factor
    mood["last_mood_update"] = time.time()

    # Clean expired triggers (older than 10 minutes)
    cutoff = time.time() - 600
    mood["mood_triggers_active"] = [
        t for t in mood.get("mood_triggers_active", [])
        if t.get("timestamp", 0) > cutoff
    ]


async def _load_baseline(twin_id: UUID, db) -> dict:
    """Load emotional baseline values from dimensional scores."""
    if db is None:
        return {}

    try:
        from sqlalchemy import select
        from app.models.dimensional_score import DimensionalScore
        result = await db.execute(
            select(DimensionalScore).where(
                DimensionalScore.twin_id == twin_id,
                DimensionalScore.dimension == "EMOTIONAL",
            )
        )
        scores = result.scalars().all()
        return {s.sub_component: s.value for s in scores}
    except Exception:
        return {}
