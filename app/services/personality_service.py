"""Personality Core derivation service.

Derives Big Five scores from dimensional sub-components,
MBTI from Big Five, and CCP from communication sub-components.

See Developer Guide Section 5 (Personality Core).
Derives Big Five from dimensional sub-components. Facet-level scoring (30 explicit facets) is a future enhancement.
"""
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dimensional_score import DimensionalScore
from app.models.personality_core import PersonalityCore


# Sub-components that feed each Big Five factor (spec Section 5.1)
BIG_FIVE_SOURCES = {
    "openness": {
        "positive": ["curiosity_level", "novelty_seeking", "abstraction_level"],
        "negative": ["change_resistance"],
    },
    "conscientiousness": {
        "positive": ["structure_preference", "linearity"],
        "negative": ["risk_tolerance"],
    },
    "extraversion": {
        "positive": ["social_orientation", "social_stamina", "expressiveness", "baseline_energy"],
        "negative": [],
    },
    "agreeableness": {
        "positive": ["trust_speed", "affective_empathy", "vulnerability_comfort"],
        "negative": ["confrontation_comfort", "directness"],
    },
    "neuroticism": {
        "positive": [],
        "negative": ["stress_threshold", "regulation_capacity", "baseline_valence"],
    },
}


async def derive_personality_core(twin_id: UUID, db: AsyncSession) -> PersonalityCore:
    """Derive Big Five, MBTI, and CCP from dimensional scores.

    Returns the updated PersonalityCore record.
    """
    # Load all dimensional scores
    result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == twin_id)
    )
    scores = result.scalars().all()

    scores_map = {s.sub_component: s for s in scores}

    # Derive Big Five
    big_five = {}
    total_observations = 0
    for factor, sources in BIG_FIVE_SOURCES.items():
        values = []
        confidences = []

        for sc in sources.get("positive", []):
            if sc in scores_map and scores_map[sc].confidence > 0.2:
                values.append(scores_map[sc].value)
                confidences.append(scores_map[sc].confidence)
                total_observations += scores_map[sc].observation_count

        for sc in sources.get("negative", []):
            if sc in scores_map and scores_map[sc].confidence > 0.2:
                values.append(100 - scores_map[sc].value)  # Invert
                confidences.append(scores_map[sc].confidence)
                total_observations += scores_map[sc].observation_count

        if values:
            score = sum(values) / len(values)
            confidence = sum(confidences) / len(confidences)
        else:
            score = 50.0
            confidence = 0.0

        big_five[factor] = {
            "score": round(score, 1),
            "confidence": round(confidence, 3),
            "std_error": round(30 * (1 - confidence), 1),
        }

    # Derive MBTI from Big Five (spec Section 5.2)
    mbti_type = ""
    mbti_type += "E" if big_five.get("extraversion", {}).get("score", 50) > 50 else "I"
    mbti_type += "N" if big_five.get("openness", {}).get("score", 50) > 50 else "S"
    mbti_type += "T" if big_five.get("agreeableness", {}).get("score", 50) < 50 else "F"
    mbti_type += "J" if big_five.get("conscientiousness", {}).get("score", 50) > 50 else "P"

    mbti = {
        "type": mbti_type,
        "dichotomies": {
            "extraversion_introversion": {"score": big_five.get("extraversion", {}).get("score", 50), "pole": mbti_type[0]},
            "sensing_intuition": {"score": big_five.get("openness", {}).get("score", 50), "pole": mbti_type[1]},
            "thinking_feeling": {"score": big_five.get("agreeableness", {}).get("score", 50), "pole": mbti_type[2]},
            "judging_perceiving": {"score": big_five.get("conscientiousness", {}).get("score", 50), "pole": mbti_type[3]},
        },
        "derivation_confidence": round(min(c.get("confidence", 0) for c in big_five.values()) if big_five else 0, 3),
    }

    # Derive CCP (spec Section 5.4)
    ccp = {
        "vocabulary_range": scores_map.get("vocabulary_level", type("", (), {"value": 50})).value,
        "explanation_depth": scores_map.get("verbosity", type("", (), {"value": 50})).value,
        "abstraction_comfort": scores_map.get("abstraction_level", type("", (), {"value": 50})).value,
        "reasoning_style": "hybrid",
        "confidence": 0.0,
    }
    if "vocabulary_level" in scores_map:
        ccp["confidence"] = scores_map["vocabulary_level"].confidence

    # Compute overall derivation confidence
    all_confidences = [v.get("confidence", 0) for v in big_five.values()]
    derivation_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    # Update or create personality core
    core_result = await db.execute(
        select(PersonalityCore).where(
            PersonalityCore.twin_id == twin_id,
            PersonalityCore.is_current == True,
        )
    )
    core = core_result.scalar_one_or_none()

    if core:
        core.big_five = big_five
        core.mbti = mbti
        core.cognitive_complexity = ccp
        core.derivation_confidence = round(derivation_confidence, 3)
        core.derived_from_observations = total_observations
    else:
        core = PersonalityCore(
            twin_id=twin_id,
            version=1,
            big_five=big_five,
            mbti=mbti,
            cognitive_complexity=ccp,
            derivation_confidence=round(derivation_confidence, 3),
            derived_from_observations=total_observations,
            is_current=True,
        )
        db.add(core)

    await db.flush()
    return core
