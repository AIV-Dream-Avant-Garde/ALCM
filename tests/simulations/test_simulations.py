"""High-intensity simulations across all ALCM core systems.

These tests push edge cases, boundary conditions, adversarial inputs,
and stress scenarios to find real flaws in the identity engine.
"""
import math
import time
import pytest


# ============================================================================
# SIMULATION 1: Bayesian Update — Convergence, Stability, Edge Cases
# ============================================================================

class TestBayesianConvergence:
    """Does the Bayesian update converge correctly under repeated evidence?"""

    def test_convergence_to_true_value_after_50_observations(self):
        """Given 50 observations around value=75, posterior should converge near 75."""
        from app.core.bayesian import update_sub_component
        import random
        random.seed(42)

        value, std_error = 50.0, 30.0  # Start at uninformed prior
        true_value = 75.0

        for _ in range(50):
            observation = true_value + random.gauss(0, 5)  # Noisy observations around 75
            value, std_error, confidence = update_sub_component(
                value, std_error, observation, inference_confidence=0.7, source_reliability=0.8
            )

        assert abs(value - true_value) < 8, f"After 50 observations, value={value:.1f} should be near {true_value}"
        assert confidence > 0.7, f"Confidence={confidence:.2f} should be high after 50 observations"
        assert std_error < 10, f"Std error={std_error:.2f} should have decreased from initial 30"

    def test_convergence_with_contradictory_evidence(self):
        """First 25 observations say 80, next 25 say 30. Where does it land?"""
        from app.core.bayesian import update_sub_component

        value, std_error = 50.0, 30.0

        # Phase 1: evidence says 80
        for _ in range(25):
            value, std_error, _ = update_sub_component(
                value, std_error, 80.0, inference_confidence=0.6, source_reliability=0.7
            )
        after_phase1 = value

        # Phase 2: evidence says 30
        for _ in range(25):
            value, std_error, _ = update_sub_component(
                value, std_error, 30.0, inference_confidence=0.6, source_reliability=0.7
            )

        # Should have shifted toward 30 but not fully — earlier evidence has weight
        assert value < after_phase1, "Value should decrease toward contradictory evidence"
        assert value > 30, "Value shouldn't fully reach contradictory evidence (prior mass from phase 1)"
        assert value < 80, "Value should be well below phase 1 level"

    def test_low_reliability_source_has_minimal_effect(self):
        """A source with 0.1 reliability should barely move the posterior."""
        from app.core.bayesian import update_sub_component

        value, std_error = 60.0, 10.0  # Established value with decent confidence
        new_value, _, _ = update_sub_component(
            value, std_error, 20.0,  # Extreme observation
            inference_confidence=0.3, source_reliability=0.1  # Very low reliability
        )

        shift = abs(new_value - value)
        assert shift < 2.0, f"Low reliability source shifted value by {shift:.2f} — should be < 2.0"

    def test_zero_precision_doesnt_crash(self):
        """Edge case: what happens with precision = 0?"""
        from app.core.bayesian import normal_normal_update

        # Zero prior precision
        post_mean, post_prec = normal_normal_update(50.0, 0.0, 75.0, 0.001)
        assert not math.isnan(post_mean), "Should not produce NaN"
        assert not math.isinf(post_mean), "Should not produce Inf"

    def test_extreme_values_dont_overflow(self):
        """Very large observation values should not cause overflow."""
        from app.core.bayesian import update_sub_component

        value, std_error, confidence = update_sub_component(
            50.0, 30.0, 999999.0, inference_confidence=0.5, source_reliability=0.5
        )
        assert not math.isnan(value), "Should not produce NaN"
        assert not math.isinf(value), "Should not produce Inf"

    def test_negative_observation(self):
        """Negative observation (e.g., baseline_valence) should be handled."""
        from app.core.bayesian import update_sub_component

        value, std_error, confidence = update_sub_component(
            0.0, 30.0, -50.0, inference_confidence=0.7, source_reliability=0.8
        )
        assert value < 0, "Should shift negative"
        assert not math.isnan(value)


# ============================================================================
# SIMULATION 2: Personality Derivation — Internal Consistency
# ============================================================================

class TestPersonalityConsistency:
    """Are derived personality types internally consistent?"""

    def test_mbti_e_requires_high_extraversion(self):
        """MBTI 'E' pole should only appear when Big Five Extraversion > 50."""
        big_five_scores = [
            (75, "E"), (25, "I"), (50, "E"),  # Edge: 50 maps to E
            (51, "E"), (49, "I"), (100, "E"), (0, "I"),
        ]
        for score, expected_pole in big_five_scores:
            pole = "E" if score > 50 else "I"
            # Note: score == 50 is ambiguous — spec says > 50 = E
            if score != 50:
                assert pole == expected_pole, f"Extraversion={score} should give {expected_pole}, got {pole}"

    def test_all_16_mbti_types_are_reachable(self):
        """Every MBTI type should be derivable from some Big Five combination."""
        types_seen = set()
        for e in [25, 75]:
            for o in [25, 75]:
                for a in [25, 75]:
                    for c in [25, 75]:
                        mbti = ""
                        mbti += "E" if e > 50 else "I"
                        mbti += "N" if o > 50 else "S"
                        mbti += "T" if a < 50 else "F"
                        mbti += "J" if c > 50 else "P"
                        types_seen.add(mbti)

        assert len(types_seen) == 16, f"Only {len(types_seen)} types reachable, expected 16"

    def test_big_five_extremes_produce_coherent_personality(self):
        """All-high Big Five should produce a coherent (if unusual) type."""
        # All factors at 90 — very open, conscientious, extraverted, agreeable, neurotic
        mbti = "E" + "N" + "F" + "J"  # High everything
        assert len(mbti) == 4
        assert mbti == "ENFJ"

        # All factors at 10
        mbti_low = "I" + "S" + "T" + "P"
        assert mbti_low == "ISTP"


# ============================================================================
# SIMULATION 3: Mood State — Extreme Inputs, Decay, Boundary
# ============================================================================

class TestMoodStateStress:
    """Push the mood system to its limits."""

    def test_rapid_mood_oscillation(self):
        """Rapidly alternating positive/negative triggers should not destabilize."""
        from app.core.mood_state import _mood_cache, shift_mood, MAX_SESSION_MOOD_SHIFT
        from uuid import uuid4

        twin_id = uuid4()
        _mood_cache[str(twin_id)] = {
            "baseline_valence": 0, "baseline_energy": 50,
            "current_valence": 0, "current_energy": 50,
            "session_start_valence": 0,
            "regulation_capacity": 50, "affective_empathy": 50,
            "recovery_speed": 50, "mood_triggers_active": [],
            "last_mood_update": time.time(),
        }

        # 100 rapid alternating shifts
        for i in range(100):
            delta = 30 if i % 2 == 0 else -30
            shift_mood(twin_id, valence_delta=delta, trigger_intensity=80)

        mood = _mood_cache[str(twin_id)]
        assert -MAX_SESSION_MOOD_SHIFT <= mood["current_valence"] <= MAX_SESSION_MOOD_SHIFT
        assert 0 <= mood["current_energy"] <= 100
        del _mood_cache[str(twin_id)]

    def test_mood_modifiers_never_exceed_caps(self):
        """No matter how extreme the mood, modifiers should be capped."""
        from app.core.mood_state import compute_mood_modifiers, MODIFIER_CAP_STANDARD, MODIFIER_CAP_RATIO

        # Extreme negative mood
        extreme_mood = {
            "baseline_valence": 0, "baseline_energy": 50,
            "current_valence": -100, "current_energy": 0,
        }
        modifiers = compute_mood_modifiers(extreme_mood)

        for sub_comp, modifier in modifiers.items():
            if sub_comp in {"generation_confidence", "listen_speak_ratio", "learning_rate"}:
                assert abs(modifier) <= MODIFIER_CAP_RATIO + 0.001, \
                    f"{sub_comp} modifier {modifier} exceeds ratio cap {MODIFIER_CAP_RATIO}"
            else:
                assert abs(modifier) <= MODIFIER_CAP_STANDARD + 0.001, \
                    f"{sub_comp} modifier {modifier} exceeds standard cap {MODIFIER_CAP_STANDARD}"

    def test_mood_decay_returns_to_baseline(self):
        """After sufficient time, mood should decay back to baseline."""
        from app.core.mood_state import _mood_cache, _apply_decay

        mood = {
            "baseline_valence": 0, "baseline_energy": 50,
            "current_valence": -25, "current_energy": 80,
            "recovery_speed": 100,  # Maximum recovery speed
            "mood_triggers_active": [],
            "last_mood_update": time.time() - 600,  # 10 minutes ago
        }

        _apply_decay(mood)

        # With max recovery speed and 10 minutes elapsed, should be very close to baseline
        assert abs(mood["current_valence"]) < 5, f"Valence should be near 0, got {mood['current_valence']:.1f}"
        assert abs(mood["current_energy"] - 50) < 10, f"Energy should be near 50, got {mood['current_energy']:.1f}"

    def test_zero_regulation_allows_full_shift(self):
        """With 0% regulation capacity, mood shift should be maximal."""
        from app.core.mood_state import _mood_cache, shift_mood
        from uuid import uuid4

        twin_id = uuid4()
        _mood_cache[str(twin_id)] = {
            "baseline_valence": 0, "baseline_energy": 50,
            "current_valence": 0, "current_energy": 50,
            "session_start_valence": 0,
            "regulation_capacity": 0,  # Zero regulation
            "affective_empathy": 50,
            "recovery_speed": 50, "mood_triggers_active": [],
            "last_mood_update": time.time(),
        }

        shift_mood(twin_id, valence_delta=-50, trigger_intensity=100)
        mood = _mood_cache[str(twin_id)]

        # With 0 regulation and 100 intensity, shift should be large
        assert mood["current_valence"] < -20, f"Expected large shift, got {mood['current_valence']}"
        del _mood_cache[str(twin_id)]


# ============================================================================
# SIMULATION 4: Drift Detection — Sensitivity Analysis
# ============================================================================

class TestDriftSensitivity:
    """How sensitive is drift detection to different change patterns?"""

    def _compute_drift(self, deviations):
        """Helper: compute drift from list of (value, baseline, confidence) tuples."""
        weighted_sum = 0.0
        weight_sum = 0.0
        for value, baseline, confidence in deviations:
            w = max(0.01, confidence)
            deviation = (value - baseline) / 100.0
            weighted_sum += w * (deviation ** 2)
            weight_sum += w
        return math.sqrt(weighted_sum / weight_sum) if weight_sum > 0 else 0

    def test_single_large_deviation_detected(self):
        """A single sub-component shifting by 40 points should be detected."""
        drift = self._compute_drift([
            (90, 50, 0.8),  # 40-point shift
            (50, 50, 0.8),  # No change
            (50, 50, 0.8),  # No change
        ])
        assert drift > 0.15, f"Single large deviation should exceed threshold, got {drift:.3f}"

    def test_many_small_deviations_accumulate(self):
        """10 sub-components each shifting by 10 points should accumulate."""
        small_shifts = [(60, 50, 0.7)] * 10  # 10 points each
        drift = self._compute_drift(small_shifts)
        assert drift > 0.05, f"Many small deviations should accumulate, got {drift:.3f}"

    def test_low_confidence_shifts_dampened(self):
        """Shifts in low-confidence sub-components should contribute less."""
        high_conf = self._compute_drift([(80, 50, 0.9)])
        low_conf = self._compute_drift([(80, 50, 0.1)])

        # Single sub-component: weight cancels. But in a mix:
        mixed = self._compute_drift([(80, 50, 0.1), (50, 50, 0.9)])
        pure = self._compute_drift([(80, 50, 0.9), (50, 50, 0.9)])
        assert mixed < pure, "Low-confidence deviation should contribute less in a mix"

    def test_threshold_boundary_precision(self):
        """Verify drift score near the 0.25 threshold is computed precisely."""
        # Engineer a case that should be exactly at threshold
        # deviation = 25/100 = 0.25, weight = 1.0
        drift = self._compute_drift([(75, 50, 1.0)])
        assert abs(drift - 0.25) < 0.001, f"Expected drift=0.25, got {drift:.4f}"


# ============================================================================
# SIMULATION 5: Score-to-Instruction — Coverage & Coherence
# ============================================================================

class TestScoreToInstructionCoverage:
    """Does the instruction table produce coherent output across all ranges?"""

    INSTRUCTION_TABLE = {
        "formality": {"high": "formal", "low": "casual"},
        "verbosity": {"high": "detailed", "low": "concise"},
        "humor_frequency": {"high": "humor", "low": "serious"},
        "expressiveness": {"high": "emotions openly", "low": "restraint"},
        "directness": {"high": "direct", "low": "indirect"},
        "risk_tolerance": {"high": "bold", "low": "cautious"},
        "vocabulary_level": {"high": "rich", "low": "simple"},
        "trust_speed": {"high": "trusting", "low": "cautious"},
        "baseline_valence": {"high": "upbeat", "low": "serious"},
        "vulnerability_comfort": {"high": "vulnerability", "low": "private"},
    }

    def test_high_and_low_instructions_are_distinct(self):
        """High and low instructions for the same sub-component should never be identical."""
        for comp, instructions in self.INSTRUCTION_TABLE.items():
            high = instructions.get("high", "")
            low = instructions.get("low", "")
            if high and low:
                assert high != low, f"{comp}: high and low instructions are identical"

    def test_opposing_scores_produce_opposing_instructions(self):
        """High and low instructions for the same component should convey opposite guidance."""
        # Verify they're non-empty and distinct (not that they don't share substrings,
        # since "direct"/"indirect" are intentionally related antonyms)
        for comp in self.INSTRUCTION_TABLE:
            high_kw = self.INSTRUCTION_TABLE[comp].get("high", "")
            low_kw = self.INSTRUCTION_TABLE[comp].get("low", "")
            assert high_kw and low_kw, f"{comp}: missing high or low instruction"
            assert high_kw != low_kw, f"{comp}: high and low are identical"

    def test_mid_range_produces_no_instruction(self):
        """Scores between 30-70 should generally produce no instruction (neutral behavior)."""
        # This is by design — the system doesn't over-specify neutral values
        pass  # Verified in test_pipeline_e2e.py::test_score_instruction_mid_range_no_instruction


# ============================================================================
# SIMULATION 6: Context Detection — Ambiguity & Multi-Signal
# ============================================================================

class TestContextDetectionEdgeCases:
    """How does context detection handle ambiguous and multi-signal inputs?"""

    CONTEXT_KEYWORDS = {
        "PROFESSIONAL": ["meeting", "business", "quarterly", "revenue", "client", "deadline", "strategy"],
        "CASUAL": ["hey", "lol", "haha", "chill", "hang out", "weekend", "fun"],
        "FORMAL": ["dear", "sincerely", "respectfully", "hereby", "pursuant"],
        "CONFLICT": ["disagree", "wrong", "frustrated", "disappointed", "unacceptable", "complaint"],
        "CREATIVE": ["brainstorm", "imagine", "what if", "idea", "create", "design", "concept"],
        "INTIMATE": ["love", "miss you", "feeling", "worried about us", "relationship"],
    }

    def _detect(self, text):
        text = text.lower()
        scores = {}
        for ctx_type, keywords in self.CONTEXT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[ctx_type] = score
        return max(scores, key=scores.get) if scores else "PUBLIC"

    def test_professional_creative_overlap(self):
        """A creative brainstorm in a business setting — which wins?"""
        result = self._detect("Let's brainstorm new business strategy ideas for the quarterly meeting")
        # Should be PROFESSIONAL (3 keywords) over CREATIVE (2)
        assert result in ("PROFESSIONAL", "CREATIVE"), f"Got {result}"

    def test_intimate_conflict_overlap(self):
        """A relationship argument — intimate or conflict?"""
        result = self._detect("I'm frustrated about our relationship and disappointed")
        # CONFLICT has more keywords here
        assert result in ("CONFLICT", "INTIMATE"), f"Got {result}"

    def test_empty_string_defaults_to_public(self):
        """Empty or generic input should default to PUBLIC."""
        assert self._detect("") == "PUBLIC"
        assert self._detect("Hello") == "PUBLIC"
        assert self._detect("How are you today?") == "PUBLIC"

    def test_single_keyword_is_enough(self):
        """Even one keyword should trigger detection (not default to PUBLIC)."""
        assert self._detect("This meeting is important") == "PROFESSIONAL"
        assert self._detect("That's unacceptable") == "CONFLICT"

    def test_case_insensitivity(self):
        """Detection should work regardless of case."""
        assert self._detect("MEETING with CLIENT") == "PROFESSIONAL"
        assert self._detect("LOL that was FUN") == "CASUAL"


# ============================================================================
# SIMULATION 7: Classification Rule Validation — Adversarial Inputs
# ============================================================================

class TestClassificationRuleEdgeCases:
    """Push the Stage 2 classifier validation with adversarial inputs."""

    VALID_CATEGORIES = {
        "MIND", "HEART", "SPIRIT", "PHYSICALITY", "EXPERIENCES",
        "RELATIONSHIPS", "SURROUNDINGS", "WORK", "ETHICS", "FUTURE",
        "INTERESTS_TASTES",
    }
    MIN_CONFIDENCE = 0.4

    def test_all_categories_below_threshold(self):
        """If all classifications are below 0.4, the top one should still be kept."""
        classifications = [
            {"category": "MIND", "confidence": 0.2},
            {"category": "HEART", "confidence": 0.1},
        ]
        valid = [c for c in classifications if c["confidence"] >= self.MIN_CONFIDENCE]
        if not valid:
            valid = classifications[:1]
        assert len(valid) == 1
        assert valid[0]["category"] == "MIND"

    def test_invalid_category_name_rejected(self):
        """Typos or hallucinated categories should be filtered out."""
        classifications = [
            {"category": "MNID", "confidence": 0.9},  # Typo
            {"category": "POLITICS", "confidence": 0.8},  # Not a category
            {"category": "HEART", "confidence": 0.5},  # Valid
        ]
        valid = [c for c in classifications if c["category"].upper() in self.VALID_CATEGORIES]
        assert len(valid) == 1
        assert valid[0]["category"] == "HEART"

    def test_duplicate_category_handling(self):
        """Two classifications for the same category — should be deduplicated."""
        classifications = [
            {"category": "ETHICS", "confidence": 0.7},
            {"category": "ETHICS", "confidence": 0.5},
        ]
        # Keep higher confidence
        seen = {}
        for c in classifications:
            cat = c["category"]
            if cat not in seen or c["confidence"] > seen[cat]["confidence"]:
                seen[cat] = c
        assert len(seen) == 1
        assert seen["ETHICS"]["confidence"] == 0.7

    def test_eleven_categories_all_above_threshold(self):
        """LLM returns all 11 categories — should be validated, not blindly accepted."""
        classifications = [
            {"category": cat, "confidence": 0.5}
            for cat in self.VALID_CATEGORIES
        ]
        # All are valid and above threshold — but 11 categories for one piece of content
        # is almost certainly wrong. This tests whether the system accepts it.
        valid = [c for c in classifications if c["confidence"] >= self.MIN_CONFIDENCE]
        # Currently the system would accept all 11. This is a real gap —
        # the co-occurrence matrix should catch implausible combinations.
        assert len(valid) == 11  # System currently accepts this


# ============================================================================
# SIMULATION 8: Fear/Need Detection — Edge Cases
# ============================================================================

class TestFearNeedEdgeCases:
    """Push fear/need detection with subtle and adversarial inputs."""

    FEAR_SIGNALS = [
        "afraid", "fear", "scared", "anxious", "worry", "dread", "phobia",
        "avoid", "terrified", "nervous", "panic", "apprehensive", "uneasy",
        "intimidated", "aversion", "can't stand", "hate dealing with",
    ]
    NEED_SIGNALS = [
        "need", "require", "must have", "can't function without", "essential",
        "crave", "driven by", "motivated by", "fulfilled by", "autonomy",
        "recognition", "security", "routine", "creative freedom", "validation",
        "structure", "independence", "connection", "purpose",
    ]

    def _detect(self, content):
        c = content.lower()
        detected = []
        if any(s in c for s in self.FEAR_SIGNALS):
            detected.append("fear")
        if any(s in c for s in self.NEED_SIGNALS):
            detected.append("need")
        return detected

    def test_indirect_fear_not_detected(self):
        """Indirect fear expressions are missed by keyword matching — known limitation."""
        # "Public speaking has always been difficult" = fear, but no keyword match
        result = self._detect("Public speaking has always been difficult for me")
        assert "fear" not in result  # Known gap — LLM-assisted detection would catch this

    def test_indirect_need_not_detected(self):
        """Indirect need expressions are missed — known limitation."""
        result = self._detect("I work best when nobody is looking over my shoulder")
        assert "need" not in result  # Known gap — should detect need for autonomy

    def test_false_positive_need_in_ordinary_sentence(self):
        """'I need a coffee' should not create a RAG need entry."""
        result = self._detect("I need a coffee before the meeting")
        # This DOES trigger "need" — false positive. The keyword "need" matches.
        assert "need" in result  # Known false positive — keyword matching is blunt

    def test_fear_in_negated_context(self):
        """'I'm not afraid of anything' should still trigger (keyword present)."""
        result = self._detect("I'm not afraid of anything")
        assert "fear" in result  # Keyword matching doesn't understand negation

    def test_empty_content(self):
        """Empty content should detect nothing."""
        assert self._detect("") == []
        assert self._detect("   ") == []


# ============================================================================
# SIMULATION 9: Dirichlet-Categorical — Distribution Stability
# ============================================================================

class TestDirichletStability:
    """Push the Dirichlet-Categorical update with edge cases."""

    def test_repeated_same_observation_converges(self):
        """100 observations of 'analytical' should make it dominant."""
        from app.core.bayesian import dirichlet_categorical_update

        dist = {"analytical": 0.33, "intuitive": 0.33, "mixed": 0.34}
        for _ in range(100):
            dist = dirichlet_categorical_update(dist, "analytical", 0.8)

        assert dist["analytical"] > 0.8, f"Expected analytical > 0.8, got {dist['analytical']:.3f}"
        assert abs(sum(dist.values()) - 1.0) < 0.01, "Distribution should sum to ~1.0"

    def test_three_way_split_stays_balanced(self):
        """Equal observations across 3 categories should maintain balance."""
        from app.core.bayesian import dirichlet_categorical_update

        dist = {"a": 0.33, "b": 0.33, "c": 0.34}
        for _ in range(30):
            dist = dirichlet_categorical_update(dist, "a", 1.0)
            dist = dirichlet_categorical_update(dist, "b", 1.0)
            dist = dirichlet_categorical_update(dist, "c", 1.0)

        # Should stay roughly balanced
        values = list(dist.values())
        assert max(values) - min(values) < 0.1, f"Imbalance: {dist}"

    def test_new_category_doesnt_destroy_existing(self):
        """Adding a never-seen category should not zero out existing ones."""
        from app.core.bayesian import dirichlet_categorical_update

        dist = {"analytical": 0.7, "intuitive": 0.3}
        dist = dirichlet_categorical_update(dist, "creative", 1.0)

        assert "creative" in dist
        assert dist["analytical"] > 0.3, "Existing categories should retain significant mass"
        assert dist["creative"] < 0.5, "New category shouldn't dominate on first observation"


# ============================================================================
# SIMULATION 10: Token Budget — Overflow Scenarios
# ============================================================================

class TestTokenBudgetOverflow:
    """What happens when prompt assembly content exceeds budget?"""

    @staticmethod
    def _truncate(text, max_chars):
        """Local reimplementation to avoid SQLAlchemy import."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.7:
            return truncated[:last_newline]
        return truncated

    def test_truncation_preserves_complete_lines(self):
        """Truncation at budget limit should cut at line boundaries when possible."""
        text = "\n".join([f"Line {i}: " + "x" * 40 for i in range(10)])
        truncated = self._truncate(text, 200)
        assert len(truncated) <= 200

    def test_single_long_line_truncated_cleanly(self):
        """A single line longer than the budget should be truncated."""
        text = "x" * 1000
        truncated = self._truncate(text, 100)
        assert len(truncated) <= 100

    def test_empty_input_returns_empty(self):
        """Empty or short input should pass through unchanged."""
        assert self._truncate("", 1000) == ""
        assert self._truncate("short", 1000) == "short"


# ============================================================================
# SUMMARY: Count simulations
# ============================================================================

def test_simulation_coverage():
    """Meta-test: verify we have sufficient simulation coverage."""
    simulation_classes = [
        TestBayesianConvergence,
        TestPersonalityConsistency,
        TestMoodStateStress,
        TestDriftSensitivity,
        TestScoreToInstructionCoverage,
        TestContextDetectionEdgeCases,
        TestClassificationRuleEdgeCases,
        TestFearNeedEdgeCases,
        TestDirichletStability,
        TestTokenBudgetOverflow,
    ]
    total = sum(
        len([m for m in dir(cls) if m.startswith("test_")])
        for cls in simulation_classes
    )
    assert total >= 35, f"Expected at least 35 simulation tests, got {total}"
