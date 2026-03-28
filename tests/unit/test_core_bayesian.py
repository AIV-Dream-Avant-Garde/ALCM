"""Unit tests for the Bayesian update core module."""
from app.core.bayesian import (
    normal_normal_update,
    update_sub_component,
    dirichlet_categorical_update,
    compute_confidence,
    compute_observation_precision,
    INITIAL_STD_ERROR,
)


def test_normal_normal_update_basic():
    """Prior at 50, observe 70 — posterior should shift toward 70."""
    prior_mean = 50.0
    prior_precision = 1.0 / (30.0 ** 2)  # Initial std_error = 30
    observation = 70.0
    obs_precision = (0.6 * 0.8) / (30.0 ** 2)

    post_mean, post_prec = normal_normal_update(
        prior_mean, prior_precision, observation, obs_precision
    )

    assert post_mean > 50.0, "Posterior should shift toward observation"
    assert post_mean < 70.0, "Posterior should not jump to observation"
    assert post_prec > prior_precision, "Precision should increase"


def test_normal_normal_update_equal_precision():
    """Equal precision — posterior should be midpoint."""
    post_mean, _ = normal_normal_update(40.0, 1.0, 60.0, 1.0)
    assert abs(post_mean - 50.0) < 0.01


def test_update_sub_component():
    """Full update should return valid value, std_error, and confidence."""
    new_val, new_se, new_conf = update_sub_component(
        current_value=50.0,
        current_std_error=30.0,
        observation=75.0,
        inference_confidence=0.7,
        source_reliability=0.8,
    )

    assert 50.0 < new_val < 75.0, "Value should shift toward observation"
    assert new_se < 30.0, "Std error should decrease"
    assert new_conf > 0.0, "Confidence should be positive"
    assert new_conf < 1.0, "Confidence should be < 1"


def test_update_sub_component_high_confidence():
    """High confidence observation should shift value more."""
    val_low, _, _ = update_sub_component(50.0, 30.0, 80.0, 0.3, 0.5)
    val_high, _, _ = update_sub_component(50.0, 30.0, 80.0, 0.9, 1.0)

    assert val_high > val_low, "Higher confidence should produce larger shift"


def test_compute_confidence():
    """Confidence should be 0 at initial std_error and approach 1 as std_error approaches 0."""
    assert compute_confidence(INITIAL_STD_ERROR) == 0.0
    assert compute_confidence(0.0) == 1.0
    assert 0.0 < compute_confidence(15.0) < 1.0


def test_compute_observation_precision():
    """Observation precision scales with confidence and reliability."""
    p1 = compute_observation_precision(0.5, 0.5)
    p2 = compute_observation_precision(1.0, 1.0)
    assert p2 > p1


def test_dirichlet_categorical_update():
    """Updating a categorical distribution should boost the observed category."""
    dist = {"analytical": 0.5, "intuitive": 0.3, "mixed": 0.2}
    updated = dirichlet_categorical_update(dist, "intuitive", 1.0)

    assert updated["intuitive"] > dist["intuitive"], "Observed category should increase"
    assert abs(sum(updated.values()) - 1.0) < 0.01, "Distribution should sum to ~1.0"


def test_dirichlet_categorical_new_category():
    """Adding a new category to the distribution."""
    dist = {"analytical": 0.6, "intuitive": 0.4}
    updated = dirichlet_categorical_update(dist, "creative", 1.0)

    assert "creative" in updated
    assert sum(updated.values()) > 0.99


def test_dirichlet_categorical_empty():
    """Empty distribution should handle gracefully."""
    updated = dirichlet_categorical_update({}, "analytical")
    assert updated == {"analytical": 1.0}
