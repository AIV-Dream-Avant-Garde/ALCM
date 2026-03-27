"""Bayesian updating for ALCM dimensional scores.

Normal-Normal conjugate model for continuous sub-components.
Dirichlet-Categorical model for enum sub-components.
See Developer Guide Section 14 (Adaptive Learning System).
"""
import math
from typing import Tuple

# Initial std_error for new sub-components (spec: Section 14.4)
INITIAL_STD_ERROR = 30.0


def normal_normal_update(
    prior_mean: float,
    prior_precision: float,
    observation: float,
    observation_precision: float,
) -> Tuple[float, float]:
    """Normal-Normal Bayesian update for continuous sub-components.

    Args:
        prior_mean: Current posterior mean (the sub-component value).
        prior_precision: 1 / std_error^2 of the current posterior.
        observation: New observed value from classification/feedback.
        observation_precision: Precision of the observation
            = (inference_confidence * source_reliability) / INITIAL_STD_ERROR^2.

    Returns:
        (posterior_mean, posterior_precision)
    """
    posterior_precision = prior_precision + observation_precision
    if posterior_precision == 0:
        return prior_mean, prior_precision
    posterior_mean = (
        prior_mean * prior_precision + observation * observation_precision
    ) / posterior_precision
    return posterior_mean, posterior_precision


def precision_to_std_error(precision: float) -> float:
    """Convert precision to standard error."""
    if precision <= 0:
        return INITIAL_STD_ERROR
    return 1.0 / math.sqrt(precision)


def std_error_to_precision(std_error: float) -> float:
    """Convert standard error to precision."""
    if std_error <= 0:
        return 0.0
    return 1.0 / (std_error ** 2)


def compute_confidence(std_error: float) -> float:
    """Compute confidence from std_error.

    confidence = 1 - (std_error / INITIAL_STD_ERROR)
    Clamped to [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - (std_error / INITIAL_STD_ERROR)))


def compute_observation_precision(
    inference_confidence: float,
    source_reliability: float,
) -> float:
    """Compute observation precision from classification confidence and source reliability.

    observation_precision = (inference_confidence * source_reliability) / INITIAL_STD_ERROR^2
    """
    return (inference_confidence * source_reliability) / (INITIAL_STD_ERROR ** 2)


def update_sub_component(
    current_value: float,
    current_std_error: float,
    observation: float,
    inference_confidence: float,
    source_reliability: float,
) -> Tuple[float, float, float]:
    """Full Bayesian update for a single sub-component.

    Args:
        current_value: Current mean value of the sub-component.
        current_std_error: Current std_error of the sub-component.
        observation: New observed value.
        inference_confidence: Confidence of the LLM inference (0-1).
        source_reliability: Reliability weight of the data source (0-1).

    Returns:
        (new_value, new_std_error, new_confidence)
    """
    prior_precision = std_error_to_precision(current_std_error)
    obs_precision = compute_observation_precision(inference_confidence, source_reliability)

    new_value, new_precision = normal_normal_update(
        current_value, prior_precision, observation, obs_precision
    )
    new_std_error = precision_to_std_error(new_precision)
    new_confidence = compute_confidence(new_std_error)

    return new_value, new_std_error, new_confidence


def dirichlet_categorical_update(
    current_dist: dict,
    observed_category: str,
    reliability_weight: float = 1.0,
) -> dict:
    """Dirichlet-Categorical update for enum sub-components.

    Each observation increments the corresponding alpha parameter
    by the source's reliability weight. The posterior mean gives
    the expected probability for each category.

    Args:
        current_dist: Current category distribution, e.g.
            {"analytical": 0.45, "intuitive": 0.35, "mixed": 0.20}
        observed_category: The observed category value.
        reliability_weight: Source reliability (0-1).

    Returns:
        Updated distribution (normalized).
    """
    if not current_dist:
        return {observed_category: 1.0}

    # Convert probabilities to pseudo-counts (alpha parameters)
    # Use a base concentration of 10 for numerical stability
    base_concentration = 10.0
    alphas = {k: v * base_concentration for k, v in current_dist.items()}

    # Add observation
    if observed_category in alphas:
        alphas[observed_category] += reliability_weight
    else:
        alphas[observed_category] = reliability_weight

    # Normalize back to probabilities
    total = sum(alphas.values())
    if total == 0:
        return current_dist
    return {k: v / total for k, v in alphas.items()}
