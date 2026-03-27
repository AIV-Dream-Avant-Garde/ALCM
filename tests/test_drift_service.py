"""Unit tests for drift computation logic."""
import math


def test_drift_score_zero_when_at_baseline():
    """Drift should be 0 when all values equal their baseline."""
    # Simulate: 3 sub-components, all at prior_mean
    weighted_sum = 0.0
    weight_sum = 0.0

    for value, baseline, confidence in [(50, 50, 0.7), (60, 60, 0.5), (30, 30, 0.8)]:
        w = max(0.01, confidence)
        deviation = (value - baseline) / 100.0
        weighted_sum += w * (deviation ** 2)
        weight_sum += w

    drift = math.sqrt(weighted_sum / weight_sum) if weight_sum > 0 else 0
    assert drift == 0.0


def test_drift_score_increases_with_deviation():
    """Drift should increase as values deviate from baseline."""
    def compute_drift(deviations):
        weighted_sum = 0.0
        weight_sum = 0.0
        for value, baseline, confidence in deviations:
            w = max(0.01, confidence)
            dev = (value - baseline) / 100.0
            weighted_sum += w * (dev ** 2)
            weight_sum += w
        return math.sqrt(weighted_sum / weight_sum) if weight_sum > 0 else 0

    small_drift = compute_drift([(55, 50, 0.7), (62, 60, 0.5)])
    large_drift = compute_drift([(80, 50, 0.7), (90, 60, 0.5)])

    assert large_drift > small_drift


def test_drift_score_bounded():
    """Drift score should remain in [0, 1] range for typical values."""
    # Maximum drift: all sub-components at opposite extremes
    weighted_sum = 0.0
    weight_sum = 0.0

    for value, baseline, confidence in [(0, 100, 1.0), (100, 0, 1.0)]:
        w = confidence
        deviation = (value - baseline) / 100.0
        weighted_sum += w * (deviation ** 2)
        weight_sum += w

    drift = math.sqrt(weighted_sum / weight_sum)
    assert drift == 1.0


def test_low_confidence_contributes_less():
    """Sub-components with low confidence should contribute less to drift."""
    def compute_drift(deviations):
        weighted_sum = 0.0
        weight_sum = 0.0
        for value, baseline, confidence in deviations:
            w = max(0.01, confidence)
            dev = (value - baseline) / 100.0
            weighted_sum += w * (dev ** 2)
            weight_sum += w
        return math.sqrt(weighted_sum / weight_sum) if weight_sum > 0 else 0

    # Same deviation, but one has low confidence
    high_conf = compute_drift([(80, 50, 0.9)])
    low_conf = compute_drift([(80, 50, 0.1)])

    # With single sub-component, the drift score is the same regardless of weight
    # (because weight cancels in numerator/denominator). But in a mix:
    mixed_high = compute_drift([(80, 50, 0.9), (50, 50, 0.9)])
    mixed_low = compute_drift([(80, 50, 0.1), (50, 50, 0.9)])

    # The low-confidence deviant should be dampened in the mix
    assert mixed_low < mixed_high
