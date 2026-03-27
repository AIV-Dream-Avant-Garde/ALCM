"""Personality drift detection service.

Computes drift from baseline snapshot using weighted Euclidean distance.
See Developer Guide Section 4.4.

drift_score = sqrt(Σ_i (w_i × ((current_i - baseline_i) / range_i)²)) / sqrt(Σ_i w_i)
"""
import math
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.twin_profile import TwinProfile
from app.models.dimensional_score import DimensionalScore


async def compute_drift(twin_id: UUID, db: AsyncSession) -> dict:
    """Compute personality drift from the last approved baseline.

    Returns:
        {
            "drift_score": 0.0-1.0,
            "per_dimension_drift": {"cognitive": 0.08, ...},
            "threshold_status": "WITHIN_BOUNDS" | "EXCEEDED",
        }
    """
    # Load current dimensional scores
    result = await db.execute(
        select(DimensionalScore).where(DimensionalScore.twin_id == twin_id)
    )
    scores = result.scalars().all()

    if not scores:
        return {
            "drift_score": 0.0,
            "per_dimension_drift": {},
            "threshold_status": "WITHIN_BOUNDS",
        }

    # Load baseline from twin profile
    profile_result = await db.execute(
        select(TwinProfile).where(TwinProfile.id == twin_id)
    )
    profile = profile_result.scalar_one_or_none()

    # If no baseline set, use the prior_mean as baseline (initial calibration)
    # In production, baseline_snapshot_ref would point to a stored snapshot
    # For now, we compute drift from prior_mean (the initial value)

    # Compute per-dimension drift
    dim_drifts = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    for s in scores:
        # Weight by confidence (low-confidence sub-components contribute less)
        w = max(0.01, s.confidence)

        # Determine the baseline value and range
        baseline = s.prior_mean if s.prior_mean is not None else 50.0

        # Range depends on distribution type
        if s.distribution_type == "CATEGORICAL":
            continue  # Skip categorical for drift (different math)

        # For continuous: range is 100 for 0-100 scale, 1.0 for 0-1 scale
        if s.sub_component in ("listen_speak_ratio", "learning_rate", "generation_confidence"):
            value_range = 1.0
        else:
            value_range = 100.0

        deviation = (s.value - baseline) / value_range if value_range > 0 else 0
        weighted_deviation_sq = w * (deviation ** 2)

        weighted_sum += weighted_deviation_sq
        weight_sum += w

        # Accumulate per-dimension
        dim = s.dimension
        if dim not in dim_drifts:
            dim_drifts[dim] = {"weighted_sum": 0.0, "weight_sum": 0.0}
        dim_drifts[dim]["weighted_sum"] += weighted_deviation_sq
        dim_drifts[dim]["weight_sum"] += w

    # Overall drift score
    if weight_sum > 0:
        drift_score = math.sqrt(weighted_sum / weight_sum)
    else:
        drift_score = 0.0

    # Per-dimension drift
    per_dim = {}
    for dim, data in dim_drifts.items():
        if data["weight_sum"] > 0:
            per_dim[dim.lower()] = round(math.sqrt(data["weighted_sum"] / data["weight_sum"]), 4)
        else:
            per_dim[dim.lower()] = 0.0

    drift_score = round(min(1.0, drift_score), 4)
    threshold = 0.25

    return {
        "drift_score": drift_score,
        "per_dimension_drift": per_dim,
        "threshold_status": "WITHIN_BOUNDS" if drift_score < threshold else "EXCEEDED",
    }
