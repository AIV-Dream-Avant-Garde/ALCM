"""Drift detection schemas."""
from pydantic import BaseModel
from typing import Optional


class DriftResponse(BaseModel):
    drift_score: float
    threshold: float = 0.25
    threshold_status: str = "WITHIN_BOUNDS"
    per_dimension_drift: dict = {}
    baseline_set_at: Optional[str] = None
    last_checked: Optional[str] = None
