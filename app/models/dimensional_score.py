"""Bayesian dimensional scores — dimensional_scores table."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..database import Base


class DimensionalScore(Base):
    __tablename__ = "dimensional_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    twin_id = Column(UUID(as_uuid=True), ForeignKey("twin_profiles.id", ondelete="CASCADE"), nullable=False)

    # Dimension: COGNITIVE | EMOTIONAL | SOCIAL | EVOLUTIONARY | VISUAL
    dimension = Column(String(50), nullable=False)
    sub_component = Column(String(100), nullable=False)

    # Bayesian posterior (Normal-Normal for continuous)
    value = Column(Float, nullable=False, default=50.0)
    confidence = Column(Float, nullable=False, default=0.0)
    std_error = Column(Float, nullable=False, default=30.0)
    prior_mean = Column(Float, default=50.0)
    prior_variance = Column(Float, default=900.0)  # 30^2
    observation_count = Column(Integer, default=0)

    # For enum sub-components (Dirichlet-Categorical)
    distribution_type = Column(String(20), default="CONTINUOUS")  # CONTINUOUS | CATEGORICAL
    categorical_dist = Column(JSONB)
    # {"analytical": 0.45, "intuitive": 0.35, "mixed": 0.20}

    # Context modulation overrides
    context_overrides = Column(JSONB, default=dict)
    # {"professional": 78, "casual": 52, "intimate": 41}

    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationship
    twin = relationship("TwinProfile", back_populates="dimensional_scores")

    __table_args__ = (
        UniqueConstraint("twin_id", "dimension", "sub_component", name="uq_dims_twin_dim_sub"),
        Index("idx_dims_twin", "twin_id"),
        Index("idx_dims_twin_dim", "twin_id", "dimension"),
    )
