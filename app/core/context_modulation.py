"""Context-specific value resolution.

Phase 1: Basic lookup from context_modulation table.
Phase 2: Dynamic context detection from conversation signals.
See Developer Guide Section 4.6.
"""
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def resolve_value(
    twin_id: UUID,
    sub_component: str,
    context_type: Optional[str],
    default_value: float,
    db: AsyncSession,
) -> float:
    """Resolve a sub-component value for a given context.

    If a context-specific override exists with sufficient confidence,
    use it. Otherwise, return the default value.

    Args:
        twin_id: The twin's ID.
        sub_component: The sub-component name (e.g., "formality").
        context_type: Detected context type (e.g., "PROFESSIONAL").
        default_value: The default sub-component value.
        db: Database session.

    Returns:
        The resolved value for this context.
    """
    if not context_type:
        return default_value

    from app.models.context_modulation import ContextModulation

    result = await db.execute(
        select(ContextModulation).where(
            ContextModulation.twin_id == twin_id,
            ContextModulation.context_type == context_type,
            ContextModulation.sub_component == sub_component,
        )
    )
    override = result.scalar_one_or_none()

    if override and override.confidence > 0.3:
        return override.override_value

    return default_value


async def get_all_overrides(
    twin_id: UUID,
    context_type: str,
    db: AsyncSession,
) -> dict:
    """Get all context overrides for a twin and context type.

    Returns:
        Dict mapping sub_component -> override_value.
    """
    from app.models.context_modulation import ContextModulation

    result = await db.execute(
        select(ContextModulation).where(
            ContextModulation.twin_id == twin_id,
            ContextModulation.context_type == context_type,
            ContextModulation.confidence > 0.3,
        )
    )
    overrides = result.scalars().all()
    return {o.sub_component: o.override_value for o in overrides}
