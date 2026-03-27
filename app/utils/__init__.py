"""Shared utilities for the ALCM API."""
import uuid as _uuid
from fastapi import HTTPException


def parse_uuid(value: str, field_name: str = "id") -> _uuid.UUID:
    """Parse a string to UUID, raising 400 on invalid format."""
    try:
        return _uuid.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name} format: {value}")
