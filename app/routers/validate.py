"""Validation endpoint — POST /validate."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..services.validation_service import validate_consistency
from ..schemas.common import ValidateRequest, ValidateResponse

router = APIRouter(tags=["validate"])


@router.post("/validate", response_model=ValidateResponse)
async def validate_content(req: ValidateRequest, db: AsyncSession = Depends(get_db)):
    """Check if sample content is consistent with twin's personality profile."""
    return await validate_consistency(req, db)
