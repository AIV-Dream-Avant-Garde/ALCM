"""Attribution endpoint — POST /attribute."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..services.attribution_service import attribute_data
from ..schemas.attribute import AttributeRequest, AttributeResponse

router = APIRouter(tags=["attribute"])


@router.post("/attribute", response_model=AttributeResponse)
async def attribute_endpoint(req: AttributeRequest, db: AsyncSession = Depends(get_db)):
    """Apply classified data to dimensional scores via Bayesian update."""
    return await attribute_data(req, db)
