"""Package delivery endpoint — GET /twin/{id}/package?scope=..."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..services.package_service import assemble_package
from ..schemas.common import PackageResponse

router = APIRouter(tags=["package"])


@router.get("/twin/{twin_id}/package", response_model=PackageResponse)
async def get_package(
    twin_id: str,
    scope: str = Query(..., description="Comma-separated delivery modules"),
    db: AsyncSession = Depends(get_db),
):
    """Deliver scoped identity data from normalized tables."""
    return await assemble_package(twin_id, scope, db)
