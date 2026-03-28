"""Feedback endpoint — POST /twin/{id}/feedback."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..services.learning_service import process_feedback
from ..schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])


@router.post("/twin/{twin_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    twin_id: str, req: FeedbackRequest, db: AsyncSession = Depends(get_db),
):
    """Process interaction feedback for adaptive learning."""
    return await process_feedback(twin_id, req, db)
