"""Processing job schemas."""
from pydantic import BaseModel
from typing import Optional


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
