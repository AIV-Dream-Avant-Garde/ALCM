"""Package delivery schemas."""
from pydantic import BaseModel
from typing import Optional, Any, Dict


class PackageResponse(BaseModel):
    twin_id: str
    version: int = 1
    seal_hash: Optional[str] = None
    generated_at: str
    modules: Dict[str, Any] = {}
