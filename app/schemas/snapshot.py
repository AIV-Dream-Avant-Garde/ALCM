"""Snapshot schemas."""
from pydantic import BaseModel


class SnapshotResponse(BaseModel):
    snapshot_ref: str
    seal_hash: str
    version_number: int
    created_at: str
