"""ALCM API authentication — bearer token verification.

The platform service must include Authorization: Bearer <token> on every request.
The /health endpoint is exempt (used for Docker health checks).
"""
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings

logger = logging.getLogger(__name__)


class ALCMAuthMiddleware(BaseHTTPMiddleware):
    """Verify bearer token on all ALCM API requests except /health and /docs."""

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        settings = get_settings()
        expected_token = settings.alcm_service_token

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid Authorization header"},
            )

        token = auth_header[7:]
        if token != expected_token:
            logger.warning(f"Invalid ALCM auth token from {request.client.host}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid authentication token"},
            )

        return await call_next(request)
