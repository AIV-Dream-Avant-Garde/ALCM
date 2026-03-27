"""Basic in-memory rate limiting for the ALCM API.

Phase 3: Simple token bucket per IP. Production: Redis-backed rate limiting.
"""
import time
import logging
from collections import defaultdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Rate limit: 100 requests per minute per IP
RATE_LIMIT = 100
WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter using sliding window counters."""

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app):
        super().__init__(app)
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - WINDOW_SECONDS

        # Clean old entries and count recent requests
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= RATE_LIMIT:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": "Too many requests — retry after delay.",
                        "status": 429,
                        "details": {"retry_after_seconds": WINDOW_SECONDS},
                    }
                },
                headers={"Retry-After": str(WINDOW_SECONDS)},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
