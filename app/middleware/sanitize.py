"""Input sanitization middleware — basic prompt injection protection.

Checks incoming request bodies for common prompt injection patterns.
This is a first line of defense, not a complete solution.
"""
import logging
import re
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Patterns that suggest prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+in\s+.*mode",
    r"system\s*:\s*you\s+are",
    r"<\|.*\|>",  # Token boundary markers
    r"\[INST\]",  # Instruction markers
    r"<<SYS>>",   # System prompt markers
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Max request body size (5MB)
MAX_BODY_SIZE = 5 * 1024 * 1024


class SanitizationMiddleware(BaseHTTPMiddleware):
    """Check request bodies for injection patterns and enforce size limits."""

    async def dispatch(self, request: Request, call_next):
        # Skip non-mutating requests
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)

        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE:
            raise HTTPException(status_code=413, detail=f"Request body too large (max {MAX_BODY_SIZE // 1024 // 1024}MB)")

        # Read and check body for injection patterns
        body = await request.body()
        if len(body) > MAX_BODY_SIZE:
            raise HTTPException(status_code=413, detail=f"Request body too large")

        body_text = body.decode("utf-8", errors="ignore")
        for pattern in COMPILED_PATTERNS:
            if pattern.search(body_text):
                logger.warning(
                    f"Potential prompt injection blocked from {request.client.host}: "
                    f"pattern={pattern.pattern}, path={request.url.path}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Request contains potentially unsafe content"
                )

        return await call_next(request)
