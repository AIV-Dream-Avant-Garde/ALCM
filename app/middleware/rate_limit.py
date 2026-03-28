"""Rate limiting middleware — Redis-backed with in-memory fallback.

Uses Redis INCR + EXPIRE for distributed rate limiting across instances.
Falls back to in-memory sliding window if Redis is unavailable.
"""
import time
import logging
from collections import defaultdict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..config import get_settings

logger = logging.getLogger(__name__)

# Redis client (lazy-initialized)
_redis = None
_redis_available = None


async def _get_redis():
    """Get Redis client, or None if unavailable."""
    global _redis, _redis_available

    if _redis_available is False:
        return None
    if _redis is not None:
        return _redis

    try:
        import redis.asyncio as aioredis
        settings = get_settings()
        redis_url = getattr(settings, "redis_url", None) or "redis://localhost:6379/0"
        _redis = aioredis.from_url(redis_url, decode_responses=True)
        await _redis.ping()
        _redis_available = True
        logger.info("Rate limiting: using Redis")
        return _redis
    except Exception:
        _redis_available = False
        logger.info("Rate limiting: Redis unavailable, using in-memory fallback")
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Distributed rate limiter with Redis + in-memory fallback."""

    EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app):
        super().__init__(app)
        self._memory_requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        settings = get_settings()
        limit = settings.rate_limit_requests
        window = settings.rate_limit_window_seconds
        client_ip = request.client.host if request.client else "unknown"

        # Try Redis first
        redis = await _get_redis()
        if redis:
            allowed = await self._check_redis(redis, client_ip, limit, window)
        else:
            allowed = self._check_memory(client_ip, limit, window)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": "Too many requests — retry after delay.",
                        "status": 429,
                        "details": {"retry_after_seconds": window},
                    }
                },
                headers={"Retry-After": str(window)},
            )

        return await call_next(request)

    async def _check_redis(self, redis, client_ip: str, limit: int, window: int) -> bool:
        """Redis-based sliding window counter."""
        key = f"alcm:ratelimit:{client_ip}"
        try:
            current = await redis.incr(key)
            if current == 1:
                await redis.expire(key, window)
            return current <= limit
        except Exception as e:
            logger.warning(f"Redis rate limit check failed: {e}")
            return self._check_memory(client_ip, limit, window)

    def _check_memory(self, client_ip: str, limit: int, window: int) -> bool:
        """In-memory sliding window fallback."""
        now = time.time()
        window_start = now - window
        self._memory_requests[client_ip] = [
            t for t in self._memory_requests[client_ip] if t > window_start
        ]
        if len(self._memory_requests[client_ip]) >= limit:
            return False
        self._memory_requests[client_ip].append(now)
        return True
