"""Request tracing middleware — assigns unique request ID, logs structured request/response data."""
import json
import logging
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("alcm.access")


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID to every request and logs structured access data."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:12])
        start_time = time.time()

        # Store request_id on request state for use in services
        request.state.request_id = request_id

        # Extract twin_id from path if present
        twin_id = None
        path_parts = request.url.path.split("/")
        if "twin" in path_parts:
            idx = path_parts.index("twin")
            if idx + 1 < len(path_parts):
                twin_id = path_parts[idx + 1]

        response = await call_next(request)
        duration_ms = int((time.time() - start_time) * 1000)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Structured log entry
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client_ip": request.client.host if request.client else None,
            "twin_id": twin_id,
        }

        if response.status_code >= 500:
            logger.error(json.dumps(log_data))
        elif response.status_code >= 400:
            logger.warning(json.dumps(log_data))
        else:
            logger.info(json.dumps(log_data))

        return response
