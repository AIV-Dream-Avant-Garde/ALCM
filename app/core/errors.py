"""Standardized ALCM API error responses.

All errors follow the format from Technical Spec Section 2.1:
{
  "error": {
    "code": "TWIN_NOT_FOUND",
    "message": "Twin with the specified ID does not exist.",
    "status": 404,
    "details": {}
  }
}
"""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ALCMError(HTTPException):
    """Base ALCM error with spec-compliant structure."""

    def __init__(self, code: str, message: str, status: int, details: dict = None):
        self.code = code
        self.message = message
        self.error_status = status
        self.details = details or {}
        super().__init__(status_code=status, detail=message)


class TwinNotFound(ALCMError):
    def __init__(self, twin_id: str = ""):
        super().__init__(
            code="TWIN_NOT_FOUND",
            message="Twin with the specified ID does not exist.",
            status=404,
            details={"twin_id": twin_id} if twin_id else {},
        )


class ClassificationFailed(ALCMError):
    def __init__(self, message: str = "Content could not be classified into any psychographic category."):
        super().__init__(code="CLASSIFICATION_FAILED", message=message, status=422)


class GenerationFailed(ALCMError):
    def __init__(self, message: str = "Score-to-Generation pipeline failed."):
        super().__init__(code="GENERATION_FAILED", message=message, status=500)


class InsufficientData(ALCMError):
    def __init__(self, message: str = "Twin does not have enough data to fulfill the request."):
        super().__init__(code="INSUFFICIENT_DATA", message=message, status=422)


class TwinLocked(ALCMError):
    def __init__(self):
        super().__init__(code="TWIN_LOCKED", message="Twin is in locked state — no interactions permitted.", status=423)


class SuccessorHold(ALCMError):
    def __init__(self):
        super().__init__(code="SUCCESSOR_HOLD", message="Twin is in protected hold (succession event).", status=423)


class GuardrailViolation(ALCMError):
    def __init__(self, message: str = "Request would violate configured behavioral guardrails."):
        super().__init__(code="GUARDRAIL_VIOLATION", message=message, status=403)


class ScopeViolation(ALCMError):
    def __init__(self, message: str = "Requested package module not in the deal's data scope."):
        super().__init__(code="SCOPE_VIOLATION", message=message, status=403)


class InvalidRequest(ALCMError):
    def __init__(self, message: str = "Malformed request body or missing required fields."):
        super().__init__(code="INVALID_REQUEST", message=message, status=400)


class RateLimited(ALCMError):
    def __init__(self):
        super().__init__(code="RATE_LIMITED", message="Too many requests — retry after delay.", status=429)


class ServiceUnavailable(ALCMError):
    def __init__(self, message: str = "ALCM API is temporarily unavailable."):
        super().__init__(code="SERVICE_UNAVAILABLE", message=message, status=503)


def alcm_error_handler(request: Request, exc: ALCMError) -> JSONResponse:
    """Convert ALCMError to spec-compliant JSON response."""
    return JSONResponse(
        status_code=exc.error_status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "status": exc.error_status,
                "details": exc.details,
            }
        },
    )
