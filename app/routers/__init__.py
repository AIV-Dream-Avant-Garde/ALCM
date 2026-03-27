"""ALCM API routers."""
from .twin import router as twin_router
from .classify import router as classify_router
from .generate import router as generate_router
from .health import router as health_router
from .package import router as package_router
from .validate import router as validate_router
from .guardrails import router as guardrails_router
from .feedback import router as feedback_router
from .media import router as media_router
from .speech import router as speech_router
from .drift import router as drift_router
from .attribute import router as attribute_router
from .jobs import router as jobs_router
from .snapshot import router as snapshot_router
from .admin import router as admin_router

__all__ = [
    "twin_router", "classify_router", "generate_router",
    "health_router", "package_router", "validate_router",
    "guardrails_router", "feedback_router", "media_router",
    "speech_router", "drift_router", "attribute_router",
    "jobs_router", "snapshot_router", "admin_router",
]
