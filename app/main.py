"""ALCM API — Identity Intelligence Engine."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
from .database import init_db

logger = logging.getLogger(__name__)
from .routers import (
    twin_router, classify_router, generate_router,
    health_router, package_router, validate_router,
    guardrails_router, feedback_router, media_router,
    speech_router, drift_router, attribute_router,
    jobs_router, snapshot_router, admin_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ALCM API...")
    try:
        await init_db()
        logger.info("ALCM database connected")
    except Exception as e:
        logger.error(f"ALCM database connection failed: {e}", exc_info=True)

    from .services.llm_provider import get_llm_provider
    provider = get_llm_provider()
    logger.info(f"LLM provider: {type(provider).__name__} (configured: {provider.is_configured})")

    yield
    logger.info("Shutting down ALCM API...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="ALCM API",
        description="Identity Intelligence Engine — psychographic classification, personality generation, identity validation",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Input sanitization (size limits + injection protection)
    from .middleware.sanitize import SanitizationMiddleware
    app.add_middleware(SanitizationMiddleware)

    # Auth — bearer token required on all endpoints except /health
    from .middleware.auth import ALCMAuthMiddleware
    app.add_middleware(ALCMAuthMiddleware)

    # CORS — only the platform service should call this
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(twin_router)
    app.include_router(classify_router)
    app.include_router(generate_router)
    app.include_router(package_router)
    app.include_router(validate_router)
    app.include_router(guardrails_router)
    app.include_router(feedback_router)
    app.include_router(media_router)
    app.include_router(speech_router)
    app.include_router(drift_router)
    app.include_router(attribute_router)
    app.include_router(jobs_router)
    app.include_router(snapshot_router)
    app.include_router(admin_router)

    return app


app = create_app()
