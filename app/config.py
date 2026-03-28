"""ALCM API settings loaded from environment variables."""
import logging
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://aiv:aiv_password@localhost:5432/alcm_db"

    # LLM Provider (gemini for dev, claude for production)
    llm_provider: str = "gemini"  # "gemini" | "claude"

    # Gemini (development default)
    google_genai_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Anthropic/Claude (production)
    anthropic_api_key: str = ""
    anthropic_model_primary: str = "claude-sonnet-4-20250514"
    anthropic_model_classification: str = "claude-haiku-4-5-20251001"

    # LLM timeouts (seconds)
    llm_request_timeout: int = 60
    llm_stream_timeout: int = 120

    # Embedding model (for RAG vector search)
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # Scraping config
    max_research_snippet_length: int = 2500

    # ElevenLabs (behind abstraction)
    elevenlabs_api_key: str = ""

    # Auth — NO DEFAULT. Must be set via .env or environment variable.
    alcm_service_token: str = ""

    # Storage
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "alcm-media"
    minio_secure: bool = False

    # Processing
    max_concurrent_jobs: int = 10
    job_timeout_seconds: int = 300

    # Generation
    generation_timeout_seconds: int = 30
    max_prompt_tokens: int = 27000
    max_output_tokens: int = 4000

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Sanitization
    max_body_size_bytes: int = 5 * 1024 * 1024  # 5MB

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def validate_settings(settings: Settings) -> list[str]:
    """Validate critical settings on startup. Returns list of warnings."""
    warnings = []
    if not settings.alcm_service_token:
        raise ValueError(
            "ALCM_SERVICE_TOKEN is not set. "
            "Set it in .env or as an environment variable. "
            "The API cannot start without authentication configured."
        )
    if len(settings.alcm_service_token) < 32:
        warnings.append("ALCM_SERVICE_TOKEN is shorter than 32 characters. Use a stronger token in production.")

    if settings.llm_provider == "gemini" and not settings.google_genai_api_key:
        warnings.append("LLM_PROVIDER is 'gemini' but GOOGLE_GENAI_API_KEY is not set.")
    if settings.llm_provider == "claude" and not settings.anthropic_api_key:
        warnings.append("LLM_PROVIDER is 'claude' but ANTHROPIC_API_KEY is not set.")

    return warnings


@lru_cache()
def get_settings() -> Settings:
    return Settings()
