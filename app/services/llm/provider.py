"""AIProvider abstract base, exceptions, and factory function."""
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class LLMNotConfiguredError(Exception):
    pass


class LLMRequestError(Exception):
    pass


class AIProvider(ABC):
    """Abstract base for LLM providers."""

    @property
    @abstractmethod
    def is_configured(self) -> bool: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str: ...

    @abstractmethod
    async def classify(self, content: str, instructions: str) -> dict: ...

    @abstractmethod
    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]: ...


@lru_cache()
def get_llm_provider() -> AIProvider:
    """Get the resilient LLM provider with primary + fallback."""
    from app.config import get_settings
    from .gemini import GeminiProvider
    from .claude import ClaudeProvider
    from .resilient import ResilientLLMProvider

    settings = get_settings()
    claude = ClaudeProvider()
    gemini = GeminiProvider()

    if settings.llm_provider == "claude":
        primary, fallback = claude, gemini
    else:
        primary, fallback = gemini, claude

    fb = fallback if fallback.is_configured else None

    logger.info(
        f"LLM provider: primary={primary.provider_name} "
        f"(configured={primary.is_configured}), "
        f"fallback={fb.provider_name if fb else 'none'}"
    )

    return ResilientLLMProvider(primary, fb)
