"""Resilient LLM provider wrapper + circuit breaker."""
import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from .provider import AIProvider, LLMNotConfiguredError, LLMRequestError

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
BACKOFF_BASE = 1.0
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_SECONDS = 60


class CircuitBreaker:
    """Prevents repeated calls to a failing provider."""

    def __init__(self, name: str):
        self.name = name
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= CIRCUIT_BREAKER_THRESHOLD:
            self.is_open = True
            logger.warning(f"Circuit breaker OPEN for {self.name} after {self.failure_count} failures")

    def should_allow(self) -> bool:
        if not self.is_open:
            return True
        if time.time() - self.last_failure_time > CIRCUIT_BREAKER_RESET_SECONDS:
            logger.info(f"Circuit breaker HALF-OPEN for {self.name} — allowing probe request")
            return True
        return False


class ResilientLLMProvider(AIProvider):
    """Wraps primary + fallback providers with retry and circuit breaker."""

    def __init__(self, primary: AIProvider, fallback: Optional[AIProvider] = None):
        self.primary = primary
        self.fallback = fallback

    @property
    def is_configured(self) -> bool:
        return self.primary.is_configured or (self.fallback is not None and self.fallback.is_configured)

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    async def _call_with_retry(self, provider: AIProvider, method: str, *args, **kwargs):
        if not provider.is_configured:
            raise LLMNotConfiguredError(f"{provider.provider_name} not configured")
        if not provider.circuit.should_allow():
            raise LLMRequestError(f"{provider.provider_name} circuit breaker open")

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await getattr(provider, method)(*args, **kwargs)
            except LLMNotConfiguredError:
                raise
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"{provider.provider_name}.{method} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{provider.provider_name}.{method} exhausted {MAX_RETRIES + 1} attempts")

        raise LLMRequestError(f"{provider.provider_name} failed after retries: {last_error}")

    async def _call_with_fallback(self, method: str, *args, **kwargs):
        try:
            return await self._call_with_retry(self.primary, method, *args, **kwargs)
        except (LLMRequestError, LLMNotConfiguredError) as primary_error:
            if self.fallback is None or not self.fallback.is_configured:
                raise
            logger.warning(f"Primary ({self.primary.provider_name}) failed: {primary_error}. Falling back to {self.fallback.provider_name}.")

        return await self._call_with_retry(self.fallback, method, *args, **kwargs)

    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        return await self._call_with_fallback("generate", prompt, context=context,
                                               temperature=temperature, max_tokens=max_tokens)

    async def classify(self, content: str, instructions: str) -> dict:
        return await self._call_with_fallback("classify", content, instructions)

    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        providers = [self.primary]
        if self.fallback and self.fallback.is_configured:
            providers.append(self.fallback)

        for provider in providers:
            if not provider.is_configured or not provider.circuit.should_allow():
                continue
            try:
                async for chunk in provider.generate_stream(
                    prompt, context=context, temperature=temperature, max_tokens=max_tokens
                ):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"{provider.provider_name} stream failed: {e}")
                continue

        raise LLMRequestError("All providers failed for streaming")
