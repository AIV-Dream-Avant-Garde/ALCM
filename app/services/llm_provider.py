"""LLM Provider abstraction with retry, fallback, circuit breaker, and config-driven timeouts.

Supports Gemini (dev) and Claude (production). If the primary provider fails,
automatically falls back to the secondary. Exponential backoff on retries.
Circuit breaker prevents hammering a down provider.
"""
import asyncio
import json
import logging
import time
import httpx
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
BACKOFF_BASE = 1.0  # seconds
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive failures before opening circuit
CIRCUIT_BREAKER_RESET_SECONDS = 60


class LLMNotConfiguredError(Exception):
    pass


class LLMRequestError(Exception):
    pass


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


class GeminiProvider(AIProvider):
    """Gemini implementation — default for development."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.google_genai_api_key
        self.model = settings.gemini_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.circuit = CircuitBreaker("gemini")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def _get_client(self, timeout: float = None):
        settings = get_settings()
        t = timeout or settings.llm_request_timeout
        return httpx.AsyncClient(timeout=httpx.Timeout(t, connect=10.0))

    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        if not self.is_configured:
            raise LLMNotConfiguredError("Gemini API key not set")

        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        contents = []
        if context:
            contents.append({"role": "user", "parts": [{"text": context}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I have this context."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        async with self._get_client() as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        self.circuit.record_success()
                        return part["text"]

        self.circuit.record_failure()
        raise LLMRequestError(f"Gemini returned {response.status_code}: {response.text[:200]}")

    async def classify(self, content: str, instructions: str) -> dict:
        result = await self.generate(
            prompt=f"{instructions}\n\nContent to classify:\n{content}",
            temperature=0.3, max_tokens=2000,
        )
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return {"raw_response": result, "error": "Failed to parse as JSON"}

    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise LLMNotConfiguredError("Gemini API key not set")

        settings = get_settings()
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

        contents = []
        if context:
            contents.append({"role": "user", "parts": [{"text": context}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        async with self._get_client(timeout=settings.llm_stream_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    self.circuit.record_failure()
                    return
                self.circuit.record_success()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        for candidate in data.get("candidates", []):
                            for part in candidate.get("content", {}).get("parts", []):
                                if "text" in part:
                                    yield part["text"]
                    except json.JSONDecodeError:
                        continue


class ClaudeProvider(AIProvider):
    """Claude implementation — production provider."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.anthropic_api_key
        self.model_primary = settings.anthropic_model_primary
        self.model_classification = settings.anthropic_model_classification
        self.circuit = CircuitBreaker("claude")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def provider_name(self) -> str:
        return "claude"

    def _get_client(self, timeout: float = None):
        settings = get_settings()
        t = timeout or settings.llm_request_timeout
        return httpx.AsyncClient(timeout=httpx.Timeout(t, connect=10.0))

    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        if not self.is_configured:
            raise LLMNotConfiguredError("Anthropic API key not set")

        messages = []
        if context:
            messages.append({"role": "user", "content": context})
            messages.append({"role": "assistant", "content": "Understood. I have this context."})
        messages.append({"role": "user", "content": prompt})

        async with self._get_client() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2024-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model_primary,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                },
            )

        if response.status_code == 200:
            result = response.json()
            for block in result.get("content", []):
                if block.get("type") == "text":
                    self.circuit.record_success()
                    return block["text"]

        self.circuit.record_failure()
        raise LLMRequestError(f"Claude returned {response.status_code}: {response.text[:200]}")

    async def classify(self, content: str, instructions: str) -> dict:
        if not self.is_configured:
            raise LLMNotConfiguredError("Anthropic API key not set")

        messages = [{"role": "user", "content": f"{instructions}\n\nContent to classify:\n{content}"}]

        async with self._get_client() as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2024-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model_classification,
                    "max_tokens": 2000,
                    "temperature": 0.3,
                    "messages": messages,
                },
            )

        if response.status_code == 200:
            result = response.json()
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text = block["text"]
                    break
            self.circuit.record_success()
            try:
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                return json.loads(cleaned.strip())
            except json.JSONDecodeError:
                return {"raw_response": text, "error": "Failed to parse as JSON"}

        self.circuit.record_failure()
        raise LLMRequestError(f"Claude classify returned {response.status_code}")

    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise LLMNotConfiguredError("Anthropic API key not set")

        settings = get_settings()
        messages = []
        if context:
            messages.append({"role": "user", "content": context})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": prompt})

        async with self._get_client(timeout=settings.llm_stream_timeout) as client:
            async with client.stream(
                "POST", "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2024-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model_primary,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "messages": messages,
                },
            ) as response:
                if response.status_code != 200:
                    self.circuit.record_failure()
                    return
                self.circuit.record_success()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta["text"]
                    except json.JSONDecodeError:
                        continue


class ResilientLLMProvider(AIProvider):
    """Wraps primary + fallback providers with retry and circuit breaker.

    On every call:
    1. Check circuit breaker — skip provider if open
    2. Try primary with exponential backoff (up to MAX_RETRIES)
    3. If primary exhausted, try fallback with same retry logic
    4. If both fail, raise LLMRequestError
    """

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
        """Call a provider method with exponential backoff retry."""
        if not provider.is_configured:
            raise LLMNotConfiguredError(f"{provider.provider_name} not configured")
        if not provider.circuit.should_allow():
            raise LLMRequestError(f"{provider.provider_name} circuit breaker open")

        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await getattr(provider, method)(*args, **kwargs)
                return result
            except LLMNotConfiguredError:
                raise
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"{provider.provider_name}.{method} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{provider.provider_name}.{method} exhausted {MAX_RETRIES + 1} attempts")

        raise LLMRequestError(f"{provider.provider_name} failed after retries: {last_error}")

    async def _call_with_fallback(self, method: str, *args, **kwargs):
        """Try primary, fall back to secondary if primary fails."""
        # Try primary
        try:
            return await self._call_with_retry(self.primary, method, *args, **kwargs)
        except (LLMRequestError, LLMNotConfiguredError) as primary_error:
            if self.fallback is None or not self.fallback.is_configured:
                raise
            logger.warning(
                f"Primary ({self.primary.provider_name}) failed: {primary_error}. "
                f"Falling back to {self.fallback.provider_name}."
            )

        # Try fallback
        return await self._call_with_retry(self.fallback, method, *args, **kwargs)

    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        return await self._call_with_fallback("generate", prompt, context=context,
                                               temperature=temperature, max_tokens=max_tokens)

    async def classify(self, content: str, instructions: str) -> dict:
        return await self._call_with_fallback("classify", content, instructions)

    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        # Streaming doesn't support fallback mid-stream — try primary, then fallback
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


@lru_cache()
def get_llm_provider() -> AIProvider:
    """Get the resilient LLM provider with primary + fallback."""
    settings = get_settings()

    claude = ClaudeProvider()
    gemini = GeminiProvider()

    if settings.llm_provider == "claude":
        primary, fallback = claude, gemini
    else:
        primary, fallback = gemini, claude

    # Only set fallback if it's actually configured
    fb = fallback if fallback.is_configured else None

    logger.info(
        f"LLM provider: primary={primary.provider_name} "
        f"(configured={primary.is_configured}), "
        f"fallback={fb.provider_name if fb else 'none'}"
    )

    return ResilientLLMProvider(primary, fb)
