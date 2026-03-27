"""LLM Provider abstraction — swap between Gemini and Claude."""
import json
import logging
import httpx
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from functools import lru_cache

from ..config import get_settings

logger = logging.getLogger(__name__)


class LLMNotConfiguredError(Exception):
    """Raised when the LLM provider API key is missing."""
    pass


class LLMRequestError(Exception):
    """Raised when an LLM API call fails."""
    pass


class AIProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        ...

    @abstractmethod
    async def classify(self, content: str, instructions: str) -> dict:
        ...

    @abstractmethod
    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        ...


class GeminiProvider(AIProvider):
    """Gemini implementation — default for development (cheapest)."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.google_genai_api_key
        self.model = settings.gemini_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self, timeout: float = None):
        settings = get_settings()
        t = timeout or settings.llm_request_timeout
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
        return httpx.AsyncClient(timeout=httpx.Timeout(t, connect=10.0), transport=transport)

    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        if not self.is_configured:
            raise LLMNotConfiguredError("Gemini API key not set. Configure GOOGLE_GENAI_API_KEY in .env")

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

        try:
            async with self._get_client() as client:
                response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                for candidate in result.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "text" in part:
                            return part["text"]
            logger.warning(f"Gemini returned {response.status_code}: {response.text[:200]}")
            return ""
        except httpx.ConnectError as e:
            logger.error(f"Gemini connection failed: {e}")
            raise LLMRequestError(f"Cannot reach Gemini API: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Gemini request timed out: {e}")
            raise LLMRequestError(f"Gemini API timed out: {e}")
        except Exception as e:
            logger.error(f"Gemini generate failed: {type(e).__name__}: {e}")
            raise LLMRequestError(f"Gemini generation failed: {e}")

    async def classify(self, content: str, instructions: str) -> dict:
        result = await self.generate(
            prompt=f"{instructions}\n\nContent to classify:\n{content}",
            temperature=0.3,
            max_tokens=2000,
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_response": result, "error": "Failed to parse as JSON"}

    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise LLMNotConfiguredError("Gemini API key not set for streaming")

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

        async with self._get_client(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    return
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
    """Claude implementation — production provider (best for personality embodiment)."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.anthropic_api_key
        self.model_primary = settings.anthropic_model_primary
        self.model_classification = settings.anthropic_model_classification

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def generate(self, prompt: str, context: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 2000) -> str:
        if not self.is_configured:
            raise LLMNotConfiguredError("Anthropic API key not set. Configure ANTHROPIC_API_KEY in .env")

        messages = []
        if context:
            messages.append({"role": "user", "content": context})
            messages.append({"role": "assistant", "content": "Understood. I have this context."})
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
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
                        return block["text"]
            logger.warning(f"Claude returned {response.status_code}: {response.text[:200]}")
            return ""
        except httpx.ConnectError as e:
            logger.error(f"Claude connection failed: {e}")
            raise LLMRequestError(f"Cannot reach Claude API: {e}")
        except httpx.TimeoutException as e:
            logger.error(f"Claude request timed out: {e}")
            raise LLMRequestError(f"Claude API timed out: {e}")
        except Exception as e:
            logger.error(f"Claude generate failed: {type(e).__name__}: {e}")
            raise LLMRequestError(f"Claude generation failed: {e}")

    async def classify(self, content: str, instructions: str) -> dict:
        if not self.is_configured:
            raise LLMNotConfiguredError("Anthropic API key not set for classification")

        messages = [{"role": "user", "content": f"{instructions}\n\nContent to classify:\n{content}"}]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
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
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw_response": text, "error": "Failed to parse as JSON"}
                return {}
        except httpx.ConnectError as e:
            raise LLMRequestError(f"Cannot reach Claude API: {e}")
        except httpx.TimeoutException as e:
            raise LLMRequestError(f"Claude classification timed out: {e}")

    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 2000) -> AsyncGenerator[str, None]:
        if not self.is_configured:
            raise LLMNotConfiguredError("Anthropic API key not set for streaming")

        messages = []
        if context:
            messages.append({"role": "user", "content": context})
            messages.append({"role": "assistant", "content": "Understood."})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
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
                    return
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


@lru_cache()
def get_llm_provider() -> AIProvider:
    """Get the configured LLM provider."""
    settings = get_settings()
    if settings.llm_provider == "claude" and settings.anthropic_api_key:
        print("Using Claude (Anthropic) as LLM provider")
        return ClaudeProvider()
    print("Using Gemini (Google) as LLM provider")
    return GeminiProvider()
