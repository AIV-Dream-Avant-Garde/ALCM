"""Claude LLM provider implementation."""
import json
import logging
import httpx
from typing import AsyncGenerator, Optional

from app.config import get_settings
from .provider import AIProvider, LLMNotConfiguredError, LLMRequestError
from .resilient import CircuitBreaker

logger = logging.getLogger(__name__)


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
                    "anthropic-version": "2023-06-01",
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
                    "anthropic-version": "2023-06-01",
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
