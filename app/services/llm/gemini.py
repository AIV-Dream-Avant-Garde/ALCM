"""Gemini LLM provider implementation."""
import json
import logging
import httpx
from typing import AsyncGenerator, Optional

from app.config import get_settings
from .provider import AIProvider, LLMNotConfiguredError, LLMRequestError
from .resilient import CircuitBreaker

logger = logging.getLogger(__name__)


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
