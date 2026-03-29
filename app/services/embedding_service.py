"""Embedding generation service for RAG vector search.

Uses Google's text-embedding-004 model via the Gemini API.
Falls back to no-op if GOOGLE_GENAI_API_KEY is not set.
"""
import logging
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 768  # Google text-embedding-004 outputs 768 dimensions


async def generate_embedding(text: str) -> Optional[list]:
    """Generate a vector embedding for the given text.

    Returns:
        List of floats (768 dimensions) or None if service unavailable.
    """
    settings = get_settings()
    if not settings.google_genai_api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.embedding_model}:embedContent",
                headers={"Content-Type": "application/json"},
                params={"key": settings.google_genai_api_key},
                json={
                    "model": f"models/{settings.embedding_model}",
                    "content": {"parts": [{"text": text[:8000]}]},
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]["values"]
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return None
