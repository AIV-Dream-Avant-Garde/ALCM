"""Embedding generation service for RAG vector search.

Uses OpenAI's text-embedding-3-small by default (configurable).
Falls back to no-op if OPENAI_API_KEY is not set.
"""
import logging
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSION = 1536


async def generate_embedding(text: str) -> Optional[list[float]]:
    """Generate a vector embedding for the given text.

    Returns:
        List of floats (1536 dimensions) or None if service unavailable.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.embedding_model,
                    "input": text[:8000],  # Truncate to model limit
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        return None
