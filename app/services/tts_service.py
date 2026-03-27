"""
TTS Service — Voice synthesis via ElevenLabs (abstracted for provider swap).

Moved from platform's voice_cloning_service.py. The platform NEVER calls
ElevenLabs directly — all voice operations go through the ALCM API.

Branded as "AIV voice technology" in all UI — no ElevenLabs mention.
"""

import os
import subprocess
import tempfile
import asyncio
import logging
from typing import Optional
from functools import lru_cache

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class TTSService:
    """Voice cloning and text-to-speech via ElevenLabs (behind abstraction)."""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.elevenlabs_api_key
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not configured. Voice features disabled.")

    @property
    def headers(self) -> dict:
        return {"xi-api-key": self.api_key}

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def create_voice_clone(
        self,
        name: str,
        audio_data: bytes,
        description: Optional[str] = None,
        audio_format: str = "wav",
    ) -> Optional[str]:
        """Create a voice clone from audio data. Returns voice_id or None."""
        if not self.is_configured:
            return None

        mime_types = {
            "wav": "audio/wav", "mp3": "audio/mpeg", "webm": "audio/webm",
            "ogg": "audio/ogg", "flac": "audio/flac",
        }
        mime_type = mime_types.get(audio_format.lower(), "audio/wav")

        logger.info(f"Creating voice clone '{name}' ({audio_format}, {len(audio_data)} bytes)")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                files = {"files": (f"voice_sample.{audio_format}", audio_data, mime_type)}
                data = {"name": name}
                if description:
                    data["description"] = description

                response = await client.post(
                    f"{self.BASE_URL}/voices/add",
                    headers=self.headers, files=files, data=data,
                )

                if response.status_code == 200:
                    voice_id = response.json().get("voice_id")
                    logger.info(f"Voice clone created: {voice_id}")
                    return voice_id
                else:
                    logger.error(f"Voice clone API error: {response.status_code} - {response.text[:300]}")
                    return None
        except Exception as e:
            logger.error(f"Voice clone failed: {e}")
            return None

    async def create_voice_clone_from_url(
        self, name: str, audio_url: str, description: Optional[str] = None,
    ) -> Optional[str]:
        """Create a voice clone from an audio URL (S3/MinIO)."""
        if not self.is_configured:
            return None
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(audio_url)
                if resp.status_code != 200:
                    logger.error(f"Failed to download audio from {audio_url}")
                    return None
                return await self.create_voice_clone(name, resp.content, description)
        except Exception as e:
            logger.error(f"Voice clone from URL failed: {e}")
            return None

    async def text_to_speech(
        self,
        voice_id: str,
        text: str,
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.45,
        similarity_boost: float = 0.90,
        style: float = 0.20,
    ) -> Optional[bytes]:
        """Generate speech from text using a cloned voice. Returns mp3 bytes."""
        if not self.is_configured:
            return None
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/text-to-speech/{voice_id}",
                    headers={**self.headers, "Content-Type": "application/json"},
                    json={
                        "text": text,
                        "model_id": model_id,
                        "voice_settings": {
                            "stability": stability,
                            "similarity_boost": similarity_boost,
                            "style": style,
                            "use_speaker_boost": True,
                        },
                    },
                )
                if response.status_code == 200:
                    return response.content
                logger.error(f"TTS error: {response.status_code} - {response.text[:200]}")
                return None
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice."""
        if not self.is_configured or not voice_id:
            return False
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.BASE_URL}/voices/{voice_id}", headers=self.headers,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Delete voice failed: {e}")
            return False


async def create_encrypted_voice_clone(
    name: str, audio_data: bytes, description: Optional[str] = None, audio_format: str = "wav",
) -> Optional[str]:
    """Create a voice clone and return the encrypted voice_id for storage."""
    from ..utils.encryption import encrypt_value
    tts = get_tts_service()
    voice_id = await tts.create_voice_clone(name, audio_data, description, audio_format)
    if voice_id:
        return encrypt_value(voice_id)
    return None


async def extract_audio_from_video(video_bytes: bytes, video_format: str = "webm") -> Optional[bytes]:
    """Extract audio track from video using ffmpeg. Runs in thread to avoid blocking."""

    def _run_ffmpeg() -> Optional[bytes]:
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{video_format}", delete=False) as vf:
                vf.write(video_bytes)
                video_path = vf.name

            audio_path = video_path.replace(f".{video_format}", ".wav")

            result = subprocess.run(
                ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", "-y", audio_path],
                capture_output=True, timeout=60,
            )

            if result.returncode == 0 and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(audio_path)
                os.unlink(video_path)
                return audio_bytes
            else:
                logger.error(f"FFmpeg error: {result.stderr.decode()[:200]}")
                try:
                    os.unlink(video_path)
                except Exception:
                    pass
                return None
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None

    return await asyncio.to_thread(_run_ffmpeg)


@lru_cache()
def get_tts_service() -> TTSService:
    return TTSService()
