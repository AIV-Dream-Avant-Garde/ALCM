"""Encryption utility for sensitive ALCM data (voice IDs, embeddings).

Uses Fernet symmetric encryption. In production, the key should be
stored in a secrets manager (AWS KMS, HashiCorp Vault, etc.).

Usage:
    from app.utils.encryption import encrypt_value, decrypt_value
    encrypted = encrypt_value("sk-elevenlabs-voice-id-123")
    original = decrypt_value(encrypted)
"""

import base64
import hashlib
import logging
from typing import Optional

from cryptography.fernet import Fernet

from ..config import get_settings

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """Derive a Fernet key from the ALCM auth token (deterministic)."""
    settings = get_settings()
    # Use the auth token as seed material for the encryption key
    key_material = settings.alcm_auth_token.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(key_material).digest())
    return Fernet(key)


def encrypt_value(plaintext: str) -> str:
    """Encrypt a string value. Returns base64-encoded ciphertext."""
    if not plaintext:
        return ""
    try:
        f = _get_fernet()
        return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return plaintext  # Fallback to plaintext in dev


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a previously encrypted value."""
    if not ciphertext:
        return ""
    try:
        f = _get_fernet()
        return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as e:
        # If decryption fails, it might be plaintext (migration period)
        logger.warning(f"Decryption failed (may be plaintext): {e}")
        return ciphertext
