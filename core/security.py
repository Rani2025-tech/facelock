"""Cryptographic encryption and integrity protection for biometric templates.

Uses AES-128 / Fernet symmetric encryption to secure 128-dimensional face embedding
vectors at rest, preventing plaintext biometric extraction or unauthorized template tampering.
"""

import base64
import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
import numpy as np

from config import BASE_DIR, EXPECTED_EMBEDDING_SHAPE

logger = logging.getLogger(__name__)

# Default master key file location
_KEY_FILE: Path = BASE_DIR / "data" / ".facelock_secret.key"


def get_or_create_master_key(key_file: Optional[Path] = None) -> bytes:
    """Retrieves or generates the persistent 256-bit Fernet encryption key.

    Args:
        key_file: Optional path to the key file. Defaults to `data/.facelock_secret.key`.

    Returns:
        A 32-byte base64-encoded URL-safe key.
    """
    if key_file is None:
        key_file = _KEY_FILE

    key_file = Path(key_file)
    if key_file.exists() and key_file.stat().st_size > 0:
        return key_file.read_bytes().strip()

    # Generate new Fernet key
    key_file.parent.mkdir(parents=True, exist_ok=True)
    new_key = Fernet.generate_key()
    key_file.write_bytes(new_key)
    logger.info("Generated new biometric master encryption key at '%s'.", key_file)
    return new_key


def encrypt_embedding(
    embedding: np.ndarray,
    key: Optional[bytes] = None,
) -> bytes:
    """Encrypts a 128-D NumPy embedding vector into an authenticated ciphertext payload.

    Args:
        embedding: A 1-D float64 NumPy array of shape (128,).
        key: Optional 32-byte base64 encryption key.

    Returns:
        Encrypted ciphertext bytes containing the serialized embedding and HMAC tag.

    Raises:
        ValueError: If embedding is None or not shape (128,).
    """
    if embedding is None or embedding.ndim != 1 or embedding.shape != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"Embedding must be a 1-D array of shape {EXPECTED_EMBEDDING_SHAPE}, "
            f"got {getattr(embedding, 'shape', None)}."
        )

    if key is None:
        key = get_or_create_master_key()

    fernet = Fernet(key)
    raw_bytes = embedding.astype(np.float64).tobytes()
    return fernet.encrypt(raw_bytes)


def decrypt_embedding(
    ciphertext: bytes,
    key: Optional[bytes] = None,
) -> np.ndarray:
    """Decrypts and validates an encrypted biometric payload back into a 128-D NumPy vector.

    Args:
        ciphertext: Encrypted payload bytes.
        key: Optional 32-byte base64 encryption key.

    Returns:
        A (128,) float64 NumPy array.

    Raises:
        ValueError: If ciphertext is corrupted, tampered with, or cannot be decrypted.
    """
    if not ciphertext or not isinstance(ciphertext, bytes):
        raise ValueError("Ciphertext must be non-empty bytes.")

    if key is None:
        key = get_or_create_master_key()

    fernet = Fernet(key)
    try:
        decrypted_raw = fernet.decrypt(ciphertext)
    except (InvalidToken, Exception) as exc:
        logger.error("Biometric decryption failed: corrupted ciphertext or invalid key.")
        raise ValueError(f"Biometric template decryption failed (corrupted or tampered): {exc}") from exc

    embedding = np.frombuffer(decrypted_raw, dtype=np.float64)
    if embedding.ndim != 1 or embedding.shape != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"Decrypted payload has invalid shape {embedding.shape}; expected {EXPECTED_EMBEDDING_SHAPE}."
        )

    return embedding
