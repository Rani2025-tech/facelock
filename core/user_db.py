"""SQLite-backed user repository and encrypted biometric template storage.

Manages structured user account metadata (user_id, display_name, timestamps) in SQLite,
while storing encrypted 128-D biometric vectors in dedicated per-user directories on disk.
"""

import datetime
import logging
import os
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import BASE_DIR
from core.audit_logger import SecurityEventType, log_security_event
from core.security import decrypt_embedding, encrypt_embedding

logger = logging.getLogger(__name__)

# Default locations
DEFAULT_DATA_DIR: Path = BASE_DIR / "data"
DEFAULT_DB_PATH: Path = DEFAULT_DATA_DIR / "facelock_users.db"
DEFAULT_USERS_DIR: Path = DEFAULT_DATA_DIR / "users"

# User ID regex validation (letters, numbers, underscores, dashes, 2-32 chars)
_USER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{2,32}$")


@dataclass
class UserRecord:
    """Represents a registered user profile.

    Attributes:
        user_id: Unique alphanumeric identifier.
        display_name: Human-friendly full name or handle.
        created_at: ISO 8601 creation timestamp.
        updated_at: ISO 8601 last modified timestamp.
        template_file: Path to the encrypted biometric template file.
    """

    user_id: str
    display_name: str
    created_at: str
    updated_at: str
    template_file: str


def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Establishes an SQLite connection and creates the schema if necessary."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            template_file TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def validate_user_id(user_id: str) -> str:
    """Validates and cleans a user ID string.

    Args:
        user_id: The candidate user ID string.

    Returns:
        The stripped user ID.

    Raises:
        ValueError: If user_id is empty or contains invalid characters.
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID must be a non-empty string.")
    cleaned = user_id.strip()
    if not _USER_ID_REGEX.match(cleaned):
        raise ValueError(
            f"User ID '{cleaned}' is invalid. Must be 2-32 characters (alphanumeric, '_' or '-')."
        )
    return cleaned


def create_user(
    user_id: str,
    display_name: str,
    embedding: np.ndarray,
    db_path: Optional[Path] = None,
    users_dir: Optional[Path] = None,
) -> UserRecord:
    """Enrolls a new user with an encrypted biometric template and stores metadata in SQLite.

    Args:
        user_id: Unique user identifier.
        display_name: Display name.
        embedding: Normalized 128-D face embedding vector.
        db_path: Optional SQLite database file path.
        users_dir: Optional directory for storing encrypted template files.

    Returns:
        The newly created UserRecord.

    Raises:
        ValueError: If user_id already exists or embedding is invalid.
    """
    clean_id = validate_user_id(user_id)
    clean_name = display_name.strip() if display_name else clean_id

    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if users_dir is None:
        users_dir = DEFAULT_USERS_DIR

    db_path = Path(db_path)
    users_dir = Path(users_dir)

    conn = _get_connection(db_path)
    try:
        # Check for existing user
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (clean_id,))
        if cursor.fetchone() is not None:
            raise ValueError(f"User with ID '{clean_id}' already exists.")

        # Encrypt embedding
        ciphertext = encrypt_embedding(embedding)

        # Save encrypted template file atomically
        user_dir = users_dir / clean_id
        user_dir.mkdir(parents=True, exist_ok=True)
        template_file = user_dir / "template.enc"
        tmp_file = user_dir / "template.tmp.enc"

        tmp_file.write_bytes(ciphertext)
        os.replace(tmp_file, template_file)

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.execute(
            """
            INSERT INTO users (user_id, display_name, created_at, updated_at, template_file)
            VALUES (?, ?, ?, ?, ?)
            """,
            (clean_id, clean_name, now_iso, now_iso, str(template_file)),
        )
        conn.commit()

        logger.info("Successfully enrolled user '%s' (%s).", clean_name, clean_id)
        return UserRecord(
            user_id=clean_id,
            display_name=clean_name,
            created_at=now_iso,
            updated_at=now_iso,
            template_file=str(template_file),
        )
    finally:
        conn.close()


def create_or_update_user(
    user_id: str,
    display_name: str,
    embedding: np.ndarray,
    db_path: Optional[Path] = None,
    users_dir: Optional[Path] = None,
) -> UserRecord:
    """Enrolls a new user or updates the biometric template of an existing user.

    Args:
        user_id: Unique user identifier.
        display_name: Display name.
        embedding: Normalized 128-D face embedding vector.
        db_path: Optional SQLite database file path.
        users_dir: Optional directory for storing encrypted template files.

    Returns:
        The created or updated UserRecord.
    """
    clean_id = validate_user_id(user_id)
    clean_name = display_name.strip() if display_name else clean_id

    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if users_dir is None:
        users_dir = DEFAULT_USERS_DIR

    db_path = Path(db_path)
    users_dir = Path(users_dir)

    conn = _get_connection(db_path)
    try:
        cursor = conn.cursor()
        ciphertext = encrypt_embedding(embedding)

        user_dir = users_dir / clean_id
        user_dir.mkdir(parents=True, exist_ok=True)
        template_file = user_dir / "template.enc"
        tmp_file = user_dir / "template.tmp.enc"

        tmp_file.write_bytes(ciphertext)
        os.replace(tmp_file, template_file)

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        cursor.execute("SELECT user_id, created_at FROM users WHERE user_id = ?", (clean_id,))
        row = cursor.fetchone()
        if row is not None:
            created_at = row[1]
            cursor.execute(
                "UPDATE users SET display_name = ?, updated_at = ?, template_file = ? WHERE user_id = ?",
                (clean_name, now_iso, str(template_file), clean_id),
            )
        else:
            created_at = now_iso
            cursor.execute(
                "INSERT INTO users (user_id, display_name, created_at, updated_at, template_file) VALUES (?, ?, ?, ?, ?)",
                (clean_id, clean_name, created_at, now_iso, str(template_file)),
            )
        conn.commit()

        logger.info("Created or updated user '%s' (%s).", clean_name, clean_id)
        return UserRecord(
            user_id=clean_id,
            display_name=clean_name,
            created_at=created_at,
            updated_at=now_iso,
            template_file=str(template_file),
        )
    finally:
        conn.close()


def get_user(
    user_id: str,
    db_path: Optional[Path] = None,
) -> Optional[UserRecord]:
    """Retrieves a single user record from SQLite by user_id."""
    clean_id = user_id.strip()
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = _get_connection(Path(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, display_name, created_at, updated_at, template_file FROM users WHERE user_id = ?",
            (clean_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return UserRecord(
            user_id=row[0],
            display_name=row[1],
            created_at=row[2],
            updated_at=row[3],
            template_file=row[4],
        )
    finally:
        conn.close()


def list_users(db_path: Optional[Path] = None) -> List[UserRecord]:
    """Retrieves all registered user profiles ordered by creation date."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = _get_connection(Path(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id, display_name, created_at, updated_at, template_file FROM users ORDER BY created_at ASC"
        )
        rows = cursor.fetchall()
        return [
            UserRecord(
                user_id=r[0],
                display_name=r[1],
                created_at=r[2],
                updated_at=r[3],
                template_file=r[4],
            )
            for r in rows
        ]
    finally:
        conn.close()


def delete_user(
    user_id: str,
    db_path: Optional[Path] = None,
    users_dir: Optional[Path] = None,
) -> bool:
    """Deletes a user record from SQLite and purges their encrypted biometric template directory.

    Args:
        user_id: Identifier of the user to remove.
        db_path: Optional SQLite path.
        users_dir: Optional users folder path.

    Returns:
        True if the user was found and deleted, False otherwise.
    """
    clean_id = user_id.strip()
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if users_dir is None:
        users_dir = DEFAULT_USERS_DIR

    conn = _get_connection(Path(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id = ?", (clean_id,))
        deleted = cursor.rowcount > 0
        conn.commit()

        # Delete user directory on disk
        user_dir = Path(users_dir) / clean_id
        if user_dir.exists():
            shutil.rmtree(user_dir, ignore_errors=True)

        if deleted:
            logger.info("Deleted user '%s' and purged biometric templates.", clean_id)
        return deleted
    finally:
        conn.close()


def load_user_embedding(
    user_id: str,
    db_path: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Loads and decrypts a user's 128-D biometric template from disk.

    Args:
        user_id: User identifier.
        db_path: Optional database path.

    Returns:
        Decrypted (128,) float64 NumPy array, or None if user/file not found.

    Raises:
        ValueError: If template file is corrupted or fails cryptographic validation.
    """
    user = get_user(user_id, db_path=db_path)
    if user is None:
        return None

    template_path = Path(user.template_file)
    if not template_path.exists() or template_path.stat().st_size == 0:
        logger.error("Template file missing or empty for user '%s': %s", user_id, template_path)
        return None

    ciphertext = template_path.read_bytes()
    return decrypt_embedding(ciphertext)


def load_all_user_embeddings(
    db_path: Optional[Path] = None,
) -> Dict[str, Tuple[UserRecord, np.ndarray]]:
    """Loads and decrypts all enrolled user biometric templates in one pass.

    Returns:
        Dictionary mapping user_id -> (UserRecord, 128-D embedding).
        Corrupted user records are skipped with logged warnings.
    """
    users = list_users(db_path=db_path)
    result: Dict[str, Tuple[UserRecord, np.ndarray]] = {}

    for user in users:
        try:
            emb = load_user_embedding(user.user_id, db_path=db_path)
            if emb is not None:
                result[user.user_id] = (user, emb)
        except Exception as exc:
            logger.warning(
                "Skipping corrupted template for user '%s' (%s): %s",
                user.display_name,
                user.user_id,
                exc,
            )
            log_security_event(
                SecurityEventType.STORAGE_ERROR,
                user_id=user.user_id,
                status="ERROR",
                reason=f"Corrupted template skipped: {exc}",
            )

    return result


def user_count(db_path: Optional[Path] = None) -> int:
    """Returns the total number of registered users."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = _get_connection(Path(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        row = cursor.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()
