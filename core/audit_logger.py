"""Security event logging and audit trail system for FaceLock.

Provides structured, append-only security event logging with log rotation,
strict biometric/credential privacy sanitization, and structured querying.

Security Events Logged:
- REGISTRATION_SUCCESS
- REGISTRATION_FAILURE
- LIVENESS_SUCCESS
- LIVENESS_FAILURE
- AUTHENTICATION_SUCCESS
- AUTHENTICATION_FAILURE
- CAMERA_ERROR
- STORAGE_ERROR

PRIVACY GUARANTEE:
Logs MUST NOT and DO NOT contain passwords, raw encryption keys,
raw biometric embedding vectors, facial landmark coordinates, or raw images.
"""

import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np

from config import (
    AUDIT_LOG_BACKUP_COUNT,
    AUDIT_LOG_MAX_BYTES,
    DEFAULT_AUDIT_LOG_FILE,
    DEFAULT_LOGS_DIR,
)

# ── Event Types Enum ──────────────────────────────────────────────────────────

class SecurityEventType(str, Enum):
    """Enumeration of all auditable security lifecycle events."""

    REGISTRATION_SUCCESS = "REGISTRATION_SUCCESS"
    REGISTRATION_FAILURE = "REGISTRATION_FAILURE"
    LIVENESS_SUCCESS = "LIVENESS_SUCCESS"
    LIVENESS_FAILURE = "LIVENESS_FAILURE"
    AUTHENTICATION_SUCCESS = "AUTHENTICATION_SUCCESS"
    AUTHENTICATION_FAILURE = "AUTHENTICATION_FAILURE"
    CAMERA_ERROR = "CAMERA_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"


# Blacklisted parameter keys that could contain sensitive credentials/biometrics
_SENSITIVE_KEY_SUBSTRINGS: Set[str] = {
    "password",
    "secret",
    "key",
    "token",
    "embedding",
    "vector",
    "template",
    "biometric",
    "landmark",
    "coordinate",
    "frame_data",
    "raw_data",
    "face_encoding",
}


@dataclass
class AuditRecord:
    """Represents a structured, privacy-compliant security log record.

    Attributes:
        timestamp: ISO 8601 formatted timestamp string.
        event_type: The SecurityEventType or custom event string.
        user_id: Associated user identifier, or None if anonymous/unidentified.
        status: Status code ("SUCCESS", "FAILURE", "ERROR", "WARNING", "INFO").
        reason: Explanatory message or diagnostic detail.
        details: Sanitized metadata dictionary (metrics, sample counts, thresholds).
    """

    timestamp: str
    event_type: str
    user_id: Optional[str]
    status: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the record to a serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "status": self.status,
            "reason": self.reason,
            "details": self.details,
        }

    def to_json(self) -> str:
        """Serializes the record to a single-line JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Optional["AuditRecord"]:
        """Deserializes an AuditRecord from a JSON string, returning None if invalid."""
        try:
            data = json.loads(line.strip())
            return cls(
                timestamp=data.get("timestamp", ""),
                event_type=data.get("event_type", "UNKNOWN"),
                user_id=data.get("user_id"),
                status=data.get("status", "INFO"),
                reason=data.get("reason", ""),
                details=data.get("details", {}),
            )
        except (json.JSONDecodeError, AttributeError):
            return None


def sanitize_details(details: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitizes event metadata to strictly prevent biometric/key leakage.

    Removes or redacts any fields matching sensitive keywords, raw numpy arrays,
    binary data, or high-dimensional float collections.
    """
    if not details or not isinstance(details, dict):
        return {}

    sanitized: Dict[str, Any] = {}

    for key, value in details.items():
        key_str = str(key).lower()

        # Check for sensitive key names
        if any(sub in key_str for sub in _SENSITIVE_KEY_SUBSTRINGS):
            continue

        # Prevent numpy array or bytes leakage
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (bytes, bytearray)):
            continue

        # Convert numpy numeric types to standard Python types
        if isinstance(value, (np.floating, float)):
            sanitized[key] = round(float(value), 4)
        elif isinstance(value, (np.integer, int)):
            sanitized[key] = int(value)
        elif isinstance(value, (np.bool_, bool)):
            sanitized[key] = bool(value)
        elif isinstance(value, (str, list, tuple)):
            if isinstance(value, (list, tuple)):
                # If it looks like a large embedding vector, drop it
                if len(value) > 16:
                    continue
                sanitized[key] = list(value)
            else:
                sanitized[key] = str(value)[:256]
        else:
            sanitized[key] = str(value)[:128]

    return sanitized


# ── Logger Manager ────────────────────────────────────────────────────────────

_audit_logger_instance: Optional[logging.Logger] = None
_current_log_path: Optional[Path] = None


def get_audit_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """Configures and returns a dedicated rotating file logger for audit events."""
    global _audit_logger_instance, _current_log_path

    target_path = Path(log_file) if log_file else DEFAULT_AUDIT_LOG_FILE

    # Reconfigure if target path changed
    if _audit_logger_instance is not None and _current_log_path == target_path:
        return _audit_logger_instance

    logger_name = f"facelock.audit.{abs(hash(str(target_path)))}"
    audit_logger = logging.getLogger(logger_name)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Avoid polluting standard root stdout logger

    # Ensure parent directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Clear existing handlers
    audit_logger.handlers.clear()

    # Create rotating file handler
    handler = RotatingFileHandler(
        filename=str(target_path),
        maxBytes=AUDIT_LOG_MAX_BYTES,
        backupCount=AUDIT_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    audit_logger.addHandler(handler)

    _audit_logger_instance = audit_logger
    _current_log_path = target_path

    return audit_logger


def log_security_event(
    event_type: Union[SecurityEventType, str],
    user_id: Optional[str] = None,
    status: str = "INFO",
    reason: str = "",
    details: Optional[Dict[str, Any]] = None,
    log_file: Optional[Path] = None,
) -> AuditRecord:
    """Logs a sanitized security event to the audit trail file.

    Args:
        event_type: Event classification from SecurityEventType or string.
        user_id: Optional subject identifier.
        status: Event outcome (SUCCESS, FAILURE, ERROR, WARNING, INFO).
        reason: Diagnostic or human-readable explanation.
        details: Optional non-sensitive telemetry parameters.
        log_file: Optional path override for audit log destination.

    Returns:
        The generated AuditRecord instance.
    """
    event_name = event_type.value if isinstance(event_type, SecurityEventType) else str(event_type)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clean_details = sanitize_details(details)

    record = AuditRecord(
        timestamp=timestamp,
        event_type=event_name,
        user_id=user_id,
        status=status.upper(),
        reason=str(reason),
        details=clean_details,
    )

    try:
        logger_inst = get_audit_logger(log_file=log_file)
        logger_inst.info(record.to_json())
    except Exception as exc:
        # Fallback to stderr if file logging fails, never crash the host application
        logging.getLogger(__name__).error("Failed to write security audit log: %s", exc)

    return record


def get_audit_records(
    limit: int = 100,
    log_file: Optional[Path] = None,
) -> List[AuditRecord]:
    """Reads and parses the most recent security audit records from the log file.

    Args:
        limit: Maximum number of recent records to return.
        log_file: Optional path override to audit log file.

    Returns:
        A list of parsed AuditRecord objects in chronological order.
    """
    target_path = Path(log_file) if log_file else DEFAULT_AUDIT_LOG_FILE
    if not target_path.exists():
        return []

    records: List[AuditRecord] = []
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines[-limit:]:
            rec = AuditRecord.from_json(line)
            if rec is not None:
                records.append(rec)
    except Exception as exc:
        logging.getLogger(__name__).warning("Error reading audit records from '%s': %s", target_path, exc)

    return records


def clear_audit_logs(log_file: Optional[Path] = None) -> None:
    """Clears the audit log file (used primarily for test isolation)."""
    global _audit_logger_instance
    target_path = Path(log_file) if log_file else DEFAULT_AUDIT_LOG_FILE
    if _audit_logger_instance is not None:
        for handler in _audit_logger_instance.handlers:
            handler.close()
        _audit_logger_instance.handlers.clear()
        _audit_logger_instance = None

    if target_path.exists():
        try:
            target_path.unlink()
        except OSError:
            pass
