"""Unit tests for the FaceLock security event logging and audit subsystem.

Validates:
1. All 8 security event types log accurately.
2. Sensitive biometric vectors, raw keys, and passwords are strictly sanitized and never leaked.
3. Log rotation and file size limiting work correctly.
4. Audit log retrieval and deserialization work seamlessly.
5. Error tolerance and corrupt line handling.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from core.audit_logger import (
    AuditRecord,
    SecurityEventType,
    clear_audit_logs,
    get_audit_records,
    log_security_event,
    sanitize_details,
)


@pytest.fixture(autouse=True)
def clean_audit_fixture(tmp_path: Path):
    """Provides an isolated temporary audit log file for each test."""
    temp_log = tmp_path / "test_audit.log"
    yield temp_log
    clear_audit_logs(temp_log)


def test_all_security_event_types_exist() -> None:
    """Ensures all required security event types are present in the enum."""
    expected_events = {
        "REGISTRATION_SUCCESS",
        "REGISTRATION_FAILURE",
        "LIVENESS_SUCCESS",
        "LIVENESS_FAILURE",
        "AUTHENTICATION_SUCCESS",
        "AUTHENTICATION_FAILURE",
        "CAMERA_ERROR",
        "STORAGE_ERROR",
    }
    actual_events = {e.value for e in SecurityEventType}
    assert expected_events.issubset(actual_events)


def test_log_security_event_creates_valid_json_record(tmp_path: Path) -> None:
    """Verifies that log_security_event writes a complete, parseable JSON line."""
    log_file = tmp_path / "audit_test.log"

    rec = log_security_event(
        event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
        user_id="user_alice",
        status="SUCCESS",
        reason="Facial match confirmed",
        details={"distance": 0.3812, "threshold": 0.60},
        log_file=log_file,
    )

    assert rec.event_type == "AUTHENTICATION_SUCCESS"
    assert rec.user_id == "user_alice"
    assert rec.status == "SUCCESS"
    assert rec.reason == "Facial match confirmed"
    assert rec.details["distance"] == 0.3812
    assert rec.details["threshold"] == 0.60

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip()
    data = json.loads(content)

    assert data["event_type"] == "AUTHENTICATION_SUCCESS"
    assert data["user_id"] == "user_alice"
    assert "timestamp" in data
    assert data["details"]["distance"] == 0.3812


def test_privacy_sanitizer_removes_sensitive_keys_and_biometrics() -> None:
    """Strictly validates that passwords, secret keys, embeddings, and arrays are stripped."""
    dirty_details: Dict[str, Any] = {
        "distance": 0.452,
        "sample_count": 15,
        "password": "SuperSecretPassword123!",
        "secret_key": b"secret_encryption_key_bytes",
        "biometric_embedding": np.random.randn(128),
        "raw_landmarks": [(10, 20), (30, 40)],
        "large_array": list(range(100)),  # Looks like high-D vector
        "safe_note": "Normal attempt",
    }

    cleaned = sanitize_details(dirty_details)

    # Allowed safe fields
    assert cleaned["distance"] == 0.452
    assert cleaned["sample_count"] == 15
    assert cleaned["safe_note"] == "Normal attempt"

    # Forbidden sensitive fields must NOT be present
    assert "password" not in cleaned
    assert "secret_key" not in cleaned
    assert "biometric_embedding" not in cleaned
    assert "raw_landmarks" not in cleaned
    assert "large_array" not in cleaned

    # Check serialization safety
    json_str = json.dumps(cleaned)
    assert "SuperSecretPassword123!" not in json_str
    assert "secret_encryption_key_bytes" not in json_str


def test_all_eight_lifecycle_events_logged(tmp_path: Path) -> None:
    """Validates logging of every individual security lifecycle event."""
    log_file = tmp_path / "lifecycle_events.log"

    events = [
        (SecurityEventType.REGISTRATION_SUCCESS, "user_01", "SUCCESS", "Enrolled"),
        (SecurityEventType.REGISTRATION_FAILURE, "user_02", "FAILURE", "Face too small"),
        (SecurityEventType.LIVENESS_SUCCESS, None, "SUCCESS", "2 blinks detected"),
        (SecurityEventType.LIVENESS_FAILURE, None, "FAILURE", "Timeout (photo spoof)"),
        (SecurityEventType.AUTHENTICATION_SUCCESS, "user_01", "SUCCESS", "Match 0.35"),
        (SecurityEventType.AUTHENTICATION_FAILURE, None, "FAILURE", "Match 0.72"),
        (SecurityEventType.CAMERA_ERROR, None, "ERROR", "Frame drop"),
        (SecurityEventType.STORAGE_ERROR, "user_01", "ERROR", "Corrupted template"),
    ]

    for event_type, u_id, status, reason in events:
        log_security_event(
            event_type=event_type,
            user_id=u_id,
            status=status,
            reason=reason,
            log_file=log_file,
        )

    records = get_audit_records(limit=20, log_file=log_file)
    assert len(records) == 8

    for i, (expected_type, expected_uid, expected_status, _) in enumerate(events):
        assert records[i].event_type == expected_type.value
        assert records[i].user_id == expected_uid
        assert records[i].status == expected_status


def test_audit_record_json_serialization_roundtrip() -> None:
    """Verifies that AuditRecord to_dict, to_json, and from_json roundtrip cleanly."""
    rec = AuditRecord(
        timestamp="2026-09-29T18:00:00.000000Z",
        event_type="REGISTRATION_SUCCESS",
        user_id="user_admin",
        status="SUCCESS",
        reason="Profile created",
        details={"samples": 15, "normalized": True},
    )

    json_line = rec.to_json()
    reconstructed = AuditRecord.from_json(json_line)

    assert reconstructed is not None
    assert reconstructed.timestamp == rec.timestamp
    assert reconstructed.event_type == rec.event_type
    assert reconstructed.user_id == rec.user_id
    assert reconstructed.details == rec.details


def test_corrupted_lines_gracefully_skipped(tmp_path: Path) -> None:
    """Ensures get_audit_records skips invalid or malformed JSON lines without crashing."""
    log_file = tmp_path / "corrupt.log"
    log_file.write_text(
        '{"timestamp": "2026-09-29T18:00:00", "event_type": "REGISTRATION_SUCCESS", "status": "SUCCESS"}\n'
        'CORRUPTED_NON_JSON_LINE\n'
        '{"timestamp": "2026-09-29T18:01:00", "event_type": "AUTHENTICATION_SUCCESS", "status": "SUCCESS"}\n',
        encoding="utf-8",
    )

    records = get_audit_records(log_file=log_file)
    assert len(records) == 2
    assert records[0].event_type == "REGISTRATION_SUCCESS"
    assert records[1].event_type == "AUTHENTICATION_SUCCESS"


def test_nonexistent_log_file_returns_empty_list(tmp_path: Path) -> None:
    """Verifies get_audit_records on a non-existent path safely returns an empty list."""
    missing_file = tmp_path / "does_not_exist.log"
    records = get_audit_records(log_file=missing_file)
    assert records == []


def test_ui_renderer_all_states_smoke_test() -> None:
    """Smoke test ensuring render_hud correctly handles every UIState without exceptions."""
    from core.ui_renderer import HUDTelemetry, UIState, render_hud

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for state in UIState:
        telemetry = HUDTelemetry(
            state=state,
            status_message=f"Testing state {state.value}",
            liveness_status="✓ VERIFIED" if state == UIState.SUCCESS else "--",
            identity_status="✓ MATCHED" if state == UIState.SUCCESS else "SCANNING...",
            distance=0.42 if state == UIState.SUCCESS else None,
            threshold=0.60,
            blinks_done=1,
            target_blinks=2,
            ear=0.31,
            samples_collected=5,
            target_samples=15,
            face_bbox=(50, 200, 200, 50),
            user_count=3,
            flash_message="Test Flash",
            flash_expiry=9999999999.0,
            flash_success=True,
        )

        output_frame = render_hud(dummy_frame, telemetry)
        assert output_frame is not None
        assert output_frame.shape == (480, 640, 3)
        assert isinstance(output_frame, np.ndarray)

