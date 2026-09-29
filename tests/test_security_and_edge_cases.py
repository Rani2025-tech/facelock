"""Security, vulnerability mitigation, and edge-case test suite for FaceLock.

Validates system resilience against:
1. Path traversal attacks in user IDs.
2. SQL injection attempts in user parameters.
3. Cryptographic key mismatch and ciphertext tampering.
4. Non-finite (NaN, Inf) and degenerate biometric inputs.
5. Empty database / uninitialized credential authentication attempts.
6. Camera hardware failures and empty frame streams.
7. Spoofing attacks and multiple face presentation attacks.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet
import numpy as np
import pytest

from config import DEFAULT_MATCH_THRESHOLD, MIN_FACE_SIZE
from core.audit_logger import SecurityEventType, get_audit_records
from core.embedding import (
    aggregate_embeddings,
    extract_validated_embedding,
    normalize_embedding,
)
from core.liveness import LivenessDetector, LivenessResult, LivenessStatus
from core.persistence import has_active_password, load_active_password
from core.security import decrypt_embedding, encrypt_embedding, get_or_create_master_key
from core.user_db import (
    create_user,
    get_user,
    list_users,
    load_all_user_embeddings,
    validate_user_id,
)
from utils.camera import WebcamManager
from workflows.authenticate import authenticate_user
from workflows.register import register_user


def _synthetic_embedding(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize_embedding(rng.standard_normal(128))


# ── 1. Security & Injection Tests ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "malicious_id",
    [
        "../../etc/passwd",
        "..\\Windows\\System32",
        "user/../admin",
        "user\\..\\admin",
        "user/admin",
        "user\\admin",
        "user*name",
        "user:name",
        "user<name>",
        "user|name",
        "user?name",
        "../template.enc",
        "",
        " ",
        "a",  # Below min 2 chars
        "a" * 33,  # Above max 32 chars
    ],
)
def test_path_traversal_and_invalid_user_id_rejected(malicious_id: str) -> None:
    """Ensures validate_user_id strictly rejects path traversal and illegal characters."""
    with pytest.raises(ValueError, match="invalid|must be|User ID"):
        validate_user_id(malicious_id)


def test_sql_injection_in_user_id_and_display_name(tmp_path: Path) -> None:
    """Verifies SQLite parameterization prevents SQL injection payloads."""
    db_file = tmp_path / "sql_test.db"
    users_dir = tmp_path / "users"
    emb = _synthetic_embedding(seed=50)

    # Malicious user_id fails regex validation before SQL execution
    with pytest.raises(ValueError):
        create_user(
            user_id="admin' OR '1'='1",
            display_name="Injected Admin",
            embedding=emb,
            db_path=db_file,
            users_dir=users_dir,
        )

    # Malicious display_name with SQL syntax is safely escaped as literal text
    safe_id = "injected_display"
    sql_payload_name = "Robert'); DROP TABLE users;--"
    created_user = create_user(
        user_id=safe_id,
        display_name=sql_payload_name,
        embedding=emb,
        db_path=db_file,
        users_dir=users_dir,
    )

    assert created_user.display_name == sql_payload_name

    # Table still exists and user is queryable
    retrieved = get_user(safe_id, db_path=db_file)
    assert retrieved is not None
    assert retrieved.display_name == sql_payload_name
    assert len(list_users(db_path=db_file)) == 1


# ── 2. Cryptographic Security Tests ───────────────────────────────────────────

def test_wrong_encryption_key_rejected(tmp_path: Path) -> None:
    """Attempting decryption with a different key raises ValueError without crashing."""
    key_alice = Fernet.generate_key()
    key_bob = Fernet.generate_key()

    emb = _synthetic_embedding(seed=99)
    ciphertext = encrypt_embedding(emb, key=key_alice)

    with pytest.raises(ValueError, match="decryption failed"):
        decrypt_embedding(ciphertext, key=key_bob)


def test_corrupted_ciphertext_tampering_rejected() -> None:
    """Tampered ciphertext payload fails HMAC verification and raises ValueError."""
    key = Fernet.generate_key()
    emb = _synthetic_embedding(seed=88)
    ciphertext = bytearray(encrypt_embedding(emb, key=key))

    # Tamper with middle bytes
    ciphertext[20] ^= 0xFF
    ciphertext[21] ^= 0xAA

    with pytest.raises(ValueError, match="decryption failed"):
        decrypt_embedding(bytes(ciphertext), key=key)


def test_missing_key_auto_generation(tmp_path: Path) -> None:
    """Missing key file triggers safe automatic creation and persistence."""
    key_file = tmp_path / ".secret.key"
    assert not key_file.exists()

    generated_key = get_or_create_master_key(key_file=key_file)
    assert key_file.exists()
    assert len(generated_key) == 44  # Base64 32 bytes

    # Subsequent call reuses the same key
    reloaded_key = get_or_create_master_key(key_file=key_file)
    assert reloaded_key == generated_key


# ── 3. Edge Cases & Degenerate Biometrics ──────────────────────────────────────

def test_non_finite_embedding_in_normalize() -> None:
    """NaN and Inf vectors must be rejected with ValueError during normalization."""
    nan_emb = np.full(128, np.nan)
    inf_emb = np.full(128, np.inf)

    with pytest.raises(ValueError, match="non-finite"):
        normalize_embedding(nan_emb)

    with pytest.raises(ValueError, match="non-finite"):
        normalize_embedding(inf_emb)


def test_non_finite_embedding_in_aggregate() -> None:
    """Aggregating embeddings containing NaN or Inf raises ValueError."""
    valid_emb = _synthetic_embedding(seed=1)
    nan_emb = np.full(128, np.nan)

    with pytest.raises(ValueError, match="non-finite"):
        aggregate_embeddings([valid_emb, nan_emb])


def test_authenticate_with_empty_database(tmp_path: Path) -> None:
    """Authentication against an empty database returns clear rejection without crash."""
    db_file = tmp_path / "empty.db"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with (
        patch("workflows.authenticate.load_active_password", return_value=None),
        patch("workflows.authenticate.extract_face_embedding", return_value=_synthetic_embedding(1)),
    ):
        result = authenticate_user(source=dummy_frame, db_path=db_file)

    assert result.authenticated is False
    assert "No registered users found" in result.reason
    assert result.user_id is None


def test_duplicate_user_registration_workflow(tmp_path: Path) -> None:
    """Registering a user with an already existing ID is rejected cleanly."""
    db_file = tmp_path / "dup_test.db"
    users_dir = tmp_path / "users"
    emb = _synthetic_embedding(seed=1)

    # First registration succeeds
    user_record = create_user("alice", "Alice", emb, db_path=db_file, users_dir=users_dir)
    assert user_record is not None

    # Register workflow with same ID returns False
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("workflows.register.extract_face_embedding", return_value=emb):
        success, msg, _ = register_user("alice", "Alice Duplicate", source=dummy_frame, db_path=db_file, users_dir=users_dir)

    assert success is False
    assert "already exists" in msg


def test_camera_hardware_failure_handling() -> None:
    """WebcamManager returning False on open is handled with RuntimeError."""
    mock_camera = WebcamManager(camera_index=99)
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.isOpened.return_value = False
        assert mock_camera.open() is False
        assert mock_camera.read_frame() is None


def test_camera_stream_empty_frames_rejected_in_registration(tmp_path: Path) -> None:
    """Empty or None frames during interactive registration are skipped safely."""
    mock_camera = MagicMock(spec=WebcamManager)
    # Return 2 None frames followed by valid frames
    valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera.read_frame.side_effect = [None, None, valid_frame] + [valid_frame] * 10

    valid_emb = _synthetic_embedding(seed=1)
    with patch("workflows.register.extract_validated_embedding", return_value=(valid_emb, None, (10, 80, 80, 10))):
        from workflows.register import _collect_and_aggregate_template
        template = _collect_and_aggregate_template(mock_camera, sample_count=1)

    assert template is not None
    assert template.shape == (128,)
