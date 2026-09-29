"""Hardware-independent unit and integration tests for multi-user biometric system.

Validates:
- Biometric template encryption & decryption roundtrip.
- Multi-user SQLite enrollment, listing, and deletion.
- Correct 1:N identity resolution (Alice -> Alice, Bob -> Bob).
- Unrecognized impostor rejection.
- Duplicate user ID prevention.
- Empty database pre-flight handling.
- Corrupted encrypted template recovery/skipping.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import DEFAULT_MATCH_THRESHOLD, EXPECTED_EMBEDDING_SHAPE
from core.security import decrypt_embedding, encrypt_embedding
from core.user_db import (
    create_user,
    delete_user,
    get_user,
    list_users,
    load_all_user_embeddings,
    load_user_embedding,
    user_count,
    validate_user_id,
)
from workflows.authenticate import AuthenticationResult, authenticate_user
from workflows.register import register_user


def _synthetic_embedding(seed: int = 42, noise: float = 0.0) -> np.ndarray:
    """Generates a normalized 128-D synthetic embedding vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float64)
    if noise > 0.0:
        v += noise * rng.standard_normal(128).astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float64)


# ── Security & Encryption Unit Tests ─────────────────────────────────────────

def test_encryption_decryption_roundtrip(tmp_path: Path) -> None:
    """Verifies that an embedding vector encrypts and decrypts bit-for-bit identically."""
    original = _synthetic_embedding(seed=123)
    key_file = tmp_path / "secret.key"

    with patch("core.security._KEY_FILE", key_file):
        ciphertext = encrypt_embedding(original)
        assert isinstance(ciphertext, bytes)
        assert len(ciphertext) > len(original.tobytes())  # AES padding + IV + HMAC

        decrypted = decrypt_embedding(ciphertext)
        assert decrypted.shape == EXPECTED_EMBEDDING_SHAPE
        np.testing.assert_array_almost_equal(decrypted, original)


def test_decrypt_corrupted_ciphertext_raises_error() -> None:
    """Verifies ValueError when attempting to decrypt corrupted/tampered bytes."""
    corrupted = b"this_is_not_a_valid_fernet_token_and_will_fail"
    with pytest.raises(ValueError, match="decryption failed"):
        decrypt_embedding(corrupted)


# ── User DB CRUD Tests ────────────────────────────────────────────────────────

def test_user_id_validation() -> None:
    """Verifies user ID format validation rules."""
    assert validate_user_id("alice") == "alice"
    assert validate_user_id("user_123-admin") == "user_123-admin"

    with pytest.raises(ValueError, match="invalid"):
        validate_user_id("a")  # Too short (<2)

    with pytest.raises(ValueError, match="invalid"):
        validate_user_id("user@domain.com")  # Invalid chars

    with pytest.raises(ValueError, match="non-empty"):
        validate_user_id("")


def test_create_and_get_user(tmp_path: Path) -> None:
    """Tests creating a user record and retrieving it from SQLite."""
    db_path = tmp_path / "test_users.db"
    users_dir = tmp_path / "users"
    emb = _synthetic_embedding(seed=10)

    user = create_user(
        user_id="alice",
        display_name="Alice Smith",
        embedding=emb,
        db_path=db_path,
        users_dir=users_dir,
    )

    assert user.user_id == "alice"
    assert user.display_name == "Alice Smith"
    assert Path(user.template_file).exists()

    fetched = get_user("alice", db_path=db_path)
    assert fetched is not None
    assert fetched.user_id == "alice"
    assert fetched.display_name == "Alice Smith"

    assert user_count(db_path=db_path) == 1


def test_duplicate_user_creation_rejected(tmp_path: Path) -> None:
    """Verifies that creating a user with an existing user_id raises ValueError."""
    db_path = tmp_path / "test_users.db"
    users_dir = tmp_path / "users"
    emb = _synthetic_embedding(seed=10)

    create_user("bob", "Bob Jones", emb, db_path=db_path, users_dir=users_dir)

    with pytest.raises(ValueError, match="already exists"):
        create_user("bob", "Bob Other", emb, db_path=db_path, users_dir=users_dir)


def test_list_and_delete_users(tmp_path: Path) -> None:
    """Verifies listing all users and deleting specific user records."""
    db_path = tmp_path / "test_users.db"
    users_dir = tmp_path / "users"

    emb1 = _synthetic_embedding(seed=1)
    emb2 = _synthetic_embedding(seed=2)

    create_user("user1", "User One", emb1, db_path=db_path, users_dir=users_dir)
    create_user("user2", "User Two", emb2, db_path=db_path, users_dir=users_dir)

    users = list_users(db_path=db_path)
    assert len(users) == 2
    assert [u.user_id for u in users] == ["user1", "user2"]

    # Delete user1
    deleted = delete_user("user1", db_path=db_path, users_dir=users_dir)
    assert deleted is True
    assert user_count(db_path=db_path) == 1
    assert get_user("user1", db_path=db_path) == None
    assert not (users_dir / "user1").exists()


# ── 1:N Multi-User Authentication Tests ───────────────────────────────────────

def test_multi_user_authentication_resolution(tmp_path: Path) -> None:
    """Verifies 1:N multi-user matching resolves the correct identity among multiple enrolled users."""
    db_path = tmp_path / "test_users.db"
    users_dir = tmp_path / "users"

    alice_clean = _synthetic_embedding(seed=100)
    alice_noisy = _synthetic_embedding(seed=100, noise=0.03)

    bob_clean = _synthetic_embedding(seed=200)
    bob_noisy = _synthetic_embedding(seed=200, noise=0.03)

    charlie_clean = _synthetic_embedding(seed=300)

    # Enroll Alice and Bob
    create_user("alice", "Alice Wonder", alice_clean, db_path=db_path, users_dir=users_dir)
    create_user("bob", "Bob Builder", bob_clean, db_path=db_path, users_dir=users_dir)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Test Alice authentication
    with patch("workflows.authenticate.extract_face_embedding", return_value=alice_noisy):
        res_alice = authenticate_user(dummy_frame, db_path=db_path)
        assert res_alice.authenticated is True
        assert res_alice.user_id == "alice"
        assert res_alice.display_name == "Alice Wonder"
        assert res_alice.distance < DEFAULT_MATCH_THRESHOLD

    # 2. Test Bob authentication
    with patch("workflows.authenticate.extract_face_embedding", return_value=bob_noisy):
        res_bob = authenticate_user(dummy_frame, db_path=db_path)
        assert res_bob.authenticated is True
        assert res_bob.user_id == "bob"
        assert res_bob.display_name == "Bob Builder"
        assert res_bob.distance < DEFAULT_MATCH_THRESHOLD

    # 3. Test Unknown Subject (Charlie) rejection
    with patch("workflows.authenticate.extract_face_embedding", return_value=charlie_clean):
        res_charlie = authenticate_user(dummy_frame, db_path=db_path)
        assert res_charlie.authenticated is False
        assert res_charlie.user_id is None
        assert "ACCESS DENIED" in res_charlie.reason


def test_multi_user_empty_database(tmp_path: Path) -> None:
    """Verifies clean rejection when attempting authentication on an empty database."""
    db_path = tmp_path / "empty_users.db"
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("workflows.authenticate.load_active_password", return_value=None):
        res = authenticate_user(dummy_frame, db_path=db_path)
        assert res.authenticated is False
        assert "No registered users found" in res.reason


def test_corrupted_template_handling(tmp_path: Path) -> None:
    """Verifies that a corrupted template file on disk is safely skipped without crashing the system."""
    db_path = tmp_path / "test_users.db"
    users_dir = tmp_path / "users"

    emb = _synthetic_embedding(seed=50)
    user = create_user("corrupted_user", "Corrupted User", emb, db_path=db_path, users_dir=users_dir)

    # Corrupt the template on disk
    template_path = Path(user.template_file)
    template_path.write_bytes(b"corrupted_junk_bytes")

    # load_all_user_embeddings should skip the corrupted user with a warning
    all_users = load_all_user_embeddings(db_path=db_path)
    assert len(all_users) == 0


def test_register_user_workflow(tmp_path: Path) -> None:
    """Verifies register_user end-to-end workflow."""
    db_path = tmp_path / "test_users.db"
    users_dir = tmp_path / "users"
    emb = _synthetic_embedding(seed=88)

    single_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("workflows.register.extract_face_embedding", return_value=emb):
        success, msg, user = register_user(
            user_id="diana",
            display_name="Diana Prince",
            source=single_frame,
            db_path=db_path,
            users_dir=users_dir,
        )
        assert success is True
        assert user is not None
        assert user.user_id == "diana"
        assert "registered successfully" in msg
