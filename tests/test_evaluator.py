"""Automated test suite for the face recognition authentication system.

All tests are fully isolated: no webcam hardware, no real face images, and no
writes to the production active_password.npy file. Synthetic 128-d embeddings
stand in for real face encodings throughout.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import core.persistence as persistence_module
from core.matcher import DEFAULT_MATCH_THRESHOLD, compare_embeddings
from core.persistence import (
    has_active_password,
    load_active_password,
    save_active_password,
)
from workflows.authenticate import authenticate_face
from workflows.register import register_face_password


# ── Synthetic embedding helper ────────────────────────────────────────────────

def generate_synthetic_embedding(seed: int = 42, noise_level: float = 0.0) -> np.ndarray:
    """Generates a normalised 128-dimensional float64 embedding vector.

    Optionally adds Gaussian noise to simulate the same identity captured under
    different lighting or pose conditions, then re-normalises to unit length.

    Args:
        seed: RNG seed for reproducibility. Different seeds produce embeddings
            that are far apart in Euclidean space (simulating different people).
        noise_level: Standard deviation of Gaussian noise added before
            re-normalisation. A value of 0.05 keeps the perturbed vector within
            the 0.6 match threshold of the original.

    Returns:
        A (128,) float64 NumPy array with unit L2 norm.
    """
    rng = np.random.default_rng(seed)
    vector: np.ndarray = rng.standard_normal(128).astype(np.float64)

    if noise_level > 0.0:
        vector = vector + noise_level * rng.standard_normal(128).astype(np.float64)

    return vector / np.linalg.norm(vector)


# ── Sanity-check the synthetic generator assumptions ─────────────────────────

def test_same_seed_with_noise_is_within_threshold() -> None:
    """Verifies that low noise on the same seed stays below the match threshold.

    This validates the True-Positive simulation assumption used in the
    authentication matrix test.
    """
    base = generate_synthetic_embedding(seed=100)
    noisy = generate_synthetic_embedding(seed=100, noise_level=0.05)
    _, distance = compare_embeddings(base, noisy)
    assert distance < DEFAULT_MATCH_THRESHOLD, (
        f"Same-identity simulation failed: distance {distance:.4f} >= {DEFAULT_MATCH_THRESHOLD}"
    )


def test_different_seeds_exceed_threshold() -> None:
    """Verifies that two different seeds produce embeddings beyond the match threshold.

    This validates the True-Negative simulation assumption used in the
    authentication matrix test.
    """
    person_a = generate_synthetic_embedding(seed=1)
    person_b = generate_synthetic_embedding(seed=2)
    _, distance = compare_embeddings(person_a, person_b)
    assert distance >= DEFAULT_MATCH_THRESHOLD, (
        f"Different-identity simulation failed: distance {distance:.4f} < {DEFAULT_MATCH_THRESHOLD}"
    )


# ── Test Case 1: Persistence lifecycle ───────────────────────────────────────

def test_persistence_lifecycle(tmp_path: Path) -> None:
    """Tests the full save → exists → load roundtrip for the password file.

    Verifies that:
    - save_active_password writes the file and returns True.
    - has_active_password detects the non-empty file.
    - load_active_password returns an array bit-for-bit identical to what was saved.

    Args:
        tmp_path: Pytest-provided temporary directory, unique per test run.
    """
    target = tmp_path / "test_pass.npy"
    embedding = generate_synthetic_embedding(seed=42)

    # Redirect _BASE_DIR so _safe_resolve accepts paths inside tmp_path.
    with patch.object(persistence_module, "_BASE_DIR", tmp_path):
        saved = save_active_password(embedding, filepath=target)
        assert saved is True, "save_active_password should return True on success."

        assert has_active_password(filepath=target) is True, (
            "has_active_password should return True after a successful save."
        )

        loaded = load_active_password(filepath=target)
        assert loaded is not None, "load_active_password should not return None for a valid file."
        np.testing.assert_array_equal(
            loaded, embedding,
            err_msg="Loaded embedding does not match the saved embedding.",
        )


def test_load_returns_none_when_file_absent(tmp_path: Path) -> None:
    """Verifies load_active_password returns None when no file exists.

    Args:
        tmp_path: Pytest-provided temporary directory, unique per test run.
    """
    missing = tmp_path / "nonexistent.npy"
    with patch.object(persistence_module, "_BASE_DIR", tmp_path):
        assert load_active_password(filepath=missing) is None


def test_has_active_password_false_when_absent(tmp_path: Path) -> None:
    """Verifies has_active_password returns False when no file exists.

    Args:
        tmp_path: Pytest-provided temporary directory, unique per test run.
    """
    missing = tmp_path / "nonexistent.npy"
    with patch.object(persistence_module, "_BASE_DIR", tmp_path):
        assert has_active_password(filepath=missing) is False


# ── Test Case 2: Authentication matrix ───────────────────────────────────────

def test_authentication_matrix(tmp_path: Path) -> None:
    """Full 5-step authentication matrix: TP, TN, reset, post-reset TP and TN.

    Uses monkeypatching to:
    - Redirect _BASE_DIR so all persistence calls stay inside tmp_path.
    - Replace extract_face_embedding in both workflow modules so no real
      camera frame or face detection is needed.

    Steps:
        1. Save Person A as the active password.
        2. True Positive  — Person A (with noise) authenticates successfully.
        3. True Negative  — Person B is rejected.
        4. Password Reset — Overwrite password with Person B.
        5. Post-Reset     — Person B now passes; Person A now fails.

    Args:
        tmp_path: Pytest-provided temporary directory, unique per test run.
    """
    password_file = tmp_path / "active_password.npy"

    person_a_clean = generate_synthetic_embedding(seed=100)
    person_a_noisy = generate_synthetic_embedding(seed=100, noise_level=0.05)
    person_b_clean = generate_synthetic_embedding(seed=999)

    # Dummy frame — shape only; content is irrelevant because extract_face_embedding
    # is patched out in both workflow modules.
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.register.extract_face_embedding") as mock_register_embed,
        patch("workflows.authenticate.extract_face_embedding") as mock_auth_embed,
        patch("workflows.authenticate.load_active_password") as mock_load,
    ):
        # ── Step 1: Register Person A ─────────────────────────────────────────
        mock_register_embed.return_value = person_a_clean
        success, msg = register_face_password(dummy_frame, filepath=password_file)
        assert success is True, f"Registration failed unexpectedly: {msg}"

        # ── Step 2: True Positive — Person A (noisy) should pass ─────────────
        mock_auth_embed.return_value = person_a_noisy
        mock_load.return_value = person_a_clean
        is_auth, distance, _ = authenticate_face(dummy_frame)
        assert is_auth is True, (
            f"True Positive failed: Person A (noisy) was rejected at distance {distance:.4f}."
        )
        assert distance < DEFAULT_MATCH_THRESHOLD

        # ── Step 3: True Negative — Person B should be rejected ───────────────
        mock_auth_embed.return_value = person_b_clean
        mock_load.return_value = person_a_clean
        is_auth, distance, _ = authenticate_face(dummy_frame)
        assert is_auth is False, (
            f"True Negative failed: Person B was incorrectly granted access at distance {distance:.4f}."
        )
        assert distance >= DEFAULT_MATCH_THRESHOLD

        # ── Step 4: Password Reset — overwrite with Person B ──────────────────
        mock_register_embed.return_value = person_b_clean
        success, msg = register_face_password(dummy_frame, filepath=password_file)
        assert success is True, f"Password reset failed unexpectedly: {msg}"

        # ── Step 5a: Post-Reset — Person B should now pass ────────────────────
        mock_auth_embed.return_value = person_b_clean
        mock_load.return_value = person_b_clean
        is_auth, distance, _ = authenticate_face(dummy_frame)
        assert is_auth is True, (
            f"Post-reset True Positive failed: Person B rejected at distance {distance:.4f}."
        )

        # ── Step 5b: Post-Reset — Person A should now fail ────────────────────
        mock_auth_embed.return_value = person_a_clean
        mock_load.return_value = person_b_clean
        is_auth, distance, _ = authenticate_face(dummy_frame)
        assert is_auth is False, (
            f"Post-reset True Negative failed: Person A still granted access at distance {distance:.4f}."
        )
