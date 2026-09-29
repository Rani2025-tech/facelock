"""Hardware-independent test suite for multi-sample face registration.

Validates:
- Embedding normalization and aggregation mathematics.
- Face validation rules (single face, multiple faces, face size bounds).
- Multi-frame collection, filtering, and persistence.
- Backward compatibility with legacy single-frame inputs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import DEFAULT_MATCH_THRESHOLD, EXPECTED_EMBEDDING_SHAPE, MIN_FACE_SIZE
import core.persistence as persistence_module
from core.embedding import (
    aggregate_embeddings,
    extract_face_embedding,
    extract_validated_embedding,
    normalize_embedding,
)
from core.matcher import compare_embeddings
from core.persistence import load_active_password
from utils.camera import WebcamManager
from workflows.register import (
    capture_and_register_from_camera,
    register_face_password,
    register_multiframe_face_password,
)


def _generate_synthetic_vector(seed: int = 42, noise: float = 0.0) -> np.ndarray:
    """Helper to generate a normalized 128-D synthetic embedding vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(128).astype(np.float64)
    if noise > 0.0:
        vec += noise * rng.standard_normal(128).astype(np.float64)
    return vec / np.linalg.norm(vec)


# ── Normalization & Aggregation Unit Tests ────────────────────────────────────

def test_normalize_embedding_success() -> None:
    """Verifies that arbitrary non-zero vectors normalize to unit L2 norm."""
    vec = np.array([3.0] * 128, dtype=np.float64)
    normalized = normalize_embedding(vec)
    assert normalized.shape == EXPECTED_EMBEDDING_SHAPE
    assert np.isclose(np.linalg.norm(normalized), 1.0)


def test_normalize_embedding_invalid_shape_or_zero() -> None:
    """Verifies that invalid shapes or zero vectors raise ValueError."""
    with pytest.raises(ValueError, match="1-D array of shape"):
        normalize_embedding(np.zeros((64,), dtype=np.float64))

    with pytest.raises(ValueError, match="zero-magnitude"):
        normalize_embedding(np.zeros((128,), dtype=np.float64))

    with pytest.raises(ValueError, match="must not be None"):
        normalize_embedding(None)  # type: ignore


def test_aggregate_embeddings_reduces_noise() -> None:
    """Verifies that aggregating 15 noisy samples produces a stable normalized centroid."""
    base = _generate_synthetic_vector(seed=77)
    noisy_samples = [_generate_synthetic_vector(seed=77, noise=0.08) for _ in range(15)]

    template = aggregate_embeddings(noisy_samples)

    assert template.shape == EXPECTED_EMBEDDING_SHAPE
    assert np.isclose(np.linalg.norm(template), 1.0)

    # The aggregated template should match the base identity within default threshold
    is_match, distance = compare_embeddings(base, template)
    assert is_match is True
    assert distance < DEFAULT_MATCH_THRESHOLD


def test_aggregate_embeddings_empty_raises() -> None:
    """Verifies that aggregating empty list raises ValueError."""
    with pytest.raises(ValueError, match="empty collection"):
        aggregate_embeddings([])


# ── Face Validation & Extraction Tests ────────────────────────────────────────

def test_extract_validated_embedding_no_face() -> None:
    """Verifies diagnostic error message when no face is found."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with patch("face_recognition.face_locations", return_value=[]):
        emb, err, bbox = extract_validated_embedding(dummy_frame)
        assert emb is None
        assert err == "Face not detected — try again"
        assert bbox is None


def test_extract_validated_embedding_multiple_faces() -> None:
    """Verifies rejection when multiple faces are visible."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_boxes = [(50, 150, 150, 50), (50, 350, 150, 250)]
    with patch("face_recognition.face_locations", return_value=mock_boxes):
        emb, err, bbox = extract_validated_embedding(dummy_frame)
        assert emb is None
        assert "Multiple faces detected" in (err or "")
        assert bbox is None


def test_extract_validated_embedding_face_too_small() -> None:
    """Verifies rejection when face bounding box is below MIN_FACE_SIZE."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # 30x30 bounding box, below 60px minimum
    small_box = [(50, 80, 80, 50)]
    with patch("face_recognition.face_locations", return_value=small_box):
        emb, err, bbox = extract_validated_embedding(dummy_frame, min_face_size=MIN_FACE_SIZE)
        assert emb is None
        assert "Face too small" in (err or "")
        assert bbox == small_box[0]


def test_extract_validated_embedding_success() -> None:
    """Verifies successful extraction when one valid face is detected."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    valid_box = [(50, 180, 180, 50)]  # 130x130 box
    synthetic_emb = _generate_synthetic_vector(seed=42)

    with (
        patch("face_recognition.face_locations", return_value=valid_box),
        patch("face_recognition.face_encodings", return_value=[synthetic_emb]),
    ):
        emb, err, bbox = extract_validated_embedding(dummy_frame)
        assert emb is not None
        assert err is None
        assert bbox == valid_box[0]
        np.testing.assert_array_equal(emb, synthetic_emb)


# ── Multi-Frame Registration Workflow Tests ───────────────────────────────────

def test_multiframe_registration_success(tmp_path: Path) -> None:
    """Tests multi-frame registration collecting 15 valid samples."""
    target_file = tmp_path / "active_password.npy"
    samples = [_generate_synthetic_vector(seed=42, noise=0.03) for _ in range(15)]
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(15)]

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.register.extract_validated_embedding") as mock_extract,
    ):
        mock_extract.side_effect = [(s, None, (10, 80, 80, 10)) for s in samples]

        success, msg = register_multiframe_face_password(
            frames=frames, sample_count=15, filepath=target_file
        )
        assert success is True
        assert "successfully" in msg

        loaded = load_active_password(filepath=target_file)
        assert loaded is not None
        assert loaded.shape == (128,)
        assert np.isclose(np.linalg.norm(loaded), 1.0)


def test_multiframe_registration_insufficient_samples(tmp_path: Path) -> None:
    """Tests registration abortion when valid frames are fewer than sample_count."""
    target_file = tmp_path / "active_password.npy"
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    samples = [_generate_synthetic_vector(seed=42) for _ in range(5)]

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.register.extract_validated_embedding") as mock_extract,
    ):
        mock_extract.side_effect = [(s, None, (10, 80, 80, 10)) for s in samples]

        success, msg = register_multiframe_face_password(
            frames=frames, sample_count=15, filepath=target_file
        )
        assert success is False
        assert "insufficient valid samples" in msg


def test_multiframe_registration_filters_invalid_frames(tmp_path: Path) -> None:
    """Tests that invalid frames are skipped and only valid ones are counted."""
    target_file = tmp_path / "active_password.npy"
    valid_samples = [_generate_synthetic_vector(seed=42) for _ in range(15)]

    # Interleave 5 invalid frames, 15 valid frames, 5 invalid frames
    frame_stream = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(25)]
    effects = (
        [(None, "Face not detected — try again", None)] * 5
        + [(s, None, (10, 80, 80, 10)) for s in valid_samples]
        + [(None, "Multiple faces detected", None)] * 5
    )

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.register.extract_validated_embedding") as mock_extract,
    ):
        mock_extract.side_effect = effects

        success, msg = register_multiframe_face_password(
            frames=frame_stream, sample_count=15, filepath=target_file
        )
        assert success is True
        assert "successfully" in msg


def test_camera_stream_registration_with_progress_callback(tmp_path: Path) -> None:
    """Tests camera streaming registration with progress callback."""
    target_file = tmp_path / "active_password.npy"
    mock_camera = MagicMock(spec=WebcamManager)
    mock_camera.read_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

    sample = _generate_synthetic_vector(seed=123)
    progress_calls = []

    def on_progress(frame, status_text, current, total, bbox):
        progress_calls.append((status_text, current, total))

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.register.extract_validated_embedding") as mock_extract,
    ):
        mock_extract.return_value = (sample, None, (20, 120, 120, 20))

        success, msg = capture_and_register_from_camera(
            camera=mock_camera,
            sample_count=5,
            filepath=target_file,
            on_progress=on_progress,
        )
        assert success is True
        assert len(progress_calls) == 5
        assert progress_calls[-1][1] == 5  # Final sample count reached


def test_unified_register_face_password_backward_compatibility(tmp_path: Path) -> None:
    """Verifies that passing a single 3D frame behaves cleanly in backward-compatible mode."""
    target_file = tmp_path / "active_password.npy"
    single_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sample = _generate_synthetic_vector(seed=999)

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.register.extract_face_embedding", return_value=sample),
    ):
        success, msg = register_face_password(single_frame, filepath=target_file)
        assert success is True
        assert "successfully" in msg

        loaded = load_active_password(filepath=target_file)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, sample)
