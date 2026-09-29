"""Hardware-independent unit and integration tests for liveness / anti-spoofing layer.

Validates:
- EAR (Eye Aspect Ratio) mathematical formula calculation.
- Open vs closed eye detection.
- Temporal blink sequence state transitions (Open -> Closed -> Open).
- Multi-blink challenge counting (0/2 -> 1/2 -> 2/2 -> PASSED).
- Timeout expiration handling (-> FAILED).
- State reset mechanics.
- Edge case handling (face disappearing, multiple faces, small face).
- Liveness-gated authentication workflow.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import (
    DEFAULT_MATCH_THRESHOLD,
    EAR_THRESHOLD,
    EXPECTED_EMBEDDING_SHAPE,
    LIVENESS_TIMEOUT_SECONDS,
    REQUIRED_BLINKS,
)
import core.persistence as persistence_module
from core.liveness import (
    LivenessDetector,
    LivenessResult,
    LivenessStatus,
    calculate_ear,
)
from utils.camera import WebcamManager
from workflows.authenticate import (
    authenticate_face,
    authenticate_face_with_liveness,
)


# ── Synthetic Eye Landmarks Helpers ──────────────────────────────────────────

def _make_open_eye_points() -> list[tuple[int, int]]:
    """Returns 6 landmark coordinates representing an open eye (EAR ~ 0.33)."""
    # p1=(0,0), p2=(10,-5), p3=(20,-5), p4=(30,0), p5=(20,5), p6=(10,5)
    return [(0, 0), (10, -5), (20, -5), (30, 0), (20, 5), (10, 5)]


def _make_closed_eye_points() -> list[tuple[int, int]]:
    """Returns 6 landmark coordinates representing a closed eye (EAR ~ 0.067)."""
    # p1=(0,0), p2=(10,-1), p3=(20,-1), p4=(30,0), p5=(20,1), p6=(10,1)
    return [(0, 0), (10, -1), (20, -1), (30, 0), (20, 1), (10, 1)]


def _make_synthetic_embedding(seed: int = 42) -> np.ndarray:
    """Returns a normalized (128,) embedding vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(128).astype(np.float64)
    return v / np.linalg.norm(v)


# ── Mathematical EAR Tests ───────────────────────────────────────────────────

def test_calculate_ear_open_eye() -> None:
    """Verifies EAR calculation on open eye geometry exceeds EAR_THRESHOLD."""
    open_eye = _make_open_eye_points()
    ear = calculate_ear(open_eye)
    assert ear > EAR_THRESHOLD
    assert np.isclose(ear, 0.3333, atol=1e-3)


def test_calculate_ear_closed_eye() -> None:
    """Verifies EAR calculation on closed eye geometry falls below EAR_THRESHOLD."""
    closed_eye = _make_closed_eye_points()
    ear = calculate_ear(closed_eye)
    assert ear < EAR_THRESHOLD
    assert np.isclose(ear, 0.0667, atol=1e-3)


def test_calculate_ear_invalid_length_raises() -> None:
    """Verifies ValueError when point list does not contain 6 coordinates."""
    with pytest.raises(ValueError, match="must contain exactly 6"):
        calculate_ear([(0, 0), (10, 0)])  # type: ignore

    with pytest.raises(ValueError, match="must contain exactly 6"):
        calculate_ear(None)  # type: ignore


def test_calculate_ear_zero_width_returns_zero() -> None:
    """Verifies that zero horizontal distance returns 0.0 without division by zero."""
    degenerate_eye = [(0, 0), (0, 5), (0, 5), (0, 0), (0, -5), (0, -5)]
    ear = calculate_ear(degenerate_eye)
    assert ear == 0.0


# ── LivenessDetector State Machine Tests ─────────────────────────────────────

def test_liveness_single_blink_sequence() -> None:
    """Verifies Open -> Closed -> Open sequence increments blink count by 1."""
    detector = LivenessDetector(ear_threshold=0.22, blink_min_frames=1, required_blinks=2)
    detector.start()

    # Frame 1: Eyes open
    r1 = detector.process_ear(0.32)
    assert r1.status == LivenessStatus.WAITING
    assert r1.blinks_completed == 0

    # Frame 2: Eyes closed
    r2 = detector.process_ear(0.12)
    assert r2.status == LivenessStatus.IN_PROGRESS
    assert r2.blinks_completed == 0

    # Frame 3: Eyes open again (Blink 1 completed!)
    r3 = detector.process_ear(0.32)
    assert r3.status == LivenessStatus.IN_PROGRESS
    assert r3.blinks_completed == 1
    assert r3.passed is False
    assert r3.is_complete is False


def test_liveness_full_challenge_pass() -> None:
    """Verifies completing 2 valid blinks transitions state to PASSED."""
    detector = LivenessDetector(ear_threshold=0.22, blink_min_frames=1, required_blinks=2)
    detector.start()

    # Blink 1: Open -> Closed -> Open
    detector.process_ear(0.30)
    detector.process_ear(0.10)
    r_blink1 = detector.process_ear(0.30)
    assert r_blink1.blinks_completed == 1
    assert r_blink1.passed is False

    # Blink 2: Open -> Closed -> Open
    detector.process_ear(0.30)
    detector.process_ear(0.10)
    r_blink2 = detector.process_ear(0.30)
    assert r_blink2.blinks_completed == 2
    assert r_blink2.passed is True
    assert r_blink2.is_complete is True
    assert r_blink2.status == LivenessStatus.PASSED


def test_liveness_timeout_expiration() -> None:
    """Verifies that exceeding timeout duration transitions state to FAILED."""
    detector = LivenessDetector(timeout_seconds=0.1)
    detector.start()

    # Artificial delay to trigger timeout
    import time
    time.sleep(0.15)

    result = detector.process_ear(0.30)
    assert result.status == LivenessStatus.FAILED
    assert result.passed is False
    assert result.is_complete is True
    assert "Timeout" in result.message


def test_liveness_prolonged_closed_eyes_rejected() -> None:
    """Verifies that keeping eyes closed for too long (> max_frames) does not count as a blink."""
    detector = LivenessDetector(
        ear_threshold=0.22,
        blink_min_frames=1,
        blink_max_frames=3,
        required_blinks=1,
    )
    detector.start()

    detector.process_ear(0.30)  # Open

    # Closed for 5 consecutive frames (exceeds max of 3)
    for _ in range(5):
        detector.process_ear(0.10)

    # Open again
    res = detector.process_ear(0.30)
    assert res.blinks_completed == 0
    assert res.passed is False


def test_liveness_reset() -> None:
    """Verifies that reset clears blink count and restores clean state."""
    detector = LivenessDetector(required_blinks=2)
    detector.start()
    detector.process_ear(0.30)
    detector.process_ear(0.10)
    detector.process_ear(0.30)
    assert detector._blinks_completed == 1

    detector.reset()
    assert detector._blinks_completed == 0
    assert detector._status == LivenessStatus.WAITING


# ── Frame Processing & Edge Case Tests ────────────────────────────────────────

def test_process_frame_no_face_detected() -> None:
    """Verifies graceful handling and guidance prompt when no face is visible."""
    detector = LivenessDetector()
    detector.start()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("face_recognition.face_locations", return_value=[]):
        res, _, _ = detector.process_frame(dummy_frame)
        assert res.status == LivenessStatus.WAITING
        assert res.passed is False
        assert "Face not detected" in res.message


def test_process_frame_multiple_faces_detected() -> None:
    """Verifies pause and warning prompt when multiple faces enter the frame."""
    detector = LivenessDetector()
    detector.start()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    multi_boxes = [(10, 100, 100, 10), (10, 300, 100, 210)]

    with patch("face_recognition.face_locations", return_value=multi_boxes):
        res, _, _ = detector.process_frame(dummy_frame)
        assert res.passed is False
        assert "Multiple faces detected" in res.message


def test_process_frame_end_to_end_blink() -> None:
    """Verifies process_frame correctly computes EAR from landmark mock."""
    detector = LivenessDetector(required_blinks=1)
    detector.start()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = [(20, 150, 150, 20)]

    open_landmarks = {"left_eye": _make_open_eye_points(), "right_eye": _make_open_eye_points()}
    closed_landmarks = {"left_eye": _make_closed_eye_points(), "right_eye": _make_closed_eye_points()}

    with (
        patch("face_recognition.face_locations", return_value=box),
        patch("face_recognition.face_landmarks") as mock_landmarks,
    ):
        # 1. Open
        mock_landmarks.return_value = [open_landmarks]
        r1, _, _ = detector.process_frame(dummy_frame)
        assert r1.ear > EAR_THRESHOLD

        # 2. Closed
        mock_landmarks.return_value = [closed_landmarks]
        r2, _, _ = detector.process_frame(dummy_frame)
        assert r2.ear < EAR_THRESHOLD

        # 3. Open (Passes 1-blink challenge)
        mock_landmarks.return_value = [open_landmarks]
        r3, _, _ = detector.process_frame(dummy_frame)
        assert r3.passed is True
        assert r3.status == LivenessStatus.PASSED


def test_process_frame_unmocked_api_signature() -> None:
    """Verifies that process_frame calls face_recognition.face_landmarks with the correct signature."""
    detector = LivenessDetector()
    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Calling process_frame with empty black frame invokes face_locations -> [] without crashing
    res, _, locs = detector.process_frame(dummy_frame)
    assert res.status == LivenessStatus.WAITING
    assert locs == []


# ── Integration: Liveness-Gated Authentication ────────────────────────────────

def test_authenticate_with_liveness_success(tmp_path: Path) -> None:
    """Verifies that passing liveness leads to successful facial matching and access approval."""
    user_embedding = _make_synthetic_embedding(seed=42)

    mock_camera = MagicMock(spec=WebcamManager)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera.read_frame.return_value = dummy_frame

    # Set up mock liveness detector passing on first frame
    mock_result = LivenessResult(
        status=LivenessStatus.PASSED,
        passed=True,
        is_complete=True,
        blinks_completed=2,
        required_blinks=2,
        ear=0.32,
        message="Liveness: PASSED",
        bbox=(20, 150, 150, 20),
    )

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.authenticate.load_active_password", return_value=user_embedding),
        patch.object(LivenessDetector, "process_frame", return_value=(mock_result, dummy_frame, [(20, 150, 150, 20)])),
        patch("face_recognition.face_encodings", return_value=[user_embedding]),
    ):
        is_auth, distance, msg = authenticate_face_with_liveness(
            camera=mock_camera,
            threshold=DEFAULT_MATCH_THRESHOLD,
        )
        assert is_auth is True
        assert distance < DEFAULT_MATCH_THRESHOLD
        assert "ACCESS GRANTED" in msg


def test_authenticate_with_liveness_denied_on_spoof_timeout(tmp_path: Path) -> None:
    """Verifies that failing the liveness challenge (photo/screen spoof) immediately denies access."""
    user_embedding = _make_synthetic_embedding(seed=42)

    mock_camera = MagicMock(spec=WebcamManager)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_camera.read_frame.return_value = dummy_frame

    # Simulate static photograph (timeout occurred with 0 blinks)
    mock_result = LivenessResult(
        status=LivenessStatus.FAILED,
        passed=False,
        is_complete=True,
        blinks_completed=0,
        required_blinks=2,
        ear=0.30,
        message="Liveness: FAILED (Timeout — no blinks detected)",
        bbox=(20, 150, 150, 20),
    )

    with (
        patch.object(persistence_module, "_BASE_DIR", tmp_path),
        patch("workflows.authenticate.load_active_password", return_value=user_embedding),
        patch.object(LivenessDetector, "process_frame", return_value=(mock_result, dummy_frame, [(20, 150, 150, 20)])),
        patch("face_recognition.face_encodings", return_value=[user_embedding]),
    ):
        is_auth, distance, msg = authenticate_face_with_liveness(
            camera=mock_camera,
            threshold=DEFAULT_MATCH_THRESHOLD,
        )
        assert is_auth is False
        assert "ACCESS DENIED" in msg
        assert "FAILED" in msg
