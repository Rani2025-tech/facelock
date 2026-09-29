"""Liveness and anti-spoofing detection using Eye Aspect Ratio (EAR).

Implements a temporal blink-detection state machine based on the landmark analysis
introduced by Soukupová and Čech (2016). Validates that a candidate subject is a
living human performing dynamic eye blinks, mitigating 2D static photograph and
replay screen presentation attacks.
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import face_recognition
import numpy as np

from config import (
    BLINK_MAX_FRAMES,
    BLINK_MIN_FRAMES,
    EAR_THRESHOLD,
    LIVENESS_TIMEOUT_SECONDS,
    MIN_FACE_SIZE,
    REQUIRED_BLINKS,
)

logger = logging.getLogger(__name__)


def calculate_ear(
    eye_points: Union[List[Tuple[int, int]], np.ndarray],
) -> float:
    r"""Calculates the Eye Aspect Ratio (EAR) for a single eye given 6 landmark coordinates.

    Mathematical Formulation:
        Let the 6 ordered (x, y) landmark points for an eye be p_1, p_2, p_3, p_4, p_5, p_6:
        - p_1, p_4: Horizontal corner landmarks (outer and inner canthi).
        - p_2, p_6: Upper and lower vertical landmarks near the outer side.
        - p_3, p_5: Upper and lower vertical landmarks near the inner side.

        The Eye Aspect Ratio is defined as:
            EAR = ( ||p_2 - p_6||_2 + ||p_3 - p_5||_2 ) / ( 2.0 * ||p_1 - p_4||_2 )

        When the eye is open, the vertical distances are large and EAR is typically in the
        range 0.25 to 0.38. When the eyelid closes during a blink, the vertical distances
        drop close to zero, causing EAR to plunge below the threshold (~0.20 - 0.22).

    Args:
        eye_points: A sequence of 6 (x, y) coordinates representing an eye.

    Returns:
        A positive float representing the EAR scalar. Returns 0.0 if points are invalid
        or horizontal distance is zero.

    Raises:
        ValueError: If eye_points does not contain exactly 6 points.
    """
    if eye_points is None or len(eye_points) != 6:
        raise ValueError(
            f"Eye landmarks must contain exactly 6 (x, y) points, got {len(eye_points) if eye_points else 0}."
        )

    pts = np.array(eye_points, dtype=np.float64)

    # Compute Euclidean distances between vertical landmark pairs
    vertical_a = float(np.linalg.norm(pts[1] - pts[5]))  # ||p_2 - p_6||
    vertical_b = float(np.linalg.norm(pts[2] - pts[4]))  # ||p_3 - p_5||

    # Compute Euclidean distance between horizontal corner landmarks
    horizontal = float(np.linalg.norm(pts[0] - pts[3]))  # ||p_1 - p_4||

    if horizontal <= 1e-6:
        return 0.0

    ear: float = (vertical_a + vertical_b) / (2.0 * horizontal)
    return float(round(ear, 4))


class LivenessStatus(Enum):
    """Enumeration of liveness detector state machine outcomes."""

    WAITING = "WAITING"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"


@dataclass
class LivenessResult:
    """Telemetry data object returned after evaluating a single video frame.

    Attributes:
        status: LivenessStatus state (WAITING, IN_PROGRESS, PASSED, FAILED).
        passed: True only if the required number of blinks was successfully completed.
        is_complete: True if the challenge has terminated (either passed or failed).
        blinks_completed: Count of completed valid blinks so far.
        required_blinks: Target blink count.
        ear: Computed average EAR for the current frame.
        message: Human-readable status or guidance prompt.
        bbox: Detected face bounding box (top, right, bottom, left) or None.
    """

    status: LivenessStatus
    passed: bool
    is_complete: bool
    blinks_completed: int
    required_blinks: int
    ear: float
    message: str
    bbox: Optional[Tuple[int, int, int, int]] = None


class LivenessDetector:
    """State machine tracking temporal eye closures to detect live human blinks.

    A valid blink is defined by the strict temporal sequence:
        EYES_OPEN -> EYES_CLOSED (held for >= blink_min_frames and <= blink_max_frames) -> EYES_OPEN

    Features:
    - Enforces temporal sequence to reject single-frame sensor blips or static photos with closed eyes.
    - Tracks countdown timer against `timeout_seconds`.
    - Resettable for multiple authentication attempts.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        blink_min_frames: int = BLINK_MIN_FRAMES,
        blink_max_frames: int = BLINK_MAX_FRAMES,
        required_blinks: int = REQUIRED_BLINKS,
        timeout_seconds: float = LIVENESS_TIMEOUT_SECONDS,
    ) -> None:
        """Initializes the liveness challenge detector.

        Args:
            ear_threshold: EAR cutoff below which eyes are considered closed (default: 0.22).
            blink_min_frames: Minimum consecutive closed frames to qualify as a closure.
            blink_max_frames: Maximum consecutive closed frames before closure is disqualified.
            required_blinks: Total completed blinks required to pass (default: 2).
            timeout_seconds: Maximum challenge duration before timeout (default: 8.0s).
        """
        self.ear_threshold: float = ear_threshold
        self.blink_min_frames: int = blink_min_frames
        self.blink_max_frames: int = blink_max_frames
        self.required_blinks: int = required_blinks
        self.timeout_seconds: float = timeout_seconds

        self.reset()

    def reset(self) -> None:
        """Resets the state machine and timers for a fresh challenge session."""
        self._start_time: Optional[float] = None
        self._blinks_completed: int = 0
        self._consecutive_closed_frames: int = 0
        self._is_eye_currently_closed: bool = False
        self._status: LivenessStatus = LivenessStatus.WAITING
        self._last_ear: float = 0.0

    def start(self) -> None:
        """Explicitly starts the challenge session and records the start timestamp."""
        self.reset()
        self._start_time = time.time()
        self._status = LivenessStatus.WAITING
        logger.info(
            "Liveness challenge started: %d blinks required within %.1fs.",
            self.required_blinks,
            self.timeout_seconds,
        )

    def process_ear(self, ear: float, bbox: Optional[Tuple[int, int, int, int]] = None) -> LivenessResult:
        """Updates the blink state machine given an already computed EAR scalar.

        Args:
            ear: Computed average Eye Aspect Ratio for the frame.
            bbox: Optional face bounding box.

        Returns:
            A LivenessResult detailing the updated state.
        """
        if self._start_time is None:
            self.start()

        self._last_ear = ear
        elapsed = time.time() - (self._start_time or time.time())

        # Check for challenge timeout
        if elapsed > self.timeout_seconds:
            self._status = LivenessStatus.FAILED
            logger.warning("Liveness challenge failed: timed out after %.2fs.", elapsed)
            return LivenessResult(
                status=LivenessStatus.FAILED,
                passed=False,
                is_complete=True,
                blinks_completed=self._blinks_completed,
                required_blinks=self.required_blinks,
                ear=ear,
                message="Liveness: FAILED (Timeout — no blinks detected)",
                bbox=bbox,
            )

        # State transition: Eyes Closed
        if ear < self.ear_threshold:
            self._consecutive_closed_frames += 1
            self._is_eye_currently_closed = True
            self._status = LivenessStatus.IN_PROGRESS
            status_msg = f"Liveness: Blink {self._blinks_completed}/{self.required_blinks}"
            return LivenessResult(
                status=self._status,
                passed=False,
                is_complete=False,
                blinks_completed=self._blinks_completed,
                required_blinks=self.required_blinks,
                ear=ear,
                message=status_msg,
                bbox=bbox,
            )

        # State transition: Eyes Open (ear >= threshold)
        if self._is_eye_currently_closed:
            # Check if previous closure duration satisfied blink criteria
            if (
                self.blink_min_frames
                <= self._consecutive_closed_frames
                <= self.blink_max_frames
            ):
                self._blinks_completed += 1
                logger.info(
                    "Blink %d/%d registered (closed for %d frames).",
                    self._blinks_completed,
                    self.required_blinks,
                    self._consecutive_closed_frames,
                )
            elif self._consecutive_closed_frames > self.blink_max_frames:
                logger.warning(
                    "Eye closure rejected: held too long (%d frames > %d max).",
                    self._consecutive_closed_frames,
                    self.blink_max_frames,
                )

            # Reset closure trackers
            self._consecutive_closed_frames = 0
            self._is_eye_currently_closed = False

        # Check if target blinks achieved
        if self._blinks_completed >= self.required_blinks:
            self._status = LivenessStatus.PASSED
            logger.info("Liveness challenge PASSED (%d blinks).", self._blinks_completed)
            return LivenessResult(
                status=LivenessStatus.PASSED,
                passed=True,
                is_complete=True,
                blinks_completed=self._blinks_completed,
                required_blinks=self.required_blinks,
                ear=ear,
                message="Liveness: PASSED",
                bbox=bbox,
            )

        if self._blinks_completed == 0:
            status_msg = "Liveness: Waiting (Blink twice to verify)"
        else:
            status_msg = f"Liveness: Blink {self._blinks_completed}/{self.required_blinks}"

        return LivenessResult(
            status=LivenessStatus.WAITING if self._blinks_completed == 0 else LivenessStatus.IN_PROGRESS,
            passed=False,
            is_complete=False,
            blinks_completed=self._blinks_completed,
            required_blinks=self.required_blinks,
            ear=ear,
            message=status_msg,
            bbox=bbox,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
        min_face_size: int = MIN_FACE_SIZE,
    ) -> Tuple[LivenessResult, np.ndarray, List[Tuple[int, int, int, int]]]:
        """Evaluates a raw BGR frame, detecting faces and computing EAR in one pass.

        Reuses RGB conversion and face location bounding boxes if provided to eliminate
        redundant computations.

        Args:
            frame: A BGR frame as a NumPy array.
            rgb_frame: Optional pre-converted RGB array.
            face_locations: Optional pre-computed face bounding boxes.
            min_face_size: Minimum face bounding box dimensions (default: 60px).

        Returns:
            A tuple of (LivenessResult, rgb_frame, face_locations).
        """
        if frame is None or frame.size == 0:
            return (
                LivenessResult(
                    status=LivenessStatus.FAILED,
                    passed=False,
                    is_complete=True,
                    blinks_completed=self._blinks_completed,
                    required_blinks=self.required_blinks,
                    ear=0.0,
                    message="Liveness: Invalid video frame",
                ),
                rgb_frame if rgb_frame is not None else np.zeros((0,), dtype=np.uint8),
                [],
            )

        if rgb_frame is None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if face_locations is None:
            face_locations = face_recognition.face_locations(rgb_frame)

        # Handle face presence constraints
        if len(face_locations) == 0:
            return (
                LivenessResult(
                    status=LivenessStatus.WAITING,
                    passed=False,
                    is_complete=False,
                    blinks_completed=self._blinks_completed,
                    required_blinks=self.required_blinks,
                    ear=0.0,
                    message="Face not detected — keep face in frame",
                    bbox=None,
                ),
                rgb_frame,
                face_locations,
            )

        if len(face_locations) > 1:
            return (
                LivenessResult(
                    status=LivenessStatus.WAITING,
                    passed=False,
                    is_complete=False,
                    blinks_completed=self._blinks_completed,
                    required_blinks=self.required_blinks,
                    ear=0.0,
                    message="Multiple faces detected — only 1 person allowed",
                    bbox=None,
                ),
                rgb_frame,
                face_locations,
            )

        bbox = face_locations[0]
        top, right, bottom, left = bbox
        if (bottom - top) < min_face_size or (right - left) < min_face_size:
            return (
                LivenessResult(
                    status=LivenessStatus.WAITING,
                    passed=False,
                    is_complete=False,
                    blinks_completed=self._blinks_completed,
                    required_blinks=self.required_blinks,
                    ear=0.0,
                    message="Face too small — move closer to camera",
                    bbox=bbox,
                ),
                rgb_frame,
                face_locations,
            )

        # Extract facial landmarks for eye points
        landmarks_list = face_recognition.face_landmarks(
            rgb_frame, face_locations=face_locations
        )
        if not landmarks_list:
            return (
                LivenessResult(
                    status=LivenessStatus.WAITING,
                    passed=False,
                    is_complete=False,
                    blinks_completed=self._blinks_completed,
                    required_blinks=self.required_blinks,
                    ear=0.0,
                    message="Eyes not visible — face the camera",
                    bbox=bbox,
                ),
                rgb_frame,
                face_locations,
            )

        landmarks = landmarks_list[0]
        left_eye = landmarks.get("left_eye")
        right_eye = landmarks.get("right_eye")

        if not left_eye or not right_eye or len(left_eye) != 6 or len(right_eye) != 6:
            return (
                LivenessResult(
                    status=LivenessStatus.WAITING,
                    passed=False,
                    is_complete=False,
                    blinks_completed=self._blinks_completed,
                    required_blinks=self.required_blinks,
                    ear=0.0,
                    message="Eye landmarks incomplete",
                    bbox=bbox,
                ),
                rgb_frame,
                face_locations,
            )

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = float(round((left_ear + right_ear) / 2.0, 4))

        result = self.process_ear(avg_ear, bbox=bbox)
        return result, rgb_frame, face_locations
