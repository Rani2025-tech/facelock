"""Face authentication workflow integrated with multi-user 1:N biometric identification.

Enforces a 4-stage sequential security pipeline:
1. User Enrollment Pre-flight: Verifies registered biometric users exist in SQLite/encrypted store.
2. Liveness Challenge: Requires natural eye blinks to defeat static photo/screen spoofing.
3. Feature Extraction: Extracts candidate 128-D vector reusing face location landmarks.
4. 1:N Biometric Matching: Computes Euclidean distances across all enrolled templates and
   identifies the closest user (L2 <= threshold).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import face_recognition
import numpy as np

from config import (
    DEFAULT_MATCH_THRESHOLD,
    LIVENESS_TIMEOUT_SECONDS,
    MATCH_THRESHOLD,
    REQUIRED_BLINKS,
)
from core.audit_logger import SecurityEventType, log_security_event
from core.embedding import extract_face_embedding
from core.liveness import LivenessDetector, LivenessResult, LivenessStatus
from core.matcher import compare_embeddings
from core.persistence import load_active_password
from core.user_db import UserRecord, load_all_user_embeddings
from utils.camera import WebcamManager

logger = logging.getLogger(__name__)

# Sentinel distance returned when biometric comparison cannot proceed
_NO_ATTEMPT_DISTANCE: float = 1.0

# Type alias for liveness progress callback during live authentication
# (frame, status_message, blinks_done, target_blinks, bbox, ear)
AuthProgressCallback = Callable[
    [np.ndarray, str, int, int, Optional[Tuple[int, int, int, int]], float],
    None,
]


@dataclass
class AuthenticationResult:
    """Represents the outcome of a 1:N biometric authentication attempt.

    Attributes:
        authenticated: True only if liveness and face match both succeed.
        user_id: Identified user ID if authenticated, else None.
        display_name: Human-friendly name if authenticated, else None.
        distance: Minimum Euclidean distance among enrolled users, or 1.0 if aborted.
        reason: Diagnostic outcome description.
    """

    authenticated: bool
    user_id: Optional[str]
    display_name: Optional[str]
    distance: float
    reason: str


def authenticate_user(
    source: Union[np.ndarray, WebcamManager],
    threshold: float = MATCH_THRESHOLD,
    db_path: Optional[Path] = None,
    on_progress: Optional[AuthProgressCallback] = None,
) -> AuthenticationResult:
    """Authenticates and identifies a subject against the multi-user biometric database.

    Args:
        source: Video source (WebcamManager for streaming or single BGR frame).
        threshold: Maximum Euclidean distance for positive match (default from config).
        db_path: Optional path to SQLite database.
        on_progress: Optional UI progress callback.

    Returns:
        An AuthenticationResult detailing authorization status, user identity, and distance.
    """
    enrolled_users = load_all_user_embeddings(db_path=db_path)
    active_single_template = load_active_password()

    candidates: Dict[str, Tuple[UserRecord, np.ndarray]] = dict(enrolled_users)
    if active_single_template is not None and "primary_user" not in candidates and "active_user" not in candidates:
        candidates["active_user"] = (
            UserRecord(
                user_id="active_user",
                display_name="Active User",
                created_at="",
                updated_at="",
                template_file="active_password.npy",
            ),
            active_single_template,
        )

    if len(candidates) == 0:
        logger.warning("Authentication attempted with zero enrolled users.")
        msg = "No registered users found. Please register first."
        log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,
            status="FAILURE",
            reason=msg,
        )
        return AuthenticationResult(
            authenticated=False,
            user_id=None,
            display_name=None,
            distance=_NO_ATTEMPT_DISTANCE,
            reason=msg,
        )

    candidate_embedding: Optional[np.ndarray] = None

    if isinstance(source, WebcamManager):
        # Interactive camera liveness verification loop
        detector = LivenessDetector(
            required_blinks=REQUIRED_BLINKS,
            timeout_seconds=LIVENESS_TIMEOUT_SECONDS,
        )
        detector.start()
        logger.info("Starting liveness-gated multi-user authentication loop.")

        while True:
            frame = source.read_frame()
            if frame is None:
                continue

            result, rgb_frame, face_locations = detector.process_frame(frame)

            if on_progress is not None:
                on_progress(
                    frame,
                    result.message,
                    result.blinks_completed,
                    result.required_blinks,
                    result.bbox,
                    result.ear,
                )

            if result.status == LivenessStatus.PASSED:
                logger.info("Liveness PASSED. Extracting candidate facial vector.")
                log_security_event(
                    SecurityEventType.LIVENESS_SUCCESS,
                    status="SUCCESS",
                    reason="Liveness challenge passed",
                    details={"blinks": result.blinks_completed, "ear": round(result.ear, 3)},
                )
                encodings = face_recognition.face_encodings(
                    rgb_frame, known_face_locations=face_locations
                )
                if not encodings:
                    msg = "Face encoding failed after liveness check."
                    log_security_event(
                        SecurityEventType.AUTHENTICATION_FAILURE,
                        status="FAILURE",
                        reason=msg,
                    )
                    return AuthenticationResult(
                        authenticated=False,
                        user_id=None,
                        display_name=None,
                        distance=_NO_ATTEMPT_DISTANCE,
                        reason=msg,
                    )
                candidate_embedding = encodings[0].astype(np.float64)
                break

            if result.status == LivenessStatus.FAILED:
                logger.warning("Liveness challenge FAILED: %s", result.message)
                log_security_event(
                    SecurityEventType.LIVENESS_FAILURE,
                    status="FAILURE",
                    reason=result.message,
                    details={"ear": round(result.ear, 3), "blinks": result.blinks_completed},
                )
                return AuthenticationResult(
                    authenticated=False,
                    user_id=None,
                    display_name=None,
                    distance=_NO_ATTEMPT_DISTANCE,
                    reason=f"ACCESS DENIED ({result.message})",
                )

    elif isinstance(source, np.ndarray) and source.ndim == 3:
        # Direct single-frame evaluation (for testing or legacy flows)
        candidate_embedding = extract_face_embedding(source)
        if candidate_embedding is None:
            return AuthenticationResult(
                authenticated=False,
                user_id=None,
                display_name=None,
                distance=_NO_ATTEMPT_DISTANCE,
                reason="No face detected in camera frame.",
            )
    else:
        return AuthenticationResult(
            authenticated=False,
            user_id=None,
            display_name=None,
            distance=_NO_ATTEMPT_DISTANCE,
            reason="Invalid video source provided.",
        )

    # 1:N Biometric Matching across all candidate templates
    best_match_user: Optional[UserRecord] = None
    min_distance = float("inf")

    for u_id, (u_record, u_template) in candidates.items():
        dist = float(np.linalg.norm(candidate_embedding - u_template))
        if dist < min_distance:
            min_distance = dist
            best_match_user = u_record

    min_distance = round(min_distance, 4)

    if min_distance <= threshold and best_match_user is not None:
        if best_match_user.user_id == "active_user":
            msg = f"ACCESS GRANTED (Distance: {min_distance:.4f})"
        else:
            msg = f"ACCESS GRANTED (Welcome, {best_match_user.display_name} | Distance: {min_distance:.4f})"
        logger.info("1:N Match Found: '%s' (%s) at distance %.4f <= %.2f", best_match_user.display_name, best_match_user.user_id, min_distance, threshold)
        log_security_event(
            SecurityEventType.AUTHENTICATION_SUCCESS,
            user_id=best_match_user.user_id,
            status="SUCCESS",
            reason=f"Authenticated as {best_match_user.display_name}",
            details={"distance": min_distance, "threshold": threshold},
        )
        return AuthenticationResult(
            authenticated=True,
            user_id=best_match_user.user_id,
            display_name=best_match_user.display_name,
            distance=min_distance,
            reason=msg,
        )
    else:
        msg = f"ACCESS DENIED (Unrecognized User | Min Distance: {min_distance:.4f})"
        logger.info("1:N Match Rejected: min_distance %.4f > threshold %.2f", min_distance, threshold)
        log_security_event(
            SecurityEventType.AUTHENTICATION_FAILURE,
            status="FAILURE",
            reason=msg,
            details={"min_distance": min_distance, "threshold": threshold},
        )
        return AuthenticationResult(
            authenticated=False,
            user_id=None,
            display_name=None,
            distance=min_distance,
            reason=msg,
        )


def authenticate_face(
    source: Union[np.ndarray, WebcamManager],
    threshold: float = MATCH_THRESHOLD,
    on_progress: Optional[AuthProgressCallback] = None,
) -> Tuple[bool, float, str]:
    """Unified entry point for face authentication.

    Preserves exact backward compatibility with previous interface:
    returns (is_authenticated, distance, status_message).
    """
    res = authenticate_user(source=source, threshold=threshold, on_progress=on_progress)
    return res.authenticated, res.distance, res.reason


def authenticate_face_with_liveness(
    camera: WebcamManager,
    threshold: float = MATCH_THRESHOLD,
    required_blinks: int = REQUIRED_BLINKS,
    timeout_seconds: float = LIVENESS_TIMEOUT_SECONDS,
    on_progress: Optional[AuthProgressCallback] = None,
) -> Tuple[bool, float, str]:
    """Authenticates a face with liveness challenge gating."""
    res = authenticate_user(source=camera, threshold=threshold, on_progress=on_progress)
    return res.authenticated, res.distance, res.reason

