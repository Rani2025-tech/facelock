"""Multi-frame face registration and biometric credential creation workflow.

Captures and validates multiple facial embedding samples (default: 15) across
consecutive frames, discards invalid or noisy frames (no face, multiple faces,
too small), aggregates the valid samples into an averaged centroid, normalizes
the resulting vector, encrypts the template at rest, and enrolls the user record in SQLite.
"""

import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from config import (
    DEFAULT_PASSWORD_FILE,
    MAX_REGISTRATION_ATTEMPTS,
    SAMPLE_COUNT,
)
from core.audit_logger import SecurityEventType, log_security_event
from core.embedding import (
    aggregate_embeddings,
    extract_face_embedding,
    extract_validated_embedding,
)
from core.persistence import save_active_password
from core.user_db import UserRecord, create_user, get_user, validate_user_id
from utils.camera import WebcamManager

logger = logging.getLogger(__name__)

# Type alias for the registration progress callback:
# (frame, status_message, current_sample_count, target_sample_count, bbox)
ProgressCallback = Callable[
    [np.ndarray, str, int, int, Optional[Tuple[int, int, int, int]]], None
]


def register_user(
    user_id: str,
    display_name: str,
    source: Union[np.ndarray, Iterable[np.ndarray], WebcamManager],
    sample_count: int = SAMPLE_COUNT,
    db_path: Optional[Path] = None,
    users_dir: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[bool, str, Optional[UserRecord]]:
    """Enrolls a new named user with multi-sample facial biometric verification.

    Args:
        user_id: Unique user identifier (alphanumeric, 2-32 chars).
        display_name: Full display name.
        source: Video source (WebcamManager, list of frames, or single frame).
        sample_count: Number of valid frames required (default: 15).
        db_path: Optional SQLite database path.
        users_dir: Optional template storage directory.
        on_progress: Optional UI progress callback.

    Returns:
        A tuple of (success, message, user_record).
    """
    try:
        clean_id = validate_user_id(user_id)
    except ValueError as exc:
        log_security_event(
            SecurityEventType.REGISTRATION_FAILURE,
            user_id=user_id,
            status="FAILURE",
            reason=str(exc),
        )
        return False, str(exc), None

    # Check for duplicate registration
    existing = get_user(clean_id, db_path=db_path)
    if existing is not None:
        msg = f"User with ID '{clean_id}' already exists. Choose a different ID or delete the user."
        logger.warning(msg)
        log_security_event(
            SecurityEventType.REGISTRATION_FAILURE,
            user_id=clean_id,
            status="FAILURE",
            reason=msg,
        )
        return False, msg, None

    # Extract template from source
    template = _collect_and_aggregate_template(source, sample_count=sample_count, on_progress=on_progress)
    if template is None:
        msg = "Registration failed: could not capture valid face samples."
        log_security_event(
            SecurityEventType.REGISTRATION_FAILURE,
            user_id=clean_id,
            status="FAILURE",
            reason=msg,
        )
        return False, msg, None

    try:
        user = create_user(
            user_id=clean_id,
            display_name=display_name,
            embedding=template,
            db_path=db_path,
            users_dir=users_dir,
        )
        # Also update active_password.npy to maintain single-user backward compatibility
        save_active_password(template)
        logger.info("Successfully enrolled user '%s' (%s).", display_name, clean_id)
        log_security_event(
            SecurityEventType.REGISTRATION_SUCCESS,
            user_id=clean_id,
            status="SUCCESS",
            reason=f"User '{display_name}' enrolled successfully",
            details={"sample_count": sample_count},
        )
        return True, f"User '{display_name}' ({clean_id}) registered successfully!", user
    except Exception as exc:
        logger.error("Failed to enroll user '%s': %s", clean_id, exc)
        log_security_event(
            SecurityEventType.STORAGE_ERROR,
            user_id=clean_id,
            status="ERROR",
            reason=f"Failed to persist user profile: {exc}",
        )
        return False, f"Failed to persist user profile: {exc}", None


def _collect_and_aggregate_template(
    source: Union[np.ndarray, Iterable[np.ndarray], WebcamManager],
    sample_count: int = SAMPLE_COUNT,
    on_progress: Optional[ProgressCallback] = None,
) -> Optional[np.ndarray]:
    """Helper that captures and aggregates 128-D template from various input sources."""
    if isinstance(source, WebcamManager):
        collected: List[np.ndarray] = []
        attempt = 0
        max_attempts = MAX_REGISTRATION_ATTEMPTS

        while len(collected) < sample_count and attempt < max_attempts:
            attempt += 1
            frame = source.read_frame()
            if frame is None:
                continue

            emb, err, bbox = extract_validated_embedding(frame)
            if emb is not None:
                collected.append(emb)
                status_text = f"Sample {len(collected)}/{sample_count} captured"
            else:
                status_text = err or "Face not detected — try again"

            if on_progress is not None:
                on_progress(frame, status_text, len(collected), sample_count, bbox)

        if len(collected) < sample_count:
            return None
        return aggregate_embeddings(collected)

    if isinstance(source, (list, tuple)):
        collected = []
        for frame in source:
            if frame is None or frame.size == 0:
                continue
            emb, _, _ = extract_validated_embedding(frame)
            if emb is not None:
                collected.append(emb)
                if len(collected) >= sample_count:
                    break
        if len(collected) < sample_count:
            return None
        return aggregate_embeddings(collected[:sample_count])

    if isinstance(source, np.ndarray) and source.ndim == 3:
        emb = extract_face_embedding(source)
        if emb is None:
            return None
        return aggregate_embeddings([emb])

    return None


def register_multiframe_face_password(
    frames: Iterable[np.ndarray],
    sample_count: int = SAMPLE_COUNT,
    filepath: Union[str, Path] = DEFAULT_PASSWORD_FILE,
) -> Tuple[bool, str]:
    """Registers a face password from an iterable sequence of BGR frames."""
    collected_embeddings: List[np.ndarray] = []

    logger.info("Multi-frame registration started. Target: %d valid samples.", sample_count)

    for frame in frames:
        if frame is None or frame.size == 0:
            continue

        embedding, err_msg, _ = extract_validated_embedding(frame)
        if embedding is not None:
            collected_embeddings.append(embedding)
            if len(collected_embeddings) >= sample_count:
                break

    if len(collected_embeddings) < sample_count:
        msg = (
            f"Registration aborted: insufficient valid samples "
            f"({len(collected_embeddings)}/{sample_count} collected)."
        )
        logger.warning(msg)
        log_security_event(
            SecurityEventType.REGISTRATION_FAILURE,
            user_id="primary_user",
            status="FAILURE",
            reason=msg,
        )
        return False, msg

    try:
        final_template = aggregate_embeddings(collected_embeddings[:sample_count])
    except ValueError as exc:
        logger.error("Embedding aggregation failed: %s", exc)
        log_security_event(
            SecurityEventType.REGISTRATION_FAILURE,
            user_id="primary_user",
            status="FAILURE",
            reason=f"Aggregation error: {exc}",
        )
        return False, f"Aggregation error: {exc}"

    saved = save_active_password(final_template, filepath=filepath)
    if not saved:
        logger.error("Failed to persist aggregated embedding to '%s'.", filepath)
        log_security_event(
            SecurityEventType.STORAGE_ERROR,
            user_id="primary_user",
            status="ERROR",
            reason=f"Failed to persist face embedding to {filepath}",
        )
        return False, "Failed to persist face embedding to disk."

    logger.info(
        "Face password registered/updated successfully from %d samples at '%s'.",
        sample_count,
        filepath,
    )
    log_security_event(
        SecurityEventType.REGISTRATION_SUCCESS,
        user_id="primary_user",
        status="SUCCESS",
        reason="Face password registered/updated successfully",
        details={"sample_count": sample_count},
    )
    return True, "Face password registered/updated successfully!"


def capture_and_register_from_camera(
    camera: WebcamManager,
    sample_count: int = SAMPLE_COUNT,
    max_attempts: int = MAX_REGISTRATION_ATTEMPTS,
    filepath: Union[str, Path] = DEFAULT_PASSWORD_FILE,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[bool, str]:
    """Interactively captures frames from an active camera and registers a face."""
    collected_embeddings: List[np.ndarray] = []
    attempt_count = 0

    logger.info("Interactive camera registration started (target=%d).", sample_count)

    while len(collected_embeddings) < sample_count and attempt_count < max_attempts:
        attempt_count += 1
        frame = camera.read_frame()
        if frame is None:
            continue

        embedding, error_msg, bbox = extract_validated_embedding(frame)

        if embedding is not None:
            collected_embeddings.append(embedding)
            status_text = f"Sample {len(collected_embeddings)}/{sample_count} captured"
        else:
            status_text = error_msg or "Face not detected — try again"

        if on_progress is not None:
            on_progress(
                frame,
                status_text,
                len(collected_embeddings),
                sample_count,
                bbox,
            )

    if len(collected_embeddings) < sample_count:
        msg = (
            f"Registration timed out: only {len(collected_embeddings)}/{sample_count} "
            f"valid samples captured."
        )
        logger.warning(msg)
        return False, msg

    try:
        final_template = aggregate_embeddings(collected_embeddings)
    except ValueError as exc:
        logger.error("Embedding aggregation failed: %s", exc)
        return False, f"Aggregation error: {exc}"

    saved = save_active_password(final_template, filepath=filepath)
    if not saved:
        logger.error("Failed to persist face embedding to disk.")
        return False, "Failed to persist face embedding to disk."

    logger.info("Face password registered successfully at '%s'.", filepath)
    return True, "Face password registered/updated successfully!"


def register_face_password(
    source: Union[np.ndarray, Iterable[np.ndarray], WebcamManager],
    sample_count: int = SAMPLE_COUNT,
    filepath: Union[str, Path] = DEFAULT_PASSWORD_FILE,
    on_progress: Optional[ProgressCallback] = None,
) -> Tuple[bool, str]:
    """Unified entry point for face password registration and reset."""
    if isinstance(source, WebcamManager):
        return capture_and_register_from_camera(
            camera=source,
            sample_count=sample_count,
            filepath=filepath,
            on_progress=on_progress,
        )

    if isinstance(source, (list, tuple)):
        return register_multiframe_face_password(
            frames=source,
            sample_count=sample_count,
            filepath=filepath,
        )

    if isinstance(source, np.ndarray) and source.ndim == 3:
        embedding = extract_face_embedding(source)
        if embedding is None:
            logger.warning("Registration failed: no face detected in the supplied frame.")
            return False, "No face detected in camera frame. Password registration aborted."

        template = aggregate_embeddings([embedding])
        saved = save_active_password(template, filepath=filepath)
        if not saved:
            logger.error("Registration failed: could not persist embedding to '%s'.", filepath)
            return False, "Failed to persist face embedding to disk."

        logger.info("Face password registered/updated successfully at '%s'.", filepath)
        return True, "Face password registered/updated successfully!"

    if hasattr(source, "__iter__"):
        return register_multiframe_face_password(
            frames=source,
            sample_count=sample_count,
            filepath=filepath,
        )

    logger.error("Invalid source type provided to register_face_password: %s", type(source))
    return False, "Invalid image source provided for registration."
