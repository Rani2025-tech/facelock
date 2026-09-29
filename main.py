"""FaceLock — AI Facial Authentication and Secure Biometric Recovery System.

Main application entrypoint running the real-time OpenCV webcam HUD loop,
anti-spoofing liveness verification challenge, 1:N multi-user biometric
identification, multi-frame registration, and security event auditing.
"""

import logging
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    EAR_THRESHOLD,
    MATCH_THRESHOLD,
    REQUIRED_BLINKS,
    SAMPLE_COUNT,
)
from core.audit_logger import SecurityEventType, log_security_event
from core.persistence import has_active_password, load_active_password
from core.ui_renderer import (
    HUDTelemetry,
    UIState,
    render_hud,
)
from core.user_db import create_or_update_user, list_users, user_count
from utils.camera import WebcamManager
from workflows.authenticate import authenticate_user
from workflows.register import register_face_password, register_user

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Flash message display duration (seconds)
_FLASH_DURATION: float = 4.0

# Active global telemetry state
_telemetry = HUDTelemetry()


# ── Progress Callbacks ────────────────────────────────────────────────────────

def _on_registration_progress(
    frame: np.ndarray,
    status_text: str,
    current_count: int,
    total_count: int,
    bbox: Optional[Tuple[int, int, int, int]],
) -> None:
    """Renders live multi-sample registration telemetry onto the active OpenCV window."""
    _telemetry.state = UIState.REGISTERING
    _telemetry.samples_collected = current_count
    _telemetry.target_samples = total_count
    _telemetry.face_bbox = bbox
    _telemetry.status_message = status_text
    _telemetry.identity_status = f"ENROLLING ({current_count}/{total_count})"

    annotated = render_hud(frame, _telemetry)
    cv2.imshow("Face Auth System", annotated)
    cv2.waitKey(1)


def _on_auth_progress(
    frame: np.ndarray,
    status_text: str,
    blinks_done: int,
    target_blinks: int,
    bbox: Optional[Tuple[int, int, int, int]],
    ear: float,
) -> None:
    """Renders live liveness challenge telemetry and eye-blink feedback onto the active OpenCV window."""
    _telemetry.state = UIState.LIVENESS_CHECK
    _telemetry.blinks_done = blinks_done
    _telemetry.target_blinks = target_blinks
    _telemetry.face_bbox = bbox
    _telemetry.ear = ear
    _telemetry.status_message = status_text

    if blinks_done >= target_blinks:
        _telemetry.liveness_status = "✓ VERIFIED"
    elif blinks_done > 0:
        _telemetry.liveness_status = f"BLINKING ({blinks_done}/{target_blinks})"
    else:
        _telemetry.liveness_status = "CHALLENGE ACTIVE"

    _telemetry.identity_status = "SCANNING..."

    annotated = render_hud(frame, _telemetry)
    cv2.imshow("Face Auth System", annotated)
    cv2.waitKey(1)


# ── Main Application Loop ─────────────────────────────────────────────────────

def run() -> None:
    """Starts the interactive webcam loop for face registration and authentication.

    Interactive Hotkeys:
        [R] — Quick Face Password Registration / Reset (Primary User).
        [U] — Enroll a New Named User Profile in SQLite.
        [L] — List All Registered User Profiles in Console & HUD.
        [A] — Start 1:N Liveness-Verified Face Authentication.
        [Q] — Quit the Application Gracefully.
    """
    global _telemetry

    try:
        with WebcamManager(camera_index=0) as camera:
            logger.info("Webcam initialized successfully. Launching FaceLock HUD.")

            while True:
                frame: Optional[np.ndarray] = camera.read_frame()
                if frame is None:
                    _telemetry.state = UIState.ERROR
                    _telemetry.status_message = "Camera frame drop — retrying..."
                    logger.warning("Empty frame received from camera — retrying.")
                    time.sleep(0.05)
                    continue

                # Refresh enrolled user count in telemetry
                _telemetry.user_count = user_count()
                if _telemetry.user_count == 0 and has_active_password():
                    _telemetry.user_count = 1

                # Reset state to IDLE if flash message expired
                if _telemetry.flash_expiry > 0 and time.time() >= _telemetry.flash_expiry:
                    _telemetry.flash_message = ""
                    _telemetry.flash_expiry = 0.0
                    _telemetry.state = UIState.IDLE
                    _telemetry.status_message = "Ready — Press [A] to Authenticate or [R] to Register"
                    _telemetry.face_bbox = None

                # Poll keyboard events
                key = cv2.waitKey(1) & 0xFF

                if key == ord("r"):
                    # Quick registration / reset
                    _telemetry.state = UIState.REGISTERING
                    _telemetry.liveness_status = "--"
                    _telemetry.identity_status = "ENROLLING PRIMARY USER"
                    _telemetry.distance = None

                    success, msg = register_face_password(
                        camera,
                        sample_count=SAMPLE_COUNT,
                        on_progress=_on_registration_progress,
                    )

                    if success:
                        saved_emb = load_active_password()
                        if saved_emb is not None:
                            create_or_update_user(
                                user_id="primary_user",
                                display_name="Primary User",
                                embedding=saved_emb,
                            )
                        _telemetry.state = UIState.SUCCESS
                        _telemetry.identity_status = "✓ REGISTERED"
                    else:
                        _telemetry.state = UIState.ERROR
                        _telemetry.identity_status = "✗ ENROLLMENT FAILED"

                    _telemetry.flash_message = msg
                    _telemetry.flash_expiry = time.time() + _FLASH_DURATION
                    _telemetry.flash_success = success

                elif key == ord("u"):
                    # Multi-user enrollment
                    next_id = f"user_{user_count() + 1:02d}"
                    display_name = f"User {user_count() + 1}"
                    logger.info("Enrolling user '%s' (%s)...", display_name, next_id)

                    _telemetry.state = UIState.REGISTERING
                    _telemetry.liveness_status = "--"
                    _telemetry.identity_status = f"ENROLLING {display_name}"
                    _telemetry.distance = None

                    success, msg, _ = register_user(
                        user_id=next_id,
                        display_name=display_name,
                        source=camera,
                        sample_count=SAMPLE_COUNT,
                        on_progress=_on_registration_progress,
                    )

                    if success:
                        _telemetry.state = UIState.SUCCESS
                        _telemetry.identity_status = f"✓ ENROLLED ({display_name})"
                    else:
                        _telemetry.state = UIState.ERROR
                        _telemetry.identity_status = "✗ ENROLLMENT FAILED"

                    _telemetry.flash_message = msg
                    _telemetry.flash_expiry = time.time() + _FLASH_DURATION
                    _telemetry.flash_success = success

                elif key == ord("l"):
                    # List enrolled users
                    users = list_users()
                    if users:
                        names = ", ".join([f"{u.display_name} ({u.user_id})" for u in users])
                        _telemetry.flash_message = f"Enrolled ({len(users)}): {names[:42]}..."
                        _telemetry.flash_success = True
                        print("\n" + "=" * 40)
                        print(f"  FACELOCK ENROLLED PROFILES ({len(users)})")
                        print("=" * 40)
                        for u in users:
                            print(f"  • {u.display_name} [ID: {u.user_id}] (Enrolled: {u.created_at[:19]})")
                        print("=" * 40 + "\n")
                    else:
                        _telemetry.flash_message = "No users enrolled in database."
                        _telemetry.flash_success = False

                    _telemetry.flash_expiry = time.time() + _FLASH_DURATION

                elif key == ord("a"):
                    # 1:N multi-user authentication with liveness verification
                    _telemetry.state = UIState.AUTHENTICATING
                    _telemetry.liveness_status = "PENDING"
                    _telemetry.identity_status = "SCANNING..."
                    _telemetry.distance = None

                    auth_res = authenticate_user(
                        camera,
                        on_progress=_on_auth_progress,
                    )

                    _telemetry.distance = auth_res.distance if auth_res.distance != 1.0 else None

                    if auth_res.authenticated:
                        _telemetry.state = UIState.SUCCESS
                        _telemetry.liveness_status = "✓ VERIFIED"
                        _telemetry.identity_status = f"✓ MATCHED ({auth_res.display_name or 'User'})"
                    else:
                        _telemetry.state = UIState.DENIED
                        if "liveness" in auth_res.reason.lower() or "spoof" in auth_res.reason.lower():
                            _telemetry.liveness_status = "✗ FAILED (SPOOF)"
                        else:
                            _telemetry.liveness_status = "✓ VERIFIED"
                        _telemetry.identity_status = "✗ UNRECOGNIZED"

                    _telemetry.flash_message = auth_res.reason
                    _telemetry.flash_success = auth_res.authenticated
                    _telemetry.flash_expiry = time.time() + _FLASH_DURATION

                elif key == ord("q"):
                    logger.info("Quit command received. Shutting down FaceLock.")
                    break

                # ── Render active HUD ─────────────────────────────────────────
                hud_canvas = render_hud(frame, _telemetry)
                cv2.imshow("Face Auth System", hud_canvas)

                # Handle window close (X) button
                if cv2.getWindowProperty("Face Auth System", cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Application window closed by user.")
                    break

    except RuntimeError as exc:
        logger.error("Camera hardware error: %s", exc)
        log_security_event(
            SecurityEventType.CAMERA_ERROR,
            status="ERROR",
            reason=str(exc),
        )
        sys.exit(1)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
