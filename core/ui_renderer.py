"""HUD and UI Rendering Engine for the FaceLock Biometric System.

Renders an interactive, telemetry-rich OpenCV Heads-Up Display (HUD)
supporting all 8 operational system states:
- IDLE: Idle monitoring, scanning camera for subjects
- DETECTING: Face detected and aligned in viewfinder
- REGISTERING: Multi-frame sample collection with live progress bar
- LIVENESS_CHECK: Dynamic blink detection challenge and EAR telemetry
- AUTHENTICATING: 1:N vector comparison against biometric repository
- SUCCESS: Visual access granted badge with matched user identity and distance
- DENIED: Visual access denied alert with spoof or threshold rejection diagnostic
- ERROR: Camera error, multiple faces detected, or storage failure
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    COLOR_AMBER,
    COLOR_BACKGROUND,
    COLOR_CARD_BG,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_PRIMARY,
    DEFAULT_MATCH_THRESHOLD,
    EAR_THRESHOLD,
    FOOTER_HEIGHT,
    HEADER_HEIGHT,
    HUD_FONT,
    MATCH_THRESHOLD,
    REQUIRED_BLINKS,
    SAMPLE_COUNT,
)


class UIState(str, Enum):
    """Enumeration of all visual HUD operational states."""

    IDLE = "IDLE"
    DETECTING = "DETECTING"
    REGISTERING = "REGISTERING"
    LIVENESS_CHECK = "LIVENESS_CHECK"
    AUTHENTICATING = "AUTHENTICATING"
    SUCCESS = "SUCCESS"
    DENIED = "DENIED"
    ERROR = "ERROR"


@dataclass
class HUDTelemetry:
    """State and telemetry payload passed to the HUD renderer."""

    state: UIState = UIState.IDLE
    status_message: str = "Ready — Press [A] to Authenticate or [R] to Register"
    liveness_status: str = "--"
    identity_status: str = "--"
    distance: Optional[float] = None
    threshold: float = MATCH_THRESHOLD
    blinks_done: int = 0
    target_blinks: int = REQUIRED_BLINKS
    ear: float = 0.0
    samples_collected: int = 0
    target_samples: int = SAMPLE_COUNT
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    user_count: int = 0
    flash_message: str = ""
    flash_expiry: float = 0.0
    flash_success: bool = False


# ── HUD Drawing Subroutines ───────────────────────────────────────────────────

def _draw_semi_transparent_rect(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.75,
) -> None:
    """Draws a semi-transparent rectangle overlay on the image."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    sub_img = image[y1:y2, x1:x2]
    overlay = np.full_like(sub_img, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub_img, 1.0 - alpha, 0, sub_img)


def _draw_header(frame: np.ndarray, telemetry: HUDTelemetry) -> None:
    """Renders the top branding header and enrolled user telemetry."""
    w = frame.shape[1]

    # Semi-transparent top bar
    _draw_semi_transparent_rect(frame, (0, 0), (w, HEADER_HEIGHT), COLOR_BACKGROUND, alpha=0.88)
    cv2.line(frame, (0, HEADER_HEIGHT), (w, HEADER_HEIGHT), (45, 55, 65), 1)

    # Title & Subtitle
    title_text = "FACELOCK"
    subtitle_text = "AI Facial Authentication System"

    # Centered Title
    title_size = cv2.getTextSize(title_text, HUD_FONT, 0.72, 2)[0]
    title_x = (w - title_size[0]) // 2
    cv2.putText(frame, title_text, (title_x, 26), HUD_FONT, 0.72, COLOR_CYAN, 2, cv2.LINE_AA)

    sub_size = cv2.getTextSize(subtitle_text, HUD_FONT, 0.40, 1)[0]
    sub_x = (w - sub_size[0]) // 2
    cv2.putText(frame, subtitle_text, (sub_x, 48), HUD_FONT, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # User enrollment indicator (Top-Right)
    user_str = f"Enrolled: {telemetry.user_count} Users" if telemetry.user_count > 0 else "Enrolled: 0 (Register First)"
    user_col = COLOR_GREEN if telemetry.user_count > 0 else COLOR_AMBER
    cv2.putText(frame, user_str, (w - 200, 24), HUD_FONT, 0.40, user_col, 1, cv2.LINE_AA)

    # State Badge (Top-Left)
    state_col = _get_state_color(telemetry.state)
    cv2.putText(frame, f"STATE: {telemetry.state.value}", (14, 24), HUD_FONT, 0.44, state_col, 1, cv2.LINE_AA)


def _draw_footer(frame: np.ndarray) -> None:
    """Renders the bottom navigation bar with interactive hotkey prompts."""
    h, w = frame.shape[:2]
    bar_y = h - FOOTER_HEIGHT

    _draw_semi_transparent_rect(frame, (0, bar_y), (w, h), COLOR_BACKGROUND, alpha=0.88)
    cv2.line(frame, (0, bar_y), (w, bar_y), (45, 55, 65), 1)

    legend_text = "[R] Register   [U] New User   [L] List Users   [A] Authenticate   [Q] Quit"
    text_size = cv2.getTextSize(legend_text, HUD_FONT, 0.44, 1)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, legend_text, (text_x, bar_y + 24), HUD_FONT, 0.44, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)


def _draw_face_target_box(frame: np.ndarray, bbox: Tuple[int, int, int, int], state: UIState) -> None:
    """Draws sleek corner brackets around the detected face bounding box."""
    top, right, bottom, left = bbox
    color = _get_state_color(state)
    line_len = max(15, min((right - left) // 4, (bottom - top) // 4))
    thickness = 2

    # Corner brackets (Top-Left)
    cv2.line(frame, (left, top), (left + line_len, top), color, thickness)
    cv2.line(frame, (left, top), (left, top + line_len), color, thickness)

    # Top-Right
    cv2.line(frame, (right, top), (right - line_len, top), color, thickness)
    cv2.line(frame, (right, top), (right, top + line_len), color, thickness)

    # Bottom-Left
    cv2.line(frame, (left, bottom), (left + line_len, bottom), color, thickness)
    cv2.line(frame, (left, bottom), (left, bottom - line_len), color, thickness)

    # Bottom-Right
    cv2.line(frame, (right, bottom), (right - line_len, bottom), color, thickness)
    cv2.line(frame, (right, bottom), (right, bottom - line_len), color, thickness)


def _draw_telemetry_card(frame: np.ndarray, telemetry: HUDTelemetry) -> None:
    """Renders the structured telemetry panel showing Liveness, Identity, Distance, and Status."""
    h, w = frame.shape[:2]

    card_w = min(420, w - 24)
    card_h = 108
    card_x = 12
    card_y = h - FOOTER_HEIGHT - card_h - 10

    _draw_semi_transparent_rect(
        frame,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        COLOR_CARD_BG,
        alpha=0.82,
    )
    cv2.rectangle(
        frame,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        (50, 60, 72),
        1,
    )

    # Row 1: Liveness Status
    row1_y = card_y + 24
    cv2.putText(frame, "LIVENESS:", (card_x + 12, row1_y), HUD_FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    liv_color = _get_status_label_color(telemetry.liveness_status)
    cv2.putText(frame, telemetry.liveness_status, (card_x + 110, row1_y), HUD_FONT, 0.44, liv_color, 1, cv2.LINE_AA)

    # Live EAR telemetry indicator (if available)
    if telemetry.ear > 0.0:
        eye_state = "CLOSED" if telemetry.ear < EAR_THRESHOLD else "OPEN"
        ear_str = f"EAR: {telemetry.ear:.2f} [{eye_state}]"
        cv2.putText(frame, ear_str, (card_x + card_w - 140, row1_y), HUD_FONT, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Row 2: Identity / Match Status
    row2_y = card_y + 48
    cv2.putText(frame, "IDENTITY:", (card_x + 12, row2_y), HUD_FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    id_color = _get_status_label_color(telemetry.identity_status)
    cv2.putText(frame, telemetry.identity_status[:28], (card_x + 110, row2_y), HUD_FONT, 0.44, id_color, 1, cv2.LINE_AA)

    # Row 3: Biometric Distance Metric
    row3_y = card_y + 72
    cv2.putText(frame, "DISTANCE:", (card_x + 12, row3_y), HUD_FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    if telemetry.distance is not None:
        dist_color = COLOR_GREEN if telemetry.distance <= telemetry.threshold else COLOR_RED
        dist_str = f"{telemetry.distance:.4f}  (Threshold: <= {telemetry.threshold:.2f})"
        cv2.putText(frame, dist_str, (card_x + 110, row3_y), HUD_FONT, 0.44, dist_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"--  (Threshold: <= {telemetry.threshold:.2f})", (card_x + 110, row3_y), HUD_FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Row 4: Primary Status State Badge
    row4_y = card_y + 96
    cv2.putText(frame, "STATUS:", (card_x + 12, row4_y), HUD_FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    state_col = _get_state_color(telemetry.state)
    state_label = telemetry.state.value
    if telemetry.state == UIState.SUCCESS:
        state_label = "AUTHENTICATED"
    elif telemetry.state == UIState.DENIED:
        state_label = "ACCESS DENIED"
    elif telemetry.state == UIState.REGISTERING:
        state_label = f"ENROLLING ({telemetry.samples_collected}/{telemetry.target_samples})"
    elif telemetry.state == UIState.LIVENESS_CHECK:
        state_label = f"CHALLENGE ({telemetry.blinks_done}/{telemetry.target_blinks} BLINKS)"

    cv2.putText(frame, state_label, (card_x + 110, row4_y), HUD_FONT, 0.48, state_col, 2, cv2.LINE_AA)


def _draw_progress_bar(frame: np.ndarray, telemetry: HUDTelemetry) -> None:
    """Renders a progress bar across the screen when registering samples."""
    h, w = frame.shape[:2]
    bar_x = 12
    bar_y = h - FOOTER_HEIGHT - 6
    bar_w = w - 24
    bar_h = 6

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 42, 50), -1)
    if telemetry.target_samples > 0 and telemetry.samples_collected > 0:
        pct = min(1.0, telemetry.samples_collected / telemetry.target_samples)
        fill_w = int(bar_w * pct)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COLOR_CYAN, -1)


def _draw_transient_banner(frame: np.ndarray, telemetry: HUDTelemetry) -> None:
    """Renders high-visibility transient message banner."""
    if not telemetry.flash_message or time.time() >= telemetry.flash_expiry:
        return

    h, w = frame.shape[:2]
    banner_y = HEADER_HEIGHT + 14
    banner_h = 36

    banner_col = COLOR_GREEN if telemetry.flash_success else COLOR_RED
    _draw_semi_transparent_rect(frame, (12, banner_y), (w - 12, banner_y + banner_h), (15, 18, 22), alpha=0.90)
    cv2.rectangle(frame, (12, banner_y), (w - 12, banner_y + banner_h), banner_col, 1)

    text_size = cv2.getTextSize(telemetry.flash_message, HUD_FONT, 0.52, 1)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(
        frame,
        telemetry.flash_message,
        (max(20, text_x), banner_y + 24),
        HUD_FONT,
        0.52,
        banner_col,
        2,
        cv2.LINE_AA,
    )


# ── Color & Utility Helpers ───────────────────────────────────────────────────

def _get_state_color(state: UIState) -> Tuple[int, int, int]:
    """Maps UI operational state to distinct thematic BGR color tokens."""
    if state == UIState.SUCCESS:
        return COLOR_GREEN
    if state == UIState.DENIED:
        return COLOR_RED
    if state in (UIState.REGISTERING, UIState.LIVENESS_CHECK):
        return COLOR_AMBER
    if state == UIState.ERROR:
        return COLOR_RED
    if state in (UIState.DETECTING, UIState.AUTHENTICATING):
        return COLOR_CYAN
    return COLOR_TEXT_PRIMARY


def _get_status_label_color(label: str) -> Tuple[int, int, int]:
    """Returns color based on status text keywords."""
    low = label.lower()
    if "verif" in low or "match" in low or "pass" in low or "grant" in low:
        return COLOR_GREEN
    if "fail" in low or "deni" in low or "unrec" in low or "spoof" in low:
        return COLOR_RED
    if "pend" in low or "scan" in low or "wait" in low or "check" in low:
        return COLOR_AMBER
    return COLOR_TEXT_PRIMARY


# ── Public Entrypoint ─────────────────────────────────────────────────────────

def render_hud(frame: np.ndarray, telemetry: HUDTelemetry) -> np.ndarray:
    """Renders the complete FaceLock HUD interface onto the given camera frame.

    Args:
        frame: BGR NumPy video frame.
        telemetry: Current operational state and telemetry metadata.

    Returns:
        Annotated BGR frame ready for cv2.imshow().
    """
    canvas = frame.copy()

    # 1. Face bounding box with alignment corner brackets
    if telemetry.face_bbox is not None:
        _draw_face_target_box(canvas, telemetry.face_bbox, telemetry.state)

    # 2. Header and Footer
    _draw_header(canvas, telemetry)
    _draw_footer(canvas)

    # 3. Telemetry card
    _draw_telemetry_card(canvas, telemetry)

    # 4. Progress bar during registration
    if telemetry.state == UIState.REGISTERING:
        _draw_progress_bar(canvas, telemetry)

    # 5. Transient alert banner (if active)
    _draw_transient_banner(canvas, telemetry)

    return canvas
