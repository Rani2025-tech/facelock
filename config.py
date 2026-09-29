"""Central configuration module for the FaceLock biometric system.

Defines all tunable parameters for sample collection, face quality constraints,
liveness / anti-spoofing detection, matching thresholds, and persistence locations.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR: Path = Path(__file__).resolve().parent

# ── Registration Configuration ────────────────────────────────────────────────
DEFAULT_SAMPLE_COUNT: int = 15
SAMPLE_COUNT: int = DEFAULT_SAMPLE_COUNT  # Number of valid face samples required
MIN_FACE_SIZE: int = 60                   # Min width and height (px) for bounding box
MAX_REGISTRATION_ATTEMPTS: int = 150     # Frame limit before registration times out

# ── Liveness & Anti-Spoofing Configuration ───────────────────────────────────
# EAR (Eye Aspect Ratio) below this threshold indicates closed eyelids.
# Normal open eyes range between 0.25 - 0.38; closed eyes drop below 0.20 - 0.23.
EAR_THRESHOLD: float = 0.22

# Minimum consecutive frames eyes must remain closed to register a valid blink.
# Filters single-frame image noise or frame drop artifacts.
BLINK_MIN_FRAMES: int = 1

# Maximum consecutive frames eyes can remain closed before being treated as a static
# photo with closed eyes or sleeping/squinting rather than a dynamic blink.
BLINK_MAX_FRAMES: int = 25

# Number of complete eye blinks required to fulfill the liveness challenge.
REQUIRED_BLINKS: int = 2

# Maximum duration (in seconds) allowed to complete the liveness challenge.
LIVENESS_TIMEOUT_SECONDS: float = 8.0

# ── Matching & Embedding Configuration ────────────────────────────────────────
DEFAULT_MATCH_THRESHOLD: float = 0.6
# Dynamic match threshold, can be overridden via environment or evaluation configs
MATCH_THRESHOLD: float = float(
    os.getenv("FACELOCK_MATCH_THRESHOLD", str(DEFAULT_MATCH_THRESHOLD))
)
EXPECTED_EMBEDDING_SHAPE: tuple[int, ...] = (128,)

# ── Persistence & Audit Logging Configuration ────────────────────────────────
DEFAULT_PASSWORD_FILE: Path = BASE_DIR / "active_password.npy"
DEFAULT_DATA_DIR: Path = BASE_DIR / "data"
DEFAULT_LOGS_DIR: Path = DEFAULT_DATA_DIR / "logs"
DEFAULT_AUDIT_LOG_FILE: Path = DEFAULT_LOGS_DIR / "security_audit.log"
AUDIT_LOG_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB rotation threshold
AUDIT_LOG_BACKUP_COUNT: int = 5            # 5 backup archives

# ── UI & HUD Design System Configuration ──────────────────────────────────────
# Color tokens (BGR format for OpenCV)
COLOR_BACKGROUND: tuple[int, int, int] = (20, 24, 28)      # Deep Slate Dark
COLOR_TEXT_PRIMARY: tuple[int, int, int] = (245, 245, 245) # Clean White
COLOR_TEXT_MUTED: tuple[int, int, int] = (160, 160, 160)   # Cool Grey
COLOR_CYAN: tuple[int, int, int] = (245, 195, 30)          # Cyber Cyan
COLOR_GREEN: tuple[int, int, int] = (80, 220, 100)         # Emerald Green
COLOR_RED: tuple[int, int, int] = (60, 60, 240)            # Crimson Red
COLOR_AMBER: tuple[int, int, int] = (30, 175, 255)         # Warm Amber
COLOR_CARD_BG: tuple[int, int, int] = (32, 38, 45)         # Elevated HUD Card

# Overlay heights and font scales
HEADER_HEIGHT: int = 68
FOOTER_HEIGHT: int = 38
HUD_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX

