import logging
from contextlib import contextmanager
from typing import Generator, Optional

import cv2
import numpy as np

from core.audit_logger import SecurityEventType, log_security_event

logger = logging.getLogger(__name__)


class WebcamManager:
    """Manages the lifecycle of an OpenCV webcam capture handle.

    Wraps cv2.VideoCapture to ensure the hardware resource is always released,
    even when exceptions occur in the calling code. Supports both explicit
    open/release calls and the context manager protocol.

    Attributes:
        camera_index: The OS-level index of the webcam to open.
        cap: The underlying cv2.VideoCapture instance, or None if not opened.
    """

    def __init__(self, camera_index: int = 0) -> None:
        """Initialises the manager without opening the camera.

        Args:
            camera_index: Index of the webcam device. Defaults to 0 (primary camera).
        """
        self.camera_index: int = camera_index
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Opens the webcam and acquires the capture handle.

        Args:
            None

        Returns:
            True if the camera was opened successfully, False otherwise.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error("Failed to open camera at index %d.", self.camera_index)
            log_security_event(
                SecurityEventType.CAMERA_ERROR,
                status="ERROR",
                reason=f"Failed to open camera at index {self.camera_index}",
                details={"camera_index": self.camera_index},
            )
            self.cap = None
            return False
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a single BGR frame from the open webcam.

        Returns:
            A BGR NumPy array (H, W, 3) on success, or None if the read fails
            or the camera is not open.
        """
        if self.cap is None or not self.cap.isOpened():
            logger.warning("read_frame() called on a camera that is not open.")
            return None

        ret: bool
        frame: np.ndarray
        ret, frame = self.cap.read()

        if not ret or frame is None:
            logger.warning("Frame read failed on camera index %d.", self.camera_index)
            return None

        return frame

    def release(self) -> None:
        """Releases the webcam handle and resets internal state.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "WebcamManager":
        """Opens the camera when entering a `with` block.

        Returns:
            This WebcamManager instance.

        Raises:
            RuntimeError: If the camera could not be opened.
        """
        if not self.open():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}.")
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Releases the camera when exiting a `with` block, regardless of exceptions.

        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Exception traceback, if any.
        """
        self.release()


@contextmanager
def get_camera_stream(index: int = 0) -> Generator[WebcamManager, None, None]:
    """Context manager that yields an open WebcamManager and guarantees cleanup.

    Preferred over direct WebcamManager usage when the camera is needed for a
    single bounded scope (e.g. a registration or authentication workflow).

    Args:
        index: Index of the webcam device. Defaults to 0.

    Yields:
        An open WebcamManager instance ready for read_frame() calls.

    Raises:
        RuntimeError: If the camera at the given index cannot be opened.

    Example:
        >>> with get_camera_stream(index=0) as camera:
        ...     frame = camera.read_frame()
    """
    manager = WebcamManager(camera_index=index)
    try:
        if not manager.open():
            raise RuntimeError(f"Could not open camera at index {index}.")
        yield manager
    finally:
        manager.release()
