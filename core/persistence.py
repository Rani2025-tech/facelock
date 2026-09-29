import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Trusted base directory: all password files must resolve inside this folder.
_BASE_DIR: Path = Path(__file__).resolve().parent.parent

DEFAULT_PASSWORD_FILE: Path = _BASE_DIR / "active_password.npy"
EXPECTED_EMBEDDING_SHAPE: tuple[int, ...] = (128,)


def _safe_resolve(filepath: Union[str, Path]) -> Path:
    """Resolves filepath to an absolute path and validates it stays within _BASE_DIR.

    Args:
        filepath: A relative or absolute path supplied by the caller.

    Returns:
        The resolved absolute Path.

    Raises:
        ValueError: If the resolved path escapes the trusted base directory.
    """
    resolved = Path(filepath).resolve()
    if not str(resolved).startswith(str(_BASE_DIR)):
        raise ValueError(
            f"Filepath '{filepath}' resolves outside the trusted base directory."
        )
    return resolved


def save_active_password(
    embedding: np.ndarray,
    filepath: Union[str, Path] = DEFAULT_PASSWORD_FILE,
) -> bool:
    """Persists a 128-dimensional face embedding to disk as a .npy file.

    Validates the embedding shape before writing. The save is performed to a
    sibling temp file first, then atomically renamed to prevent a corrupted
    state if the process is interrupted mid-write.

    Args:
        embedding: A 1-D float64 NumPy array of shape (128,).
        filepath: Destination path for the .npy file.
            Defaults to DEFAULT_PASSWORD_FILE.

    Returns:
        True if the file was written successfully, False on any I/O failure.

    Raises:
        ValueError: If embedding is None, not 1-dimensional, or not shape (128,).
    """
    if embedding is None or embedding.ndim != 1 or embedding.shape != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"embedding must be a 1-D array of shape {EXPECTED_EMBEDDING_SHAPE}, "
            f"got {getattr(embedding, 'shape', None)}."
        )

    filepath = _safe_resolve(filepath)
    tmp_path = filepath.with_suffix(".tmp.npy")

    try:
        np.save(tmp_path, embedding)
        os.replace(tmp_path, filepath)
        return True
    except (IOError, OSError) as exc:
        logger.error("Failed to save active password to '%s': %s", filepath, exc)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False


def load_active_password(
    filepath: Union[str, Path] = DEFAULT_PASSWORD_FILE,
) -> Optional[np.ndarray]:
    """Loads the stored 128-dimensional face embedding from disk.

    Returns None cleanly when no password has been registered yet (file absent)
    or when the file is corrupted / contains an unexpected shape.

    Args:
        filepath: Path to the .npy file to load.
            Defaults to DEFAULT_PASSWORD_FILE.

    Returns:
        A (128,) float64 NumPy array if the file exists and is valid,
        or None otherwise.
    """
    filepath = _safe_resolve(filepath)

    if not filepath.exists():
        return None

    try:
        with open(filepath, "rb") as fh:
            embedding: np.ndarray = np.load(fh)
    except (IOError, OSError, ValueError) as exc:
        logger.error("Failed to load active password from '%s': %s", filepath, exc)
        return None

    if embedding.ndim != 1 or embedding.shape != EXPECTED_EMBEDDING_SHAPE:
        logger.error(
            "Corrupted password file '%s': expected shape %s, got %s.",
            filepath,
            EXPECTED_EMBEDDING_SHAPE,
            embedding.shape,
        )
        return None

    return embedding


def has_active_password(
    filepath: Union[str, Path] = DEFAULT_PASSWORD_FILE,
) -> bool:
    """Checks whether a non-empty active password file exists on disk.

    Does not validate the file contents — use load_active_password() for that.

    Args:
        filepath: Path to the .npy file to check.
            Defaults to DEFAULT_PASSWORD_FILE.

    Returns:
        True if the file exists and has a non-zero size, False otherwise.
    """
    filepath = _safe_resolve(filepath)
    return filepath.exists() and filepath.stat().st_size > 0
