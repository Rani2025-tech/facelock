from typing import Tuple

import numpy as np

DEFAULT_MATCH_THRESHOLD: float = 0.6

_EXPECTED_EMBEDDING_SHAPE: Tuple[int, ...] = (128,)


def compare_embeddings(
    known_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Tuple[bool, float]:
    """Computes Euclidean distance between two face embeddings and determines a match.

    Uses L2 (Euclidean) distance in 128-dimensional embedding space. A distance
    below the threshold indicates the two embeddings belong to the same identity.

    Args:
        known_embedding: The stored reference face embedding, shape (128,).
        candidate_embedding: The live candidate face embedding, shape (128,).
        threshold: Maximum Euclidean distance to consider a positive match.
            Defaults to 0.6, the standard dlib/face_recognition boundary.

    Returns:
        A tuple of (is_match, distance) where:
            - is_match: True if distance < threshold, False otherwise.
            - distance: The rounded Euclidean distance as a plain Python float.

    Raises:
        ValueError: If either embedding is None, not 1-dimensional, or not shape (128,).
    """
    _validate_embedding(known_embedding, "known_embedding")
    _validate_embedding(candidate_embedding, "candidate_embedding")

    distance: float = round(
        float(np.linalg.norm(known_embedding - candidate_embedding)), 6
    )
    is_match: bool = distance < threshold

    return is_match, distance


def _validate_embedding(embedding: np.ndarray, name: str) -> None:
    """Validates that an embedding is a non-None, 1-D array of shape (128,).

    Args:
        embedding: The array to validate.
        name: The parameter name, used in error messages.

    Raises:
        ValueError: If the embedding fails any validation check.
    """
    if embedding is None:
        raise ValueError(f"{name} must not be None.")
    if embedding.ndim != 1:
        raise ValueError(
            f"{name} must be 1-dimensional, got ndim={embedding.ndim}."
        )
    if embedding.shape != _EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"{name} must have shape {_EXPECTED_EMBEDDING_SHAPE}, got {embedding.shape}."
        )
    if not np.isfinite(embedding).all():
        raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
