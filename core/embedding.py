"""Core facial feature extraction, quality validation, and template aggregation.

Provides functions to extract 128-dimensional biometric embeddings from video frames
using dlib's ResNet face encoder, enforce quality constraints (single face, minimum
bounding box size), and aggregate multiple frame samples into a robust, normalized
biometric template.
"""

from typing import List, Optional, Tuple, Union

import cv2
import face_recognition
import numpy as np

from config import EXPECTED_EMBEDDING_SHAPE, MIN_FACE_SIZE


def extract_validated_embedding(
    frame: np.ndarray,
    min_face_size: int = MIN_FACE_SIZE,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[Tuple[int, int, int, int]]]:
    """Extracts a 128-D face embedding with quality validation and diagnostics.

    Validates that:
    1. Exactly one face is visible in the frame (rejects zero or multiple faces).
    2. The detected face bounding box satisfies the minimum dimension threshold.

    Args:
        frame: A BGR image as a NumPy array (H, W, 3).
        min_face_size: Minimum height and width (in pixels) for the face bounding box.
            Defaults to MIN_FACE_SIZE (60px).

    Returns:
        A tuple of (embedding, error_message, bounding_box) where:
            - embedding: (128,) float64 NumPy array if valid, else None.
            - error_message: Human-readable rejection reason if invalid, else None.
            - bounding_box: Tuple of (top, right, bottom, left) coordinates if a face
              was located, else None.
    """
    if frame is None or frame.size == 0:
        return None, "Invalid or empty video frame.", None

    rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        return None, "Face not detected — try again", None

    if len(face_locations) > 1:
        return None, "Multiple faces detected — only one person should be visible", None

    bbox = face_locations[0]
    top, right, bottom, left = bbox
    height = bottom - top
    width = right - left

    if height < min_face_size or width < min_face_size:
        return (
            None,
            f"Face too small ({width}x{height}px, min {min_face_size}px) — move closer",
            bbox,
        )

    encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
    if not encodings:
        return None, "Failed to encode face features — try again", bbox

    return encodings[0].astype(np.float64), None, bbox


def extract_face_embedding(frame: np.ndarray) -> Optional[np.ndarray]:
    """Extracts a 128-dimensional face embedding from a BGR frame.

    Preserves backward compatibility with legacy single-frame callers and test suites.

    Args:
        frame: A BGR image as a NumPy array.

    Returns:
        A (128,) float64 NumPy array representing the face embedding, or None
        if no valid single face is detected.
    """
    embedding, _, _ = extract_validated_embedding(frame)
    return embedding


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Projects a 128-dimensional embedding vector to unit L2 norm.

    Args:
        embedding: A 1-D array of shape (128,).

    Returns:
        A (128,) float64 NumPy array with ||v||_2 = 1.0.

    Raises:
        ValueError: If the embedding is None, not 1-D, not shape (128,), or has zero norm.
    """
    if embedding is None:
        raise ValueError("Embedding must not be None.")
    if embedding.ndim != 1 or embedding.shape != EXPECTED_EMBEDDING_SHAPE:
        raise ValueError(
            f"Embedding must be a 1-D array of shape {EXPECTED_EMBEDDING_SHAPE}, "
            f"got {getattr(embedding, 'shape', None)}."
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite values (NaN or Inf).")

    norm: float = float(np.linalg.norm(embedding))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-magnitude embedding vector.")

    return (embedding / norm).astype(np.float64)


def aggregate_embeddings(
    embeddings: Union[List[np.ndarray], np.ndarray],
) -> np.ndarray:
    r"""Aggregates multiple 128-D face embeddings into a single robust template.

    Mathematical Formulation:
        1. Mean Vector Calculation:
           \bar{v} = (1 / N) * \sum_{i=1}^{N} v_i
        2. Hypersphere Projection (Unit L2 Normalization):
           v_template = \bar{v} / ||\bar{v}||_2

    Why Averaging Multiple Samples Improves Robustness:
        - Sensor Noise Cancellation: High-frequency camera noise and pixel quantization
          errors average out to zero across multiple captured frames.
        - Pose & Micro-Expression Invariance: Capturing frames across a temporal window
          integrates slight natural head micromovements, eye gaze variations, and muscle
          twitches into a central biometric centroid.
        - Lighting & Exposure Smoothing: Mitigates momentary auto-exposure shifts and
          shadow fluctuations from room lighting.
        - Metric Geometry Alignment: Normalizing the mean vector back onto the unit
          hypersphere ensures exact calibration with Euclidean distance thresholding (d <= 0.6).

    Args:
        embeddings: A list of (128,) arrays or an (N, 128) 2D NumPy array with N >= 1.

    Returns:
        A normalized (128,) float64 NumPy array representing the biometric template.

    Raises:
        ValueError: If embeddings is empty or contains arrays not matching shape (128,).
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Cannot aggregate an empty collection of embeddings.")

    array_list = list(embeddings)
    for idx, emb in enumerate(array_list):
        if emb is None or emb.ndim != 1 or emb.shape != EXPECTED_EMBEDDING_SHAPE:
            raise ValueError(
                f"Embedding at index {idx} has invalid shape {getattr(emb, 'shape', None)}; "
                f"expected {EXPECTED_EMBEDDING_SHAPE}."
            )

    # Compute element-wise mean across all valid samples
    stacked = np.stack(array_list, axis=0).astype(np.float64)
    mean_vector = np.mean(stacked, axis=0)

    # Normalize to unit L2 norm
    return normalize_embedding(mean_vector)
