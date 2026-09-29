"""Unit tests for core.matcher and biometric metric space mathematical properties.

Verifies:
1. Metric space axioms:
   - Non-negativity: d(u, v) >= 0
   - Identity of indiscernibles: d(u, u) == 0
   - Symmetry: d(u, v) == d(v, u)
   - Triangle inequality: d(u, w) <= d(u, v) + d(v, w)
2. Threshold classification behavior.
3. Input validation and non-finite / invalid shape rejection.
"""

import numpy as np
import pytest

from core.embedding import normalize_embedding
from core.matcher import DEFAULT_MATCH_THRESHOLD, compare_embeddings


def _random_unit_vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(128)
    return normalize_embedding(vec)


def test_matcher_identity_property() -> None:
    """d(u, u) must be exactly 0.0, resulting in a positive match."""
    vec_a = _random_unit_vector(seed=10)
    is_match, distance = compare_embeddings(vec_a, vec_a)

    assert is_match is True
    assert distance == 0.0


def test_matcher_symmetry_property() -> None:
    """d(u, v) == d(v, u) for any arbitrary pair of embeddings."""
    vec_a = _random_unit_vector(seed=10)
    vec_b = _random_unit_vector(seed=20)

    _, dist_ab = compare_embeddings(vec_a, vec_b)
    _, dist_ba = compare_embeddings(vec_b, vec_a)

    assert dist_ab == pytest.approx(dist_ba, abs=1e-6)


def test_matcher_triangle_inequality() -> None:
    """d(u, w) <= d(u, v) + d(v, w) across arbitrary vectors."""
    u = _random_unit_vector(seed=101)
    v = _random_unit_vector(seed=202)
    w = _random_unit_vector(seed=303)

    _, d_uw = compare_embeddings(u, w)
    _, d_uv = compare_embeddings(u, v)
    _, d_vw = compare_embeddings(v, w)

    assert d_uw <= (d_uv + d_vw + 1e-6)


def test_matcher_non_negativity() -> None:
    """Euclidean distance must always be non-negative."""
    vec_a = _random_unit_vector(seed=42)
    vec_b = _random_unit_vector(seed=84)

    _, dist = compare_embeddings(vec_a, vec_b)
    assert dist >= 0.0


def test_matcher_threshold_boundary() -> None:
    """Verifies strict threshold classification boundary (d < threshold)."""
    vec_a = _random_unit_vector(seed=1)

    # Synthetic candidate at exactly known distance from vec_a
    # By perturbing one dimension slightly
    vec_b = vec_a.copy()
    vec_b[0] += 0.1
    vec_b = normalize_embedding(vec_b)

    _, actual_dist = compare_embeddings(vec_a, vec_b)

    # Test with threshold set strictly above and below actual distance
    is_match_above, _ = compare_embeddings(vec_a, vec_b, threshold=actual_dist + 0.01)
    is_match_below, _ = compare_embeddings(vec_a, vec_b, threshold=actual_dist - 0.01)

    assert is_match_above is True
    assert is_match_below is False


def test_matcher_rejects_nan_and_inf_vectors() -> None:
    """Verifies that non-finite values (NaN, Inf, -Inf) are rejected with ValueError."""
    valid_vec = _random_unit_vector(seed=7)

    nan_vec = valid_vec.copy()
    nan_vec[10] = np.nan

    inf_vec = valid_vec.copy()
    inf_vec[20] = np.inf

    neginf_vec = valid_vec.copy()
    neginf_vec[30] = -np.inf

    with pytest.raises(ValueError, match="non-finite"):
        compare_embeddings(nan_vec, valid_vec)

    with pytest.raises(ValueError, match="non-finite"):
        compare_embeddings(valid_vec, inf_vec)

    with pytest.raises(ValueError, match="non-finite"):
        compare_embeddings(neginf_vec, valid_vec)


def test_matcher_rejects_invalid_shapes_and_none() -> None:
    """Verifies shape and type validation in compare_embeddings."""
    valid_vec = _random_unit_vector(seed=12)

    with pytest.raises(ValueError, match="must not be None"):
        compare_embeddings(None, valid_vec)  # type: ignore

    with pytest.raises(ValueError, match="1-dimensional"):
        compare_embeddings(np.zeros((64, 2)), valid_vec)

    with pytest.raises(ValueError, match="shape"):
        compare_embeddings(np.zeros((64,)), valid_vec)
