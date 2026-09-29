"""Hardware-independent test suite for biometric evaluation metrics and plots.

Validates:
- False Accept Rate (FAR) calculation.
- False Reject Rate (FRR) calculation.
- Confusion matrix calculations (TP, FN, TN, FP).
- Classification metrics: Precision, Recall, F1 Score, and Accuracy.
- Threshold sweep and Equal Error Rate (EER) identification.
- Edge cases (empty distance arrays, zero divisions, extreme thresholds).
- Synthetic dataset generation abstraction.
- Plot generation utilities (saving PNGs headlessly).
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from evaluation.metrics import (
    EvaluationDataset,
    EvaluationPair,
    calculate_confusion_matrix,
    calculate_far,
    calculate_frr,
    calculate_metrics_at_threshold,
    evaluate_threshold_sweep,
    generate_synthetic_evaluation_dataset,
)
from evaluation.plots import (
    generate_all_plots,
    plot_distance_distributions,
    plot_far_frr_curve,
    plot_roc_curve,
)


# ── Metric Calculation Tests ──────────────────────────────────────────────────

def test_calculate_far_exact_values() -> None:
    """Verifies FAR calculation against hand-computed ground truth."""
    # 4 impostor distances: 0.40, 0.55, 0.70, 0.90
    impostor_distances = np.array([0.40, 0.55, 0.70, 0.90], dtype=np.float64)

    # Threshold 0.30: None are <= 0.30 -> FAR = 0/4 = 0.0
    assert calculate_far(impostor_distances, threshold=0.30) == 0.0

    # Threshold 0.60: 0.40 and 0.55 are <= 0.60 -> FAR = 2/4 = 0.5
    assert calculate_far(impostor_distances, threshold=0.60) == 0.5

    # Threshold 1.00: All 4 are <= 1.00 -> FAR = 4/4 = 1.0
    assert calculate_far(impostor_distances, threshold=1.00) == 1.0


def test_calculate_frr_exact_values() -> None:
    """Verifies FRR calculation against hand-computed ground truth."""
    # 4 genuine distances: 0.20, 0.35, 0.50, 0.65
    genuine_distances = np.array([0.20, 0.35, 0.50, 0.65], dtype=np.float64)

    # Threshold 0.10: All 4 are > 0.10 -> FRR = 4/4 = 1.0
    assert calculate_frr(genuine_distances, threshold=0.10) == 1.0

    # Threshold 0.40: 0.50 and 0.65 are > 0.40 -> FRR = 2/4 = 0.5
    assert calculate_frr(genuine_distances, threshold=0.40) == 0.5

    # Threshold 0.80: None are > 0.80 -> FRR = 0/4 = 0.0
    assert calculate_frr(genuine_distances, threshold=0.80) == 0.0


def test_calculate_confusion_matrix() -> None:
    """Verifies TP, FN, TN, FP counting logic."""
    gen = np.array([0.20, 0.40, 0.70])      # At tau=0.5: 0.20,0.40 <= 0.5 (TP=2), 0.70 > 0.5 (FN=1)
    imp = np.array([0.30, 0.60, 0.80, 0.90]) # At tau=0.5: 0.30 <= 0.5 (FP=1), 0.60,0.80,0.90 > 0.5 (TN=3)

    cm = calculate_confusion_matrix(gen, imp, threshold=0.50)
    assert cm["tp"] == 2
    assert cm["fn"] == 1
    assert cm["tn"] == 3
    assert cm["fp"] == 1


def test_classification_metrics_full_suite() -> None:
    """Verifies Precision, Recall, Accuracy, and F1 calculations."""
    gen = np.array([0.30, 0.40])       # TP=2, FN=0
    imp = np.array([0.50, 0.80, 0.90]) # at tau=0.60: 0.50 <= 0.60 (FP=1), TN=2

    m = calculate_metrics_at_threshold(gen, imp, threshold=0.60)
    assert m["tp"] == 2
    assert m["fn"] == 0
    assert m["fp"] == 1
    assert m["tn"] == 2

    # Accuracy = (2 + 2) / 5 = 0.80
    assert np.isclose(m["accuracy"], 0.80)

    # Precision = 2 / (2 + 1) = 2/3 = 0.6667
    assert np.isclose(m["precision"], 2 / 3, atol=1e-3)

    # Recall = 2 / (2 + 0) = 1.0
    assert np.isclose(m["recall"], 1.0)

    # F1 = 2 * ( (2/3) * 1.0 ) / ( 2/3 + 1.0 ) = (4/3) / (5/3) = 4/5 = 0.80
    assert np.isclose(m["f1_score"], 0.80)


# ── Edge Case Tests ───────────────────────────────────────────────────────────

def test_metrics_empty_arrays_handle_cleanly() -> None:
    """Verifies that empty distance arrays return safe 0.0 values without throwing errors."""
    empty = np.array([], dtype=np.float64)
    assert calculate_far(empty, threshold=0.6) == 0.0
    assert calculate_frr(empty, threshold=0.6) == 0.0

    m = calculate_metrics_at_threshold(empty, empty, threshold=0.6)
    assert m["accuracy"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1_score"] == 0.0


def test_metrics_no_positive_predictions_zero_division_guard() -> None:
    """Verifies division-by-zero guard when TP + FP == 0 (threshold too small)."""
    gen = np.array([0.80, 0.90])
    imp = np.array([0.85, 0.95])
    # Threshold 0.10: TP=0, FP=0
    m = calculate_metrics_at_threshold(gen, imp, threshold=0.10)
    assert m["precision"] == 0.0
    assert m["f1_score"] == 0.0


# ── Threshold Sweep & EER Tests ───────────────────────────────────────────────

def test_evaluate_threshold_sweep() -> None:
    """Verifies threshold sweep table generation and EER detection."""
    # Clearly separated distributions
    gen = np.array([0.25, 0.30, 0.35, 0.40], dtype=np.float64)
    imp = np.array([0.70, 0.75, 0.80, 0.85], dtype=np.float64)

    sweep = evaluate_threshold_sweep(
        genuine_distances=gen,
        impostor_distances=imp,
        min_threshold=0.20,
        max_threshold=0.90,
        step=0.05,
    )

    assert "sweep_table" in sweep
    assert len(sweep["sweep_table"]) > 0
    assert "eer_threshold" in sweep
    assert "auc_roc" in sweep

    # Perfect separation means best accuracy should be 1.0 at intermediate thresholds
    assert sweep["best_accuracy"] == 1.0
    assert 0.40 <= sweep["best_accuracy_threshold"] <= 0.70


# ── Dataset Abstraction Tests ─────────────────────────────────────────────────

def test_evaluation_dataset_methods() -> None:
    """Verifies EvaluationDataset pair storage and distance extraction."""
    v1 = np.ones(128, dtype=np.float64)
    v2 = np.ones(128, dtype=np.float64)
    v3 = np.zeros(128, dtype=np.float64)

    pair_gen = EvaluationPair(v1, v2, is_same_identity=True, subject_id_1="A", subject_id_2="A")
    pair_imp = EvaluationPair(v1, v3, is_same_identity=False, subject_id_1="A", subject_id_2="B")

    dataset = EvaluationDataset(
        genuine_pairs=[pair_gen],
        impostor_pairs=[pair_imp],
        is_synthetic=True,
    )

    assert dataset.total_pairs == 2
    assert len(dataset.get_genuine_distances()) == 1
    assert len(dataset.get_impostor_distances()) == 1
    assert dataset.get_genuine_distances()[0] == 0.0
    assert np.isclose(dataset.get_impostor_distances()[0], np.sqrt(128))


def test_generate_synthetic_evaluation_dataset() -> None:
    """Verifies that generate_synthetic_evaluation_dataset produces valid normalized pairs."""
    dataset = generate_synthetic_evaluation_dataset(
        num_subjects=5,
        samples_per_subject=4,
        seed=100,
    )
    assert dataset.is_synthetic is True
    # Genuine pairs: 5 subjects * (4*3/2) = 5 * 6 = 30 pairs
    assert len(dataset.genuine_pairs) == 30
    # Impostor pairs: 5*4/2 = 10 pairs
    assert len(dataset.impostor_pairs) == 10

    gen_dists = dataset.get_genuine_distances()
    imp_dists = dataset.get_impostor_distances()

    assert np.all(gen_dists >= 0.0)
    assert np.all(imp_dists >= 0.0)
    # Average genuine distance should be significantly lower than impostor distance
    assert np.mean(gen_dists) < np.mean(imp_dists)


# ── Plotting Utility Tests ────────────────────────────────────────────────────

def test_plots_generate_and_save_png(tmp_path: Path) -> None:
    """Verifies that all Matplotlib figures generate and write valid PNG files headlessly."""
    gen = np.array([0.30, 0.35, 0.40, 0.45], dtype=np.float64)
    imp = np.array([1.10, 1.20, 1.30, 1.40], dtype=np.float64)

    sweep = evaluate_threshold_sweep(gen, imp, min_threshold=0.2, max_threshold=0.8, step=0.05)

    plot_dir = tmp_path / "evaluation_plots"

    dist_file = plot_distance_distributions(gen, imp, threshold=0.6, output_path=plot_dir / "dist.png")
    assert Path(dist_file).exists()
    assert Path(dist_file).stat().st_size > 1000  # Valid PNG image

    far_file = plot_far_frr_curve(sweep, output_path=plot_dir / "far_frr.png")
    assert Path(far_file).exists()
    assert Path(far_file).stat().st_size > 1000

    roc_file = plot_roc_curve(sweep, output_path=plot_dir / "roc.png")
    assert Path(roc_file).exists()
    assert Path(roc_file).stat().st_size > 1000

    all_plots = generate_all_plots(sweep, gen, imp, output_dir=plot_dir)
    assert len(all_plots) == 3
    for p in all_plots.values():
        assert Path(p).exists()
