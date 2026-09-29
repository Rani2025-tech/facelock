"""Biometric evaluation metrics, dataset abstractions, and threshold sweep analytics.

Implements standard biometric verification metrics:
- False Accept Rate (FAR) / False Match Rate (FMR)
- False Reject Rate (FRR) / False Non-Match Rate (FNMR)
- Equal Error Rate (EER) and Operating Point Identification
- Precision, Recall (True Accept Rate), F1 Score, and Classification Accuracy
- Synthetic benchmark generation for reproducible offline test validation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config import EXPECTED_EMBEDDING_SHAPE, MATCH_THRESHOLD


@dataclass
class EvaluationPair:
    """A single comparison pair between two 128-dimensional biometric embeddings.

    Attributes:
        embedding_1: First 128-D vector.
        embedding_2: Second 128-D vector.
        is_same_identity: True for genuine (intra-subject) pair, False for impostor.
        subject_id_1: Identifier for first subject.
        subject_id_2: Identifier for second subject.
        distance: Computed Euclidean (L2) distance between embedding_1 and embedding_2.
    """

    embedding_1: np.ndarray
    embedding_2: np.ndarray
    is_same_identity: bool
    subject_id_1: str = ""
    subject_id_2: str = ""
    distance: float = 0.0

    def __post_init__(self) -> None:
        if self.distance == 0.0 and self.embedding_1 is not None and self.embedding_2 is not None:
            self.distance = float(np.linalg.norm(self.embedding_1 - self.embedding_2))


@dataclass
class EvaluationDataset:
    """Container holding genuine and impostor comparison pairs for biometric evaluation.

    Attributes:
        genuine_pairs: List of genuine comparison pairs (same subject).
        impostor_pairs: List of impostor comparison pairs (different subjects).
        is_synthetic: Flag indicating whether embeddings are synthetically simulated.
        dataset_name: Human-readable identifier for the dataset.
    """

    genuine_pairs: List[EvaluationPair] = field(default_factory=list)
    impostor_pairs: List[EvaluationPair] = field(default_factory=list)
    is_synthetic: bool = False
    dataset_name: str = "Unspecified Dataset"

    def get_genuine_distances(self) -> np.ndarray:
        """Returns 1-D array of Euclidean distances for all genuine pairs."""
        if not self.genuine_pairs:
            return np.array([], dtype=np.float64)
        return np.array([p.distance for p in self.genuine_pairs], dtype=np.float64)

    def get_impostor_distances(self) -> np.ndarray:
        """Returns 1-D array of Euclidean distances for all impostor pairs."""
        if not self.impostor_pairs:
            return np.array([], dtype=np.float64)
        return np.array([p.distance for p in self.impostor_pairs], dtype=np.float64)

    @property
    def total_pairs(self) -> int:
        """Total count of comparison pairs."""
        return len(self.genuine_pairs) + len(self.impostor_pairs)


# ── Metric Formulations ───────────────────────────────────────────────────────

def calculate_far(impostor_distances: np.ndarray, threshold: float) -> float:
    r"""Calculates the False Accept Rate (FAR) for a given distance threshold.

    Mathematical Definition:
        FAR(\tau) = \frac{|\{ d \in \mathcal{D}_{\text{impostor}} \mid d \le \tau \}|}{|\mathcal{D}_{\text{impostor}}|}

    FAR measures the probability that an unauthorized impostor is incorrectly
    authenticated as the registered user (security vulnerability).

    Args:
        impostor_distances: Array of Euclidean distances from impostor comparisons.
        threshold: Decision boundary distance \tau.

    Returns:
        A float in [0.0, 1.0] representing FAR. Returns 0.0 if impostor array is empty.
    """
    if impostor_distances is None or len(impostor_distances) == 0:
        return 0.0
    false_accepts = int(np.sum(impostor_distances <= threshold))
    return float(false_accepts / len(impostor_distances))


def calculate_frr(genuine_distances: np.ndarray, threshold: float) -> float:
    r"""Calculates the False Reject Rate (FRR) for a given distance threshold.

    Mathematical Definition:
        FRR(\tau) = \frac{|\{ d \in \mathcal{D}_{\text{genuine}} \mid d > \tau \}|}{|\mathcal{D}_{\text{genuine}}|}

    FRR measures the probability that a legitimate genuine user is incorrectly
    rejected (usability friction).

    Args:
        genuine_distances: Array of Euclidean distances from genuine comparisons.
        threshold: Decision boundary distance \tau.

    Returns:
        A float in [0.0, 1.0] representing FRR. Returns 0.0 if genuine array is empty.
    """
    if genuine_distances is None or len(genuine_distances) == 0:
        return 0.0
    false_rejects = int(np.sum(genuine_distances > threshold))
    return float(false_rejects / len(genuine_distances))


def calculate_confusion_matrix(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    threshold: float,
) -> Dict[str, int]:
    r"""Computes True Positives, False Negatives, True Negatives, and False Positives.

    Binary Classification Mapping:
        - Positive Class: Genuine Match (d <= threshold)
        - Negative Class: Impostor Rejection (d > threshold)

        TP: Genuine comparison where d <= threshold (Correctly Granted)
        FN: Genuine comparison where d > threshold (Falsely Rejected)
        TN: Impostor comparison where d > threshold (Correctly Blocked)
        FP: Impostor comparison where d <= threshold (Falsely Granted)

    Args:
        genuine_distances: 1-D array of genuine Euclidean distances.
        impostor_distances: 1-D array of impostor Euclidean distances.
        threshold: Matching threshold distance.

    Returns:
        Dictionary with keys 'tp', 'fn', 'tn', 'fp'.
    """
    gen = np.asarray(genuine_distances, dtype=np.float64)
    imp = np.asarray(impostor_distances, dtype=np.float64)

    tp = int(np.sum(gen <= threshold)) if len(gen) > 0 else 0
    fn = int(np.sum(gen > threshold)) if len(gen) > 0 else 0
    tn = int(np.sum(imp > threshold)) if len(imp) > 0 else 0
    fp = int(np.sum(imp <= threshold)) if len(imp) > 0 else 0

    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def calculate_metrics_at_threshold(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    r"""Computes full classification and biometric error metrics at a specific threshold.

    Computes:
        - FAR = FP / (FP + TN)
        - FRR = FN / (TP + FN)
        - Accuracy = (TP + TN) / (TP + TN + FP + FN)
        - Precision = TP / (TP + FP)
        - Recall (TAR) = TP / (TP + FN) = 1.0 - FRR
        - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        genuine_distances: 1-D array of genuine Euclidean distances.
        impostor_distances: 1-D array of impostor Euclidean distances.
        threshold: Distance threshold to evaluate.

    Returns:
        Dictionary containing metric scalars.
    """
    cm = calculate_confusion_matrix(genuine_distances, impostor_distances, threshold)
    tp, fn, tn, fp = cm["tp"], cm["fn"], cm["tn"], cm["fp"]
    total = tp + fn + tn + fp

    far = calculate_far(impostor_distances, threshold)
    frr = calculate_frr(genuine_distances, threshold)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Accept Rate (TAR)
    f1_score = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": float(round(threshold, 4)),
        "far": float(round(far, 4)),
        "frr": float(round(frr, 4)),
        "accuracy": float(round(accuracy, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1_score": float(round(f1_score, 4)),
        "tp": float(tp),
        "fn": float(fn),
        "tn": float(tn),
        "fp": float(fp),
    }


def evaluate_threshold_sweep(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    min_threshold: float = 0.20,
    max_threshold: float = 0.80,
    step: float = 0.01,
) -> Dict[str, Any]:
    """Sweeps thresholds across a range to identify EER and optimal operational points.

    Equal Error Rate (EER):
        The operating threshold where FAR equals FRR:
        EER = \text{argmin}_\tau |\text{FAR}(\tau) - \text{FRR}(\tau)|

    Args:
        genuine_distances: 1-D array of genuine distances.
        impostor_distances: 1-D array of impostor distances.
        min_threshold: Starting threshold (default: 0.20).
        max_threshold: Ending threshold (default: 0.80).
        step: Increment step (default: 0.01).

    Returns:
        Dictionary containing sweep table, EER threshold, best accuracy, and best F1.
    """
    thresholds = np.arange(min_threshold, max_threshold + (step / 2.0), step)
    sweep_table: List[Dict[str, float]] = []

    best_eer_diff = float("inf")
    eer_threshold = float(MATCH_THRESHOLD)
    eer_value = 0.0

    best_acc = -1.0
    best_acc_threshold = float(MATCH_THRESHOLD)

    best_f1 = -1.0
    best_f1_threshold = float(MATCH_THRESHOLD)

    for thresh in thresholds:
        m = calculate_metrics_at_threshold(genuine_distances, impostor_distances, float(thresh))
        sweep_table.append(m)

        # Track EER (|FAR - FRR| min)
        diff = abs(m["far"] - m["frr"])
        if diff < best_eer_diff:
            best_eer_diff = diff
            eer_threshold = m["threshold"]
            eer_value = (m["far"] + m["frr"]) / 2.0

        # Track Best Accuracy
        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_acc_threshold = m["threshold"]

        # Track Best F1
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_f1_threshold = m["threshold"]

    # Compute Approximate Area Under ROC Curve (AUC) using trapezoidal rule
    # Sort points by FAR ascending
    sorted_by_far = sorted(sweep_table, key=lambda x: x["far"])
    far_vals = [pt["far"] for pt in sorted_by_far]
    tar_vals = [pt["recall"] for pt in sorted_by_far]  # TAR = Recall

    # Ensure boundaries (0,0) and (1,1) for AUC estimation
    if far_vals[0] > 0.0:
        far_vals.insert(0, 0.0)
        tar_vals.insert(0, 0.0)
    if far_vals[-1] < 1.0:
        far_vals.append(1.0)
        tar_vals.append(1.0)

    auc_roc = float(np.trapezoid(tar_vals, far_vals))

    default_metrics = calculate_metrics_at_threshold(
        genuine_distances, impostor_distances, float(MATCH_THRESHOLD)
    )

    return {
        "sweep_table": sweep_table,
        "eer_threshold": float(round(eer_threshold, 4)),
        "eer_value": float(round(eer_value, 4)),
        "best_accuracy_threshold": float(round(best_acc_threshold, 4)),
        "best_accuracy": float(round(best_acc, 4)),
        "best_f1_threshold": float(round(best_f1_threshold, 4)),
        "best_f1": float(round(best_f1, 4)),
        "auc_roc": float(round(auc_roc, 4)),
        "default_metrics": default_metrics,
        "genuine_count": len(genuine_distances),
        "impostor_count": len(impostor_distances),
    }


# ── Synthetic Dataset Generation ──────────────────────────────────────────────

def generate_synthetic_evaluation_dataset(
    num_subjects: int = 30,
    samples_per_subject: int = 6,
    intra_class_noise: float = 0.035,
    seed: int = 42,
) -> EvaluationDataset:
    r"""Generates a reproducible synthetic benchmark dataset for offline metric verification.

    DISCLAIMER:
        This dataset uses mathematically simulated 128-D Gaussian embeddings normalized
        to unit L2 norm. It serves STRICTLY to validate evaluation pipelines, unit tests,
        and metric sweep algorithms without storing private personal face imagery.
        It does NOT represent real-world physical camera biometric distributions.

    Args:
        num_subjects: Number of distinct identities (default: 30).
        samples_per_subject: Embeddings generated per identity (default: 6).
        intra_class_noise: Gaussian std added to simulate same-identity variance (default: 0.08).
        seed: Random seed for deterministic reproducibility.

    Returns:
        An EvaluationDataset populated with genuine and impostor EvaluationPairs.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate base centroid vectors for each identity (far apart on hypersphere)
    base_centroids = {}
    subject_samples: Dict[str, List[np.ndarray]] = {}

    for s_idx in range(num_subjects):
        s_id = f"Subject_{s_idx + 1:03d}"
        raw_base = rng.standard_normal(128).astype(np.float64)
        base_norm = raw_base / np.linalg.norm(raw_base)
        base_centroids[s_id] = base_norm

        # Generate K samples per subject with intra-class perturbation
        samples = []
        for _ in range(samples_per_subject):
            noise = intra_class_noise * rng.standard_normal(128).astype(np.float64)
            perturbed = base_norm + noise
            normalized_sample = (perturbed / np.linalg.norm(perturbed)).astype(np.float64)
            samples.append(normalized_sample)
        subject_samples[s_id] = samples

    genuine_pairs: List[EvaluationPair] = []
    impostor_pairs: List[EvaluationPair] = []

    subject_ids = list(subject_samples.keys())

    # 2. Build Genuine Pairs: all (s_i, s_j) for the same subject
    for s_id, samples in subject_samples.items():
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                genuine_pairs.append(
                    EvaluationPair(
                        embedding_1=samples[i],
                        embedding_2=samples[j],
                        is_same_identity=True,
                        subject_id_1=s_id,
                        subject_id_2=s_id,
                    )
                )

    # 3. Build Impostor Pairs: pairs sampled across different subjects
    for i in range(len(subject_ids)):
        for j in range(i + 1, len(subject_ids)):
            s_a = subject_ids[i]
            s_b = subject_ids[j]
            # Pair primary samples across identities
            impostor_pairs.append(
                EvaluationPair(
                    embedding_1=subject_samples[s_a][0],
                    embedding_2=subject_samples[s_b][0],
                    is_same_identity=False,
                    subject_id_1=s_a,
                    subject_id_2=s_b,
                )
            )

    return EvaluationDataset(
        genuine_pairs=genuine_pairs,
        impostor_pairs=impostor_pairs,
        is_synthetic=True,
        dataset_name=f"Synthetic-Benchmark (N={num_subjects}, K={samples_per_subject}, Seed={seed})",
    )
