"""Visualization and plotting utilities for biometric evaluation analysis.

Generates publication-quality charts using Matplotlib:
1. Genuine vs. Impostor Distance Distributions (Histograms & KDE)
2. FAR & FRR vs. Threshold Curve (with EER operational point)
3. Receiver Operating Characteristic (ROC) Curve (TAR vs. FAR with AUC)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib

# Use headless backend to avoid GUI window popups during CLI execution and testing
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette constants
_COLOR_GENUINE = "#2ecc71"   # Emerald green
_COLOR_IMPOSTOR = "#e74c3c"  # Coral red
_COLOR_FAR = "#e74c3c"       # Red
_COLOR_FRR = "#2980b9"       # Blue
_COLOR_EER = "#8e44ad"       # Purple
_COLOR_THRESHOLD = "#f39c12" # Amber


def plot_distance_distributions(
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    threshold: float = 0.60,
    output_path: Optional[Union[str, Path]] = None,
    is_synthetic: bool = True,
) -> str:
    """Plots overlapping histogram distributions of genuine and impostor Euclidean distances.

    Args:
        genuine_distances: 1-D array of genuine distances.
        impostor_distances: 1-D array of impostor distances.
        threshold: Decision threshold to highlight (default: 0.60).
        output_path: Filepath where the plot image should be saved.
        is_synthetic: Whether to label the figure as synthetic benchmark data.

    Returns:
        Path to the saved image file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "plots" / "distance_distributions.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    # Bin genuine distances
    if len(genuine_distances) > 0:
        ax.hist(
            genuine_distances,
            bins=35,
            density=True,
            alpha=0.6,
            color=_COLOR_GENUINE,
            edgecolor="black",
            linewidth=0.5,
            label=f"Genuine Pairs (μ={np.mean(genuine_distances):.3f}, σ={np.std(genuine_distances):.3f})",
        )

    # Bin impostor distances
    if len(impostor_distances) > 0:
        ax.hist(
            impostor_distances,
            bins=35,
            density=True,
            alpha=0.6,
            color=_COLOR_IMPOSTOR,
            edgecolor="black",
            linewidth=0.5,
            label=f"Impostor Pairs (μ={np.mean(impostor_distances):.3f}, σ={np.std(impostor_distances):.3f})",
        )

    # Decision threshold line
    ax.axvline(
        x=threshold,
        color=_COLOR_THRESHOLD,
        linestyle="--",
        linewidth=2.0,
        label=f"Match Threshold (τ = {threshold:.2f})",
    )

    title = "Euclidean Distance Distribution: Genuine vs. Impostor Pairs"
    if is_synthetic:
        title += "\n[SYNTHETIC BENCHMARK DATASET]"

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Euclidean Distance (L2 Norm in 128-D Embedding Space)", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return str(output_path)


def plot_far_frr_curve(
    sweep_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    is_synthetic: bool = True,
) -> str:
    """Plots False Accept Rate (FAR) and False Reject Rate (FRR) vs. Threshold.

    Args:
        sweep_results: Dictionary output from evaluate_threshold_sweep().
        output_path: Destination path for plot.
        is_synthetic: Whether to label data as synthetic.

    Returns:
        Path to saved PNG file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "plots" / "far_frr_curve.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = sweep_results["sweep_table"]
    thresholds = [pt["threshold"] for pt in table]
    far_vals = [pt["far"] for pt in table]
    frr_vals = [pt["frr"] for pt in table]

    eer_thresh = sweep_results["eer_threshold"]
    eer_val = sweep_results["eer_value"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    ax.plot(thresholds, far_vals, color=_COLOR_FAR, linewidth=2.2, label="FAR (False Accept Rate)")
    ax.plot(thresholds, frr_vals, color=_COLOR_FRR, linewidth=2.2, label="FRR (False Reject Rate)")

    # EER intersection
    ax.axvline(
        x=eer_thresh,
        color=_COLOR_EER,
        linestyle=":",
        linewidth=2.0,
        label=f"EER Point (τ = {eer_thresh:.3f}, Error = {eer_val*100:.2f}%)",
    )
    ax.scatter([eer_thresh], [eer_val], color=_COLOR_EER, s=60, zorder=5)

    title = "FAR & FRR vs. Match Threshold Curve"
    if is_synthetic:
        title += "\n[SYNTHETIC BENCHMARK DATASET]"

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Distance Threshold (τ)", fontsize=10)
    ax.set_ylabel("Error Rate (0.0 – 1.0)", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return str(output_path)


def plot_roc_curve(
    sweep_results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    is_synthetic: bool = True,
) -> str:
    """Plots the Receiver Operating Characteristic (ROC) curve (TAR vs. FAR).

    Args:
        sweep_results: Dictionary output from evaluate_threshold_sweep().
        output_path: Destination path for plot.
        is_synthetic: Whether to label data as synthetic.

    Returns:
        Path to saved PNG file.
    """
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "plots" / "roc_curve.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = sweep_results["sweep_table"]
    # Sort points by FAR
    sorted_pts = sorted(table, key=lambda x: x["far"])
    far_vals = [pt["far"] for pt in sorted_pts]
    tar_vals = [pt["recall"] for pt in sorted_pts]  # TAR = Recall

    # Boundary extensions
    if far_vals[0] > 0.0:
        far_vals.insert(0, 0.0)
        tar_vals.insert(0, 0.0)
    if far_vals[-1] < 1.0:
        far_vals.append(1.0)
        tar_vals.append(1.0)

    auc_score = sweep_results.get("auc_roc", 1.0)

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    ax.plot(
        far_vals,
        tar_vals,
        color="#27ae60",
        linewidth=2.5,
        label=f"FaceLock Biometric ROC (AUC = {auc_score:.4f})",
    )
    # Random guess baseline
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.50)")

    title = "Receiver Operating Characteristic (ROC) Curve"
    if is_synthetic:
        title += "\n[SYNTHETIC BENCHMARK DATASET]"

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Accept Rate (FAR / False Positive Rate)", fontsize=10)
    ax.set_ylabel("True Accept Rate (TAR / Recall / 1 - FRR)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return str(output_path)


def generate_all_plots(
    sweep_results: Dict[str, Any],
    genuine_distances: np.ndarray,
    impostor_distances: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None,
    is_synthetic: bool = True,
) -> Dict[str, str]:
    """Generates and saves all three evaluation charts into the target directory.

    Args:
        sweep_results: Output from evaluate_threshold_sweep().
        genuine_distances: 1-D array of genuine distances.
        impostor_distances: 1-D array of impostor distances.
        output_dir: Folder to save PNG files (defaults to evaluation/plots/).
        is_synthetic: Whether data is synthetic.

    Returns:
        Dictionary mapping plot identifiers to file paths.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dist_path = plot_distance_distributions(
        genuine_distances,
        impostor_distances,
        threshold=sweep_results.get("eer_threshold", 0.60),
        output_path=output_dir / "distance_distributions.png",
        is_synthetic=is_synthetic,
    )

    far_frr_path = plot_far_frr_curve(
        sweep_results,
        output_path=output_dir / "far_frr_curve.png",
        is_synthetic=is_synthetic,
    )

    roc_path = plot_roc_curve(
        sweep_results,
        output_path=output_dir / "roc_curve.png",
        is_synthetic=is_synthetic,
    )

    return {
        "distance_distributions": dist_path,
        "far_frr_curve": far_frr_path,
        "roc_curve": roc_path,
    }
