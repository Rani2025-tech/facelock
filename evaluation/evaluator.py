"""CLI evaluator and empirical threshold analysis runner for FaceLock.

Usage:
    python -m evaluation.evaluator
    python -m evaluation.evaluator --min-threshold 0.30 --max-threshold 0.70 --step 0.01
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from config import MATCH_THRESHOLD
from evaluation.metrics import (
    EvaluationDataset,
    evaluate_threshold_sweep,
    generate_synthetic_evaluation_dataset,
)
from evaluation.plots import generate_all_plots


def run_evaluation(
    dataset: Optional[EvaluationDataset] = None,
    min_threshold: float = 0.20,
    max_threshold: float = 0.80,
    step: float = 0.01,
    output_dir: Optional[Path] = None,
    save_plots: bool = True,
) -> dict:
    """Executes a full biometric evaluation run and produces an analytical report.

    Args:
        dataset: EvaluationDataset instance. If None, generates a synthetic benchmark.
        min_threshold: Lower threshold boundary.
        max_threshold: Upper threshold boundary.
        step: Threshold step size.
        output_dir: Directory where PNG plots will be written.
        save_plots: Whether to render and save matplotlib figures.

    Returns:
        Dictionary containing the complete evaluation results.
    """
    if dataset is None:
        dataset = generate_synthetic_evaluation_dataset(
            num_subjects=35,
            samples_per_subject=6,
            intra_class_noise=0.035,
            seed=42,
        )

    genuine_dist = dataset.get_genuine_distances()
    impostor_dist = dataset.get_impostor_distances()

    sweep = evaluate_threshold_sweep(
        genuine_distances=genuine_dist,
        impostor_distances=impostor_dist,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        step=step,
    )

    plot_paths = {}
    if save_plots:
        plot_paths = generate_all_plots(
            sweep_results=sweep,
            genuine_distances=genuine_dist,
            impostor_distances=impostor_dist,
            output_dir=output_dir,
            is_synthetic=dataset.is_synthetic,
        )

    # Print Formatted Report to Console
    _print_evaluation_report(dataset, sweep, plot_paths)

    return {"sweep": sweep, "plot_paths": plot_paths, "dataset": dataset}


def _print_evaluation_report(dataset: EvaluationDataset, sweep: dict, plot_paths: dict) -> None:
    """Prints a structured ASCII report of the biometric evaluation results."""
    gen_dist = dataset.get_genuine_distances()
    imp_dist = dataset.get_impostor_distances()

    print("\n" + "=" * 78)
    print("      FACELOCK BIOMETRIC EVALUATION & THRESHOLD ANALYSIS REPORT")
    print("=" * 78)

    if dataset.is_synthetic:
        print("\n [!] NOTICE: EVALUATION CONDUCTED ON SYNTHETIC BENCHMARK DATASET")
        print("     Dataset: " + dataset.dataset_name)
        print("     (Synthetic Gaussian 128-D vectors used to validate metric math offline)\n")
    else:
        print("\n Dataset: " + dataset.dataset_name + "\n")

    print("-" * 78)
    print(" 1. DATASET STATISTICS")
    print("-" * 78)
    print(f" Total Genuine Pairs (Same Subject):    {len(gen_dist):>6d}")
    print(f" Total Impostor Pairs (Diff Subject):   {len(imp_dist):>6d}")
    print(f" Total Biometric Comparisons:           {len(gen_dist) + len(imp_dist):>6d}")
    print(f" Genuine Distance (Mean +/- Std):       {np.mean(gen_dist):.4f} +/- {np.std(gen_dist):.4f}  [Min: {np.min(gen_dist):.4f}, Max: {np.max(gen_dist):.4f}]")
    print(f" Impostor Distance (Mean +/- Std):      {np.mean(imp_dist):.4f} +/- {np.std(imp_dist):.4f}  [Min: {np.min(imp_dist):.4f}, Max: {np.max(imp_dist):.4f}]")

    print("\n" + "-" * 78)
    print(" 2. OPTIMAL OPERATING POINTS & EMPIRICAL THRESHOLDS")
    print("-" * 78)
    print(f" Equal Error Rate (EER) Threshold:      tau = {sweep['eer_threshold']:.4f}  (EER = {sweep['eer_value']*100:.2f}%)")
    print(f" Best Accuracy Threshold:               tau = {sweep['best_accuracy_threshold']:.4f}  (Accuracy = {sweep['best_accuracy']*100:.2f}%)")
    print(f" Best F1-Score Threshold:               tau = {sweep['best_f1_threshold']:.4f}  (F1 = {sweep['best_f1']:.4f})")
    print(f" ROC Area Under Curve (AUC):            AUC = {sweep['auc_roc']:.4f}")

    print("\n" + "-" * 78)
    print(f" 3. DEFAULT THRESHOLD BENCHMARK (tau = {MATCH_THRESHOLD:.2f})")
    print("-" * 78)
    def_m = sweep["default_metrics"]
    print(f" False Accept Rate (FAR):               {def_m['far']*100:>6.2f}%  (Security Risk: unauthorized access granted)")
    print(f" False Reject Rate (FRR):               {def_m['frr']*100:>6.2f}%  (Usability Friction: genuine user blocked)")
    print(f" Overall Verification Accuracy:         {def_m['accuracy']*100:>6.2f}%")
    print(f" Verification Precision:                {def_m['precision']*100:>6.2f}%")
    print(f" Verification Recall (TAR):             {def_m['recall']*100:>6.2f}%")
    print(f" F1-Score:                              {def_m['f1_score']:>8.4f}")
    print(f" Confusion Matrix:                      TP={int(def_m['tp'])}, FN={int(def_m['fn'])}, TN={int(def_m['tn'])}, FP={int(def_m['fp'])}")

    print("\n" + "-" * 78)
    print(" 4. THRESHOLD SWEEP EXCERPT")
    print("-" * 78)
    print(f" {'Threshold (tau)':<15} {'FAR (%)':<12} {'FRR (%)':<12} {'Accuracy (%)':<14} {'F1-Score':<10}")
    print(" " + "-" * 65)

    # Print evenly spaced rows from sweep table
    table = sweep["sweep_table"]
    sample_indices = np.linspace(0, len(table) - 1, min(9, len(table)), dtype=int)
    for idx in sample_indices:
        row = table[idx]
        marker = " <-- [EER]" if abs(row["threshold"] - sweep["eer_threshold"]) < 0.005 else ""
        print(f" {row['threshold']:<15.3f} {row['far']*100:<12.2f} {row['frr']*100:<12.2f} {row['accuracy']*100:<14.2f} {row['f1_score']:<10.4f}{marker}")

    if plot_paths:
        print("\n" + "-" * 78)
        print(" 5. GENERATED VISUALIZATION PLOTS")
        print("-" * 78)
        for name, path in plot_paths.items():
            print(f" - {name:<25}: {path}")

    print("=" * 78 + "\n")


def main() -> None:
    """CLI entry point for python -m evaluation.evaluator."""
    parser = argparse.ArgumentParser(
        description="FaceLock Biometric Threshold Evaluator and Metrics Suite"
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=0.20,
        help="Minimum distance threshold to evaluate (default: 0.20)",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=0.80,
        help="Maximum distance threshold to evaluate (default: 0.80)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Step size for threshold sweep (default: 0.01)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated plots (default: evaluation/plots/)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generating PNG plot files",
    )

    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    run_evaluation(
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        output_dir=out_path,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
