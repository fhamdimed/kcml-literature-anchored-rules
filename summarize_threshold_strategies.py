#!/usr/bin/env python3
"""Summarize common, per-lambda optimized, and fixed-threshold results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "balanced_accuracy",
    "recall_sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "patient_violation_rate",
    "soft_rule_violation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    root = Path(args.results_dir)

    mode_files = {
        "primary_common": root / "combined_selected_common_threshold_test_results.csv",
        "secondary_per_lambda_optimized": root
        / "combined_selected_optimized_threshold_test_results.csv",
        "sensitivity_fixed": root
        / "combined_selected_fixed_threshold_test_results.csv",
    }
    frames: list[pd.DataFrame] = []
    for mode, path in mode_files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "reporting_strategy", mode)
        frames.append(frame)
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(
        root / "combined_selected_operating_point_comparison.csv", index=False
    )

    common = pd.read_csv(root / "combined_common_threshold_all_lambda_results.csv")
    audit = pd.read_csv(root / "combined_lambda_selection_audit.csv")
    selected = audit.loc[audit["selected"].astype(bool), [
        "algorithm",
        "penalty_multiplier",
    ]].drop_duplicates()
    test = common.loc[common["split"].eq("test")].copy()
    selected_test = test.merge(
        selected,
        on=["algorithm", "penalty_multiplier"],
        how="inner",
    )
    baseline = test.loc[np.isclose(test["penalty_multiplier"], 0.0)].copy()
    paired = selected_test.merge(
        baseline,
        on="algorithm",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    for metric in METRICS:
        left = f"{metric}_selected"
        right = f"{metric}_baseline"
        if left in paired.columns and right in paired.columns:
            paired[f"delta_{metric}"] = paired[left] - paired[right]
    paired.to_csv(
        root / "combined_selected_vs_unpenalized_common_threshold.csv",
        index=False,
    )

    display = [
        "reporting_strategy",
        "algorithm",
        "penalty_multiplier",
        "threshold",
        *METRICS,
    ]
    display = [column for column in display if column in comparison.columns]
    print("\nSelected models across operating-point strategies")
    print(comparison[display].to_string(index=False))

    delta_columns = [
        "algorithm",
        "penalty_multiplier_selected",
        "threshold_selected",
        *[f"delta_{metric}" for metric in METRICS],
    ]
    delta_columns = [column for column in delta_columns if column in paired.columns]
    print("\nPrimary common-threshold: selected minus unpenalized")
    print(paired[delta_columns].to_string(index=False))


if __name__ == "__main__":
    main()
