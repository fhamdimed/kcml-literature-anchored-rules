#!/usr/bin/env python3
"""Post-hoc holdout analyses from saved KCML patient-level predictions.

This script does not retrain any model. It reads the patient-level test
predictions produced by ``run_all_algorithms.py`` and generates:

1. calibration summaries and per-algorithm calibration plots comparing the
   unpenalized and validation-selected models;
2. paired, class-stratified bootstrap confidence intervals for primary
   holdout metrics and selected-minus-unpenalized differences; and
3. a hypothetical confirmatory-referral summary in which every patient
   classified positive at the common threshold is assumed to be referred for
   molecular confirmation.

The referral calculation is an operating-point translation, not a clinical
utility analysis or recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kcml.metrics import calibration_intercept_slope


PREFERRED_ALGORITHM_ORDER = ["xgboost", "lightgbm", "logistic", "neural"]

BOOTSTRAP_METRICS = [
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
    "log_loss",
    "binary_violation_rate",
    "soft_rule_violation",
]


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return cleaned.lower() or "algorithm"


def discover_algorithm_dirs(results_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.is_dir()
        and (path / "selected_lambda_results.csv").exists()
        and (path / "predictions").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No algorithm result directories with saved predictions were found in {results_dir}. "
            "Expected subdirectories such as xgboost/, lightgbm/, logistic/, and neural/."
        )

    order = {name: i for i, name in enumerate(PREFERRED_ALGORITHM_ORDER)}
    return sorted(candidates, key=lambda p: (order.get(p.name, 999), p.name))


def selected_lambda_from_dir(algorithm_dir: Path) -> tuple[str, float, float]:
    selected = pd.read_csv(algorithm_dir / "selected_lambda_results.csv")
    if selected.empty:
        raise ValueError(f"Empty selected_lambda_results.csv in {algorithm_dir}")

    lambdas = selected["penalty_multiplier"].dropna().astype(float).unique()
    if len(lambdas) != 1:
        raise ValueError(
            f"Expected one selected lambda in {algorithm_dir}, found {lambdas.tolist()}"
        )
    selected_lambda = float(lambdas[0])

    test_rows = selected.loc[selected["split"].eq("test")]
    if len(test_rows) != 1:
        raise ValueError(
            f"Expected one selected test row in {algorithm_dir}, found {len(test_rows)}"
        )
    algorithm_name = str(test_rows.iloc[0]["algorithm"])
    common_threshold = float(test_rows.iloc[0]["threshold"])
    return algorithm_name, selected_lambda, common_threshold


def prediction_file(algorithm_dir: Path, penalty: float) -> Path:
    tag = f"{float(penalty):g}"
    matches = sorted((algorithm_dir / "predictions").glob(f"*_lambda_{tag}_test.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one test prediction file for lambda={tag} in "
            f"{algorithm_dir / 'predictions'}, found {len(matches)}"
        )
    return matches[0]


def validate_prediction_frame(df: pd.DataFrame, path: Path) -> None:
    required = {
        "patient_id",
        "true_label",
        "predicted_probability",
        "baseline_common_threshold",
        "predicted_label_common",
        "any_rule_violation_common",
        "soft_rule_violation_score",
        "active_rule_weight_patient",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def assert_paired(baseline: pd.DataFrame, selected: pd.DataFrame, algorithm: str) -> None:
    if len(baseline) != len(selected):
        raise ValueError(f"{algorithm}: baseline and selected test files have different lengths")

    if not np.array_equal(
        baseline["true_label"].to_numpy(), selected["true_label"].to_numpy()
    ):
        raise ValueError(f"{algorithm}: true-label order differs between baseline and selected")

    if not np.array_equal(
        baseline["patient_id"].astype(str).to_numpy(),
        selected["patient_id"].astype(str).to_numpy(),
    ):
        raise ValueError(f"{algorithm}: patient order differs between baseline and selected")


def weighted_soft_violation(df: pd.DataFrame, indices: np.ndarray | None = None) -> float:
    if indices is None:
        soft = df["soft_rule_violation_score"].to_numpy(dtype=float)
        weight = df["active_rule_weight_patient"].to_numpy(dtype=float)
    else:
        soft = df["soft_rule_violation_score"].to_numpy(dtype=float)[indices]
        weight = df["active_rule_weight_patient"].to_numpy(dtype=float)[indices]
    denominator = float(np.sum(weight))
    if denominator <= 0:
        return float("nan")
    return float(np.sum(soft * weight) / denominator)


def metric_vector(df: pd.DataFrame, indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        y = df["true_label"].to_numpy(dtype=int)
        p = df["predicted_probability"].to_numpy(dtype=float)
        pred = df["predicted_label_common"].to_numpy(dtype=int)
        violation = df["any_rule_violation_common"].to_numpy(dtype=float)
    else:
        y = df["true_label"].to_numpy(dtype=int)[indices]
        p = df["predicted_probability"].to_numpy(dtype=float)[indices]
        pred = df["predicted_label_common"].to_numpy(dtype=int)[indices]
        violation = df["any_rule_violation_common"].to_numpy(dtype=float)[indices]

    negative = y == 0
    positive = y == 1
    tn = int(np.sum((pred == 0) & negative))
    fp = int(np.sum((pred == 1) & negative))
    fn = int(np.sum((pred == 0) & positive))
    tp = int(np.sum((pred == 1) & positive))
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-8, 1 - 1e-8), labels=[0, 1])),
        "binary_violation_rate": float(np.mean(violation)),
        "soft_rule_violation": weighted_soft_violation(df, indices),
    }
    if np.unique(y).size == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["average_precision"] = float(average_precision_score(y, p))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")
    return result


def percentile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(finite, alpha / 2)),
        float(np.quantile(finite, 1 - alpha / 2)),
    )


def paired_stratified_bootstrap(
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    y = baseline["true_label"].to_numpy(dtype=int)
    negative_indices = np.flatnonzero(y == 0)
    positive_indices = np.flatnonzero(y == 1)
    if len(negative_indices) == 0 or len(positive_indices) == 0:
        raise ValueError("Both outcome classes are required for stratified bootstrap")

    rng = np.random.default_rng(seed)
    storage: dict[str, dict[str, list[float]]] = {
        metric: {"baseline": [], "selected": [], "delta": []}
        for metric in BOOTSTRAP_METRICS
    }

    for _ in range(n_bootstrap):
        sampled_negative = rng.choice(
            negative_indices, size=len(negative_indices), replace=True
        )
        sampled_positive = rng.choice(
            positive_indices, size=len(positive_indices), replace=True
        )
        sampled = np.concatenate([sampled_negative, sampled_positive])

        base_metrics = metric_vector(baseline, sampled)
        selected_metrics = metric_vector(selected, sampled)
        for metric in BOOTSTRAP_METRICS:
            base_value = base_metrics[metric]
            selected_value = selected_metrics[metric]
            storage[metric]["baseline"].append(base_value)
            storage[metric]["selected"].append(selected_value)
            storage[metric]["delta"].append(selected_value - base_value)

    return {
        metric: {
            role: np.asarray(values, dtype=float)
            for role, values in roles.items()
        }
        for metric, roles in storage.items()
    }


def calibration_table(
    df: pd.DataFrame,
    algorithm: str,
    role: str,
    penalty: float,
    bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = df["true_label"].to_numpy(dtype=int)
    p = np.clip(df["predicted_probability"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)
    intercept, slope = calibration_intercept_slope(y, p)

    summary = {
        "algorithm": algorithm,
        "model_role": role,
        "penalty_multiplier": float(penalty),
        "n_patients": int(len(df)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_index = np.digitize(p, edges[1:-1], right=False)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        mask = bin_index == index
        n_bin = int(np.sum(mask))
        rows.append(
            {
                "algorithm": algorithm,
                "model_role": role,
                "penalty_multiplier": float(penalty),
                "bin_index": index + 1,
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "n_bin": n_bin,
                "mean_predicted_probability": (
                    float(np.mean(p[mask])) if n_bin else float("nan")
                ),
                "observed_positive_fraction": (
                    float(np.mean(y[mask])) if n_bin else float("nan")
                ),
            }
        )
    return summary, pd.DataFrame(rows)


def save_calibration_plot(curves: pd.DataFrame, algorithm: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Ideal calibration")

    for role, group in curves.groupby("model_role", sort=False):
        valid = group["n_bin"].gt(0) & group["mean_predicted_probability"].notna()
        g = group.loc[valid]
        if g.empty:
            continue
        label = "Unpenalized" if role == "unpenalized" else "Selected"
        axis.plot(
            g["mean_predicted_probability"],
            g["observed_positive_fraction"],
            marker="o",
            linewidth=1.5,
            label=label,
        )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive fraction")
    axis.set_title(f"Calibration: {algorithm}")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def confusion_counts(df: pd.DataFrame) -> dict[str, int | float]:
    y = df["true_label"].to_numpy(dtype=int)
    pred = df["predicted_label_common"].to_numpy(dtype=int)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_positive": tp + fp,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def referral_row(
    algorithm: str,
    selected_lambda: float,
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    base = confusion_counts(baseline)
    sel = confusion_counts(selected)
    n = len(baseline)
    delta_referrals = int(sel["predicted_positive"] - base["predicted_positive"])
    delta_tp = int(sel["tp"] - base["tp"])
    delta_fp = int(sel["fp"] - base["fp"])

    return {
        "algorithm": algorithm,
        "selected_penalty_multiplier": float(selected_lambda),
        "n_test": int(n),
        "n_true_positive": int(np.sum(baseline["true_label"].to_numpy(dtype=int) == 1)),
        "n_true_negative": int(np.sum(baseline["true_label"].to_numpy(dtype=int) == 0)),
        "common_threshold": float(baseline["baseline_common_threshold"].iloc[0]),
        "baseline_referrals": int(base["predicted_positive"]),
        "selected_referrals": int(sel["predicted_positive"]),
        "additional_referrals": delta_referrals,
        "baseline_referrals_per_1000": float(base["predicted_positive"] / n * 1000),
        "selected_referrals_per_1000": float(sel["predicted_positive"] / n * 1000),
        "additional_referrals_per_1000": float(delta_referrals / n * 1000),
        "baseline_tp": int(base["tp"]),
        "selected_tp": int(sel["tp"]),
        "additional_true_positives": delta_tp,
        "baseline_fp": int(base["fp"]),
        "selected_fp": int(sel["fp"]),
        "additional_false_positives": delta_fp,
        "baseline_fn": int(base["fn"]),
        "selected_fn": int(sel["fn"]),
        "baseline_tn": int(base["tn"]),
        "selected_tn": int(sel["tn"]),
        "baseline_sensitivity": float(base["sensitivity"]),
        "selected_sensitivity": float(sel["sensitivity"]),
        "delta_sensitivity": float(sel["sensitivity"] - base["sensitivity"]),
        "baseline_specificity": float(base["specificity"]),
        "selected_specificity": float(sel["specificity"]),
        "delta_specificity": float(sel["specificity"] - base["specificity"]),
        "additional_referrals_per_additional_true_positive": (
            float(delta_referrals / delta_tp) if delta_tp > 0 else float("nan")
        ),
        "additional_false_positives_per_additional_true_positive": (
            float(delta_fp / delta_tp) if delta_tp > 0 else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analysis_outputs/common_threshold_robust"),
        help="Root directory produced by run_all_algorithms.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/holdout_analysis"),
        help="Directory for calibration, bootstrap, and referral outputs",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Number of paired class-stratified bootstrap replicates (default: 2000)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260905,
        help="Random seed for the paired bootstrap",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of fixed-width calibration bins (default: 10)",
    )
    args = parser.parse_args()

    if args.bootstrap_replicates < 200:
        raise ValueError("--bootstrap-replicates should be at least 200")
    if args.calibration_bins < 3:
        raise ValueError("--calibration-bins should be at least 3")
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    calibration_dir = args.output / "calibration_plots"
    calibration_dir.mkdir(parents=True, exist_ok=True)

    calibration_summaries: list[dict[str, Any]] = []
    calibration_curves: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    referral_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    algorithm_dirs = discover_algorithm_dirs(args.results_dir)

    for algorithm_index, algorithm_dir in enumerate(algorithm_dirs):
        algorithm_name, selected_lambda, selected_threshold = selected_lambda_from_dir(
            algorithm_dir
        )
        baseline_path = prediction_file(algorithm_dir, 0.0)
        selected_path = prediction_file(algorithm_dir, selected_lambda)
        baseline = pd.read_csv(baseline_path)
        selected = pd.read_csv(selected_path)
        validate_prediction_frame(baseline, baseline_path)
        validate_prediction_frame(selected, selected_path)
        assert_paired(baseline, selected, algorithm_name)

        base_thresholds = baseline["baseline_common_threshold"].astype(float).unique()
        selected_thresholds = selected["baseline_common_threshold"].astype(float).unique()
        if len(base_thresholds) != 1 or len(selected_thresholds) != 1:
            raise ValueError(f"{algorithm_name}: expected one common threshold per file")
        if not np.isclose(base_thresholds[0], selected_thresholds[0], atol=1e-12):
            raise ValueError(
                f"{algorithm_name}: selected and baseline files use different common thresholds"
            )
        if not np.isclose(base_thresholds[0], selected_threshold, atol=1e-12):
            raise ValueError(
                f"{algorithm_name}: prediction threshold does not match selected_lambda_results.csv"
            )

        source_records.append(
            {
                "algorithm": algorithm_name,
                "algorithm_directory": str(algorithm_dir),
                "baseline_prediction_file": str(baseline_path),
                "selected_prediction_file": str(selected_path),
                "selected_penalty_multiplier": selected_lambda,
                "common_threshold": float(base_thresholds[0]),
            }
        )

        base_summary, base_curve = calibration_table(
            baseline, algorithm_name, "unpenalized", 0.0, args.calibration_bins
        )
        calibration_summaries.append(base_summary)
        calibration_curves.append(base_curve)

        if np.isclose(selected_lambda, 0.0):
            selected_summary = dict(base_summary)
            selected_summary["model_role"] = "selected"
            selected_summary["penalty_multiplier"] = 0.0
            selected_curve = base_curve.copy()
            selected_curve["model_role"] = "selected"
            selected_curve["penalty_multiplier"] = 0.0
        else:
            selected_summary, selected_curve = calibration_table(
                selected,
                algorithm_name,
                "selected",
                selected_lambda,
                args.calibration_bins,
            )
        calibration_summaries.append(selected_summary)
        calibration_curves.append(selected_curve)

        curves_for_plot = pd.concat([base_curve, selected_curve], ignore_index=True)
        save_calibration_plot(
            curves_for_plot,
            algorithm_name,
            calibration_dir / f"calibration_{safe_name(algorithm_name)}.pdf",
        )

        point_baseline = metric_vector(baseline)
        point_selected = metric_vector(selected)
        bootstrap = paired_stratified_bootstrap(
            baseline,
            selected,
            n_bootstrap=args.bootstrap_replicates,
            seed=args.bootstrap_seed + algorithm_index,
        )
        for metric in BOOTSTRAP_METRICS:
            base_low, base_high = percentile_interval(bootstrap[metric]["baseline"])
            sel_low, sel_high = percentile_interval(bootstrap[metric]["selected"])
            delta_low, delta_high = percentile_interval(bootstrap[metric]["delta"])
            bootstrap_rows.append(
                {
                    "algorithm": algorithm_name,
                    "selected_penalty_multiplier": selected_lambda,
                    "metric": metric,
                    "baseline_estimate": point_baseline[metric],
                    "baseline_ci_low": base_low,
                    "baseline_ci_high": base_high,
                    "selected_estimate": point_selected[metric],
                    "selected_ci_low": sel_low,
                    "selected_ci_high": sel_high,
                    "delta_selected_minus_baseline": (
                        point_selected[metric] - point_baseline[metric]
                    ),
                    "delta_ci_low": delta_low,
                    "delta_ci_high": delta_high,
                    "bootstrap_replicates": args.bootstrap_replicates,
                }
            )

        referral_rows.append(
            referral_row(algorithm_name, selected_lambda, baseline, selected)
        )

        print(
            f"{algorithm_name}: selected lambda={selected_lambda:g}; "
            f"common threshold={base_thresholds[0]:.6g}; "
            f"bootstrap={args.bootstrap_replicates}",
            flush=True,
        )

    calibration_summary_df = pd.DataFrame(calibration_summaries)
    calibration_curves_df = pd.concat(calibration_curves, ignore_index=True)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    referral_df = pd.DataFrame(referral_rows)

    calibration_summary_df.to_csv(args.output / "calibration_summary.csv", index=False)
    calibration_curves_df.to_csv(args.output / "calibration_curves.csv", index=False)
    bootstrap_df.to_csv(args.output / "bootstrap_metric_intervals.csv", index=False)
    referral_df.to_csv(args.output / "confirmatory_referral_summary.csv", index=False)

    manifest = {
        "analysis": "holdout calibration, paired bootstrap uncertainty, and hypothetical confirmatory-referral translation",
        "results_dir": str(args.results_dir),
        "output_dir": str(args.output),
        "no_model_retraining": True,
        "evaluation_mode": "baseline-derived common threshold",
        "bootstrap": {
            "method": "paired patient-level bootstrap stratified by true outcome class",
            "replicates": args.bootstrap_replicates,
            "base_seed": args.bootstrap_seed,
            "confidence_interval": "percentile 95% interval",
            "metrics": BOOTSTRAP_METRICS,
        },
        "calibration": {
            "intercept_slope": "logistic recalibration of clipped predicted log-odds using kcml.metrics.calibration_intercept_slope",
            "curve_bins": args.calibration_bins,
            "curve_binning": "fixed-width probability bins over [0,1]",
        },
        "confirmatory_referral_interpretation": (
            "Hypothetical operating-point translation assuming every common-threshold "
            "predicted-positive classification is referred for molecular confirmation. "
            "This is not a clinical utility analysis or deployment recommendation."
        ),
        "algorithms": source_records,
    }
    (args.output / "holdout_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nWrote holdout analysis outputs to: {args.output}")
    print("  calibration_summary.csv")
    print("  calibration_curves.csv")
    print("  calibration_plots/*.pdf")
    print("  bootstrap_metric_intervals.csv")
    print("  confirmatory_referral_summary.csv")
    print("  holdout_analysis_manifest.json")


if __name__ == "__main__":
    main()
