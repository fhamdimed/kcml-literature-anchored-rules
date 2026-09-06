#!/usr/bin/env python3
"""Reviewer-response sensitivity analyses for KCML governance and rule weights.

This script adds two secondary analyses without changing the prespecified primary
analysis:

1. Safeguard sensitivity re-applies alternative validation eligibility criteria
   to already-trained all-lambda results. No model retraining is required.
2. Rule-weight sensitivity refits the all-rule models under alternative *relative*
   rule-weight schemes while keeping the split, model settings, lambda grid and
   primary governance policy unchanged.

The default configurations correspond to the sensitivity analyses proposed for
Reviewer 2, Major Issue 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kcml.runtime import configure_runtime, finish_cli

PROJECT_ROOT = Path(__file__).resolve().parent
configure_runtime(PROJECT_ROOT)

from kcml.data import load_cohort_data, make_holdout_splits
from kcml.experiment import run_penalty_grid, safe_name
from kcml.factories import (
    make_lightgbm_factory,
    make_logistic_factory,
    make_neural_factory,
    make_xgboost_factory,
)
from kcml.metrics import choose_penalty_constrained


PRIMARY_POLICY = {
    "balanced_accuracy_tolerance": 0.01,
    "auc_tolerance": 0.01,
    "average_precision_tolerance": 0.02,
    "minimum_violation_reduction": 0.40,
    "minimum_soft_violation_reduction": 0.0,
}

SAFEGUARD_SCENARIOS: tuple[tuple[str, dict[str, float]], ...] = (
    ("primary", {}),
    ("ba_strict_0.005", {"balanced_accuracy_tolerance": 0.005}),
    ("ba_permissive_0.020", {"balanced_accuracy_tolerance": 0.020}),
    ("auc_strict_0.005", {"auc_tolerance": 0.005}),
    ("auc_permissive_0.020", {"auc_tolerance": 0.020}),
    ("ap_strict_0.010", {"average_precision_tolerance": 0.010}),
    ("ap_permissive_0.030", {"average_precision_tolerance": 0.030}),
    ("violation_strict_0.50", {"minimum_violation_reduction": 0.50}),
    ("violation_permissive_0.30", {"minimum_violation_reduction": 0.30}),
)

WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "primary": {"LR01": 1.0, "LR02": 1.0, "LR03": 1.0, "LR04": 1.0, "LR05": 1.5},
    "equal": {"LR01": 1.0, "LR02": 1.0, "LR03": 1.0, "LR04": 1.0, "LR05": 1.0},
    "lr05_stronger": {"LR01": 1.0, "LR02": 1.0, "LR03": 1.0, "LR04": 1.0, "LR05": 2.0},
    "lr02_lr04_downweighted": {"LR01": 1.0, "LR02": 0.5, "LR03": 1.0, "LR04": 0.5, "LR05": 1.5},
}

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "logistic": "Logistic Regression",
    "neural": "Neural Network (MLP)",
}


def _policy_for(overrides: dict[str, float]) -> dict[str, float]:
    policy = dict(PRIMARY_POLICY)
    policy.update(overrides)
    return policy


def _select_test_row(
    all_lambda_rows: pd.DataFrame,
    algorithm: str,
    selected_lambda: float,
) -> pd.Series | None:
    rows = all_lambda_rows[
        (all_lambda_rows["algorithm"] == algorithm)
        & (all_lambda_rows["split"] == "test")
        & np.isclose(all_lambda_rows["penalty_multiplier"], selected_lambda)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _baseline_test_row(all_lambda_rows: pd.DataFrame, algorithm: str) -> pd.Series | None:
    return _select_test_row(all_lambda_rows, algorithm, 0.0)


def _delta(selected: pd.Series | None, baseline: pd.Series | None, column: str) -> float:
    if selected is None or baseline is None or column not in selected or column not in baseline:
        return float("nan")
    try:
        return float(selected[column]) - float(baseline[column])
    except Exception:
        return float("nan")


def safeguard_sensitivity_for_table(all_lambda_rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "algorithm",
        "penalty_multiplier",
        "split",
        "balanced_accuracy",
        "patient_violation_rate",
        "roc_auc",
        "average_precision",
        "soft_rule_violation",
    }
    missing = required - set(all_lambda_rows.columns)
    if missing:
        raise ValueError(f"All-lambda result table is missing columns: {sorted(missing)}")

    output_rows: list[dict[str, object]] = []
    algorithms = list(dict.fromkeys(all_lambda_rows["algorithm"].astype(str)))

    for algorithm in algorithms:
        algorithm_rows = all_lambda_rows[all_lambda_rows["algorithm"] == algorithm]
        validation = algorithm_rows[algorithm_rows["split"] == "validation"].copy()
        baseline_test = _baseline_test_row(all_lambda_rows, algorithm)

        if validation.empty:
            continue

        for scenario_name, overrides in SAFEGUARD_SCENARIOS:
            policy = _policy_for(overrides)
            selected_lambda, audit, reason = choose_penalty_constrained(
                validation,
                balanced_accuracy_tolerance=policy["balanced_accuracy_tolerance"],
                auc_tolerance=policy["auc_tolerance"],
                average_precision_tolerance=policy["average_precision_tolerance"],
                minimum_violation_reduction=policy["minimum_violation_reduction"],
                minimum_soft_violation_reduction=policy[
                    "minimum_soft_violation_reduction"
                ],
            )
            selected_test = _select_test_row(all_lambda_rows, algorithm, selected_lambda)
            selected_validation = audit.loc[
                np.isclose(audit["penalty_multiplier"], selected_lambda)
            ].iloc[0]

            row: dict[str, object] = {
                "algorithm": algorithm,
                "scenario": scenario_name,
                "selected_lambda": selected_lambda,
                **policy,
                "selection_reason": reason,
                "validation_balanced_accuracy": float(
                    selected_validation["balanced_accuracy"]
                ),
                "validation_roc_auc": float(selected_validation["roc_auc"]),
                "validation_average_precision": float(
                    selected_validation["average_precision"]
                ),
                "validation_patient_violation_rate": float(
                    selected_validation["patient_violation_rate"]
                ),
                "validation_soft_rule_violation": float(
                    selected_validation["soft_rule_violation"]
                ),
            }

            metric_map = {
                "balanced_accuracy": "balanced_accuracy",
                "sensitivity": "recall_sensitivity",
                "specificity": "specificity",
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
                "patient_violation_rate": "patient_violation_rate",
                "soft_rule_violation": "soft_rule_violation",
                "log_loss": "log_loss",
                "brier_score": "brier_score",
            }
            for output_name, source_name in metric_map.items():
                if selected_test is not None and source_name in selected_test:
                    row[f"test_{output_name}"] = float(selected_test[source_name])
                    row[f"test_delta_{output_name}"] = _delta(
                        selected_test, baseline_test, source_name
                    )

            output_rows.append(row)

    return pd.DataFrame(output_rows)


def run_safeguard_sensitivity(
    primary_results_dir: Path,
    output_dir: Path,
    cv_root: Path | None = None,
) -> None:
    primary_file = primary_results_dir / "combined_common_threshold_all_lambda_results.csv"
    if not primary_file.exists():
        raise FileNotFoundError(
            f"Primary all-lambda result file not found: {primary_file}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(primary_file)
    primary_summary = safeguard_sensitivity_for_table(primary)
    primary_summary.to_csv(output_dir / "safeguard_sensitivity_holdout.csv", index=False)

    if cv_root is not None:
        cv_rows: list[pd.DataFrame] = []
        for algorithm_key in ("xgboost", "lightgbm", "logistic", "neural"):
            algorithm_root = cv_root / algorithm_key
            if not algorithm_root.exists():
                continue
            for fold_dir in sorted(algorithm_root.glob("fold_*")):
                result_file = fold_dir / "common_threshold_all_lambda_results.csv"
                if not result_file.exists():
                    continue
                fold_summary = safeguard_sensitivity_for_table(pd.read_csv(result_file))
                fold_summary.insert(0, "algorithm_key", algorithm_key)
                try:
                    fold_number = int(fold_dir.name.split("_")[-1])
                except ValueError:
                    fold_number = fold_dir.name
                fold_summary.insert(1, "fold", fold_number)
                cv_rows.append(fold_summary)

        if cv_rows:
            cv = pd.concat(cv_rows, ignore_index=True)
            cv.to_csv(output_dir / "safeguard_sensitivity_cv_folds.csv", index=False)
            frequency = (
                cv.groupby(["algorithm_key", "algorithm", "scenario", "selected_lambda"])
                .size()
                .rename("n_folds")
                .reset_index()
            )
            frequency.to_csv(
                output_dir / "safeguard_sensitivity_cv_selection_frequency.csv",
                index=False,
            )

    metadata = {
        "analysis": "reviewer safeguard sensitivity",
        "primary_results_dir": str(primary_results_dir),
        "cv_root": None if cv_root is None else str(cv_root),
        "primary_policy": PRIMARY_POLICY,
        "scenarios": [
            {"name": name, **_policy_for(overrides)}
            for name, overrides in SAFEGUARD_SCENARIOS
        ],
        "note": (
            "Each sensitivity scenario changes one governance criterion at a time; "
            "the underlying trained models and predictions are unchanged."
        ),
    }
    (output_dir / "safeguard_sensitivity_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _build_factory(
    algorithm_key: str,
    rule_weights: dict[str, float],
    seed: int,
    tree_rounds: int,
    tree_learning_rate: float,
    tree_early_stopping: int,
    nn_max_epochs: int,
    nn_patience: int,
    nn_restarts: int,
    nn_device: str,
    quiet: bool,
):
    common = dict(
        enabled_rules=None,
        rule_weights=rule_weights,
        rule_control="none",
        random_state=seed,
        verbose=not quiet,
    )
    if algorithm_key == "xgboost":
        return make_xgboost_factory(
            n_estimators=tree_rounds,
            learning_rate=tree_learning_rate,
            early_stopping_rounds=tree_early_stopping,
            **common,
        )
    if algorithm_key == "lightgbm":
        return make_lightgbm_factory(
            n_estimators=tree_rounds,
            learning_rate=tree_learning_rate,
            early_stopping_rounds=tree_early_stopping,
            **common,
        )
    if algorithm_key == "logistic":
        return make_logistic_factory(**common)
    if algorithm_key == "neural":
        return make_neural_factory(
            architecture="mlp",
            hidden_sizes=(32, 16),
            dropout=0.10,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            max_epochs=nn_max_epochs,
            patience=nn_patience,
            n_restarts=nn_restarts,
            restart_aggregation="mean_probability",
            deterministic=True,
            device=nn_device,
            **common,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm_key}")


def run_weight_sensitivity(
    data_path: Path,
    output_dir: Path,
    algorithms: Iterable[str],
    lambdas: Iterable[float],
    seed: int,
    tree_rounds: int,
    tree_learning_rate: float,
    tree_early_stopping: int,
    nn_max_epochs: int,
    nn_patience: int,
    nn_restarts: int,
    nn_device: str,
    quiet: bool,
    schemes: Iterable[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort_data(data_path, feature_order=("MCV", "MCH", "HBA2", "HBF"))
    splits = make_holdout_splits(
        cohort,
        test_size=0.20,
        validation_fraction_of_total=0.20,
        random_state=seed,
    )
    splits.assignments().to_csv(output_dir / "shared_split_assignments.csv", index=False)

    selected_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []

    for scheme_name in schemes:
        if scheme_name not in WEIGHT_SCHEMES:
            raise ValueError(
                f"Unknown weight scheme {scheme_name!r}; choose from {sorted(WEIGHT_SCHEMES)}"
            )
        weights = WEIGHT_SCHEMES[scheme_name]
        for algorithm_key in algorithms:
            factory = _build_factory(
                algorithm_key,
                weights,
                seed,
                tree_rounds,
                tree_learning_rate,
                tree_early_stopping,
                nn_max_epochs,
                nn_patience,
                nn_restarts,
                nn_device,
                quiet,
            )
            result_dir = output_dir / scheme_name / algorithm_key
            result = run_penalty_grid(
                algorithm_name=DISPLAY_NAMES[algorithm_key],
                model_factory=factory,
                splits=splits,
                output_dir=result_dir,
                penalty_multipliers=lambdas,
                enabled_rules=None,
                rule_weights=weights,
                threshold=0.5,
                selection_tolerance=PRIMARY_POLICY["balanced_accuracy_tolerance"],
                selection_auc_tolerance=PRIMARY_POLICY["auc_tolerance"],
                selection_ap_tolerance=PRIMARY_POLICY["average_precision_tolerance"],
                minimum_violation_reduction=PRIMARY_POLICY[
                    "minimum_violation_reduction"
                ],
                minimum_soft_violation_reduction=PRIMARY_POLICY[
                    "minimum_soft_violation_reduction"
                ],
                save_models=False,
                save_predictions=True,
            )
            selected = dict(result["selected_common_test"])
            selected.update(
                {
                    "weight_scheme": scheme_name,
                    "algorithm_key": algorithm_key,
                    "selected_lambda": float(result["selected_lambda"]),
                    **{f"weight_{k}": v for k, v in weights.items()},
                }
            )
            selected_rows.append(selected)

            common = result["common_threshold_results"]
            baseline = common[
                (common["split"] == "test")
                & np.isclose(common["penalty_multiplier"], 0.0)
            ].iloc[0]
            selected_test = common[
                (common["split"] == "test")
                & np.isclose(
                    common["penalty_multiplier"], float(result["selected_lambda"])
                )
            ].iloc[0]
            delta_row: dict[str, object] = {
                "weight_scheme": scheme_name,
                "algorithm_key": algorithm_key,
                "algorithm": DISPLAY_NAMES[algorithm_key],
                "selected_lambda": float(result["selected_lambda"]),
            }
            metric_map = {
                "balanced_accuracy": "balanced_accuracy",
                "sensitivity": "recall_sensitivity",
                "specificity": "specificity",
                "roc_auc": "roc_auc",
                "average_precision": "average_precision",
                "patient_violation_rate": "patient_violation_rate",
                "soft_rule_violation": "soft_rule_violation",
                "log_loss": "log_loss",
                "brier_score": "brier_score",
            }
            for output_name, source_name in metric_map.items():
                if source_name in common.columns:
                    delta_row[f"delta_{output_name}"] = float(selected_test[source_name]) - float(
                        baseline[source_name]
                    )
            delta_rows.append(delta_row)

    pd.DataFrame(selected_rows).to_csv(
        output_dir / "rule_weight_sensitivity_selected_test.csv", index=False
    )
    pd.DataFrame(delta_rows).to_csv(
        output_dir / "rule_weight_sensitivity_deltas.csv", index=False
    )
    manifest = {
        "analysis": "reviewer rule-weight sensitivity",
        "data": str(data_path),
        "algorithms": list(algorithms),
        "lambda_grid": [float(x) for x in lambdas],
        "seed": seed,
        "weight_schemes": {name: WEIGHT_SCHEMES[name] for name in schemes},
        "primary_governance_policy": PRIMARY_POLICY,
        "note": (
            "Only relative rule weights change. Data split, model settings, lambda "
            "grid, and primary governance safeguards remain fixed."
        ),
    }
    (output_dir / "rule_weight_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    safeguard = subparsers.add_parser(
        "safeguards", help="Re-select lambdas under alternative safeguard thresholds"
    )
    safeguard.add_argument("--primary-results-dir", required=True, type=Path)
    safeguard.add_argument("--output", required=True, type=Path)
    safeguard.add_argument(
        "--cv-root",
        type=Path,
        default=None,
        help="Optional repeated-CV root containing xgboost/lightgbm/logistic/neural",
    )

    weights = subparsers.add_parser(
        "weights", help="Refit models under alternative relative rule weights"
    )
    weights.add_argument("--data", required=True, type=Path)
    weights.add_argument("--output", required=True, type=Path)
    weights.add_argument(
        "--algorithms",
        nargs="+",
        choices=list(DISPLAY_NAMES),
        default=list(DISPLAY_NAMES),
    )
    weights.add_argument(
        "--schemes",
        nargs="+",
        choices=list(WEIGHT_SCHEMES),
        default=list(WEIGHT_SCHEMES),
    )
    weights.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2],
    )
    weights.add_argument("--seed", type=int, default=42)
    weights.add_argument("--tree-rounds", type=int, default=500)
    weights.add_argument("--tree-learning-rate", type=float, default=0.05)
    weights.add_argument("--tree-early-stopping", type=int, default=50)
    weights.add_argument("--nn-max-epochs", type=int, default=300)
    weights.add_argument("--nn-patience", type=int, default=30)
    weights.add_argument("--nn-restarts", type=int, default=3)
    weights.add_argument("--nn-device", default="cpu")
    weights.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    if args.mode == "safeguards":
        run_safeguard_sensitivity(
            primary_results_dir=args.primary_results_dir,
            output_dir=args.output,
            cv_root=args.cv_root,
        )
        finish_cli(used_torch=False)
        return

    run_weight_sensitivity(
        data_path=args.data,
        output_dir=args.output,
        algorithms=args.algorithms,
        lambdas=args.lambdas,
        seed=args.seed,
        tree_rounds=args.tree_rounds,
        tree_learning_rate=args.tree_learning_rate,
        tree_early_stopping=args.tree_early_stopping,
        nn_max_epochs=args.nn_max_epochs,
        nn_patience=args.nn_patience,
        nn_restarts=args.nn_restarts,
        nn_device=args.nn_device,
        quiet=args.quiet,
        schemes=args.schemes,
    )
    finish_cli(used_torch="neural" in args.algorithms)


if __name__ == "__main__":
    main()
