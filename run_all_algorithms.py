#!/usr/bin/env python3
"""Run all requested KCML algorithms on one deterministic shared split.

Primary evaluation uses a threshold learned once from the unpenalized
validation model for each algorithm and then held fixed across every lambda.
Per-lambda optimized thresholds and threshold 0.5 are retained as secondary
and sensitivity analyses. Neural algorithms ensemble independently early-
stopped restarts and run in a fresh Python subprocess by default on macOS.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from kcml.runtime import configure_runtime, finish_cli

PROJECT_ROOT = Path(__file__).resolve().parent
configure_runtime(PROJECT_ROOT)

import pandas as pd

from kcml.cli import add_common_arguments
from kcml.data import load_cohort_data, make_holdout_splits
from kcml.experiment import run_penalty_grid, safe_name
from kcml.factories import (
    make_lightgbm_factory,
    make_logistic_factory,
    make_xgboost_factory,
)

NEURAL_KEYS = {"neural", "neural_linear"}


def _append_many(command: list[str], flag: str, values) -> None:
    values = list(values)
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def _run_neural_subprocess(
    *,
    key: str,
    args: argparse.Namespace,
    output_dir: Path,
    expected_assignments: pd.DataFrame,
) -> dict[str, object]:
    """Run one neural configuration in a clean interpreter and reload outputs."""
    architecture = "mlp" if key == "neural" else "linear"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_neural_network.py"),
        "--data",
        str(args.data),
        "--output",
        str(output_dir),
        "--architecture",
        architecture,
        "--seed",
        str(args.seed),
        "--test-size",
        str(args.test_size),
        "--validation-size",
        str(args.validation_size),
        "--threshold",
        str(args.threshold),
        "--selection-tolerance",
        str(args.selection_tolerance),
        "--selection-auc-tolerance",
        str(args.selection_auc_tolerance),
        "--selection-ap-tolerance",
        str(args.selection_ap_tolerance),
        "--minimum-violation-reduction",
        str(args.minimum_violation_reduction),
        "--minimum-soft-violation-reduction",
        str(args.minimum_soft_violation_reduction),
        "--max-epochs",
        str(args.nn_max_epochs),
        "--patience",
        str(args.nn_patience),
        "--restarts",
        str(args.nn_restarts),
        "--restart-aggregation",
        str(args.nn_restart_aggregation),
        "--threads",
        str(args.nn_threads),
        "--device",
        str(args.nn_device),
    ]
    _append_many(command, "--lambdas", args.lambdas)
    _append_many(command, "--hidden-sizes", args.nn_hidden_sizes)
    if args.features:
        _append_many(command, "--features", args.features)
    if args.rules is not None:
        command.append("--rules")
        command.extend(args.rules)
    if args.rule_control != "none":
        command.extend(["--rule-control", args.rule_control])
    if args.nn_nondeterministic:
        command.append("--nondeterministic")
    if args.quiet:
        command.append("--quiet")

    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(name, str(args.nn_threads))
    env["KCML_TORCH_THREADS"] = str(args.nn_threads)

    print(
        f"Launching {key} in an isolated Python process "
        f"(threads={args.nn_threads}, device={args.nn_device})...",
        flush=True,
    )
    subprocess.run(command, check=True, env=env)

    required = {
        "fixed_results": output_dir / "all_lambda_results.csv",
        "common_results": output_dir / "common_threshold_all_lambda_results.csv",
        "optimized_results": output_dir
        / "threshold_optimized_all_lambda_results.csv",
        "selected_common": output_dir / "selected_lambda_results.csv",
        "selected_optimized": output_dir
        / "selected_lambda_optimized_threshold_results.csv",
        "selected_fixed": output_dir / "selected_lambda_fixed_threshold_results.csv",
        "audit": output_dir / "lambda_selection_audit.csv",
        "splits": output_dir / "split_assignments.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Neural subprocess finished without expected outputs:\n  "
            + "\n  ".join(missing)
        )

    child_assignments = pd.read_csv(required["splits"])
    compare_columns = [
        column
        for column in ("row_index", "split")
        if column in child_assignments.columns
        and column in expected_assignments.columns
    ]
    if not compare_columns:
        raise RuntimeError("Could not verify shared split assignments")
    left = expected_assignments[compare_columns].reset_index(drop=True)
    right = child_assignments[compare_columns].reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError(
            "The isolated neural subprocess produced different split assignments. "
            "Results were not combined."
        )

    selected_common = pd.read_csv(required["selected_common"])
    selected_optimized = pd.read_csv(required["selected_optimized"])
    selected_fixed = pd.read_csv(required["selected_fixed"])
    common_test = selected_common.loc[
        selected_common["split"] == "test"
    ].iloc[0].to_dict()
    optimized_test = selected_optimized.loc[
        selected_optimized["split"] == "test"
    ].iloc[0].to_dict()
    fixed_test = selected_fixed.loc[
        selected_fixed["split"] == "test"
    ].iloc[0].to_dict()
    common_validation = selected_common.loc[
        selected_common["split"] == "validation"
    ].iloc[0].to_dict()
    optimized_validation = selected_optimized.loc[
        selected_optimized["split"] == "validation"
    ].iloc[0].to_dict()

    return {
        "results": pd.read_csv(required["fixed_results"]),
        "common_threshold_results": pd.read_csv(required["common_results"]),
        "threshold_optimized_results": pd.read_csv(required["optimized_results"]),
        "selected_test": common_test,
        "selected_common_test": common_test,
        "selected_optimized_test": optimized_test,
        "selected_fixed_test": fixed_test,
        "selected_validation": common_validation,
        "selected_common_threshold": float(common_validation["threshold"]),
        "selected_optimized_threshold": float(optimized_validation["threshold"]),
        "selection_audit": pd.read_csv(required["audit"]),
    }


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["xgboost", "lightgbm", "logistic", "neural", "neural_linear"],
        default=["xgboost", "lightgbm", "logistic", "neural"],
    )
    parser.add_argument("--tree-rounds", type=int, default=500)
    parser.add_argument("--tree-learning-rate", type=float, default=0.05)
    parser.add_argument("--tree-early-stopping", type=int, default=50)
    parser.add_argument("--l2-strength", type=float, default=1.0)
    parser.add_argument("--nn-hidden-sizes", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--nn-max-epochs", type=int, default=500)
    parser.add_argument("--nn-patience", type=int, default=50)
    parser.add_argument("--nn-restarts", type=int, default=3)
    parser.add_argument("--nn-device", default="cpu")
    parser.add_argument(
        "--nn-restart-aggregation",
        choices=["mean_probability", "best"],
        default="mean_probability",
        help="Default: ensemble all independently early-stopped neural restarts",
    )
    parser.add_argument(
        "--nn-nondeterministic",
        action="store_true",
        help="Disable strict deterministic PyTorch operations",
    )
    parser.add_argument(
        "--nn-threads",
        type=int,
        default=2,
        help="CPU threads used by the isolated neural process (default: 2)",
    )
    parser.add_argument(
        "--neural-in-process",
        action="store_true",
        help=(
            "Debugging only: run neural models in this interpreter. The default "
            "isolated subprocess is safer on macOS."
        ),
    )
    args = parser.parse_args()
    if args.nn_threads < 1:
        raise ValueError("--nn-threads must be at least 1")

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort_data(
        args.data,
        feature_order=args.features or ("MCV", "MCH", "HBA2", "HBF"),
    )
    splits = make_holdout_splits(
        cohort,
        test_size=args.test_size,
        validation_fraction_of_total=args.validation_size,
        random_state=args.seed,
    )
    shared_assignments = splits.assignments()

    def build_core_configuration(key: str):
        if key == "xgboost":
            return (
                "XGBoost",
                make_xgboost_factory(
                    enabled_rules=args.rules,
                    rule_control=args.rule_control,
                    random_state=args.seed,
                    n_estimators=args.tree_rounds,
                    learning_rate=args.tree_learning_rate,
                    early_stopping_rounds=args.tree_early_stopping,
                    verbose=not args.quiet,
                ),
            )
        if key == "lightgbm":
            return (
                "LightGBM",
                make_lightgbm_factory(
                    enabled_rules=args.rules,
                    rule_control=args.rule_control,
                    random_state=args.seed,
                    n_estimators=args.tree_rounds,
                    learning_rate=args.tree_learning_rate,
                    early_stopping_rounds=args.tree_early_stopping,
                    verbose=not args.quiet,
                ),
            )
        if key == "logistic":
            return (
                "Logistic Regression",
                make_logistic_factory(
                    enabled_rules=args.rules,
                    rule_control=args.rule_control,
                    random_state=args.seed,
                    l2_strength=args.l2_strength,
                    verbose=not args.quiet,
                ),
            )
        raise ValueError(f"Unsupported non-neural algorithm: {key}")

    all_fixed_rows: list[pd.DataFrame] = []
    all_common_rows: list[pd.DataFrame] = []
    all_optimized_rows: list[pd.DataFrame] = []
    selected_common_rows: list[dict[str, object]] = []
    selected_optimized_rows: list[dict[str, object]] = []
    selected_fixed_rows: list[dict[str, object]] = []
    selection_audits: list[pd.DataFrame] = []

    for key in args.algorithms:
        if key in NEURAL_KEYS and not args.neural_in_process:
            result = _run_neural_subprocess(
                key=key,
                args=args,
                output_dir=root / safe_name(key),
                expected_assignments=shared_assignments,
            )
        else:
            if key in NEURAL_KEYS:
                from kcml.factories import make_neural_factory

                display_name = (
                    "Neural Network (MLP)"
                    if key == "neural"
                    else "Neural Linear Check"
                )
                factory = make_neural_factory(
                    enabled_rules=args.rules,
                    rule_control=args.rule_control,
                    random_state=args.seed,
                    architecture="mlp" if key == "neural" else "linear",
                    hidden_sizes=args.nn_hidden_sizes if key == "neural" else (),
                    dropout=0.10 if key == "neural" else 0.0,
                    max_epochs=args.nn_max_epochs,
                    patience=args.nn_patience,
                    n_restarts=args.nn_restarts,
                    restart_aggregation=args.nn_restart_aggregation,
                    deterministic=not args.nn_nondeterministic,
                    device=args.nn_device,
                    verbose=not args.quiet,
                )
            else:
                display_name, factory = build_core_configuration(key)

            result = run_penalty_grid(
                algorithm_name=display_name,
                model_factory=factory,
                splits=splits,
                output_dir=root / safe_name(key),
                penalty_multipliers=args.lambdas,
                enabled_rules=args.rules,
                threshold=args.threshold,
                selection_tolerance=args.selection_tolerance,
                selection_auc_tolerance=args.selection_auc_tolerance,
                selection_ap_tolerance=args.selection_ap_tolerance,
                minimum_violation_reduction=args.minimum_violation_reduction,
                minimum_soft_violation_reduction=(
                    args.minimum_soft_violation_reduction
                ),
            )

        all_fixed_rows.append(result["results"])
        all_common_rows.append(result["common_threshold_results"])
        all_optimized_rows.append(result["threshold_optimized_results"])
        selected_common_rows.append(result["selected_common_test"])
        selected_optimized_rows.append(result["selected_optimized_test"])
        selected_fixed_rows.append(result["selected_fixed_test"])
        audit = result["selection_audit"].copy()
        audit.insert(0, "algorithm_key", key)
        selection_audits.append(audit)

    pd.concat(all_fixed_rows, ignore_index=True).to_csv(
        root / "combined_all_lambda_results.csv", index=False
    )
    pd.concat(all_common_rows, ignore_index=True).to_csv(
        root / "combined_common_threshold_all_lambda_results.csv", index=False
    )
    pd.concat(all_optimized_rows, ignore_index=True).to_csv(
        root / "combined_threshold_optimized_all_lambda_results.csv", index=False
    )

    # Primary selected result uses the baseline-derived common threshold.
    pd.DataFrame(selected_common_rows).to_csv(
        root / "combined_selected_test_results.csv", index=False
    )
    pd.DataFrame(selected_common_rows).to_csv(
        root / "combined_selected_common_threshold_test_results.csv", index=False
    )
    pd.DataFrame(selected_optimized_rows).to_csv(
        root / "combined_selected_optimized_threshold_test_results.csv", index=False
    )
    pd.DataFrame(selected_fixed_rows).to_csv(
        root / "combined_selected_fixed_threshold_test_results.csv", index=False
    )
    pd.concat(selection_audits, ignore_index=True).to_csv(
        root / "combined_lambda_selection_audit.csv", index=False
    )
    shared_assignments.to_csv(root / "shared_split_assignments.csv", index=False)

    finish_cli(
        used_torch=bool(
            args.neural_in_process
            and any(k in NEURAL_KEYS for k in args.algorithms)
        )
    )


if __name__ == "__main__":
    main()
