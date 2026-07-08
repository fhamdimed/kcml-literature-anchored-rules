#!/usr/bin/env python3
"""Run all-rule, single-rule, leave-one-out, and negative-control experiments."""

import argparse
import os
from pathlib import Path

from kcml.runtime import configure_runtime, finish_cli

configure_runtime(Path(__file__).resolve().parent)

import pandas as pd

from kcml.cli import DEFAULT_LAMBDAS
from kcml.data import load_cohort_data, make_holdout_splits
from kcml.experiment import run_penalty_grid, safe_name
from kcml.factories import (
    make_lightgbm_factory,
    make_logistic_factory,
    make_neural_factory,
    make_xgboost_factory,
)
from kcml.rules import RuleCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Default: MCV MCH HBA2 HBF; may add HGB but retain all four rule variables",
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["xgboost", "lightgbm", "logistic", "neural"],
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--selection-tolerance", type=float, default=0.01)
    parser.add_argument("--selection-auc-tolerance", type=float, default=0.01)
    parser.add_argument("--selection-ap-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-violation-reduction", type=float, default=0.40)
    parser.add_argument(
        "--minimum-soft-violation-reduction", type=float, default=0.0
    )
    parser.add_argument("--nn-max-epochs", type=int, default=500)
    parser.add_argument("--nn-patience", type=int, default=50)
    parser.add_argument("--nn-restarts", type=int, default=3)
    parser.add_argument("--nn-device", default="cpu")
    parser.add_argument("--nn-threads", type=int, default=2)
    parser.add_argument(
        "--nn-restart-aggregation",
        choices=["mean_probability", "best"],
        default="mean_probability",
    )
    parser.add_argument("--nn-nondeterministic", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.nn_threads < 1:
        raise ValueError("--nn-threads must be at least 1")
    os.environ["KCML_TORCH_THREADS"] = str(args.nn_threads)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(args.nn_threads))

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort_data(args.data, feature_order=args.features or ("MCV", "MCH", "HBA2", "HBF"))
    splits = make_holdout_splits(
        cohort,
        test_size=args.test_size,
        validation_fraction_of_total=args.validation_size,
        random_state=args.seed,
    )
    all_rules = list(RuleCatalog().primary_names)
    configurations = [("all_rules", all_rules, "none")]
    configurations.extend((f"single_{rule}", [rule], "none") for rule in all_rules)
    configurations.extend(
        (f"leave_out_{rule}", [item for item in all_rules if item != rule], "none")
        for rule in all_rules
    )
    configurations.extend(
        [
            ("control_permuted", all_rules, "permuted"),
            ("control_column_permuted", all_rules, "column_permuted"),
            ("control_reversed", all_rules, "reversed"),
        ]
    )

    def factory_for(rules, control):
        common = dict(
            enabled_rules=rules,
            rule_control=control,
            random_state=args.seed,
            verbose=not args.quiet,
        )
        if args.algorithm == "xgboost":
            return make_xgboost_factory(**common)
        if args.algorithm == "lightgbm":
            return make_lightgbm_factory(**common)
        if args.algorithm == "logistic":
            return make_logistic_factory(**common)
        return make_neural_factory(
            max_epochs=args.nn_max_epochs,
            patience=args.nn_patience,
            n_restarts=args.nn_restarts,
            restart_aggregation=args.nn_restart_aggregation,
            deterministic=not args.nn_nondeterministic,
            device=args.nn_device,
            **common,
        )

    selected = []
    for label, rules, control in configurations:
        result = run_penalty_grid(
            algorithm_name=f"{args.algorithm}:{label}",
            model_factory=factory_for(rules, control),
            splits=splits,
            output_dir=root / safe_name(label),
            penalty_multipliers=args.lambdas,
            enabled_rules=rules,
            selection_tolerance=args.selection_tolerance,
            selection_auc_tolerance=args.selection_auc_tolerance,
            selection_ap_tolerance=args.selection_ap_tolerance,
            minimum_violation_reduction=args.minimum_violation_reduction,
            minimum_soft_violation_reduction=(
                args.minimum_soft_violation_reduction
            ),
            save_models=False,
            save_predictions=False,
        )
        row = dict(result["selected_test"])
        row["ablation"] = label
        row["training_rule_control"] = control
        row["training_rules"] = ",".join(rules)
        selected.append(row)

    pd.DataFrame(selected).to_csv(root / "ablation_selected_test_results.csv", index=False)
    finish_cli(
        used_torch=args.algorithm in {"neural", "neural_linear"}
    )


if __name__ == "__main__":
    main()
