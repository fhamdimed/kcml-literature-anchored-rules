#!/usr/bin/env python3
"""Run repeated outer cross-validation with inner lambda selection."""

import argparse
import os
from pathlib import Path

from kcml.runtime import configure_runtime, finish_cli

configure_runtime(Path(__file__).resolve().parent)

from kcml.cli import DEFAULT_LAMBDAS
from kcml.data import load_cohort_data
from kcml.experiment import run_repeated_nested_holdout
from kcml.factories import (
    make_lightgbm_factory,
    make_logistic_factory,
    make_neural_factory,
    make_xgboost_factory,
)


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
        choices=["xgboost", "lightgbm", "logistic", "neural", "neural_linear"],
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--rules", nargs="*", default=None)
    parser.add_argument(
        "--rule-control",
        choices=["none", "permuted", "column_permuted", "reversed"],
        default="none",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-tolerance", type=float, default=0.01)
    parser.add_argument("--selection-auc-tolerance", type=float, default=0.01)
    parser.add_argument("--selection-ap-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-violation-reduction", type=float, default=0.40)
    parser.add_argument(
        "--minimum-soft-violation-reduction", type=float, default=0.0
    )
    parser.add_argument("--tree-rounds", type=int, default=500)
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

    cohort = load_cohort_data(args.data, feature_order=args.features or ("MCV", "MCH", "HBA2", "HBF"))

    def builder(enabled_rules):
        common = dict(
            enabled_rules=enabled_rules,
            rule_control=args.rule_control,
            random_state=args.seed,
            verbose=not args.quiet,
        )
        if args.algorithm == "xgboost":
            return make_xgboost_factory(n_estimators=args.tree_rounds, **common)
        if args.algorithm == "lightgbm":
            return make_lightgbm_factory(n_estimators=args.tree_rounds, **common)
        if args.algorithm == "logistic":
            return make_logistic_factory(**common)
        architecture = "linear" if args.algorithm == "neural_linear" else "mlp"
        return make_neural_factory(
            architecture=architecture,
            max_epochs=args.nn_max_epochs,
            patience=args.nn_patience,
            n_restarts=args.nn_restarts,
            restart_aggregation=args.nn_restart_aggregation,
            deterministic=not args.nn_nondeterministic,
            device=args.nn_device,
            **common,
        )

    display_names = {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "logistic": "Logistic Regression",
        "neural": "Neural Network (MLP)",
        "neural_linear": "Neural Linear Check",
    }
    run_repeated_nested_holdout(
        cohort=cohort,
        algorithm_name=display_names[args.algorithm],
        model_factory_builder=builder,
        output_dir=Path(args.output),
        penalty_multipliers=args.lambdas,
        enabled_rules=args.rules,
        n_splits=args.folds,
        n_repeats=args.repeats,
        validation_fraction_within_training=args.validation_fraction,
        random_state=args.seed,
        selection_tolerance=args.selection_tolerance,
        selection_auc_tolerance=args.selection_auc_tolerance,
        selection_ap_tolerance=args.selection_ap_tolerance,
        minimum_violation_reduction=args.minimum_violation_reduction,
        minimum_soft_violation_reduction=(
            args.minimum_soft_violation_reduction
        ),
    )
    finish_cli(
        used_torch=args.algorithm in {"neural", "neural_linear"}
    )


if __name__ == "__main__":
    main()
