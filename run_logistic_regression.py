#!/usr/bin/env python3
"""Run the shared penalty grid with convex logistic regression."""

from pathlib import Path

from kcml.runtime import configure_runtime

configure_runtime(Path(__file__).resolve().parent)
import argparse

from kcml.cli import add_common_arguments
from kcml.data import load_cohort_data, make_holdout_splits
from kcml.experiment import run_penalty_grid
from kcml.factories import make_logistic_factory


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--l2-strength", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    args = parser.parse_args()

    cohort = load_cohort_data(args.data, feature_order=args.features or ("MCV", "MCH", "HBA2", "HBF"))
    splits = make_holdout_splits(
        cohort,
        test_size=args.test_size,
        validation_fraction_of_total=args.validation_size,
        random_state=args.seed,
    )
    factory = make_logistic_factory(
        enabled_rules=args.rules,
        rule_control=args.rule_control,
        random_state=args.seed,
        l2_strength=args.l2_strength,
        max_iter=args.max_iter,
        verbose=not args.quiet,
    )
    run_penalty_grid(
        "Logistic Regression",
        factory,
        splits,
        args.output,
        args.lambdas,
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


if __name__ == "__main__":
    main()
