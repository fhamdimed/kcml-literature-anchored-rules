#!/usr/bin/env python3
"""Run the shared penalty grid with a PyTorch linear model or MLP."""

import os
from pathlib import Path

from kcml.runtime import configure_runtime, finish_cli

configure_runtime(Path(__file__).resolve().parent)
import argparse

from kcml.cli import add_common_arguments
from kcml.data import load_cohort_data, make_holdout_splits
from kcml.experiment import run_penalty_grid
from kcml.factories import make_neural_factory


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--architecture", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument(
        "--restart-aggregation",
        choices=["mean_probability", "best"],
        default="mean_probability",
        help="Default: average probabilities across all independently early-stopped restarts",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--nondeterministic", action="store_true",
        help="Disable strict deterministic PyTorch operations (not recommended for final analyses)",
    )
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    os.environ["KCML_TORCH_THREADS"] = str(args.threads)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(args.threads))

    cohort = load_cohort_data(args.data, feature_order=args.features or ("MCV", "MCH", "HBA2", "HBF"))
    splits = make_holdout_splits(
        cohort,
        test_size=args.test_size,
        validation_fraction_of_total=args.validation_size,
        random_state=args.seed,
    )
    factory = make_neural_factory(
        enabled_rules=args.rules,
        rule_control=args.rule_control,
        random_state=args.seed,
        architecture=args.architecture,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_restarts=args.restarts,
        restart_aggregation=args.restart_aggregation,
        deterministic=not args.nondeterministic,
        device=args.device,
        verbose=not args.quiet,
    )
    display_name = "Neural Network (MLP)" if args.architecture == "mlp" else "Neural Linear Check"
    run_penalty_grid(
        display_name,
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
    finish_cli(used_torch=True)


if __name__ == "__main__":
    main()
