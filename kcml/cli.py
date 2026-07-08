"""Common command-line arguments."""

from __future__ import annotations

import argparse


DEFAULT_LAMBDAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data", required=True, help="Path to the cohort CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Model predictors. Default: MCV MCH HBA2 HBF. "
            "Sensitivity sets may add HGB but must retain all four rule variables."
        ),
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
        help="Penalty multipliers to evaluate; must include 0",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Conventional fixed threshold used only for sensitivity reporting "
            "(default: 0.5). The primary threshold is learned once from the "
            "lambda=0 validation model and held common across all lambdas."
        ),
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.01,
        help="Allowed balanced-accuracy deficit from the validation optimum",
    )
    parser.add_argument(
        "--selection-auc-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed ROC-AUC loss relative to validation lambda=0",
    )
    parser.add_argument(
        "--selection-ap-tolerance",
        type=float,
        default=0.02,
        help="Maximum allowed average-precision loss relative to validation lambda=0",
    )
    parser.add_argument(
        "--minimum-violation-reduction",
        type=float,
        default=0.40,
        help=(
            "Minimum fractional reduction in validation patient-level rule "
            "violations required for a penalized lambda"
        ),
    )
    parser.add_argument(
        "--minimum-soft-violation-reduction",
        type=float,
        default=0.0,
        help=(
            "Minimum fractional reduction in threshold-free soft rule "
            "violation required for a penalized lambda. Default 0 requires "
            "that soft discordance does not worsen."
        ),
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        default=None,
        help="Subset of primary rules, e.g. LR01 LR03 LR05; omit for all",
    )
    parser.add_argument(
        "--rule-control",
        choices=["none", "permuted", "column_permuted", "reversed"],
        default="none",
        help="Negative-control transformation used during training",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser
