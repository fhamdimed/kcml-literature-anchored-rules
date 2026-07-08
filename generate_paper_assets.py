#!/usr/bin/env python3
"""Generate manuscript tables and figures from archived KCML analysis outputs.

This wrapper is intended to be run from the repository root, but it also works
when called from another directory because paths are resolved relative to this
file by default.

Default expected repository layout
----------------------------------
kcml-literature-anchored-rules/
├── generate_paper_assets.py
├── analysis_outputs/
│   ├── common_threshold_robust/
│   ├── repeated_cv/
│   └── rule_ablations/
├── data/
│   └── cleaned_phenotype_cohort/
└── paper/
    └── scripts/
        └── generate_common_threshold_assets.py

Examples
--------
Run with default paths:

    python generate_paper_assets.py

Run with explicit paths:

    python generate_paper_assets.py \
      --cleaned-data-root data/cleaned_phenotype_cohort \
      --robust-root analysis_outputs/common_threshold_robust \
      --cv-root analysis_outputs/repeated_cv \
      --ablation-root analysis_outputs/rule_ablations \
      --output-root paper
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the directory containing this wrapper script."""
    return Path(__file__).resolve().parent


def _as_abs(path_text: str | None, default: Path, root: Path) -> Path:
    """Resolve a user path against the repository root when relative."""
    if path_text is None:
        return default.resolve()
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _check_dir(path: Path, label: str) -> bool:
    if path.is_dir():
        return True
    print(f"[ERROR] Missing {label}: {path}", file=sys.stderr)
    return False


def _check_file(path: Path, label: str) -> bool:
    if path.is_file():
        return True
    print(f"[ERROR] Missing {label}: {path}", file=sys.stderr)
    return False


def _print_tree_hint(root: Path) -> None:
    print("\nExpected repository layout:", file=sys.stderr)
    print(f"  {root}/", file=sys.stderr)
    print("  ├── generate_paper_assets.py", file=sys.stderr)
    print("  ├── analysis_outputs/", file=sys.stderr)
    print("  │   ├── common_threshold_robust/", file=sys.stderr)
    print("  │   ├── repeated_cv/", file=sys.stderr)
    print("  │   └── rule_ablations/", file=sys.stderr)
    print("  ├── data/", file=sys.stderr)
    print("  │   └── cleaned_phenotype_cohort/", file=sys.stderr)
    print("  └── paper/", file=sys.stderr)
    print("      └── scripts/", file=sys.stderr)
    print("          └── generate_common_threshold_assets.py", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate manuscript tables and figures from KCML robust, "
            "repeated-CV and rule-ablation outputs."
        )
    )
    parser.add_argument(
        "--asset-script",
        default=None,
        help=(
            "Path to generate_common_threshold_assets.py. "
            "Default: paper/scripts/generate_common_threshold_assets.py"
        ),
    )
    parser.add_argument(
        "--cleaned-data-root",
        default=None,
        help=(
            "Path to lean cleaned cohort outputs used to regenerate "
            "table_cleaning.tex. Default: data/cleaned_phenotype_cohort"
        ),
    )
    parser.add_argument(
        "--robust-root",
        default=None,
        help="Path to common-threshold robust analysis outputs.",
    )
    parser.add_argument(
        "--cv-root",
        default=None,
        help="Path to repeated cross-validation outputs.",
    )
    parser.add_argument(
        "--ablation-root",
        default=None,
        help="Path to rule-ablation outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Path where paper figures/tables/source_data should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command but do not execute it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()

    asset_script = _as_abs(
        args.asset_script,
        root / "paper" / "scripts" / "generate_common_threshold_assets.py",
        root,
    )
    robust_root = _as_abs(
        args.robust_root,
        root / "analysis_outputs" / "common_threshold_robust",
        root,
    )
    cv_root = _as_abs(
        args.cv_root,
        root / "analysis_outputs" / "repeated_cv",
        root,
    )
    ablation_root = _as_abs(
        args.ablation_root,
        root / "analysis_outputs" / "rule_ablations",
        root,
    )
    cleaned_data_root = _as_abs(
        args.cleaned_data_root,
        root / "data" / "cleaned_phenotype_cohort",
        root,
    )
    output_root = _as_abs(
        args.output_root,
        root / "paper",
        root,
    )

    print("Resolved paths:")
    print(f"  Repository root : {root}")
    print(f"  Asset script    : {asset_script}")
    print(f"  Cleaned data    : {cleaned_data_root}")
    print(f"  Robust outputs  : {robust_root}")
    print(f"  CV outputs      : {cv_root}")
    print(f"  Ablation outputs: {ablation_root}")
    print(f"  Output root     : {output_root}")

    ok = True
    ok &= _check_file(asset_script, "asset-generation script")
    ok &= _check_dir(cleaned_data_root, "cleaned cohort output directory")
    ok &= _check_dir(robust_root, "robust analysis output directory")
    ok &= _check_dir(cv_root, "repeated-CV output directory")
    ok &= _check_dir(ablation_root, "rule-ablation output directory")

    if not ok:
        _print_tree_hint(root)
        print(
            "\nFix one of the following:\n"
            "  1. Move the analysis outputs into analysis_outputs/ with the expected names;\n"
            "  2. Move generate_common_threshold_assets.py into paper/scripts/;\n"
            "  3. Run prepare_thalassemia_dataset.py to create data/cleaned_phenotype_cohort;\n"
            "  4. Or pass explicit paths with --asset-script, --cleaned-data-root, "
            "--robust-root, --cv-root, --ablation-root and --output-root.\n",
            file=sys.stderr,
        )
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(asset_script),
        "--cleaned-data-root",
        str(cleaned_data_root),
        "--robust-root",
        str(robust_root),
        "--cv-root",
        str(cv_root),
        "--ablation-root",
        str(ablation_root),
        "--output-root",
        str(output_root),
    ]

    print("\nCommand:")
    print(" ".join(command))

    if args.dry_run:
        print("\nDry run requested; command was not executed.")
        return 0

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        print(
            "\n[ERROR] Python could not start the asset-generation process.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 127
    except subprocess.CalledProcessError as exc:
        print(
            "\n[ERROR] Asset-generation script failed.",
            file=sys.stderr,
        )
        print(f"Exit status: {exc.returncode}", file=sys.stderr)
        print(
            "\nRun with --dry-run to inspect resolved paths, or run the printed "
            "command directly to see the full script-level error.",
            file=sys.stderr,
        )
        return exc.returncode

    print("\nDone. Generated manuscript assets under:")
    print(f"  {output_root / 'figures'}")
    print(f"  {output_root / 'tables'}")
    print(f"  {output_root / 'source_data'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
