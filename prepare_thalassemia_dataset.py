#!/usr/bin/env python3
"""
prepare_thalassemia_dataset.py

Prepare the reproducible analytical dataset from genotype_datadeposition.xlsx.

Default primary source
----------------------
Workbook: data/genotype_datadeposition.xlsx
Sheet:    Phenotype cohort

Target
------
0 = Pure beta-thalassemia trait
1 = Alpha-beta co-inheritance

Key cleaning decisions
----------------------
1. Genotype is used only to construct the outcome and is excluded from the
   model predictors.
2. Blank HbF is interpreted as zero only when HbA and HbA2 are observed and
   |100 - HbA - HbA2| is within a configurable tolerance (default: 0.5).
3. Other blank HbF values remain unresolved by default. They are not silently
   filled during cohort preparation.
4. HbA is reconstructed only when HbA2 and cleaned HbF are available and the
   resulting value is biologically plausible.
5. No predictor imputation is performed here. Remaining imputation must be
   fitted inside each training fold in the downstream modelling pipeline.
6. Duplicate patient IDs stop execution by default to avoid possible leakage.

Example
-------
python prepare_thalassemia_dataset.py \
    --input data/genotype_datadeposition.xlsx \
    --sheet "Phenotype cohort" \
    --output-dir data/cleaned_phenotype_cohort
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SHEET_ROWS = {
    "Prevalence cohort": 14_146,
    "Phenotype cohort": 13_888,
}

PURE_BTT_GENOTYPES = {
    "αα/αα,βm/βn",
}

COINHERITANCE_GENOTYPES = {
    "α0/αα,βm/βn",
    "α+/αα,βm/βn",
    "α+/α0,βm/βn",
}

PRIMARY_PREDICTOR_COLUMNS = [
    "HGB",
    "MCV",
    "MCH",
    "HBA",
    "HBA2",
    "HBF",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the thalassemia workbook and create a model-ready CSV "
            "with auditable HbF blank handling."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/genotype_datadeposition.xlsx"),
        help="Path to genotype_datadeposition.xlsx.",
    )
    parser.add_argument(
        "--sheet",
        default="Phenotype cohort",
        help='Workbook sheet to process. Default: "Phenotype cohort".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cleaned_phenotype_cohort"),
        help="Directory for cleaned data outputs.",
    )
    parser.add_argument(
        "--hbf-zero-tolerance",
        type=float,
        default=0.5,
        help=(
            "Interpret blank HbF as zero when "
            "abs(100 - HbA - HbA2) is no greater than this value."
        ),
    )
    parser.add_argument(
        "--fraction-sum-tolerance",
        type=float,
        default=1.0,
        help=(
            "Flag complete cleaned hemoglobin fractions when their sum differs "
            "from 100 by more than this value."
        ),
    )
    parser.add_argument(
        "--reconstruct-positive-hbf-residuals",
        action="store_true",
        help=(
            "Sensitivity option: reconstruct a blank HbF as "
            "100 - HbA - HbA2 when the residual is positive and plausible. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--max-reconstructed-hbf",
        type=float,
        default=20.0,
        help=(
            "Maximum positive HbF residual that may be reconstructed when "
            "--reconstruct-positive-hbf-residuals is enabled."
        ),
    )
    parser.add_argument(
        "--assume-double-blank-hbf-zero",
        action="store_true",
        help=(
            "Sensitivity option: when HbA and HbF are both blank but HbA2 is "
            "observed, set HbF=0 and reconstruct HbA=100-HbA2. Disabled by "
            "default because the residual cannot be checked."
        ),
    )
    parser.add_argument(
        "--minimum-age",
        type=float,
        default=None,
        help=(
            "Optional minimum age. Rows below this age are excluded. Missing "
            "age is retained and reported unless you clean it separately."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-ids",
        action="store_true",
        help=(
            "Continue when duplicate patient IDs are found. The safer default "
            "is to stop execution because duplicate IDs may cause leakage."
        ),
    )

    args = parser.parse_args()

    if args.hbf_zero_tolerance < 0:
        parser.error("--hbf-zero-tolerance must be non-negative")
    if args.fraction_sum_tolerance < 0:
        parser.error("--fraction-sum-tolerance must be non-negative")
    if args.max_reconstructed_hbf <= 0:
        parser.error("--max-reconstructed-hbf must be positive")

    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_column_name(column: object) -> str:
    text = unicodedata.normalize("NFKC", str(column)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_genotype(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None

    replacements = {
        "，": ",",
        "；": ";",
        "／": "/",
        "＋": "+",
        "−": "-",
        "–": "-",
        "Α": "α",
        "Β": "β",
        "alpha": "α",
        "beta": "β",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", "", text)
    text = text.replace("^", "")
    return text.lower()


def locate_columns(dataframe: pd.DataFrame) -> dict[str, str]:
    normalized_to_original = {
        normalize_column_name(column): str(column)
        for column in dataframe.columns
    }

    aliases = {
        "patient_id": ["id", "patientid", "participantid", "subjectid"],
        "sex": ["gender", "sex"],
        "age": ["ageyear", "ageyears", "age"],
        "genotype": ["genotypegroup", "genotype"],
        "HGB": ["hbgl", "hgb", "hb", "hemoglobin", "haemoglobin"],
        "MCV": ["mcvfl", "mcv"],
        "MCH": ["mchpg", "mch"],
        "HBA": ["hba", "hbapercent"],
        "HBA2": ["hba2", "hba2percent"],
        "HBF": ["hbf", "hbfpercent"],
    }

    result: dict[str, str] = {}
    missing: list[str] = []

    for canonical, candidate_aliases in aliases.items():
        matched: str | None = None
        for alias in candidate_aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_to_original:
                matched = normalized_to_original[normalized_alias]
                break

        if matched is None:
            missing.append(canonical)
        else:
            result[canonical] = matched

    if missing:
        raise KeyError(
            "Could not locate required columns: "
            f"{missing}. Available columns: {list(dataframe.columns)}"
        )

    return result


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_canonical_dataframe(
    source_df: pd.DataFrame,
    column_map: dict[str, str],
) -> pd.DataFrame:
    row_count = len(source_df)

    df = pd.DataFrame(
        {
            "source_row_number": np.arange(2, row_count + 2, dtype=int),
            "patient_id": source_df[column_map["patient_id"]],
            "sex": source_df[column_map["sex"]],
            "age": source_df[column_map["age"]],
            "genotype_original": source_df[column_map["genotype"]],
            "HGB_source": source_df[column_map["HGB"]],
            "MCV_source": source_df[column_map["MCV"]],
            "MCH_source": source_df[column_map["MCH"]],
            "HBA_source": source_df[column_map["HBA"]],
            "HBA2_source": source_df[column_map["HBA2"]],
            "HBF_source": source_df[column_map["HBF"]],
        }
    )

    df["patient_id"] = df["patient_id"].astype("string").str.strip()
    df["sex"] = df["sex"].astype("string").str.strip()

    for column in [
        "age",
        "HGB_source",
        "MCV_source",
        "MCH_source",
        "HBA_source",
        "HBA2_source",
        "HBF_source",
    ]:
        df[column] = safe_numeric(df[column])

    df["genotype_normalized"] = df["genotype_original"].map(normalize_genotype)

    label_map = {genotype: 0 for genotype in PURE_BTT_GENOTYPES}
    label_map.update({genotype: 1 for genotype in COINHERITANCE_GENOTYPES})

    df["label"] = df["genotype_normalized"].map(label_map)
    df["class_name"] = df["label"].map(
        {
            0: "Pure BTT",
            1: "Alpha-beta co-inheritance",
        }
    )

    return df


def clean_hemoglobin_fractions(
    df: pd.DataFrame,
    *,
    zero_tolerance: float,
    fraction_sum_tolerance: float,
    reconstruct_positive_residuals: bool,
    max_reconstructed_hbf: float,
    assume_double_blank_hbf_zero: bool,
) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["HBA_source_blank"] = cleaned["HBA_source"].isna().astype("int8")
    cleaned["HBF_source_blank"] = cleaned["HBF_source"].isna().astype("int8")

    cleaned["HGB"] = cleaned["HGB_source"]
    cleaned["MCV"] = cleaned["MCV_source"]
    cleaned["MCH"] = cleaned["MCH_source"]
    cleaned["HBA"] = cleaned["HBA_source"]
    cleaned["HBA2"] = cleaned["HBA2_source"]
    cleaned["HBF"] = cleaned["HBF_source"]

    cleaned["HBF_residual_from_HBA_HBA2"] = (
        100.0 - cleaned["HBA_source"] - cleaned["HBA2_source"]
    )

    cleaned["HBF_cleaning_status"] = np.where(
        cleaned["HBF_source"].notna(),
        "observed",
        "unresolved_blank",
    )
    cleaned["HBA_cleaning_status"] = np.where(
        cleaned["HBA_source"].notna(),
        "observed",
        "unresolved_blank",
    )

    hbf_blank_with_hba_hba2 = (
        cleaned["HBF_source"].isna()
        & cleaned["HBA_source"].notna()
        & cleaned["HBA2_source"].notna()
    )

    blank_as_zero = (
        hbf_blank_with_hba_hba2
        & cleaned["HBF_residual_from_HBA_HBA2"].abs().le(zero_tolerance)
    )

    cleaned.loc[blank_as_zero, "HBF"] = 0.0
    cleaned.loc[blank_as_zero, "HBF_cleaning_status"] = "blank_interpreted_as_zero"

    positive_plausible_residual = (
        hbf_blank_with_hba_hba2
        & cleaned["HBF_residual_from_HBA_HBA2"].gt(zero_tolerance)
        & cleaned["HBF_residual_from_HBA_HBA2"].le(max_reconstructed_hbf)
    )

    if reconstruct_positive_residuals:
        cleaned.loc[
            positive_plausible_residual,
            "HBF",
        ] = cleaned.loc[
            positive_plausible_residual,
            "HBF_residual_from_HBA_HBA2",
        ]
        cleaned.loc[
            positive_plausible_residual,
            "HBF_cleaning_status",
        ] = "reconstructed_from_positive_residual"

    both_hba_hbf_blank = (
        cleaned["HBA_source"].isna()
        & cleaned["HBF_source"].isna()
        & cleaned["HBA2_source"].notna()
    )

    if assume_double_blank_hbf_zero:
        cleaned.loc[both_hba_hbf_blank, "HBF"] = 0.0
        cleaned.loc[both_hba_hbf_blank, "HBA"] = (
            100.0 - cleaned.loc[both_hba_hbf_blank, "HBA2"]
        )
        cleaned.loc[both_hba_hbf_blank, "HBF_cleaning_status"] = (
            "double_blank_assumed_zero"
        )
        cleaned.loc[both_hba_hbf_blank, "HBA_cleaning_status"] = (
            "reconstructed_from_HBA2_and_zero_HBF"
        )

    hba_reconstructable = (
        cleaned["HBA"].isna()
        & cleaned["HBA2"].notna()
        & cleaned["HBF"].notna()
    )

    candidate_hba = 100.0 - cleaned["HBA2"] - cleaned["HBF"]
    plausible_hba = hba_reconstructable & candidate_hba.between(0.0, 100.0)

    cleaned.loc[plausible_hba, "HBA"] = candidate_hba.loc[plausible_hba]
    cleaned.loc[plausible_hba, "HBA_cleaning_status"] = (
        "reconstructed_from_HBA2_HBF"
    )

    cleaned["HBF_zero_from_blank"] = (
        cleaned["HBF_cleaning_status"].eq("blank_interpreted_as_zero")
        | cleaned["HBF_cleaning_status"].eq("double_blank_assumed_zero")
    ).astype("int8")

    cleaned["HBF_reconstructed_from_residual"] = cleaned[
        "HBF_cleaning_status"
    ].eq("reconstructed_from_positive_residual").astype("int8")

    cleaned["HBF_unresolved"] = cleaned["HBF"].isna().astype("int8")
    cleaned["HBA_unresolved"] = cleaned["HBA"].isna().astype("int8")

    cleaned["Hb_sum"] = cleaned[["HBA", "HBA2", "HBF"]].sum(
        axis=1,
        min_count=3,
    )
    cleaned["Hb_sum_error"] = cleaned["Hb_sum"] - 100.0
    cleaned["Hb_sum_anomaly"] = (
        cleaned["Hb_sum"].notna()
        & cleaned["Hb_sum_error"].abs().gt(fraction_sum_tolerance)
    ).astype("int8")

    cleaned["HBF_blank_nonzero_residual_anomaly"] = (
        hbf_blank_with_hba_hba2
        & ~blank_as_zero
        & cleaned["HBF"].isna()
    ).astype("int8")

    return cleaned


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).to_dict().items()
    }


def missing_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {column: int(df[column].isna().sum()) for column in columns}


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input workbook not found: {args.input.resolve()}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("Preparing thalassemia analytical dataset")
    print("=" * 76)
    print(f"Input workbook : {args.input.resolve()}")
    print(f"Input sheet    : {args.sheet}")
    print(f"Output folder  : {args.output_dir.resolve()}")
    print()

    workbook = pd.ExcelFile(args.input, engine="openpyxl")
    if args.sheet not in workbook.sheet_names:
        raise ValueError(
            f"Sheet {args.sheet!r} not found. Available sheets: "
            f"{workbook.sheet_names}"
        )

    source_df = pd.read_excel(args.input, sheet_name=args.sheet, engine="openpyxl")

    original_rows = len(source_df)
    expected_rows = EXPECTED_SHEET_ROWS.get(args.sheet)

    print(f"Rows in source sheet: {original_rows:,}")
    if expected_rows is not None:
        status = "OK" if original_rows == expected_rows else "WARNING"
        print(f"[{status}] Expected rows for {args.sheet!r}: {expected_rows:,}")
    print()

    column_map = locate_columns(source_df)
    print("Detected columns:")
    for canonical, source in column_map.items():
        print(f"  {canonical:12s} <- {source}")
    print()

    df = make_canonical_dataframe(source_df, column_map)

    target_mask = df["label"].notna()
    age_excluded_mask = pd.Series(False, index=df.index)
    if args.minimum_age is not None:
        age_excluded_mask = df["age"].notna() & df["age"].lt(args.minimum_age)

    analytical_mask = target_mask & ~age_excluded_mask
    analytical = df.loc[analytical_mask].copy()
    analytical["label"] = analytical["label"].astype("int8")

    duplicate_mask = (
        analytical["patient_id"].notna()
        & analytical["patient_id"].duplicated(keep=False)
    )
    duplicate_rows = analytical.loc[
        duplicate_mask,
        [
            "patient_id",
            "source_row_number",
            "genotype_original",
            "label",
        ],
    ].sort_values(["patient_id", "source_row_number"])

    if len(duplicate_rows) and not args.allow_duplicate_ids:
        preview = duplicate_rows.head(20).to_string(index=False)
        raise RuntimeError(
            f"Found {len(duplicate_rows):,} rows with duplicated patient IDs. "
            "Review the following first rows and rerun with --allow-duplicate-ids "
            "only if repeated rows are intentional and downstream splitting is "
            f"grouped by patient_id:\n{preview}"
        )

    cleaned = clean_hemoglobin_fractions(
        analytical,
        zero_tolerance=args.hbf_zero_tolerance,
        fraction_sum_tolerance=args.fraction_sum_tolerance,
        reconstruct_positive_residuals=args.reconstruct_positive_hbf_residuals,
        max_reconstructed_hbf=args.max_reconstructed_hbf,
        assume_double_blank_hbf_zero=args.assume_double_blank_hbf_zero,
    )

    model_columns = [
        "patient_id",
        "HGB",
        "MCV",
        "MCH",
        "HBA",
        "HBA2",
        "HBF",
        "HBA_source_blank",
        "HBF_source_blank",
        "HBF_zero_from_blank",
        "HBF_unresolved",
        "Hb_sum",
        "Hb_sum_anomaly",
        "label",
    ]
    model_df = cleaned[model_columns].copy().reset_index(drop=True)

    cohort_flow = pd.DataFrame(
        [
            {
                "stage": f"Original sheet: {args.sheet}",
                "number_of_records": original_rows,
            },
            {
                "stage": "Excluded: genotype outside binary target definition",
                "number_of_records": int((~target_mask).sum()),
            },
            {
                "stage": "Excluded: below configured minimum age",
                "number_of_records": int(age_excluded_mask.sum()),
            },
            {
                "stage": "Final analytical cohort",
                "number_of_records": len(cleaned),
            },
            {
                "stage": "Pure beta-thalassemia trait",
                "number_of_records": int((cleaned["label"] == 0).sum()),
            },
            {
                "stage": "Alpha-beta co-inheritance",
                "number_of_records": int((cleaned["label"] == 1).sum()),
            },
        ]
    )

    output_paths = {
        "model_matrix": args.output_dir / "thalassemia_model_matrix_clean.csv",
        "cohort_flow": args.output_dir / "cohort_flow.csv",
        "manifest": args.output_dir / "analysis_manifest.json",
    }

    write_csv(model_df, output_paths["model_matrix"])
    write_csv(cohort_flow, output_paths["cohort_flow"])

    genotype_distribution = (
        cleaned.groupby(["label", "class_name", "genotype_normalized"], dropna=False)
        .size()
        .reset_index(name="number_of_records")
        .sort_values(["label", "number_of_records"], ascending=[True, False])
        .reset_index(drop=True)
    )

    hbf_status_counts = value_counts_dict(cleaned["HBF_cleaning_status"])
    hba_status_counts = value_counts_dict(cleaned["HBA_cleaning_status"])
    genotype_distribution_records = genotype_distribution.to_dict(orient="records")

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "script_mode": "repository_lean_outputs",
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "input_sheet": args.sheet,
        "expected_source_rows": expected_rows,
        "observed_source_rows": original_rows,
        "target_definition": {
            "0": "Pure beta-thalassemia trait",
            "1": "Alpha-beta co-inheritance",
        },
        "pure_btt_genotypes": sorted(PURE_BTT_GENOTYPES),
        "coinheritance_genotypes": sorted(COINHERITANCE_GENOTYPES),
        "cleaning_parameters": {
            "hbf_zero_tolerance": args.hbf_zero_tolerance,
            "fraction_sum_tolerance": args.fraction_sum_tolerance,
            "reconstruct_positive_hbf_residuals": (
                args.reconstruct_positive_hbf_residuals
            ),
            "max_reconstructed_hbf": args.max_reconstructed_hbf,
            "assume_double_blank_hbf_zero": args.assume_double_blank_hbf_zero,
            "minimum_age": args.minimum_age,
            "allow_duplicate_ids": args.allow_duplicate_ids,
        },
        "counts": {
            "source_rows": original_rows,
            "excluded_outside_target_genotypes": int((~target_mask).sum()),
            "excluded_below_minimum_age": int(age_excluded_mask.sum()),
            "analytical_rows": len(cleaned),
            "pure_btt": int((cleaned["label"] == 0).sum()),
            "alpha_beta_coinheritance": int((cleaned["label"] == 1).sum()),
            "positive_class_prevalence": (
                float(cleaned["label"].mean()) if len(cleaned) else None
            ),
            "hbf_source_blank": int(cleaned["HBF_source_blank"].sum()),
            "hbf_blank_interpreted_as_zero": int(
                cleaned["HBF_zero_from_blank"].sum()
            ),
            "hbf_reconstructed_from_positive_residual": int(
                cleaned["HBF_reconstructed_from_residual"].sum()
            ),
            "hbf_unresolved": int(cleaned["HBF_unresolved"].sum()),
            "hba_unresolved": int(cleaned["HBA_unresolved"].sum()),
            "fraction_sum_anomalies": int(cleaned["Hb_sum_anomaly"].sum()),
            "duplicate_rows": len(duplicate_rows),
        },
        "missing_values_in_model_matrix": missing_counts(
            model_df,
            ["HGB", "MCV", "MCH", "HBA", "HBA2", "HBF", "Hb_sum"],
        ),
        "hbf_cleaning_status_counts": hbf_status_counts,
        "hba_cleaning_status_counts": hba_status_counts,
        "genotype_distribution": genotype_distribution_records,
        "recommended_primary_predictors": ["MCV", "MCH", "HBA2", "HBF"],
        "optional_predictor": "HGB",
        "qc_only_variables": ["HBA", "Hb_sum", "Hb_sum_anomaly"],
        "outputs": {
            name: str(path.resolve())
            for name, path in output_paths.items()
        },
    }

    hashable_outputs = {
        name: path
        for name, path in output_paths.items()
        if name != "manifest" and path.exists()
    }
    manifest["output_sha256"] = {
        path.name: sha256_file(path)
        for path in hashable_outputs.values()
    }

    with output_paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    # Add manifest hash after writing the initial manifest.
    manifest["output_sha256"][output_paths["manifest"].name] = sha256_file(
        output_paths["manifest"]
    )
    with output_paths["manifest"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("=" * 76)
    print("Final cohort summary")
    print("=" * 76)
    print(cohort_flow.to_string(index=False))
    print()
    print("HbF cleaning summary:")
    print(cleaned["HBF_cleaning_status"].value_counts(dropna=False).to_string())
    print()
    print(f"Unresolved cleaned HbF values : {int(cleaned['HBF_unresolved'].sum()):,}")
    print(f"Unresolved cleaned HbA values : {int(cleaned['HBA_unresolved'].sum()):,}")
    print(f"Hb fraction-sum anomalies     : {int(cleaned['Hb_sum_anomaly'].sum()):,}")
    print(f"Duplicate-ID rows             : {len(duplicate_rows):,}")
    print()
    print("Recommended primary predictors:")
    print("  MCV, MCH, HBA2, HBF")
    print("Optional sensitivity predictor:")
    print("  HGB")
    print("QC-only variables:")
    print("  HBA, Hb_sum, Hb_sum_anomaly")
    print()
    print("Output files:")
    for path in output_paths.values():
        print(f"  {path.resolve()}")
    print()
    print("SUCCESS: repository-lean cleaned cohort files were written.")
    print(
        "No downstream predictor imputation was performed. Fit any remaining "
        "imputation inside each training fold only."
    )


if __name__ == "__main__":
    main()
