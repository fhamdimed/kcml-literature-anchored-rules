"""Cohort loading and reproducible data splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Primary phenotype predictors used in the manuscript analysis.  The cleaned
# cohort file may contain additional audit/QC columns, but they are not model
# inputs unless explicitly requested with --features.
DEFAULT_FEATURE_ORDER = ("MCV", "MCH", "HBA2", "HBF")
REQUIRED_RULE_FEATURES = ("MCV", "MCH", "HBA2", "HBF")


@dataclass
class Cohort:
    frame: pd.DataFrame
    features: list[str]
    X: pd.DataFrame
    y: pd.Series
    patient_ids: pd.Series


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    ids_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    ids_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    ids_test: pd.Series

    @property
    def feature_names(self) -> list[str]:
        return self.X_train.columns.tolist()

    def assignments(self) -> pd.DataFrame:
        pieces = []
        for name, ids, y in (
            ("train", self.ids_train, self.y_train),
            ("validation", self.ids_validation, self.y_validation),
            ("test", self.ids_test, self.y_test),
        ):
            pieces.append(
                pd.DataFrame(
                    {
                        "row_index": ids.index,
                        "patient_id": ids.to_numpy(),
                        "label": y.to_numpy(dtype=int),
                        "split": name,
                    }
                )
            )
        return pd.concat(pieces, ignore_index=True)


def _coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _validate_cleaned_hbf_contract(df: pd.DataFrame) -> None:
    """Validate optional audit flags written by prepare_thalassemia_dataset.py.

    These columns are not predictors.  They are checked only to prevent an
    accidentally inconsistent cleaned file from entering model training.
    """
    if "HBF_zero_from_blank" in df.columns:
        flag = pd.to_numeric(df["HBF_zero_from_blank"], errors="coerce").fillna(0)
        flagged = flag != 0
        inconsistent = flagged & (df["HBF"].isna() | ~np.isclose(df["HBF"], 0.0))
        if inconsistent.any():
            raise ValueError(
                "Cleaned-data inconsistency: HBF_zero_from_blank=1 but HBF is "
                f"not zero for {int(inconsistent.sum())} rows"
            )
        print(f"HbF blanks interpreted as zero: {int(flagged.sum())}")

    if "HBF_unresolved" in df.columns:
        flag = pd.to_numeric(df["HBF_unresolved"], errors="coerce").fillna(0)
        flagged = flag != 0
        inconsistent = flagged & df["HBF"].notna()
        if inconsistent.any():
            raise ValueError(
                "Cleaned-data inconsistency: HBF_unresolved=1 but HBF is present "
                f"for {int(inconsistent.sum())} rows"
            )
        print(f"Unresolved HbF values retained for fold-wise imputation: {int(flagged.sum())}")

    if "Hb_sum_anomaly" in df.columns:
        flag = pd.to_numeric(df["Hb_sum_anomaly"], errors="coerce").fillna(0)
        print(f"Hb fraction-sum QC anomalies retained: {int((flag != 0).sum())}")


def load_cohort_data(
    filepath: str | Path,
    feature_order: Sequence[str] = DEFAULT_FEATURE_ORDER,
) -> Cohort:
    """Load the cleaned phenotype cohort without re-cleaning it.

    The expected primary input is ``thalassemia_model_matrix_clean.csv`` from
    ``prepare_thalassemia_dataset.py``.  Blank HbF values already interpreted
    as zero are therefore ordinary numeric zeros.  Remaining unresolved model
    values are retained and imputed later using medians learned from the
    training partition only.

    Clinical rules are evaluated on the original, non-imputed DataFrame passed
    to each model.  A threshold rule requiring a genuinely unavailable value is
    inactive for that patient.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    if "label" not in df.columns:
        raise ValueError("Target column 'label' was not found")

    requested_features = list(feature_order)
    if not requested_features:
        raise ValueError("feature_order must contain at least one feature")
    if len(set(requested_features)) != len(requested_features):
        raise ValueError("feature_order contains duplicate feature names")

    missing_columns = [name for name in requested_features if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Requested model features were not found: {missing_columns}")

    missing_rule_columns = [
        column for column in REQUIRED_RULE_FEATURES if column not in df.columns
    ]
    if missing_rule_columns:
        raise ValueError(
            "Primary rules require these columns in the cleaned CSV: "
            f"{missing_rule_columns}"
        )

    # The current architecture uses the same original-unit frame for model
    # features and rule masks.  Sensitivity feature sets may add HGB, but must
    # retain the four rule variables.
    omitted_rule_features = [
        column for column in REQUIRED_RULE_FEATURES if column not in requested_features
    ]
    if omitted_rule_features:
        raise ValueError(
            "Every experiment must retain the four rule variables. Missing from "
            f"--features: {omitted_rule_features}"
        )

    numeric_candidates = set(requested_features) | {
        "label",
        "HBA",
        "HGB",
        "Hb_sum",
        "HBF_zero_from_blank",
        "HBF_unresolved",
        "Hb_sum_anomaly",
    }
    _coerce_numeric(df, sorted(numeric_candidates.intersection(df.columns)))
    _validate_cleaned_hbf_contract(df)

    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    missing_target_count = before - len(df)
    if missing_target_count:
        print(f"Dropped {missing_target_count} rows with missing target")

    if not df["label"].isin([0, 1]).all():
        invalid = sorted(df.loc[~df["label"].isin([0, 1]), "label"].unique())
        raise ValueError(f"label must contain only 0 and 1; found {invalid}")
    df["label"] = df["label"].astype(int)

    # A row with no value for any selected predictor cannot support prediction.
    no_measurement = df[requested_features].isna().all(axis=1)
    if no_measurement.any():
        print(
            f"Dropped {int(no_measurement.sum())} rows with no observed selected predictor"
        )
        df = df.loc[~no_measurement].copy()

    if df.empty:
        raise ValueError("No rows remain after loading the cleaned cohort")
    if df["label"].nunique() != 2:
        raise ValueError("The cleaned cohort must contain both classes")

    X = df[requested_features].astype(float).copy()
    y = df["label"].astype(int).copy()
    if "patient_id" in df.columns:
        patient_ids = df["patient_id"].copy()
        duplicated = patient_ids.notna() & patient_ids.duplicated(keep=False)
        if duplicated.any():
            raise ValueError(
                f"Found {int(duplicated.sum())} rows with duplicated patient IDs; "
                "resolve duplicates or use grouped splitting before training"
            )
    else:
        patient_ids = pd.Series(df.index, index=df.index, name="patient_id")

    missing_counts = X.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    print(f"Loaded {len(df)} patients from {filepath}")
    print(f"Features: {requested_features}")
    print(f"Class-1 prevalence: {y.mean():.4f}")
    if len(missing_counts):
        print("Missing values retained for training-only median imputation:")
        for column, count in missing_counts.items():
            print(f"  {column}: {int(count)} ({100.0 * count / len(df):.2f}%)")
    else:
        print("No missing model-feature values remain")

    return Cohort(df, requested_features, X, y, patient_ids)


def make_holdout_splits(
    cohort: Cohort,
    test_size: float = 0.20,
    validation_fraction_of_total: float = 0.20,
    random_state: int = 42,
) -> DataSplits:
    """Create stratified train/validation/test splits.

    ``validation_fraction_of_total`` is expressed relative to the full cohort.
    With both values equal to 0.20, the resulting proportions are 60/20/20.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    if not 0 < validation_fraction_of_total < 1 - test_size:
        raise ValueError(
            "validation_fraction_of_total must be in (0, 1 - test_size)"
        )

    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = (
        train_test_split(
            cohort.X,
            cohort.y,
            cohort.patient_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=cohort.y,
        )
    )

    validation_relative = validation_fraction_of_total / (1.0 - test_size)
    X_train, X_validation, y_train, y_validation, ids_train, ids_validation = (
        train_test_split(
            X_train_val,
            y_train_val,
            ids_train_val,
            test_size=validation_relative,
            random_state=random_state,
            stratify=y_train_val,
        )
    )

    return DataSplits(
        X_train=X_train.copy(),
        y_train=y_train.copy(),
        ids_train=ids_train.copy(),
        X_validation=X_validation.copy(),
        y_validation=y_validation.copy(),
        ids_validation=ids_validation.copy(),
        X_test=X_test.copy(),
        y_test=y_test.copy(),
        ids_test=ids_test.copy(),
    )


def make_splits_from_indices(
    cohort: Cohort,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> DataSplits:
    """Create ``DataSplits`` from positional row indices."""
    return DataSplits(
        X_train=cohort.X.iloc[train_indices].copy(),
        y_train=cohort.y.iloc[train_indices].copy(),
        ids_train=cohort.patient_ids.iloc[train_indices].copy(),
        X_validation=cohort.X.iloc[validation_indices].copy(),
        y_validation=cohort.y.iloc[validation_indices].copy(),
        ids_validation=cohort.patient_ids.iloc[validation_indices].copy(),
        X_test=cohort.X.iloc[test_indices].copy(),
        y_test=cohort.y.iloc[test_indices].copy(),
        ids_test=cohort.patient_ids.iloc[test_indices].copy(),
    )
