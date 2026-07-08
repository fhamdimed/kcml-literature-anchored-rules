"""Leakage-safe numeric preprocessing shared by all model families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


@dataclass
class MedianFeatureImputer:
    """Median-impute model features using statistics learned on training data only.

    Clinical rules must be evaluated on the original, non-imputed DataFrame.  This
    class is only for the matrix supplied to the predictive model.
    """

    feature_names: list[str] | None = None
    imputer: SimpleImputer | None = None

    def fit(self, X: pd.DataFrame, feature_names: Sequence[str] | None = None) -> "MedianFeatureImputer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        names = list(feature_names) if feature_names is not None else X.columns.tolist()
        missing = [name for name in names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        numeric = X[names].apply(pd.to_numeric, errors="coerce")
        self.feature_names = names
        self.imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )
        self.imputer.fit(numeric)
        return self

    def transform_array(self, X: pd.DataFrame, dtype: np.dtype | type = float) -> np.ndarray:
        if self.imputer is None or self.feature_names is None:
            raise ValueError("The imputer has not been fitted")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        numeric = X[self.feature_names].apply(pd.to_numeric, errors="coerce")
        transformed = self.imputer.transform(numeric)
        return np.asarray(transformed, dtype=dtype)

    def transform_frame(self, X: pd.DataFrame, dtype: np.dtype | type = float) -> pd.DataFrame:
        values = self.transform_array(X, dtype=dtype)
        assert self.feature_names is not None
        return pd.DataFrame(values, columns=self.feature_names, index=X.index)

    @property
    def statistics_(self) -> np.ndarray:
        if self.imputer is None:
            raise ValueError("The imputer has not been fitted")
        return np.asarray(self.imputer.statistics_, dtype=float)
