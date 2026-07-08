"""Rule-penalized XGBoost implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

from ..preprocessing import MedianFeatureImputer
from ..rules import (
    RuleCatalog,
    bce_from_logits,
    rule_gradient_hessian,
    rule_loss_from_logits,
    sigmoid,
)


class RulePenalizedXGBoost:
    def __init__(
        self,
        penalty_multiplier: float,
        enabled_rules: Iterable[str] | None = None,
        rule_weights: Mapping[str, float] | None = None,
        rule_control: str = "none",
        random_state: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        verbose: bool = True,
    ) -> None:
        if penalty_multiplier < 0:
            raise ValueError("penalty_multiplier must be non-negative")
        self.penalty_multiplier = float(penalty_multiplier)
        self.enabled_rules = tuple(enabled_rules) if enabled_rules is not None else None
        self.rule_weights = dict(rule_weights or {})
        self.rule_control = rule_control
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.verbose = bool(verbose)
        self.catalog = RuleCatalog()
        self.imputer = MedianFeatureImputer()
        self.feature_names: list[str] | None = None
        self.model: xgb.Booster | None = None
        self.evals_result: dict[str, dict[str, list[float]]] = {}

    def _make_dmatrix(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        if self.feature_names is None:
            raise ValueError("Feature names have not been initialized")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X_original = X[self.feature_names].apply(pd.to_numeric, errors="coerce")
        X_model = self.imputer.transform_frame(X_original)
        matrix = xgb.DMatrix(
            X_model,
            label=None if y is None else np.asarray(y, dtype=float),
            feature_names=self.feature_names,
        )
        matrix.rule_bundle = self.catalog.build_bundle(
            X_original,
            enabled_rules=self.enabled_rules,
            weight_overrides=self.rule_weights,
            control=self.rule_control,
            random_state=self.random_state,
        )
        return matrix

    def _objective(self, raw_scores: np.ndarray, dataset: xgb.DMatrix):
        labels = dataset.get_label().astype(float)
        probabilities = sigmoid(raw_scores)
        base_gradient = probabilities - labels
        base_hessian = probabilities * (1.0 - probabilities)
        rule_gradient, rule_hessian = rule_gradient_hessian(
            raw_scores,
            dataset.rule_bundle,
            self.penalty_multiplier,
        )
        return (
            base_gradient + rule_gradient,
            np.maximum(base_hessian + rule_hessian, 1e-8),
        )

    def _metric(self, raw_scores: np.ndarray, dataset: xgb.DMatrix):
        labels = dataset.get_label().astype(float)
        data_loss = float(np.mean(bce_from_logits(raw_scores, labels)))
        combined = data_loss + self.penalty_multiplier * rule_loss_from_logits(
            raw_scores, dataset.rule_bundle
        )
        return "combined_loss", float(combined)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> "RulePenalizedXGBoost":
        self.feature_names = X.columns.tolist()
        self.imputer.fit(X, self.feature_names)
        train_matrix = self._make_dmatrix(X, y)
        validation_matrix = self._make_dmatrix(X_validation, y_validation)

        params = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda,
            "tree_method": "hist",
            "seed": self.random_state,
            "base_score": 0.0,
            "disable_default_eval_metric": 1,
            "verbosity": 1 if self.verbose else 0,
        }
        callbacks = []
        if self.early_stopping_rounds > 0:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_rounds,
                    metric_name="combined_loss",
                    data_name="validation",
                    maximize=False,
                    save_best=True,
                )
            )

        self.evals_result = {}
        self.model = xgb.train(
            params=params,
            dtrain=train_matrix,
            num_boost_round=self.n_estimators,
            evals=[(train_matrix, "train"), (validation_matrix, "validation")],
            obj=self._objective,
            custom_metric=self._metric,
            evals_result=self.evals_result,
            callbacks=callbacks or None,
            verbose_eval=100 if self.verbose else False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        matrix = self._make_dmatrix(X)
        raw_scores = self.model.predict(matrix, output_margin=True)
        return sigmoid(raw_scores)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path) + ".model.json")
        metadata = {
            "algorithm": "xgboost",
            "penalty_multiplier": self.penalty_multiplier,
            "enabled_rules": self.enabled_rules,
            "rule_weights": self.rule_weights,
            "rule_control": self.rule_control,
            "feature_names": self.feature_names,
            "imputation_strategy": "training_median",
            "imputer_statistics": self.imputer.statistics_.tolist(),
            "boosted_rounds": self.model.num_boosted_rounds(),
        }
        Path(str(path) + ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
