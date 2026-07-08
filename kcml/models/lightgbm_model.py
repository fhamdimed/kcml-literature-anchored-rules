"""Rule-penalized LightGBM implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..preprocessing import MedianFeatureImputer
from ..rules import (
    RuleCatalog,
    bce_from_logits,
    rule_gradient_hessian,
    rule_loss_from_logits,
    sigmoid,
)


class RulePenalizedLightGBM:
    def __init__(
        self,
        penalty_multiplier: float,
        enabled_rules: Iterable[str] | None = None,
        rule_weights: Mapping[str, float] | None = None,
        rule_control: str = "none",
        random_state: int = 42,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 1,
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
        self.num_leaves = int(num_leaves)
        self.max_depth = int(max_depth)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.feature_fraction = float(feature_fraction)
        self.bagging_fraction = float(bagging_fraction)
        self.bagging_freq = int(bagging_freq)
        self.reg_lambda = float(reg_lambda)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.verbose = bool(verbose)
        self.catalog = RuleCatalog()
        self.imputer = MedianFeatureImputer()
        self.feature_names: list[str] | None = None
        self.model: lgb.Booster | None = None
        self.evals_result: dict[str, dict[str, list[float]]] = {}

    def _make_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        reference: lgb.Dataset | None = None,
    ) -> lgb.Dataset:
        if self.feature_names is None:
            raise ValueError("Feature names have not been initialized")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X_original = X[self.feature_names].apply(pd.to_numeric, errors="coerce")
        X_model = self.imputer.transform_frame(X_original)
        dataset = lgb.Dataset(
            X_model,
            label=np.asarray(y, dtype=float),
            feature_name=self.feature_names,
            reference=reference,
            free_raw_data=False,
        )
        dataset.rule_bundle = self.catalog.build_bundle(
            X_original,
            enabled_rules=self.enabled_rules,
            weight_overrides=self.rule_weights,
            control=self.rule_control,
            random_state=self.random_state,
        )
        return dataset

    def _objective(self, raw_scores: np.ndarray, dataset: lgb.Dataset):
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

    def _metric(self, raw_scores: np.ndarray, dataset: lgb.Dataset):
        labels = dataset.get_label().astype(float)
        data_loss = float(np.mean(bce_from_logits(raw_scores, labels)))
        combined = data_loss + self.penalty_multiplier * rule_loss_from_logits(
            raw_scores, dataset.rule_bundle
        )
        return "combined_loss", float(combined), False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> "RulePenalizedLightGBM":
        self.feature_names = X.columns.tolist()
        self.imputer.fit(X, self.feature_names)
        train_dataset = self._make_dataset(X, y)
        validation_dataset = self._make_dataset(
            X_validation, y_validation, reference=train_dataset
        )

        params = {
            "objective": self._objective,
            "metric": "None",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.reg_lambda,
            "boost_from_average": False,
            "seed": self.random_state,
            "feature_fraction_seed": self.random_state,
            "bagging_seed": self.random_state,
            "data_random_seed": self.random_state,
            "verbosity": 1 if self.verbose else -1,
        }
        callbacks = [
            lgb.record_evaluation(self.evals_result),
            lgb.log_evaluation(period=100 if self.verbose else 0),
        ]
        if self.early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    first_metric_only=True,
                    verbose=self.verbose,
                )
            )

        self.evals_result = {}
        callbacks[0] = lgb.record_evaluation(self.evals_result)
        self.model = lgb.train(
            params=params,
            train_set=train_dataset,
            num_boost_round=self.n_estimators,
            valid_sets=[train_dataset, validation_dataset],
            valid_names=["train", "validation"],
            feval=self._metric,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        if self.feature_names is None:
            raise ValueError("Feature names are unavailable")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X_model = self.imputer.transform_frame(X[self.feature_names])
        raw_scores = self.model.predict(
            X_model,
            raw_score=True,
            num_iteration=self.model.best_iteration or self.model.current_iteration(),
        )
        return sigmoid(raw_scores)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(
            str(path) + ".model.txt",
            num_iteration=self.model.best_iteration or self.model.current_iteration(),
        )
        metadata = {
            "algorithm": "lightgbm",
            "penalty_multiplier": self.penalty_multiplier,
            "enabled_rules": self.enabled_rules,
            "rule_weights": self.rule_weights,
            "rule_control": self.rule_control,
            "feature_names": self.feature_names,
            "imputation_strategy": "training_median",
            "imputer_statistics": self.imputer.statistics_.tolist(),
            "best_iteration": self.model.best_iteration,
        }
        Path(str(path) + ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
