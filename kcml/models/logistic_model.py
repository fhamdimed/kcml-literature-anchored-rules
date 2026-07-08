"""Convex rule-penalized logistic regression optimized with L-BFGS-B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from ..preprocessing import MedianFeatureImputer
from ..rules import RuleCatalog, bce_from_logits, rule_loss_from_logits, sigmoid


class RulePenalizedLogisticRegression:
    def __init__(
        self,
        penalty_multiplier: float,
        enabled_rules: Iterable[str] | None = None,
        rule_weights: Mapping[str, float] | None = None,
        rule_control: str = "none",
        random_state: int = 42,
        l2_strength: float = 1.0,
        max_iter: int = 5000,
        tolerance: float = 1e-8,
        verbose: bool = True,
    ) -> None:
        if penalty_multiplier < 0:
            raise ValueError("penalty_multiplier must be non-negative")
        if l2_strength < 0:
            raise ValueError("l2_strength must be non-negative")
        self.penalty_multiplier = float(penalty_multiplier)
        self.enabled_rules = tuple(enabled_rules) if enabled_rules is not None else None
        self.rule_weights = dict(rule_weights or {})
        self.rule_control = rule_control
        self.random_state = int(random_state)
        self.l2_strength = float(l2_strength)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.verbose = bool(verbose)
        self.catalog = RuleCatalog()
        self.imputer = MedianFeatureImputer()
        self.scaler = StandardScaler()
        self.feature_names: list[str] | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.optimization_result_: dict[str, object] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_validation: pd.DataFrame | None = None,
        y_validation: pd.Series | None = None,
    ) -> "RulePenalizedLogisticRegression":
        del X_validation, y_validation  # Convex optimizer does not require early stopping.
        self.feature_names = X.columns.tolist()
        X_original = X[self.feature_names].apply(pd.to_numeric, errors="coerce")
        self.imputer.fit(X_original, self.feature_names)
        X_imputed = self.imputer.transform_array(X_original)
        X_scaled = self.scaler.fit_transform(X_imputed)
        labels = np.asarray(y, dtype=float).reshape(-1)
        if len(labels) != len(X_scaled):
            raise ValueError("X and y have different row counts")

        bundle = self.catalog.build_bundle(
            X_original,
            enabled_rules=self.enabled_rules,
            weight_overrides=self.rule_weights,
            control=self.rule_control,
            random_state=self.random_state,
        )
        n_rows, n_features = X_scaled.shape

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = parameters[:n_features]
            intercept = parameters[n_features]
            raw_scores = X_scaled @ coefficients + intercept
            probabilities = sigmoid(raw_scores)

            data_loss = float(np.mean(bce_from_logits(raw_scores, labels)))
            score_gradient = probabilities - labels

            if bundle.mask.shape[1] and self.penalty_multiplier > 0:
                strength = bundle.mask * bundle.weights[None, :]
                rule_residual = probabilities[:, None] - bundle.targets[None, :]
                score_gradient = score_gradient + self.penalty_multiplier * np.sum(
                    strength * rule_residual, axis=1
                )
                rule_loss = rule_loss_from_logits(raw_scores, bundle)
            else:
                rule_loss = 0.0

            regularization = 0.5 * self.l2_strength * float(
                np.dot(coefficients, coefficients)
            )
            total_loss = (
                data_loss + self.penalty_multiplier * rule_loss + regularization
            )

            score_gradient /= n_rows
            coefficient_gradient = (
                X_scaled.T @ score_gradient + self.l2_strength * coefficients
            )
            intercept_gradient = float(score_gradient.sum())
            gradient = np.concatenate(
                [coefficient_gradient, np.asarray([intercept_gradient])]
            )
            return float(total_loss), gradient

        initial = np.zeros(n_features + 1, dtype=float)
        result = minimize(
            objective,
            x0=initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.max_iter,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
                "maxls": 50,
            },
        )
        if not result.success:
            raise RuntimeError(
                "Logistic regression optimization failed: "
                f"status={result.status}, message={result.message}"
            )

        self.coef_ = result.x[:n_features].copy()
        self.intercept_ = float(result.x[n_features])
        self.optimization_result_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "final_objective": float(result.fun),
            "gradient_norm": float(np.linalg.norm(result.jac)),
        }
        if self.verbose:
            print(
                "Logistic optimization complete: "
                f"iterations={result.nit}, objective={result.fun:.6f}"
            )
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None or self.feature_names is None:
            raise ValueError("Model has not been fitted")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        X_imputed = self.imputer.transform_array(X[self.feature_names])
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled @ self.coef_ + self.intercept_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def save(self, path: str | Path) -> None:
        if self.coef_ is None or self.intercept_ is None or self.feature_names is None:
            raise ValueError("Model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path) + ".model.npz",
            coefficients=self.coef_,
            intercept=np.asarray([self.intercept_]),
            imputer_statistics=self.imputer.statistics_,
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            feature_names=np.asarray(self.feature_names, dtype=object),
        )
        metadata = {
            "algorithm": "logistic_regression",
            "penalty_multiplier": self.penalty_multiplier,
            "enabled_rules": self.enabled_rules,
            "rule_weights": self.rule_weights,
            "rule_control": self.rule_control,
            "l2_strength": self.l2_strength,
            "feature_names": self.feature_names,
            "imputation_strategy": "training_median",
            "imputer_statistics": self.imputer.statistics_.tolist(),
            "optimization": self.optimization_result_,
        }
        Path(str(path) + ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def coefficient_table(self) -> pd.DataFrame:
        if self.coef_ is None or self.feature_names is None:
            raise ValueError("Model has not been fitted")
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "standardized_coefficient": self.coef_,
                "odds_ratio_per_sd": np.exp(self.coef_),
            }
        ).sort_values("standardized_coefficient", ascending=False)
