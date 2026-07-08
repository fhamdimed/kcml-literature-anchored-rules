"""Literature-anchored phenotype rules and shared rule-penalty utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleBundle:
    """Numerical representation of rules for one ordered patient matrix."""

    mask: np.ndarray  # shape: (n_patients, n_rules), values in {0, 1}
    targets: np.ndarray  # shape: (n_rules,), values in {0, 1}
    weights: np.ndarray  # shape: (n_rules,), non-negative
    names: tuple[str, ...]

    def validate(self, n_rows: int | None = None) -> None:
        if self.mask.ndim != 2:
            raise ValueError("rule mask must be a 2D array")
        if n_rows is not None and self.mask.shape[0] != n_rows:
            raise ValueError("rule mask row count does not match the data")
        n_rules = self.mask.shape[1]
        if self.targets.shape != (n_rules,):
            raise ValueError("rule targets must have shape (n_rules,)")
        if self.weights.shape != (n_rules,):
            raise ValueError("rule weights must have shape (n_rules,)")
        if len(self.names) != n_rules:
            raise ValueError("rule names do not match the mask width")
        if np.any(self.weights < 0):
            raise ValueError("rule weights must be non-negative")


class RuleCatalog:
    """Central definition of the literature-anchored phenotype rules."""

    def __init__(self) -> None:
        # penalty_if_negative: when the rule fires, class 0 is discouraged,
        # so the soft rule target is class 1.
        # penalty_if_positive: when the rule fires, class 1 is discouraged,
        # so the soft rule target is class 0.
        self.primary_rules: dict[str, dict[str, object]] = {
            "LR01": {
                "condition": lambda df: df["HBA2"] <= 3.5,
                "interpretation": "Low HbA2 - atypical for beta-carrier",
                "penalty_if_negative": 1.0,
                "weight": 1.0,
            },
            "LR02": {
                "condition": lambda df: df["HBF"] >= 5.0,
                "interpretation": "Elevated HbF - possible HPFH/delta-beta",
                "penalty_if_positive": 1.0,
                "weight": 1.0,
            },
            "LR03": {
                "condition": lambda df: (df["MCV"] >= 79) | (df["MCH"] >= 27),
                "interpretation": "Normalized indices - possible alpha coinheritance",
                "penalty_if_negative": 1.0,
                "weight": 1.0,
            },
            "LR04": {
                "condition": lambda df: (df["MCV"] < 55) | (df["MCH"] < 15),
                "interpretation": "Extreme phenotype - severe modifier or unusual genotype",
                "penalty_if_negative": 1.0,
                "weight": 1.0,
            },
            "LR05": {
                "condition": lambda df: (
                    (df["MCV"] >= 70)
                    & (df["MCH"] >= 22)
                    & (df["HBA2"] > 3.5)
                ),
                "interpretation": "Literature-anchored alpha-modifier phenotype",
                "penalty_if_negative": 1.0,
                "weight": 1.5,
            },
        }

        self.composite_rules: dict[str, dict[str, object]] = {
            "CR01": {
                "condition": lambda df: (
                    ((df["MCV"] >= 79) | (df["MCH"] >= 27))
                    & (df["HBA2"] <= 3.5)
                ),
                "interpretation": "Normalized indices with low HbA2",
            },
            "CR02": {
                "condition": lambda df: (
                    ((df["MCV"] >= 79) | (df["MCH"] >= 27))
                    & (df["HBF"] >= 5.0)
                ),
                "interpretation": "Normalized indices with high HbF",
            },
        }

        self.quality_rules: dict[str, dict[str, object]] = {
            "QC01": {
                "condition": self._check_hb_sum,
                "interpretation": "Hemoglobin sum outside tolerance",
            }
        }

        self.same_source_rules: dict[str, dict[str, object]] = {
            "LR06": {
                "condition": lambda df: (df["MCV"] > 66.47) | (df["MCH"] > 21.59),
                "interpretation": "Huizhou high-index alpha-beta screen",
            },
            "LR07": {
                "condition": lambda df: (df["MCV"] <= 56.70) | (df["MCH"] <= 18.30),
                "interpretation": "Huizhou severe-index pattern",
            },
        }

    @property
    def primary_names(self) -> tuple[str, ...]:
        return tuple(self.primary_rules)

    @staticmethod
    def _apply_rule(name: str, rule: Mapping[str, object], df: pd.DataFrame) -> pd.Series:
        try:
            condition = rule["condition"]
            result = condition(df)  # type: ignore[operator]
            return pd.Series(result, index=df.index).fillna(False).astype(bool)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"Warning: rule {name} failed: {exc}")
            return pd.Series(False, index=df.index, dtype=bool)

    @staticmethod
    def _check_hb_sum(df: pd.DataFrame) -> pd.Series:
        required = ["HBA", "HBA2", "HBF"]
        if not all(column in df.columns for column in required):
            return pd.Series(False, index=df.index, dtype=bool)
        complete = ~df[required].isna().any(axis=1)
        total_ok = (df["HBA"] + df["HBA2"] + df["HBF"]).between(97, 103)
        return complete & ~total_ok

    def _resolve_primary_names(self, enabled_rules: Iterable[str] | None) -> tuple[str, ...]:
        if enabled_rules is None:
            return self.primary_names
        names = tuple(enabled_rules)
        unknown = sorted(set(names) - set(self.primary_rules))
        if unknown:
            raise ValueError(f"Unknown primary rule names: {unknown}")
        if len(set(names)) != len(names):
            raise ValueError("enabled_rules contains duplicates")
        return names

    def apply_primary_rules(
        self,
        df: pd.DataFrame,
        enabled_rules: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        names = self._resolve_primary_names(enabled_rules)
        result = pd.DataFrame(index=df.index)
        for name in names:
            result[f"{name}_fires"] = self._apply_rule(
                name, self.primary_rules[name], df
            ).astype(np.int8)
        return result

    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        result: dict[str, pd.Series] = {}
        for catalog in (
            self.primary_rules,
            self.composite_rules,
            self.quality_rules,
            self.same_source_rules,
        ):
            for name, rule in catalog.items():
                result[name] = self._apply_rule(name, rule, df)
        return pd.DataFrame(result, index=df.index)

    def build_bundle(
        self,
        df: pd.DataFrame,
        enabled_rules: Iterable[str] | None = None,
        weight_overrides: Mapping[str, float] | None = None,
        control: str = "none",
        random_state: int = 42,
    ) -> RuleBundle:
        """Create the shared mask, target, and weight representation.

        Controls are intended for negative-control experiments:
        - ``none``: use the true patient-rule assignments and targets.
        - ``permuted``: permute patient rows of the rule mask.
        - ``column_permuted``: independently permute each rule column.
        - ``reversed``: reverse every rule target while retaining fire masks.
        """

        names = self._resolve_primary_names(enabled_rules)
        fired = self.apply_primary_rules(df, names)
        mask = fired[[f"{name}_fires" for name in names]].to_numpy(dtype=float)

        targets: list[float] = []
        weights: list[float] = []
        overrides = dict(weight_overrides or {})

        for name in names:
            rule = self.primary_rules[name]
            has_negative = "penalty_if_negative" in rule
            has_positive = "penalty_if_positive" in rule
            if has_negative == has_positive:
                raise ValueError(
                    f"Rule {name} must define exactly one penalty direction"
                )
            targets.append(1.0 if has_negative else 0.0)
            weight = float(overrides.get(name, rule.get("weight", 1.0)))
            if weight < 0:
                raise ValueError(f"Rule weight for {name} must be non-negative")
            weights.append(weight)

        targets_array = np.asarray(targets, dtype=float)
        weights_array = np.asarray(weights, dtype=float)
        rng = np.random.default_rng(random_state)

        if control == "none":
            pass
        elif control == "permuted":
            if len(mask):
                mask = mask[rng.permutation(len(mask))]
        elif control == "column_permuted":
            for column in range(mask.shape[1]):
                mask[:, column] = mask[rng.permutation(len(mask)), column]
        elif control == "reversed":
            targets_array = 1.0 - targets_array
        else:
            raise ValueError(
                "control must be one of: none, permuted, column_permuted, reversed"
            )

        bundle = RuleBundle(
            mask=mask,
            targets=targets_array,
            weights=weights_array,
            names=names,
        )
        bundle.validate(len(df))
        return bundle


def sigmoid(raw_scores: np.ndarray | Sequence[float]) -> np.ndarray:
    raw = np.clip(np.asarray(raw_scores, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-raw))


def logits_from_probabilities(probabilities: np.ndarray | Sequence[float]) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(probabilities) - np.log1p(-probabilities)


def bce_from_logits(raw_scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise binary cross-entropy."""
    raw_scores = np.asarray(raw_scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    return np.logaddexp(0.0, raw_scores) - targets * raw_scores


def rule_loss_from_logits(raw_scores: np.ndarray, bundle: RuleBundle) -> float:
    """Patient-normalized weighted rule BCE used by every model family."""
    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    bundle.validate(len(raw_scores))
    if bundle.mask.shape[1] == 0 or len(raw_scores) == 0:
        return 0.0
    losses = bce_from_logits(
        raw_scores[:, None],
        bundle.targets[None, :],
    )
    weighted = losses * bundle.mask * bundle.weights[None, :]
    return float(weighted.sum() / len(raw_scores))


def rule_gradient_hessian(
    raw_scores: np.ndarray,
    bundle: RuleBundle,
    penalty_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Derivative of lambda times the shared rule BCE with respect to logits."""
    raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    bundle.validate(len(raw_scores))
    if penalty_multiplier < 0:
        raise ValueError("penalty_multiplier must be non-negative")
    probabilities = sigmoid(raw_scores)
    if bundle.mask.shape[1] == 0 or penalty_multiplier == 0:
        zeros = np.zeros_like(probabilities)
        return zeros, zeros

    strength = bundle.mask * bundle.weights[None, :]
    residuals = probabilities[:, None] - bundle.targets[None, :]
    gradient = penalty_multiplier * np.sum(strength * residuals, axis=1)
    logistic_hessian = probabilities * (1.0 - probabilities)
    hessian = penalty_multiplier * logistic_hessian * np.sum(strength, axis=1)
    return gradient, hessian
