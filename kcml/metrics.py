"""Shared predictive, calibration, threshold, and rule-consistency metrics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .rules import (
    RuleBundle,
    RuleCatalog,
    bce_from_logits,
    logits_from_probabilities,
    rule_loss_from_logits,
    sigmoid,
)


def calibration_intercept_slope(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Estimate calibration intercept and slope by logistic recalibration."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    original_logits = logits_from_probabilities(p)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        recalibrated_logits = intercept + slope * original_logits
        q = sigmoid(recalibrated_logits)
        loss = float(np.mean(bce_from_logits(recalibrated_logits, y)))
        residual = q - y
        gradient = np.asarray(
            [np.mean(residual), np.mean(residual * original_logits)],
            dtype=float,
        )
        return loss, gradient

    result = minimize(
        objective,
        x0=np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


def choose_balanced_accuracy_threshold(
    y_true: Iterable[int],
    probabilities: np.ndarray,
) -> float:
    """Select a validation threshold that maximizes balanced accuracy.

    This is equivalent to maximizing Youden's index. Candidate thresholds are
    the exact empirical ROC thresholds. When several thresholds have the same
    balanced accuracy (within numerical precision), the threshold closest to
    0.5 is selected, with the larger threshold used as the final deterministic
    tie-breaker. Rule violations are deliberately not used for threshold
    selection, so threshold tuning remains a purely predictive procedure.
    """
    y = np.asarray(list(y_true), dtype=int).reshape(-1)
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-8, 1 - 1e-8)
    if len(y) != len(p):
        raise ValueError("y_true and probabilities must have equal length")
    if len(y) == 0 or np.unique(y).size < 2:
        return 0.5

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y,
        p,
        drop_intermediate=False,
    )
    balanced_accuracy = 0.5 * (true_positive_rate + (1.0 - false_positive_rate))
    best_value = float(np.nanmax(balanced_accuracy))
    candidate_indices = np.flatnonzero(
        np.isclose(balanced_accuracy, best_value, rtol=0.0, atol=1e-12)
    )

    candidates: list[tuple[float, float, float]] = []
    for index in candidate_indices:
        threshold = float(thresholds[index])
        if not np.isfinite(threshold):
            # Probabilities are clipped below 1, so threshold 1 gives the same
            # all-negative classification represented by the infinite ROC cut.
            threshold = 1.0
        threshold = float(np.clip(threshold, 0.0, 1.0))
        candidates.append((abs(threshold - 0.5), -threshold, threshold))

    candidates.sort()
    return float(candidates[0][2])


def calculate_rule_violations(
    predictions: np.ndarray,
    bundle: RuleBundle,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Return per-rule counts, patient flags, and patient-by-rule violation mask."""
    predictions = np.asarray(predictions, dtype=int).reshape(-1)
    bundle.validate(len(predictions))
    if bundle.mask.shape[1] == 0:
        empty = np.zeros((len(predictions), 0), dtype=bool)
        return {}, np.zeros(len(predictions), dtype=bool), empty

    predicted_matrix = predictions[:, None]
    target_matrix = bundle.targets[None, :]
    violations = (bundle.mask > 0) & (predicted_matrix != target_matrix)
    counts = {
        f"{name}_{'neg' if target == 1 else 'pos'}": int(violations[:, index].sum())
        for index, (name, target) in enumerate(zip(bundle.names, bundle.targets))
    }
    any_violation = violations.any(axis=1)
    return counts, any_violation, violations


def evaluate_predictions(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    X_original: pd.DataFrame,
    threshold: float = 0.5,
    rule_catalog: RuleCatalog | None = None,
    enabled_rules: Iterable[str] | None = None,
    weight_overrides: dict[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate a model using the true, non-permuted clinical rules."""
    y = np.asarray(list(y_true), dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(y) != len(p) or len(y) != len(X_original):
        raise ValueError("y_true, probabilities, and X_original must have equal length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    p = np.clip(p, 1e-8, 1 - 1e-8)
    predictions = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }
    if np.unique(y).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, p))
        metrics["average_precision"] = float(average_precision_score(y, p))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    intercept, slope = calibration_intercept_slope(y, p)
    metrics["calibration_intercept"] = intercept
    metrics["calibration_slope"] = slope

    catalog = rule_catalog or RuleCatalog()
    bundle = catalog.build_bundle(
        X_original,
        enabled_rules=enabled_rules,
        weight_overrides=weight_overrides,
        control="none",
    )
    counts, any_violation, violation_matrix = calculate_rule_violations(
        predictions, bundle
    )
    total_violations = int(violation_matrix.sum())
    weighted_activation = bundle.mask * bundle.weights[None, :]
    active_rule_weight = float(weighted_activation.sum())
    active_rule_count = int(bundle.mask.sum())

    # Threshold-free directional discordance.  For a positive-target rule the
    # contradiction is (1 - p); for a negative-target rule it is p.  The
    # weighted average is bounded in [0, 1] and is directly interpretable as
    # the mean probability mass assigned against active literature rules.
    directional_discordance = np.abs(p[:, None] - bundle.targets[None, :])
    weighted_soft_matrix = directional_discordance * weighted_activation
    soft_rule_violation = (
        float(weighted_soft_matrix.sum() / active_rule_weight)
        if active_rule_weight
        else 0.0
    )
    unweighted_soft_rule_violation = (
        float((directional_discordance * bundle.mask).sum() / active_rule_count)
        if active_rule_count
        else 0.0
    )
    patient_active_weight = weighted_activation.sum(axis=1)
    patient_soft_score = np.divide(
        weighted_soft_matrix.sum(axis=1),
        patient_active_weight,
        out=np.zeros(len(y), dtype=float),
        where=patient_active_weight > 0,
    )
    active_patients = patient_active_weight > 0

    raw_scores = logits_from_probabilities(p)
    metrics.update(
        {
            "patient_violation_rate": float(any_violation.mean()),
            "total_rule_patient_violations": total_violations,
            "violations_per_patient": float(total_violations / len(y)),
            "active_rule_count": active_rule_count,
            "active_rule_weight": active_rule_weight,
            "soft_rule_violation": soft_rule_violation,
            "unweighted_soft_rule_violation": unweighted_soft_rule_violation,
            "mean_patient_soft_rule_violation": (
                float(patient_soft_score[active_patients].mean())
                if np.any(active_patients)
                else 0.0
            ),
            "patients_with_active_rules": int(active_patients.sum()),
            "rule_bce_per_patient": float(rule_loss_from_logits(raw_scores, bundle)),
            "active_rule_violation_rate": (
                float(total_violations / active_rule_count)
                if active_rule_count
                else 0.0
            ),
        }
    )
    for name, count in counts.items():
        metrics[f"violation_{name}"] = count
    for index, name in enumerate(bundle.names):
        active = bundle.mask[:, index] > 0
        metrics[f"soft_violation_{name}"] = (
            float(directional_discordance[active, index].mean())
            if np.any(active)
            else 0.0
        )

    detail = pd.DataFrame(
        {
            "true_label": y,
            "predicted_probability": p,
            "predicted_label": predictions,
            "any_rule_violation": any_violation.astype(int),
            "soft_rule_violation_score": patient_soft_score,
            "active_rule_weight_patient": patient_active_weight,
        },
        index=X_original.index,
    )
    for index, name in enumerate(bundle.names):
        detail[f"flag_{name}"] = bundle.mask[:, index].astype(int)
        detail[f"violation_{name}"] = violation_matrix[:, index].astype(int)
        detail[f"soft_violation_{name}"] = (
            directional_discordance[:, index] * bundle.mask[:, index]
        )

    return metrics, detail


def _passes_floor(value: float, baseline: float, tolerance: float) -> bool:
    """Return whether value is no more than tolerance below baseline."""
    if np.isnan(baseline):
        return True
    if np.isnan(value):
        return False
    return bool(value >= baseline - tolerance)


def choose_penalty_constrained(
    validation_rows: pd.DataFrame,
    balanced_accuracy_tolerance: float = 0.01,
    auc_tolerance: float = 0.01,
    average_precision_tolerance: float = 0.02,
    minimum_violation_reduction: float = 0.40,
    minimum_soft_violation_reduction: float = 0.0,
) -> tuple[float, pd.DataFrame, str]:
    """Choose the smallest adequate lambda using validation data only.

    A penalized model is eligible when all of the following hold:

    * balanced accuracy is within ``balanced_accuracy_tolerance`` of the best;
    * ROC AUC is no more than ``auc_tolerance`` below the lambda-zero model;
    * average precision is no more than ``average_precision_tolerance`` below
      the lambda-zero model;
    * patient-level rule violations are reduced by at least
      ``minimum_violation_reduction`` relative to lambda zero;
    * threshold-free soft rule discordance is not worsened (or is reduced by
      ``minimum_soft_violation_reduction`` when a positive value is supplied).

    The smallest eligible penalty is selected. If no penalized model satisfies
    every condition, the unpenalized baseline is retained. The returned audit
    table records every eligibility component for transparent reporting.
    """
    if validation_rows.empty:
        raise ValueError("validation_rows is empty")
    required = {
        "penalty_multiplier",
        "balanced_accuracy",
        "patient_violation_rate",
        "roc_auc",
        "average_precision",
        "soft_rule_violation",
    }
    missing = required - set(validation_rows.columns)
    if missing:
        raise ValueError(f"Missing validation columns: {sorted(missing)}")
    if not 0 <= minimum_violation_reduction <= 1:
        raise ValueError("minimum_violation_reduction must be between 0 and 1")
    if not 0 <= minimum_soft_violation_reduction <= 1:
        raise ValueError(
            "minimum_soft_violation_reduction must be between 0 and 1"
        )

    table = validation_rows.copy().sort_values("penalty_multiplier").reset_index(drop=True)
    baseline_rows = table[np.isclose(table["penalty_multiplier"], 0.0)]
    if baseline_rows.empty:
        raise ValueError("The lambda grid must include 0 for constrained selection")
    baseline = baseline_rows.iloc[0]
    best_accuracy = float(table["balanced_accuracy"].max())
    baseline_violation_rate = float(baseline["patient_violation_rate"])
    baseline_soft_violation = float(baseline["soft_rule_violation"])

    if baseline_violation_rate > 0:
        table["violation_reduction_fraction"] = (
            baseline_violation_rate - table["patient_violation_rate"]
        ) / baseline_violation_rate
    else:
        table["violation_reduction_fraction"] = np.where(
            table["patient_violation_rate"] <= 0,
            0.0,
            -np.inf,
        )


    if baseline_soft_violation > 0:
        table["soft_violation_reduction_fraction"] = (
            baseline_soft_violation - table["soft_rule_violation"]
        ) / baseline_soft_violation
    else:
        table["soft_violation_reduction_fraction"] = np.where(
            table["soft_rule_violation"] <= 0,
            0.0,
            -np.inf,
        )

    table["passes_balanced_accuracy"] = (
        table["balanced_accuracy"] >= best_accuracy - balanced_accuracy_tolerance
    )
    table["passes_auc"] = [
        _passes_floor(float(value), float(baseline["roc_auc"]), auc_tolerance)
        for value in table["roc_auc"]
    ]
    table["passes_average_precision"] = [
        _passes_floor(
            float(value),
            float(baseline["average_precision"]),
            average_precision_tolerance,
        )
        for value in table["average_precision"]
    ]
    table["passes_violation_reduction"] = (
        table["violation_reduction_fraction"] >= minimum_violation_reduction
    )
    table["passes_soft_violation_reduction"] = (
        table["soft_violation_reduction_fraction"]
        >= minimum_soft_violation_reduction
    )
    table["is_penalized"] = table["penalty_multiplier"] > 0
    table["eligible"] = (
        table["is_penalized"]
        & table["passes_balanced_accuracy"]
        & table["passes_auc"]
        & table["passes_average_precision"]
        & table["passes_violation_reduction"]
        & table["passes_soft_violation_reduction"]
    )

    eligible = table[table["eligible"]].copy()
    if eligible.empty:
        selected_lambda = 0.0
        reason = (
            "No penalized lambda satisfied all validation constraints; "
            "the unpenalized baseline was retained."
        )
    else:
        selected_lambda = float(eligible["penalty_multiplier"].min())
        reason = (
            "Selected the smallest penalized lambda satisfying balanced-accuracy, "
            "ROC-AUC, average-precision, and violation-reduction constraints."
        )

    table["selected"] = np.isclose(table["penalty_multiplier"], selected_lambda)
    table["best_validation_balanced_accuracy"] = best_accuracy
    table["baseline_validation_roc_auc"] = float(baseline["roc_auc"])
    table["baseline_validation_average_precision"] = float(
        baseline["average_precision"]
    )
    table["baseline_validation_patient_violation_rate"] = baseline_violation_rate
    table["baseline_validation_soft_rule_violation"] = baseline_soft_violation
    table["balanced_accuracy_tolerance"] = balanced_accuracy_tolerance
    table["auc_tolerance"] = auc_tolerance
    table["average_precision_tolerance"] = average_precision_tolerance
    table["minimum_violation_reduction"] = minimum_violation_reduction
    table["minimum_soft_violation_reduction"] = (
        minimum_soft_violation_reduction
    )
    table["selection_reason"] = reason

    return selected_lambda, table, reason


def choose_penalty_from_validation(
    validation_rows: pd.DataFrame,
    balanced_accuracy_tolerance: float = 0.01,
) -> float:
    """Legacy selector retained for compatibility with older result scripts."""
    if validation_rows.empty:
        raise ValueError("validation_rows is empty")
    required = {
        "penalty_multiplier",
        "balanced_accuracy",
        "patient_violation_rate",
        "log_loss",
    }
    missing = required - set(validation_rows.columns)
    if missing:
        raise ValueError(f"Missing validation columns: {sorted(missing)}")

    best_accuracy = validation_rows["balanced_accuracy"].max()
    eligible = validation_rows[
        validation_rows["balanced_accuracy"]
        >= best_accuracy - balanced_accuracy_tolerance
    ].copy()
    eligible = eligible.sort_values(
        ["patient_violation_rate", "log_loss", "penalty_multiplier"],
        ascending=[True, True, True],
    )
    return float(eligible.iloc[0]["penalty_multiplier"])
