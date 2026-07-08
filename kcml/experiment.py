"""Shared experiment runners for holdout, repeated CV, and ablation studies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .data import Cohort, DataSplits, make_splits_from_indices
from .metrics import (
    choose_balanced_accuracy_threshold,
    choose_penalty_constrained,
    evaluate_predictions,
)
from .rules import RuleCatalog

ModelFactory = Callable[[float], Any]


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return cleaned or "item"


def _save_penalty_plot(
    results: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    validation_rows = results[results["split"] == "validation"].sort_values(
        "penalty_multiplier"
    )
    if validation_rows.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        validation_rows["penalty_multiplier"],
        validation_rows["balanced_accuracy"],
        marker="o",
        label="Balanced accuracy",
    )
    axis.plot(
        validation_rows["penalty_multiplier"],
        validation_rows["patient_violation_rate"],
        marker="o",
        label="Binary patient violation rate",
    )
    if "soft_rule_violation" in validation_rows.columns:
        axis.plot(
            validation_rows["penalty_multiplier"],
            validation_rows["soft_rule_violation"],
            marker="o",
            label="Soft rule violation",
        )
    axis.plot(
        validation_rows["penalty_multiplier"],
        validation_rows["roc_auc"],
        marker="o",
        label="ROC AUC",
    )
    axis.set_xlabel("Penalty multiplier (lambda)")
    axis.set_ylabel("Validation metric value")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _prediction_export(
    fixed_detail: pd.DataFrame,
    common_detail: pd.DataFrame,
    optimized_detail: pd.DataFrame,
    ids_part: pd.Series,
    algorithm_name: str,
    penalty: float,
    fixed_threshold: float,
    common_threshold: float,
    optimized_threshold: float,
) -> pd.DataFrame:
    """Combine all three patient-level operating-point evaluations."""
    output = pd.DataFrame(index=fixed_detail.index)
    output["row_index"] = output.index
    output["patient_id"] = ids_part.loc[output.index].to_numpy()
    output["algorithm"] = algorithm_name
    output["penalty_multiplier"] = penalty
    output["true_label"] = fixed_detail["true_label"].to_numpy()
    output["predicted_probability"] = fixed_detail[
        "predicted_probability"
    ].to_numpy()

    output["fixed_threshold"] = fixed_threshold
    output["predicted_label_fixed"] = fixed_detail["predicted_label"].to_numpy()
    output["any_rule_violation_fixed"] = fixed_detail[
        "any_rule_violation"
    ].to_numpy()

    output["baseline_common_threshold"] = common_threshold
    output["predicted_label_common"] = common_detail[
        "predicted_label"
    ].to_numpy()
    output["any_rule_violation_common"] = common_detail[
        "any_rule_violation"
    ].to_numpy()

    output["validation_selected_threshold"] = optimized_threshold
    output["predicted_label_optimized"] = optimized_detail[
        "predicted_label"
    ].to_numpy()
    output["any_rule_violation_optimized"] = optimized_detail[
        "any_rule_violation"
    ].to_numpy()

    # Soft violation is threshold-free and therefore identical across modes.
    output["soft_rule_violation_score"] = fixed_detail[
        "soft_rule_violation_score"
    ].to_numpy()
    output["active_rule_weight_patient"] = fixed_detail[
        "active_rule_weight_patient"
    ].to_numpy()

    for column in fixed_detail.columns:
        if column.startswith("flag_"):
            output[column] = fixed_detail[column].to_numpy()
        elif column.startswith("violation_"):
            output[f"{column}_fixed"] = fixed_detail[column].to_numpy()
        elif column.startswith("soft_violation_"):
            output[column] = fixed_detail[column].to_numpy()
    for column in common_detail.columns:
        if column.startswith("violation_"):
            output[f"{column}_common"] = common_detail[column].to_numpy()
    for column in optimized_detail.columns:
        if column.startswith("violation_"):
            output[f"{column}_optimized"] = optimized_detail[column].to_numpy()

    return output.reset_index(drop=True)


def run_penalty_grid(
    algorithm_name: str,
    model_factory: ModelFactory,
    splits: DataSplits,
    output_dir: str | Path,
    penalty_multipliers: Iterable[float],
    enabled_rules: Iterable[str] | None = None,
    rule_weights: dict[str, float] | None = None,
    threshold: float = 0.5,
    selection_tolerance: float = 0.01,
    selection_auc_tolerance: float = 0.01,
    selection_ap_tolerance: float = 0.02,
    minimum_violation_reduction: float = 0.40,
    minimum_soft_violation_reduction: float = 0.0,
    save_models: bool = True,
    save_predictions: bool = True,
) -> dict[str, Any]:
    """Train all lambda values and evaluate three operating-point strategies.

    Primary analysis
        A single threshold is selected from the unpenalized (lambda=0)
        validation predictions by maximizing balanced accuracy.  That threshold
        is then held fixed for every lambda within the algorithm and is frozen
        for test evaluation.  Lambda selection uses these common-threshold
        validation metrics.  Consequently, changes in recall, specificity, and
        binary rule violations reflect the training penalty rather than
        per-lambda threshold retuning.

    Secondary analysis
        A separate validation-balanced-accuracy threshold is selected for each
        lambda and applied to the corresponding test predictions.

    Sensitivity analysis
        Every lambda is evaluated at the conventional fixed threshold supplied
        by ``threshold`` (default 0.5).

    All three analyses also report a threshold-free weighted soft rule-
    violation metric.  The metric is the mean probability mass assigned
    against active literature-derived rule directions and is bounded in [0, 1].
    """
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    lambdas = sorted({float(value) for value in penalty_multipliers})
    if not lambdas:
        raise ValueError("At least one penalty multiplier is required")
    if any(value < 0 for value in lambdas):
        raise ValueError("Penalty multipliers must be non-negative")
    if not any(np.isclose(value, 0.0) for value in lambdas):
        raise ValueError("The lambda grid must include 0 for baseline comparison")

    catalog = RuleCatalog()
    models: dict[float, Any] = {}
    validation_probabilities_by_lambda: dict[float, np.ndarray] = {}
    test_probabilities_by_lambda: dict[float, np.ndarray] = {}
    optimized_threshold_by_lambda: dict[float, float] = {}

    # Fit first, then derive the common threshold strictly from lambda=0.
    for penalty in lambdas:
        print("=" * 72)
        print(f"{algorithm_name}: lambda={penalty:g}")
        print("=" * 72)
        model = model_factory(penalty)
        model.fit(
            splits.X_train,
            splits.y_train,
            splits.X_validation,
            splits.y_validation,
        )
        models[penalty] = model
        validation_probabilities = model.predict_proba(splits.X_validation)
        test_probabilities = model.predict_proba(splits.X_test)
        validation_probabilities_by_lambda[penalty] = validation_probabilities
        test_probabilities_by_lambda[penalty] = test_probabilities
        optimized_threshold_by_lambda[penalty] = choose_balanced_accuracy_threshold(
            splits.y_validation,
            validation_probabilities,
        )

        if save_models and hasattr(model, "save"):
            model.save(models_dir / f"{safe_name(algorithm_name)}_lambda_{penalty:g}")
        if hasattr(model, "coefficient_table"):
            model.coefficient_table().to_csv(
                output_dir / f"coefficients_lambda_{penalty:g}.csv", index=False
            )

    baseline_lambda = min(
        lambdas,
        key=lambda value: abs(value),
    )
    if not np.isclose(baseline_lambda, 0.0):
        raise RuntimeError("Unable to identify lambda=0 baseline")
    common_threshold = choose_balanced_accuracy_threshold(
        splits.y_validation,
        validation_probabilities_by_lambda[baseline_lambda],
    )
    print(
        f"Primary common threshold for {algorithm_name}: {common_threshold:.6g} "
        "(selected once from lambda=0 validation predictions)",
        flush=True,
    )

    fixed_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    optimized_rows: list[dict[str, Any]] = []

    for penalty in lambdas:
        model = models[penalty]
        optimized_threshold = optimized_threshold_by_lambda[penalty]
        for split_name, X_part, y_part, ids_part, probabilities in (
            (
                "validation",
                splits.X_validation,
                splits.y_validation,
                splits.ids_validation,
                validation_probabilities_by_lambda[penalty],
            ),
            (
                "test",
                splits.X_test,
                splits.y_test,
                splits.ids_test,
                test_probabilities_by_lambda[penalty],
            ),
        ):
            fixed_metrics, fixed_detail = evaluate_predictions(
                y_true=y_part,
                probabilities=probabilities,
                X_original=X_part,
                threshold=threshold,
                rule_catalog=catalog,
                enabled_rules=enabled_rules,
                weight_overrides=rule_weights,
            )
            fixed_rows.append(
                {
                    "algorithm": algorithm_name,
                    "penalty_multiplier": penalty,
                    "split": split_name,
                    "evaluation_mode": "fixed_threshold_0.5"
                    if np.isclose(threshold, 0.5)
                    else "fixed_threshold",
                    "threshold_selected_on": "prespecified",
                    "n_patients": len(X_part),
                    **fixed_metrics,
                }
            )

            common_metrics, common_detail = evaluate_predictions(
                y_true=y_part,
                probabilities=probabilities,
                X_original=X_part,
                threshold=common_threshold,
                rule_catalog=catalog,
                enabled_rules=enabled_rules,
                weight_overrides=rule_weights,
            )
            common_rows.append(
                {
                    "algorithm": algorithm_name,
                    "penalty_multiplier": penalty,
                    "split": split_name,
                    "evaluation_mode": "baseline_common_threshold",
                    "threshold_selected_on": "lambda_0_validation",
                    "n_patients": len(X_part),
                    **common_metrics,
                }
            )

            optimized_metrics, optimized_detail = evaluate_predictions(
                y_true=y_part,
                probabilities=probabilities,
                X_original=X_part,
                threshold=optimized_threshold,
                rule_catalog=catalog,
                enabled_rules=enabled_rules,
                weight_overrides=rule_weights,
            )
            optimized_rows.append(
                {
                    "algorithm": algorithm_name,
                    "penalty_multiplier": penalty,
                    "split": split_name,
                    "evaluation_mode": "per_lambda_validation_optimized_threshold",
                    "threshold_selected_on": "same_lambda_validation",
                    "n_patients": len(X_part),
                    **optimized_metrics,
                }
            )

            if save_predictions:
                export = _prediction_export(
                    fixed_detail=fixed_detail,
                    common_detail=common_detail,
                    optimized_detail=optimized_detail,
                    ids_part=ids_part,
                    algorithm_name=algorithm_name,
                    penalty=penalty,
                    fixed_threshold=threshold,
                    common_threshold=common_threshold,
                    optimized_threshold=optimized_threshold,
                )
                if hasattr(model, "restart_probabilities"):
                    restart_probabilities = model.restart_probabilities(X_part)
                    for restart_index in range(restart_probabilities.shape[1]):
                        export[
                            f"predicted_probability_restart_{restart_index + 1}"
                        ] = restart_probabilities[:, restart_index]
                    export["restart_probability_sd"] = restart_probabilities.std(
                        axis=1, ddof=0
                    )
                export.to_csv(
                    predictions_dir
                    / f"{safe_name(algorithm_name)}_lambda_{penalty:g}_{split_name}.csv",
                    index=False,
                )

    fixed_results = pd.DataFrame(fixed_rows)
    common_results = pd.DataFrame(common_rows)
    optimized_results = pd.DataFrame(optimized_rows)

    # Backward-compatible name: all_lambda_results.csv remains the fixed
    # threshold sensitivity analysis.
    fixed_results.to_csv(output_dir / "all_lambda_results.csv", index=False)
    common_results.to_csv(
        output_dir / "common_threshold_all_lambda_results.csv", index=False
    )
    optimized_results.to_csv(
        output_dir / "threshold_optimized_all_lambda_results.csv", index=False
    )
    splits.assignments().to_csv(output_dir / "split_assignments.csv", index=False)

    # Primary lambda selection uses the baseline-derived common threshold.
    validation_rows = common_results[
        common_results["split"] == "validation"
    ].copy()
    selected_lambda, selection_audit, selection_reason = choose_penalty_constrained(
        validation_rows,
        balanced_accuracy_tolerance=selection_tolerance,
        auc_tolerance=selection_auc_tolerance,
        average_precision_tolerance=selection_ap_tolerance,
        minimum_violation_reduction=minimum_violation_reduction,
        minimum_soft_violation_reduction=minimum_soft_violation_reduction,
    )
    selection_audit["selection_evaluation_mode"] = "baseline_common_threshold"
    selection_audit["baseline_common_threshold"] = common_threshold
    selection_audit.to_csv(output_dir / "lambda_selection_audit.csv", index=False)

    selected_common_rows = common_results[
        np.isclose(common_results["penalty_multiplier"], selected_lambda)
    ].copy()
    selected_optimized_rows = optimized_results[
        np.isclose(optimized_results["penalty_multiplier"], selected_lambda)
    ].copy()
    selected_fixed_rows = fixed_results[
        np.isclose(fixed_results["penalty_multiplier"], selected_lambda)
    ].copy()

    # Primary selected output and explicit aliases.
    selected_common_rows.to_csv(output_dir / "selected_lambda_results.csv", index=False)
    selected_common_rows.to_csv(
        output_dir / "selected_lambda_common_threshold_results.csv", index=False
    )
    selected_optimized_rows.to_csv(
        output_dir / "selected_lambda_optimized_threshold_results.csv", index=False
    )
    selected_fixed_rows.to_csv(
        output_dir / "selected_lambda_fixed_threshold_results.csv", index=False
    )

    selected_common_test = selected_common_rows[
        selected_common_rows["split"] == "test"
    ].iloc[0].to_dict()
    selected_optimized_test = selected_optimized_rows[
        selected_optimized_rows["split"] == "test"
    ].iloc[0].to_dict()
    selected_fixed_test = selected_fixed_rows[
        selected_fixed_rows["split"] == "test"
    ].iloc[0].to_dict()
    selected_common_validation = selected_common_rows[
        selected_common_rows["split"] == "validation"
    ].iloc[0].to_dict()
    selected_optimized_validation = selected_optimized_rows[
        selected_optimized_rows["split"] == "validation"
    ].iloc[0].to_dict()

    manifest = {
        "algorithm": algorithm_name,
        "penalty_multipliers": lambdas,
        "selected_penalty_multiplier": selected_lambda,
        "selection_reason": selection_reason,
        "primary_analysis": "baseline_common_threshold",
        "selection_rule": {
            "primary_threshold_selection": (
                "Maximize validation balanced accuracy for lambda=0 once; "
                "apply that same threshold to all lambdas and freeze it for test."
            ),
            "secondary_threshold_selection": (
                "For each lambda separately, maximize validation balanced accuracy."
            ),
            "sensitivity_threshold": threshold,
            "lambda_selection": (
                "Using common-threshold validation metrics, select the smallest "
                "penalized lambda satisfying all safeguards; retain lambda=0 if "
                "none is eligible."
            ),
            "balanced_accuracy_tolerance": selection_tolerance,
            "auc_tolerance_from_lambda_zero": selection_auc_tolerance,
            "average_precision_tolerance_from_lambda_zero": selection_ap_tolerance,
            "minimum_patient_violation_reduction": minimum_violation_reduction,
            "minimum_soft_violation_reduction": minimum_soft_violation_reduction,
        },
        "baseline_common_threshold": common_threshold,
        "fixed_sensitivity_threshold": threshold,
        "selected_lambda_own_validation_threshold": selected_optimized_validation[
            "threshold"
        ],
        "features": splits.feature_names,
        "imputation": "training-partition median; rules use original non-imputed values",
        "enabled_rules": list(enabled_rules) if enabled_rules is not None else None,
        "rule_weights": rule_weights,
        "soft_rule_violation_definition": (
            "Weighted mean absolute difference between predicted probability and "
            "the target direction of active rules; bounded in [0,1]."
        ),
        "selected_validation_metrics_common_threshold": selected_common_validation,
        "selected_test_metrics_common_threshold": selected_common_test,
        "selected_test_metrics_per_lambda_optimized_threshold": selected_optimized_test,
        "selected_test_metrics_fixed_threshold": selected_fixed_test,
    }
    selected_model = models[selected_lambda]
    if hasattr(selected_model, "restart_aggregation"):
        manifest["neural_restart_protocol"] = {
            "n_restarts": getattr(selected_model, "n_restarts", None),
            "restart_aggregation": getattr(
                selected_model, "restart_aggregation", None
            ),
            "deterministic_algorithms": getattr(
                selected_model, "deterministic", None
            ),
            "restart_summary": getattr(
                selected_model, "restart_summary_", None
            ),
            "ensemble_validation_loss": getattr(
                selected_model, "ensemble_validation_loss_", None
            ),
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    _save_penalty_plot(
        common_results,
        output_dir / "penalty_response_common_threshold.png",
        title=(
            f"{algorithm_name}: baseline-derived common threshold "
            f"({common_threshold:.4g})"
        ),
    )
    _save_penalty_plot(
        fixed_results,
        output_dir / "penalty_response_fixed_threshold.png",
        title=f"{algorithm_name}: fixed threshold {threshold:g}",
    )
    _save_penalty_plot(
        optimized_results,
        output_dir / "penalty_response_validation_optimized.png",
        title=f"{algorithm_name}: per-lambda validation-optimized thresholds",
    )
    # Primary plot name points to the common-threshold analysis.
    _save_penalty_plot(
        common_results,
        output_dir / "penalty_response.png",
        title=(
            f"{algorithm_name}: primary common-threshold analysis "
            f"({common_threshold:.4g})"
        ),
    )

    print(
        f"Selected lambda for {algorithm_name}: {selected_lambda:g}; "
        f"primary common threshold={common_threshold:.6g}; "
        f"selected-lambda own optimized threshold="
        f"{selected_optimized_validation['threshold']:.6g}",
        flush=True,
    )
    return {
        "algorithm": algorithm_name,
        "selected_lambda": selected_lambda,
        "selected_threshold": float(common_threshold),
        "selected_common_threshold": float(common_threshold),
        "selected_optimized_threshold": float(
            selected_optimized_validation["threshold"]
        ),
        "results": fixed_results,
        "common_threshold_results": common_results,
        "threshold_optimized_results": optimized_results,
        "selected_test": selected_common_test,
        "selected_common_test": selected_common_test,
        "selected_optimized_test": selected_optimized_test,
        "selected_fixed_test": selected_fixed_test,
        "selected_validation": selected_common_validation,
        "selected_common_validation": selected_common_validation,
        "selected_optimized_validation": selected_optimized_validation,
        "selection_audit": selection_audit,
        "models": models,
    }


def run_repeated_nested_holdout(
    cohort: Cohort,
    algorithm_name: str,
    model_factory_builder: Callable[[Iterable[str] | None], ModelFactory],
    output_dir: str | Path,
    penalty_multipliers: Iterable[float],
    enabled_rules: Iterable[str] | None = None,
    rule_weights: dict[str, float] | None = None,
    n_splits: int = 5,
    n_repeats: int = 2,
    validation_fraction_within_training: float = 0.20,
    random_state: int = 42,
    threshold: float = 0.5,
    selection_tolerance: float = 0.01,
    selection_auc_tolerance: float = 0.01,
    selection_ap_tolerance: float = 0.02,
    minimum_violation_reduction: float = 0.40,
    minimum_soft_violation_reduction: float = 0.0,
) -> pd.DataFrame:
    """Repeated outer CV with inner common-threshold and lambda selection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    selected_rows: list[dict[str, Any]] = []

    for fold_number, (development_indices, test_indices) in enumerate(
        splitter.split(cohort.X, cohort.y), start=1
    ):
        development_labels = cohort.y.iloc[development_indices]
        train_indices, validation_indices = train_test_split(
            development_indices,
            test_size=validation_fraction_within_training,
            random_state=random_state + fold_number,
            stratify=development_labels,
        )
        splits = make_splits_from_indices(
            cohort,
            np.asarray(train_indices),
            np.asarray(validation_indices),
            np.asarray(test_indices),
        )
        fold_dir = output_dir / f"fold_{fold_number:03d}"
        result = run_penalty_grid(
            algorithm_name=algorithm_name,
            model_factory=model_factory_builder(enabled_rules),
            splits=splits,
            output_dir=fold_dir,
            penalty_multipliers=penalty_multipliers,
            enabled_rules=enabled_rules,
            rule_weights=rule_weights,
            threshold=threshold,
            selection_tolerance=selection_tolerance,
            selection_auc_tolerance=selection_auc_tolerance,
            selection_ap_tolerance=selection_ap_tolerance,
            minimum_violation_reduction=minimum_violation_reduction,
            minimum_soft_violation_reduction=minimum_soft_violation_reduction,
            save_models=False,
            save_predictions=False,
        )
        row = dict(result["selected_common_test"])
        row["fold"] = fold_number
        row["selected_penalty_multiplier"] = result["selected_lambda"]
        row["baseline_common_threshold"] = result["selected_common_threshold"]
        row["selected_lambda_optimized_threshold"] = result[
            "selected_optimized_threshold"
        ]
        selected_rows.append(row)

    summary = pd.DataFrame(selected_rows)
    summary.to_csv(output_dir / "repeated_cv_selected_test_results.csv", index=False)
    numeric = summary.select_dtypes(include=[np.number])
    aggregate = pd.DataFrame(
        {
            "mean": numeric.mean(),
            "std": numeric.std(ddof=1),
            "median": numeric.median(),
            "q025": numeric.quantile(0.025),
            "q975": numeric.quantile(0.975),
        }
    )
    aggregate.to_csv(output_dir / "repeated_cv_aggregate.csv")
    return summary
