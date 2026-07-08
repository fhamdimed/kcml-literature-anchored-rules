#!/usr/bin/env python3
"""Generate manuscript assets for the common-threshold KCML analyses."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISPLAY_ORDER = [
    "XGBoost",
    "LightGBM",
    "Logistic Regression",
    "Neural Network (MLP)",
]
SHORT = {
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Logistic Regression": "Logistic",
    "Neural Network (MLP)": "Neural ensemble",
}
KEY_ORDER = ["xgboost", "lightgbm", "logistic", "neural"]
KEY_TO_DISPLAY = dict(zip(KEY_ORDER, DISPLAY_ORDER))
ABLATION_ORDER = [
    "all_rules",
    "single_LR01", "single_LR02", "single_LR03", "single_LR04", "single_LR05",
    "leave_out_LR01", "leave_out_LR02", "leave_out_LR03", "leave_out_LR04", "leave_out_LR05",
    "control_permuted", "control_column_permuted", "control_reversed",
]
ABLATION_LABELS = {
    "all_rules": "All rules",
    "single_LR01": "LR01 only",
    "single_LR02": "LR02 only",
    "single_LR03": "LR03 only",
    "single_LR04": "LR04 only",
    "single_LR05": "LR05 only",
    "leave_out_LR01": "Without LR01",
    "leave_out_LR02": "Without LR02",
    "leave_out_LR03": "Without LR03",
    "leave_out_LR04": "Without LR04",
    "leave_out_LR05": "Without LR05",
    "control_permuted": "Profile permutation",
    "control_column_permuted": "Column permutation",
    "control_reversed": "Reversed targets",
}


def latex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def load_cv(cv_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    delta_rows: list[dict[str, float | int | str]] = []
    for key in KEY_ORDER:
        selected = pd.read_csv(cv_root / key / "repeated_cv_selected_test_results.csv")
        selected["algorithm_key"] = key
        selected_frames.append(selected)
        for fold_dir in sorted((cv_root / key).glob("fold_*")):
            fold = int(re.search(r"fold_(\d+)", fold_dir.name).group(1))
            common = pd.read_csv(fold_dir / "common_threshold_all_lambda_results.csv")
            baseline = common.loc[
                common["split"].eq("test")
                & np.isclose(common["penalty_multiplier"], 0.0)
            ].copy()
            if len(baseline) != 1:
                raise ValueError(f"Expected one baseline row for {key} fold {fold}")
            baseline["fold"] = fold
            baseline["algorithm_key"] = key
            baseline_frames.append(baseline)
        baseline_all = pd.concat(
            [x for x in baseline_frames if x["algorithm_key"].iloc[0] == key],
            ignore_index=True,
        )
        merged = selected.merge(
            baseline_all[
                [
                    "fold", "balanced_accuracy", "recall_sensitivity", "specificity",
                    "roc_auc", "average_precision", "log_loss", "brier_score",
                    "patient_violation_rate", "soft_rule_violation",
                ]
            ],
            on="fold",
            suffixes=("_selected", "_baseline"),
            validate="one_to_one",
        )
        for _, row in merged.iterrows():
            out: dict[str, float | int | str] = {
                "algorithm": row["algorithm"],
                "algorithm_key": key,
                "fold": int(row["fold"]),
                "selected_penalty_multiplier": float(row["penalty_multiplier"]),
            }
            for metric in [
                "balanced_accuracy", "recall_sensitivity", "specificity",
                "roc_auc", "average_precision", "log_loss", "brier_score",
                "patient_violation_rate", "soft_rule_violation",
            ]:
                out[f"delta_{metric}"] = float(
                    row[f"{metric}_selected"] - row[f"{metric}_baseline"]
                )
            delta_rows.append(out)
    return (
        pd.concat(selected_frames, ignore_index=True),
        pd.concat(baseline_frames, ignore_index=True),
        pd.DataFrame(delta_rows),
    )


def summarize_cv(selected: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in DISPLAY_ORDER:
        group = selected.loc[selected["algorithm"].eq(algorithm)].copy()
        dgroup = deltas.loc[deltas["algorithm"].eq(algorithm)].copy()
        penalized = dgroup.loc[dgroup["selected_penalty_multiplier"] > 0]
        counts = Counter(float(v) for v in group["penalty_multiplier"])
        frequency = ", ".join(f"{lam:g}: {counts[lam]}" for lam in sorted(counts))
        row: dict[str, object] = {
            "algorithm": algorithm,
            "selected_lambda_frequency": frequency,
            "penalized_folds": int((group["penalty_multiplier"] > 0).sum()),
            "n_folds": int(len(group)),
        }
        for metric in [
            "balanced_accuracy", "recall_sensitivity", "specificity", "roc_auc",
            "average_precision", "log_loss", "brier_score",
            "patient_violation_rate", "soft_rule_violation", "threshold",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
            row[f"{metric}_q025"] = float(group[metric].quantile(0.025))
            row[f"{metric}_q975"] = float(group[metric].quantile(0.975))
        for metric in [
            "balanced_accuracy", "recall_sensitivity", "specificity", "roc_auc",
            "average_precision", "patient_violation_rate", "soft_rule_violation",
        ]:
            row[f"delta_{metric}_all_mean"] = float(dgroup[f"delta_{metric}"].mean())
            row[f"delta_{metric}_penalized_mean"] = (
                float(penalized[f"delta_{metric}"].mean()) if len(penalized) else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_ablation(ablation_root: Path) -> pd.DataFrame:
    frames = []
    for key in KEY_ORDER:
        df = pd.read_csv(ablation_root / key / "ablation_selected_test_results.csv")
        df["algorithm_key"] = key
        df["algorithm_display"] = KEY_TO_DISPLAY[key]
        all_row = df.loc[df["ablation"].eq("all_rules")].iloc[0]
        for metric in [
            "balanced_accuracy", "recall_sensitivity", "specificity", "roc_auc",
            "average_precision", "patient_violation_rate", "soft_rule_violation",
        ]:
            df[f"delta_{metric}_vs_all"] = df[metric] - all_row[metric]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_ablation(ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm in DISPLAY_ORDER:
        group = ablation.loc[ablation["algorithm_display"].eq(algorithm)]
        singles = group.loc[group["ablation"].str.startswith("single_")]
        leaveouts = group.loc[group["ablation"].str.startswith("leave_out_")]
        controls = group.loc[group["ablation"].str.startswith("control_")]
        all_row = group.loc[group["ablation"].eq("all_rules")].iloc[0]
        lr05 = group.loc[group["ablation"].eq("leave_out_LR05")].iloc[0]
        rows.append({
            "algorithm": algorithm,
            "all_rules_lambda": float(all_row["penalty_multiplier"]),
            "single_rule_penalized": int((singles["penalty_multiplier"] > 0).sum()),
            "leave_out_penalized": int((leaveouts["penalty_multiplier"] > 0).sum()),
            "negative_control_penalized": int((controls["penalty_multiplier"] > 0).sum()),
            "leave_out_lr05_lambda": float(lr05["penalty_multiplier"]),
            "leave_out_lr05_delta_ba": float(lr05["delta_balanced_accuracy_vs_all"]),
            "leave_out_lr05_delta_ap": float(lr05["delta_average_precision_vs_all"]),
            "zero_penalty_leaveouts": ", ".join(
                x.replace("leave_out_", "")
                for x in leaveouts.loc[np.isclose(leaveouts["penalty_multiplier"], 0.0), "ablation"]
            ) or "None",
        })
    return pd.DataFrame(rows)


def write_primary_table(selected: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Algorithm & $\lambda$ & $\tau_0$ & Bal. acc. & Sens. & Spec. & ROC AUC & AP & Binary $V$ & Soft $V$ \\",
        r"\midrule",
    ]
    for alg in DISPLAY_ORDER:
        r = selected.loc[selected["algorithm"].eq(alg)].iloc[0]
        lines.append(
            f"{latex_escape(SHORT[alg])} & {r['penalty_multiplier']:.2g} & "
            f"{r['threshold']:.3f} & {r['balanced_accuracy']:.3f} & "
            f"{r['recall_sensitivity']:.3f} & {r['specificity']:.3f} & "
            f"{r['roc_auc']:.3f} & {r['average_precision']:.3f} & "
            f"{r['patient_violation_rate']:.3f} & {r['soft_rule_violation']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_table(comp: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Algorithm & $\lambda$ & $\Delta$ Bal. acc. & $\Delta$ Sens. & $\Delta$ Spec. & $\Delta$ AUC & Binary $V$ reduction & Soft $V$ reduction \\",
        r"\midrule",
    ]
    for alg in DISPLAY_ORDER:
        r = comp.loc[comp["algorithm"].eq(alg)].iloc[0]
        rel_b = 0.0 if r["patient_violation_rate_baseline"] == 0 else (
            r["patient_violation_rate_baseline"] - r["patient_violation_rate_selected"]
        ) / r["patient_violation_rate_baseline"]
        rel_s = 0.0 if r["soft_rule_violation_baseline"] == 0 else (
            r["soft_rule_violation_baseline"] - r["soft_rule_violation_selected"]
        ) / r["soft_rule_violation_baseline"]
        lines.append(
            f"{latex_escape(SHORT[alg])} & {r['penalty_multiplier_selected']:.2g} & "
            f"{r['delta_balanced_accuracy']:+.3f} & "
            f"{r['delta_recall_sensitivity']:+.3f} & "
            f"{r['delta_specificity']:+.3f} & "
            f"{r['delta_roc_auc']:+.3f} & "
            f"{100*rel_b:.1f}\\% & {100*rel_s:.1f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_operating_table(op: pd.DataFrame, path: Path) -> None:
    role_names = {
        "primary_common": "Common baseline-derived",
        "secondary_per_lambda_optimized": r"Per-$\lambda$ optimized",
        "sensitivity_fixed": "Fixed 0.5",
    }
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Algorithm & Operating point & Threshold & Bal. acc. & Sens. & Spec. & Binary $V$ \\",
        r"\midrule",
    ]
    for alg in DISPLAY_ORDER:
        g = op.loc[op["algorithm"].eq(alg)]
        for idx, role in enumerate(["primary_common", "secondary_per_lambda_optimized", "sensitivity_fixed"]):
            r = g.loc[g["reporting_strategy"].eq(role)].iloc[0]
            alg_text = latex_escape(SHORT[alg]) if idx == 0 else ""
            lines.append(
                f"{alg_text} & {role_names[role]} & {r['threshold']:.3f} & "
                f"{r['balanced_accuracy']:.3f} & {r['recall_sensitivity']:.3f} & "
                f"{r['specificity']:.3f} & {r['patient_violation_rate']:.3f} \\\\"
            )
        if alg != DISPLAY_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cv_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lp{2.4cm}rrrrrr}",
        r"\toprule",
        r"Algorithm & Selected $\lambda$ frequency & Bal. acc. & Sens. & Spec. & ROC AUC & Binary $V$ & Soft $V$ \\",
        r"\midrule",
    ]
    for alg in DISPLAY_ORDER:
        r = summary.loc[summary["algorithm"].eq(alg)].iloc[0]
        lines.append(
            f"{latex_escape(SHORT[alg])} & {latex_escape(r['selected_lambda_frequency'])} & "
            f"{r['balanced_accuracy_mean']:.3f} $\\pm$ {r['balanced_accuracy_sd']:.3f} & "
            f"{r['recall_sensitivity_mean']:.3f} $\\pm$ {r['recall_sensitivity_sd']:.3f} & "
            f"{r['specificity_mean']:.3f} $\\pm$ {r['specificity_sd']:.3f} & "
            f"{r['roc_auc_mean']:.3f} $\\pm$ {r['roc_auc_sd']:.3f} & "
            f"{r['patient_violation_rate_mean']:.3f} $\\pm$ {r['patient_violation_rate_sd']:.3f} & "
            f"{r['soft_rule_violation_mean']:.3f} $\\pm$ {r['soft_rule_violation_sd']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cv_effect_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Algorithm & Penalized folds & $\Delta$ Bal. acc. & $\Delta$ Sens. & $\Delta$ Spec. & $\Delta$ Binary $V$ & $\Delta$ Soft $V$ \\",
        r"\midrule",
    ]
    for alg in DISPLAY_ORDER:
        r = summary.loc[summary["algorithm"].eq(alg)].iloc[0]
        lines.append(
            f"{latex_escape(SHORT[alg])} & {int(r['penalized_folds'])}/{int(r['n_folds'])} & "
            f"{r['delta_balanced_accuracy_penalized_mean']:+.3f} & "
            f"{r['delta_recall_sensitivity_penalized_mean']:+.3f} & "
            f"{r['delta_specificity_penalized_mean']:+.3f} & "
            f"{r['delta_patient_violation_rate_penalized_mean']:+.3f} & "
            f"{r['delta_soft_rule_violation_penalized_mean']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccccp{3.5cm}}",
        r"\toprule",
        r"Algorithm & All-rule $\lambda$ & Penalized single rules & Penalized leave-outs & Penalized controls & Leave-outs yielding $\lambda=0$ \\",
        r"\midrule",
    ]
    for alg in DISPLAY_ORDER:
        r = summary.loc[summary["algorithm"].eq(alg)].iloc[0]
        lines.append(
            f"{latex_escape(SHORT[alg])} & {r['all_rules_lambda']:.2g} & "
            f"{int(r['single_rule_penalized'])}/5 & {int(r['leave_out_penalized'])}/5 & "
            f"{int(r['negative_control_penalized'])}/3 & "
            f"{latex_escape(r['zero_penalty_leaveouts'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def format_int(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{int(value):,}"
    

def read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cleaning_table(
    cleaned_data_root: Path,
    tables_dir: Path,
    source_dir: Path,
) -> None:
    """Write the manuscript cleaning table.
    It is generated from the repository-lean cleaned-data outputs:
    thalassemia_model_matrix_clean.csv, cohort_flow.csv and
    analysis_manifest.json.
    """

    cleaned_data_root = cleaned_data_root.resolve()
    manifest_path = cleaned_data_root / "analysis_manifest.json"
    cohort_flow_path = cleaned_data_root / "cohort_flow.csv"
    model_matrix_path = cleaned_data_root / "thalassemia_model_matrix_clean.csv"

    required = [manifest_path, cohort_flow_path, model_matrix_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        print(
            "WARNING: table_cleaning.tex was not generated because the cleaned "
            "cohort files were not found:",
        )
        for path in missing:
            print(f"  missing: {path}")
        print(
            "Run prepare_thalassemia_dataset.py first, or pass "
            "--cleaned-data-root explicitly."
        )
        return

    manifest = read_manifest(manifest_path)
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}

    model_df = pd.read_csv(model_matrix_path)

    primary_predictors = ["MCV", "MCH", "HBA2", "HBF"]
    missing_predictors = [
        column for column in primary_predictors if column not in model_df.columns
    ]
    if missing_predictors:
        raise ValueError(
            "Cannot build table_cleaning.tex; missing predictor columns in "
            f"{model_matrix_path}: {missing_predictors}"
        )

    retained_mask = ~model_df[primary_predictors].isna().all(axis=1)
    retained_df = model_df.loc[retained_mask].copy()

    rows = [
        {
            "Item": "Source phenotype cohort",
            "Count": format_int(counts.get("source_rows", len(model_df))),
        },
        {
            "Item": "Analytical cohort",
            "Count": format_int(len(retained_df)),
        },
        {
            "Item": "Blank HbF interpreted as zero",
            "Count": format_int(counts.get("hbf_blank_interpreted_as_zero")),
        },
        {
            "Item": "Unresolved HbF before all-missing exclusion",
            "Count": format_int(counts.get("hbf_unresolved")),
        },
        {
            "Item": "Fraction-sum anomaly flag",
            "Count": format_int(counts.get("fraction_sum_anomalies")),
        },
        {
            "Item": "Missing MCV in analytical cohort",
            "Count": format_int(retained_df["MCV"].isna().sum()),
        },
        {
            "Item": "Missing MCH in analytical cohort",
            "Count": format_int(retained_df["MCH"].isna().sum()),
        },
        {
            "Item": r"Missing HbA$_2$ in analytical cohort",
            "Count": format_int(retained_df["HBA2"].isna().sum()),
        },
        {
            "Item": "Missing HbF in analytical cohort",
            "Count": format_int(retained_df["HBF"].isna().sum()),
        },
    ]

    summary = pd.DataFrame(rows)
    source_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(source_dir / "table_cleaning_source.csv", index=False)
    shutil.copy2(cohort_flow_path, source_dir / "cohort_flow.csv")
    shutil.copy2(manifest_path, source_dir / "analysis_manifest.json")

    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Item & Count \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row['Item']} & {row['Count']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (tables_dir / "table_cleaning.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_validation(audit: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("recall_sensitivity", "Sensitivity / recall"),
        ("patient_violation_rate", "Binary violation rate"),
        ("soft_rule_violation", "Soft rule violation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    for ax, (metric, ylabel), panel in zip(axes.flat, metrics, "abcd"):
        for alg in DISPLAY_ORDER:
            g = audit.loc[audit["algorithm"].eq(alg)].sort_values("penalty_multiplier")
            line, = ax.plot(g["penalty_multiplier"], g[metric], marker="o", label=SHORT[alg])
            s = g.loc[g["selected"]].iloc[0]
            ax.scatter([s["penalty_multiplier"]], [s[metric]], marker="*", s=120,
                       color=line.get_color(), edgecolor="black", linewidth=0.5, zorder=4)
        ax.set_xlabel("Penalty multiplier, $\\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({panel})", loc="left", fontweight="bold")
        style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, fontsize=8,
               loc='upper center', bbox_to_anchor=(0.5, 1.02))
    
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

def grouped_baseline_selected(ax: plt.Axes, comp: pd.DataFrame, base_col: str,
                              selected_col: str, ylabel: str, panel: str) -> None:
    c = comp.set_index("algorithm").reindex(DISPLAY_ORDER).reset_index()
    x = np.arange(len(c)); w = 0.36
    ax.bar(x-w/2, c[base_col], w, label="Unpenalized")
    ax.bar(x+w/2, c[selected_col], w, label="Selected")
    ax.set_xticks(x, [SHORT[a] for a in DISPLAY_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({panel})", loc="left", fontweight="bold")
    style_axes(ax)


def figure_holdout(comp: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7))
    grouped_baseline_selected(axes[0,0], comp, "balanced_accuracy_baseline",
                              "balanced_accuracy_selected", "Balanced accuracy", "a")
    grouped_baseline_selected(axes[0,1], comp, "recall_sensitivity_baseline",
                              "recall_sensitivity_selected", "Sensitivity / recall", "b")
    grouped_baseline_selected(axes[1,0], comp, "patient_violation_rate_baseline",
                              "patient_violation_rate_selected", "Binary violation rate", "c")
    grouped_baseline_selected(axes[1,1], comp, "soft_rule_violation_baseline",
                              "soft_rule_violation_selected", "Soft rule violation", "d")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=8,
               loc='upper center', bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def figure_operating_points(op: pd.DataFrame, output: Path) -> None:
    roles = ["primary_common", "secondary_per_lambda_optimized", "sensitivity_fixed"]
    #role_labels = ["Common", "Per-$\\lambda$ optimized", "Threshold 0.5"]
    role_labels = ["Common threshold", "Per-$\\lambda$ optimized threshold", "Fixed 0.5 threshold"]
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("recall_sensitivity", "Sensitivity / recall"),
        ("specificity", "Specificity"),
        ("patient_violation_rate", "Binary violation rate"),
    ]
    fig, axes = plt.subplots(2,2,figsize=(10.8,7.7))
    for ax,(metric,ylabel),panel in zip(axes.flat,metrics,"abcd"):
        x=np.arange(len(DISPLAY_ORDER)); w=0.24
        for j,(role,label) in enumerate(zip(roles,role_labels)):
            vals=[]
            for alg in DISPLAY_ORDER:
                vals.append(float(op.loc[(op.algorithm==alg)&(op.reporting_strategy==role),metric].iloc[0]))
            ax.bar(x+(j-1)*w,vals,w,label=label)
        ax.set_xticks(x,[SHORT[a] for a in DISPLAY_ORDER],rotation=15,ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({panel})",loc="left",fontweight="bold")
        style_axes(ax)
        handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=8,
               loc='upper center', bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output,bbox_inches="tight")
    plt.close(fig)


def box_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, panel: str) -> None:
    data=[df.loc[df.algorithm==alg,metric].to_numpy() for alg in DISPLAY_ORDER]
    ax.boxplot(data, tick_labels=[SHORT[a] for a in DISPLAY_ORDER], showmeans=True)
    ax.tick_params(axis="x",rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(f"({panel})",loc="left",fontweight="bold")
    style_axes(ax)


def figure_cv(selected: pd.DataFrame, output: Path) -> None:
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.8))
    box_panel(axes[0,0],selected,"balanced_accuracy","Outer-fold balanced accuracy","a")
    box_panel(axes[0,1],selected,"recall_sensitivity","Outer-fold sensitivity","b")
    box_panel(axes[1,0],selected,"patient_violation_rate","Outer-fold binary violation rate","c")
    lambdas=[0.0,0.1,0.25]
    x=np.arange(len(DISPLAY_ORDER)); bottom=np.zeros(len(DISPLAY_ORDER))
    for lam in lambdas:
        counts=[]
        for alg in DISPLAY_ORDER:
            g=selected.loc[selected.algorithm==alg]
            counts.append(int(np.isclose(g.penalty_multiplier,lam).sum()))
        axes[1,1].bar(x,counts,bottom=bottom,label=f"$\\lambda={lam:g}$")
        bottom+=np.array(counts)
    axes[1,1].set_xticks(x,[SHORT[a] for a in DISPLAY_ORDER],rotation=15,ha="right")
    axes[1,1].set_ylabel("Number of outer folds")
    axes[1,1].set_ylim(0,10)
    axes[1,1].set_title("(d)",loc="left",fontweight="bold")
    axes[1,1].legend(frameon=False, fontsize=8, ncol=3,
                     loc='upper center', bbox_to_anchor=(0.5, 1.15))
    style_axes(axes[1,1])
    fig.tight_layout()
    fig.savefig(output,bbox_inches="tight")
    plt.close(fig)


def figure_ablation(ablation: pd.DataFrame, output: Path) -> None:
    lam=np.zeros((len(ABLATION_ORDER),len(DISPLAY_ORDER)))
    dba=np.zeros_like(lam)
    for i,config in enumerate(ABLATION_ORDER):
        for j,alg in enumerate(DISPLAY_ORDER):
            r=ablation.loc[(ablation.ablation==config)&(ablation.algorithm_display==alg)].iloc[0]
            lam[i,j]=r.penalty_multiplier
            dba[i,j]=r.delta_balanced_accuracy_vs_all
    fig,axes=plt.subplots(1,2,figsize=(12.2,8.0),gridspec_kw={"width_ratios":[1,1]})
    im0=axes[0].imshow(lam,aspect="auto",cmap="viridis",vmin=0,vmax=max(0.5,lam.max()))
    im1=axes[1].imshow(dba,aspect="auto",cmap="coolwarm",vmin=-max(abs(dba.min()),abs(dba.max())),vmax=max(abs(dba.min()),abs(dba.max())))
    for ax,panel,title in [(axes[0],"a","Selected penalty"),(axes[1],"b","Change in balanced accuracy vs all rules")]:
        ax.set_yticks(np.arange(len(ABLATION_ORDER)),[ABLATION_LABELS[x] for x in ABLATION_ORDER])
        ax.set_xticks(np.arange(len(DISPLAY_ORDER)),[SHORT[a] for a in DISPLAY_ORDER],rotation=25,ha="right")
        ax.set_title(f"({panel}) {title}",loc="left",fontweight="bold")
    for i in range(lam.shape[0]):
        for j in range(lam.shape[1]):
            axes[0].text(j,i,f"{lam[i,j]:g}",ha="center",va="center",fontsize=7,color="white" if lam[i,j]>0.25 else "black")
            axes[1].text(j,i,f"{dba[i,j]:+.3f}",ha="center",va="center",fontsize=6.5,color="white" if abs(dba[i,j])>0.012 else "black")
    fig.colorbar(im0,ax=axes[0],fraction=0.035,pad=0.02,label="Selected $\\lambda$")
    fig.colorbar(im1,ax=axes[1],fraction=0.035,pad=0.02,label="$\\Delta$ balanced accuracy")
    fig.tight_layout()
    fig.savefig(output,bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--robust-root",type=Path,required=True)
    p.add_argument("--cv-root",type=Path,required=True)
    p.add_argument("--ablation-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument(
        "--cleaned-data-root",
        type=Path,
        help=(
            "Directory containing thalassemia_model_matrix_clean.csv, "
            "cohort_flow.csv and analysis_manifest.json. Default: "
            "<repository-root>/data/cleaned_phenotype_cohort, inferred "
            "from --output-root. Used to generate table_cleaning.tex."
        ),
        required=True
    )
    args=p.parse_args()
    root=args.output_root.resolve(); figures=root/"figures"; tables=root/"tables"; source=root/"source_data"
    for d in (figures,tables,source): d.mkdir(parents=True,exist_ok=True)

    robust=args.robust_root.resolve(); cv=args.cv_root.resolve(); ab=args.ablation_root.resolve()
    selected=pd.read_csv(robust/"combined_selected_common_threshold_test_results.csv")
    comp=pd.read_csv(robust/"combined_selected_vs_unpenalized_common_threshold.csv")
    audit=pd.read_csv(robust/"combined_lambda_selection_audit.csv")
    op=pd.read_csv(robust/"combined_selected_operating_point_comparison.csv")
    cv_selected,cv_baseline,cv_deltas=load_cv(cv)
    cv_summary=summarize_cv(cv_selected,cv_deltas)
    ablation=load_ablation(ab)
    ab_summary=summarize_ablation(ablation)

    # Machine-readable sources.
    for fn in [
        "combined_selected_common_threshold_test_results.csv",
        "combined_selected_optimized_threshold_test_results.csv",
        "combined_selected_fixed_threshold_test_results.csv",
        "combined_selected_operating_point_comparison.csv",
        "combined_selected_vs_unpenalized_common_threshold.csv",
        "combined_lambda_selection_audit.csv",
        "combined_common_threshold_all_lambda_results.csv",
        "combined_threshold_optimized_all_lambda_results.csv",
        "combined_all_lambda_results.csv",
    ]:
        copy_csv(robust/fn,source/fn)
    cv_selected.to_csv(source/"repeated_cv_selected_test_results_combined.csv",index=False)
    cv_baseline.to_csv(source/"repeated_cv_unpenalized_test_results_combined.csv",index=False)
    cv_deltas.to_csv(source/"repeated_cv_selected_vs_baseline_deltas.csv",index=False)
    cv_summary.to_csv(source/"repeated_cv_summary.csv",index=False)
    ablation.to_csv(source/"ablation_selected_results_combined.csv",index=False)
    ab_summary.to_csv(source/"ablation_summary.csv",index=False)

    write_primary_table(selected,tables/"table_selected_test_results.tex")
    write_delta_table(comp,tables/"table_selected_vs_baseline.tex")
    write_operating_table(op,tables/"table_operating_points.tex")
    write_cv_table(cv_summary,tables/"table_repeated_cv_summary.tex")
    write_cv_effect_table(cv_summary,tables/"table_cv_penalized_effects.tex")
    write_ablation_table(ab_summary,tables/"table_ablation_summary.tex")

    cleaned_data_root = (
        args.cleaned_data_root.resolve()
        if args.cleaned_data_root is not None
        else (root.parent / "data" / "cleaned_phenotype_cohort").resolve()
    )
    write_cleaning_table(
        cleaned_data_root=cleaned_data_root,
        tables_dir=tables,
        source_dir=source,
    )

    figure_validation(audit,figures/"figure2_validation_tradeoff.pdf")
    figure_holdout(comp,figures/"figure3_holdout_performance.pdf")
    figure_operating_points(op,figures/"figure6_operating_points.pdf")
    figure_cv(cv_selected,figures/"figure4_cross_validation.pdf")
    figure_ablation(ablation,figures/"figure5_ablation_controls.pdf")

if __name__=="__main__":
    main()
