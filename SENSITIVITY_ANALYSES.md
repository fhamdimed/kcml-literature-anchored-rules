# Governance safeguard sensitivity analyses

`run_sensitivity_analyses.py` implements two secondary robustness analyses while
leaving the prespecified primary KCML analysis unchanged.

## 1. Governance-safeguard sensitivity

This analysis reuses the already-trained common-threshold all-lambda results and
therefore does **not** retrain models. Each criterion is varied one at a time:

- balanced-accuracy tolerance: 0.005, 0.010 (primary), 0.020;
- ROC-AUC tolerance: 0.005, 0.010 (primary), 0.020;
- average-precision tolerance: 0.010, 0.020 (primary), 0.030;
- minimum relative binary-violation reduction: 0.30, 0.40 (primary), 0.50;
- soft violation remains required not to increase.

Run on the primary split:

```bash
python run_sensitivity_analyses.py safeguards \
  --primary-results-dir analysis_outputs/common_threshold_robust \
  --output analysis_outputs/sensitivity_analyses/safeguards
```

To also re-select within all repeated-CV folds:

```bash
python run_sensitivity_analyses.py safeguards \
  --primary-results-dir analysis_outputs/common_threshold_robust \
  --cv-root analysis_outputs/repeated_cv \
  --output analysis_outputs/sensitivity_analyses/safeguards
```

Main outputs:

- `safeguard_sensitivity_holdout.csv`
- `safeguard_sensitivity_cv_folds.csv`
- `safeguard_sensitivity_cv_selection_frequency.csv`
- `safeguard_sensitivity_manifest.json`

## 2. Relative rule-weight sensitivity

This analysis **does retrain models** because the weights enter the training
objective. The default schemes are:

- primary: `(1, 1, 1, 1, 1.5)`;
- equal: `(1, 1, 1, 1, 1)`;
- stronger LR05: `(1, 1, 1, 1, 2)`;
- down-weight LR02/LR04: `(1, 0.5, 1, 0.5, 1.5)`.

The last scheme is a sensitivity perturbation only and is not an estimated
clinical confidence scale.

Run the manuscript configuration:

```bash
python run_sensitivity_analyses.py weights \
  --data data/cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output analysis_outputs/sensitivity_analyses/weights \
  --algorithms xgboost lightgbm logistic neural \
  --schemes primary equal lr05_stronger lr02_lr04_downweighted \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --tree-rounds 500 \
  --tree-learning-rate 0.05 \
  --tree-early-stopping 50 \
  --nn-max-epochs 300 \
  --nn-patience 30 \
  --nn-restarts 3 \
  --nn-device cpu \
  --seed 42
```

Main outputs:

- `rule_weight_sensitivity_selected_test.csv`
- `rule_weight_sensitivity_deltas.csv`
- `rule_weight_sensitivity_manifest.json`
- per-scheme/per-algorithm all-lambda result folders.

## Interpretation

Exact stability of the selected penalty is not required. The robustness question
is whether the qualitative pattern persists: constraints are selected
selectively; retained constraints reduce encoded-rule discordance; their main
predictive effect is a sensitivity-specificity redistribution with small changes
in threshold-free discrimination; and stricter governance can appropriately
increase opt-out at lambda=0.
