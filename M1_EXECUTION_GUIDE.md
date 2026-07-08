# Apple Silicon execution guide

Use one unchanged Apple Silicon environment for the robust run, repeated cross-validation, and rule ablations.

The primary analysis uses one threshold learned from the unpenalized validation model and held fixed across all lambdas. Per-lambda optimized thresholds and threshold 0.5 are generated automatically as secondary analyses.

Create the log directory once:

```bash
mkdir -p logs
```

## 1. Robust run

```bash
python run_all_algorithms.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/common_threshold_robust \
  --algorithms xgboost lightgbm logistic neural \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --tree-rounds 500 \
  --tree-learning-rate 0.05 \
  --tree-early-stopping 50 \
  --nn-max-epochs 300 \
  --nn-patience 30 \
  --nn-restarts 3 \
  --nn-restart-aggregation mean_probability \
  --nn-device cpu \
  --nn-threads 2 \
  --seed 42 \
  2>&1 | tee logs/common_threshold_robust.log
```

After completion:

```bash
python summarize_threshold_strategies.py \
  --results-dir results/common_threshold_robust
```

## 2. Repeated cross-validation

### XGBoost

```bash
python run_repeated_cv.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/repeated_cv/xgboost \
  --algorithm xgboost \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --folds 5 \
  --repeats 2 \
  --validation-fraction 0.20 \
  --tree-rounds 500 \
  --seed 42 \
  2>&1 | tee logs/repeated_cv_xgboost.log
```

### LightGBM

```bash
python run_repeated_cv.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/repeated_cv/lightgbm \
  --algorithm lightgbm \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --folds 5 \
  --repeats 2 \
  --validation-fraction 0.20 \
  --tree-rounds 500 \
  --seed 42 \
  2>&1 | tee logs/repeated_cv_lightgbm.log
```

### Logistic regression

```bash
python run_repeated_cv.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/repeated_cv/logistic \
  --algorithm logistic \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --folds 5 \
  --repeats 2 \
  --validation-fraction 0.20 \
  --seed 42 \
  2>&1 | tee logs/repeated_cv_logistic.log
```

### Neural probability ensemble

```bash
python run_repeated_cv.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/repeated_cv/neural \
  --algorithm neural \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --folds 5 \
  --repeats 2 \
  --validation-fraction 0.20 \
  --nn-max-epochs 300 \
  --nn-patience 30 \
  --nn-restarts 3 \
  --nn-restart-aggregation mean_probability \
  --nn-device cpu \
  --nn-threads 2 \
  --seed 42 \
  2>&1 | tee logs/repeated_cv_neural.log
```

## 3. Rule ablations

### XGBoost

```bash
python run_rule_ablations.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/rule_ablations/xgboost \
  --algorithm xgboost \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --seed 42 \
  2>&1 | tee logs/rule_ablations_xgboost.log
```

### LightGBM

```bash
python run_rule_ablations.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/rule_ablations/lightgbm \
  --algorithm lightgbm \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --seed 42 \
  2>&1 | tee logs/rule_ablations_lightgbm.log
```

### Logistic regression

```bash
python run_rule_ablations.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/rule_ablations/logistic \
  --algorithm logistic \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --seed 42 \
  2>&1 | tee logs/rule_ablations_logistic.log
```

### Neural probability ensemble

```bash
python run_rule_ablations.py \
  --data cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output results/rule_ablations/neural \
  --algorithm neural \
  --lambdas 0 0.1 0.25 0.5 0.75 1 1.5 2 \
  --nn-max-epochs 300 \
  --nn-patience 30 \
  --nn-restarts 3 \
  --nn-restart-aggregation mean_probability \
  --nn-device cpu \
  --nn-threads 2 \
  --seed 42 \
  2>&1 | tee logs/rule_ablations_neural.log
```

## 4. Files to upload for manuscript revision

Create three archives containing CSV and JSON outputs:

```bash
zip -r common_threshold_robust_analysis.zip \
  results/common_threshold_robust \
  -i "*.csv" "*.json"

zip -r common_threshold_repeated_cv_analysis.zip \
  results/repeated_cv \
  -i "*.csv" "*.json"

zip -r common_threshold_rule_ablation_analysis.zip \
  results/rule_ablations \
  -i "*.csv" "*.json"
```

Upload all three archives. Also retain the execution logs, environment report, and SHA-256 hash of the cleaned CSV.
