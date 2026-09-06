# Knowledge-Constrained Machine Learning for Literature-Anchored Rules

This repository contains the code, rule definitions, processing workflow and machine-readable outputs used to reproduce the results and figures for the paper:

**Knowledge-Constrained Machine Learning for Governed Integration of Literature-Derived Rules in Thalassemia Genotype Classification**

The repository implements an algorithm-agnostic framework that integrates literature-anchored phenotype rules into supervised machine-learning training by combining:

- a shared patient-normalized rule-penalty objective,
- model-specific adapters for XGBoost, LightGBM, logistic regression and neural networks,
- common-threshold governance that separates constrained training from operating-point retuning,
- threshold-free soft rule-discordance audits,
- repeated cross-validation, rule ablations and corrupted-rule controls.

The thalassemia screening analysis is a reproducible case study using a public genotype-labelled cohort.

## Repository structure

```text
kcml-literature-anchored-rules/
├── LICENSE
├── README.md
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── data/
├── kcml/
├── paper/
├── analysis_outputs/
├── archive_inputs/
├── docs/
├── prepare_thalassemia_dataset.py
├── run_all_algorithms.py
├── run_xgboost.py
├── run_lightgbm.py
├── run_logistic_regression.py
├── run_neural_network.py
├── run_repeated_cv.py
├── run_rule_ablations.py
├── run_sensitivity_analyses.py
├── analyze_holdout_predictions.py
├── summarize_threshold_strategies.py
├── generate_paper_assets.py
├── SENSITIVITY_ANALYSES.md
└── CHANGELOG.md
```

### Main source files

- `prepare_thalassemia_dataset.py`  
  Cleans the public phenotype workbook, applies HbF-resolution logic, and writes the modelling matrix.

- `kcml/rules.py`  
  Central definition of the primary literature-anchored rules, same-source audit-only rules, quality-control rules and shared rule-penalty utilities.

- `kcml/experiment.py`  
  Main experiment engine: data splitting, imputation, model fitting, validation governance, threshold selection, metric calculation and output writing.

- `kcml/metrics.py`  
  Classification, ranking, calibration, binary rule-violation and threshold-free soft rule-discordance metrics.

- `kcml/models/xgboost_model.py`  
  XGBoost adapter implementing the supervised objective plus rule-gradient and Hessian contributions.

- `kcml/models/lightgbm_model.py`  
  LightGBM adapter implementing the supervised objective plus rule-gradient and Hessian contributions.

- `kcml/models/logistic_model.py`  
  Logistic-regression adapter with the shared rule penalty.

- `kcml/models/neural_model.py`  
  Neural-network adapter using independently early-stopped restarts and mean-probability ensembling.

- `run_all_algorithms.py`  
  Runs the primary robust experiment for all requested algorithms. XGBoost, LightGBM and logistic regression are constructed directly through `kcml.factories`; the neural model is launched through `run_neural_network.py` in a fresh Python subprocess by default to preserve the isolated PyTorch execution used in the reported workflow.

- `run_xgboost.py`  
  Standalone XGBoost command-line runner. Model construction remains centralized in `kcml.factories.make_xgboost_factory`.

- `run_lightgbm.py`  
  Standalone LightGBM command-line runner. Model construction remains centralized in `kcml.factories.make_lightgbm_factory`.

- `run_logistic_regression.py`  
  Standalone logistic-regression command-line runner. Model construction remains centralized in `kcml.factories.make_logistic_factory`.

- `run_neural_network.py`  
  Standalone neural command-line runner and the isolated subprocess entry point used by `run_all_algorithms.py`. The neural implementation remains centralized in `kcml.factories.make_neural_factory` and `kcml/models/neural_model.py`.

- `run_repeated_cv.py`  
  Runs repeated train/validation/test resampling with the same common-threshold governance.

- `run_rule_ablations.py`  
  Runs all-rules, single-rule, leave-one-rule-out and corrupted-rule control experiments.

- `run_sensitivity_analyses.py`  
  Runs secondary sensitivity analyses for governance safeguards and relative rule weights without altering the prespecified primary KCML analysis. Safeguard sensitivity re-selects from existing all-lambda results; rule-weight sensitivity refits models because rule weights enter the training objective.

- `analyze_holdout_predictions.py`  
  Reuses archived primary holdout predictions to generate calibration summaries and curves, paired bootstrap uncertainty intervals and a hypothetical confirmatory-referral translation. It does not retrain models.

- `SENSITIVITY_ANALYSES.md`  
  Documents the sensitivity configurations, execution commands and generated outputs.

- `summarize_threshold_strategies.py`  
  Generates compact comparison tables for common-threshold, per-penalty optimized-threshold and fixed-0.5 operating points.

- `paper/scripts/generate_common_threshold_assets.py`  
  Low-level manuscript asset generator. It requires explicit result-directory arguments.

- `generate_paper_assets.py`  
  Convenience wrapper that regenerates manuscript tables, figures and source CSVs from the archived result folders using repository-default paths.

## Python version

This repository was executed using:

- **Python 3.10**

## Requirements

Create the conda environment with:

```bash
conda env create -f environment.yml
conda activate kcml_clean
```

Alternatively, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The repository separates core and optional neural dependencies:

```text
requirements-core.txt
requirements-neural.txt
requirements.txt
```

## Input data

The original genotype and phenotype workbook is not redistributed here. Download it from the supplementary material accompanying the source cohort publication and place it in:

```text
data/genotype_datadeposition.xlsx
```

Then generate the cleaned model matrix with:

```bash
python prepare_thalassemia_dataset.py \
  --input data/genotype_datadeposition.xlsx \
  --output data/cleaned_phenotype_cohort
```

This creates files such as:

```text
data/cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv
data/cleaned_phenotype_cohort/cleaning_summary.json
data/cleaned_phenotype_cohort/rule_audit_flags.csv
```

## Reproducing the primary experiment

After generating the cleaned model matrix, run:

```bash
python run_all_algorithms.py \
  --data data/cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output analysis_outputs/common_threshold_robust \
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
  --seed 42
```

Then summarize the three operating-point strategies:

```bash
python summarize_threshold_strategies.py \
  --results-dir analysis_outputs/common_threshold_robust
```

Important primary outputs include:

```text
analysis_outputs/common_threshold_robust/combined_common_threshold_all_lambda_results.csv
analysis_outputs/common_threshold_robust/combined_selected_test_results.csv
analysis_outputs/common_threshold_robust/combined_selected_vs_unpenalized_common_threshold.csv
analysis_outputs/common_threshold_robust/combined_selected_operating_point_comparison.csv
```

## Repeated cross-validation and ablations

The exact commands used for repeated cross-validation and rule-ablation experiments are documented in:

```text
M1_EXECUTION_GUIDE.md
```

The final workflow includes:

- repeated cross-validation for XGBoost, LightGBM, logistic regression and neural networks;
- all-rules, single-rule and leave-one-rule-out configurations;
- permuted, column-permuted and reversed-target corrupted-rule controls;
- common-threshold, per-penalty optimized-threshold and fixed-0.5 analyses;
- threshold-free soft rule-discordance metrics.

## Sensitivity analyses

The revision analyses assess whether constraint selection depends on the investigator-defined governance safeguards or on one specific relative rule-weight configuration. They are secondary analyses and do not redefine the prespecified primary analysis.

### Governance-safeguard sensitivity

This analysis reuses the existing all-lambda results and therefore does not retrain models. Each governance criterion is varied one at a time around the primary setting.

```bash
python run_sensitivity_analyses.py safeguards \
  --primary-results-dir analysis_outputs/common_threshold_robust \
  --cv-root analysis_outputs/repeated_cv \
  --output analysis_outputs/sensitivity_analyses/safeguards
```

Main outputs include:

```text
safeguard_sensitivity_holdout.csv
safeguard_sensitivity_cv_folds.csv
safeguard_sensitivity_cv_selection_frequency.csv
safeguard_sensitivity_manifest.json
```

### Relative rule-weight sensitivity

This analysis refits the models because rule weights enter the training objective. The tested schemes include the primary weights, equal weights, stronger LR05 weighting and reduced relative influence of LR02 and LR04. Alternative weights are sensitivity perturbations rather than estimated clinical confidence values.

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

Main outputs include:

```text
rule_weight_sensitivity_selected_test.csv
rule_weight_sensitivity_deltas.csv
rule_weight_sensitivity_manifest.json
```

See `SENSITIVITY_ANALYSES.md` for the complete definitions and interpretation notes.

## Holdout calibration, uncertainty and referral analysis

The archived patient-level primary holdout predictions can be reused without model retraining to assess calibration, paired bootstrap uncertainty and the operating-point implications for hypothetical molecular-confirmation referrals.

```bash
python analyze_holdout_predictions.py \
  --results-dir analysis_outputs/common_threshold_robust \
  --output analysis_outputs/holdout_analysis \
  --bootstrap-replicates 2000 \
  --calibration-bins 10
```

Main outputs include:

```text
calibration_summary.csv
calibration_curves.csv
bootstrap_metric_intervals.csv
confirmatory_referral_summary.csv
holdout_analysis_manifest.json
```

Bootstrap intervals use paired patient-level resampling stratified by true outcome class. The confirmatory-referral summary assumes that every common-threshold predicted-positive case would be referred for molecular confirmation and should be interpreted as an operating-point translation rather than a clinical utility analysis or deployment recommendation.

## Generating manuscript figures and tables

To regenerate tables and figures from the archived CSV outputs using the repository-default paths:

```bash
python generate_paper_assets.py
```

This wrapper reads from:

```text
data/cleaned_phenotype_cohort/
analysis_outputs/common_threshold_robust/
analysis_outputs/repeated_cv/
analysis_outputs/rule_ablations/
```

and writes updated assets to:

```text
paper/figures/
paper/tables/
paper/source_data/
```

The equivalent explicit command is:

```bash
python paper/scripts/generate_common_threshold_assets.py \
  --cleaned-data-root data/cleaned_phenotype_cohort \
  --robust-root analysis_outputs/common_threshold_robust \
  --cv-root analysis_outputs/repeated_cv \
  --ablation-root analysis_outputs/rule_ablations \
  --output-root paper
```

## Rule provenance

The paper's primary analyses use only the five prespecified rules LR01--LR05 implemented in `kcml/rules.py`.

Detailed rule provenance files are provided in:

```text
docs/RULE_PROVENANCE.csv
docs/RULE_PROVENANCE.md
docs/rules_uploaded_for_provenance.py
```

These files document the epistemic status, literature basis, operationalization and intended use of each rule-like object maintained in the implementation.

## Important reproducibility notes

### 1. Common-threshold governance

The primary analysis learns the decision threshold from the unpenalized validation model and holds it fixed across all penalty strengths for the same algorithm. This isolates the effect of constrained training from per-lambda threshold retuning.

### 2. Neural-network repeatability

The neural model uses three independently early-stopped restarts and averages predicted probabilities across restarts. All reported neural runs used CPU execution. By default, `run_all_algorithms.py` launches `run_neural_network.py` in a fresh Python subprocess, while the underlying neural implementation remains centralized in `kcml.factories` and `kcml/models/neural_model.py`.

### 3. Source workbook

The source workbook should be downloaded from the public cohort publication. This repository provides the cleaning code and analysis workflow, but does not redistribute the original workbook.

### 4. Derived outputs

The `analysis_outputs/` directory contains machine-readable CSV files used to support the reported tables, figures and secondary sensitivity analyses. Large model binaries are intentionally not required for paper reproduction.

## Suggested execution order

### If regenerating everything from the public source workbook:

```bash
python prepare_thalassemia_dataset.py \
  --input data/genotype_datadeposition.xlsx \
  --output data/cleaned_phenotype_cohort

python run_all_algorithms.py \
  --data data/cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
  --output analysis_outputs/common_threshold_robust \
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
  --seed 42

python summarize_threshold_strategies.py \
  --results-dir analysis_outputs/common_threshold_robust
```

Then run the repeated-CV and ablation commands in `M1_EXECUTION_GUIDE.md`. The secondary governance and rule-weight sensitivity analyses can then be run with `run_sensitivity_analyses.py` as described above; they do not replace the primary analysis.

## License

This repository is distributed under the MIT License. See `LICENSE`.

## Citation

If you use this repository, please cite the corresponding paper and the archived software release.

## Contact

Fayçal Hamdi, Ph.D  
Email: fhamdimed@gmail.com
