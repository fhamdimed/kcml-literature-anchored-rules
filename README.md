# Knowledge-Constrained Machine Learning for Literature-Anchored Rules

This repository contains the code, rule definitions, processing workflow and machine-readable outputs used to reproduce the results and figures for the paper:

**Knowledge-Constrained Machine Learning: Translating Literature-Derived Rules into Governed Model Training for Thalassemia Screening**

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
├── run_repeated_cv.py
├── run_rule_ablations.py
├── summarize_threshold_strategies.py
└── generate_paper_assets.py
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
  Runs the primary robust experiment for all requested algorithms.

- `run_repeated_cv.py`  
  Runs repeated train/validation/test resampling with the same common-threshold governance.

- `run_rule_ablations.py`  
  Runs all-rules, single-rule, leave-one-rule-out and corrupted-rule control experiments.

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
  --seed 42
```

Then summarize the three operating-point strategies:

```bash
python summarize_threshold_strategies.py \
  --results-dir results/common_threshold_robust
```

Important primary outputs include:

```text
results/common_threshold_robust/combined_common_threshold_all_lambda_results.csv
results/common_threshold_robust/combined_selected_test_results.csv
results/common_threshold_robust/combined_selected_vs_unpenalized_common_threshold.csv
results/common_threshold_robust/combined_selected_operating_point_comparison.csv
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

## Generating manuscript figures and tables

To regenerate tables and figures from the archived CSV outputs using the repository-default paths:

```bash
python scripts/generate_paper_assets.py
```

This wrapper reads from:

```text
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

The neural model uses three independently early-stopped restarts and averages predicted probabilities across restarts. All reported neural runs used CPU execution.

### 3. Source workbook

The source workbook should be downloaded from the public cohort publication. This repository provides the cleaning code and analysis workflow, but does not redistribute the original workbook.

### 4. Derived outputs

The `analysis_outputs/` directory contains machine-readable CSV files used to support the reported tables and figures. Large model binaries are intentionally not required for paper reproduction.

## Suggested execution order

### If regenerating everything from the public source workbook:

```bash
python prepare_thalassemia_dataset.py \
  --input data/genotype_datadeposition.xlsx \
  --output data/cleaned_phenotype_cohort

python run_all_algorithms.py \
  --data data/cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv \
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
  --seed 42

python summarize_threshold_strategies.py \
  --results-dir results/common_threshold_robust
```

Then run the repeated-CV and ablation commands in `M1_EXECUTION_GUIDE.md`.

## License

This repository is distributed under the MIT License. See `LICENSE`.

## Citation

If you use this repository, please cite the corresponding paper and the archived software release.

## Contact

Fayçal Hamdi, Ph.D  
Email: fhamdimed@gmail.com
