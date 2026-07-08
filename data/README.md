# Data directory

The original genotype and phenotype workbook is **not redistributed here**.

Users should download the public Chen et al. source workbook from ScienceDB:

```text
https://doi.org/10.57760/sciencedb.21443
```

After downloading, place the workbook in this directory as:

```text
data/genotype_datadeposition.xlsx
```

## Preparing the cleaned cohort

From the repository root, run:

```bash
python prepare_thalassemia_dataset.py \
  --input data/genotype_datadeposition.xlsx \
  --output-dir data/cleaned_phenotype_cohort
```

By default, the script processes the `Phenotype cohort` sheet.

## Cleaned analysis files

The cleaning script writes only the principal derived files required for
reproduction:

```text
data/cleaned_phenotype_cohort/
├── thalassemia_model_matrix_clean.csv
├── cohort_flow.csv
└── analysis_manifest.json
```

### File descriptions

- `thalassemia_model_matrix_clean.csv`  
  Model-ready phenotype matrix used by the training, cross-validation and
  ablation scripts.

- `cohort_flow.csv`  
  Stepwise summary from the deposited phenotype cohort to the final analytical
  cohort.

- `analysis_manifest.json`  
  Machine-readable record of the source workbook hash, cleaning parameters,
  cohort counts, missingness in the model matrix, genotype distribution,
  HbF/HbA cleaning summaries, output hashes and recommended predictor set.

## Downstream use

The modelling scripts expect the cleaned matrix at:

```text
data/cleaned_phenotype_cohort/thalassemia_model_matrix_clean.csv
```

No predictor imputation is performed during cohort preparation. Any remaining
missing predictor values are imputed only within the relevant training partition
by the downstream modelling pipeline.
