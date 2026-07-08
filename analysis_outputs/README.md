# Analysis outputs

This directory contains the uploaded machine-readable outputs from the final common-threshold KCML experiments:

- `common_threshold_robust/`: primary holdout experiment for XGBoost, LightGBM, logistic regression and neural ensemble.
- `repeated_cv/`: repeated cross-validation outputs.
- `rule_ablations/`: single-rule, leave-one-out and corrupted-rule control experiments.

The principal tables for manuscript checking are:

- `common_threshold_robust/combined_selected_test_results.csv`
- `common_threshold_robust/combined_selected_operating_point_comparison.csv`
- `common_threshold_robust/combined_selected_vs_unpenalized_common_threshold.csv`
- `repeated_cv/repeated_cv_selected_test_results.csv`
- `repeated_cv/repeated_cv_aggregate.csv`
- `rule_ablations/ablation_selected_test_results.csv`

The archived input ZIP files supplied for packaging are retained in `archive_inputs/` for traceability.
