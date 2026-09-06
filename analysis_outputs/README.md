# Analysis outputs

This directory contains the uploaded machine-readable outputs from the final common-threshold KCML experiments:

- `common_threshold_robust/`: primary holdout experiment for XGBoost, LightGBM, logistic regression and neural ensemble.
- `repeated_cv/`: repeated cross-validation outputs.
- `rule_ablations/`: single-rule, leave-one-out and corrupted-rule control experiments.
- `sensitivity_analyses/`: governance-safeguard and relative rule-weight sensitivity analyses.
- `holdout_analysis/`: calibration, paired bootstrap uncertainty and hypothetical confirmatory-referral analyses based on archived holdout predictions.

The principal tables for manuscript checking are:

- `common_threshold_robust/combined_selected_test_results.csv`
- `common_threshold_robust/combined_selected_operating_point_comparison.csv`
- `common_threshold_robust/combined_selected_vs_unpenalized_common_threshold.csv`
- `repeated_cv/repeated_cv_selected_test_results.csv`
- `repeated_cv/repeated_cv_aggregate.csv`
- `rule_ablations/ablation_selected_test_results.csv`
- `holdout_analysis/calibration_summary.csv`
- `holdout_analysis/bootstrap_metric_intervals.csv`
- `holdout_analysis/confirmatory_referral_summary.csv`
- `sensitivity_analyses/safeguards/safeguard_sensitivity_holdout.csv`
- `sensitivity_analyses/safeguards/safeguard_sensitivity_cv_selection_frequency.csv`
- `sensitivity_analyses/weights/rule_weight_sensitivity_selected_test.csv`
- `sensitivity_analyses/weights/rule_weight_sensitivity_deltas.csv`

The archived input ZIP files supplied for packaging are retained in `archive_inputs/` for traceability.
