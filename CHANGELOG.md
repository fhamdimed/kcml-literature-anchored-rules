# v1.1.0

## Added

- Governance-safeguard sensitivity analysis.
- Relative rule-weight sensitivity analysis.
- `run_sensitivity_analyses.py` for the secondary sensitivity analyses.
- `SENSITIVITY_ANALYSES.md` documenting sensitivity configurations, commands and outputs.
- Standalone algorithm runners:
  - `run_xgboost.py`
  - `run_lightgbm.py`
  - `run_logistic_regression.py`
  - `run_neural_network.py`

## Fixed

- Restored `run_neural_network.py`, which is referenced by the default isolated-neural execution path in `run_all_algorithms.py`.
- Clarified repository documentation so that standalone runners are distinguished from the shared model implementations in `kcml.factories` and `kcml/models/`.
- Standardized README examples on the `analysis_outputs/` result paths used by the archived analyses and manuscript-asset workflow.
- Corrected the README command for the root-level `generate_paper_assets.py` wrapper.

## Unchanged

- KCML training objective.
- Primary rule definitions and primary rule weights.
- Primary data split.
- Lambda grid.
- Original primary governance policy.
- Model architectures and model-specific optimization adapters.
- Originally reported primary results.

## Notes

The sensitivity analyses are secondary analyses added for robustness assessment. They do not redefine or replace the prespecified primary analysis. The standalone algorithm runners delegate model construction to the shared factories and do not introduce duplicate model implementations.
