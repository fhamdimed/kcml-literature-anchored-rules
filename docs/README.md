# Rule provenance and epistemic status

`docs/RULE_PROVENANCE.csv` records the provenance and epistemic status of each rule-like object maintained in the implementation.

The primary KCML training bundle includes LR01--LR05 only. LR06 and LR07 reproduce selected same-source Chen/Huizhou thresholds and are retained only for optional descriptive auditing; they are excluded from the primary training penalty and primary holdout evaluation.

The table deliberately separates:

1. the published clinical or laboratory statement;
2. the Boolean KCML activation condition;
3. the preferred binary target used by the case-study penalty;
4. whether the threshold was directly cited or operationally selected;
5. whether the rule is a primary training rule, audit-only rule or quality-control flag.

This separation is important because not every rule has the same epistemic status. LR01 is closest to a directly literature-anchored directional rule, whereas LR02 and LR04 are operational binary mappings of more ambiguous phenotype signals, and LR05 is an investigator-specified composite formalization of a literature-supported pattern.
