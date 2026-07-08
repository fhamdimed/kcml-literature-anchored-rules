"""Factory helpers used by command-line experiment scripts.

Imports are intentionally lazy so that XGBoost, LightGBM, and logistic
regression can run even when PyTorch is not installed or is temporarily broken.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def make_xgboost_factory(
    enabled_rules: Iterable[str] | None = None,
    rule_weights: Mapping[str, float] | None = None,
    rule_control: str = "none",
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
):
    from .models.xgboost_model import RulePenalizedXGBoost

    return lambda penalty: RulePenalizedXGBoost(
        penalty_multiplier=penalty,
        enabled_rules=enabled_rules,
        rule_weights=rule_weights,
        rule_control=rule_control,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )


def make_lightgbm_factory(
    enabled_rules: Iterable[str] | None = None,
    rule_weights: Mapping[str, float] | None = None,
    rule_control: str = "none",
    random_state: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 15,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
):
    from .models.lightgbm_model import RulePenalizedLightGBM

    return lambda penalty: RulePenalizedLightGBM(
        penalty_multiplier=penalty,
        enabled_rules=enabled_rules,
        rule_weights=rule_weights,
        rule_control=rule_control,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )


def make_logistic_factory(
    enabled_rules: Iterable[str] | None = None,
    rule_weights: Mapping[str, float] | None = None,
    rule_control: str = "none",
    random_state: int = 42,
    l2_strength: float = 1.0,
    max_iter: int = 5000,
    verbose: bool = True,
):
    from .models.logistic_model import RulePenalizedLogisticRegression

    return lambda penalty: RulePenalizedLogisticRegression(
        penalty_multiplier=penalty,
        enabled_rules=enabled_rules,
        rule_weights=rule_weights,
        rule_control=rule_control,
        random_state=random_state,
        l2_strength=l2_strength,
        max_iter=max_iter,
        verbose=verbose,
    )


def make_neural_factory(
    enabled_rules: Iterable[str] | None = None,
    rule_weights: Mapping[str, float] | None = None,
    rule_control: str = "none",
    random_state: int = 42,
    architecture: str = "mlp",
    hidden_sizes: Sequence[int] = (32, 16),
    dropout: float = 0.10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 500,
    patience: int = 50,
    n_restarts: int = 3,
    restart_aggregation: str = "mean_probability",
    deterministic: bool = True,
    device: str = "cpu",
    verbose: bool = True,
):
    try:
        from .models.neural_model import RulePenalizedNeuralNetwork
    except Exception as exc:
        raise RuntimeError(
            "The neural model could not import PyTorch. The other algorithms can "
            "still be run by omitting `neural` and `neural_linear`. For the neural "
            "model, create the clean environment described in README.md."
        ) from exc

    return lambda penalty: RulePenalizedNeuralNetwork(
        penalty_multiplier=penalty,
        enabled_rules=enabled_rules,
        rule_weights=rule_weights,
        rule_control=rule_control,
        random_state=random_state,
        architecture=architecture,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        n_restarts=n_restarts,
        restart_aggregation=restart_aggregation,
        deterministic=deterministic,
        device=device,
        verbose=verbose,
    )
