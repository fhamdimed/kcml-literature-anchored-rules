"""Model implementations with lazy imports."""

from __future__ import annotations

__all__ = [
    "RulePenalizedXGBoost",
    "RulePenalizedLightGBM",
    "RulePenalizedLogisticRegression",
    "RulePenalizedNeuralNetwork",
]


def __getattr__(name: str):
    if name == "RulePenalizedXGBoost":
        from .xgboost_model import RulePenalizedXGBoost
        return RulePenalizedXGBoost
    if name == "RulePenalizedLightGBM":
        from .lightgbm_model import RulePenalizedLightGBM
        return RulePenalizedLightGBM
    if name == "RulePenalizedLogisticRegression":
        from .logistic_model import RulePenalizedLogisticRegression
        return RulePenalizedLogisticRegression
    if name == "RulePenalizedNeuralNetwork":
        from .neural_model import RulePenalizedNeuralNetwork
        return RulePenalizedNeuralNetwork
    raise AttributeError(name)
