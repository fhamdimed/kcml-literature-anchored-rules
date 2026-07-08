"""PyTorch linear and multilayer neural models with shared rule penalties.

Every restart is trained and early-stopped independently.  By default,
predictions are the arithmetic mean of the restart-level probabilities.  This
reduces sensitivity to a single narrowly winning restart and is more stable
than selecting one restart by its validation loss.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..preprocessing import MedianFeatureImputer
from ..rules import RuleCatalog


def _configure_torch_threads() -> None:
    """Apply a conservative per-process CPU thread limit when requested."""
    raw = os.environ.get("KCML_TORCH_THREADS", "").strip()
    if not raw:
        return
    try:
        threads = max(1, int(raw))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ValueError as exc:
        raise ValueError("KCML_TORCH_THREADS must be an integer >= 1") from exc


def _configure_determinism(enabled: bool) -> None:
    """Request deterministic PyTorch operations in the current environment."""
    if not enabled:
        return
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _require_numpy_bridge() -> None:
    """Fail early with an actionable message for NumPy/PyTorch ABI mismatches."""
    try:
        probe = np.zeros(1, dtype=np.float32)
        torch.from_numpy(probe)
    except Exception as exc:
        raise RuntimeError(
            "PyTorch cannot exchange arrays with the installed NumPy build. "
            "This usually means PyTorch was compiled against NumPy 1.x while "
            "NumPy 2.x is installed. Create a clean environment or reinstall "
            "with `python -m pip install --force-reinstall 'numpy>=1.26,<2'` "
            "and then reinstall PyTorch. See README.md for exact commands."
        ) from exc


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _LinearLogit(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.output = nn.Linear(n_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(1)


class _MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class RulePenalizedNeuralNetwork:
    def __init__(
        self,
        penalty_multiplier: float,
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
        gradient_clip: float = 5.0,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        if penalty_multiplier < 0:
            raise ValueError("penalty_multiplier must be non-negative")
        if architecture not in {"linear", "mlp"}:
            raise ValueError("architecture must be 'linear' or 'mlp'")
        if restart_aggregation not in {"mean_probability", "best"}:
            raise ValueError(
                "restart_aggregation must be 'mean_probability' or 'best'"
            )
        if n_restarts < 1:
            raise ValueError("n_restarts must be at least 1")

        self.penalty_multiplier = float(penalty_multiplier)
        self.enabled_rules = tuple(enabled_rules) if enabled_rules is not None else None
        self.rule_weights = dict(rule_weights or {})
        self.rule_control = rule_control
        self.random_state = int(random_state)
        self.architecture = architecture
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.n_restarts = int(n_restarts)
        self.restart_aggregation = restart_aggregation
        self.deterministic = bool(deterministic)
        self.gradient_clip = float(gradient_clip)
        self.device = _resolve_device(device)
        self.verbose = bool(verbose)

        self.catalog = RuleCatalog()
        self.imputer = MedianFeatureImputer()
        self.scaler = StandardScaler()
        self.feature_names: list[str] | None = None

        # ``model`` and ``best_restart_`` are retained for backwards-compatible
        # diagnostics.  Predictions use ``models_`` when mean aggregation is active.
        self.model: nn.Module | None = None
        self.models_: list[nn.Module] = []
        self.training_history_: list[dict[str, float]] = []
        self.restart_summary_: list[dict[str, float | int]] = []
        self.best_restart_: int | None = None
        self.best_validation_loss_: float | None = None
        self.ensemble_validation_loss_: float | None = None

    def _build_model(self, n_features: int) -> nn.Module:
        if self.architecture == "linear":
            return _LinearLogit(n_features)
        return _MLP(n_features, self.hidden_sizes, self.dropout)

    def _combined_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rule_mask: torch.Tensor,
        rule_targets: torch.Tensor,
        rule_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data_loss = F.binary_cross_entropy_with_logits(logits, labels)
        if rule_mask.shape[1] == 0 or self.penalty_multiplier == 0:
            rule_loss = logits.new_tensor(0.0)
        else:
            expanded_logits = logits[:, None].expand(-1, rule_mask.shape[1])
            expanded_targets = rule_targets[None, :].expand_as(expanded_logits)
            per_rule = F.binary_cross_entropy_with_logits(
                expanded_logits, expanded_targets, reduction="none"
            )
            rule_loss = (
                per_rule * rule_mask * rule_weights[None, :]
            ).sum() / logits.shape[0]
        total = data_loss + self.penalty_multiplier * rule_loss
        return total, data_loss, rule_loss

    @staticmethod
    def _probabilities_to_logits(probabilities: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(probabilities.dtype).eps
        return torch.logit(probabilities.clamp(min=eps, max=1.0 - eps))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_validation: pd.DataFrame,
        y_validation: pd.Series,
    ) -> "RulePenalizedNeuralNetwork":
        _configure_torch_threads()
        _configure_determinism(self.deterministic)
        _require_numpy_bridge()

        if self.verbose:
            print(
                f"Preparing neural tensors: device={self.device}, "
                f"train_rows={len(X)}, validation_rows={len(X_validation)}, "
                f"features={X.shape[1]}, "
                f"restart_aggregation={self.restart_aggregation}",
                flush=True,
            )

        self.feature_names = X.columns.tolist()
        X_train_original = X[self.feature_names].apply(pd.to_numeric, errors="coerce")
        X_validation_original = X_validation[self.feature_names].apply(
            pd.to_numeric, errors="coerce"
        )
        self.imputer.fit(X_train_original, self.feature_names)
        X_train_imputed = self.imputer.transform_array(
            X_train_original, dtype=np.float32
        )
        X_validation_imputed = self.imputer.transform_array(
            X_validation_original, dtype=np.float32
        )
        X_train = self.scaler.fit_transform(X_train_imputed).astype(np.float32)
        X_val = self.scaler.transform(X_validation_imputed).astype(np.float32)
        y_train = np.asarray(y, dtype=np.float32)
        y_val = np.asarray(y_validation, dtype=np.float32)

        train_bundle = self.catalog.build_bundle(
            X_train_original,
            enabled_rules=self.enabled_rules,
            weight_overrides=self.rule_weights,
            control=self.rule_control,
            random_state=self.random_state,
        )
        validation_bundle = self.catalog.build_bundle(
            X_validation_original,
            enabled_rules=self.enabled_rules,
            weight_overrides=self.rule_weights,
            control=self.rule_control,
            random_state=self.random_state,
        )

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(train_bundle.mask.astype(np.float32)),
        )
        validation_tensors = (
            torch.from_numpy(X_val).to(self.device),
            torch.from_numpy(y_val).to(self.device),
            torch.from_numpy(validation_bundle.mask.astype(np.float32)).to(self.device),
        )
        rule_targets = torch.from_numpy(train_bundle.targets.astype(np.float32)).to(
            self.device
        )
        rule_weights = torch.from_numpy(train_bundle.weights.astype(np.float32)).to(
            self.device
        )
        validation_targets = torch.from_numpy(
            validation_bundle.targets.astype(np.float32)
        ).to(self.device)
        validation_weights = torch.from_numpy(
            validation_bundle.weights.astype(np.float32)
        ).to(self.device)

        fitted_models: list[nn.Module] = []
        all_history: list[dict[str, float]] = []
        restart_summary: list[dict[str, float | int]] = []
        best_individual_loss = float("inf")
        best_individual_restart = 0

        for restart in range(self.n_restarts):
            seed = self.random_state + restart
            if self.verbose:
                print(
                    f"Starting neural restart {restart + 1}/{self.n_restarts} "
                    f"(seed={seed})",
                    flush=True,
                )
            _set_seed(seed)
            model = self._build_model(X_train.shape[1]).to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            generator = torch.Generator()
            generator.manual_seed(seed)
            loader = DataLoader(
                train_dataset,
                batch_size=min(self.batch_size, len(train_dataset)),
                shuffle=True,
                generator=generator,
                num_workers=0,
                drop_last=False,
            )

            best_loss = float("inf")
            best_epoch = 0
            best_state: dict[str, torch.Tensor] | None = None
            no_improvement = 0

            for epoch in range(1, self.max_epochs + 1):
                model.train()
                train_total_sum = 0.0
                train_count = 0
                for features, labels, rule_mask in loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    rule_mask = rule_mask.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    total_loss, _, _ = self._combined_loss(
                        logits, labels, rule_mask, rule_targets, rule_weights
                    )
                    total_loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip
                        )
                    optimizer.step()
                    train_total_sum += float(total_loss.detach()) * len(features)
                    train_count += len(features)

                model.eval()
                with torch.no_grad():
                    val_logits = model(validation_tensors[0])
                    val_total, val_data, val_rule = self._combined_loss(
                        val_logits,
                        validation_tensors[1],
                        validation_tensors[2],
                        validation_targets,
                        validation_weights,
                    )
                validation_loss = float(val_total.detach().cpu())
                row = {
                    "restart": float(restart),
                    "seed": float(seed),
                    "epoch": float(epoch),
                    "train_combined_loss": train_total_sum / max(train_count, 1),
                    "validation_combined_loss": validation_loss,
                    "validation_data_loss": float(val_data.detach().cpu()),
                    "validation_rule_loss": float(val_rule.detach().cpu()),
                }
                all_history.append(row)

                if validation_loss < best_loss - 1e-8:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1

                if self.verbose and (epoch == 1 or epoch % 100 == 0):
                    print(
                        f"NN restart={restart + 1}/{self.n_restarts}, "
                        f"epoch={epoch}, validation_loss={validation_loss:.6f}",
                        flush=True,
                    )
                if no_improvement >= self.patience:
                    break

            if best_state is None:
                raise RuntimeError("Neural network training did not produce a model state")

            fitted = self._build_model(X_train.shape[1]).to(self.device)
            fitted.load_state_dict(best_state)
            fitted.eval()
            fitted_models.append(fitted)
            restart_summary.append(
                {
                    "restart": restart,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_combined_loss": best_loss,
                }
            )
            if best_loss < best_individual_loss:
                best_individual_loss = best_loss
                best_individual_restart = restart

            if self.verbose:
                print(
                    f"Completed neural restart {restart + 1}/{self.n_restarts}; "
                    f"best_epoch={best_epoch}, best_validation_loss={best_loss:.6f}",
                    flush=True,
                )

        self.models_ = fitted_models
        self.best_restart_ = best_individual_restart
        self.model = fitted_models[best_individual_restart]
        self.best_validation_loss_ = best_individual_loss
        self.training_history_ = all_history
        self.restart_summary_ = restart_summary

        with torch.no_grad():
            if self.restart_aggregation == "mean_probability":
                restart_probabilities = torch.stack(
                    [torch.sigmoid(model(validation_tensors[0])) for model in self.models_],
                    dim=0,
                )
                ensemble_probabilities = restart_probabilities.mean(dim=0)
                ensemble_logits = self._probabilities_to_logits(ensemble_probabilities)
            else:
                ensemble_logits = self.model(validation_tensors[0])
            ensemble_total, _, _ = self._combined_loss(
                ensemble_logits,
                validation_tensors[1],
                validation_tensors[2],
                validation_targets,
                validation_weights,
            )
        self.ensemble_validation_loss_ = float(ensemble_total.detach().cpu())

        if self.verbose:
            if self.restart_aggregation == "mean_probability":
                print(
                    f"Ensembled {len(self.models_)} neural restarts by mean "
                    f"probability; ensemble validation combined loss="
                    f"{self.ensemble_validation_loss_:.6f}; lowest individual "
                    f"restart={self.best_restart_ + 1}",
                    flush=True,
                )
            else:
                print(
                    f"Selected neural restart {self.best_restart_ + 1}; "
                    f"best validation combined loss={self.best_validation_loss_:.6f}",
                    flush=True,
                )
        return self

    def _scaled_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        if not self.models_ or self.feature_names is None:
            raise ValueError("Model has not been fitted")
        missing = [name for name in self.feature_names if name not in X.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        _require_numpy_bridge()
        imputed = self.imputer.transform_array(
            X[self.feature_names], dtype=np.float32
        )
        scaled = self.scaler.transform(imputed).astype(np.float32)
        return torch.from_numpy(scaled).to(self.device)

    def restart_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Return one probability column per fitted restart."""
        features = self._scaled_tensor(X)
        with torch.no_grad():
            probabilities = []
            for model in self.models_:
                model.eval()
                probabilities.append(torch.sigmoid(model(features)).detach().cpu().numpy())
        return np.column_stack(probabilities).astype(float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        restart_probabilities = self.restart_probabilities(X)
        if self.restart_aggregation == "mean_probability":
            return restart_probabilities.mean(axis=1)
        if self.best_restart_ is None:
            raise ValueError("Best restart is unavailable")
        return restart_probabilities[:, self.best_restart_]

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.clip(self.predict_proba(X), 1e-12, 1.0 - 1e-12)
        return np.log(probabilities / (1.0 - probabilities))

    def save(self, path: str | Path) -> None:
        if not self.models_ or self.feature_names is None:
            raise ValueError("Model has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dicts": [model.state_dict() for model in self.models_],
            # Retain one state for older readers; predictions in this package use
            # all states when restart_aggregation is mean_probability.
            "state_dict": self.model.state_dict() if self.model is not None else None,
            "architecture": self.architecture,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "feature_names": self.feature_names,
            "imputer_statistics": self.imputer.statistics_,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "penalty_multiplier": self.penalty_multiplier,
            "enabled_rules": self.enabled_rules,
            "rule_weights": self.rule_weights,
            "rule_control": self.rule_control,
            "restart_aggregation": self.restart_aggregation,
            "restart_summary": self.restart_summary_,
        }
        torch.save(checkpoint, str(path) + ".model.pt")
        metadata = {
            "algorithm": f"neural_network_{self.architecture}",
            "penalty_multiplier": self.penalty_multiplier,
            "enabled_rules": self.enabled_rules,
            "rule_weights": self.rule_weights,
            "rule_control": self.rule_control,
            "feature_names": self.feature_names,
            "imputation_strategy": "training_median",
            "imputer_statistics": self.imputer.statistics_.tolist(),
            "restart_aggregation": self.restart_aggregation,
            "n_fitted_restarts": len(self.models_),
            "restart_summary": self.restart_summary_,
            "lowest_loss_restart": self.best_restart_,
            "lowest_individual_validation_loss": self.best_validation_loss_,
            "ensemble_validation_loss": self.ensemble_validation_loss_,
            "deterministic_algorithms": self.deterministic,
            "device_used": str(self.device),
        }
        Path(str(path) + ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if self.training_history_:
            pd.DataFrame(self.training_history_).to_csv(
                str(path) + ".history.csv", index=False
            )
        if self.restart_summary_:
            pd.DataFrame(self.restart_summary_).to_csv(
                str(path) + ".restart_summary.csv", index=False
            )
