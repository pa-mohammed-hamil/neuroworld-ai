
"""
NeuroWorld AI
File: neuroworld/prediction/next_state.py

Neural Next-State Prediction Engine.

Features:
    - State-action-conditioned dynamics
    - Residual next-state prediction
    - Deterministic prediction
    - Gaussian probabilistic prediction
    - Deep ensemble uncertainty
    - Epistemic and aleatoric uncertainty
    - Supervised model training
    - Bootstrap ensemble training
    - Gradient clipping
    - Evaluation metrics
    - Checkpoint save/load
    - NumPy inference adapter

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.prediction.next_state

Inputs:
    states:  [B, state_dim]
    actions: [B, action_dim]

Outputs:
    next_states: [B, state_dim]

This module predicts latent states.
It does not directly control physical devices.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class NextStateError(Exception):
    """Base next-state prediction exception."""


class NextStateValidationError(NextStateError):
    """Invalid model input or target."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class NextStateConfig:
    state_dim: int = 128
    action_dim: int = 8

    hidden_dim: int = 256
    num_hidden_layers: int = 3
    ensemble_size: int = 5

    stochastic: bool = True

    min_log_variance: float = -10.0
    max_log_variance: float = 2.0

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    reconstruction_weight: float = 1.0
    nll_weight: float = 0.1

    bootstrap_ensemble: bool = True

    uncertainty_penalty: float = 0.0

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive_fields = (
            "state_dim",
            "action_dim",
            "hidden_dim",
            "num_hidden_layers",
            "ensemble_size",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay cannot be negative"
            )

        if (
            self.min_log_variance
            >= self.max_log_variance
        ):
            raise ValueError(
                "Invalid log variance bounds"
            )

        if (
            self.reconstruction_weight < 0
            or self.nll_weight < 0
            or self.uncertainty_penalty < 0
        ):
            raise ValueError(
                "Loss weights and penalties "
                "must be nonnegative"
            )


# ============================================================
# VALIDATION
# ============================================================

def prepare_states(
    states: Tensor | np.ndarray,
    *,
    state_dim: int,
    device: torch.device,
) -> Tensor:
    states = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    if states.ndim == 1:
        states = states.unsqueeze(0)

    if (
        states.ndim != 2
        or states.shape[1] != state_dim
    ):
        raise NextStateValidationError(
            f"Expected states [B,{state_dim}]"
        )

    if states.shape[0] == 0:
        raise NextStateValidationError(
            "Empty state batch"
        )

    if not torch.isfinite(states).all():
        raise NextStateValidationError(
            "States contain non-finite values"
        )

    return states


def prepare_actions(
    actions: Tensor | np.ndarray,
    *,
    action_dim: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    actions = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=device,
    )

    if actions.ndim == 1:
        actions = actions.unsqueeze(0)

    if actions.shape != (
        batch_size,
        action_dim,
    ):
        raise NextStateValidationError(
            f"Expected actions "
            f"[{batch_size},{action_dim}]"
        )

    if not torch.isfinite(actions).all():
        raise NextStateValidationError(
            "Actions contain non-finite values"
        )

    return actions


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualDynamicsBlock(nn.Module):
    """
    Residual MLP block for latent dynamics.
    """

    def __init__(
        self,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.SiLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.activation = nn.SiLU()

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        return self.activation(
            features
            + self.network(features)
        )


# ============================================================
# NEXT-STATE DYNAMICS NETWORK
# ============================================================

class NextStateDynamicsNetwork(nn.Module):
    """
    Residual state transition predictor.

    Learns:
        delta_state = f(state, action)

        next_state = state + delta_state

    Predicts diagonal Gaussian uncertainty
    when stochastic mode is enabled.
    """

    def __init__(
        self,
        config: NextStateConfig,
    ) -> None:
        super().__init__()

        self.config = config

        input_dim = (
            config.state_dim
            + config.action_dim
        )

        self.input_layer = nn.Sequential(
            nn.Linear(
                input_dim,
                config.hidden_dim,
            ),
            nn.LayerNorm(
                config.hidden_dim
            ),
            nn.SiLU(),
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualDynamicsBlock(
                    config.hidden_dim
                )
                for _ in range(
                    config.num_hidden_layers
                )
            ]
        )

        self.delta_head = nn.Linear(
            config.hidden_dim,
            config.state_dim,
        )

        self.log_variance_head = nn.Linear(
            config.hidden_dim,
            config.state_dim,
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        features = torch.cat(
            [states, actions],
            dim=-1,
        )

        hidden = self.input_layer(
            features
        )

        hidden = self.residual_blocks(
            hidden
        )

        delta = self.delta_head(
            hidden
        )

        next_state_mean = (
            states + delta
        )

        log_variance = (
            self.log_variance_head(
                hidden
            ).clamp(
                self.config.min_log_variance,
                self.config.max_log_variance,
            )
        )

        return (
            next_state_mean,
            log_variance,
        )


# ============================================================
# PREDICTION OUTPUT
# ============================================================

@dataclass
class NextStatePrediction:
    next_states: Tensor
    mean_states: Tensor

    epistemic_uncertainty: Tensor
    aleatoric_uncertainty: Tensor
    total_uncertainty: Tensor

    ensemble_means: Tensor

    def detach(self) -> "NextStatePrediction":
        return NextStatePrediction(
            next_states=self.next_states.detach(),
            mean_states=self.mean_states.detach(),
            epistemic_uncertainty=(
                self.epistemic_uncertainty.detach()
            ),
            aleatoric_uncertainty=(
                self.aleatoric_uncertainty.detach()
            ),
            total_uncertainty=(
                self.total_uncertainty.detach()
            ),
            ensemble_means=(
                self.ensemble_means.detach()
            ),
        )

    def to_numpy(self) -> dict[str, np.ndarray]:
        def convert(
            tensor: Tensor,
        ) -> np.ndarray:
            return (
                tensor.detach()
                .cpu()
                .numpy()
            )

        return {
            "next_states": convert(
                self.next_states
            ),
            "mean_states": convert(
                self.mean_states
            ),
            "epistemic_uncertainty": convert(
                self.epistemic_uncertainty
            ),
            "aleatoric_uncertainty": convert(
                self.aleatoric_uncertainty
            ),
            "total_uncertainty": convert(
                self.total_uncertainty
            ),
            "ensemble_means": convert(
                self.ensemble_means
            ),
        }


# ============================================================
# NEXT-STATE ENSEMBLE MODEL
# ============================================================

class NextStatePredictionModel(nn.Module):
    """
    Ensemble-based next-state prediction.

    Epistemic variance:
        Variance across ensemble predictions.

    Aleatoric variance:
        Mean predicted transition variance.

    Total variance:
        Epistemic + aleatoric.
    """

    def __init__(
        self,
        config: NextStateConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or NextStateConfig()
        )

        self.networks = nn.ModuleList(
            [
                NextStateDynamicsNetwork(
                    self.config
                )
                for _ in range(
                    self.config.ensemble_size
                )
            ]
        )

        self.to(
            torch.device(
                self.config.device
            )
        )

    @property
    def device(self) -> torch.device:
        return next(
            self.parameters()
        ).device

    def forward(
        self,
        states: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
        uncertainty_penalty: bool = False,
    ) -> NextStatePrediction:
        states = prepare_states(
            states,
            state_dim=self.config.state_dim,
            device=self.device,
        )

        actions = prepare_actions(
            actions,
            action_dim=self.config.action_dim,
            batch_size=states.shape[0],
            device=self.device,
        )

        means = []
        variances = []

        for network in self.networks:
            mean, log_variance = network(
                states,
                actions,
            )

            means.append(
                mean
            )

            if self.config.stochastic:
                variances.append(
                    log_variance.exp()
                )

            else:
                variances.append(
                    torch.zeros_like(mean)
                )

        ensemble_means = torch.stack(
            means,
            dim=0,
        )

        ensemble_variances = torch.stack(
            variances,
            dim=0,
        )

        mean_states = ensemble_means.mean(
            dim=0
        )

        epistemic_variance = (
            ensemble_means.var(
                dim=0,
                unbiased=False,
            )
        )

        aleatoric_variance = (
            ensemble_variances.mean(
                dim=0
            )
        )

        total_variance = (
            epistemic_variance
            + aleatoric_variance
        )

        epistemic_std = (
            epistemic_variance
            .clamp_min(0)
            .sqrt()
        )

        aleatoric_std = (
            aleatoric_variance
            .clamp_min(0)
            .sqrt()
        )

        total_std = (
            total_variance
            .clamp_min(0)
            .sqrt()
        )

        if (
            deterministic
            or not self.config.stochastic
        ):
            next_states = mean_states

        else:
            # Sample one ensemble member per
            # batch item to preserve diversity.
            batch_size = (
                states.shape[0]
            )

            member_indices = torch.randint(
                0,
                self.config.ensemble_size,
                (batch_size,),
                device=self.device,
            )

            batch_indices = torch.arange(
                batch_size,
                device=self.device,
            )

            selected_means = ensemble_means[
                member_indices,
                batch_indices,
            ]

            selected_variances = (
                ensemble_variances[
                    member_indices,
                    batch_indices,
                ]
            )

            next_states = (
                selected_means
                + torch.randn_like(
                    selected_means
                )
                * selected_variances.sqrt()
            )

        if uncertainty_penalty:
            # A conservative latent-state
            # adjustment. Normally uncertainty
            # penalties belong in reward/planning,
            # not physical state transitions.
            next_states = (
                next_states
                - self.config.uncertainty_penalty
                * total_std
            )

        return NextStatePrediction(
            next_states=next_states,
            mean_states=mean_states,
            epistemic_uncertainty=epistemic_std,
            aleatoric_uncertainty=aleatoric_std,
            total_uncertainty=total_std,
            ensemble_means=ensemble_means,
        )

    @torch.no_grad()
    def predict(
        self,
        states: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
    ) -> NextStatePrediction:
        previous_mode = self.training

        try:
            self.eval()

            return self.forward(
                states,
                actions,
                deterministic=deterministic,
            ).detach()

        finally:
            self.train(
                previous_mode
            )

    @torch.no_grad()
    def predict_single(
        self,
        state: Tensor | np.ndarray,
        action: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
    ) -> dict[str, np.ndarray | float]:
        prediction = self.predict(
            state,
            action,
            deterministic=deterministic,
        )

        return {
            "next_state": (
                prediction.next_states[0]
                .cpu()
                .numpy()
            ),
            "mean_state": (
                prediction.mean_states[0]
                .cpu()
                .numpy()
            ),
            "epistemic_uncertainty": float(
                prediction.epistemic_uncertainty[0]
                .mean()
                .item()
            ),
            "aleatoric_uncertainty": float(
                prediction.aleatoric_uncertainty[0]
                .mean()
                .item()
            ),
            "total_uncertainty": float(
                prediction.total_uncertainty[0]
                .mean()
                .item()
            ),
        }


# ============================================================
# TRAINING METRICS
# ============================================================

@dataclass
class NextStateTrainingMetrics:
    total_loss: float
    reconstruction_loss: float
    nll_loss: float

    gradient_norm: float

    mean_epistemic_uncertainty: float
    mean_aleatoric_uncertainty: float

    training_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# NEXT-STATE TRAINER
# ============================================================

class NextStateTrainer:
    """
    Supervised training of next-state
    dynamics ensembles.

    Supports:
        - MSE reconstruction loss
        - Gaussian NLL
        - Bootstrap resampling
        - Independent optimizers
    """

    def __init__(
        self,
        model: NextStatePredictionModel,
    ) -> None:
        self.model = model
        self.config = model.config

        self.optimizers = [
            torch.optim.AdamW(
                network.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            for network in model.networks
        ]

        self.training_steps = 0

    def train_step(
        self,
        states: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        next_states: Tensor | np.ndarray,
    ) -> NextStateTrainingMetrics:
        self.model.train()

        states = prepare_states(
            states,
            state_dim=self.config.state_dim,
            device=self.model.device,
        )

        actions = prepare_actions(
            actions,
            action_dim=self.config.action_dim,
            batch_size=states.shape[0],
            device=self.model.device,
        )

        targets = prepare_states(
            next_states,
            state_dim=self.config.state_dim,
            device=self.model.device,
        )

        if targets.shape != states.shape:
            raise NextStateValidationError(
                "Target shape mismatch"
            )

        batch_size = (
            states.shape[0]
        )

        total_losses = []
        reconstruction_losses = []
        nll_losses = []
        gradient_norms = []

        for network, optimizer in zip(
            self.model.networks,
            self.optimizers,
        ):
            if self.config.bootstrap_ensemble:
                indices = torch.randint(
                    0,
                    batch_size,
                    (batch_size,),
                    device=self.model.device,
                )

                train_states = states[
                    indices
                ]

                train_actions = actions[
                    indices
                ]

                train_targets = targets[
                    indices
                ]

            else:
                train_states = states
                train_actions = actions
                train_targets = targets

            (
                predicted_mean,
                log_variance,
            ) = network(
                train_states,
                train_actions,
            )

            reconstruction_loss = F.mse_loss(
                predicted_mean,
                train_targets,
            )

            if self.config.stochastic:
                nll_loss = (
                    0.5
                    * (
                        math.log(2.0 * math.pi)
                        + log_variance
                        + (
                            (
                                train_targets
                                - predicted_mean
                            ).square()
                            / log_variance.exp()
                        )
                    )
                ).mean()

            else:
                nll_loss = (
                    reconstruction_loss * 0.0
                )

            total_loss = (
                self.config.reconstruction_weight
                * reconstruction_loss
                + self.config.nll_weight
                * nll_loss
            )

            if not torch.isfinite(
                total_loss
            ):
                raise NextStateError(
                    "Non-finite training loss"
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            total_loss.backward()

            gradient_norm = (
                nn.utils.clip_grad_norm_(
                    network.parameters(),
                    self.config.max_grad_norm,
                )
            )

            optimizer.step()

            total_losses.append(
                total_loss.detach()
            )

            reconstruction_losses.append(
                reconstruction_loss.detach()
            )

            nll_losses.append(
                nll_loss.detach()
            )

            gradient_norms.append(
                gradient_norm.detach()
            )

        self.training_steps += 1

        with torch.no_grad():
            prediction = self.model(
                states,
                actions,
                deterministic=True,
            )

        return NextStateTrainingMetrics(
            total_loss=float(
                torch.stack(
                    total_losses
                ).mean().item()
            ),
            reconstruction_loss=float(
                torch.stack(
                    reconstruction_losses
                ).mean().item()
            ),
            nll_loss=float(
                torch.stack(
                    nll_losses
                ).mean().item()
            ),
            gradient_norm=float(
                torch.stack(
                    gradient_norms
                ).mean().item()
            ),
            mean_epistemic_uncertainty=float(
                prediction.epistemic_uncertainty
                .mean().item()
            ),
            mean_aleatoric_uncertainty=float(
                prediction.aleatoric_uncertainty
                .mean().item()
            ),
            training_step=self.training_steps,
        )

    @torch.no_grad()
    def evaluate(
        self,
        states: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        next_states: Tensor | np.ndarray,
    ) -> dict[str, float]:
        self.model.eval()

        prediction = self.model.predict(
            states,
            actions,
        )

        targets = prepare_states(
            next_states,
            state_dim=self.config.state_dim,
            device=self.model.device,
        )

        if targets.shape != prediction.mean_states.shape:
            raise NextStateValidationError(
                "Evaluation target shape mismatch"
            )

        errors = (
            prediction.mean_states
            - targets
        )

        mse = (
            errors.square().mean()
        )

        mae = (
            errors.abs().mean()
        )

        return {
            "mse": float(
                mse.item()
            ),
            "rmse": float(
                mse.sqrt().item()
            ),
            "mae": float(
                mae.item()
            ),
            "mean_epistemic_uncertainty": float(
                prediction.epistemic_uncertainty
                .mean().item()
            ),
            "mean_aleatoric_uncertainty": float(
                prediction.aleatoric_uncertainty
                .mean().item()
            ),
            "mean_total_uncertainty": float(
                prediction.total_uncertainty
                .mean().item()
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [
                optimizer.state_dict()
                for optimizer in self.optimizers
            ],
            "training_steps": self.training_steps,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if (
            len(state["optimizers"])
            != len(self.optimizers)
        ):
            raise NextStateValidationError(
                "Optimizer ensemble size mismatch"
            )

        for optimizer, saved_state in zip(
            self.optimizers,
            state["optimizers"],
        ):
            optimizer.load_state_dict(
                saved_state
            )

        self.training_steps = int(
            state["training_steps"]
        )


# ============================================================
# NUMPY ADAPTER
# ============================================================

class NextStateAdapter:
    """
    NumPy-compatible interface for
    imagination and planning systems.
    """

    def __init__(
        self,
        model: NextStatePredictionModel,
    ) -> None:
        self.model = model

    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        prediction = self.model.predict_single(
            state,
            action,
            deterministic=deterministic,
        )

        return prediction[
            "next_state"
        ]

    def predict_with_uncertainty(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        return self.model.predict_single(
            state,
            action,
        )


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_next_state_model(
    model: NextStatePredictionModel,
    path: str | Path,
    *,
    trainer: NextStateTrainer | None = None,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: dict[str, Any] = {
        "config": asdict(
            model.config
        ),
        "model": model.state_dict(),
    }

    if trainer is not None:
        checkpoint["trainer"] = (
            trainer.state_dict()
        )

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_next_state_model(
    path: str | Path,
    *,
    device: str = "cpu",
    restore_trainer: bool = False,
) -> (
    NextStatePredictionModel
    | tuple[
        NextStatePredictionModel,
        NextStateTrainer,
    ]
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    config_data = dict(
        checkpoint["config"]
    )

    config_data["device"] = device

    model = NextStatePredictionModel(
        NextStateConfig(
            **config_data
        )
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    if not restore_trainer:
        return model

    trainer = NextStateTrainer(
        model
    )

    if "trainer" in checkpoint:
        trainer.load_state_dict(
            checkpoint["trainer"]
        )

    return model, trainer


# ============================================================
# FACTORY
# ============================================================

def create_next_state_predictor(
    config: NextStateConfig | None = None,
) -> NextStatePredictionModel:
    return NextStatePredictionModel(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - NEXT STATE PREDICTION"
    )
    print("=" * 65)

    config = NextStateConfig(
        state_dim=16,
        action_dim=4,
        hidden_dim=64,
        num_hidden_layers=2,
        ensemble_size=3,
        stochastic=True,
        learning_rate=1e-3,
    )

    model = NextStatePredictionModel(
        config
    )

    trainer = NextStateTrainer(
        model
    )

    batch_size = 128

    states = torch.randn(
        batch_size,
        config.state_dim,
    )

    actions = torch.randn(
        batch_size,
        config.action_dim,
    ) * 0.2

    # Synthetic ground-truth dynamics.
    next_states = states.clone()

    next_states[
        :, :config.action_dim
    ] += actions

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for step in range(30):
        metrics = trainer.train_step(
            states,
            actions,
            next_states,
        )

        if step in (0, 9, 19, 29):
            print(
                f"Training step {step + 1}:",
                metrics.to_dict(),
            )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        states[:4],
        actions[:4],
    )

    print(
        "Next states:",
        tuple(
            prediction.next_states.shape
        ),
    )

    print(
        "Epistemic uncertainty:",
        tuple(
            prediction.epistemic_uncertainty.shape
        ),
    )

    # --------------------------------------------------------
    # SINGLE TRANSITION
    # --------------------------------------------------------

    single_prediction = (
        model.predict_single(
            states[0],
            actions[0],
        )
    )

    print(
        "Single predicted state:",
        single_prediction["next_state"].shape,
    )

    print(
        "Total uncertainty:",
        single_prediction["total_uncertainty"],
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluation = trainer.evaluate(
        states,
        actions,
        next_states,
    )

    print(
        "Evaluation:",
        evaluation,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = save_next_state_model(
        model,
        "models/next_state_demo.pt",
        trainer=trainer,
    )

    restored_model = load_next_state_model(
        checkpoint
    )

    print(
        "Restored model:",
        type(restored_model).__name__,
    )

    print(
        "Next-state prediction demonstration completed"
    )


if __name__ == "__main__":
    main()
