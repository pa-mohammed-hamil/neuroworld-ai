
"""
NeuroWorld AI
File: neuroworld/prediction/future_state.py

Future World State Prediction Engine.

Features:
    - Action-conditioned latent state dynamics
    - Deterministic and stochastic prediction
    - Multi-step autoregressive rollouts
    - GRU-based temporal dynamics
    - Gaussian next-state distribution
    - Monte Carlo future uncertainty
    - Teacher forcing during training
    - State prediction loss
    - Gaussian negative log likelihood
    - Multi-horizon evaluation
    - Checkpoint save/load
    - Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.prediction.future_state

Inputs:
    initial_state: [B, state_dim]
    actions:       [B, horizon, action_dim]

Outputs:
    future_states: [B, horizon, state_dim]

Note:
    The model predicts latent state vectors.
    It does not directly generate video frames.
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

class FutureStateError(Exception):
    """Base future-state prediction exception."""


class FutureStateValidationError(FutureStateError):
    """Invalid state or action data."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FutureStateConfig:
    state_dim: int = 128
    action_dim: int = 8

    hidden_dim: int = 256
    num_gru_layers: int = 2

    max_prediction_horizon: int = 50

    stochastic: bool = True

    min_log_variance: float = -10.0
    max_log_variance: float = 2.0

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    reconstruction_weight: float = 1.0
    nll_weight: float = 0.1

    teacher_forcing_ratio: float = 0.5

    mc_samples: int = 20

    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "hidden_dim",
            "num_gru_layers",
            "max_prediction_horizon",
            "mc_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if (
            self.min_log_variance
            >= self.max_log_variance
        ):
            raise ValueError(
                "Invalid log variance bounds"
            )

        if not (
            0.0 <= self.teacher_forcing_ratio <= 1.0
        ):
            raise ValueError(
                "teacher_forcing_ratio must be in [0,1]"
            )

        for name in (
            "learning_rate",
            "max_grad_norm",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        for name in (
            "weight_decay",
            "reconstruction_weight",
            "nll_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


# ============================================================
# INPUT VALIDATION
# ============================================================

def prepare_initial_state(
    state: Tensor | np.ndarray,
    config: FutureStateConfig,
    device: torch.device,
) -> Tensor:
    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    )

    if state.ndim == 1:
        state = state.unsqueeze(0)

    if (
        state.ndim != 2
        or state.shape[-1] != config.state_dim
    ):
        raise FutureStateValidationError(
            f"Expected initial state "
            f"[B,{config.state_dim}]"
        )

    if not torch.isfinite(state).all():
        raise FutureStateValidationError(
            "Initial state contains non-finite values"
        )

    return state


def prepare_action_sequence(
    actions: Tensor | np.ndarray,
    config: FutureStateConfig,
    device: torch.device,
    *,
    batch_size: int,
) -> Tensor:
    actions = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=device,
    )

    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    if actions.ndim != 3:
        raise FutureStateValidationError(
            "Actions must have shape [B,H,A]"
        )

    if actions.shape[0] != batch_size:
        raise FutureStateValidationError(
            "Action batch size mismatch"
        )

    if actions.shape[-1] != config.action_dim:
        raise FutureStateValidationError(
            "Action dimension mismatch"
        )

    if not (
        1
        <= actions.shape[1]
        <= config.max_prediction_horizon
    ):
        raise FutureStateValidationError(
            "Invalid prediction horizon"
        )

    if not torch.isfinite(actions).all():
        raise FutureStateValidationError(
            "Actions contain non-finite values"
        )

    return actions


# ============================================================
# STATE-ACTION ENCODER
# ============================================================

class StateActionEncoder(nn.Module):
    """
    Encodes current world state and action
    into a learned transition representation.
    """

    def __init__(
        self,
        config: FutureStateConfig,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.state_dim
                + config.action_dim,
                config.hidden_dim,
            ),
            nn.LayerNorm(
                config.hidden_dim
            ),
            nn.SiLU(),

            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [state, action],
            dim=-1,
        )

        return self.network(
            features
        )


# ============================================================
# RECURRENT DYNAMICS MODEL
# ============================================================

class RecurrentFutureDynamics(nn.Module):
    """
    GRU-based state transition model.

    The GRU hidden state carries temporal
    context across imagined future steps.
    """

    def __init__(
        self,
        config: FutureStateConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.encoder = StateActionEncoder(
            config
        )

        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_gru_layers,
            batch_first=True,
        )

        self.initial_hidden = nn.Sequential(
            nn.Linear(
                config.state_dim,
                config.hidden_dim
                * config.num_gru_layers,
            ),
            nn.Tanh(),
        )

        self.delta_head = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dim,
                config.state_dim,
            ),
        )

        self.log_variance_head = nn.Linear(
            config.hidden_dim,
            config.state_dim,
        )

    def initialize_hidden(
        self,
        initial_state: Tensor,
    ) -> Tensor:
        batch_size = (
            initial_state.shape[0]
        )

        hidden = self.initial_hidden(
            initial_state
        )

        hidden = hidden.reshape(
            batch_size,
            self.config.num_gru_layers,
            self.config.hidden_dim,
        )

        return hidden.permute(
            1, 0, 2
        ).contiguous()

    def forward_step(
        self,
        state: Tensor,
        action: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(
            state,
            action,
        )

        gru_output, next_hidden = self.gru(
            encoded.unsqueeze(1),
            hidden,
        )

        features = gru_output[
            :, -1
        ]

        delta = self.delta_head(
            features
        )

        mean_next_state = (
            state + delta
        )

        log_variance = (
            self.log_variance_head(
                features
            ).clamp(
                self.config.min_log_variance,
                self.config.max_log_variance,
            )
        )

        return (
            mean_next_state,
            log_variance,
            next_hidden,
        )


# ============================================================
# PREDICTION OUTPUT
# ============================================================

@dataclass
class FutureStateOutput:
    future_states: Tensor
    mean_states: Tensor
    log_variances: Tensor

    uncertainty: Tensor | None = None
    final_hidden: Tensor | None = None

    def detach(self) -> "FutureStateOutput":
        return FutureStateOutput(
            future_states=(
                self.future_states.detach()
            ),
            mean_states=(
                self.mean_states.detach()
            ),
            log_variances=(
                self.log_variances.detach()
            ),
            uncertainty=(
                self.uncertainty.detach()
                if self.uncertainty is not None
                else None
            ),
            final_hidden=(
                self.final_hidden.detach()
                if self.final_hidden is not None
                else None
            ),
        )


# ============================================================
# FUTURE STATE PREDICTION MODEL
# ============================================================

class FutureStatePredictionModel(nn.Module):
    """
    Predicts multiple future world states.

    Autoregressive dynamics:
        s(t+1) = f(s(t), a(t), hidden)

    Supports:
        deterministic mean prediction
        stochastic Gaussian sampling
        teacher-forced training
    """

    def __init__(
        self,
        config: FutureStateConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or FutureStateConfig()
        )

        self.dynamics = RecurrentFutureDynamics(
            self.config
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

    # --------------------------------------------------------
    # SINGLE TRANSITION
    # --------------------------------------------------------

    def predict_step(
        self,
        state: Tensor,
        action: Tensor,
        hidden: Tensor | None = None,
        *,
        deterministic: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if hidden is None:
            hidden = (
                self.dynamics.initialize_hidden(
                    state
                )
            )

        (
            mean,
            log_variance,
            next_hidden,
        ) = self.dynamics.forward_step(
            state,
            action,
            hidden,
        )

        if (
            self.config.stochastic
            and not deterministic
        ):
            std = torch.exp(
                0.5 * log_variance
            )

            next_state = (
                mean
                + torch.randn_like(mean)
                * std
            )

        else:
            next_state = mean

        return (
            next_state,
            mean,
            log_variance,
            next_hidden,
        )

    # --------------------------------------------------------
    # MULTI-STEP ROLLOUT
    # --------------------------------------------------------

    def forward(
        self,
        initial_state: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
        target_states: (
            Tensor | np.ndarray | None
        ) = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> FutureStateOutput:
        if not (
            0.0
            <= teacher_forcing_ratio
            <= 1.0
        ):
            raise FutureStateValidationError(
                "Invalid teacher_forcing_ratio"
            )

        state = prepare_initial_state(
            initial_state,
            self.config,
            self.device,
        )

        actions = prepare_action_sequence(
            actions,
            self.config,
            self.device,
            batch_size=state.shape[0],
        )

        batch_size, horizon, _ = (
            actions.shape
        )

        targets = None

        if target_states is not None:
            targets = torch.as_tensor(
                target_states,
                dtype=torch.float32,
                device=self.device,
            )

            if targets.ndim == 2:
                targets = targets.unsqueeze(0)

            expected_shape = (
                batch_size,
                horizon,
                self.config.state_dim,
            )

            if targets.shape != expected_shape:
                raise FutureStateValidationError(
                    f"Expected targets {expected_shape}"
                )

            if not torch.isfinite(targets).all():
                raise FutureStateValidationError(
                    "Targets contain non-finite values"
                )

        hidden = (
            self.dynamics.initialize_hidden(
                state
            )
        )

        future_states = []
        mean_states = []
        log_variances = []

        for step in range(horizon):
            action = actions[
                :, step
            ]

            (
                next_state,
                mean,
                log_variance,
                hidden,
            ) = self.predict_step(
                state,
                action,
                hidden,
                deterministic=deterministic,
            )

            future_states.append(
                next_state
            )

            mean_states.append(
                mean
            )

            log_variances.append(
                log_variance
            )

            state = next_state

            if (
                self.training
                and targets is not None
                and teacher_forcing_ratio > 0
            ):
                teacher_mask = (
                    torch.rand(
                        batch_size,
                        1,
                        device=self.device,
                    )
                    < teacher_forcing_ratio
                )

                state = torch.where(
                    teacher_mask,
                    targets[:, step],
                    state,
                )

        future_states = torch.stack(
            future_states,
            dim=1,
        )

        mean_states = torch.stack(
            mean_states,
            dim=1,
        )

        log_variances = torch.stack(
            log_variances,
            dim=1,
        )

        return FutureStateOutput(
            future_states=future_states,
            mean_states=mean_states,
            log_variances=log_variances,
            uncertainty=torch.exp(
                0.5 * log_variances
            ),
            final_hidden=hidden,
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        initial_state: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
        estimate_uncertainty: bool = False,
        mc_samples: int | None = None,
    ) -> FutureStateOutput:
        if mc_samples is None:
            mc_samples = (
                self.config.mc_samples
            )

        if mc_samples <= 0:
            raise FutureStateValidationError(
                "mc_samples must be positive"
            )

        previous_mode = self.training

        try:
            self.eval()

            if not estimate_uncertainty:
                return self.forward(
                    initial_state,
                    actions,
                    deterministic=deterministic,
                ).detach()

            if not self.config.stochastic:
                output = self.forward(
                    initial_state,
                    actions,
                    deterministic=True,
                )

                # A deterministic model cannot
                # estimate predictive variance
                # through stochastic sampling.
                output.uncertainty = (
                    torch.zeros_like(
                        output.future_states
                    )
                )

                return output.detach()

            samples = [
                self.forward(
                    initial_state,
                    actions,
                    deterministic=False,
                )
                for _ in range(mc_samples)
            ]

            trajectories = torch.stack(
                [
                    sample.future_states
                    for sample in samples
                ],
                dim=0,
            )

            means = torch.stack(
                [
                    sample.mean_states
                    for sample in samples
                ],
                dim=0,
            )

            log_variances = torch.stack(
                [
                    sample.log_variances
                    for sample in samples
                ],
                dim=0,
            )

            return FutureStateOutput(
                future_states=(
                    trajectories.mean(
                        dim=0
                    )
                ),
                mean_states=means.mean(
                    dim=0
                ),
                log_variances=(
                    log_variances.mean(
                        dim=0
                    )
                ),
                uncertainty=(
                    trajectories.std(
                        dim=0,
                        unbiased=False,
                    )
                ),
            ).detach()

        finally:
            self.train(
                previous_mode
            )


# ============================================================
# FUTURE STATE LOSS
# ============================================================

def gaussian_nll_loss(
    mean: Tensor,
    log_variance: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Diagonal Gaussian negative
    log likelihood.

    Constant log(2*pi) is included.
    """

    return (
        0.5
        * (
            math.log(2.0 * math.pi)
            + log_variance
            + (
                (target - mean).square()
                / log_variance.exp()
            )
        )
    ).mean()


def calculate_future_state_loss(
    output: FutureStateOutput,
    targets: Tensor,
    config: FutureStateConfig,
) -> dict[str, Tensor]:
    if (
        output.future_states.shape
        != targets.shape
    ):
        raise FutureStateValidationError(
            "Prediction and target shapes differ"
        )

    reconstruction_loss = F.mse_loss(
        output.mean_states,
        targets,
    )

    if config.stochastic:
        nll_loss = gaussian_nll_loss(
            output.mean_states,
            output.log_variances,
            targets,
        )

    else:
        nll_loss = (
            reconstruction_loss * 0.0
        )

    total_loss = (
        config.reconstruction_weight
        * reconstruction_loss
        + config.nll_weight
        * nll_loss
    )

    return {
        "total_loss": total_loss,
        "reconstruction_loss": (
            reconstruction_loss
        ),
        "nll_loss": nll_loss,
    }


# ============================================================
# TRAINING BATCH
# ============================================================

@dataclass
class FutureStateTrainingBatch:
    initial_states: Tensor
    actions: Tensor
    target_states: Tensor


# ============================================================
# TRAINER
# ============================================================

class FutureStateTrainer:
    """
    Trains multi-step action-conditioned
    future state prediction.
    """

    def __init__(
        self,
        model: FutureStatePredictionModel,
    ) -> None:
        self.model = model
        self.config = model.config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.training_steps = 0

    def _prepare_batch(
        self,
        batch: FutureStateTrainingBatch,
    ) -> FutureStateTrainingBatch:
        initial_states = prepare_initial_state(
            batch.initial_states,
            self.config,
            self.model.device,
        )

        actions = prepare_action_sequence(
            batch.actions,
            self.config,
            self.model.device,
            batch_size=initial_states.shape[0],
        )

        targets = torch.as_tensor(
            batch.target_states,
            dtype=torch.float32,
            device=self.model.device,
        )

        if targets.ndim == 2:
            targets = targets.unsqueeze(0)

        expected_shape = (
            initial_states.shape[0],
            actions.shape[1],
            self.config.state_dim,
        )

        if targets.shape != expected_shape:
            raise FutureStateValidationError(
                f"Expected targets {expected_shape}"
            )

        if not torch.isfinite(targets).all():
            raise FutureStateValidationError(
                "Targets contain non-finite values"
            )

        return FutureStateTrainingBatch(
            initial_states=initial_states,
            actions=actions,
            target_states=targets,
        )

    def train_step(
        self,
        batch: FutureStateTrainingBatch,
    ) -> dict[str, float]:
        self.model.train()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.initial_states,
            batch.actions,
            deterministic=True,
            target_states=batch.target_states,
            teacher_forcing_ratio=(
                self.config.teacher_forcing_ratio
            ),
        )

        losses = calculate_future_state_loss(
            output,
            batch.target_states,
            self.config,
        )

        if not torch.isfinite(
            losses["total_loss"]
        ):
            raise FutureStateError(
                "Non-finite training loss"
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        losses["total_loss"].backward()

        gradient_norm = (
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        self.training_steps += 1

        return {
            **{
                name: float(
                    value.detach().item()
                )
                for name, value in losses.items()
            },
            "gradient_norm": float(
                gradient_norm.detach().item()
            ),
            "training_step": (
                self.training_steps
            ),
        }

    @torch.no_grad()
    def evaluate(
        self,
        batch: FutureStateTrainingBatch,
    ) -> dict[str, float]:
        self.model.eval()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.initial_states,
            batch.actions,
            deterministic=True,
        )

        losses = calculate_future_state_loss(
            output,
            batch.target_states,
            self.config,
        )

        errors = (
            output.future_states
            - batch.target_states
        )

        mse_by_horizon = (
            errors.square()
            .mean(dim=(0, 2))
        )

        mae = (
            errors.abs().mean()
        )

        metrics = {
            name: float(
                value.item()
            )
            for name, value in losses.items()
        }

        metrics["mae"] = float(
            mae.item()
        )

        metrics["rmse"] = float(
            errors.square()
            .mean()
            .sqrt()
            .item()
        )

        for index, value in enumerate(
            mse_by_horizon
        ):
            metrics[
                f"horizon_{index + 1}_mse"
            ] = float(
                value.item()
            )

        return metrics


# ============================================================
# NUMPY ADAPTER
# ============================================================

class FutureStateAdapter:
    """
    Adapter for imagination and planning
    modules that operate on NumPy arrays.
    """

    def __init__(
        self,
        model: FutureStatePredictionModel,
    ) -> None:
        self.model = model

    def predict_trajectory(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        *,
        stochastic: bool = False,
    ) -> np.ndarray:
        result = self.model.predict(
            initial_state,
            actions,
            deterministic=not stochastic,
        )

        return (
            result.future_states[0]
            .cpu()
            .numpy()
        )

    def predict_next_state(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        actions = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(
            1, 1, -1
        )

        return self.predict_trajectory(
            np.asarray(
                state,
                dtype=np.float32,
            ),
            actions,
        )[0]


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_future_state_model(
    model: FutureStatePredictionModel,
    path: str | Path,
    *,
    trainer: FutureStateTrainer | None = None,
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
        checkpoint["optimizer"] = (
            trainer.optimizer.state_dict()
        )

        checkpoint["training_steps"] = (
            trainer.training_steps
        )

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_future_state_model(
    path: str | Path,
    *,
    device: str = "cpu",
    restore_trainer: bool = False,
) -> (
    FutureStatePredictionModel
    | tuple[
        FutureStatePredictionModel,
        FutureStateTrainer,
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

    model = FutureStatePredictionModel(
        FutureStateConfig(
            **config_data
        )
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    if not restore_trainer:
        return model

    trainer = FutureStateTrainer(
        model
    )

    if "optimizer" in checkpoint:
        trainer.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    trainer.training_steps = int(
        checkpoint.get(
            "training_steps",
            0,
        )
    )

    return model, trainer


# ============================================================
# FACTORY
# ============================================================

def create_future_state_predictor(
    config: FutureStateConfig | None = None,
) -> FutureStatePredictionModel:
    return FutureStatePredictionModel(
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
        "NEUROWORLD AI - FUTURE STATE PREDICTION"
    )
    print("=" * 65)

    config = FutureStateConfig(
        state_dim=16,
        action_dim=4,
        hidden_dim=64,
        num_gru_layers=2,
        max_prediction_horizon=10,
        stochastic=True,
        mc_samples=5,
    )

    model = FutureStatePredictionModel(
        config
    )

    trainer = FutureStateTrainer(
        model
    )

    batch_size = 16
    horizon = 5

    initial_states = torch.randn(
        batch_size,
        config.state_dim,
    )

    actions = torch.randn(
        batch_size,
        horizon,
        config.action_dim,
    ) * 0.2

    # Synthetic transition dynamics:
    # first action_dim state features change
    # in proportion to the action.

    target_states = []

    current_state = initial_states.clone()

    for step in range(horizon):
        next_state = current_state.clone()

        next_state[
            :, :config.action_dim
        ] += actions[:, step]

        target_states.append(
            next_state
        )

        current_state = next_state

    target_states = torch.stack(
        target_states,
        dim=1,
    )

    batch = FutureStateTrainingBatch(
        initial_states=initial_states,
        actions=actions,
        target_states=target_states,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for step in range(20):
        metrics = trainer.train_step(
            batch
        )

        if step in (0, 9, 19):
            print(
                f"Training step {step + 1}:",
                metrics,
            )

    # --------------------------------------------------------
    # DETERMINISTIC PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        initial_states[:2],
        actions[:2],
        deterministic=True,
    )

    print(
        "Future state shape:",
        tuple(
            prediction.future_states.shape
        ),
    )

    # --------------------------------------------------------
    # STOCHASTIC UNCERTAINTY
    # --------------------------------------------------------

    uncertain_prediction = model.predict(
        initial_states[:2],
        actions[:2],
        estimate_uncertainty=True,
        mc_samples=5,
    )

    print(
        "Uncertainty shape:",
        tuple(
            uncertain_prediction.uncertainty.shape
        ),
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluation = trainer.evaluate(
        batch
    )

    print(
        "Evaluation:",
        evaluation,
    )

    # --------------------------------------------------------
    # NUMPY ADAPTER
    # --------------------------------------------------------

    adapter = FutureStateAdapter(
        model
    )

    next_state = adapter.predict_next_state(
        initial_states[0].numpy(),
        actions[0, 0].numpy(),
    )

    print(
        "Adapter next-state shape:",
        next_state.shape,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = save_future_state_model(
        model,
        "models/future_state_demo.pt",
        trainer=trainer,
    )

    restored_model = load_future_state_model(
        checkpoint
    )

    print(
        "Restored model:",
        type(restored_model).__name__,
    )

    print(
        "Future state prediction demonstration completed"
    )


if __name__ == "__main__":
    main()
