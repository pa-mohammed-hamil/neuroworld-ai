
"""
NeuroWorld AI
File: neuroworld/prediction/trajectory_prediction.py

Neural Trajectory Prediction & Motion Forecasting.

Features:
    - 2D and 3D trajectory prediction
    - Historical position and velocity encoding
    - GRU temporal motion modeling
    - Action-conditioned future trajectories
    - Multi-modal future prediction
    - Gaussian trajectory uncertainty
    - Best-of-K trajectory training
    - Teacher-forced supervised learning
    - ADE and FDE evaluation
    - Collision-distance analysis
    - Trajectory sampling
    - Checkpoint save/load
    - NumPy adapter
    - Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.prediction.trajectory_prediction

Inputs:
    positions: [B, T_history, spatial_dim]
    actions:   [B, T_future, action_dim]

Outputs:
    trajectories:
        [B, num_modes, T_future, spatial_dim]

Notes:
    - Positions must use consistent units.
    - Timesteps must represent a consistent interval.
    - Predictions are learned estimates.
    - Collision-distance checks are geometric
      approximations, not safety guarantees.
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

class TrajectoryPredictionError(Exception):
    """Base trajectory prediction exception."""


class TrajectoryValidationError(TrajectoryPredictionError):
    """Invalid trajectory input."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrajectoryPredictionConfig:
    spatial_dim: int = 2
    action_dim: int = 4

    hidden_dim: int = 128
    num_gru_layers: int = 2

    num_modes: int = 5

    max_history_steps: int = 50
    max_future_steps: int = 50

    dropout: float = 0.1

    min_log_variance: float = -8.0
    max_log_variance: float = 3.0

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    regression_weight: float = 1.0
    nll_weight: float = 0.1
    mode_loss_weight: float = 0.2

    teacher_forcing_ratio: float = 0.5

    timestep_seconds: float = 0.1

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        for name in (
            "action_dim",
            "hidden_dim",
            "num_gru_layers",
            "num_modes",
            "max_history_steps",
            "max_future_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if not (
            0 <= self.teacher_forcing_ratio <= 1
        ):
            raise ValueError(
                "teacher_forcing_ratio must be in [0,1]"
            )

        if (
            self.min_log_variance
            >= self.max_log_variance
        ):
            raise ValueError(
                "Invalid log variance bounds"
            )

        if (
            self.learning_rate <= 0
            or self.max_grad_norm <= 0
            or self.timestep_seconds <= 0
        ):
            raise ValueError(
                "Learning rate, gradient norm, "
                "and timestep must be positive"
            )

        for name in (
            "weight_decay",
            "regression_weight",
            "nll_weight",
            "mode_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


# ============================================================
# INPUT PREPARATION
# ============================================================

def prepare_history(
    positions: Tensor | np.ndarray,
    config: TrajectoryPredictionConfig,
    device: torch.device,
) -> Tensor:
    positions = torch.as_tensor(
        positions,
        dtype=torch.float32,
        device=device,
    )

    if positions.ndim == 2:
        positions = positions.unsqueeze(0)

    if positions.ndim != 3:
        raise TrajectoryValidationError(
            "History must have shape [B,T,D]"
        )

    if positions.shape[-1] != config.spatial_dim:
        raise TrajectoryValidationError(
            "Invalid spatial dimension"
        )

    if not (
        2
        <= positions.shape[1]
        <= config.max_history_steps
    ):
        raise TrajectoryValidationError(
            "History requires 2 or more steps "
            "within the configured maximum"
        )

    if not torch.isfinite(positions).all():
        raise TrajectoryValidationError(
            "History contains non-finite values"
        )

    return positions


def prepare_future_actions(
    actions: Tensor | np.ndarray | None,
    *,
    batch_size: int,
    future_steps: int,
    config: TrajectoryPredictionConfig,
    device: torch.device,
) -> Tensor:
    if actions is None:
        return torch.zeros(
            batch_size,
            future_steps,
            config.action_dim,
            device=device,
        )

    actions = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=device,
    )

    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    expected = (
        batch_size,
        future_steps,
        config.action_dim,
    )

    if actions.shape != expected:
        raise TrajectoryValidationError(
            f"Expected actions {expected}"
        )

    if not torch.isfinite(actions).all():
        raise TrajectoryValidationError(
            "Actions contain non-finite values"
        )

    return actions


def prepare_targets(
    targets: Tensor | np.ndarray,
    *,
    batch_size: int,
    future_steps: int,
    config: TrajectoryPredictionConfig,
    device: torch.device,
) -> Tensor:
    targets = torch.as_tensor(
        targets,
        dtype=torch.float32,
        device=device,
    )

    if targets.ndim == 2:
        targets = targets.unsqueeze(0)

    expected = (
        batch_size,
        future_steps,
        config.spatial_dim,
    )

    if targets.shape != expected:
        raise TrajectoryValidationError(
            f"Expected targets {expected}"
        )

    if not torch.isfinite(targets).all():
        raise TrajectoryValidationError(
            "Targets contain non-finite values"
        )

    return targets


# ============================================================
# MOTION FEATURE ENCODER
# ============================================================

class MotionFeatureEncoder(nn.Module):
    """
    Encodes:
        position
        velocity
        acceleration

    All features use consistent spatial
    dimensions and timestep units.
    """

    def __init__(
        self,
        config: TrajectoryPredictionConfig,
    ) -> None:
        super().__init__()

        self.config = config

        input_dim = (
            config.spatial_dim * 3
        )

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
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
        positions: Tensor,
    ) -> Tensor:
        dt = (
            self.config.timestep_seconds
        )

        velocity = torch.zeros_like(
            positions
        )

        velocity[:, 1:] = (
            positions[:, 1:]
            - positions[:, :-1]
        ) / dt

        acceleration = torch.zeros_like(
            positions
        )

        acceleration[:, 2:] = (
            velocity[:, 2:]
            - velocity[:, 1:-1]
        ) / dt

        features = torch.cat(
            [
                positions,
                velocity,
                acceleration,
            ],
            dim=-1,
        )

        return self.network(
            features
        )


# ============================================================
# TEMPORAL MOTION ENCODER
# ============================================================

class TemporalMotionEncoder(nn.Module):
    """
    Encodes historical motion with a GRU.
    """

    def __init__(
        self,
        config: TrajectoryPredictionConfig,
    ) -> None:
        super().__init__()

        self.features = MotionFeatureEncoder(
            config
        )

        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_gru_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_gru_layers > 1
                else 0.0
            ),
        )

    def forward(
        self,
        positions: Tensor,
    ) -> Tensor:
        features = self.features(
            positions
        )

        _, hidden = self.gru(
            features
        )

        return hidden


# ============================================================
# MULTI-MODAL TRAJECTORY DECODER
# ============================================================

class MultiModalTrajectoryDecoder(nn.Module):
    """
    Predicts multiple possible futures.

    Each mode has its own learned embedding.
    The decoder predicts position deltas,
    Gaussian variance, and mode scores.
    """

    def __init__(
        self,
        config: TrajectoryPredictionConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.mode_embeddings = nn.Embedding(
            config.num_modes,
            config.hidden_dim,
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                config.spatial_dim
                + config.action_dim
                + config.hidden_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
        )

        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_gru_layers,
            batch_first=True,
            dropout=(
                config.dropout
                if config.num_gru_layers > 1
                else 0.0
            ),
        )

        self.delta_head = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dim,
                config.spatial_dim,
            ),
        )

        self.log_variance_head = nn.Linear(
            config.hidden_dim,
            config.spatial_dim,
        )

        self.mode_head = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(
                config.hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        initial_position: Tensor,
        encoded_hidden: Tensor,
        actions: Tensor,
        *,
        targets: Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, future_steps, _ = (
            actions.shape
        )

        all_trajectories = []
        all_log_variances = []
        all_mode_logits = []

        for mode_index in range(
            self.config.num_modes
        ):
            mode_ids = torch.full(
                (batch_size,),
                mode_index,
                dtype=torch.long,
                device=actions.device,
            )

            mode_embedding = (
                self.mode_embeddings(
                    mode_ids
                )
            )

            hidden = encoded_hidden.clone()

            hidden = hidden.clone()

            hidden[-1] = (
                hidden[-1]
                + mode_embedding
            )

            current_position = (
                initial_position
            )

            mode_trajectory = []
            mode_variances = []

            for step in range(
                future_steps
            ):
                decoder_input = torch.cat(
                    [
                        current_position,
                        actions[:, step],
                        mode_embedding,
                    ],
                    dim=-1,
                )

                decoder_input = (
                    self.input_projection(
                        decoder_input
                    )
                )

                output, hidden = self.gru(
                    decoder_input.unsqueeze(1),
                    hidden,
                )

                features = output[
                    :, -1
                ]

                delta = self.delta_head(
                    features
                )

                next_position = (
                    current_position + delta
                )

                log_variance = (
                    self.log_variance_head(
                        features
                    ).clamp(
                        self.config.min_log_variance,
                        self.config.max_log_variance,
                    )
                )

                mode_trajectory.append(
                    next_position
                )

                mode_variances.append(
                    log_variance
                )

                current_position = (
                    next_position
                )

                if (
                    self.training
                    and targets is not None
                    and teacher_forcing_ratio > 0
                ):
                    teacher_mask = (
                        torch.rand(
                            batch_size,
                            1,
                            device=actions.device,
                        )
                        < teacher_forcing_ratio
                    )

                    current_position = torch.where(
                        teacher_mask,
                        targets[:, step],
                        current_position,
                    )

            trajectory = torch.stack(
                mode_trajectory,
                dim=1,
            )

            variances = torch.stack(
                mode_variances,
                dim=1,
            )

            mode_score = (
                self.mode_head(
                    hidden[-1]
                ).squeeze(-1)
            )

            all_trajectories.append(
                trajectory
            )

            all_log_variances.append(
                variances
            )

            all_mode_logits.append(
                mode_score
            )

        trajectories = torch.stack(
            all_trajectories,
            dim=1,
        )

        log_variances = torch.stack(
            all_log_variances,
            dim=1,
        )

        mode_logits = torch.stack(
            all_mode_logits,
            dim=1,
        )

        return (
            trajectories,
            log_variances,
            mode_logits,
        )


# ============================================================
# PREDICTION OUTPUT
# ============================================================

@dataclass
class TrajectoryPredictionOutput:
    trajectories: Tensor
    log_variances: Tensor

    mode_logits: Tensor
    mode_probabilities: Tensor

    expected_trajectory: Tensor
    uncertainty: Tensor

    def detach(self) -> "TrajectoryPredictionOutput":
        return TrajectoryPredictionOutput(
            trajectories=(
                self.trajectories.detach()
            ),
            log_variances=(
                self.log_variances.detach()
            ),
            mode_logits=(
                self.mode_logits.detach()
            ),
            mode_probabilities=(
                self.mode_probabilities.detach()
            ),
            expected_trajectory=(
                self.expected_trajectory.detach()
            ),
            uncertainty=(
                self.uncertainty.detach()
            ),
        )


# ============================================================
# MAIN TRAJECTORY PREDICTION MODEL
# ============================================================

class TrajectoryPredictionModel(nn.Module):
    """
    Neural multi-modal trajectory predictor.

    Input:
        historical positions
        optional future actions

    Output:
        multiple future trajectories
        mode probabilities
        predictive uncertainty
    """

    def __init__(
        self,
        config: TrajectoryPredictionConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or TrajectoryPredictionConfig()
        )

        self.encoder = TemporalMotionEncoder(
            self.config
        )

        self.decoder = MultiModalTrajectoryDecoder(
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

    def forward(
        self,
        history_positions: Tensor | np.ndarray,
        *,
        future_steps: int,
        actions: Tensor | np.ndarray | None = None,
        target_positions: (
            Tensor | np.ndarray | None
        ) = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> TrajectoryPredictionOutput:
        if not (
            1
            <= future_steps
            <= self.config.max_future_steps
        ):
            raise TrajectoryValidationError(
                "Invalid future_steps"
            )

        history = prepare_history(
            history_positions,
            self.config,
            self.device,
        )

        batch_size = (
            history.shape[0]
        )

        actions = prepare_future_actions(
            actions,
            batch_size=batch_size,
            future_steps=future_steps,
            config=self.config,
            device=self.device,
        )

        targets = None

        if target_positions is not None:
            targets = prepare_targets(
                target_positions,
                batch_size=batch_size,
                future_steps=future_steps,
                config=self.config,
                device=self.device,
            )

        hidden = self.encoder(
            history
        )

        initial_position = history[
            :, -1
        ]

        (
            trajectories,
            log_variances,
            mode_logits,
        ) = self.decoder(
            initial_position,
            hidden,
            actions,
            targets=targets,
            teacher_forcing_ratio=(
                teacher_forcing_ratio
            ),
        )

        mode_probabilities = F.softmax(
            mode_logits,
            dim=1,
        )

        weights = mode_probabilities[
            :, :, None, None
        ]

        expected_trajectory = (
            trajectories * weights
        ).sum(
            dim=1
        )

        epistemic_variance = (
            (
                trajectories
                - expected_trajectory.unsqueeze(1)
            ).square()
            * weights
        ).sum(
            dim=1
        )

        aleatoric_variance = (
            log_variances.exp()
            * weights
        ).sum(
            dim=1
        )

        uncertainty = (
            epistemic_variance
            + aleatoric_variance
        ).clamp_min(0).sqrt()

        return TrajectoryPredictionOutput(
            trajectories=trajectories,
            log_variances=log_variances,
            mode_logits=mode_logits,
            mode_probabilities=mode_probabilities,
            expected_trajectory=expected_trajectory,
            uncertainty=uncertainty,
        )

    @torch.no_grad()
    def predict(
        self,
        history_positions: Tensor | np.ndarray,
        *,
        future_steps: int,
        actions: Tensor | np.ndarray | None = None,
    ) -> TrajectoryPredictionOutput:
        previous_mode = self.training

        try:
            self.eval()

            return self.forward(
                history_positions,
                future_steps=future_steps,
                actions=actions,
            ).detach()

        finally:
            self.train(
                previous_mode
            )

    @torch.no_grad()
    def predict_best_trajectory(
        self,
        history_positions: Tensor | np.ndarray,
        *,
        future_steps: int,
        actions: Tensor | np.ndarray | None = None,
    ) -> Tensor:
        prediction = self.predict(
            history_positions,
            future_steps=future_steps,
            actions=actions,
        )

        best_modes = (
            prediction.mode_probabilities.argmax(
                dim=1
            )
        )

        batch_indices = torch.arange(
            prediction.trajectories.shape[0],
            device=self.device,
        )

        return prediction.trajectories[
            batch_indices,
            best_modes,
        ]

    @torch.no_grad()
    def sample_trajectories(
        self,
        history_positions: Tensor | np.ndarray,
        *,
        future_steps: int,
        actions: Tensor | np.ndarray | None = None,
        num_samples: int = 10,
    ) -> Tensor:
        if num_samples <= 0:
            raise TrajectoryValidationError(
                "num_samples must be positive"
            )

        prediction = self.predict(
            history_positions,
            future_steps=future_steps,
            actions=actions,
        )

        batch_size = (
            prediction.trajectories.shape[0]
        )

        mode_indices = torch.multinomial(
            prediction.mode_probabilities,
            num_samples=num_samples,
            replacement=True,
        )

        batch_indices = torch.arange(
            batch_size,
            device=self.device,
        ).unsqueeze(1).expand(
            -1,
            num_samples,
        )

        selected_means = (
            prediction.trajectories[
                batch_indices,
                mode_indices,
            ]
        )

        selected_log_variances = (
            prediction.log_variances[
                batch_indices,
                mode_indices,
            ]
        )

        noise = torch.randn_like(
            selected_means
        )

        return (
            selected_means
            + noise
            * torch.exp(
                0.5
                * selected_log_variances
            )
        )


# ============================================================
# TRAJECTORY LOSS
# ============================================================

def calculate_trajectory_loss(
    output: TrajectoryPredictionOutput,
    targets: Tensor,
    config: TrajectoryPredictionConfig,
) -> dict[str, Tensor]:
    """
    Best-of-K training.

    Select the mode closest to the
    ground-truth trajectory.

    Regression:
        mean displacement error

    NLL:
        Gaussian likelihood of best mode

    Mode loss:
        classify best matching mode
    """

    batch_size, num_modes, future_steps, spatial_dim = (
        output.trajectories.shape
    )

    expanded_targets = targets.unsqueeze(
        1
    )

    displacement_errors = (
        output.trajectories
        - expanded_targets
    ).norm(
        dim=-1
    )

    ade_per_mode = (
        displacement_errors.mean(
            dim=-1
        )
    )

    best_mode_indices = (
        ade_per_mode.argmin(
            dim=1
        )
    )

    batch_indices = torch.arange(
        batch_size,
        device=targets.device,
    )

    best_trajectories = (
        output.trajectories[
            batch_indices,
            best_mode_indices,
        ]
    )

    best_log_variances = (
        output.log_variances[
            batch_indices,
            best_mode_indices,
        ]
    )

    regression_loss = F.smooth_l1_loss(
        best_trajectories,
        targets,
    )

    nll_loss = (
        0.5
        * (
            math.log(2.0 * math.pi)
            + best_log_variances
            + (
                (
                    targets
                    - best_trajectories
                ).square()
                / best_log_variances.exp()
            )
        )
    ).mean()

    mode_loss = F.cross_entropy(
        output.mode_logits,
        best_mode_indices,
    )

    total_loss = (
        config.regression_weight
        * regression_loss
        + config.nll_weight
        * nll_loss
        + config.mode_loss_weight
        * mode_loss
    )

    return {
        "total_loss": total_loss,
        "regression_loss": regression_loss,
        "nll_loss": nll_loss,
        "mode_loss": mode_loss,
    }


# ============================================================
# TRAJECTORY METRICS
# ============================================================

def trajectory_metrics(
    output: TrajectoryPredictionOutput,
    targets: Tensor,
) -> dict[str, float]:
    """
    ADE:
        Average Displacement Error

    FDE:
        Final Displacement Error

    minADE / minFDE:
        Best matching mode.
    """

    errors = (
        output.trajectories
        - targets.unsqueeze(1)
    ).norm(
        dim=-1
    )

    ade_per_mode = errors.mean(
        dim=-1
    )

    fde_per_mode = errors[
        :, :, -1
    ]

    best_mode = (
        output.mode_probabilities.argmax(
            dim=1
        )
    )

    batch_indices = torch.arange(
        targets.shape[0],
        device=targets.device,
    )

    ade = ade_per_mode[
        batch_indices,
        best_mode,
    ].mean()

    fde = fde_per_mode[
        batch_indices,
        best_mode,
    ].mean()

    min_ade = (
        ade_per_mode.min(
            dim=1
        ).values.mean()
    )

    min_fde = (
        fde_per_mode.min(
            dim=1
        ).values.mean()
    )

    return {
        "ade": float(
            ade.item()
        ),
        "fde": float(
            fde.item()
        ),
        "min_ade": float(
            min_ade.item()
        ),
        "min_fde": float(
            min_fde.item()
        ),
        "mean_uncertainty": float(
            output.uncertainty.mean().item()
        ),
    }


# ============================================================
# COLLISION DISTANCE ANALYSIS
# ============================================================

def calculate_collision_distances(
    trajectory_a: Tensor | np.ndarray,
    trajectory_b: Tensor | np.ndarray,
) -> Tensor:
    """
    Calculate point-to-point distances
    between synchronized trajectories.

    Inputs:
        [B,T,D] or [T,D]

    Output:
        [B,T]
    """

    a = torch.as_tensor(
        trajectory_a,
        dtype=torch.float32,
    )

    b = torch.as_tensor(
        trajectory_b,
        dtype=torch.float32,
        device=a.device,
    )

    if a.ndim == 2:
        a = a.unsqueeze(0)

    if b.ndim == 2:
        b = b.unsqueeze(0)

    if a.shape != b.shape:
        raise TrajectoryValidationError(
            "Trajectory shapes must match"
        )

    if a.ndim != 3:
        raise TrajectoryValidationError(
            "Expected trajectories [B,T,D]"
        )

    return torch.linalg.vector_norm(
        a - b,
        dim=-1,
    )


def detect_possible_collision(
    trajectory_a: Tensor | np.ndarray,
    trajectory_b: Tensor | np.ndarray,
    *,
    safety_distance: float = 1.0,
) -> dict[str, Any]:
    """
    Detect proximity below a threshold.

    Does not account for:
        object geometry
        swept collision volumes
        sensor uncertainty
        real-time control latency
    """

    if safety_distance <= 0:
        raise TrajectoryValidationError(
            "safety_distance must be positive"
        )

    distances = calculate_collision_distances(
        trajectory_a,
        trajectory_b,
    )

    possible_collision = (
        distances < safety_distance
    )

    return {
        "possible_collision": (
            possible_collision.any(
                dim=1
            ).cpu().tolist()
        ),
        "minimum_distance": (
            distances.min(
                dim=1
            ).values.cpu().tolist()
        ),
        "collision_steps": [
            torch.where(
                row
            )[0].cpu().tolist()
            for row in possible_collision
        ],
    }


# ============================================================
# TRAINING BATCH
# ============================================================

@dataclass
class TrajectoryTrainingBatch:
    history_positions: Tensor
    target_positions: Tensor
    actions: Tensor | None = None


# ============================================================
# TRAINER
# ============================================================

class TrajectoryPredictionTrainer:
    """
    Supervised multi-modal trajectory
    prediction trainer.
    """

    def __init__(
        self,
        model: TrajectoryPredictionModel,
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
        batch: TrajectoryTrainingBatch,
    ) -> TrajectoryTrainingBatch:
        history = prepare_history(
            batch.history_positions,
            self.config,
            self.model.device,
        )

        targets = torch.as_tensor(
            batch.target_positions,
            dtype=torch.float32,
            device=self.model.device,
        )

        if targets.ndim == 2:
            targets = targets.unsqueeze(0)

        if targets.ndim != 3:
            raise TrajectoryValidationError(
                "Targets must be [B,T,D]"
            )

        if not (
            1
            <= targets.shape[1]
            <= self.config.max_future_steps
        ):
            raise TrajectoryValidationError(
                "Invalid target horizon"
            )

        targets = prepare_targets(
            targets,
            batch_size=history.shape[0],
            future_steps=targets.shape[1],
            config=self.config,
            device=self.model.device,
        )

        actions = prepare_future_actions(
            batch.actions,
            batch_size=history.shape[0],
            future_steps=targets.shape[1],
            config=self.config,
            device=self.model.device,
        )

        return TrajectoryTrainingBatch(
            history_positions=history,
            target_positions=targets,
            actions=actions,
        )

    def train_step(
        self,
        batch: TrajectoryTrainingBatch,
    ) -> dict[str, float]:
        self.model.train()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.history_positions,
            future_steps=(
                batch.target_positions.shape[1]
            ),
            actions=batch.actions,
            target_positions=(
                batch.target_positions
            ),
            teacher_forcing_ratio=(
                self.config.teacher_forcing_ratio
            ),
        )

        losses = calculate_trajectory_loss(
            output,
            batch.target_positions,
            self.config,
        )

        if not torch.isfinite(
            losses["total_loss"]
        ):
            raise TrajectoryPredictionError(
                "Non-finite trajectory loss"
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
        batch: TrajectoryTrainingBatch,
    ) -> dict[str, float]:
        self.model.eval()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.history_positions,
            future_steps=(
                batch.target_positions.shape[1]
            ),
            actions=batch.actions,
            teacher_forcing_ratio=0.0,
        )

        losses = calculate_trajectory_loss(
            output,
            batch.target_positions,
            self.config,
        )

        metrics = trajectory_metrics(
            output,
            batch.target_positions,
        )

        metrics.update(
            {
                name: float(
                    value.item()
                )
                for name, value in losses.items()
            }
        )

        return metrics


# ============================================================
# NUMPY ADAPTER
# ============================================================

class TrajectoryPredictionAdapter:
    """
    NumPy-compatible trajectory predictor
    for planners and simulation systems.
    """

    def __init__(
        self,
        model: TrajectoryPredictionModel,
    ) -> None:
        self.model = model

    def predict_trajectory(
        self,
        history_positions: np.ndarray,
        *,
        future_steps: int,
        actions: np.ndarray | None = None,
    ) -> np.ndarray:
        trajectory = (
            self.model.predict_best_trajectory(
                history_positions,
                future_steps=future_steps,
                actions=actions,
            )
        )

        return (
            trajectory[0]
            .cpu()
            .numpy()
        )

    def predict_all_modes(
        self,
        history_positions: np.ndarray,
        *,
        future_steps: int,
        actions: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        prediction = self.model.predict(
            history_positions,
            future_steps=future_steps,
            actions=actions,
        )

        return {
            "trajectories": (
                prediction.trajectories[0]
                .cpu()
                .numpy()
            ),
            "mode_probabilities": (
                prediction.mode_probabilities[0]
                .cpu()
                .numpy()
            ),
            "expected_trajectory": (
                prediction.expected_trajectory[0]
                .cpu()
                .numpy()
            ),
            "uncertainty": (
                prediction.uncertainty[0]
                .cpu()
                .numpy()
            ),
        }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_trajectory_prediction_model(
    model: TrajectoryPredictionModel,
    path: str | Path,
    *,
    trainer: TrajectoryPredictionTrainer | None = None,
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


def load_trajectory_prediction_model(
    path: str | Path,
    *,
    device: str = "cpu",
    restore_trainer: bool = False,
) -> (
    TrajectoryPredictionModel
    | tuple[
        TrajectoryPredictionModel,
        TrajectoryPredictionTrainer,
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

    model = TrajectoryPredictionModel(
        TrajectoryPredictionConfig(
            **config_data
        )
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    if not restore_trainer:
        return model

    trainer = TrajectoryPredictionTrainer(
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

def create_trajectory_predictor(
    config: TrajectoryPredictionConfig | None = None,
) -> TrajectoryPredictionModel:
    return TrajectoryPredictionModel(
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
        "NEUROWORLD AI - TRAJECTORY PREDICTION"
    )
    print("=" * 65)

    config = TrajectoryPredictionConfig(
        spatial_dim=2,
        action_dim=2,
        hidden_dim=64,
        num_gru_layers=2,
        num_modes=3,
        max_history_steps=20,
        max_future_steps=10,
        timestep_seconds=0.1,
    )

    model = TrajectoryPredictionModel(
        config
    )

    trainer = TrajectoryPredictionTrainer(
        model
    )

    batch_size = 8
    history_steps = 10
    future_steps = 5

    # --------------------------------------------------------
    # SYNTHETIC LINEAR MOTION
    # --------------------------------------------------------

    initial_positions = torch.randn(
        batch_size,
        1,
        2,
    )

    velocities = torch.randn(
        batch_size,
        1,
        2,
    ) * 0.1

    history_times = torch.arange(
        history_steps,
    ).float().view(
        1,
        history_steps,
        1,
    )

    history = (
        initial_positions
        + history_times * velocities
    )

    future_times = torch.arange(
        history_steps,
        history_steps + future_steps,
    ).float().view(
        1,
        future_steps,
        1,
    )

    targets = (
        initial_positions
        + future_times * velocities
    )

    actions = torch.zeros(
        batch_size,
        future_steps,
        config.action_dim,
    )

    batch = TrajectoryTrainingBatch(
        history_positions=history,
        target_positions=targets,
        actions=actions,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for step in range(10):
        metrics = trainer.train_step(
            batch
        )

        if step in (0, 4, 9):
            print(
                f"Training step {step + 1}:",
                metrics,
            )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        history[:2],
        future_steps=future_steps,
        actions=actions[:2],
    )

    print(
        "Trajectory shape:",
        tuple(
            prediction.trajectories.shape
        ),
    )

    print(
        "Mode probabilities:",
        prediction.mode_probabilities[0]
        .cpu()
        .numpy(),
    )

    print(
        "Uncertainty shape:",
        tuple(
            prediction.uncertainty.shape
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
    # COLLISION ANALYSIS
    # --------------------------------------------------------

    trajectory_a = prediction.expected_trajectory[
        0
    ]

    trajectory_b = prediction.expected_trajectory[
        1
    ]

    collision = detect_possible_collision(
        trajectory_a,
        trajectory_b,
        safety_distance=0.5,
    )

    print(
        "Possible collision:",
        collision,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = save_trajectory_prediction_model(
        model,
        "models/trajectory_prediction_demo.pt",
        trainer=trainer,
    )

    restored_model = (
        load_trajectory_prediction_model(
            checkpoint
        )
    )

    print(
        "Restored model:",
        type(restored_model).__name__,
    )

    print(
        "Trajectory prediction demonstration completed"
    )


if __name__ == "__main__":
    main()
