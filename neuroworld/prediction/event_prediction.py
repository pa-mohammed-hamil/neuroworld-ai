
"""
NeuroWorld AI
File: neuroworld/prediction/event_prediction.py

Future Event Prediction and Temporal Intelligence.

Features:
    1. Temporal Transformer encoder
    2. Multi-label future event prediction
    3. Multi-horizon event forecasting
    4. Time-to-event prediction
    5. Event probability estimation
    6. Monte Carlo dropout uncertainty
    7. Event risk classification
    8. State/action sequence fusion
    9. Padding-aware temporal attention
   10. Supervised event model training
   11. Gradient clipping
   12. Evaluation metrics
   13. Checkpoint save/load
   14. Demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.prediction.event_prediction

Inputs:
    states: [B, T, state_dim]
    actions: [B, T, action_dim]

Outputs:
    probabilities: [B, H, E]
    time_to_event: [B, H, E]

Where:
    B = batch size
    T = history length
    H = prediction horizons
    E = number of event types

Predictions are statistical estimates,
not verified future events.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class EventPredictionError(Exception):
    """Base event prediction exception."""


class EventValidationError(EventPredictionError):
    """Invalid event prediction input."""


# ============================================================
# EVENT TYPES
# ============================================================

class EventType(str, Enum):
    COLLISION = "collision"
    OBSTACLE_ENCOUNTER = "obstacle_encounter"
    OBJECT_INTERACTION = "object_interaction"
    GOAL_REACHED = "goal_reached"
    NAVIGATION_FAILURE = "navigation_failure"
    UNSTABLE_MOTION = "unstable_motion"
    OBJECT_FALL = "object_fall"
    HUMAN_PROXIMITY = "human_proximity"
    SENSOR_FAILURE = "sensor_failure"
    LOW_ENERGY = "low_energy"
    UNKNOWN_ANOMALY = "unknown_anomaly"


class EventRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EventPredictionConfig:
    state_dim: int = 128
    action_dim: int = 8

    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_transformer_layers: int = 3

    feedforward_dim: int = 512
    dropout: float = 0.1

    max_sequence_length: int = 128

    prediction_horizons: tuple[int, ...] = (
        1, 3, 5, 10
    )

    event_names: tuple[str, ...] = tuple(
        event.value for event in EventType
    )

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    event_loss_weight: float = 1.0
    time_loss_weight: float = 0.2

    event_threshold: float = 0.5

    moderate_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.65
    critical_risk_threshold: float = 0.85

    mc_dropout_samples: int = 10

    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "hidden_dim",
            "num_attention_heads",
            "num_transformer_layers",
            "feedforward_dim",
            "max_sequence_length",
            "mc_dropout_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if (
            self.hidden_dim
            % self.num_attention_heads
            != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible "
                "by num_attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if not self.prediction_horizons:
            raise ValueError(
                "At least one prediction horizon required"
            )

        if any(
            horizon <= 0
            for horizon in self.prediction_horizons
        ):
            raise ValueError(
                "Prediction horizons must be positive"
            )

        if len(set(self.prediction_horizons)) != len(
            self.prediction_horizons
        ):
            raise ValueError(
                "Prediction horizons must be unique"
            )

        if not self.event_names:
            raise ValueError(
                "At least one event type required"
            )

        if len(set(self.event_names)) != len(
            self.event_names
        ):
            raise ValueError(
                "Event names must be unique"
            )

        if not (
            0
            <= self.moderate_risk_threshold
            < self.high_risk_threshold
            < self.critical_risk_threshold
            <= 1
        ):
            raise ValueError(
                "Invalid risk thresholds"
            )

        if not 0 <= self.event_threshold <= 1:
            raise ValueError(
                "event_threshold must be in [0,1]"
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
            "event_loss_weight",
            "time_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


# ============================================================
# INPUT VALIDATION
# ============================================================

def prepare_sequence(
    states: Tensor | np.ndarray,
    actions: Tensor | np.ndarray,
    *,
    config: EventPredictionConfig,
    device: torch.device,
    padding_mask: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    states = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=device,
    )

    actions = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=device,
    )

    if states.ndim == 2:
        states = states.unsqueeze(0)

    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    if states.ndim != 3 or actions.ndim != 3:
        raise EventValidationError(
            "Expected sequences [B,T,D]"
        )

    if states.shape[:2] != actions.shape[:2]:
        raise EventValidationError(
            "State/action sequence shapes mismatch"
        )

    if states.shape[-1] != config.state_dim:
        raise EventValidationError(
            "Invalid state dimension"
        )

    if actions.shape[-1] != config.action_dim:
        raise EventValidationError(
            "Invalid action dimension"
        )

    batch_size, sequence_length, _ = states.shape

    if sequence_length == 0:
        raise EventValidationError(
            "Sequence cannot be empty"
        )

    if sequence_length > config.max_sequence_length:
        raise EventValidationError(
            "Sequence exceeds max_sequence_length"
        )

    if (
        not torch.isfinite(states).all()
        or not torch.isfinite(actions).all()
    ):
        raise EventValidationError(
            "Sequence contains non-finite values"
        )

    if padding_mask is None:
        padding_mask = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=device,
        )

    else:
        padding_mask = torch.as_tensor(
            padding_mask,
            dtype=torch.bool,
            device=device,
        )

        if padding_mask.shape != (
            batch_size,
            sequence_length,
        ):
            raise EventValidationError(
                "Invalid padding mask shape"
            )

    if padding_mask.all(dim=1).any():
        raise EventValidationError(
            "Each sequence requires at least "
            "one valid timestep"
        )

    return states, actions, padding_mask


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal temporal position embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_length: int,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        frequencies = torch.exp(
            torch.arange(
                0,
                hidden_dim,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(10000.0)
                / hidden_dim
            )
        )

        encoding = torch.zeros(
            max_length,
            hidden_dim,
        )

        encoding[:, 0::2] = torch.sin(
            positions * frequencies
        )

        cos_width = encoding[:, 1::2].shape[1]

        encoding[:, 1::2] = torch.cos(
            positions * frequencies[:cos_width]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        sequence_length = features.shape[1]

        return (
            features
            + self.encoding[:, :sequence_length]
        )


# ============================================================
# TEMPORAL EVENT ENCODER
# ============================================================

class TemporalEventEncoder(nn.Module):
    """
    Encodes temporal world-state/action history
    using Transformer attention.
    """

    def __init__(
        self,
        config: EventPredictionConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Sequential(
            nn.Linear(
                config.state_dim
                + config.action_dim,
                config.hidden_dim,
            ),
            nn.LayerNorm(
                config.hidden_dim
            ),
            nn.GELU(),
        )

        self.position_encoding = (
            SinusoidalPositionEncoding(
                config.hidden_dim,
                config.max_sequence_length,
            )
        )

        transformer_layer = (
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=(
                    config.feedforward_dim
                ),
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=(
                config.num_transformer_layers
            ),
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(
            config.hidden_dim
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [states, actions],
            dim=-1,
        )

        features = self.input_projection(
            features
        )

        features = self.position_encoding(
            features
        )

        encoded = self.transformer(
            features,
            src_key_padding_mask=padding_mask,
        )

        # Select the final non-padded timestep.
        # This supports arbitrary padding patterns
        # as long as the final valid position exists.
        sequence_length = states.shape[1]

        positions = torch.arange(
            sequence_length,
            device=states.device,
        ).unsqueeze(0)

        valid_positions = positions.masked_fill(
            padding_mask,
            -1,
        )

        last_indices = valid_positions.max(
            dim=1
        ).values

        batch_indices = torch.arange(
            states.shape[0],
            device=states.device,
        )

        context = encoded[
            batch_indices,
            last_indices,
        ]

        return self.output_norm(
            context
        )


# ============================================================
# EVENT PREDICTION OUTPUT
# ============================================================

@dataclass
class EventPredictionOutput:
    logits: Tensor
    probabilities: Tensor

    time_to_event: Tensor
    context_embedding: Tensor

    uncertainty: Tensor | None = None

    def detach(self) -> "EventPredictionOutput":
        return EventPredictionOutput(
            logits=self.logits.detach(),
            probabilities=(
                self.probabilities.detach()
            ),
            time_to_event=(
                self.time_to_event.detach()
            ),
            context_embedding=(
                self.context_embedding.detach()
            ),
            uncertainty=(
                self.uncertainty.detach()
                if self.uncertainty is not None
                else None
            ),
        )


# ============================================================
# NEURAL EVENT PREDICTION MODEL
# ============================================================

class EventPredictionModel(nn.Module):
    """
    Multi-horizon event prediction network.

    For each horizon, predicts:
        - Event occurrence probabilities
        - Nonnegative time-to-event estimates

    Event probabilities are multi-label:
    multiple event types may occur together.
    """

    def __init__(
        self,
        config: EventPredictionConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or EventPredictionConfig()
        )

        self.encoder = TemporalEventEncoder(
            self.config
        )

        num_horizons = len(
            self.config.prediction_horizons
        )

        num_events = len(
            self.config.event_names
        )

        output_dim = (
            num_horizons * num_events
        )

        self.event_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_dim,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                self.config.dropout
            ),
            nn.Linear(
                self.config.hidden_dim,
                output_dim,
            ),
        )

        self.time_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_dim,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                self.config.dropout
            ),
            nn.Linear(
                self.config.hidden_dim,
                output_dim,
            ),
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
        padding_mask: (
            Tensor | np.ndarray | None
        ) = None,
    ) -> EventPredictionOutput:
        (
            states,
            actions,
            padding_mask,
        ) = prepare_sequence(
            states,
            actions,
            config=self.config,
            device=self.device,
            padding_mask=padding_mask,
        )

        context = self.encoder(
            states,
            actions,
            padding_mask,
        )

        batch_size = (
            context.shape[0]
        )

        num_horizons = len(
            self.config.prediction_horizons
        )

        num_events = len(
            self.config.event_names
        )

        logits = self.event_head(
            context
        ).reshape(
            batch_size,
            num_horizons,
            num_events,
        )

        probabilities = torch.sigmoid(
            logits
        )

        time_to_event = F.softplus(
            self.time_head(
                context
            )
        ).reshape(
            batch_size,
            num_horizons,
            num_events,
        )

        return EventPredictionOutput(
            logits=logits,
            probabilities=probabilities,
            time_to_event=time_to_event,
            context_embedding=context,
        )

    # --------------------------------------------------------
    # MONTE CARLO DROPOUT
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        states: Tensor | np.ndarray,
        actions: Tensor | np.ndarray,
        *,
        padding_mask: (
            Tensor | np.ndarray | None
        ) = None,
        estimate_uncertainty: bool = True,
        mc_samples: int | None = None,
    ) -> EventPredictionOutput:
        if mc_samples is None:
            mc_samples = (
                self.config.mc_dropout_samples
            )

        if mc_samples <= 0:
            raise EventValidationError(
                "mc_samples must be positive"
            )

        previous_mode = self.training

        try:
            if not estimate_uncertainty:
                self.eval()

                return self.forward(
                    states,
                    actions,
                    padding_mask=padding_mask,
                ).detach()

            # Activate dropout for stochastic
            # inference without gradients.
            self.train()

            predictions = [
                self.forward(
                    states,
                    actions,
                    padding_mask=padding_mask,
                )
                for _ in range(mc_samples)
            ]

            probabilities = torch.stack(
                [
                    prediction.probabilities
                    for prediction in predictions
                ],
                dim=0,
            )

            times = torch.stack(
                [
                    prediction.time_to_event
                    for prediction in predictions
                ],
                dim=0,
            )

            contexts = torch.stack(
                [
                    prediction.context_embedding
                    for prediction in predictions
                ],
                dim=0,
            )

            mean_probabilities = (
                probabilities.mean(
                    dim=0
                )
            )

            uncertainty = (
                probabilities.std(
                    dim=0,
                    unbiased=False,
                )
            )

            # Mean probability is the ensemble
            # estimate; logits are reconstructed
            # only for output convenience.
            mean_logits = torch.logit(
                mean_probabilities.clamp(
                    1e-6,
                    1.0 - 1e-6,
                )
            )

            return EventPredictionOutput(
                logits=mean_logits,
                probabilities=mean_probabilities,
                time_to_event=times.mean(dim=0),
                context_embedding=(
                    contexts.mean(dim=0)
                ),
                uncertainty=uncertainty,
            ).detach()

        finally:
            self.train(
                previous_mode
            )


# ============================================================
# EVENT RISK ASSESSMENT
# ============================================================

@dataclass
class PredictedEvent:
    event_name: str
    horizon: int

    probability: float
    time_to_event: float

    risk_level: EventRiskLevel

    uncertainty: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(
            self
        )

        result["risk_level"] = (
            self.risk_level.value
        )

        return result


class EventRiskAssessor:
    """
    Converts event probabilities into
    configurable risk categories.

    Thresholds are illustrative and must
    be calibrated for each event type.
    """

    def __init__(
        self,
        config: EventPredictionConfig,
    ) -> None:
        self.config = config

    def classify(
        self,
        probability: float,
    ) -> EventRiskLevel:
        if (
            probability
            >= self.config.critical_risk_threshold
        ):
            return EventRiskLevel.CRITICAL

        if (
            probability
            >= self.config.high_risk_threshold
        ):
            return EventRiskLevel.HIGH

        if (
            probability
            >= self.config.moderate_risk_threshold
        ):
            return EventRiskLevel.MODERATE

        return EventRiskLevel.LOW

    def summarize(
        self,
        output: EventPredictionOutput,
        *,
        batch_index: int = 0,
        threshold: float | None = None,
    ) -> list[PredictedEvent]:
        if threshold is None:
            threshold = (
                self.config.event_threshold
            )

        probabilities = (
            output.probabilities[
                batch_index
            ]
            .detach()
            .cpu()
            .numpy()
        )

        times = (
            output.time_to_event[
                batch_index
            ]
            .detach()
            .cpu()
            .numpy()
        )

        uncertainties = None

        if output.uncertainty is not None:
            uncertainties = (
                output.uncertainty[
                    batch_index
                ]
                .detach()
                .cpu()
                .numpy()
            )

        events = []

        for horizon_index, horizon in enumerate(
            self.config.prediction_horizons
        ):
            for event_index, event_name in enumerate(
                self.config.event_names
            ):
                probability = float(
                    probabilities[
                        horizon_index,
                        event_index,
                    ]
                )

                if probability < threshold:
                    continue

                uncertainty = None

                if uncertainties is not None:
                    uncertainty = float(
                        uncertainties[
                            horizon_index,
                            event_index,
                        ]
                    )

                events.append(
                    PredictedEvent(
                        event_name=event_name,
                        horizon=horizon,
                        probability=probability,
                        time_to_event=float(
                            times[
                                horizon_index,
                                event_index,
                            ]
                        ),
                        risk_level=self.classify(
                            probability
                        ),
                        uncertainty=uncertainty,
                    )
                )

        return sorted(
            events,
            key=lambda event: (
                -event.probability,
                event.horizon,
            ),
        )


# ============================================================
# TRAINING BATCH
# ============================================================

@dataclass
class EventTrainingBatch:
    states: Tensor
    actions: Tensor

    event_labels: Tensor
    time_targets: Tensor

    padding_mask: Tensor | None = None
    time_valid_mask: Tensor | None = None


# ============================================================
# TRAINER
# ============================================================

class EventPredictionTrainer:
    """
    Supervised event prediction trainer.

    Event loss:
        Binary cross entropy with logits

    Time loss:
        Smooth L1 on observed events
        with valid time targets

    Missing event times must be masked.
    """

    def __init__(
        self,
        model: EventPredictionModel,
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
        batch: EventTrainingBatch,
    ) -> EventTrainingBatch:
        device = self.model.device

        states = torch.as_tensor(
            batch.states,
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            batch.actions,
            dtype=torch.float32,
            device=device,
        )

        labels = torch.as_tensor(
            batch.event_labels,
            dtype=torch.float32,
            device=device,
        )

        times = torch.as_tensor(
            batch.time_targets,
            dtype=torch.float32,
            device=device,
        )

        if states.ndim == 2:
            states = states.unsqueeze(0)

        if actions.ndim == 2:
            actions = actions.unsqueeze(0)

        expected_shape = (
            states.shape[0],
            len(
                self.config.prediction_horizons
            ),
            len(
                self.config.event_names
            ),
        )

        if labels.shape != expected_shape:
            raise EventValidationError(
                f"Expected event labels "
                f"{expected_shape}"
            )

        if times.shape != expected_shape:
            raise EventValidationError(
                f"Expected time targets "
                f"{expected_shape}"
            )

        if (
            not torch.isfinite(labels).all()
            or (labels < 0).any()
            or (labels > 1).any()
        ):
            raise EventValidationError(
                "Event labels must be in [0,1]"
            )

        if batch.time_valid_mask is None:
            time_valid_mask = (
                labels > 0.5
            )

        else:
            time_valid_mask = torch.as_tensor(
                batch.time_valid_mask,
                dtype=torch.bool,
                device=device,
            )

            if time_valid_mask.shape != expected_shape:
                raise EventValidationError(
                    "Invalid time_valid_mask shape"
                )

        if (
            not torch.isfinite(
                times[time_valid_mask]
            ).all()
            or (
                times[time_valid_mask] < 0
            ).any()
        ):
            raise EventValidationError(
                "Valid time targets must be "
                "finite and nonnegative"
            )

        padding_mask = None

        if batch.padding_mask is not None:
            padding_mask = torch.as_tensor(
                batch.padding_mask,
                dtype=torch.bool,
                device=device,
            )

        return EventTrainingBatch(
            states=states,
            actions=actions,
            event_labels=labels,
            time_targets=times,
            padding_mask=padding_mask,
            time_valid_mask=time_valid_mask,
        )

    def _losses(
        self,
        batch: EventTrainingBatch,
        output: EventPredictionOutput,
    ) -> tuple[Tensor, Tensor, Tensor]:
        event_loss = (
            F.binary_cross_entropy_with_logits(
                output.logits,
                batch.event_labels,
            )
        )

        mask = batch.time_valid_mask

        if mask.any():
            time_loss = F.smooth_l1_loss(
                output.time_to_event[mask],
                batch.time_targets[mask],
            )

        else:
            time_loss = (
                output.time_to_event.sum()
                * 0.0
            )

        total_loss = (
            self.config.event_loss_weight
            * event_loss
            + self.config.time_loss_weight
            * time_loss
        )

        return (
            total_loss,
            event_loss,
            time_loss,
        )

    def train_step(
        self,
        batch: EventTrainingBatch,
    ) -> dict[str, float]:
        self.model.train()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.states,
            batch.actions,
            padding_mask=batch.padding_mask,
        )

        (
            total_loss,
            event_loss,
            time_loss,
        ) = self._losses(
            batch,
            output,
        )

        if not torch.isfinite(total_loss):
            raise EventPredictionError(
                "Non-finite training loss"
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        gradient_norm = (
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        self.training_steps += 1

        return {
            "total_loss": float(
                total_loss.detach().item()
            ),
            "event_loss": float(
                event_loss.detach().item()
            ),
            "time_loss": float(
                time_loss.detach().item()
            ),
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
        batch: EventTrainingBatch,
    ) -> dict[str, float]:
        self.model.eval()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.states,
            batch.actions,
            padding_mask=batch.padding_mask,
        )

        (
            total_loss,
            event_loss,
            time_loss,
        ) = self._losses(
            batch,
            output,
        )

        predictions = (
            output.probabilities
            >= self.config.event_threshold
        )

        targets = (
            batch.event_labels > 0.5
        )

        true_positive = (
            predictions & targets
        ).sum().float()

        false_positive = (
            predictions & ~targets
        ).sum().float()

        false_negative = (
            ~predictions & targets
        ).sum().float()

        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            ).clamp_min(1.0)
        )

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            ).clamp_min(1.0)
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision + recall
            ).clamp_min(1e-8)
        )

        return {
            "total_loss": float(
                total_loss.item()
            ),
            "event_loss": float(
                event_loss.item()
            ),
            "time_loss": float(
                time_loss.item()
            ),
            "precision": float(
                precision.item()
            ),
            "recall": float(
                recall.item()
            ),
            "f1": float(
                f1.item()
            ),
            "mean_event_probability": float(
                output.probabilities.mean().item()
            ),
        }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_event_prediction_model(
    model: EventPredictionModel,
    path: str | Path,
    *,
    trainer: EventPredictionTrainer | None = None,
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


def load_event_prediction_model(
    path: str | Path,
    *,
    device: str = "cpu",
    restore_trainer: bool = False,
) -> (
    EventPredictionModel
    | tuple[
        EventPredictionModel,
        EventPredictionTrainer,
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

    model = EventPredictionModel(
        EventPredictionConfig(
            **config_data
        )
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    if not restore_trainer:
        return model

    trainer = EventPredictionTrainer(
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

def create_event_predictor(
    config: EventPredictionConfig | None = None,
) -> EventPredictionModel:
    return EventPredictionModel(
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
        "NEUROWORLD AI - EVENT PREDICTION"
    )
    print("=" * 65)

    config = EventPredictionConfig(
        state_dim=32,
        action_dim=4,
        hidden_dim=64,
        num_attention_heads=4,
        num_transformer_layers=2,
        feedforward_dim=128,
        max_sequence_length=32,
        prediction_horizons=(1, 3, 5),
        mc_dropout_samples=5,
    )

    model = EventPredictionModel(
        config
    )

    trainer = EventPredictionTrainer(
        model
    )

    batch_size = 16
    sequence_length = 12

    num_horizons = len(
        config.prediction_horizons
    )

    num_events = len(
        config.event_names
    )

    states = torch.randn(
        batch_size,
        sequence_length,
        config.state_dim,
    )

    actions = torch.randn(
        batch_size,
        sequence_length,
        config.action_dim,
    )

    # Synthetic labels for demonstration.
    labels = torch.randint(
        0,
        2,
        (
            batch_size,
            num_horizons,
            num_events,
        ),
    ).float()

    time_targets = torch.rand(
        batch_size,
        num_horizons,
        num_events,
    ) * 5.0

    training_batch = EventTrainingBatch(
        states=states,
        actions=actions,
        event_labels=labels,
        time_targets=time_targets,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for step in range(10):
        metrics = trainer.train_step(
            training_batch
        )

        if step in (0, 4, 9):
            print(
                f"Training step {step + 1}:",
                metrics,
            )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    output = model.predict(
        states[:2],
        actions[:2],
        estimate_uncertainty=True,
    )

    print(
        "Event probability shape:",
        output.probabilities.shape,
    )

    print(
        "Time-to-event shape:",
        output.time_to_event.shape,
    )

    print(
        "Uncertainty shape:",
        output.uncertainty.shape,
    )

    # --------------------------------------------------------
    # EVENT RISK ASSESSMENT
    # --------------------------------------------------------

    assessor = EventRiskAssessor(
        config
    )

    predicted_events = assessor.summarize(
        output,
        batch_index=0,
        threshold=0.5,
    )

    print(
        "Predicted events:",
        [
            event.to_dict()
            for event in predicted_events[:5]
        ],
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluation = trainer.evaluate(
        training_batch
    )

    print(
        "Evaluation:",
        evaluation,
    )

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    checkpoint = save_event_prediction_model(
        model,
        "models/event_prediction_demo.pt",
        trainer=trainer,
    )

    restored_model = (
        load_event_prediction_model(
            checkpoint
        )
    )

    print(
        "Restored model:",
        type(restored_model).__name__,
    )

    print(
        "Event prediction demonstration completed"
    )


if __name__ == "__main__":
    main()
