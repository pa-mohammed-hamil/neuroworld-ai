
"""
NeuroWorld AI
File: training/losses/prediction.py

Prediction losses for world models, video prediction,
robotics, and latent state forecasting.

Features:
- MSE / MAE / Huber / Charbonnier
- Gaussian negative log-likelihood
- Multi-step rollout prediction
- Temporal difference consistency
- Velocity and acceleration prediction
- Latent state prediction
- Action-conditioned prediction
- Discounted future prediction
- Optional teacher-target distillation
- Sequence and spatial masking
- Custom loss registration
- Weighted loss aggregation
- Training metrics

Requires:
    PyTorch >= 2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PredictionLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class PredictionLossConfig:
    prediction_weight: float = 1.0
    rollout_weight: float = 0.5
    temporal_weight: float = 0.2
    velocity_weight: float = 0.1
    acceleration_weight: float = 0.05
    latent_weight: float = 0.5
    action_weight: float = 0.2
    gaussian_nll_weight: float = 0.0
    teacher_weight: float = 0.0

    loss_type: str = "mse"

    huber_delta: float = 1.0
    charbonnier_epsilon: float = 1e-3

    rollout_discount: float = 0.99
    max_rollout_steps: int | None = None

    min_logvar: float = -20.0
    max_logvar: float = 10.0

    detach_teacher: bool = True
    normalize_weights: bool = False
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        if self.loss_type not in {
            "mse",
            "mae",
            "huber",
            "charbonnier",
        }:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}"
            )

        weights = (
            self.prediction_weight,
            self.rollout_weight,
            self.temporal_weight,
            self.velocity_weight,
            self.acceleration_weight,
            self.latent_weight,
            self.action_weight,
            self.gaussian_nll_weight,
            self.teacher_weight,
        )

        if any(
            not math.isfinite(w) or w < 0
            for w in weights
        ):
            raise ValueError(
                "All weights must be finite and nonnegative"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if self.charbonnier_epsilon <= 0:
            raise ValueError(
                "charbonnier_epsilon must be positive"
            )

        if not 0 < self.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in (0,1]"
            )

        if (
            self.max_rollout_steps is not None
            and self.max_rollout_steps < 1
        ):
            raise ValueError(
                "max_rollout_steps must be >= 1"
            )

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "min_logvar must be less than max_logvar"
            )


class PredictionLoss(nn.Module):
    """
    Configurable prediction objective.

    Supported outputs:
        predictions
        predicted_frames
        predicted_next_states
        predicted_rollout
        predicted_trajectory
        predicted_latents
        predicted_velocities
        predicted_accelerations
        action_conditioned_predictions
        prediction_mean
        prediction_logvar
        student_predictions
        teacher_predictions

    Supported targets:
        targets
        target_frames
        next_states
        rollout_targets
        trajectory_targets
        latent_targets
        velocities
        accelerations
        action_conditioned_targets
        teacher_predictions
        mask
        sequence_mask
        frame_mask

    Sequence convention:
        [B,T,...]

    The loss only evaluates components
    with positive weights and available data.
    """

    def __init__(
        self,
        config: PredictionLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or PredictionLossConfig()
        )

        self.weights = {
            "prediction": (
                self.config.prediction_weight
            ),
            "rollout": self.config.rollout_weight,
            "temporal": self.config.temporal_weight,
            "velocity": self.config.velocity_weight,
            "acceleration": (
                self.config.acceleration_weight
            ),
            "latent": self.config.latent_weight,
            "action": self.config.action_weight,
            "gaussian_nll": (
                self.config.gaussian_nll_weight
            ),
            "teacher": self.config.teacher_weight,
        }

        self.custom_losses: dict[
            str, PredictionLossFunction
        ] = {}

        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _get(
        values: Mapping[str, Any],
        *keys: str,
    ) -> Tensor | None:
        for key in keys:
            value = values.get(key)

            if isinstance(value, Tensor):
                return value

        return None

    def _prediction(
        self,
        outputs: Mapping[str, Any],
    ) -> Tensor | None:
        return self._get(
            outputs,
            "predictions",
            "predicted_frames",
            "predicted_next_states",
            "prediction_mean",
            "predicted_states",
        )

    def _target(
        self,
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        return self._get(
            targets,
            "targets",
            "target_frames",
            "next_states",
            "prediction_target",
            "frames",
        )

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = PredictionLoss._get(
            targets,
            "mask",
            "sequence_mask",
            "frame_mask",
        )

        if mask is None:
            return None

        mask = mask.to(
            device=reference.device,
            dtype=torch.bool,
        )

        if mask.ndim == 0:
            return mask.expand_as(reference)

        if mask.shape[0] != reference.shape[0]:
            raise ValueError(
                "Mask batch dimension mismatch"
            )

        if (
            mask.ndim >= 2
            and reference.ndim >= 3
            and mask.shape[1] != reference.shape[1]
        ):
            if mask.shape[1] > reference.shape[1]:
                mask = mask[
                    :,
                    -reference.shape[1]:,
                ]
            else:
                raise ValueError(
                    "Mask sequence length mismatch"
                )

        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                reference.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Mask cannot broadcast to prediction"
            ) from exc

    @staticmethod
    def _masked_mean(
        values: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is None:
            return values.mean()

        weights = mask.to(
            device=values.device,
            dtype=values.dtype,
        )

        return (
            (values * weights).sum()
            / weights.sum().clamp_min(1.0)
        )

    def _elementwise(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        prediction = prediction.float()
        target = target.float()

        difference = prediction - target

        if self.config.loss_type == "mse":
            return difference.square()

        if self.config.loss_type == "mae":
            return difference.abs()

        if self.config.loss_type == "huber":
            return F.huber_loss(
                prediction,
                target,
                delta=self.config.huber_delta,
                reduction="none",
            )

        epsilon = self.config.charbonnier_epsilon

        return torch.sqrt(
            difference.square() + epsilon**2
        ) - epsilon

    def _regression(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self._masked_mean(
            self._elementwise(
                prediction,
                target,
            ),
            mask,
        )

    def prediction_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._prediction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def rollout_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Discounted multi-step future prediction.

        Expected shape:
            [B,T,...]
        """
        prediction = self._get(
            outputs,
            "predicted_rollout",
            "rollout_predictions",
            "predicted_trajectory",
        )

        target = self._get(
            targets,
            "rollout_targets",
            "trajectory_targets",
        )

        if prediction is None or target is None:
            return None

        if prediction.shape != target.shape:
            raise ValueError(
                "Rollout shapes must match"
            )

        if prediction.ndim < 3:
            raise ValueError(
                "Rollout requires [B,T,...]"
            )

        steps = prediction.shape[1]

        if self.config.max_rollout_steps is not None:
            steps = min(
                steps,
                self.config.max_rollout_steps,
            )

        prediction = prediction[:, :steps]
        target = target[:, :steps]

        errors = self._elementwise(
            prediction,
            target,
        )

        discount = torch.pow(
            torch.tensor(
                self.config.rollout_discount,
                device=prediction.device,
                dtype=errors.dtype,
            ),
            torch.arange(
                steps,
                device=prediction.device,
                dtype=errors.dtype,
            ),
        )

        discount = discount.reshape(
            1,
            steps,
            *([1] * (prediction.ndim - 2)),
        ).expand_as(errors)

        mask = self._mask(
            targets,
            prediction,
        )

        if mask is not None:
            discount = (
                discount
                * mask.to(discount.dtype)
            )

        return (
            (errors * discount).sum()
            / discount.sum().clamp_min(1e-12)
        )

    def temporal_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Match changes between consecutive
        predicted and target states/frames.
        """
        prediction = self._prediction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        if prediction.shape != target.shape:
            raise ValueError(
                "Temporal prediction shapes differ"
            )

        if (
            prediction.ndim < 3
            or prediction.shape[1] < 2
        ):
            return None

        predicted_delta = (
            prediction[:, 1:]
            - prediction[:, :-1]
        )

        target_delta = (
            target[:, 1:]
            - target[:, :-1]
        )

        mask = self._mask(
            targets,
            prediction,
        )

        if mask is not None:
            mask = (
                mask[:, 1:]
                & mask[:, :-1]
            )

        return self._regression(
            predicted_delta,
            target_delta,
            mask,
        )

    def velocity_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_velocities",
            "velocity_predictions",
        )

        target = self._get(
            targets,
            "velocities",
            "velocity_targets",
        )

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def acceleration_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_accelerations",
            "acceleration_predictions",
        )

        target = self._get(
            targets,
            "accelerations",
            "acceleration_targets",
        )

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def latent_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_latents",
            "latent_predictions",
            "predicted_next_latents",
        )

        target = self._get(
            targets,
            "latent_targets",
            "next_latents",
        )

        if target is None:
            target = self._get(
                outputs,
                "target_latents",
            )

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def action_conditioned_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Evaluate future predictions conditioned
        on supplied control actions.
        """
        prediction = self._get(
            outputs,
            "action_conditioned_predictions",
            "action_conditioned_states",
        )

        target = self._get(
            targets,
            "action_conditioned_targets",
            "next_states",
        )

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def gaussian_nll_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Gaussian negative log-likelihood
        for probabilistic predictions.

        Outputs:
            prediction_mean
            prediction_logvar
        """
        mean = self._get(
            outputs,
            "prediction_mean",
            "predicted_mean",
        )

        logvar = self._get(
            outputs,
            "prediction_logvar",
            "predicted_logvar",
        )

        target = self._target(targets)

        if (
            mean is None
            or logvar is None
            or target is None
        ):
            return None

        if not (
            mean.shape == logvar.shape == target.shape
        ):
            raise ValueError(
                "Gaussian prediction shapes differ"
            )

        mean = mean.float()
        target = target.float()

        logvar = logvar.float().clamp(
            min=self.config.min_logvar,
            max=self.config.max_logvar,
        )

        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (target - mean).square()
            * torch.exp(-logvar)
        )

        return self._masked_mean(
            nll,
            self._mask(targets, mean),
        )

    def teacher_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        student = self._get(
            outputs,
            "student_predictions",
        )

        teacher = self._get(
            targets,
            "teacher_predictions",
        )

        if teacher is None:
            teacher = self._get(
                outputs,
                "teacher_predictions",
            )

        if student is None or teacher is None:
            return None

        if self.config.detach_teacher:
            teacher = teacher.detach()

        return self._regression(
            student,
            teacher,
            self._mask(targets, student),
        )

    def register_loss(
        self,
        name: str,
        function: PredictionLossFunction,
        *,
        weight: float = 1.0,
    ) -> None:
        if not name:
            raise ValueError(
                "Loss name cannot be empty"
            )

        if name in self.weights:
            raise ValueError(
                f"Loss '{name}' already exists"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Weight must be finite and nonnegative"
            )

        self.custom_losses[name] = function
        self.weights[name] = float(weight)

    def set_weight(
        self,
        name: str,
        weight: float,
    ) -> None:
        if name not in self.weights:
            raise KeyError(
                f"Unknown prediction loss: {name}"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Weight must be finite and nonnegative"
            )

        self.weights[name] = float(weight)

    def get_weight(
        self,
        name: str,
    ) -> float:
        return self.weights[name]

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        return_components: bool = True,
    ) -> (
        Tensor
        | tuple[Tensor, dict[str, Tensor]]
    ):
        functions = {
            "prediction": self.prediction_loss,
            "rollout": self.rollout_loss,
            "temporal": self.temporal_loss,
            "velocity": self.velocity_loss,
            "acceleration": self.acceleration_loss,
            "latent": self.latent_loss,
            "action": self.action_conditioned_loss,
            "gaussian_nll": self.gaussian_nll_loss,
            "teacher": self.teacher_loss,
        }

        components: dict[str, Tensor] = {}

        for name, function in functions.items():
            if self.weights[name] <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is not None:
                components[name] = loss

        for name, function in self.custom_losses.items():
            if self.weights[name] <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is not None:
                components[name] = loss

        if not components:
            raise ValueError(
                "No active prediction loss components. "
                "Check outputs, targets and weights."
            )

        for name, loss in components.items():
            if not isinstance(loss, Tensor):
                raise TypeError(
                    f"{name} must return a Tensor"
                )

            if loss.ndim != 0:
                raise ValueError(
                    f"{name} must return a scalar loss"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Non-finite prediction loss: {name}"
                )

        normalizer = 1.0

        if self.config.normalize_weights:
            normalizer = sum(
                self.weights[name]
                for name in components
            )

        weighted = {
            name: (
                loss * self.weights[name]
                / normalizer
            )
            for name, loss in components.items()
        }

        total = torch.stack(
            list(weighted.values())
        ).sum()

        self.last_metrics = {
            "prediction_total": float(
                total.detach().item()
            ),
            **{
                f"prediction/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"prediction_weighted/{name}": float(
                    loss.detach().item()
                )
                for name, loss in weighted.items()
            },
        }

        if return_components:
            return total, {
                "total": total,
                **components,
            }

        return total

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return dict(
            self.last_metrics
        )

    def reset_metrics(self) -> None:
        self.last_metrics.clear()


def create_prediction_loss(
    *,
    prediction_weight: float = 1.0,
    rollout_weight: float = 0.5,
    temporal_weight: float = 0.2,
    velocity_weight: float = 0.1,
    acceleration_weight: float = 0.05,
    latent_weight: float = 0.5,
    action_weight: float = 0.2,
    gaussian_nll_weight: float = 0.0,
    teacher_weight: float = 0.0,
    loss_type: str = "mse",
) -> PredictionLoss:
    return PredictionLoss(
        PredictionLossConfig(
            prediction_weight=prediction_weight,
            rollout_weight=rollout_weight,
            temporal_weight=temporal_weight,
            velocity_weight=velocity_weight,
            acceleration_weight=acceleration_weight,
            latent_weight=latent_weight,
            action_weight=action_weight,
            gaussian_nll_weight=gaussian_nll_weight,
            teacher_weight=teacher_weight,
            loss_type=loss_type,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 12
    state_dim = 128
    action_dim = 32
    latent_dim = 64

    class DemoPredictionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(
                    state_dim + action_dim,
                    256,
                ),
                nn.ReLU(),
                nn.Linear(
                    256,
                    256,
                ),
                nn.ReLU(),
            )

            self.state_head = nn.Linear(
                256,
                state_dim,
            )

            self.latent_head = nn.Linear(
                256,
                latent_dim,
            )

        def forward(
            self,
            states: Tensor,
            actions: Tensor,
        ) -> dict[str, Tensor]:
            features = torch.cat(
                [states, actions],
                dim=-1,
            )

            hidden = self.encoder(features)

            return {
                "predictions": self.state_head(
                    hidden
                ),
                "predicted_latents": (
                    self.latent_head(hidden)
                ),
            }

    model = DemoPredictionModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = PredictionLoss(
        PredictionLossConfig(
            prediction_weight=1.0,
            rollout_weight=0.0,
            temporal_weight=0.2,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            latent_weight=0.5,
            action_weight=0.0,
            gaussian_nll_weight=0.0,
            teacher_weight=0.0,
        )
    )

    for step in range(5):
        states = torch.randn(
            batch_size,
            sequence_length,
            state_dim,
        )

        actions = torch.randn(
            batch_size,
            sequence_length,
            action_dim,
        )

        target_states = (
            states
            + 0.05 * torch.randn_like(states)
        )

        latent_targets = torch.randn(
            batch_size,
            sequence_length,
            latent_dim,
        )

        outputs = model(
            states,
            actions,
        )

        targets = {
            "next_states": target_states,
            "latent_targets": latent_targets,
            "mask": torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
            ),
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss, components = criterion(
            outputs,
            targets,
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )
