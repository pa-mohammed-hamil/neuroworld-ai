"""Temporal loss."""
import torch

def temporal_loss(prediction:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    if prediction.shape!=target.shape: raise ValueError('Shape mismatch')
    if prediction.numel()==0: raise ValueError('Empty tensors')
    return (prediction[:,1:]-prediction[:,:-1]-target[:,1:]+target[:,:-1]).abs().mean()

"""
NeuroWorld AI
File: training/losses/temporal.py

Temporal losses for world models, video prediction,
latent dynamics, physical AI and robotics.

Supported objectives:
- Temporal prediction
- First-order temporal consistency
- Second-order temporal consistency
- Velocity consistency
- Acceleration consistency
- Multi-step temporal rollout
- Temporal latent alignment
- Temporal smoothness
- Teacher-student temporal distillation
- Time-discounted prediction error

Tensor conventions:
    States:  [B, T, D]
    Videos:  [B, T, C, H, W]
    Latents: [B, T, D]

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


TemporalLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class TemporalLossConfig:
    prediction_weight: float = 1.0
    first_order_weight: float = 0.2
    second_order_weight: float = 0.1
    velocity_weight: float = 0.1
    acceleration_weight: float = 0.05
    rollout_weight: float = 0.5
    latent_weight: float = 0.2
    smoothness_weight: float = 0.0
    distillation_weight: float = 0.0

    loss_type: str = "mse"
    huber_delta: float = 1.0
    charbonnier_epsilon: float = 1e-3

    discount_factor: float = 0.99
    max_rollout_steps: int | None = None

    normalize_weights: bool = False
    detach_teacher: bool = True
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        if self.loss_type not in {
            "mse",
            "mae",
            "huber",
            "charbonnier",
        }:
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type}"
            )

        weight_names = (
            "prediction_weight",
            "first_order_weight",
            "second_order_weight",
            "velocity_weight",
            "acceleration_weight",
            "rollout_weight",
            "latent_weight",
            "smoothness_weight",
            "distillation_weight",
        )

        for name in weight_names:
            weight = getattr(self, name)

            if (
                not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and nonnegative"
                )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if self.charbonnier_epsilon <= 0:
            raise ValueError(
                "charbonnier_epsilon must be positive"
            )

        if not 0 < self.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in (0,1]"
            )

        if (
            self.max_rollout_steps is not None
            and self.max_rollout_steps < 1
        ):
            raise ValueError(
                "max_rollout_steps must be >= 1"
            )


class TemporalLoss(nn.Module):
    """
    Multi-objective temporal training loss.

    Expected outputs:
        predictions
        predicted_frames
        predicted_states
        predicted_velocities
        predicted_accelerations
        predicted_rollout
        predicted_latents
        student_predictions
        teacher_predictions

    Expected targets:
        targets
        target_frames
        target_states
        velocities
        accelerations
        rollout_targets
        latent_targets
        teacher_predictions
        mask
        sequence_mask

    Missing optional objectives are skipped.
    """

    def __init__(
        self,
        config: TemporalLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or TemporalLossConfig()
        )

        self.weights = {
            "prediction": (
                self.config.prediction_weight
            ),
            "first_order": (
                self.config.first_order_weight
            ),
            "second_order": (
                self.config.second_order_weight
            ),
            "velocity": (
                self.config.velocity_weight
            ),
            "acceleration": (
                self.config.acceleration_weight
            ),
            "rollout": (
                self.config.rollout_weight
            ),
            "latent": (
                self.config.latent_weight
            ),
            "smoothness": (
                self.config.smoothness_weight
            ),
            "distillation": (
                self.config.distillation_weight
            ),
        }

        self.custom_losses: dict[
            str, TemporalLossFunction
        ] = {}

        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _get(
        mapping: Mapping[str, Any],
        *names: str,
    ) -> Tensor | None:
        for name in names:
            value = mapping.get(name)

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
            "predicted_states",
            "predicted_next_states",
        )

    def _target(
        self,
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        return self._get(
            targets,
            "targets",
            "target_frames",
            "target_states",
            "next_states",
        )

    @staticmethod
    def _validate_pair(
        prediction: Tensor,
        target: Tensor,
    ) -> None:
        if prediction.shape != target.shape:
            raise ValueError(
                "Temporal tensor shapes differ: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = TemporalLoss._get(
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
                "Mask batch size mismatch"
            )

        if (
            reference.ndim == 5
            and mask.ndim == 4
            and mask.shape[1] == reference.shape[1]
            and mask.shape[-2:] == reference.shape[-2:]
        ):
            mask = mask.unsqueeze(2)

        elif (
            reference.ndim == 4
            and mask.ndim == 3
            and mask.shape[-2:] == reference.shape[-2:]
        ):
            mask = mask.unsqueeze(1)

        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                reference.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Temporal mask cannot broadcast "
                "to tensor shape"
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
        self._validate_pair(
            prediction,
            target,
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

    @staticmethod
    def _temporal_difference(
        values: Tensor,
        order: int = 1,
    ) -> Tensor:
        if values.ndim < 3:
            raise ValueError(
                "Expected sequence tensor [B,T,...]"
            )

        if order < 1:
            raise ValueError(
                "Difference order must be >= 1"
            )

        result = values

        for _ in range(order):
            if result.shape[1] < 2:
                raise ValueError(
                    "Insufficient timesteps "
                    "for temporal difference"
                )

            result = (
                result[:, 1:]
                - result[:, :-1]
            )

        return result

    @staticmethod
    def _difference_mask(
        mask: Tensor | None,
        order: int,
    ) -> Tensor | None:
        if mask is None:
            return None

        result = mask

        for _ in range(order):
            result = (
                result[:, 1:]
                & result[:, :-1]
            )

        return result

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

    def first_order_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._prediction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        self._validate_pair(
            prediction,
            target,
        )

        if (
            prediction.ndim < 3
            or prediction.shape[1] < 2
        ):
            return None

        predicted_difference = (
            self._temporal_difference(
                prediction,
                order=1,
            )
        )

        target_difference = (
            self._temporal_difference(
                target,
                order=1,
            )
        )

        mask = self._difference_mask(
            self._mask(targets, prediction),
            order=1,
        )

        return self._regression(
            predicted_difference,
            target_difference,
            mask,
        )

    def second_order_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._prediction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        self._validate_pair(
            prediction,
            target,
        )

        if (
            prediction.ndim < 3
            or prediction.shape[1] < 3
        ):
            return None

        predicted_difference = (
            self._temporal_difference(
                prediction,
                order=2,
            )
        )

        target_difference = (
            self._temporal_difference(
                target,
                order=2,
            )
        )

        mask = self._difference_mask(
            self._mask(targets, prediction),
            order=2,
        )

        return self._regression(
            predicted_difference,
            target_difference,
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

    def rollout_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
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

        self._validate_pair(
            prediction,
            target,
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

        discount = (
            self.config.discount_factor
            ** torch.arange(
                steps,
                device=errors.device,
                dtype=errors.dtype,
            )
        )

        discount = discount.reshape(
            1,
            steps,
            *([1] * (errors.ndim - 2)),
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

    def latent_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_latents",
            "latent_predictions",
        )

        target = self._get(
            targets,
            "latent_targets",
            "target_latents",
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

    def smoothness_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Penalizes rapid temporal changes.

        Optional regularizer; disable for
        trajectories with legitimate abrupt
        movements or scene transitions.
        """
        prediction = self._prediction(outputs)

        if prediction is None:
            return None

        if (
            prediction.ndim < 3
            or prediction.shape[1] < 2
        ):
            return None

        differences = self._temporal_difference(
            prediction,
            order=1,
        )

        mask = self._difference_mask(
            self._mask(targets, prediction),
            order=1,
        )

        return self._masked_mean(
            differences.float().square(),
            mask,
        )

    def distillation_loss(
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
        function: TemporalLossFunction,
        *,
        weight: float = 1.0,
    ) -> None:
        if not name:
            raise ValueError(
                "Custom loss name cannot be empty"
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
                f"Unknown temporal loss: {name}"
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
            "first_order": self.first_order_loss,
            "second_order": self.second_order_loss,
            "velocity": self.velocity_loss,
            "acceleration": self.acceleration_loss,
            "rollout": self.rollout_loss,
            "latent": self.latent_loss,
            "smoothness": self.smoothness_loss,
            "distillation": self.distillation_loss,
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
                "No active temporal loss components. "
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
                    f"Non-finite temporal loss: {name}"
                )

        normalizer = 1.0

        if self.config.normalize_weights:
            normalizer = sum(
                self.weights[name]
                for name in components
            )

        weighted = {
            name: (
                loss
                * self.weights[name]
                / normalizer
            )
            for name, loss in components.items()
        }

        total = torch.stack(
            list(weighted.values())
        ).sum()

        self.last_metrics = {
            "temporal_total": float(
                total.detach().item()
            ),
            **{
                f"temporal/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"temporal_weighted/{name}": float(
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
        return dict(self.last_metrics)

    def reset_metrics(self) -> None:
        self.last_metrics.clear()


def create_temporal_loss(
    *,
    prediction_weight: float = 1.0,
    first_order_weight: float = 0.2,
    second_order_weight: float = 0.1,
    velocity_weight: float = 0.1,
    acceleration_weight: float = 0.05,
    rollout_weight: float = 0.5,
    latent_weight: float = 0.2,
    smoothness_weight: float = 0.0,
    distillation_weight: float = 0.0,
    loss_type: str = "mse",
) -> TemporalLoss:
    return TemporalLoss(
        TemporalLossConfig(
            prediction_weight=prediction_weight,
            first_order_weight=first_order_weight,
            second_order_weight=second_order_weight,
            velocity_weight=velocity_weight,
            acceleration_weight=acceleration_weight,
            rollout_weight=rollout_weight,
            latent_weight=latent_weight,
            smoothness_weight=smoothness_weight,
            distillation_weight=distillation_weight,
            loss_type=loss_type,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 12
    state_dim = 128
    action_dim = 32

    class DemoTemporalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.network = nn.Sequential(
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
                nn.Linear(
                    256,
                    state_dim,
                ),
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

            prediction = self.network(
                features
            )

            return {
                "predictions": prediction,
                "predicted_rollout": prediction,
            }

    model = DemoTemporalModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = TemporalLoss(
        TemporalLossConfig(
            prediction_weight=1.0,
            first_order_weight=0.2,
            second_order_weight=0.1,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            rollout_weight=0.5,
            latent_weight=0.0,
            smoothness_weight=0.0,
            distillation_weight=0.0,
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

        outputs = model(
            states,
            actions,
        )

        targets = {
            "targets": target_states,
            "rollout_targets": target_states,
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
