
"""
NeuroWorld AI
File: training/losses/consistency.py

Consistency losses for:
- World models
- Video prediction
- Robot trajectory prediction
- Latent dynamics
- Action-conditioned models
- Physical AI
- Multi-step rollout training

Features:
- Temporal consistency
- State transition consistency
- Latent consistency
- Action consistency
- Cycle consistency
- Multi-step rollout consistency
- Teacher-student consistency
- Forward-backward consistency
- Masked sequence support
- MSE, MAE, Huber losses
- Weighted loss combination
- Custom loss registration
- NaN/Inf detection
- Per-component metrics

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


ConsistencyFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class ConsistencyLossConfig:
    temporal_weight: float = 1.0
    transition_weight: float = 1.0
    latent_weight: float = 0.5
    action_weight: float = 0.5
    cycle_weight: float = 0.5
    rollout_weight: float = 1.0
    teacher_weight: float = 0.5
    forward_backward_weight: float = 0.5

    loss_type: str = "mse"
    huber_delta: float = 1.0

    rollout_discount: float = 0.99
    max_rollout_steps: int | None = None

    normalize_weights: bool = False
    detach_teacher: bool = True
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        weights = (
            self.temporal_weight,
            self.transition_weight,
            self.latent_weight,
            self.action_weight,
            self.cycle_weight,
            self.rollout_weight,
            self.teacher_weight,
            self.forward_backward_weight,
        )

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "All weights must be finite "
                "and nonnegative"
            )

        if self.loss_type not in {
            "mse", "mae", "huber"
        }:
            raise ValueError(
                "loss_type must be mse, mae or huber"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if not 0 < self.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in (0, 1]"
            )

        if (
            self.max_rollout_steps is not None
            and self.max_rollout_steps < 1
        ):
            raise ValueError(
                "max_rollout_steps must be >= 1"
            )


class ConsistencyLoss(nn.Module):
    """
    Combined consistency loss.

    Expected dictionary keys:

    outputs:
        predicted_frames
        predicted_next_states
        latent_predictions
        latent_targets
        predicted_actions
        reconstructed_states
        rollout_states
        student_predictions
        teacher_predictions
        forward_states
        backward_reconstructed_states

    targets:
        target_frames
        states
        next_states
        actions
        latent_states
        rollout_targets
        mask
        frame_mask

    Only available components with
    positive weights are evaluated.
    """

    def __init__(
        self,
        config: ConsistencyLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or ConsistencyLossConfig()
        )

        self.weights = {
            "temporal": self.config.temporal_weight,
            "transition": self.config.transition_weight,
            "latent": self.config.latent_weight,
            "action": self.config.action_weight,
            "cycle": self.config.cycle_weight,
            "rollout": self.config.rollout_weight,
            "teacher": self.config.teacher_weight,
            "forward_backward": (
                self.config.forward_backward_weight
            ),
        }

        self.custom_losses: dict[
            str, ConsistencyFunction
        ] = {}

        self.last_metrics: dict[
            str, float
        ] = {}

    @staticmethod
    def _get(
        values: Mapping[str, Any],
        *names: str,
    ) -> Tensor | None:
        for name in names:
            value = values.get(name)

            if isinstance(value, Tensor):
                return value

        return None

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = ConsistencyLoss._get(
            targets,
            "mask",
            "frame_mask",
        )

        if mask is None:
            return None

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
                mask.bool(),
                reference.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Mask cannot broadcast "
                "to reference shape"
            ) from exc

    @staticmethod
    def _masked_mean(
        values: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if mask is None:
            return values.mean()

        mask = mask.to(
            device=values.device,
            dtype=values.dtype,
        )

        return (
            (values * mask).sum()
            / mask.sum().clamp_min(1.0)
        )

    def _elementwise_loss(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Consistency tensor shapes differ: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        prediction = prediction.float()
        target = target.float()

        if self.config.loss_type == "mse":
            return (
                prediction - target
            ).square()

        if self.config.loss_type == "mae":
            return (
                prediction - target
            ).abs()

        return F.huber_loss(
            prediction,
            target,
            delta=self.config.huber_delta,
            reduction="none",
        )

    def _regression(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        loss = self._elementwise_loss(
            prediction,
            target,
        )

        return self._masked_mean(
            loss,
            mask,
        )

    def temporal_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Compare frame-to-frame changes.

        Encourages predicted motion to
        match target motion.
        """
        prediction = self._get(
            outputs,
            "predicted_frames",
        )

        target = self._get(
            targets,
            "target_frames",
        )

        if prediction is None or target is None:
            return None

        if prediction.shape != target.shape:
            raise ValueError(
                "Temporal prediction/target mismatch"
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

    def transition_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Enforce agreement with observed
        next-state transitions.
        """
        prediction = self._get(
            outputs,
            "predicted_next_states",
            "next_states",
        )

        target = self._get(
            targets,
            "next_states",
        )

        if prediction is None or target is None:
            return None

        mask = self._mask(
            targets,
            prediction,
        )

        return self._regression(
            prediction,
            target,
            mask,
        )

    def latent_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Align predicted latent states
        with target latent representations.
        """
        prediction = self._get(
            outputs,
            "latent_predictions",
            "predicted_latents",
        )

        target = self._get(
            outputs,
            "latent_targets",
        )

        if target is None:
            target = self._get(
                targets,
                "latent_states",
                "latent_targets",
            )

        if prediction is None or target is None:
            return None

        mask = self._mask(
            targets,
            prediction,
        )

        return self._regression(
            prediction,
            target,
            mask,
        )

    def action_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Compare reconstructed or predicted
        continuous actions with target actions.
        """
        prediction = self._get(
            outputs,
            "predicted_actions",
            "reconstructed_actions",
        )

        target = self._get(
            targets,
            "actions",
        )

        if prediction is None or target is None:
            return None

        mask = self._mask(
            targets,
            prediction,
        )

        return self._regression(
            prediction,
            target,
            mask,
        )

    def cycle_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Compare cycle-reconstructed states
        against their original states.
        """
        reconstructed = self._get(
            outputs,
            "reconstructed_states",
            "cycle_reconstruction",
        )

        original = self._get(
            targets,
            "states",
        )

        if reconstructed is None or original is None:
            return None

        mask = self._mask(
            targets,
            reconstructed,
        )

        return self._regression(
            reconstructed,
            original,
            mask,
        )

    def rollout_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Discounted multi-step rollout loss.

        Expected:
            rollout_states: [B,T,D]
            rollout_targets: [B,T,D]
        """
        prediction = self._get(
            outputs,
            "rollout_states",
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
                "Rollout prediction/target mismatch"
            )

        if prediction.ndim < 3:
            raise ValueError(
                "Rollout tensors must have "
                "shape [B,T,...]"
            )

        steps = prediction.shape[1]

        if self.config.max_rollout_steps is not None:
            steps = min(
                steps,
                self.config.max_rollout_steps,
            )

        prediction = prediction[:, :steps]
        target = target[:, :steps]

        element_loss = self._elementwise_loss(
            prediction,
            target,
        )

        mask = self._mask(
            targets,
            prediction,
        )

        discounts = torch.pow(
            torch.tensor(
                self.config.rollout_discount,
                device=prediction.device,
                dtype=torch.float32,
            ),
            torch.arange(
                steps,
                device=prediction.device,
                dtype=torch.float32,
            ),
        )

        discount_shape = (
            1,
            steps,
            *([1] * (prediction.ndim - 2)),
        )

        discounts = discounts.reshape(
            discount_shape
        ).expand_as(element_loss)

        if mask is not None:
            discounts = (
                discounts
                * mask.to(discounts.dtype)
            )

        denominator = discounts.sum().clamp_min(
            1e-12
        )

        return (
            element_loss * discounts
        ).sum() / denominator

    def teacher_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Student-teacher prediction agreement.

        Teacher predictions are detached
        by default.
        """
        student = self._get(
            outputs,
            "student_predictions",
        )

        teacher = self._get(
            outputs,
            "teacher_predictions",
        )

        if student is None or teacher is None:
            return None

        if self.config.detach_teacher:
            teacher = teacher.detach()

        mask = self._mask(
            targets,
            student,
        )

        return self._regression(
            student,
            teacher,
            mask,
        )

    def forward_backward_consistency(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Check that backward reconstruction
        recovers the initial state after
        a forward transition.
        """
        reconstructed = self._get(
            outputs,
            "backward_reconstructed_states",
        )

        original = self._get(
            targets,
            "states",
        )

        if reconstructed is None or original is None:
            return None

        mask = self._mask(
            targets,
            reconstructed,
        )

        return self._regression(
            reconstructed,
            original,
            mask,
        )

    def register_loss(
        self,
        name: str,
        function: ConsistencyFunction,
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
                "Weight must be finite "
                "and nonnegative"
            )

        self.custom_losses[name] = function
        self.weights[name] = weight

    def set_weight(
        self,
        name: str,
        weight: float,
    ) -> None:
        if name not in self.weights:
            raise KeyError(
                f"Unknown loss: {name}"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Weight must be finite "
                "and nonnegative"
            )

        self.weights[name] = float(weight)

    @staticmethod
    def _reference_tensor(
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor:
        for values in (outputs, targets):
            for value in values.values():
                if isinstance(value, Tensor):
                    return value

        raise ValueError(
            "At least one Tensor is required"
        )

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
            "temporal": self.temporal_consistency,
            "transition": self.transition_consistency,
            "latent": self.latent_consistency,
            "action": self.action_consistency,
            "cycle": self.cycle_consistency,
            "rollout": self.rollout_consistency,
            "teacher": self.teacher_consistency,
            "forward_backward": (
                self.forward_backward_consistency
            ),
        }

        components: dict[
            str, Tensor
        ] = {}

        for name, function in functions.items():
            if self.weights[name] <= 0:
                continue

            value = function(
                outputs,
                targets,
            )

            if value is not None:
                components[name] = value

        for name, function in self.custom_losses.items():
            if self.weights[name] <= 0:
                continue

            value = function(
                outputs,
                targets,
            )

            if value is not None:
                components[name] = value

        if not components:
            raise ValueError(
                "No active consistency losses. "
                "Check tensor keys and weights."
            )

        for name, value in components.items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"{name} must return a Tensor"
                )

            if value.ndim != 0:
                raise ValueError(
                    f"{name} loss must be scalar"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(value).item()
            ):
                raise FloatingPointError(
                    f"Non-finite consistency loss: {name}"
                )

        active_weight_sum = sum(
            self.weights[name]
            for name in components
        )

        divisor = (
            active_weight_sum
            if self.config.normalize_weights
            else 1.0
        )

        reference = self._reference_tensor(
            outputs,
            targets,
        )

        total = reference.new_zeros(
            (),
            dtype=torch.float32,
        )

        weighted_components = {}

        for name, value in components.items():
            weighted = (
                value
                * self.weights[name]
                / divisor
            )

            weighted_components[name] = weighted

            total = total + weighted

        self.last_metrics = {
            "consistency_total": float(
                total.detach().item()
            ),
            **{
                f"consistency/{name}": float(
                    value.detach().item()
                )
                for name, value in components.items()
            },
            **{
                f"consistency_weighted/{name}": float(
                    value.detach().item()
                )
                for name, value
                in weighted_components.items()
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


def create_consistency_loss(
    *,
    temporal_weight: float = 1.0,
    transition_weight: float = 1.0,
    latent_weight: float = 0.5,
    action_weight: float = 0.5,
    cycle_weight: float = 0.5,
    rollout_weight: float = 1.0,
    teacher_weight: float = 0.5,
    forward_backward_weight: float = 0.5,
) -> ConsistencyLoss:
    return ConsistencyLoss(
        ConsistencyLossConfig(
            temporal_weight=temporal_weight,
            transition_weight=transition_weight,
            latent_weight=latent_weight,
            action_weight=action_weight,
            cycle_weight=cycle_weight,
            rollout_weight=rollout_weight,
            teacher_weight=teacher_weight,
            forward_backward_weight=(
                forward_backward_weight
            ),
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    sequence_length = 8
    state_dim = 128
    action_dim = 32
    latent_dim = 64

    model = nn.Sequential(
        nn.Linear(
            state_dim + action_dim,
            256,
        ),
        nn.ReLU(),
        nn.Linear(
            256,
            state_dim,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

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

    next_states = (
        states
        + 0.1 * torch.randn_like(states)
    )

    latent_targets = torch.randn(
        batch_size,
        sequence_length,
        latent_dim,
    )

    latent_projector = nn.Linear(
        state_dim,
        latent_dim,
    )

    optimizer.add_param_group(
        {
            "params": (
                latent_projector.parameters()
            )
        }
    )

    criterion = ConsistencyLoss(
        ConsistencyLossConfig(
            temporal_weight=0.0,
            transition_weight=1.0,
            latent_weight=0.5,
            action_weight=0.0,
            cycle_weight=0.0,
            rollout_weight=0.0,
            teacher_weight=0.0,
            forward_backward_weight=0.0,
        )
    )

    for step in range(5):
        optimizer.zero_grad(
            set_to_none=True
        )

        inputs = torch.cat(
            [states, actions],
            dim=-1,
        )

        predicted_states = model(
            inputs
        )

        predicted_latents = latent_projector(
            predicted_states
        )

        outputs = {
            "predicted_next_states": (
                predicted_states
            ),
            "latent_predictions": (
                predicted_latents
            ),
        }

        targets = {
            "next_states": next_states,
            "latent_states": latent_targets,
            "mask": torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
            ),
        }

        total_loss, components = criterion(
            outputs,
            targets,
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters())
            + list(latent_projector.parameters()),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )
