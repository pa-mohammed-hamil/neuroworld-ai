
"""
NeuroWorld AI
File: training/losses/dynamics.py

Dynamics losses for world-model and robotics training.

Features:
- Deterministic state-transition loss
- Gaussian negative log-likelihood
- Latent dynamics consistency
- Multi-step rollout loss
- Action-conditioned dynamics loss
- Velocity consistency
- Acceleration consistency
- Energy conservation loss
- Forward/backward cycle consistency
- Uncertainty calibration
- Masked sequence losses
- Weighted multi-objective training
- Custom loss registration
- Loss component logging

Requires:
    torch >= 2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DynamicsLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class DynamicsLossConfig:
    transition_weight: float = 1.0
    gaussian_nll_weight: float = 0.0
    latent_weight: float = 0.5
    rollout_weight: float = 1.0
    action_weight: float = 0.2
    velocity_weight: float = 0.2
    acceleration_weight: float = 0.1
    energy_weight: float = 0.05
    cycle_weight: float = 0.2
    uncertainty_weight: float = 0.1

    loss_type: str = "mse"
    huber_delta: float = 1.0

    rollout_discount: float = 0.99
    max_rollout_steps: int | None = None

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    normalize_weights: bool = False
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        weights = (
            self.transition_weight,
            self.gaussian_nll_weight,
            self.latent_weight,
            self.rollout_weight,
            self.action_weight,
            self.velocity_weight,
            self.acceleration_weight,
            self.energy_weight,
            self.cycle_weight,
            self.uncertainty_weight,
        )

        if any(
            not math.isfinite(w) or w < 0
            for w in weights
        ):
            raise ValueError(
                "All loss weights must be "
                "finite and nonnegative"
            )

        if self.loss_type not in {
            "mse",
            "mae",
            "huber",
        }:
            raise ValueError(
                "Unsupported loss_type"
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

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "min_logvar must be below max_logvar"
            )


class DynamicsLoss(nn.Module):
    """
    Multi-objective dynamics training loss.

    Expected outputs:

        predicted_next_states
        next_state_mean
        next_state_logvar
        predicted_latents
        predicted_rollout
        action_conditioned_states
        predicted_velocities
        predicted_accelerations
        predicted_energy
        reconstructed_states
        predicted_uncertainty

    Expected targets:

        next_states
        latent_targets
        rollout_targets
        action_conditioned_targets
        velocities
        accelerations
        energy
        states
        mask

    Only components with available tensors
    and positive weights are evaluated.

    Missing optional components are skipped.
    """

    def __init__(
        self,
        config: DynamicsLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or DynamicsLossConfig()
        )

        self.weights = {
            "transition": (
                self.config.transition_weight
            ),
            "gaussian_nll": (
                self.config.gaussian_nll_weight
            ),
            "latent": self.config.latent_weight,
            "rollout": self.config.rollout_weight,
            "action": self.config.action_weight,
            "velocity": self.config.velocity_weight,
            "acceleration": (
                self.config.acceleration_weight
            ),
            "energy": self.config.energy_weight,
            "cycle": self.config.cycle_weight,
            "uncertainty": (
                self.config.uncertainty_weight
            ),
        }

        self.custom_losses: dict[
            str, DynamicsLossFunction
        ] = {}

        self.last_metrics: dict[
            str, float
        ] = {}

    @staticmethod
    def _get(
        mapping: Mapping[str, Any],
        *keys: str,
    ) -> Tensor | None:
        for key in keys:
            value = mapping.get(key)

            if isinstance(value, Tensor):
                return value

        return None

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = DynamicsLoss._get(
            targets,
            "mask",
            "sequence_mask",
        )

        if mask is None:
            return None

        if mask.shape[0] != reference.shape[0]:
            raise ValueError(
                "Mask batch size mismatch"
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
                "Mask cannot broadcast to tensor"
            ) from exc

    @staticmethod
    def _masked_mean(
        values: Tensor,
        mask: Tensor | None,
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
        return self._masked_mean(
            self._elementwise(
                prediction,
                target,
            ),
            mask,
        )

    def transition_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_next_states",
            "next_states",
            "next_state_mean",
            "dynamics_prediction",
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

    def gaussian_nll_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Gaussian negative log-likelihood
        for probabilistic state transitions.

        Model predicts:
            mean:   [B, ..., D]
            logvar: [B, ..., D]
        """
        mean = self._get(
            outputs,
            "next_state_mean",
            "predicted_mean",
        )

        logvar = self._get(
            outputs,
            "next_state_logvar",
            "predicted_logvar",
        )

        target = self._get(
            targets,
            "next_states",
        )

        if (
            mean is None
            or logvar is None
            or target is None
        ):
            return None

        if (
            mean.shape != logvar.shape
            or mean.shape != target.shape
        ):
            raise ValueError(
                "Gaussian dynamics tensor "
                "shapes must match"
            )

        mean = mean.float()
        target = target.float()

        logvar = logvar.float().clamp(
            min=self.config.min_logvar,
            max=self.config.max_logvar,
        )

        variance = torch.exp(
            logvar
        )

        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (target - mean).square()
            / variance
        )

        mask = self._mask(
            targets,
            mean,
        )

        return self._masked_mean(
            nll,
            mask,
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
            "next_latents",
        )

        if target is None:
            target = self._get(
                outputs,
                "target_latents",
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

    def rollout_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Discounted multi-step dynamics loss.

        Expected shape:
            [batch, time, state_dim]
        """
        prediction = self._get(
            outputs,
            "predicted_rollout",
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
                "Rollout shapes must match"
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

        element_loss = self._elementwise(
            prediction,
            target,
        )

        discount = torch.pow(
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

        shape = (
            1,
            steps,
            *([1] * (prediction.ndim - 2)),
        )

        discount = discount.reshape(
            shape
        ).expand_as(element_loss)

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
            element_loss * discount
        ).sum() / discount.sum().clamp_min(
            1e-12
        )

    def action_conditioned_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Evaluate transitions generated
        under explicit action conditioning.
        """
        prediction = self._get(
            outputs,
            "action_conditioned_states",
            "action_conditioned_prediction",
        )

        target = self._get(
            targets,
            "action_conditioned_targets",
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

    def velocity_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_velocities",
            "predicted_velocity",
        )

        target = self._get(
            targets,
            "velocities",
            "velocity_targets",
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

    def acceleration_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_accelerations",
            "predicted_acceleration",
        )

        target = self._get(
            targets,
            "accelerations",
            "acceleration_targets",
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

    def energy_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Compare predicted and reference energy.

        The caller is responsible for computing
        energy using an appropriate physical
        model and consistent units.
        """
        prediction = self._get(
            outputs,
            "predicted_energy",
        )

        target = self._get(
            targets,
            "energy",
            "energy_targets",
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

    def cycle_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Forward-backward cycle consistency.

        A backward dynamics model should
        reconstruct the original state.
        """
        reconstruction = self._get(
            outputs,
            "reconstructed_states",
            "backward_reconstructed_states",
        )

        original = self._get(
            targets,
            "states",
        )

        if reconstruction is None or original is None:
            return None

        mask = self._mask(
            targets,
            reconstruction,
        )

        return self._regression(
            reconstruction,
            original,
            mask,
        )

    def uncertainty_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Calibrate predicted uncertainty
        against observed squared error.

        predicted_uncertainty represents
        variance, not standard deviation.

        The target squared error is detached
        to avoid changing the mean predictor
        through this auxiliary objective.
        """
        uncertainty = self._get(
            outputs,
            "predicted_uncertainty",
            "predicted_variance",
        )

        prediction = self._get(
            outputs,
            "predicted_next_states",
            "next_state_mean",
        )

        target = self._get(
            targets,
            "next_states",
        )

        if (
            uncertainty is None
            or prediction is None
            or target is None
        ):
            return None

        if (
            uncertainty.shape != prediction.shape
            or prediction.shape != target.shape
        ):
            raise ValueError(
                "Uncertainty tensor shapes differ"
            )

        squared_error = (
            prediction.float()
            - target.float()
        ).square().detach()

        uncertainty = uncertainty.float()

        mask = self._mask(
            targets,
            uncertainty,
        )

        return self._regression(
            uncertainty,
            squared_error,
            mask,
        )

    def register_loss(
        self,
        name: str,
        function: DynamicsLossFunction,
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
        self.weights[name] = float(weight)

    def set_weight(
        self,
        name: str,
        weight: float,
    ) -> None:
        if name not in self.weights:
            raise KeyError(
                f"Unknown dynamics loss: {name}"
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
        """
        Calculate weighted dynamics loss.

        Returns:
            total_loss

        Or:
            total_loss, component_losses
        """
        functions = {
            "transition": self.transition_loss,
            "gaussian_nll": self.gaussian_nll_loss,
            "latent": self.latent_loss,
            "rollout": self.rollout_loss,
            "action": self.action_conditioned_loss,
            "velocity": self.velocity_loss,
            "acceleration": self.acceleration_loss,
            "energy": self.energy_loss,
            "cycle": self.cycle_loss,
            "uncertainty": self.uncertainty_loss,
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
                "No active dynamics loss components. "
                "Check model outputs, targets, "
                "and configured weights."
            )

        for name, loss in components.items():
            if not isinstance(loss, Tensor):
                raise TypeError(
                    f"{name} must return a Tensor"
                )

            if loss.ndim != 0:
                raise ValueError(
                    f"{name} must return a scalar"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Non-finite dynamics loss: {name}"
                )

        weight_sum = sum(
            self.weights[name]
            for name in components
        )

        divisor = (
            weight_sum
            if self.config.normalize_weights
            else 1.0
        )

        weighted = {
            name: (
                loss * self.weights[name] / divisor
            )
            for name, loss in components.items()
        }

        total = torch.stack(
            list(weighted.values())
        ).sum()

        self.last_metrics = {
            "dynamics_total": float(
                total.detach().item()
            ),
            **{
                f"dynamics/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"dynamics_weighted/{name}": float(
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


def create_dynamics_loss(
    *,
    transition_weight: float = 1.0,
    gaussian_nll_weight: float = 0.0,
    latent_weight: float = 0.5,
    rollout_weight: float = 1.0,
    action_weight: float = 0.2,
    velocity_weight: float = 0.2,
    acceleration_weight: float = 0.1,
    energy_weight: float = 0.05,
    cycle_weight: float = 0.2,
    uncertainty_weight: float = 0.1,
) -> DynamicsLoss:
    return DynamicsLoss(
        DynamicsLossConfig(
            transition_weight=transition_weight,
            gaussian_nll_weight=gaussian_nll_weight,
            latent_weight=latent_weight,
            rollout_weight=rollout_weight,
            action_weight=action_weight,
            velocity_weight=velocity_weight,
            acceleration_weight=acceleration_weight,
            energy_weight=energy_weight,
            cycle_weight=cycle_weight,
            uncertainty_weight=uncertainty_weight,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 10
    state_dim = 128
    action_dim = 32
    latent_dim = 64

    class DemoDynamicsModel(nn.Module):
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
            inputs = torch.cat(
                [states, actions],
                dim=-1,
            )

            hidden = self.encoder(inputs)

            return {
                "predicted_next_states": (
                    self.state_head(hidden)
                ),
                "predicted_latents": (
                    self.latent_head(hidden)
                ),
            }

    model = DemoDynamicsModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = DynamicsLoss(
        DynamicsLossConfig(
            transition_weight=1.0,
            gaussian_nll_weight=0.0,
            latent_weight=0.5,
            rollout_weight=0.0,
            action_weight=0.0,
            velocity_weight=0.0,
            acceleration_weight=0.0,
            energy_weight=0.0,
            cycle_weight=0.0,
            uncertainty_weight=0.0,
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

        next_states = (
            states
            + 0.1 * torch.randn_like(states)
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
            "next_states": next_states,
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