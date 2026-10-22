"""
NeuroWorld AI
File: training/losses/combined.py

Combined loss functions for:
- World models
- Dynamics prediction
- Video prediction
- Image reconstruction
- Reward prediction
- Policy learning
- Latent representation learning
- Physics-aware training
- Multi-task learning

Features:
- Weighted multi-objective loss
- Reconstruction loss
- Dynamics prediction loss
- Temporal consistency loss
- KL divergence loss
- Reward prediction loss
- Policy loss
- Physics consistency loss
- Contrastive loss
- Custom loss registration
- Masked sequence losses
- Loss normalization
- Dynamic loss weights
- Warmup-based loss scheduling
- NaN/Inf detection
- Per-component logging

Dependencies:
    torch >= 2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor,
]


@dataclass
class CombinedLossConfig:
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    temporal_weight: float = 0.2
    dynamics_weight: float = 1.0
    physics_weight: float = 0.1
    reward_weight: float = 0.5
    policy_weight: float = 0.2
    kl_weight: float = 0.01
    contrastive_weight: float = 0.1

    reconstruction_type: str = "mse"
    prediction_type: str = "mse"

    huber_delta: float = 1.0

    kl_free_nats: float = 0.0
    kl_warmup_steps: int = 1000

    normalize_weights: bool = False
    detect_nonfinite: bool = True

    contrastive_temperature: float = 0.07

    custom_weights: dict[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        weights = [
            self.reconstruction_weight,
            self.prediction_weight,
            self.temporal_weight,
            self.dynamics_weight,
            self.physics_weight,
            self.reward_weight,
            self.policy_weight,
            self.kl_weight,
            self.contrastive_weight,
            *self.custom_weights.values(),
        ]

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Loss weights must be finite and nonnegative"
            )

        supported = {
            "mse",
            "mae",
            "huber",
            "smooth_l1",
        }

        if self.reconstruction_type not in supported:
            raise ValueError(
                "Invalid reconstruction loss type"
            )

        if self.prediction_type not in supported:
            raise ValueError(
                "Invalid prediction loss type"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if self.kl_free_nats < 0:
            raise ValueError(
                "kl_free_nats must be >= 0"
            )

        if self.kl_warmup_steps < 0:
            raise ValueError(
                "kl_warmup_steps must be >= 0"
            )

        if self.contrastive_temperature <= 0:
            raise ValueError(
                "contrastive_temperature must be positive"
            )


class CombinedLoss(nn.Module):
    """
    Weighted multi-objective training loss.

    Supported output keys:

        reconstruction
        predictions
        predicted_frames
        next_states
        predicted_next_states
        dynamics_prediction
        predicted_rewards
        reward_predictions
        policy_logits
        policy_log_probs
        latent_mean
        latent_logvar
        posterior_mean
        posterior_logvar
        prior_mean
        prior_logvar
        embeddings_a
        embeddings_b
        physics_loss

    Supported target keys:

        frames
        images
        reconstruction_target
        target_frames
        next_states
        rewards
        actions
        policy_targets
        advantages
        states
        mask
        frame_mask
        physics_target

    Only losses with available inputs and
    positive weights are activated.

    Missing inputs do not generate fake losses.
    """

    COMPONENT_NAMES = (
        "reconstruction",
        "prediction",
        "temporal",
        "dynamics",
        "physics",
        "reward",
        "policy",
        "kl",
        "contrastive",
    )

    def __init__(
        self,
        config: CombinedLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or CombinedLossConfig()
        )

        self.step_count = 0

        self.weights = {
            "reconstruction": (
                self.config.reconstruction_weight
            ),
            "prediction": (
                self.config.prediction_weight
            ),
            "temporal": (
                self.config.temporal_weight
            ),
            "dynamics": (
                self.config.dynamics_weight
            ),
            "physics": (
                self.config.physics_weight
            ),
            "reward": (
                self.config.reward_weight
            ),
            "policy": (
                self.config.policy_weight
            ),
            "kl": self.config.kl_weight,
            "contrastive": (
                self.config.contrastive_weight
            ),
        }

        self.weights.update(
            self.config.custom_weights
        )

        self.custom_losses: dict[
            str, LossFunction
        ] = {}

        self.last_components: dict[
            str, float
        ] = {}

        self.last_weighted_components: dict[
            str, float
        ] = {}

    @staticmethod
    def _get_tensor(
        values: Mapping[str, Any],
        *names: str,
    ) -> Tensor | None:
        for name in names:
            value = values.get(name)

            if isinstance(value, Tensor):
                return value

        return None

    @staticmethod
    def _get_mask(
        targets: Mapping[str, Any],
        prediction: Tensor,
        *,
        mask_names: tuple[str, ...] = (
            "mask",
            "frame_mask",
        ),
    ) -> Tensor | None:
        mask = CombinedLoss._get_tensor(
            targets,
            *mask_names,
        )

        if mask is None:
            return None

        if mask.shape[0] != prediction.shape[0]:
            raise ValueError(
                "Mask batch size mismatch"
            )

        # Video datasets may supply a mask
        # covering context + future frames.
        # Select the prediction horizon.
        if (
            mask.ndim >= 2
            and prediction.ndim >= 3
            and mask.shape[1] != prediction.shape[1]
        ):
            if mask.shape[1] > prediction.shape[1]:
                mask = mask[
                    :,
                    -prediction.shape[1]:,
                ]
            else:
                raise ValueError(
                    "Mask sequence length is shorter "
                    "than prediction sequence length"
                )

        while mask.ndim < prediction.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask.bool(),
                prediction.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Mask cannot broadcast to prediction"
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

        denominator = mask.sum().clamp_min(1.0)

        return (
            (values * mask).sum()
            / denominator
        )

    def _regression_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        *,
        loss_type: str = "mse",
        mask: Tensor | None = None,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shape mismatch: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        prediction = prediction.float()
        target = target.float()

        if loss_type == "mse":
            element_loss = (
                prediction - target
            ).square()

        elif loss_type == "mae":
            element_loss = (
                prediction - target
            ).abs()

        elif loss_type == "huber":
            element_loss = F.huber_loss(
                prediction,
                target,
                delta=self.config.huber_delta,
                reduction="none",
            )

        elif loss_type == "smooth_l1":
            element_loss = F.smooth_l1_loss(
                prediction,
                target,
                beta=self.config.huber_delta,
                reduction="none",
            )

        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )

        return self._masked_mean(
            element_loss,
            mask,
        )

    def reconstruction_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get_tensor(
            outputs,
            "reconstruction",
            "reconstructed_frames",
            "reconstructed_images",
        )

        target = self._get_tensor(
            targets,
            "reconstruction_target",
            "frames",
            "images",
        )

        if prediction is None or target is None:
            return None

        mask = self._get_mask(
            targets,
            prediction,
        )

        return self._regression_loss(
            prediction,
            target,
            loss_type=self.config.reconstruction_type,
            mask=mask,
        )

    def prediction_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get_tensor(
            outputs,
            "predictions",
            "predicted_frames",
        )

        target = self._get_tensor(
            targets,
            "target_frames",
            "prediction_target",
        )

        if prediction is None or target is None:
            return None

        mask = self._get_mask(
            targets,
            prediction,
        )

        return self._regression_loss(
            prediction,
            target,
            loss_type=self.config.prediction_type,
            mask=mask,
        )

    def dynamics_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get_tensor(
            outputs,
            "dynamics_prediction",
            "predicted_next_states",
            "next_states",
        )

        target = self._get_tensor(
            targets,
            "next_states",
        )

        if prediction is None or target is None:
            return None

        mask = self._get_mask(
            targets,
            prediction,
            mask_names=("mask",),
        )

        return self._regression_loss(
            prediction,
            target,
            loss_type="mse",
            mask=mask,
        )

    def temporal_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get_tensor(
            outputs,
            "predicted_frames",
            "predictions",
        )

        target = self._get_tensor(
            targets,
            "target_frames",
        )

        if prediction is None or target is None:
            return None

        if prediction.shape != target.shape:
            raise ValueError(
                "Temporal prediction shape mismatch"
            )

        if prediction.ndim < 3:
            return None

        if prediction.shape[1] < 2:
            return None

        predicted_delta = (
            prediction[:, 1:].float()
            - prediction[:, :-1].float()
        )

        target_delta = (
            target[:, 1:].float()
            - target[:, :-1].float()
        )

        mask = self._get_mask(
            targets,
            prediction,
        )

        delta_mask = None

        if mask is not None:
            delta_mask = (
                mask[:, 1:]
                & mask[:, :-1]
            )

        return self._regression_loss(
            predicted_delta,
            target_delta,
            loss_type="mae",
            mask=delta_mask,
        )

    def reward_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get_tensor(
            outputs,
            "predicted_rewards",
            "reward_predictions",
        )

        target = self._get_tensor(
            targets,
            "rewards",
        )

        if prediction is None or target is None:
            return None

        if (
            prediction.ndim == target.ndim + 1
            and prediction.shape[-1] == 1
        ):
            prediction = prediction.squeeze(-1)

        mask = self._get_mask(
            targets,
            prediction,
            mask_names=("mask",),
        )

        return self._regression_loss(
            prediction,
            target,
            loss_type="mse",
            mask=mask,
        )

    def physics_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        explicit_loss = self._get_tensor(
            outputs,
            "physics_loss",
        )

        if explicit_loss is not None:
            if explicit_loss.ndim != 0:
                raise ValueError(
                    "physics_loss must be scalar"
                )

            return explicit_loss

        prediction = self._get_tensor(
            outputs,
            "physics_prediction",
        )

        target = self._get_tensor(
            targets,
            "physics_target",
        )

        if prediction is None or target is None:
            return None

        return self._regression_loss(
            prediction,
            target,
            loss_type="mse",
        )

    def policy_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Supports:
        - Discrete action cross-entropy
        - Policy-gradient objective using
          log probabilities and advantages

        PPO-specific clipping and value losses
        should be supplied as custom losses.
        """
        log_probs = self._get_tensor(
            outputs,
            "policy_log_probs",
        )

        advantages = self._get_tensor(
            targets,
            "advantages",
        )

        if (
            log_probs is not None
            and advantages is not None
        ):
            if log_probs.shape != advantages.shape:
                raise ValueError(
                    "Policy log-probability "
                    "and advantage shapes differ"
                )

            mask = self._get_mask(
                targets,
                log_probs,
                mask_names=("mask",),
            )

            return self._masked_mean(
                -log_probs.float()
                * advantages.float().detach(),
                mask,
            )

        logits = self._get_tensor(
            outputs,
            "policy_logits",
        )

        actions = self._get_tensor(
            targets,
            "policy_targets",
            "actions",
        )

        if logits is None or actions is None:
            return None

        if logits.ndim < 2:
            raise ValueError(
                "Policy logits must have action dimension"
            )

        if actions.shape != logits.shape[:-1]:
            raise ValueError(
                "Discrete action target shape mismatch"
            )

        if actions.is_floating_point():
            raise TypeError(
                "Discrete action labels must be integers"
            )

        losses = F.cross_entropy(
            logits.float().reshape(
                -1,
                logits.shape[-1],
            ),
            actions.long().reshape(-1),
            reduction="none",
        ).reshape(actions.shape)

        mask = self._get_mask(
            targets,
            losses,
            mask_names=("mask",),
        )

        return self._masked_mean(
            losses,
            mask,
        )

    def kl_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        KL(q || p) between diagonal Gaussians.

        If prior parameters are absent,
        use a standard normal prior.
        """
        mean = self._get_tensor(
            outputs,
            "posterior_mean",
            "latent_mean",
        )

        logvar = self._get_tensor(
            outputs,
            "posterior_logvar",
            "latent_logvar",
        )

        if mean is None or logvar is None:
            return None

        if mean.shape != logvar.shape:
            raise ValueError(
                "KL mean/logvar shape mismatch"
            )

        prior_mean = self._get_tensor(
            outputs,
            "prior_mean",
        )

        prior_logvar = self._get_tensor(
            outputs,
            "prior_logvar",
        )

        if (
            (prior_mean is None)
            != (prior_logvar is None)
        ):
            raise ValueError(
                "Provide both prior_mean "
                "and prior_logvar"
            )

        mean = mean.float()
        logvar = logvar.float()

        if prior_mean is None:
            prior_mean = torch.zeros_like(mean)
            prior_logvar = torch.zeros_like(logvar)

        else:
            if (
                prior_mean.shape != mean.shape
                or prior_logvar.shape != logvar.shape
            ):
                raise ValueError(
                    "Prior and posterior shapes differ"
                )

            prior_mean = prior_mean.float()
            prior_logvar = prior_logvar.float()

        variance_ratio = torch.exp(
            logvar - prior_logvar
        )

        mean_difference = (
            mean - prior_mean
        ).square() / torch.exp(prior_logvar)

        kl = 0.5 * (
            prior_logvar
            - logvar
            + variance_ratio
            + mean_difference
            - 1.0
        )

        # Sum across latent dimension.
        kl = kl.sum(dim=-1)

        if self.config.kl_free_nats > 0:
            kl = kl.clamp_min(
                self.config.kl_free_nats
            )

        mask = self._get_mask(
            targets,
            kl,
            mask_names=("mask",),
        )

        return self._masked_mean(
            kl,
            mask,
        )

    def contrastive_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Symmetric InfoNCE loss.

        Positive pairs must share the
        same batch index.

        Other examples in the batch
        are treated as negatives.
        """
        embeddings_a = self._get_tensor(
            outputs,
            "embeddings_a",
        )

        embeddings_b = self._get_tensor(
            outputs,
            "embeddings_b",
        )

        if (
            embeddings_a is None
            or embeddings_b is None
        ):
            return None

        if embeddings_a.shape != embeddings_b.shape:
            raise ValueError(
                "Contrastive embedding shapes differ"
            )

        if embeddings_a.ndim != 2:
            raise ValueError(
                "Contrastive embeddings must be [B,D]"
            )

        if embeddings_a.shape[0] < 2:
            return None

        a = F.normalize(
            embeddings_a.float(),
            dim=-1,
        )

        b = F.normalize(
            embeddings_b.float(),
            dim=-1,
        )

        logits = (
            a @ b.transpose(0, 1)
        ) / self.config.contrastive_temperature

        labels = torch.arange(
            logits.shape[0],
            device=logits.device,
        )

        loss_ab = F.cross_entropy(
            logits,
            labels,
        )

        loss_ba = F.cross_entropy(
            logits.transpose(0, 1),
            labels,
        )

        return 0.5 * (
            loss_ab + loss_ba
        )

    def register_loss(
        self,
        name: str,
        function: LossFunction,
        *,
        weight: float = 1.0,
    ) -> None:
        if not name:
            raise ValueError(
                "Loss name cannot be empty"
            )

        if name in self.COMPONENT_NAMES:
            raise ValueError(
                "Cannot override built-in loss"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Custom loss weight must be "
                "finite and nonnegative"
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

    def get_weight(
        self,
        name: str,
    ) -> float:
        return self.weights[name]

    def _effective_weight(
        self,
        name: str,
    ) -> float:
        weight = self.weights.get(
            name,
            0.0,
        )

        if name == "kl":
            warmup = (
                self.config.kl_warmup_steps
            )

            if warmup > 0:
                progress = min(
                    1.0,
                    self.step_count / warmup,
                )

                weight *= progress

        return weight

    def _reference_tensor(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor:
        for mapping in (outputs, targets):
            for value in mapping.values():
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
        step: int | None = None,
        return_components: bool = True,
    ) -> (
        Tensor
        | tuple[Tensor, dict[str, Tensor]]
    ):
        """
        Compute weighted combined loss.

        Args:
            outputs:
                Model prediction dictionary.

            targets:
                Training batch dictionary.

            step:
                Optional global training step
                for KL warmup scheduling.

            return_components:
                Return component losses
                alongside total loss.

        Returns:
            total_loss or
            (total_loss, component_losses)
        """
        if step is not None:
            if step < 0:
                raise ValueError(
                    "step must be nonnegative"
                )

            self.step_count = int(step)

        reference = self._reference_tensor(
            outputs,
            targets,
        )

        total_loss = reference.new_zeros(
            (),
            dtype=torch.float32,
        )

        functions = {
            "reconstruction": self.reconstruction_loss,
            "prediction": self.prediction_loss,
            "temporal": self.temporal_loss,
            "dynamics": self.dynamics_loss,
            "physics": self.physics_loss,
            "reward": self.reward_loss,
            "policy": self.policy_loss,
            "kl": self.kl_loss,
            "contrastive": self.contrastive_loss,
        }

        raw_components: dict[
            str, Tensor
        ] = {}

        effective_weights: dict[
            str, float
        ] = {}

        for name, function in functions.items():
            weight = self._effective_weight(
                name
            )

            if weight <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is None:
                continue

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
                    f"Non-finite {name} loss"
                )

            raw_components[name] = loss
            effective_weights[name] = weight

        for name, function in self.custom_losses.items():
            weight = self._effective_weight(
                name
            )

            if weight <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if not isinstance(loss, Tensor):
                raise TypeError(
                    f"Custom loss '{name}' "
                    "must return a Tensor"
                )

            if loss.ndim != 0:
                raise ValueError(
                    f"Custom loss '{name}' "
                    "must be scalar"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Non-finite custom loss: {name}"
                )

            raw_components[name] = loss
            effective_weights[name] = weight

        if not raw_components:
            raise ValueError(
                "No active loss components. "
                "Check output keys, target keys, "
                "and configured weights."
            )

        weight_normalizer = 1.0

        if self.config.normalize_weights:
            weight_normalizer = sum(
                effective_weights.values()
            )

            if weight_normalizer <= 0:
                raise ValueError(
                    "Total active loss weight is zero"
                )

        weighted_components = {}

        for name, loss in raw_components.items():
            weight = (
                effective_weights[name]
                / weight_normalizer
            )

            weighted_loss = (
                loss * weight
            )

            weighted_components[name] = (
                weighted_loss
            )

            total_loss = (
                total_loss + weighted_loss
            )

        self.last_components = {
            name: float(
                loss.detach().item()
            )
            for name, loss in raw_components.items()
        }

        self.last_weighted_components = {
            name: float(
                loss.detach().item()
            )
            for name, loss in weighted_components.items()
        }

        if step is None:
            self.step_count += 1

        if return_components:
            components = {
                "total": total_loss,
                **raw_components,
            }

            return total_loss, components

        return total_loss

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return {
            "total": sum(
                self.last_weighted_components.values()
            ),
            **{
                f"loss/{name}": value
                for name, value
                in self.last_components.items()
            },
            **{
                f"weighted/{name}": value
                for name, value
                in self.last_weighted_components.items()
            },
        }

    def reset(self) -> None:
        self.step_count = 0
        self.last_components.clear()
        self.last_weighted_components.clear()

    def get_extra_state(
        self,
    ) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "weights": dict(self.weights),
        }

    def set_extra_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.step_count = int(
            state.get(
                "step_count",
                0,
            )
        )

        restored_weights = state.get(
            "weights",
            {},
        )

        for name, weight in restored_weights.items():
            if name in self.weights:
                self.set_weight(
                    name,
                    float(weight),
                )


def create_combined_loss(
    *,
    reconstruction_weight: float = 1.0,
    prediction_weight: float = 1.0,
    dynamics_weight: float = 1.0,
    kl_weight: float = 0.01,
    temporal_weight: float = 0.2,
    physics_weight: float = 0.1,
    reward_weight: float = 0.5,
    policy_weight: float = 0.2,
    contrastive_weight: float = 0.1,
) -> CombinedLoss:
    return CombinedLoss(
        CombinedLossConfig(
            reconstruction_weight=(
                reconstruction_weight
            ),
            prediction_weight=(
                prediction_weight
            ),
            dynamics_weight=(
                dynamics_weight
            ),
            kl_weight=kl_weight,
            temporal_weight=temporal_weight,
            physics_weight=physics_weight,
            reward_weight=reward_weight,
            policy_weight=policy_weight,
            contrastive_weight=(
                contrastive_weight
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

    next_states = torch.randn(
        batch_size,
        sequence_length,
        state_dim,
    )

    inputs = torch.cat(
        [states, actions],
        dim=-1,
    )

    predictions = model(
        inputs
    )

    outputs = {
        "predicted_next_states": predictions,
        "latent_mean": torch.randn(
            batch_size,
            sequence_length,
            latent_dim,
            requires_grad=True,
        ),
        "latent_logvar": torch.zeros(
            batch_size,
            sequence_length,
            latent_dim,
            requires_grad=True,
        ),
    }

    targets = {
        "next_states": next_states,
        "mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
        ),
    }

    criterion = CombinedLoss(
        CombinedLossConfig(
            reconstruction_weight=0.0,
            prediction_weight=0.0,
            temporal_weight=0.0,
            dynamics_weight=1.0,
            physics_weight=0.0,
            reward_weight=0.0,
            policy_weight=0.0,
            kl_weight=0.01,
            contrastive_weight=0.0,
            kl_warmup_steps=100,
        )
    )

    for step in range(5):
        optimizer.zero_grad(
            set_to_none=True
        )

        outputs["predicted_next_states"] = (
            model(inputs)
        )

        total_loss, components = criterion(
            outputs,
            targets,
            step=step,
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