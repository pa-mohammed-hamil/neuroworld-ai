
"""
NeuroWorld AI
File: training/losses/reward.py

Reward-model training losses for:
- World models and imagined trajectories
- Model-based reinforcement learning
- Autonomous robotics
- Physical AI and action planning
- Reward prediction and preference learning

Supported objectives:
- Reward regression (MSE / MAE / Huber)
- Gaussian reward negative log-likelihood
- Binary reward classification
- Categorical reward classification
- Pairwise preference learning (Bradley-Terry)
- Reward ranking with margin
- Discounted return prediction
- Temporal reward consistency
- Reward distillation
- Reward smoothness
- Reward uncertainty calibration

Features:
- Sequence masks
- Variable-length trajectories
- Configurable objective weights
- Custom loss registration
- Non-finite loss detection
- Per-component metrics
- Differentiable PyTorch objectives

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


RewardLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class RewardLossConfig:
    regression_weight: float = 1.0
    gaussian_nll_weight: float = 0.0
    binary_weight: float = 0.0
    categorical_weight: float = 0.0
    preference_weight: float = 0.0
    ranking_weight: float = 0.0
    return_weight: float = 0.0
    temporal_weight: float = 0.0
    distillation_weight: float = 0.0
    smoothness_weight: float = 0.0
    uncertainty_weight: float = 0.0

    loss_type: str = "huber"
    huber_delta: float = 1.0

    gamma: float = 0.99
    preference_temperature: float = 1.0
    ranking_margin: float = 0.1

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
        }:
            raise ValueError(
                f"Unsupported loss_type: {self.loss_type}"
            )

        weights = (
            self.regression_weight,
            self.gaussian_nll_weight,
            self.binary_weight,
            self.categorical_weight,
            self.preference_weight,
            self.ranking_weight,
            self.return_weight,
            self.temporal_weight,
            self.distillation_weight,
            self.smoothness_weight,
            self.uncertainty_weight,
        )

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Loss weights must be finite "
                "and nonnegative"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma must be in [0,1]"
            )

        if self.preference_temperature <= 0:
            raise ValueError(
                "preference_temperature must be positive"
            )

        if self.ranking_margin < 0:
            raise ValueError(
                "ranking_margin must be nonnegative"
            )

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "min_logvar must be below max_logvar"
            )


def discounted_returns(
    rewards: Tensor,
    *,
    gamma: float = 0.99,
    dones: Tensor | None = None,
    mask: Tensor | None = None,
    bootstrap_value: Tensor | None = None,
) -> Tensor:
    """
    Compute discounted returns.

    Args:
        rewards: [B,T]
        dones: [B,T] true episode termination
        mask: [B,T] valid timesteps
        bootstrap_value: [B] value after final step

    Returns:
        [B,T] discounted returns

    Time-limit truncation should be handled
    through bootstrapping, not by marking
    the transition as terminal.
    """
    if rewards.ndim != 2:
        raise ValueError(
            "rewards must have shape [B,T]"
        )

    if not 0 <= gamma <= 1:
        raise ValueError(
            "gamma must be in [0,1]"
        )

    rewards = rewards.float()
    batch_size, steps = rewards.shape

    if dones is None:
        dones = torch.zeros_like(
            rewards,
            dtype=torch.bool,
        )
    else:
        if dones.shape != rewards.shape:
            raise ValueError(
                "dones must match rewards"
            )
        dones = dones.bool()

    if mask is not None:
        if mask.shape != rewards.shape:
            raise ValueError(
                "mask must match rewards"
            )
        mask = mask.bool()

    if bootstrap_value is None:
        running = torch.zeros(
            batch_size,
            device=rewards.device,
            dtype=rewards.dtype,
        )
    else:
        if bootstrap_value.shape != (batch_size,):
            raise ValueError(
                "bootstrap_value must have shape [B]"
            )
        running = bootstrap_value.float()

    result = torch.zeros_like(rewards)

    for t in reversed(range(steps)):
        running = (
            rewards[:, t]
            + gamma
            * (~dones[:, t]).to(rewards.dtype)
            * running
        )

        if mask is not None:
            running = torch.where(
                mask[:, t],
                running,
                torch.zeros_like(running),
            )

        result[:, t] = running

    return result


class RewardLoss(nn.Module):
    """
    Multi-objective reward-model loss.

    Expected outputs:
        predicted_rewards
        reward_mean
        reward_logvar
        binary_reward_logits
        reward_class_logits
        preferred_scores
        rejected_scores
        predicted_returns
        student_rewards
        teacher_rewards
        predicted_reward_variance

    Expected targets:
        rewards
        reward_targets
        binary_labels
        reward_classes
        preference_labels
        returns
        teacher_rewards
        mask
        sequence_mask

    Preference labels:
        1.0 = first trajectory preferred
        0.0 = second trajectory preferred
        0.5 = equal preference

    Missing optional objectives are skipped.
    """

    def __init__(
        self,
        config: RewardLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or RewardLossConfig()
        )

        self.weights = {
            "regression": (
                self.config.regression_weight
            ),
            "gaussian_nll": (
                self.config.gaussian_nll_weight
            ),
            "binary": self.config.binary_weight,
            "categorical": (
                self.config.categorical_weight
            ),
            "preference": (
                self.config.preference_weight
            ),
            "ranking": self.config.ranking_weight,
            "return": self.config.return_weight,
            "temporal": self.config.temporal_weight,
            "distillation": (
                self.config.distillation_weight
            ),
            "smoothness": (
                self.config.smoothness_weight
            ),
            "uncertainty": (
                self.config.uncertainty_weight
            ),
        }

        self.custom_losses: dict[
            str, RewardLossFunction
        ] = {}

        self.last_metrics: dict[str, float] = {}

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
    def _squeeze_reward(
        value: Tensor,
    ) -> Tensor:
        if (
            value.ndim >= 2
            and value.shape[-1] == 1
        ):
            return value.squeeze(-1)
        return value

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = RewardLoss._get(
            targets,
            "mask",
            "sequence_mask",
        )

        if mask is None:
            return None

        mask = mask.to(
            device=reference.device,
            dtype=torch.bool,
        )

        if (
            mask.ndim == reference.ndim + 1
            and mask.shape[-1] == 1
        ):
            mask = mask.squeeze(-1)

        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                reference.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Reward mask shape mismatch"
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
                "Reward prediction/target shape mismatch: "
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

    def regression_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_rewards",
            "reward_predictions",
            "reward_mean",
        )

        target = self._get(
            targets,
            "rewards",
            "reward_targets",
        )

        if prediction is None or target is None:
            return None

        prediction = self._squeeze_reward(
            prediction
        )

        target = self._squeeze_reward(
            target
        )

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
        mean = self._get(
            outputs,
            "reward_mean",
        )

        logvar = self._get(
            outputs,
            "reward_logvar",
        )

        target = self._get(
            targets,
            "rewards",
            "reward_targets",
        )

        if (
            mean is None
            or logvar is None
            or target is None
        ):
            return None

        mean = self._squeeze_reward(mean)
        logvar = self._squeeze_reward(logvar)
        target = self._squeeze_reward(target)

        if not (
            mean.shape == logvar.shape == target.shape
        ):
            raise ValueError(
                "Gaussian reward tensor shapes differ"
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

    def binary_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        logits = self._get(
            outputs,
            "binary_reward_logits",
            "reward_binary_logits",
        )

        labels = self._get(
            targets,
            "binary_labels",
            "reward_binary_labels",
        )

        if logits is None or labels is None:
            return None

        logits = self._squeeze_reward(logits)
        labels = self._squeeze_reward(labels)

        if logits.shape != labels.shape:
            raise ValueError(
                "Binary reward shapes differ"
            )

        labels = labels.float()

        if not torch.all(
            (labels >= 0) & (labels <= 1)
        ).item():
            raise ValueError(
                "Binary labels must be in [0,1]"
            )

        loss = F.binary_cross_entropy_with_logits(
            logits.float(),
            labels,
            reduction="none",
        )

        return self._masked_mean(
            loss,
            self._mask(targets, loss),
        )

    def categorical_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        logits = self._get(
            outputs,
            "reward_class_logits",
            "reward_logits",
        )

        labels = self._get(
            targets,
            "reward_classes",
            "reward_class_labels",
        )

        if logits is None or labels is None:
            return None

        if logits.ndim < 2:
            raise ValueError(
                "Categorical logits require [...,C]"
            )

        if labels.shape != logits.shape[:-1]:
            raise ValueError(
                "Categorical reward label shape mismatch"
            )

        if labels.is_floating_point():
            raise TypeError(
                "Categorical reward labels "
                "must contain integer class indices"
            )

        losses = F.cross_entropy(
            logits.float().reshape(
                -1,
                logits.shape[-1],
            ),
            labels.long().reshape(-1),
            reduction="none",
        ).reshape(labels.shape)

        return self._masked_mean(
            losses,
            self._mask(targets, losses),
        )

    def preference_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Bradley-Terry pairwise preference loss.

        Scores may be:
            [B] trajectory scores
            [B,T] per-step reward scores

        For sequence scores, valid timesteps
        are summed before comparison.
        """
        preferred = self._get(
            outputs,
            "preferred_scores",
            "chosen_rewards",
        )

        rejected = self._get(
            outputs,
            "rejected_scores",
            "rejected_rewards",
        )

        if preferred is None or rejected is None:
            return None

        if preferred.shape != rejected.shape:
            raise ValueError(
                "Preference score shapes differ"
            )

        if preferred.ndim == 2:
            mask = self._get(
                targets,
                "preference_mask",
            )

            if mask is not None:
                if mask.shape != preferred.shape:
                    raise ValueError(
                        "preference_mask shape mismatch"
                    )

                preferred = (
                    preferred
                    * mask.to(preferred.dtype)
                )

                rejected = (
                    rejected
                    * mask.to(rejected.dtype)
                )

            preferred = preferred.sum(dim=1)
            rejected = rejected.sum(dim=1)

        if preferred.ndim != 1:
            raise ValueError(
                "Preference scores must reduce to [B]"
            )

        labels = self._get(
            targets,
            "preference_labels",
        )

        if labels is None:
            labels = torch.ones_like(
                preferred
            )

        labels = labels.float()

        if labels.shape != preferred.shape:
            raise ValueError(
                "Preference label shape mismatch"
            )

        if not torch.all(
            (labels >= 0) & (labels <= 1)
        ).item():
            raise ValueError(
                "Preference labels must be in [0,1]"
            )

        score_difference = (
            preferred.float()
            - rejected.float()
        ) / self.config.preference_temperature

        losses = F.binary_cross_entropy_with_logits(
            score_difference,
            labels,
            reduction="none",
        )

        return losses.mean()

    def ranking_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Margin-based pairwise reward ranking.

        Supports optional labels:
            +1: first score should be larger
            -1: second score should be larger
        """
        first = self._get(
            outputs,
            "preferred_scores",
            "chosen_rewards",
        )

        second = self._get(
            outputs,
            "rejected_scores",
            "rejected_rewards",
        )

        if first is None or second is None:
            return None

        if first.shape != second.shape:
            raise ValueError(
                "Ranking score shapes differ"
            )

        if first.ndim == 2:
            first = first.sum(dim=1)
            second = second.sum(dim=1)

        if first.ndim != 1:
            raise ValueError(
                "Ranking scores must reduce to [B]"
            )

        labels = self._get(
            targets,
            "ranking_labels",
        )

        if labels is None:
            labels = torch.ones_like(
                first
            )

        if labels.shape != first.shape:
            raise ValueError(
                "Ranking label shape mismatch"
            )

        if not torch.all(
            (labels == 1) | (labels == -1)
        ).item():
            raise ValueError(
                "Ranking labels must be +1 or -1"
            )

        return F.margin_ranking_loss(
            first.float(),
            second.float(),
            labels.float(),
            margin=self.config.ranking_margin,
            reduction="mean",
        )

    def return_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_returns",
            "return_predictions",
        )

        if prediction is None:
            return None

        prediction = self._squeeze_reward(
            prediction
        )

        target = self._get(
            targets,
            "returns",
            "return_targets",
        )

        if target is None:
            rewards = self._get(
                targets,
                "rewards",
            )

            if rewards is None:
                return None

            rewards = self._squeeze_reward(
                rewards
            )

            if rewards.ndim != 2:
                return None

            dones = self._get(
                targets,
                "dones",
                "terminated",
            )

            mask = self._get(
                targets,
                "mask",
                "sequence_mask",
            )

            target = discounted_returns(
                rewards,
                gamma=self.config.gamma,
                dones=dones,
                mask=mask,
            )

        target = self._squeeze_reward(
            target
        ).detach()

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def temporal_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Match temporal reward differences.

        This is a supervised difference loss,
        not an assumption that rewards
        must remain constant.
        """
        prediction = self._get(
            outputs,
            "predicted_rewards",
            "reward_predictions",
        )

        target = self._get(
            targets,
            "rewards",
            "reward_targets",
        )

        if prediction is None or target is None:
            return None

        prediction = self._squeeze_reward(
            prediction
        )

        target = self._squeeze_reward(
            target
        )

        if prediction.shape != target.shape:
            raise ValueError(
                "Temporal reward shapes differ"
            )

        if (
            prediction.ndim != 2
            or prediction.shape[1] < 2
        ):
            return None

        predicted_difference = (
            prediction[:, 1:]
            - prediction[:, :-1]
        )

        target_difference = (
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
            predicted_difference,
            target_difference,
            mask,
        )

    def distillation_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        student = self._get(
            outputs,
            "student_rewards",
            "predicted_rewards",
        )

        teacher = self._get(
            targets,
            "teacher_rewards",
        )

        if teacher is None:
            teacher = self._get(
                outputs,
                "teacher_rewards",
            )

        if student is None or teacher is None:
            return None

        student = self._squeeze_reward(
            student
        )

        teacher = self._squeeze_reward(
            teacher
        )

        if self.config.detach_teacher:
            teacher = teacher.detach()

        return self._regression(
            student,
            teacher,
            self._mask(targets, student),
        )

    def smoothness_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Optional reward smoothness regularizer.

        Use cautiously for tasks with
        legitimate abrupt reward events.
        """
        prediction = self._get(
            outputs,
            "predicted_rewards",
            "reward_predictions",
        )

        if prediction is None:
            return None

        prediction = self._squeeze_reward(
            prediction
        )

        if (
            prediction.ndim != 2
            or prediction.shape[1] < 2
        ):
            return None

        differences = (
            prediction[:, 1:]
            - prediction[:, :-1]
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

        return self._masked_mean(
            differences.float().square(),
            mask,
        )

    def uncertainty_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Calibrate predicted reward variance
        to observed squared prediction error.
        """
        variance = self._get(
            outputs,
            "predicted_reward_variance",
            "reward_variance",
        )

        prediction = self._get(
            outputs,
            "predicted_rewards",
            "reward_mean",
        )

        target = self._get(
            targets,
            "rewards",
            "reward_targets",
        )

        if (
            variance is None
            or prediction is None
            or target is None
        ):
            return None

        variance = self._squeeze_reward(
            variance
        )

        prediction = self._squeeze_reward(
            prediction
        )

        target = self._squeeze_reward(
            target
        )

        if not (
            variance.shape
            == prediction.shape
            == target.shape
        ):
            raise ValueError(
                "Reward uncertainty shapes differ"
            )

        squared_error = (
            prediction.float()
            - target.float()
        ).square().detach()

        return self._regression(
            variance.float(),
            squared_error,
            self._mask(targets, variance),
        )

    def register_loss(
        self,
        name: str,
        function: RewardLossFunction,
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
                f"Unknown reward loss: {name}"
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
            "regression": self.regression_loss,
            "gaussian_nll": self.gaussian_nll_loss,
            "binary": self.binary_loss,
            "categorical": self.categorical_loss,
            "preference": self.preference_loss,
            "ranking": self.ranking_loss,
            "return": self.return_loss,
            "temporal": self.temporal_loss,
            "distillation": self.distillation_loss,
            "smoothness": self.smoothness_loss,
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
                "No active reward loss components. "
                "Check model outputs, targets "
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
                    f"Non-finite reward loss: {name}"
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
            "reward_total": float(
                total.detach().item()
            ),
            **{
                f"reward/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"reward_weighted/{name}": float(
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


def create_reward_loss(
    *,
    regression_weight: float = 1.0,
    gaussian_nll_weight: float = 0.0,
    binary_weight: float = 0.0,
    categorical_weight: float = 0.0,
    preference_weight: float = 0.0,
    ranking_weight: float = 0.0,
    return_weight: float = 0.0,
    temporal_weight: float = 0.0,
    distillation_weight: float = 0.0,
    smoothness_weight: float = 0.0,
    uncertainty_weight: float = 0.0,
    loss_type: str = "huber",
) -> RewardLoss:
    return RewardLoss(
        RewardLossConfig(
            regression_weight=regression_weight,
            gaussian_nll_weight=gaussian_nll_weight,
            binary_weight=binary_weight,
            categorical_weight=categorical_weight,
            preference_weight=preference_weight,
            ranking_weight=ranking_weight,
            return_weight=return_weight,
            temporal_weight=temporal_weight,
            distillation_weight=distillation_weight,
            smoothness_weight=smoothness_weight,
            uncertainty_weight=uncertainty_weight,
            loss_type=loss_type,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 12
    state_dim = 128
    action_dim = 32

    class DemoRewardModel(nn.Module):
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

            self.reward_head = nn.Linear(
                256,
                1,
            )

            self.variance_head = nn.Linear(
                256,
                1,
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

            hidden = self.encoder(
                features
            )

            rewards = self.reward_head(
                hidden
            ).squeeze(-1)

            variance = F.softplus(
                self.variance_head(hidden)
            ).squeeze(-1) + 1e-6

            return {
                "predicted_rewards": rewards,
                "predicted_reward_variance": variance,
            }

    model = DemoRewardModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = RewardLoss(
        RewardLossConfig(
            regression_weight=1.0,
            uncertainty_weight=0.1,
            temporal_weight=0.2,
            return_weight=0.0,
            smoothness_weight=0.0,
            loss_type="huber",
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

        target_rewards = (
            0.5 * states[..., 0]
            - 0.2 * actions[..., 0]
            + 0.05 * torch.randn(
                batch_size,
                sequence_length,
            )
        )

        outputs = model(
            states,
            actions,
        )

        targets = {
            "rewards": target_rewards,
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
