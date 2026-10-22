
"""
NeuroWorld AI
File: training/losses/policy.py

Policy optimization losses for:
- World-model-based reinforcement learning
- Autonomous robotics
- Physical AI
- Actor-critic agents
- Discrete and continuous control
- Imagined trajectory optimization

Supported objectives:
- REINFORCE / policy gradient
- Advantage actor-critic (A2C)
- PPO clipped surrogate
- Behavior cloning (discrete)
- Behavior cloning (continuous)
- Gaussian policy negative log-likelihood
- Entropy regularization
- Value function regression
- Clipped PPO value function loss
- Approximate KL regularization
- Imagined-return policy optimization

Features:
- Masked trajectories
- Generalized advantage estimation (GAE)
- Discounted returns
- Advantage normalization
- Gradient-safe loss aggregation
- Configurable objective weights
- Custom policy losses
- Per-component metrics

Dependencies:
    torch >= 2.1
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical, Normal


PolicyLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class PolicyLossConfig:
    objective: str = "ppo"

    policy_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    kl_weight: float = 0.0
    behavior_cloning_weight: float = 0.0

    clip_epsilon: float = 0.2
    value_clip_epsilon: float | None = 0.2

    gamma: float = 0.99
    gae_lambda: float = 0.95

    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8

    max_log_ratio: float = 20.0
    max_log_std: float = 2.0
    min_log_std: float = -20.0

    value_loss_type: str = "mse"
    huber_delta: float = 1.0

    normalize_weights: bool = False
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        supported = {
            "reinforce",
            "a2c",
            "ppo",
            "behavior_cloning",
            "imagined",
        }

        if self.objective not in supported:
            raise ValueError(
                f"Unsupported objective: {self.objective}"
            )

        for name in (
            "policy_weight",
            "value_weight",
            "entropy_weight",
            "kl_weight",
            "behavior_cloning_weight",
        ):
            weight = getattr(self, name)

            if (
                not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and nonnegative"
                )

        if not 0 < self.clip_epsilon < 1:
            raise ValueError(
                "clip_epsilon must be in (0,1)"
            )

        if (
            self.value_clip_epsilon is not None
            and self.value_clip_epsilon <= 0
        ):
            raise ValueError(
                "value_clip_epsilon must be positive"
            )

        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma must be in [0,1]"
            )

        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(
                "gae_lambda must be in [0,1]"
            )

        if self.advantage_epsilon <= 0:
            raise ValueError(
                "advantage_epsilon must be positive"
            )

        if self.max_log_ratio <= 0:
            raise ValueError(
                "max_log_ratio must be positive"
            )

        if self.min_log_std >= self.max_log_std:
            raise ValueError(
                "min_log_std must be below max_log_std"
            )

        if self.value_loss_type not in {
            "mse",
            "huber",
        }:
            raise ValueError(
                "Invalid value_loss_type"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )


def masked_mean(
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if mask is None:
        return values.mean()

    mask = mask.to(
        device=values.device,
        dtype=torch.bool,
    )

    try:
        mask = torch.broadcast_to(
            mask,
            values.shape,
        )
    except RuntimeError as exc:
        raise ValueError(
            "Mask shape is not broadcastable"
        ) from exc

    weights = mask.to(values.dtype)

    return (
        (values * weights).sum()
        / weights.sum().clamp_min(1.0)
    )


def normalize_advantages(
    advantages: Tensor,
    mask: Tensor | None = None,
    epsilon: float = 1e-8,
) -> Tensor:
    advantages = advantages.float()

    if mask is None:
        mean = advantages.mean()
        variance = advantages.var(
            unbiased=False
        )
    else:
        mask = torch.broadcast_to(
            mask.to(
                device=advantages.device,
                dtype=torch.bool,
            ),
            advantages.shape,
        )

        weights = mask.to(
            advantages.dtype
        )

        count = weights.sum().clamp_min(1.0)

        mean = (
            advantages * weights
        ).sum() / count

        variance = (
            (advantages - mean).square()
            * weights
        ).sum() / count

    normalized = (
        advantages - mean
    ) / torch.sqrt(
        variance + epsilon
    )

    if mask is not None:
        normalized = normalized.masked_fill(
            ~mask,
            0.0,
        )

    return normalized


def compute_discounted_returns(
    rewards: Tensor,
    dones: Tensor,
    *,
    gamma: float = 0.99,
    bootstrap_value: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Discounted returns for [B,T] rewards.

    dones:
        True when the episode terminates.

    bootstrap_value:
        [B] estimated value after final step.

    Note:
        Time-limit truncation should generally
        bootstrap rather than be marked as a
        true terminal state.
    """
    if rewards.ndim != 2:
        raise ValueError(
            "rewards must have shape [B,T]"
        )

    if dones.shape != rewards.shape:
        raise ValueError(
            "dones must match rewards"
        )

    batch_size, steps = rewards.shape

    rewards = rewards.float()
    dones = dones.bool()

    if bootstrap_value is None:
        running = torch.zeros(
            batch_size,
            device=rewards.device,
            dtype=rewards.dtype,
        )
    else:
        if bootstrap_value.shape != (batch_size,):
            raise ValueError(
                "bootstrap_value must be [B]"
            )

        running = bootstrap_value.float()

    if mask is not None:
        mask = torch.broadcast_to(
            mask.to(
                device=rewards.device,
                dtype=torch.bool,
            ),
            rewards.shape,
        )

    returns = torch.zeros_like(
        rewards
    )

    for t in reversed(range(steps)):
        running = (
            rewards[:, t]
            + gamma
            * (~dones[:, t]).float()
            * running
        )

        if mask is not None:
            running = torch.where(
                mask[:, t],
                running,
                torch.zeros_like(running),
            )

        returns[:, t] = running

    return returns


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    *,
    next_value: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Generalized Advantage Estimation.

    Shapes:
        rewards:    [B,T]
        values:     [B,T]
        dones:      [B,T]
        next_value: [B]

    Returns:
        advantages: [B,T]
        returns:    [B,T]

    Inputs are detached because GAE is
    used as a fixed policy target.
    """
    if rewards.ndim != 2:
        raise ValueError(
            "rewards must have shape [B,T]"
        )

    if (
        values.shape != rewards.shape
        or dones.shape != rewards.shape
    ):
        raise ValueError(
            "GAE tensor shapes must match"
        )

    batch_size, steps = rewards.shape

    if next_value.shape != (batch_size,):
        raise ValueError(
            "next_value must have shape [B]"
        )

    rewards = rewards.detach().float()
    values = values.detach().float()
    dones = dones.detach().bool()
    next_value = next_value.detach().float()

    if mask is not None:
        mask = torch.broadcast_to(
            mask.to(
                device=rewards.device,
                dtype=torch.bool,
            ),
            rewards.shape,
        )

    advantages = torch.zeros_like(
        rewards
    )

    last_advantage = torch.zeros(
        batch_size,
        device=rewards.device,
        dtype=rewards.dtype,
    )

    for t in reversed(range(steps)):
        next_values = (
            next_value
            if t == steps - 1
            else values[:, t + 1]
        )

        nonterminal = (
            ~dones[:, t]
        ).float()

        if (
            mask is not None
            and t < steps - 1
        ):
            nonterminal = (
                nonterminal
                * mask[:, t + 1].float()
            )

        delta = (
            rewards[:, t]
            + gamma
            * nonterminal
            * next_values
            - values[:, t]
        )

        last_advantage = (
            delta
            + gamma
            * gae_lambda
            * nonterminal
            * last_advantage
        )

        if mask is not None:
            last_advantage = torch.where(
                mask[:, t],
                last_advantage,
                torch.zeros_like(
                    last_advantage
                ),
            )

        advantages[:, t] = last_advantage

    returns = (
        advantages + values
    )

    return advantages, returns


class PolicyLoss(nn.Module):
    """
    Policy optimization objective.

    Expected outputs:
        policy_logits
        action_mean
        action_log_std
        log_probs
        entropy
        values
        predicted_values
        imagined_returns

    Expected targets:
        actions
        advantages
        returns
        old_log_probs
        old_values
        old_policy_logits
        rewards
        dones
        mask
        behavior_actions

    Important:
        PPO requires old_log_probs collected
        from the behavior policy before
        updating the model.
    """

    def __init__(
        self,
        config: PolicyLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or PolicyLossConfig()
        )

        self.weights = {
            "policy": self.config.policy_weight,
            "value": self.config.value_weight,
            "entropy": self.config.entropy_weight,
            "kl": self.config.kl_weight,
            "behavior_cloning": (
                self.config.behavior_cloning_weight
            ),
        }

        self.custom_losses: dict[
            str, PolicyLossFunction
        ] = {}

        self.last_metrics: dict[str, float] = {}

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

    def _mask(
        self,
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = self._get(
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

        try:
            return torch.broadcast_to(
                mask,
                reference.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Policy mask shape mismatch"
            ) from exc

    def _distribution(
        self,
        outputs: Mapping[str, Any],
    ) -> Categorical | Normal | None:
        logits = self._get(
            outputs,
            "policy_logits",
        )

        if logits is not None:
            return Categorical(
                logits=logits.float()
            )

        mean = self._get(
            outputs,
            "action_mean",
            "policy_mean",
        )

        log_std = self._get(
            outputs,
            "action_log_std",
            "policy_log_std",
        )

        if mean is None or log_std is None:
            return None

        log_std = log_std.float().clamp(
            min=self.config.min_log_std,
            max=self.config.max_log_std,
        )

        return Normal(
            loc=mean.float(),
            scale=log_std.exp(),
        )

    def _log_prob_and_entropy(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> tuple[Tensor | None, Tensor | None]:
        explicit_log_probs = self._get(
            outputs,
            "log_probs",
            "policy_log_probs",
        )

        explicit_entropy = self._get(
            outputs,
            "entropy",
            "policy_entropy",
        )

        distribution = self._distribution(
            outputs
        )

        actions = self._get(
            targets,
            "actions",
        )

        if (
            actions is None
            or distribution is None
        ):
            return (
                explicit_log_probs,
                explicit_entropy,
            )

        if isinstance(distribution, Categorical):
            if actions.shape != (
                distribution.logits.shape[:-1]
            ):
                raise ValueError(
                    "Discrete actions must match "
                    "policy batch dimensions"
                )

            if actions.is_floating_point():
                raise TypeError(
                    "Discrete actions must be integers"
                )

            log_probs = distribution.log_prob(
                actions.long()
            )

            entropy = distribution.entropy()

        else:
            if actions.shape != distribution.loc.shape:
                raise ValueError(
                    "Continuous actions must match "
                    "policy mean shape"
                )

            log_probs = (
                distribution.log_prob(
                    actions.float()
                )
                .sum(dim=-1)
            )

            entropy = (
                distribution.entropy()
                .sum(dim=-1)
            )

        return (
            explicit_log_probs
            if explicit_log_probs is not None
            else log_probs,
            explicit_entropy
            if explicit_entropy is not None
            else entropy,
        )

    def _advantages(
        self,
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        advantages = self._get(
            targets,
            "advantages",
        )

        if advantages is None:
            return None

        advantages = advantages.detach().float()

        if self.config.normalize_advantages:
            advantages = normalize_advantages(
                advantages,
                mask=self._mask(
                    targets,
                    advantages,
                ),
                epsilon=(
                    self.config.advantage_epsilon
                ),
            )

        return advantages

    def reinforce_loss(
        self,
        log_probs: Tensor,
        returns: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        if log_probs.shape != returns.shape:
            raise ValueError(
                "REINFORCE shapes must match"
            )

        return -masked_mean(
            log_probs.float()
            * returns.detach().float(),
            mask,
        )

    def actor_critic_loss(
        self,
        log_probs: Tensor,
        advantages: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        if log_probs.shape != advantages.shape:
            raise ValueError(
                "Actor-critic shapes must match"
            )

        return -masked_mean(
            log_probs.float()
            * advantages.detach().float(),
            mask,
        )

    def ppo_loss(
        self,
        log_probs: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not (
            log_probs.shape
            == old_log_probs.shape
            == advantages.shape
        ):
            raise ValueError(
                "PPO input shapes must match"
            )

        log_ratio = (
            log_probs.float()
            - old_log_probs.detach().float()
        )

        ratio = torch.exp(
            log_ratio.clamp(
                min=-self.config.max_log_ratio,
                max=self.config.max_log_ratio,
            )
        )

        advantages = (
            advantages.detach().float()
        )

        surrogate_unclipped = (
            ratio * advantages
        )

        clipped_ratio = ratio.clamp(
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )

        surrogate_clipped = (
            clipped_ratio * advantages
        )

        objective = torch.minimum(
            surrogate_unclipped,
            surrogate_clipped,
        )

        loss = -masked_mean(
            objective,
            mask,
        )

        with torch.no_grad():
            approx_kl = masked_mean(
                (
                    torch.exp(log_ratio)
                    - 1.0
                    - log_ratio
                ),
                mask,
            )

            clip_fraction = masked_mean(
                (
                    (ratio - 1.0).abs()
                    > self.config.clip_epsilon
                ).float(),
                mask,
            )

        return loss, {
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "ratio_mean": masked_mean(
                ratio.detach(),
                mask,
            ),
        }

    def behavior_cloning_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        actions = self._get(
            targets,
            "behavior_actions",
            "expert_actions",
            "actions",
        )

        if actions is None:
            return None

        distribution = self._distribution(
            outputs
        )

        if distribution is None:
            return None

        if isinstance(distribution, Categorical):
            if actions.shape != (
                distribution.logits.shape[:-1]
            ):
                raise ValueError(
                    "Behavior action shape mismatch"
                )

            if actions.is_floating_point():
                raise TypeError(
                    "Discrete behavior actions "
                    "must be integers"
                )

            losses = -distribution.log_prob(
                actions.long()
            )

        else:
            if actions.shape != distribution.loc.shape:
                raise ValueError(
                    "Continuous behavior action "
                    "shape mismatch"
                )

            losses = -distribution.log_prob(
                actions.float()
            ).sum(dim=-1)

        return masked_mean(
            losses,
            self._mask(targets, losses),
        )

    def value_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        values = self._get(
            outputs,
            "values",
            "predicted_values",
        )

        returns = self._get(
            targets,
            "returns",
            "value_targets",
        )

        if values is None or returns is None:
            return None

        if (
            values.ndim == returns.ndim + 1
            and values.shape[-1] == 1
        ):
            values = values.squeeze(-1)

        if values.shape != returns.shape:
            raise ValueError(
                "Value prediction/target mismatch"
            )

        values = values.float()
        returns = returns.detach().float()

        if self.config.value_loss_type == "huber":
            losses = F.huber_loss(
                values,
                returns,
                delta=self.config.huber_delta,
                reduction="none",
            )

        else:
            losses = (
                values - returns
            ).square()

        old_values = self._get(
            targets,
            "old_values",
        )

        if (
            self.config.objective == "ppo"
            and old_values is not None
            and self.config.value_clip_epsilon is not None
        ):
            if (
                old_values.ndim == values.ndim + 1
                and old_values.shape[-1] == 1
            ):
                old_values = old_values.squeeze(-1)

            if old_values.shape != values.shape:
                raise ValueError(
                    "Old value shape mismatch"
                )

            old_values = old_values.detach().float()

            clipped_values = (
                old_values
                + (
                    values - old_values
                ).clamp(
                    -self.config.value_clip_epsilon,
                    self.config.value_clip_epsilon,
                )
            )

            if self.config.value_loss_type == "huber":
                clipped_losses = F.huber_loss(
                    clipped_values,
                    returns,
                    delta=self.config.huber_delta,
                    reduction="none",
                )

            else:
                clipped_losses = (
                    clipped_values - returns
                ).square()

            losses = torch.maximum(
                losses,
                clipped_losses,
            )

        return masked_mean(
            losses,
            self._mask(targets, losses),
        )

    def entropy_loss(
        self,
        entropy: Tensor | None,
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        if entropy is None:
            return None

        return -masked_mean(
            entropy.float(),
            self._mask(targets, entropy),
        )

    def kl_regularization(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        log_probs: Tensor | None,
    ) -> Tensor | None:
        """
        Approximate KL(old || current)
        using behavior-policy samples.

        For exact categorical KL, supply
        old_policy_logits.
        """
        current_logits = self._get(
            outputs,
            "policy_logits",
        )

        old_logits = self._get(
            targets,
            "old_policy_logits",
        )

        if (
            current_logits is not None
            and old_logits is not None
        ):
            if current_logits.shape != old_logits.shape:
                raise ValueError(
                    "Policy KL logits shape mismatch"
                )

            old_log_prob = F.log_softmax(
                old_logits.detach().float(),
                dim=-1,
            )

            current_log_prob = F.log_softmax(
                current_logits.float(),
                dim=-1,
            )

            old_prob = old_log_prob.exp()

            kl = (
                old_prob
                * (
                    old_log_prob
                    - current_log_prob
                )
            ).sum(dim=-1)

            return masked_mean(
                kl,
                self._mask(targets, kl),
            )

        old_log_probs = self._get(
            targets,
            "old_log_probs",
        )

        if (
            old_log_probs is None
            or log_probs is None
        ):
            return None

        if old_log_probs.shape != log_probs.shape:
            raise ValueError(
                "KL log-probability shape mismatch"
            )

        log_ratio = (
            log_probs.float()
            - old_log_probs.detach().float()
        )

        approx_kl = (
            torch.exp(
                log_ratio.clamp(
                    -self.config.max_log_ratio,
                    self.config.max_log_ratio,
                )
            )
            - 1.0
            - log_ratio
        )

        return masked_mean(
            approx_kl,
            self._mask(targets, approx_kl),
        )

    def imagined_policy_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        log_probs: Tensor | None,
    ) -> Tensor | None:
        """
        Policy optimization over imagined
        world-model trajectories.

        If imagined_returns are differentiable
        through the policy and world model,
        optimize them directly.

        Otherwise, use score-function
        gradients through log_probs.
        """
        imagined_returns = self._get(
            outputs,
            "imagined_returns",
        )

        if imagined_returns is None:
            imagined_returns = self._get(
                targets,
                "imagined_returns",
            )

        if imagined_returns is None:
            return None

        if imagined_returns.requires_grad:
            return -masked_mean(
                imagined_returns.float(),
                self._mask(
                    targets,
                    imagined_returns,
                ),
            )

        if log_probs is None:
            raise ValueError(
                "Non-differentiable imagined returns "
                "require policy log probabilities"
            )

        if imagined_returns.shape != log_probs.shape:
            raise ValueError(
                "Imagined return shape mismatch"
            )

        return self.reinforce_loss(
            log_probs,
            imagined_returns,
            mask=self._mask(
                targets,
                log_probs,
            ),
        )

    def register_loss(
        self,
        name: str,
        function: PolicyLossFunction,
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
                f"Unknown policy loss: {name}"
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
        log_probs, entropy = (
            self._log_prob_and_entropy(
                outputs,
                targets,
            )
        )

        advantages = self._advantages(
            targets
        )

        returns = self._get(
            targets,
            "returns",
        )

        old_log_probs = self._get(
            targets,
            "old_log_probs",
        )

        components: dict[str, Tensor] = {}
        diagnostics: dict[str, Tensor] = {}

        objective = self.config.objective

        if self.weights["policy"] > 0:
            if objective == "ppo":
                if (
                    log_probs is None
                    or old_log_probs is None
                    or advantages is None
                ):
                    raise ValueError(
                        "PPO requires log_probs, "
                        "old_log_probs and advantages"
                    )

                policy_loss, diagnostics = (
                    self.ppo_loss(
                        log_probs,
                        old_log_probs,
                        advantages,
                        mask=self._mask(
                            targets,
                            log_probs,
                        ),
                    )
                )

                components["policy"] = policy_loss

            elif objective == "a2c":
                if (
                    log_probs is None
                    or advantages is None
                ):
                    raise ValueError(
                        "A2C requires log_probs "
                        "and advantages"
                    )

                components["policy"] = (
                    self.actor_critic_loss(
                        log_probs,
                        advantages,
                        mask=self._mask(
                            targets,
                            log_probs,
                        ),
                    )
                )

            elif objective == "reinforce":
                if (
                    log_probs is None
                    or returns is None
                ):
                    raise ValueError(
                        "REINFORCE requires log_probs "
                        "and returns"
                    )

                components["policy"] = (
                    self.reinforce_loss(
                        log_probs,
                        returns,
                        mask=self._mask(
                            targets,
                            log_probs,
                        ),
                    )
                )

            elif objective == "imagined":
                imagined_loss = (
                    self.imagined_policy_loss(
                        outputs,
                        targets,
                        log_probs,
                    )
                )

                if imagined_loss is None:
                    raise ValueError(
                        "Imagined objective requires "
                        "imagined_returns"
                    )

                components["policy"] = imagined_loss

            elif objective == "behavior_cloning":
                cloning_loss = (
                    self.behavior_cloning_loss(
                        outputs,
                        targets,
                    )
                )

                if cloning_loss is None:
                    raise ValueError(
                        "Behavior cloning requires "
                        "policy distribution and actions"
                    )

                components["policy"] = cloning_loss

        if self.weights["value"] > 0:
            value_loss = self.value_loss(
                outputs,
                targets,
            )

            if value_loss is not None:
                components["value"] = value_loss

        if self.weights["entropy"] > 0:
            entropy_loss = self.entropy_loss(
                entropy,
                targets,
            )

            if entropy_loss is not None:
                components["entropy"] = entropy_loss

        if self.weights["kl"] > 0:
            kl_loss = self.kl_regularization(
                outputs,
                targets,
                log_probs,
            )

            if kl_loss is not None:
                components["kl"] = kl_loss

        if self.weights["behavior_cloning"] > 0:
            if objective != "behavior_cloning":
                cloning_loss = (
                    self.behavior_cloning_loss(
                        outputs,
                        targets,
                    )
                )

                if cloning_loss is not None:
                    components[
                        "behavior_cloning"
                    ] = cloning_loss

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
                "No active policy loss components"
            )

        for name, loss in components.items():
            if not isinstance(loss, Tensor):
                raise TypeError(
                    f"{name} must return a Tensor"
                )

            if loss.ndim != 0:
                raise ValueError(
                    f"{name} loss must be scalar"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Non-finite policy loss: {name}"
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
            "policy_total": float(
                total.detach().item()
            ),
            **{
                f"policy/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"policy_weighted/{name}": float(
                    loss.detach().item()
                )
                for name, loss in weighted.items()
            },
            **{
                f"policy/{name}": float(
                    value.detach().item()
                )
                for name, value in diagnostics.items()
            },
        }

        if return_components:
            return total, {
                "total": total,
                **components,
                **diagnostics,
            }

        return total

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return dict(self.last_metrics)

    def reset_metrics(self) -> None:
        self.last_metrics.clear()


def create_policy_loss(
    *,
    objective: str = "ppo",
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    kl_weight: float = 0.0,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> PolicyLoss:
    return PolicyLoss(
        PolicyLossConfig(
            objective=objective,
            policy_weight=policy_weight,
            value_weight=value_weight,
            entropy_weight=entropy_weight,
            kl_weight=kl_weight,
            clip_epsilon=clip_epsilon,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 16
    observation_dim = 32
    num_actions = 6

    class DemoActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(
                    observation_dim,
                    128,
                ),
                nn.Tanh(),
                nn.Linear(
                    128,
                    128,
                ),
                nn.Tanh(),
            )

            self.actor = nn.Linear(
                128,
                num_actions,
            )

            self.critic = nn.Linear(
                128,
                1,
            )

        def forward(
            self,
            observations: Tensor,
        ) -> dict[str, Tensor]:
            hidden = self.encoder(
                observations
            )

            return {
                "policy_logits": self.actor(
                    hidden
                ),
                "values": self.critic(
                    hidden
                ).squeeze(-1),
            }

    model = DemoActorCritic()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = PolicyLoss(
        PolicyLossConfig(
            objective="ppo",
            policy_weight=1.0,
            value_weight=0.5,
            entropy_weight=0.01,
            kl_weight=0.0,
            clip_epsilon=0.2,
            normalize_advantages=True,
        )
    )

    observations = torch.randn(
        batch_size,
        sequence_length,
        observation_dim,
    )

    rewards = torch.randn(
        batch_size,
        sequence_length,
    )

    dones = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.bool,
    )

    dones[:, -1] = True

    # Collect behavior-policy statistics
    # before the PPO optimization updates.
    with torch.no_grad():
        old_outputs = model(
            observations
        )

        old_distribution = Categorical(
            logits=old_outputs["policy_logits"]
        )

        actions = old_distribution.sample()

        old_log_probs = (
            old_distribution.log_prob(
                actions
            )
        )

        old_values = (
            old_outputs["values"]
        )

        advantages, returns = compute_gae(
            rewards,
            old_values,
            dones,
            next_value=torch.zeros(
                batch_size
            ),
            gamma=0.99,
            gae_lambda=0.95,
        )

    targets = {
        "actions": actions,
        "advantages": advantages,
        "returns": returns,
        "old_log_probs": old_log_probs,
        "old_values": old_values,
        "mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
        ),
    }

    for step in range(5):
        outputs = model(
            observations
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, components = criterion(
            outputs,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )
