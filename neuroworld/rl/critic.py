
"""
NeuroWorld AI
File: neuroworld/rl/critic.py

Advanced Critic Networks and Value Estimation.

Features:
- State-value critic V(s)
- Action-value critic Q(s,a)
- Discrete-action Q critic
- Hybrid-action Q critic
- Twin Q critics
- Distributional quantile critic
- Target network synchronization
- Soft target updates (Polyak averaging)
- TD(0) value learning
- Double-Q Bellman targets
- Generalized Advantage Estimation (GAE)
- Lambda returns
- Gradient clipping
- Critic checkpoint save/load
- Batch inference and evaluation
- Reproducible demonstration

Dependencies:
    pip install torch numpy

Conventions:
    States:             [B, state_dim]
    Continuous actions: [B, action_dim]
    Discrete actions:   [B]
    Rewards:            [B]
    Done flags:         [B]

Important:
    This module provides neural value estimators.
    It does not independently guarantee policy
    safety or correctness.
"""

from __future__ import annotations

import copy
import math

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class CriticError(Exception):
    """Base critic exception."""


class CriticValidationError(CriticError):
    """Invalid critic input or configuration."""


# ============================================================
# CRITIC TYPES
# ============================================================

class CriticType(str, Enum):
    STATE_VALUE = "state_value"
    CONTINUOUS_Q = "continuous_q"
    DISCRETE_Q = "discrete_q"
    HYBRID_Q = "hybrid_q"
    TWIN_Q = "twin_q"
    DISTRIBUTIONAL = "distributional"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CriticConfig:
    state_dim: int = 128
    action_dim: int = 8
    num_discrete_actions: int = 12

    hidden_dim: int = 256
    num_hidden_layers: int = 3

    critic_type: CriticType = CriticType.STATE_VALUE

    gamma: float = 0.99
    gae_lambda: float = 0.95

    tau: float = 0.005

    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    max_grad_norm: float = 1.0

    num_quantiles: int = 32
    quantile_huber_kappa: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "critic_type",
            CriticType(self.critic_type),
        )

        for name in (
            "state_dim",
            "action_dim",
            "num_discrete_actions",
            "hidden_dim",
            "num_hidden_layers",
            "num_quantiles",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma must be in [0,1]"
            )

        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(
                "gae_lambda must be in [0,1]"
            )

        if not 0 < self.tau <= 1:
            raise ValueError(
                "tau must be in (0,1]"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive"
            )

        if self.quantile_huber_kappa <= 0:
            raise ValueError(
                "quantile_huber_kappa must be positive"
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_states(
    states: Tensor,
    state_dim: int,
) -> Tensor:
    if not isinstance(states, Tensor):
        raise CriticValidationError(
            "states must be a Tensor"
        )

    if states.ndim == 1:
        states = states.unsqueeze(0)

    if (
        states.ndim != 2
        or states.shape[1] != state_dim
    ):
        raise CriticValidationError(
            f"Expected states [B,{state_dim}]"
        )

    if not torch.is_floating_point(states):
        raise CriticValidationError(
            "States must be floating point"
        )

    if not torch.isfinite(states).all():
        raise CriticValidationError(
            "States contain non-finite values"
        )

    return states


def validate_continuous_actions(
    actions: Tensor,
    batch_size: int,
    action_dim: int,
) -> Tensor:
    if not isinstance(actions, Tensor):
        raise CriticValidationError(
            "actions must be a Tensor"
        )

    if actions.ndim == 1:
        actions = actions.unsqueeze(0)

    if actions.shape != (
        batch_size,
        action_dim,
    ):
        raise CriticValidationError(
            "Continuous action shape mismatch"
        )

    if not torch.isfinite(actions).all():
        raise CriticValidationError(
            "Actions contain non-finite values"
        )

    return actions


def validate_discrete_actions(
    actions: Tensor,
    batch_size: int,
    num_actions: int,
) -> Tensor:
    if not isinstance(actions, Tensor):
        raise CriticValidationError(
            "actions must be a Tensor"
        )

    actions = actions.reshape(-1)

    if actions.shape[0] != batch_size:
        raise CriticValidationError(
            "Discrete action batch mismatch"
        )

    if actions.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise CriticValidationError(
            "Discrete actions must be integers"
        )

    actions = actions.long()

    if (
        torch.any(actions < 0)
        or torch.any(actions >= num_actions)
    ):
        raise CriticValidationError(
            "Discrete action out of range"
        )

    return actions


def validate_vector(
    values: Tensor,
    batch_size: int,
    name: str,
) -> Tensor:
    if not isinstance(values, Tensor):
        raise CriticValidationError(
            f"{name} must be a Tensor"
        )

    values = values.reshape(-1)

    if values.shape[0] != batch_size:
        raise CriticValidationError(
            f"{name} batch size mismatch"
        )

    if not torch.isfinite(values).all():
        raise CriticValidationError(
            f"{name} contains non-finite values"
        )

    return values


# ============================================================
# SHARED MLP BACKBONE
# ============================================================

def build_critic_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    current_dim = input_dim

    for _ in range(num_hidden_layers):
        layers.extend(
            [
                nn.Linear(
                    current_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
        )

        current_dim = hidden_dim

    layers.append(
        nn.Linear(
            current_dim,
            output_dim,
        )
    )

    return nn.Sequential(*layers)


# ============================================================
# STATE VALUE CRITIC
# ============================================================

class StateValueCritic(nn.Module):
    """
    Estimates:
        V(s) = expected discounted return
    """

    def __init__(
        self,
        config: CriticConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = build_critic_mlp(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_hidden_layers=(
                config.num_hidden_layers
            ),
        )

    def forward(
        self,
        states: Tensor,
    ) -> Tensor:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        return self.network(
            states
        ).squeeze(-1)


# ============================================================
# CONTINUOUS ACTION Q CRITIC
# ============================================================

class ContinuousQCritic(nn.Module):
    """
    Estimates Q(s,a) for continuous actions.
    """

    def __init__(
        self,
        config: CriticConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = build_critic_mlp(
            input_dim=(
                config.state_dim
                + config.action_dim
            ),
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_hidden_layers=(
                config.num_hidden_layers
            ),
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        actions = validate_continuous_actions(
            actions,
            states.shape[0],
            self.config.action_dim,
        )

        features = torch.cat(
            [states, actions],
            dim=-1,
        )

        return self.network(
            features
        ).squeeze(-1)


# ============================================================
# DISCRETE ACTION Q CRITIC
# ============================================================

class DiscreteQCritic(nn.Module):
    """
    Produces Q values for every action.

    Input:
        states [B,S]

    Output:
        q_values [B,A]
    """

    def __init__(
        self,
        config: CriticConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = build_critic_mlp(
            input_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            output_dim=(
                config.num_discrete_actions
            ),
            num_hidden_layers=(
                config.num_hidden_layers
            ),
        )

    def forward(
        self,
        states: Tensor,
    ) -> Tensor:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        return self.network(
            states
        )

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        q_values = self.forward(
            states
        )

        actions = validate_discrete_actions(
            actions,
            q_values.shape[0],
            self.config.num_discrete_actions,
        )

        return q_values.gather(
            dim=1,
            index=actions.unsqueeze(-1),
        ).squeeze(-1)


# ============================================================
# HYBRID ACTION Q CRITIC
# ============================================================

class HybridQCritic(nn.Module):
    """
    Q(s, continuous_action, discrete_action).

    Discrete actions are one-hot encoded.
    """

    def __init__(
        self,
        config: CriticConfig,
    ) -> None:
        super().__init__()

        self.config = config

        input_dim = (
            config.state_dim
            + config.action_dim
            + config.num_discrete_actions
        )

        self.network = build_critic_mlp(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_hidden_layers=(
                config.num_hidden_layers
            ),
        )

    def forward(
        self,
        states: Tensor,
        continuous_actions: Tensor,
        discrete_actions: Tensor,
    ) -> Tensor:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        continuous_actions = (
            validate_continuous_actions(
                continuous_actions,
                states.shape[0],
                self.config.action_dim,
            )
        )

        discrete_actions = (
            validate_discrete_actions(
                discrete_actions,
                states.shape[0],
                self.config.num_discrete_actions,
            )
        )

        one_hot = F.one_hot(
            discrete_actions,
            num_classes=(
                self.config.num_discrete_actions
            ),
        ).to(
            dtype=states.dtype,
            device=states.device,
        )

        features = torch.cat(
            [
                states,
                continuous_actions,
                one_hot,
            ],
            dim=-1,
        )

        return self.network(
            features
        ).squeeze(-1)


# ============================================================
# TWIN Q CRITIC
# ============================================================

class TwinQCritic(nn.Module):
    """
    Two independent continuous-action
    Q critics.

    Useful for TD3 and SAC-style training.
    """

    def __init__(
        self,
        config: CriticConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.q1 = ContinuousQCritic(
            config
        )

        self.q2 = ContinuousQCritic(
            config
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1(states, actions),
            self.q2(states, actions),
        )

    def minimum(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        q1, q2 = self.forward(
            states,
            actions,
        )

        return torch.minimum(
            q1,
            q2,
        )


# ============================================================
# DISTRIBUTIONAL QUANTILE CRITIC
# ============================================================

class DistributionalCritic(nn.Module):
    """
    Quantile-based distributional Q critic.

    Each output represents a return quantile
    at a fixed quantile fraction.

    Uses continuous state-action input.
    """

    def __init__(
        self,
        config: CriticConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = build_critic_mlp(
            input_dim=(
                config.state_dim
                + config.action_dim
            ),
            hidden_dim=config.hidden_dim,
            output_dim=config.num_quantiles,
            num_hidden_layers=(
                config.num_hidden_layers
            ),
        )

        fractions = (
            torch.arange(
                config.num_quantiles,
                dtype=torch.float32,
            )
            + 0.5
        ) / config.num_quantiles

        self.register_buffer(
            "quantile_fractions",
            fractions,
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        actions = validate_continuous_actions(
            actions,
            states.shape[0],
            self.config.action_dim,
        )

        features = torch.cat(
            [states, actions],
            dim=-1,
        )

        return self.network(
            features
        )

    def expected_q(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        return self.forward(
            states,
            actions,
        ).mean(dim=-1)


# ============================================================
# QUANTILE HUBER LOSS
# ============================================================

def quantile_huber_loss(
    predicted_quantiles: Tensor,
    target_quantiles: Tensor,
    quantile_fractions: Tensor,
    kappa: float = 1.0,
) -> Tensor:
    """
    Quantile regression Huber loss.

    predicted_quantiles: [B,N]
    target_quantiles:    [B,M]
    quantile_fractions:  [N]

    Returns scalar loss.
    """

    if (
        predicted_quantiles.ndim != 2
        or target_quantiles.ndim != 2
    ):
        raise CriticValidationError(
            "Quantiles must be 2D"
        )

    if (
        predicted_quantiles.shape[0]
        != target_quantiles.shape[0]
    ):
        raise CriticValidationError(
            "Quantile batch mismatch"
        )

    if (
        quantile_fractions.numel()
        != predicted_quantiles.shape[1]
    ):
        raise CriticValidationError(
            "Quantile fraction count mismatch"
        )

    if kappa <= 0:
        raise CriticValidationError(
            "kappa must be positive"
        )

    errors = (
        target_quantiles.detach().unsqueeze(1)
        - predicted_quantiles.unsqueeze(2)
    )

    absolute_errors = errors.abs()

    huber = torch.where(
        absolute_errors <= kappa,
        0.5 * errors.square(),
        kappa * (
            absolute_errors
            - 0.5 * kappa
        ),
    )

    fractions = quantile_fractions.view(
        1,
        -1,
        1,
    )

    asymmetric_weight = (
        fractions
        - (
            errors.detach() < 0
        ).to(
            fractions.dtype
        )
    ).abs()

    return (
        asymmetric_weight
        * huber
        / kappa
    ).mean(dim=2).sum(dim=1).mean()


# ============================================================
# TEMPORAL DIFFERENCE TARGETS
# ============================================================

def td_target(
    rewards: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    gamma: float = 0.99,
) -> Tensor:
    """
    One-step Bellman target:

        y = r + gamma*(1-terminated)*V(s')

    terminated is 1 for true terminal states.

    Time-limit truncations should normally
    continue bootstrapping.
    """

    if not 0 <= gamma <= 1:
        raise CriticValidationError(
            "gamma must be in [0,1]"
        )

    batch_size = rewards.numel()

    rewards = validate_vector(
        rewards,
        batch_size,
        "rewards",
    )

    next_values = validate_vector(
        next_values,
        batch_size,
        "next_values",
    )

    terminated = validate_vector(
        terminated,
        batch_size,
        "terminated",
    )

    if (
        torch.any(terminated < 0)
        or torch.any(terminated > 1)
    ):
        raise CriticValidationError(
            "terminated must be in [0,1]"
        )

    return (
        rewards
        + gamma
        * (1.0 - terminated)
        * next_values
    )


# ============================================================
# GENERALIZED ADVANTAGE ESTIMATION
# ============================================================

def generalized_advantage_estimation(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    episode_ends: Tensor | None = None,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Generalized Advantage Estimation.

    Shapes:
        rewards:      [T] or [T,B]
        values:       same
        next_values:  same
        terminated:   same
        episode_ends: same, optional

    terminated:
        True environment terminal condition.
        Controls value bootstrapping.

    episode_ends:
        Terminal OR truncated episode boundary.
        Controls advantage recursion.

    Returns:
        advantages
        returns = advantages + values
    """

    if not (
        0 <= gamma <= 1
        and 0 <= gae_lambda <= 1
    ):
        raise CriticValidationError(
            "Invalid GAE parameters"
        )

    if rewards.ndim not in (1, 2):
        raise CriticValidationError(
            "Expected [T] or [T,B] rewards"
        )

    if not (
        rewards.shape == values.shape
        == next_values.shape
        == terminated.shape
    ):
        raise CriticValidationError(
            "GAE tensor shapes must match"
        )

    if episode_ends is None:
        episode_ends = terminated

    if episode_ends.shape != rewards.shape:
        raise CriticValidationError(
            "episode_ends shape mismatch"
        )

    if not torch.isfinite(rewards).all():
        raise CriticValidationError(
            "Non-finite rewards"
        )

    if not torch.isfinite(values).all():
        raise CriticValidationError(
            "Non-finite values"
        )

    if not torch.isfinite(next_values).all():
        raise CriticValidationError(
            "Non-finite next_values"
        )

    terminated = terminated.to(
        rewards.dtype
    )

    episode_ends = episode_ends.to(
        rewards.dtype
    )

    deltas = (
        rewards
        + gamma
        * (1.0 - terminated)
        * next_values
        - values
    )

    advantages = torch.zeros_like(
        rewards
    )

    running_advantage = (
        torch.zeros_like(
            rewards[0]
        )
    )

    for step in reversed(
        range(rewards.shape[0])
    ):
        continuation = (
            1.0 - episode_ends[step]
        )

        running_advantage = (
            deltas[step]
            + gamma
            * gae_lambda
            * continuation
            * running_advantage
        )

        advantages[step] = (
            running_advantage
        )

    returns = (
        advantages + values
    )

    if normalize:
        mean = advantages.mean()

        std = advantages.std(
            unbiased=False
        )

        advantages = (
            advantages - mean
        ) / (
            std + 1e-8
        )

    return (
        advantages,
        returns,
    )


# ============================================================
# TARGET NETWORK MANAGEMENT
# ============================================================

def hard_update(
    target: nn.Module,
    source: nn.Module,
) -> None:
    """
    Copy all parameters and buffers.
    """

    target.load_state_dict(
        source.state_dict()
    )


@torch.no_grad()
def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float,
) -> None:
    """
    Polyak averaging:

        target = (1-tau)*target + tau*source

    Floating-point parameters and buffers
    are averaged; other buffers are copied.
    """

    if not 0 < tau <= 1:
        raise CriticValidationError(
            "tau must be in (0,1]"
        )

    target_state = (
        target.state_dict()
    )

    source_state = (
        source.state_dict()
    )

    if target_state.keys() != source_state.keys():
        raise CriticValidationError(
            "Target/source state mismatch"
        )

    for name, target_tensor in (
        target_state.items()
    ):
        source_tensor = (
            source_state[name]
        )

        if (
            target_tensor.shape
            != source_tensor.shape
        ):
            raise CriticValidationError(
                f"Shape mismatch: {name}"
            )

        if torch.is_floating_point(
            target_tensor
        ):
            target_tensor.lerp_(
                source_tensor,
                tau,
            )

        else:
            target_tensor.copy_(
                source_tensor
            )


# ============================================================
# CRITIC FACTORY
# ============================================================

def create_critic(
    config: CriticConfig,
) -> nn.Module:
    critic_type = CriticType(
        config.critic_type
    )

    factories = {
        CriticType.STATE_VALUE:
            StateValueCritic,

        CriticType.CONTINUOUS_Q:
            ContinuousQCritic,

        CriticType.DISCRETE_Q:
            DiscreteQCritic,

        CriticType.HYBRID_Q:
            HybridQCritic,

        CriticType.TWIN_Q:
            TwinQCritic,

        CriticType.DISTRIBUTIONAL:
            DistributionalCritic,
    }

    return factories[
        critic_type
    ](config)


# ============================================================
# CRITIC TRAINER
# ============================================================

class CriticTrainer:
    """
    Critic optimization utilities.

    Supports:
        State-value regression
        Continuous Q regression
        Discrete Q regression
        Hybrid Q regression
        Twin Q regression
        Distributional quantile regression

    Bellman target computation is
    kept separate from optimization.
    """

    def __init__(
        self,
        critic: nn.Module,
        config: CriticConfig,
    ) -> None:
        self.critic = critic
        self.config = config

        self.optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.target_critic = copy.deepcopy(
            critic
        )

        self.target_critic.eval()

        for parameter in (
            self.target_critic.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        self.training_steps = 0

    def _optimize(
        self,
        loss: Tensor,
    ) -> dict[str, float]:
        if not torch.isfinite(loss):
            raise CriticError(
                "Non-finite critic loss"
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        grad_norm = (
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        self.training_steps += 1

        soft_update(
            self.target_critic,
            self.critic,
            self.config.tau,
        )

        return {
            "loss": float(
                loss.detach().item()
            ),
            "grad_norm": float(
                grad_norm.detach().item()
            ),
            "training_steps": (
                self.training_steps
            ),
        }

    def train_state_value(
        self,
        states: Tensor,
        target_returns: Tensor,
    ) -> dict[str, float]:
        if not isinstance(
            self.critic,
            StateValueCritic,
        ):
            raise CriticError(
                "StateValueCritic required"
            )

        self.critic.train()

        predicted = self.critic(
            states
        )

        targets = validate_vector(
            target_returns,
            predicted.shape[0],
            "target_returns",
        ).detach()

        loss = F.mse_loss(
            predicted,
            targets,
        )

        return self._optimize(
            loss
        )

    def train_continuous_q(
        self,
        states: Tensor,
        actions: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        if not isinstance(
            self.critic,
            ContinuousQCritic,
        ):
            raise CriticError(
                "ContinuousQCritic required"
            )

        self.critic.train()

        predicted = self.critic(
            states,
            actions,
        )

        targets = validate_vector(
            targets,
            predicted.shape[0],
            "targets",
        ).detach()

        loss = F.mse_loss(
            predicted,
            targets,
        )

        return self._optimize(
            loss
        )

    def train_discrete_q(
        self,
        states: Tensor,
        actions: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        if not isinstance(
            self.critic,
            DiscreteQCritic,
        ):
            raise CriticError(
                "DiscreteQCritic required"
            )

        self.critic.train()

        predicted = (
            self.critic.evaluate_actions(
                states,
                actions,
            )
        )

        targets = validate_vector(
            targets,
            predicted.shape[0],
            "targets",
        ).detach()

        loss = F.mse_loss(
            predicted,
            targets,
        )

        return self._optimize(
            loss
        )

    def train_hybrid_q(
        self,
        states: Tensor,
        continuous_actions: Tensor,
        discrete_actions: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        if not isinstance(
            self.critic,
            HybridQCritic,
        ):
            raise CriticError(
                "HybridQCritic required"
            )

        self.critic.train()

        predicted = self.critic(
            states,
            continuous_actions,
            discrete_actions,
        )

        targets = validate_vector(
            targets,
            predicted.shape[0],
            "targets",
        ).detach()

        loss = F.mse_loss(
            predicted,
            targets,
        )

        return self._optimize(
            loss
        )

    def train_twin_q(
        self,
        states: Tensor,
        actions: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        if not isinstance(
            self.critic,
            TwinQCritic,
        ):
            raise CriticError(
                "TwinQCritic required"
            )

        self.critic.train()

        q1, q2 = self.critic(
            states,
            actions,
        )

        targets = validate_vector(
            targets,
            q1.shape[0],
            "targets",
        ).detach()

        loss_q1 = F.mse_loss(
            q1,
            targets,
        )

        loss_q2 = F.mse_loss(
            q2,
            targets,
        )

        metrics = self._optimize(
            loss_q1 + loss_q2
        )

        metrics.update(
            {
                "q1_loss": float(
                    loss_q1.detach()
                ),
                "q2_loss": float(
                    loss_q2.detach()
                ),
            }
        )

        return metrics

    def train_distributional(
        self,
        states: Tensor,
        actions: Tensor,
        target_quantiles: Tensor,
    ) -> dict[str, float]:
        if not isinstance(
            self.critic,
            DistributionalCritic,
        ):
            raise CriticError(
                "DistributionalCritic required"
            )

        self.critic.train()

        predicted = self.critic(
            states,
            actions,
        )

        loss = quantile_huber_loss(
            predicted,
            target_quantiles,
            self.critic.quantile_fractions,
            kappa=(
                self.config.quantile_huber_kappa
            ),
        )

        return self._optimize(
            loss
        )

    @torch.no_grad()
    def compute_state_td_targets(
        self,
        rewards: Tensor,
        next_states: Tensor,
        terminated: Tensor,
    ) -> Tensor:
        if not isinstance(
            self.target_critic,
            StateValueCritic,
        ):
            raise CriticError(
                "StateValueCritic target required"
            )

        next_values = (
            self.target_critic(
                next_states
            )
        )

        return td_target(
            rewards,
            next_values,
            terminated,
            self.config.gamma,
        )

    @torch.no_grad()
    def compute_twin_q_targets(
        self,
        rewards: Tensor,
        next_states: Tensor,
        next_actions: Tensor,
        terminated: Tensor,
        *,
        next_log_probabilities: Tensor | None = None,
        entropy_temperature: float = 0.0,
    ) -> Tensor:
        """
        SAC-style clipped double-Q target.

        y = r + gamma*(1-done)*
            (min(Q1',Q2') - alpha*log_pi)

        For TD3-style targets, use alpha=0
        and provide appropriately smoothed
        target-policy actions.
        """

        if not isinstance(
            self.target_critic,
            TwinQCritic,
        ):
            raise CriticError(
                "TwinQCritic target required"
            )

        if entropy_temperature < 0:
            raise CriticValidationError(
                "entropy_temperature must be >= 0"
            )

        next_q = (
            self.target_critic.minimum(
                next_states,
                next_actions,
            )
        )

        if next_log_probabilities is not None:
            log_probs = validate_vector(
                next_log_probabilities,
                next_q.shape[0],
                "next_log_probabilities",
            )

            next_q = (
                next_q
                - entropy_temperature
                * log_probs
            )

        elif entropy_temperature != 0:
            raise CriticValidationError(
                "Log probabilities required "
                "when entropy temperature > 0"
            )

        return td_target(
            rewards,
            next_q,
            terminated,
            self.config.gamma,
        )

    def synchronize_target(
        self,
    ) -> None:
        hard_update(
            self.target_critic,
            self.critic,
        )


# ============================================================
# CHECKPOINTS
# ============================================================

def save_critic(
    critic: nn.Module,
    config: CriticConfig,
    path: str | Path,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_data = asdict(
        config
    )

    config_data["critic_type"] = (
        config.critic_type.value
    )

    torch.save(
        {
            "config": config_data,
            "state_dict": critic.state_dict(),
        },
        path,
    )

    return path


def load_critic(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, CriticConfig]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    config = CriticConfig(
        **checkpoint["config"]
    )

    critic = create_critic(
        config
    ).to(device)

    critic.load_state_dict(
        checkpoint["state_dict"]
    )

    critic.eval()

    return critic, config


def save_critic_trainer(
    trainer: CriticTrainer,
    path: str | Path,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_data = asdict(
        trainer.config
    )

    config_data["critic_type"] = (
        trainer.config.critic_type.value
    )

    torch.save(
        {
            "config": config_data,
            "critic_state": (
                trainer.critic.state_dict()
            ),
            "target_state": (
                trainer.target_critic.state_dict()
            ),
            "optimizer_state": (
                trainer.optimizer.state_dict()
            ),
            "training_steps": (
                trainer.training_steps
            ),
        },
        path,
    )

    return path


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)

    print("=" * 60)
    print("NEUROWORLD AI - CRITIC NETWORK")
    print("=" * 60)

    # --------------------------------------------------------
    # STATE VALUE CRITIC
    # --------------------------------------------------------

    value_config = CriticConfig(
        state_dim=32,
        action_dim=4,
        hidden_dim=128,
        critic_type=CriticType.STATE_VALUE,
    )

    value_critic = create_critic(
        value_config
    )

    states = torch.randn(
        16,
        32,
    )

    values = value_critic(
        states
    )

    print(
        "State values shape:",
        values.shape,
    )

    value_trainer = CriticTrainer(
        value_critic,
        value_config,
    )

    returns = torch.randn(
        16
    )

    value_metrics = (
        value_trainer.train_state_value(
            states,
            returns,
        )
    )

    print(
        "Value training:",
        value_metrics,
    )

    # --------------------------------------------------------
    # GAE
    # --------------------------------------------------------

    rewards = torch.randn(
        8,
        4,
    )

    baseline_values = torch.randn(
        8,
        4,
    )

    next_values = torch.randn(
        8,
        4,
    )

    terminated = torch.zeros(
        8,
        4,
    )

    terminated[3, 1] = 1.0

    advantages, lambda_returns = (
        generalized_advantage_estimation(
            rewards,
            baseline_values,
            next_values,
            terminated,
        )
    )

    print(
        "GAE advantages:",
        advantages.shape,
    )

    print(
        "Lambda returns:",
        lambda_returns.shape,
    )

    # --------------------------------------------------------
    # TWIN Q CRITIC
    # --------------------------------------------------------

    twin_config = CriticConfig(
        state_dim=32,
        action_dim=4,
        hidden_dim=128,
        critic_type=CriticType.TWIN_Q,
    )

    twin_critic = create_critic(
        twin_config
    )

    actions = torch.randn(
        16,
        4,
    )

    q1, q2 = twin_critic(
        states,
        actions,
    )

    print(
        "Twin Q shapes:",
        q1.shape,
        q2.shape,
    )

    twin_trainer = CriticTrainer(
        twin_critic,
        twin_config,
    )

    twin_metrics = (
        twin_trainer.train_twin_q(
            states,
            actions,
            torch.randn(16),
        )
    )

    print(
        "Twin Q training:",
        twin_metrics,
    )

    # --------------------------------------------------------
    # DISTRIBUTIONAL CRITIC
    # --------------------------------------------------------

    distributional_config = CriticConfig(
        state_dim=32,
        action_dim=4,
        hidden_dim=128,
        num_quantiles=32,
        critic_type=CriticType.DISTRIBUTIONAL,
    )

    distributional = create_critic(
        distributional_config
    )

    quantiles = distributional(
        states,
        actions,
    )

    print(
        "Quantile output shape:",
        quantiles.shape,
    )

    distributional_trainer = CriticTrainer(
        distributional,
        distributional_config,
    )

    target_quantiles = torch.randn(
        16,
        32,
    )

    distributional_metrics = (
        distributional_trainer.train_distributional(
            states,
            actions,
            target_quantiles,
        )
    )

    print(
        "Distributional training:",
        distributional_metrics,
    )

    # --------------------------------------------------------
    # SAVE AND LOAD
    # --------------------------------------------------------

    checkpoint = save_critic(
        twin_critic,
        twin_config,
        "models/critic_demo.pt",
    )

    loaded_critic, loaded_config = (
        load_critic(
            checkpoint
        )
    )

    print(
        "Loaded critic:",
        type(loaded_critic).__name__,
    )

    print(
        "Loaded type:",
        loaded_config.critic_type.value,
    )

    print(
        "Critic demonstration completed"
    )


if __name__ == "__main__":
    main()
