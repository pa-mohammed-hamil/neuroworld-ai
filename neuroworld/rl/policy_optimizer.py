
"""
NeuroWorld AI
File: neuroworld/rl/policy_optimizer.py

Advanced Reinforcement Learning Policy Optimizer.

Supported algorithms:
    PPO  - Proximal Policy Optimization
    A2C  - Advantage Actor-Critic
    SAC  - Soft Actor-Critic
    TD3  - Twin Delayed DDPG

Features:
    - Continuous-action PPO
    - Discrete-action PPO
    - Hybrid-action PPO
    - A2C policy optimization
    - SAC actor and twin-Q critic optimization
    - TD3 deterministic policy optimization
    - Generalized Advantage Estimation
    - Clipped PPO objectives
    - Entropy regularization
    - Automatic SAC entropy tuning
    - Twin-Q target networks
    - TD3 target policy smoothing
    - Delayed policy updates
    - Gradient clipping
    - Training metrics
    - Checkpoint save/load
    - Reproducible demonstration

Dependencies:
    pip install torch numpy

This implementation uses the existing
NeuroWorld actor and critic interfaces.

Real-world robot commands require an
independent safety controller.
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

from .actor import (
    ActorConfig,
    ActorType,
    ContinuousActor,
    DiscreteActor,
    HybridActor,
    create_actor,
)

from .critic import (
    CriticConfig,
    CriticType,
    StateValueCritic,
    TwinQCritic,
    generalized_advantage_estimation,
    soft_update,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class PolicyOptimizerError(Exception):
    """Base optimizer exception."""


class PolicyValidationError(PolicyOptimizerError):
    """Invalid policy optimization input."""


# ============================================================
# ALGORITHMS
# ============================================================

class PolicyAlgorithm(str, Enum):
    PPO = "ppo"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PolicyOptimizerConfig:
    algorithm: PolicyAlgorithm = PolicyAlgorithm.PPO

    state_dim: int = 128
    action_dim: int = 8
    num_discrete_actions: int = 12

    actor_type: ActorType = ActorType.CONTINUOUS

    hidden_dim: int = 256

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    tau: float = 0.005

    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64

    value_loss_coefficient: float = 0.5
    entropy_coefficient: float = 0.01

    max_grad_norm: float = 1.0

    initial_alpha: float = 0.2
    automatic_entropy_tuning: bool = True
    target_entropy: float | None = None

    td3_policy_delay: int = 2
    td3_target_noise: float = 0.2
    td3_noise_clip: float = 0.5

    action_low: float = -1.0
    action_high: float = 1.0

    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "algorithm",
            PolicyAlgorithm(self.algorithm),
        )

        object.__setattr__(
            self,
            "actor_type",
            ActorType(self.actor_type),
        )

        for name in (
            "state_dim",
            "action_dim",
            "num_discrete_actions",
            "hidden_dim",
            "ppo_epochs",
            "ppo_minibatch_size",
            "td3_policy_delay",
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

        if not 0 < self.ppo_clip_epsilon < 1:
            raise ValueError(
                "Invalid PPO clipping epsilon"
            )

        if self.action_low >= self.action_high:
            raise ValueError(
                "Invalid action bounds"
            )

        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "entropy_learning_rate",
            "max_grad_norm",
            "initial_alpha",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        for name in (
            "value_loss_coefficient",
            "entropy_coefficient",
            "td3_target_noise",
            "td3_noise_clip",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if (
            self.algorithm
            in (PolicyAlgorithm.SAC, PolicyAlgorithm.TD3)
            and self.actor_type != ActorType.CONTINUOUS
        ):
            raise ValueError(
                "SAC and TD3 require continuous actions"
            )


# ============================================================
# ROLLOUT BATCH
# ============================================================

@dataclass
class PolicyBatch:
    states: Tensor
    actions: Tensor | None = None

    rewards: Tensor | None = None
    next_states: Tensor | None = None
    terminated: Tensor | None = None

    old_log_probabilities: Tensor | None = None
    advantages: Tensor | None = None
    returns: Tensor | None = None

    discrete_actions: Tensor | None = None

    weights: Tensor | None = None

    def to(
        self,
        device: torch.device,
    ) -> "PolicyBatch":
        def move(
            value: Tensor | None,
        ) -> Tensor | None:
            if value is None:
                return None

            return value.to(device)

        return PolicyBatch(
            states=move(self.states),
            actions=move(self.actions),
            rewards=move(self.rewards),
            next_states=move(self.next_states),
            terminated=move(self.terminated),
            old_log_probabilities=move(
                self.old_log_probabilities
            ),
            advantages=move(self.advantages),
            returns=move(self.returns),
            discrete_actions=move(
                self.discrete_actions
            ),
            weights=move(self.weights),
        )


# ============================================================
# OPTIMIZATION METRICS
# ============================================================

@dataclass
class PolicyOptimizationMetrics:
    algorithm: str
    actor_loss: float
    critic_loss: float
    entropy: float

    alpha: float = 0.0
    approximate_kl: float = 0.0
    clip_fraction: float = 0.0

    mean_q: float = 0.0
    mean_advantage: float = 0.0

    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0

    actor_updated: bool = True
    training_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# DETERMINISTIC TD3 ACTOR
# ============================================================

class DeterministicActor(nn.Module):
    """
    Deterministic policy for TD3.

    Action range:
        [action_low, action_high]
    """

    def __init__(
        self,
        config: PolicyOptimizerConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                config.state_dim,
                config.hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                config.hidden_dim,
                config.action_dim,
            ),
            nn.Tanh(),
        )

        self.action_scale = (
            config.action_high
            - config.action_low
        ) / 2.0

        self.action_bias = (
            config.action_high
            + config.action_low
        ) / 2.0

    def forward(
        self,
        states: Tensor,
    ) -> Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)

        return (
            self.network(states)
            * self.action_scale
            + self.action_bias
        )


# ============================================================
# MAIN POLICY OPTIMIZER
# ============================================================

class PolicyOptimizer:
    """
    Unified optimization interface.

    PPO/A2C:
        Stochastic actor + state-value critic

    SAC:
        Continuous stochastic actor
        + twin-Q critic

    TD3:
        Deterministic actor
        + twin-Q critic
    """

    def __init__(
        self,
        config: PolicyOptimizerConfig | None = None,
    ) -> None:
        self.config = (
            config or PolicyOptimizerConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.training_steps = 0

        self.actor: nn.Module
        self.critic: nn.Module

        self.target_actor: nn.Module | None = None
        self.target_critic: nn.Module | None = None

        self.log_alpha: Tensor | None = None
        self.alpha_optimizer: (
            torch.optim.Optimizer | None
        ) = None

        self._build_models()

    # --------------------------------------------------------
    # MODEL CONSTRUCTION
    # --------------------------------------------------------

    def _build_models(self) -> None:
        config = self.config

        actor_config = ActorConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            num_discrete_actions=(
                config.num_discrete_actions
            ),
            hidden_dim=config.hidden_dim,
            actor_type=config.actor_type,
            action_low=config.action_low,
            action_high=config.action_high,
            learning_rate=(
                config.actor_learning_rate
            ),
        )

        critic_config = CriticConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            num_discrete_actions=(
                config.num_discrete_actions
            ),
            hidden_dim=config.hidden_dim,
            gamma=config.gamma,
            tau=config.tau,
            critic_type=(
                CriticType.STATE_VALUE
                if config.algorithm in (
                    PolicyAlgorithm.PPO,
                    PolicyAlgorithm.A2C,
                )
                else CriticType.TWIN_Q
            ),
        )

        if config.algorithm == PolicyAlgorithm.TD3:
            self.actor = DeterministicActor(
                config
            ).to(self.device)

        else:
            self.actor = create_actor(
                actor_config
            ).to(self.device)

        if config.algorithm in (
            PolicyAlgorithm.PPO,
            PolicyAlgorithm.A2C,
        ):
            self.critic = StateValueCritic(
                critic_config
            ).to(self.device)

        else:
            self.critic = TwinQCritic(
                critic_config
            ).to(self.device)

            self.target_critic = copy.deepcopy(
                self.critic
            )

            self.target_critic.eval()

            for parameter in (
                self.target_critic.parameters()
            ):
                parameter.requires_grad_(False)

        if config.algorithm == PolicyAlgorithm.TD3:
            self.target_actor = copy.deepcopy(
                self.actor
            )

            self.target_actor.eval()

            for parameter in (
                self.target_actor.parameters()
            ):
                parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.critic_learning_rate,
        )

        if config.algorithm == PolicyAlgorithm.SAC:
            self.log_alpha = torch.tensor(
                math.log(config.initial_alpha),
                device=self.device,
                requires_grad=(
                    config.automatic_entropy_tuning
                ),
            )

            if config.automatic_entropy_tuning:
                self.alpha_optimizer = (
                    torch.optim.Adam(
                        [self.log_alpha],
                        lr=(
                            config.entropy_learning_rate
                        ),
                    )
                )

    # --------------------------------------------------------
    # ALPHA
    # --------------------------------------------------------

    @property
    def alpha(self) -> Tensor:
        if self.log_alpha is None:
            return torch.zeros(
                (),
                device=self.device,
            )

        return self.log_alpha.exp()

    # --------------------------------------------------------
    # POLICY ACTION SELECTION
    # --------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        states = states.to(
            self.device
        )

        if self.config.algorithm == PolicyAlgorithm.TD3:
            return self.actor(
                states
            )

        output = self.actor(
            states,
            deterministic=deterministic,
        )

        if self.config.actor_type == ActorType.CONTINUOUS:
            return output.continuous_action

        if self.config.actor_type == ActorType.DISCRETE:
            return output.discrete_action

        return (
            output.continuous_action,
            output.discrete_action,
        )

    # --------------------------------------------------------
    # ACTION EVALUATION
    # --------------------------------------------------------

    def _evaluate_policy(
        self,
        states: Tensor,
        actions: Tensor | None,
        discrete_actions: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.config.actor_type == ActorType.CONTINUOUS:
            if actions is None:
                raise PolicyValidationError(
                    "Continuous actions required"
                )

            return self.actor.evaluate_actions(
                states,
                actions,
            )

        if self.config.actor_type == ActorType.DISCRETE:
            if discrete_actions is None:
                raise PolicyValidationError(
                    "Discrete actions required"
                )

            return self.actor.evaluate_actions(
                states,
                discrete_actions,
            )

        if (
            actions is None
            or discrete_actions is None
        ):
            raise PolicyValidationError(
                "Hybrid actions required"
            )

        return self.actor.evaluate_actions(
            states,
            actions,
            discrete_actions,
        )

    # --------------------------------------------------------
    # PPO OPTIMIZATION
    # --------------------------------------------------------

    def update_ppo(
        self,
        batch: PolicyBatch,
    ) -> PolicyOptimizationMetrics:
        if self.config.algorithm != PolicyAlgorithm.PPO:
            raise PolicyOptimizerError(
                "Optimizer is not configured for PPO"
            )

        batch = batch.to(
            self.device
        )

        if (
            batch.old_log_probabilities is None
            or batch.advantages is None
            or batch.returns is None
        ):
            raise PolicyValidationError(
                "PPO requires old log probabilities, "
                "advantages and returns"
            )

        states = batch.states

        old_log_probs = (
            batch.old_log_probabilities
            .detach()
            .reshape(-1)
        )

        advantages = (
            batch.advantages
            .detach()
            .reshape(-1)
        )

        returns = (
            batch.returns
            .detach()
            .reshape(-1)
        )

        if not (
            states.shape[0]
            == old_log_probs.shape[0]
            == advantages.shape[0]
            == returns.shape[0]
        ):
            raise PolicyValidationError(
                "PPO batch size mismatch"
            )

        advantages = (
            advantages
            - advantages.mean()
        ) / (
            advantages.std(
                unbiased=False
            )
            + 1e-8
        )

        batch_size = states.shape[0]

        actor_losses = []
        critic_losses = []
        entropies = []
        approximate_kls = []
        clip_fractions = []

        actor_grad_norm = 0.0
        critic_grad_norm = 0.0

        for _ in range(
            self.config.ppo_epochs
        ):
            permutation = torch.randperm(
                batch_size,
                device=self.device,
            )

            for start in range(
                0,
                batch_size,
                self.config.ppo_minibatch_size,
            ):
                indices = permutation[
                    start:
                    start
                    + self.config.ppo_minibatch_size
                ]

                mb_states = states[indices]

                mb_actions = (
                    batch.actions[indices]
                    if batch.actions is not None
                    else None
                )

                mb_discrete_actions = (
                    batch.discrete_actions[indices]
                    if batch.discrete_actions is not None
                    else None
                )

                mb_old_log_probs = (
                    old_log_probs[indices]
                )

                mb_advantages = (
                    advantages[indices]
                )

                mb_returns = (
                    returns[indices]
                )

                (
                    new_log_probs,
                    entropy,
                ) = self._evaluate_policy(
                    mb_states,
                    mb_actions,
                    mb_discrete_actions,
                )

                log_ratio = (
                    new_log_probs
                    - mb_old_log_probs
                )

                ratio = torch.exp(
                    log_ratio
                )

                surrogate_1 = (
                    ratio * mb_advantages
                )

                surrogate_2 = (
                    torch.clamp(
                        ratio,
                        1.0
                        - self.config.ppo_clip_epsilon,
                        1.0
                        + self.config.ppo_clip_epsilon,
                    )
                    * mb_advantages
                )

                actor_loss = -torch.minimum(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                entropy_mean = entropy.mean()

                actor_objective = (
                    actor_loss
                    - self.config.entropy_coefficient
                    * entropy_mean
                )

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                actor_objective.backward()

                actor_norm = (
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        self.config.max_grad_norm,
                    )
                )

                self.actor_optimizer.step()

                predicted_values = self.critic(
                    mb_states
                )

                critic_loss = F.mse_loss(
                    predicted_values,
                    mb_returns,
                )

                self.critic_optimizer.zero_grad(
                    set_to_none=True
                )

                (
                    self.config.value_loss_coefficient
                    * critic_loss
                ).backward()

                critic_norm = (
                    nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        self.config.max_grad_norm,
                    )
                )

                self.critic_optimizer.step()

                with torch.no_grad():
                    approximate_kl = (
                        ratio
                        - 1.0
                        - log_ratio
                    ).mean()

                    clip_fraction = (
                        (
                            (ratio - 1.0).abs()
                            > self.config.ppo_clip_epsilon
                        )
                        .float()
                        .mean()
                    )

                actor_losses.append(
                    actor_loss.detach()
                )

                critic_losses.append(
                    critic_loss.detach()
                )

                entropies.append(
                    entropy_mean.detach()
                )

                approximate_kls.append(
                    approximate_kl.detach()
                )

                clip_fractions.append(
                    clip_fraction.detach()
                )

                actor_grad_norm = float(
                    actor_norm.detach()
                )

                critic_grad_norm = float(
                    critic_norm.detach()
                )

        self.training_steps += 1

        def average(
            values: list[Tensor],
        ) -> float:
            return float(
                torch.stack(
                    values
                ).mean().item()
            )

        return PolicyOptimizationMetrics(
            algorithm="ppo",
            actor_loss=average(
                actor_losses
            ),
            critic_loss=average(
                critic_losses
            ),
            entropy=average(
                entropies
            ),
            approximate_kl=average(
                approximate_kls
            ),
            clip_fraction=average(
                clip_fractions
            ),
            mean_advantage=float(
                advantages.mean()
            ),
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            training_step=self.training_steps,
        )

    # --------------------------------------------------------
    # A2C OPTIMIZATION
    # --------------------------------------------------------

    def update_a2c(
        self,
        batch: PolicyBatch,
    ) -> PolicyOptimizationMetrics:
        if self.config.algorithm != PolicyAlgorithm.A2C:
            raise PolicyOptimizerError(
                "Optimizer is not configured for A2C"
            )

        batch = batch.to(
            self.device
        )

        if (
            batch.advantages is None
            or batch.returns is None
        ):
            raise PolicyValidationError(
                "A2C requires advantages and returns"
            )

        advantages = (
            batch.advantages
            .detach()
            .reshape(-1)
        )

        returns = (
            batch.returns
            .detach()
            .reshape(-1)
        )

        log_probs, entropy = (
            self._evaluate_policy(
                batch.states,
                batch.actions,
                batch.discrete_actions,
            )
        )

        actor_loss = -(
            log_probs * advantages
        ).mean()

        entropy_mean = entropy.mean()

        actor_objective = (
            actor_loss
            - self.config.entropy_coefficient
            * entropy_mean
        )

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_objective.backward()

        actor_norm = (
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.actor_optimizer.step()

        values = self.critic(
            batch.states
        )

        critic_loss = F.mse_loss(
            values,
            returns,
        )

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        (
            self.config.value_loss_coefficient
            * critic_loss
        ).backward()

        critic_norm = (
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.critic_optimizer.step()

        self.training_steps += 1

        return PolicyOptimizationMetrics(
            algorithm="a2c",
            actor_loss=float(
                actor_loss.detach()
            ),
            critic_loss=float(
                critic_loss.detach()
            ),
            entropy=float(
                entropy_mean.detach()
            ),
            mean_advantage=float(
                advantages.mean()
            ),
            actor_grad_norm=float(
                actor_norm.detach()
            ),
            critic_grad_norm=float(
                critic_norm.detach()
            ),
            training_step=self.training_steps,
        )

    # --------------------------------------------------------
    # SAC OPTIMIZATION
    # --------------------------------------------------------

    def update_sac(
        self,
        batch: PolicyBatch,
    ) -> PolicyOptimizationMetrics:
        if self.config.algorithm != PolicyAlgorithm.SAC:
            raise PolicyOptimizerError(
                "Optimizer is not configured for SAC"
            )

        batch = batch.to(
            self.device
        )

        if any(
            value is None
            for value in (
                batch.actions,
                batch.rewards,
                batch.next_states,
                batch.terminated,
            )
        ):
            raise PolicyValidationError(
                "SAC requires complete transitions"
            )

        states = batch.states
        actions = batch.actions

        rewards = batch.rewards.reshape(-1)
        terminated = batch.terminated.reshape(-1)
        next_states = batch.next_states

        weights = (
            batch.weights.reshape(-1)
            if batch.weights is not None
            else torch.ones_like(rewards)
        )

        with torch.no_grad():
            next_output = self.actor(
                next_states
            )

            next_actions = (
                next_output.continuous_action
            )

            next_log_probs = (
                next_output.log_probability
            )

            next_q = (
                self.target_critic.minimum(
                    next_states,
                    next_actions,
                )
            )

            target_values = (
                next_q
                - self.alpha.detach()
                * next_log_probs
            )

            bellman_targets = (
                rewards
                + self.config.gamma
                * (1.0 - terminated)
                * target_values
            )

        q1, q2 = self.critic(
            states,
            actions,
        )

        critic_loss = (
            weights
            * (
                (q1 - bellman_targets).square()
                + (q2 - bellman_targets).square()
            )
        ).mean()

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        critic_norm = (
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.critic_optimizer.step()

        # Freeze critic parameters while
        # preserving action gradients.
        for parameter in (
            self.critic.parameters()
        ):
            parameter.requires_grad_(False)

        try:
            output = self.actor(
                states
            )

            policy_actions = (
                output.continuous_action
            )

            log_probs = (
                output.log_probability
            )

            policy_q = (
                self.critic.minimum(
                    states,
                    policy_actions,
                )
            )

            actor_loss = (
                self.alpha.detach()
                * log_probs
                - policy_q
            ).mean()

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            actor_norm = (
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.config.max_grad_norm,
                )
            )

            self.actor_optimizer.step()

        finally:
            for parameter in (
                self.critic.parameters()
            ):
                parameter.requires_grad_(True)

        if (
            self.config.automatic_entropy_tuning
            and self.alpha_optimizer is not None
        ):
            target_entropy = (
                self.config.target_entropy
                if self.config.target_entropy is not None
                else -float(
                    self.config.action_dim
                )
            )

            alpha_loss = -(
                self.log_alpha
                * (
                    log_probs.detach()
                    + target_entropy
                )
            ).mean()

            self.alpha_optimizer.zero_grad(
                set_to_none=True
            )

            alpha_loss.backward()

            self.alpha_optimizer.step()

        soft_update(
            self.target_critic,
            self.critic,
            self.config.tau,
        )

        self.training_steps += 1

        return PolicyOptimizationMetrics(
            algorithm="sac",
            actor_loss=float(
                actor_loss.detach()
            ),
            critic_loss=float(
                critic_loss.detach()
            ),
            entropy=float(
                (-log_probs).mean().detach()
            ),
            alpha=float(
                self.alpha.detach()
            ),
            mean_q=float(
                policy_q.mean().detach()
            ),
            actor_grad_norm=float(
                actor_norm.detach()
            ),
            critic_grad_norm=float(
                critic_norm.detach()
            ),
            training_step=self.training_steps,
        )

    # --------------------------------------------------------
    # TD3 OPTIMIZATION
    # --------------------------------------------------------

    def update_td3(
        self,
        batch: PolicyBatch,
    ) -> PolicyOptimizationMetrics:
        if self.config.algorithm != PolicyAlgorithm.TD3:
            raise PolicyOptimizerError(
                "Optimizer is not configured for TD3"
            )

        batch = batch.to(
            self.device
        )

        if any(
            value is None
            for value in (
                batch.actions,
                batch.rewards,
                batch.next_states,
                batch.terminated,
            )
        ):
            raise PolicyValidationError(
                "TD3 requires complete transitions"
            )

        states = batch.states
        actions = batch.actions
        rewards = batch.rewards.reshape(-1)
        terminated = batch.terminated.reshape(-1)
        next_states = batch.next_states

        with torch.no_grad():
            next_actions = self.target_actor(
                next_states
            )

            noise = (
                torch.randn_like(
                    next_actions
                )
                * self.config.td3_target_noise
            )

            noise = noise.clamp(
                -self.config.td3_noise_clip,
                self.config.td3_noise_clip,
            )

            next_actions = (
                next_actions + noise
            ).clamp(
                self.config.action_low,
                self.config.action_high,
            )

            next_q = (
                self.target_critic.minimum(
                    next_states,
                    next_actions,
                )
            )

            targets = (
                rewards
                + self.config.gamma
                * (1.0 - terminated)
                * next_q
            )

        q1, q2 = self.critic(
            states,
            actions,
        )

        weights = (
            batch.weights.reshape(-1)
            if batch.weights is not None
            else torch.ones_like(rewards)
        )

        critic_loss = (
            weights
            * (
                (q1 - targets).square()
                + (q2 - targets).square()
            )
        ).mean()

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        critic_norm = (
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.critic_optimizer.step()

        self.training_steps += 1

        actor_updated = (
            self.training_steps
            % self.config.td3_policy_delay
            == 0
        )

        actor_loss = torch.zeros(
            (),
            device=self.device,
        )

        actor_norm = torch.zeros(
            (),
            device=self.device,
        )

        if actor_updated:
            for parameter in (
                self.critic.parameters()
            ):
                parameter.requires_grad_(False)

            try:
                policy_actions = self.actor(
                    states
                )

                actor_loss = -self.critic.q1(
                    states,
                    policy_actions,
                ).mean()

                self.actor_optimizer.zero_grad(
                    set_to_none=True
                )

                actor_loss.backward()

                actor_norm = (
                    nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        self.config.max_grad_norm,
                    )
                )

                self.actor_optimizer.step()

            finally:
                for parameter in (
                    self.critic.parameters()
                ):
                    parameter.requires_grad_(True)

            soft_update(
                self.target_actor,
                self.actor,
                self.config.tau,
            )

            soft_update(
                self.target_critic,
                self.critic,
                self.config.tau,
            )

        return PolicyOptimizationMetrics(
            algorithm="td3",
            actor_loss=float(
                actor_loss.detach()
            ),
            critic_loss=float(
                critic_loss.detach()
            ),
            entropy=0.0,
            mean_q=float(
                torch.minimum(
                    q1.detach(),
                    q2.detach(),
                ).mean()
            ),
            actor_grad_norm=float(
                actor_norm.detach()
            ),
            critic_grad_norm=float(
                critic_norm.detach()
            ),
            actor_updated=actor_updated,
            training_step=self.training_steps,
        )

    # --------------------------------------------------------
    # UNIFIED UPDATE
    # --------------------------------------------------------

    def update(
        self,
        batch: PolicyBatch,
    ) -> PolicyOptimizationMetrics:
        if self.config.algorithm == PolicyAlgorithm.PPO:
            return self.update_ppo(batch)

        if self.config.algorithm == PolicyAlgorithm.A2C:
            return self.update_a2c(batch)

        if self.config.algorithm == PolicyAlgorithm.SAC:
            return self.update_sac(batch)

        if self.config.algorithm == PolicyAlgorithm.TD3:
            return self.update_td3(batch)

        raise PolicyOptimizerError(
            "Unsupported algorithm"
        )

    # --------------------------------------------------------
    # GAE HELPER
    # --------------------------------------------------------

    @torch.no_grad()
    def calculate_advantages(
        self,
        rewards: Tensor,
        states: Tensor,
        next_states: Tensor,
        terminated: Tensor,
        *,
        episode_ends: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.config.algorithm not in (
            PolicyAlgorithm.PPO,
            PolicyAlgorithm.A2C,
        ):
            raise PolicyOptimizerError(
                "GAE requires a state-value critic"
            )

        values = self.critic(
            states.to(self.device)
        )

        next_values = self.critic(
            next_states.to(self.device)
        )

        return generalized_advantage_estimation(
            rewards.to(self.device),
            values,
            next_values,
            terminated.to(self.device),
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            episode_ends=(
                episode_ends.to(self.device)
                if episode_ends is not None
                else None
            ),
        )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> Path:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload: dict[str, Any] = {
            "config": {
                **asdict(self.config),
                "algorithm": (
                    self.config.algorithm.value
                ),
                "actor_type": (
                    self.config.actor_type.value
                ),
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": (
                self.actor_optimizer.state_dict()
            ),
            "critic_optimizer": (
                self.critic_optimizer.state_dict()
            ),
            "training_steps": (
                self.training_steps
            ),
        }

        if self.target_actor is not None:
            payload["target_actor"] = (
                self.target_actor.state_dict()
            )

        if self.target_critic is not None:
            payload["target_critic"] = (
                self.target_critic.state_dict()
            )

        if self.log_alpha is not None:
            payload["log_alpha"] = (
                self.log_alpha.detach().clone()
            )

        if self.alpha_optimizer is not None:
            payload["alpha_optimizer"] = (
                self.alpha_optimizer.state_dict()
            )

        torch.save(
            payload,
            path,
        )

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "PolicyOptimizer":
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        config_data = dict(
            checkpoint["config"]
        )

        config_data["device"] = device

        optimizer = cls(
            PolicyOptimizerConfig(
                **config_data
            )
        )

        optimizer.actor.load_state_dict(
            checkpoint["actor"]
        )

        optimizer.critic.load_state_dict(
            checkpoint["critic"]
        )

        optimizer.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        optimizer.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        if (
            optimizer.target_actor is not None
            and "target_actor" in checkpoint
        ):
            optimizer.target_actor.load_state_dict(
                checkpoint["target_actor"]
            )

        if (
            optimizer.target_critic is not None
            and "target_critic" in checkpoint
        ):
            optimizer.target_critic.load_state_dict(
                checkpoint["target_critic"]
            )

        if (
            optimizer.log_alpha is not None
            and "log_alpha" in checkpoint
        ):
            with torch.no_grad():
                optimizer.log_alpha.copy_(
                    checkpoint["log_alpha"]
                )

        if (
            optimizer.alpha_optimizer is not None
            and "alpha_optimizer" in checkpoint
        ):
            optimizer.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer"]
            )

        optimizer.training_steps = int(
            checkpoint["training_steps"]
        )

        return optimizer


# ============================================================
# FACTORY
# ============================================================

def create_policy_optimizer(
    config: PolicyOptimizerConfig | None = None,
) -> PolicyOptimizer:
    return PolicyOptimizer(config)


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - POLICY OPTIMIZER"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # PPO DEMONSTRATION
    # --------------------------------------------------------

    ppo_config = PolicyOptimizerConfig(
        algorithm=PolicyAlgorithm.PPO,
        state_dim=16,
        action_dim=4,
        hidden_dim=128,
        ppo_epochs=2,
        ppo_minibatch_size=16,
    )

    ppo = PolicyOptimizer(
        ppo_config
    )

    states = torch.randn(
        32,
        16,
    )

    with torch.no_grad():
        rollout = ppo.actor(
            states
        )

    ppo_batch = PolicyBatch(
        states=states,
        actions=(
            rollout.continuous_action.detach()
        ),
        old_log_probabilities=(
            rollout.log_probability.detach()
        ),
        advantages=torch.randn(32),
        returns=torch.randn(32),
    )

    ppo_metrics = ppo.update(
        ppo_batch
    )

    print(
        "PPO:",
        ppo_metrics.to_dict(),
    )

    # --------------------------------------------------------
    # SAC DEMONSTRATION
    # --------------------------------------------------------

    sac_config = PolicyOptimizerConfig(
        algorithm=PolicyAlgorithm.SAC,
        state_dim=16,
        action_dim=4,
        hidden_dim=128,
    )

    sac = PolicyOptimizer(
        sac_config
    )

    sac_batch = PolicyBatch(
        states=torch.randn(32, 16),
        actions=torch.rand(32, 4) * 2 - 1,
        rewards=torch.randn(32),
        next_states=torch.randn(32, 16),
        terminated=torch.zeros(32),
    )

    sac_metrics = sac.update(
        sac_batch
    )

    print(
        "SAC:",
        sac_metrics.to_dict(),
    )

    # --------------------------------------------------------
    # TD3 DEMONSTRATION
    # --------------------------------------------------------

    td3_config = PolicyOptimizerConfig(
        algorithm=PolicyAlgorithm.TD3,
        state_dim=16,
        action_dim=4,
        hidden_dim=128,
    )

    td3 = PolicyOptimizer(
        td3_config
    )

    td3_batch = PolicyBatch(
        states=torch.randn(32, 16),
        actions=torch.rand(32, 4) * 2 - 1,
        rewards=torch.randn(32),
        next_states=torch.randn(32, 16),
        terminated=torch.zeros(32),
    )

    for _ in range(2):
        td3_metrics = td3.update(
            td3_batch
        )

    print(
        "TD3:",
        td3_metrics.to_dict(),
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = ppo.save(
        "models/policy_optimizer_demo.pt"
    )

    restored = PolicyOptimizer.load(
        checkpoint
    )

    print(
        "Restored algorithm:",
        restored.config.algorithm.value,
    )

    print(
        "Policy optimizer demonstration completed"
    )


if __name__ == "__main__":
    main()
