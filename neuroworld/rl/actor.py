
"""
NeuroWorld AI
File: neuroworld/rl/actor.py

Actor Networks for Model-Based Reinforcement Learning.

Features:
- Continuous stochastic actor (Squashed Gaussian)
- Discrete categorical actor
- Hybrid continuous/discrete actor
- Deterministic inference
- Stochastic action sampling
- Tanh action squashing with corrected log-probability
- Configurable action bounds
- Entropy estimation
- Policy-gradient optimization
- PPO clipped objective
- Gradient clipping
- Actor checkpoint save/load
- Batch inference
- Reproducible demo

Dependencies:
    pip install torch numpy

Action conventions:
    Continuous: [batch, action_dim]
    Discrete:   [batch]
    Hybrid:     continuous + discrete

Safety:
    Actor outputs are proposed actions only.
    A separate safety controller must validate
    actions before physical execution.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import (
    Categorical,
    Normal,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ActorError(Exception):
    """Base actor exception."""


class ActorValidationError(ActorError):
    """Invalid actor input or configuration."""


# ============================================================
# ACTION TYPES
# ============================================================

class ActorType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ActorConfig:
    state_dim: int = 128

    action_dim: int = 8
    num_discrete_actions: int = 12

    hidden_dim: int = 256
    num_hidden_layers: int = 3

    actor_type: ActorType = ActorType.CONTINUOUS

    log_std_min: float = -5.0
    log_std_max: float = 2.0

    action_low: float = -1.0
    action_high: float = 1.0

    learning_rate: float = 3e-4
    weight_decay: float = 0.0

    max_grad_norm: float = 1.0
    ppo_clip_epsilon: float = 0.2

    entropy_coefficient: float = 0.01

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError(
                "state_dim must be positive"
            )

        if self.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if self.num_discrete_actions <= 1:
            raise ValueError(
                "num_discrete_actions must exceed 1"
            )

        if (
            self.hidden_dim <= 0
            or self.num_hidden_layers <= 0
        ):
            raise ValueError(
                "Invalid network dimensions"
            )

        if self.log_std_min >= self.log_std_max:
            raise ValueError(
                "Invalid log_std range"
            )

        if self.action_low >= self.action_high:
            raise ValueError(
                "Invalid action bounds"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if not 0 < self.ppo_clip_epsilon < 1:
            raise ValueError(
                "Invalid PPO clip epsilon"
            )


# ============================================================
# OUTPUT STRUCTURES
# ============================================================

@dataclass
class ActorOutput:
    continuous_action: Tensor | None
    discrete_action: Tensor | None

    log_probability: Tensor
    entropy: Tensor

    mean: Tensor | None = None
    log_std: Tensor | None = None
    logits: Tensor | None = None

    def detach(self) -> "ActorOutput":
        def maybe_detach(
            tensor: Tensor | None,
        ) -> Tensor | None:
            return (
                tensor.detach()
                if tensor is not None
                else None
            )

        return ActorOutput(
            continuous_action=maybe_detach(
                self.continuous_action
            ),
            discrete_action=maybe_detach(
                self.discrete_action
            ),
            log_probability=(
                self.log_probability.detach()
            ),
            entropy=self.entropy.detach(),
            mean=maybe_detach(self.mean),
            log_std=maybe_detach(self.log_std),
            logits=maybe_detach(self.logits),
        )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_states(
    states: Tensor,
    state_dim: int,
) -> Tensor:
    if not isinstance(states, Tensor):
        raise ActorValidationError(
            "states must be a torch.Tensor"
        )

    if states.ndim == 1:
        states = states.unsqueeze(0)

    if (
        states.ndim != 2
        or states.shape[-1] != state_dim
    ):
        raise ActorValidationError(
            f"Expected states [B,{state_dim}]"
        )

    if not torch.is_floating_point(states):
        raise ActorValidationError(
            "states must be floating point"
        )

    if not torch.isfinite(states).all():
        raise ActorValidationError(
            "states contain non-finite values"
        )

    return states


# ============================================================
# MLP BACKBONE
# ============================================================

class ActorBackbone(nn.Module):
    def __init__(
        self,
        config: ActorConfig,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []

        input_dim = config.state_dim

        for _ in range(
            config.num_hidden_layers
        ):
            layers.extend(
                [
                    nn.Linear(
                        input_dim,
                        config.hidden_dim,
                    ),
                    nn.LayerNorm(
                        config.hidden_dim
                    ),
                    nn.SiLU(),
                ]
            )

            input_dim = config.hidden_dim

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        states: Tensor,
    ) -> Tensor:
        return self.network(
            states
        )


# ============================================================
# SQUASHED GAUSSIAN DISTRIBUTION
# ============================================================

class SquashedGaussian:
    """
    Gaussian transformed by tanh and
    scaled to configured action bounds.

    log_prob uses the change-of-variables
    correction for tanh and scaling.
    """

    def __init__(
        self,
        mean: Tensor,
        log_std: Tensor,
        action_low: float,
        action_high: float,
    ) -> None:
        self.mean = mean
        self.log_std = log_std

        self.std = log_std.exp()

        self.normal = Normal(
            mean,
            self.std,
        )

        self.scale = (
            action_high - action_low
        ) / 2.0

        self.bias = (
            action_high + action_low
        ) / 2.0

    def _transform(
        self,
        raw_action: Tensor,
    ) -> Tensor:
        return (
            torch.tanh(raw_action)
            * self.scale
            + self.bias
        )

    def _log_prob_raw(
        self,
        raw_action: Tensor,
    ) -> Tensor:
        # Stable tanh Jacobian:
        # log(1 - tanh(x)^2)
        # = 2(log(2) - x - softplus(-2x))
        correction = (
            2.0
            * (
                math.log(2.0)
                - raw_action
                - nn.functional.softplus(
                    -2.0 * raw_action
                )
            )
        )

        log_prob = (
            self.normal.log_prob(
                raw_action
            )
            - correction
            - math.log(self.scale)
        )

        return log_prob.sum(
            dim=-1
        )

    def sample(
        self,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if deterministic:
            raw_action = self.mean

        else:
            raw_action = self.normal.rsample()

        action = self._transform(
            raw_action
        )

        log_probability = (
            self._log_prob_raw(
                raw_action
            )
        )

        return (
            action,
            log_probability,
        )

    def log_prob(
        self,
        action: Tensor,
    ) -> Tensor:
        normalized = (
            action - self.bias
        ) / self.scale

        normalized = normalized.clamp(
            -1 + 1e-6,
            1 - 1e-6,
        )

        raw_action = torch.atanh(
            normalized
        )

        return self._log_prob_raw(
            raw_action
        )

    def entropy_estimate(
        self,
        samples: int = 1,
    ) -> Tensor:
        """
        Monte Carlo entropy estimate.

        Squashed Gaussian entropy has
        no simple closed form.
        """

        if samples <= 0:
            raise ActorValidationError(
                "samples must be positive"
            )

        entropies = []

        for _ in range(samples):
            _, log_probability = (
                self.sample()
            )

            entropies.append(
                -log_probability
            )

        return torch.stack(
            entropies
        ).mean(dim=0)


# ============================================================
# CONTINUOUS ACTOR
# ============================================================

class ContinuousActor(nn.Module):
    def __init__(
        self,
        config: ActorConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.backbone = ActorBackbone(
            config
        )

        self.mean_head = nn.Linear(
            config.hidden_dim,
            config.action_dim,
        )

        self.log_std_head = nn.Linear(
            config.hidden_dim,
            config.action_dim,
        )

    def distribution(
        self,
        states: Tensor,
    ) -> tuple[
        SquashedGaussian,
        Tensor,
        Tensor,
    ]:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        hidden = self.backbone(
            states
        )

        mean = self.mean_head(
            hidden
        )

        log_std = self.log_std_head(
            hidden
        ).clamp(
            self.config.log_std_min,
            self.config.log_std_max,
        )

        distribution = SquashedGaussian(
            mean=mean,
            log_std=log_std,
            action_low=(
                self.config.action_low
            ),
            action_high=(
                self.config.action_high
            ),
        )

        return (
            distribution,
            mean,
            log_std,
        )

    def forward(
        self,
        states: Tensor,
        deterministic: bool = False,
    ) -> ActorOutput:
        distribution, mean, log_std = (
            self.distribution(
                states
            )
        )

        action, log_probability = (
            distribution.sample(
                deterministic=deterministic
            )
        )

        entropy = (
            distribution.entropy_estimate()
        )

        return ActorOutput(
            continuous_action=action,
            discrete_action=None,
            log_probability=log_probability,
            entropy=entropy,
            mean=mean,
            log_std=log_std,
        )

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        distribution, _, _ = (
            self.distribution(
                states
            )
        )

        log_probability = (
            distribution.log_prob(
                actions
            )
        )

        entropy = (
            distribution.entropy_estimate()
        )

        return (
            log_probability,
            entropy,
        )


# ============================================================
# DISCRETE ACTOR
# ============================================================

class DiscreteActor(nn.Module):
    def __init__(
        self,
        config: ActorConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.backbone = ActorBackbone(
            config
        )

        self.logits_head = nn.Linear(
            config.hidden_dim,
            config.num_discrete_actions,
        )

    def distribution(
        self,
        states: Tensor,
    ) -> tuple[Categorical, Tensor]:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        hidden = self.backbone(
            states
        )

        logits = self.logits_head(
            hidden
        )

        return (
            Categorical(logits=logits),
            logits,
        )

    def forward(
        self,
        states: Tensor,
        deterministic: bool = False,
    ) -> ActorOutput:
        distribution, logits = (
            self.distribution(
                states
            )
        )

        if deterministic:
            action = logits.argmax(
                dim=-1
            )

        else:
            action = distribution.sample()

        return ActorOutput(
            continuous_action=None,
            discrete_action=action,
            log_probability=(
                distribution.log_prob(
                    action
                )
            ),
            entropy=distribution.entropy(),
            logits=logits,
        )

    def evaluate_actions(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        distribution, _ = (
            self.distribution(
                states
            )
        )

        return (
            distribution.log_prob(
                actions.long()
            ),
            distribution.entropy(),
        )


# ============================================================
# HYBRID ACTOR
# ============================================================

class HybridActor(nn.Module):
    """
    Shared state backbone with independent
    continuous and discrete action heads.

    The joint log probability is the sum
    of both branch log probabilities.
    """

    def __init__(
        self,
        config: ActorConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.backbone = ActorBackbone(
            config
        )

        self.mean_head = nn.Linear(
            config.hidden_dim,
            config.action_dim,
        )

        self.log_std_head = nn.Linear(
            config.hidden_dim,
            config.action_dim,
        )

        self.logits_head = nn.Linear(
            config.hidden_dim,
            config.num_discrete_actions,
        )

    def distributions(
        self,
        states: Tensor,
    ) -> tuple[
        SquashedGaussian,
        Categorical,
        Tensor,
        Tensor,
        Tensor,
    ]:
        states = validate_states(
            states,
            self.config.state_dim,
        )

        hidden = self.backbone(
            states
        )

        mean = self.mean_head(
            hidden
        )

        log_std = self.log_std_head(
            hidden
        ).clamp(
            self.config.log_std_min,
            self.config.log_std_max,
        )

        logits = self.logits_head(
            hidden
        )

        continuous = SquashedGaussian(
            mean,
            log_std,
            self.config.action_low,
            self.config.action_high,
        )

        discrete = Categorical(
            logits=logits
        )

        return (
            continuous,
            discrete,
            mean,
            log_std,
            logits,
        )

    def forward(
        self,
        states: Tensor,
        deterministic: bool = False,
    ) -> ActorOutput:
        (
            continuous,
            discrete,
            mean,
            log_std,
            logits,
        ) = self.distributions(
            states
        )

        continuous_action, continuous_log_prob = (
            continuous.sample(
                deterministic=deterministic
            )
        )

        if deterministic:
            discrete_action = logits.argmax(
                dim=-1
            )

        else:
            discrete_action = discrete.sample()

        discrete_log_prob = (
            discrete.log_prob(
                discrete_action
            )
        )

        entropy = (
            continuous.entropy_estimate()
            + discrete.entropy()
        )

        return ActorOutput(
            continuous_action=continuous_action,
            discrete_action=discrete_action,
            log_probability=(
                continuous_log_prob
                + discrete_log_prob
            ),
            entropy=entropy,
            mean=mean,
            log_std=log_std,
            logits=logits,
        )

    def evaluate_actions(
        self,
        states: Tensor,
        continuous_actions: Tensor,
        discrete_actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        (
            continuous,
            discrete,
            _,
            _,
            _,
        ) = self.distributions(
            states
        )

        log_probability = (
            continuous.log_prob(
                continuous_actions
            )
            + discrete.log_prob(
                discrete_actions.long()
            )
        )

        entropy = (
            continuous.entropy_estimate()
            + discrete.entropy()
        )

        return (
            log_probability,
            entropy,
        )


# ============================================================
# ACTOR FACTORY
# ============================================================

def create_actor(
    config: ActorConfig,
) -> nn.Module:
    actor_type = ActorType(
        config.actor_type
    )

    if actor_type == ActorType.CONTINUOUS:
        return ContinuousActor(
            config
        )

    if actor_type == ActorType.DISCRETE:
        return DiscreteActor(
            config
        )

    if actor_type == ActorType.HYBRID:
        return HybridActor(
            config
        )

    raise ActorValidationError(
        f"Unsupported actor type: {actor_type}"
    )


# ============================================================
# PPO TRAINER
# ============================================================

class ActorTrainer:
    """
    PPO-style actor optimization.

    Required rollout data:
        states
        actions
        old_log_probabilities
        advantages

    Advantages should be computed
    externally by GAE or another
    advantage estimator.
    """

    def __init__(
        self,
        actor: nn.Module,
        config: ActorConfig,
    ) -> None:
        self.actor = actor
        self.config = config

        self.optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train_step(
        self,
        states: Tensor,
        old_log_probabilities: Tensor,
        advantages: Tensor,
        *,
        continuous_actions: Tensor | None = None,
        discrete_actions: Tensor | None = None,
    ) -> dict[str, float]:
        self.actor.train()

        states = validate_states(
            states,
            self.config.state_dim,
        )

        batch_size = states.shape[0]

        old_log_probabilities = (
            old_log_probabilities.reshape(-1)
        )

        advantages = advantages.reshape(-1)

        if (
            old_log_probabilities.shape[0]
            != batch_size
            or advantages.shape[0]
            != batch_size
        ):
            raise ActorValidationError(
                "Rollout batch size mismatch"
            )

        actor_type = ActorType(
            self.config.actor_type
        )

        if actor_type == ActorType.CONTINUOUS:
            if continuous_actions is None:
                raise ActorValidationError(
                    "Continuous actions required"
                )

            new_log_prob, entropy = (
                self.actor.evaluate_actions(
                    states,
                    continuous_actions,
                )
            )

        elif actor_type == ActorType.DISCRETE:
            if discrete_actions is None:
                raise ActorValidationError(
                    "Discrete actions required"
                )

            new_log_prob, entropy = (
                self.actor.evaluate_actions(
                    states,
                    discrete_actions,
                )
            )

        else:
            if (
                continuous_actions is None
                or discrete_actions is None
            ):
                raise ActorValidationError(
                    "Both action types required"
                )

            new_log_prob, entropy = (
                self.actor.evaluate_actions(
                    states,
                    continuous_actions,
                    discrete_actions,
                )
            )

        # PPO importance sampling ratio.
        log_ratio = (
            new_log_prob
            - old_log_probabilities.detach()
        )

        ratio = torch.exp(
            log_ratio
        )

        surrogate_1 = (
            ratio * advantages.detach()
        )

        surrogate_2 = (
            torch.clamp(
                ratio,
                1.0
                - self.config.ppo_clip_epsilon,
                1.0
                + self.config.ppo_clip_epsilon,
            )
            * advantages.detach()
        )

        policy_loss = -torch.minimum(
            surrogate_1,
            surrogate_2,
        ).mean()

        entropy_bonus = entropy.mean()

        total_loss = (
            policy_loss
            - self.config.entropy_coefficient
            * entropy_bonus
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        grad_norm = (
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        approximate_kl = (
            (
                ratio
                - 1.0
                - log_ratio
            )
            .mean()
            .detach()
        )

        clip_fraction = (
            (
                (ratio - 1.0).abs()
                > self.config.ppo_clip_epsilon
            )
            .float()
            .mean()
            .detach()
        )

        return {
            "loss": float(
                total_loss.detach()
            ),
            "policy_loss": float(
                policy_loss.detach()
            ),
            "entropy": float(
                entropy_bonus.detach()
            ),
            "approximate_kl": float(
                approximate_kl
            ),
            "clip_fraction": float(
                clip_fraction
            ),
            "grad_norm": float(
                grad_norm.detach()
                if isinstance(
                    grad_norm,
                    Tensor,
                )
                else grad_norm
            ),
        }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_actor(
    actor: nn.Module,
    config: ActorConfig,
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

    config_data["actor_type"] = (
        ActorType(
            config.actor_type
        ).value
    )

    torch.save(
        {
            "config": config_data,
            "state_dict": actor.state_dict(),
        },
        path,
    )

    return path


def load_actor(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, ActorConfig]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    config_data = checkpoint[
        "config"
    ]

    config_data["actor_type"] = (
        ActorType(
            config_data["actor_type"]
        )
    )

    config = ActorConfig(
        **config_data
    )

    actor = create_actor(
        config
    ).to(device)

    actor.load_state_dict(
        checkpoint["state_dict"]
    )

    actor.eval()

    return (
        actor,
        config,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(
        42
    )

    config = ActorConfig(
        state_dim=32,
        action_dim=4,
        num_discrete_actions=6,
        hidden_dim=128,
        actor_type=ActorType.HYBRID,
    )

    actor = create_actor(
        config
    )

    states = torch.randn(
        16,
        config.state_dim,
    )

    print("=" * 60)
    print("NEUROWORLD AI - ACTOR NETWORK")
    print("=" * 60)

    # Generate rollout actions.
    with torch.no_grad():
        output = actor(
            states
        )

    print(
        "Continuous action shape:",
        output.continuous_action.shape,
    )

    print(
        "Discrete action shape:",
        output.discrete_action.shape,
    )

    print(
        "Log-probability shape:",
        output.log_probability.shape,
    )

    # Example normalized advantages.
    # Replace with real GAE advantages.
    advantages = torch.randn(
        16
    )

    trainer = ActorTrainer(
        actor,
        config,
    )

    metrics = trainer.train_step(
        states=states,
        continuous_actions=(
            output.continuous_action.detach()
        ),
        discrete_actions=(
            output.discrete_action.detach()
        ),
        old_log_probabilities=(
            output.log_probability.detach()
        ),
        advantages=advantages,
    )

    print(
        "Training metrics:",
        metrics,
    )

    # Deterministic inference.
    actor.eval()

    with torch.no_grad():
        prediction = actor(
            states[:2],
            deterministic=True,
        )

    print(
        "Deterministic continuous actions:",
        prediction.continuous_action,
    )

    print(
        "Deterministic discrete actions:",
        prediction.discrete_action,
    )

    checkpoint_path = save_actor(
        actor,
        config,
        "models/actor_demo.pt",
    )

    loaded_actor, loaded_config = (
        load_actor(
            checkpoint_path
        )
    )

    print(
        "Loaded actor:",
        type(loaded_actor).__name__,
    )

    print(
        "Loaded actor type:",
        loaded_config.actor_type.value,
    )

    print(
        "Actor demonstration completed"
    )


if __name__ == "__main__":
    main()
