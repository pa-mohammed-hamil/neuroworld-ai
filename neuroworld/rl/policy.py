
"""
NeuroWorld AI
File: neuroworld/rl/policy.py

Unified Reinforcement Learning Policy Engine.

Features:
    1. Continuous-action policies
    2. Discrete-action policies
    3. Hybrid-action policies
    4. Stochastic action sampling
    5. Deterministic inference
    6. Epsilon-greedy exploration
    7. Gaussian exploration noise
    8. Exploration decay schedules
    9. Action clipping and validation
   10. Optional action safety callbacks
   11. Batched policy inference
   12. Action log probabilities
   13. Policy evaluation
   14. Checkpoint save/load
   15. Policy statistics

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.rl.policy

The safety callback is a policy-level filter,
not a substitute for independent hardware
safety enforcement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class PolicyError(Exception):
    """Base policy exception."""


class PolicyValidationError(PolicyError):
    """Invalid policy input or output."""


class UnsafeActionError(PolicyError):
    """Action rejected by safety validation."""


# ============================================================
# ENUMERATIONS
# ============================================================

class PolicyType(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"


class ExplorationStrategy(str, Enum):
    NONE = "none"
    EPSILON_GREEDY = "epsilon_greedy"
    GAUSSIAN = "gaussian"


class ExplorationSchedule(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PolicyConfig:
    state_dim: int = 128
    action_dim: int = 8
    num_discrete_actions: int = 12

    hidden_dim: int = 256

    policy_type: PolicyType = PolicyType.CONTINUOUS

    action_low: float = -1.0
    action_high: float = 1.0

    log_std_min: float = -5.0
    log_std_max: float = 2.0

    exploration_strategy: ExplorationStrategy = (
        ExplorationStrategy.GAUSSIAN
    )

    exploration_schedule: ExplorationSchedule = (
        ExplorationSchedule.LINEAR
    )

    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05

    initial_noise_std: float = 0.2
    final_noise_std: float = 0.01

    exploration_decay_steps: int = 100_000

    max_action_norm: float | None = None

    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_type",
            PolicyType(self.policy_type),
        )

        object.__setattr__(
            self,
            "exploration_strategy",
            ExplorationStrategy(
                self.exploration_strategy
            ),
        )

        object.__setattr__(
            self,
            "exploration_schedule",
            ExplorationSchedule(
                self.exploration_schedule
            ),
        )

        for name in (
            "state_dim",
            "action_dim",
            "num_discrete_actions",
            "hidden_dim",
            "exploration_decay_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.action_low >= self.action_high:
            raise ValueError(
                "action_low must be less than action_high"
            )

        if self.log_std_min >= self.log_std_max:
            raise ValueError(
                "Invalid log standard deviation bounds"
            )

        for name in (
            "initial_epsilon",
            "final_epsilon",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be in [0,1]"
                )

        for name in (
            "initial_noise_std",
            "final_noise_std",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )

        if (
            self.max_action_norm is not None
            and self.max_action_norm <= 0
        ):
            raise ValueError(
                "max_action_norm must be positive"
            )


# ============================================================
# POLICY OUTPUT
# ============================================================

@dataclass
class PolicyOutput:
    continuous_action: Tensor | None
    discrete_action: Tensor | None

    log_probability: Tensor | None
    entropy: Tensor | None

    mean: Tensor | None = None
    log_std: Tensor | None = None
    logits: Tensor | None = None

    deterministic: bool = False
    exploration_applied: bool = False

    def detach(self) -> "PolicyOutput":
        def detach_tensor(
            value: Tensor | None,
        ) -> Tensor | None:
            return (
                value.detach()
                if value is not None
                else None
            )

        return PolicyOutput(
            continuous_action=detach_tensor(
                self.continuous_action
            ),
            discrete_action=detach_tensor(
                self.discrete_action
            ),
            log_probability=detach_tensor(
                self.log_probability
            ),
            entropy=detach_tensor(
                self.entropy
            ),
            mean=detach_tensor(self.mean),
            log_std=detach_tensor(self.log_std),
            logits=detach_tensor(self.logits),
            deterministic=self.deterministic,
            exploration_applied=(
                self.exploration_applied
            ),
        )


# ============================================================
# EXPLORATION SCHEDULER
# ============================================================

class ExplorationScheduler:
    """
    Decays epsilon and Gaussian noise
    according to an exploration schedule.
    """

    def __init__(
        self,
        config: PolicyConfig,
    ) -> None:
        self.config = config
        self.steps = 0

    def _progress(self) -> float:
        return min(
            self.steps
            / self.config.exploration_decay_steps,
            1.0,
        )

    def _schedule(
        self,
        initial: float,
        final: float,
    ) -> float:
        progress = self._progress()

        if (
            self.config.exploration_schedule
            == ExplorationSchedule.CONSTANT
        ):
            return initial

        if (
            self.config.exploration_schedule
            == ExplorationSchedule.LINEAR
        ):
            return (
                initial
                + (final - initial) * progress
            )

        # Exponential interpolation, including
        # zero-valued endpoints.
        if initial > 0 and final > 0:
            return (
                initial
                * (final / initial) ** progress
            )

        return (
            initial
            + (final - initial) * progress
        )

    @property
    def epsilon(self) -> float:
        return self._schedule(
            self.config.initial_epsilon,
            self.config.final_epsilon,
        )

    @property
    def noise_std(self) -> float:
        return self._schedule(
            self.config.initial_noise_std,
            self.config.final_noise_std,
        )

    def step(
        self,
        count: int = 1,
    ) -> None:
        if count < 0:
            raise ValueError(
                "count must be nonnegative"
            )

        self.steps += count

    def reset(self) -> None:
        self.steps = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "steps": self.steps
        }

    def load_state_dict(
        self,
        state: dict[str, int],
    ) -> None:
        self.steps = int(
            state["steps"]
        )


# ============================================================
# NEURAL POLICY NETWORK
# ============================================================

class PolicyNetwork(nn.Module):
    """
    Shared feature backbone with action heads.

    Continuous:
        Gaussian mean and log_std

    Discrete:
        Categorical logits

    Hybrid:
        Gaussian and categorical heads
    """

    def __init__(
        self,
        config: PolicyConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.backbone = nn.Sequential(
            nn.Linear(
                config.state_dim,
                config.hidden_dim,
            ),
            nn.LayerNorm(
                config.hidden_dim
            ),
            nn.SiLU(),

            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.SiLU(),
        )

        if config.policy_type in (
            PolicyType.CONTINUOUS,
            PolicyType.HYBRID,
        ):
            self.mean_head = nn.Linear(
                config.hidden_dim,
                config.action_dim,
            )

            self.log_std_head = nn.Linear(
                config.hidden_dim,
                config.action_dim,
            )

        else:
            self.mean_head = None
            self.log_std_head = None

        if config.policy_type in (
            PolicyType.DISCRETE,
            PolicyType.HYBRID,
        ):
            self.logits_head = nn.Linear(
                config.hidden_dim,
                config.num_discrete_actions,
            )

        else:
            self.logits_head = None

    def forward(
        self,
        states: Tensor,
    ) -> dict[str, Tensor]:
        if states.ndim == 1:
            states = states.unsqueeze(0)

        if (
            states.ndim != 2
            or states.shape[-1]
            != self.config.state_dim
        ):
            raise PolicyValidationError(
                "Invalid state shape"
            )

        features = self.backbone(
            states
        )

        outputs = {}

        if self.mean_head is not None:
            outputs["mean"] = (
                self.mean_head(
                    features
                )
            )

            outputs["log_std"] = (
                self.log_std_head(
                    features
                ).clamp(
                    self.config.log_std_min,
                    self.config.log_std_max,
                )
            )

        if self.logits_head is not None:
            outputs["logits"] = (
                self.logits_head(
                    features
                )
            )

        return outputs


# ============================================================
# CONTINUOUS ACTION DISTRIBUTION
# ============================================================

class SquashedGaussianDistribution:
    """
    Tanh-squashed Gaussian distribution.

    Supports scaled continuous actions.

    Uses the change-of-variables
    correction for action log probabilities.
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

        self.normal = Normal(
            mean,
            log_std.exp(),
        )

        self.scale = (
            action_high - action_low
        ) / 2.0

        self.bias = (
            action_high + action_low
        ) / 2.0

    def sample(
        self,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if deterministic:
            raw_action = self.mean

        else:
            raw_action = (
                self.normal.rsample()
            )

        squashed = torch.tanh(
            raw_action
        )

        action = (
            squashed * self.scale
            + self.bias
        )

        log_prob = self._log_prob_raw(
            raw_action
        )

        return (
            action,
            log_prob,
        )

    def _log_prob_raw(
        self,
        raw_action: Tensor,
    ) -> Tensor:
        base_log_prob = (
            self.normal.log_prob(
                raw_action
            )
        )

        # Stable tanh Jacobian correction.
        log_tanh_jacobian = (
            2.0
            * (
                math.log(2.0)
                - raw_action
                - F.softplus(
                    -2.0 * raw_action
                )
            )
        )

        log_scale = math.log(
            self.scale
        )

        return (
            base_log_prob
            - log_tanh_jacobian
            - log_scale
        ).sum(
            dim=-1
        )

    def log_prob(
        self,
        actions: Tensor,
    ) -> Tensor:
        normalized = (
            actions - self.bias
        ) / self.scale

        normalized = normalized.clamp(
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )

        raw_action = torch.atanh(
            normalized
        )

        return self._log_prob_raw(
            raw_action
        )

    def entropy_estimate(
        self,
    ) -> Tensor:
        _, log_prob = self.sample()

        return -log_prob


# ============================================================
# POLICY ENGINE
# ============================================================

SafetyCallback = Callable[
    [np.ndarray | int | tuple[np.ndarray, int]],
    bool,
]


class PolicyEngine(nn.Module):
    """
    Unified policy inference engine.

    Neural optimization should be handled
    by policy_optimizer.py.

    This module focuses on:
        action generation
        action validation
        exploration
        inference
        evaluation
        checkpointing
    """

    def __init__(
        self,
        config: PolicyConfig | None = None,
        *,
        safety_callback: SafetyCallback | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or PolicyConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.network = PolicyNetwork(
            self.config
        ).to(
            self.device
        )

        self.exploration = ExplorationScheduler(
            self.config
        )

        self.safety_callback = (
            safety_callback
        )

        self.inference_count = 0
        self.rejected_actions = 0

    # --------------------------------------------------------
    # STATE VALIDATION
    # --------------------------------------------------------

    def _prepare_states(
        self,
        states: Tensor | np.ndarray,
    ) -> Tensor:
        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        if states.ndim == 1:
            states = states.unsqueeze(0)

        if (
            states.ndim != 2
            or states.shape[-1]
            != self.config.state_dim
        ):
            raise PolicyValidationError(
                f"Expected states [B,"
                f"{self.config.state_dim}]"
            )

        if not torch.isfinite(
            states
        ).all():
            raise PolicyValidationError(
                "States contain non-finite values"
            )

        return states

    # --------------------------------------------------------
    # CONTINUOUS DISTRIBUTION
    # --------------------------------------------------------

    def _continuous_distribution(
        self,
        network_output: dict[str, Tensor],
    ) -> SquashedGaussianDistribution:
        return SquashedGaussianDistribution(
            mean=network_output["mean"],
            log_std=network_output["log_std"],
            action_low=self.config.action_low,
            action_high=self.config.action_high,
        )

    # --------------------------------------------------------
    # ACTION SAMPLING
    # --------------------------------------------------------

    def forward(
        self,
        states: Tensor | np.ndarray,
        *,
        deterministic: bool = False,
    ) -> PolicyOutput:
        states = self._prepare_states(
            states
        )

        outputs = self.network(
            states
        )

        continuous_action = None
        discrete_action = None

        log_probabilities = []
        entropies = []

        if self.config.policy_type in (
            PolicyType.CONTINUOUS,
            PolicyType.HYBRID,
        ):
            distribution = (
                self._continuous_distribution(
                    outputs
                )
            )

            (
                continuous_action,
                continuous_log_prob,
            ) = distribution.sample(
                deterministic=deterministic
            )

            log_probabilities.append(
                continuous_log_prob
            )

            entropies.append(
                distribution.entropy_estimate()
            )

        if self.config.policy_type in (
            PolicyType.DISCRETE,
            PolicyType.HYBRID,
        ):
            categorical = Categorical(
                logits=outputs["logits"]
            )

            if deterministic:
                discrete_action = (
                    outputs["logits"].argmax(
                        dim=-1
                    )
                )

            else:
                discrete_action = (
                    categorical.sample()
                )

            log_probabilities.append(
                categorical.log_prob(
                    discrete_action
                )
            )

            entropies.append(
                categorical.entropy()
            )

        log_probability = torch.stack(
            log_probabilities,
            dim=0,
        ).sum(
            dim=0
        )

        entropy = torch.stack(
            entropies,
            dim=0,
        ).sum(
            dim=0
        )

        return PolicyOutput(
            continuous_action=continuous_action,
            discrete_action=discrete_action,
            log_probability=log_probability,
            entropy=entropy,
            mean=outputs.get("mean"),
            log_std=outputs.get("log_std"),
            logits=outputs.get("logits"),
            deterministic=deterministic,
        )

    # --------------------------------------------------------
    # LOG PROBABILITY EVALUATION
    # --------------------------------------------------------

    def evaluate_actions(
        self,
        states: Tensor | np.ndarray,
        *,
        continuous_actions: Tensor | None = None,
        discrete_actions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        states = self._prepare_states(
            states
        )

        outputs = self.network(
            states
        )

        log_probs = []
        entropies = []

        if self.config.policy_type in (
            PolicyType.CONTINUOUS,
            PolicyType.HYBRID,
        ):
            if continuous_actions is None:
                raise PolicyValidationError(
                    "Continuous actions required"
                )

            continuous_actions = (
                continuous_actions.to(
                    self.device
                )
            )

            distribution = (
                self._continuous_distribution(
                    outputs
                )
            )

            log_probs.append(
                distribution.log_prob(
                    continuous_actions
                )
            )

            entropies.append(
                distribution.entropy_estimate()
            )

        if self.config.policy_type in (
            PolicyType.DISCRETE,
            PolicyType.HYBRID,
        ):
            if discrete_actions is None:
                raise PolicyValidationError(
                    "Discrete actions required"
                )

            discrete_actions = (
                discrete_actions.to(
                    self.device
                ).long().reshape(-1)
            )

            categorical = Categorical(
                logits=outputs["logits"]
            )

            log_probs.append(
                categorical.log_prob(
                    discrete_actions
                )
            )

            entropies.append(
                categorical.entropy()
            )

        return (
            torch.stack(
                log_probs
            ).sum(dim=0),
            torch.stack(
                entropies
            ).sum(dim=0),
        )

    # --------------------------------------------------------
    # ACTION LIMITING
    # --------------------------------------------------------

    def _limit_continuous_actions(
        self,
        actions: Tensor,
    ) -> Tensor:
        actions = actions.clamp(
            self.config.action_low,
            self.config.action_high,
        )

        if self.config.max_action_norm is not None:
            norms = torch.linalg.vector_norm(
                actions,
                dim=-1,
                keepdim=True,
            )

            scale = (
                self.config.max_action_norm
                / norms.clamp_min(1e-8)
            ).clamp(max=1.0)

            actions = actions * scale

        return actions

    # --------------------------------------------------------
    # EXPLORATION
    # --------------------------------------------------------

    def _apply_exploration(
        self,
        output: PolicyOutput,
    ) -> PolicyOutput:
        strategy = (
            self.config.exploration_strategy
        )

        if strategy == ExplorationStrategy.NONE:
            return output

        applied = False

        if strategy == ExplorationStrategy.GAUSSIAN:
            if output.continuous_action is not None:
                noise = (
                    torch.randn_like(
                        output.continuous_action
                    )
                    * self.exploration.noise_std
                )

                output.continuous_action = (
                    self._limit_continuous_actions(
                        output.continuous_action
                        + noise
                    )
                )

                applied = True

        elif strategy == ExplorationStrategy.EPSILON_GREEDY:
            epsilon = self.exploration.epsilon

            if output.discrete_action is not None:
                random_mask = (
                    torch.rand(
                        output.discrete_action.shape,
                        device=self.device,
                    )
                    < epsilon
                )

                random_actions = torch.randint(
                    0,
                    self.config.num_discrete_actions,
                    output.discrete_action.shape,
                    device=self.device,
                )

                output.discrete_action = torch.where(
                    random_mask,
                    random_actions,
                    output.discrete_action,
                )

                applied = bool(
                    random_mask.any().item()
                )

            if output.continuous_action is not None:
                batch_size = (
                    output.continuous_action.shape[0]
                )

                random_mask = (
                    torch.rand(
                        batch_size,
                        device=self.device,
                    )
                    < epsilon
                )

                random_actions = (
                    torch.empty_like(
                        output.continuous_action
                    ).uniform_(
                        self.config.action_low,
                        self.config.action_high,
                    )
                )

                output.continuous_action = (
                    torch.where(
                        random_mask.unsqueeze(-1),
                        random_actions,
                        output.continuous_action,
                    )
                )

                applied = (
                    applied
                    or bool(
                        random_mask.any().item()
                    )
                )

        output.exploration_applied = applied

        # External exploration changes the action
        # distribution. Do not report the original
        # policy log probability as valid.
        if applied:
            output.log_probability = None

        return output

    # --------------------------------------------------------
    # SAFETY CALLBACK
    # --------------------------------------------------------

    def _check_safety(
        self,
        output: PolicyOutput,
    ) -> None:
        if self.safety_callback is None:
            return

        continuous = (
            output.continuous_action
        )

        discrete = (
            output.discrete_action
        )

        batch_size = (
            continuous.shape[0]
            if continuous is not None
            else discrete.shape[0]
        )

        for index in range(batch_size):
            if (
                continuous is not None
                and discrete is not None
            ):
                action: Any = (
                    continuous[index]
                    .detach()
                    .cpu()
                    .numpy(),
                    int(
                        discrete[index].item()
                    ),
                )

            elif continuous is not None:
                action = (
                    continuous[index]
                    .detach()
                    .cpu()
                    .numpy()
                )

            else:
                action = int(
                    discrete[index].item()
                )

            approved = self.safety_callback(
                action
            )

            if not approved:
                self.rejected_actions += 1

                raise UnsafeActionError(
                    f"Action rejected at batch "
                    f"index {index}"
                )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        states: Tensor | np.ndarray,
        *,
        deterministic: bool = False,
        explore: bool = False,
        advance_schedule: bool = True,
    ) -> PolicyOutput:
        self.eval()

        output = self.forward(
            states,
            deterministic=deterministic,
        )

        if explore and not deterministic:
            output = self._apply_exploration(
                output
            )

        if output.continuous_action is not None:
            limited = (
                self._limit_continuous_actions(
                    output.continuous_action
                )
            )

            if not torch.equal(
                limited,
                output.continuous_action,
            ):
                output.log_probability = None

            output.continuous_action = limited

        self._check_safety(
            output
        )

        if explore and advance_schedule:
            self.exploration.step()

        self.inference_count += (
            output.continuous_action.shape[0]
            if output.continuous_action is not None
            else output.discrete_action.shape[0]
        )

        return output.detach()

    # --------------------------------------------------------
    # SINGLE STATE ACTION
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        state: Tensor | np.ndarray,
        *,
        deterministic: bool = True,
        explore: bool = False,
    ) -> np.ndarray | int | tuple[np.ndarray, int]:
        output = self.act(
            state,
            deterministic=deterministic,
            explore=explore,
        )

        continuous = output.continuous_action
        discrete = output.discrete_action

        if (
            continuous is not None
            and discrete is not None
        ):
            return (
                continuous[0]
                .cpu()
                .numpy(),
                int(
                    discrete[0].item()
                ),
            )

        if continuous is not None:
            return (
                continuous[0]
                .cpu()
                .numpy()
            )

        return int(
            discrete[0].item()
        )

    # --------------------------------------------------------
    # POLICY STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        return {
            "policy_type": (
                self.config.policy_type.value
            ),
            "state_dim": (
                self.config.state_dim
            ),
            "action_dim": (
                self.config.action_dim
            ),
            "inference_count": (
                self.inference_count
            ),
            "rejected_actions": (
                self.rejected_actions
            ),
            "exploration_steps": (
                self.exploration.steps
            ),
            "epsilon": (
                self.exploration.epsilon
            ),
            "noise_std": (
                self.exploration.noise_std
            ),
            "device": str(
                self.device
            ),
        }

    # --------------------------------------------------------
    # CHECKPOINT SAVE
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

        checkpoint = {
            "config": {
                **asdict(self.config),
                "policy_type": (
                    self.config.policy_type.value
                ),
                "exploration_strategy": (
                    self.config.exploration_strategy.value
                ),
                "exploration_schedule": (
                    self.config.exploration_schedule.value
                ),
            },
            "network": (
                self.network.state_dict()
            ),
            "exploration": (
                self.exploration.state_dict()
            ),
            "inference_count": (
                self.inference_count
            ),
            "rejected_actions": (
                self.rejected_actions
            ),
        }

        torch.save(
            checkpoint,
            path,
        )

        return path

    # --------------------------------------------------------
    # CHECKPOINT LOAD
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        safety_callback: SafetyCallback | None = None,
    ) -> "PolicyEngine":
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        config_data = dict(
            checkpoint["config"]
        )

        config_data["device"] = device

        engine = cls(
            PolicyConfig(
                **config_data
            ),
            safety_callback=safety_callback,
        )

        engine.network.load_state_dict(
            checkpoint["network"]
        )

        engine.exploration.load_state_dict(
            checkpoint["exploration"]
        )

        engine.inference_count = int(
            checkpoint.get(
                "inference_count",
                0,
            )
        )

        engine.rejected_actions = int(
            checkpoint.get(
                "rejected_actions",
                0,
            )
        )

        engine.eval()

        return engine


# ============================================================
# FACTORY
# ============================================================

def create_policy(
    config: PolicyConfig | None = None,
    *,
    safety_callback: SafetyCallback | None = None,
) -> PolicyEngine:
    return PolicyEngine(
        config,
        safety_callback=safety_callback,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print("NEUROWORLD AI - POLICY ENGINE")
    print("=" * 65)

    # --------------------------------------------------------
    # CONTINUOUS POLICY
    # --------------------------------------------------------

    continuous_config = PolicyConfig(
        state_dim=32,
        action_dim=4,
        hidden_dim=128,
        policy_type=PolicyType.CONTINUOUS,
        exploration_strategy=(
            ExplorationStrategy.GAUSSIAN
        ),
    )

    continuous_policy = create_policy(
        continuous_config
    )

    states = torch.randn(
        8,
        32,
    )

    continuous_output = continuous_policy.act(
        states,
        explore=True,
    )

    print(
        "Continuous action shape:",
        continuous_output.continuous_action.shape,
    )

    # --------------------------------------------------------
    # DISCRETE POLICY
    # --------------------------------------------------------

    discrete_config = PolicyConfig(
        state_dim=32,
        action_dim=4,
        num_discrete_actions=6,
        hidden_dim=128,
        policy_type=PolicyType.DISCRETE,
        exploration_strategy=(
            ExplorationStrategy.EPSILON_GREEDY
        ),
    )

    discrete_policy = create_policy(
        discrete_config
    )

    discrete_output = discrete_policy.act(
        states,
        explore=True,
    )

    print(
        "Discrete actions:",
        discrete_output.discrete_action,
    )

    # --------------------------------------------------------
    # HYBRID POLICY
    # --------------------------------------------------------

    hybrid_config = PolicyConfig(
        state_dim=32,
        action_dim=4,
        num_discrete_actions=6,
        hidden_dim=128,
        policy_type=PolicyType.HYBRID,
        exploration_strategy=(
            ExplorationStrategy.GAUSSIAN
        ),
    )

    hybrid_policy = create_policy(
        hybrid_config
    )

    hybrid_output = hybrid_policy.act(
        states,
        deterministic=False,
    )

    print(
        "Hybrid continuous:",
        hybrid_output.continuous_action.shape,
    )

    print(
        "Hybrid discrete:",
        hybrid_output.discrete_action.shape,
    )

    # --------------------------------------------------------
    # ACTION SAFETY VALIDATION
    # --------------------------------------------------------

    def safe_action(
        action: Any,
    ) -> bool:
        if isinstance(action, np.ndarray):
            return bool(
                np.all(
                    np.isfinite(action)
                )
            )

        return True

    safe_policy = create_policy(
        continuous_config,
        safety_callback=safe_action,
    )

    action = safe_policy.predict(
        np.zeros(
            32,
            dtype=np.float32,
        )
    )

    print(
        "Safe predicted action:",
        action,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = continuous_policy.save(
        "models/policy_demo.pt"
    )

    restored_policy = PolicyEngine.load(
        checkpoint
    )

    print(
        "Restored policy statistics:",
        restored_policy.statistics(),
    )

    print(
        "Policy demonstration completed"
    )


if __name__ == "__main__":
    main()
