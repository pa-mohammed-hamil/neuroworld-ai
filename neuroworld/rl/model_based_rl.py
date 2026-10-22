
"""
NeuroWorld AI
File: neuroworld/rl/model_based_rl.py

Model-Based Reinforcement Learning Engine.

Features:
    - Continuous-action actor-critic learning
    - Twin Q critics
    - Soft Actor-Critic (SAC) objectives
    - Automatic entropy temperature tuning
    - Real and imagined replay
    - Neural world-model adapters
    - Uncertainty-filtered imagination
    - Optional intrinsic rewards
    - World-model training callback
    - Policy training from mixed experiences
    - TD-error priority updates
    - Target-network Polyak averaging
    - Gradient clipping
    - Training metrics
    - Checkpoint save/load
    - Reproducible demo

Dependencies:
    pip install torch numpy

Integration:
    neuroworld.rl.critic
    neuroworld.rl.imagined_replay
    neuroworld.rl.intrinsic_reward
    neuroworld.world_model.world_model

This module assumes continuous actions in [-1, 1].

Important:
    This is a learning engine, not a robot
    safety controller. Real-world actions
    require independent safety constraints.
"""

from __future__ import annotations

import copy
import math
import random

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from torch import Tensor, nn
from torch.nn import functional as F

from .critic import (
    CriticConfig,
    CriticType,
    TwinQCritic,
    soft_update,
)

from .imagined_replay import (
    ExperienceSource,
    ImaginedReplayBuffer,
    ImaginedReplayConfig,
    ImaginedStep,
)

from .intrinsic_reward import (
    IntrinsicRewardConfig,
    IntrinsicRewardEngine,
    IntrinsicRewardTrainer,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelBasedRLError(Exception):
    """Base model-based RL exception."""


class ModelBasedRLValidationError(
    ModelBasedRLError
):
    """Invalid configuration or transition."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ModelBasedRLConfig:
    state_dim: int = 128
    action_dim: int = 8

    hidden_dim: int = 256

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.005

    batch_size: int = 256
    max_grad_norm: float = 1.0

    real_capacity: int = 100_000
    imagined_capacity: int = 200_000

    imagined_ratio: float = 0.5
    imagination_horizon: int = 5

    max_imagination_uncertainty: float = 0.5
    uncertainty_penalty: float = 0.1

    initial_random_steps: int = 1000
    min_replay_size: int = 256

    actor_update_interval: int = 1
    target_update_interval: int = 1

    initial_entropy_temperature: float = 0.2
    automatic_entropy_tuning: bool = True

    target_entropy: float | None = None

    use_intrinsic_rewards: bool = False
    intrinsic_reward_scale: float = 0.1

    priority_alpha: float = 0.6
    priority_beta: float = 0.4

    seed: int = 42

    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "hidden_dim",
            "batch_size",
            "real_capacity",
            "imagined_capacity",
            "imagination_horizon",
            "min_replay_size",
            "actor_update_interval",
            "target_update_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.initial_random_steps < 0:
            raise ValueError(
                "initial_random_steps cannot be negative"
            )

        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma must be in [0,1]"
            )

        if not 0 < self.tau <= 1:
            raise ValueError(
                "tau must be in (0,1]"
            )

        if not 0 <= self.imagined_ratio <= 1:
            raise ValueError(
                "imagined_ratio must be in [0,1]"
            )

        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "entropy_learning_rate",
            "max_grad_norm",
            "initial_entropy_temperature",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        for name in (
            "max_imagination_uncertainty",
            "uncertainty_penalty",
            "intrinsic_reward_scale",
            "priority_alpha",
            "priority_beta",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if self.priority_beta > 1:
            raise ValueError(
                "priority_beta must be <= 1"
            )


# ============================================================
# CONTINUOUS STOCHASTIC ACTOR
# ============================================================

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(nn.Module):
    """
    Tanh-squashed Gaussian policy.

    Outputs continuous actions in [-1,1].

    Uses reparameterization for SAC updates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(
                state_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.log_std_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

    def forward(
        self,
        states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if states.ndim == 1:
            states = states.unsqueeze(0)

        if (
            states.ndim != 2
            or states.shape[-1] != self.state_dim
        ):
            raise ModelBasedRLValidationError(
                "Invalid actor state shape"
            )

        features = self.backbone(
            states
        )

        mean = self.mean_head(
            features
        )

        log_std = self.log_std_head(
            features
        ).clamp(
            LOG_STD_MIN,
            LOG_STD_MAX,
        )

        return mean, log_std

    def sample(
        self,
        states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mean, log_std = self.forward(
            states
        )

        std = log_std.exp()

        distribution = (
            torch.distributions.Normal(
                mean,
                std,
            )
        )

        raw_actions = distribution.rsample()

        actions = torch.tanh(
            raw_actions
        )

        log_probabilities = (
            distribution.log_prob(
                raw_actions
            )
            - 2.0
            * (
                math.log(2.0)
                - raw_actions
                - F.softplus(
                    -2.0 * raw_actions
                )
            )
        ).sum(
            dim=-1
        )

        return (
            actions,
            log_probabilities,
        )

    @torch.no_grad()
    def act(
        self,
        states: Tensor,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        if deterministic:
            mean, _ = self.forward(
                states
            )

            return torch.tanh(
                mean
            )

        actions, _ = self.sample(
            states
        )

        return actions


# ============================================================
# WORLD MODEL PROTOCOL
# ============================================================

class WorldModelAdapter(Protocol):
    """
    Required world-model adapter.

    predict_step(state, action) -> ImaginedStep

    A trained world model must predict:
        next_state
        reward
        terminated
        uncertainty
    """

    def predict_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> ImaginedStep:
        ...


# ============================================================
# POLICY ADAPTER
# ============================================================

class SACPolicyAdapter:
    """
    Connects GaussianActor with
    ImaginedReplayBuffer rollout generation.
    """

    def __init__(
        self,
        actor: GaussianActor,
        device: torch.device,
        deterministic: bool = False,
    ) -> None:
        self.actor = actor
        self.device = device
        self.deterministic = deterministic

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action = self.actor.act(
            tensor,
            deterministic=self.deterministic,
        )

        return (
            action.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )


# ============================================================
# TRAINING METRICS
# ============================================================

@dataclass
class RLTrainingMetrics:
    critic_loss: float
    actor_loss: float
    entropy_loss: float

    entropy_temperature: float
    mean_q_value: float
    mean_log_probability: float

    mean_reward: float
    mean_uncertainty: float

    real_samples: int
    imagined_samples: int

    critic_gradient_norm: float
    actor_gradient_norm: float

    training_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(
            self
        )


# ============================================================
# MODEL-BASED RL ENGINE
# ============================================================

class ModelBasedRLEngine:
    """
    SAC-inspired model-based RL coordinator.

    Main responsibilities:
        1. Store real transitions
        2. Generate imagined transitions
        3. Sample mixed replay batches
        4. Train twin Q critics
        5. Improve stochastic actor
        6. Update entropy temperature
        7. Update target critics
        8. Track metrics
    """

    def __init__(
        self,
        config: ModelBasedRLConfig | None = None,
        *,
        world_model: WorldModelAdapter | None = None,
    ) -> None:
        self.config = (
            config or ModelBasedRLConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        random.seed(
            self.config.seed
        )

        np.random.seed(
            self.config.seed
        )

        torch.manual_seed(
            self.config.seed
        )

        # ----------------------------------------------------
        # ACTOR
        # ----------------------------------------------------

        self.actor = GaussianActor(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(
            self.device
        )

        self.actor_optimizer = (
            torch.optim.Adam(
                self.actor.parameters(),
                lr=self.config.actor_learning_rate,
            )
        )

        # ----------------------------------------------------
        # TWIN Q CRITIC
        # ----------------------------------------------------

        critic_config = CriticConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            critic_type=CriticType.TWIN_Q,
            learning_rate=(
                self.config.critic_learning_rate
            ),
            gamma=self.config.gamma,
            tau=self.config.tau,
        )

        self.critic = TwinQCritic(
            critic_config
        ).to(
            self.device
        )

        self.target_critic = copy.deepcopy(
            self.critic
        )

        self.target_critic.eval()

        for parameter in (
            self.target_critic.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        self.critic_optimizer = (
            torch.optim.Adam(
                self.critic.parameters(),
                lr=self.config.critic_learning_rate,
            )
        )

        # ----------------------------------------------------
        # ENTROPY TEMPERATURE
        # ----------------------------------------------------

        self.log_alpha = torch.tensor(
            math.log(
                self.config.initial_entropy_temperature
            ),
            dtype=torch.float32,
            device=self.device,
            requires_grad=(
                self.config.automatic_entropy_tuning
            ),
        )

        if self.config.automatic_entropy_tuning:
            self.alpha_optimizer = (
                torch.optim.Adam(
                    [self.log_alpha],
                    lr=self.config.entropy_learning_rate,
                )
            )

        else:
            self.alpha_optimizer = None

        self.target_entropy = (
            self.config.target_entropy
            if self.config.target_entropy is not None
            else -float(
                self.config.action_dim
            )
        )

        # ----------------------------------------------------
        # REPLAY
        # ----------------------------------------------------

        replay_config = ImaginedReplayConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            real_capacity=self.config.real_capacity,
            imagined_capacity=(
                self.config.imagined_capacity
            ),
            batch_size=self.config.batch_size,
            imagined_ratio=self.config.imagined_ratio,
            gamma=self.config.gamma,
            priority_alpha=self.config.priority_alpha,
            importance_beta=self.config.priority_beta,
            imagination_horizon=(
                self.config.imagination_horizon
            ),
            max_imagination_uncertainty=(
                self.config.max_imagination_uncertainty
            ),
            uncertainty_penalty=(
                self.config.uncertainty_penalty
            ),
            seed=self.config.seed,
        )

        self.replay = ImaginedReplayBuffer(
            replay_config
        )

        # ----------------------------------------------------
        # OPTIONAL INTRINSIC REWARDS
        # ----------------------------------------------------

        self.intrinsic_engine = None
        self.intrinsic_trainer = None

        if self.config.use_intrinsic_rewards:
            intrinsic_config = (
                IntrinsicRewardConfig(
                    state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim,
                    hidden_dim=self.config.hidden_dim,
                    intrinsic_scale=(
                        self.config.intrinsic_reward_scale
                    ),
                )
            )

            self.intrinsic_engine = (
                IntrinsicRewardEngine(
                    intrinsic_config
                ).to(
                    self.device
                )
            )

            self.intrinsic_trainer = (
                IntrinsicRewardTrainer(
                    self.intrinsic_engine
                )
            )

        # ----------------------------------------------------
        # WORLD MODEL
        # ----------------------------------------------------

        self.world_model = world_model

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        self.environment_steps = 0
        self.training_steps = 0

        self.metrics_history: list[
            dict[str, Any]
        ] = []

    # --------------------------------------------------------
    # ENTROPY TEMPERATURE
    # --------------------------------------------------------

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        state: Any,
        *,
        deterministic: bool = False,
        exploration: bool = True,
    ) -> np.ndarray:
        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        if state_array.shape != (
            self.config.state_dim,
        ):
            raise ModelBasedRLValidationError(
                "Invalid state dimension"
            )

        if not np.isfinite(
            state_array
        ).all():
            raise ModelBasedRLValidationError(
                "State contains non-finite values"
            )

        if (
            exploration
            and self.environment_steps
            < self.config.initial_random_steps
        ):
            return np.random.uniform(
                -1.0,
                1.0,
                size=self.config.action_dim,
            ).astype(np.float32)

        tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.actor.eval()

        action = self.actor.act(
            tensor,
            deterministic=deterministic,
        )

        return (
            action.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # --------------------------------------------------------
    # STORE REAL TRANSITION
    # --------------------------------------------------------

    def store_real_transition(
        self,
        *,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        terminated: bool = False,
        truncated: bool = False,
        episode_id: str = "",
        step_index: int = 0,
    ) -> int:
        index = self.replay.add_real(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
            episode_id=episode_id,
            step_index=step_index,
        )

        self.environment_steps += 1

        return index

    # --------------------------------------------------------
    # GENERATE IMAGINED TRANSITIONS
    # --------------------------------------------------------

    def imagine(
        self,
        *,
        initial_states: Any | None = None,
        num_start_states: int = 32,
        horizon: int | None = None,
        model_version: str | None = None,
    ) -> dict[str, int]:
        if self.world_model is None:
            raise ModelBasedRLError(
                "World model is not configured"
            )

        if initial_states is None:
            initial_states = (
                self.replay.sample_real_start_states(
                    num_start_states
                )
            )

        policy = SACPolicyAdapter(
            self.actor,
            self.device,
        )

        return (
            self.replay.generate_imagined_rollouts(
                world_model=self.world_model,
                policy=policy,
                initial_states=initial_states,
                horizon=horizon,
                model_version=model_version,
            )
        )

    # --------------------------------------------------------
    # REPLAY READINESS
    # --------------------------------------------------------

    def ready_to_train(self) -> bool:
        return (
            len(self.replay)
            >= self.config.min_replay_size
        )

    # --------------------------------------------------------
    # CRITIC TRAINING
    # --------------------------------------------------------

    def _update_critic(
        self,
        batch: dict[str, Tensor],
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        terminated = batch["terminated"]
        weights = batch["weights"]

        with torch.no_grad():
            (
                next_actions,
                next_log_probabilities,
            ) = self.actor.sample(
                next_states
            )

            target_q = (
                self.target_critic.minimum(
                    next_states,
                    next_actions,
                )
            )

            target_q = (
                target_q
                - self.alpha.detach()
                * next_log_probabilities
            )

            bellman_target = (
                rewards
                + self.config.gamma
                * (1.0 - terminated)
                * target_q
            )

        q1, q2 = self.critic(
            states,
            actions,
        )

        td_error_1 = (
            q1 - bellman_target
        )

        td_error_2 = (
            q2 - bellman_target
        )

        critic_loss = (
            weights
            * (
                td_error_1.square()
                + td_error_2.square()
            )
        ).mean()

        self.critic_optimizer.zero_grad(
            set_to_none=True
        )

        critic_loss.backward()

        grad_norm = (
            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.critic_optimizer.step()

        td_errors = (
            0.5
            * (
                td_error_1.detach().abs()
                + td_error_2.detach().abs()
            )
        )

        return (
            critic_loss.detach(),
            grad_norm.detach(),
            td_errors,
        )

    # --------------------------------------------------------
    # ACTOR TRAINING
    # --------------------------------------------------------

    def _update_actor(
        self,
        states: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        # Freeze critic parameters while
        # allowing gradients through Q(s,a)
        # to reach the actor actions.

        for parameter in (
            self.critic.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        try:
            (
                actions,
                log_probabilities,
            ) = self.actor.sample(
                states
            )

            q_values = self.critic.minimum(
                states,
                actions,
            )

            actor_loss = (
                self.alpha.detach()
                * log_probabilities
                - q_values
            ).mean()

            self.actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            grad_norm = (
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
                parameter.requires_grad_(
                    True
                )

        return (
            actor_loss.detach(),
            grad_norm.detach(),
            log_probabilities.detach(),
        )

    # --------------------------------------------------------
    # ENTROPY TEMPERATURE TRAINING
    # --------------------------------------------------------

    def _update_entropy(
        self,
        log_probabilities: Tensor,
    ) -> Tensor:
        if (
            not self.config.automatic_entropy_tuning
        ):
            return torch.zeros(
                (),
                device=self.device,
            )

        alpha_loss = -(
            self.log_alpha
            * (
                log_probabilities
                + self.target_entropy
            ).detach()
        ).mean()

        assert self.alpha_optimizer is not None

        self.alpha_optimizer.zero_grad(
            set_to_none=True
        )

        alpha_loss.backward()

        self.alpha_optimizer.step()

        return alpha_loss.detach()

    # --------------------------------------------------------
    # MAIN TRAINING STEP
    # --------------------------------------------------------

    def train_step(
        self,
    ) -> RLTrainingMetrics:
        if not self.ready_to_train():
            raise ModelBasedRLError(
                "Insufficient replay experiences"
            )

        replay_batch = self.replay.sample(
            batch_size=self.config.batch_size,
        )

        batch = replay_batch.to_torch(
            self.device
        )

        # ----------------------------------------------------
        # OPTIONAL INTRINSIC REWARD
        # ----------------------------------------------------

        if self.intrinsic_engine is not None:
            self.intrinsic_engine.eval()

            intrinsic_output = (
                self.intrinsic_engine.compute(
                    batch["states"],
                    batch["actions"],
                    batch["next_states"],
                    extrinsic_rewards=(
                        batch["rewards"]
                    ),
                    uncertainties=(
                        batch["uncertainties"]
                    ),
                    update_memory=False,
                    update_normalizers=False,
                )
            )

            batch["rewards"] = (
                intrinsic_output.combined_reward
            )

        # ----------------------------------------------------
        # CRITIC UPDATE
        # ----------------------------------------------------

        (
            critic_loss,
            critic_grad_norm,
            td_errors,
        ) = self._update_critic(
            batch
        )

        self.replay.update_priorities(
            replay_batch.indices,
            td_errors.cpu().numpy(),
        )

        # ----------------------------------------------------
        # ACTOR UPDATE
        # ----------------------------------------------------

        actor_loss = torch.zeros(
            (),
            device=self.device,
        )

        actor_grad_norm = torch.zeros(
            (),
            device=self.device,
        )

        entropy_loss = torch.zeros(
            (),
            device=self.device,
        )

        mean_log_probability = torch.zeros(
            (),
            device=self.device,
        )

        if (
            self.training_steps
            % self.config.actor_update_interval
            == 0
        ):
            (
                actor_loss,
                actor_grad_norm,
                log_probabilities,
            ) = self._update_actor(
                batch["states"]
            )

            mean_log_probability = (
                log_probabilities.mean()
            )

            entropy_loss = (
                self._update_entropy(
                    log_probabilities
                )
            )

        # ----------------------------------------------------
        # TARGET UPDATE
        # ----------------------------------------------------

        if (
            self.training_steps
            % self.config.target_update_interval
            == 0
        ):
            soft_update(
                self.target_critic,
                self.critic,
                self.config.tau,
            )

        self.training_steps += 1

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        with torch.no_grad():
            q_values = (
                self.critic.minimum(
                    batch["states"],
                    batch["actions"],
                )
            )

        real_samples = sum(
            source == ExperienceSource.REAL
            for source in replay_batch.sources
        )

        imagined_samples = (
            len(replay_batch)
            - real_samples
        )

        metrics = RLTrainingMetrics(
            critic_loss=float(
                critic_loss.item()
            ),
            actor_loss=float(
                actor_loss.item()
            ),
            entropy_loss=float(
                entropy_loss.item()
            ),
            entropy_temperature=float(
                self.alpha.detach().item()
            ),
            mean_q_value=float(
                q_values.mean().item()
            ),
            mean_log_probability=float(
                mean_log_probability.item()
            ),
            mean_reward=float(
                batch["rewards"].mean().item()
            ),
            mean_uncertainty=float(
                batch["uncertainties"].mean().item()
            ),
            real_samples=real_samples,
            imagined_samples=imagined_samples,
            critic_gradient_norm=float(
                critic_grad_norm.item()
            ),
            actor_gradient_norm=float(
                actor_grad_norm.item()
            ),
            training_step=self.training_steps,
        )

        self.metrics_history.append(
            metrics.to_dict()
        )

        return metrics

    # --------------------------------------------------------
    # TRAIN MULTIPLE STEPS
    # --------------------------------------------------------

    def train(
        self,
        gradient_steps: int = 100,
    ) -> list[RLTrainingMetrics]:
        if gradient_steps <= 0:
            raise ModelBasedRLValidationError(
                "gradient_steps must be positive"
            )

        results = []

        for _ in range(gradient_steps):
            results.append(
                self.train_step()
            )

        return results

    # --------------------------------------------------------
    # WORLD MODEL TRAINING CALLBACK
    # --------------------------------------------------------

    def train_world_model(
        self,
        train_fn: Callable[
            [dict[str, Tensor]],
            dict[str, float],
        ],
        *,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """
        Trains an external world model using
        real transitions only.

        train_fn(batch) must perform the
        world-model optimization step.

        This keeps the engine compatible
        with different world-model designs.
        """

        if batch_size is None:
            batch_size = (
                self.config.batch_size
            )

        (
            transitions,
            _,
            _,
            _,
        ) = self.replay.real_buffer.sample(
            batch_size,
            beta=0.0,
        )

        batch = {
            "states": torch.as_tensor(
                np.stack(
                    [t.state for t in transitions]
                ),
                device=self.device,
            ),
            "actions": torch.as_tensor(
                np.stack(
                    [t.action for t in transitions]
                ),
                device=self.device,
            ),
            "rewards": torch.as_tensor(
                np.asarray(
                    [t.reward for t in transitions],
                    dtype=np.float32,
                ),
                device=self.device,
            ),
            "next_states": torch.as_tensor(
                np.stack(
                    [t.next_state for t in transitions]
                ),
                device=self.device,
            ),
            "terminated": torch.as_tensor(
                np.asarray(
                    [t.terminated for t in transitions],
                    dtype=np.float32,
                ),
                device=self.device,
            ),
        }

        return train_fn(
            batch
        )

    # --------------------------------------------------------
    # INTRINSIC MODEL TRAINING
    # --------------------------------------------------------

    def train_intrinsic_models(
        self,
    ) -> dict[str, float]:
        if self.intrinsic_trainer is None:
            raise ModelBasedRLError(
                "Intrinsic rewards are disabled"
            )

        (
            transitions,
            _,
            _,
            _,
        ) = self.replay.real_buffer.sample(
            self.config.batch_size,
            beta=0.0,
        )

        states = torch.as_tensor(
            np.stack(
                [t.state for t in transitions]
            ),
            device=self.device,
        )

        actions = torch.as_tensor(
            np.stack(
                [t.action for t in transitions]
            ),
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.stack(
                [t.next_state for t in transitions]
            ),
            device=self.device,
        )

        return (
            self.intrinsic_trainer.train_step(
                states,
                actions,
                next_states,
            )
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "environment_steps": (
                self.environment_steps
            ),
            "training_steps": (
                self.training_steps
            ),
            "entropy_temperature": float(
                self.alpha.detach().item()
            ),
            "replay": (
                self.replay.statistics()
            ),
            "world_model_connected": (
                self.world_model is not None
            ),
            "intrinsic_rewards_enabled": (
                self.intrinsic_engine is not None
            ),
        }

    # --------------------------------------------------------
    # SAVE CHECKPOINT
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

        checkpoint: dict[str, Any] = {
            "config": asdict(
                self.config
            ),
            "actor": (
                self.actor.state_dict()
            ),
            "critic": (
                self.critic.state_dict()
            ),
            "target_critic": (
                self.target_critic.state_dict()
            ),
            "actor_optimizer": (
                self.actor_optimizer.state_dict()
            ),
            "critic_optimizer": (
                self.critic_optimizer.state_dict()
            ),
            "log_alpha": (
                self.log_alpha.detach().clone()
            ),
            "environment_steps": (
                self.environment_steps
            ),
            "training_steps": (
                self.training_steps
            ),
            "metrics_history": (
                self.metrics_history
            ),
        }

        if self.alpha_optimizer is not None:
            checkpoint["alpha_optimizer"] = (
                self.alpha_optimizer.state_dict()
            )

        if self.intrinsic_engine is not None:
            checkpoint["intrinsic_engine"] = (
                self.intrinsic_engine.state_dict()
            )

        torch.save(
            checkpoint,
            path,
        )

        return path

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        world_model: WorldModelAdapter | None = None,
        device: str = "cpu",
    ) -> "ModelBasedRLEngine":
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
            ModelBasedRLConfig(
                **config_data
            ),
            world_model=world_model,
        )

        engine.actor.load_state_dict(
            checkpoint["actor"]
        )

        engine.critic.load_state_dict(
            checkpoint["critic"]
        )

        engine.target_critic.load_state_dict(
            checkpoint["target_critic"]
        )

        engine.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        engine.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        with torch.no_grad():
            engine.log_alpha.copy_(
                checkpoint["log_alpha"]
            )

        if (
            engine.alpha_optimizer is not None
            and "alpha_optimizer" in checkpoint
        ):
            engine.alpha_optimizer.load_state_dict(
                checkpoint["alpha_optimizer"]
            )

        if (
            engine.intrinsic_engine is not None
            and "intrinsic_engine" in checkpoint
        ):
            engine.intrinsic_engine.load_state_dict(
                checkpoint["intrinsic_engine"]
            )

        engine.environment_steps = int(
            checkpoint["environment_steps"]
        )

        engine.training_steps = int(
            checkpoint["training_steps"]
        )

        engine.metrics_history = list(
            checkpoint.get(
                "metrics_history",
                [],
            )
        )

        return engine


# ============================================================
# DEMO WORLD MODEL
# ============================================================

class DemoWorldModel:
    """
    Synthetic dynamics used for demonstration.

    This is not a trained neural world model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

    def predict_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> ImaginedStep:
        next_state = state.copy()

        count = min(
            self.state_dim,
            self.action_dim,
        )

        next_state[:count] += (
            0.1 * action[:count]
        )

        reward = -float(
            np.linalg.norm(
                next_state
            )
        )

        uncertainty = (
            0.05
            + 0.02
            * float(
                np.linalg.norm(action)
            )
        )

        terminated = bool(
            np.linalg.norm(next_state) < 0.1
        )

        return ImaginedStep(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            uncertainty=uncertainty,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - MODEL-BASED RL ENGINE"
    )
    print("=" * 65)

    config = ModelBasedRLConfig(
        state_dim=16,
        action_dim=4,
        hidden_dim=128,
        batch_size=32,
        min_replay_size=32,
        initial_random_steps=20,
        real_capacity=1000,
        imagined_capacity=2000,
        imagined_ratio=0.5,
        imagination_horizon=5,
        use_intrinsic_rewards=True,
    )

    world_model = DemoWorldModel(
        config.state_dim,
        config.action_dim,
    )

    engine = ModelBasedRLEngine(
        config,
        world_model=world_model,
    )

    rng = np.random.default_rng(
        42
    )

    # --------------------------------------------------------
    # COLLECT REAL EXPERIENCE
    # --------------------------------------------------------

    for step in range(100):
        state = rng.normal(
            size=config.state_dim
        ).astype(np.float32)

        action = engine.select_action(
            state
        )

        next_state = state.copy()

        next_state[:config.action_dim] += (
            0.1 * action
        )

        reward = -float(
            np.linalg.norm(
                next_state
            )
        )

        engine.store_real_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            episode_id="demo_episode",
            step_index=step,
        )

    print(
        "Real transitions collected:",
        len(engine.replay.real_buffer),
    )

    # --------------------------------------------------------
    # TRAIN INTRINSIC MODELS
    # --------------------------------------------------------

    intrinsic_metrics = (
        engine.train_intrinsic_models()
    )

    print(
        "Intrinsic training:",
        intrinsic_metrics,
    )

    # --------------------------------------------------------
    # GENERATE IMAGINED EXPERIENCE
    # --------------------------------------------------------

    imagination_metrics = engine.imagine(
        num_start_states=20,
        horizon=5,
        model_version="demo_world_v1",
    )

    print(
        "Imagination:",
        imagination_metrics,
    )

    # --------------------------------------------------------
    # TRAIN ACTOR AND CRITIC
    # --------------------------------------------------------

    for step in range(10):
        metrics = engine.train_step()

        if step in (0, 4, 9):
            print(
                f"RL training step {step + 1}:",
                metrics.to_dict(),
            )

    # --------------------------------------------------------
    # POLICY INFERENCE
    # --------------------------------------------------------

    test_state = np.zeros(
        config.state_dim,
        dtype=np.float32,
    )

    action = engine.select_action(
        test_state,
        deterministic=True,
        exploration=False,
    )

    print(
        "Deterministic action:",
        action,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print(
        "Engine statistics:",
        engine.statistics(),
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = engine.save(
        "models/model_based_rl_demo.pt"
    )

    restored_engine = (
        ModelBasedRLEngine.load(
            checkpoint_path,
            world_model=world_model,
        )
    )

    print(
        "Restored training steps:",
        restored_engine.training_steps,
    )

    print(
        "Model-based RL demonstration completed"
    )


if __name__ == "__main__":
    main()
