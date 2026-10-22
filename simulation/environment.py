
```python
"""
NeuroWorld AI
File: neuroworld/environment/environment.py

World-Model Simulation Environment

Features:
- Gymnasium-style reset and step API
- Continuous, discrete, multi-discrete and hybrid actions
- Action encoding using ActionSpace
- Latent-state world-model simulation
- Goal-conditioned rewards
- Action penalties
- Predictive uncertainty penalties
- Episode termination and truncation
- Custom reward and termination functions
- Deterministic or stochastic predictions
- Environment history
- Simulation rollouts
- Seeded reproducibility
- CPU/CUDA compatibility

Dependencies:
    pip install torch

This environment uses a learned world model.
It is not a physical robot controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from neuroworld.environment.action_space import (
    ActionSpace,
    ActionSpaceConfig,
    HybridAction,
)


Action = Tensor | HybridAction

RewardFunction = Callable[
    [Tensor, Tensor, Tensor],
    Tensor,
]

TerminationFunction = Callable[
    [Tensor],
    Tensor,
]


@dataclass
class EnvironmentConfig:
    latent_dim: int = 128
    action_dim: int = 32

    max_episode_steps: int = 200

    goal_tolerance: float = 0.5
    goal_reward: float = 10.0
    step_penalty: float = 0.01
    action_penalty: float = 0.001

    goal_distance_weight: float = 1.0
    uncertainty_penalty: float = 0.0

    deterministic: bool = True
    temperature: float = 1.0

    terminate_on_goal: bool = True
    terminate_on_invalid_state: bool = True

    max_state_norm: Optional[float] = None

    history_capacity: int = 1000
    seed: Optional[int] = 42


@dataclass
class EnvironmentStep:
    observation: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    info: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EnvironmentTransition:
    state: Tensor
    action: Tensor
    next_state: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor


@dataclass
class EnvironmentStatistics:
    episode_steps: int
    total_steps: int
    mean_episode_reward: float
    completed_episodes: int
    active_environments: int


class WorldEnvironment:
    """
    Batched latent-space simulation environment.

    State:
        [B,latent_dim]

    Encoded action:
        [B,action_dim]

    Reward:
        [B]

    Terminated:
        [B]

    Truncated:
        [B]

    This is a batched environment.

    Individual batch elements that finish
    remain inactive until reset().
    """

    def __init__(
        self,
        world_model: nn.Module,
        config: Optional[
            EnvironmentConfig
        ] = None,
        action_space: Optional[
            ActionSpace
        ] = None,
        reward_fn: Optional[
            RewardFunction
        ] = None,
        termination_fn: Optional[
            TerminationFunction
        ] = None,
    ):
        self.config = (
            config or EnvironmentConfig()
        )

        self._validate_config()

        self.world_model = world_model

        self.action_space = (
            action_space
            if action_space is not None
            else ActionSpace(
                ActionSpaceConfig(
                    action_type="continuous",
                    action_dim=self.config.action_dim,
                )
            )
        )

        if (
            self.action_space.encoded_dim
            != self.config.action_dim
        ):
            raise ValueError(
                "Environment action_dim must "
                "match ActionSpace.encoded_dim"
            )

        self.reward_fn = reward_fn
        self.termination_fn = termination_fn

        self.state: Optional[Tensor] = None
        self.goal_state: Optional[Tensor] = None

        self.terminated: Optional[Tensor] = None
        self.truncated: Optional[Tensor] = None

        self.episode_rewards: Optional[
            Tensor
        ] = None

        self.episode_steps = 0
        self.total_steps = 0
        self.completed_episodes = 0

        self.history: List[
            EnvironmentTransition
        ] = []

        self._generator = torch.Generator(
            device="cpu"
        )

        if self.config.seed is not None:
            self.seed(self.config.seed)

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if cfg.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if cfg.max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be positive"
            )

        if cfg.history_capacity <= 0:
            raise ValueError(
                "history_capacity must be positive"
            )

        if cfg.goal_tolerance < 0:
            raise ValueError(
                "goal_tolerance must be nonnegative"
            )

        if cfg.temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        if (
            cfg.max_state_norm is not None
            and cfg.max_state_norm <= 0
        ):
            raise ValueError(
                "max_state_norm must be positive"
            )

        for name in (
            "goal_reward",
            "step_penalty",
            "action_penalty",
            "goal_distance_weight",
            "uncertainty_penalty",
        ):
            if not math.isfinite(
                getattr(cfg, name)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

    def seed(
        self,
        seed: int,
    ) -> None:
        self._generator.manual_seed(
            int(seed)
        )

    def _model_device(self) -> torch.device:
        parameter = next(
            self.world_model.parameters(),
            None,
        )

        if parameter is not None:
            return parameter.device

        buffer = next(
            self.world_model.buffers(),
            None,
        )

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    def _model_dtype(self) -> torch.dtype:
        parameter = next(
            self.world_model.parameters(),
            None,
        )

        if parameter is not None:
            return parameter.dtype

        return torch.float32

    def _validate_state(
        self,
        state: Tensor,
    ) -> None:
        if (
            state.ndim != 2
            or state.shape[1]
            != self.config.latent_dim
        ):
            raise ValueError(
                "State must be [B,latent_dim]"
            )

        if not torch.is_floating_point(state):
            raise TypeError(
                "State must be floating point"
            )

        if not torch.isfinite(state).all().item():
            raise ValueError(
                "Initial state contains NaN or Inf"
            )

    def _sample_initial_state(
        self,
        batch_size: int,
    ) -> Tensor:
        """
        Generate a synthetic initial latent state.

        For meaningful experiments, provide
        real encoded observations instead.
        """

        samples = torch.randn(
            batch_size,
            self.config.latent_dim,
            generator=self._generator,
            device="cpu",
            dtype=torch.float32,
        )

        return samples.to(
            device=self._model_device(),
            dtype=self._model_dtype(),
        )

    def reset(
        self,
        batch_size: int = 1,
        initial_state: Optional[
            Tensor
        ] = None,
        goal_state: Optional[
            Tensor
        ] = None,
        seed: Optional[int] = None,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Reset the simulation environment.

        Returns:
            observation [B,D]
            info dictionary
        """

        if seed is not None:
            self.seed(seed)

        if initial_state is None:
            if batch_size <= 0:
                raise ValueError(
                    "batch_size must be positive"
                )

            initial_state = (
                self._sample_initial_state(
                    batch_size
                )
            )

        self._validate_state(
            initial_state
        )

        if initial_state.device != self._model_device():
            raise ValueError(
                "Initial state must be on "
                "the world-model device"
            )

        if initial_state.dtype != self._model_dtype():
            raise ValueError(
                "Initial state dtype must match "
                "the world model"
            )

        if goal_state is not None:
            self._validate_state(
                goal_state
            )

            if goal_state.shape != initial_state.shape:
                raise ValueError(
                    "Goal shape must match initial state"
                )

            if (
                goal_state.device != initial_state.device
                or goal_state.dtype != initial_state.dtype
            ):
                raise ValueError(
                    "Goal device and dtype must "
                    "match initial state"
                )

        self.state = (
            initial_state.detach().clone()
        )

        self.goal_state = (
            goal_state.detach().clone()
            if goal_state is not None
            else None
        )

        batch_size = self.state.shape[0]

        self.terminated = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.state.device,
        )

        self.truncated = torch.zeros_like(
            self.terminated
        )

        self.episode_rewards = torch.zeros(
            batch_size,
            device=self.state.device,
            dtype=self.state.dtype,
        )

        self.episode_steps = 0
        self.history.clear()

        return self.state.clone(), {
            "batch_size": batch_size,
            "goal_enabled": (
                self.goal_state is not None
            ),
        }

    def _encode_action(
        self,
        action: Action,
    ) -> Tensor:
        """
        Validate and encode actions.

        Returns:
            [B,action_dim]
        """

        if self.state is None:
            raise RuntimeError(
                "Call reset() before step()"
            )

        validation = self.action_space.validate(
            action
        )

        if not validation.valid.all().item():
            raise ValueError(
                "Action contains invalid values"
            )

        encoded = self.action_space.encode(
            action
        )

        expected = (
            self.state.shape[0],
            self.config.action_dim,
        )

        if encoded.shape != expected:
            raise ValueError(
                f"Encoded action must be {expected}"
            )

        return encoded.to(
            device=self.state.device,
            dtype=self.state.dtype,
        )

    @torch.no_grad()
    def _predict_transition(
        self,
        state: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Predict next latent state.

        Supports:
            model(state, action) -> next_state

            model(state, action) ->
                (mean, logvar)

        Uncertainty is represented as
        mean predictive variance [B].
        """

        prediction = self.world_model(
            state,
            action,
        )

        if isinstance(prediction, tuple):
            if len(prediction) < 2:
                raise ValueError(
                    "World model tuple must "
                    "contain mean and logvar"
                )

            mean, logvar = prediction[:2]

            if (
                mean.shape != state.shape
                or logvar.shape != state.shape
            ):
                raise ValueError(
                    "World-model output shape mismatch"
                )

            logvar = logvar.clamp(
                min=-20.0,
                max=10.0,
            )

            uncertainty = (
                logvar.exp().mean(dim=-1)
            )

            if self.config.deterministic:
                next_state = mean
            else:
                noise = torch.randn_like(
                    mean
                )

                next_state = (
                    mean
                    + noise
                    * torch.exp(
                        0.5 * logvar
                    )
                    * self.config.temperature
                )

        else:
            next_state = prediction

            if next_state.shape != state.shape:
                raise ValueError(
                    "World-model output shape mismatch"
                )

            uncertainty = torch.zeros(
                state.shape[0],
                device=state.device,
                dtype=state.dtype,
            )

        return next_state, uncertainty

    def _compute_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        uncertainty: Tensor,
    ) -> Tensor:
        cfg = self.config

        batch_size = state.shape[0]

        reward = torch.full(
            (batch_size,),
            -cfg.step_penalty,
            device=state.device,
            dtype=state.dtype,
        )

        action_cost = (
            action.square().mean(dim=-1)
        )

        reward = (
            reward
            - cfg.action_penalty * action_cost
            - cfg.uncertainty_penalty * uncertainty
        )

        if self.goal_state is not None:
            goal_distance = (
                torch.linalg.vector_norm(
                    next_state
                    - self.goal_state,
                    dim=-1,
                )
            )

            reward = (
                reward
                - cfg.goal_distance_weight
                * goal_distance
            )

            reached = (
                goal_distance
                <= cfg.goal_tolerance
            )

            reward = (
                reward
                + reached.to(reward.dtype)
                * cfg.goal_reward
            )

        if self.reward_fn is not None:
            custom_reward = self.reward_fn(
                state,
                action,
                next_state,
            )

            if custom_reward.shape != (
                batch_size,
            ):
                raise ValueError(
                    "Custom reward must return [B]"
                )

            reward = reward + custom_reward

        return reward

    def _compute_termination(
        self,
        next_state: Tensor,
    ) -> Tensor:
        cfg = self.config

        batch_size = next_state.shape[0]

        terminated = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=next_state.device,
        )

        if (
            self.goal_state is not None
            and cfg.terminate_on_goal
        ):
            distances = (
                torch.linalg.vector_norm(
                    next_state
                    - self.goal_state,
                    dim=-1,
                )
            )

            terminated |= (
                distances <= cfg.goal_tolerance
            )

        if cfg.max_state_norm is not None:
            norms = torch.linalg.vector_norm(
                next_state,
                dim=-1,
            )

            terminated |= (
                norms > cfg.max_state_norm
            )

        if self.termination_fn is not None:
            custom_terminated = (
                self.termination_fn(
                    next_state
                )
            )

            if (
                custom_terminated.shape
                != (batch_size,)
                or custom_terminated.dtype
                != torch.bool
            ):
                raise ValueError(
                    "Termination function must "
                    "return boolean [B]"
                )

            terminated |= custom_terminated

        return terminated

    @torch.no_grad()
    def step(
        self,
        action: Action,
    ) -> EnvironmentStep:
        """
        Advance the simulated environment.

        Returns:
            observation [B,D]
            reward [B]
            terminated [B]
            truncated [B]
            info

        Finished batch elements remain
        inactive until the next reset().
        """

        if self.state is None:
            raise RuntimeError(
                "Call reset() before step()"
            )

        assert self.terminated is not None
        assert self.truncated is not None
        assert self.episode_rewards is not None

        active = ~(
            self.terminated
            | self.truncated
        )

        if not active.any().item():
            raise RuntimeError(
                "All environments finished; "
                "call reset()"
            )

        encoded_action = (
            self._encode_action(action)
        )

        previous_state = (
            self.state.clone()
        )

        next_state = previous_state.clone()

        uncertainty = torch.zeros(
            previous_state.shape[0],
            device=previous_state.device,
            dtype=previous_state.dtype,
        )

        active_indices = active.nonzero(
            as_tuple=True
        )[0]

        predicted, predicted_uncertainty = (
            self._predict_transition(
                previous_state[active],
                encoded_action[active],
            )
        )

        finite_prediction = torch.isfinite(
            predicted
        ).all(dim=-1)

        invalid = ~finite_prediction

        if (
            invalid.any().item()
            and not self.config
            .terminate_on_invalid_state
        ):
            raise FloatingPointError(
                "World model predicted NaN or Inf"
            )

        # Preserve previous state for invalid
        # predictions, rather than storing NaNs.
        predicted = torch.where(
            finite_prediction.unsqueeze(-1),
            predicted,
            previous_state[active],
        )

        predicted_uncertainty = torch.where(
            torch.isfinite(
                predicted_uncertainty
            ),
            predicted_uncertainty,
            torch.zeros_like(
                predicted_uncertainty
            ),
        )

        next_state[active] = predicted

        uncertainty[active] = (
            predicted_uncertainty
        )

        reward = torch.zeros(
            previous_state.shape[0],
            device=previous_state.device,
            dtype=previous_state.dtype,
        )

        # Compute rewards with full-batch goal
        # alignment, then mask inactive entries.
        computed_reward = self._compute_reward(
            previous_state,
            encoded_action,
            next_state,
            uncertainty,
        )

        reward[active] = (
            computed_reward[active]
        )

        # Compute termination on the full batch
        # to preserve goal batch alignment.
        new_terminated = (
            self._compute_termination(
                next_state
            )
        )

        invalid_full = torch.zeros_like(
            active
        )

        invalid_full[active_indices] = invalid

        new_terminated |= invalid_full

        self.terminated |= (
            active & new_terminated
        )

        self.episode_steps += 1
        self.total_steps += int(
            active.sum().item()
        )

        if (
            self.episode_steps
            >= self.config.max_episode_steps
        ):
            self.truncated |= (
                active & ~self.terminated
            )

        self.episode_rewards += reward

        self.state = next_state

        newly_finished = (
            active
            & (
                self.terminated
                | self.truncated
            )
        )

        self.completed_episodes += int(
            newly_finished.sum().item()
        )

        self.history.append(
            EnvironmentTransition(
                state=previous_state.detach().cpu(),
                action=encoded_action.detach().cpu(),
                next_state=next_state.detach().cpu(),
                reward=reward.detach().cpu(),
                terminated=self.terminated.detach().cpu(),
                truncated=self.truncated.detach().cpu(),
            )
        )

        if (
            len(self.history)
            > self.config.history_capacity
        ):
            self.history.pop(0)

        return EnvironmentStep(
            observation=next_state.clone(),
            reward=reward,
            terminated=self.terminated.clone(),
            truncated=self.truncated.clone(),
            info={
                "uncertainty": uncertainty,
                "active_mask": active,
                "invalid_prediction": invalid_full,
                "episode_steps": self.episode_steps,
                "episode_rewards": (
                    self.episode_rewards.clone()
                ),
            },
        )

    @torch.no_grad()
    def rollout(
        self,
        actions: Tensor,
        initial_state: Optional[
            Tensor
        ] = None,
        goal_state: Optional[
            Tensor
        ] = None,
    ) -> Dict[str, Tensor]:
        """
        Simulate an action sequence.

        actions:
            [B,H,A] encoded action vectors

        This method uses encoded actions
        directly and is intended for planning.

        It does not modify the active
        environment episode.
        """

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,H,A]"
            )

        if (
            actions.shape[-1]
            != self.config.action_dim
        ):
            raise ValueError(
                "Action dimension mismatch"
            )

        if initial_state is None:
            if self.state is None:
                raise RuntimeError(
                    "Initial state is required"
                )

            initial_state = self.state

        self._validate_state(
            initial_state
        )

        if (
            actions.shape[0]
            != initial_state.shape[0]
        ):
            raise ValueError(
                "Batch size mismatch"
            )

        if goal_state is not None:
            if goal_state.shape != initial_state.shape:
                raise ValueError(
                    "Goal shape mismatch"
                )

        # Temporarily use a local goal for
        # reward computation without mutating
        # the active environment goal.
        original_goal = self.goal_state

        try:
            self.goal_state = goal_state

            current = initial_state.clone()

            states = [current]
            rewards = []
            uncertainties = []

            for step_index in range(
                actions.shape[1]
            ):
                action = actions[
                    :,
                    step_index,
                ]

                next_state, uncertainty = (
                    self._predict_transition(
                        current,
                        action,
                    )
                )

                reward = self._compute_reward(
                    current,
                    action,
                    next_state,
                    uncertainty,
                )

                states.append(next_state)
                rewards.append(reward)
                uncertainties.append(
                    uncertainty
                )

                current = next_state

            return {
                "states": torch.stack(
                    states,
                    dim=1,
                ),
                "rewards": torch.stack(
                    rewards,
                    dim=1,
                ),
                "uncertainties": torch.stack(
                    uncertainties,
                    dim=1,
                ),
                "actions": actions,
            }

        finally:
            self.goal_state = original_goal

    def statistics(
        self,
    ) -> EnvironmentStatistics:
        if self.state is None:
            return EnvironmentStatistics(
                episode_steps=0,
                total_steps=self.total_steps,
                mean_episode_reward=0.0,
                completed_episodes=(
                    self.completed_episodes
                ),
                active_environments=0,
            )

        assert self.episode_rewards is not None
        assert self.terminated is not None
        assert self.truncated is not None

        active = ~(
            self.terminated
            | self.truncated
        )

        return EnvironmentStatistics(
            episode_steps=self.episode_steps,
            total_steps=self.total_steps,
            mean_episode_reward=float(
                self.episode_rewards.mean()
            ),
            completed_episodes=(
                self.completed_episodes
            ),
            active_environments=int(
                active.sum().item()
            ),
        )


if __name__ == "__main__":

    from neuroworld.dynamics.transition_model import (
        TransitionModel,
        TransitionModelConfig,
    )

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    latent_dim = 128
    action_dim = 32

    model = TransitionModel(
        TransitionModelConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            backend="stochastic",
        )
    ).to(device)

    action_space = ActionSpace(
        ActionSpaceConfig(
            action_type="continuous",
            action_dim=action_dim,
            device=str(device),
        )
    )

    environment = WorldEnvironment(
        world_model=model,
        config=EnvironmentConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            max_episode_steps=20,
            deterministic=True,
        ),
        action_space=action_space,
    )

    batch_size = 4

    initial_state = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    goal_state = torch.zeros_like(
        initial_state
    )

    observation, info = environment.reset(
        initial_state=initial_state,
        goal_state=goal_state,
    )

    print(
        "Initial observation:",
        observation.shape,
    )

    for step_index in range(20):
        action = action_space.sample(
            batch_size=batch_size
        )

        result = environment.step(
            action
        )

        print(
            f"Step {step_index + 1}:",
            "reward=",
            result.reward.mean().item(),
            "terminated=",
            result.terminated.tolist(),
            "truncated=",
            result.truncated.tolist(),
        )

        if (
            result.terminated
            | result.truncated
        ).all().item():
            break

    print(
        "Environment statistics:",
        environment.statistics(),
    )

    # Independent planning rollout.
    encoded_actions = torch.randn(
        batch_size,
        10,
        action_dim,
        device=device,
    ).clamp(-1.0, 1.0)

    imagined = environment.rollout(
        actions=encoded_actions,
        initial_state=initial_state,
        goal_state=goal_state,
    )

    print(
        "Imagined states:",
        imagined["states"].shape,
    )

    print(
        "Imagined rewards:",
        imagined["rewards"].shape,
    )
```
