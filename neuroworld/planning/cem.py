
"""
NeuroWorld AI
File: neuroworld/planning/cem.py

Cross-Entropy Method (CEM) Planning Engine.

Features:
    - Continuous action-sequence optimization
    - Model-based planning
    - Cross-Entropy Method optimization
    - Gaussian action-distribution updates
    - Elite trajectory selection
    - Reward-maximizing planning
    - Action bounds and constraints
    - Uncertainty-aware planning
    - Receding-horizon MPC
    - Warm-start optimization
    - Batch candidate evaluation
    - Custom world-model adapters
    - Planning diagnostics
    - NumPy-compatible interface
    - Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.planning.cem

Conventions:
    state: [state_dim]
    candidate actions: [N, H, action_dim]
    predicted states: [N, H, state_dim]

N = candidate population
H = planning horizon

CEM optimizes a learned or simulated model.
Its plans are only as reliable as that model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class CEMError(Exception):
    """Base CEM planning exception."""


class CEMValidationError(CEMError):
    """Invalid planner input or configuration."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CEMConfig:
    state_dim: int = 128
    action_dim: int = 8

    horizon: int = 15
    population_size: int = 256
    elite_size: int = 32
    iterations: int = 8

    action_min: float = -1.0
    action_max: float = 1.0

    initial_std: float = 0.5
    minimum_std: float = 0.05
    maximum_std: float = 1.0

    smoothing: float = 0.2

    discount_factor: float = 0.99

    uncertainty_penalty: float = 0.1
    action_penalty: float = 0.001

    warm_start: bool = True
    terminal_value_weight: float = 1.0

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "horizon",
            "population_size",
            "elite_size",
            "iterations",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.elite_size > self.population_size:
            raise ValueError(
                "elite_size cannot exceed population_size"
            )

        if self.action_min >= self.action_max:
            raise ValueError(
                "action_min must be less than action_max"
            )

        if not (
            0 < self.minimum_std
            <= self.initial_std
            <= self.maximum_std
        ):
            raise ValueError(
                "Invalid standard deviation bounds"
            )

        if not 0 <= self.smoothing < 1:
            raise ValueError(
                "smoothing must be in [0,1)"
            )

        if not 0 < self.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in (0,1]"
            )

        for name in (
            "uncertainty_penalty",
            "action_penalty",
            "terminal_value_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

class WorldModelProtocol(Protocol):
    """
    Required world-model interface.

    next_state:
        [N,state_dim]

    reward:
        [N]

    uncertainty:
        [N] or [N,state_dim]

    done:
        [N]

    All tensors should use the same device.
    """

    def predict_step(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        ...


# ============================================================
# WORLD MODEL ADAPTER
# ============================================================

class CEMWorldModelAdapter:
    """
    Adapts independent dynamics, reward,
    uncertainty and termination functions.

    Dynamics function:
        f(states, actions) -> next_states

    Reward function:
        r(states, actions, next_states) -> rewards

    Uncertainty function:
        u(states, actions, next_states) -> scores

    Termination function:
        d(states, actions, next_states) -> bool
    """

    def __init__(
        self,
        dynamics_fn: Callable[
            [Tensor, Tensor],
            Tensor,
        ],
        reward_fn: Callable[
            [Tensor, Tensor, Tensor],
            Tensor,
        ],
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
        termination_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn
        self.uncertainty_fn = uncertainty_fn
        self.termination_fn = termination_fn

    def predict_step(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        next_states = self.dynamics_fn(
            states,
            actions,
        )

        rewards = self.reward_fn(
            states,
            actions,
            next_states,
        )

        if self.uncertainty_fn is None:
            uncertainty = torch.zeros(
                states.shape[0],
                device=states.device,
            )
        else:
            uncertainty = self.uncertainty_fn(
                states,
                actions,
                next_states,
            )

        if self.termination_fn is None:
            done = torch.zeros(
                states.shape[0],
                dtype=torch.bool,
                device=states.device,
            )
        else:
            done = self.termination_fn(
                states,
                actions,
                next_states,
            )

        return (
            next_states,
            rewards,
            uncertainty,
            done,
        )


# ============================================================
# PLANNING OUTPUT
# ============================================================

@dataclass
class CEMPlanningResult:
    best_action_sequence: Tensor
    first_action: Tensor

    predicted_states: Tensor

    best_score: float
    expected_reward: float
    uncertainty_cost: float

    final_mean: Tensor
    final_std: Tensor

    elite_scores: Tensor

    optimization_history: list[dict[str, float]] = field(
        default_factory=list
    )

    def to_numpy(self) -> dict[str, Any]:
        def convert(tensor: Tensor) -> np.ndarray:
            return (
                tensor.detach()
                .cpu()
                .numpy()
            )

        return {
            "best_action_sequence": convert(
                self.best_action_sequence
            ),
            "first_action": convert(
                self.first_action
            ),
            "predicted_states": convert(
                self.predicted_states
            ),
            "best_score": self.best_score,
            "expected_reward": self.expected_reward,
            "uncertainty_cost": self.uncertainty_cost,
            "final_mean": convert(
                self.final_mean
            ),
            "final_std": convert(
                self.final_std
            ),
            "elite_scores": convert(
                self.elite_scores
            ),
            "optimization_history": (
                self.optimization_history
            ),
        }


# ============================================================
# CEM PLANNER
# ============================================================

class CEMPlanner:
    """
    Cross-Entropy Method planner.

    Optimization:
        1. Initialize action distribution
        2. Sample candidate action sequences
        3. Simulate candidate trajectories
        4. Evaluate discounted rewards
        5. Select elite candidates
        6. Update Gaussian distribution
        7. Repeat until iterations complete

    The best action sequence is returned.
    """

    def __init__(
        self,
        world_model: WorldModelProtocol,
        config: CEMConfig | None = None,
        *,
        terminal_value_fn: (
            Callable[[Tensor], Tensor]
            | None
        ) = None,
    ) -> None:
        self.config = config or CEMConfig()

        self.world_model = world_model
        self.terminal_value_fn = (
            terminal_value_fn
        )

        self.device = torch.device(
            self.config.device
        )

        self._generator = torch.Generator(
            device=self.device
        )

        self._generator.manual_seed(
            self.config.seed
        )

        self._warm_start_mean: Tensor | None = None

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _prepare_state(
        self,
        state: Tensor | np.ndarray,
    ) -> Tensor:
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        if state.ndim == 2:
            if state.shape[0] != 1:
                raise CEMValidationError(
                    "Planner accepts one initial state"
                )

            state = state[0]

        if state.shape != (
            self.config.state_dim,
        ):
            raise CEMValidationError(
                f"Expected state [{self.config.state_dim}]"
            )

        if not torch.isfinite(state).all():
            raise CEMValidationError(
                "Initial state contains non-finite values"
            )

        return state

    # --------------------------------------------------------
    # DISTRIBUTION INITIALIZATION
    # --------------------------------------------------------

    def _initialize_distribution(
        self,
    ) -> tuple[Tensor, Tensor]:
        shape = (
            self.config.horizon,
            self.config.action_dim,
        )

        mean = torch.zeros(
            shape,
            device=self.device,
        )

        if (
            self.config.warm_start
            and self._warm_start_mean is not None
        ):
            mean = (
                self._warm_start_mean.clone()
            )

        mean = mean.clamp(
            self.config.action_min,
            self.config.action_max,
        )

        std = torch.full(
            shape,
            self.config.initial_std,
            device=self.device,
        )

        return mean, std

    # --------------------------------------------------------
    # ACTION SAMPLING
    # --------------------------------------------------------

    def _sample_actions(
        self,
        mean: Tensor,
        std: Tensor,
    ) -> Tensor:
        noise = torch.randn(
            self.config.population_size,
            self.config.horizon,
            self.config.action_dim,
            device=self.device,
            generator=self._generator,
        )

        samples = (
            mean.unsqueeze(0)
            + std.unsqueeze(0) * noise
        )

        return samples.clamp(
            self.config.action_min,
            self.config.action_max,
        )

    # --------------------------------------------------------
    # MODEL OUTPUT VALIDATION
    # --------------------------------------------------------

    def _validate_model_output(
        self,
        next_states: Tensor,
        rewards: Tensor,
        uncertainty: Tensor,
        done: Tensor,
        *,
        population_size: int,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        uncertainty = torch.as_tensor(
            uncertainty,
            dtype=torch.float32,
            device=self.device,
        )

        done = torch.as_tensor(
            done,
            dtype=torch.bool,
            device=self.device,
        )

        expected_state_shape = (
            population_size,
            self.config.state_dim,
        )

        if next_states.shape != expected_state_shape:
            raise CEMValidationError(
                "World model returned invalid "
                "next-state shape"
            )

        if rewards.shape != (
            population_size,
        ):
            raise CEMValidationError(
                "World model returned invalid reward shape"
            )

        if uncertainty.ndim == 2:
            if uncertainty.shape[0] != population_size:
                raise CEMValidationError(
                    "Invalid uncertainty batch size"
                )

            uncertainty = uncertainty.mean(
                dim=-1
            )

        if uncertainty.shape != (
            population_size,
        ):
            raise CEMValidationError(
                "Invalid uncertainty shape"
            )

        if done.shape != (
            population_size,
        ):
            raise CEMValidationError(
                "Invalid termination shape"
            )

        if (
            not torch.isfinite(next_states).all()
            or not torch.isfinite(rewards).all()
            or not torch.isfinite(uncertainty).all()
        ):
            raise CEMValidationError(
                "World model returned non-finite values"
            )

        if (uncertainty < 0).any():
            raise CEMValidationError(
                "Uncertainty cannot be negative"
            )

        return (
            next_states,
            rewards,
            uncertainty,
            done,
        )

    # --------------------------------------------------------
    # CANDIDATE EVALUATION
    # --------------------------------------------------------

    @torch.no_grad()
    def evaluate_candidates(
        self,
        initial_state: Tensor | np.ndarray,
        action_sequences: Tensor,
    ) -> dict[str, Tensor]:
        initial_state = self._prepare_state(
            initial_state
        )

        action_sequences = torch.as_tensor(
            action_sequences,
            dtype=torch.float32,
            device=self.device,
        )

        if action_sequences.ndim != 3:
            raise CEMValidationError(
                "Expected candidate actions [N,H,A]"
            )

        population_size, horizon, action_dim = (
            action_sequences.shape
        )

        if (
            population_size <= 0
            or horizon != self.config.horizon
            or action_dim != self.config.action_dim
        ):
            raise CEMValidationError(
                "Candidate action dimensions mismatch"
            )

        if not torch.isfinite(
            action_sequences
        ).all():
            raise CEMValidationError(
                "Candidate actions contain non-finite values"
            )

        if (
            (action_sequences < self.config.action_min).any()
            or (action_sequences > self.config.action_max).any()
        ):
            raise CEMValidationError(
                "Candidate actions violate bounds"
            )

        states = initial_state.unsqueeze(
            0
        ).expand(
            population_size,
            -1,
        ).clone()

        alive = torch.ones(
            population_size,
            dtype=torch.bool,
            device=self.device,
        )

        total_reward = torch.zeros(
            population_size,
            device=self.device,
        )

        total_uncertainty = torch.zeros_like(
            total_reward
        )

        total_action_cost = torch.zeros_like(
            total_reward
        )

        predicted_states = []

        discount = 1.0

        for step in range(
            self.config.horizon
        ):
            actions = action_sequences[
                :, step
            ]

            (
                next_states,
                rewards,
                uncertainty,
                done,
            ) = self.world_model.predict_step(
                states,
                actions,
            )

            (
                next_states,
                rewards,
                uncertainty,
                done,
            ) = self._validate_model_output(
                next_states,
                rewards,
                uncertainty,
                done,
                population_size=population_size,
            )

            alive_float = alive.float()

            total_reward += (
                discount
                * rewards
                * alive_float
            )

            total_uncertainty += (
                discount
                * uncertainty
                * alive_float
            )

            action_cost = (
                actions.square()
                .sum(dim=-1)
            )

            total_action_cost += (
                discount
                * action_cost
                * alive_float
            )

            # Keep terminal states fixed.
            states = torch.where(
                alive.unsqueeze(-1),
                next_states,
                states,
            )

            predicted_states.append(
                states
            )

            alive = (
                alive & ~done
            )

            discount *= (
                self.config.discount_factor
            )

        if self.terminal_value_fn is not None:
            terminal_values = (
                self.terminal_value_fn(
                    states
                )
            )

            terminal_values = torch.as_tensor(
                terminal_values,
                dtype=torch.float32,
                device=self.device,
            )

            if terminal_values.shape != (
                population_size,
            ):
                raise CEMValidationError(
                    "Invalid terminal value shape"
                )

            if not torch.isfinite(
                terminal_values
            ).all():
                raise CEMValidationError(
                    "Non-finite terminal values"
                )

            total_reward += (
                discount
                * self.config.terminal_value_weight
                * terminal_values
                * alive.float()
            )

        uncertainty_cost = (
            self.config.uncertainty_penalty
            * total_uncertainty
        )

        action_penalty_cost = (
            self.config.action_penalty
            * total_action_cost
        )

        scores = (
            total_reward
            - uncertainty_cost
            - action_penalty_cost
        )

        trajectories = torch.stack(
            predicted_states,
            dim=1,
        )

        return {
            "scores": scores,
            "rewards": total_reward,
            "uncertainty_cost": uncertainty_cost,
            "action_cost": action_penalty_cost,
            "trajectories": trajectories,
        }

    # --------------------------------------------------------
    # OPTIMIZATION
    # --------------------------------------------------------

    @torch.no_grad()
    def plan(
        self,
        initial_state: Tensor | np.ndarray,
    ) -> CEMPlanningResult:
        state = self._prepare_state(
            initial_state
        )

        mean, std = (
            self._initialize_distribution()
        )

        best_score = -math.inf
        best_sequence = None
        best_trajectory = None

        best_reward = 0.0
        best_uncertainty_cost = 0.0

        optimization_history = []
        final_elite_scores = None

        for iteration in range(
            self.config.iterations
        ):
            candidates = self._sample_actions(
                mean,
                std,
            )

            # Always evaluate the current
            # distribution mean as a candidate.
            candidates[0] = mean.clamp(
                self.config.action_min,
                self.config.action_max,
            )

            evaluation = self.evaluate_candidates(
                state,
                candidates,
            )

            scores = evaluation[
                "scores"
            ]

            elite_scores, elite_indices = (
                torch.topk(
                    scores,
                    k=self.config.elite_size,
                    largest=True,
                )
            )

            elite_actions = candidates[
                elite_indices
            ]

            elite_mean = elite_actions.mean(
                dim=0
            )

            elite_std = elite_actions.std(
                dim=0,
                unbiased=False,
            ).clamp(
                self.config.minimum_std,
                self.config.maximum_std,
            )

            smoothing = (
                self.config.smoothing
            )

            mean = (
                smoothing * mean
                + (1.0 - smoothing)
                * elite_mean
            )

            std = (
                smoothing * std
                + (1.0 - smoothing)
                * elite_std
            ).clamp(
                self.config.minimum_std,
                self.config.maximum_std,
            )

            iteration_best_index = int(
                scores.argmax().item()
            )

            iteration_best_score = float(
                scores[
                    iteration_best_index
                ].item()
            )

            if (
                iteration_best_score
                > best_score
            ):
                best_score = (
                    iteration_best_score
                )

                best_sequence = (
                    candidates[
                        iteration_best_index
                    ].clone()
                )

                best_trajectory = (
                    evaluation["trajectories"][
                        iteration_best_index
                    ].clone()
                )

                best_reward = float(
                    evaluation["rewards"][
                        iteration_best_index
                    ].item()
                )

                best_uncertainty_cost = float(
                    evaluation["uncertainty_cost"][
                        iteration_best_index
                    ].item()
                )

            optimization_history.append(
                {
                    "iteration": float(
                        iteration + 1
                    ),
                    "best_score": (
                        iteration_best_score
                    ),
                    "mean_score": float(
                        scores.mean().item()
                    ),
                    "elite_mean_score": float(
                        elite_scores.mean().item()
                    ),
                    "distribution_std": float(
                        std.mean().item()
                    ),
                }
            )

            final_elite_scores = (
                elite_scores.clone()
            )

        if best_sequence is None:
            raise CEMError(
                "CEM failed to produce a plan"
            )

        if self.config.warm_start:
            self._warm_start_mean = torch.cat(
                [
                    best_sequence[1:],
                    best_sequence[-1:],
                ],
                dim=0,
            ).detach().clone()

        return CEMPlanningResult(
            best_action_sequence=best_sequence,
            first_action=best_sequence[0],
            predicted_states=best_trajectory,
            best_score=best_score,
            expected_reward=best_reward,
            uncertainty_cost=(
                best_uncertainty_cost
            ),
            final_mean=mean.clone(),
            final_std=std.clone(),
            elite_scores=final_elite_scores,
            optimization_history=(
                optimization_history
            ),
        )

    # --------------------------------------------------------
    # RECEDING-HORIZON CONTROL
    # --------------------------------------------------------

    def act(
        self,
        current_state: Tensor | np.ndarray,
    ) -> Tensor:
        """
        Replan and return the first action.

        MPC usage:
            observe state
            plan
            execute first action
            observe again
            replan
        """

        result = self.plan(
            current_state
        )

        return result.first_action

    def reset(self) -> None:
        self._warm_start_mean = None


# ============================================================
# SIMPLE MODEL-BASED MPC CONTROLLER
# ============================================================

class CEMMPCController:
    """
    Wraps the CEM planner for repeated
    receding-horizon planning.
    """

    def __init__(
        self,
        planner: CEMPlanner,
    ) -> None:
        self.planner = planner
        self.last_plan: (
            CEMPlanningResult | None
        ) = None

        self.control_steps = 0

    def select_action(
        self,
        current_state: Tensor | np.ndarray,
    ) -> Tensor:
        self.last_plan = self.planner.plan(
            current_state
        )

        self.control_steps += 1

        return (
            self.last_plan.first_action
        )

    def reset(self) -> None:
        self.planner.reset()
        self.last_plan = None
        self.control_steps = 0


# ============================================================
# NUMPY PLANNER ADAPTER
# ============================================================

class CEMPlannerAdapter:
    """
    NumPy-compatible interface for
    backend APIs and simulation modules.
    """

    def __init__(
        self,
        planner: CEMPlanner,
    ) -> None:
        self.planner = planner

    def plan(
        self,
        state: np.ndarray,
    ) -> dict[str, Any]:
        result = self.planner.plan(
            state
        )

        return result.to_numpy()

    def select_action(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        action = self.planner.act(
            state
        )

        return (
            action.detach()
            .cpu()
            .numpy()
        )


# ============================================================
# EXAMPLE WORLD MODEL
# ============================================================

class SimpleNavigationWorldModel:
    """
    Simple 2D point-mass navigation model.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    Dynamics:
        velocity += action * dt
        position += velocity * dt

    Reward:
        negative distance to goal

    This is a demonstration environment,
    not a learned neural world model.
    """

    def __init__(
        self,
        goal: tuple[float, float] = (5.0, 5.0),
        timestep: float = 0.1,
        maximum_speed: float = 3.0,
    ) -> None:
        self.goal = goal
        self.timestep = timestep
        self.maximum_speed = maximum_speed

    def predict_step(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        positions = states[
            :, :2
        ]

        velocities = states[
            :, 2:4
        ]

        next_velocities = (
            velocities
            + actions * self.timestep
        )

        speed = torch.linalg.vector_norm(
            next_velocities,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        speed_scale = (
            self.maximum_speed / speed
        ).clamp(max=1.0)

        next_velocities = (
            next_velocities
            * speed_scale
        )

        next_positions = (
            positions
            + next_velocities
            * self.timestep
        )

        next_states = torch.cat(
            [
                next_positions,
                next_velocities,
            ],
            dim=-1,
        )

        goal = torch.tensor(
            self.goal,
            dtype=states.dtype,
            device=states.device,
        )

        distance_to_goal = (
            torch.linalg.vector_norm(
                next_positions - goal,
                dim=-1,
            )
        )

        rewards = (
            -distance_to_goal
            - 0.01
            * actions.square().sum(
                dim=-1
            )
        )

        uncertainty = torch.full_like(
            rewards,
            0.01,
        )

        done = (
            distance_to_goal < 0.2
        )

        return (
            next_states,
            rewards,
            uncertainty,
            done,
        )


# ============================================================
# FACTORY
# ============================================================

def create_cem_planner(
    world_model: WorldModelProtocol,
    config: CEMConfig | None = None,
    *,
    terminal_value_fn: (
        Callable[[Tensor], Tensor]
        | None
    ) = None,
) -> CEMPlanner:
    return CEMPlanner(
        world_model=world_model,
        config=config,
        terminal_value_fn=terminal_value_fn,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - CEM PLANNING ENGINE"
    )
    print("=" * 65)

    config = CEMConfig(
        state_dim=4,
        action_dim=2,
        horizon=20,
        population_size=256,
        elite_size=32,
        iterations=8,
        action_min=-1.0,
        action_max=1.0,
        initial_std=0.5,
        uncertainty_penalty=0.1,
        warm_start=True,
    )

    world_model = (
        SimpleNavigationWorldModel(
            goal=(5.0, 5.0),
        )
    )

    planner = CEMPlanner(
        world_model,
        config,
    )

    initial_state = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    result = planner.plan(
        initial_state
    )

    print(
        "Best score:",
        result.best_score,
    )

    print(
        "First action:",
        result.first_action.cpu().numpy(),
    )

    print(
        "Action sequence shape:",
        tuple(
            result.best_action_sequence.shape
        ),
    )

    print(
        "Predicted states shape:",
        tuple(
            result.predicted_states.shape
        ),
    )

    print(
        "Predicted final position:",
        result.predicted_states[-1, :2]
        .cpu()
        .numpy(),
    )

    # --------------------------------------------------------
    # MPC
    # --------------------------------------------------------

    controller = CEMMPCController(
        planner
    )

    current_state = torch.tensor(
        initial_state
    )

    for step in range(5):
        action = controller.select_action(
            current_state
        )

        (
            next_state,
            reward,
            uncertainty,
            done,
        ) = world_model.predict_step(
            current_state.unsqueeze(0),
            action.unsqueeze(0),
        )

        current_state = (
            next_state[0]
        )

        print(
            f"MPC step {step + 1}:",
            "action=",
            action.cpu().numpy(),
            "position=",
            current_state[:2].cpu().numpy(),
            "reward=",
            float(reward[0].item()),
        )

        if bool(done[0].item()):
            print(
                "Goal reached"
            )
            break

    print(
        "CEM planning demonstration completed"
    )


if __name__ == "__main__":
    main()
