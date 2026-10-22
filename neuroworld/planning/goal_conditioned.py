
"""
NeuroWorld AI
File: neuroworld/planning/goal_conditioned.py

Goal-Conditioned Planning Engine.

Features:
    1. Goal representation and validation
    2. Goal-conditioned reward functions
    3. Euclidean and weighted state distances
    4. Partial-state goal matching
    5. Goal reachability estimation
    6. CEM-based action-sequence planning
    7. World-model trajectory rollouts
    8. Uncertainty-aware action selection
    9. Multi-goal priority selection
   10. Receding-horizon MPC
   11. Goal progress tracking
   12. Goal completion detection
   13. NumPy integration
   14. Synthetic navigation demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.planning.goal_conditioned

Integration:
    neuroworld/planning/cem.py
    neuroworld/prediction/next_state.py
    neuroworld/rl/reward_model.py

Important:
    Learned world-model predictions are not
    guaranteed to match real-world dynamics.

    Safety-critical actions require independent
    validation and authorized control systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor

from neuroworld.planning.cem import (
    CEMConfig,
    CEMPlanner,
    CEMPlanningResult,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class GoalPlanningError(Exception):
    """Base goal-planning exception."""


class GoalValidationError(GoalPlanningError):
    """Invalid goal or state representation."""


# ============================================================
# GOAL STATUS
# ============================================================

class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    REACHED = "reached"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GoalConditionedConfig:
    state_dim: int = 128
    action_dim: int = 8

    planning_horizon: int = 15
    population_size: int = 256
    elite_size: int = 32
    cem_iterations: int = 8

    action_min: float = -1.0
    action_max: float = 1.0

    goal_tolerance: float = 0.25

    distance_reward_weight: float = 1.0
    progress_reward_weight: float = 2.0
    success_reward: float = 10.0

    action_penalty_weight: float = 0.01
    uncertainty_penalty_weight: float = 0.1

    discount_factor: float = 0.99

    reachability_temperature: float = 1.0

    max_control_steps: int = 100

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "planning_horizon",
            "population_size",
            "elite_size",
            "cem_iterations",
            "max_control_steps",
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
                "Invalid action bounds"
            )

        if self.goal_tolerance <= 0:
            raise ValueError(
                "goal_tolerance must be positive"
            )

        if not 0 < self.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in (0,1]"
            )

        if self.reachability_temperature <= 0:
            raise ValueError(
                "reachability_temperature must be positive"
            )

        for name in (
            "distance_reward_weight",
            "progress_reward_weight",
            "success_reward",
            "action_penalty_weight",
            "uncertainty_penalty_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# GOAL REPRESENTATION
# ============================================================

@dataclass
class Goal:
    goal_id: str
    target_state: Tensor | np.ndarray

    tolerance: float = 0.25
    priority: int = 1

    description: str = ""

    # Select which state dimensions matter.
    state_mask: Tensor | np.ndarray | None = None

    # Optional per-dimension distance weights.
    dimension_weights: (
        Tensor | np.ndarray | None
    ) = None

    status: GoalStatus = GoalStatus.PENDING

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise GoalValidationError(
                "goal_id cannot be empty"
            )

        if self.tolerance <= 0:
            raise GoalValidationError(
                "Goal tolerance must be positive"
            )

        if self.priority < 0:
            raise GoalValidationError(
                "Goal priority cannot be negative"
            )


# ============================================================
# GOAL DISTANCE
# ============================================================

class GoalDistanceMetric:
    """
    Weighted Euclidean distance.

    Supports partial-state goals using masks.

    For example:
        State = [x, y, vx, vy]

        Goal may constrain only x and y.
    """

    def __init__(
        self,
        state_dim: int,
        device: torch.device,
    ) -> None:
        self.state_dim = state_dim
        self.device = device

    def _vector(
        self,
        value: Tensor | np.ndarray,
        name: str,
    ) -> Tensor:
        vector = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.device,
        )

        if vector.shape != (
            self.state_dim,
        ):
            raise GoalValidationError(
                f"{name} must have shape "
                f"[{self.state_dim}]"
            )

        if not torch.isfinite(vector).all():
            raise GoalValidationError(
                f"{name} contains non-finite values"
            )

        return vector

    def distance(
        self,
        states: Tensor,
        goal: Goal,
    ) -> Tensor:
        target = self._vector(
            goal.target_state,
            "target_state",
        )

        if states.ndim == 1:
            states = states.unsqueeze(0)

        if (
            states.ndim != 2
            or states.shape[1] != self.state_dim
        ):
            raise GoalValidationError(
                "States must be [B,state_dim]"
            )

        if not torch.isfinite(states).all():
            raise GoalValidationError(
                "States contain non-finite values"
            )

        if goal.state_mask is None:
            mask = torch.ones(
                self.state_dim,
                device=self.device,
            )
        else:
            mask = self._vector(
                goal.state_mask,
                "state_mask",
            )

            if not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise GoalValidationError(
                    "state_mask must contain 0 or 1"
                )

        if mask.sum() == 0:
            raise GoalValidationError(
                "At least one goal dimension "
                "must be selected"
            )

        if goal.dimension_weights is None:
            weights = torch.ones(
                self.state_dim,
                device=self.device,
            )
        else:
            weights = self._vector(
                goal.dimension_weights,
                "dimension_weights",
            )

            if (weights < 0).any():
                raise GoalValidationError(
                    "Distance weights must be nonnegative"
                )

        effective_weights = (
            mask * weights
        )

        if effective_weights.sum() <= 0:
            raise GoalValidationError(
                "Active goal weights must be positive"
            )

        squared_difference = (
            states - target
        ).square()

        return (
            (
                squared_difference
                * effective_weights
            ).sum(dim=-1)
        ).sqrt()

    def reached(
        self,
        states: Tensor,
        goal: Goal,
    ) -> Tensor:
        return (
            self.distance(
                states,
                goal,
            )
            <= goal.tolerance
        )


# ============================================================
# GOAL REWARD FUNCTION
# ============================================================

class GoalRewardFunction:
    """
    Goal-conditioned reward shaping.

    Components:
        negative distance
        progress toward goal
        success bonus
        action penalty
    """

    def __init__(
        self,
        metric: GoalDistanceMetric,
        config: GoalConditionedConfig,
    ) -> None:
        self.metric = metric
        self.config = config

    def calculate(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        goal: Goal,
    ) -> Tensor:
        current_distance = (
            self.metric.distance(
                states,
                goal,
            )
        )

        next_distance = (
            self.metric.distance(
                next_states,
                goal,
            )
        )

        progress = (
            current_distance
            - next_distance
        )

        reached = (
            next_distance
            <= goal.tolerance
        )

        distance_reward = (
            -self.config.distance_reward_weight
            * next_distance
        )

        progress_reward = (
            self.config.progress_reward_weight
            * progress
        )

        success_bonus = (
            reached.float()
            * self.config.success_reward
        )

        action_penalty = (
            self.config.action_penalty_weight
            * actions.square().sum(
                dim=-1
            )
        )

        return (
            distance_reward
            + progress_reward
            + success_bonus
            - action_penalty
        )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

class GoalDynamicsProtocol(Protocol):
    """
    Dynamics model used by goal planner.

    predict_next_state:
        states: [B,state_dim]
        actions: [B,action_dim]

    returns:
        next_states: [B,state_dim]
    """

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        ...


# ============================================================
# GOAL-CONDITIONED WORLD MODEL
# ============================================================

class GoalConditionedWorldModel:
    """
    Adapts state dynamics to CEM.

    Rewards and termination are generated
    from the active goal.
    """

    def __init__(
        self,
        dynamics: GoalDynamicsProtocol,
        goal: Goal,
        metric: GoalDistanceMetric,
        reward_function: GoalRewardFunction,
        *,
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.dynamics = dynamics
        self.goal = goal
        self.metric = metric
        self.reward_function = (
            reward_function
        )

        self.uncertainty_fn = (
            uncertainty_fn
        )

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
        next_states = (
            self.dynamics.predict_next_state(
                states,
                actions,
            )
        )

        rewards = (
            self.reward_function.calculate(
                states,
                actions,
                next_states,
                self.goal,
            )
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

        done = self.metric.reached(
            next_states,
            self.goal,
        )

        return (
            next_states,
            rewards,
            uncertainty,
            done,
        )


# ============================================================
# GOAL REACHABILITY
# ============================================================

@dataclass
class GoalReachabilityResult:
    goal_id: str
    initial_distance: float
    predicted_final_distance: float

    predicted_reachable: bool
    reachability_score: float

    planning_score: float


class GoalReachabilityEstimator:
    """
    Estimate whether a goal appears reachable
    within the current planning horizon.

    The score is a heuristic based on
    predicted final distance.

    It is not a calibrated probability.
    """

    def __init__(
        self,
        metric: GoalDistanceMetric,
        config: GoalConditionedConfig,
    ) -> None:
        self.metric = metric
        self.config = config

    def evaluate(
        self,
        initial_state: Tensor,
        result: CEMPlanningResult,
        goal: Goal,
    ) -> GoalReachabilityResult:
        initial_distance = float(
            self.metric.distance(
                initial_state.unsqueeze(0),
                goal,
            )[0].item()
        )

        predicted_final_distance = float(
            self.metric.distance(
                result.predicted_states[-1].unsqueeze(0),
                goal,
            )[0].item()
        )

        # Smooth distance-based heuristic.
        score = float(
            torch.exp(
                torch.tensor(
                    -predicted_final_distance
                    / self.config.reachability_temperature
                )
            ).item()
        )

        return GoalReachabilityResult(
            goal_id=goal.goal_id,
            initial_distance=initial_distance,
            predicted_final_distance=(
                predicted_final_distance
            ),
            predicted_reachable=(
                predicted_final_distance
                <= goal.tolerance
            ),
            reachability_score=score,
            planning_score=result.best_score,
        )


# ============================================================
# GOAL PLANNING RESULT
# ============================================================

@dataclass
class GoalPlanningResult:
    goal_id: str
    goal_status: GoalStatus

    first_action: Tensor
    action_sequence: Tensor
    predicted_states: Tensor

    initial_distance: float
    final_distance: float

    predicted_reachable: bool
    reachability_score: float

    planning_score: float
    uncertainty_cost: float

    optimization_history: list[
        dict[str, float]
    ]

    def to_numpy(self) -> dict[str, Any]:
        def convert(
            value: Tensor,
        ) -> np.ndarray:
            return (
                value.detach()
                .cpu()
                .numpy()
            )

        return {
            "goal_id": self.goal_id,
            "goal_status": self.goal_status.value,
            "first_action": convert(
                self.first_action
            ),
            "action_sequence": convert(
                self.action_sequence
            ),
            "predicted_states": convert(
                self.predicted_states
            ),
            "initial_distance": (
                self.initial_distance
            ),
            "final_distance": (
                self.final_distance
            ),
            "predicted_reachable": (
                self.predicted_reachable
            ),
            "reachability_score": (
                self.reachability_score
            ),
            "planning_score": (
                self.planning_score
            ),
            "uncertainty_cost": (
                self.uncertainty_cost
            ),
            "optimization_history": (
                self.optimization_history
            ),
        }


# ============================================================
# GOAL-CONDITIONED PLANNER
# ============================================================

class GoalConditionedPlanner:
    """
    Goal-conditioned planning with CEM.

    Pipeline:
        current state
            |
        active goal
            |
        goal reward function
            |
        world-model rollouts
            |
        CEM action optimization
            |
        predicted goal reachability
            |
        first MPC action
    """

    def __init__(
        self,
        dynamics: GoalDynamicsProtocol,
        config: GoalConditionedConfig | None = None,
        *,
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.config = (
            config or GoalConditionedConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.dynamics = dynamics
        self.uncertainty_fn = (
            uncertainty_fn
        )

        self.metric = GoalDistanceMetric(
            self.config.state_dim,
            self.device,
        )

        self.reward_function = GoalRewardFunction(
            self.metric,
            self.config,
        )

        self.reachability = (
            GoalReachabilityEstimator(
                self.metric,
                self.config,
            )
        )

        self._active_goal: Goal | None = None
        self._planner: CEMPlanner | None = None

    # --------------------------------------------------------
    # CEM CONFIGURATION
    # --------------------------------------------------------

    def _cem_config(self) -> CEMConfig:
        return CEMConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            horizon=self.config.planning_horizon,
            population_size=(
                self.config.population_size
            ),
            elite_size=self.config.elite_size,
            iterations=self.config.cem_iterations,
            action_min=self.config.action_min,
            action_max=self.config.action_max,
            discount_factor=(
                self.config.discount_factor
            ),
            uncertainty_penalty=(
                self.config.uncertainty_penalty_weight
            ),
            action_penalty=0.0,
            warm_start=True,
            seed=self.config.seed,
            device=self.config.device,
        )

    # --------------------------------------------------------
    # GOAL MANAGEMENT
    # --------------------------------------------------------

    def set_goal(
        self,
        goal: Goal,
    ) -> None:
        # Validate goal dimensions and weights
        # before creating a planner.
        reference_state = torch.zeros(
            1,
            self.config.state_dim,
            device=self.device,
        )

        self.metric.distance(
            reference_state,
            goal,
        )

        world_model = (
            GoalConditionedWorldModel(
                dynamics=self.dynamics,
                goal=goal,
                metric=self.metric,
                reward_function=(
                    self.reward_function
                ),
                uncertainty_fn=(
                    self.uncertainty_fn
                ),
            )
        )

        self._planner = CEMPlanner(
            world_model,
            self._cem_config(),
        )

        self._active_goal = goal
        goal.status = GoalStatus.ACTIVE

    def clear_goal(self) -> None:
        if self._active_goal is not None:
            if self._active_goal.status == GoalStatus.ACTIVE:
                self._active_goal.status = (
                    GoalStatus.CANCELLED
                )

        self._active_goal = None
        self._planner = None

    @property
    def active_goal(self) -> Goal | None:
        return self._active_goal

    # --------------------------------------------------------
    # PLANNING
    # --------------------------------------------------------

    def plan(
        self,
        current_state: Tensor | np.ndarray,
        *,
        goal: Goal | None = None,
    ) -> GoalPlanningResult:
        if goal is not None:
            if self._active_goal is not goal:
                self.set_goal(
                    goal
                )

        if (
            self._active_goal is None
            or self._planner is None
        ):
            raise GoalPlanningError(
                "No active goal configured"
            )

        state = torch.as_tensor(
            current_state,
            dtype=torch.float32,
            device=self.device,
        )

        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]

        if state.shape != (
            self.config.state_dim,
        ):
            raise GoalValidationError(
                "Invalid current state dimensions"
            )

        if not torch.isfinite(state).all():
            raise GoalValidationError(
                "Current state contains non-finite values"
            )

        goal = self._active_goal

        if bool(
            self.metric.reached(
                state.unsqueeze(0),
                goal,
            )[0].item()
        ):
            goal.status = GoalStatus.REACHED

            zero_actions = torch.zeros(
                self.config.planning_horizon,
                self.config.action_dim,
                device=self.device,
            )

            repeated_states = state.unsqueeze(
                0
            ).expand(
                self.config.planning_horizon,
                -1,
            ).clone()

            return GoalPlanningResult(
                goal_id=goal.goal_id,
                goal_status=GoalStatus.REACHED,
                first_action=zero_actions[0],
                action_sequence=zero_actions,
                predicted_states=repeated_states,
                initial_distance=0.0,
                final_distance=0.0,
                predicted_reachable=True,
                reachability_score=1.0,
                planning_score=0.0,
                uncertainty_cost=0.0,
                optimization_history=[],
            )

        cem_result = self._planner.plan(
            state
        )

        reachability = self.reachability.evaluate(
            state,
            cem_result,
            goal,
        )

        return GoalPlanningResult(
            goal_id=goal.goal_id,
            goal_status=GoalStatus.ACTIVE,
            first_action=(
                cem_result.first_action
            ),
            action_sequence=(
                cem_result.best_action_sequence
            ),
            predicted_states=(
                cem_result.predicted_states
            ),
            initial_distance=(
                reachability.initial_distance
            ),
            final_distance=(
                reachability.predicted_final_distance
            ),
            predicted_reachable=(
                reachability.predicted_reachable
            ),
            reachability_score=(
                reachability.reachability_score
            ),
            planning_score=(
                cem_result.best_score
            ),
            uncertainty_cost=(
                cem_result.uncertainty_cost
            ),
            optimization_history=(
                cem_result.optimization_history
            ),
        )

    def select_action(
        self,
        current_state: Tensor | np.ndarray,
    ) -> Tensor:
        result = self.plan(
            current_state
        )

        return result.first_action

    # --------------------------------------------------------
    # GOAL PROGRESS
    # --------------------------------------------------------

    def get_progress(
        self,
        current_state: Tensor | np.ndarray,
    ) -> dict[str, Any]:
        if self._active_goal is None:
            raise GoalPlanningError(
                "No active goal"
            )

        state = torch.as_tensor(
            current_state,
            dtype=torch.float32,
            device=self.device,
        )

        if state.ndim == 1:
            state = state.unsqueeze(0)

        distance = float(
            self.metric.distance(
                state,
                self._active_goal,
            )[0].item()
        )

        reached = (
            distance
            <= self._active_goal.tolerance
        )

        if reached:
            self._active_goal.status = (
                GoalStatus.REACHED
            )

        return {
            "goal_id": (
                self._active_goal.goal_id
            ),
            "distance": distance,
            "reached": reached,
            "status": (
                self._active_goal.status.value
            ),
        }


# ============================================================
# MULTI-GOAL MANAGER
# ============================================================

class MultiGoalManager:
    """
    Manages multiple candidate goals.

    Goals are ranked by:
        1. Higher priority
        2. Shorter current distance

    This is a simple priority heuristic,
    not a full hierarchical task planner.
    """

    def __init__(
        self,
        metric: GoalDistanceMetric,
    ) -> None:
        self.metric = metric
        self.goals: dict[str, Goal] = {}

    def add_goal(
        self,
        goal: Goal,
    ) -> None:
        if goal.goal_id in self.goals:
            raise GoalValidationError(
                "Duplicate goal_id"
            )

        self.goals[
            goal.goal_id
        ] = goal

    def remove_goal(
        self,
        goal_id: str,
    ) -> bool:
        return (
            self.goals.pop(
                goal_id,
                None,
            )
            is not None
        )

    def select_goal(
        self,
        current_state: Tensor,
    ) -> Goal | None:
        candidates = [
            goal
            for goal in self.goals.values()
            if goal.status in (
                GoalStatus.PENDING,
                GoalStatus.ACTIVE,
            )
        ]

        if not candidates:
            return None

        ranked = []

        for goal in candidates:
            distance = float(
                self.metric.distance(
                    current_state.unsqueeze(0),
                    goal,
                )[0].item()
            )

            ranked.append(
                (
                    -goal.priority,
                    distance,
                    goal.goal_id,
                    goal,
                )
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        return ranked[0][3]


# ============================================================
# GOAL-CONDITIONED MPC CONTROLLER
# ============================================================

class GoalConditionedMPC:
    """
    Receding-horizon goal controller.

    At each control step:
        1. Observe current state
        2. Check goal completion
        3. Optimize future action sequence
        4. Return first action
        5. Repeat after new observation

    Does not directly execute hardware commands.
    """

    def __init__(
        self,
        planner: GoalConditionedPlanner,
    ) -> None:
        self.planner = planner
        self.control_steps = 0

        self.last_result: (
            GoalPlanningResult | None
        ) = None

    def select_action(
        self,
        current_state: Tensor | np.ndarray,
    ) -> Tensor:
        if (
            self.control_steps
            >= self.planner.config.max_control_steps
        ):
            raise GoalPlanningError(
                "Maximum control steps exceeded"
            )

        self.last_result = (
            self.planner.plan(
                current_state
            )
        )

        self.control_steps += 1

        return (
            self.last_result.first_action
        )

    def reset(self) -> None:
        self.control_steps = 0
        self.last_result = None

        if self.planner._planner is not None:
            self.planner._planner.reset()


# ============================================================
# NUMPY ADAPTER
# ============================================================

class GoalConditionedAdapter:
    """
    NumPy interface for FastAPI,
    simulation, and agent integration.
    """

    def __init__(
        self,
        planner: GoalConditionedPlanner,
    ) -> None:
        self.planner = planner

    def set_goal(
        self,
        goal_id: str,
        target_state: np.ndarray,
        *,
        tolerance: float = 0.25,
        priority: int = 1,
        state_mask: np.ndarray | None = None,
    ) -> None:
        goal = Goal(
            goal_id=goal_id,
            target_state=target_state,
            tolerance=tolerance,
            priority=priority,
            state_mask=state_mask,
        )

        self.planner.set_goal(
            goal
        )

    def plan(
        self,
        current_state: np.ndarray,
    ) -> dict[str, Any]:
        return (
            self.planner.plan(
                current_state
            ).to_numpy()
        )

    def select_action(
        self,
        current_state: np.ndarray,
    ) -> np.ndarray:
        return (
            self.planner.select_action(
                current_state
            )
            .detach()
            .cpu()
            .numpy()
        )


# ============================================================
# SYNTHETIC NAVIGATION DYNAMICS
# ============================================================

class SimpleGoalNavigationDynamics:
    """
    Example 2D point-mass dynamics.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    Not a trained world model.
    """

    def __init__(
        self,
        timestep: float = 0.1,
        max_speed: float = 3.0,
    ) -> None:
        if timestep <= 0 or max_speed <= 0:
            raise ValueError(
                "Invalid navigation parameters"
            )

        self.timestep = timestep
        self.max_speed = max_speed

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
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

        speeds = (
            torch.linalg.vector_norm(
                next_velocities,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        speed_scale = (
            self.max_speed
            / speeds
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

        return torch.cat(
            [
                next_positions,
                next_velocities,
            ],
            dim=-1,
        )


# ============================================================
# FACTORY
# ============================================================

def create_goal_conditioned_planner(
    dynamics: GoalDynamicsProtocol,
    config: GoalConditionedConfig | None = None,
    *,
    uncertainty_fn: (
        Callable[
            [Tensor, Tensor, Tensor],
            Tensor,
        ]
        | None
    ) = None,
) -> GoalConditionedPlanner:
    return GoalConditionedPlanner(
        dynamics=dynamics,
        config=config,
        uncertainty_fn=uncertainty_fn,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - GOAL-CONDITIONED PLANNING"
    )
    print("=" * 65)

    config = GoalConditionedConfig(
        state_dim=4,
        action_dim=2,
        planning_horizon=20,
        population_size=128,
        elite_size=16,
        cem_iterations=5,
        action_min=-1.0,
        action_max=1.0,
        goal_tolerance=0.25,
        max_control_steps=20,
    )

    dynamics = SimpleGoalNavigationDynamics(
        timestep=0.1
    )

    planner = GoalConditionedPlanner(
        dynamics,
        config,
    )

    # Goal constrains only position.
    goal = Goal(
        goal_id="navigation_goal_001",
        target_state=np.array(
            [
                2.0,
                2.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        tolerance=0.25,
        priority=10,
        state_mask=np.array(
            [
                1.0,
                1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        ),
        description="Navigate to target location",
    )

    planner.set_goal(
        goal
    )

    initial_state = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # --------------------------------------------------------
    # GOAL-CONDITIONED PLANNING
    # --------------------------------------------------------

    result = planner.plan(
        initial_state
    )

    print(
        "Goal ID:",
        result.goal_id,
    )

    print(
        "Initial distance:",
        result.initial_distance,
    )

    print(
        "Predicted final distance:",
        result.final_distance,
    )

    print(
        "Predicted reachable:",
        result.predicted_reachable,
    )

    print(
        "First action:",
        result.first_action.cpu().numpy(),
    )

    print(
        "Action sequence shape:",
        tuple(
            result.action_sequence.shape
        ),
    )

    # --------------------------------------------------------
    # MULTI-GOAL MANAGEMENT
    # --------------------------------------------------------

    manager = MultiGoalManager(
        planner.metric
    )

    manager.add_goal(
        goal
    )

    manager.add_goal(
        Goal(
            goal_id="secondary_goal",
            target_state=np.array(
                [
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            priority=5,
            state_mask=np.array(
                [
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        )
    )

    selected_goal = manager.select_goal(
        initial_state
    )

    print(
        "Selected goal:",
        selected_goal.goal_id
        if selected_goal
        else None,
    )

    # --------------------------------------------------------
    # MPC SIMULATION
    # --------------------------------------------------------

    controller = GoalConditionedMPC(
        planner
    )

    current_state = (
        initial_state.clone()
    )

    for step in range(5):
        action = controller.select_action(
            current_state
        )

        next_state = (
            dynamics.predict_next_state(
                current_state.unsqueeze(0),
                action.unsqueeze(0),
            )
        )[0]

        current_state = next_state

        progress = planner.get_progress(
            current_state
        )

        print(
            f"Step {step + 1}:",
            "position=",
            current_state[:2].cpu().numpy(),
            "distance=",
            progress["distance"],
        )

        if progress["reached"]:
            print(
                "Goal reached"
            )
            break

    print(
        "Goal-conditioned planning demonstration completed"
    )


if __name__ == "__main__":
    main()
