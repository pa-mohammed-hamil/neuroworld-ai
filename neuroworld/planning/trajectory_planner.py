
"""
NeuroWorld AI
File: neuroworld/planning/trajectory_planner.py

Advanced Trajectory Planning Engine.

Features:
    - Goal-conditioned trajectory planning
    - Neural world-model rollouts
    - Cross-Entropy Method (CEM)
    - Position and velocity tracking
    - Obstacle clearance constraints
    - Action and acceleration limits
    - Predictive uncertainty penalties
    - Terminal goal optimization
    - Smooth trajectory optimization
    - Warm-start replanning
    - Multi-candidate trajectory evaluation
    - NumPy/PyTorch compatibility
    - Structured planning results

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.planning.trajectory_planner

This planner is intended for simulation and
research. It does not actuate robots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class TrajectoryPlanningError(Exception):
    """Base trajectory planning exception."""


class TrajectoryValidationError(
    TrajectoryPlanningError
):
    """Invalid planning input or model output."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrajectoryPlannerConfig:
    state_dim: int = 4
    action_dim: int = 2
    position_dim: int = 2

    horizon: int = 20
    timestep: float = 0.1

    population_size: int = 256
    elite_size: int = 32
    iterations: int = 8

    action_min: float = -1.0
    action_max: float = 1.0
    max_action_change: float = 0.5

    initial_std: float = 0.5
    minimum_std: float = 0.05
    maximum_std: float = 1.0
    smoothing: float = 0.2

    goal_weight: float = 1.0
    terminal_goal_weight: float = 8.0
    progress_weight: float = 1.0

    action_weight: float = 0.01
    smoothness_weight: float = 0.05
    uncertainty_weight: float = 0.2

    clearance_weight: float = 2.0
    minimum_clearance: float = 0.3

    goal_tolerance: float = 0.2
    maximum_uncertainty: float = 1.0

    warm_start: bool = True

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            "state_dim",
            "action_dim",
            "position_dim",
            "horizon",
            "population_size",
            "elite_size",
            "iterations",
        )

        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.position_dim > self.state_dim:
            raise ValueError(
                "position_dim cannot exceed state_dim"
            )

        if self.elite_size > self.population_size:
            raise ValueError(
                "elite_size cannot exceed population_size"
            )

        if self.timestep <= 0:
            raise ValueError(
                "timestep must be positive"
            )

        if self.action_min >= self.action_max:
            raise ValueError(
                "Invalid action bounds"
            )

        if self.max_action_change <= 0:
            raise ValueError(
                "max_action_change must be positive"
            )

        if not (
            0 < self.minimum_std
            <= self.initial_std
            <= self.maximum_std
        ):
            raise ValueError(
                "Invalid CEM standard deviations"
            )

        if not 0 <= self.smoothing < 1:
            raise ValueError(
                "smoothing must be in [0,1)"
            )

        for name in (
            "goal_weight",
            "terminal_goal_weight",
            "progress_weight",
            "action_weight",
            "smoothness_weight",
            "uncertainty_weight",
            "clearance_weight",
            "minimum_clearance",
            "goal_tolerance",
            "maximum_uncertainty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

class TrajectoryDynamics(Protocol):
    """
    Batched state transition model.

    Input:
        states: [N,state_dim]
        actions: [N,action_dim]

    Output:
        next_states: [N,state_dim]
    """

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        ...


# ============================================================
# NEURAL DYNAMICS ADAPTER
# ============================================================

class NeuralTrajectoryDynamicsAdapter:
    """
    Adapter for NextStatePredictionModel.

    Uses the ensemble mean prediction.
    """

    def __init__(
        self,
        model: Any,
    ) -> None:
        self.model = model

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        output = self.model.predict(
            states,
            actions,
            deterministic=True,
        )

        if isinstance(output, Tensor):
            return output

        if hasattr(output, "mean_states"):
            return output.mean_states

        if hasattr(output, "next_states"):
            return output.next_states

        raise TrajectoryPlanningError(
            "Unsupported dynamics prediction output"
        )


# ============================================================
# OBSTACLE
# ============================================================

@dataclass(frozen=True)
class CircularObstacle:
    center: tuple[float, ...]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(
                "Obstacle radius must be positive"
            )


# ============================================================
# TRAJECTORY GOAL
# ============================================================

@dataclass
class TrajectoryGoal:
    position: Tensor | np.ndarray

    tolerance: float = 0.2

    desired_velocity: (
        Tensor | np.ndarray | None
    ) = None

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise ValueError(
                "Goal tolerance must be positive"
            )


# ============================================================
# INPUT HELPERS
# ============================================================

def _tensor(
    value: Any,
    *,
    device: torch.device,
    name: str,
) -> Tensor:
    result = torch.as_tensor(
        value,
        dtype=torch.float32,
        device=device,
    )

    if not torch.isfinite(result).all():
        raise TrajectoryValidationError(
            f"{name} contains non-finite values"
        )

    return result


# ============================================================
# OBSTACLE GEOMETRY
# ============================================================

class ObstacleGeometry:
    """
    Computes point-to-circle clearance.

    Clearance:
        distance(position, center)
        - obstacle_radius
        - robot_radius

    Negative clearance indicates overlap.

    For position_dim=3, the same formula
    treats obstacles as spheres.
    """

    def __init__(
        self,
        position_dim: int,
        robot_radius: float = 0.0,
    ) -> None:
        if robot_radius < 0:
            raise ValueError(
                "robot_radius must be nonnegative"
            )

        self.position_dim = position_dim
        self.robot_radius = robot_radius

    def minimum_clearance(
        self,
        positions: Tensor,
        obstacles: list[CircularObstacle],
    ) -> Tensor:
        """
        positions: [...,position_dim]

        Returns:
            clearance: [...]
        """
        if not obstacles:
            return torch.full(
                positions.shape[:-1],
                float("inf"),
                dtype=positions.dtype,
                device=positions.device,
            )

        clearances = []

        for obstacle in obstacles:
            center = torch.as_tensor(
                obstacle.center,
                dtype=positions.dtype,
                device=positions.device,
            )

            if center.shape != (
                self.position_dim,
            ):
                raise TrajectoryValidationError(
                    "Obstacle center dimension mismatch"
                )

            distance = torch.linalg.vector_norm(
                positions - center,
                dim=-1,
            )

            clearance = (
                distance
                - obstacle.radius
                - self.robot_radius
            )

            clearances.append(
                clearance
            )

        return torch.stack(
            clearances,
            dim=-1,
        ).min(dim=-1).values


# ============================================================
# ACTION PROJECTOR
# ============================================================

class TrajectoryActionProjector:
    """
    Applies action bounds and rate limits
    sequentially across the horizon.
    """

    def __init__(
        self,
        config: TrajectoryPlannerConfig,
    ) -> None:
        self.config = config

    def project(
        self,
        actions: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        if actions.ndim != 3:
            raise TrajectoryValidationError(
                "Actions must be [N,H,A]"
            )

        if actions.shape[1:] != (
            self.config.horizon,
            self.config.action_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid action sequence dimensions"
            )

        if previous_action.shape != (
            self.config.action_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid previous action"
            )

        last = previous_action.unsqueeze(
            0
        ).expand(
            actions.shape[0],
            -1,
        )

        projected = []

        for step in range(
            self.config.horizon
        ):
            current = actions[
                :, step
            ].clamp(
                self.config.action_min,
                self.config.action_max,
            )

            current = torch.maximum(
                current,
                last
                - self.config.max_action_change,
            )

            current = torch.minimum(
                current,
                last
                + self.config.max_action_change,
            )

            current = current.clamp(
                self.config.action_min,
                self.config.action_max,
            )

            projected.append(
                current
            )

            last = current

        return torch.stack(
            projected,
            dim=1,
        )


# ============================================================
# ROLLOUT RESULT
# ============================================================

@dataclass
class TrajectoryRollout:
    states: Tensor
    positions: Tensor
    scores: Tensor

    goal_costs: Tensor
    terminal_costs: Tensor
    action_costs: Tensor
    smoothness_costs: Tensor
    uncertainty_costs: Tensor
    clearance_costs: Tensor

    minimum_clearances: Tensor
    maximum_uncertainties: Tensor

    feasible: Tensor


# ============================================================
# TRAJECTORY ROLLOUT EVALUATOR
# ============================================================

class TrajectoryRolloutEvaluator:
    """
    Simulates and scores candidate trajectories.

    Candidate scoring:
        goal distance
        goal progress
        terminal distance
        action energy
        action smoothness
        uncertainty
        obstacle clearance

    Unsafe predicted trajectories are
    marked infeasible.

    Collision checks use sampled predicted
    states, not continuous swept geometry.
    """

    def __init__(
        self,
        dynamics: TrajectoryDynamics,
        config: TrajectoryPlannerConfig,
        *,
        obstacles: (
            list[CircularObstacle] | None
        ) = None,
        robot_radius: float = 0.0,
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.dynamics = dynamics
        self.config = config

        self.obstacles = (
            obstacles or []
        )

        self.uncertainty_fn = uncertainty_fn

        self.device = torch.device(
            config.device
        )

        self.geometry = ObstacleGeometry(
            config.position_dim,
            robot_radius,
        )

    @torch.no_grad()
    def evaluate(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal: TrajectoryGoal,
    ) -> TrajectoryRollout:
        initial_state = _tensor(
            initial_state,
            device=self.device,
            name="initial_state",
        )

        actions = _tensor(
            actions,
            device=self.device,
            name="actions",
        )

        if initial_state.shape != (
            self.config.state_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid initial state dimensions"
            )

        if (
            actions.ndim != 3
            or actions.shape[1:]
            != (
                self.config.horizon,
                self.config.action_dim,
            )
        ):
            raise TrajectoryValidationError(
                "Actions must be [N,H,A]"
            )

        population = actions.shape[0]

        if population == 0:
            raise TrajectoryValidationError(
                "Empty candidate population"
            )

        target = _tensor(
            goal.position,
            device=self.device,
            name="goal.position",
        )

        if target.shape != (
            self.config.position_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid goal position dimensions"
            )

        states = initial_state.unsqueeze(
            0
        ).expand(
            population,
            -1,
        ).clone()

        state_history = [
            states.clone()
        ]

        position_history = [
            states[:, :self.config.position_dim]
            .clone()
        ]

        goal_costs = torch.zeros(
            population,
            device=self.device,
        )

        action_costs = torch.zeros_like(
            goal_costs
        )

        smoothness_costs = torch.zeros_like(
            goal_costs
        )

        uncertainty_costs = torch.zeros_like(
            goal_costs
        )

        clearance_costs = torch.zeros_like(
            goal_costs
        )

        feasible = torch.ones(
            population,
            dtype=torch.bool,
            device=self.device,
        )

        minimum_clearances = (
            self.geometry.minimum_clearance(
                position_history[0],
                self.obstacles,
            )
        )

        feasible &= (
            minimum_clearances
            >= self.config.minimum_clearance
        )

        maximum_uncertainties = (
            torch.zeros_like(
                goal_costs
            )
        )

        previous_distance = (
            torch.linalg.vector_norm(
                states[
                    :, :self.config.position_dim
                ]
                - target,
                dim=-1,
            )
        )

        progress_reward = torch.zeros_like(
            goal_costs
        )

        previous_action = None

        for step in range(
            self.config.horizon
        ):
            action = actions[
                :, step
            ]

            next_states = (
                self.dynamics.predict_next_state(
                    states,
                    action,
                )
            )

            next_states = _tensor(
                next_states,
                device=self.device,
                name="predicted_states",
            )

            if next_states.shape != (
                population,
                self.config.state_dim,
            ):
                raise TrajectoryValidationError(
                    "Dynamics returned invalid shape"
                )

            position = next_states[
                :, :self.config.position_dim
            ]

            distance = torch.linalg.vector_norm(
                position - target,
                dim=-1,
            )

            goal_costs += (
                self.config.goal_weight
                * distance
            )

            progress_reward += (
                self.config.progress_weight
                * (
                    previous_distance
                    - distance
                )
            )

            action_costs += (
                self.config.action_weight
                * action.square()
                .sum(dim=-1)
            )

            if previous_action is not None:
                smoothness_costs += (
                    self.config.smoothness_weight
                    * (
                        action
                        - previous_action
                    )
                    .square()
                    .sum(dim=-1)
                )

            if self.uncertainty_fn is None:
                uncertainty = torch.zeros(
                    population,
                    device=self.device,
                )
            else:
                uncertainty = _tensor(
                    self.uncertainty_fn(
                        states,
                        action,
                        next_states,
                    ),
                    device=self.device,
                    name="uncertainty",
                )

                if uncertainty.ndim == 2:
                    if uncertainty.shape[0] != population:
                        raise TrajectoryValidationError(
                            "Invalid uncertainty batch"
                        )

                    uncertainty = uncertainty.mean(
                        dim=-1
                    )

                if uncertainty.shape != (
                    population,
                ):
                    raise TrajectoryValidationError(
                        "Uncertainty must be [N]"
                    )

                if (uncertainty < 0).any():
                    raise TrajectoryValidationError(
                        "Uncertainty must be nonnegative"
                    )

            uncertainty_costs += (
                self.config.uncertainty_weight
                * uncertainty
            )

            maximum_uncertainties = torch.maximum(
                maximum_uncertainties,
                uncertainty,
            )

            feasible &= (
                uncertainty
                <= self.config.maximum_uncertainty
            )

            clearance = (
                self.geometry.minimum_clearance(
                    position,
                    self.obstacles,
                )
            )

            minimum_clearances = torch.minimum(
                minimum_clearances,
                clearance,
            )

            feasible &= (
                clearance
                >= self.config.minimum_clearance
            )

            if self.obstacles:
                clearance_deficit = (
                    self.config.minimum_clearance
                    - clearance
                ).clamp_min(0.0)

                clearance_costs += (
                    self.config.clearance_weight
                    * clearance_deficit.square()
                )

            state_history.append(
                next_states
            )

            position_history.append(
                position
            )

            previous_distance = distance
            previous_action = action
            states = next_states

        terminal_distance = (
            torch.linalg.vector_norm(
                states[
                    :, :self.config.position_dim
                ]
                - target,
                dim=-1,
            )
        )

        terminal_costs = (
            self.config.terminal_goal_weight
            * terminal_distance
        )

        # Optional terminal velocity penalty.
        if goal.desired_velocity is not None:
            velocity_target = _tensor(
                goal.desired_velocity,
                device=self.device,
                name="goal.desired_velocity",
            )

            start_index = (
                self.config.position_dim
            )

            end_index = (
                2 * self.config.position_dim
            )

            if (
                end_index > self.config.state_dim
                or velocity_target.shape
                != (self.config.position_dim,)
            ):
                raise TrajectoryValidationError(
                    "Velocity goal requires velocity "
                    "components after position components"
                )

            velocity_error = (
                states[
                    :, start_index:end_index
                ]
                - velocity_target
            )

            terminal_costs += (
                self.config.terminal_goal_weight
                * velocity_error.square()
                .sum(dim=-1)
            )

        scores = (
            progress_reward
            - goal_costs
            - terminal_costs
            - action_costs
            - smoothness_costs
            - uncertainty_costs
            - clearance_costs
        )

        scores = torch.where(
            feasible,
            scores,
            torch.full_like(
                scores,
                float("-inf"),
            ),
        )

        return TrajectoryRollout(
            states=torch.stack(
                state_history,
                dim=1,
            ),
            positions=torch.stack(
                position_history,
                dim=1,
            ),
            scores=scores,
            goal_costs=goal_costs,
            terminal_costs=terminal_costs,
            action_costs=action_costs,
            smoothness_costs=smoothness_costs,
            uncertainty_costs=uncertainty_costs,
            clearance_costs=clearance_costs,
            minimum_clearances=(
                minimum_clearances
            ),
            maximum_uncertainties=(
                maximum_uncertainties
            ),
            feasible=feasible,
        )


# ============================================================
# PLANNING RESULT
# ============================================================

@dataclass
class TrajectoryPlanningResult:
    success: bool

    first_action: Tensor | None
    action_sequence: Tensor | None

    states: Tensor | None
    positions: Tensor | None

    best_score: float | None

    final_goal_distance: float | None

    predicted_goal_reached: bool

    minimum_clearance: float | None
    maximum_uncertainty: float | None

    feasible_candidates: int

    optimization_history: list[
        dict[str, float]
    ] = field(default_factory=list)

    message: str = ""

    def to_numpy(self) -> dict[str, Any]:
        def convert(
            value: Tensor | None,
        ) -> np.ndarray | None:
            if value is None:
                return None

            return (
                value.detach()
                .cpu()
                .numpy()
            )

        return {
            "success": self.success,
            "first_action": convert(
                self.first_action
            ),
            "action_sequence": convert(
                self.action_sequence
            ),
            "states": convert(
                self.states
            ),
            "positions": convert(
                self.positions
            ),
            "best_score": self.best_score,
            "final_goal_distance": (
                self.final_goal_distance
            ),
            "predicted_goal_reached": (
                self.predicted_goal_reached
            ),
            "minimum_clearance": (
                self.minimum_clearance
            ),
            "maximum_uncertainty": (
                self.maximum_uncertainty
            ),
            "feasible_candidates": (
                self.feasible_candidates
            ),
            "optimization_history": (
                self.optimization_history
            ),
            "message": self.message,
        }


# ============================================================
# TRAJECTORY PLANNER
# ============================================================

class TrajectoryPlanner:
    """
    CEM trajectory planner.

    Optimization loop:
        1. Sample action sequences
        2. Enforce action constraints
        3. Roll out world model
        4. Reject infeasible trajectories
        5. Select elite candidates
        6. Update sampling distribution
        7. Return best trajectory
    """

    def __init__(
        self,
        dynamics: TrajectoryDynamics,
        config: (
            TrajectoryPlannerConfig | None
        ) = None,
        *,
        obstacles: (
            list[CircularObstacle] | None
        ) = None,
        robot_radius: float = 0.0,
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.config = (
            config or TrajectoryPlannerConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.evaluator = (
            TrajectoryRolloutEvaluator(
                dynamics,
                self.config,
                obstacles=obstacles,
                robot_radius=robot_radius,
                uncertainty_fn=uncertainty_fn,
            )
        )

        self.projector = (
            TrajectoryActionProjector(
                self.config
            )
        )

        self.generator = torch.Generator(
            device=self.device
        )

        self.generator.manual_seed(
            self.config.seed
        )

        self.previous_action = torch.zeros(
            self.config.action_dim,
            device=self.device,
        )

        self._warm_mean: Tensor | None = None

    def reset(self) -> None:
        self.previous_action.zero_()
        self._warm_mean = None

    def acknowledge_action(
        self,
        action: Tensor | np.ndarray,
    ) -> None:
        action = _tensor(
            action,
            device=self.device,
            name="applied_action",
        )

        if action.shape != (
            self.config.action_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid applied action shape"
            )

        if (
            (action < self.config.action_min).any()
            or (action > self.config.action_max).any()
        ):
            raise TrajectoryValidationError(
                "Applied action violates bounds"
            )

        self.previous_action = (
            action.clone()
        )

    def _initial_distribution(
        self,
    ) -> tuple[Tensor, Tensor]:
        shape = (
            self.config.horizon,
            self.config.action_dim,
        )

        if (
            self.config.warm_start
            and self._warm_mean is not None
        ):
            mean = self._warm_mean.clone()
        else:
            mean = torch.zeros(
                shape,
                device=self.device,
            )

        std = torch.full(
            shape,
            self.config.initial_std,
            device=self.device,
        )

        return mean, std

    @torch.no_grad()
    def plan(
        self,
        initial_state: Tensor | np.ndarray,
        goal: TrajectoryGoal,
    ) -> TrajectoryPlanningResult:
        state = _tensor(
            initial_state,
            device=self.device,
            name="initial_state",
        )

        if state.shape != (
            self.config.state_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid initial state shape"
            )

        target = _tensor(
            goal.position,
            device=self.device,
            name="goal.position",
        )

        if target.shape != (
            self.config.position_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid goal shape"
            )

        mean, std = (
            self._initial_distribution()
        )

        best_score = float("-inf")
        best_actions = None
        best_states = None
        best_positions = None
        best_clearance = None
        best_uncertainty = None

        history = []
        maximum_feasible = 0

        for iteration in range(
            self.config.iterations
        ):
            noise = torch.randn(
                self.config.population_size,
                self.config.horizon,
                self.config.action_dim,
                generator=self.generator,
                device=self.device,
            )

            candidates = (
                mean.unsqueeze(0)
                + std.unsqueeze(0)
                * noise
            )

            candidates = self.projector.project(
                candidates,
                self.previous_action,
            )

            # Include current distribution mean.
            candidates[0] = self.projector.project(
                mean.unsqueeze(0),
                self.previous_action,
            )[0]

            rollout = self.evaluator.evaluate(
                state,
                candidates,
                goal,
            )

            feasible_indices = torch.nonzero(
                rollout.feasible,
                as_tuple=False,
            ).flatten()

            feasible_count = int(
                feasible_indices.numel()
            )

            maximum_feasible = max(
                maximum_feasible,
                feasible_count,
            )

            if feasible_count == 0:
                history.append(
                    {
                        "iteration": float(
                            iteration + 1
                        ),
                        "feasible_candidates": 0.0,
                    }
                )

                continue

            feasible_scores = rollout.scores[
                feasible_indices
            ]

            elite_count = min(
                self.config.elite_size,
                feasible_count,
            )

            elite_values, elite_positions = (
                torch.topk(
                    feasible_scores,
                    k=elite_count,
                )
            )

            elite_indices = feasible_indices[
                elite_positions
            ]

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
                + (1 - smoothing)
                * elite_mean
            )

            std = (
                smoothing * std
                + (1 - smoothing)
                * elite_std
            ).clamp(
                self.config.minimum_std,
                self.config.maximum_std,
            )

            local_best = int(
                feasible_indices[
                    feasible_scores.argmax()
                ].item()
            )

            local_score = float(
                rollout.scores[
                    local_best
                ].item()
            )

            if local_score > best_score:
                best_score = local_score

                best_actions = candidates[
                    local_best
                ].clone()

                best_states = rollout.states[
                    local_best
                ].clone()

                best_positions = rollout.positions[
                    local_best
                ].clone()

                best_clearance = float(
                    rollout.minimum_clearances[
                        local_best
                    ].item()
                )

                best_uncertainty = float(
                    rollout.maximum_uncertainties[
                        local_best
                    ].item()
                )

            history.append(
                {
                    "iteration": float(
                        iteration + 1
                    ),
                    "best_score": local_score,
                    "elite_mean_score": float(
                        elite_values.mean().item()
                    ),
                    "feasible_candidates": float(
                        feasible_count
                    ),
                    "mean_std": float(
                        std.mean().item()
                    ),
                }
            )

        if best_actions is None:
            return TrajectoryPlanningResult(
                success=False,
                first_action=None,
                action_sequence=None,
                states=None,
                positions=None,
                best_score=None,
                final_goal_distance=None,
                predicted_goal_reached=False,
                minimum_clearance=None,
                maximum_uncertainty=None,
                feasible_candidates=0,
                optimization_history=history,
                message=(
                    "No feasible trajectory found"
                ),
            )

        final_distance = float(
            torch.linalg.vector_norm(
                best_positions[-1] - target
            ).item()
        )

        if self.config.warm_start:
            self._warm_mean = torch.cat(
                [
                    best_actions[1:],
                    best_actions[-1:],
                ],
                dim=0,
            ).clone()

        return TrajectoryPlanningResult(
            success=True,
            first_action=best_actions[0],
            action_sequence=best_actions,
            states=best_states,
            positions=best_positions,
            best_score=best_score,
            final_goal_distance=final_distance,
            predicted_goal_reached=(
                final_distance <= goal.tolerance
            ),
            minimum_clearance=best_clearance,
            maximum_uncertainty=best_uncertainty,
            feasible_candidates=maximum_feasible,
            optimization_history=history,
            message="Feasible trajectory generated",
        )


# ============================================================
# UNIFIED PLANNER ADAPTER
# ============================================================

class UnifiedTrajectoryPlannerAdapter:
    """
    Integrates with:
        neuroworld.planning.planner.UnifiedPlanner

    Converts a PlanningRequest into a
    trajectory-planning request.
    """

    def __init__(
        self,
        planner: TrajectoryPlanner,
    ) -> None:
        self.planner = planner

    def plan(
        self,
        request: Any,
    ) -> Any:
        from neuroworld.planning.planner import (
            StrategyPlan,
        )

        if request.goal_state is None:
            raise TrajectoryValidationError(
                "Trajectory planning requires a goal"
            )

        goal_state = _tensor(
            request.goal_state,
            device=self.planner.device,
            name="goal_state",
        )

        goal = TrajectoryGoal(
            position=goal_state[
                :self.planner.config.position_dim
            ]
        )

        result = self.planner.plan(
            request.current_state,
            goal,
        )

        if not result.success:
            raise TrajectoryPlanningError(
                result.message
            )

        return StrategyPlan(
            actions=result.action_sequence,
            states=result.states,
            predicted_score=result.best_score,
            metadata={
                "source": "trajectory",
                "minimum_clearance": (
                    result.minimum_clearance
                ),
                "maximum_uncertainty": (
                    result.maximum_uncertainty
                ),
                "predicted_goal_reached": (
                    result.predicted_goal_reached
                ),
            },
        )


# ============================================================
# NUMPY ADAPTER
# ============================================================

class TrajectoryPlannerAdapter:
    """
    NumPy interface for backend APIs.
    """

    def __init__(
        self,
        planner: TrajectoryPlanner,
    ) -> None:
        self.planner = planner

    def plan(
        self,
        state: np.ndarray,
        goal_position: np.ndarray,
    ) -> dict[str, Any]:
        result = self.planner.plan(
            state,
            TrajectoryGoal(
                position=goal_position
            ),
        )

        return result.to_numpy()


# ============================================================
# SYNTHETIC DYNAMICS
# ============================================================

class PointMassDynamics:
    """
    Synthetic point-mass model.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    Motion:
        velocity_next = velocity + a*dt
        position_next = position + velocity_next*dt

    This model is for demonstration only.
    """

    def __init__(
        self,
        timestep: float = 0.1,
    ) -> None:
        if timestep <= 0:
            raise ValueError(
                "timestep must be positive"
            )

        self.timestep = timestep

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        positions = states[:, :2]
        velocities = states[:, 2:4]

        next_velocities = (
            velocities
            + actions * self.timestep
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

def create_trajectory_planner(
    dynamics: TrajectoryDynamics,
    config: (
        TrajectoryPlannerConfig | None
    ) = None,
    *,
    obstacles: (
        list[CircularObstacle] | None
    ) = None,
    robot_radius: float = 0.0,
    uncertainty_fn: (
        Callable[
            [Tensor, Tensor, Tensor],
            Tensor,
        ]
        | None
    ) = None,
) -> TrajectoryPlanner:
    return TrajectoryPlanner(
        dynamics,
        config,
        obstacles=obstacles,
        robot_radius=robot_radius,
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
        "NEUROWORLD AI - TRAJECTORY PLANNER"
    )
    print("=" * 65)

    config = TrajectoryPlannerConfig(
        state_dim=4,
        action_dim=2,
        position_dim=2,
        horizon=25,
        timestep=0.2,
        population_size=192,
        elite_size=24,
        iterations=8,
        goal_tolerance=0.25,
        minimum_clearance=0.15,
    )

    dynamics = PointMassDynamics(
        timestep=config.timestep
    )

    obstacles = [
        CircularObstacle(
            center=(1.0, 0.5),
            radius=0.25,
        ),
        CircularObstacle(
            center=(1.5, 1.5),
            radius=0.2,
        ),
    ]

    planner = TrajectoryPlanner(
        dynamics,
        config,
        obstacles=obstacles,
        robot_radius=0.1,
    )

    initial_state = torch.tensor(
        [0.0, 0.0, 0.0, 0.0]
    )

    goal = TrajectoryGoal(
        position=torch.tensor(
            [2.0, 2.0]
        ),
        tolerance=0.25,
        desired_velocity=torch.zeros(2),
    )

    result = planner.plan(
        initial_state,
        goal,
    )

    print(
        "Success:",
        result.success,
    )

    print(
        "Message:",
        result.message,
    )

    if result.success:
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
                result.action_sequence.shape
            ),
        )

        print(
            "Trajectory shape:",
            tuple(
                result.states.shape
            ),
        )

        print(
            "Final goal distance:",
            result.final_goal_distance,
        )

        print(
            "Predicted goal reached:",
            result.predicted_goal_reached,
        )

        print(
            "Minimum clearance:",
            result.minimum_clearance,
        )

        print(
            "Maximum uncertainty:",
            result.maximum_uncertainty,
        )

        print(
            "Feasible candidates:",
            result.feasible_candidates,
        )

    print(
        "Trajectory planning demonstration completed"
    )


if __name__ == "__main__":
    main()
