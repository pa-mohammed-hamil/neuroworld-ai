
"""
NeuroWorld AI
File: neuroworld/planning/latent_planner.py

Latent Space Planning & Imagination Engine.

Features:
    - Goal-conditioned latent planning
    - Learned world-model dynamics adapters
    - Cross-Entropy Method (CEM)
    - Batch candidate action optimization
    - Latent trajectory rollouts
    - Discounted reward optimization
    - Terminal goal-distance penalties
    - Predictive uncertainty penalties
    - Action magnitude and smoothness penalties
    - Receding-horizon MPC
    - Warm-start optimization
    - Goal reachability diagnostics
    - NumPy/PyTorch compatibility
    - Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.planning.latent_planner

Inputs:
    initial_latent: [latent_dim]
    goal_latent:    [latent_dim]
    actions:        [N, H, action_dim]

Outputs:
    optimized actions: [H, action_dim]
    predicted latents: [H+1, latent_dim]

IMPORTANT:
    The world model must be trained before
    its predictions are used for planning.

    The planner generates candidate actions;
    it does not execute physical commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class LatentPlanningError(Exception):
    """Base latent-planning exception."""


class LatentValidationError(LatentPlanningError):
    """Invalid state, goal, or action input."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LatentPlannerConfig:
    latent_dim: int = 128
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

    goal_distance_weight: float = 1.0
    terminal_goal_weight: float = 5.0
    goal_success_bonus: float = 10.0
    goal_tolerance: float = 0.25

    reward_weight: float = 1.0
    uncertainty_weight: float = 0.1
    action_cost_weight: float = 0.001
    action_smoothness_weight: float = 0.01

    warm_start: bool = True

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "latent_dim",
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
                "Invalid action bounds"
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

        if not 0 < self.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in (0,1]"
            )

        if self.goal_tolerance <= 0:
            raise ValueError(
                "goal_tolerance must be positive"
            )

        for name in (
            "goal_distance_weight",
            "terminal_goal_weight",
            "goal_success_bonus",
            "reward_weight",
            "uncertainty_weight",
            "action_cost_weight",
            "action_smoothness_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

class LatentDynamicsProtocol(Protocol):
    """
    Batched latent dynamics interface.

    Input:
        latents: [N, latent_dim]
        actions: [N, action_dim]

    Output:
        next_latents: [N, latent_dim]
    """

    def predict_next_latent(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> Tensor:
        ...


class NeuralLatentDynamicsAdapter:
    """
    Adapter for a neural next-state model.

    Supports a model exposing:
        predict(states, actions)

    The prediction may be a Tensor or an
    object containing `.mean_states`.

    The adapter uses the mean prediction.
    """

    def __init__(
        self,
        model: Any,
    ) -> None:
        self.model = model

    def predict_next_latent(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> Tensor:
        output = self.model.predict(
            latents,
            actions,
            deterministic=True,
        )

        if isinstance(output, Tensor):
            return output

        if hasattr(
            output,
            "mean_states",
        ):
            return output.mean_states

        if hasattr(
            output,
            "next_states",
        ):
            return output.next_states

        raise LatentPlanningError(
            "Unsupported neural dynamics output"
        )


# ============================================================
# GOAL REPRESENTATION
# ============================================================

@dataclass
class LatentGoal:
    target: Tensor | np.ndarray

    tolerance: float = 0.25

    mask: Tensor | np.ndarray | None = None

    weights: Tensor | np.ndarray | None = None

    goal_id: str = "default_goal"

    def __post_init__(self) -> None:
        if self.tolerance <= 0:
            raise LatentValidationError(
                "Goal tolerance must be positive"
            )

        if not self.goal_id:
            raise LatentValidationError(
                "goal_id cannot be empty"
            )


# ============================================================
# GOAL DISTANCE
# ============================================================

class LatentGoalMetric:
    """
    Weighted Euclidean distance.

    Supports partial latent-state goals
    through masks and weights.

    Latent distance is meaningful only
    when the representation is trained
    to preserve relevant goal structure.
    """

    def __init__(
        self,
        config: LatentPlannerConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(
            config.device
        )

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
            self.config.latent_dim,
        ):
            raise LatentValidationError(
                f"{name} must be "
                f"[{self.config.latent_dim}]"
            )

        if not torch.isfinite(vector).all():
            raise LatentValidationError(
                f"{name} contains non-finite values"
            )

        return vector

    def distance(
        self,
        latents: Tensor,
        goal: LatentGoal,
    ) -> Tensor:
        target = self._vector(
            goal.target,
            "goal.target",
        )

        if latents.ndim == 1:
            latents = latents.unsqueeze(0)

        if (
            latents.ndim != 2
            or latents.shape[-1]
            != self.config.latent_dim
        ):
            raise LatentValidationError(
                "Latents must be [N,latent_dim]"
            )

        if goal.mask is None:
            mask = torch.ones_like(
                target
            )
        else:
            mask = self._vector(
                goal.mask,
                "goal.mask",
            )

            if not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise LatentValidationError(
                    "Goal mask must be binary"
                )

        if goal.weights is None:
            weights = torch.ones_like(
                target
            )
        else:
            weights = self._vector(
                goal.weights,
                "goal.weights",
            )

            if (weights < 0).any():
                raise LatentValidationError(
                    "Goal weights must be nonnegative"
                )

        effective_weights = (
            mask * weights
        )

        if effective_weights.sum() <= 0:
            raise LatentValidationError(
                "At least one positive goal weight required"
            )

        difference = (
            latents - target
        )

        return (
            difference.square()
            .mul(effective_weights)
            .sum(dim=-1)
            .sqrt()
        )

    def reached(
        self,
        latents: Tensor,
        goal: LatentGoal,
    ) -> Tensor:
        return (
            self.distance(
                latents,
                goal,
            )
            <= goal.tolerance
        )


# ============================================================
# ROLLOUT RESULT
# ============================================================

@dataclass
class LatentRollout:
    latent_states: Tensor

    rewards: Tensor
    uncertainties: Tensor
    goal_distances: Tensor

    scores: Tensor

    discounted_rewards: Tensor
    uncertainty_costs: Tensor
    action_costs: Tensor
    smoothness_costs: Tensor
    terminal_costs: Tensor

    def detach(self) -> "LatentRollout":
        return LatentRollout(
            **{
                name: value.detach()
                for name, value in vars(self).items()
            }
        )


# ============================================================
# LATENT ROLLOUT EVALUATOR
# ============================================================

class LatentRolloutEvaluator:
    """
    Simulates action sequences in latent space.

    Scores each candidate using:
        discounted task reward
        goal-distance shaping
        terminal goal distance
        goal completion bonus
        uncertainty penalty
        action cost
        action smoothness cost

    Candidate rollouts are independent.
    """

    def __init__(
        self,
        dynamics: LatentDynamicsProtocol,
        config: LatentPlannerConfig,
        *,
        reward_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
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

        self.reward_fn = reward_fn
        self.uncertainty_fn = uncertainty_fn

        self.device = torch.device(
            config.device
        )

        self.goal_metric = LatentGoalMetric(
            config
        )

    def _validate_vector_batch(
        self,
        tensor: Tensor,
        batch_size: int,
        name: str,
    ) -> Tensor:
        tensor = torch.as_tensor(
            tensor,
            dtype=torch.float32,
            device=self.device,
        )

        if tensor.shape != (
            batch_size,
        ):
            raise LatentValidationError(
                f"{name} must be [{batch_size}]"
            )

        if not torch.isfinite(tensor).all():
            raise LatentValidationError(
                f"{name} contains non-finite values"
            )

        return tensor

    @torch.no_grad()
    def evaluate(
        self,
        initial_latent: Tensor | np.ndarray,
        action_sequences: Tensor,
        goal: LatentGoal,
    ) -> LatentRollout:
        initial_latent = torch.as_tensor(
            initial_latent,
            dtype=torch.float32,
            device=self.device,
        )

        if initial_latent.shape != (
            self.config.latent_dim,
        ):
            raise LatentValidationError(
                "Invalid initial latent shape"
            )

        if not torch.isfinite(
            initial_latent
        ).all():
            raise LatentValidationError(
                "Initial latent contains non-finite values"
            )

        action_sequences = torch.as_tensor(
            action_sequences,
            dtype=torch.float32,
            device=self.device,
        )

        if action_sequences.ndim != 3:
            raise LatentValidationError(
                "Actions must be [N,H,A]"
            )

        population_size, horizon, action_dim = (
            action_sequences.shape
        )

        if (
            population_size == 0
            or horizon != self.config.horizon
            or action_dim != self.config.action_dim
        ):
            raise LatentValidationError(
                "Invalid candidate action dimensions"
            )

        if not torch.isfinite(
            action_sequences
        ).all():
            raise LatentValidationError(
                "Actions contain non-finite values"
            )

        if (
            (action_sequences < self.config.action_min).any()
            or (action_sequences > self.config.action_max).any()
        ):
            raise LatentValidationError(
                "Candidate actions exceed configured bounds"
            )

        latents = initial_latent.unsqueeze(
            0
        ).expand(
            population_size,
            -1,
        ).clone()

        latent_history = [
            latents.clone()
        ]

        reward_history = []
        uncertainty_history = []
        distance_history = []

        total_discounted_reward = torch.zeros(
            population_size,
            device=self.device,
        )

        total_uncertainty_cost = torch.zeros_like(
            total_discounted_reward
        )

        total_action_cost = torch.zeros_like(
            total_discounted_reward
        )

        total_smoothness_cost = torch.zeros_like(
            total_discounted_reward
        )

        discount = 1.0

        previous_actions = None

        for step in range(horizon):
            actions = action_sequences[
                :, step
            ]

            next_latents = (
                self.dynamics.predict_next_latent(
                    latents,
                    actions,
                )
            )

            next_latents = torch.as_tensor(
                next_latents,
                dtype=torch.float32,
                device=self.device,
            )

            if next_latents.shape != (
                population_size,
                self.config.latent_dim,
            ):
                raise LatentValidationError(
                    "Dynamics returned invalid latent shape"
                )

            if not torch.isfinite(
                next_latents
            ).all():
                raise LatentValidationError(
                    "Dynamics returned non-finite latents"
                )

            distances = (
                self.goal_metric.distance(
                    next_latents,
                    goal,
                )
            )

            if self.reward_fn is None:
                rewards = torch.zeros(
                    population_size,
                    device=self.device,
                )
            else:
                rewards = self._validate_vector_batch(
                    self.reward_fn(
                        latents,
                        actions,
                        next_latents,
                    ),
                    population_size,
                    "Rewards",
                )

            if self.uncertainty_fn is None:
                uncertainties = torch.zeros(
                    population_size,
                    device=self.device,
                )
            else:
                uncertainties = (
                    self._validate_vector_batch(
                        self.uncertainty_fn(
                            latents,
                            actions,
                            next_latents,
                        ),
                        population_size,
                        "Uncertainty",
                    )
                )

                if (uncertainties < 0).any():
                    raise LatentValidationError(
                        "Uncertainty must be nonnegative"
                    )

            goal_reward = (
                -self.config.goal_distance_weight
                * distances
            )

            combined_reward = (
                self.config.reward_weight
                * rewards
                + goal_reward
            )

            total_discounted_reward += (
                discount
                * combined_reward
            )

            total_uncertainty_cost += (
                discount
                * self.config.uncertainty_weight
                * uncertainties
            )

            total_action_cost += (
                discount
                * self.config.action_cost_weight
                * actions.square().sum(
                    dim=-1
                )
            )

            if previous_actions is not None:
                action_difference = (
                    actions
                    - previous_actions
                )

                total_smoothness_cost += (
                    discount
                    * self.config.action_smoothness_weight
                    * action_difference.square().sum(
                        dim=-1
                    )
                )

            reward_history.append(
                combined_reward
            )

            uncertainty_history.append(
                uncertainties
            )

            distance_history.append(
                distances
            )

            latent_history.append(
                next_latents
            )

            latents = next_latents

            previous_actions = actions

            discount *= (
                self.config.discount_factor
            )

        final_distance = (
            self.goal_metric.distance(
                latents,
                goal,
            )
        )

        terminal_cost = (
            self.config.terminal_goal_weight
            * final_distance
        )

        success_bonus = (
            (final_distance <= goal.tolerance)
            .float()
            * self.config.goal_success_bonus
        )

        scores = (
            total_discounted_reward
            - total_uncertainty_cost
            - total_action_cost
            - total_smoothness_cost
            - terminal_cost
            + success_bonus
        )

        return LatentRollout(
            latent_states=torch.stack(
                latent_history,
                dim=1,
            ),
            rewards=torch.stack(
                reward_history,
                dim=1,
            ),
            uncertainties=torch.stack(
                uncertainty_history,
                dim=1,
            ),
            goal_distances=torch.stack(
                distance_history,
                dim=1,
            ),
            scores=scores,
            discounted_rewards=(
                total_discounted_reward
            ),
            uncertainty_costs=(
                total_uncertainty_cost
            ),
            action_costs=(
                total_action_cost
            ),
            smoothness_costs=(
                total_smoothness_cost
            ),
            terminal_costs=(
                terminal_cost
            ),
        )


# ============================================================
# PLANNING RESULT
# ============================================================

@dataclass
class LatentPlanningResult:
    first_action: Tensor
    action_sequence: Tensor

    latent_trajectory: Tensor

    best_score: float
    initial_goal_distance: float
    final_goal_distance: float

    predicted_goal_reached: bool

    discounted_reward: float
    uncertainty_cost: float
    action_cost: float
    smoothness_cost: float
    terminal_cost: float

    final_mean: Tensor
    final_std: Tensor

    optimization_history: list[
        dict[str, float]
    ] = field(default_factory=list)

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
            "first_action": convert(
                self.first_action
            ),
            "action_sequence": convert(
                self.action_sequence
            ),
            "latent_trajectory": convert(
                self.latent_trajectory
            ),
            "best_score": self.best_score,
            "initial_goal_distance": (
                self.initial_goal_distance
            ),
            "final_goal_distance": (
                self.final_goal_distance
            ),
            "predicted_goal_reached": (
                self.predicted_goal_reached
            ),
            "discounted_reward": (
                self.discounted_reward
            ),
            "uncertainty_cost": (
                self.uncertainty_cost
            ),
            "action_cost": (
                self.action_cost
            ),
            "smoothness_cost": (
                self.smoothness_cost
            ),
            "terminal_cost": (
                self.terminal_cost
            ),
            "final_mean": convert(
                self.final_mean
            ),
            "final_std": convert(
                self.final_std
            ),
            "optimization_history": (
                self.optimization_history
            ),
        }


# ============================================================
# LATENT CEM PLANNER
# ============================================================

class LatentCEMPlanner:
    """
    Cross-Entropy Method planning in
    learned latent-state space.

    CEM optimization:
        1. Initialize action distribution
        2. Sample candidate sequences
        3. Simulate latent trajectories
        4. Evaluate goal and reward scores
        5. Select elite sequences
        6. Update distribution
        7. Return highest-scoring sequence

    Warm-start is shifted after planning
    for receding-horizon operation.
    """

    def __init__(
        self,
        evaluator: LatentRolloutEvaluator,
        config: LatentPlannerConfig,
    ) -> None:
        self.evaluator = evaluator
        self.config = config

        self.device = torch.device(
            config.device
        )

        self.generator = torch.Generator(
            device=self.device
        )

        self.generator.manual_seed(
            config.seed
        )

        self._warm_mean: Tensor | None = None

    def reset(self) -> None:
        self._warm_mean = None

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

    def _sample_candidates(
        self,
        mean: Tensor,
        std: Tensor,
    ) -> Tensor:
        noise = torch.randn(
            self.config.population_size,
            self.config.horizon,
            self.config.action_dim,
            device=self.device,
            generator=self.generator,
        )

        candidates = (
            mean.unsqueeze(0)
            + noise * std.unsqueeze(0)
        )

        return candidates.clamp(
            self.config.action_min,
            self.config.action_max,
        )

    @torch.no_grad()
    def plan(
        self,
        initial_latent: Tensor | np.ndarray,
        goal: LatentGoal,
    ) -> LatentPlanningResult:
        initial_latent = torch.as_tensor(
            initial_latent,
            dtype=torch.float32,
            device=self.device,
        )

        if initial_latent.shape != (
            self.config.latent_dim,
        ):
            raise LatentValidationError(
                "Invalid initial latent shape"
            )

        initial_distance = float(
            self.evaluator.goal_metric.distance(
                initial_latent.unsqueeze(0),
                goal,
            )[0].item()
        )

        mean, std = (
            self._initial_distribution()
        )

        best_score = float("-inf")
        best_sequence = None
        best_rollout = None
        best_index = None

        optimization_history = []

        for iteration in range(
            self.config.iterations
        ):
            candidates = (
                self._sample_candidates(
                    mean,
                    std,
                )
            )

            candidates[0] = mean

            rollout = (
                self.evaluator.evaluate(
                    initial_latent,
                    candidates,
                    goal,
                )
            )

            scores = rollout.scores

            elite_scores, elite_indices = (
                torch.topk(
                    scores,
                    k=self.config.elite_size,
                )
            )

            elite_actions = (
                candidates[
                    elite_indices
                ]
            )

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

            iteration_best_index = int(
                scores.argmax().item()
            )

            iteration_best_score = float(
                scores[
                    iteration_best_index
                ].item()
            )

            if iteration_best_score > best_score:
                best_score = (
                    iteration_best_score
                )

                best_sequence = (
                    candidates[
                        iteration_best_index
                    ].clone()
                )

                best_rollout = (
                    rollout.detach()
                )

                best_index = (
                    iteration_best_index
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
                    "mean_action_std": float(
                        std.mean().item()
                    ),
                }
            )

        if (
            best_sequence is None
            or best_rollout is None
            or best_index is None
        ):
            raise LatentPlanningError(
                "CEM failed to find an action sequence"
            )

        final_latent = (
            best_rollout.latent_states[
                best_index,
                -1,
            ]
        )

        final_distance = float(
            self.evaluator.goal_metric.distance(
                final_latent.unsqueeze(0),
                goal,
            )[0].item()
        )

        if self.config.warm_start:
            self._warm_mean = torch.cat(
                [
                    best_sequence[1:],
                    best_sequence[-1:],
                ],
                dim=0,
            ).clone()

        return LatentPlanningResult(
            first_action=best_sequence[0],
            action_sequence=best_sequence,
            latent_trajectory=(
                best_rollout.latent_states[
                    best_index
                ]
            ),
            best_score=best_score,
            initial_goal_distance=(
                initial_distance
            ),
            final_goal_distance=(
                final_distance
            ),
            predicted_goal_reached=(
                final_distance
                <= goal.tolerance
            ),
            discounted_reward=float(
                best_rollout.discounted_rewards[
                    best_index
                ].item()
            ),
            uncertainty_cost=float(
                best_rollout.uncertainty_costs[
                    best_index
                ].item()
            ),
            action_cost=float(
                best_rollout.action_costs[
                    best_index
                ].item()
            ),
            smoothness_cost=float(
                best_rollout.smoothness_costs[
                    best_index
                ].item()
            ),
            terminal_cost=float(
                best_rollout.terminal_costs[
                    best_index
                ].item()
            ),
            final_mean=mean.clone(),
            final_std=std.clone(),
            optimization_history=(
                optimization_history
            ),
        )


# ============================================================
# HIGH-LEVEL LATENT PLANNER
# ============================================================

class LatentPlanner:
    """
    High-level interface for planning.

    Connects:
        neural world model
        goal representation
        reward model
        uncertainty model
        CEM optimization
        MPC replanning
    """

    def __init__(
        self,
        dynamics: LatentDynamicsProtocol,
        config: LatentPlannerConfig | None = None,
        *,
        reward_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor],
                Tensor,
            ]
            | None
        ) = None,
    ) -> None:
        self.config = (
            config or LatentPlannerConfig()
        )

        self.evaluator = LatentRolloutEvaluator(
            dynamics,
            self.config,
            reward_fn=reward_fn,
            uncertainty_fn=uncertainty_fn,
        )

        self.optimizer = LatentCEMPlanner(
            self.evaluator,
            self.config,
        )

        self.active_goal: (
            LatentGoal | None
        ) = None

    def set_goal(
        self,
        goal: LatentGoal,
    ) -> None:
        # Validate goal representation.
        reference = torch.zeros(
            1,
            self.config.latent_dim,
            device=self.config.device,
        )

        self.evaluator.goal_metric.distance(
            reference,
            goal,
        )

        self.active_goal = goal

        # Avoid reusing action distributions
        # from a different target.
        self.optimizer.reset()

    def plan(
        self,
        initial_latent: Tensor | np.ndarray,
        *,
        goal: LatentGoal | None = None,
    ) -> LatentPlanningResult:
        if goal is not None:
            if self.active_goal is not goal:
                self.set_goal(goal)

        if self.active_goal is None:
            raise LatentPlanningError(
                "No active goal configured"
            )

        return self.optimizer.plan(
            initial_latent,
            self.active_goal,
        )

    def select_action(
        self,
        initial_latent: Tensor | np.ndarray,
    ) -> Tensor:
        result = self.plan(
            initial_latent
        )

        return result.first_action

    def reset(self) -> None:
        self.optimizer.reset()


# ============================================================
# LATENT MPC CONTROLLER
# ============================================================

class LatentMPCController:
    """
    Receding-horizon controller.

    Each step:
        observe latent
        optimize actions
        select first action
        observe new latent
        replan

    This class does not actuate hardware.
    """

    def __init__(
        self,
        planner: LatentPlanner,
    ) -> None:
        self.planner = planner

        self.last_result: (
            LatentPlanningResult | None
        ) = None

        self.control_steps = 0

    def select_action(
        self,
        current_latent: Tensor | np.ndarray,
    ) -> Tensor:
        self.last_result = (
            self.planner.plan(
                current_latent
            )
        )

        self.control_steps += 1

        return (
            self.last_result.first_action
        )

    def reset(self) -> None:
        self.control_steps = 0
        self.last_result = None
        self.planner.reset()


# ============================================================
# NUMPY ADAPTER
# ============================================================

class LatentPlannerAdapter:
    """
    NumPy interface for FastAPI,
    agents, and simulation services.
    """

    def __init__(
        self,
        planner: LatentPlanner,
    ) -> None:
        self.planner = planner

    def set_goal(
        self,
        target: np.ndarray,
        *,
        tolerance: float = 0.25,
        goal_id: str = "default_goal",
    ) -> None:
        self.planner.set_goal(
            LatentGoal(
                target=target,
                tolerance=tolerance,
                goal_id=goal_id,
            )
        )

    def plan(
        self,
        latent: np.ndarray,
    ) -> dict[str, Any]:
        return (
            self.planner.plan(
                latent
            ).to_numpy()
        )

    def select_action(
        self,
        latent: np.ndarray,
    ) -> np.ndarray:
        return (
            self.planner.select_action(
                latent
            )
            .detach()
            .cpu()
            .numpy()
        )


# ============================================================
# SYNTHETIC LATENT DYNAMICS
# ============================================================

class SimpleLatentDynamics:
    """
    Synthetic dynamics for demonstration.

    First action_dim latent features evolve
    directly with action.

    Remaining latent dimensions stay fixed.

    This is not a trained world model.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        timestep: float = 0.1,
    ) -> None:
        if latent_dim < action_dim:
            raise ValueError(
                "latent_dim must be >= action_dim"
            )

        if timestep <= 0:
            raise ValueError(
                "timestep must be positive"
            )

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.timestep = timestep

    def predict_next_latent(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> Tensor:
        next_latents = latents.clone()

        next_latents[
            :, :self.action_dim
        ] += (
            actions * self.timestep
        )

        return next_latents


# ============================================================
# FACTORY
# ============================================================

def create_latent_planner(
    dynamics: LatentDynamicsProtocol,
    config: LatentPlannerConfig | None = None,
    *,
    reward_fn: (
        Callable[
            [Tensor, Tensor, Tensor],
            Tensor,
        ]
        | None
    ) = None,
    uncertainty_fn: (
        Callable[
            [Tensor, Tensor, Tensor],
            Tensor,
        ]
        | None
    ) = None,
) -> LatentPlanner:
    return LatentPlanner(
        dynamics,
        config,
        reward_fn=reward_fn,
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
        "NEUROWORLD AI - LATENT PLANNING"
    )
    print("=" * 65)

    config = LatentPlannerConfig(
        latent_dim=8,
        action_dim=2,
        horizon=12,
        population_size=128,
        elite_size=16,
        iterations=6,
        action_min=-1.0,
        action_max=1.0,
        goal_tolerance=0.25,
    )

    dynamics = SimpleLatentDynamics(
        latent_dim=8,
        action_dim=2,
        timestep=0.2,
    )

    planner = LatentPlanner(
        dynamics,
        config,
    )

    initial_latent = torch.zeros(
        config.latent_dim
    )

    target = torch.zeros(
        config.latent_dim
    )

    target[0] = 1.0
    target[1] = 1.0

    # Only first two dimensions matter.
    mask = torch.zeros(
        config.latent_dim
    )

    mask[:2] = 1.0

    goal = LatentGoal(
        target=target,
        tolerance=0.25,
        mask=mask,
        goal_id="latent_navigation_goal",
    )

    planner.set_goal(
        goal
    )

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    result = planner.plan(
        initial_latent
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
            result.action_sequence.shape
        ),
    )

    print(
        "Latent trajectory shape:",
        tuple(
            result.latent_trajectory.shape
        ),
    )

    print(
        "Initial goal distance:",
        result.initial_goal_distance,
    )

    print(
        "Final goal distance:",
        result.final_goal_distance,
    )

    print(
        "Predicted goal reached:",
        result.predicted_goal_reached,
    )

    # --------------------------------------------------------
    # MPC SIMULATION
    # --------------------------------------------------------

    controller = LatentMPCController(
        planner
    )

    current_latent = (
        initial_latent.clone()
    )

    for step in range(5):
        action = controller.select_action(
            current_latent
        )

        current_latent = (
            dynamics.predict_next_latent(
                current_latent.unsqueeze(0),
                action.unsqueeze(0),
            )
        )[0]

        distance = (
            planner.evaluator.goal_metric.distance(
                current_latent.unsqueeze(0),
                goal,
            )
        )[0]

        print(
            f"MPC step {step + 1}:",
            "latent=",
            current_latent[:2].cpu().numpy(),
            "distance=",
            float(distance.item()),
        )

        if float(distance.item()) <= goal.tolerance:
            print(
                "Latent goal reached"
            )
            break

    print(
        "Latent planning demonstration completed"
    )


if __name__ == "__main__":
    main()
