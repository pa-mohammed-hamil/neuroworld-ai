
"""
NeuroWorld AI
File: neuroworld/planning/mpc.py

Model Predictive Control (MPC) Engine.

Features:
    1. Receding-horizon optimization
    2. Cross-Entropy Method (CEM) planning
    3. Learned dynamics integration
    4. Goal-conditioned rewards
    5. Batched future trajectory simulation
    6. Predictive uncertainty penalties
    7. Action bounds and rate constraints
    8. Candidate trajectory safety screening
    9. Terminal-state value function
   10. Warm-start planning
   11. Emergency fallback action
   12. Planning diagnostics and history
   13. NumPy and PyTorch interfaces
   14. Synthetic simulation demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.planning.mpc

Model interface:
    predict_step(states, actions)
        -> next_states, rewards, uncertainty, done

State shape:
    [state_dim]

Candidate action shape:
    [population, horizon, action_dim]

This module plans actions only.
It does not operate physical hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class MPCError(Exception):
    """Base MPC exception."""


class MPCValidationError(MPCError):
    """Invalid MPC inputs or configuration."""


# ============================================================
# STATUS
# ============================================================

class MPCStatus(str, Enum):
    READY = "ready"
    PLANNING = "planning"
    SUCCESS = "success"
    FALLBACK = "fallback"
    FAILED = "failed"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MPCConfig:
    state_dim: int = 128
    action_dim: int = 8

    horizon: int = 15
    population_size: int = 256
    elite_size: int = 32
    iterations: int = 8

    action_min: float = -1.0
    action_max: float = 1.0

    max_action_change: float = 0.5

    initial_std: float = 0.5
    min_std: float = 0.05
    max_std: float = 1.0

    smoothing: float = 0.2
    discount_factor: float = 0.99

    uncertainty_weight: float = 0.1
    action_weight: float = 0.001
    smoothness_weight: float = 0.01

    terminal_value_weight: float = 1.0

    max_acceptable_uncertainty: float = 0.5

    warm_start: bool = True

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
                "Invalid action bounds"
            )

        if self.max_action_change <= 0:
            raise ValueError(
                "max_action_change must be positive"
            )

        if not (
            0 < self.min_std
            <= self.initial_std
            <= self.max_std
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

        for name in (
            "uncertainty_weight",
            "action_weight",
            "smoothness_weight",
            "terminal_value_weight",
            "max_acceptable_uncertainty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

class MPCWorldModel(Protocol):
    """
    Batched world-model interface.

    Inputs:
        states: [N,state_dim]
        actions: [N,action_dim]

    Returns:
        next_states: [N,state_dim]
        rewards: [N]
        uncertainty: [N]
        done: [N]
    """

    def predict_step(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ...


# ============================================================
# MODEL ADAPTER
# ============================================================

class MPCWorldModelAdapter:
    """
    Construct an MPC world model from
    independent prediction functions.
    """

    def __init__(
        self,
        dynamics_fn: Callable[
            [Tensor, Tensor], Tensor
        ],
        reward_fn: Callable[
            [Tensor, Tensor, Tensor], Tensor
        ],
        uncertainty_fn: (
            Callable[
                [Tensor, Tensor, Tensor], Tensor
            ]
            | None
        ) = None,
        termination_fn: (
            Callable[
                [Tensor, Tensor, Tensor], Tensor
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        next_states = self.dynamics_fn(
            states, actions
        )

        rewards = self.reward_fn(
            states, actions, next_states
        )

        if self.uncertainty_fn is None:
            uncertainty = torch.zeros(
                states.shape[0],
                device=states.device,
            )
        else:
            uncertainty = self.uncertainty_fn(
                states, actions, next_states
            )

        if self.termination_fn is None:
            done = torch.zeros(
                states.shape[0],
                dtype=torch.bool,
                device=states.device,
            )
        else:
            done = self.termination_fn(
                states, actions, next_states
            )

        return next_states, rewards, uncertainty, done


# ============================================================
# SAFETY CONSTRAINTS
# ============================================================

class MPCSafetyConstraint(Protocol):
    """
    Return one Boolean per candidate state.

    True:
        Candidate satisfies this constraint.

    False:
        Candidate violates this constraint.
    """

    def is_safe(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> Tensor:
        ...


class StateBoundsConstraint:
    """
    Checks that predicted states remain
    inside configured state bounds.
    """

    def __init__(
        self,
        lower: Tensor | np.ndarray,
        upper: Tensor | np.ndarray,
    ) -> None:
        self.lower = torch.as_tensor(
            lower,
            dtype=torch.float32,
        )

        self.upper = torch.as_tensor(
            upper,
            dtype=torch.float32,
        )

        if self.lower.shape != self.upper.shape:
            raise MPCValidationError(
                "State bound shapes must match"
            )

        if not torch.all(
            self.lower <= self.upper
        ):
            raise MPCValidationError(
                "Lower bounds exceed upper bounds"
            )

    def is_safe(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> Tensor:
        lower = self.lower.to(
            next_states.device
        )

        upper = self.upper.to(
            next_states.device
        )

        return (
            (next_states >= lower)
            & (next_states <= upper)
        ).all(dim=-1)


# ============================================================
# ACTION CONSTRAINTS
# ============================================================

class MPCActionConstraint:
    """
    Enforces:
        action minimum
        action maximum
        maximum change per control step

    Rate limits are applied sequentially
    across each sampled action sequence.
    """

    def __init__(
        self,
        config: MPCConfig,
    ) -> None:
        self.config = config

    def project(
        self,
        actions: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        if actions.ndim != 3:
            raise MPCValidationError(
                "Actions must be [N,H,A]"
            )

        if previous_action.shape != (
            self.config.action_dim,
        ):
            raise MPCValidationError(
                "Invalid previous action shape"
            )

        projected = []

        last_action = previous_action.unsqueeze(
            0
        ).expand(
            actions.shape[0], -1
        )

        for step in range(
            actions.shape[1]
        ):
            current = actions[
                :, step
            ]

            current = current.clamp(
                self.config.action_min,
                self.config.action_max,
            )

            lower_rate = (
                last_action
                - self.config.max_action_change
            )

            upper_rate = (
                last_action
                + self.config.max_action_change
            )

            current = torch.maximum(
                current,
                lower_rate,
            )

            current = torch.minimum(
                current,
                upper_rate,
            )

            current = current.clamp(
                self.config.action_min,
                self.config.action_max,
            )

            projected.append(
                current
            )

            last_action = current

        return torch.stack(
            projected,
            dim=1,
        )


# ============================================================
# TRAJECTORY EVALUATION RESULT
# ============================================================

@dataclass
class MPCTrajectoryEvaluation:
    scores: Tensor
    rewards: Tensor
    uncertainty_costs: Tensor
    action_costs: Tensor
    smoothness_costs: Tensor

    trajectories: Tensor
    valid_mask: Tensor

    max_uncertainty: Tensor


# ============================================================
# TRAJECTORY EVALUATOR
# ============================================================

class MPCTrajectoryEvaluator:
    """
    Simulates candidate action sequences.

    Applies:
        discounted rewards
        uncertainty penalties
        action magnitude penalties
        action smoothness penalties
        safety screening
        terminal value estimates

    Invalid trajectories receive -inf scores.
    """

    def __init__(
        self,
        world_model: MPCWorldModel,
        config: MPCConfig,
        *,
        safety_constraints: (
            list[MPCSafetyConstraint] | None
        ) = None,
        terminal_value_fn: (
            Callable[[Tensor], Tensor] | None
        ) = None,
    ) -> None:
        self.world_model = world_model
        self.config = config

        self.safety_constraints = (
            safety_constraints or []
        )

        self.terminal_value_fn = (
            terminal_value_fn
        )

        self.device = torch.device(
            config.device
        )

    def _as_vector(
        self,
        value: Tensor,
        size: int,
        name: str,
        *,
        boolean: bool = False,
    ) -> Tensor:
        value = torch.as_tensor(
            value,
            dtype=(
                torch.bool
                if boolean
                else torch.float32
            ),
            device=self.device,
        )

        if value.shape != (size,):
            raise MPCValidationError(
                f"{name} must be [{size}]"
            )

        if (
            not boolean
            and not torch.isfinite(value).all()
        ):
            raise MPCValidationError(
                f"{name} contains non-finite values"
            )

        return value

    @torch.no_grad()
    def evaluate(
        self,
        initial_state: Tensor,
        action_sequences: Tensor,
        previous_action: Tensor,
    ) -> MPCTrajectoryEvaluation:
        population, horizon, action_dim = (
            action_sequences.shape
        )

        if (
            horizon != self.config.horizon
            or action_dim != self.config.action_dim
        ):
            raise MPCValidationError(
                "Invalid candidate action shape"
            )

        states = initial_state.unsqueeze(
            0
        ).expand(
            population, -1
        ).clone()

        alive = torch.ones(
            population,
            dtype=torch.bool,
            device=self.device,
        )

        valid = torch.ones_like(
            alive
        )

        total_rewards = torch.zeros(
            population,
            device=self.device,
        )

        uncertainty_costs = (
            torch.zeros_like(
                total_rewards
            )
        )

        action_costs = torch.zeros_like(
            total_rewards
        )

        smoothness_costs = torch.zeros_like(
            total_rewards
        )

        max_uncertainty = (
            torch.zeros_like(
                total_rewards
            )
        )

        trajectories = [
            states.clone()
        ]

        last_action = (
            previous_action.unsqueeze(0)
            .expand(population, -1)
        )

        discount = 1.0

        for step in range(horizon):
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

            next_states = torch.as_tensor(
                next_states,
                dtype=torch.float32,
                device=self.device,
            )

            if next_states.shape != (
                population,
                self.config.state_dim,
            ):
                raise MPCValidationError(
                    "Invalid world-model state output"
                )

            if not torch.isfinite(
                next_states
            ).all():
                raise MPCValidationError(
                    "Non-finite predicted states"
                )

            rewards = self._as_vector(
                rewards,
                population,
                "Rewards",
            )

            uncertainty = torch.as_tensor(
                uncertainty,
                dtype=torch.float32,
                device=self.device,
            )

            if uncertainty.ndim == 2:
                if uncertainty.shape[0] != population:
                    raise MPCValidationError(
                        "Invalid uncertainty batch size"
                    )

                uncertainty = uncertainty.mean(
                    dim=-1
                )

            uncertainty = self._as_vector(
                uncertainty,
                population,
                "Uncertainty",
            )

            if (uncertainty < 0).any():
                raise MPCValidationError(
                    "Uncertainty must be nonnegative"
                )

            done = self._as_vector(
                done,
                population,
                "Done",
                boolean=True,
            )

            active = (
                alive & valid
            )

            safe = torch.ones_like(
                valid
            )

            for constraint in self.safety_constraints:
                constraint_result = (
                    constraint.is_safe(
                        states,
                        actions,
                        next_states,
                    )
                )

                constraint_result = self._as_vector(
                    constraint_result,
                    population,
                    "Safety constraint",
                    boolean=True,
                )

                safe &= constraint_result

            safe &= (
                uncertainty
                <= self.config.max_acceptable_uncertainty
            )

            valid &= (
                ~active | safe
            )

            active_float = (
                active.float()
            )

            total_rewards += (
                discount
                * rewards
                * active_float
            )

            uncertainty_costs += (
                discount
                * self.config.uncertainty_weight
                * uncertainty
                * active_float
            )

            action_costs += (
                discount
                * self.config.action_weight
                * actions.square().sum(
                    dim=-1
                )
                * active_float
            )

            smoothness_costs += (
                discount
                * self.config.smoothness_weight
                * (actions - last_action)
                .square()
                .sum(dim=-1)
                * active_float
            )

            max_uncertainty = torch.maximum(
                max_uncertainty,
                torch.where(
                    active,
                    uncertainty,
                    torch.zeros_like(
                        uncertainty
                    ),
                ),
            )

            states = torch.where(
                active.unsqueeze(-1),
                next_states,
                states,
            )

            trajectories.append(
                states.clone()
            )

            alive &= (
                ~done & valid
            )

            last_action = actions

            discount *= (
                self.config.discount_factor
            )

        if self.terminal_value_fn is not None:
            terminal_values = self._as_vector(
                self.terminal_value_fn(
                    states
                ),
                population,
                "Terminal values",
            )

            total_rewards += (
                discount
                * self.config.terminal_value_weight
                * terminal_values
                * alive.float()
            )

        scores = (
            total_rewards
            - uncertainty_costs
            - action_costs
            - smoothness_costs
        )

        scores = torch.where(
            valid,
            scores,
            torch.full_like(
                scores,
                float("-inf"),
            ),
        )

        return MPCTrajectoryEvaluation(
            scores=scores,
            rewards=total_rewards,
            uncertainty_costs=uncertainty_costs,
            action_costs=action_costs,
            smoothness_costs=smoothness_costs,
            trajectories=torch.stack(
                trajectories,
                dim=1,
            ),
            valid_mask=valid,
            max_uncertainty=max_uncertainty,
        )


# ============================================================
# MPC RESULT
# ============================================================

@dataclass
class MPCResult:
    status: MPCStatus

    action: Tensor
    action_sequence: Tensor
    predicted_trajectory: Tensor

    best_score: float
    expected_reward: float
    uncertainty_cost: float

    max_predicted_uncertainty: float

    feasible_candidates: int

    optimization_history: list[
        dict[str, float]
    ] = field(default_factory=list)

    message: str = ""

    def to_numpy(self) -> dict[str, Any]:
        def convert(
            tensor: Tensor,
        ) -> np.ndarray:
            return (
                tensor.detach()
                .cpu()
                .numpy()
            )

        return {
            "status": self.status.value,
            "action": convert(
                self.action
            ),
            "action_sequence": convert(
                self.action_sequence
            ),
            "predicted_trajectory": convert(
                self.predicted_trajectory
            ),
            "best_score": self.best_score,
            "expected_reward": (
                self.expected_reward
            ),
            "uncertainty_cost": (
                self.uncertainty_cost
            ),
            "max_predicted_uncertainty": (
                self.max_predicted_uncertainty
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
# MPC CONTROLLER
# ============================================================

class ModelPredictiveController:
    """
    CEM-based Model Predictive Controller.

    Each control cycle:
        1. Observe current state
        2. Sample future action sequences
        3. Project action constraints
        4. Simulate trajectories
        5. Reject unsafe predictions
        6. Select elite trajectories
        7. Update action distribution
        8. Return first optimized action

    Actual execution is handled externally.
    """

    def __init__(
        self,
        world_model: MPCWorldModel,
        config: MPCConfig | None = None,
        *,
        safety_constraints: (
            list[MPCSafetyConstraint] | None
        ) = None,
        terminal_value_fn: (
            Callable[[Tensor], Tensor] | None
        ) = None,
        fallback_action: (
            Tensor | np.ndarray | None
        ) = None,
    ) -> None:
        self.config = (
            config or MPCConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.evaluator = MPCTrajectoryEvaluator(
            world_model,
            self.config,
            safety_constraints=safety_constraints,
            terminal_value_fn=terminal_value_fn,
        )

        self.action_constraint = (
            MPCActionConstraint(
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

        if fallback_action is None:
            self.fallback_action = (
                torch.zeros_like(
                    self.previous_action
                )
            )
        else:
            self.fallback_action = torch.as_tensor(
                fallback_action,
                dtype=torch.float32,
                device=self.device,
            )

            if self.fallback_action.shape != (
                self.config.action_dim,
            ):
                raise MPCValidationError(
                    "Invalid fallback action shape"
                )

            if not torch.isfinite(
                self.fallback_action
            ).all():
                raise MPCValidationError(
                    "Fallback action contains non-finite values"
                )

        if (
            (self.fallback_action < self.config.action_min).any()
            or (self.fallback_action > self.config.action_max).any()
        ):
            raise MPCValidationError(
                "Fallback action violates action bounds"
            )

        self._warm_mean: Tensor | None = None

        self.status = MPCStatus.READY

        self.last_result: MPCResult | None = None

        self.control_steps = 0

    def _prepare_state(
        self,
        state: Tensor | np.ndarray,
    ) -> Tensor:
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        if state.shape != (
            self.config.state_dim,
        ):
            raise MPCValidationError(
                "Invalid state dimensions"
            )

        if not torch.isfinite(state).all():
            raise MPCValidationError(
                "State contains non-finite values"
            )

        return state

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
            + std.unsqueeze(0)
            * noise
        )

        return self.action_constraint.project(
            candidates,
            self.previous_action,
        )

    def _make_fallback(
        self,
        state: Tensor,
        *,
        message: str,
        history: list[dict[str, float]],
    ) -> MPCResult:
        # The fallback is a configured
        # planning output, not a guarantee
        # that physical execution is safe.
        action = self.fallback_action.clone()

        sequence = action.unsqueeze(
            0
        ).expand(
            self.config.horizon,
            -1,
        ).clone()

        trajectory = state.unsqueeze(
            0
        ).expand(
            self.config.horizon + 1,
            -1,
        ).clone()

        return MPCResult(
            status=MPCStatus.FALLBACK,
            action=action,
            action_sequence=sequence,
            predicted_trajectory=trajectory,
            best_score=float("-inf"),
            expected_reward=0.0,
            uncertainty_cost=0.0,
            max_predicted_uncertainty=0.0,
            feasible_candidates=0,
            optimization_history=history,
            message=message,
        )

    @torch.no_grad()
    def plan(
        self,
        current_state: Tensor | np.ndarray,
    ) -> MPCResult:
        state = self._prepare_state(
            current_state
        )

        self.status = MPCStatus.PLANNING

        mean, std = (
            self._initial_distribution()
        )

        best_score = float("-inf")
        best_actions = None
        best_trajectory = None

        best_reward = 0.0
        best_uncertainty_cost = 0.0
        best_max_uncertainty = 0.0

        history = []
        feasible_candidates = 0

        for iteration in range(
            self.config.iterations
        ):
            candidates = (
                self._sample_candidates(
                    mean,
                    std,
                )
            )

            evaluation = (
                self.evaluator.evaluate(
                    state,
                    candidates,
                    self.previous_action,
                )
            )

            valid_indices = torch.nonzero(
                evaluation.valid_mask,
                as_tuple=False,
            ).flatten()

            feasible_candidates = max(
                feasible_candidates,
                int(
                    valid_indices.numel()
                ),
            )

            if valid_indices.numel() == 0:
                history.append(
                    {
                        "iteration": float(
                            iteration + 1
                        ),
                        "feasible_candidates": 0.0,
                    }
                )
                continue

            valid_scores = (
                evaluation.scores[
                    valid_indices
                ]
            )

            elite_count = min(
                self.config.elite_size,
                valid_indices.numel(),
            )

            elite_values, elite_positions = (
                torch.topk(
                    valid_scores,
                    k=elite_count,
                )
            )

            elite_indices = valid_indices[
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
                self.config.min_std,
                self.config.max_std,
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
                self.config.min_std,
                self.config.max_std,
            )

            iteration_best = int(
                valid_indices[
                    valid_scores.argmax()
                ].item()
            )

            iteration_score = float(
                evaluation.scores[
                    iteration_best
                ].item()
            )

            if iteration_score > best_score:
                best_score = iteration_score

                best_actions = candidates[
                    iteration_best
                ].clone()

                best_trajectory = (
                    evaluation.trajectories[
                        iteration_best
                    ].clone()
                )

                best_reward = float(
                    evaluation.rewards[
                        iteration_best
                    ].item()
                )

                best_uncertainty_cost = float(
                    evaluation.uncertainty_costs[
                        iteration_best
                    ].item()
                )

                best_max_uncertainty = float(
                    evaluation.max_uncertainty[
                        iteration_best
                    ].item()
                )

            history.append(
                {
                    "iteration": float(
                        iteration + 1
                    ),
                    "best_score": (
                        iteration_score
                    ),
                    "elite_mean_score": float(
                        elite_values.mean().item()
                    ),
                    "feasible_candidates": float(
                        valid_indices.numel()
                    ),
                    "distribution_std": float(
                        std.mean().item()
                    ),
                }
            )

        if best_actions is None:
            result = self._make_fallback(
                state,
                message=(
                    "No feasible candidate trajectory found"
                ),
                history=history,
            )

            self.status = MPCStatus.FALLBACK
            self.last_result = result
            return result

        if self.config.warm_start:
            self._warm_mean = torch.cat(
                [
                    best_actions[1:],
                    best_actions[-1:],
                ],
                dim=0,
            ).clone()

        result = MPCResult(
            status=MPCStatus.SUCCESS,
            action=best_actions[0],
            action_sequence=best_actions,
            predicted_trajectory=best_trajectory,
            best_score=best_score,
            expected_reward=best_reward,
            uncertainty_cost=best_uncertainty_cost,
            max_predicted_uncertainty=(
                best_max_uncertainty
            ),
            feasible_candidates=(
                feasible_candidates
            ),
            optimization_history=history,
            message="Feasible MPC plan generated",
        )

        self.status = MPCStatus.SUCCESS
        self.last_result = result

        return result

    def select_action(
        self,
        current_state: Tensor | np.ndarray,
    ) -> Tensor:
        result = self.plan(
            current_state
        )

        # Do not advance controller state
        # until the external system confirms
        # the action actually applied.
        return result.action

    def acknowledge_action(
        self,
        applied_action: Tensor | np.ndarray,
    ) -> None:
        action = torch.as_tensor(
            applied_action,
            dtype=torch.float32,
            device=self.device,
        )

        if action.shape != (
            self.config.action_dim,
        ):
            raise MPCValidationError(
                "Invalid acknowledged action shape"
            )

        if not torch.isfinite(action).all():
            raise MPCValidationError(
                "Acknowledged action is non-finite"
            )

        if (
            (action < self.config.action_min).any()
            or (action > self.config.action_max).any()
        ):
            raise MPCValidationError(
                "Acknowledged action violates bounds"
            )

        self.previous_action = action.clone()

        self.control_steps += 1

    def reset(self) -> None:
        self.previous_action.zero_()
        self._warm_mean = None
        self.status = MPCStatus.READY
        self.last_result = None
        self.control_steps = 0


# ============================================================
# NUMPY ADAPTER
# ============================================================

class MPCAdapter:
    """
    NumPy interface for backend services.
    """

    def __init__(
        self,
        controller: ModelPredictiveController,
    ) -> None:
        self.controller = controller

    def plan(
        self,
        state: np.ndarray,
    ) -> dict[str, Any]:
        return (
            self.controller.plan(
                state
            ).to_numpy()
        )

    def select_action(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        return (
            self.controller.select_action(
                state
            )
            .detach()
            .cpu()
            .numpy()
        )

    def acknowledge_action(
        self,
        action: np.ndarray,
    ) -> None:
        self.controller.acknowledge_action(
            action
        )


# ============================================================
# SYNTHETIC NAVIGATION WORLD MODEL
# ============================================================

class SimpleNavigationMPCWorldModel:
    """
    Example point-mass navigation model.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    Goal:
        [goal_x, goal_y]

    Not a trained neural world model.
    """

    def __init__(
        self,
        goal: tuple[float, float] = (2.0, 2.0),
        timestep: float = 0.1,
    ) -> None:
        self.goal = goal
        self.timestep = timestep

    def predict_step(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

        distance = torch.linalg.vector_norm(
            next_positions - goal,
            dim=-1,
        )

        rewards = (
            -distance
            - 0.01
            * actions.square().sum(
                dim=-1
            )
        )

        uncertainty = torch.full_like(
            rewards,
            0.01,
        )

        done = distance < 0.15

        return (
            next_states,
            rewards,
            uncertainty,
            done,
        )


# ============================================================
# FACTORY
# ============================================================

def create_mpc_controller(
    world_model: MPCWorldModel,
    config: MPCConfig | None = None,
    *,
    safety_constraints: (
        list[MPCSafetyConstraint] | None
    ) = None,
    terminal_value_fn: (
        Callable[[Tensor], Tensor] | None
    ) = None,
    fallback_action: (
        Tensor | np.ndarray | None
    ) = None,
) -> ModelPredictiveController:
    return ModelPredictiveController(
        world_model,
        config,
        safety_constraints=safety_constraints,
        terminal_value_fn=terminal_value_fn,
        fallback_action=fallback_action,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - MODEL PREDICTIVE CONTROL"
    )
    print("=" * 65)

    config = MPCConfig(
        state_dim=4,
        action_dim=2,
        horizon=20,
        population_size=128,
        elite_size=16,
        iterations=6,
        action_min=-1.0,
        action_max=1.0,
        max_action_change=0.4,
        max_acceptable_uncertainty=0.5,
    )

    world_model = (
        SimpleNavigationMPCWorldModel(
            goal=(2.0, 2.0),
        )
    )

    # Keep predicted positions inside
    # a simple rectangular workspace.
    safety_constraint = StateBoundsConstraint(
        lower=[
            -1.0,
            -1.0,
            -5.0,
            -5.0,
        ],
        upper=[
            5.0,
            5.0,
            5.0,
            5.0,
        ],
    )

    controller = ModelPredictiveController(
        world_model,
        config,
        safety_constraints=[
            safety_constraint
        ],
        fallback_action=np.zeros(
            2,
            dtype=np.float32,
        ),
    )

    current_state = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # --------------------------------------------------------
    # INITIAL MPC PLAN
    # --------------------------------------------------------

    result = controller.plan(
        current_state
    )

    print(
        "Planning status:",
        result.status.value,
    )

    print(
        "Best score:",
        result.best_score,
    )

    print(
        "First action:",
        result.action.cpu().numpy(),
    )

    print(
        "Predicted trajectory shape:",
        tuple(
            result.predicted_trajectory.shape
        ),
    )

    print(
        "Feasible candidates:",
        result.feasible_candidates,
    )

    # --------------------------------------------------------
    # RECEDING-HORIZON SIMULATION
    # --------------------------------------------------------

    for step in range(5):
        result = controller.plan(
            current_state
        )

        if result.status != MPCStatus.SUCCESS:
            print(
                "MPC fallback:",
                result.message,
            )
            break

        action = result.action

        (
            next_states,
            rewards,
            uncertainty,
            done,
        ) = world_model.predict_step(
            current_state.unsqueeze(0),
            action.unsqueeze(0),
        )

        # In a real system, acknowledgment
        # must come from the authorized
        # action-execution layer.
        controller.acknowledge_action(
            action
        )

        current_state = (
            next_states[0]
        )

        print(
            f"Control step {step + 1}:",
            "action=",
            action.cpu().numpy(),
            "position=",
            current_state[:2].cpu().numpy(),
            "reward=",
            float(rewards[0].item()),
        )

        if bool(done[0].item()):
            print(
                "Goal reached"
            )
            break

    print(
        "MPC demonstration completed"
    )


if __name__ == "__main__":
    main()
