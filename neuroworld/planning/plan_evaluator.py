
"""
NeuroWorld AI
File: neuroworld/planning/plan_evaluator.py

Intelligent Plan Evaluation Engine.

Features:
    1. Goal-distance evaluation
    2. Goal completion estimation
    3. Trajectory quality assessment
    4. Reward and return evaluation
    5. Action magnitude evaluation
    6. Action smoothness evaluation
    7. Predictive uncertainty evaluation
    8. State and action constraint checking
    9. Collision-clearance evaluation
   10. Safety screening
   11. Risk classification
   12. Multi-plan comparison and ranking
   13. Weighted plan scoring
   14. Plan acceptance/rejection
   15. JSON-compatible reports
   16. NumPy/PyTorch integration
   17. Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.planning.plan_evaluator

Conventions:
    states: [H+1, state_dim]
    actions: [H, action_dim]
    rewards: [H]
    uncertainties: [H] or [H, state_dim]

This evaluator assesses predicted plans.
It does not certify physical safety.
"""

from __future__ import annotations

import json
import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class PlanEvaluationError(Exception):
    """Base plan evaluation exception."""


class PlanValidationError(PlanEvaluationError):
    """Invalid plan inputs or dimensions."""


# ============================================================
# ENUMERATIONS
# ============================================================

class PlanRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PlanDecision(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PlanEvaluationConfig:
    state_dim: int = 128
    action_dim: int = 8

    discount_factor: float = 0.99
    goal_tolerance: float = 0.25

    goal_weight: float = 0.35
    reward_weight: float = 0.20
    smoothness_weight: float = 0.10
    efficiency_weight: float = 0.10
    uncertainty_weight: float = 0.15
    clearance_weight: float = 0.10

    max_acceptable_uncertainty: float = 0.5
    minimum_clearance: float = 0.2

    maximum_action_change: float = 0.5

    moderate_risk_threshold: float = 0.25
    high_risk_threshold: float = 0.50
    critical_risk_threshold: float = 0.75

    minimum_acceptance_score: float = 0.65

    device: str = "cpu"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError(
                "State/action dimensions must be positive"
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
            "goal_weight",
            "reward_weight",
            "smoothness_weight",
            "efficiency_weight",
            "uncertainty_weight",
            "clearance_weight",
            "max_acceptable_uncertainty",
            "minimum_clearance",
            "maximum_action_change",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )

        if not (
            0
            <= self.moderate_risk_threshold
            < self.high_risk_threshold
            < self.critical_risk_threshold
            <= 1
        ):
            raise ValueError(
                "Invalid risk thresholds"
            )

        if not 0 <= self.minimum_acceptance_score <= 1:
            raise ValueError(
                "Acceptance score must be in [0,1]"
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )


# ============================================================
# PLAN INPUT
# ============================================================

@dataclass
class PlanCandidate:
    plan_id: str

    states: Tensor | np.ndarray
    actions: Tensor | np.ndarray

    rewards: Tensor | np.ndarray | None = None

    uncertainties: (
        Tensor | np.ndarray | None
    ) = None

    goal_state: (
        Tensor | np.ndarray | None
    ) = None

    goal_mask: (
        Tensor | np.ndarray | None
    ) = None

    goal_weights: (
        Tensor | np.ndarray | None
    ) = None

    # Minimum obstacle clearance per state.
    clearances: (
        Tensor | np.ndarray | None
    ) = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise PlanValidationError(
                "plan_id cannot be empty"
            )


# ============================================================
# VALIDATED PLAN
# ============================================================

@dataclass
class ValidatedPlan:
    plan_id: str

    states: Tensor
    actions: Tensor

    rewards: Tensor | None
    uncertainties: Tensor | None

    goal_state: Tensor | None
    goal_mask: Tensor | None
    goal_weights: Tensor | None

    clearances: Tensor | None

    metadata: dict[str, Any]


# ============================================================
# INPUT VALIDATOR
# ============================================================

class PlanInputValidator:
    """
    Validates plan shapes, finiteness,
    and optional evaluation inputs.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(
            config.device
        )

    def tensor(
        self,
        value: Any,
        name: str,
    ) -> Tensor:
        tensor = torch.as_tensor(
            value,
            dtype=torch.float32,
            device=self.device,
        )

        if tensor.numel() == 0:
            raise PlanValidationError(
                f"{name} cannot be empty"
            )

        if not torch.isfinite(tensor).all():
            raise PlanValidationError(
                f"{name} contains non-finite values"
            )

        return tensor

    def validate(
        self,
        candidate: PlanCandidate,
    ) -> ValidatedPlan:
        states = self.tensor(
            candidate.states,
            "states",
        )

        actions = self.tensor(
            candidate.actions,
            "actions",
        )

        if (
            states.ndim != 2
            or states.shape[1]
            != self.config.state_dim
        ):
            raise PlanValidationError(
                "States must be [H+1,state_dim]"
            )

        if (
            actions.ndim != 2
            or actions.shape[1]
            != self.config.action_dim
        ):
            raise PlanValidationError(
                "Actions must be [H,action_dim]"
            )

        horizon = actions.shape[0]

        if states.shape[0] != horizon + 1:
            raise PlanValidationError(
                "Expected H+1 states for H actions"
            )

        rewards = None

        if candidate.rewards is not None:
            rewards = self.tensor(
                candidate.rewards,
                "rewards",
            )

            if rewards.shape != (horizon,):
                raise PlanValidationError(
                    "Rewards must be [H]"
                )

        uncertainties = None

        if candidate.uncertainties is not None:
            uncertainties = self.tensor(
                candidate.uncertainties,
                "uncertainties",
            )

            if uncertainties.shape == (
                horizon,
                self.config.state_dim,
            ):
                uncertainties = uncertainties.mean(
                    dim=-1
                )

            if uncertainties.shape != (
                horizon,
            ):
                raise PlanValidationError(
                    "Uncertainties must be [H] "
                    "or [H,state_dim]"
                )

            if (uncertainties < 0).any():
                raise PlanValidationError(
                    "Uncertainties must be nonnegative"
                )

        goal_state = None
        goal_mask = None
        goal_weights = None

        if candidate.goal_state is not None:
            goal_state = self.tensor(
                candidate.goal_state,
                "goal_state",
            )

            if goal_state.shape != (
                self.config.state_dim,
            ):
                raise PlanValidationError(
                    "Goal state dimensions mismatch"
                )

            if candidate.goal_mask is None:
                goal_mask = torch.ones_like(
                    goal_state
                )
            else:
                goal_mask = self.tensor(
                    candidate.goal_mask,
                    "goal_mask",
                )

                if goal_mask.shape != goal_state.shape:
                    raise PlanValidationError(
                        "Goal mask dimensions mismatch"
                    )

                if not torch.all(
                    (goal_mask == 0)
                    | (goal_mask == 1)
                ):
                    raise PlanValidationError(
                        "Goal mask must be binary"
                    )

            if candidate.goal_weights is None:
                goal_weights = torch.ones_like(
                    goal_state
                )
            else:
                goal_weights = self.tensor(
                    candidate.goal_weights,
                    "goal_weights",
                )

                if goal_weights.shape != goal_state.shape:
                    raise PlanValidationError(
                        "Goal weights dimensions mismatch"
                    )

                if (goal_weights < 0).any():
                    raise PlanValidationError(
                        "Goal weights must be nonnegative"
                    )

            if (
                (goal_mask * goal_weights)
                .sum()
                <= 0
            ):
                raise PlanValidationError(
                    "Goal requires positive active weights"
                )

        elif (
            candidate.goal_mask is not None
            or candidate.goal_weights is not None
        ):
            raise PlanValidationError(
                "Goal mask/weights require goal_state"
            )

        clearances = None

        if candidate.clearances is not None:
            clearances = self.tensor(
                candidate.clearances,
                "clearances",
            )

            if clearances.shape != (
                horizon + 1,
            ):
                raise PlanValidationError(
                    "Clearances must be [H+1]"
                )

            if (clearances < 0).any():
                raise PlanValidationError(
                    "Clearances must be nonnegative"
                )

        return ValidatedPlan(
            plan_id=candidate.plan_id,
            states=states,
            actions=actions,
            rewards=rewards,
            uncertainties=uncertainties,
            goal_state=goal_state,
            goal_mask=goal_mask,
            goal_weights=goal_weights,
            clearances=clearances,
            metadata=candidate.metadata,
        )


# ============================================================
# GOAL EVALUATOR
# ============================================================

class GoalEvaluator:
    """
    Evaluates predicted progress toward a goal.

    Goal completion is based on the final state.
    Minimum trajectory distance is also reported.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        plan: ValidatedPlan,
    ) -> dict[str, Any]:
        if plan.goal_state is None:
            return {
                "available": False,
                "goal_reached": None,
                "initial_distance": None,
                "final_distance": None,
                "minimum_distance": None,
                "goal_score": None,
            }

        weights = (
            plan.goal_mask
            * plan.goal_weights
        )

        distances = (
            (
                (plan.states - plan.goal_state)
                .square()
                * weights
            )
            .sum(dim=-1)
            .sqrt()
        )

        initial_distance = float(
            distances[0].item()
        )

        final_distance = float(
            distances[-1].item()
        )

        minimum_distance = float(
            distances.min().item()
        )

        goal_reached = (
            final_distance
            <= self.config.goal_tolerance
        )

        # Normalized heuristic score.
        goal_score = math.exp(
            -final_distance
            / max(
                self.config.goal_tolerance,
                self.config.epsilon,
            )
        )

        return {
            "available": True,
            "goal_reached": goal_reached,
            "goal_reached_anytime": bool(
                (
                    distances
                    <= self.config.goal_tolerance
                ).any().item()
            ),
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "minimum_distance": minimum_distance,
            "goal_score": goal_score,
        }


# ============================================================
# REWARD EVALUATOR
# ============================================================

class RewardEvaluator:
    """
    Computes cumulative and discounted reward.

    Reward normalization uses a bounded
    heuristic transform. It is not an
    empirically calibrated utility score.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        plan: ValidatedPlan,
    ) -> dict[str, Any]:
        if plan.rewards is None:
            return {
                "available": False,
                "cumulative_reward": None,
                "discounted_reward": None,
                "reward_score": None,
            }

        horizon = plan.rewards.shape[0]

        discounts = (
            self.config.discount_factor
            ** torch.arange(
                horizon,
                device=plan.rewards.device,
            )
        )

        cumulative_reward = float(
            plan.rewards.sum().item()
        )

        discounted_reward = float(
            (
                plan.rewards
                * discounts
            ).sum().item()
        )

        # Assumes rewards are reasonably scaled.
        normalized_return = (
            discounted_reward
            / max(horizon, 1)
        )

        reward_score = (
            math.tanh(normalized_return)
            + 1.0
        ) / 2.0

        return {
            "available": True,
            "cumulative_reward": cumulative_reward,
            "discounted_reward": discounted_reward,
            "reward_score": reward_score,
        }


# ============================================================
# ACTION QUALITY
# ============================================================

class ActionQualityEvaluator:
    """
    Measures action energy and smoothness.

    Smoothness:
        mean squared difference between
        consecutive actions.

    Efficiency:
        mean squared action magnitude.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        plan: ValidatedPlan,
    ) -> dict[str, Any]:
        actions = plan.actions

        mean_action_energy = float(
            actions.square()
            .sum(dim=-1)
            .mean()
            .item()
        )

        efficiency_score = math.exp(
            -mean_action_energy
        )

        if actions.shape[0] > 1:
            differences = (
                actions[1:]
                - actions[:-1]
            )

            mean_change = float(
                differences.abs()
                .max(dim=-1)
                .values
                .mean()
                .item()
            )

            smoothness_cost = float(
                differences.square()
                .sum(dim=-1)
                .mean()
                .item()
            )

            maximum_change = float(
                differences.abs()
                .max()
                .item()
            )

        else:
            mean_change = 0.0
            smoothness_cost = 0.0
            maximum_change = 0.0

        smoothness_score = math.exp(
            -smoothness_cost
        )

        rate_limit_satisfied = (
            maximum_change
            <= self.config.maximum_action_change
        )

        return {
            "mean_action_energy": (
                mean_action_energy
            ),
            "efficiency_score": (
                efficiency_score
            ),
            "smoothness_cost": (
                smoothness_cost
            ),
            "smoothness_score": (
                smoothness_score
            ),
            "mean_action_change": (
                mean_change
            ),
            "maximum_action_change": (
                maximum_change
            ),
            "rate_limit_satisfied": (
                rate_limit_satisfied
            ),
        }


# ============================================================
# UNCERTAINTY EVALUATOR
# ============================================================

class PlanUncertaintyEvaluator:
    """
    Evaluates predicted uncertainty.

    Uncertainty values should be expressed
    in a consistent scale across plans.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        plan: ValidatedPlan,
    ) -> dict[str, Any]:
        if plan.uncertainties is None:
            return {
                "available": False,
                "mean_uncertainty": None,
                "maximum_uncertainty": None,
                "uncertainty_score": None,
                "acceptable": None,
            }

        mean_uncertainty = float(
            plan.uncertainties.mean().item()
        )

        maximum_uncertainty = float(
            plan.uncertainties.max().item()
        )

        uncertainty_score = math.exp(
            -mean_uncertainty
        )

        acceptable = (
            maximum_uncertainty
            <= self.config.max_acceptable_uncertainty
        )

        return {
            "available": True,
            "mean_uncertainty": (
                mean_uncertainty
            ),
            "maximum_uncertainty": (
                maximum_uncertainty
            ),
            "uncertainty_score": (
                uncertainty_score
            ),
            "acceptable": acceptable,
        }


# ============================================================
# COLLISION CLEARANCE
# ============================================================

class CollisionClearanceEvaluator:
    """
    Evaluates obstacle clearance estimates.

    clearances:
        minimum obstacle distance at each state

    Clearances must be supplied by a collision
    or geometry module. This class does not
    compute collision geometry itself.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        plan: ValidatedPlan,
    ) -> dict[str, Any]:
        if plan.clearances is None:
            return {
                "available": False,
                "minimum_clearance": None,
                "mean_clearance": None,
                "clearance_score": None,
                "collision_risk": None,
            }

        minimum_clearance = float(
            plan.clearances.min().item()
        )

        mean_clearance = float(
            plan.clearances.mean().item()
        )

        collision_risk = (
            minimum_clearance
            < self.config.minimum_clearance
        )

        # Heuristic saturating clearance score.
        scale = max(
            self.config.minimum_clearance,
            self.config.epsilon,
        )

        clearance_score = (
            1.0
            - math.exp(
                -minimum_clearance / scale
            )
        )

        return {
            "available": True,
            "minimum_clearance": (
                minimum_clearance
            ),
            "mean_clearance": (
                mean_clearance
            ),
            "clearance_score": (
                clearance_score
            ),
            "collision_risk": (
                collision_risk
            ),
        }


# ============================================================
# CUSTOM SAFETY CONSTRAINTS
# ============================================================

SafetyConstraint = Callable[
    [Tensor, Tensor],
    bool | Tensor,
]


class PlanSafetyEvaluator:
    """
    Evaluates independent plan constraints.

    Each constraint receives:
        states [H+1,D]
        actions [H,A]

    and returns:
        bool or scalar Boolean Tensor.

    Constraint functions must be deterministic
    and must not execute hardware commands.
    """

    def __init__(
        self,
        constraints: (
            dict[str, SafetyConstraint] | None
        ) = None,
    ) -> None:
        self.constraints = (
            constraints or {}
        )

    def evaluate(
        self,
        plan: ValidatedPlan,
    ) -> dict[str, Any]:
        results = {}

        for name, constraint in (
            self.constraints.items()
        ):
            outcome = constraint(
                plan.states,
                plan.actions,
            )

            if isinstance(outcome, Tensor):
                if outcome.numel() != 1:
                    raise PlanEvaluationError(
                        f"Constraint {name} must "
                        "return one Boolean"
                    )

                outcome = bool(
                    outcome.item()
                )

            if not isinstance(
                outcome,
                (bool, np.bool_),
            ):
                raise PlanEvaluationError(
                    f"Constraint {name} returned "
                    "a non-Boolean value"
                )

            results[name] = bool(
                outcome
            )

        passed = all(
            results.values()
        )

        return {
            "constraints_checked": (
                len(results)
            ),
            "constraint_results": results,
            "all_constraints_passed": passed,
        }


# ============================================================
# SCORING OUTPUT
# ============================================================

@dataclass
class PlanEvaluationResult:
    plan_id: str

    score: float
    decision: PlanDecision
    risk_level: PlanRiskLevel

    goal_reached: bool | None

    valid: bool

    component_scores: dict[str, float]
    metrics: dict[str, Any]

    rejection_reasons: list[str]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "score": self.score,
            "decision": self.decision.value,
            "risk_level": self.risk_level.value,
            "goal_reached": self.goal_reached,
            "valid": self.valid,
            "component_scores": (
                self.component_scores
            ),
            "metrics": self.metrics,
            "rejection_reasons": (
                self.rejection_reasons
            ),
            "metadata": self.metadata,
        }

    def to_json(
        self,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str,
        )


# ============================================================
# MAIN PLAN EVALUATOR
# ============================================================

class PlanEvaluator:
    """
    Unified plan evaluation engine.

    Produces:
        goal metrics
        reward metrics
        action quality metrics
        uncertainty metrics
        clearance metrics
        safety constraint results
        composite score
        risk level
        acceptance decision

    Missing goal, uncertainty, or clearance
    data is not treated as proof of safety.
    """

    def __init__(
        self,
        config: PlanEvaluationConfig | None = None,
        *,
        safety_constraints: (
            dict[str, SafetyConstraint] | None
        ) = None,
    ) -> None:
        self.config = (
            config or PlanEvaluationConfig()
        )

        self.validator = PlanInputValidator(
            self.config
        )

        self.goal_evaluator = GoalEvaluator(
            self.config
        )

        self.reward_evaluator = RewardEvaluator(
            self.config
        )

        self.action_evaluator = (
            ActionQualityEvaluator(
                self.config
            )
        )

        self.uncertainty_evaluator = (
            PlanUncertaintyEvaluator(
                self.config
            )
        )

        self.clearance_evaluator = (
            CollisionClearanceEvaluator(
                self.config
            )
        )

        self.safety_evaluator = (
            PlanSafetyEvaluator(
                safety_constraints
            )
        )

    def _risk_level(
        self,
        risk_score: float,
    ) -> PlanRiskLevel:
        if (
            risk_score
            >= self.config.critical_risk_threshold
        ):
            return PlanRiskLevel.CRITICAL

        if (
            risk_score
            >= self.config.high_risk_threshold
        ):
            return PlanRiskLevel.HIGH

        if (
            risk_score
            >= self.config.moderate_risk_threshold
        ):
            return PlanRiskLevel.MODERATE

        return PlanRiskLevel.LOW

    @torch.no_grad()
    def evaluate(
        self,
        candidate: PlanCandidate,
    ) -> PlanEvaluationResult:
        plan = self.validator.validate(
            candidate
        )

        goal = self.goal_evaluator.evaluate(
            plan
        )

        reward = self.reward_evaluator.evaluate(
            plan
        )

        action = self.action_evaluator.evaluate(
            plan
        )

        uncertainty = (
            self.uncertainty_evaluator.evaluate(
                plan
            )
        )

        clearance = (
            self.clearance_evaluator.evaluate(
                plan
            )
        )

        safety = self.safety_evaluator.evaluate(
            plan
        )

        # ----------------------------------------------------
        # WEIGHTED COMPONENT SCORES
        # ----------------------------------------------------

        available_scores = {}
        available_weights = {}

        components = {
            "goal": (
                goal["goal_score"],
                self.config.goal_weight,
            ),
            "reward": (
                reward["reward_score"],
                self.config.reward_weight,
            ),
            "smoothness": (
                action["smoothness_score"],
                self.config.smoothness_weight,
            ),
            "efficiency": (
                action["efficiency_score"],
                self.config.efficiency_weight,
            ),
            "uncertainty": (
                uncertainty["uncertainty_score"],
                self.config.uncertainty_weight,
            ),
            "clearance": (
                clearance["clearance_score"],
                self.config.clearance_weight,
            ),
        }

        for name, (
            component_score,
            weight,
        ) in components.items():
            if component_score is not None:
                available_scores[name] = float(
                    component_score
                )

                available_weights[name] = float(
                    weight
                )

        total_weight = sum(
            available_weights.values()
        )

        if total_weight <= 0:
            raise PlanEvaluationError(
                "No positive-weight evaluation "
                "components are available"
            )

        composite_score = sum(
            available_scores[name]
            * available_weights[name]
            for name in available_scores
        ) / total_weight

        composite_score = max(
            0.0,
            min(1.0, composite_score),
        )

        # ----------------------------------------------------
        # HARD CONSTRAINTS
        # ----------------------------------------------------

        rejection_reasons = []

        if not action["rate_limit_satisfied"]:
            rejection_reasons.append(
                "Action rate limit exceeded"
            )

        if (
            uncertainty["available"]
            and not uncertainty["acceptable"]
        ):
            rejection_reasons.append(
                "Predicted uncertainty exceeds limit"
            )

        if (
            clearance["available"]
            and clearance["collision_risk"]
        ):
            rejection_reasons.append(
                "Obstacle clearance below threshold"
            )

        if not safety["all_constraints_passed"]:
            rejection_reasons.append(
                "Custom safety constraint failed"
            )

        valid = (
            len(rejection_reasons) == 0
        )

        # ----------------------------------------------------
        # RISK ESTIMATION
        # ----------------------------------------------------

        risk_components = [
            1.0
            - action["smoothness_score"],
            1.0
            - action["efficiency_score"],
        ]

        if uncertainty["available"]:
            risk_components.append(
                1.0
                - uncertainty["uncertainty_score"]
            )

        if clearance["available"]:
            risk_components.append(
                1.0
                - clearance["clearance_score"]
            )

        risk_score = (
            sum(risk_components)
            / len(risk_components)
        )

        if not valid:
            risk_score = max(
                risk_score,
                self.config.critical_risk_threshold,
            )

        risk_level = self._risk_level(
            risk_score
        )

        # ----------------------------------------------------
        # ACCEPTANCE DECISION
        # ----------------------------------------------------

        if not valid:
            decision = PlanDecision.REJECT

        elif (
            composite_score
            >= self.config.minimum_acceptance_score
            and risk_level
            in (
                PlanRiskLevel.LOW,
                PlanRiskLevel.MODERATE,
            )
        ):
            # Incomplete safety evidence requires
            # additional review.
            if (
                uncertainty["available"]
                and clearance["available"]
                and safety["constraints_checked"] > 0
            ):
                decision = PlanDecision.ACCEPT
            else:
                decision = PlanDecision.REVIEW

        else:
            decision = PlanDecision.REVIEW

        metrics = {
            "goal": goal,
            "reward": reward,
            "action": action,
            "uncertainty": uncertainty,
            "clearance": clearance,
            "safety": safety,
            "risk_score": risk_score,
        }

        return PlanEvaluationResult(
            plan_id=plan.plan_id,
            score=composite_score,
            decision=decision,
            risk_level=risk_level,
            goal_reached=goal["goal_reached"],
            valid=valid,
            component_scores=available_scores,
            metrics=metrics,
            rejection_reasons=rejection_reasons,
            metadata=plan.metadata,
        )


# ============================================================
# MULTI-PLAN RANKING
# ============================================================

@dataclass
class PlanRankingResult:
    ranked_plans: list[PlanEvaluationResult]

    best_plan_id: str | None
    accepted_count: int
    review_count: int
    rejected_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_plan_id": self.best_plan_id,
            "accepted_count": (
                self.accepted_count
            ),
            "review_count": (
                self.review_count
            ),
            "rejected_count": (
                self.rejected_count
            ),
            "ranked_plans": [
                result.to_dict()
                for result in self.ranked_plans
            ],
        }


class MultiPlanRanker:
    """
    Evaluate and rank multiple plans.

    Ordering:
        ACCEPT > REVIEW > REJECT

    Within each category:
        higher composite score first.
    """

    def __init__(
        self,
        evaluator: PlanEvaluator,
    ) -> None:
        self.evaluator = evaluator

    def rank(
        self,
        candidates: list[PlanCandidate],
    ) -> PlanRankingResult:
        if not candidates:
            raise PlanValidationError(
                "No plans provided"
            )

        results = [
            self.evaluator.evaluate(
                candidate
            )
            for candidate in candidates
        ]

        priority = {
            PlanDecision.ACCEPT: 0,
            PlanDecision.REVIEW: 1,
            PlanDecision.REJECT: 2,
        }

        results.sort(
            key=lambda result: (
                priority[result.decision],
                -result.score,
                result.plan_id,
            )
        )

        accepted = sum(
            result.decision
            == PlanDecision.ACCEPT
            for result in results
        )

        review = sum(
            result.decision
            == PlanDecision.REVIEW
            for result in results
        )

        rejected = sum(
            result.decision
            == PlanDecision.REJECT
            for result in results
        )

        # Do not recommend a rejected plan
        # as the best candidate.
        best_plan_id = next(
            (
                result.plan_id
                for result in results
                if result.decision
                != PlanDecision.REJECT
            ),
            None,
        )

        return PlanRankingResult(
            ranked_plans=results,
            best_plan_id=best_plan_id,
            accepted_count=accepted,
            review_count=review,
            rejected_count=rejected,
        )


# ============================================================
# NUMPY / API ADAPTER
# ============================================================

class PlanEvaluatorAdapter:
    """
    Convenient interface for FastAPI,
    simulation, and frontend dashboards.
    """

    def __init__(
        self,
        evaluator: PlanEvaluator,
    ) -> None:
        self.evaluator = evaluator

    def evaluate(
        self,
        *,
        plan_id: str,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | None = None,
        uncertainties: np.ndarray | None = None,
        goal_state: np.ndarray | None = None,
        goal_mask: np.ndarray | None = None,
        clearances: np.ndarray | None = None,
    ) -> dict[str, Any]:
        candidate = PlanCandidate(
            plan_id=plan_id,
            states=states,
            actions=actions,
            rewards=rewards,
            uncertainties=uncertainties,
            goal_state=goal_state,
            goal_mask=goal_mask,
            clearances=clearances,
        )

        result = self.evaluator.evaluate(
            candidate
        )

        return result.to_dict()


# ============================================================
# SYNTHETIC DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - PLAN EVALUATION ENGINE"
    )
    print("=" * 65)

    config = PlanEvaluationConfig(
        state_dim=4,
        action_dim=2,
        goal_tolerance=0.25,
        max_acceptable_uncertainty=0.5,
        minimum_clearance=0.2,
        maximum_action_change=0.5,
    )

    # --------------------------------------------------------
    # CUSTOM SAFETY CONSTRAINT
    # --------------------------------------------------------

    def workspace_constraint(
        states: Tensor,
        actions: Tensor,
    ) -> bool:
        positions = states[:, :2]

        return bool(
            (
                (positions >= -1.0)
                & (positions <= 5.0)
            ).all().item()
        )

    evaluator = PlanEvaluator(
        config,
        safety_constraints={
            "workspace_bounds": (
                workspace_constraint
            ),
        },
    )

    horizon = 10

    start = torch.tensor(
        [0.0, 0.0, 0.0, 0.0]
    )

    goal = torch.tensor(
        [2.0, 2.0, 0.0, 0.0]
    )

    # --------------------------------------------------------
    # PLAN A
    # --------------------------------------------------------

    states_a = torch.stack(
        [
            start
            + (goal - start)
            * (step / horizon)
            for step in range(horizon + 1)
        ]
    )

    actions_a = torch.full(
        (horizon, 2),
        0.2,
    )

    plan_a = PlanCandidate(
        plan_id="plan_A",
        states=states_a,
        actions=actions_a,
        rewards=torch.ones(
            horizon
        ),
        uncertainties=torch.full(
            (horizon,),
            0.1,
        ),
        goal_state=goal,
        goal_mask=torch.tensor(
            [1.0, 1.0, 0.0, 0.0]
        ),
        clearances=torch.full(
            (horizon + 1,),
            0.8,
        ),
        metadata={
            "planner": "CEM",
        },
    )

    # --------------------------------------------------------
    # PLAN B
    # --------------------------------------------------------

    states_b = torch.stack(
        [
            start
            + (goal - start)
            * (0.7 * step / horizon)
            for step in range(horizon + 1)
        ]
    )

    actions_b = torch.full(
        (horizon, 2),
        0.35,
    )

    plan_b = PlanCandidate(
        plan_id="plan_B",
        states=states_b,
        actions=actions_b,
        rewards=torch.full(
            (horizon,),
            0.5,
        ),
        uncertainties=torch.full(
            (horizon,),
            0.2,
        ),
        goal_state=goal,
        goal_mask=torch.tensor(
            [1.0, 1.0, 0.0, 0.0]
        ),
        clearances=torch.full(
            (horizon + 1,),
            0.4,
        ),
        metadata={
            "planner": "MPC",
        },
    )

    # --------------------------------------------------------
    # PLAN C - UNSAFE CLEARANCE
    # --------------------------------------------------------

    states_c = states_a.clone()

    actions_c = torch.full(
        (horizon, 2),
        0.4,
    )

    plan_c = PlanCandidate(
        plan_id="plan_C",
        states=states_c,
        actions=actions_c,
        rewards=torch.full(
            (horizon,),
            1.5,
        ),
        uncertainties=torch.full(
            (horizon,),
            0.1,
        ),
        goal_state=goal,
        goal_mask=torch.tensor(
            [1.0, 1.0, 0.0, 0.0]
        ),
        clearances=torch.full(
            (horizon + 1,),
            0.05,
        ),
        metadata={
            "planner": "Experimental",
        },
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    for candidate in (
        plan_a,
        plan_b,
        plan_c,
    ):
        result = evaluator.evaluate(
            candidate
        )

        print(
            f"\nPlan: {result.plan_id}"
        )

        print(
            "Score:",
            round(result.score, 4),
        )

        print(
            "Decision:",
            result.decision.value,
        )

        print(
            "Risk:",
            result.risk_level.value,
        )

        print(
            "Goal reached:",
            result.goal_reached,
        )

        print(
            "Rejection reasons:",
            result.rejection_reasons,
        )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    ranker = MultiPlanRanker(
        evaluator
    )

    ranking = ranker.rank(
        [
            plan_a,
            plan_b,
            plan_c,
        ]
    )

    print(
        "\nBest plan:",
        ranking.best_plan_id,
    )

    print(
        "Accepted:",
        ranking.accepted_count,
    )

    print(
        "Review:",
        ranking.review_count,
    )

    print(
        "Rejected:",
        ranking.rejected_count,
    )

    print(
        "\nPlan evaluation demonstration completed"
    )


if __name__ == "__main__":
    main()
