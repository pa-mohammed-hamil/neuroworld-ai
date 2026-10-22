
"""
NeuroWorld AI
File: neuroworld/planning/planner.py

Unified Intelligent Planning Engine.

Features:
    - Multiple planning strategies
    - CEM / MPC / latent / trajectory planner adapters
    - Goal-conditioned planning
    - Dynamic strategy selection
    - Plan validation
    - Plan quality evaluation
    - Risk and safety screening
    - Fallback strategy
    - Plan ranking
    - Human approval requirements
    - Execution authorization checks
    - Planning history and diagnostics
    - NumPy / PyTorch compatibility

This module generates and evaluates plans.
It does not execute physical robot commands.

Install:
    pip install numpy torch

Run:
    python -m neuroworld.planning.planner
"""

from __future__ import annotations

import json
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class PlanningError(Exception):
    """Base planning error."""


class PlanningValidationError(PlanningError):
    """Invalid planning request."""


class StrategyUnavailableError(PlanningError):
    """Requested planning strategy unavailable."""


# ============================================================
# ENUMERATIONS
# ============================================================

class PlanningStrategy(str, Enum):
    AUTO = "auto"
    CEM = "cem"
    MPC = "mpc"
    LATENT = "latent"
    TRAJECTORY = "trajectory"
    GOAL_CONDITIONED = "goal_conditioned"


class PlanningStatus(str, Enum):
    SUCCESS = "success"
    REVIEW = "review"
    REJECTED = "rejected"
    FAILED = "failed"


class PlanningRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PlannerConfig:
    state_dim: int = 128
    action_dim: int = 8

    default_strategy: PlanningStrategy = (
        PlanningStrategy.AUTO
    )

    minimum_plan_score: float = 0.65

    maximum_planning_time_seconds: float = 10.0

    allow_fallback: bool = True

    require_evaluation: bool = True
    require_human_approval: bool = True

    reject_high_risk: bool = True

    max_history: int = 100

    action_min: float = -1.0
    action_max: float = 1.0

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError(
                "state_dim must be positive"
            )

        if self.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if not (
            0.0
            <= self.minimum_plan_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_plan_score must be in [0,1]"
            )

        if self.maximum_planning_time_seconds <= 0:
            raise ValueError(
                "Planning time must be positive"
            )

        if self.max_history <= 0:
            raise ValueError(
                "max_history must be positive"
            )

        if self.action_min >= self.action_max:
            raise ValueError(
                "Invalid action bounds"
            )


# ============================================================
# REQUEST
# ============================================================

@dataclass
class PlanningRequest:
    current_state: Tensor | np.ndarray

    goal_state: Tensor | np.ndarray | None = None

    strategy: PlanningStrategy = (
        PlanningStrategy.AUTO
    )

    goal_mask: Tensor | np.ndarray | None = None

    context: dict[str, Any] = field(
        default_factory=dict
    )

    request_id: str = field(
        default_factory=lambda: str(
            uuid.uuid4()
        )
    )


# ============================================================
# STRATEGY OUTPUT
# ============================================================

@dataclass
class StrategyPlan:
    """
    Common output from any planner.

    states:
        [H+1, state_dim]

    actions:
        [H, action_dim]

    Optional evaluation signals:
        rewards [H]
        uncertainties [H]
        clearances [H+1]
    """

    actions: Tensor | np.ndarray
    states: Tensor | np.ndarray

    rewards: Tensor | np.ndarray | None = None

    uncertainties: (
        Tensor | np.ndarray | None
    ) = None

    clearances: (
        Tensor | np.ndarray | None
    ) = None

    predicted_score: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# STRATEGY PROTOCOL
# ============================================================

class PlanningStrategyProtocol(Protocol):
    def plan(
        self,
        request: PlanningRequest,
    ) -> StrategyPlan:
        ...


# ============================================================
# GENERIC FUNCTION ADAPTER
# ============================================================

class FunctionPlannerAdapter:
    """
    Wraps a planning function.

    Expected function:
        fn(request) -> StrategyPlan

    Useful for connecting custom planners.
    """

    def __init__(
        self,
        fn: Callable[
            [PlanningRequest],
            StrategyPlan,
        ],
    ) -> None:
        self.fn = fn

    def plan(
        self,
        request: PlanningRequest,
    ) -> StrategyPlan:
        return self.fn(request)


# ============================================================
# MPC ADAPTER
# ============================================================

class MPCPlannerAdapter:
    """
    Adapter for ModelPredictiveController
    from neuroworld.planning.mpc.

    The MPC controller must be configured
    with matching state/action dimensions.
    """

    def __init__(
        self,
        controller: Any,
    ) -> None:
        self.controller = controller

    def plan(
        self,
        request: PlanningRequest,
    ) -> StrategyPlan:
        result = self.controller.plan(
            request.current_state
        )

        status = getattr(
            result,
            "status",
            None,
        )

        status_value = getattr(
            status,
            "value",
            status,
        )

        if status_value != "success":
            raise PlanningError(
                "MPC did not produce a feasible plan"
            )

        return StrategyPlan(
            actions=result.action_sequence,
            states=result.predicted_trajectory,
            predicted_score=result.best_score,
            metadata={
                "source": "mpc",
                "max_predicted_uncertainty": (
                    result.max_predicted_uncertainty
                ),
            },
        )


# ============================================================
# LATENT PLANNER ADAPTER
# ============================================================

class LatentPlannerAdapter:
    """
    Adapter for LatentPlanner.

    IMPORTANT:
        The unified planner's state_dim
        must equal the latent planner's
        latent_dim.

    Goal mask support is forwarded.
    """

    def __init__(
        self,
        planner: Any,
    ) -> None:
        self.planner = planner

    def plan(
        self,
        request: PlanningRequest,
    ) -> StrategyPlan:
        if request.goal_state is None:
            raise PlanningValidationError(
                "Latent planning requires a goal"
            )

        from neuroworld.planning.latent_planner import (
            LatentGoal,
        )

        goal = LatentGoal(
            target=request.goal_state,
            mask=request.goal_mask,
        )

        result = self.planner.plan(
            request.current_state,
            goal=goal,
        )

        return StrategyPlan(
            actions=result.action_sequence,
            states=result.latent_trajectory,
            predicted_score=result.best_score,
            metadata={
                "source": "latent",
                "predicted_goal_reached": (
                    result.predicted_goal_reached
                ),
            },
        )


# ============================================================
# PLANNING RESULT
# ============================================================

@dataclass
class UnifiedPlanningResult:
    request_id: str

    status: PlanningStatus
    strategy: PlanningStrategy

    actions: Tensor | None
    predicted_states: Tensor | None

    score: float | None
    risk: PlanningRisk

    evaluation: dict[str, Any] | None

    approval_required: bool
    approved: bool

    elapsed_seconds: float

    message: str

    attempted_strategies: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Tensor):
                return (
                    value.detach()
                    .cpu()
                    .tolist()
                )

            if isinstance(value, np.ndarray):
                return value.tolist()

            if isinstance(value, Enum):
                return value.value

            if isinstance(value, dict):
                return {
                    str(k): convert(v)
                    for k, v in value.items()
                }

            if isinstance(value, (list, tuple)):
                return [
                    convert(v)
                    for v in value
                ]

            if isinstance(value, np.generic):
                return value.item()

            return value

        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "strategy": self.strategy.value,
            "actions": convert(self.actions),
            "predicted_states": convert(
                self.predicted_states
            ),
            "score": self.score,
            "risk": self.risk.value,
            "evaluation": convert(
                self.evaluation
            ),
            "approval_required": (
                self.approval_required
            ),
            "approved": self.approved,
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "message": self.message,
            "attempted_strategies": (
                self.attempted_strategies
            ),
            "metadata": convert(
                self.metadata
            ),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            indent=2,
        )


# ============================================================
# STRATEGY SELECTOR
# ============================================================

class PlanningStrategySelector:
    """
    Chooses a planning strategy based on
    request context and available planners.

    AUTO preference:
        real-time -> MPC
        latent -> LATENT
        explicit goal -> GOAL_CONDITIONED
        otherwise -> CEM / TRAJECTORY
    """

    def select(
        self,
        request: PlanningRequest,
        available: set[PlanningStrategy],
    ) -> list[PlanningStrategy]:
        if request.strategy != PlanningStrategy.AUTO:
            if request.strategy not in available:
                raise StrategyUnavailableError(
                    f"Strategy unavailable: "
                    f"{request.strategy.value}"
                )

            return [request.strategy]

        preferences = []

        if request.context.get(
            "real_time",
            False,
        ):
            preferences.append(
                PlanningStrategy.MPC
            )

        if request.context.get(
            "latent_space",
            False,
        ):
            preferences.append(
                PlanningStrategy.LATENT
            )

        if request.goal_state is not None:
            preferences.append(
                PlanningStrategy.GOAL_CONDITIONED
            )

        preferences.extend(
            [
                PlanningStrategy.MPC,
                PlanningStrategy.CEM,
                PlanningStrategy.TRAJECTORY,
                PlanningStrategy.LATENT,
                PlanningStrategy.GOAL_CONDITIONED,
            ]
        )

        ordered = []

        for strategy in preferences:
            if (
                strategy in available
                and strategy not in ordered
            ):
                ordered.append(
                    strategy
                )

        return ordered


# ============================================================
# UNIFIED PLANNER
# ============================================================

class UnifiedPlanner:
    """
    Central planning orchestrator.

    Responsibilities:
        register planning strategies
        choose appropriate planner
        validate candidate plans
        run evaluation
        reject unsafe plans
        apply fallback strategy
        return structured result

    No physical actuation is performed.
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
        *,
        evaluator: Any | None = None,
    ) -> None:
        self.config = (
            config or PlannerConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.evaluator = evaluator

        self.strategies: dict[
            PlanningStrategy,
            PlanningStrategyProtocol,
        ] = {}

        self.selector = (
            PlanningStrategySelector()
        )

        self.history: list[
            UnifiedPlanningResult
        ] = []

    def register_strategy(
        self,
        strategy: PlanningStrategy,
        planner: PlanningStrategyProtocol,
    ) -> None:
        if strategy == PlanningStrategy.AUTO:
            raise PlanningValidationError(
                "AUTO cannot be registered"
            )

        self.strategies[strategy] = planner

    def unregister_strategy(
        self,
        strategy: PlanningStrategy,
    ) -> None:
        self.strategies.pop(
            strategy,
            None,
        )

    def available_strategies(
        self,
    ) -> list[str]:
        return [
            strategy.value
            for strategy in self.strategies
        ]

    def _validate_request(
        self,
        request: PlanningRequest,
    ) -> None:
        state = torch.as_tensor(
            request.current_state,
            dtype=torch.float32,
            device=self.device,
        )

        if state.shape != (
            self.config.state_dim,
        ):
            raise PlanningValidationError(
                "Current state dimensions mismatch"
            )

        if not torch.isfinite(state).all():
            raise PlanningValidationError(
                "Current state contains non-finite values"
            )

        if request.goal_state is not None:
            goal = torch.as_tensor(
                request.goal_state,
                dtype=torch.float32,
                device=self.device,
            )

            if goal.shape != (
                self.config.state_dim,
            ):
                raise PlanningValidationError(
                    "Goal state dimensions mismatch"
                )

            if not torch.isfinite(goal).all():
                raise PlanningValidationError(
                    "Goal contains non-finite values"
                )

        if request.goal_mask is not None:
            if request.goal_state is None:
                raise PlanningValidationError(
                    "Goal mask requires goal state"
                )

            mask = torch.as_tensor(
                request.goal_mask,
                dtype=torch.float32,
                device=self.device,
            )

            if mask.shape != (
                self.config.state_dim,
            ):
                raise PlanningValidationError(
                    "Goal mask dimensions mismatch"
                )

            if not torch.all(
                (mask == 0) | (mask == 1)
            ):
                raise PlanningValidationError(
                    "Goal mask must be binary"
                )

    def _validate_plan(
        self,
        plan: StrategyPlan,
    ) -> tuple[Tensor, Tensor]:
        actions = torch.as_tensor(
            plan.actions,
            dtype=torch.float32,
            device=self.device,
        )

        states = torch.as_tensor(
            plan.states,
            dtype=torch.float32,
            device=self.device,
        )

        if (
            actions.ndim != 2
            or actions.shape[1]
            != self.config.action_dim
        ):
            raise PlanningValidationError(
                "Actions must be [H,action_dim]"
            )

        if (
            states.ndim != 2
            or states.shape[1]
            != self.config.state_dim
        ):
            raise PlanningValidationError(
                "States must be [H+1,state_dim]"
            )

        if (
            actions.shape[0] == 0
            or states.shape[0]
            != actions.shape[0] + 1
        ):
            raise PlanningValidationError(
                "Inconsistent planning horizon"
            )

        if (
            not torch.isfinite(actions).all()
            or not torch.isfinite(states).all()
        ):
            raise PlanningValidationError(
                "Plan contains non-finite values"
            )

        if (
            (actions < self.config.action_min).any()
            or (actions > self.config.action_max).any()
        ):
            raise PlanningValidationError(
                "Action bounds violated"
            )

        return actions, states

    def _evaluate(
        self,
        request: PlanningRequest,
        plan: StrategyPlan,
    ) -> tuple[
        float | None,
        PlanningRisk,
        dict[str, Any] | None,
        PlanningStatus,
    ]:
        if self.evaluator is None:
            if self.config.require_evaluation:
                raise PlanningError(
                    "Plan evaluator is required"
                )

            return (
                None,
                PlanningRisk.UNKNOWN,
                None,
                PlanningStatus.REVIEW,
            )

        from neuroworld.planning.plan_evaluator import (
            PlanCandidate,
        )

        candidate = PlanCandidate(
            plan_id=request.request_id,
            states=plan.states,
            actions=plan.actions,
            rewards=plan.rewards,
            uncertainties=plan.uncertainties,
            goal_state=request.goal_state,
            goal_mask=request.goal_mask,
            clearances=plan.clearances,
            metadata=plan.metadata,
        )

        evaluation_result = (
            self.evaluator.evaluate(
                candidate
            )
        )

        score = float(
            evaluation_result.score
        )

        risk = PlanningRisk(
            evaluation_result.risk_level.value
        )

        decision = (
            evaluation_result.decision.value
        )

        if decision == "reject":
            status = PlanningStatus.REJECTED

        elif (
            self.config.reject_high_risk
            and risk
            in (
                PlanningRisk.HIGH,
                PlanningRisk.CRITICAL,
            )
        ):
            status = PlanningStatus.REJECTED

        elif (
            score
            < self.config.minimum_plan_score
        ):
            status = PlanningStatus.REVIEW

        elif decision == "accept":
            status = PlanningStatus.SUCCESS

        else:
            status = PlanningStatus.REVIEW

        return (
            score,
            risk,
            evaluation_result.to_dict(),
            status,
        )

    def _remember(
        self,
        result: UnifiedPlanningResult,
    ) -> None:
        self.history.append(
            result
        )

        if (
            len(self.history)
            > self.config.max_history
        ):
            self.history = self.history[
                -self.config.max_history:
            ]

    def plan(
        self,
        request: PlanningRequest,
    ) -> UnifiedPlanningResult:
        started = time.perf_counter()

        self._validate_request(
            request
        )

        available = set(
            self.strategies.keys()
        )

        strategies = self.selector.select(
            request,
            available,
        )

        if not strategies:
            raise StrategyUnavailableError(
                "No planning strategies registered"
            )

        attempted = []
        errors = []

        for strategy in strategies:
            elapsed = (
                time.perf_counter()
                - started
            )

            if (
                elapsed
                >= self.config.maximum_planning_time_seconds
            ):
                errors.append(
                    "Planning time budget exceeded"
                )
                break

            attempted.append(
                strategy.value
            )

            try:
                strategy_planner = (
                    self.strategies[
                        strategy
                    ]
                )

                candidate = (
                    strategy_planner.plan(
                        request
                    )
                )

                actions, states = (
                    self._validate_plan(
                        candidate
                    )
                )

                (
                    score,
                    risk,
                    evaluation,
                    status,
                ) = self._evaluate(
                    request,
                    candidate,
                )

                if status == PlanningStatus.REJECTED:
                    errors.append(
                        f"{strategy.value}: "
                        "plan rejected by evaluation"
                    )

                    if self.config.allow_fallback:
                        continue

                result = UnifiedPlanningResult(
                    request_id=request.request_id,
                    status=status,
                    strategy=strategy,
                    actions=actions,
                    predicted_states=states,
                    score=score,
                    risk=risk,
                    evaluation=evaluation,
                    approval_required=(
                        self.config.require_human_approval
                    ),
                    approved=False,
                    elapsed_seconds=(
                        time.perf_counter()
                        - started
                    ),
                    message=(
                        "Plan generated; execution "
                        "requires separate authorization"
                    ),
                    attempted_strategies=attempted.copy(),
                    metadata=candidate.metadata,
                )

                self._remember(
                    result
                )

                return result

            except (
                PlanningError,
                ValueError,
                RuntimeError,
            ) as exc:
                errors.append(
                    f"{strategy.value}: {exc}"
                )

                if not self.config.allow_fallback:
                    break

        result = UnifiedPlanningResult(
            request_id=request.request_id,
            status=PlanningStatus.FAILED,
            strategy=(
                attempted
                and PlanningStrategy(
                    attempted[-1]
                )
                or PlanningStrategy.AUTO
            ),
            actions=None,
            predicted_states=None,
            score=None,
            risk=PlanningRisk.UNKNOWN,
            evaluation=None,
            approval_required=True,
            approved=False,
            elapsed_seconds=(
                time.perf_counter()
                - started
            ),
            message="; ".join(errors)
            or "No valid plan found",
            attempted_strategies=attempted,
        )

        self._remember(
            result
        )

        return result

    def recent_history(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        return [
            result.to_dict()
            for result in self.history[-limit:]
        ]


# ============================================================
# SYNTHETIC PLANNER
# ============================================================

class LinearGoalPlanner:
    """
    Synthetic demonstration planner.

    Produces a simple straight-line state
    interpolation and zero actions.

    This is NOT a physically valid dynamics
    controller. Use only for testing the
    unified planner orchestration.
    """

    def __init__(
        self,
        horizon: int = 10,
        action_dim: int = 2,
    ) -> None:
        self.horizon = horizon
        self.action_dim = action_dim

    def plan(
        self,
        request: PlanningRequest,
    ) -> StrategyPlan:
        if request.goal_state is None:
            raise PlanningValidationError(
                "Goal required"
            )

        start = torch.as_tensor(
            request.current_state,
            dtype=torch.float32,
        )

        goal = torch.as_tensor(
            request.goal_state,
            dtype=torch.float32,
        )

        states = torch.stack(
            [
                start
                + (goal - start)
                * (step / self.horizon)
                for step in range(
                    self.horizon + 1
                )
            ]
        )

        actions = torch.zeros(
            self.horizon,
            self.action_dim,
        )

        return StrategyPlan(
            actions=actions,
            states=states,
            rewards=torch.ones(
                self.horizon
            ),
            uncertainties=torch.full(
                (self.horizon,),
                0.05,
            ),
            clearances=torch.full(
                (self.horizon + 1,),
                1.0,
            ),
            metadata={
                "planner": "synthetic_linear",
                "demo_only": True,
            },
        )


# ============================================================
# FACTORY
# ============================================================

def create_unified_planner(
    config: PlannerConfig | None = None,
    *,
    evaluator: Any | None = None,
) -> UnifiedPlanner:
    return UnifiedPlanner(
        config=config,
        evaluator=evaluator,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from neuroworld.planning.plan_evaluator import (
        PlanEvaluator,
        PlanEvaluationConfig,
    )

    config = PlannerConfig(
        state_dim=4,
        action_dim=2,
        minimum_plan_score=0.60,
        require_evaluation=True,
        require_human_approval=True,
    )

    evaluator = PlanEvaluator(
        PlanEvaluationConfig(
            state_dim=4,
            action_dim=2,
        ),
        safety_constraints={
            "workspace": lambda states, actions: bool(
                (
                    (states[:, :2] >= -5.0)
                    & (states[:, :2] <= 5.0)
                ).all().item()
            ),
        },
    )

    planner = UnifiedPlanner(
        config,
        evaluator=evaluator,
    )

    planner.register_strategy(
        PlanningStrategy.TRAJECTORY,
        LinearGoalPlanner(
            horizon=10,
            action_dim=2,
        ),
    )

    request = PlanningRequest(
        current_state=np.array(
            [0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        goal_state=np.array(
            [2.0, 2.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        goal_mask=np.array(
            [1.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        context={
            "real_time": False,
        },
    )

    result = planner.plan(
        request
    )

    print("=" * 65)
    print(
        "NEUROWORLD AI - UNIFIED PLANNER"
    )
    print("=" * 65)

    print(
        "Status:",
        result.status.value,
    )

    print(
        "Strategy:",
        result.strategy.value,
    )

    print(
        "Score:",
        result.score,
    )

    print(
        "Risk:",
        result.risk.value,
    )

    print(
        "Approval required:",
        result.approval_required,
    )

    print(
        "Execution approved:",
        result.approved,
    )

    print(
        "Elapsed seconds:",
        result.elapsed_seconds,
    )

    print(
        "Attempted strategies:",
        result.attempted_strategies,
    )

    if result.actions is not None:
        print(
            "Action shape:",
            tuple(result.actions.shape),
        )

    if result.predicted_states is not None:
        print(
            "Predicted states shape:",
            tuple(result.predicted_states.shape),
        )

    print(
        "\nPlanning demonstration completed"
    )


if __name__ == "__main__":
    main()
