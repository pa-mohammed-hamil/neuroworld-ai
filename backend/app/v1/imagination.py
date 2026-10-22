
"""
NeuroWorld AI
backend/app/core/imagination.py

Core imagination engine for neural world models.

Features:
- Action-conditioned future imagination
- Latent-state trajectory rollouts
- Multiple future branches
- Counterfactual imagination
- Uncertainty-aware termination
- Reward accumulation
- Goal-conditioned trajectory ranking
- Branch comparison
- Deterministic seeded action sampling
- Thread-safe imagination history
- JSON-compatible results

This module simulates predictions through a
world model. It does not execute real-world
actions or directly control physical robots.

Dependencies:
    pip install numpy
"""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


# ============================================================
# EXCEPTIONS
# ============================================================

class ImaginationError(Exception):
    """Base imagination error."""


class ImaginationValidationError(ImaginationError):
    """Invalid imagination input."""


class ImaginationModelError(ImaginationError):
    """World model prediction failed."""


class ImaginationNotFoundError(ImaginationError):
    """Requested imagination result was not found."""


# ============================================================
# ENUMS
# ============================================================

class ImaginationStatus(str, Enum):
    COMPLETED = "completed"
    UNCERTAINTY_LIMIT = "uncertainty_limit"
    TERMINAL_STATE = "terminal_state"
    MODEL_ERROR = "model_error"


class ActionSamplingStrategy(str, Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    ZERO = "zero"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def validate_vector(
    value: Sequence[float] | Array,
    *,
    name: str,
    dimension: int,
) -> Array:
    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ImaginationValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if array.ndim != 1:
        raise ImaginationValidationError(
            f"{name} must be one-dimensional"
        )

    if array.shape[0] != dimension:
        raise ImaginationValidationError(
            f"{name} must have dimension {dimension}"
        )

    if not np.all(np.isfinite(array)):
        raise ImaginationValidationError(
            f"{name} contains NaN or infinity"
        )

    return array.copy()


def validate_positive_integer(
    value: int,
    name: str,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ImaginationValidationError(
            f"{name} must be between 1 and {maximum}"
        )

    return value


def validate_finite_number(
    value: float,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ImaginationValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ImaginationValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ImaginationValidationError(
            f"{name} must be finite"
        )

    return number


def json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())

    if isinstance(value, np.generic):
        return json_compatible(value.item())

    if isinstance(value, dict):
        return {
            str(key): json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_compatible(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ImaginationValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ImaginationValidationError(
        f"Unsupported JSON value: {type(value).__name__}"
    )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

@dataclass(frozen=True)
class ModelTransition:
    next_state: Array
    reward: float = 0.0
    uncertainty: float = 0.0
    terminal: bool = False


@runtime_checkable
class WorldModelProtocol(Protocol):
    name: str
    state_dim: int
    action_dim: int

    def predict_transition(
        self,
        state: Array,
        action: Array,
    ) -> ModelTransition:
        ...


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ImaginationConfig:
    max_horizon: int = 128
    max_branches: int = 32
    max_history: int = 100

    action_min: float = -1.0
    action_max: float = 1.0

    uncertainty_threshold: float = 1.0
    stop_on_uncertainty: bool = True
    stop_on_terminal: bool = True

    reward_weight: float = 1.0
    goal_weight: float = 1.0
    uncertainty_penalty: float = 0.5
    action_penalty: float = 0.01

    def __post_init__(self) -> None:
        validate_positive_integer(
            self.max_horizon,
            "max_horizon",
            100000,
        )

        validate_positive_integer(
            self.max_branches,
            "max_branches",
            10000,
        )

        validate_positive_integer(
            self.max_history,
            "max_history",
            100000,
        )

        for name in (
            "action_min",
            "action_max",
            "uncertainty_threshold",
            "reward_weight",
            "goal_weight",
            "uncertainty_penalty",
            "action_penalty",
        ):
            validate_finite_number(
                getattr(self, name),
                name,
            )

        if self.action_min >= self.action_max:
            raise ImaginationValidationError(
                "action_min must be less than action_max"
            )

        if self.uncertainty_threshold < 0:
            raise ImaginationValidationError(
                "uncertainty_threshold must be nonnegative"
            )

        for name in (
            "reward_weight",
            "goal_weight",
            "uncertainty_penalty",
            "action_penalty",
        ):
            if getattr(self, name) < 0:
                raise ImaginationValidationError(
                    f"{name} cannot be negative"
                )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass(frozen=True)
class ImaginedStep:
    step_index: int
    state: list[float]
    action: list[float]
    next_state: list[float]
    reward: float
    uncertainty: float
    terminal: bool


@dataclass(frozen=True)
class ImaginedBranch:
    branch_id: str
    status: str
    steps: list[ImaginedStep]
    final_state: list[float]
    total_reward: float
    mean_uncertainty: float
    maximum_uncertainty: float
    goal_distance: float | None
    score: float
    error_message: str | None = None


@dataclass(frozen=True)
class ImaginationResult:
    imagination_id: str
    world_id: str
    model_name: str
    initial_state: list[float]
    goal_state: list[float] | None
    branches: list[ImaginedBranch]
    best_branch_id: str | None
    created_at: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class CounterfactualResult:
    counterfactual_id: str
    world_id: str
    factual_branch: ImaginedBranch
    alternative_branch: ImaginedBranch
    final_state_difference: list[float]
    final_state_distance: float
    reward_difference: float
    created_at: str


# ============================================================
# IMAGINATION ENGINE
# ============================================================

class ImaginationEngine:
    """
    Predict future states using a world model.

    The engine only evaluates model-generated
    trajectories. It never executes actions
    in the physical world.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: ImaginationConfig | None = None,
    ) -> None:
        if not isinstance(
            model,
            WorldModelProtocol,
        ):
            raise ImaginationValidationError(
                "Model must implement WorldModelProtocol"
            )

        if (
            isinstance(model.state_dim, bool)
            or not isinstance(model.state_dim, int)
            or model.state_dim < 1
        ):
            raise ImaginationValidationError(
                "Invalid model state_dim"
            )

        if (
            isinstance(model.action_dim, bool)
            or not isinstance(model.action_dim, int)
            or model.action_dim < 1
        ):
            raise ImaginationValidationError(
                "Invalid model action_dim"
            )

        self.model = model
        self.config = config or ImaginationConfig()

        self._history: dict[
            str, ImaginationResult
        ] = {}

        self._history_order: list[str] = []
        self._lock = threading.RLock()

    # --------------------------------------------------------
    # TRANSITION PREDICTION
    # --------------------------------------------------------

    def _predict(
        self,
        state: Array,
        action: Array,
    ) -> ModelTransition:
        try:
            transition = (
                self.model.predict_transition(
                    state.copy(),
                    action.copy(),
                )
            )

            next_state = validate_vector(
                transition.next_state,
                name="predicted next state",
                dimension=self.model.state_dim,
            )

            reward = validate_finite_number(
                transition.reward,
                "reward",
            )

            uncertainty = validate_finite_number(
                transition.uncertainty,
                "uncertainty",
            )

            if uncertainty < 0:
                raise ImaginationValidationError(
                    "Uncertainty cannot be negative"
                )

            if not isinstance(
                transition.terminal,
                (bool, np.bool_),
            ):
                raise ImaginationValidationError(
                    "terminal must be boolean"
                )

            return ModelTransition(
                next_state=next_state,
                reward=reward,
                uncertainty=uncertainty,
                terminal=bool(transition.terminal),
            )

        except Exception as exc:
            raise ImaginationModelError(
                f"World model prediction failed: "
                f"{type(exc).__name__}"
            ) from exc

    # --------------------------------------------------------
    # ACTION SAMPLING
    # --------------------------------------------------------

    def sample_actions(
        self,
        *,
        horizon: int,
        strategy: ActionSamplingStrategy = (
            ActionSamplingStrategy.UNIFORM
        ),
        rng: np.random.Generator | None = None,
    ) -> Array:
        validate_positive_integer(
            horizon,
            "horizon",
            self.config.max_horizon,
        )

        if not isinstance(
            strategy,
            ActionSamplingStrategy,
        ):
            raise ImaginationValidationError(
                "Invalid action sampling strategy"
            )

        rng = rng or np.random.default_rng()

        shape = (
            horizon,
            self.model.action_dim,
        )

        if strategy == ActionSamplingStrategy.ZERO:
            actions = np.zeros(
                shape,
                dtype=np.float64,
            )

        elif strategy == ActionSamplingStrategy.GAUSSIAN:
            center = (
                self.config.action_min
                + self.config.action_max
            ) / 2.0

            scale = (
                self.config.action_max
                - self.config.action_min
            ) / 6.0

            actions = rng.normal(
                loc=center,
                scale=scale,
                size=shape,
            )

            actions = np.clip(
                actions,
                self.config.action_min,
                self.config.action_max,
            )

        else:
            actions = rng.uniform(
                self.config.action_min,
                self.config.action_max,
                size=shape,
            )

        return np.asarray(
            actions,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # VALIDATE ACTION SEQUENCE
    # --------------------------------------------------------

    def _validate_actions(
        self,
        actions: Sequence[Sequence[float]] | Array,
    ) -> Array:
        try:
            array = np.asarray(
                actions,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ImaginationValidationError(
                "Invalid action sequence"
            ) from exc

        if array.ndim != 2:
            raise ImaginationValidationError(
                "Actions must be a 2D array"
            )

        if array.shape[1] != self.model.action_dim:
            raise ImaginationValidationError(
                "Action dimension mismatch"
            )

        validate_positive_integer(
            array.shape[0],
            "action horizon",
            self.config.max_horizon,
        )

        if not np.all(np.isfinite(array)):
            raise ImaginationValidationError(
                "Actions contain NaN or infinity"
            )

        if (
            np.any(array < self.config.action_min)
            or np.any(array > self.config.action_max)
        ):
            raise ImaginationValidationError(
                "Actions exceed configured bounds"
            )

        return array.copy()

    # --------------------------------------------------------
    # SCORE BRANCH
    # --------------------------------------------------------

    def _score_branch(
        self,
        *,
        final_state: Array,
        actions: Array,
        total_reward: float,
        mean_uncertainty: float,
        goal_state: Array | None,
    ) -> tuple[float, float | None]:
        goal_distance = None

        score = (
            self.config.reward_weight
            * total_reward
        )

        score -= (
            self.config.uncertainty_penalty
            * mean_uncertainty
        )

        if actions.size:
            action_cost = float(
                np.mean(
                    np.sum(
                        actions ** 2,
                        axis=1,
                    )
                )
            )

            score -= (
                self.config.action_penalty
                * action_cost
            )

        if goal_state is not None:
            goal_distance = float(
                np.linalg.norm(
                    final_state - goal_state
                )
            )

            score -= (
                self.config.goal_weight
                * goal_distance
            )

        if not math.isfinite(score):
            raise ImaginationValidationError(
                "Branch score is non-finite"
            )

        return float(score), goal_distance

    # --------------------------------------------------------
    # IMAGINE SINGLE BRANCH
    # --------------------------------------------------------

    def rollout(
        self,
        *,
        initial_state: Sequence[float] | Array,
        actions: Sequence[Sequence[float]] | Array,
        goal_state: Sequence[float] | Array | None = None,
    ) -> ImaginedBranch:
        state = validate_vector(
            initial_state,
            name="initial_state",
            dimension=self.model.state_dim,
        )

        action_array = self._validate_actions(
            actions
        )

        goal = (
            validate_vector(
                goal_state,
                name="goal_state",
                dimension=self.model.state_dim,
            )
            if goal_state is not None
            else None
        )

        steps: list[ImaginedStep] = []
        uncertainties: list[float] = []
        executed_actions: list[Array] = []

        total_reward = 0.0
        status = ImaginationStatus.COMPLETED

        for index, action in enumerate(
            action_array
        ):
            transition = self._predict(
                state,
                action,
            )

            steps.append(
                ImaginedStep(
                    step_index=index,
                    state=state.tolist(),
                    action=action.tolist(),
                    next_state=(
                        transition.next_state.tolist()
                    ),
                    reward=transition.reward,
                    uncertainty=transition.uncertainty,
                    terminal=transition.terminal,
                )
            )

            uncertainties.append(
                transition.uncertainty
            )

            executed_actions.append(
                action.copy()
            )

            total_reward += transition.reward
            state = transition.next_state.copy()

            if (
                self.config.stop_on_uncertainty
                and transition.uncertainty
                > self.config.uncertainty_threshold
            ):
                status = (
                    ImaginationStatus.UNCERTAINTY_LIMIT
                )
                break

            if (
                self.config.stop_on_terminal
                and transition.terminal
            ):
                status = (
                    ImaginationStatus.TERMINAL_STATE
                )
                break

        mean_uncertainty = (
            float(np.mean(uncertainties))
            if uncertainties
            else 0.0
        )

        maximum_uncertainty = (
            max(uncertainties)
            if uncertainties
            else 0.0
        )

        actual_actions = np.asarray(
            executed_actions,
            dtype=np.float64,
        )

        score, goal_distance = (
            self._score_branch(
                final_state=state,
                actions=actual_actions,
                total_reward=total_reward,
                mean_uncertainty=mean_uncertainty,
                goal_state=goal,
            )
        )

        return ImaginedBranch(
            branch_id=uuid.uuid4().hex,
            status=status.value,
            steps=steps,
            final_state=state.tolist(),
            total_reward=total_reward,
            mean_uncertainty=mean_uncertainty,
            maximum_uncertainty=maximum_uncertainty,
            goal_distance=goal_distance,
            score=score,
        )

    # --------------------------------------------------------
    # MULTI-BRANCH IMAGINATION
    # --------------------------------------------------------

    def imagine(
        self,
        *,
        world_id: str,
        initial_state: Sequence[float] | Array,
        horizon: int = 12,
        branch_count: int = 8,
        goal_state: Sequence[float] | Array | None = None,
        sampling_strategy: ActionSamplingStrategy = (
            ActionSamplingStrategy.UNIFORM
        ),
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ImaginationResult:
        if not isinstance(world_id, str) or not (
            world_id.strip()
        ):
            raise ImaginationValidationError(
                "world_id is required"
            )

        validate_positive_integer(
            horizon,
            "horizon",
            self.config.max_horizon,
        )

        validate_positive_integer(
            branch_count,
            "branch_count",
            self.config.max_branches,
        )

        initial = validate_vector(
            initial_state,
            name="initial_state",
            dimension=self.model.state_dim,
        )

        goal = (
            validate_vector(
                goal_state,
                name="goal_state",
                dimension=self.model.state_dim,
            )
            if goal_state is not None
            else None
        )

        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise ImaginationValidationError(
                "metadata must be a dictionary"
            )

        safe_metadata = json_compatible(
            metadata or {}
        )

        rng = np.random.default_rng(
            seed
        )

        branches = []

        for _ in range(branch_count):
            actions = self.sample_actions(
                horizon=horizon,
                strategy=sampling_strategy,
                rng=rng,
            )

            branch = self.rollout(
                initial_state=initial,
                actions=actions,
                goal_state=goal,
            )

            branches.append(branch)

        branches.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        result = ImaginationResult(
            imagination_id=uuid.uuid4().hex,
            world_id=world_id,
            model_name=self.model.name,
            initial_state=initial.tolist(),
            goal_state=(
                goal.tolist()
                if goal is not None
                else None
            ),
            branches=branches,
            best_branch_id=(
                branches[0].branch_id
                if branches
                else None
            ),
            created_at=utc_now(),
            metadata=safe_metadata,
        )

        self._save_history(
            result
        )

        return result

    # --------------------------------------------------------
    # COUNTERFACTUAL IMAGINATION
    # --------------------------------------------------------

    def counterfactual(
        self,
        *,
        world_id: str,
        initial_state: Sequence[float] | Array,
        factual_actions: (
            Sequence[Sequence[float]] | Array
        ),
        alternative_actions: (
            Sequence[Sequence[float]] | Array
        ),
        goal_state: Sequence[float] | Array | None = None,
    ) -> CounterfactualResult:
        if not isinstance(world_id, str) or not (
            world_id.strip()
        ):
            raise ImaginationValidationError(
                "world_id is required"
            )

        initial = validate_vector(
            initial_state,
            name="initial_state",
            dimension=self.model.state_dim,
        )

        factual = self.rollout(
            initial_state=initial,
            actions=factual_actions,
            goal_state=goal_state,
        )

        alternative = self.rollout(
            initial_state=initial,
            actions=alternative_actions,
            goal_state=goal_state,
        )

        difference = (
            np.asarray(
                alternative.final_state,
                dtype=np.float64,
            )
            - np.asarray(
                factual.final_state,
                dtype=np.float64,
            )
        )

        return CounterfactualResult(
            counterfactual_id=uuid.uuid4().hex,
            world_id=world_id,
            factual_branch=factual,
            alternative_branch=alternative,
            final_state_difference=(
                difference.tolist()
            ),
            final_state_distance=float(
                np.linalg.norm(difference)
            ),
            reward_difference=(
                alternative.total_reward
                - factual.total_reward
            ),
            created_at=utc_now(),
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def _save_history(
        self,
        result: ImaginationResult,
    ) -> None:
        with self._lock:
            self._history[
                result.imagination_id
            ] = copy.deepcopy(result)

            self._history_order.append(
                result.imagination_id
            )

            while len(
                self._history_order
            ) > self.config.max_history:
                oldest = self._history_order.pop(
                    0
                )

                self._history.pop(
                    oldest,
                    None,
                )

    def get_result(
        self,
        imagination_id: str,
    ) -> ImaginationResult:
        with self._lock:
            result = self._history.get(
                imagination_id
            )

            if result is None:
                raise ImaginationNotFoundError(
                    "Imagination result not found"
                )

            return copy.deepcopy(result)

    def list_results(
        self,
        *,
        world_id: str | None = None,
        limit: int = 50,
    ) -> list[ImaginationResult]:
        validate_positive_integer(
            limit,
            "limit",
            self.config.max_history,
        )

        with self._lock:
            results = [
                self._history[item]
                for item in reversed(
                    self._history_order
                )
            ]

            if world_id is not None:
                results = [
                    item
                    for item in results
                    if item.world_id == world_id
                ]

            return copy.deepcopy(
                results[:limit]
            )

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
            self._history_order.clear()

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_json(
        self,
        result: ImaginationResult,
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = json_compatible(
            asdict(result)
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return path

    # --------------------------------------------------------
    # ENGINE STATUS
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            history_count = len(
                self._history
            )

        return {
            "engine": "NeuroWorld Imagination",
            "model_name": self.model.name,
            "state_dimension": self.model.state_dim,
            "action_dimension": self.model.action_dim,
            "max_horizon": self.config.max_horizon,
            "max_branches": self.config.max_branches,
            "history_count": history_count,
            "physical_actuation_enabled": False,
        }


# ============================================================
# DEMO WORLD MODEL
# ============================================================

class DemoWorldModel:
    """
    Deterministic synthetic 2D dynamics.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    This is not a trained neural network.
    """

    name = "demo_synthetic_world_model"
    state_dim = 4
    action_dim = 2

    def __init__(
        self,
        dt: float = 0.1,
    ) -> None:
        self.dt = validate_finite_number(
            dt,
            "dt",
        )

        if self.dt <= 0:
            raise ImaginationValidationError(
                "dt must be positive"
            )

    def predict_transition(
        self,
        state: Array,
        action: Array,
    ) -> ModelTransition:
        x, y, vx, vy = state
        ax, ay = action

        next_vx = vx + ax * self.dt
        next_vy = vy + ay * self.dt

        next_x = x + next_vx * self.dt
        next_y = y + next_vy * self.dt

        next_state = np.array(
            [
                next_x,
                next_y,
                next_vx,
                next_vy,
            ],
            dtype=np.float64,
        )

        reward = -float(
            np.linalg.norm(
                next_state[:2]
                - np.array([5.0, 5.0])
            )
        )

        uncertainty = min(
            1.0,
            float(
                0.02
                + 0.03
                * np.linalg.norm(action)
            ),
        )

        return ModelTransition(
            next_state=next_state,
            reward=reward,
            uncertainty=uncertainty,
            terminal=False,
        )


# ============================================================
# FACTORY
# ============================================================

def create_imagination_engine(
    model: WorldModelProtocol | None = None,
    config: ImaginationConfig | None = None,
) -> ImaginationEngine:
    return ImaginationEngine(
        model=model or DemoWorldModel(),
        config=config,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = create_imagination_engine(
        config=ImaginationConfig(
            max_horizon=64,
            max_branches=16,
            max_history=100,
            uncertainty_threshold=0.8,
        )
    )

    initial_state = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    goal_state = [
        5.0,
        5.0,
        0.0,
        0.0,
    ]

    result = engine.imagine(
        world_id="demo_navigation_world",
        initial_state=initial_state,
        goal_state=goal_state,
        horizon=12,
        branch_count=8,
        sampling_strategy=(
            ActionSamplingStrategy.UNIFORM
        ),
        seed=42,
        metadata={
            "environment": "synthetic_navigation",
            "simulation_only": True,
        },
    )

    print("=" * 60)
    print("NEUROWORLD AI - IMAGINATION ENGINE")
    print("=" * 60)

    print(
        "Imagination ID:",
        result.imagination_id,
    )

    print(
        "Model:",
        result.model_name,
    )

    print(
        "Generated branches:",
        len(result.branches),
    )

    best = result.branches[0]

    print(
        "Best branch:",
        best.branch_id,
    )

    print(
        "Best score:",
        round(best.score, 4),
    )

    print(
        "Final state:",
        best.final_state,
    )

    print(
        "Goal distance:",
        best.goal_distance,
    )

    counterfactual = engine.counterfactual(
        world_id="demo_navigation_world",
        initial_state=initial_state,
        factual_actions=[
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
        ],
        alternative_actions=[
            [-0.2, 0.2],
            [-0.3, 0.3],
            [-0.4, 0.4],
        ],
        goal_state=goal_state,
    )

    print(
        "Counterfactual distance:",
        counterfactual.final_state_distance,
    )

    report = engine.export_json(
        result,
        "experiments/reports/imagination_result.json",
    )

    print(
        "Report:",
        report,
    )

    print(
        "Engine status:",
        engine.status(),
    )


if __name__ == "__main__":
    main()
