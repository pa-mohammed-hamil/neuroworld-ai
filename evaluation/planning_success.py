
"""
NeuroWorld AI
evaluation/planning_success.py

Planning Success Evaluation for Neural World Models.

Features:
- Goal achievement rate
- Final goal distance
- Goal improvement
- Planning efficiency
- Trajectory prediction accuracy
- Collision avoidance
- Action constraint compliance
- Planning latency
- Uncertainty analysis
- Multi-step execution evaluation
- Model-based planning comparison
- JSON / CSV report export
- Reproducible synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.planning_success

Planner interface:
    planner.name
    planner.plan(initial_state, goal_state)

Expected plan:
    plan.actions

World model interface:
    model.predict_transition(state, action)

Environment interface:
    environment.reset(initial_state)
    environment.step(action)

IMPORTANT:
    Planning success is measured against environment
    outcomes, not merely the planner's predictions.

    The demonstration uses synthetic dynamics and
    does not establish real-world robotic safety.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np


logger = logging.getLogger(
    "neuroworld.evaluation.planning_success"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class PlanningEvaluationError(Exception):
    pass


class PlanningValidationError(PlanningEvaluationError):
    pass


class PlanningExecutionError(PlanningEvaluationError):
    pass


# ============================================================
# INTERFACES
# ============================================================

class PlannerProtocol(Protocol):
    name: str

    def plan(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
    ) -> Any:
        ...


class WorldModelProtocol(Protocol):
    name: str
    state_dim: int
    action_dim: int

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        ...


class EvaluationEnvironmentProtocol(Protocol):
    def reset(
        self,
        initial_state: np.ndarray,
    ) -> np.ndarray:
        ...

    def step(
        self,
        action: np.ndarray,
    ) -> Any:
        ...


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PlanningSuccessConfig:
    goal_tolerance: float = 0.25

    max_episodes: int = 1000
    max_horizon: int = 128

    action_min: float = -1.0
    action_max: float = 1.0

    collision_penalty: float = 1.0

    goal_error_scale: float = 1.0
    trajectory_error_scale: float = 1.0

    latency_scale_ms: float = 1000.0

    uncertainty_threshold: float = 1.0

    pass_threshold: float = 0.70

    random_seed: int = 42
    epsilon: float = 1e-8

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "goal_success_rate": 0.35,
            "goal_improvement": 0.15,
            "planning_efficiency": 0.10,
            "trajectory_accuracy": 0.10,
            "collision_avoidance": 0.10,
            "action_compliance": 0.10,
            "latency_score": 0.05,
            "uncertainty_score": 0.05,
        }
    )

    def __post_init__(self):
        positive = (
            "goal_tolerance",
            "goal_error_scale",
            "trajectory_error_scale",
            "latency_scale_ms",
            "epsilon",
        )

        for name in positive:
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if self.max_episodes < 1:
            raise ValueError(
                "max_episodes must be positive"
            )

        if self.max_horizon < 1:
            raise ValueError(
                "max_horizon must be positive"
            )

        if not (
            math.isfinite(self.action_min)
            and math.isfinite(self.action_max)
            and self.action_min < self.action_max
        ):
            raise ValueError(
                "Invalid action bounds"
            )

        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError(
                "uncertainty_threshold must be in [0, 1]"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        allowed = {
            "goal_success_rate",
            "goal_improvement",
            "planning_efficiency",
            "trajectory_accuracy",
            "collision_avoidance",
            "action_compliance",
            "latency_score",
            "uncertainty_score",
        }

        if (
            not self.weights
            or set(self.weights) - allowed
            or any(
                not math.isfinite(weight) or weight < 0
                for weight in self.weights.values()
            )
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Invalid metric weights"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class PlanningScenario:
    scenario_id: str

    initial_state: np.ndarray
    goal_state: np.ndarray

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        state_dim: int,
    ) -> None:
        if not self.scenario_id.strip():
            raise PlanningValidationError(
                "scenario_id cannot be empty"
            )

        initial = np.asarray(
            self.initial_state,
            dtype=np.float64,
        )

        goal = np.asarray(
            self.goal_state,
            dtype=np.float64,
        )

        if (
            initial.shape != (state_dim,)
            or goal.shape != (state_dim,)
        ):
            raise PlanningValidationError(
                "Invalid scenario state dimensions"
            )

        if not (
            np.all(np.isfinite(initial))
            and np.all(np.isfinite(goal))
        ):
            raise PlanningValidationError(
                "Scenario states must be finite"
            )

        self.initial_state = initial.copy()
        self.goal_state = goal.copy()


@dataclass
class PlanningDataset:
    scenarios: list[PlanningScenario]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        state_dim: int,
        max_episodes: int,
    ) -> None:
        if not 1 <= len(self.scenarios) <= max_episodes:
            raise PlanningValidationError(
                "Invalid scenario count"
            )

        ids = []

        for scenario in self.scenarios:
            scenario.validate(state_dim)
            ids.append(scenario.scenario_id)

        if len(ids) != len(set(ids)):
            raise PlanningValidationError(
                "Scenario IDs must be unique"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class PlanningEpisodeMetrics:
    scenario_id: str

    success: bool
    initial_goal_distance: float
    final_goal_distance: float

    goal_improvement: float
    planning_efficiency: float

    trajectory_rmse: float | None

    collision_count: int
    action_violation_count: int

    planning_latency_ms: float

    mean_uncertainty: float | None

    executed_steps: int

    terminated_early: bool


@dataclass
class PlanningSuccessMetrics:
    goal_success_rate: float
    goal_improvement: float

    planning_efficiency: float

    trajectory_accuracy: float | None

    collision_avoidance: float
    action_compliance: float

    latency_score: float

    uncertainty_score: float | None

    overall_score: float


@dataclass
class PlanningSuccessResult:
    evaluation_id: str

    planner_name: str
    world_model_name: str

    scenario_count: int

    metrics: PlanningSuccessMetrics
    episode_metrics: list[PlanningEpisodeMetrics]

    passed: bool
    created_at: str

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPERS
# ============================================================

def goal_distance(
    state: np.ndarray,
    goal: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            state - goal
        )
    )


def error_to_score(
    error: float,
    scale: float,
) -> float:
    return float(
        np.clip(
            math.exp(-error / scale),
            0.0,
            1.0,
        )
    )


def safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(
        np.mean(values)
    )


# ============================================================
# PLANNING SUCCESS EVALUATOR
# ============================================================

class PlanningSuccessEvaluator:
    """
    Evaluate planner performance through
    actual environment execution.

    The world model is used only to compare
    predicted and executed trajectories.

    The environment determines actual outcomes.
    """

    def __init__(
        self,
        planner: PlannerProtocol,
        world_model: WorldModelProtocol,
        environment_factory: Any,
        config: PlanningSuccessConfig | None = None,
    ) -> None:
        self.planner = planner
        self.world_model = world_model

        self.environment_factory = (
            environment_factory
        )

        self.config = (
            config or PlanningSuccessConfig()
        )

        self._results: dict[
            str,
            PlanningSuccessResult,
        ] = {}

    # --------------------------------------------------------
    # WORLD MODEL PREDICTION
    # --------------------------------------------------------

    def _predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float | None]:
        try:
            transition = (
                self.world_model.predict_transition(
                    state.copy(),
                    action.copy(),
                )
            )

            next_state = np.asarray(
                transition.next_state,
                dtype=np.float64,
            )

            raw_uncertainty = getattr(
                transition,
                "uncertainty",
                None,
            )

            uncertainty = (
                float(raw_uncertainty)
                if raw_uncertainty is not None
                else None
            )

        except Exception as exc:
            raise PlanningExecutionError(
                f"World model prediction failed: {exc}"
            ) from exc

        if next_state.shape != (
            self.world_model.state_dim,
        ):
            raise PlanningExecutionError(
                "Invalid predicted state dimensions"
            )

        if not np.all(np.isfinite(next_state)):
            raise PlanningExecutionError(
                "Non-finite world-model prediction"
            )

        if uncertainty is not None and (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise PlanningExecutionError(
                "Invalid uncertainty value"
            )

        return next_state, uncertainty

    # --------------------------------------------------------
    # PLAN VALIDATION
    # --------------------------------------------------------

    def _extract_actions(
        self,
        plan: Any,
    ) -> np.ndarray:
        try:
            actions = np.asarray(
                plan.actions,
                dtype=np.float64,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise PlanningExecutionError(
                "Planner must return numeric actions"
            ) from exc

        if actions.ndim != 2:
            raise PlanningExecutionError(
                "Plan actions must be 2D"
            )

        if actions.shape[1] != self.world_model.action_dim:
            raise PlanningExecutionError(
                "Invalid plan action dimensions"
            )

        if not (
            1 <= len(actions)
            <= self.config.max_horizon
        ):
            raise PlanningExecutionError(
                "Invalid planning horizon"
            )

        if not np.all(np.isfinite(actions)):
            raise PlanningExecutionError(
                "Plan contains non-finite actions"
            )

        return actions

    # --------------------------------------------------------
    # SINGLE SCENARIO EVALUATION
    # --------------------------------------------------------

    def evaluate_episode(
        self,
        scenario: PlanningScenario,
    ) -> PlanningEpisodeMetrics:
        scenario.validate(
            self.world_model.state_dim
        )

        environment = (
            self.environment_factory()
        )

        initial_state = np.asarray(
            environment.reset(
                scenario.initial_state.copy()
            ),
            dtype=np.float64,
        )

        if initial_state.shape != (
            self.world_model.state_dim,
        ):
            raise PlanningExecutionError(
                "Environment returned invalid state"
            )

        start = time.perf_counter()

        try:
            plan = self.planner.plan(
                initial_state.copy(),
                scenario.goal_state.copy(),
            )
        except Exception as exc:
            raise PlanningExecutionError(
                f"Planner failed: {exc}"
            ) from exc

        planning_latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        actions = self._extract_actions(
            plan
        )

        initial_distance = goal_distance(
            initial_state,
            scenario.goal_state,
        )

        state = initial_state.copy()

        predicted_state = initial_state.copy()

        actual_trajectory = []
        predicted_trajectory = []

        uncertainties = []

        collision_count = 0
        action_violation_count = 0

        terminated_early = False

        for step_index, action in enumerate(actions):
            violations = np.count_nonzero(
                (action < self.config.action_min)
                | (action > self.config.action_max)
            )

            action_violation_count += int(
                violations
            )

            # Fail closed: do not execute an
            # out-of-bounds action.
            if violations:
                terminated_early = True
                break

            predicted_state, uncertainty = (
                self._predict(
                    predicted_state,
                    action,
                )
            )

            if uncertainty is not None:
                uncertainties.append(
                    uncertainty
                )

            try:
                step_result = environment.step(
                    action.copy()
                )

                actual_state = np.asarray(
                    step_result.next_state,
                    dtype=np.float64,
                )

                done = bool(
                    getattr(
                        step_result,
                        "done",
                        False,
                    )
                )

                collision = bool(
                    getattr(
                        step_result,
                        "collision",
                        False,
                    )
                )

            except Exception as exc:
                raise PlanningExecutionError(
                    f"Environment execution failed: {exc}"
                ) from exc

            if actual_state.shape != (
                self.world_model.state_dim,
            ):
                raise PlanningExecutionError(
                    "Environment returned invalid state"
                )

            if not np.all(np.isfinite(actual_state)):
                raise PlanningExecutionError(
                    "Environment returned non-finite state"
                )

            collision_count += int(collision)

            state = actual_state

            actual_trajectory.append(
                state.copy()
            )

            predicted_trajectory.append(
                predicted_state.copy()
            )

            if done:
                terminated_early = (
                    step_index + 1 < len(actions)
                )
                break

        final_distance = goal_distance(
            state,
            scenario.goal_state,
        )

        success = (
            final_distance
            <= self.config.goal_tolerance
            and collision_count == 0
            and action_violation_count == 0
        )

        goal_improvement = (
            initial_distance - final_distance
        )

        normalized_improvement = float(
            np.clip(
                goal_improvement
                / max(
                    initial_distance,
                    self.config.epsilon,
                ),
                0.0,
                1.0,
            )
        )

        executed_steps = len(
            actual_trajectory
        )

        # Efficiency proxy: fewer executed steps
        # relative to the submitted plan.
        planning_efficiency = (
            float(
                executed_steps
                / len(actions)
            )
            if success and executed_steps > 0
            else 0.0
        )

        trajectory_rmse = None

        if actual_trajectory:
            actual_array = np.asarray(
                actual_trajectory,
                dtype=np.float64,
            )

            predicted_array = np.asarray(
                predicted_trajectory,
                dtype=np.float64,
            )

            trajectory_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            predicted_array
                            - actual_array
                        ) ** 2
                    )
                )
            )

        return PlanningEpisodeMetrics(
            scenario_id=scenario.scenario_id,
            success=success,
            initial_goal_distance=initial_distance,
            final_goal_distance=final_distance,
            goal_improvement=normalized_improvement,
            planning_efficiency=planning_efficiency,
            trajectory_rmse=trajectory_rmse,
            collision_count=collision_count,
            action_violation_count=action_violation_count,
            planning_latency_ms=planning_latency_ms,
            mean_uncertainty=safe_mean(
                uncertainties
            ),
            executed_steps=executed_steps,
            terminated_early=terminated_early,
        )

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: PlanningDataset,
    ) -> PlanningSuccessResult:
        dataset.validate(
            self.world_model.state_dim,
            self.config.max_episodes,
        )

        episodes = [
            self.evaluate_episode(
                scenario
            )
            for scenario in dataset.scenarios
        ]

        scenario_count = len(
            episodes
        )

        success_rate = float(
            np.mean(
                [
                    episode.success
                    for episode in episodes
                ]
            )
        )

        improvement = float(
            np.mean(
                [
                    episode.goal_improvement
                    for episode in episodes
                ]
            )
        )

        efficiency = float(
            np.mean(
                [
                    episode.planning_efficiency
                    for episode in episodes
                ]
            )
        )

        trajectory_errors = [
            episode.trajectory_rmse
            for episode in episodes
            if episode.trajectory_rmse is not None
        ]

        trajectory_rmse = safe_mean(
            trajectory_errors
        )

        trajectory_accuracy = (
            error_to_score(
                trajectory_rmse,
                self.config.trajectory_error_scale,
            )
            if trajectory_rmse is not None
            else None
        )

        collision_free_count = sum(
            episode.collision_count == 0
            for episode in episodes
        )

        collision_avoidance = (
            collision_free_count
            / scenario_count
        )

        action_compliance = float(
            np.mean(
                [
                    episode.action_violation_count == 0
                    for episode in episodes
                ]
            )
        )

        mean_latency = float(
            np.mean(
                [
                    episode.planning_latency_ms
                    for episode in episodes
                ]
            )
        )

        latency_score = error_to_score(
            mean_latency,
            self.config.latency_scale_ms,
        )

        uncertainties = [
            episode.mean_uncertainty
            for episode in episodes
            if episode.mean_uncertainty is not None
        ]

        uncertainty_score = (
            float(
                np.mean(
                    [
                        uncertainty
                        <= self.config.uncertainty_threshold
                        for uncertainty in uncertainties
                    ]
                )
            )
            if uncertainties
            else None
        )

        scores: dict[str, float | None] = {
            "goal_success_rate": success_rate,
            "goal_improvement": improvement,
            "planning_efficiency": efficiency,
            "trajectory_accuracy": trajectory_accuracy,
            "collision_avoidance": collision_avoidance,
            "action_compliance": action_compliance,
            "latency_score": latency_score,
            "uncertainty_score": uncertainty_score,
        }

        active_weights = {
            name: weight
            for name, weight in self.config.weights.items()
            if scores[name] is not None
        }

        total_weight = sum(
            active_weights.values()
        )

        if total_weight <= 0:
            raise PlanningValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = PlanningSuccessMetrics(
            goal_success_rate=success_rate,
            goal_improvement=improvement,
            planning_efficiency=efficiency,
            trajectory_accuracy=trajectory_accuracy,
            collision_avoidance=collision_avoidance,
            action_compliance=action_compliance,
            latency_score=latency_score,
            uncertainty_score=uncertainty_score,
            overall_score=float(overall_score),
        )

        result = PlanningSuccessResult(
            evaluation_id=uuid.uuid4().hex,
            planner_name=self.planner.name,
            world_model_name=self.world_model.name,
            scenario_count=scenario_count,
            metrics=metrics,
            episode_metrics=episodes,
            passed=(
                overall_score
                >= self.config.pass_threshold
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            details={
                "weights": active_weights,
                "goal_tolerance": (
                    self.config.goal_tolerance
                ),
                "action_bounds": [
                    self.config.action_min,
                    self.config.action_max,
                ],
                "mean_planning_latency_ms": mean_latency,
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "interpretation": (
                    "Planning performance measured "
                    "against executed environment outcomes"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Planning evaluation: planner=%s "
            "success=%.4f score=%.4f",
            self.planner.name,
            success_rate,
            overall_score,
        )

        return result

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> PlanningSuccessResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise PlanningEvaluationError(
                "Planning result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[PlanningSuccessResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[PlanningSuccessResult],
    ) -> list[PlanningSuccessResult]:
        return sorted(
            results,
            key=lambda result: (
                result.metrics.overall_score
            ),
            reverse=True,
        )

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_json(
        self,
        evaluation_id: str,
        output_path: str | Path,
    ) -> Path:
        result = self.get_result(
            evaluation_id
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(result),
                file,
                indent=2,
                allow_nan=False,
            )

        return path

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    @staticmethod
    def export_csv(
        results: list[PlanningSuccessResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise PlanningValidationError(
                "No results to export"
            )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for result in results:
            row = {
                "evaluation_id": result.evaluation_id,
                "planner_name": result.planner_name,
                "world_model_name": (
                    result.world_model_name
                ),
                "scenario_count": result.scenario_count,
                "passed": result.passed,
                "created_at": result.created_at,
            }

            row.update(
                asdict(result.metrics)
            )

            rows.append(row)

        fieldnames = list(
            dict.fromkeys(
                key
                for row in rows
                for key in row
            )
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_planning_success_evaluator(
    planner: PlannerProtocol,
    world_model: WorldModelProtocol,
    environment_factory: Any,
    *,
    config: PlanningSuccessConfig | None = None,
) -> PlanningSuccessEvaluator:
    return PlanningSuccessEvaluator(
        planner=planner,
        world_model=world_model,
        environment_factory=environment_factory,
        config=config,
    )


# ============================================================
# DEMONSTRATION ENVIRONMENT
# ============================================================

class DemoStepResult:
    def __init__(
        self,
        next_state: np.ndarray,
        done: bool,
        collision: bool = False,
    ) -> None:
        self.next_state = next_state
        self.done = done
        self.collision = collision


class DemoEnvironment:
    """
    Simple 2D point environment.

    State:
        [x, y]

    Action:
        [dx, dy]

    Transition:
        next_state = state + 0.5 * action
    """

    def __init__(self):
        self.state = np.zeros(
            2,
            dtype=np.float64,
        )

    def reset(
        self,
        initial_state: np.ndarray,
    ) -> np.ndarray:
        self.state = initial_state.copy()
        return self.state.copy()

    def step(
        self,
        action: np.ndarray,
    ) -> DemoStepResult:
        self.state = (
            self.state + 0.5 * action
        )

        return DemoStepResult(
            next_state=self.state.copy(),
            done=False,
            collision=False,
        )


# ============================================================
# DEMONSTRATION WORLD MODEL
# ============================================================

class DemoWorldModel:
    name = "demo_planning_world_model"

    state_dim = 2
    action_dim = 2

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            next_state=state + 0.5 * action,
            uncertainty=0.05,
            terminal=False,
        )


# ============================================================
# DEMONSTRATION PLANNER
# ============================================================

class DemoPlanner:
    """
    Simple proportional goal-directed planner.

    Not a trained model-based planning algorithm.
    """

    name = "demo_goal_directed_planner"

    def __init__(
        self,
        horizon: int = 20,
    ):
        self.horizon = horizon

    def plan(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
    ) -> Any:
        from types import SimpleNamespace

        state = initial_state.copy()
        actions = []

        for _ in range(self.horizon):
            difference = (
                goal_state - state
            )

            action = np.clip(
                difference,
                -1.0,
                1.0,
            )

            actions.append(
                action.copy()
            )

            state = (
                state + 0.5 * action
            )

        return SimpleNamespace(
            actions=np.asarray(
                actions,
                dtype=np.float64,
            )
        )


# ============================================================
# SYNTHETIC SCENARIOS
# ============================================================

def generate_synthetic_planning_dataset(
    *,
    scenario_count: int = 50,
    seed: int = 42,
) -> PlanningDataset:
    if scenario_count < 1:
        raise ValueError(
            "scenario_count must be positive"
        )

    rng = np.random.default_rng(seed)

    scenarios = []

    for index in range(scenario_count):
        initial = rng.uniform(
            -3.0,
            3.0,
            size=2,
        )

        goal = rng.uniform(
            -3.0,
            3.0,
            size=2,
        )

        scenarios.append(
            PlanningScenario(
                scenario_id=f"scenario_{index:04d}",
                initial_state=initial,
                goal_state=goal,
            )
        )

    return PlanningDataset(
        scenarios=scenarios,
        metadata={
            "source": "synthetic_2d_navigation",
            "seed": seed,
        },
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    planner = DemoPlanner(
        horizon=20
    )

    world_model = DemoWorldModel()

    dataset = generate_synthetic_planning_dataset(
        scenario_count=50,
        seed=42,
    )

    evaluator = create_planning_success_evaluator(
        planner=planner,
        world_model=world_model,
        environment_factory=DemoEnvironment,
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — PLANNING SUCCESS")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/planning_success.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/planning_success.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
