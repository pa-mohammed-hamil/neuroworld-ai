
"""
NeuroWorld AI
evaluation/physical_consistency.py

Physical Consistency Evaluation for Neural World Models.

Features:
- Position-velocity consistency
- Newton's second law consistency
- Gravity consistency
- Kinetic and potential energy conservation
- Momentum conservation
- Ground penetration detection
- Collision impulse consistency
- Collision restitution consistency
- Multi-step physical consistency
- Per-metric diagnostics
- Weighted overall scoring
- JSON / CSV reports
- Reproducible demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.physical_consistency

Assumed state representation:
    [px, py, vx, vy]

Assumed action representation:
    [ax, ay]

Coordinate convention:
    Positive Y points upward.
    Gravity acts in negative Y.

These assumptions are configurable through
PhysicalStateLayout and PhysicalConsistencyConfig.

IMPORTANT:
    Conservation metrics apply only when the
    corresponding external forces and collision
    conditions are known.

    This evaluator tests model predictions against
    configured physics assumptions. It does not
    certify real-world physical safety.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np


logger = logging.getLogger(
    "neuroworld.evaluation.physical_consistency"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class PhysicalConsistencyError(Exception):
    pass


class PhysicalValidationError(PhysicalConsistencyError):
    pass


class PhysicalModelError(PhysicalConsistencyError):
    pass


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PhysicalStateLayout:
    position_x: int = 0
    position_y: int = 1
    velocity_x: int = 2
    velocity_y: int = 3

    action_x: int = 0
    action_y: int = 1

    def validate(
        self,
        state_dim: int,
        action_dim: int,
    ) -> None:
        state_indices = (
            self.position_x,
            self.position_y,
            self.velocity_x,
            self.velocity_y,
        )

        action_indices = (
            self.action_x,
            self.action_y,
        )

        if (
            len(set(state_indices)) != 4
            or min(state_indices) < 0
            or max(state_indices) >= state_dim
        ):
            raise PhysicalValidationError(
                "Invalid state layout"
            )

        if (
            len(set(action_indices)) != 2
            or min(action_indices) < 0
            or max(action_indices) >= action_dim
        ):
            raise PhysicalValidationError(
                "Invalid action layout"
            )


@dataclass(frozen=True)
class PhysicalConsistencyConfig:
    time_step: float = 0.1
    mass: float = 1.0

    gravity: float = 9.81
    ground_height: float = 0.0

    restitution: float = 0.8

    position_error_scale: float = 0.1
    velocity_error_scale: float = 0.1
    energy_error_scale: float = 1.0
    momentum_error_scale: float = 0.1
    penetration_error_scale: float = 0.1

    collision_height_tolerance: float = 0.05
    collision_velocity_tolerance: float = 0.05

    max_samples: int = 10000
    max_horizon: int = 256

    pass_threshold: float = 0.70

    epsilon: float = 1e-8

    layout: PhysicalStateLayout = field(
        default_factory=PhysicalStateLayout
    )

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "kinematic_consistency": 0.20,
            "dynamics_consistency": 0.20,
            "gravity_consistency": 0.10,
            "energy_consistency": 0.15,
            "momentum_consistency": 0.10,
            "ground_consistency": 0.15,
            "collision_consistency": 0.10,
        }
    )

    def __post_init__(self):
        positive_fields = (
            "time_step",
            "mass",
            "position_error_scale",
            "velocity_error_scale",
            "energy_error_scale",
            "momentum_error_scale",
            "penetration_error_scale",
            "epsilon",
        )

        for name in positive_fields:
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if (
            not math.isfinite(self.gravity)
            or self.gravity < 0
        ):
            raise ValueError(
                "gravity must be nonnegative"
            )

        if not 0 <= self.restitution <= 1:
            raise ValueError(
                "restitution must be in [0, 1]"
            )

        if (
            self.collision_height_tolerance < 0
            or self.collision_velocity_tolerance < 0
        ):
            raise ValueError(
                "Collision tolerances must be nonnegative"
            )

        if self.max_samples < 1 or self.max_horizon < 1:
            raise ValueError(
                "Evaluation limits must be positive"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        allowed = {
            "kinematic_consistency",
            "dynamics_consistency",
            "gravity_consistency",
            "energy_consistency",
            "momentum_consistency",
            "ground_consistency",
            "collision_consistency",
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
                "Invalid physical metric weights"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class PhysicalConsistencyDataset:
    """
    states:
        (N, state_dim)

    actions:
        (N, action_dim)

    Optional:
        reference_next_states:
            (N, state_dim)

        collision_mask:
            (N,) boolean

        isolated_mask:
            (N,) boolean
            True means no external force or impulse.

        trajectory_actions:
            (N, H, action_dim)
    """

    states: np.ndarray
    actions: np.ndarray

    reference_next_states: np.ndarray | None = None

    collision_mask: np.ndarray | None = None
    isolated_mask: np.ndarray | None = None

    trajectory_actions: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        state_dim: int,
        action_dim: int,
        *,
        max_samples: int,
        max_horizon: int,
    ) -> None:
        try:
            states = np.asarray(
                self.states,
                dtype=np.float64,
            )

            actions = np.asarray(
                self.actions,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise PhysicalValidationError(
                "Dataset arrays must be numeric"
            ) from exc

        if states.ndim != 2:
            raise PhysicalValidationError(
                "states must be 2D"
            )

        n = states.shape[0]

        if not 1 <= n <= max_samples:
            raise PhysicalValidationError(
                "Invalid sample count"
            )

        if states.shape != (n, state_dim):
            raise PhysicalValidationError(
                "Invalid state dimensions"
            )

        if actions.shape != (n, action_dim):
            raise PhysicalValidationError(
                "Invalid action dimensions"
            )

        if not (
            np.all(np.isfinite(states))
            and np.all(np.isfinite(actions))
        ):
            raise PhysicalValidationError(
                "Dataset contains non-finite values"
            )

        self.states = states.copy()
        self.actions = actions.copy()

        if self.reference_next_states is not None:
            targets = np.asarray(
                self.reference_next_states,
                dtype=np.float64,
            )

            if (
                targets.shape != (n, state_dim)
                or not np.all(np.isfinite(targets))
            ):
                raise PhysicalValidationError(
                    "Invalid reference_next_states"
                )

            self.reference_next_states = targets.copy()

        for name in (
            "collision_mask",
            "isolated_mask",
        ):
            values = getattr(self, name)

            if values is not None:
                mask = np.asarray(values)

                if (
                    mask.shape != (n,)
                    or not np.issubdtype(
                        mask.dtype,
                        np.bool_,
                    )
                ):
                    raise PhysicalValidationError(
                        f"{name} must be boolean with shape ({n},)"
                    )

                setattr(
                    self,
                    name,
                    mask.copy(),
                )

        if self.trajectory_actions is not None:
            trajectory = np.asarray(
                self.trajectory_actions,
                dtype=np.float64,
            )

            if trajectory.ndim != 3:
                raise PhysicalValidationError(
                    "trajectory_actions must be 3D"
                )

            horizon = trajectory.shape[1]

            if (
                not 1 <= horizon <= max_horizon
                or trajectory.shape != (
                    n,
                    horizon,
                    action_dim,
                )
                or not np.all(np.isfinite(trajectory))
            ):
                raise PhysicalValidationError(
                    "Invalid trajectory_actions"
                )

            self.trajectory_actions = trajectory.copy()


# ============================================================
# RESULTS
# ============================================================

@dataclass
class PhysicalConsistencyMetrics:
    kinematic_consistency: float
    dynamics_consistency: float

    gravity_consistency: float | None
    energy_consistency: float | None

    momentum_consistency: float | None
    ground_consistency: float

    collision_consistency: float | None

    overall_score: float


@dataclass
class PhysicalConsistencyResult:
    evaluation_id: str
    model_name: str

    sample_count: int

    metrics: PhysicalConsistencyMetrics

    passed: bool
    created_at: str

    raw_metrics: dict[str, float]

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPERS
# ============================================================

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


def mean_or_none(
    values: np.ndarray,
) -> float | None:
    if values.size == 0:
        return None

    return float(
        np.mean(values)
    )


# ============================================================
# EVALUATOR
# ============================================================

class PhysicalConsistencyEvaluator:
    """
    Evaluate predictions against configured
    two-dimensional physical constraints.

    The model must predict state transitions.

    The evaluator does not control hardware.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: PhysicalConsistencyConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or PhysicalConsistencyConfig()
        )

        self.config.layout.validate(
            model.state_dim,
            model.action_dim,
        )

        self._results: dict[
            str,
            PhysicalConsistencyResult,
        ] = {}

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    def _predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        try:
            transition = (
                self.model.predict_transition(
                    state.copy(),
                    action.copy(),
                )
            )

            next_state = np.asarray(
                transition.next_state,
                dtype=np.float64,
            )

        except Exception as exc:
            raise PhysicalModelError(
                f"Prediction failed: {exc}"
            ) from exc

        if next_state.shape != (
            self.model.state_dim,
        ):
            raise PhysicalModelError(
                "Invalid predicted state dimension"
            )

        if not np.all(np.isfinite(next_state)):
            raise PhysicalModelError(
                "Prediction contains non-finite values"
            )

        return next_state

    # --------------------------------------------------------
    # EXTRACT PHYSICAL VARIABLES
    # --------------------------------------------------------

    def _extract(
        self,
        states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        layout = self.config.layout

        positions = states[
            :,
            [
                layout.position_x,
                layout.position_y,
            ],
        ]

        velocities = states[
            :,
            [
                layout.velocity_x,
                layout.velocity_y,
            ],
        ]

        return positions, velocities

    def _extract_actions(
        self,
        actions: np.ndarray,
    ) -> np.ndarray:
        layout = self.config.layout

        return actions[
            :,
            [
                layout.action_x,
                layout.action_y,
            ],
        ]

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    def _predict_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        return np.stack(
            [
                self._predict(state, action)
                for state, action in zip(
                    states,
                    actions,
                )
            ]
        )

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: PhysicalConsistencyDataset,
    ) -> PhysicalConsistencyResult:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        predicted_states = self._predict_batch(
            dataset.states,
            dataset.actions,
        )

        positions, velocities = self._extract(
            dataset.states
        )

        next_positions, next_velocities = self._extract(
            predicted_states
        )

        actions = self._extract_actions(
            dataset.actions
        )

        dt = self.config.time_step
        gravity_vector = np.array(
            [0.0, -self.config.gravity],
            dtype=np.float64,
        )

        acceleration = (
            actions + gravity_vector
        )

        # ----------------------------------------------------
        # Kinematics: p' = p + v*dt + 0.5*a*dt^2
        # ----------------------------------------------------

        expected_positions = (
            positions
            + velocities * dt
            + 0.5 * acceleration * dt * dt
        )

        kinematic_errors = np.linalg.norm(
            next_positions - expected_positions,
            axis=1,
        )

        kinematic_error = float(
            np.mean(kinematic_errors)
        )

        kinematic_score = error_to_score(
            kinematic_error,
            self.config.position_error_scale,
        )

        # ----------------------------------------------------
        # Dynamics: v' = v + a*dt
        # ----------------------------------------------------

        expected_velocities = (
            velocities + acceleration * dt
        )

        velocity_errors = np.linalg.norm(
            next_velocities - expected_velocities,
            axis=1,
        )

        dynamics_error = float(
            np.mean(velocity_errors)
        )

        dynamics_score = error_to_score(
            dynamics_error,
            self.config.velocity_error_scale,
        )

        # ----------------------------------------------------
        # Gravity consistency
        # ----------------------------------------------------

        # Gravity consistency is meaningful only
        # for samples without contact impulses.
        if dataset.collision_mask is not None:
            free_flight_mask = (
                ~dataset.collision_mask
            )
        else:
            free_flight_mask = np.ones(
                len(dataset.states),
                dtype=bool,
            )

        gravity_score = None
        gravity_error = None

        if np.any(free_flight_mask):
            expected_vy = (
                velocities[:, 1]
                + (
                    actions[:, 1]
                    - self.config.gravity
                ) * dt
            )

            errors = np.abs(
                next_velocities[:, 1]
                - expected_vy
            )

            gravity_error = float(
                np.mean(
                    errors[free_flight_mask]
                )
            )

            gravity_score = error_to_score(
                gravity_error,
                self.config.velocity_error_scale,
            )

        # ----------------------------------------------------
        # Mechanical energy conservation
        # ----------------------------------------------------

        # Valid only when no non-conservative
        # external work or collision occurs.
        energy_score = None
        energy_error = None

        if dataset.isolated_mask is not None:
            isolated = dataset.isolated_mask.copy()

            if dataset.collision_mask is not None:
                isolated &= (
                    ~dataset.collision_mask
                )

            # Actions represent external acceleration,
            # so conservation applies only when actions
            # are approximately zero.
            isolated &= (
                np.linalg.norm(actions, axis=1)
                <= self.config.epsilon
            )

            if np.any(isolated):
                initial_kinetic = (
                    0.5
                    * self.config.mass
                    * np.sum(
                        velocities ** 2,
                        axis=1,
                    )
                )

                final_kinetic = (
                    0.5
                    * self.config.mass
                    * np.sum(
                        next_velocities ** 2,
                        axis=1,
                    )
                )

                initial_potential = (
                    self.config.mass
                    * self.config.gravity
                    * positions[:, 1]
                )

                final_potential = (
                    self.config.mass
                    * self.config.gravity
                    * next_positions[:, 1]
                )

                energy_differences = np.abs(
                    (
                        final_kinetic
                        + final_potential
                    )
                    - (
                        initial_kinetic
                        + initial_potential
                    )
                )

                energy_error = float(
                    np.mean(
                        energy_differences[isolated]
                    )
                )

                energy_score = error_to_score(
                    energy_error,
                    self.config.energy_error_scale,
                )

        # ----------------------------------------------------
        # Momentum conservation
        # ----------------------------------------------------

        # Momentum conservation is only evaluated
        # when gravity is zero and external actions
        # are approximately zero.
        momentum_score = None
        momentum_error = None

        if (
            dataset.isolated_mask is not None
            and self.config.gravity <= self.config.epsilon
        ):
            isolated = dataset.isolated_mask.copy()

            isolated &= (
                np.linalg.norm(actions, axis=1)
                <= self.config.epsilon
            )

            if dataset.collision_mask is not None:
                isolated &= (
                    ~dataset.collision_mask
                )

            if np.any(isolated):
                initial_momentum = (
                    self.config.mass
                    * velocities
                )

                final_momentum = (
                    self.config.mass
                    * next_velocities
                )

                momentum_differences = np.linalg.norm(
                    final_momentum - initial_momentum,
                    axis=1,
                )

                momentum_error = float(
                    np.mean(
                        momentum_differences[isolated]
                    )
                )

                momentum_score = error_to_score(
                    momentum_error,
                    self.config.momentum_error_scale,
                )

        # ----------------------------------------------------
        # Ground penetration
        # ----------------------------------------------------

        penetration = np.maximum(
            self.config.ground_height
            - next_positions[:, 1],
            0.0,
        )

        penetration_error = float(
            np.mean(penetration)
        )

        ground_score = error_to_score(
            penetration_error,
            self.config.penetration_error_scale,
        )

        # ----------------------------------------------------
        # Collision restitution
        # ----------------------------------------------------

        collision_score = None
        collision_error = None

        if (
            dataset.collision_mask is not None
            and np.any(dataset.collision_mask)
        ):
            collision_indices = np.flatnonzero(
                dataset.collision_mask
            )

            errors = []

            for index in collision_indices:
                incoming_vy = velocities[index, 1]
                outgoing_vy = next_velocities[index, 1]

                if (
                    incoming_vy
                    < -self.config.collision_velocity_tolerance
                ):
                    # Simplified instantaneous ground
                    # collision assumption.
                    expected_outgoing = (
                        -self.config.restitution
                        * incoming_vy
                    )

                    errors.append(
                        abs(
                            outgoing_vy
                            - expected_outgoing
                        )
                    )

            if errors:
                collision_error = float(
                    np.mean(errors)
                )

                collision_score = error_to_score(
                    collision_error,
                    self.config.velocity_error_scale,
                )

        # ----------------------------------------------------
        # Weighted score
        # ----------------------------------------------------

        scores: dict[str, float | None] = {
            "kinematic_consistency": kinematic_score,
            "dynamics_consistency": dynamics_score,
            "gravity_consistency": gravity_score,
            "energy_consistency": energy_score,
            "momentum_consistency": momentum_score,
            "ground_consistency": ground_score,
            "collision_consistency": collision_score,
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
            raise PhysicalValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = PhysicalConsistencyMetrics(
            kinematic_consistency=kinematic_score,
            dynamics_consistency=dynamics_score,
            gravity_consistency=gravity_score,
            energy_consistency=energy_score,
            momentum_consistency=momentum_score,
            ground_consistency=ground_score,
            collision_consistency=collision_score,
            overall_score=float(overall_score),
        )

        raw_metrics = {
            "kinematic_mean_error": kinematic_error,
            "dynamics_mean_error": dynamics_error,
            "ground_penetration_mean": penetration_error,
        }

        optional_errors = {
            "gravity_mean_error": gravity_error,
            "energy_mean_error": energy_error,
            "momentum_mean_error": momentum_error,
            "collision_mean_error": collision_error,
        }

        raw_metrics.update(
            {
                name: value
                for name, value in optional_errors.items()
                if value is not None
            }
        )

        if dataset.reference_next_states is not None:
            reference_rmse = float(
                np.sqrt(
                    np.mean(
                        (
                            predicted_states
                            - dataset.reference_next_states
                        ) ** 2
                    )
                )
            )

            raw_metrics[
                "reference_prediction_rmse"
            ] = reference_rmse

        result = PhysicalConsistencyResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            sample_count=len(dataset.states),
            metrics=metrics,
            passed=(
                overall_score
                >= self.config.pass_threshold
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            raw_metrics=raw_metrics,
            details={
                "weights": active_weights,
                "time_step": dt,
                "gravity": self.config.gravity,
                "mass": self.config.mass,
                "restitution": self.config.restitution,
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "interpretation": (
                    "Agreement with configured 2D "
                    "physics assumptions"
                ),
                "limitations": (
                    "Collision samples require a separate "
                    "impulse-aware dynamics assessment; "
                    "simple free-flight kinematics do not "
                    "apply during impact"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Physical consistency evaluated: "
            "model=%s score=%.4f passed=%s",
            self.model.name,
            overall_score,
            result.passed,
        )

        return result

    # --------------------------------------------------------
    # MULTI-STEP TRAJECTORY EVALUATION
    # --------------------------------------------------------

    def evaluate_trajectory(
        self,
        dataset: PhysicalConsistencyDataset,
    ) -> list[PhysicalConsistencyResult]:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        if dataset.trajectory_actions is None:
            raise PhysicalValidationError(
                "trajectory_actions are required"
            )

        current_states = dataset.states.copy()
        results = []

        horizon = (
            dataset.trajectory_actions.shape[1]
        )

        for step in range(horizon):
            step_actions = (
                dataset.trajectory_actions[:, step, :]
            )

            step_dataset = PhysicalConsistencyDataset(
                states=current_states,
                actions=step_actions,
                metadata={
                    **dataset.metadata,
                    "horizon_step": step + 1,
                },
            )

            result = self.evaluate(
                step_dataset
            )

            results.append(result)

            current_states = self._predict_batch(
                current_states,
                step_actions,
            )

        return results

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> PhysicalConsistencyResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise PhysicalConsistencyError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[PhysicalConsistencyResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[PhysicalConsistencyResult],
    ) -> list[PhysicalConsistencyResult]:
        return sorted(
            results,
            key=lambda item: (
                item.metrics.overall_score
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
        results: list[PhysicalConsistencyResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise PhysicalValidationError(
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
                "model_name": result.model_name,
                "sample_count": result.sample_count,
                "passed": result.passed,
                "created_at": result.created_at,
            }

            row.update(
                asdict(result.metrics)
            )

            row.update(
                result.raw_metrics
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

def create_physical_consistency_evaluator(
    model: WorldModelProtocol,
    *,
    config: PhysicalConsistencyConfig | None = None,
) -> PhysicalConsistencyEvaluator:
    return PhysicalConsistencyEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# DEMONSTRATION MODEL
# ============================================================

class DemoPhysicsWorldModel:
    """
    Synthetic ballistic world model.

    State:
        [x, y, vx, vy]

    Action:
        [ax, ay]

    Uses constant-acceleration integration.

    This is not a trained neural world model.
    """

    name = "demo_physics_world_model"
    state_dim = 4
    action_dim = 2

    def __init__(
        self,
        *,
        dt: float = 0.1,
        gravity: float = 9.81,
    ) -> None:
        self.dt = dt
        self.gravity = gravity

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        from types import SimpleNamespace

        x, y, vx, vy = state
        ax, ay = action

        acceleration_y = ay - self.gravity

        next_x = (
            x + vx * self.dt
            + 0.5 * ax * self.dt ** 2
        )

        next_y = (
            y + vy * self.dt
            + 0.5 * acceleration_y * self.dt ** 2
        )

        next_vx = vx + ax * self.dt
        next_vy = vy + acceleration_y * self.dt

        return SimpleNamespace(
            next_state=np.array(
                [
                    next_x,
                    next_y,
                    next_vx,
                    next_vy,
                ],
                dtype=np.float64,
            ),
            uncertainty=0.05,
            terminal=False,
            reward=0.0,
        )


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_physics_dataset(
    *,
    sample_count: int = 128,
    seed: int = 42,
) -> PhysicalConsistencyDataset:
    if sample_count < 1:
        raise ValueError(
            "sample_count must be positive"
        )

    rng = np.random.default_rng(seed)

    states = np.column_stack(
        [
            rng.uniform(
                -5.0, 5.0, sample_count
            ),
            rng.uniform(
                10.0, 30.0, sample_count
            ),
            rng.uniform(
                -2.0, 2.0, sample_count
            ),
            rng.uniform(
                -2.0, 2.0, sample_count
            ),
        ]
    )

    actions = np.zeros(
        (sample_count, 2),
        dtype=np.float64,
    )

    return PhysicalConsistencyDataset(
        states=states,
        actions=actions,
        isolated_mask=np.ones(
            sample_count,
            dtype=bool,
        ),
        collision_mask=np.zeros(
            sample_count,
            dtype=bool,
        ),
        metadata={
            "source": "synthetic_ballistic_environment",
            "seed": seed,
        },
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    model = DemoPhysicsWorldModel()

    dataset = (
        generate_synthetic_physics_dataset(
            sample_count=128,
            seed=42,
        )
    )

    evaluator = (
        create_physical_consistency_evaluator(
            model
        )
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — PHYSICAL CONSISTENCY")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/physical_consistency.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/physical_consistency.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
