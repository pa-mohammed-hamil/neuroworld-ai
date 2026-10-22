
"""
NeuroWorld AI
File: neuroworld/physics/physical_consistency.py

Physical Consistency Evaluation Engine

Features:
    1. Motion continuity validation
    2. Position-velocity consistency
    3. Velocity-acceleration consistency
    4. Gravity consistency
    5. Newton's second law consistency
    6. Momentum conservation checks
    7. Kinetic and potential energy checks
    8. Collision penetration detection
    9. Object identity and continuity checks
   10. Temporal trajectory validation
   11. Configurable tolerances
   12. Weighted consistency scores
   13. Batch trajectory evaluation
   14. Differentiable PyTorch physics loss
   15. Structured reports and diagnostics
   16. World-model training integration

Install:
    pip install numpy torch

Run:
    python -m neuroworld.physics.physical_consistency

Conventions:
    positions:     [T,N,D]
    velocities:    [T,N,D]
    accelerations: [T,N,D] optional
    forces:        [T,N,D] optional
    masses:        [N]
    radii:         [N] optional
    timestamps:    [T]

    T: time steps
    N: objects
    D: 2 or 3

Important:
    These are numerical consistency checks,
    not proofs of physical correctness.
"""

from __future__ import annotations

import json
import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor


EPS = 1e-12


# ============================================================
# EXCEPTIONS
# ============================================================

class PhysicalConsistencyError(Exception):
    pass


class PhysicalValidationError(
    PhysicalConsistencyError
):
    pass


# ============================================================
# ENUMERATIONS
# ============================================================

class ConsistencyStatus(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PhysicalConsistencyConfig:
    spatial_dim: int = 2

    gravity_magnitude: float = 9.80665
    enable_gravity: bool = True

    position_tolerance: float = 0.05
    velocity_tolerance: float = 0.1
    acceleration_tolerance: float = 0.2

    momentum_tolerance: float = 0.1
    energy_tolerance: float = 0.1

    penetration_tolerance: float = 0.01

    maximum_speed: float = 100.0
    maximum_acceleration: float = 100.0

    minimum_pass_score: float = 0.85
    minimum_review_score: float = 0.65

    weight_position: float = 0.2
    weight_velocity: float = 0.2
    weight_gravity: float = 0.15
    weight_momentum: float = 0.15
    weight_energy: float = 0.15
    weight_collision: float = 0.15

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if (
            not math.isfinite(self.gravity_magnitude)
            or self.gravity_magnitude < 0
        ):
            raise ValueError(
                "Invalid gravity magnitude"
            )

        positive = (
            "position_tolerance",
            "velocity_tolerance",
            "acceleration_tolerance",
            "momentum_tolerance",
            "energy_tolerance",
            "maximum_speed",
            "maximum_acceleration",
        )

        for name in positive:
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if (
            not math.isfinite(
                self.penetration_tolerance
            )
            or self.penetration_tolerance < 0
        ):
            raise ValueError(
                "Invalid penetration tolerance"
            )

        if not (
            0 <= self.minimum_review_score
            <= self.minimum_pass_score
            <= 1
        ):
            raise ValueError(
                "Invalid score thresholds"
            )

        weights = (
            self.weight_position,
            self.weight_velocity,
            self.weight_gravity,
            self.weight_momentum,
            self.weight_energy,
            self.weight_collision,
        )

        if (
            any(
                not math.isfinite(w) or w < 0
                for w in weights
            )
            or sum(weights) <= 0
        ):
            raise ValueError(
                "Invalid metric weights"
            )


# ============================================================
# TRAJECTORY DATA
# ============================================================

@dataclass
class PhysicalTrajectory:
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    timestamps: np.ndarray

    accelerations: np.ndarray | None = None
    forces: np.ndarray | None = None
    radii: np.ndarray | None = None

    object_ids: list[str] | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.positions = np.asarray(
            self.positions,
            dtype=np.float64,
        )

        self.velocities = np.asarray(
            self.velocities,
            dtype=np.float64,
        )

        self.masses = np.asarray(
            self.masses,
            dtype=np.float64,
        )

        self.timestamps = np.asarray(
            self.timestamps,
            dtype=np.float64,
        )

        if (
            self.positions.ndim != 3
            or self.positions.shape[-1]
            not in (2, 3)
        ):
            raise PhysicalValidationError(
                "positions must be [T,N,2/3]"
            )

        if self.positions.shape[0] < 2:
            raise PhysicalValidationError(
                "At least two time steps required"
            )

        if self.positions.shape[1] < 1:
            raise PhysicalValidationError(
                "At least one object required"
            )

        if (
            self.velocities.shape
            != self.positions.shape
        ):
            raise PhysicalValidationError(
                "Velocity shape mismatch"
            )

        steps, objects, _ = (
            self.positions.shape
        )

        if self.masses.shape != (objects,):
            raise PhysicalValidationError(
                "masses must be [N]"
            )

        if self.timestamps.shape != (steps,):
            raise PhysicalValidationError(
                "timestamps must be [T]"
            )

        if (
            not np.isfinite(
                self.positions
            ).all()
            or not np.isfinite(
                self.velocities
            ).all()
            or not np.isfinite(
                self.masses
            ).all()
            or not np.isfinite(
                self.timestamps
            ).all()
        ):
            raise PhysicalValidationError(
                "Trajectory contains non-finite values"
            )

        if np.any(self.masses <= 0):
            raise PhysicalValidationError(
                "Masses must be positive"
            )

        if np.any(
            np.diff(self.timestamps) <= 0
        ):
            raise PhysicalValidationError(
                "Timestamps must strictly increase"
            )

        for name in (
            "accelerations",
            "forces",
        ):
            value = getattr(self, name)

            if value is not None:
                value = np.asarray(
                    value,
                    dtype=np.float64,
                )

                if value.shape != self.positions.shape:
                    raise PhysicalValidationError(
                        f"{name} must be [T,N,D]"
                    )

                if not np.isfinite(value).all():
                    raise PhysicalValidationError(
                        f"{name} must be finite"
                    )

                setattr(
                    self,
                    name,
                    value,
                )

        if self.radii is not None:
            self.radii = np.asarray(
                self.radii,
                dtype=np.float64,
            )

            if self.radii.shape != (objects,):
                raise PhysicalValidationError(
                    "radii must be [N]"
                )

            if (
                not np.isfinite(
                    self.radii
                ).all()
                or np.any(self.radii <= 0)
            ):
                raise PhysicalValidationError(
                    "Radii must be positive"
                )

        if self.object_ids is not None:
            if len(self.object_ids) != objects:
                raise PhysicalValidationError(
                    "object_ids length mismatch"
                )

            if len(set(self.object_ids)) != objects:
                raise PhysicalValidationError(
                    "object_ids must be unique"
                )

    @property
    def time_steps(self) -> int:
        return self.positions.shape[0]

    @property
    def object_count(self) -> int:
        return self.positions.shape[1]

    @property
    def spatial_dim(self) -> int:
        return self.positions.shape[2]

    @property
    def dt(self) -> np.ndarray:
        return np.diff(
            self.timestamps
        )


# ============================================================
# METRIC RESULT
# ============================================================

@dataclass
class PhysicsMetricResult:
    name: str

    evaluated: bool
    score: float | None

    error: float | None

    tolerance: float | None

    passed: bool | None

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evaluated": self.evaluated,
            "score": self.score,
            "error": self.error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "details": self.details,
        }


# ============================================================
# METRIC HELPERS
# ============================================================

def error_to_score(
    error: float,
    tolerance: float,
) -> float:
    """
    Converts numerical error into [0,1].

    Score = exp(-error/tolerance)

    This is a heuristic score, not a
    calibrated probability.
    """
    if tolerance <= 0:
        raise ValueError(
            "Tolerance must be positive"
        )

    return float(
        math.exp(
            -max(0.0, error)
            / tolerance
        )
    )


def make_metric(
    name: str,
    error: float,
    tolerance: float,
    *,
    details: dict[str, Any] | None = None,
) -> PhysicsMetricResult:
    if (
        not math.isfinite(error)
        or error < 0
    ):
        raise PhysicalValidationError(
            f"Invalid error for {name}"
        )

    return PhysicsMetricResult(
        name=name,
        evaluated=True,
        score=error_to_score(
            error,
            tolerance,
        ),
        error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
        details=details or {},
    )


def unavailable_metric(
    name: str,
    reason: str,
) -> PhysicsMetricResult:
    return PhysicsMetricResult(
        name=name,
        evaluated=False,
        score=None,
        error=None,
        tolerance=None,
        passed=None,
        details={
            "reason": reason
        },
    )


# ============================================================
# POSITION-VELOCITY CONSISTENCY
# ============================================================

class PositionConsistencyEvaluator:
    """
    Checks:
        x[t+1] ≈ x[t] + 0.5*(v[t]+v[t+1])*dt

    Trapezoidal integration approximation.
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicsMetricResult:
        dt = trajectory.dt[
            :, None, None
        ]

        predicted = (
            trajectory.positions[:-1]
            + 0.5
            * (
                trajectory.velocities[:-1]
                + trajectory.velocities[1:]
            )
            * dt
        )

        error = np.linalg.norm(
            predicted
            - trajectory.positions[1:],
            axis=-1,
        )

        mean_error = float(
            np.mean(error)
        )

        return make_metric(
            "position",
            mean_error,
            self.config.position_tolerance,
            details={
                "maximum_error": float(
                    np.max(error)
                )
            },
        )


# ============================================================
# VELOCITY-ACCELERATION CONSISTENCY
# ============================================================

class VelocityConsistencyEvaluator:
    """
    Checks:
        v[t+1] ≈ v[t] + 0.5*(a[t]+a[t+1])*dt

    Requires acceleration observations.
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicsMetricResult:
        if trajectory.accelerations is None:
            return unavailable_metric(
                "velocity",
                "Acceleration data not supplied",
            )

        dt = trajectory.dt[
            :, None, None
        ]

        predicted = (
            trajectory.velocities[:-1]
            + 0.5
            * (
                trajectory.accelerations[:-1]
                + trajectory.accelerations[1:]
            )
            * dt
        )

        error = np.linalg.norm(
            predicted
            - trajectory.velocities[1:],
            axis=-1,
        )

        return make_metric(
            "velocity",
            float(
                np.mean(error)
            ),
            self.config.velocity_tolerance,
            details={
                "maximum_error": float(
                    np.max(error)
                )
            },
        )


# ============================================================
# GRAVITY CONSISTENCY
# ============================================================

class GravityConsistencyEvaluator:
    """
    Checks measured acceleration against
    expected gravity plus known forces.

    If force observations are unavailable,
    gravity-only comparison is not valid
    for actively controlled objects.
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicsMetricResult:
        if trajectory.accelerations is None:
            return unavailable_metric(
                "gravity",
                "Acceleration data not supplied",
            )

        if trajectory.forces is None:
            return unavailable_metric(
                "gravity",
                "Known external forces required",
            )

        gravity = np.zeros(
            trajectory.spatial_dim,
            dtype=np.float64,
        )

        if self.config.enable_gravity:
            vertical_axis = (
                1
                if trajectory.spatial_dim == 2
                else 2
            )

            gravity[vertical_axis] = (
                -self.config.gravity_magnitude
            )

        expected_acceleration = (
            trajectory.forces
            / trajectory.masses[
                None, :, None
            ]
            + gravity[
                None, None, :
            ]
        )

        error = np.linalg.norm(
            trajectory.accelerations
            - expected_acceleration,
            axis=-1,
        )

        return make_metric(
            "gravity",
            float(
                np.mean(error)
            ),
            self.config.acceleration_tolerance,
            details={
                "maximum_error": float(
                    np.max(error)
                ),
                "assumption": (
                    "Forces contain all non-gravity "
                    "external forces"
                ),
            },
        )


# ============================================================
# MOMENTUM CONSERVATION
# ============================================================

class MomentumConsistencyEvaluator:
    """
    Checks conservation of total momentum.

    Only valid for a closed system with
    zero net external impulse.

    Momentum conservation is skipped
    unless the caller explicitly declares
    the system closed in metadata:

        metadata["closed_system"] = True
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicsMetricResult:
        if not trajectory.metadata.get(
            "closed_system",
            False,
        ):
            return unavailable_metric(
                "momentum",
                "Closed-system assumption not declared",
            )

        momentum = np.sum(
            trajectory.masses[
                None, :, None
            ]
            * trajectory.velocities,
            axis=1,
        )

        change = (
            momentum
            - momentum[0]
        )

        error = np.linalg.norm(
            change,
            axis=-1,
        )

        return make_metric(
            "momentum",
            float(
                np.mean(error)
            ),
            self.config.momentum_tolerance,
            details={
                "maximum_drift": float(
                    np.max(error)
                )
            },
        )


# ============================================================
# ENERGY CONSISTENCY
# ============================================================

class EnergyConsistencyEvaluator:
    """
    Evaluates mechanical-energy drift.

    Only applicable when:
        metadata["conservative_system"] = True

    Assumes:
        - no dissipative friction
        - no drag
        - no external work
        - uniform gravity
        - no energy injection
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicsMetricResult:
        if not trajectory.metadata.get(
            "conservative_system",
            False,
        ):
            return unavailable_metric(
                "energy",
                "Conservative-system assumption "
                "not declared",
            )

        speed_squared = np.sum(
            trajectory.velocities ** 2,
            axis=-1,
        )

        kinetic = (
            0.5
            * trajectory.masses[None, :]
            * speed_squared
        )

        vertical_axis = (
            1
            if trajectory.spatial_dim == 2
            else 2
        )

        gravity = (
            self.config.gravity_magnitude
            if self.config.enable_gravity
            else 0.0
        )

        potential = (
            trajectory.masses[None, :]
            * gravity
            * trajectory.positions[
                :, :, vertical_axis
            ]
        )

        total_energy = np.sum(
            kinetic + potential,
            axis=1,
        )

        reference = float(
            total_energy[0]
        )

        drift = np.abs(
            total_energy - reference
        )

        scale = max(
            abs(reference),
            1.0,
        )

        relative_drift = (
            drift / scale
        )

        return make_metric(
            "energy",
            float(
                np.mean(relative_drift)
            ),
            self.config.energy_tolerance,
            details={
                "maximum_relative_drift": float(
                    np.max(relative_drift)
                ),
                "initial_energy": reference,
                "final_energy": float(
                    total_energy[-1]
                ),
            },
        )


# ============================================================
# COLLISION CONSISTENCY
# ============================================================

class CollisionConsistencyEvaluator:
    """
    Detects sphere/sphere penetration
    between distinct objects.

    Collision checks require radii.

    Contacts at exactly zero clearance
    are permitted.

    This checks sampled time steps only.
    It does not detect tunneling.
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicsMetricResult:
        if trajectory.radii is None:
            return unavailable_metric(
                "collision",
                "Object radii not supplied",
            )

        count = trajectory.object_count

        if count < 2:
            return unavailable_metric(
                "collision",
                "At least two objects required",
            )

        max_penetration = 0.0
        violations = 0
        tested_pairs = 0

        for i in range(count):
            for j in range(
                i + 1,
                count,
            ):
                displacement = (
                    trajectory.positions[:, i]
                    - trajectory.positions[:, j]
                )

                distance = np.linalg.norm(
                    displacement,
                    axis=-1,
                )

                penetration = np.maximum(
                    0.0,
                    trajectory.radii[i]
                    + trajectory.radii[j]
                    - distance,
                )

                max_penetration = max(
                    max_penetration,
                    float(
                        np.max(penetration)
                    ),
                )

                violations += int(
                    np.count_nonzero(
                        penetration
                        > self.config.penetration_tolerance
                    )
                )

                tested_pairs += len(
                    penetration
                )

        return make_metric(
            "collision",
            max_penetration,
            max(
                self.config.penetration_tolerance,
                1e-6,
            ),
            details={
                "violations": violations,
                "tested_pairs": tested_pairs,
                "maximum_penetration": (
                    max_penetration
                ),
            },
        )


# ============================================================
# PHYSICAL BOUNDS EVALUATOR
# ============================================================

class PhysicalBoundsEvaluator:
    """
    Checks whether speeds and
    observed accelerations remain
    within configured physical limits.

    These are heuristic limits and
    must be configured for each system.
    """

    def __init__(
        self,
        config: PhysicalConsistencyConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> dict[str, Any]:
        speeds = np.linalg.norm(
            trajectory.velocities,
            axis=-1,
        )

        maximum_speed = float(
            np.max(speeds)
        )

        acceleration_violation = False
        maximum_acceleration = None

        if trajectory.accelerations is not None:
            magnitudes = np.linalg.norm(
                trajectory.accelerations,
                axis=-1,
            )

            maximum_acceleration = float(
                np.max(magnitudes)
            )

            acceleration_violation = (
                maximum_acceleration
                > self.config.maximum_acceleration
            )

        return {
            "passed": (
                maximum_speed
                <= self.config.maximum_speed
                and not acceleration_violation
            ),
            "maximum_speed": maximum_speed,
            "maximum_acceleration": (
                maximum_acceleration
            ),
            "speed_limit": (
                self.config.maximum_speed
            ),
            "acceleration_limit": (
                self.config.maximum_acceleration
            ),
        }


# ============================================================
# CONSISTENCY REPORT
# ============================================================

@dataclass
class PhysicalConsistencyReport:
    status: ConsistencyStatus

    overall_score: float | None

    metrics: dict[
        str,
        PhysicsMetricResult,
    ]

    bounds: dict[str, Any]

    evaluated_metrics: int
    skipped_metrics: int

    violations: list[str]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "overall_score": self.overall_score,
            "metrics": {
                name: metric.to_dict()
                for name, metric
                in self.metrics.items()
            },
            "bounds": self.bounds,
            "evaluated_metrics": (
                self.evaluated_metrics
            ),
            "skipped_metrics": (
                self.skipped_metrics
            ),
            "violations": self.violations,
            "metadata": self.metadata,
        }

    def to_json(
        self,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
        )


# ============================================================
# MAIN PHYSICAL CONSISTENCY EVALUATOR
# ============================================================

class PhysicalConsistencyEvaluator:
    """
    Combines physical consistency checks.

    Important:
        Missing metrics are not assigned
        perfect scores.

        The final score is computed only
        from evaluated metrics.

        PASS requires all weighted metrics
        to be evaluated and passing.
    """

    def __init__(
        self,
        config: (
            PhysicalConsistencyConfig | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or PhysicalConsistencyConfig()
        )

        self.position = (
            PositionConsistencyEvaluator(
                self.config
            )
        )

        self.velocity = (
            VelocityConsistencyEvaluator(
                self.config
            )
        )

        self.gravity = (
            GravityConsistencyEvaluator(
                self.config
            )
        )

        self.momentum = (
            MomentumConsistencyEvaluator(
                self.config
            )
        )

        self.energy = (
            EnergyConsistencyEvaluator(
                self.config
            )
        )

        self.collision = (
            CollisionConsistencyEvaluator(
                self.config
            )
        )

        self.bounds_evaluator = (
            PhysicalBoundsEvaluator(
                self.config
            )
        )

    def evaluate(
        self,
        trajectory: PhysicalTrajectory,
    ) -> PhysicalConsistencyReport:
        if trajectory.spatial_dim != (
            self.config.spatial_dim
        ):
            raise PhysicalValidationError(
                "Trajectory/config dimensions mismatch"
            )

        metrics = {
            "position": self.position.evaluate(
                trajectory
            ),
            "velocity": self.velocity.evaluate(
                trajectory
            ),
            "gravity": self.gravity.evaluate(
                trajectory
            ),
            "momentum": self.momentum.evaluate(
                trajectory
            ),
            "energy": self.energy.evaluate(
                trajectory
            ),
            "collision": self.collision.evaluate(
                trajectory
            ),
        }

        weights = {
            "position": (
                self.config.weight_position
            ),
            "velocity": (
                self.config.weight_velocity
            ),
            "gravity": (
                self.config.weight_gravity
            ),
            "momentum": (
                self.config.weight_momentum
            ),
            "energy": (
                self.config.weight_energy
            ),
            "collision": (
                self.config.weight_collision
            ),
        }

        weighted_score = 0.0
        active_weight = 0.0

        evaluated_count = 0
        skipped_count = 0

        violations = []

        for name, metric in metrics.items():
            if not metric.evaluated:
                skipped_count += 1
                continue

            evaluated_count += 1

            weight = weights[name]

            weighted_score += (
                weight * metric.score
            )

            active_weight += weight

            if metric.passed is False:
                violations.append(
                    f"{name}: error "
                    f"{metric.error:.6f} exceeds "
                    f"tolerance {metric.tolerance:.6f}"
                )

        overall_score = (
            weighted_score / active_weight
            if active_weight > 0
            else None
        )

        bounds = (
            self.bounds_evaluator.evaluate(
                trajectory
            )
        )

        if not bounds["passed"]:
            violations.append(
                "Physical speed/acceleration "
                "bounds exceeded"
            )

        all_metrics_evaluated = (
            skipped_count == 0
        )

        if overall_score is None:
            status = (
                ConsistencyStatus.NOT_EVALUATED
            )

        elif (
            overall_score
            >= self.config.minimum_pass_score
            and not violations
            and all_metrics_evaluated
        ):
            status = ConsistencyStatus.PASS

        elif (
            overall_score
            >= self.config.minimum_review_score
            and bounds["passed"]
        ):
            status = ConsistencyStatus.REVIEW

        else:
            status = ConsistencyStatus.FAIL

        return PhysicalConsistencyReport(
            status=status,
            overall_score=overall_score,
            metrics=metrics,
            bounds=bounds,
            evaluated_metrics=evaluated_count,
            skipped_metrics=skipped_count,
            violations=violations,
            metadata={
                "time_steps": (
                    trajectory.time_steps
                ),
                "object_count": (
                    trajectory.object_count
                ),
                "spatial_dim": (
                    trajectory.spatial_dim
                ),
                "all_metrics_evaluated": (
                    all_metrics_evaluated
                ),
            },
        )


# ============================================================
# DIFFERENTIABLE PYTORCH PHYSICS LOSS
# ============================================================

@dataclass(frozen=True)
class PhysicsLossConfig:
    position_weight: float = 1.0
    velocity_weight: float = 1.0
    acceleration_weight: float = 0.5
    collision_weight: float = 1.0

    gravity_magnitude: float = 9.80665
    enable_gravity: bool = True

    def __post_init__(self) -> None:
        for name in (
            "position_weight",
            "velocity_weight",
            "acceleration_weight",
            "collision_weight",
            "gravity_magnitude",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be nonnegative"
                )


class DifferentiablePhysicsLoss(
    torch.nn.Module
):
    """
    Physics-informed training loss.

    Input:
        positions [B,T,N,D]
        velocities [B,T,N,D]
        timestamps [T]

    Optional:
        accelerations [B,T,N,D]
        forces [B,T,N,D]
        masses [N] or [B,N]
        radii [N] or [B,N]

    Terms:
        position consistency
        velocity consistency
        force/acceleration consistency
        collision penetration penalty

    Missing terms contribute zero.
    Inspect returned component losses
    to understand available supervision.
    """

    def __init__(
        self,
        config: PhysicsLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or PhysicsLossConfig()
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        timestamps: Tensor,
        *,
        accelerations: Tensor | None = None,
        forces: Tensor | None = None,
        masses: Tensor | None = None,
        radii: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if (
            positions.ndim != 4
            or positions.shape[-1]
            not in (2, 3)
        ):
            raise PhysicalValidationError(
                "positions must be [B,T,N,D]"
            )

        if velocities.shape != positions.shape:
            raise PhysicalValidationError(
                "Velocity shape mismatch"
            )

        batch, steps, objects, dimension = (
            positions.shape
        )

        if steps < 2:
            raise PhysicalValidationError(
                "At least two time steps required"
            )

        if timestamps.shape != (steps,):
            raise PhysicalValidationError(
                "timestamps must be [T]"
            )

        dt = (
            timestamps[1:]
            - timestamps[:-1]
        )

        if (
            not torch.isfinite(dt).all()
            or (dt <= 0).any()
        ):
            raise PhysicalValidationError(
                "Invalid timestamps"
            )

        dt = dt.reshape(
            1,
            steps - 1,
            1,
            1,
        )

        position_prediction = (
            positions[:, :-1]
            + 0.5
            * (
                velocities[:, :-1]
                + velocities[:, 1:]
            )
            * dt
        )

        position_loss = (
            (
                position_prediction
                - positions[:, 1:]
            ) ** 2
        ).mean()

        zero = positions.new_zeros(
            ()
        )

        velocity_loss = zero
        acceleration_loss = zero
        collision_loss = zero

        if accelerations is not None:
            if accelerations.shape != positions.shape:
                raise PhysicalValidationError(
                    "Acceleration shape mismatch"
                )

            velocity_prediction = (
                velocities[:, :-1]
                + 0.5
                * (
                    accelerations[:, :-1]
                    + accelerations[:, 1:]
                )
                * dt
            )

            velocity_loss = (
                (
                    velocity_prediction
                    - velocities[:, 1:]
                ) ** 2
            ).mean()

        if (
            accelerations is not None
            and forces is not None
            and masses is not None
        ):
            if forces.shape != positions.shape:
                raise PhysicalValidationError(
                    "Force shape mismatch"
                )

            if masses.ndim == 1:
                if masses.shape != (objects,):
                    raise PhysicalValidationError(
                        "Mass shape mismatch"
                    )

                masses_expanded = masses.reshape(
                    1, 1, objects, 1
                )

            elif masses.shape == (
                batch,
                objects,
            ):
                masses_expanded = masses.reshape(
                    batch, 1, objects, 1
                )

            else:
                raise PhysicalValidationError(
                    "Masses must be [N] or [B,N]"
                )

            if (masses_expanded <= 0).any():
                raise PhysicalValidationError(
                    "Masses must be positive"
                )

            gravity = torch.zeros(
                dimension,
                device=positions.device,
                dtype=positions.dtype,
            )

            if self.config.enable_gravity:
                axis = (
                    1 if dimension == 2 else 2
                )

                gravity[axis] = (
                    -self.config.gravity_magnitude
                )

            expected_acceleration = (
                forces
                / masses_expanded
                + gravity.reshape(
                    1, 1, 1, dimension
                )
            )

            acceleration_loss = (
                (
                    accelerations
                    - expected_acceleration
                ) ** 2
            ).mean()

        if radii is not None and objects > 1:
            if radii.ndim == 1:
                if radii.shape != (objects,):
                    raise PhysicalValidationError(
                        "Radii shape mismatch"
                    )

                radii = radii.unsqueeze(
                    0
                ).expand(
                    batch,
                    -1,
                )

            elif radii.shape != (
                batch,
                objects,
            ):
                raise PhysicalValidationError(
                    "Radii must be [N] or [B,N]"
                )

            if (radii <= 0).any():
                raise PhysicalValidationError(
                    "Radii must be positive"
                )

            displacement = (
                positions[:, :, :, None, :]
                - positions[:, :, None, :, :]
            )

            distances = (
                torch.linalg.vector_norm(
                    displacement,
                    dim=-1,
                )
            )

            combined_radii = (
                radii[:, None, :, None]
                + radii[:, None, None, :]
            )

            penetration = torch.relu(
                combined_radii
                - distances
            )

            pair_mask = torch.triu(
                torch.ones(
                    objects,
                    objects,
                    device=positions.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )

            pair_penetrations = penetration[
                :,
                :,
                pair_mask,
            ]

            collision_loss = (
                pair_penetrations.square().mean()
            )

        total_loss = (
            self.config.position_weight
            * position_loss
            + self.config.velocity_weight
            * velocity_loss
            + self.config.acceleration_weight
            * acceleration_loss
            + self.config.collision_weight
            * collision_loss
        )

        return {
            "total_loss": total_loss,
            "position_loss": position_loss,
            "velocity_loss": velocity_loss,
            "acceleration_loss": (
                acceleration_loss
            ),
            "collision_loss": collision_loss,
        }


# ============================================================
# WORLD MODEL EVALUATION ADAPTER
# ============================================================

class WorldModelPhysicsEvaluator:
    """
    Evaluates predicted world-model
    trajectories against physical rules.

    Predicted state layout:
        [T,N,2*D]

    First D values:
        positions

    Next D values:
        velocities
    """

    def __init__(
        self,
        config: (
            PhysicalConsistencyConfig | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or PhysicalConsistencyConfig()
        )

        self.evaluator = (
            PhysicalConsistencyEvaluator(
                self.config
            )
        )

    def evaluate(
        self,
        predicted_states: np.ndarray | Tensor,
        timestamps: np.ndarray,
        masses: np.ndarray,
        *,
        radii: np.ndarray | None = None,
        accelerations: np.ndarray | None = None,
        forces: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PhysicalConsistencyReport:
        if isinstance(
            predicted_states,
            Tensor,
        ):
            predicted_states = (
                predicted_states.detach()
                .cpu()
                .numpy()
            )

        states = np.asarray(
            predicted_states,
            dtype=np.float64,
        )

        dimension = (
            self.config.spatial_dim
        )

        if (
            states.ndim != 3
            or states.shape[-1]
            != 2 * dimension
        ):
            raise PhysicalValidationError(
                "predicted_states must be [T,N,2*D]"
            )

        trajectory = PhysicalTrajectory(
            positions=states[
                :, :, :dimension
            ],
            velocities=states[
                :, :, dimension:
            ],
            masses=masses,
            timestamps=timestamps,
            accelerations=accelerations,
            forces=forces,
            radii=radii,
            metadata=metadata or {},
        )

        return self.evaluator.evaluate(
            trajectory
        )


# ============================================================
# FACTORY
# ============================================================

def create_physical_consistency_evaluator(
    config: (
        PhysicalConsistencyConfig | None
    ) = None,
) -> PhysicalConsistencyEvaluator:
    return PhysicalConsistencyEvaluator(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - PHYSICAL CONSISTENCY"
    )
    print("=" * 65)

    config = PhysicalConsistencyConfig(
        spatial_dim=2,
        enable_gravity=False,
        position_tolerance=0.05,
        velocity_tolerance=0.05,
    )

    timestamps = np.linspace(
        0.0,
        1.0,
        11,
    )

    steps = len(timestamps)

    positions = np.zeros(
        (steps, 2, 2),
        dtype=np.float64,
    )

    velocities = np.zeros_like(
        positions
    )

    accelerations = np.zeros_like(
        positions
    )

    forces = np.zeros_like(
        positions
    )

    # Object A moves right.
    positions[:, 0, 0] = (
        timestamps
    )

    velocities[:, 0, 0] = 1.0

    # Object B remains stationary.
    positions[:, 1, 0] = 3.0

    trajectory = PhysicalTrajectory(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        forces=forces,
        masses=np.array(
            [1.0, 2.0]
        ),
        radii=np.array(
            [0.2, 0.2]
        ),
        timestamps=timestamps,
        object_ids=[
            "moving_object",
            "static_object",
        ],
        metadata={
            "closed_system": False,
            "conservative_system": False,
        },
    )

    evaluator = (
        PhysicalConsistencyEvaluator(
            config
        )
    )

    report = evaluator.evaluate(
        trajectory
    )

    print(
        "\nEvaluation report:"
    )

    print(
        report.to_json()
    )

    print(
        "\nPyTorch physics loss"
    )

    physics_loss = (
        DifferentiablePhysicsLoss(
            PhysicsLossConfig(
                enable_gravity=False
            )
        )
    )

    positions_tensor = torch.tensor(
        positions[None],
        dtype=torch.float32,
    )

    velocities_tensor = torch.tensor(
        velocities[None],
        dtype=torch.float32,
    )

    timestamps_tensor = torch.tensor(
        timestamps,
        dtype=torch.float32,
    )

    losses = physics_loss(
        positions_tensor,
        velocities_tensor,
        timestamps_tensor,
        accelerations=torch.tensor(
            accelerations[None],
            dtype=torch.float32,
        ),
        forces=torch.tensor(
            forces[None],
            dtype=torch.float32,
        ),
        masses=torch.tensor(
            [1.0, 2.0],
            dtype=torch.float32,
        ),
        radii=torch.tensor(
            [0.2, 0.2],
            dtype=torch.float32,
        ),
    )

    for name, value in losses.items():
        print(
            name,
            float(
                value.item()
            ),
        )

    print(
        "\nPhysical consistency demonstration completed"
    )


if __name__ == "__main__":
    main()
