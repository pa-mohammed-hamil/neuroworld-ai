
"""
NeuroWorld AI
File: neuroworld/physics/collision_model.py

Collision Detection & Prediction Engine

Features:
    1. 2D circle / 3D sphere collisions
    2. Axis-aligned bounding box collisions
    3. Circle/sphere-to-box collisions
    4. Signed separation distance
    5. Contact normal and penetration depth
    6. Moving-object collision prediction
    7. Time-to-collision (TTC)
    8. Continuous swept-sphere detection
    9. Trajectory collision screening
   10. Safety margins and robot footprint
   11. Batch point-clearance evaluation
   12. Planning safety constraints
   13. Collision-risk reports
   14. NumPy / PyTorch integration

Install:
    pip install numpy torch

Run:
    python -m neuroworld.physics.collision_model

Scope:
    This is a geometric prediction module.
    It does not simulate contact forces,
    deformable objects, or collision response.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor


EPS = 1e-9


# ============================================================
# ENUMERATIONS
# ============================================================

class ShapeType(str, Enum):
    SPHERE = "sphere"
    BOX = "box"


class CollisionRisk(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CollisionConfig:
    spatial_dim: int = 2

    safety_margin: float = 0.2
    robot_radius: float = 0.15

    prediction_horizon: float = 5.0
    prediction_timestep: float = 0.1

    critical_ttc: float = 0.5
    high_ttc: float = 1.5
    moderate_ttc: float = 3.0

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        for name in (
            "safety_margin",
            "robot_radius",
            "prediction_horizon",
            "prediction_timestep",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )

        if self.prediction_horizon <= 0:
            raise ValueError(
                "prediction_horizon must be positive"
            )

        if self.prediction_timestep <= 0:
            raise ValueError(
                "prediction_timestep must be positive"
            )

        if not (
            0 < self.critical_ttc
            < self.high_ttc
            < self.moderate_ttc
        ):
            raise ValueError(
                "Invalid TTC thresholds"
            )


# ============================================================
# VECTOR HELPERS
# ============================================================

def as_vector(
    value: Any,
    dimension: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=np.float64,
    )

    if vector.shape != (dimension,):
        raise ValueError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            f"{name} contains non-finite values"
        )

    return vector


def normalized(
    vector: np.ndarray,
) -> np.ndarray:
    length = float(
        np.linalg.norm(vector)
    )

    if length <= EPS:
        output = np.zeros_like(vector)
        output[0] = 1.0
        return output

    return vector / length


# ============================================================
# COLLISION SHAPES
# ============================================================

@dataclass
class Sphere:
    """
    Circle in 2D, sphere in 3D.
    """

    center: np.ndarray | list[float]
    radius: float

    velocity: np.ndarray | list[float] | None = None

    object_id: str = "sphere"

    def __post_init__(self) -> None:
        self.center = np.asarray(
            self.center,
            dtype=np.float64,
        )

        if self.center.shape not in (
            (2,),
            (3,),
        ):
            raise ValueError(
                "Sphere center must be 2D or 3D"
            )

        if not np.isfinite(
            self.center
        ).all():
            raise ValueError(
                "Sphere center must be finite"
            )

        if (
            not math.isfinite(self.radius)
            or self.radius <= 0
        ):
            raise ValueError(
                "Sphere radius must be positive"
            )

        if self.velocity is None:
            self.velocity = np.zeros_like(
                self.center
            )
        else:
            self.velocity = as_vector(
                self.velocity,
                len(self.center),
                "velocity",
            )


@dataclass
class AxisAlignedBox:
    """
    2D rectangle or 3D AABB.

    Boxes do not rotate.
    """

    minimum: np.ndarray | list[float]
    maximum: np.ndarray | list[float]

    object_id: str = "box"

    def __post_init__(self) -> None:
        self.minimum = np.asarray(
            self.minimum,
            dtype=np.float64,
        )

        self.maximum = np.asarray(
            self.maximum,
            dtype=np.float64,
        )

        if (
            self.minimum.shape
            != self.maximum.shape
            or self.minimum.shape
            not in ((2,), (3,))
        ):
            raise ValueError(
                "Box dimensions must match"
            )

        if not (
            np.isfinite(self.minimum).all()
            and np.isfinite(self.maximum).all()
        ):
            raise ValueError(
                "Box coordinates must be finite"
            )

        if np.any(
            self.minimum >= self.maximum
        ):
            raise ValueError(
                "Box minimum must be below maximum"
            )

    @property
    def center(self) -> np.ndarray:
        return (
            self.minimum + self.maximum
        ) / 2.0


CollisionShape = Sphere | AxisAlignedBox


# ============================================================
# COLLISION RESULT
# ============================================================

@dataclass
class CollisionResult:
    object_a: str
    object_b: str

    collided: bool
    signed_distance: float

    penetration_depth: float
    normal: np.ndarray

    contact_point: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_a": self.object_a,
            "object_b": self.object_b,
            "collided": self.collided,
            "signed_distance": (
                self.signed_distance
            ),
            "penetration_depth": (
                self.penetration_depth
            ),
            "normal": self.normal.tolist(),
            "contact_point": (
                self.contact_point.tolist()
            ),
        }


# ============================================================
# SPHERE-SPHERE COLLISION
# ============================================================

def sphere_sphere_collision(
    a: Sphere,
    b: Sphere,
) -> CollisionResult:
    if a.center.shape != b.center.shape:
        raise ValueError(
            "Sphere dimensions must match"
        )

    displacement = (
        b.center - a.center
    )

    center_distance = float(
        np.linalg.norm(displacement)
    )

    signed_distance = (
        center_distance
        - a.radius
        - b.radius
    )

    normal = normalized(
        displacement
    )

    contact_point = (
        a.center
        + normal * a.radius
    )

    return CollisionResult(
        object_a=a.object_id,
        object_b=b.object_id,
        collided=signed_distance <= 0,
        signed_distance=signed_distance,
        penetration_depth=max(
            0.0,
            -signed_distance,
        ),
        normal=normal,
        contact_point=contact_point,
    )


# ============================================================
# BOX-BOX COLLISION
# ============================================================

def box_box_collision(
    a: AxisAlignedBox,
    b: AxisAlignedBox,
) -> CollisionResult:
    if a.minimum.shape != b.minimum.shape:
        raise ValueError(
            "Box dimensions must match"
        )

    # Signed separation on each axis.
    gaps = np.maximum(
        b.minimum - a.maximum,
        a.minimum - b.maximum,
    )

    separated = np.any(
        gaps > 0
    )

    direction = (
        b.center - a.center
    )

    if separated:
        positive_gaps = np.maximum(
            gaps,
            0.0,
        )

        signed_distance = float(
            np.linalg.norm(
                positive_gaps
            )
        )

        normal = np.zeros_like(
            direction
        )

        for axis in range(
            len(direction)
        ):
            if positive_gaps[axis] > 0:
                normal[axis] = (
                    1.0
                    if direction[axis] >= 0
                    else -1.0
                ) * positive_gaps[axis]

        normal = normalized(
            normal
        )

        penetration_depth = 0.0

    else:
        overlap = np.minimum(
            a.maximum,
            b.maximum,
        ) - np.maximum(
            a.minimum,
            b.minimum,
        )

        axis = int(
            np.argmin(overlap)
        )

        penetration_depth = float(
            overlap[axis]
        )

        signed_distance = (
            -penetration_depth
        )

        normal = np.zeros_like(
            direction
        )

        normal[axis] = (
            1.0
            if direction[axis] >= 0
            else -1.0
        )

    contact_point = np.clip(
        b.center,
        a.minimum,
        a.maximum,
    )

    return CollisionResult(
        object_a=a.object_id,
        object_b=b.object_id,
        collided=not separated,
        signed_distance=signed_distance,
        penetration_depth=penetration_depth,
        normal=normal,
        contact_point=contact_point,
    )


# ============================================================
# SPHERE-BOX COLLISION
# ============================================================

def sphere_box_collision(
    sphere: Sphere,
    box: AxisAlignedBox,
) -> CollisionResult:
    if sphere.center.shape != box.minimum.shape:
        raise ValueError(
            "Sphere and box dimensions mismatch"
        )

    closest = np.clip(
        sphere.center,
        box.minimum,
        box.maximum,
    )

    delta = (
        closest - sphere.center
    )

    distance = float(
        np.linalg.norm(delta)
    )

    if distance > EPS:
        signed_distance = (
            distance - sphere.radius
        )

        normal = (
            delta / distance
        )

        contact_point = closest

    else:
        # Sphere center is inside the box.
        distance_to_min = (
            sphere.center - box.minimum
        )

        distance_to_max = (
            box.maximum - sphere.center
        )

        nearest_min_axis = int(
            np.argmin(distance_to_min)
        )

        nearest_max_axis = int(
            np.argmin(distance_to_max)
        )

        min_distance = float(
            distance_to_min[
                nearest_min_axis
            ]
        )

        max_distance = float(
            distance_to_max[
                nearest_max_axis
            ]
        )

        normal = np.zeros_like(
            sphere.center
        )

        contact_point = (
            sphere.center.copy()
        )

        if min_distance <= max_distance:
            axis = nearest_min_axis
            normal[axis] = -1.0
            contact_point[axis] = (
                box.minimum[axis]
            )
            face_distance = min_distance

        else:
            axis = nearest_max_axis
            normal[axis] = 1.0
            contact_point[axis] = (
                box.maximum[axis]
            )
            face_distance = max_distance

        signed_distance = -(
            face_distance
            + sphere.radius
        )

    return CollisionResult(
        object_a=sphere.object_id,
        object_b=box.object_id,
        collided=signed_distance <= 0,
        signed_distance=signed_distance,
        penetration_depth=max(
            0.0,
            -signed_distance,
        ),
        normal=normal,
        contact_point=contact_point,
    )


# ============================================================
# GENERAL COLLISION DETECTOR
# ============================================================

class CollisionDetector:
    """
    Dispatches collision detection
    based on shape types.
    """

    def detect(
        self,
        a: CollisionShape,
        b: CollisionShape,
    ) -> CollisionResult:
        if isinstance(a, Sphere) and isinstance(
            b,
            Sphere,
        ):
            return sphere_sphere_collision(
                a,
                b,
            )

        if isinstance(
            a,
            AxisAlignedBox,
        ) and isinstance(
            b,
            AxisAlignedBox,
        ):
            return box_box_collision(
                a,
                b,
            )

        if isinstance(a, Sphere) and isinstance(
            b,
            AxisAlignedBox,
        ):
            return sphere_box_collision(
                a,
                b,
            )

        if isinstance(
            a,
            AxisAlignedBox,
        ) and isinstance(
            b,
            Sphere,
        ):
            result = sphere_box_collision(
                b,
                a,
            )

            return CollisionResult(
                object_a=a.object_id,
                object_b=b.object_id,
                collided=result.collided,
                signed_distance=(
                    result.signed_distance
                ),
                penetration_depth=(
                    result.penetration_depth
                ),
                normal=-result.normal,
                contact_point=(
                    result.contact_point
                ),
            )

        raise TypeError(
            "Unsupported collision shape pair"
        )


# ============================================================
# TIME-TO-COLLISION
# ============================================================

@dataclass
class TimeToCollisionResult:
    will_collide: bool

    time_to_collision: float | None

    closest_approach_time: float

    closest_distance: float

    risk: CollisionRisk

    def to_dict(self) -> dict[str, Any]:
        return {
            "will_collide": self.will_collide,
            "time_to_collision": (
                self.time_to_collision
            ),
            "closest_approach_time": (
                self.closest_approach_time
            ),
            "closest_distance": (
                self.closest_distance
            ),
            "risk": self.risk.value,
        }


class TimeToCollisionPredictor:
    """
    Analytical collision prediction
    for two spheres moving at constant velocity.

    Uses relative position and velocity.

    Does not model acceleration or
    changing trajectories.
    """

    def __init__(
        self,
        config: CollisionConfig,
    ) -> None:
        self.config = config

    def _risk(
        self,
        ttc: float | None,
    ) -> CollisionRisk:
        if ttc is None:
            return CollisionRisk.NONE

        if ttc <= self.config.critical_ttc:
            return CollisionRisk.CRITICAL

        if ttc <= self.config.high_ttc:
            return CollisionRisk.HIGH

        if ttc <= self.config.moderate_ttc:
            return CollisionRisk.MODERATE

        return CollisionRisk.LOW

    def predict(
        self,
        a: Sphere,
        b: Sphere,
    ) -> TimeToCollisionResult:
        if a.center.shape != b.center.shape:
            raise ValueError(
                "Sphere dimensions mismatch"
            )

        relative_position = (
            b.center - a.center
        )

        relative_velocity = (
            b.velocity - a.velocity
        )

        combined_radius = (
            a.radius
            + b.radius
            + self.config.safety_margin
        )

        aa = float(
            np.dot(
                relative_velocity,
                relative_velocity,
            )
        )

        bb = 2.0 * float(
            np.dot(
                relative_position,
                relative_velocity,
            )
        )

        cc = (
            float(
                np.dot(
                    relative_position,
                    relative_position,
                )
            )
            - combined_radius ** 2
        )

        horizon = (
            self.config.prediction_horizon
        )

        if aa <= EPS:
            closest_time = 0.0
        else:
            closest_time = float(
                np.clip(
                    -bb / (2.0 * aa),
                    0.0,
                    horizon,
                )
            )

        closest_vector = (
            relative_position
            + relative_velocity
            * closest_time
        )

        closest_distance = max(
            0.0,
            float(
                np.linalg.norm(
                    closest_vector
                )
            ) - combined_radius,
        )

        if cc <= 0:
            ttc = 0.0

        elif aa <= EPS:
            ttc = None

        else:
            discriminant = (
                bb * bb
                - 4.0 * aa * cc
            )

            if discriminant < 0:
                ttc = None
            else:
                root = math.sqrt(
                    max(0.0, discriminant)
                )

                entry = (
                    -bb - root
                ) / (2.0 * aa)

                if 0 <= entry <= horizon:
                    ttc = float(entry)
                else:
                    ttc = None

        return TimeToCollisionResult(
            will_collide=ttc is not None,
            time_to_collision=ttc,
            closest_approach_time=(
                closest_time
            ),
            closest_distance=(
                closest_distance
            ),
            risk=self._risk(ttc),
        )


# ============================================================
# CONTINUOUS SEGMENT COLLISION
# ============================================================

def segment_sphere_clearance(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: Sphere,
    robot_radius: float = 0.0,
) -> float:
    """
    Minimum clearance between a moving
    spherical robot and a stationary sphere.

    Robot center travels along a straight
    line segment from start to end.

    This catches collisions between
    discretely sampled trajectory states.
    """

    if robot_radius < 0:
        raise ValueError(
            "robot_radius must be nonnegative"
        )

    dimension = len(obstacle.center)

    start = as_vector(
        start,
        dimension,
        "start",
    )

    end = as_vector(
        end,
        dimension,
        "end",
    )

    segment = end - start

    length_squared = float(
        np.dot(
            segment,
            segment,
        )
    )

    if length_squared <= EPS:
        closest = start

    else:
        fraction = float(
            np.dot(
                obstacle.center - start,
                segment,
            ) / length_squared
        )

        fraction = float(
            np.clip(
                fraction,
                0.0,
                1.0,
            )
        )

        closest = (
            start
            + fraction * segment
        )

    distance = float(
        np.linalg.norm(
            closest - obstacle.center
        )
    )

    return (
        distance
        - obstacle.radius
        - robot_radius
    )


# ============================================================
# POINT CLEARANCE
# ============================================================

def point_obstacle_clearance(
    point: np.ndarray,
    obstacle: CollisionShape,
    robot_radius: float = 0.0,
) -> float:
    """
    Signed clearance between a spherical
    robot centered at point and an obstacle.
    """

    if robot_radius < 0:
        raise ValueError(
            "robot_radius must be nonnegative"
        )

    if isinstance(obstacle, Sphere):
        point = as_vector(
            point,
            len(obstacle.center),
            "point",
        )

        return (
            float(
                np.linalg.norm(
                    point - obstacle.center
                )
            )
            - obstacle.radius
            - robot_radius
        )

    point = as_vector(
        point,
        len(obstacle.minimum),
        "point",
    )

    outside = np.maximum(
        np.maximum(
            obstacle.minimum - point,
            point - obstacle.maximum,
        ),
        0.0,
    )

    outside_distance = float(
        np.linalg.norm(outside)
    )

    if np.any(outside > 0):
        return (
            outside_distance
            - robot_radius
        )

    inside_distance = float(
        np.min(
            np.minimum(
                point - obstacle.minimum,
                obstacle.maximum - point,
            )
        )
    )

    return -(
        inside_distance
        + robot_radius
    )


# ============================================================
# TRAJECTORY COLLISION REPORT
# ============================================================

@dataclass
class TrajectoryCollisionReport:
    collision_detected: bool

    minimum_clearance: float

    collision_steps: list[int]

    closest_obstacle_id: str | None

    risk: CollisionRisk

    point_clearances: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "collision_detected": (
                self.collision_detected
            ),
            "minimum_clearance": (
                self.minimum_clearance
            ),
            "collision_steps": (
                self.collision_steps
            ),
            "closest_obstacle_id": (
                self.closest_obstacle_id
            ),
            "risk": self.risk.value,
            "point_clearances": (
                self.point_clearances
            ),
        }


# ============================================================
# TRAJECTORY COLLISION CHECKER
# ============================================================

class TrajectoryCollisionChecker:
    """
    Checks a planned robot trajectory.

    Supports:
        point clearances for spheres and AABBs
        swept-segment checks for spheres

    For boxes, intermediate segment points
    are sampled; exact swept-box geometry
    is not implemented here.
    """

    def __init__(
        self,
        config: CollisionConfig,
        obstacles: Iterable[CollisionShape],
    ) -> None:
        self.config = config
        self.obstacles = list(
            obstacles
        )

        for obstacle in self.obstacles:
            dimension = (
                len(obstacle.center)
                if isinstance(obstacle, Sphere)
                else len(obstacle.minimum)
            )

            if dimension != config.spatial_dim:
                raise ValueError(
                    "Obstacle spatial dimensions mismatch"
                )

    def _risk(
        self,
        clearance: float,
    ) -> CollisionRisk:
        if clearance <= 0:
            return CollisionRisk.CRITICAL

        if clearance < (
            self.config.safety_margin
        ):
            return CollisionRisk.HIGH

        if clearance < (
            self.config.safety_margin * 2
        ):
            return CollisionRisk.MODERATE

        return CollisionRisk.LOW

    def check(
        self,
        positions: np.ndarray | Tensor,
    ) -> TrajectoryCollisionReport:
        if isinstance(positions, Tensor):
            positions = (
                positions.detach()
                .cpu()
                .numpy()
            )

        positions = np.asarray(
            positions,
            dtype=np.float64,
        )

        if (
            positions.ndim != 2
            or positions.shape[1]
            != self.config.spatial_dim
            or positions.shape[0] == 0
        ):
            raise ValueError(
                "Positions must be [T,spatial_dim]"
            )

        if not np.isfinite(
            positions
        ).all():
            raise ValueError(
                "Positions contain non-finite values"
            )

        if not self.obstacles:
            return TrajectoryCollisionReport(
                collision_detected=False,
                minimum_clearance=float("inf"),
                collision_steps=[],
                closest_obstacle_id=None,
                risk=CollisionRisk.NONE,
                point_clearances=[
                    float("inf")
                    for _ in positions
                ],
            )

        minimum_clearance = float("inf")
        closest_obstacle_id = None

        point_clearances = []
        collision_steps = set()

        for index, position in enumerate(
            positions
        ):
            step_clearance = float("inf")

            for obstacle in self.obstacles:
                clearance = (
                    point_obstacle_clearance(
                        position,
                        obstacle,
                        self.config.robot_radius,
                    )
                )

                step_clearance = min(
                    step_clearance,
                    clearance,
                )

                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    closest_obstacle_id = (
                        obstacle.object_id
                    )

                if clearance <= 0:
                    collision_steps.add(
                        index
                    )

            point_clearances.append(
                step_clearance
            )

        # Continuous straight-line checks
        # for spherical obstacles.
        for index in range(
            len(positions) - 1
        ):
            start = positions[index]
            end = positions[index + 1]

            for obstacle in self.obstacles:
                if isinstance(obstacle, Sphere):
                    clearance = (
                        segment_sphere_clearance(
                            start,
                            end,
                            obstacle,
                            self.config.robot_radius,
                        )
                    )

                    if clearance < minimum_clearance:
                        minimum_clearance = clearance
                        closest_obstacle_id = (
                            obstacle.object_id
                        )

                    if clearance <= 0:
                        collision_steps.add(
                            index + 1
                        )

                else:
                    # Approximate segment-to-AABB
                    # clearance by dense sampling.
                    # Not a formal continuous check.
                    for fraction in np.linspace(
                        0.0,
                        1.0,
                        21,
                    ):
                        point = (
                            start
                            + fraction * (
                                end - start
                            )
                        )

                        clearance = (
                            point_obstacle_clearance(
                                point,
                                obstacle,
                                self.config.robot_radius,
                            )
                        )

                        if clearance < minimum_clearance:
                            minimum_clearance = clearance
                            closest_obstacle_id = (
                                obstacle.object_id
                            )

                        if clearance <= 0:
                            collision_steps.add(
                                index + 1
                            )

        return TrajectoryCollisionReport(
            collision_detected=(
                minimum_clearance <= 0
            ),
            minimum_clearance=(
                minimum_clearance
            ),
            collision_steps=sorted(
                collision_steps
            ),
            closest_obstacle_id=(
                closest_obstacle_id
            ),
            risk=self._risk(
                minimum_clearance
            ),
            point_clearances=(
                point_clearances
            ),
        )


# ============================================================
# BATCH TORCH CLEARANCE
# ============================================================

class TorchCollisionClearance:
    """
    Batched differentiable clearance
    against stationary spheres.

    positions:
        [..., spatial_dim]

    output:
        [...]
    """

    def __init__(
        self,
        config: CollisionConfig,
        obstacles: list[Sphere],
    ) -> None:
        self.config = config
        self.obstacles = obstacles

        for obstacle in obstacles:
            if (
                len(obstacle.center)
                != config.spatial_dim
            ):
                raise ValueError(
                    "Obstacle dimension mismatch"
                )

    def minimum_clearance(
        self,
        positions: Tensor,
    ) -> Tensor:
        if positions.shape[-1] != (
            self.config.spatial_dim
        ):
            raise ValueError(
                "Invalid position dimensions"
            )

        if not self.obstacles:
            return torch.full(
                positions.shape[:-1],
                float("inf"),
                dtype=positions.dtype,
                device=positions.device,
            )

        centers = torch.tensor(
            np.stack(
                [
                    obstacle.center
                    for obstacle in self.obstacles
                ]
            ),
            dtype=positions.dtype,
            device=positions.device,
        )

        radii = torch.tensor(
            [
                obstacle.radius
                for obstacle in self.obstacles
            ],
            dtype=positions.dtype,
            device=positions.device,
        )

        distances = torch.linalg.vector_norm(
            positions.unsqueeze(-2)
            - centers,
            dim=-1,
        )

        clearances = (
            distances
            - radii
            - self.config.robot_radius
        )

        return clearances.min(
            dim=-1
        ).values


# ============================================================
# PLANNER CONSTRAINT ADAPTER
# ============================================================

class CollisionSafetyConstraint:
    """
    Compatible with the MPCSafetyConstraint
    interface from planning/mpc.py.

    Checks predicted next-state positions
    against stationary spherical obstacles.
    """

    def __init__(
        self,
        config: CollisionConfig,
        obstacles: list[Sphere],
        *,
        minimum_clearance: float | None = None,
    ) -> None:
        self.config = config

        self.minimum_clearance = (
            config.safety_margin
            if minimum_clearance is None
            else minimum_clearance
        )

        if self.minimum_clearance < 0:
            raise ValueError(
                "minimum_clearance must be nonnegative"
            )

        self.clearance = (
            TorchCollisionClearance(
                config,
                obstacles,
            )
        )

    def is_safe(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> Tensor:
        positions = next_states[
            :, :self.config.spatial_dim
        ]

        clearances = (
            self.clearance.minimum_clearance(
                positions
            )
        )

        return (
            clearances
            >= self.minimum_clearance
        )


# ============================================================
# MAIN COLLISION MODEL
# ============================================================

class CollisionModel:
    """
    High-level collision detection API.
    """

    def __init__(
        self,
        config: CollisionConfig | None = None,
    ) -> None:
        self.config = (
            config or CollisionConfig()
        )

        self.detector = (
            CollisionDetector()
        )

        self.ttc_predictor = (
            TimeToCollisionPredictor(
                self.config
            )
        )

    def detect(
        self,
        a: CollisionShape,
        b: CollisionShape,
    ) -> CollisionResult:
        return self.detector.detect(
            a,
            b,
        )

    def predict_collision(
        self,
        a: Sphere,
        b: Sphere,
    ) -> TimeToCollisionResult:
        return self.ttc_predictor.predict(
            a,
            b,
        )

    def evaluate_trajectory(
        self,
        positions: np.ndarray | Tensor,
        obstacles: Iterable[CollisionShape],
    ) -> TrajectoryCollisionReport:
        checker = (
            TrajectoryCollisionChecker(
                self.config,
                obstacles,
            )
        )

        return checker.check(
            positions
        )


# ============================================================
# FACTORY
# ============================================================

def create_collision_model(
    config: CollisionConfig | None = None,
) -> CollisionModel:
    return CollisionModel(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - COLLISION MODEL"
    )
    print("=" * 65)

    config = CollisionConfig(
        spatial_dim=2,
        robot_radius=0.15,
        safety_margin=0.2,
        prediction_horizon=5.0,
    )

    model = CollisionModel(
        config
    )

    robot = Sphere(
        center=[0.0, 0.0],
        radius=0.15,
        velocity=[1.0, 0.0],
        object_id="robot",
    )

    moving_obstacle = Sphere(
        center=[2.0, 0.0],
        radius=0.3,
        velocity=[-0.2, 0.0],
        object_id="moving_obstacle",
    )

    static_obstacle = Sphere(
        center=[1.0, 0.5],
        radius=0.25,
        object_id="static_obstacle",
    )

    box = AxisAlignedBox(
        minimum=[1.5, 1.5],
        maximum=[2.5, 2.5],
        object_id="box_obstacle",
    )

    print("\n1. Sphere collision detection")

    collision = model.detect(
        robot,
        static_obstacle,
    )

    print(
        collision.to_dict()
    )

    print("\n2. Time-to-collision prediction")

    ttc = model.predict_collision(
        robot,
        moving_obstacle,
    )

    print(
        ttc.to_dict()
    )

    print("\n3. Sphere-box collision")

    box_collision = model.detect(
        robot,
        box,
    )

    print(
        box_collision.to_dict()
    )

    print("\n4. Trajectory collision screening")

    trajectory = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )

    report = model.evaluate_trajectory(
        trajectory,
        [
            static_obstacle,
            box,
        ],
    )

    print(
        report.to_dict()
    )

    print("\n5. Batched PyTorch clearance")

    torch_clearance = (
        TorchCollisionClearance(
            config,
            [static_obstacle],
        )
    )

    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
        ],
        dtype=torch.float32,
    )

    print(
        torch_clearance.minimum_clearance(
            points
        )
    )

    print(
        "\nCollision model demonstration completed"
    )


if __name__ == "__main__":
    main()
