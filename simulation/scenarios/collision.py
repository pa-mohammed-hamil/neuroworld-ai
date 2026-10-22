
```python
"""
NeuroWorld AI
File: neuroworld/physics/collision.py

2D Collision Detection and Prediction Engine

Features:
- Circle-circle collision detection
- Circle-rectangle collision detection
- Rectangle-rectangle collision detection (AABB)
- Robot and obstacle collision checks
- Multi-robot collision detection
- Swept-circle collision detection
- Continuous collision detection for linear motion
- Time-to-collision (TTC) estimation
- Collision risk scoring
- Safety margin handling
- World boundary checks
- Trajectory collision prediction
- Collision reports and statistics

Coordinates:
    x, y: meters
    vx, vy: meters/second
    radius: meters
    time: seconds

No external dependencies.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


EPSILON = 1e-9

Point2D = Tuple[float, float]


class CollisionType(str, Enum):
    NONE = "none"
    CIRCLE_CIRCLE = "circle_circle"
    CIRCLE_RECTANGLE = "circle_rectangle"
    RECTANGLE_RECTANGLE = "rectangle_rectangle"
    ROBOT_OBSTACLE = "robot_obstacle"
    ROBOT_ROBOT = "robot_robot"
    BOUNDARY = "boundary"
    PREDICTED = "predicted"


@dataclass
class CollisionConfig:
    safety_margin: float = 0.1
    prediction_horizon: float = 5.0
    risk_distance: float = 2.0
    critical_ttc: float = 1.0
    warning_ttc: float = 3.0
    epsilon: float = EPSILON


@dataclass
class CircleBody:
    body_id: str
    x: float
    y: float
    radius: float

    vx: float = 0.0
    vy: float = 0.0

    is_dynamic: bool = False

    @property
    def position(self) -> Point2D:
        return self.x, self.y

    @property
    def velocity(self) -> Point2D:
        return self.vx, self.vy


@dataclass
class RectangleBody:
    body_id: str
    x: float
    y: float
    width: float
    height: float

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )


@dataclass
class CollisionResult:
    collided: bool
    collision_type: CollisionType

    body_a: str
    body_b: str

    distance: float
    penetration_depth: float = 0.0

    contact_point: Optional[Point2D] = None
    normal: Optional[Point2D] = None

    time_to_collision: Optional[float] = None
    risk_score: float = 0.0


@dataclass
class TrajectoryCollisionResult:
    safe: bool
    first_collision_step: Optional[int]
    first_collision_time: Optional[float]
    collisions: List[CollisionResult] = field(
        default_factory=list
    )


@dataclass
class CollisionStatistics:
    total_checks: int = 0
    detected_collisions: int = 0
    predicted_collisions: int = 0
    rejected_trajectories: int = 0


class CollisionEngine:
    """
    Geometry-based collision detection engine.

    This engine does not resolve forces,
    friction, impacts or rigid-body dynamics.
    """

    def __init__(
        self,
        config: Optional[CollisionConfig] = None,
    ):
        self.config = config or CollisionConfig()

        self._validate_config()

        self.statistics = CollisionStatistics()

    def _validate_config(self) -> None:
        cfg = self.config

        values = (
            cfg.safety_margin,
            cfg.prediction_horizon,
            cfg.risk_distance,
            cfg.critical_ttc,
            cfg.warning_ttc,
            cfg.epsilon,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                "Collision configuration must be finite"
            )

        if cfg.safety_margin < 0:
            raise ValueError(
                "Safety margin cannot be negative"
            )

        if cfg.prediction_horizon <= 0:
            raise ValueError(
                "Prediction horizon must be positive"
            )

        if cfg.risk_distance <= 0:
            raise ValueError(
                "Risk distance must be positive"
            )

        if cfg.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if not (
            0 < cfg.critical_ttc
            <= cfg.warning_ttc
        ):
            raise ValueError(
                "Invalid TTC thresholds"
            )

    @staticmethod
    def distance(
        a: Point2D,
        b: Point2D,
    ) -> float:
        return math.hypot(
            b[0] - a[0],
            b[1] - a[1],
        )

    @staticmethod
    def _clamp(
        value: float,
        lower: float,
        upper: float,
    ) -> float:
        return max(lower, min(value, upper))

    def _risk_score(
        self,
        clearance: float,
        ttc: Optional[float] = None,
    ) -> float:
        """
        Heuristic risk score in [0,1].

        This is not a calibrated
        collision probability.
        """

        if clearance <= 0:
            return 1.0

        distance_risk = max(
            0.0,
            1.0 - clearance / self.config.risk_distance,
        )

        if ttc is None:
            return distance_risk

        if ttc <= self.config.critical_ttc:
            time_risk = 1.0
        elif ttc >= self.config.warning_ttc:
            time_risk = 0.0
        else:
            time_risk = (
                self.config.warning_ttc - ttc
            ) / (
                self.config.warning_ttc
                - self.config.critical_ttc
                + self.config.epsilon
            )

        return self._clamp(
            max(distance_risk, time_risk),
            0.0,
            1.0,
        )

    def circle_circle(
        self,
        a: CircleBody,
        b: CircleBody,
        safety_margin: Optional[float] = None,
    ) -> CollisionResult:
        self.statistics.total_checks += 1

        margin = (
            self.config.safety_margin
            if safety_margin is None
            else safety_margin
        )

        if margin < 0:
            raise ValueError(
                "Safety margin cannot be negative"
            )

        if a.radius < 0 or b.radius < 0:
            raise ValueError(
                "Body radius cannot be negative"
            )

        dx = b.x - a.x
        dy = b.y - a.y

        center_distance = math.hypot(dx, dy)

        physical_radius = a.radius + b.radius
        safe_radius = physical_radius + margin

        collided = (
            center_distance <= safe_radius
        )

        penetration = max(
            0.0,
            physical_radius - center_distance,
        )

        if center_distance > self.config.epsilon:
            nx = dx / center_distance
            ny = dy / center_distance
        else:
            nx, ny = 1.0, 0.0

        contact_point = (
            a.x + nx * a.radius,
            a.y + ny * a.radius,
        )

        clearance = (
            center_distance - safe_radius
        )

        if collided:
            self.statistics.detected_collisions += 1

        return CollisionResult(
            collided=collided,
            collision_type=CollisionType.CIRCLE_CIRCLE,
            body_a=a.body_id,
            body_b=b.body_id,
            distance=center_distance,
            penetration_depth=penetration,
            contact_point=contact_point,
            normal=(nx, ny),
            risk_score=self._risk_score(clearance),
        )

    def circle_rectangle(
        self,
        circle: CircleBody,
        rectangle: RectangleBody,
        safety_margin: Optional[float] = None,
    ) -> CollisionResult:
        self.statistics.total_checks += 1

        margin = (
            self.config.safety_margin
            if safety_margin is None
            else safety_margin
        )

        if (
            circle.radius < 0
            or rectangle.width <= 0
            or rectangle.height <= 0
            or margin < 0
        ):
            raise ValueError(
                "Invalid geometry or safety margin"
            )

        left, bottom, right, top = (
            rectangle.bounds
        )

        closest_x = self._clamp(
            circle.x, left, right
        )
        closest_y = self._clamp(
            circle.y, bottom, top
        )

        dx = circle.x - closest_x
        dy = circle.y - closest_y

        distance = math.hypot(dx, dy)

        effective_radius = (
            circle.radius + margin
        )

        collided = distance <= effective_radius

        penetration = max(
            0.0,
            circle.radius - distance,
        )

        if distance > self.config.epsilon:
            normal = (
                dx / distance,
                dy / distance,
            )
        else:
            # Circle center lies within or
            # directly on the rectangle.
            distances = [
                (circle.x - left, (-1.0, 0.0)),
                (right - circle.x, (1.0, 0.0)),
                (circle.y - bottom, (0.0, -1.0)),
                (top - circle.y, (0.0, 1.0)),
            ]

            nearest = min(
                distances,
                key=lambda item: item[0],
            )

            normal = nearest[1]

            if (
                left <= circle.x <= right
                and bottom <= circle.y <= top
            ):
                penetration = (
                    circle.radius
                    + max(0.0, nearest[0])
                )

        if collided:
            self.statistics.detected_collisions += 1

        return CollisionResult(
            collided=collided,
            collision_type=CollisionType.CIRCLE_RECTANGLE,
            body_a=circle.body_id,
            body_b=rectangle.body_id,
            distance=distance,
            penetration_depth=penetration,
            contact_point=(
                closest_x,
                closest_y,
            ),
            normal=normal,
            risk_score=self._risk_score(
                distance - effective_radius
            ),
        )

    def rectangle_rectangle(
        self,
        a: RectangleBody,
        b: RectangleBody,
        safety_margin: Optional[float] = None,
    ) -> CollisionResult:
        self.statistics.total_checks += 1

        margin = (
            self.config.safety_margin
            if safety_margin is None
            else safety_margin
        )

        if (
            a.width <= 0
            or a.height <= 0
            or b.width <= 0
            or b.height <= 0
            or margin < 0
        ):
            raise ValueError(
                "Invalid rectangle geometry"
            )

        a_left, a_bottom, a_right, a_top = a.bounds
        b_left, b_bottom, b_right, b_top = b.bounds

        separation_x = max(
            b_left - a_right,
            a_left - b_right,
            0.0,
        )

        separation_y = max(
            b_bottom - a_top,
            a_bottom - b_top,
            0.0,
        )

        distance = math.hypot(
            separation_x,
            separation_y,
        )

        collided = distance <= margin

        overlap_x = min(
            a_right, b_right
        ) - max(a_left, b_left)

        overlap_y = min(
            a_top, b_top
        ) - max(a_bottom, b_bottom)

        penetration = (
            min(overlap_x, overlap_y)
            if overlap_x > 0 and overlap_y > 0
            else 0.0
        )

        if collided:
            self.statistics.detected_collisions += 1

        return CollisionResult(
            collided=collided,
            collision_type=(
                CollisionType.RECTANGLE_RECTANGLE
            ),
            body_a=a.body_id,
            body_b=b.body_id,
            distance=distance,
            penetration_depth=penetration,
            risk_score=self._risk_score(
                distance - margin
            ),
        )

    def swept_circle(
        self,
        moving: CircleBody,
        start: Point2D,
        end: Point2D,
        obstacle: CircleBody,
        safety_margin: Optional[float] = None,
    ) -> CollisionResult:
        """
        Check whether a circle moving along
        a straight line intersects another
        stationary circle.

        Prevents discrete-step tunneling.
        """

        margin = (
            self.config.safety_margin
            if safety_margin is None
            else safety_margin
        )

        if margin < 0:
            raise ValueError(
                "Safety margin cannot be negative"
            )

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length_squared = dx * dx + dy * dy

        if length_squared <= self.config.epsilon:
            closest = start
        else:
            projection = (
                (
                    (obstacle.x - start[0]) * dx
                    + (obstacle.y - start[1]) * dy
                ) / length_squared
            )

            t = self._clamp(
                projection,
                0.0,
                1.0,
            )

            closest = (
                start[0] + t * dx,
                start[1] + t * dy,
            )

        proxy = CircleBody(
            body_id=moving.body_id,
            x=closest[0],
            y=closest[1],
            radius=moving.radius,
        )

        result = self.circle_circle(
            proxy,
            obstacle,
            safety_margin=margin,
        )

        result.contact_point = closest

        return result

    def time_to_collision(
        self,
        a: CircleBody,
        b: CircleBody,
        horizon: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute first collision time for
        two circles with constant velocities.

        Returns:
            Time in seconds, or None.

        Assumptions:
            Constant linear velocity.
            No acceleration or turning.
        """

        horizon = (
            self.config.prediction_horizon
            if horizon is None
            else horizon
        )

        if horizon < 0:
            raise ValueError(
                "Horizon cannot be negative"
            )

        px = b.x - a.x
        py = b.y - a.y

        vx = b.vx - a.vx
        vy = b.vy - a.vy

        radius = (
            a.radius
            + b.radius
            + self.config.safety_margin
        )

        c = (
            px * px
            + py * py
            - radius * radius
        )

        if c <= 0:
            return 0.0

        coefficient_a = (
            vx * vx + vy * vy
        )

        if coefficient_a <= self.config.epsilon:
            return None

        coefficient_b = 2.0 * (
            px * vx + py * vy
        )

        discriminant = (
            coefficient_b * coefficient_b
            - 4.0 * coefficient_a * c
        )

        if discriminant < 0:
            return None

        root = math.sqrt(
            discriminant
        )

        first_time = (
            -coefficient_b - root
        ) / (
            2.0 * coefficient_a
        )

        if 0 <= first_time <= horizon:
            self.statistics.predicted_collisions += 1
            return first_time

        return None

    def predict_collision(
        self,
        a: CircleBody,
        b: CircleBody,
        horizon: Optional[float] = None,
    ) -> CollisionResult:
        """
        Predict collision and compute risk.
        """

        ttc = self.time_to_collision(
            a,
            b,
            horizon=horizon,
        )

        distance = self.distance(
            a.position,
            b.position,
        )

        clearance = (
            distance
            - a.radius
            - b.radius
            - self.config.safety_margin
        )

        return CollisionResult(
            collided=ttc is not None,
            collision_type=CollisionType.PREDICTED,
            body_a=a.body_id,
            body_b=b.body_id,
            distance=distance,
            time_to_collision=ttc,
            risk_score=self._risk_score(
                clearance,
                ttc,
            ),
        )

    def check_world_bounds(
        self,
        body: CircleBody,
        world_width: float,
        world_height: float,
    ) -> CollisionResult:
        """
        Check robot/world boundary overlap.
        """

        if world_width <= 0 or world_height <= 0:
            raise ValueError(
                "World dimensions must be positive"
            )

        margin = self.config.safety_margin

        left = body.x - body.radius - margin
        right = body.x + body.radius + margin
        bottom = body.y - body.radius - margin
        top = body.y + body.radius + margin

        violated = (
            left < 0
            or right > world_width
            or bottom < 0
            or top > world_height
        )

        clearance = min(
            left,
            world_width - right,
            bottom,
            world_height - top,
        )

        self.statistics.total_checks += 1

        if violated:
            self.statistics.detected_collisions += 1

        return CollisionResult(
            collided=violated,
            collision_type=CollisionType.BOUNDARY,
            body_a=body.body_id,
            body_b="world_boundary",
            distance=max(0.0, clearance),
            penetration_depth=max(
                0.0,
                -clearance,
            ),
            risk_score=self._risk_score(
                clearance
            ),
        )

    def check_robots(
        self,
        robots: Sequence[CircleBody],
        obstacles: Sequence[CircleBody],
    ) -> List[CollisionResult]:
        """
        Detect robot-obstacle and
        robot-robot collisions.
        """

        collisions = []

        for robot in robots:
            for obstacle in obstacles:
                result = self.circle_circle(
                    robot,
                    obstacle,
                )

                if result.collided:
                    result.collision_type = (
                        CollisionType.ROBOT_OBSTACLE
                    )

                    collisions.append(result)

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                result = self.circle_circle(
                    robots[i],
                    robots[j],
                )

                if result.collided:
                    result.collision_type = (
                        CollisionType.ROBOT_ROBOT
                    )

                    collisions.append(result)

        return collisions

    def check_trajectory(
        self,
        robot: CircleBody,
        trajectory: Sequence[Point2D],
        obstacles: Sequence[CircleBody],
        timestep: float = 0.1,
    ) -> TrajectoryCollisionResult:
        """
        Check a planned robot trajectory.

        Uses swept-circle tests between
        consecutive trajectory points.

        Assumes obstacles are stationary
        throughout the trajectory.
        """

        if timestep <= 0:
            raise ValueError(
                "timestep must be positive"
            )

        if not trajectory:
            raise ValueError(
                "Trajectory cannot be empty"
            )

        collisions = []

        first_step = None
        first_time = None

        for index, point in enumerate(
            trajectory
        ):
            if index == 0:
                segment_start = point
            else:
                segment_start = trajectory[
                    index - 1
                ]

            segment_end = point

            for obstacle in obstacles:
                result = self.swept_circle(
                    moving=robot,
                    start=segment_start,
                    end=segment_end,
                    obstacle=obstacle,
                )

                if result.collided:
                    collisions.append(result)

                    if first_step is None:
                        first_step = index
                        first_time = (
                            index * timestep
                        )

        safe = len(collisions) == 0

        if not safe:
            self.statistics.rejected_trajectories += 1

        return TrajectoryCollisionResult(
            safe=safe,
            first_collision_step=first_step,
            first_collision_time=first_time,
            collisions=collisions,
        )

    def safe_motion(
        self,
        robot: CircleBody,
        target_position: Point2D,
        obstacles: Sequence[CircleBody],
        world_width: float,
        world_height: float,
    ) -> bool:
        """
        Check whether a proposed straight-line
        robot movement is safe.

        Includes swept collision checks
        and world boundary validation.
        """

        target = CircleBody(
            body_id=robot.body_id,
            x=target_position[0],
            y=target_position[1],
            radius=robot.radius,
        )

        if self.check_world_bounds(
            target,
            world_width,
            world_height,
        ).collided:
            return False

        if self.check_world_bounds(
            robot,
            world_width,
            world_height,
        ).collided:
            return False

        for obstacle in obstacles:
            result = self.swept_circle(
                moving=robot,
                start=robot.position,
                end=target_position,
                obstacle=obstacle,
            )

            if result.collided:
                return False

        return True

    def reset_statistics(self) -> None:
        self.statistics = CollisionStatistics()

    def summary(self) -> Dict[str, int]:
        return {
            "total_checks": (
                self.statistics.total_checks
            ),
            "detected_collisions": (
                self.statistics.detected_collisions
            ),
            "predicted_collisions": (
                self.statistics.predicted_collisions
            ),
            "rejected_trajectories": (
                self.statistics.rejected_trajectories
            ),
        }


if __name__ == "__main__":

    engine = CollisionEngine(
        CollisionConfig(
            safety_margin=0.1,
            prediction_horizon=5.0,
        )
    )

    robot = CircleBody(
        body_id="robot_1",
        x=1.0,
        y=1.0,
        radius=0.25,
        vx=0.5,
        vy=0.0,
        is_dynamic=True,
    )

    obstacle = CircleBody(
        body_id="obstacle_1",
        x=3.0,
        y=1.0,
        radius=0.4,
    )

    # 1. Current collision detection
    current = engine.circle_circle(
        robot,
        obstacle,
    )

    print(
        "Current collision:",
        current.collided,
    )

    # 2. Future collision prediction
    prediction = engine.predict_collision(
        robot,
        obstacle,
    )

    print(
        "Predicted collision:",
        prediction.collided,
    )

    print(
        "Time to collision:",
        prediction.time_to_collision,
    )

    print(
        "Risk score:",
        prediction.risk_score,
    )

    # 3. Planned trajectory validation
    trajectory = [
        (1.0, 1.0),
        (1.5, 1.0),
        (2.0, 1.0),
        (2.5, 1.0),
        (3.0, 1.0),
        (4.0, 1.0),
    ]

    trajectory_result = (
        engine.check_trajectory(
            robot,
            trajectory,
            obstacles=[obstacle],
            timestep=0.1,
        )
    )

    print(
        "Trajectory safe:",
        trajectory_result.safe,
    )

    print(
        "First collision step:",
        trajectory_result.first_collision_step,
    )

    # 4. Safe motion validation
    safe = engine.safe_motion(
        robot=robot,
        target_position=(1.5, 2.0),
        obstacles=[obstacle],
        world_width=10.0,
        world_height=10.0,
    )

    print(
        "Motion accepted:",
        safe,
    )

    print(
        "Statistics:",
        engine.summary(),
    )
```
