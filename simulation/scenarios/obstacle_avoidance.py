
```python
"""
NeuroWorld AI
File: neuroworld/environment/obstacle_avoidance.py

2D obstacle avoidance using Dynamic Window Approach (DWA).

Dependencies:
    Python 3.10+

Features:
    - Static and moving obstacle avoidance
    - Differential-drive robot motion
    - Velocity and acceleration constraints
    - Local trajectory prediction
    - Swept collision detection
    - Goal-directed trajectory scoring
    - Emergency stop requests
    - Blocked and stuck detection
    - Simulation diagnostics

Simulation only; not a hardware safety controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


EPS = 1e-9


class AvoidanceStatus(str, Enum):
    CLEAR = "clear"
    AVOIDING = "avoiding"
    WARNING = "warning"
    BLOCKED = "blocked"
    STUCK = "stuck"
    GOAL_REACHED = "goal_reached"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class AvoidanceConfig:
    robot_radius: float = 0.25
    safety_margin: float = 0.15

    min_linear_speed: float = 0.0
    max_linear_speed: float = 0.8
    max_angular_speed: float = 1.5

    max_linear_acceleration: float = 0.8
    max_angular_acceleration: float = 2.0

    control_dt: float = 0.1
    prediction_dt: float = 0.05
    prediction_horizon: float = 2.0

    linear_samples: int = 11
    angular_samples: int = 21

    goal_tolerance: float = 0.2
    warning_clearance: float = 0.8

    heading_weight: float = 1.5
    progress_weight: float = 2.0
    clearance_weight: float = 1.0
    speed_weight: float = 0.3
    smoothness_weight: float = 0.15

    stuck_steps: int = 30
    stuck_distance: float = 0.03

    world_width: Optional[float] = None
    world_height: Optional[float] = None

    allow_reverse: bool = False


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass
class Obstacle:
    obstacle_id: str
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    dynamic: bool = False


@dataclass
class VelocityCommand:
    linear: float = 0.0
    angular: float = 0.0


@dataclass
class TrajectoryPoint:
    time: float
    x: float
    y: float
    yaw: float


@dataclass
class TrajectoryEvaluation:
    safe: bool
    command: VelocityCommand
    trajectory: list[TrajectoryPoint]
    score: float
    minimum_clearance: float
    final_distance_to_goal: float
    collision_obstacle: Optional[str] = None


@dataclass
class AvoidanceDecision:
    status: AvoidanceStatus
    command: VelocityCommand
    trajectory: list[TrajectoryPoint]
    score: float
    minimum_clearance: float
    distance_to_goal: float
    evaluated_candidates: int
    safe_candidates: int
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ObstacleAvoidanceEngine:
    """Collision-aware local navigation planner."""

    def __init__(
        self,
        config: Optional[AvoidanceConfig] = None,
    ) -> None:
        self.config = config or AvoidanceConfig()
        self._validate_config()

        self.emergency_stop_active = False
        self.previous_position: Optional[
            tuple[float, float]
        ] = None
        self.stuck_counter = 0

        self.total_plans = 0
        self.blocked_plans = 0
        self.emergency_stops = 0

    def _validate_config(self) -> None:
        c = self.config

        positive = (
            c.robot_radius,
            c.max_linear_speed,
            c.max_angular_speed,
            c.max_linear_acceleration,
            c.max_angular_acceleration,
            c.control_dt,
            c.prediction_dt,
            c.prediction_horizon,
            c.goal_tolerance,
        )

        if any(
            not math.isfinite(x) or x <= 0
            for x in positive
        ):
            raise ValueError(
                "Positive configuration fields "
                "must be finite and greater than zero"
            )

        if (
            c.safety_margin < 0
            or c.warning_clearance < 0
            or c.stuck_distance < 0
        ):
            raise ValueError(
                "Margins must be nonnegative"
            )

        if c.linear_samples < 2 or c.angular_samples < 2:
            raise ValueError(
                "Sampling counts must be at least 2"
            )

        if c.stuck_steps <= 0:
            raise ValueError(
                "stuck_steps must be positive"
            )

        if not c.allow_reverse and c.min_linear_speed < 0:
            raise ValueError(
                "Reverse movement is disabled"
            )

        if c.min_linear_speed > c.max_linear_speed:
            raise ValueError(
                "Invalid linear speed limits"
            )

        for size in (c.world_width, c.world_height):
            if size is not None and (
                not math.isfinite(size) or size <= 0
            ):
                raise ValueError(
                    "World dimensions must be positive"
                )

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def distance(
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _validate_inputs(
        self,
        robot: RobotState,
        goal: tuple[float, float],
        obstacles: Sequence[Obstacle],
    ) -> None:
        robot_values = (
            robot.x,
            robot.y,
            robot.yaw,
            robot.linear_velocity,
            robot.angular_velocity,
        )

        if not all(map(math.isfinite, robot_values)):
            raise ValueError("Robot state must be finite")

        if (
            len(goal) != 2
            or not all(map(math.isfinite, goal))
        ):
            raise ValueError("Goal must be a finite 2D point")

        for obstacle in obstacles:
            values = (
                obstacle.x,
                obstacle.y,
                obstacle.radius,
                obstacle.vx,
                obstacle.vy,
            )

            if not all(map(math.isfinite, values)):
                raise ValueError(
                    f"Invalid obstacle: {obstacle.obstacle_id}"
                )

            if obstacle.radius < 0:
                raise ValueError(
                    "Obstacle radius cannot be negative"
                )

    def _dynamic_window(
        self,
        robot: RobotState,
    ) -> tuple[float, float, float, float]:
        c = self.config

        dt = c.control_dt

        minimum_speed = (
            -c.max_linear_speed
            if c.allow_reverse
            else c.min_linear_speed
        )

        v_min = max(
            minimum_speed,
            robot.linear_velocity
            - c.max_linear_acceleration * dt,
        )

        v_max = min(
            c.max_linear_speed,
            robot.linear_velocity
            + c.max_linear_acceleration * dt,
        )

        w_min = max(
            -c.max_angular_speed,
            robot.angular_velocity
            - c.max_angular_acceleration * dt,
        )

        w_max = min(
            c.max_angular_speed,
            robot.angular_velocity
            + c.max_angular_acceleration * dt,
        )

        return v_min, v_max, w_min, w_max

    @staticmethod
    def _sample_range(
        lower: float,
        upper: float,
        count: int,
    ) -> list[float]:
        if lower > upper:
            return []

        if abs(upper - lower) <= EPS:
            return [lower]

        return [
            lower + (upper - lower) * i / (count - 1)
            for i in range(count)
        ]

    def _candidate_commands(
        self,
        robot: RobotState,
    ) -> list[VelocityCommand]:
        v_min, v_max, w_min, w_max = (
            self._dynamic_window(robot)
        )

        linear = self._sample_range(
            v_min, v_max, self.config.linear_samples
        )

        angular = self._sample_range(
            w_min, w_max, self.config.angular_samples
        )

        if v_min <= 0 <= v_max:
            linear.append(0.0)

        if w_min <= 0 <= w_max:
            angular.append(0.0)

        linear = sorted(set(round(v, 9) for v in linear))
        angular = sorted(set(round(w, 9) for w in angular))

        return [
            VelocityCommand(v, w)
            for v in linear
            for w in angular
        ]

    def _integrate(
        self,
        pose: TrajectoryPoint,
        command: VelocityCommand,
        dt: float,
    ) -> TrajectoryPoint:
        v = command.linear
        w = command.angular

        new_yaw = self.wrap_angle(pose.yaw + w * dt)

        if abs(w) <= EPS:
            new_x = pose.x + v * math.cos(pose.yaw) * dt
            new_y = pose.y + v * math.sin(pose.yaw) * dt
        else:
            new_x = (
                pose.x
                + v / w
                * (
                    math.sin(pose.yaw + w * dt)
                    - math.sin(pose.yaw)
                )
            )

            new_y = (
                pose.y
                - v / w
                * (
                    math.cos(pose.yaw + w * dt)
                    - math.cos(pose.yaw)
                )
            )

        return TrajectoryPoint(
            time=pose.time + dt,
            x=new_x,
            y=new_y,
            yaw=new_yaw,
        )

    def predict_trajectory(
        self,
        robot: RobotState,
        command: VelocityCommand,
    ) -> list[TrajectoryPoint]:
        c = self.config

        trajectory = [
            TrajectoryPoint(
                time=0.0,
                x=robot.x,
                y=robot.y,
                yaw=robot.yaw,
            )
        ]

        while (
            trajectory[-1].time
            < c.prediction_horizon - EPS
        ):
            dt = min(
                c.prediction_dt,
                c.prediction_horizon - trajectory[-1].time,
            )

            trajectory.append(
                self._integrate(
                    trajectory[-1],
                    command,
                    dt,
                )
            )

        return trajectory

    @staticmethod
    def _obstacle_position(
        obstacle: Obstacle,
        time: float,
    ) -> tuple[float, float]:
        if not obstacle.dynamic:
            return obstacle.x, obstacle.y

        return (
            obstacle.x + obstacle.vx * time,
            obstacle.y + obstacle.vy * time,
        )

    @staticmethod
    def _moving_segment_distance(
        robot_start: tuple[float, float],
        robot_end: tuple[float, float],
        obstacle_start: tuple[float, float],
        obstacle_end: tuple[float, float],
    ) -> float:
        """
        Minimum separation between two
        linearly moving points over a step.
        """

        rx = robot_start[0] - obstacle_start[0]
        ry = robot_start[1] - obstacle_start[1]

        dx = (
            robot_end[0] - robot_start[0]
            - obstacle_end[0] + obstacle_start[0]
        )

        dy = (
            robot_end[1] - robot_start[1]
            - obstacle_end[1] + obstacle_start[1]
        )

        squared_speed = dx * dx + dy * dy

        if squared_speed <= EPS:
            fraction = 0.0
        else:
            fraction = max(
                0.0,
                min(
                    1.0,
                    -(rx * dx + ry * dy) / squared_speed,
                ),
            )

        return math.hypot(
            rx + fraction * dx,
            ry + fraction * dy,
        )

    def _boundary_clearance(
        self,
        x: float,
        y: float,
    ) -> float:
        c = self.config
        margin = c.robot_radius + c.safety_margin

        clearance = math.inf

        if c.world_width is not None:
            clearance = min(
                clearance,
                x - margin,
                c.world_width - x - margin,
            )

        if c.world_height is not None:
            clearance = min(
                clearance,
                y - margin,
                c.world_height - y - margin,
            )

        return clearance

    def _trajectory_safety(
        self,
        trajectory: Sequence[TrajectoryPoint],
        obstacles: Sequence[Obstacle],
    ) -> tuple[bool, float, Optional[str]]:
        c = self.config

        minimum_clearance = math.inf

        for index, pose in enumerate(trajectory):
            boundary = self._boundary_clearance(
                pose.x, pose.y
            )

            minimum_clearance = min(
                minimum_clearance, boundary
            )

            if boundary <= 0:
                return (
                    False,
                    minimum_clearance,
                    "world_boundary",
                )

            for obstacle in obstacles:
                obstacle_now = self._obstacle_position(
                    obstacle,
                    pose.time,
                )

                required = (
                    c.robot_radius
                    + obstacle.radius
                    + c.safety_margin
                )

                clearance = (
                    self.distance(
                        (pose.x, pose.y),
                        obstacle_now,
                    )
                    - required
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance,
                )

                if clearance <= 0:
                    return (
                        False,
                        minimum_clearance,
                        obstacle.obstacle_id,
                    )

                if index == 0:
                    continue

                previous = trajectory[index - 1]

                obstacle_before = (
                    self._obstacle_position(
                        obstacle,
                        previous.time,
                    )
                )

                swept_distance = (
                    self._moving_segment_distance(
                        (previous.x, previous.y),
                        (pose.x, pose.y),
                        obstacle_before,
                        obstacle_now,
                    )
                )

                swept_clearance = (
                    swept_distance - required
                )

                minimum_clearance = min(
                    minimum_clearance,
                    swept_clearance,
                )

                if swept_clearance <= 0:
                    return (
                        False,
                        minimum_clearance,
                        obstacle.obstacle_id,
                    )

        return True, minimum_clearance, None

    def _score(
        self,
        robot: RobotState,
        goal: tuple[float, float],
        command: VelocityCommand,
        trajectory: Sequence[TrajectoryPoint],
        clearance: float,
    ) -> float:
        c = self.config
        final = trajectory[-1]

        initial_distance = self.distance(
            robot.position, goal
        )

        final_distance = self.distance(
            (final.x, final.y), goal
        )

        desired_heading = math.atan2(
            goal[1] - final.y,
            goal[0] - final.x,
        )

        heading_error = abs(
            self.wrap_angle(
                desired_heading - final.yaw
            )
        )

        heading_score = 1.0 - heading_error / math.pi

        progress_score = (
            initial_distance - final_distance
        )

        clearance_score = (
            1.0
            if math.isinf(clearance)
            else 1.0 - math.exp(-max(0.0, clearance))
        )

        speed_score = (
            command.linear / c.max_linear_speed
        )

        smoothness_cost = (
            abs(
                command.linear
                - robot.linear_velocity
            )
            + abs(
                command.angular
                - robot.angular_velocity
            )
        )

        return (
            c.heading_weight * heading_score
            + c.progress_weight * progress_score
            + c.clearance_weight * clearance_score
            + c.speed_weight * speed_score
            - c.smoothness_weight * smoothness_cost
        )

    def evaluate(
        self,
        robot: RobotState,
        goal: tuple[float, float],
        obstacles: Sequence[Obstacle],
        command: VelocityCommand,
    ) -> TrajectoryEvaluation:
        trajectory = self.predict_trajectory(
            robot, command
        )

        safe, clearance, collision_id = (
            self._trajectory_safety(
                trajectory, obstacles
            )
        )

        final = trajectory[-1]

        final_distance = self.distance(
            (final.x, final.y), goal
        )

        score = (
            self._score(
                robot,
                goal,
                command,
                trajectory,
                clearance,
            )
            if safe
            else -math.inf
        )

        return TrajectoryEvaluation(
            safe=safe,
            command=command,
            trajectory=trajectory,
            score=score,
            minimum_clearance=clearance,
            final_distance_to_goal=final_distance,
            collision_obstacle=collision_id,
        )

    def _stuck(
        self,
        robot: RobotState,
    ) -> bool:
        if self.previous_position is None:
            self.previous_position = robot.position
            self.stuck_counter = 0
            return False

        moved = self.distance(
            self.previous_position,
            robot.position,
        )

        if moved >= self.config.stuck_distance:
            self.previous_position = robot.position
            self.stuck_counter = 0
            return False

        self.stuck_counter += 1

        return (
            self.stuck_counter
            >= self.config.stuck_steps
        )

    def emergency_stop(self) -> None:
        if not self.emergency_stop_active:
            self.emergency_stops += 1

        self.emergency_stop_active = True

    def clear_emergency_stop(
        self,
        authorized: bool = False,
    ) -> None:
        if not authorized:
            raise PermissionError(
                "Authorized reset required"
            )

        self.emergency_stop_active = False
        self.previous_position = None
        self.stuck_counter = 0

    def plan(
        self,
        robot: RobotState,
        goal: tuple[float, float],
        obstacles: Sequence[Obstacle],
    ) -> AvoidanceDecision:
        self._validate_inputs(
            robot, goal, obstacles
        )

        self.total_plans += 1

        goal_distance = self.distance(
            robot.position, goal
        )

        stop = VelocityCommand()

        if self.emergency_stop_active:
            return AvoidanceDecision(
                status=AvoidanceStatus.EMERGENCY_STOP,
                command=stop,
                trajectory=[],
                score=-math.inf,
                minimum_clearance=0.0,
                distance_to_goal=goal_distance,
                evaluated_candidates=0,
                safe_candidates=0,
                reason="Emergency stop active",
            )

        if goal_distance <= self.config.goal_tolerance:
            return AvoidanceDecision(
                status=AvoidanceStatus.GOAL_REACHED,
                command=stop,
                trajectory=[],
                score=0.0,
                minimum_clearance=math.inf,
                distance_to_goal=goal_distance,
                evaluated_candidates=0,
                safe_candidates=0,
                reason="Goal reached",
            )

        current_safety = self._trajectory_safety(
            [
                TrajectoryPoint(
                    time=0.0,
                    x=robot.x,
                    y=robot.y,
                    yaw=robot.yaw,
                )
            ],
            obstacles,
        )

        if not current_safety[0]:
            self.emergency_stop()

            return AvoidanceDecision(
                status=AvoidanceStatus.EMERGENCY_STOP,
                command=stop,
                trajectory=[],
                score=-math.inf,
                minimum_clearance=current_safety[1],
                distance_to_goal=goal_distance,
                evaluated_candidates=0,
                safe_candidates=0,
                reason="Unsafe current robot position",
            )

        if self._stuck(robot):
            return AvoidanceDecision(
                status=AvoidanceStatus.STUCK,
                command=stop,
                trajectory=[],
                score=-math.inf,
                minimum_clearance=current_safety[1],
                distance_to_goal=goal_distance,
                evaluated_candidates=0,
                safe_candidates=0,
                reason="Robot movement below threshold",
            )

        commands = self._candidate_commands(robot)

        evaluations = [
            self.evaluate(
                robot,
                goal,
                obstacles,
                command,
            )
            for command in commands
        ]

        safe_options = [
            evaluation
            for evaluation in evaluations
            if evaluation.safe
        ]

        if not safe_options:
            self.blocked_plans += 1

            return AvoidanceDecision(
                status=AvoidanceStatus.BLOCKED,
                command=stop,
                trajectory=[],
                score=-math.inf,
                minimum_clearance=current_safety[1],
                distance_to_goal=goal_distance,
                evaluated_candidates=len(commands),
                safe_candidates=0,
                reason="No safe local trajectory",
            )

        best = max(
            safe_options,
            key=lambda item: (
                item.score,
                item.minimum_clearance,
            ),
        )

        status = (
            AvoidanceStatus.AVOIDING
            if best.minimum_clearance
            <= self.config.warning_clearance
            else AvoidanceStatus.CLEAR
        )

        return AvoidanceDecision(
            status=status,
            command=best.command,
            trajectory=best.trajectory,
            score=best.score,
            minimum_clearance=best.minimum_clearance,
            distance_to_goal=goal_distance,
            evaluated_candidates=len(commands),
            safe_candidates=len(safe_options),
            reason="Safe trajectory selected",
            diagnostics={
                "predicted_final_distance": (
                    best.final_distance_to_goal
                ),
                "moving_obstacles": sum(
                    obstacle.dynamic
                    for obstacle in obstacles
                ),
            },
        )

    def reset(self) -> None:
        self.emergency_stop_active = False
        self.previous_position = None
        self.stuck_counter = 0
        self.total_plans = 0
        self.blocked_plans = 0
        self.emergency_stops = 0

    def statistics(self) -> dict[str, Any]:
        return {
            "total_plans": self.total_plans,
            "blocked_plans": self.blocked_plans,
            "emergency_stops": self.emergency_stops,
            "stuck_counter": self.stuck_counter,
            "emergency_stop_active": (
                self.emergency_stop_active
            ),
        }


if __name__ == "__main__":
    engine = ObstacleAvoidanceEngine(
        AvoidanceConfig(
            world_width=10.0,
            world_height=10.0,
        )
    )

    robot = RobotState(
        x=1.0,
        y=1.0,
        yaw=0.0,
    )

    goal = (8.0, 8.0)

    obstacles = [
        Obstacle(
            obstacle_id="static_1",
            x=3.0,
            y=3.0,
            radius=0.6,
        ),
        Obstacle(
            obstacle_id="moving_1",
            x=5.0,
            y=2.0,
            radius=0.3,
            vx=-0.1,
            vy=0.15,
            dynamic=True,
        ),
    ]

    dt = engine.config.control_dt

    for step in range(200):
        decision = engine.plan(
            robot,
            goal,
            obstacles,
        )

        print(
            f"Step {step:03d} | "
            f"Status: {decision.status.value} | "
            f"v={decision.command.linear:.3f} | "
            f"w={decision.command.angular:.3f} | "
            f"Goal distance={decision.distance_to_goal:.3f}"
        )

        if decision.status in (
            AvoidanceStatus.GOAL_REACHED,
            AvoidanceStatus.EMERGENCY_STOP,
            AvoidanceStatus.BLOCKED,
            AvoidanceStatus.STUCK,
        ):
            break

        current = TrajectoryPoint(
            time=0.0,
            x=robot.x,
            y=robot.y,
            yaw=robot.yaw,
        )

        next_pose = engine._integrate(
            current,
            decision.command,
            dt,
        )

        robot.x = next_pose.x
        robot.y = next_pose.y
        robot.yaw = next_pose.yaw
        robot.linear_velocity = decision.command.linear
        robot.angular_velocity = decision.command.angular

        for obstacle in obstacles:
            if obstacle.dynamic:
                obstacle.x += obstacle.vx * dt
                obstacle.y += obstacle.vy * dt

    print("Final position:", robot.position)
    print("Statistics:", engine.statistics())
```
