
```python
"""
NeuroWorld AI
File: neuroworld/environment/navigation.py

Navigation Environment

Features:
- 2D navigation environment
- Robot pose and goal management
- Occupancy-grid integration
- NavigationAgent integration
- Collision-aware motion
- Differential-drive kinematics
- Waypoint and path tracking
- Dynamic obstacle updates
- Goal-reached detection
- Navigation progress metrics
- Stuck detection
- Emergency stop
- Navigation history
- Simulation-ready step interface

Dependencies:
    pip install numpy

This module simulates navigation.
It does not send commands to physical robots.
"""

from __future__ import annotations

import math
import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroworld.agents.navigation_agent import (
    NavigationAgent,
    NavigationAgentConfig,
    NavigationGoal,
    RobotPose,
    OccupancyGrid,
)

from neuroworld.physics.collision import (
    CircleBody,
    CollisionConfig,
    CollisionEngine,
)

from neuroworld.environment.scenario import (
    Scenario,
)


Point2D = Tuple[float, float]


class NavigationEnvironmentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    GOAL_REACHED = "goal_reached"
    BLOCKED = "blocked"
    STUCK = "stuck"
    COLLISION_PREVENTED = "collision_prevented"
    EMERGENCY_STOP = "emergency_stop"
    TIMEOUT = "timeout"


@dataclass
class NavigationEnvironmentConfig:
    timestep: float = 0.1
    max_steps: int = 1000

    robot_radius: float = 0.25
    safety_margin: float = 0.1

    max_linear_speed: float = 0.6
    max_angular_speed: float = 1.2

    goal_tolerance: float = 0.2

    stuck_distance_threshold: float = 0.05
    stuck_steps: int = 30

    enable_collision_detection: bool = True
    stop_on_collision: bool = True

    update_dynamic_obstacles: bool = True
    replan_on_obstacle_change: bool = True

    history_capacity: int = 10000


@dataclass
class NavigationObservation:
    x: float
    y: float
    yaw: float

    goal_x: float
    goal_y: float

    distance_to_goal: float
    heading_error: float

    linear_velocity: float
    angular_velocity: float

    step: int
    simulation_time: float

    status: NavigationEnvironmentStatus


@dataclass
class NavigationStepResult:
    observation: NavigationObservation

    reward: float

    terminated: bool
    truncated: bool

    collision_prevented: bool = False
    replanned: bool = False

    info: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class NavigationHistoryEntry:
    step: int
    simulation_time: float

    x: float
    y: float
    yaw: float

    linear_velocity: float
    angular_velocity: float

    distance_to_goal: float
    reward: float

    status: str


@dataclass
class NavigationEnvironmentReport:
    status: str

    total_steps: int
    simulation_time: float

    goal_reached: bool
    final_distance_to_goal: float

    distance_traveled: float

    collision_preventions: int
    replans: int

    trajectory_points: int


class NavigationEnvironment:
    """
    Single-robot navigation environment.

    Uses:
        Scenario -> world geometry
        NavigationAgent -> navigation decisions
        CollisionEngine -> motion validation

    Example:
        env = NavigationEnvironment(scenario)
        observation = env.reset()

        while True:
            result = env.step()

            if result.terminated or result.truncated:
                break
    """

    def __init__(
        self,
        scenario: Scenario,
        config: Optional[
            NavigationEnvironmentConfig
        ] = None,
        navigation_config: Optional[
            NavigationAgentConfig
        ] = None,
    ):
        self.scenario = scenario

        self.config = (
            config
            or NavigationEnvironmentConfig()
        )

        self._validate_config()

        self.navigation_config = (
            navigation_config
            or NavigationAgentConfig(
                robot_radius=self.config.robot_radius,
                safety_margin=self.config.safety_margin,
                max_linear_speed=(
                    self.config.max_linear_speed
                ),
                max_angular_speed=(
                    self.config.max_angular_speed
                ),
                goal_tolerance=(
                    self.config.goal_tolerance
                ),
            )
        )

        self.collision_engine = CollisionEngine(
            CollisionConfig(
                safety_margin=(
                    self.config.safety_margin
                )
            )
        )

        self.navigation_agent: Optional[
            NavigationAgent
        ] = None

        self.robot_id: Optional[str] = None
        self.pose: Optional[RobotPose] = None
        self.goal: Optional[NavigationGoal] = None

        self.status = (
            NavigationEnvironmentStatus.IDLE
        )

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.step_count = 0
        self.simulation_time = 0.0

        self.distance_traveled = 0.0
        self.collision_preventions = 0
        self.replans = 0

        self.stuck_counter = 0
        self.stuck_reference: Optional[
            Point2D
        ] = None

        self.trajectory: List[
            Point2D
        ] = []

        self.history: List[
            NavigationHistoryEntry
        ] = []

        self.emergency_stop_active = False

    def _validate_config(self) -> None:
        cfg = self.config

        for name in (
            "timestep",
            "robot_radius",
            "max_linear_speed",
            "max_angular_speed",
        ):
            value = getattr(cfg, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        if cfg.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        if cfg.safety_margin < 0:
            raise ValueError(
                "safety_margin must be nonnegative"
            )

        if cfg.goal_tolerance < 0:
            raise ValueError(
                "goal_tolerance must be nonnegative"
            )

        if cfg.stuck_steps <= 0:
            raise ValueError(
                "stuck_steps must be positive"
            )

        if cfg.stuck_distance_threshold < 0:
            raise ValueError(
                "stuck_distance_threshold "
                "must be nonnegative"
            )

        if cfg.history_capacity <= 0:
            raise ValueError(
                "history_capacity must be positive"
            )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (
            (angle + math.pi)
            % (2.0 * math.pi)
        ) - math.pi

    def _distance_to_goal(self) -> float:
        assert self.pose is not None
        assert self.goal is not None

        return math.hypot(
            self.goal.x - self.pose.x,
            self.goal.y - self.pose.y,
        )

    def _heading_error(self) -> float:
        assert self.pose is not None
        assert self.goal is not None

        desired_heading = math.atan2(
            self.goal.y - self.pose.y,
            self.goal.x - self.pose.x,
        )

        return self._wrap_angle(
            desired_heading - self.pose.yaw
        )

    def _goal_reached(self) -> bool:
        assert self.goal is not None

        tolerance = (
            self.goal.tolerance
            if self.goal.tolerance is not None
            else self.config.goal_tolerance
        )

        return (
            self._distance_to_goal()
            <= tolerance
        )

    def _create_grid(self) -> OccupancyGrid:
        return OccupancyGrid(
            grid=self.scenario.to_occupancy_grid(),
            resolution=(
                self.scenario.config.map_resolution
            ),
            origin=(0.0, 0.0),
        )

    def _observation(
        self,
    ) -> NavigationObservation:
        assert self.pose is not None
        assert self.goal is not None

        return NavigationObservation(
            x=self.pose.x,
            y=self.pose.y,
            yaw=self.pose.yaw,
            goal_x=self.goal.x,
            goal_y=self.goal.y,
            distance_to_goal=(
                self._distance_to_goal()
            ),
            heading_error=(
                self._heading_error()
            ),
            linear_velocity=(
                self.linear_velocity
            ),
            angular_velocity=(
                self.angular_velocity
            ),
            step=self.step_count,
            simulation_time=(
                self.simulation_time
            ),
            status=self.status,
        )

    def reset(
        self,
        robot_id: Optional[str] = None,
        regenerate_scenario: bool = False,
    ) -> NavigationObservation:
        """
        Reset navigation for a selected robot.

        If regenerate_scenario=True,
        rebuild the scenario first.
        """

        if regenerate_scenario:
            self.scenario.reset()

        if not self.scenario.state.robots:
            raise RuntimeError(
                "Scenario contains no robots"
            )

        if robot_id is None:
            robot = (
                self.scenario.state.robots[0]
            )
        else:
            robot = next(
                (
                    item
                    for item
                    in self.scenario.state.robots
                    if item.robot_id == robot_id
                ),
                None,
            )

            if robot is None:
                raise KeyError(
                    f"Robot not found: {robot_id}"
                )

        self.robot_id = robot.robot_id

        self.pose = RobotPose(
            x=robot.x,
            y=robot.y,
            yaw=robot.yaw,
        )

        assigned_goal = next(
            (
                item
                for item
                in self.scenario.state.goals
                if item.assigned_robot
                == self.robot_id
            ),
            None,
        )

        if assigned_goal is None:
            raise RuntimeError(
                "Selected robot has no assigned goal"
            )

        self.goal = NavigationGoal(
            x=assigned_goal.x,
            y=assigned_goal.y,
            tolerance=assigned_goal.tolerance,
        )

        self.navigation_agent = NavigationAgent(
            occupancy_grid=self._create_grid(),
            config=self.navigation_config,
        )

        self.step_count = 0
        self.simulation_time = 0.0

        self.distance_traveled = 0.0
        self.collision_preventions = 0
        self.replans = 0

        self.stuck_counter = 0
        self.stuck_reference = (
            self.pose.x,
            self.pose.y,
        )

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.trajectory = [
            (self.pose.x, self.pose.y)
        ]

        self.history.clear()

        self.emergency_stop_active = False

        if self._goal_reached():
            self.status = (
                NavigationEnvironmentStatus.GOAL_REACHED
            )
        else:
            planned = (
                self.navigation_agent.set_goal(
                    self.goal,
                    self.pose,
                )
            )

            self.status = (
                NavigationEnvironmentStatus.NAVIGATING
                if planned
                else NavigationEnvironmentStatus.BLOCKED
            )

        return self._observation()

    def _robot_body(
        self,
    ) -> CircleBody:
        assert self.pose is not None

        return CircleBody(
            body_id=self.robot_id or "robot",
            x=self.pose.x,
            y=self.pose.y,
            radius=self.config.robot_radius,
        )

    def _obstacle_bodies(
        self,
    ) -> List[CircleBody]:
        return [
            CircleBody(
                body_id=obstacle.obstacle_id,
                x=obstacle.x,
                y=obstacle.y,
                radius=obstacle.radius,
                vx=obstacle.velocity_x,
                vy=obstacle.velocity_y,
                is_dynamic=obstacle.dynamic,
            )
            for obstacle
            in self.scenario.state.obstacles
        ]

    def _motion_safe(
        self,
        new_x: float,
        new_y: float,
    ) -> bool:
        assert self.pose is not None

        if not self.config.enable_collision_detection:
            return True

        return self.collision_engine.safe_motion(
            robot=self._robot_body(),
            target_position=(new_x, new_y),
            obstacles=self._obstacle_bodies(),
            world_width=(
                self.scenario.config.world_width
            ),
            world_height=(
                self.scenario.config.world_height
            ),
        )

    def _integrate(
        self,
        linear: float,
        angular: float,
    ) -> RobotPose:
        """
        Exact constant-velocity integration
        of the unicycle motion model.
        """

        assert self.pose is not None

        dt = self.config.timestep

        linear = float(
            np.clip(
                linear,
                -self.config.max_linear_speed,
                self.config.max_linear_speed,
            )
        )

        angular = float(
            np.clip(
                angular,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )

        x = self.pose.x
        y = self.pose.y
        yaw = self.pose.yaw

        next_yaw = self._wrap_angle(
            yaw + angular * dt
        )

        if abs(angular) < 1e-8:
            next_x = (
                x + linear * math.cos(yaw) * dt
            )

            next_y = (
                y + linear * math.sin(yaw) * dt
            )
        else:
            next_x = (
                x
                + linear / angular
                * (
                    math.sin(next_yaw)
                    - math.sin(yaw)
                )
            )

            next_y = (
                y
                - linear / angular
                * (
                    math.cos(next_yaw)
                    - math.cos(yaw)
                )
            )

        return RobotPose(
            x=next_x,
            y=next_y,
            yaw=next_yaw,
        )

    def _check_stuck(self) -> bool:
        assert self.pose is not None

        if self.stuck_reference is None:
            self.stuck_reference = (
                self.pose.x,
                self.pose.y,
            )
            return False

        distance = math.hypot(
            self.pose.x - self.stuck_reference[0],
            self.pose.y - self.stuck_reference[1],
        )

        if (
            distance
            >= self.config.stuck_distance_threshold
        ):
            self.stuck_counter = 0
            self.stuck_reference = (
                self.pose.x,
                self.pose.y,
            )
            return False

        self.stuck_counter += 1

        return (
            self.stuck_counter
            >= self.config.stuck_steps
        )

    def _sync_scenario_robot(self) -> None:
        assert self.pose is not None

        for robot in self.scenario.state.robots:
            if robot.robot_id == self.robot_id:
                robot.x = self.pose.x
                robot.y = self.pose.y
                robot.yaw = self.pose.yaw
                break

    def _record_history(
        self,
        reward: float,
    ) -> None:
        assert self.pose is not None

        self.history.append(
            NavigationHistoryEntry(
                step=self.step_count,
                simulation_time=(
                    self.simulation_time
                ),
                x=self.pose.x,
                y=self.pose.y,
                yaw=self.pose.yaw,
                linear_velocity=(
                    self.linear_velocity
                ),
                angular_velocity=(
                    self.angular_velocity
                ),
                distance_to_goal=(
                    self._distance_to_goal()
                ),
                reward=reward,
                status=self.status.value,
            )
        )

        if (
            len(self.history)
            > self.config.history_capacity
        ):
            self.history.pop(0)

    def emergency_stop(self) -> None:
        self.emergency_stop_active = True

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.status = (
            NavigationEnvironmentStatus.EMERGENCY_STOP
        )

    def clear_emergency_stop(
        self,
        authorized: bool = False,
    ) -> None:
        if not authorized:
            raise PermissionError(
                "Authorization required"
            )

        self.emergency_stop_active = False

        self.status = (
            NavigationEnvironmentStatus.IDLE
        )

        if (
            self.navigation_agent is not None
            and self.pose is not None
            and self.goal is not None
        ):
            planned = (
                self.navigation_agent.set_goal(
                    self.goal,
                    self.pose,
                )
            )

            self.status = (
                NavigationEnvironmentStatus.NAVIGATING
                if planned
                else NavigationEnvironmentStatus.BLOCKED
            )

    def step(self) -> NavigationStepResult:
        """
        Advance navigation by one timestep.

        Reward:
            +10.0 for reaching goal
            +progress toward goal
            -0.01 per step
            -5.0 for prevented collision
            -1.0 when stuck
        """

        if (
            self.pose is None
            or self.goal is None
            or self.navigation_agent is None
        ):
            raise RuntimeError(
                "Call reset() before step()"
            )

        if self.emergency_stop_active:
            return NavigationStepResult(
                observation=self._observation(),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={
                    "reason": "emergency_stop"
                },
            )

        if self.status in (
            NavigationEnvironmentStatus.GOAL_REACHED,
            NavigationEnvironmentStatus.TIMEOUT,
            NavigationEnvironmentStatus.STUCK,
        ):
            raise RuntimeError(
                "Episode finished; call reset()"
            )

        previous_distance = (
            self._distance_to_goal()
        )

        replanned = False
        collision_prevented = False

        if self.config.update_dynamic_obstacles:
            self.scenario.step(
                self.config.timestep
            )

            if self.config.replan_on_obstacle_change:
                self.navigation_agent.grid = (
                    self._create_grid()
                )

                replanned = (
                    self.navigation_agent.plan_path(
                        self.pose
                    )
                )

                if replanned:
                    self.replans += 1

        decision = self.navigation_agent.step(
            self.pose,
            emergency_stop=False,
        )

        navigation_status = (
            decision.status.value
        )

        linear = 0.0
        angular = 0.0

        if navigation_status == "goal_reached":
            self.status = (
                NavigationEnvironmentStatus.GOAL_REACHED
            )

        elif navigation_status in (
            "blocked",
            "stuck",
        ):
            self.status = (
                NavigationEnvironmentStatus.BLOCKED
                if navigation_status == "blocked"
                else NavigationEnvironmentStatus.STUCK
            )

        else:
            linear = float(
                decision.command.linear
            )

            angular = float(
                decision.command.angular
            )

            candidate = self._integrate(
                linear,
                angular,
            )

            if self._motion_safe(
                candidate.x,
                candidate.y,
            ):
                traveled = math.hypot(
                    candidate.x - self.pose.x,
                    candidate.y - self.pose.y,
                )

                self.pose = candidate
                self.distance_traveled += traveled

                self.linear_velocity = linear
                self.angular_velocity = angular

                self.status = (
                    NavigationEnvironmentStatus.NAVIGATING
                )

            else:
                collision_prevented = True
                self.collision_preventions += 1

                self.linear_velocity = 0.0
                self.angular_velocity = 0.0

                self.status = (
                    NavigationEnvironmentStatus
                    .COLLISION_PREVENTED
                )

                if self.config.stop_on_collision:
                    self.emergency_stop()

        self.step_count += 1
        self.simulation_time += (
            self.config.timestep
        )

        self._sync_scenario_robot()

        self.trajectory.append(
            (self.pose.x, self.pose.y)
        )

        current_distance = (
            self._distance_to_goal()
        )

        progress = (
            previous_distance
            - current_distance
        )

        reward = (
            progress - 0.01
        )

        if collision_prevented:
            reward -= 5.0

        if self._goal_reached():
            self.status = (
                NavigationEnvironmentStatus.GOAL_REACHED
            )
            reward += 10.0

        if (
            self.status
            == NavigationEnvironmentStatus.NAVIGATING
            and self._check_stuck()
        ):
            self.status = (
                NavigationEnvironmentStatus.STUCK
            )
            reward -= 1.0

        terminated = self.status in (
            NavigationEnvironmentStatus.GOAL_REACHED,
            NavigationEnvironmentStatus.STUCK,
            NavigationEnvironmentStatus.EMERGENCY_STOP,
        )

        truncated = (
            self.step_count
            >= self.config.max_steps
        )

        if truncated and not terminated:
            self.status = (
                NavigationEnvironmentStatus.TIMEOUT
            )

        self._record_history(
            reward
        )

        return NavigationStepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            collision_prevented=(
                collision_prevented
            ),
            replanned=replanned,
            info={
                "navigation_status": (
                    navigation_status
                ),
                "distance_to_goal": (
                    current_distance
                ),
                "progress": progress,
                "collision_preventions": (
                    self.collision_preventions
                ),
                "replans": self.replans,
            },
        )

    def run(
        self,
        max_steps: Optional[int] = None,
    ) -> NavigationEnvironmentReport:
        """
        Run navigation until termination.
        """

        limit = (
            self.config.max_steps
            if max_steps is None
            else max_steps
        )

        if limit <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        for _ in range(limit):
            result = self.step()

            if (
                result.terminated
                or result.truncated
            ):
                break

        return self.report()

    def report(
        self,
    ) -> NavigationEnvironmentReport:
        if self.pose is None:
            raise RuntimeError(
                "Environment not initialized"
            )

        return NavigationEnvironmentReport(
            status=self.status.value,
            total_steps=self.step_count,
            simulation_time=self.simulation_time,
            goal_reached=(
                self.status
                == NavigationEnvironmentStatus
                .GOAL_REACHED
            ),
            final_distance_to_goal=(
                self._distance_to_goal()
            ),
            distance_traveled=(
                self.distance_traveled
            ),
            collision_preventions=(
                self.collision_preventions
            ),
            replans=self.replans,
            trajectory_points=len(
                self.trajectory
            ),
        )


if __name__ == "__main__":

    from neuroworld.environment.scenario import (
        ScenarioConfig,
    )

    scenario = Scenario(
        ScenarioConfig(
            name="NeuroWorld Navigation Test",
            scenario_type="random",
            world_width=10.0,
            world_height=10.0,
            num_robots=1,
            num_obstacles=10,
            seed=42,
        )
    )

    scenario.reset()

    environment = NavigationEnvironment(
        scenario=scenario,
        config=NavigationEnvironmentConfig(
            timestep=0.1,
            max_steps=500,
            robot_radius=0.25,
            safety_margin=0.1,
        ),
    )

    observation = environment.reset()

    print(
        "Initial observation:",
        observation,
    )

    for _ in range(500):
        result = environment.step()

        if result.observation.step % 25 == 0:
            print(
                "Step:",
                result.observation.step,
                "Distance:",
                round(
                    result.observation.distance_to_goal,
                    3,
                ),
                "Status:",
                result.observation.status.value,
            )

        if result.terminated or result.truncated:
            break

    print(
        "Navigation report:",
        environment.report(),
    )
```
