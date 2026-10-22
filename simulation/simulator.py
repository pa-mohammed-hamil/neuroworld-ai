
```python
"""
NeuroWorld AI
File: neuroworld/environment/simulator.py

Autonomous World Simulation Engine

Features:
- Multi-robot 2D simulation
- Scenario loading and resetting
- Differential-drive robot kinematics
- Navigation agent integration
- Dynamic obstacle updates
- Collision and boundary detection
- Emergency stop and pause/resume
- Simulation clock
- Robot trajectory tracking
- Goal detection
- Occupancy grid updates
- Matplotlib rendering
- PNG, GIF and MP4 export
- Simulation event history
- JSON simulation reports
- Reproducible simulation setup

Dependencies:
    pip install numpy matplotlib pillow torch

Optional MP4:
    pip install imageio imageio-ffmpeg

This is a software simulator, not a
physical robot or vehicle controller.
"""

from __future__ import annotations

import copy
import json
import math
import time

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroworld.environment.scenario import (
    Scenario,
    ScenarioConfig,
    ScenarioRobot,
)

from neuroworld.environment.renderer import (
    WorldRenderer,
    RendererConfig,
    WorldRenderState,
    RenderRobot,
    RenderGoal,
    RenderObstacle,
    RenderTrajectory,
)

from neuroworld.agents.navigation_agent import (
    NavigationAgent,
    NavigationAgentConfig,
    NavigationGoal,
    RobotPose,
    OccupancyGrid,
)


class SimulatorStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"


class RobotSimulationStatus(str, Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    GOAL_REACHED = "goal_reached"
    BLOCKED = "blocked"
    COLLISION = "collision"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SimulatorConfig:
    timestep: float = 0.1
    max_steps: int = 1000

    real_time: bool = False
    real_time_factor: float = 1.0

    enable_navigation: bool = True
    enable_collision_detection: bool = True
    stop_on_collision: bool = True

    enable_dynamic_obstacles: bool = True

    enable_rendering: bool = False
    render_every_n_steps: int = 1
    capture_frames: bool = False

    max_linear_speed: float = 0.6
    max_angular_speed: float = 1.2

    goal_tolerance: float = 0.2
    robot_safety_margin: float = 0.1

    history_capacity: int = 10000
    seed: int = 42


@dataclass
class RobotControl:
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


@dataclass
class RobotSimulationState:
    robot_id: str
    x: float
    y: float
    yaw: float

    linear_velocity: float = 0.0
    angular_velocity: float = 0.0

    status: RobotSimulationStatus = (
        RobotSimulationStatus.IDLE
    )

    distance_traveled: float = 0.0

    goal_reached: bool = False
    collision_count: int = 0

    trajectory: List[Tuple[float, float]] = field(
        default_factory=list
    )


@dataclass
class SimulationEvent:
    step: int
    simulation_time: float
    event_type: str
    entity_id: str
    description: str


@dataclass
class SimulationStepResult:
    step: int
    simulation_time: float
    status: SimulatorStatus

    robots: Dict[str, RobotSimulationState]

    collisions: List[Tuple[str, str]]

    all_goals_reached: bool

    events: List[SimulationEvent] = field(
        default_factory=list
    )


@dataclass
class SimulationReport:
    status: str
    total_steps: int
    simulation_time: float
    wall_time_seconds: float

    total_collisions: int
    goals_reached: int
    total_robots: int

    robot_statistics: Dict[str, Dict[str, Any]]

    event_count: int


class WorldSimulator:
    """
    Main NeuroWorld simulation coordinator.

    Architecture:

        Scenario
           |
           v
        Occupancy Grid
           |
           v
        Navigation Agents
           |
           v
        Robot Commands
           |
           v
        Kinematic Simulation
           |
           v
        Collision Detection
           |
           v
        World State Update
           |
           v
        Renderer / Reports

    Important:
        Navigation commands are proposals.
        Motion is checked against obstacles
        before it is committed.
    """

    def __init__(
        self,
        scenario: Scenario,
        config: Optional[SimulatorConfig] = None,
        renderer: Optional[WorldRenderer] = None,
        navigation_config: Optional[
            NavigationAgentConfig
        ] = None,
    ):
        self.scenario = scenario
        self.config = config or SimulatorConfig()

        self._validate_config()

        self.renderer = renderer

        if (
            self.renderer is None
            and self.config.enable_rendering
        ):
            self.renderer = WorldRenderer(
                RendererConfig(
                    world_width=(
                        scenario.config.world_width
                    ),
                    world_height=(
                        scenario.config.world_height
                    ),
                    interactive=False,
                )
            )

        self.navigation_config = (
            navigation_config
            or NavigationAgentConfig(
                robot_radius=(
                    scenario.config.robot_radius
                ),
                safety_margin=(
                    self.config.robot_safety_margin
                ),
                max_linear_speed=(
                    self.config.max_linear_speed
                ),
                max_angular_speed=(
                    self.config.max_angular_speed
                ),
            )
        )

        self.status = SimulatorStatus.IDLE

        self.step_count = 0
        self.simulation_time = 0.0
        self.wall_start_time: Optional[float] = None
        self.wall_elapsed_seconds = 0.0

        self.robot_states: Dict[
            str, RobotSimulationState
        ] = {}

        self.navigation_agents: Dict[
            str, NavigationAgent
        ] = {}

        self.manual_controls: Dict[
            str, RobotControl
        ] = {}

        self.events: List[SimulationEvent] = []
        self.history: List[SimulationStepResult] = []

        self.total_collisions = 0
        self.emergency_stop_active = False

        self._initialized = False

    def _validate_config(self) -> None:
        cfg = self.config

        if not math.isfinite(cfg.timestep) or (
            cfg.timestep <= 0
        ):
            raise ValueError(
                "timestep must be positive and finite"
            )

        if cfg.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        if (
            cfg.max_linear_speed <= 0
            or cfg.max_angular_speed <= 0
        ):
            raise ValueError(
                "Speed limits must be positive"
            )

        if cfg.real_time_factor <= 0:
            raise ValueError(
                "real_time_factor must be positive"
            )

        if cfg.render_every_n_steps <= 0:
            raise ValueError(
                "render_every_n_steps must be positive"
            )

        if cfg.history_capacity <= 0:
            raise ValueError(
                "history_capacity must be positive"
            )

        if cfg.goal_tolerance < 0:
            raise ValueError(
                "goal_tolerance must be nonnegative"
            )

        if cfg.robot_safety_margin < 0:
            raise ValueError(
                "robot_safety_margin must be nonnegative"
            )

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:
        return (
            (angle + math.pi)
            % (2 * math.pi)
        ) - math.pi

    def _record_event(
        self,
        event_type: str,
        entity_id: str,
        description: str,
    ) -> SimulationEvent:
        event = SimulationEvent(
            step=self.step_count,
            simulation_time=self.simulation_time,
            event_type=event_type,
            entity_id=entity_id,
            description=description,
        )

        self.events.append(event)

        return event

    def _create_occupancy_grid(
        self,
    ) -> OccupancyGrid:
        return OccupancyGrid(
            grid=self.scenario.to_occupancy_grid(),
            resolution=(
                self.scenario.config.map_resolution
            ),
            origin=(0.0, 0.0),
        )

    def _goal_for_robot(
        self,
        robot_id: str,
    ) -> Optional[NavigationGoal]:
        for goal in self.scenario.state.goals:
            if goal.assigned_robot == robot_id:
                return NavigationGoal(
                    x=goal.x,
                    y=goal.y,
                    tolerance=goal.tolerance,
                )

        return None

    def reset(
        self,
        regenerate_scenario: bool = True,
    ) -> Dict[str, RobotSimulationState]:
        """
        Reset simulation.

        regenerate_scenario=True:
            Rebuild the scenario from its seed.

        regenerate_scenario=False:
            Use the currently loaded scenario.
        """

        if regenerate_scenario:
            self.scenario.reset()

        self.step_count = 0
        self.simulation_time = 0.0
        self.wall_elapsed_seconds = 0.0
        self.wall_start_time = None

        self.events.clear()
        self.history.clear()

        self.robot_states.clear()
        self.navigation_agents.clear()
        self.manual_controls.clear()

        self.total_collisions = 0
        self.emergency_stop_active = False

        self.scenario.state.simulation_time = 0.0

        occupancy = self._create_occupancy_grid()

        for robot in self.scenario.state.robots:
            state = RobotSimulationState(
                robot_id=robot.robot_id,
                x=robot.x,
                y=robot.y,
                yaw=robot.yaw,
                trajectory=[
                    (robot.x, robot.y)
                ],
            )

            self.robot_states[
                robot.robot_id
            ] = state

            self.manual_controls[
                robot.robot_id
            ] = RobotControl()

            if self.config.enable_navigation:
                agent = NavigationAgent(
                    occupancy_grid=occupancy,
                    config=copy.deepcopy(
                        self.navigation_config
                    ),
                )

                goal = self._goal_for_robot(
                    robot.robot_id
                )

                if goal is not None:
                    planned = agent.set_goal(
                        goal,
                        RobotPose(
                            x=robot.x,
                            y=robot.y,
                            yaw=robot.yaw,
                        ),
                    )

                    state.status = (
                        RobotSimulationStatus.NAVIGATING
                        if planned
                        else RobotSimulationStatus.BLOCKED
                    )

                self.navigation_agents[
                    robot.robot_id
                ] = agent

        self.status = SimulatorStatus.IDLE
        self._initialized = True

        self._record_event(
            "simulation_reset",
            "simulator",
            "Simulation initialized",
        )

        return copy.deepcopy(
            self.robot_states
        )

    def set_robot_control(
        self,
        robot_id: str,
        linear_velocity: float,
        angular_velocity: float,
    ) -> None:
        """
        Set manual velocity commands.

        Used when navigation is disabled,
        or when an external controller
        supplies robot commands.
        """

        if robot_id not in self.robot_states:
            raise KeyError(
                f"Unknown robot: {robot_id}"
            )

        if not all(
            math.isfinite(value)
            for value in (
                linear_velocity,
                angular_velocity,
            )
        ):
            raise ValueError(
                "Velocity commands must be finite"
            )

        self.manual_controls[
            robot_id
        ] = RobotControl(
            linear_velocity=float(
                np.clip(
                    linear_velocity,
                    -self.config.max_linear_speed,
                    self.config.max_linear_speed,
                )
            ),
            angular_velocity=float(
                np.clip(
                    angular_velocity,
                    -self.config.max_angular_speed,
                    self.config.max_angular_speed,
                )
            ),
        )

    def _navigation_control(
        self,
        robot_id: str,
    ) -> RobotControl:
        state = self.robot_states[robot_id]

        if state.goal_reached:
            return RobotControl()

        agent = self.navigation_agents.get(
            robot_id
        )

        if agent is None:
            return self.manual_controls.get(
                robot_id,
                RobotControl(),
            )

        decision = agent.step(
            RobotPose(
                x=state.x,
                y=state.y,
                yaw=state.yaw,
            ),
            emergency_stop=(
                self.emergency_stop_active
            ),
        )

        navigation_status = (
            decision.status.value
        )

        if navigation_status == "goal_reached":
            state.goal_reached = True
            state.status = (
                RobotSimulationStatus.GOAL_REACHED
            )
            return RobotControl()

        if navigation_status in (
            "blocked",
            "stuck",
            "emergency_stop",
        ):
            state.status = (
                RobotSimulationStatus.BLOCKED
                if navigation_status
                != "emergency_stop"
                else RobotSimulationStatus.EMERGENCY_STOP
            )
            return RobotControl()

        state.status = (
            RobotSimulationStatus.NAVIGATING
        )

        return RobotControl(
            linear_velocity=(
                decision.command.linear
            ),
            angular_velocity=(
                decision.command.angular
            ),
        )

    def _integrate_motion(
        self,
        state: RobotSimulationState,
        control: RobotControl,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Differential-drive unicycle model.

        x' = v cos(yaw)
        y' = v sin(yaw)
        yaw' = angular_velocity

        Uses exact integration for
        constant velocities over dt.
        """

        v = float(
            np.clip(
                control.linear_velocity,
                -self.config.max_linear_speed,
                self.config.max_linear_speed,
            )
        )

        w = float(
            np.clip(
                control.angular_velocity,
                -self.config.max_angular_speed,
                self.config.max_angular_speed,
            )
        )

        new_yaw = self._normalize_angle(
            state.yaw + w * dt
        )

        if abs(w) < 1e-8:
            new_x = (
                state.x
                + v * math.cos(state.yaw) * dt
            )

            new_y = (
                state.y
                + v * math.sin(state.yaw) * dt
            )
        else:
            new_x = (
                state.x
                + (v / w)
                * (
                    math.sin(
                        state.yaw + w * dt
                    )
                    - math.sin(state.yaw)
                )
            )

            new_y = (
                state.y
                - (v / w)
                * (
                    math.cos(
                        state.yaw + w * dt
                    )
                    - math.cos(state.yaw)
                )
            )

        return new_x, new_y, new_yaw

    def _is_position_safe(
        self,
        robot_id: str,
        x: float,
        y: float,
    ) -> bool:
        """
        Check world boundaries and collisions
        with obstacles and other robots.
        """

        state = self.robot_states[robot_id]

        scenario_robot = next(
            robot
            for robot in self.scenario.state.robots
            if robot.robot_id == robot_id
        )

        radius = (
            scenario_robot.radius
            + self.config.robot_safety_margin
        )

        if (
            x - radius < 0
            or x + radius
            > self.scenario.config.world_width
            or y - radius < 0
            or y + radius
            > self.scenario.config.world_height
        ):
            return False

        for obstacle in self.scenario.state.obstacles:
            distance = math.hypot(
                x - obstacle.x,
                y - obstacle.y,
            )

            if distance < (
                radius + obstacle.radius
            ):
                return False

        for other_id, other in (
            self.robot_states.items()
        ):
            if other_id == robot_id:
                continue

            other_robot = next(
                robot
                for robot in self.scenario.state.robots
                if robot.robot_id == other_id
            )

            distance = math.hypot(
                x - other.x,
                y - other.y,
            )

            if distance < (
                radius + other_robot.radius
            ):
                return False

        return True

    def _move_robot(
        self,
        robot_id: str,
        control: RobotControl,
        dt: float,
    ) -> bool:
        """
        Move a robot if the proposed
        trajectory remains collision-free.

        Samples intermediate positions to
        reduce obstacle tunneling.
        """

        state = self.robot_states[robot_id]

        new_x, new_y, new_yaw = (
            self._integrate_motion(
                state,
                control,
                dt,
            )
        )

        distance = math.hypot(
            new_x - state.x,
            new_y - state.y,
        )

        resolution = (
            self.scenario.config.map_resolution
        )

        samples = max(
            2,
            math.ceil(
                distance
                / max(
                    resolution / 2,
                    1e-4,
                )
            ),
        )

        if self.config.enable_collision_detection:
            for index in range(
                1,
                samples + 1,
            ):
                fraction = index / samples

                sample_x = (
                    state.x
                    + fraction
                    * (new_x - state.x)
                )

                sample_y = (
                    state.y
                    + fraction
                    * (new_y - state.y)
                )

                if not self._is_position_safe(
                    robot_id,
                    sample_x,
                    sample_y,
                ):
                    state.status = (
                        RobotSimulationStatus.COLLISION
                    )

                    state.collision_count += 1
                    self.total_collisions += 1

                    self._record_event(
                        "collision_prevented",
                        robot_id,
                        "Proposed motion rejected "
                        "due to obstacle or boundary",
                    )

                    return False

        state.x = new_x
        state.y = new_y
        state.yaw = new_yaw

        state.linear_velocity = (
            control.linear_velocity
        )

        state.angular_velocity = (
            control.angular_velocity
        )

        state.distance_traveled += distance

        state.trajectory.append(
            (new_x, new_y)
        )

        return True

    def _sync_scenario_robots(self) -> None:
        """
        Copy simulated poses into the
        scenario entity representation.
        """

        for robot in self.scenario.state.robots:
            state = self.robot_states.get(
                robot.robot_id
            )

            if state is None:
                continue

            robot.x = state.x
            robot.y = state.y
            robot.yaw = state.yaw

    def _update_navigation_maps(self) -> None:
        """
        Update occupancy maps after
        dynamic obstacle movement.
        """

        occupancy = self._create_occupancy_grid()

        for agent in self.navigation_agents.values():
            agent.grid = occupancy

    def _check_goals(self) -> None:
        for robot_id, state in (
            self.robot_states.items()
        ):
            if state.goal_reached:
                continue

            goal = self._goal_for_robot(
                robot_id
            )

            if goal is None:
                continue

            distance = math.hypot(
                state.x - goal.x,
                state.y - goal.y,
            )

            tolerance = (
                goal.tolerance
                if goal.tolerance is not None
                else self.config.goal_tolerance
            )

            if distance <= tolerance:
                state.goal_reached = True
                state.status = (
                    RobotSimulationStatus.GOAL_REACHED
                )

                self._record_event(
                    "goal_reached",
                    robot_id,
                    "Robot reached assigned goal",
                )

    def _all_goals_reached(self) -> bool:
        if not self.robot_states:
            return False

        assigned_robot_ids = {
            goal.assigned_robot
            for goal in self.scenario.state.goals
            if goal.assigned_robot is not None
        }

        if not assigned_robot_ids:
            return False

        return all(
            self.robot_states[robot_id].goal_reached
            for robot_id in assigned_robot_ids
            if robot_id in self.robot_states
        )

    def _render(
        self,
    ) -> None:
        if self.renderer is None:
            return

        render_state = self.scenario.to_render_state()

        render_state.trajectories = [
            RenderTrajectory(
                points=list(
                    state.trajectory
                ),
                label=f"{robot_id} trajectory",
            )
            for robot_id, state
            in self.robot_states.items()
        ]

        render_state.simulation_time = (
            self.simulation_time
        )

        self.renderer.render(
            render_state,
            capture_frame=(
                self.config.capture_frames
            ),
        )

    def start(self) -> None:
        if not self._initialized:
            self.reset()

        if self.emergency_stop_active:
            raise RuntimeError(
                "Cannot start while emergency "
                "stop is active"
            )

        if self.status == SimulatorStatus.COMPLETED:
            raise RuntimeError(
                "Simulation completed; reset first"
            )

        self.status = SimulatorStatus.RUNNING

        if self.wall_start_time is None:
            self.wall_start_time = (
                time.perf_counter()
            )

    def pause(self) -> None:
        if self.status == SimulatorStatus.RUNNING:
            self.status = SimulatorStatus.PAUSED

            self._record_event(
                "simulation_paused",
                "simulator",
                "Simulation paused",
            )

    def resume(self) -> None:
        if self.status != SimulatorStatus.PAUSED:
            raise RuntimeError(
                "Simulation is not paused"
            )

        if self.emergency_stop_active:
            raise RuntimeError(
                "Emergency stop active"
            )

        self.status = SimulatorStatus.RUNNING

    def emergency_stop(self) -> None:
        self.emergency_stop_active = True
        self.status = (
            SimulatorStatus.EMERGENCY_STOP
        )

        for state in self.robot_states.values():
            state.linear_velocity = 0.0
            state.angular_velocity = 0.0
            state.status = (
                RobotSimulationStatus.EMERGENCY_STOP
            )

        self._record_event(
            "emergency_stop",
            "simulator",
            "Emergency stop activated",
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
        self.status = SimulatorStatus.PAUSED

        self._record_event(
            "emergency_stop_cleared",
            "simulator",
            "Emergency stop cleared",
        )

    def step(self) -> SimulationStepResult:
        """
        Advance the simulation one timestep.
        """

        if not self._initialized:
            raise RuntimeError(
                "Call reset() first"
            )

        if self.status != SimulatorStatus.RUNNING:
            raise RuntimeError(
                "Simulator must be running"
            )

        cfg = self.config
        dt = cfg.timestep

        step_start = time.perf_counter()
        previous_event_count = len(
            self.events
        )

        if cfg.enable_dynamic_obstacles:
            self.scenario.step(dt)
            self._update_navigation_maps()
        else:
            self.scenario.state.simulation_time += dt

        for robot_id, state in (
            self.robot_states.items()
        ):
            if state.goal_reached:
                continue

            if cfg.enable_navigation:
                control = self._navigation_control(
                    robot_id
                )
            else:
                control = self.manual_controls.get(
                    robot_id,
                    RobotControl(),
                )

            moved = self._move_robot(
                robot_id,
                control,
                dt,
            )

            if (
                not moved
                and cfg.stop_on_collision
            ):
                self.emergency_stop()
                break

        self._sync_scenario_robots()

        self.step_count += 1
        self.simulation_time += dt

        self._check_goals()

        all_reached = (
            self._all_goals_reached()
        )

        if (
            all_reached
            and not self.emergency_stop_active
        ):
            self.status = (
                SimulatorStatus.COMPLETED
            )

            self._record_event(
                "simulation_completed",
                "simulator",
                "All assigned goals reached",
            )

        elif (
            self.step_count >= cfg.max_steps
            and not self.emergency_stop_active
        ):
            self.status = (
                SimulatorStatus.COMPLETED
            )

            self._record_event(
                "max_steps_reached",
                "simulator",
                "Maximum simulation steps reached",
            )

        if (
            cfg.enable_rendering
            and self.step_count
            % cfg.render_every_n_steps == 0
        ):
            self._render()

        result = SimulationStepResult(
            step=self.step_count,
            simulation_time=(
                self.simulation_time
            ),
            status=self.status,
            robots=copy.deepcopy(
                self.robot_states
            ),
            collisions=(
                self.scenario.check_collisions()
            ),
            all_goals_reached=all_reached,
            events=copy.deepcopy(
                self.events[
                    previous_event_count:
                ]
            ),
        )

        self.history.append(result)

        if len(self.history) > cfg.history_capacity:
            self.history.pop(0)

        if cfg.real_time:
            elapsed = (
                time.perf_counter()
                - step_start
            )

            target_duration = (
                dt / cfg.real_time_factor
            )

            remaining = (
                target_duration - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)

        if self.wall_start_time is not None:
            self.wall_elapsed_seconds = (
                time.perf_counter()
                - self.wall_start_time
            )

        return result

    def run(
        self,
        max_steps: Optional[int] = None,
    ) -> SimulationReport:
        """
        Run until:
        - All goals reached
        - Maximum steps reached
        - Emergency stop
        """

        if not self._initialized:
            self.reset()

        self.start()

        limit = (
            max_steps
            if max_steps is not None
            else self.config.max_steps
        )

        if limit <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        for _ in range(limit):
            if self.status != SimulatorStatus.RUNNING:
                break

            self.step()

        if self.status == SimulatorStatus.RUNNING:
            self.pause()

        return self.report()

    def report(self) -> SimulationReport:
        robot_statistics = {}

        for robot_id, state in (
            self.robot_states.items()
        ):
            robot_statistics[robot_id] = {
                "status": state.status.value,
                "goal_reached": (
                    state.goal_reached
                ),
                "distance_traveled": (
                    state.distance_traveled
                ),
                "collision_count": (
                    state.collision_count
                ),
                "trajectory_points": len(
                    state.trajectory
                ),
                "final_pose": {
                    "x": state.x,
                    "y": state.y,
                    "yaw": state.yaw,
                },
            }

        return SimulationReport(
            status=self.status.value,
            total_steps=self.step_count,
            simulation_time=self.simulation_time,
            wall_time_seconds=(
                self.wall_elapsed_seconds
            ),
            total_collisions=(
                self.total_collisions
            ),
            goals_reached=sum(
                state.goal_reached
                for state in self.robot_states.values()
            ),
            total_robots=len(
                self.robot_states
            ),
            robot_statistics=robot_statistics,
            event_count=len(self.events),
        )

    def export_report(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "report": asdict(
                self.report()
            ),
            "events": [
                asdict(event)
                for event in self.events
            ],
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def save_image(
        self,
        path: str | Path,
    ) -> None:
        if self.renderer is None:
            raise RuntimeError(
                "Renderer is not configured"
            )

        self._render()
        self.renderer.save_image(path)

    def save_gif(
        self,
        path: str | Path,
        fps: int = 10,
    ) -> None:
        if self.renderer is None:
            raise RuntimeError(
                "Renderer is not configured"
            )

        self.renderer.save_gif(
            path,
            fps=fps,
        )

    def save_mp4(
        self,
        path: str | Path,
        fps: int = 20,
    ) -> None:
        if self.renderer is None:
            raise RuntimeError(
                "Renderer is not configured"
            )

        self.renderer.save_mp4(
            path,
            fps=fps,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":

    scenario = Scenario(
        ScenarioConfig(
            name="NeuroWorld Robot Simulation",
            scenario_type="random",
            world_width=10.0,
            world_height=10.0,
            map_resolution=0.1,
            num_robots=2,
            num_obstacles=10,
            seed=42,
        )
    )

    simulator = WorldSimulator(
        scenario=scenario,
        config=SimulatorConfig(
            timestep=0.1,
            max_steps=300,
            enable_navigation=True,
            enable_collision_detection=True,
            stop_on_collision=True,
            enable_dynamic_obstacles=True,
            enable_rendering=True,
            capture_frames=True,
            render_every_n_steps=3,
        ),
    )

    simulator.reset()

    print(
        "Initial robots:",
        list(simulator.robot_states),
    )

    report = simulator.run()

    print(
        "Simulation status:",
        report.status,
    )

    print(
        "Total steps:",
        report.total_steps,
    )

    print(
        "Goals reached:",
        report.goals_reached,
    )

    print(
        "Collisions:",
        report.total_collisions,
    )

    simulator.export_report(
        "artifacts/simulation_report.json"
    )

    simulator.save_image(
        "artifacts/simulation_final.png"
    )

    if (
        simulator.renderer is not None
        and simulator.renderer.frames
    ):
        simulator.save_gif(
            "artifacts/simulation.gif",
            fps=10,
        )

    simulator.close()
```
