
"""
NeuroWorld AI
File: neuroworld/agents/navigation_agent.py

Autonomous Navigation Agent

Features:
- 2D occupancy-grid navigation
- A* global path planning
- Obstacle inflation for robot clearance
- Dynamic obstacle updates
- Goal-conditioned navigation
- Waypoint tracking
- Local collision-aware action selection
- Replanning when paths become blocked
- Navigation progress monitoring
- Stuck detection
- Emergency stop decisions
- Optional world-model action scoring
- JSON-compatible navigation reports

Dependencies:
    Python 3.10+
    pip install numpy torch

Important:
    This is a simulation/prototyping controller.
    It is not a certified robotics safety system.

    The occupancy map must use a consistent
    coordinate frame and metric resolution.

    A separate safety controller must validate
    actions before hardware execution.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn


GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]


class NavigationStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    REPLANNING = "replanning"
    GOAL_REACHED = "goal_reached"
    BLOCKED = "blocked"
    STUCK = "stuck"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class NavigationAgentConfig:
    map_resolution: float = 0.1

    robot_radius: float = 0.25
    safety_margin: float = 0.15

    obstacle_threshold: float = 0.5

    allow_diagonal: bool = True
    prevent_corner_cutting: bool = True

    goal_tolerance: float = 0.2
    waypoint_tolerance: float = 0.15

    max_linear_speed: float = 0.6
    max_angular_speed: float = 1.2

    heading_gain: float = 2.0
    speed_gain: float = 1.0

    heading_slowdown_angle: float = 0.8

    lookahead_distance: float = 0.4

    stuck_distance_threshold: float = 0.05
    stuck_steps: int = 20

    local_prediction_horizon: float = 1.0
    local_prediction_steps: int = 10

    replan_on_obstacle: bool = True

    world_model_weight: float = 0.0


@dataclass
class RobotPose:
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class NavigationGoal:
    x: float
    y: float
    tolerance: Optional[float] = None


@dataclass
class VelocityCommand:
    linear: float
    angular: float


@dataclass
class NavigationDecision:
    status: NavigationStatus

    command: VelocityCommand

    target_waypoint: Optional[WorldPoint]

    distance_to_goal: float

    path: List[WorldPoint]

    replanned: bool = False

    reason: str = ""


@dataclass
class NavigationReport:
    status: str
    current_pose: Dict[str, float]
    goal: Optional[Dict[str, float]]
    path_length: float
    waypoint_count: int
    distance_to_goal: Optional[float]
    replans: int
    steps: int


class OccupancyGrid:
    """
    2D occupancy grid.

    Grid values:
        0.0 = free
        1.0 = occupied

    Coordinates:
        array[row, column]

        row increases with world y
        column increases with world x

    World coordinate:
        x = origin_x + (column + 0.5) * resolution
        y = origin_y + (row + 0.5) * resolution

    Unknown cells should be marked occupied
    before use in safety-sensitive planning.
    """

    def __init__(
        self,
        grid: np.ndarray,
        resolution: float = 0.1,
        origin: WorldPoint = (0.0, 0.0),
        obstacle_threshold: float = 0.5,
    ):
        array = np.asarray(
            grid,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Occupancy grid must be 2D"
            )

        if min(array.shape) < 1:
            raise ValueError(
                "Occupancy grid cannot be empty"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Occupancy grid contains NaN or Inf"
            )

        if resolution <= 0:
            raise ValueError(
                "Resolution must be positive"
            )

        if not 0 <= obstacle_threshold <= 1:
            raise ValueError(
                "obstacle_threshold must be in [0,1]"
            )

        self.grid = array.copy()

        self.resolution = float(resolution)

        self.origin = (
            float(origin[0]),
            float(origin[1]),
        )

        self.obstacle_threshold = (
            obstacle_threshold
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def in_bounds(
        self,
        cell: GridCell,
    ) -> bool:
        row, col = cell

        return (
            0 <= row < self.grid.shape[0]
            and 0 <= col < self.grid.shape[1]
        )

    def world_to_grid(
        self,
        x: float,
        y: float,
    ) -> GridCell:
        col = math.floor(
            (x - self.origin[0])
            / self.resolution
        )

        row = math.floor(
            (y - self.origin[1])
            / self.resolution
        )

        return int(row), int(col)

    def grid_to_world(
        self,
        cell: GridCell,
    ) -> WorldPoint:
        row, col = cell

        x = (
            self.origin[0]
            + (col + 0.5)
            * self.resolution
        )

        y = (
            self.origin[1]
            + (row + 0.5)
            * self.resolution
        )

        return float(x), float(y)

    def is_occupied(
        self,
        cell: GridCell,
    ) -> bool:
        if not self.in_bounds(cell):
            return True

        return bool(
            self.grid[cell]
            >= self.obstacle_threshold
        )

    def set_obstacle(
        self,
        cell: GridCell,
        occupied: bool = True,
    ) -> None:
        if not self.in_bounds(cell):
            raise ValueError(
                "Obstacle cell is outside map"
            )

        self.grid[cell] = (
            1.0 if occupied else 0.0
        )

    def update_obstacles(
        self,
        cells: Sequence[GridCell],
        occupied: bool = True,
    ) -> None:
        for cell in cells:
            self.set_obstacle(
                cell,
                occupied,
            )

    def inflated_obstacles(
        self,
        clearance: float,
    ) -> np.ndarray:
        """
        Inflate occupied cells by robot
        radius plus safety margin.

        This is a circular approximation
        of robot footprint clearance.
        """

        if clearance < 0:
            raise ValueError(
                "Clearance cannot be negative"
            )

        occupied = (
            self.grid
            >= self.obstacle_threshold
        )

        radius_cells = math.ceil(
            clearance / self.resolution
        )

        if radius_cells == 0:
            return occupied.copy()

        inflated = occupied.copy()

        obstacle_cells = np.argwhere(
            occupied
        )

        offsets = []

        for dr in range(
            -radius_cells,
            radius_cells + 1,
        ):
            for dc in range(
                -radius_cells,
                radius_cells + 1,
            ):
                distance = math.hypot(
                    dr,
                    dc,
                ) * self.resolution

                if distance <= clearance + 1e-9:
                    offsets.append(
                        (dr, dc)
                    )

        height, width = occupied.shape

        for row, col in obstacle_cells:
            for dr, dc in offsets:
                nr = int(row + dr)
                nc = int(col + dc)

                if (
                    0 <= nr < height
                    and 0 <= nc < width
                ):
                    inflated[nr, nc] = True

        return inflated


class AStarPlanner:
    """
    A* global path planner.

    Supports:
        - 4-connected movement
        - 8-connected movement
        - Euclidean heuristic
        - Diagonal corner-cut prevention
    """

    def __init__(
        self,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
    ):
        self.allow_diagonal = (
            allow_diagonal
        )

        self.prevent_corner_cutting = (
            prevent_corner_cutting
        )

    @staticmethod
    def heuristic(
        a: GridCell,
        b: GridCell,
    ) -> float:
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    def neighbors(
        self,
        cell: GridCell,
        obstacles: np.ndarray,
    ) -> List[Tuple[GridCell, float]]:
        row, col = cell

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        if self.allow_diagonal:
            directions.extend(
                [
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
            )

        height, width = obstacles.shape

        result = []

        for dr, dc in directions:
            nr = row + dr
            nc = col + dc

            if not (
                0 <= nr < height
                and 0 <= nc < width
            ):
                continue

            if obstacles[nr, nc]:
                continue

            if (
                dr != 0
                and dc != 0
                and self.prevent_corner_cutting
            ):
                if (
                    obstacles[row + dr, col]
                    or obstacles[row, col + dc]
                ):
                    continue

            cost = math.hypot(
                dr,
                dc,
            )

            result.append(
                ((nr, nc), cost)
            )

        return result

    def plan(
        self,
        start: GridCell,
        goal: GridCell,
        obstacles: np.ndarray,
    ) -> List[GridCell]:
        """
        Find shortest grid path using A*.

        Returns an empty list if no path exists.
        """

        height, width = obstacles.shape

        def valid(cell: GridCell) -> bool:
            row, col = cell

            return (
                0 <= row < height
                and 0 <= col < width
                and not obstacles[row, col]
            )

        if not valid(start) or not valid(goal):
            return []

        if start == goal:
            return [start]

        frontier = []

        heapq.heappush(
            frontier,
            (0.0, start),
        )

        came_from: Dict[
            GridCell,
            GridCell,
        ] = {}

        g_score = {
            start: 0.0
        }

        closed = set()

        while frontier:
            _, current = heapq.heappop(
                frontier
            )

            if current in closed:
                continue

            if current == goal:
                path = [current]

                while current in came_from:
                    current = came_from[current]
                    path.append(current)

                path.reverse()

                return path

            closed.add(current)

            for neighbor, cost in self.neighbors(
                current,
                obstacles,
            ):
                tentative = (
                    g_score[current] + cost
                )

                if tentative < g_score.get(
                    neighbor,
                    float("inf"),
                ):
                    came_from[neighbor] = current

                    g_score[neighbor] = (
                        tentative
                    )

                    priority = (
                        tentative
                        + self.heuristic(
                            neighbor,
                            goal,
                        )
                    )

                    heapq.heappush(
                        frontier,
                        (
                            priority,
                            neighbor,
                        ),
                    )

        return []


class NavigationAgent:
    """
    Autonomous navigation decision agent.

    Workflow:

        Pose + Goal
             |
             v
        Occupancy Grid
             |
             v
        Obstacle Inflation
             |
             v
        A* Global Planning
             |
             v
        Waypoint Tracking
             |
             v
        Local Motion Safety Check
             |
             v
        Velocity Command

    Dynamic obstacle changes trigger replanning.

    The controller returns suggested velocity
    commands. It never sends them to hardware.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        config: Optional[
            NavigationAgentConfig
        ] = None,
        world_model: Optional[
            nn.Module
        ] = None,
    ):
        self.grid = occupancy_grid

        self.config = (
            config or NavigationAgentConfig()
        )

        self.world_model = world_model

        self._validate_config()

        self.planner = AStarPlanner(
            allow_diagonal=(
                self.config.allow_diagonal
            ),
            prevent_corner_cutting=(
                self.config.prevent_corner_cutting
            ),
        )

        self.status = NavigationStatus.IDLE

        self.goal: Optional[
            NavigationGoal
        ] = None

        self.path: List[
            WorldPoint
        ] = []

        self.path_cells: List[
            GridCell
        ] = []

        self.waypoint_index = 0

        self.steps = 0
        self.replans = 0

        self._previous_pose: Optional[
            RobotPose
        ] = None

        self._stuck_counter = 0

    def _validate_config(self) -> None:
        cfg = self.config

        for name in (
            "map_resolution",
            "max_linear_speed",
            "max_angular_speed",
            "lookahead_distance",
            "local_prediction_horizon",
        ):
            if getattr(cfg, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        for name in (
            "robot_radius",
            "safety_margin",
            "goal_tolerance",
            "waypoint_tolerance",
            "stuck_distance_threshold",
        ):
            if getattr(cfg, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )

        if cfg.stuck_steps <= 0:
            raise ValueError(
                "stuck_steps must be positive"
            )

        if cfg.local_prediction_steps <= 0:
            raise ValueError(
                "local_prediction_steps "
                "must be positive"
            )

        if cfg.heading_slowdown_angle <= 0:
            raise ValueError(
                "heading_slowdown_angle "
                "must be positive"
            )

        if cfg.world_model_weight < 0:
            raise ValueError(
                "world_model_weight "
                "must be nonnegative"
            )

    @staticmethod
    def distance(
        a: WorldPoint,
        b: WorldPoint,
    ) -> float:
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    @staticmethod
    def normalize_angle(
        angle: float,
    ) -> float:
        return (
            (angle + math.pi)
            % (2.0 * math.pi)
        ) - math.pi

    def _obstacles(self) -> np.ndarray:
        clearance = (
            self.config.robot_radius
            + self.config.safety_margin
        )

        return self.grid.inflated_obstacles(
            clearance
        )

    def _goal_tolerance(self) -> float:
        if self.goal is None:
            return self.config.goal_tolerance

        return (
            self.goal.tolerance
            if self.goal.tolerance is not None
            else self.config.goal_tolerance
        )

    def _distance_to_goal(
        self,
        pose: RobotPose,
    ) -> float:
        if self.goal is None:
            return float("inf")

        return self.distance(
            (pose.x, pose.y),
            (self.goal.x, self.goal.y),
        )

    def set_goal(
        self,
        goal: NavigationGoal,
        current_pose: RobotPose,
    ) -> bool:
        """
        Set a navigation goal and plan path.

        Returns True when a path is found.
        """

        if not all(
            math.isfinite(v)
            for v in (
                goal.x,
                goal.y,
                current_pose.x,
                current_pose.y,
                current_pose.yaw,
            )
        ):
            raise ValueError(
                "Goal and pose must be finite"
            )

        if (
            goal.tolerance is not None
            and goal.tolerance < 0
        ):
            raise ValueError(
                "Goal tolerance must be nonnegative"
            )

        self.goal = goal

        self._previous_pose = None
        self._stuck_counter = 0

        return self.plan_path(
            current_pose
        )

    def plan_path(
        self,
        current_pose: RobotPose,
    ) -> bool:
        """
        Compute an A* path from current pose
        to the current goal.
        """

        if self.goal is None:
            self.status = NavigationStatus.IDLE
            return False

        self.status = NavigationStatus.PLANNING

        obstacles = self._obstacles()

        start = self.grid.world_to_grid(
            current_pose.x,
            current_pose.y,
        )

        goal = self.grid.world_to_grid(
            self.goal.x,
            self.goal.y,
        )

        cells = self.planner.plan(
            start,
            goal,
            obstacles,
        )

        if not cells:
            self.path = []
            self.path_cells = []

            self.status = NavigationStatus.BLOCKED

            return False

        self.path_cells = cells

        self.path = [
            self.grid.grid_to_world(cell)
            for cell in cells
        ]

        self.waypoint_index = (
            1 if len(self.path) > 1 else 0
        )

        self.status = (
            NavigationStatus.NAVIGATING
        )

        return True

    def update_obstacles(
        self,
        occupied_cells: Sequence[
            GridCell
        ] = (),
        cleared_cells: Sequence[
            GridCell
        ] = (),
    ) -> None:
        """
        Update dynamic obstacle observations.

        Existing map cells are modified.
        Replanning occurs on the next step.
        """

        self.grid.update_obstacles(
            occupied_cells,
            occupied=True,
        )

        self.grid.update_obstacles(
            cleared_cells,
            occupied=False,
        )

    def _path_is_blocked(self) -> bool:
        if not self.path_cells:
            return True

        obstacles = self._obstacles()

        remaining = self.path_cells[
            self.waypoint_index:
        ]

        for cell in remaining:
            if not self.grid.in_bounds(cell):
                return True

            if obstacles[cell]:
                return True

        return False

    def _advance_waypoint(
        self,
        pose: RobotPose,
    ) -> None:
        """
        Advance to the next waypoint after
        reaching the current waypoint.
        """

        while (
            self.waypoint_index
            < len(self.path) - 1
        ):
            waypoint = self.path[
                self.waypoint_index
            ]

            distance = self.distance(
                (pose.x, pose.y),
                waypoint,
            )

            if (
                distance
                > self.config.waypoint_tolerance
            ):
                break

            self.waypoint_index += 1

    def _select_waypoint(
        self,
        pose: RobotPose,
    ) -> Optional[WorldPoint]:
        """
        Select a target waypoint.

        Lookahead is limited to the current
        waypoint to avoid skipping corners
        or crossing obstacles.
        """

        if not self.path:
            return None

        self._advance_waypoint(
            pose
        )

        index = min(
            self.waypoint_index,
            len(self.path) - 1,
        )

        return self.path[index]

    def _velocity_to_waypoint(
        self,
        pose: RobotPose,
        waypoint: WorldPoint,
    ) -> VelocityCommand:
        """
        Proportional waypoint controller.
        """

        cfg = self.config

        dx = waypoint[0] - pose.x
        dy = waypoint[1] - pose.y

        distance = math.hypot(
            dx,
            dy,
        )

        desired_heading = math.atan2(
            dy,
            dx,
        )

        heading_error = self.normalize_angle(
            desired_heading - pose.yaw
        )

        angular = (
            cfg.heading_gain
            * heading_error
        )

        angular = max(
            -cfg.max_angular_speed,
            min(
                cfg.max_angular_speed,
                angular,
            ),
        )

        heading_factor = max(
            0.0,
            1.0
            - abs(heading_error)
            / cfg.heading_slowdown_angle,
        )

        linear = min(
            cfg.max_linear_speed,
            cfg.speed_gain * distance,
        )

        linear *= heading_factor

        return VelocityCommand(
            linear=float(linear),
            angular=float(angular),
        )

    def _simulate_command(
        self,
        pose: RobotPose,
        command: VelocityCommand,
    ) -> List[RobotPose]:
        """
        Predict motion with a simple
        unicycle kinematic model.

        x_next = x + v*cos(yaw)*dt
        y_next = y + v*sin(yaw)*dt
        yaw_next = yaw + omega*dt

        This does not model wheel slip,
        latency, acceleration or dynamics.
        """

        cfg = self.config

        steps = cfg.local_prediction_steps

        dt = (
            cfg.local_prediction_horizon
            / steps
        )

        x = pose.x
        y = pose.y
        yaw = pose.yaw

        predicted = []

        for _ in range(steps):
            x += (
                command.linear
                * math.cos(yaw)
                * dt
            )

            y += (
                command.linear
                * math.sin(yaw)
                * dt
            )

            yaw = self.normalize_angle(
                yaw
                + command.angular * dt
            )

            predicted.append(
                RobotPose(
                    x=x,
                    y=y,
                    yaw=yaw,
                )
            )

        return predicted

    def _command_is_safe(
        self,
        pose: RobotPose,
        command: VelocityCommand,
    ) -> bool:
        """
        Check predicted motion against the
        inflated occupancy grid.

        This is a discrete sampled check,
        not continuous collision certification.
        """

        obstacles = self._obstacles()

        predicted = self._simulate_command(
            pose,
            command,
        )

        for future_pose in predicted:
            cell = self.grid.world_to_grid(
                future_pose.x,
                future_pose.y,
            )

            if not self.grid.in_bounds(cell):
                return False

            if obstacles[cell]:
                return False

        return True

    def _safe_command(
        self,
        pose: RobotPose,
        desired: VelocityCommand,
    ) -> VelocityCommand:
        """
        Reduce forward velocity until a
        collision-free predicted command
        is found.

        Returns a zero-velocity stop if
        no candidate is safe.
        """

        candidates = [
            1.0,
            0.75,
            0.5,
            0.25,
            0.0,
        ]

        for scale in candidates:
            candidate = VelocityCommand(
                linear=(
                    desired.linear * scale
                ),
                angular=desired.angular,
            )

            if self._command_is_safe(
                pose,
                candidate,
            ):
                return candidate

        return VelocityCommand(
            linear=0.0,
            angular=0.0,
        )

    def _check_stuck(
        self,
        pose: RobotPose,
        commanded_motion: bool,
    ) -> bool:
        """
        Detect lack of position progress
        while translational motion is requested.
        """

        if self._previous_pose is None:
            self._previous_pose = (
                RobotPose(
                    pose.x,
                    pose.y,
                    pose.yaw,
                )
            )

            return False

        movement = self.distance(
            (
                self._previous_pose.x,
                self._previous_pose.y,
            ),
            (pose.x, pose.y),
        )

        if (
            commanded_motion
            and movement
            < self.config.stuck_distance_threshold
        ):
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        self._previous_pose = RobotPose(
            pose.x,
            pose.y,
            pose.yaw,
        )

        return (
            self._stuck_counter
            >= self.config.stuck_steps
        )

    def step(
        self,
        pose: RobotPose,
        emergency_stop: bool = False,
    ) -> NavigationDecision:
        """
        Run one navigation decision cycle.

        Inputs:
            Current robot pose
            Emergency stop flag

        Output:
            NavigationDecision containing
            suggested linear/angular velocities.
        """

        self.steps += 1

        stop = VelocityCommand(
            linear=0.0,
            angular=0.0,
        )

        if emergency_stop:
            self.status = (
                NavigationStatus.EMERGENCY_STOP
            )

            return NavigationDecision(
                status=self.status,
                command=stop,
                target_waypoint=None,
                distance_to_goal=(
                    self._distance_to_goal(pose)
                ),
                path=list(self.path),
                reason="Emergency stop requested",
            )

        if self.goal is None:
            self.status = NavigationStatus.IDLE

            return NavigationDecision(
                status=self.status,
                command=stop,
                target_waypoint=None,
                distance_to_goal=float("inf"),
                path=[],
                reason="No navigation goal",
            )

        distance_to_goal = (
            self._distance_to_goal(pose)
        )

        if (
            distance_to_goal
            <= self._goal_tolerance()
        ):
            self.status = (
                NavigationStatus.GOAL_REACHED
            )

            return NavigationDecision(
                status=self.status,
                command=stop,
                target_waypoint=None,
                distance_to_goal=distance_to_goal,
                path=list(self.path),
                reason="Goal reached",
            )

        replanned = False

        if (
            not self.path
            or (
                self.config.replan_on_obstacle
                and self._path_is_blocked()
            )
        ):
            self.status = (
                NavigationStatus.REPLANNING
            )

            self.replans += 1
            replanned = True

            if not self.plan_path(pose):
                return NavigationDecision(
                    status=self.status,
                    command=stop,
                    target_waypoint=None,
                    distance_to_goal=distance_to_goal,
                    path=[],
                    replanned=True,
                    reason="No collision-free path",
                )

        waypoint = self._select_waypoint(
            pose
        )

        if waypoint is None:
            self.status = NavigationStatus.BLOCKED

            return NavigationDecision(
                status=self.status,
                command=stop,
                target_waypoint=None,
                distance_to_goal=distance_to_goal,
                path=[],
                reason="No valid waypoint",
            )

        desired = self._velocity_to_waypoint(
            pose,
            waypoint,
        )

        command = self._safe_command(
            pose,
            desired,
        )

        if (
            desired.linear > 1e-6
            and command.linear <= 1e-6
            and abs(command.angular) <= 1e-6
        ):
            self.status = NavigationStatus.BLOCKED

            return NavigationDecision(
                status=self.status,
                command=stop,
                target_waypoint=waypoint,
                distance_to_goal=distance_to_goal,
                path=list(self.path),
                replanned=replanned,
                reason="Local motion blocked",
            )

        if self._check_stuck(
            pose,
            commanded_motion=(
                command.linear > 1e-4
            ),
        ):
            self.status = NavigationStatus.STUCK

            return NavigationDecision(
                status=self.status,
                command=stop,
                target_waypoint=waypoint,
                distance_to_goal=distance_to_goal,
                path=list(self.path),
                replanned=replanned,
                reason="Robot appears stuck",
            )

        self.status = (
            NavigationStatus.NAVIGATING
        )

        return NavigationDecision(
            status=self.status,
            command=command,
            target_waypoint=waypoint,
            distance_to_goal=distance_to_goal,
            path=list(self.path),
            replanned=replanned,
            reason="Navigation command generated",
        )

    def reset(self) -> None:
        """
        Reset navigation session.
        """

        self.status = NavigationStatus.IDLE

        self.goal = None
        self.path = []
        self.path_cells = []

        self.waypoint_index = 0

        self.steps = 0
        self.replans = 0

        self._previous_pose = None
        self._stuck_counter = 0

    def path_length(self) -> float:
        """
        Calculate total planned path distance.
        """

        if len(self.path) < 2:
            return 0.0

        return sum(
            self.distance(
                self.path[index],
                self.path[index + 1],
            )
            for index in range(
                len(self.path) - 1
            )
        )

    def report(
        self,
        pose: RobotPose,
    ) -> NavigationReport:
        """
        Generate navigation status report.
        """

        goal = None

        if self.goal is not None:
            goal = asdict(self.goal)

        distance = (
            self._distance_to_goal(pose)
            if self.goal is not None
            else None
        )

        return NavigationReport(
            status=self.status.value,
            current_pose=asdict(pose),
            goal=goal,
            path_length=self.path_length(),
            waypoint_count=len(self.path),
            distance_to_goal=distance,
            replans=self.replans,
            steps=self.steps,
        )


if __name__ == "__main__":

    # -----------------------------------------
    # Example: Indoor Robot Navigation
    # -----------------------------------------

    grid_array = np.zeros(
        (60, 60),
        dtype=np.float32,
    )

    # Wall with a doorway.
    grid_array[10:50, 30] = 1.0
    grid_array[27:34, 30] = 0.0

    occupancy_grid = OccupancyGrid(
        grid=grid_array,
        resolution=0.1,
        origin=(0.0, 0.0),
    )

    config = NavigationAgentConfig(
        map_resolution=0.1,
        robot_radius=0.15,
        safety_margin=0.05,
        max_linear_speed=0.5,
        max_angular_speed=1.0,
    )

    agent = NavigationAgent(
        occupancy_grid=occupancy_grid,
        config=config,
    )

    pose = RobotPose(
        x=1.0,
        y=3.0,
        yaw=0.0,
    )

    goal = NavigationGoal(
        x=5.0,
        y=3.0,
        tolerance=0.2,
    )

    success = agent.set_goal(
        goal,
        pose,
    )

    print("Path found:", success)
    print("Waypoints:", len(agent.path))
    print("Path length:", agent.path_length())

    # Example navigation simulation.
    dt = 0.1

    for step in range(300):

        decision = agent.step(
            pose
        )

        if decision.status in (
            NavigationStatus.GOAL_REACHED,
            NavigationStatus.BLOCKED,
            NavigationStatus.STUCK,
            NavigationStatus.EMERGENCY_STOP,
        ):
            print(
                "Navigation stopped:",
                decision.status.value,
            )
            break

        command = decision.command

        # Simplified simulated motion.
        pose.x += (
            command.linear
            * math.cos(pose.yaw)
            * dt
        )

        pose.y += (
            command.linear
            * math.sin(pose.yaw)
            * dt
        )

        pose.yaw = agent.normalize_angle(
            pose.yaw
            + command.angular * dt
        )

        if step % 20 == 0:
            print(
                f"Step {step}: "
                f"position=({pose.x:.2f}, "
                f"{pose.y:.2f}), "
                f"distance={decision.distance_to_goal:.2f}"
            )

    print("\nNavigation report:")
    print(asdict(agent.report(pose)))
"""

