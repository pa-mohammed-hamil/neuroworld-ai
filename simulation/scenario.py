
```python
"""
NeuroWorld AI
File: neuroworld/environment/scenario.py

Scenario Management and Generation

Features:
- Scenario configuration
- Empty-world generation
- Random obstacle generation
- Maze generation
- Corridor and doorway scenarios
- Multiple robots and goals
- Dynamic moving obstacles
- Scenario validation
- Robot-obstacle collision checking
- Reproducible random seeds
- Scenario reset and stepping
- JSON import/export
- Occupancy-grid conversion
- Renderer integration
- Navigation-agent integration

Dependencies:
    pip install numpy
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


WorldPoint = Tuple[float, float]


class ScenarioType(str, Enum):
    EMPTY = "empty"
    RANDOM = "random"
    MAZE = "maze"
    CORRIDOR = "corridor"
    WAREHOUSE = "warehouse"
    CUSTOM = "custom"


@dataclass
class ScenarioConfig:
    name: str = "NeuroWorld Scenario"

    scenario_type: str = "empty"

    world_width: float = 10.0
    world_height: float = 10.0

    map_resolution: float = 0.1

    num_robots: int = 1
    num_obstacles: int = 15

    robot_radius: float = 0.2

    obstacle_min_radius: float = 0.2
    obstacle_max_radius: float = 0.5

    safety_margin: float = 0.1

    max_generation_attempts: int = 10000

    seed: int = 42

    simulation_dt: float = 0.1


@dataclass
class ScenarioRobot:
    robot_id: str

    x: float
    y: float
    yaw: float = 0.0

    radius: float = 0.2

    goal_x: Optional[float] = None
    goal_y: Optional[float] = None


@dataclass
class ScenarioObstacle:
    obstacle_id: str

    x: float
    y: float

    radius: float = 0.3

    dynamic: bool = False

    velocity_x: float = 0.0
    velocity_y: float = 0.0


@dataclass
class ScenarioGoal:
    goal_id: str

    x: float
    y: float

    tolerance: float = 0.2

    assigned_robot: Optional[str] = None


@dataclass
class ScenarioEvent:
    timestamp: float

    event_type: str

    entity_id: str

    description: str = ""


@dataclass
class ScenarioState:
    name: str

    scenario_type: str

    world_width: float
    world_height: float

    robots: List[ScenarioRobot] = field(
        default_factory=list
    )

    obstacles: List[ScenarioObstacle] = field(
        default_factory=list
    )

    goals: List[ScenarioGoal] = field(
        default_factory=list
    )

    simulation_time: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class Scenario:
    """
    Scenario container and simulation manager.

    Responsibilities:
        - Store world entities
        - Generate procedural environments
        - Validate positions
        - Update dynamic obstacles
        - Generate occupancy grids
        - Export renderer-compatible state

    Robot movement is controlled externally.
    """

    def __init__(
        self,
        config: Optional[ScenarioConfig] = None,
    ):
        self.config = config or ScenarioConfig()

        self._validate_config()

        self._rng = random.Random(
            self.config.seed
        )

        self.state = ScenarioState(
            name=self.config.name,
            scenario_type=self.config.scenario_type,
            world_width=self.config.world_width,
            world_height=self.config.world_height,
        )

        self.events: List[ScenarioEvent] = []

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.scenario_type not in {
            item.value for item in ScenarioType
        }:
            raise ValueError(
                "Unknown scenario type"
            )

        for name in (
            "world_width",
            "world_height",
            "map_resolution",
            "robot_radius",
            "simulation_dt",
        ):
            if getattr(cfg, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if cfg.num_robots < 0:
            raise ValueError(
                "num_robots cannot be negative"
            )

        if cfg.num_obstacles < 0:
            raise ValueError(
                "num_obstacles cannot be negative"
            )

        if (
            cfg.obstacle_min_radius <= 0
            or cfg.obstacle_max_radius
            < cfg.obstacle_min_radius
        ):
            raise ValueError(
                "Invalid obstacle radius range"
            )

        if cfg.safety_margin < 0:
            raise ValueError(
                "Safety margin cannot be negative"
            )

        if cfg.max_generation_attempts <= 0:
            raise ValueError(
                "max_generation_attempts "
                "must be positive"
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

    def _inside_world(
        self,
        x: float,
        y: float,
        radius: float = 0.0,
    ) -> bool:
        return (
            radius <= x
            <= self.config.world_width - radius
            and radius <= y
            <= self.config.world_height - radius
        )

    def _position_available(
        self,
        x: float,
        y: float,
        radius: float,
        ignore_obstacle: Optional[str] = None,
    ) -> bool:
        if not self._inside_world(
            x, y, radius
        ):
            return False

        clearance = self.config.safety_margin

        for obstacle in self.state.obstacles:
            if (
                obstacle.obstacle_id
                == ignore_obstacle
            ):
                continue

            distance = self.distance(
                (x, y),
                (obstacle.x, obstacle.y),
            )

            if distance < (
                radius
                + obstacle.radius
                + clearance
            ):
                return False

        for robot in self.state.robots:
            distance = self.distance(
                (x, y),
                (robot.x, robot.y),
            )

            if distance < (
                radius
                + robot.radius
                + clearance
            ):
                return False

        return True

    def _random_position(
        self,
        radius: float,
    ) -> WorldPoint:
        x = self._rng.uniform(
            radius,
            self.config.world_width - radius,
        )

        y = self._rng.uniform(
            radius,
            self.config.world_height - radius,
        )

        return x, y

    def add_robot(
        self,
        robot: ScenarioRobot,
    ) -> None:
        if any(
            item.robot_id == robot.robot_id
            for item in self.state.robots
        ):
            raise ValueError(
                "Duplicate robot ID"
            )

        if robot.radius <= 0:
            raise ValueError(
                "Robot radius must be positive"
            )

        if not self._position_available(
            robot.x,
            robot.y,
            robot.radius,
        ):
            raise ValueError(
                "Robot position is occupied "
                "or outside the world"
            )

        self.state.robots.append(robot)

    def add_obstacle(
        self,
        obstacle: ScenarioObstacle,
    ) -> None:
        if any(
            item.obstacle_id == obstacle.obstacle_id
            for item in self.state.obstacles
        ):
            raise ValueError(
                "Duplicate obstacle ID"
            )

        if obstacle.radius <= 0:
            raise ValueError(
                "Obstacle radius must be positive"
            )

        if not self._position_available(
            obstacle.x,
            obstacle.y,
            obstacle.radius,
        ):
            raise ValueError(
                "Obstacle position unavailable"
            )

        self.state.obstacles.append(
            obstacle
        )

    def add_goal(
        self,
        goal: ScenarioGoal,
    ) -> None:
        if any(
            item.goal_id == goal.goal_id
            for item in self.state.goals
        ):
            raise ValueError(
                "Duplicate goal ID"
            )

        if not self._inside_world(
            goal.x,
            goal.y,
            goal.tolerance,
        ):
            raise ValueError(
                "Goal outside world"
            )

        if any(
            self.distance(
                (goal.x, goal.y),
                (obstacle.x, obstacle.y),
            )
            < obstacle.radius
            + goal.tolerance
            for obstacle in self.state.obstacles
        ):
            raise ValueError(
                "Goal overlaps an obstacle"
            )

        if (
            goal.assigned_robot is not None
            and not any(
                robot.robot_id
                == goal.assigned_robot
                for robot in self.state.robots
            )
        ):
            raise ValueError(
                "Assigned robot does not exist"
            )

        self.state.goals.append(goal)

    def clear(self) -> None:
        self.state.robots.clear()
        self.state.obstacles.clear()
        self.state.goals.clear()

        self.state.simulation_time = 0.0
        self.events.clear()

    def reset(self) -> ScenarioState:
        """
        Regenerate the configured scenario.

        Uses the original random seed for
        deterministic scenario generation.
        """

        self.clear()

        self._rng.seed(
            self.config.seed
        )

        scenario_type = ScenarioType(
            self.config.scenario_type
        )

        if scenario_type == ScenarioType.EMPTY:
            self._generate_empty()

        elif scenario_type == ScenarioType.RANDOM:
            self._generate_random()

        elif scenario_type == ScenarioType.MAZE:
            self._generate_maze()

        elif scenario_type == ScenarioType.CORRIDOR:
            self._generate_corridor()

        elif scenario_type == ScenarioType.WAREHOUSE:
            self._generate_warehouse()

        # CUSTOM remains empty for manual setup.

        self._generate_robots_and_goals()

        return self.state

    def _generate_empty(self) -> None:
        pass

    def _generate_random(self) -> None:
        """
        Generate circular obstacles.
        """

        attempts = 0

        while (
            len(self.state.obstacles)
            < self.config.num_obstacles
            and attempts
            < self.config.max_generation_attempts
        ):
            attempts += 1

            radius = self._rng.uniform(
                self.config.obstacle_min_radius,
                self.config.obstacle_max_radius,
            )

            x, y = self._random_position(
                radius
            )

            obstacle = ScenarioObstacle(
                obstacle_id=(
                    f"obstacle_{len(self.state.obstacles)}"
                ),
                x=x,
                y=y,
                radius=radius,
            )

            try:
                self.add_obstacle(obstacle)
            except ValueError:
                continue

        if (
            len(self.state.obstacles)
            < self.config.num_obstacles
        ):
            raise RuntimeError(
                "Unable to place requested "
                "number of obstacles"
            )

    def _generate_maze(self) -> None:
        """
        Generate staggered circular barriers.

        This creates a maze-like obstacle
        course, not a guaranteed-solvable
        perfect maze.
        """

        spacing_x = 1.5
        spacing_y = 1.5

        row_index = 0
        y = spacing_y

        while y < (
            self.config.world_height - spacing_y
        ):
            offset = (
                0.75
                if row_index % 2
                else 0.0
            )

            x = spacing_x + offset

            while x < (
                self.config.world_width - spacing_x
            ):
                obstacle = ScenarioObstacle(
                    obstacle_id=(
                        f"maze_{len(self.state.obstacles)}"
                    ),
                    x=x,
                    y=y,
                    radius=0.35,
                )

                try:
                    self.add_obstacle(obstacle)
                except ValueError:
                    pass

                x += spacing_x

            y += spacing_y
            row_index += 1

    def _generate_corridor(self) -> None:
        """
        Create two parallel rows of
        circular obstacles.
        """

        center_y = (
            self.config.world_height / 2
        )

        corridor_half_width = 1.0

        x = 1.0

        while x < (
            self.config.world_width - 1.0
        ):
            for side in (-1, 1):
                obstacle = ScenarioObstacle(
                    obstacle_id=(
                        f"corridor_{len(self.state.obstacles)}"
                    ),
                    x=x,
                    y=(
                        center_y
                        + side
                        * corridor_half_width
                    ),
                    radius=0.25,
                )

                try:
                    self.add_obstacle(obstacle)
                except ValueError:
                    pass

            x += 0.6

    def _generate_warehouse(self) -> None:
        """
        Create warehouse-like rows
        of storage obstacles.
        """

        x = 2.0

        while x < (
            self.config.world_width - 1.0
        ):
            y = 1.5

            while y < (
                self.config.world_height - 1.0
            ):
                obstacle = ScenarioObstacle(
                    obstacle_id=(
                        f"shelf_{len(self.state.obstacles)}"
                    ),
                    x=x,
                    y=y,
                    radius=0.35,
                )

                try:
                    self.add_obstacle(obstacle)
                except ValueError:
                    pass

                y += 0.8

            x += 2.0

    def _generate_robots_and_goals(self) -> None:
        """
        Place robots and goals in free space.
        """

        for index in range(
            self.config.num_robots
        ):
            placed = False

            for _ in range(
                self.config.max_generation_attempts
            ):
                x, y = self._random_position(
                    self.config.robot_radius
                )

                robot = ScenarioRobot(
                    robot_id=f"robot_{index}",
                    x=x,
                    y=y,
                    yaw=self._rng.uniform(
                        -math.pi,
                        math.pi,
                    ),
                    radius=self.config.robot_radius,
                )

                try:
                    self.add_robot(robot)
                    placed = True
                    break
                except ValueError:
                    continue

            if not placed:
                raise RuntimeError(
                    f"Could not place robot_{index}"
                )

            goal_placed = False

            for _ in range(
                self.config.max_generation_attempts
            ):
                gx, gy = self._random_position(
                    self.config.robot_radius
                )

                goal = ScenarioGoal(
                    goal_id=f"goal_{index}",
                    x=gx,
                    y=gy,
                    assigned_robot=robot.robot_id,
                )

                try:
                    self.add_goal(goal)
                    robot.goal_x = gx
                    robot.goal_y = gy
                    goal_placed = True
                    break
                except ValueError:
                    continue

            if not goal_placed:
                raise RuntimeError(
                    f"Could not place goal_{index}"
                )

    def step(
        self,
        dt: Optional[float] = None,
    ) -> ScenarioState:
        """
        Advance dynamic obstacles.

        Obstacles reflect off world boundaries.

        This uses simple kinematics and does
        not resolve obstacle-obstacle collisions.
        """

        dt = (
            self.config.simulation_dt
            if dt is None
            else dt
        )

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(
                "dt must be positive and finite"
            )

        for obstacle in self.state.obstacles:
            if not obstacle.dynamic:
                continue

            new_x = (
                obstacle.x
                + obstacle.velocity_x * dt
            )

            new_y = (
                obstacle.y
                + obstacle.velocity_y * dt
            )

            if (
                new_x - obstacle.radius < 0
                or new_x + obstacle.radius
                > self.config.world_width
            ):
                obstacle.velocity_x *= -1

                new_x = (
                    obstacle.x
                    + obstacle.velocity_x * dt
                )

            if (
                new_y - obstacle.radius < 0
                or new_y + obstacle.radius
                > self.config.world_height
            ):
                obstacle.velocity_y *= -1

                new_y = (
                    obstacle.y
                    + obstacle.velocity_y * dt
                )

            obstacle.x = min(
                max(new_x, obstacle.radius),
                self.config.world_width
                - obstacle.radius,
            )

            obstacle.y = min(
                max(new_y, obstacle.radius),
                self.config.world_height
                - obstacle.radius,
            )

        self.state.simulation_time += dt

        return self.state

    def check_collisions(
        self,
    ) -> List[Tuple[str, str]]:
        """
        Return pairs of colliding entities.

        Checks robot-obstacle and
        robot-robot collisions.
        """

        collisions = []

        robots = self.state.robots

        for robot in robots:
            for obstacle in self.state.obstacles:
                distance = self.distance(
                    (robot.x, robot.y),
                    (obstacle.x, obstacle.y),
                )

                if distance < (
                    robot.radius
                    + obstacle.radius
                ):
                    collisions.append(
                        (
                            robot.robot_id,
                            obstacle.obstacle_id,
                        )
                    )

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                first = robots[i]
                second = robots[j]

                distance = self.distance(
                    (first.x, first.y),
                    (second.x, second.y),
                )

                if distance < (
                    first.radius
                    + second.radius
                ):
                    collisions.append(
                        (
                            first.robot_id,
                            second.robot_id,
                        )
                    )

        return collisions

    def to_occupancy_grid(
        self,
    ) -> np.ndarray:
        """
        Rasterize circular obstacles into
        a 2D occupancy grid.

        Returns:
            [rows,columns]

        0 = free
        1 = occupied

        Conservative cell-center rasterization
        includes a half-cell margin.
        """

        resolution = (
            self.config.map_resolution
        )

        rows = math.ceil(
            self.config.world_height
            / resolution
        )

        cols = math.ceil(
            self.config.world_width
            / resolution
        )

        grid = np.zeros(
            (rows, cols),
            dtype=np.float32,
        )

        half_diagonal = (
            resolution * math.sqrt(2) / 2
        )

        for obstacle in self.state.obstacles:
            effective_radius = (
                obstacle.radius
                + half_diagonal
            )

            min_col = max(
                0,
                math.floor(
                    (
                        obstacle.x
                        - effective_radius
                    ) / resolution
                ),
            )

            max_col = min(
                cols - 1,
                math.ceil(
                    (
                        obstacle.x
                        + effective_radius
                    ) / resolution
                ),
            )

            min_row = max(
                0,
                math.floor(
                    (
                        obstacle.y
                        - effective_radius
                    ) / resolution
                ),
            )

            max_row = min(
                rows - 1,
                math.ceil(
                    (
                        obstacle.y
                        + effective_radius
                    ) / resolution
                ),
            )

            for row in range(
                min_row,
                max_row + 1,
            ):
                for col in range(
                    min_col,
                    max_col + 1,
                ):
                    x = (
                        col + 0.5
                    ) * resolution

                    y = (
                        row + 0.5
                    ) * resolution

                    if self.distance(
                        (x, y),
                        (
                            obstacle.x,
                            obstacle.y,
                        ),
                    ) <= effective_radius:
                        grid[row, col] = 1.0

        return grid

    def to_render_state(self):
        """
        Convert scenario into the format
        accepted by environment/renderer.py.
        """

        from neuroworld.environment.renderer import (
            WorldRenderState,
            RenderRobot,
            RenderObstacle,
            RenderGoal,
        )

        return WorldRenderState(
            robots=[
                RenderRobot(
                    robot_id=robot.robot_id,
                    x=robot.x,
                    y=robot.y,
                    yaw=robot.yaw,
                    radius=robot.radius,
                )
                for robot in self.state.robots
            ],
            obstacles=[
                RenderObstacle(
                    x=obstacle.x,
                    y=obstacle.y,
                    radius=obstacle.radius,
                )
                for obstacle in self.state.obstacles
            ],
            goals=[
                RenderGoal(
                    x=goal.x,
                    y=goal.y,
                    label=goal.goal_id,
                )
                for goal in self.state.goals
            ],
            simulation_time=(
                self.state.simulation_time
            ),
            map_resolution=(
                self.config.map_resolution
            ),
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save configuration and scenario state.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "config": asdict(self.config),
            "state": asdict(self.state),
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

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "Scenario":
        """
        Load a previously saved scenario.
        """

        payload = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        scenario = cls(
            ScenarioConfig(
                **payload["config"]
            )
        )

        state = payload["state"]

        scenario.state = ScenarioState(
            name=state["name"],
            scenario_type=state["scenario_type"],
            world_width=state["world_width"],
            world_height=state["world_height"],
            robots=[
                ScenarioRobot(**item)
                for item in state["robots"]
            ],
            obstacles=[
                ScenarioObstacle(**item)
                for item in state["obstacles"]
            ],
            goals=[
                ScenarioGoal(**item)
                for item in state["goals"]
            ],
            simulation_time=state.get(
                "simulation_time",
                0.0,
            ),
            metadata=state.get(
                "metadata",
                {},
            ),
        )

        scenario.events = [
            ScenarioEvent(**item)
            for item in payload.get(
                "events",
                [],
            )
        ]

        return scenario

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.state.name,
            "type": self.state.scenario_type,
            "world_size": [
                self.state.world_width,
                self.state.world_height,
            ],
            "robots": len(
                self.state.robots
            ),
            "obstacles": len(
                self.state.obstacles
            ),
            "dynamic_obstacles": sum(
                obstacle.dynamic
                for obstacle in self.state.obstacles
            ),
            "goals": len(
                self.state.goals
            ),
            "simulation_time": (
                self.state.simulation_time
            ),
            "collisions": (
                self.check_collisions()
            ),
        }


if __name__ == "__main__":

    scenario = Scenario(
        ScenarioConfig(
            name="Autonomous Warehouse",
            scenario_type="warehouse",
            world_width=12.0,
            world_height=10.0,
            map_resolution=0.1,
            num_robots=2,
            seed=42,
        )
    )

    scenario.reset()

    print(
        "Scenario:",
        scenario.summary(),
    )

    # Add a dynamic obstacle in free space.
    moving_obstacle = ScenarioObstacle(
        obstacle_id="moving_obstacle_1",
        x=6.0,
        y=5.0,
        radius=0.2,
        dynamic=True,
        velocity_x=0.2,
        velocity_y=0.1,
    )

    try:
        scenario.add_obstacle(
            moving_obstacle
        )
    except ValueError:
        print(
            "Moving obstacle placement unavailable"
        )

    for step_index in range(20):
        scenario.step()

        if step_index % 5 == 0:
            print(
                "Simulation time:",
                scenario.state.simulation_time,
            )

            print(
                "Collisions:",
                scenario.check_collisions(),
            )

    occupancy_grid = (
        scenario.to_occupancy_grid()
    )

    print(
        "Occupancy grid shape:",
        occupancy_grid.shape,
    )

    scenario.save(
        "artifacts/warehouse_scenario.json"
    )

    restored = Scenario.load(
        "artifacts/warehouse_scenario.json"
    )

    print(
        "Restored scenario:",
        restored.summary(),
    )
```
