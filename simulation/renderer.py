
```python
"""
NeuroWorld AI
File: neuroworld/environment/renderer.py

World Simulation Renderer

Features:
- 2D world visualization
- Occupancy-grid rendering
- Robot position and orientation
- Navigation goals and waypoints
- Actual and predicted trajectories
- Multiple robots
- Dynamic obstacles
- Uncertainty circles
- World-model state projection
- Frame capture
- PNG export
- GIF recording
- MP4 recording (optional)
- Headless rendering
- Matplotlib interactive display

Dependencies:
    pip install numpy matplotlib pillow

Optional MP4 support:
    pip install imageio imageio-ffmpeg

Notes:
    This is a visualization module.
    It does not control real robots.

    World coordinates are in meters.
    Robot yaw is measured in radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    Polygon,
)
from PIL import Image


WorldPoint = Tuple[float, float]


@dataclass
class RendererConfig:
    width: int = 10
    height: int = 8

    dpi: int = 120

    title: str = "NeuroWorld AI Simulation"

    world_width: float = 10.0
    world_height: float = 10.0

    show_grid: bool = True
    show_legend: bool = True
    show_axes: bool = True

    show_trajectories: bool = True
    show_predictions: bool = True
    show_uncertainty: bool = True

    robot_radius: float = 0.2

    background_color: str = "#f8fafc"
    obstacle_color: str = "#334155"
    robot_color: str = "#2563eb"
    goal_color: str = "#16a34a"
    path_color: str = "#f59e0b"
    predicted_path_color: str = "#9333ea"

    max_frames: int = 1000

    interactive: bool = False


@dataclass
class RenderRobot:
    robot_id: str
    x: float
    y: float
    yaw: float = 0.0

    radius: float = 0.2
    color: Optional[str] = None

    label: Optional[str] = None


@dataclass
class RenderObstacle:
    x: float
    y: float

    radius: float = 0.3

    color: Optional[str] = None


@dataclass
class RenderGoal:
    x: float
    y: float

    radius: float = 0.15
    label: str = "Goal"


@dataclass
class RenderTrajectory:
    points: List[WorldPoint]

    label: str = "Trajectory"

    color: Optional[str] = None

    dashed: bool = False

    linewidth: float = 2.0


@dataclass
class RenderUncertainty:
    x: float
    y: float

    radius: float

    alpha: float = 0.15


@dataclass
class WorldRenderState:
    robots: List[RenderRobot] = field(
        default_factory=list
    )

    obstacles: List[RenderObstacle] = field(
        default_factory=list
    )

    goals: List[RenderGoal] = field(
        default_factory=list
    )

    trajectories: List[RenderTrajectory] = field(
        default_factory=list
    )

    predicted_trajectories: List[
        RenderTrajectory
    ] = field(
        default_factory=list
    )

    uncertainties: List[
        RenderUncertainty
    ] = field(
        default_factory=list
    )

    occupancy_grid: Optional[
        np.ndarray
    ] = None

    map_resolution: float = 0.1

    map_origin: WorldPoint = (0.0, 0.0)

    simulation_time: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class WorldRenderer:
    """
    Matplotlib-based world renderer.

    Example:

        renderer = WorldRenderer()

        state = WorldRenderState(
            robots=[
                RenderRobot(
                    robot_id="robot_1",
                    x=2.0,
                    y=3.0,
                )
            ]
        )

        renderer.render(state)

        renderer.save_image("world.png")
    """

    def __init__(
        self,
        config: Optional[
            RendererConfig
        ] = None,
    ):
        self.config = (
            config or RendererConfig()
        )

        self._validate_config()

        self.figure: Optional[
            Figure
        ] = None

        self.axes: Optional[
            Axes
        ] = None

        self.frames: List[
            Image.Image
        ] = []

        self.last_state: Optional[
            WorldRenderState
        ] = None

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.width <= 0 or cfg.height <= 0:
            raise ValueError(
                "Figure dimensions must be positive"
            )

        if cfg.dpi <= 0:
            raise ValueError(
                "DPI must be positive"
            )

        if (
            cfg.world_width <= 0
            or cfg.world_height <= 0
        ):
            raise ValueError(
                "World dimensions must be positive"
            )

        if cfg.robot_radius <= 0:
            raise ValueError(
                "Robot radius must be positive"
            )

        if cfg.max_frames <= 0:
            raise ValueError(
                "max_frames must be positive"
            )

    def _initialize_figure(self) -> None:
        """
        Create a new Matplotlib figure.
        """

        if self.figure is None:
            self.figure, self.axes = (
                plt.subplots(
                    figsize=(
                        self.config.width,
                        self.config.height,
                    ),
                    dpi=self.config.dpi,
                )
            )

    def _setup_axes(
        self,
        state: WorldRenderState,
    ) -> None:
        assert self.axes is not None

        ax = self.axes
        cfg = self.config

        ax.clear()

        ax.set_facecolor(
            cfg.background_color
        )

        ax.set_xlim(
            0,
            cfg.world_width,
        )

        ax.set_ylim(
            0,
            cfg.world_height,
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.set_title(
            f"{cfg.title}\n"
            f"Simulation time: "
            f"{state.simulation_time:.2f}s"
        )

        if cfg.show_axes:
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
        else:
            ax.set_axis_off()

        if cfg.show_grid:
            ax.grid(
                True,
                alpha=0.2,
                linestyle="--",
            )

    def _draw_occupancy_grid(
        self,
        state: WorldRenderState,
    ) -> None:
        if state.occupancy_grid is None:
            return

        assert self.axes is not None

        grid = np.asarray(
            state.occupancy_grid
        )

        if grid.ndim != 2:
            raise ValueError(
                "Occupancy grid must be 2D"
            )

        if state.map_resolution <= 0:
            raise ValueError(
                "Map resolution must be positive"
            )

        rows, cols = grid.shape

        origin_x, origin_y = (
            state.map_origin
        )

        extent = [
            origin_x,
            origin_x
            + cols * state.map_resolution,
            origin_y,
            origin_y
            + rows * state.map_resolution,
        ]

        self.axes.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="Greys",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            alpha=0.65,
            zorder=1,
        )

    def _draw_obstacles(
        self,
        obstacles: Sequence[
            RenderObstacle
        ],
    ) -> None:
        assert self.axes is not None

        for obstacle in obstacles:
            if obstacle.radius < 0:
                raise ValueError(
                    "Obstacle radius cannot be negative"
                )

            patch = Circle(
                (
                    obstacle.x,
                    obstacle.y,
                ),
                obstacle.radius,
                color=(
                    obstacle.color
                    or self.config.obstacle_color
                ),
                alpha=0.85,
                zorder=4,
            )

            self.axes.add_patch(
                patch
            )

    def _draw_robot(
        self,
        robot: RenderRobot,
    ) -> None:
        """
        Draw a triangular robot marker
        oriented by yaw.
        """

        assert self.axes is not None

        if robot.radius <= 0:
            raise ValueError(
                "Robot radius must be positive"
            )

        r = robot.radius

        local_vertices = np.array(
            [
                [1.5 * r, 0.0],
                [-r, 0.85 * r],
                [-r, -0.85 * r],
            ],
            dtype=np.float64,
        )

        cos_yaw = math.cos(
            robot.yaw
        )

        sin_yaw = math.sin(
            robot.yaw
        )

        rotation = np.array(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw],
            ]
        )

        vertices = (
            local_vertices
            @ rotation.T
        )

        vertices[:, 0] += robot.x
        vertices[:, 1] += robot.y

        patch = Polygon(
            vertices,
            closed=True,
            facecolor=(
                robot.color
                or self.config.robot_color
            ),
            edgecolor="white",
            linewidth=1.5,
            zorder=8,
        )

        self.axes.add_patch(
            patch
        )

        label = (
            robot.label
            or robot.robot_id
        )

        self.axes.text(
            robot.x,
            robot.y + r * 2.0,
            label,
            fontsize=9,
            ha="center",
            va="bottom",
            zorder=9,
        )

    def _draw_goals(
        self,
        goals: Sequence[
            RenderGoal
        ],
    ) -> None:
        assert self.axes is not None

        for goal in goals:
            self.axes.scatter(
                goal.x,
                goal.y,
                marker="*",
                s=200,
                color=self.config.goal_color,
                edgecolors="white",
                linewidths=1,
                zorder=10,
            )

            self.axes.text(
                goal.x,
                goal.y + max(
                    goal.radius,
                    0.15,
                ) * 2,
                goal.label,
                fontsize=9,
                ha="center",
                zorder=10,
            )

    def _draw_trajectory(
        self,
        trajectory: RenderTrajectory,
        predicted: bool = False,
    ) -> None:
        assert self.axes is not None

        if not trajectory.points:
            return

        points = np.asarray(
            trajectory.points,
            dtype=np.float64,
        )

        if (
            points.ndim != 2
            or points.shape[1] != 2
        ):
            raise ValueError(
                "Trajectory points must be [N,2]"
            )

        color = (
            trajectory.color
            or (
                self.config.predicted_path_color
                if predicted
                else self.config.path_color
            )
        )

        linestyle = (
            "--"
            if trajectory.dashed or predicted
            else "-"
        )

        self.axes.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=trajectory.linewidth,
            label=trajectory.label,
            zorder=5,
        )

    def _draw_uncertainty(
        self,
        uncertainties: Sequence[
            RenderUncertainty
        ],
    ) -> None:
        assert self.axes is not None

        for item in uncertainties:
            if item.radius < 0:
                raise ValueError(
                    "Uncertainty radius cannot "
                    "be negative"
                )

            patch = Circle(
                (
                    item.x,
                    item.y,
                ),
                item.radius,
                color=(
                    self.config
                    .predicted_path_color
                ),
                alpha=item.alpha,
                linewidth=0,
                zorder=3,
            )

            self.axes.add_patch(
                patch
            )

    def render(
        self,
        state: WorldRenderState,
        capture_frame: bool = False,
    ) -> Figure:
        """
        Render the complete world state.

        capture_frame=True stores a frame
        for GIF/MP4 export.
        """

        self._initialize_figure()

        assert self.figure is not None
        assert self.axes is not None

        self.last_state = state

        self._setup_axes(
            state
        )

        self._draw_occupancy_grid(
            state
        )

        self._draw_obstacles(
            state.obstacles
        )

        if self.config.show_uncertainty:
            self._draw_uncertainty(
                state.uncertainties
            )

        if self.config.show_trajectories:
            for trajectory in (
                state.trajectories
            ):
                self._draw_trajectory(
                    trajectory,
                    predicted=False,
                )

        if self.config.show_predictions:
            for trajectory in (
                state.predicted_trajectories
            ):
                self._draw_trajectory(
                    trajectory,
                    predicted=True,
                )

        for goal in state.goals:
            self._draw_goals(
                [goal]
            )

        for robot in state.robots:
            self._draw_robot(
                robot
            )

        if self.config.show_legend:
            handles, labels = (
                self.axes
                .get_legend_handles_labels()
            )

            if handles:
                self.axes.legend(
                    loc="upper right"
                )

        self.figure.tight_layout()

        self.figure.canvas.draw()

        if capture_frame:
            self.capture_frame()

        if self.config.interactive:
            plt.pause(0.001)

        return self.figure

    def capture_frame(self) -> Image.Image:
        """
        Capture the current rendered figure.

        Returns a PIL image.
        """

        if self.figure is None:
            raise RuntimeError(
                "Nothing has been rendered"
            )

        self.figure.canvas.draw()

        rgba = np.asarray(
            self.figure.canvas.buffer_rgba()
        )

        frame = Image.fromarray(
            rgba,
            mode="RGBA",
        ).convert("RGB")

        if (
            len(self.frames)
            >= self.config.max_frames
        ):
            self.frames.pop(0)

        self.frames.append(
            frame.copy()
        )

        return frame

    def save_image(
        self,
        path: str | Path,
    ) -> None:
        """
        Save current figure to PNG,
        JPEG or another Matplotlib format.
        """

        if self.figure is None:
            raise RuntimeError(
                "Nothing has been rendered"
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.figure.savefig(
            path,
            dpi=self.config.dpi,
            bbox_inches="tight",
        )

    def save_gif(
        self,
        path: str | Path,
        fps: int = 10,
    ) -> None:
        """
        Save recorded frames as GIF.
        """

        if not self.frames:
            raise RuntimeError(
                "No frames have been captured"
            )

        if fps <= 0:
            raise ValueError(
                "fps must be positive"
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        duration = round(
            1000 / fps
        )

        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )

    def save_mp4(
        self,
        path: str | Path,
        fps: int = 20,
    ) -> None:
        """
        Save recorded frames as MP4.

        Requires:
            pip install imageio imageio-ffmpeg
        """

        if not self.frames:
            raise RuntimeError(
                "No frames have been captured"
            )

        if fps <= 0:
            raise ValueError(
                "fps must be positive"
            )

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Install imageio and imageio-ffmpeg "
                "to export MP4"
            ) from exc

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with imageio.get_writer(
            str(path),
            fps=fps,
            codec="libx264",
            macro_block_size=16,
        ) as writer:
            for frame in self.frames:
                writer.append_data(
                    np.asarray(
                        frame.convert("RGB")
                    )
                )

    def clear_frames(self) -> None:
        self.frames.clear()

    def show(self) -> None:
        """
        Display interactive Matplotlib window.
        """

        if self.figure is None:
            raise RuntimeError(
                "Nothing has been rendered"
            )

        plt.show()

    def close(self) -> None:
        """
        Release figure resources.
        """

        if self.figure is not None:
            plt.close(
                self.figure
            )

        self.figure = None
        self.axes = None

    def statistics(self) -> Dict[str, Any]:
        """
        Return renderer statistics.
        """

        return {
            "captured_frames": len(
                self.frames
            ),
            "figure_initialized": (
                self.figure is not None
            ),
            "last_simulation_time": (
                self.last_state.simulation_time
                if self.last_state is not None
                else None
            ),
            "robot_count": (
                len(self.last_state.robots)
                if self.last_state is not None
                else 0
            ),
            "obstacle_count": (
                len(self.last_state.obstacles)
                if self.last_state is not None
                else 0
            ),
        }


if __name__ == "__main__":

    renderer = WorldRenderer(
        RendererConfig(
            world_width=10.0,
            world_height=10.0,
            interactive=False,
        )
    )

    occupancy = np.zeros(
        (100, 100),
        dtype=np.float32,
    )

    # Wall with doorway.
    occupancy[20:80, 50] = 1.0
    occupancy[45:55, 50] = 0.0

    actual_path = []
    predicted_path = [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.5),
        (5.0, 5.0),
        (7.0, 7.0),
        (8.0, 8.0),
    ]

    # Simulated robot motion.
    for step in range(50):

        fraction = step / 49.0

        x = 1.0 + 7.0 * fraction
        y = 1.0 + 7.0 * fraction

        actual_path.append(
            (x, y)
        )

        state = WorldRenderState(
            robots=[
                RenderRobot(
                    robot_id="RA-R1",
                    x=x,
                    y=y,
                    yaw=math.pi / 4,
                    radius=0.2,
                )
            ],
            goals=[
                RenderGoal(
                    x=8.0,
                    y=8.0,
                    label="Target",
                )
            ],
            trajectories=[
                RenderTrajectory(
                    points=list(actual_path),
                    label="Actual trajectory",
                )
            ],
            predicted_trajectories=[
                RenderTrajectory(
                    points=predicted_path,
                    label="Predicted trajectory",
                    dashed=True,
                )
            ],
            uncertainties=[
                RenderUncertainty(
                    x=7.0,
                    y=7.0,
                    radius=0.4,
                )
            ],
            occupancy_grid=occupancy,
            map_resolution=0.1,
            simulation_time=step * 0.1,
        )

        renderer.render(
            state,
            capture_frame=True,
        )

    renderer.save_image(
        "artifacts/world_render.png"
    )

    renderer.save_gif(
        "artifacts/world_simulation.gif",
        fps=10,
    )

    print(
        "Renderer statistics:",
        renderer.statistics(),
    )

    renderer.close()
```
