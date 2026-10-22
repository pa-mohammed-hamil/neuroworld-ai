
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/run_simulation.py

Run robot navigation and world simulations locally.

Features:
    - Random, empty, maze, corridor, warehouse scenarios
    - Multiple robot simulation
    - Autonomous navigation
    - Static and dynamic obstacles
    - Collision detection
    - Emergency stop handling
    - Deterministic random seeds
    - Optional visualization
    - Optional GIF and MP4 export
    - Simulation progress monitoring
    - JSON metrics and trajectory export
    - Multiple experiment runs
    - Command-line interface

Dependencies:
    pip install numpy matplotlib

Optional:
    pip install pillow imageio imageio-ffmpeg

Examples:
    python scripts/run_simulation.py

    python scripts/run_simulation.py \
        --scenario random \
        --robots 2 \
        --obstacles 10 \
        --steps 500

    python scripts/run_simulation.py \
        --scenario warehouse \
        --render \
        --save-image

    python scripts/run_simulation.py \
        --scenario corridor \
        --runs 5 \
        --save-report
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from neuroworld.environment.scenario import (
    Scenario,
    ScenarioConfig,
    ScenarioType,
)

from neuroworld.environment.simulator import (
    SimulatorConfig,
    SimulatorStatus,
    WorldSimulator,
)


LOGGER = logging.getLogger("neuroworld.simulation")


@dataclass
class SimulationRunConfig:
    scenario: str = "random"

    world_width: float = 10.0
    world_height: float = 10.0
    map_resolution: float = 0.1

    robots: int = 1
    obstacles: int = 10

    steps: int = 500
    timestep: float = 0.1

    runs: int = 1
    seed: int = 42

    enable_navigation: bool = True
    enable_collision_detection: bool = True
    enable_dynamic_obstacles: bool = True

    stop_on_collision: bool = True
    stop_when_goals_reached: bool = True

    render: bool = False
    capture_frames: bool = False

    save_image: bool = False
    save_gif: bool = False
    save_mp4: bool = False

    render_interval: int = 5
    progress_interval: int = 25

    gif_fps: int = 10
    mp4_fps: int = 20

    output_dir: str = "artifacts/simulations"

    save_report: bool = True
    save_trajectories: bool = True


@dataclass
class RobotRunMetrics:
    robot_id: str
    distance_traveled: float
    goal_reached: bool
    collision_count: int
    trajectory_points: int
    final_position: list[float]


@dataclass
class SimulationRunResult:
    run_id: str
    seed: int

    status: str
    success: bool

    steps: int
    simulation_time: float
    wall_time_seconds: float

    total_robots: int
    goals_reached: int
    total_collisions: int

    robot_metrics: list[RobotRunMetrics] = field(
        default_factory=list
    )

    output_files: dict[str, str] = field(
        default_factory=dict
    )

    error: Optional[str] = None


@dataclass
class SimulationExperimentReport:
    project: str
    timestamp: str

    configuration: dict[str, Any]

    total_runs: int
    successful_runs: int
    failed_runs: int

    success_rate: float
    mean_collisions: float
    mean_steps: float
    mean_simulation_time: float

    results: list[SimulationRunResult]


class SimulationRunner:
    """
    Execute NeuroWorld robot simulations.

    Uses existing project modules rather than
    duplicating navigation and physics logic.
    """

    VALID_SCENARIOS = {
        "empty",
        "random",
        "maze",
        "corridor",
        "warehouse",
    }

    def __init__(
        self,
        config: Optional[
            SimulationRunConfig
        ] = None,
    ) -> None:
        self.config = config or SimulationRunConfig()

        self._validate_config()

        self.output_dir = self._resolve_path(
            self.config.output_dir
        )

        self.results: list[
            SimulationRunResult
        ] = []

    @staticmethod
    def _resolve_path(
        value: str,
    ) -> Path:
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def _validate_config(self) -> None:
        c = self.config

        if c.scenario not in self.VALID_SCENARIOS:
            raise ValueError(
                f"Unsupported scenario: {c.scenario}"
            )

        if c.world_width <= 0 or c.world_height <= 0:
            raise ValueError(
                "World dimensions must be positive"
            )

        if c.map_resolution <= 0:
            raise ValueError(
                "Map resolution must be positive"
            )

        if c.robots <= 0:
            raise ValueError(
                "robots must be positive"
            )

        if c.obstacles < 0:
            raise ValueError(
                "obstacles cannot be negative"
            )

        if c.steps <= 0:
            raise ValueError(
                "steps must be positive"
            )

        if c.timestep <= 0:
            raise ValueError(
                "timestep must be positive"
            )

        if c.runs <= 0:
            raise ValueError(
                "runs must be positive"
            )

        if c.render_interval <= 0:
            raise ValueError(
                "render_interval must be positive"
            )

        if c.progress_interval <= 0:
            raise ValueError(
                "progress_interval must be positive"
            )

        if c.gif_fps <= 0 or c.mp4_fps <= 0:
            raise ValueError(
                "Export FPS must be positive"
            )

        for name in (
            "world_width",
            "world_height",
            "map_resolution",
            "timestep",
        ):
            if not math.isfinite(
                getattr(c, name)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )

    def _create_scenario(
        self,
        seed: int,
    ) -> Scenario:
        c = self.config

        scenario_config = ScenarioConfig(
            name=f"NeuroWorld_{c.scenario}_{seed}",
            scenario_type=ScenarioType(
                c.scenario
            ),
            world_width=c.world_width,
            world_height=c.world_height,
            map_resolution=c.map_resolution,
            num_robots=c.robots,
            num_obstacles=c.obstacles,
            simulation_dt=c.timestep,
            seed=seed,
        )

        scenario = Scenario(
            scenario_config
        )

        scenario.reset()

        return scenario

    def _create_renderer(self) -> Any:
        if not (
            self.config.render
            or self.config.save_image
            or self.config.save_gif
            or self.config.save_mp4
        ):
            return None

        from neuroworld.environment.renderer import (
            RendererConfig,
            WorldRenderer,
        )

        renderer_config = RendererConfig(
            world_width=self.config.world_width,
            world_height=self.config.world_height,
            title="NeuroWorld AI Simulation",
            interactive=False,
        )

        return WorldRenderer(
            renderer_config
        )

    def _create_simulator(
        self,
        scenario: Scenario,
        renderer: Any,
        seed: int,
    ) -> WorldSimulator:
        c = self.config

        simulator_config = SimulatorConfig(
            timestep=c.timestep,
            max_steps=c.steps,
            enable_navigation=c.enable_navigation,
            enable_collision_detection=(
                c.enable_collision_detection
            ),
            enable_dynamic_obstacles=(
                c.enable_dynamic_obstacles
            ),
            stop_on_collision=c.stop_on_collision,
            enable_rendering=(
                renderer is not None
            ),
            capture_frames=(
                c.capture_frames
                or c.save_gif
                or c.save_mp4
            ),
            render_every_n_steps=(
                c.render_interval
            ),
            seed=seed,
        )

        return WorldSimulator(
            scenario=scenario,
            config=simulator_config,
            renderer=renderer,
        )

    @staticmethod
    def _status_value(
        status: Any,
    ) -> str:
        return str(
            getattr(
                status,
                "value",
                status,
            )
        )

    @staticmethod
    def _serialize_trajectory(
        trajectory: Any,
    ) -> list[Any]:
        """
        Convert trajectory records into
        JSON-compatible values.
        """

        output = []

        for point in trajectory:
            if hasattr(
                point,
                "__dataclass_fields__",
            ):
                output.append(
                    asdict(point)
                )

            elif isinstance(
                point,
                (list, tuple),
            ):
                output.append(
                    [
                        float(value)
                        for value in point
                    ]
                )

            elif isinstance(
                point,
                dict,
            ):
                output.append(point)

            else:
                output.append(
                    str(point)
                )

        return output

    def _extract_robot_metrics(
        self,
        simulator: WorldSimulator,
    ) -> list[RobotRunMetrics]:
        metrics = []

        robot_states = getattr(
            simulator,
            "robot_states",
            None,
        )

        if robot_states is None:
            robot_states = getattr(
                simulator,
                "robots",
                {},
            )

        if not isinstance(
            robot_states,
            dict,
        ):
            return metrics

        for robot_id, robot in (
            robot_states.items()
        ):
            trajectory = getattr(
                robot,
                "trajectory",
                [],
            )

            metrics.append(
                RobotRunMetrics(
                    robot_id=str(robot_id),
                    distance_traveled=float(
                        getattr(
                            robot,
                            "distance_traveled",
                            0.0,
                        )
                    ),
                    goal_reached=bool(
                        getattr(
                            robot,
                            "goal_reached",
                            False,
                        )
                    ),
                    collision_count=int(
                        getattr(
                            robot,
                            "collision_count",
                            0,
                        )
                    ),
                    trajectory_points=len(
                        trajectory
                    ),
                    final_position=[
                        float(
                            getattr(
                                robot,
                                "x",
                                0.0,
                            )
                        ),
                        float(
                            getattr(
                                robot,
                                "y",
                                0.0,
                            )
                        ),
                    ],
                )
            )

        return metrics

    def _save_outputs(
        self,
        run_dir: Path,
        simulator: WorldSimulator,
        renderer: Any,
        robot_metrics: list[RobotRunMetrics],
    ) -> dict[str, str]:
        c = self.config

        output_files: dict[
            str, str
        ] = {}

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if c.save_trajectories:
            robot_states = getattr(
                simulator,
                "robot_states",
                {},
            )

            trajectories = {}

            if isinstance(
                robot_states,
                dict,
            ):
                for robot_id, robot in (
                    robot_states.items()
                ):
                    trajectories[
                        str(robot_id)
                    ] = self._serialize_trajectory(
                        getattr(
                            robot,
                            "trajectory",
                            [],
                        )
                    )

            trajectory_path = (
                run_dir / "trajectories.json"
            )

            trajectory_path.write_text(
                json.dumps(
                    trajectories,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

            output_files[
                "trajectories"
            ] = str(
                trajectory_path
            )

        if renderer is not None:
            if c.save_image:
                image_path = (
                    run_dir / "final_state.png"
                )

                simulator.save_image(
                    str(image_path)
                )

                output_files[
                    "image"
                ] = str(
                    image_path
                )

            if c.save_gif:
                gif_path = (
                    run_dir / "simulation.gif"
                )

                simulator.save_gif(
                    str(gif_path),
                    fps=c.gif_fps,
                )

                output_files[
                    "gif"
                ] = str(
                    gif_path
                )

            if c.save_mp4:
                mp4_path = (
                    run_dir / "simulation.mp4"
                )

                simulator.save_mp4(
                    str(mp4_path),
                    fps=c.mp4_fps,
                )

                output_files[
                    "mp4"
                ] = str(
                    mp4_path
                )

        return output_files

    def run_once(
        self,
        run_index: int,
    ) -> SimulationRunResult:
        c = self.config

        seed = c.seed + run_index

        run_id = (
            f"run_{run_index + 1:04d}"
        )

        run_dir = (
            self.output_dir / run_id
        )

        started = time.perf_counter()

        simulator = None
        renderer = None

        try:
            LOGGER.info(
                "Starting simulation %s "
                "| scenario=%s | seed=%d",
                run_id,
                c.scenario,
                seed,
            )

            scenario = self._create_scenario(
                seed
            )

            renderer = self._create_renderer()

            simulator = self._create_simulator(
                scenario,
                renderer,
                seed,
            )

            simulator.reset(
                regenerate_scenario=False
            )

            simulator.start()

            terminal_statuses = {
                SimulatorStatus.COMPLETED,
                SimulatorStatus.EMERGENCY_STOP,
                SimulatorStatus.ERROR,
            }

            for step_index in range(
                c.steps
            ):
                step_result = simulator.step()

                if (
                    (step_index + 1)
                    % c.progress_interval
                    == 0
                ):
                    LOGGER.info(
                        "%s | step=%d | "
                        "time=%.2fs | status=%s",
                        run_id,
                        step_result.step,
                        step_result.simulation_time,
                        self._status_value(
                            step_result.status
                        ),
                    )

                if (
                    c.stop_when_goals_reached
                    and step_result.all_goals_reached
                ):
                    LOGGER.info(
                        "%s | all goals reached",
                        run_id,
                    )
                    break

                if (
                    step_result.status
                    in terminal_statuses
                ):
                    break

            report = simulator.report()

            robot_metrics = (
                self._extract_robot_metrics(
                    simulator
                )
            )

            success = (
                report.total_robots > 0
                and report.goals_reached
                == report.total_robots
                and report.total_collisions == 0
            )

            output_files = self._save_outputs(
                run_dir,
                simulator,
                renderer,
                robot_metrics,
            )

            result = SimulationRunResult(
                run_id=run_id,
                seed=seed,
                status=self._status_value(
                    report.status
                ),
                success=success,
                steps=int(
                    report.total_steps
                ),
                simulation_time=float(
                    report.simulation_time
                ),
                wall_time_seconds=(
                    time.perf_counter()
                    - started
                ),
                total_robots=int(
                    report.total_robots
                ),
                goals_reached=int(
                    report.goals_reached
                ),
                total_collisions=int(
                    report.total_collisions
                ),
                robot_metrics=robot_metrics,
                output_files=output_files,
            )

            if c.save_report:
                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                report_path = (
                    run_dir / "report.json"
                )

                report_path.write_text(
                    json.dumps(
                        asdict(result),
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )

                result.output_files[
                    "report"
                ] = str(
                    report_path
                )

            LOGGER.info(
                "%s finished | success=%s "
                "| steps=%d | collisions=%d",
                run_id,
                result.success,
                result.steps,
                result.total_collisions,
            )

            return result

        except Exception as exc:
            LOGGER.exception(
                "Simulation failed: %s",
                run_id,
            )

            return SimulationRunResult(
                run_id=run_id,
                seed=seed,
                status="error",
                success=False,
                steps=0,
                simulation_time=0.0,
                wall_time_seconds=(
                    time.perf_counter()
                    - started
                ),
                total_robots=0,
                goals_reached=0,
                total_collisions=0,
                error=str(exc),
            )

        finally:
            if simulator is not None:
                try:
                    simulator.close()
                except Exception:
                    LOGGER.exception(
                        "Failed to close simulator"
                    )

            elif renderer is not None:
                try:
                    renderer.close()
                except Exception:
                    LOGGER.exception(
                        "Failed to close renderer"
                    )

    def run_experiments(
        self,
    ) -> SimulationExperimentReport:
        self.results.clear()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index in range(
            self.config.runs
        ):
            result = self.run_once(
                index
            )

            self.results.append(
                result
            )

        total = len(
            self.results
        )

        successful = sum(
            result.success
            for result in self.results
        )

        mean_collisions = (
            sum(
                result.total_collisions
                for result in self.results
            ) / total
            if total
            else 0.0
        )

        mean_steps = (
            sum(
                result.steps
                for result in self.results
            ) / total
            if total
            else 0.0
        )

        mean_simulation_time = (
            sum(
                result.simulation_time
                for result in self.results
            ) / total
            if total
            else 0.0
        )

        experiment = SimulationExperimentReport(
            project="NeuroWorld AI",
            timestamp=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            configuration=asdict(
                self.config
            ),
            total_runs=total,
            successful_runs=successful,
            failed_runs=(
                total - successful
            ),
            success_rate=(
                successful / total
                if total
                else 0.0
            ),
            mean_collisions=mean_collisions,
            mean_steps=mean_steps,
            mean_simulation_time=(
                mean_simulation_time
            ),
            results=list(
                self.results
            ),
        )

        summary_path = (
            self.output_dir
            / "experiment_summary.json"
        )

        summary_path.write_text(
            json.dumps(
                asdict(experiment),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return experiment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI simulation runner"
        )
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "empty",
            "random",
            "maze",
            "corridor",
            "warehouse",
        ],
        default="random",
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--obstacles",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--height",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--timestep",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    parser.add_argument(
        "--save-image",
        action="store_true",
    )

    parser.add_argument(
        "--save-gif",
        action="store_true",
    )

    parser.add_argument(
        "--save-mp4",
        action="store_true",
    )

    parser.add_argument(
        "--save-report",
        action="store_true",
    )

    parser.add_argument(
        "--no-navigation",
        action="store_true",
    )

    parser.add_argument(
        "--no-dynamic-obstacles",
        action="store_true",
    )

    parser.add_argument(
        "--no-stop-on-collision",
        action="store_true",
    )

    parser.add_argument(
        "--render-interval",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--output",
        default="artifacts/simulations",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = SimulationRunConfig(
        scenario=args.scenario,
        world_width=args.width,
        world_height=args.height,
        map_resolution=args.resolution,
        robots=args.robots,
        obstacles=args.obstacles,
        steps=args.steps,
        timestep=args.timestep,
        runs=args.runs,
        seed=args.seed,
        enable_navigation=(
            not args.no_navigation
        ),
        enable_dynamic_obstacles=(
            not args.no_dynamic_obstacles
        ),
        stop_on_collision=(
            not args.no_stop_on_collision
        ),
        render=args.render,
        save_image=args.save_image,
        save_gif=args.save_gif,
        save_mp4=args.save_mp4,
        capture_frames=(
            args.save_gif
            or args.save_mp4
        ),
        render_interval=(
            args.render_interval
        ),
        progress_interval=(
            args.progress_interval
        ),
        output_dir=args.output,
        save_report=True,
    )

    runner = SimulationRunner(
        config
    )

    experiment = (
        runner.run_experiments()
    )

    print()
    print("=" * 60)
    print("          NEUROWORLD AI SIMULATION")
    print("=" * 60)

    print(
        "Scenario:",
        config.scenario,
    )

    print(
        "Runs:",
        experiment.total_runs,
    )

    print(
        "Successful:",
        experiment.successful_runs,
    )

    print(
        "Failed:",
        experiment.failed_runs,
    )

    print(
        "Success rate:",
        f"{experiment.success_rate:.2%}",
    )

    print(
        "Mean collisions:",
        f"{experiment.mean_collisions:.2f}",
    )

    print(
        "Mean steps:",
        f"{experiment.mean_steps:.2f}",
    )

    print(
        "Output:",
        runner.output_dir,
    )

    for result in experiment.results:
        if result.error:
            print(
                f"[ERROR] {result.run_id}: "
                f"{result.error}"
            )

    return (
        0
        if experiment.failed_runs == 0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
