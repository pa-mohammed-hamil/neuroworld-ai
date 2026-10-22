
```python
"""
NeuroWorld AI
File: neuroworld/environment/world_runner.py

World Simulation Runner and Experiment Orchestrator

Features:
- Single simulation execution
- Batch experiment execution
- Multiple scenario types
- Reproducible scenario seeds
- Navigation simulation
- Simulation progress callbacks
- Stop conditions
- Run status tracking
- Collision and goal metrics
- Experiment benchmarking
- JSON and CSV reports
- Optional PNG/GIF/MP4 recording
- Graceful interruption
- Run history and aggregate statistics

Dependencies:
    pip install numpy matplotlib pillow torch

Optional:
    pip install imageio imageio-ffmpeg
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
import uuid

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from neuroworld.environment.scenario import (
    Scenario,
    ScenarioConfig,
)

from neuroworld.environment.simulator import (
    WorldSimulator,
    SimulatorConfig,
    SimulatorStatus,
    SimulationReport,
)

from neuroworld.environment.renderer import (
    WorldRenderer,
    RendererConfig,
)


class WorldRunnerStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class WorldRunnerConfig:
    experiment_name: str = "neuroworld_experiment"

    scenario_type: str = "random"

    world_width: float = 10.0
    world_height: float = 10.0

    map_resolution: float = 0.1

    num_robots: int = 1
    num_obstacles: int = 10

    timestep: float = 0.1
    max_steps: int = 500

    num_runs: int = 1
    seed: int = 42

    enable_navigation: bool = True
    enable_dynamic_obstacles: bool = True

    stop_on_collision: bool = True
    stop_when_goals_reached: bool = True

    enable_rendering: bool = False
    capture_frames: bool = False

    render_every_n_steps: int = 5

    save_reports: bool = True
    save_final_image: bool = False
    save_gif: bool = False
    save_mp4: bool = False

    gif_fps: int = 10
    mp4_fps: int = 20

    output_dir: str = "artifacts/world_runs"

    progress_interval: int = 25


@dataclass
class WorldRunMetrics:
    run_id: str
    run_index: int
    scenario_type: str
    seed: int

    status: str
    termination_reason: str

    total_steps: int
    simulation_time: float
    wall_time_seconds: float

    total_robots: int
    goals_reached: int
    goal_success_rate: float

    total_collisions: int
    collision_rate: float

    total_distance_traveled: float
    mean_distance_traveled: float

    mean_steps_per_second: float

    timestamp: str

    output_files: Dict[str, str] = field(
        default_factory=dict
    )

    error: Optional[str] = None


@dataclass
class ExperimentSummary:
    experiment_name: str

    total_runs: int
    completed_runs: int
    failed_runs: int
    stopped_runs: int

    mean_goal_success_rate: float
    mean_collisions: float
    mean_simulation_time: float
    mean_wall_time_seconds: float

    total_goals_reached: int
    total_collisions: int

    best_run_id: Optional[str]

    run_metrics: List[WorldRunMetrics] = field(
        default_factory=list
    )


ProgressCallback = Callable[
    [int, int, WorldSimulator],
    None,
]


StopCondition = Callable[
    [WorldSimulator],
    bool,
]


class WorldRunner:
    """
    High-level world simulation runner.

    Execution:

        Runner Configuration
                 |
                 v
        Scenario Generation
                 |
                 v
        Simulator Initialization
                 |
                 v
        Navigation / Physics Steps
                 |
                 v
        Progress Monitoring
                 |
                 v
        Termination Check
                 |
                 v
        Metrics Collection
                 |
                 v
        Reports and Recordings

    The runner does not execute commands
    on real robots or vehicles.
    """

    def __init__(
        self,
        config: Optional[
            WorldRunnerConfig
        ] = None,
        progress_callback: Optional[
            ProgressCallback
        ] = None,
        stop_condition: Optional[
            StopCondition
        ] = None,
    ):
        self.config = (
            config or WorldRunnerConfig()
        )

        self._validate_config()

        self.progress_callback = (
            progress_callback
        )

        self.stop_condition = (
            stop_condition
        )

        self.status = WorldRunnerStatus.IDLE

        self.run_history: List[
            WorldRunMetrics
        ] = []

        self.current_simulator: Optional[
            WorldSimulator
        ] = None

        self.current_run_id: Optional[
            str
        ] = None

        self._stop_requested = False

        self.output_dir = Path(
            self.config.output_dir
        )

    def _validate_config(self) -> None:
        cfg = self.config

        for name in (
            "world_width",
            "world_height",
            "map_resolution",
            "timestep",
        ):
            value = getattr(cfg, name)

            if (
                not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        for name in (
            "max_steps",
            "num_runs",
            "render_every_n_steps",
            "progress_interval",
            "gif_fps",
            "mp4_fps",
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

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _new_run_id(
        index: int,
    ) -> str:
        return (
            f"run_{index:03d}_"
            f"{uuid.uuid4().hex[:8]}"
        )

    def _create_scenario(
        self,
        seed: int,
    ) -> Scenario:
        cfg = self.config

        scenario = Scenario(
            ScenarioConfig(
                name=(
                    f"{cfg.experiment_name}_{seed}"
                ),
                scenario_type=cfg.scenario_type,
                world_width=cfg.world_width,
                world_height=cfg.world_height,
                map_resolution=cfg.map_resolution,
                num_robots=cfg.num_robots,
                num_obstacles=cfg.num_obstacles,
                seed=seed,
                simulation_dt=cfg.timestep,
            )
        )

        scenario.reset()

        return scenario

    def _create_renderer(
        self,
        scenario: Scenario,
    ) -> Optional[WorldRenderer]:
        cfg = self.config

        needs_renderer = any(
            (
                cfg.enable_rendering,
                cfg.capture_frames,
                cfg.save_final_image,
                cfg.save_gif,
                cfg.save_mp4,
            )
        )

        if not needs_renderer:
            return None

        return WorldRenderer(
            RendererConfig(
                title=cfg.experiment_name,
                world_width=(
                    scenario.config.world_width
                ),
                world_height=(
                    scenario.config.world_height
                ),
                interactive=False,
                max_frames=max(
                    cfg.max_steps
                    // cfg.render_every_n_steps
                    + 2,
                    100,
                ),
            )
        )

    def _create_simulator(
        self,
        scenario: Scenario,
        renderer: Optional[
            WorldRenderer
        ],
    ) -> WorldSimulator:
        cfg = self.config

        simulator = WorldSimulator(
            scenario=scenario,
            config=SimulatorConfig(
                timestep=cfg.timestep,
                max_steps=cfg.max_steps,
                enable_navigation=(
                    cfg.enable_navigation
                ),
                enable_dynamic_obstacles=(
                    cfg.enable_dynamic_obstacles
                ),
                stop_on_collision=(
                    cfg.stop_on_collision
                ),
                enable_rendering=(
                    renderer is not None
                    and (
                        cfg.enable_rendering
                        or cfg.capture_frames
                        or cfg.save_gif
                        or cfg.save_mp4
                    )
                ),
                capture_frames=(
                    cfg.capture_frames
                    or cfg.save_gif
                    or cfg.save_mp4
                ),
                render_every_n_steps=(
                    cfg.render_every_n_steps
                ),
                seed=cfg.seed,
            ),
            renderer=renderer,
        )

        # Preserve the already-generated
        # scenario and its current entities.
        simulator.reset(
            regenerate_scenario=False
        )

        return simulator

    def request_stop(self) -> None:
        """
        Request a graceful stop.

        The active simulation exits after
        its current step finishes.
        """

        self._stop_requested = True

    def clear_stop_request(self) -> None:
        self._stop_requested = False

    def _notify_progress(
        self,
        simulator: WorldSimulator,
    ) -> None:
        if self.progress_callback is None:
            return

        self.progress_callback(
            simulator.step_count,
            self.config.max_steps,
            simulator,
        )

    def _run_loop(
        self,
        simulator: WorldSimulator,
    ) -> str:
        """
        Execute simulation steps.

        Returns a termination reason.
        """

        simulator.start()

        while (
            simulator.status
            == SimulatorStatus.RUNNING
        ):
            if self._stop_requested:
                simulator.pause()
                return "stop_requested"

            if (
                self.stop_condition is not None
                and self.stop_condition(simulator)
            ):
                simulator.pause()
                return "custom_stop_condition"

            if (
                simulator.step_count
                >= self.config.max_steps
            ):
                simulator.pause()
                return "max_steps"

            result = simulator.step()

            if (
                result.step
                % self.config.progress_interval
                == 0
            ):
                self._notify_progress(
                    simulator
                )

            if (
                result.all_goals_reached
                and self.config
                .stop_when_goals_reached
            ):
                return "goals_reached"

            if (
                result.status
                == SimulatorStatus.EMERGENCY_STOP
            ):
                return "emergency_stop"

            if (
                result.status
                == SimulatorStatus.COMPLETED
            ):
                if result.all_goals_reached:
                    return "goals_reached"

                return "simulator_completed"

        return simulator.status.value

    def _calculate_metrics(
        self,
        simulator: WorldSimulator,
        run_id: str,
        run_index: int,
        seed: int,
        termination_reason: str,
        wall_time: float,
    ) -> WorldRunMetrics:
        report = simulator.report()

        total_robots = report.total_robots

        distances = [
            float(
                robot["distance_traveled"]
            )
            for robot
            in report.robot_statistics.values()
        ]

        total_distance = sum(distances)

        mean_distance = (
            total_distance / len(distances)
            if distances
            else 0.0
        )

        success_rate = (
            report.goals_reached
            / total_robots
            if total_robots > 0
            else 0.0
        )

        collision_rate = (
            report.total_collisions
            / max(report.total_steps, 1)
        )

        steps_per_second = (
            report.total_steps
            / wall_time
            if wall_time > 0
            else 0.0
        )

        if termination_reason in (
            "stop_requested",
            "custom_stop_condition",
        ):
            status = "stopped"

        elif termination_reason == (
            "emergency_stop"
        ):
            status = "stopped"

        else:
            status = "completed"

        return WorldRunMetrics(
            run_id=run_id,
            run_index=run_index,
            scenario_type=(
                self.config.scenario_type
            ),
            seed=seed,
            status=status,
            termination_reason=(
                termination_reason
            ),
            total_steps=report.total_steps,
            simulation_time=(
                report.simulation_time
            ),
            wall_time_seconds=wall_time,
            total_robots=total_robots,
            goals_reached=(
                report.goals_reached
            ),
            goal_success_rate=success_rate,
            total_collisions=(
                report.total_collisions
            ),
            collision_rate=collision_rate,
            total_distance_traveled=(
                total_distance
            ),
            mean_distance_traveled=(
                mean_distance
            ),
            mean_steps_per_second=(
                steps_per_second
            ),
            timestamp=self._timestamp(),
        )

    def _save_run_outputs(
        self,
        simulator: WorldSimulator,
        metrics: WorldRunMetrics,
    ) -> None:
        cfg = self.config

        if not any(
            (
                cfg.save_reports,
                cfg.save_final_image,
                cfg.save_gif,
                cfg.save_mp4,
            )
        ):
            return

        run_dir = (
            self.output_dir
            / metrics.run_id
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if cfg.save_final_image:
            image_path = (
                run_dir / "final_world.png"
            )

            simulator.save_image(
                image_path
            )

            metrics.output_files[
                "image"
            ] = str(image_path)

        renderer = simulator.renderer

        if cfg.save_gif:
            if (
                renderer is not None
                and renderer.frames
            ):
                gif_path = (
                    run_dir / "simulation.gif"
                )

                simulator.save_gif(
                    gif_path,
                    fps=cfg.gif_fps,
                )

                metrics.output_files[
                    "gif"
                ] = str(gif_path)

        if cfg.save_mp4:
            if (
                renderer is not None
                and renderer.frames
            ):
                mp4_path = (
                    run_dir / "simulation.mp4"
                )

                simulator.save_mp4(
                    mp4_path,
                    fps=cfg.mp4_fps,
                )

                metrics.output_files[
                    "mp4"
                ] = str(mp4_path)

        if cfg.save_reports:
            simulator_report_path = (
                run_dir
                / "simulator_report.json"
            )

            simulator.export_report(
                simulator_report_path
            )

            metrics.output_files[
                "simulator_report"
            ] = str(
                simulator_report_path
            )

            metrics_path = (
                run_dir / "run_metrics.json"
            )

            metrics.output_files[
                "metrics"
            ] = str(metrics_path)

            metrics_path.write_text(
                json.dumps(
                    asdict(metrics),
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

    def run_once(
        self,
        run_index: int = 1,
        seed: Optional[int] = None,
    ) -> WorldRunMetrics:
        """
        Execute one simulation experiment.
        """

        if self.status == (
            WorldRunnerStatus.RUNNING
        ):
            raise RuntimeError(
                "Runner is already executing"
            )

        self.status = (
            WorldRunnerStatus.PREPARING
        )

        selected_seed = (
            self.config.seed
            + run_index - 1
            if seed is None
            else seed
        )

        run_id = self._new_run_id(
            run_index
        )

        self.current_run_id = run_id

        simulator = None
        start_time = time.perf_counter()

        try:
            scenario = self._create_scenario(
                selected_seed
            )

            renderer = self._create_renderer(
                scenario
            )

            simulator = self._create_simulator(
                scenario,
                renderer,
            )

            self.current_simulator = simulator

            self.status = (
                WorldRunnerStatus.RUNNING
            )

            termination_reason = (
                self._run_loop(
                    simulator
                )
            )

            wall_time = (
                time.perf_counter()
                - start_time
            )

            metrics = self._calculate_metrics(
                simulator=simulator,
                run_id=run_id,
                run_index=run_index,
                seed=selected_seed,
                termination_reason=(
                    termination_reason
                ),
                wall_time=wall_time,
            )

            self._save_run_outputs(
                simulator,
                metrics,
            )

            self.status = (
                WorldRunnerStatus.STOPPED
                if metrics.status == "stopped"
                else WorldRunnerStatus.COMPLETED
            )

        except Exception as exc:
            self.status = (
                WorldRunnerStatus.FAILED
            )

            wall_time = (
                time.perf_counter()
                - start_time
            )

            metrics = WorldRunMetrics(
                run_id=run_id,
                run_index=run_index,
                scenario_type=(
                    self.config.scenario_type
                ),
                seed=selected_seed,
                status="failed",
                termination_reason="exception",
                total_steps=(
                    simulator.step_count
                    if simulator is not None
                    else 0
                ),
                simulation_time=(
                    simulator.simulation_time
                    if simulator is not None
                    else 0.0
                ),
                wall_time_seconds=wall_time,
                total_robots=(
                    len(simulator.robot_states)
                    if simulator is not None
                    else 0
                ),
                goals_reached=0,
                goal_success_rate=0.0,
                total_collisions=0,
                collision_rate=0.0,
                total_distance_traveled=0.0,
                mean_distance_traveled=0.0,
                mean_steps_per_second=0.0,
                timestamp=self._timestamp(),
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        finally:
            if simulator is not None:
                simulator.close()

            self.current_simulator = None
            self.current_run_id = None

        self.run_history.append(
            metrics
        )

        return metrics

    def run_experiments(
        self,
        num_runs: Optional[int] = None,
    ) -> ExperimentSummary:
        """
        Execute multiple simulation runs.

        Each run uses a different seed.

        Failed runs are recorded without
        aborting the complete experiment.
        """

        count = (
            self.config.num_runs
            if num_runs is None
            else num_runs
        )

        if count <= 0:
            raise ValueError(
                "num_runs must be positive"
            )

        self.clear_stop_request()

        experiment_results = []

        for index in range(1, count + 1):
            if self._stop_requested:
                break

            result = self.run_once(
                run_index=index
            )

            experiment_results.append(
                result
            )

            if result.status == "stopped":
                break

        summary = self._summarize(
            experiment_results
        )

        if self.config.save_reports:
            self.export_experiment(
                summary
            )

        return summary

    def _summarize(
        self,
        results: List[WorldRunMetrics],
    ) -> ExperimentSummary:
        """
        Calculate aggregate experiment metrics.

        Failed runs are excluded from
        performance averages.
        """

        completed = [
            result
            for result in results
            if result.status == "completed"
        ]

        failed = [
            result
            for result in results
            if result.status == "failed"
        ]

        stopped = [
            result
            for result in results
            if result.status == "stopped"
        ]

        def mean_metric(
            name: str,
        ) -> float:
            values = [
                float(
                    getattr(result, name)
                )
                for result in completed
            ]

            return (
                statistics.mean(values)
                if values
                else 0.0
            )

        best = None

        if completed:
            best = max(
                completed,
                key=lambda result: (
                    result.goal_success_rate,
                    -result.total_collisions,
                    -result.simulation_time,
                ),
            )

        return ExperimentSummary(
            experiment_name=(
                self.config.experiment_name
            ),
            total_runs=len(results),
            completed_runs=len(completed),
            failed_runs=len(failed),
            stopped_runs=len(stopped),
            mean_goal_success_rate=(
                mean_metric(
                    "goal_success_rate"
                )
            ),
            mean_collisions=mean_metric(
                "total_collisions"
            ),
            mean_simulation_time=mean_metric(
                "simulation_time"
            ),
            mean_wall_time_seconds=(
                mean_metric(
                    "wall_time_seconds"
                )
            ),
            total_goals_reached=sum(
                result.goals_reached
                for result in completed
            ),
            total_collisions=sum(
                result.total_collisions
                for result in completed
            ),
            best_run_id=(
                best.run_id
                if best is not None
                else None
            ),
            run_metrics=results,
        )

    def export_experiment(
        self,
        summary: ExperimentSummary,
    ) -> Dict[str, str]:
        """
        Export experiment JSON and CSV.
        """

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            self.output_dir
            / "experiment_summary.json"
        )

        csv_path = (
            self.output_dir
            / "experiment_runs.csv"
        )

        json_path.write_text(
            json.dumps(
                asdict(summary),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        if summary.run_metrics:
            rows = [
                asdict(result)
                for result
                in summary.run_metrics
            ]

            fieldnames = list(
                rows[0].keys()
            )

            with csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                for row in rows:
                    row["output_files"] = (
                        json.dumps(
                            row["output_files"]
                        )
                    )

                    writer.writerow(row)

        return {
            "json": str(json_path),
            "csv": (
                str(csv_path)
                if summary.run_metrics
                else ""
            ),
        }

    def statistics(self) -> Dict[str, Any]:
        """
        Return statistics across all
        runs executed by this runner.
        """

        completed = [
            result
            for result in self.run_history
            if result.status == "completed"
        ]

        return {
            "runner_status": (
                self.status.value
            ),
            "total_runs": len(
                self.run_history
            ),
            "completed_runs": len(
                completed
            ),
            "failed_runs": sum(
                result.status == "failed"
                for result in self.run_history
            ),
            "mean_success_rate": (
                statistics.mean(
                    result.goal_success_rate
                    for result in completed
                )
                if completed
                else 0.0
            ),
            "total_collisions": sum(
                result.total_collisions
                for result in completed
            ),
        }


def print_progress(
    step: int,
    max_steps: int,
    simulator: WorldSimulator,
) -> None:
    """
    Example progress callback.
    """

    percentage = (
        100.0 * step / max_steps
    )

    goals_reached = sum(
        robot.goal_reached
        for robot
        in simulator.robot_states.values()
    )

    print(
        f"[NeuroWorld] "
        f"Step {step}/{max_steps} "
        f"({percentage:.1f}%) | "
        f"Goals: {goals_reached} | "
        f"Collisions: "
        f"{simulator.total_collisions}"
    )


if __name__ == "__main__":

    config = WorldRunnerConfig(
        experiment_name=(
            "NeuroWorld_Navigation_Benchmark"
        ),
        scenario_type="random",
        world_width=10.0,
        world_height=10.0,
        num_robots=2,
        num_obstacles=10,
        timestep=0.1,
        max_steps=300,
        num_runs=3,
        seed=42,
        enable_navigation=True,
        enable_dynamic_obstacles=True,
        stop_on_collision=True,
        enable_rendering=False,
        capture_frames=False,
        save_reports=True,
        save_final_image=True,
        save_gif=False,
        save_mp4=False,
        progress_interval=50,
    )

    runner = WorldRunner(
        config=config,
        progress_callback=print_progress,
    )

    summary = runner.run_experiments()

    print("\n=== NeuroWorld Experiment ===")

    print(
        "Experiment:",
        summary.experiment_name,
    )

    print(
        "Total runs:",
        summary.total_runs,
    )

    print(
        "Completed:",
        summary.completed_runs,
    )

    print(
        "Failed:",
        summary.failed_runs,
    )

    print(
        "Mean goal success:",
        f"{summary.mean_goal_success_rate:.2%}",
    )

    print(
        "Mean collisions:",
        summary.mean_collisions,
    )

    print(
        "Best run:",
        summary.best_run_id,
    )

    print(
        "Runner statistics:",
        runner.statistics(),
    )
```
