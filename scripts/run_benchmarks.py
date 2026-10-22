
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/run_benchmarks.py

Benchmark suites:
    1. World-model next-state prediction
    2. Multi-step trajectory prediction
    3. Predictive uncertainty calibration
    4. Navigation and obstacle avoidance

Features:
    - Load NPZ transition datasets
    - Load PyTorch model checkpoints
    - Support model factory import paths
    - Evaluate MSE, MAE, RMSE, R2
    - Evaluate trajectory ADE and FDE
    - Evaluate Gaussian NLL and coverage
    - Evaluate navigation goal success
    - Track inference latency
    - Export JSON and CSV reports
    - Enforce benchmark thresholds
    - Reproducible random seeds
    - CPU/CUDA execution
    - CI/CD-compatible exit codes

Dependencies:
    pip install numpy torch

Examples:
    python scripts/run_benchmarks.py \
        --suite prediction \
        --dataset data/trajectories/test.npz \
        --model-factory \
          neuroworld.dynamics.transition_model:TransitionModel \
        --checkpoint models/checkpoints/world_model.pt

    python scripts/run_benchmarks.py \
        --suite navigation \
        --episodes 20

    python scripts/run_benchmarks.py \
        --suite all \
        --dataset data/trajectories/test.npz \
        --model-factory \
          neuroworld.dynamics.transition_model:TransitionModel \
        --checkpoint models/checkpoints/world_model.pt
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import logging
import math
import random
import sys
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("neuroworld.benchmarks")

EPS = 1e-8


@dataclass
class BenchmarkConfig:
    suite: str = "all"

    dataset_path: str = (
        "data/trajectories/test.npz"
    )

    model_factory: Optional[str] = None
    checkpoint_path: Optional[str] = None

    output_dir: str = (
        "artifacts/evaluations/benchmarks"
    )

    device: str = "auto"
    batch_size: int = 64

    trajectory_horizon: int = 10

    navigation_episodes: int = 20
    navigation_max_steps: int = 300
    navigation_obstacles: int = 10

    seed: int = 42

    max_samples: Optional[int] = None

    strict: bool = False

    max_prediction_rmse: Optional[float] = None
    max_trajectory_ade: Optional[float] = None
    min_navigation_success: Optional[float] = None


@dataclass
class BenchmarkResult:
    benchmark_name: str
    suite: str

    success: bool

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    duration_seconds: float = 0.0
    samples_evaluated: int = 0

    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    project: str
    timestamp: str

    configuration: dict[str, Any]

    total_benchmarks: int
    successful_benchmarks: int
    failed_benchmarks: int

    overall_passed: bool
    duration_seconds: float

    results: list[BenchmarkResult]


class BenchmarkRunner:
    """Run reproducible NeuroWorld benchmark suites."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        self.config = config or BenchmarkConfig()

        self._validate_config()

        self.results: list[BenchmarkResult] = []

        self.model: Any = None
        self.torch: Any = None

        self.device = "cpu"

        self._seed_everything()

    def _validate_config(self) -> None:
        c = self.config

        if c.suite not in {
            "all",
            "prediction",
            "trajectory",
            "uncertainty",
            "navigation",
        }:
            raise ValueError(
                f"Unsupported suite: {c.suite}"
            )

        if c.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if c.trajectory_horizon <= 0:
            raise ValueError(
                "trajectory_horizon must be positive"
            )

        if c.navigation_episodes <= 0:
            raise ValueError(
                "navigation_episodes must be positive"
            )

        if c.navigation_max_steps <= 0:
            raise ValueError(
                "navigation_max_steps must be positive"
            )

        if c.navigation_obstacles < 0:
            raise ValueError(
                "navigation_obstacles cannot be negative"
            )

        if (
            c.max_samples is not None
            and c.max_samples <= 0
        ):
            raise ValueError(
                "max_samples must be positive"
            )

        if (
            c.min_navigation_success is not None
            and not 0 <= c.min_navigation_success <= 1
        ):
            raise ValueError(
                "Navigation success threshold "
                "must be between 0 and 1"
            )

    @staticmethod
    def _resolve_path(
        path: str,
    ) -> Path:
        value = Path(path).expanduser()

        if not value.is_absolute():
            value = PROJECT_ROOT / value

        return value.resolve()

    def _seed_everything(self) -> None:
        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)

    def _initialize_torch(self) -> None:
        if self.torch is not None:
            return

        import torch

        self.torch = torch

        torch.manual_seed(
            self.config.seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.config.seed
            )

        if self.config.device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = self.config.device

        if (
            self.device.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable"
            )

    @staticmethod
    def _import_factory(
        import_path: str,
    ) -> Callable[..., Any]:
        if ":" not in import_path:
            raise ValueError(
                "Model factory must use "
                "'module.path:ClassName' format"
            )

        module_name, symbol_name = (
            import_path.split(":", 1)
        )

        module = importlib.import_module(
            module_name
        )

        factory = getattr(
            module,
            symbol_name,
        )

        if not callable(factory):
            raise TypeError(
                "Model factory must be callable"
            )

        return factory

    def load_model(self) -> Any:
        if self.model is not None:
            return self.model

        if not self.config.model_factory:
            raise ValueError(
                "--model-factory is required "
                "for model benchmarks"
            )

        self._initialize_torch()

        factory = self._import_factory(
            self.config.model_factory
        )

        model = factory()

        if self.config.checkpoint_path:
            checkpoint_path = self._resolve_path(
                self.config.checkpoint_path
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    checkpoint_path
                )

            # weights_only avoids general-purpose
            # pickle object loading. Use checkpoints
            # produced by trusted training pipelines.
            checkpoint = self.torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get(
                    "model_state_dict",
                    checkpoint.get(
                        "state_dict",
                        checkpoint,
                    ),
                )
            else:
                state_dict = checkpoint

            if not isinstance(
                state_dict, dict
            ):
                raise TypeError(
                    "Checkpoint must contain "
                    "a model state dictionary"
                )

            model.load_state_dict(
                state_dict,
                strict=True,
            )

        model.to(self.device)
        model.eval()

        self.model = model

        return model

    def load_dataset(
        self,
    ) -> dict[str, np.ndarray]:
        path = self._resolve_path(
            self.config.dataset_path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            required = {
                "states",
                "actions",
                "next_states",
            }

            missing = (
                required - set(archive.files)
            )

            if missing:
                raise ValueError(
                    f"Dataset missing fields: "
                    f"{sorted(missing)}"
                )

            data = {
                key: archive[key].copy()
                for key in archive.files
            }

        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]

        if (
            states.ndim != 2
            or actions.ndim != 2
            or next_states.ndim != 2
        ):
            raise ValueError(
                "States/actions must be 2D arrays"
            )

        count = len(states)

        if (
            count == 0
            or len(actions) != count
            or len(next_states) != count
        ):
            raise ValueError(
                "Invalid transition counts"
            )

        if states.shape != next_states.shape:
            raise ValueError(
                "State and next-state dimensions differ"
            )

        for key in (
            "states",
            "actions",
            "next_states",
        ):
            if not np.isfinite(
                data[key]
            ).all():
                raise ValueError(
                    f"{key} contains non-finite values"
                )

        if self.config.max_samples:
            limit = self.config.max_samples

            data = {
                key: (
                    value[:limit]
                    if value.ndim >= 1
                    and len(value) == count
                    else value
                )
                for key, value in data.items()
            }

        return data

    def _synchronize_device(self) -> None:
        if (
            self.torch is not None
            and self.device.startswith("cuda")
        ):
            self.torch.cuda.synchronize()

    def _predict_batch(
        self,
        states: Any,
        actions: Any,
    ) -> tuple[Any, Optional[Any]]:
        """
        Supported model outputs:
            Tensor next_state
            (mean, logvar)
            Object with .mean and .logvar
            Object with .next_state
        """

        model = self.load_model()

        if hasattr(model, "predict"):
            try:
                output = model.predict(
                    states,
                    actions,
                    deterministic=True,
                )
            except TypeError:
                output = model.predict(
                    states,
                    actions,
                )
        else:
            output = model(
                states,
                actions,
            )

        torch = self.torch

        if isinstance(output, torch.Tensor):
            return output, None

        if isinstance(output, (tuple, list)):
            if not output:
                raise ValueError(
                    "Empty model output"
                )

            mean = output[0]

            logvar = (
                output[1]
                if len(output) > 1
                else None
            )

            return mean, logvar

        mean = getattr(
            output,
            "mean",
            None,
        )

        logvar = getattr(
            output,
            "logvar",
            None,
        )

        if isinstance(mean, torch.Tensor):
            return mean, logvar

        next_state = getattr(
            output,
            "next_state",
            None,
        )

        if isinstance(
            next_state, torch.Tensor
        ):
            return next_state, logvar

        raise TypeError(
            "Unsupported model prediction output"
        )

    def _batched_predictions(
        self,
        data: dict[str, np.ndarray],
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        dict[str, float],
    ]:
        torch = self.torch

        if torch is None:
            self._initialize_torch()
            torch = self.torch

        predictions = []
        logvars = []

        uncertainty_available = True

        latencies = []

        states = data["states"]
        actions = data["actions"]

        with torch.inference_mode():
            for start in range(
                0,
                len(states),
                self.config.batch_size,
            ):
                end = (
                    start
                    + self.config.batch_size
                )

                state_tensor = torch.as_tensor(
                    states[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                action_tensor = torch.as_tensor(
                    actions[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )

                self._synchronize_device()

                start_time = (
                    time.perf_counter()
                )

                mean, logvar = (
                    self._predict_batch(
                        state_tensor,
                        action_tensor,
                    )
                )

                self._synchronize_device()

                latencies.append(
                    time.perf_counter()
                    - start_time
                )

                predictions.append(
                    mean.detach()
                    .cpu()
                    .numpy()
                )

                if logvar is None:
                    uncertainty_available = False
                else:
                    logvars.append(
                        logvar.detach()
                        .cpu()
                        .numpy()
                    )

        prediction_array = np.concatenate(
            predictions,
            axis=0,
        )

        logvar_array = (
            np.concatenate(
                logvars,
                axis=0,
            )
            if uncertainty_available
            else None
        )

        latency_metrics = {
            "mean_batch_latency_ms": (
                float(
                    np.mean(latencies)
                ) * 1000
            ),
            "p95_batch_latency_ms": (
                float(
                    np.percentile(
                        latencies, 95
                    )
                ) * 1000
            ),
            "samples_per_second": (
                float(
                    len(states)
                    / max(
                        sum(latencies),
                        EPS,
                    )
                )
            ),
        }

        return (
            prediction_array,
            logvar_array,
            latency_metrics,
        )

    @staticmethod
    def _regression_metrics(
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, float]:
        if predicted.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ"
            )

        errors = (
            predicted.astype(np.float64)
            - target.astype(np.float64)
        )

        mse = float(
            np.mean(errors**2)
        )

        mae = float(
            np.mean(np.abs(errors))
        )

        rmse = math.sqrt(mse)

        total_variance = float(
            np.sum(
                (
                    target
                    - target.mean(axis=0)
                ) ** 2
            )
        )

        residual_sum = float(
            np.sum(errors**2)
        )

        r2 = (
            1.0
            - residual_sum / total_variance
            if total_variance > EPS
            else float("nan")
        )

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    def benchmark_prediction(
        self,
    ) -> BenchmarkResult:
        started = time.perf_counter()

        data = self.load_dataset()
        self.load_model()

        predicted, _, latency = (
            self._batched_predictions(
                data
            )
        )

        metrics = self._regression_metrics(
            predicted,
            data["next_states"],
        )

        metrics.update(latency)

        threshold = (
            self.config.max_prediction_rmse
        )

        passed = (
            threshold is None
            or metrics["rmse"] <= threshold
        )

        if threshold is not None:
            metrics["rmse_threshold"] = threshold

        return BenchmarkResult(
            benchmark_name="next_state_prediction",
            suite="prediction",
            success=passed,
            metrics=metrics,
            duration_seconds=(
                time.perf_counter() - started
            ),
            samples_evaluated=len(
                data["states"]
            ),
        )

    @staticmethod
    def _trajectory_windows(
        data: dict[str, np.ndarray],
        horizon: int,
    ) -> list[list[int]]:
        """
        Construct contiguous windows for
        a single robot within an episode.

        Windows cannot cross episode,
        robot, terminal, or step boundaries.
        """

        required = {
            "episode_ids",
            "robot_ids",
            "steps",
        }

        if not required.issubset(data):
            raise ValueError(
                "Trajectory benchmark requires "
                "episode_ids, robot_ids and steps"
            )

        count = len(data["states"])

        terminated = data.get(
            "terminated",
            np.zeros(
                count,
                dtype=bool,
            ),
        )

        truncated = data.get(
            "truncated",
            np.zeros(
                count,
                dtype=bool,
            ),
        )

        groups: dict[
            tuple[int, str],
            list[int],
        ] = {}

        for index in range(count):
            key = (
                int(
                    data["episode_ids"][index]
                ),
                str(
                    data["robot_ids"][index]
                ),
            )

            groups.setdefault(
                key, []
            ).append(index)

        windows = []

        for indices in groups.values():
            indices.sort(
                key=lambda i: int(
                    data["steps"][i]
                )
            )

            for start in range(
                len(indices) - horizon + 1
            ):
                window = indices[
                    start:start + horizon
                ]

                valid = True

                for position in range(
                    len(window) - 1
                ):
                    first = window[position]
                    second = window[position + 1]

                    if (
                        int(
                            data["steps"][second]
                        )
                        != int(
                            data["steps"][first]
                        ) + 1
                    ):
                        valid = False
                        break

                    if (
                        terminated[first]
                        or truncated[first]
                    ):
                        valid = False
                        break

                if valid:
                    windows.append(
                        window
                    )

        return windows

    def benchmark_trajectory(
        self,
    ) -> BenchmarkResult:
        started = time.perf_counter()

        data = self.load_dataset()
        self.load_model()

        horizon = (
            self.config.trajectory_horizon
        )

        windows = self._trajectory_windows(
            data,
            horizon,
        )

        if not windows:
            raise ValueError(
                "No contiguous trajectory windows "
                "found for the requested horizon"
            )

        torch = self.torch

        position_errors = []

        with torch.inference_mode():
            for window in windows:
                first = window[0]

                state = torch.as_tensor(
                    data["states"][
                        first:first + 1
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )

                predicted_positions = []

                target_positions = []

                for index in window:
                    action = torch.as_tensor(
                        data["actions"][
                            index:index + 1
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    state, _ = (
                        self._predict_batch(
                            state,
                            action,
                        )
                    )

                    predicted_positions.append(
                        state[
                            0, :2
                        ].detach()
                        .cpu()
                        .numpy()
                    )

                    target_positions.append(
                        data["next_states"][
                            index, :2
                        ]
                    )

                predicted_array = np.asarray(
                    predicted_positions
                )

                target_array = np.asarray(
                    target_positions
                )

                errors = np.linalg.norm(
                    predicted_array
                    - target_array,
                    axis=-1,
                )

                position_errors.append(
                    errors
                )

        error_matrix = np.asarray(
            position_errors
        )

        ade = float(
            np.mean(error_matrix)
        )

        fde = float(
            np.mean(
                error_matrix[:, -1]
            )
        )

        metrics = {
            "ade": ade,
            "fde": fde,
            "horizon": horizon,
            "trajectory_windows": len(
                windows
            ),
            "mean_error_by_step": (
                error_matrix.mean(
                    axis=0
                ).tolist()
            ),
        }

        threshold = (
            self.config.max_trajectory_ade
        )

        passed = (
            threshold is None
            or ade <= threshold
        )

        if threshold is not None:
            metrics["ade_threshold"] = threshold

        return BenchmarkResult(
            benchmark_name="trajectory_prediction",
            suite="trajectory",
            success=passed,
            metrics=metrics,
            duration_seconds=(
                time.perf_counter() - started
            ),
            samples_evaluated=len(
                windows
            ),
        )

    def benchmark_uncertainty(
        self,
    ) -> BenchmarkResult:
        started = time.perf_counter()

        data = self.load_dataset()
        self.load_model()

        predicted, logvar, _ = (
            self._batched_predictions(
                data
            )
        )

        if logvar is None:
            raise ValueError(
                "Model does not return predictive "
                "log-variance"
            )

        target = data["next_states"]

        if (
            predicted.shape != target.shape
            or logvar.shape != target.shape
        ):
            raise ValueError(
                "Uncertainty tensor shape mismatch"
            )

        logvar = np.clip(
            logvar.astype(np.float64),
            -20.0,
            10.0,
        )

        variance = np.exp(logvar)

        standard_deviation = np.sqrt(
            variance
        )

        errors = (
            target.astype(np.float64)
            - predicted.astype(np.float64)
        )

        nll = 0.5 * (
            math.log(2 * math.pi)
            + logvar
            + errors**2 / variance
        )

        normalized_errors = (
            np.abs(errors)
            / np.maximum(
                standard_deviation,
                EPS,
            )
        )

        coverage_68 = float(
            np.mean(
                normalized_errors <= 1.0
            )
        )

        coverage_95 = float(
            np.mean(
                normalized_errors <= 1.96
            )
        )

        metrics = {
            "gaussian_nll": float(
                np.mean(nll)
            ),
            "coverage_68": coverage_68,
            "coverage_95": coverage_95,
            "coverage_68_error": abs(
                coverage_68 - 0.6827
            ),
            "coverage_95_error": abs(
                coverage_95 - 0.95
            ),
            "mean_predictive_std": float(
                np.mean(
                    standard_deviation
                )
            ),
        }

        return BenchmarkResult(
            benchmark_name="uncertainty_calibration",
            suite="uncertainty",
            success=True,
            metrics=metrics,
            duration_seconds=(
                time.perf_counter() - started
            ),
            samples_evaluated=len(
                target
            ),
        )

    def benchmark_navigation(
        self,
    ) -> BenchmarkResult:
        """
        Benchmark autonomous navigation in
        generated simulation scenarios.

        Uses the existing NeuroWorld
        Scenario and WorldSimulator APIs.
        """

        started = time.perf_counter()

        from neuroworld.environment.scenario import (
            Scenario,
            ScenarioConfig,
        )

        from neuroworld.environment.simulator import (
            SimulatorConfig,
            SimulatorStatus,
            WorldSimulator,
        )

        successes = 0
        total_collisions = 0
        total_steps = 0

        episode_results = []

        for episode_index in range(
            self.config.navigation_episodes
        ):
            seed = (
                self.config.seed
                + episode_index
            )

            scenario = Scenario(
                ScenarioConfig(
                    name=(
                        f"Benchmark_{episode_index}"
                    ),
                    scenario_type="random",
                    world_width=10.0,
                    world_height=10.0,
                    num_robots=1,
                    num_obstacles=(
                        self.config.navigation_obstacles
                    ),
                    seed=seed,
                )
            )

            scenario.reset()

            simulator = WorldSimulator(
                scenario=scenario,
                config=SimulatorConfig(
                    timestep=0.1,
                    max_steps=(
                        self.config.navigation_max_steps
                    ),
                    enable_navigation=True,
                    enable_collision_detection=True,
                    stop_on_collision=True,
                    enable_rendering=False,
                    capture_frames=False,
                    seed=seed,
                ),
            )

            try:
                simulator.reset(
                    regenerate_scenario=False
                )

                simulator.start()

                for _ in range(
                    self.config.navigation_max_steps
                ):
                    result = simulator.step()

                    if result.status in {
                        SimulatorStatus.COMPLETED,
                        SimulatorStatus.EMERGENCY_STOP,
                        SimulatorStatus.ERROR,
                    }:
                        break

                report = simulator.report()

                success = (
                    report.total_robots > 0
                    and report.goals_reached
                    == report.total_robots
                )

                if success:
                    successes += 1

                total_collisions += (
                    report.total_collisions
                )

                total_steps += (
                    report.total_steps
                )

                episode_results.append(
                    {
                        "episode": episode_index,
                        "seed": seed,
                        "success": success,
                        "steps": (
                            report.total_steps
                        ),
                        "collisions": (
                            report.total_collisions
                        ),
                    }
                )

            finally:
                simulator.close()

        count = len(
            episode_results
        )

        success_rate = (
            successes / count
            if count
            else 0.0
        )

        metrics = {
            "success_rate": success_rate,
            "successful_episodes": successes,
            "episodes": count,
            "total_collisions": total_collisions,
            "mean_collisions": (
                total_collisions / count
                if count
                else 0.0
            ),
            "mean_episode_steps": (
                total_steps / count
                if count
                else 0.0
            ),
            "episode_results": episode_results,
        }

        threshold = (
            self.config.min_navigation_success
        )

        passed = (
            threshold is None
            or success_rate >= threshold
        )

        if threshold is not None:
            metrics[
                "success_rate_threshold"
            ] = threshold

        return BenchmarkResult(
            benchmark_name="navigation_success",
            suite="navigation",
            success=passed,
            metrics=metrics,
            duration_seconds=(
                time.perf_counter() - started
            ),
            samples_evaluated=count,
        )

    def _run_one(
        self,
        name: str,
        suite: str,
        function: Callable[
            [], BenchmarkResult
        ],
    ) -> None:
        started = time.perf_counter()

        LOGGER.info(
            "Running benchmark: %s",
            name,
        )

        try:
            result = function()

        except Exception as exc:
            LOGGER.exception(
                "Benchmark failed: %s",
                name,
            )

            result = BenchmarkResult(
                benchmark_name=name,
                suite=suite,
                success=False,
                duration_seconds=(
                    time.perf_counter()
                    - started
                ),
                error=str(exc),
            )

        self.results.append(
            result
        )

    def run(
        self,
    ) -> BenchmarkReport:
        self.results.clear()

        started = time.perf_counter()

        suites = {
            "prediction": (
                self.benchmark_prediction
            ),
            "trajectory": (
                self.benchmark_trajectory
            ),
            "uncertainty": (
                self.benchmark_uncertainty
            ),
            "navigation": (
                self.benchmark_navigation
            ),
        }

        selected = (
            list(suites)
            if self.config.suite == "all"
            else [self.config.suite]
        )

        for suite_name in selected:
            self._run_one(
                name=suite_name,
                suite=suite_name,
                function=suites[
                    suite_name
                ],
            )

        passed = sum(
            result.success
            for result in self.results
        )

        return BenchmarkReport(
            project="NeuroWorld AI",
            timestamp=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            configuration=asdict(
                self.config
            ),
            total_benchmarks=len(
                self.results
            ),
            successful_benchmarks=passed,
            failed_benchmarks=(
                len(self.results) - passed
            ),
            overall_passed=(
                passed == len(self.results)
                and len(self.results) > 0
            ),
            duration_seconds=(
                time.perf_counter() - started
            ),
            results=list(
                self.results
            ),
        )

    def export(
        self,
        report: BenchmarkReport,
    ) -> dict[str, str]:
        output_dir = self._resolve_path(
            self.config.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            output_dir / "benchmark_report.json"
        )

        csv_path = (
            output_dir / "benchmark_metrics.csv"
        )

        json_path.write_text(
            json.dumps(
                asdict(report),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        rows = []

        for result in report.results:
            for metric, value in (
                result.metrics.items()
            ):
                if isinstance(
                    value,
                    (int, float, str, bool),
                ):
                    rows.append(
                        {
                            "benchmark": (
                                result.benchmark_name
                            ),
                            "suite": result.suite,
                            "passed": result.success,
                            "metric": metric,
                            "value": value,
                        }
                    )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "benchmark",
                    "suite",
                    "passed",
                    "metric",
                    "value",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        return {
            "json": str(json_path),
            "csv": str(csv_path),
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI benchmark runner"
        )
    )

    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "prediction",
            "trajectory",
            "uncertainty",
            "navigation",
        ],
        default="all",
    )

    parser.add_argument(
        "--dataset",
        default=(
            "data/trajectories/test.npz"
        ),
    )

    parser.add_argument(
        "--model-factory",
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--navigation-obstacles",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default=(
            "artifacts/evaluations/benchmarks"
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-rmse",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-ade",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--min-navigation-success",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
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

    config = BenchmarkConfig(
        suite=args.suite,
        dataset_path=args.dataset,
        model_factory=args.model_factory,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        trajectory_horizon=args.horizon,
        navigation_episodes=args.episodes,
        navigation_max_steps=args.max_steps,
        navigation_obstacles=(
            args.navigation_obstacles
        ),
        seed=args.seed,
        max_samples=args.max_samples,
        strict=args.strict,
        max_prediction_rmse=args.max_rmse,
        max_trajectory_ade=args.max_ade,
        min_navigation_success=(
            args.min_navigation_success
        ),
    )

    runner = BenchmarkRunner(
        config
    )

    report = runner.run()

    output_paths = runner.export(
        report
    )

    print()
    print("=" * 60)
    print("       NEUROWORLD AI BENCHMARK RESULTS")
    print("=" * 60)

    for result in report.results:
        status = (
            "PASS"
            if result.success
            else "FAIL"
        )

        print(
            f"\n[{status}] "
            f"{result.benchmark_name}"
        )

        if result.error:
            print(
                "  Error:",
                result.error,
            )

        for name, value in (
            result.metrics.items()
        ):
            if isinstance(
                value,
                (int, float, str, bool),
            ):
                print(
                    f"  {name}: {value}"
                )

    print()
    print(
        "Overall:",
        (
            "PASSED"
            if report.overall_passed
            else "FAILED"
        ),
    )

    print(
        "JSON:",
        output_paths["json"],
    )

    print(
        "CSV:",
        output_paths["csv"],
    )

    return (
        0
        if report.overall_passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
