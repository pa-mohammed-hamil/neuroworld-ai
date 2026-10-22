
"""
NeuroWorld AI
evaluation/benchmark_runner.py

Reproducible Neural World Model Benchmark Runner.

Features:
- World-model prediction accuracy
- Multi-step trajectory prediction
- Temporal consistency
- Action controllability integration
- Uncertainty calibration
- Noise robustness
- Goal-directed planning success
- Configurable evaluation thresholds
- Weighted benchmark score
- Multiple model comparison
- JSON and CSV report export
- Reproducible synthetic test datasets

Dependencies:
    pip install numpy

Run:
    python -m evaluation.benchmark_runner

Important:
    Synthetic benchmarks test implementation behavior,
    not real-world model accuracy or safety.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from evaluation.action_controllability import (
    ActionControllabilityEvaluator,
    ControllabilityConfig,
)


logger = logging.getLogger(
    "neuroworld.evaluation.benchmark_runner"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class BenchmarkError(Exception):
    pass


class BenchmarkValidationError(BenchmarkError):
    pass


class BenchmarkModelError(BenchmarkError):
    pass


# ============================================================
# MODEL PROTOCOL
# ============================================================

class WorldModelProtocol(Protocol):
    name: str
    state_dim: int
    action_dim: int

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        ...


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    random_seed: int = 42

    sample_count: int = 128
    trajectory_horizon: int = 12

    action_scale: float = 1.0
    observation_noise_std: float = 0.05

    prediction_error_scale: float = 1.0
    trajectory_error_scale: float = 1.0

    max_samples: int = 10000
    max_horizon: int = 256

    pass_threshold: float = 0.70

    output_directory: str = (
        "experiments/reports/benchmarks"
    )

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "prediction_accuracy": 0.20,
            "trajectory_accuracy": 0.15,
            "temporal_consistency": 0.10,
            "action_controllability": 0.20,
            "uncertainty_calibration": 0.10,
            "robustness": 0.10,
            "planning_success": 0.15,
        }
    )

    def __post_init__(self):
        if not 1 <= self.sample_count <= self.max_samples:
            raise ValueError(
                "Invalid sample_count"
            )

        if not 1 <= self.trajectory_horizon <= self.max_horizon:
            raise ValueError(
                "Invalid trajectory_horizon"
            )

        for value in (
            self.action_scale,
            self.prediction_error_scale,
            self.trajectory_error_scale,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Scale parameters must be finite and positive"
                )

        if (
            not math.isfinite(self.observation_noise_std)
            or self.observation_noise_std < 0
        ):
            raise ValueError(
                "Invalid observation_noise_std"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        if not self.weights:
            raise ValueError(
                "Benchmark weights cannot be empty"
            )

        if any(
            not math.isfinite(value) or value < 0
            for value in self.weights.values()
        ):
            raise ValueError(
                "Benchmark weights must be nonnegative"
            )

        if sum(self.weights.values()) <= 0:
            raise ValueError(
                "Benchmark weights must have positive total"
            )


# ============================================================
# BENCHMARK DATASET
# ============================================================

@dataclass
class BenchmarkDataset:
    states: np.ndarray
    actions: np.ndarray
    target_next_states: np.ndarray
    goals: np.ndarray

    trajectory_actions: np.ndarray
    trajectory_targets: np.ndarray

    def validate(
        self,
        state_dim: int,
        action_dim: int,
    ) -> None:
        arrays = (
            self.states,
            self.actions,
            self.target_next_states,
            self.goals,
            self.trajectory_actions,
            self.trajectory_targets,
        )

        if not all(
            isinstance(array, np.ndarray)
            and np.issubdtype(array.dtype, np.number)
            and np.all(np.isfinite(array))
            for array in arrays
        ):
            raise BenchmarkValidationError(
                "Dataset arrays must contain finite numbers"
            )

        n = len(self.states)

        if n < 1:
            raise BenchmarkValidationError(
                "Dataset cannot be empty"
            )

        expected = {
            "states": (n, state_dim),
            "actions": (n, action_dim),
            "target_next_states": (n, state_dim),
            "goals": (n, state_dim),
        }

        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise BenchmarkValidationError(
                    f"{name} must have shape {shape}"
                )

        if self.trajectory_actions.ndim != 3:
            raise BenchmarkValidationError(
                "trajectory_actions must be 3D"
            )

        horizon = self.trajectory_actions.shape[1]

        if horizon < 1:
            raise BenchmarkValidationError(
                "Trajectory horizon must be positive"
            )

        if self.trajectory_actions.shape != (
            n, horizon, action_dim
        ):
            raise BenchmarkValidationError(
                "Invalid trajectory_actions shape"
            )

        if self.trajectory_targets.shape != (
            n, horizon, state_dim
        ):
            raise BenchmarkValidationError(
                "Invalid trajectory_targets shape"
            )


# ============================================================
# BENCHMARK RESULT
# ============================================================

@dataclass
class BenchmarkMetrics:
    prediction_accuracy: float
    trajectory_accuracy: float
    temporal_consistency: float
    action_controllability: float
    uncertainty_calibration: float
    robustness: float
    planning_success: float
    overall_score: float


@dataclass
class BenchmarkResult:
    benchmark_id: str
    model_name: str

    sample_count: int
    trajectory_horizon: int

    metrics: BenchmarkMetrics

    passed: bool
    created_at: str

    raw_metrics: dict[str, float] = field(
        default_factory=dict
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalized_error_score(
    error: float,
    scale: float,
) -> float:
    return float(
        np.clip(
            math.exp(-error / scale),
            0.0,
            1.0,
        )
    )


def calculate_rmse(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    if predicted.shape != target.shape:
        raise BenchmarkValidationError(
            "Prediction and target shapes differ"
        )

    return float(
        np.sqrt(
            np.mean(
                np.square(predicted - target)
            )
        )
    )


def safe_mean(
    values: list[float],
) -> float:
    return (
        float(np.mean(values))
        if values
        else 0.0
    )


# ============================================================
# SYNTHETIC DATASET GENERATOR
# ============================================================

def generate_synthetic_dataset(
    *,
    state_dim: int = 4,
    action_dim: int = 4,
    sample_count: int = 128,
    horizon: int = 12,
    seed: int = 42,
    action_scale: float = 1.0,
    dynamics_scale: float = 0.5,
) -> BenchmarkDataset:
    """
    Generate a deterministic synthetic benchmark.

    Dynamics:
        next_state = state + B @ action

    The matrix B maps action space to state space.

    Synthetic targets are generated independently
    from the evaluated model.
    """

    if min(
        state_dim,
        action_dim,
        sample_count,
        horizon,
    ) < 1:
        raise BenchmarkValidationError(
            "Dimensions and counts must be positive"
        )

    rng = np.random.default_rng(seed)

    states = rng.normal(
        0,
        1,
        size=(sample_count, state_dim),
    )

    actions = rng.uniform(
        -action_scale,
        action_scale,
        size=(sample_count, action_dim),
    )

    transition_matrix = np.zeros(
        (state_dim, action_dim),
        dtype=np.float64,
    )

    for index in range(
        min(state_dim, action_dim)
    ):
        transition_matrix[index, index] = (
            dynamics_scale
        )

    target_next_states = (
        states + actions @ transition_matrix.T
    )

    goals = (
        states
        + 2.0 * actions @ transition_matrix.T
    )

    trajectory_actions = rng.uniform(
        -action_scale,
        action_scale,
        size=(
            sample_count,
            horizon,
            action_dim,
        ),
    )

    trajectory_targets = np.empty(
        (
            sample_count,
            horizon,
            state_dim,
        ),
        dtype=np.float64,
    )

    current = states.copy()

    for step in range(horizon):
        current = (
            current
            + trajectory_actions[:, step, :]
            @ transition_matrix.T
        )

        trajectory_targets[:, step, :] = current

    dataset = BenchmarkDataset(
        states=states,
        actions=actions,
        target_next_states=target_next_states,
        goals=goals,
        trajectory_actions=trajectory_actions,
        trajectory_targets=trajectory_targets,
    )

    dataset.validate(
        state_dim,
        action_dim,
    )

    return dataset


# ============================================================
# BENCHMARK RUNNER
# ============================================================

class BenchmarkRunner:
    """
    Run reproducible world-model benchmarks.

    Model predictions are compared against
    independently supplied ground-truth targets.

    Scores are normalized heuristic diagnostics,
    not universal model-quality guarantees.
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.config = (
            config or BenchmarkConfig()
        )

        self._results: dict[
            str,
            BenchmarkResult,
        ] = {}

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    def _predict(
        self,
        model: WorldModelProtocol,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        try:
            transition = model.predict_transition(
                state.copy(),
                action.copy(),
            )

            next_state = np.asarray(
                transition.next_state,
                dtype=np.float64,
            )

            uncertainty = float(
                getattr(
                    transition,
                    "uncertainty",
                    0.0,
                )
            )

        except Exception as exc:
            raise BenchmarkModelError(
                f"Model prediction failed: {exc}"
            ) from exc

        if next_state.shape != (
            model.state_dim,
        ):
            raise BenchmarkModelError(
                "Model returned invalid state dimension"
            )

        if not np.all(np.isfinite(next_state)):
            raise BenchmarkModelError(
                "Model returned non-finite states"
            )

        if (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise BenchmarkModelError(
                "Model returned invalid uncertainty"
            )

        return next_state, uncertainty

    # --------------------------------------------------------
    # NEXT-STATE PREDICTION
    # --------------------------------------------------------

    def evaluate_prediction(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> tuple[float, float]:
        predictions = []

        for state, action in zip(
            dataset.states,
            dataset.actions,
        ):
            predicted, _ = self._predict(
                model,
                state,
                action,
            )

            predictions.append(predicted)

        predicted_array = np.stack(
            predictions
        )

        rmse = calculate_rmse(
            predicted_array,
            dataset.target_next_states,
        )

        score = normalized_error_score(
            rmse,
            self.config.prediction_error_scale,
        )

        return score, rmse

    # --------------------------------------------------------
    # TRAJECTORY PREDICTION
    # --------------------------------------------------------

    def evaluate_trajectory(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> tuple[float, float]:
        predicted_trajectories = []

        horizon = (
            dataset.trajectory_actions.shape[1]
        )

        for sample_index, initial_state in enumerate(
            dataset.states
        ):
            state = initial_state.copy()
            predicted_steps = []

            for step in range(horizon):
                action = (
                    dataset.trajectory_actions[
                        sample_index,
                        step,
                    ]
                )

                state, _ = self._predict(
                    model,
                    state,
                    action,
                )

                predicted_steps.append(
                    state.copy()
                )

            predicted_trajectories.append(
                predicted_steps
            )

        predictions = np.asarray(
            predicted_trajectories,
            dtype=np.float64,
        )

        rmse = calculate_rmse(
            predictions,
            dataset.trajectory_targets,
        )

        score = normalized_error_score(
            rmse,
            self.config.trajectory_error_scale,
        )

        return score, rmse

    # --------------------------------------------------------
    # TEMPORAL CONSISTENCY
    # --------------------------------------------------------

    def evaluate_temporal_consistency(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> float:
        """
        Measures repeatability of identical
        state-action transitions.

        Does not independently verify temporal
        physical correctness.
        """

        scores = []

        for state, action in zip(
            dataset.states,
            dataset.actions,
        ):
            first, _ = self._predict(
                model,
                state,
                action,
            )

            second, _ = self._predict(
                model,
                state,
                action,
            )

            difference = float(
                np.linalg.norm(
                    first - second
                )
            )

            scores.append(
                normalized_error_score(
                    difference,
                    self.config.prediction_error_scale,
                )
            )

        return safe_mean(scores)

    # --------------------------------------------------------
    # ACTION CONTROLLABILITY
    # --------------------------------------------------------

    def evaluate_action_controllability(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> tuple[float, dict[str, Any]]:
        evaluator = ActionControllabilityEvaluator(
            model=model,
            config=ControllabilityConfig(
                action_scale=self.config.action_scale,
                max_samples=self.config.max_samples,
                random_seed=self.config.random_seed,
            ),
        )

        result = evaluator.evaluate(
            states=dataset.states,
            actions=dataset.actions,
            goals=dataset.goals,
        )

        return (
            result.metrics.controllability_score,
            asdict(result.metrics),
        )

    # --------------------------------------------------------
    # UNCERTAINTY CALIBRATION
    # --------------------------------------------------------

    def evaluate_uncertainty_calibration(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> tuple[float, float]:
        """
        Compares reported uncertainty with
        observed normalized prediction error.

        This is a proxy calibration metric.
        """

        uncertainties = []
        observed_errors = []

        for state, action, target in zip(
            dataset.states,
            dataset.actions,
            dataset.target_next_states,
        ):
            predicted, uncertainty = self._predict(
                model,
                state,
                action,
            )

            error = float(
                np.linalg.norm(
                    predicted - target
                )
            )

            normalized_error = float(
                np.clip(
                    error
                    / self.config.prediction_error_scale,
                    0.0,
                    1.0,
                )
            )

            uncertainties.append(uncertainty)
            observed_errors.append(normalized_error)

        calibration_mae = float(
            np.mean(
                np.abs(
                    np.asarray(uncertainties)
                    - np.asarray(observed_errors)
                )
            )
        )

        return (
            float(
                np.clip(
                    1.0 - calibration_mae,
                    0.0,
                    1.0,
                )
            ),
            calibration_mae,
        )

    # --------------------------------------------------------
    # ROBUSTNESS
    # --------------------------------------------------------

    def evaluate_robustness(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> tuple[float, float]:
        """
        Measures prediction stability under
        small perturbations to input states.

        Robustness is relative to clean outputs;
        it does not imply prediction correctness.
        """

        rng = np.random.default_rng(
            self.config.random_seed + 100
        )

        differences = []

        for state, action in zip(
            dataset.states,
            dataset.actions,
        ):
            clean, _ = self._predict(
                model,
                state,
                action,
            )

            noisy_state = (
                state
                + rng.normal(
                    0.0,
                    self.config.observation_noise_std,
                    size=state.shape,
                )
            )

            noisy, _ = self._predict(
                model,
                noisy_state,
                action,
            )

            differences.append(
                float(
                    np.linalg.norm(
                        clean - noisy
                    )
                )
            )

        mean_difference = safe_mean(
            differences
        )

        return (
            normalized_error_score(
                mean_difference,
                self.config.prediction_error_scale,
            ),
            mean_difference,
        )

    # --------------------------------------------------------
    # GOAL-DIRECTED PLANNING PROXY
    # --------------------------------------------------------

    def evaluate_planning_success(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> float:
        """
        Measures whether supplied actions
        move the model's state closer to goals.

        This evaluates action outcomes, not
        a full planning algorithm.
        """

        successes = []

        for state, action, goal in zip(
            dataset.states,
            dataset.actions,
            dataset.goals,
        ):
            predicted, _ = self._predict(
                model,
                state,
                action,
            )

            before = np.linalg.norm(
                goal - state
            )

            after = np.linalg.norm(
                goal - predicted
            )

            successes.append(
                float(after < before)
            )

        return safe_mean(successes)

    # --------------------------------------------------------
    # COMPLETE BENCHMARK
    # --------------------------------------------------------

    def run(
        self,
        model: WorldModelProtocol,
        dataset: BenchmarkDataset,
    ) -> BenchmarkResult:
        dataset.validate(
            model.state_dim,
            model.action_dim,
        )

        if len(dataset.states) > self.config.max_samples:
            raise BenchmarkValidationError(
                "Dataset exceeds max_samples"
            )

        horizon = (
            dataset.trajectory_actions.shape[1]
        )

        if horizon > self.config.max_horizon:
            raise BenchmarkValidationError(
                "Dataset exceeds max_horizon"
            )

        prediction_score, prediction_rmse = (
            self.evaluate_prediction(
                model,
                dataset,
            )
        )

        trajectory_score, trajectory_rmse = (
            self.evaluate_trajectory(
                model,
                dataset,
            )
        )

        temporal_score = (
            self.evaluate_temporal_consistency(
                model,
                dataset,
            )
        )

        controllability_score, controllability_details = (
            self.evaluate_action_controllability(
                model,
                dataset,
            )
        )

        calibration_score, calibration_mae = (
            self.evaluate_uncertainty_calibration(
                model,
                dataset,
            )
        )

        robustness_score, robustness_difference = (
            self.evaluate_robustness(
                model,
                dataset,
            )
        )

        planning_score = (
            self.evaluate_planning_success(
                model,
                dataset,
            )
        )

        scores = {
            "prediction_accuracy": prediction_score,
            "trajectory_accuracy": trajectory_score,
            "temporal_consistency": temporal_score,
            "action_controllability": controllability_score,
            "uncertainty_calibration": calibration_score,
            "robustness": robustness_score,
            "planning_success": planning_score,
        }

        unknown_weights = (
            set(self.config.weights)
            - set(scores)
        )

        if unknown_weights:
            raise BenchmarkValidationError(
                f"Unknown metric weights: {sorted(unknown_weights)}"
            )

        total_weight = sum(
            self.config.weights.values()
        )

        overall_score = sum(
            scores[name] * weight
            for name, weight in self.config.weights.items()
        ) / total_weight

        metrics = BenchmarkMetrics(
            **scores,
            overall_score=float(overall_score),
        )

        result = BenchmarkResult(
            benchmark_id=uuid.uuid4().hex,
            model_name=model.name,
            sample_count=len(dataset.states),
            trajectory_horizon=horizon,
            metrics=metrics,
            passed=(
                overall_score
                >= self.config.pass_threshold
            ),
            created_at=utc_now(),
            raw_metrics={
                "prediction_rmse": prediction_rmse,
                "trajectory_rmse": trajectory_rmse,
                "uncertainty_calibration_mae": calibration_mae,
                "robustness_mean_difference": robustness_difference,
            },
            details={
                "random_seed": self.config.random_seed,
                "weights": dict(self.config.weights),
                "pass_threshold": self.config.pass_threshold,
                "action_controllability": controllability_details,
                "planning_success_type": (
                    "goal-directed action proxy"
                ),
                "dataset_type": (
                    "user_supplied_or_synthetic"
                ),
            },
        )

        self._results[
            result.benchmark_id
        ] = result

        logger.info(
            "Benchmark completed: model=%s score=%.4f passed=%s",
            model.name,
            overall_score,
            result.passed,
        )

        return result

    # --------------------------------------------------------
    # MULTI-MODEL COMPARISON
    # --------------------------------------------------------

    def compare_models(
        self,
        models: list[WorldModelProtocol],
        dataset: BenchmarkDataset,
    ) -> list[BenchmarkResult]:
        if not models:
            raise BenchmarkValidationError(
                "At least one model is required"
            )

        results = [
            self.run(model, dataset)
            for model in models
        ]

        return sorted(
            results,
            key=lambda result: (
                result.metrics.overall_score
            ),
            reverse=True,
        )

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        benchmark_id: str,
    ) -> BenchmarkResult:
        result = self._results.get(
            benchmark_id
        )

        if result is None:
            raise BenchmarkError(
                "Benchmark result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[BenchmarkResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_json(
        self,
        result: BenchmarkResult,
        path: str | Path,
    ) -> Path:
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(result),
                file,
                indent=2,
                allow_nan=False,
            )

        return output_path

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    def export_csv(
        self,
        results: list[BenchmarkResult],
        path: str | Path,
    ) -> Path:
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for result in results:
            row = {
                "benchmark_id": result.benchmark_id,
                "model_name": result.model_name,
                "sample_count": result.sample_count,
                "trajectory_horizon": result.trajectory_horizon,
                "passed": result.passed,
                "created_at": result.created_at,
            }

            row.update(
                asdict(result.metrics)
            )

            row.update(
                result.raw_metrics
            )

            rows.append(row)

        if not rows:
            raise BenchmarkValidationError(
                "No results available for CSV export"
            )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0]),
            )

            writer.writeheader()
            writer.writerows(rows)

        return output_path


# ============================================================
# FACTORY
# ============================================================

def create_benchmark_runner(
    config: BenchmarkConfig | None = None,
) -> BenchmarkRunner:
    return BenchmarkRunner(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

class DemoWorldModel:
    """
    Synthetic reference model matching
    the generated benchmark dynamics.

    Not a trained neural world model.
    """

    name = "demo_world_model"
    state_dim = 4
    action_dim = 4

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            next_state=state + action * 0.5,
            uncertainty=0.05,
            reward=0.0,
            terminal=False,
        )


def main() -> None:
    config = BenchmarkConfig(
        sample_count=128,
        trajectory_horizon=12,
        random_seed=42,
    )

    dataset = generate_synthetic_dataset(
        state_dim=4,
        action_dim=4,
        sample_count=config.sample_count,
        horizon=config.trajectory_horizon,
        seed=config.random_seed,
    )

    model = DemoWorldModel()

    runner = create_benchmark_runner(
        config=config
    )

    result = runner.run(
        model,
        dataset,
    )

    print("=" * 65)
    print("NEUROWORLD AI — BENCHMARK RUNNER")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    output_directory = Path(
        config.output_directory
    )

    json_path = runner.export_json(
        result,
        output_directory / "benchmark_result.json",
    )

    csv_path = runner.export_csv(
        [result],
        output_directory / "benchmark_results.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
