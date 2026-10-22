
"""
NeuroWorld AI
backend/app/services/evaluation_service.py

Evaluation service for neural world models, imagination,
planning, simulation and model-based reinforcement learning.

Features:
    - Prediction MAE / MSE / RMSE / R2
    - Trajectory ADE / FDE
    - Temporal consistency
    - Physics consistency
    - Classification accuracy / precision / recall / F1
    - Uncertainty calibration
    - Planning success rate
    - Simulation episode statistics
    - Benchmark evaluation
    - Model comparison
    - JSON report export
    - Thread-safe in-memory evaluation history

Dependencies:
    pip install numpy pydantic

Important:
    Metrics are calculated from actual supplied data.
    No trained model or dataset is bundled here.
    Persistence is in-memory unless integrated with
    the database repository layer.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# EXCEPTIONS
# ============================================================

class EvaluationServiceError(Exception):
    """Base evaluation service exception."""


class InvalidEvaluationData(EvaluationServiceError):
    """Invalid metric input."""


class EvaluationNotFound(EvaluationServiceError):
    """Evaluation record does not exist."""


# ============================================================
# DATA MODELS
# ============================================================

class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str
    model_name: str
    evaluation_type: str
    created_at: datetime
    sample_count: int = Field(ge=0)
    metrics: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    benchmark_id: str
    benchmark_name: str
    created_at: datetime
    results: list[EvaluationResult]
    summary: dict[str, float]


class ModelComparison(BaseModel):
    comparison_id: str
    created_at: datetime
    metric_name: str
    lower_is_better: bool
    ranking: list[dict[str, Any]]


@dataclass(frozen=True)
class SimulationEpisode:
    reward: float
    steps: int
    success: bool
    collisions: int = 0
    duration_seconds: float = 0.0


# ============================================================
# ARRAY VALIDATION
# ============================================================

def _array(
    values: Any,
    name: str,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidEvaluationData(
            f"{name} must contain numeric values"
        ) from exc

    if array.size == 0:
        raise InvalidEvaluationData(
            f"{name} cannot be empty"
        )

    if ndim is not None and array.ndim != ndim:
        raise InvalidEvaluationData(
            f"{name} must have {ndim} dimensions"
        )

    if not np.all(np.isfinite(array)):
        raise InvalidEvaluationData(
            f"{name} contains NaN or infinity"
        )

    return array


def _paired(
    actual: Any,
    predicted: Any,
) -> tuple[np.ndarray, np.ndarray]:
    actual_array = _array(actual, "actual")
    predicted_array = _array(predicted, "predicted")

    if actual_array.shape != predicted_array.shape:
        raise InvalidEvaluationData(
            "Actual and predicted shapes must match: "
            f"{actual_array.shape} != {predicted_array.shape}"
        )

    return actual_array, predicted_array


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise InvalidEvaluationData(
            "At least one value is required"
        )

    return float(np.mean(values))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# PREDICTION METRICS
# ============================================================

def prediction_metrics(
    actual: Any,
    predicted: Any,
) -> dict[str, float]:
    """
    Regression metrics for predictions.

    Inputs may have any matching numeric shape.
    R2 is computed over all flattened values.

    For constant targets, R2 is reported as 1.0
    for perfect predictions and 0.0 otherwise.
    """

    y_true, y_pred = _paired(actual, predicted)

    error = y_pred - y_true

    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))

    target = y_true.ravel()
    prediction = y_pred.ravel()

    ss_res = float(
        np.sum((target - prediction) ** 2)
    )

    ss_tot = float(
        np.sum((target - np.mean(target)) ** 2)
    )

    if ss_tot <= 1e-15:
        r2 = 1.0 if ss_res <= 1e-15 else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": float(r2),
    }


# ============================================================
# TRAJECTORY METRICS
# ============================================================

def trajectory_metrics(
    actual: Any,
    predicted: Any,
) -> dict[str, float]:
    """
    Evaluate predicted trajectories.

    Shape:
        [batch, time, coordinates]

    Coordinates may be 2D or 3D.

    ADE = average displacement error.
    FDE = final displacement error.
    """

    y_true, y_pred = _paired(actual, predicted)

    if y_true.ndim != 3:
        raise InvalidEvaluationData(
            "Trajectories must have shape "
            "[batch, time, coordinates]"
        )

    if y_true.shape[2] not in (2, 3):
        raise InvalidEvaluationData(
            "Trajectory coordinates must be 2D or 3D"
        )

    displacement = np.linalg.norm(
        y_pred - y_true,
        axis=-1,
    )

    ade = float(np.mean(displacement))
    fde = float(np.mean(displacement[:, -1]))

    return {
        "ade": ade,
        "fde": fde,
        "max_displacement_error": float(
            np.max(displacement)
        ),
        "trajectory_rmse": float(
            np.sqrt(np.mean(displacement ** 2))
        ),
    }


# ============================================================
# TEMPORAL CONSISTENCY
# ============================================================

def temporal_consistency_metrics(
    actual: Any,
    predicted: Any,
) -> dict[str, float]:
    """
    Compare frame-to-frame or state-to-state changes.

    Shape:
        [batch, time, features]

    At least two timesteps are required.
    """

    y_true, y_pred = _paired(actual, predicted)

    if y_true.ndim != 3 or y_true.shape[1] < 2:
        raise InvalidEvaluationData(
            "Expected [batch, time>=2, features]"
        )

    actual_changes = np.diff(
        y_true,
        axis=1,
    )

    predicted_changes = np.diff(
        y_pred,
        axis=1,
    )

    delta_error = (
        predicted_changes - actual_changes
    )

    return {
        "temporal_mae": float(
            np.mean(np.abs(delta_error))
        ),
        "temporal_mse": float(
            np.mean(delta_error ** 2)
        ),
        "temporal_rmse": float(
            np.sqrt(np.mean(delta_error ** 2))
        ),
    }


# ============================================================
# PHYSICS CONSISTENCY
# ============================================================

def physics_consistency_metrics(
    positions: Any,
    *,
    time_step: float,
    max_acceleration: float | None = None,
) -> dict[str, float]:
    """
    Compute motion diagnostics from positions.

    Shape:
        [batch, time, coordinates]

    This evaluates velocity/acceleration smoothness.
    It is not a complete physical-law validator.
    """

    values = _array(
        positions,
        "positions",
        ndim=3,
    )

    if values.shape[1] < 3:
        raise InvalidEvaluationData(
            "At least three timesteps are required"
        )

    if values.shape[2] not in (2, 3):
        raise InvalidEvaluationData(
            "Coordinates must be 2D or 3D"
        )

    if (
        not math.isfinite(time_step)
        or time_step <= 0
    ):
        raise InvalidEvaluationData(
            "time_step must be positive and finite"
        )

    velocity = np.diff(
        values,
        axis=1,
    ) / time_step

    acceleration = np.diff(
        velocity,
        axis=1,
    ) / time_step

    speed = np.linalg.norm(
        velocity,
        axis=-1,
    )

    acceleration_magnitude = np.linalg.norm(
        acceleration,
        axis=-1,
    )

    metrics = {
        "mean_speed": float(np.mean(speed)),
        "max_speed": float(np.max(speed)),
        "mean_acceleration": float(
            np.mean(acceleration_magnitude)
        ),
        "max_acceleration": float(
            np.max(acceleration_magnitude)
        ),
    }

    if max_acceleration is not None:
        if (
            not math.isfinite(max_acceleration)
            or max_acceleration < 0
        ):
            raise InvalidEvaluationData(
                "max_acceleration must be nonnegative"
            )

        violations = (
            acceleration_magnitude
            > max_acceleration
        )

        metrics["acceleration_violation_rate"] = float(
            np.mean(violations)
        )

    return metrics


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    actual: Any,
    predicted: Any,
    *,
    positive_label: int = 1,
) -> dict[str, float]:
    """
    Binary classification metrics.

    Labels must be 0 or 1.
    """

    y_true, y_pred = _paired(actual, predicted)

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    if not (
        np.all(np.isin(y_true, [0, 1]))
        and np.all(np.isin(y_pred, [0, 1]))
    ):
        raise InvalidEvaluationData(
            "Classification labels must be binary"
        )

    if positive_label not in (0, 1):
        raise InvalidEvaluationData(
            "positive_label must be 0 or 1"
        )

    tp = int(np.sum(
        (y_true == positive_label)
        & (y_pred == positive_label)
    ))

    fp = int(np.sum(
        (y_true != positive_label)
        & (y_pred == positive_label)
    ))

    fn = int(np.sum(
        (y_true == positive_label)
        & (y_pred != positive_label)
    ))

    tn = int(np.sum(
        (y_true != positive_label)
        & (y_pred != positive_label)
    ))

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


# ============================================================
# UNCERTAINTY CALIBRATION
# ============================================================

def uncertainty_calibration_metrics(
    actual: Any,
    predicted: Any,
    standard_deviation: Any,
) -> dict[str, float]:
    """
    Evaluate Gaussian predictive uncertainty.

    Assumes predicted means and standard deviations.

    Metrics:
        Gaussian NLL
        68% interval coverage
        95% interval coverage
        Mean predicted standard deviation

    These metrics are meaningful only when the model's
    predictive distribution is approximately Gaussian.
    """

    y_true, y_pred = _paired(actual, predicted)

    std = _array(
        standard_deviation,
        "standard_deviation",
    )

    if std.shape != y_true.shape:
        raise InvalidEvaluationData(
            "Standard deviation shape must match targets"
        )

    if np.any(std <= 0):
        raise InvalidEvaluationData(
            "Standard deviations must be positive"
        )

    error = y_true - y_pred

    variance = std ** 2

    nll = 0.5 * (
        np.log(2 * np.pi * variance)
        + (error ** 2) / variance
    )

    absolute_error = np.abs(error)

    return {
        "gaussian_nll": float(np.mean(nll)),
        "coverage_68": float(
            np.mean(absolute_error <= std)
        ),
        "coverage_95": float(
            np.mean(absolute_error <= 1.96 * std)
        ),
        "mean_predicted_std": float(
            np.mean(std)
        ),
        "mean_absolute_error": float(
            np.mean(absolute_error)
        ),
    }


# ============================================================
# PLANNING METRICS
# ============================================================

def planning_metrics(
    successes: Sequence[bool],
    *,
    path_lengths: Sequence[float] | None = None,
    planning_times: Sequence[float] | None = None,
) -> dict[str, float]:
    """
    Evaluate planning episodes.
    """

    if not successes:
        raise InvalidEvaluationData(
            "At least one planning result is required"
        )

    if not all(
        isinstance(value, (bool, np.bool_))
        for value in successes
    ):
        raise InvalidEvaluationData(
            "successes must contain booleans"
        )

    metrics = {
        "success_rate": float(
            np.mean(successes)
        ),
        "episode_count": float(len(successes)),
        "successful_episodes": float(
            sum(successes)
        ),
    }

    if path_lengths is not None:
        paths = _array(
            path_lengths,
            "path_lengths",
            ndim=1,
        )

        if len(paths) != len(successes):
            raise InvalidEvaluationData(
                "path_lengths count must match successes"
            )

        if np.any(paths < 0):
            raise InvalidEvaluationData(
                "Path lengths cannot be negative"
            )

        metrics["mean_path_length"] = float(
            np.mean(paths)
        )

    if planning_times is not None:
        times = _array(
            planning_times,
            "planning_times",
            ndim=1,
        )

        if len(times) != len(successes):
            raise InvalidEvaluationData(
                "planning_times count must match successes"
            )

        if np.any(times < 0):
            raise InvalidEvaluationData(
                "Planning times cannot be negative"
            )

        metrics["mean_planning_time_seconds"] = float(
            np.mean(times)
        )

    return metrics


# ============================================================
# SIMULATION METRICS
# ============================================================

def simulation_metrics(
    episodes: Sequence[SimulationEpisode],
) -> dict[str, float]:

    if not episodes:
        raise InvalidEvaluationData(
            "At least one simulation episode is required"
        )

    for episode in episodes:
        if episode.steps < 0 or episode.collisions < 0:
            raise InvalidEvaluationData(
                "Steps and collisions cannot be negative"
            )

        if not (
            math.isfinite(episode.reward)
            and math.isfinite(episode.duration_seconds)
        ):
            raise InvalidEvaluationData(
                "Episode values must be finite"
            )

        if episode.duration_seconds < 0:
            raise InvalidEvaluationData(
                "Episode duration cannot be negative"
            )

    rewards = [e.reward for e in episodes]
    steps = [e.steps for e in episodes]
    successes = [e.success for e in episodes]
    collisions = [e.collisions for e in episodes]
    durations = [
        e.duration_seconds
        for e in episodes
    ]

    return {
        "episode_count": float(len(episodes)),
        "mean_reward": _mean(rewards),
        "reward_std": float(np.std(rewards)),
        "mean_steps": _mean(steps),
        "success_rate": float(np.mean(successes)),
        "mean_collisions": _mean(collisions),
        "total_collisions": float(sum(collisions)),
        "mean_duration_seconds": _mean(durations),
    }


# ============================================================
# EVALUATION SERVICE
# ============================================================

class EvaluationService:
    """
    Central evaluation service.

    Stores evaluation results in memory.

    For persistence across restarts, connect the
    evaluation repository and database session.
    """

    def __init__(
        self,
        report_directory: str | Path | None = None,
    ) -> None:

        self._lock = threading.RLock()

        self._results: dict[
            str,
            EvaluationResult,
        ] = {}

        self._benchmarks: dict[
            str,
            BenchmarkResult,
        ] = {}

        self._report_directory = (
            Path(report_directory).resolve()
            if report_directory is not None
            else None
        )

    def _save_result(
        self,
        *,
        model_name: str,
        evaluation_type: str,
        sample_count: int,
        metrics: dict[str, float],
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:

        if not model_name.strip():
            raise InvalidEvaluationData(
                "model_name cannot be empty"
            )

        if sample_count < 0:
            raise InvalidEvaluationData(
                "sample_count cannot be negative"
            )

        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                raise InvalidEvaluationData(
                    f"Metric '{name}' is not finite"
                )

        result = EvaluationResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=model_name,
            evaluation_type=evaluation_type,
            created_at=_utcnow(),
            sample_count=sample_count,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self._results[result.evaluation_id] = result

        return result

    def evaluate_prediction(
        self,
        *,
        model_name: str,
        actual: Any,
        predicted: Any,
    ) -> EvaluationResult:

        metrics = prediction_metrics(
            actual,
            predicted,
        )

        count = _array(actual, "actual").shape[0]

        return self._save_result(
            model_name=model_name,
            evaluation_type="prediction",
            sample_count=count,
            metrics=metrics,
        )

    def evaluate_trajectory(
        self,
        *,
        model_name: str,
        actual: Any,
        predicted: Any,
    ) -> EvaluationResult:

        metrics = trajectory_metrics(
            actual,
            predicted,
        )

        count = _array(actual, "actual").shape[0]

        return self._save_result(
            model_name=model_name,
            evaluation_type="trajectory",
            sample_count=count,
            metrics=metrics,
        )

    def evaluate_temporal_consistency(
        self,
        *,
        model_name: str,
        actual: Any,
        predicted: Any,
    ) -> EvaluationResult:

        metrics = temporal_consistency_metrics(
            actual,
            predicted,
        )

        count = _array(actual, "actual").shape[0]

        return self._save_result(
            model_name=model_name,
            evaluation_type="temporal_consistency",
            sample_count=count,
            metrics=metrics,
        )

    def evaluate_physics(
        self,
        *,
        model_name: str,
        positions: Any,
        time_step: float,
        max_acceleration: float | None = None,
    ) -> EvaluationResult:

        metrics = physics_consistency_metrics(
            positions,
            time_step=time_step,
            max_acceleration=max_acceleration,
        )

        count = _array(
            positions,
            "positions",
        ).shape[0]

        return self._save_result(
            model_name=model_name,
            evaluation_type="physics_consistency",
            sample_count=count,
            metrics=metrics,
        )

    def evaluate_classification(
        self,
        *,
        model_name: str,
        actual: Any,
        predicted: Any,
    ) -> EvaluationResult:

        metrics = classification_metrics(
            actual,
            predicted,
        )

        count = _array(
            actual,
            "actual",
        ).size

        return self._save_result(
            model_name=model_name,
            evaluation_type="classification",
            sample_count=count,
            metrics=metrics,
        )

    def evaluate_uncertainty(
        self,
        *,
        model_name: str,
        actual: Any,
        predicted: Any,
        standard_deviation: Any,
    ) -> EvaluationResult:

        metrics = uncertainty_calibration_metrics(
            actual,
            predicted,
            standard_deviation,
        )

        count = _array(
            actual,
            "actual",
        ).size

        return self._save_result(
            model_name=model_name,
            evaluation_type="uncertainty_calibration",
            sample_count=count,
            metrics=metrics,
        )

    def evaluate_planning(
        self,
        *,
        model_name: str,
        successes: Sequence[bool],
        path_lengths: Sequence[float] | None = None,
        planning_times: Sequence[float] | None = None,
    ) -> EvaluationResult:

        metrics = planning_metrics(
            successes,
            path_lengths=path_lengths,
            planning_times=planning_times,
        )

        return self._save_result(
            model_name=model_name,
            evaluation_type="planning",
            sample_count=len(successes),
            metrics=metrics,
        )

    def evaluate_simulation(
        self,
        *,
        model_name: str,
        episodes: Sequence[SimulationEpisode],
    ) -> EvaluationResult:

        metrics = simulation_metrics(episodes)

        return self._save_result(
            model_name=model_name,
            evaluation_type="simulation",
            sample_count=len(episodes),
            metrics=metrics,
        )

    def get_result(
        self,
        evaluation_id: str,
    ) -> EvaluationResult:

        with self._lock:
            result = self._results.get(evaluation_id)

        if result is None:
            raise EvaluationNotFound(
                f"Evaluation '{evaluation_id}' not found"
            )

        return result

    def list_results(
        self,
        *,
        model_name: str | None = None,
        evaluation_type: str | None = None,
        limit: int = 100,
    ) -> list[EvaluationResult]:

        if not 1 <= limit <= 1000:
            raise ValueError(
                "limit must be between 1 and 1000"
            )

        with self._lock:
            results = list(self._results.values())

        if model_name is not None:
            results = [
                result
                for result in results
                if result.model_name == model_name
            ]

        if evaluation_type is not None:
            results = [
                result
                for result in results
                if result.evaluation_type == evaluation_type
            ]

        results.sort(
            key=lambda result: result.created_at,
            reverse=True,
        )

        return results[:limit]

    def compare_models(
        self,
        *,
        evaluation_ids: Sequence[str],
        metric_name: str,
        lower_is_better: bool = True,
    ) -> ModelComparison:

        if len(evaluation_ids) < 2:
            raise InvalidEvaluationData(
                "At least two evaluations are required"
            )

        results = [
            self.get_result(evaluation_id)
            for evaluation_id in evaluation_ids
        ]

        types = {
            result.evaluation_type
            for result in results
        }

        if len(types) != 1:
            raise InvalidEvaluationData(
                "Cannot compare different evaluation types"
            )

        ranking = []

        for result in results:
            if metric_name not in result.metrics:
                raise InvalidEvaluationData(
                    f"Metric '{metric_name}' is missing "
                    f"for {result.model_name}"
                )

            ranking.append({
                "evaluation_id": result.evaluation_id,
                "model_name": result.model_name,
                "metric": metric_name,
                "value": result.metrics[metric_name],
            })

        ranking.sort(
            key=lambda item: item["value"],
            reverse=not lower_is_better,
        )

        return ModelComparison(
            comparison_id=uuid.uuid4().hex,
            created_at=_utcnow(),
            metric_name=metric_name,
            lower_is_better=lower_is_better,
            ranking=ranking,
        )

    def run_benchmark(
        self,
        *,
        benchmark_name: str,
        evaluations: Sequence[
            Callable[[], EvaluationResult]
        ],
    ) -> BenchmarkResult:
        """
        Run supplied evaluation callables.

        Each callable must return an EvaluationResult.
        """

        if not benchmark_name.strip():
            raise InvalidEvaluationData(
                "benchmark_name cannot be empty"
            )

        if not evaluations:
            raise InvalidEvaluationData(
                "At least one evaluation is required"
            )

        results = []

        for evaluation in evaluations:
            result = evaluation()

            if not isinstance(result, EvaluationResult):
                raise InvalidEvaluationData(
                    "Benchmark callable must return "
                    "EvaluationResult"
                )

            results.append(result)

        # Only aggregate metrics present in every result.
        shared_metrics = set(
            results[0].metrics
        )

        for result in results[1:]:
            shared_metrics &= set(result.metrics)

        summary = {
            f"mean_{metric}": float(
                np.mean([
                    result.metrics[metric]
                    for result in results
                ])
            )
            for metric in sorted(shared_metrics)
        }

        benchmark = BenchmarkResult(
            benchmark_id=uuid.uuid4().hex,
            benchmark_name=benchmark_name,
            created_at=_utcnow(),
            results=results,
            summary=summary,
        )

        with self._lock:
            self._benchmarks[
                benchmark.benchmark_id
            ] = benchmark

        return benchmark

    def get_benchmark(
        self,
        benchmark_id: str,
    ) -> BenchmarkResult:

        with self._lock:
            benchmark = self._benchmarks.get(
                benchmark_id
            )

        if benchmark is None:
            raise EvaluationNotFound(
                f"Benchmark '{benchmark_id}' not found"
            )

        return benchmark

    def export_result(
        self,
        evaluation_id: str,
        *,
        filename: str | None = None,
    ) -> Path:
        """
        Export a stored result to the configured
        report directory.

        A generated filename is used by default.
        """

        if self._report_directory is None:
            raise EvaluationServiceError(
                "Report directory is not configured"
            )

        result = self.get_result(
            evaluation_id
        )

        directory = self._report_directory
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        if filename is None:
            filename = (
                f"evaluation_{evaluation_id}.json"
            )

        if (
            Path(filename).name != filename
            or not filename.endswith(".json")
        ):
            raise InvalidEvaluationData(
                "Invalid report filename"
            )

        destination = (
            directory / filename
        ).resolve()

        if destination.parent != directory:
            raise InvalidEvaluationData(
                "Report path escapes export directory"
            )

        # Exclusive create avoids silently overwriting
        # an existing report.
        with destination.open(
            "x",
            encoding="utf-8",
        ) as output:
            json.dump(
                result.model_dump(mode="json"),
                output,
                indent=2,
                ensure_ascii=False,
            )

        return destination


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_evaluation_service() -> EvaluationService:
    """
    Create an evaluation service with a local report
    directory.
    """

    project_root = Path(__file__).resolve().parents[3]

    return EvaluationService(
        report_directory=(
            project_root / "experiments" / "reports"
        )
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = create_evaluation_service()

    actual = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ]

    predicted = [
        [1.1, 1.9, 3.2],
        [2.1, 2.8, 4.1],
    ]

    prediction_result = service.evaluate_prediction(
        model_name="demo_world_transformer",
        actual=actual,
        predicted=predicted,
    )

    trajectory_actual = np.array([
        [[0, 0], [1, 1], [2, 2]],
        [[0, 0], [1, 0], [2, 0]],
    ])

    trajectory_predicted = np.array([
        [[0, 0], [1.1, 0.9], [2.2, 2.1]],
        [[0, 0], [0.9, 0.1], [1.8, 0.1]],
    ])

    trajectory_result = service.evaluate_trajectory(
        model_name="demo_world_transformer",
        actual=trajectory_actual,
        predicted=trajectory_predicted,
    )

    simulation_result = service.evaluate_simulation(
        model_name="demo_planning_agent",
        episodes=[
            SimulationEpisode(
                reward=10.0,
                steps=100,
                success=True,
                collisions=0,
                duration_seconds=2.1,
            ),
            SimulationEpisode(
                reward=4.0,
                steps=150,
                success=False,
                collisions=2,
                duration_seconds=3.4,
            ),
        ],
    )

    print("=" * 65)
    print("NEUROWORLD AI — EVALUATION SERVICE")
    print("=" * 65)

    for result in (
        prediction_result,
        trajectory_result,
        simulation_result,
    ):
        print()
        print(
            result.model_dump_json(indent=2)
        )

    print()
    print("Evaluation service demonstration completed.")
    print("Results above use synthetic demonstration data.")


if __name__ == "__main__":
    main()
