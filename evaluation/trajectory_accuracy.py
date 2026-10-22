
"""
NeuroWorld AI
evaluation/trajectory_accuracy.py

Trajectory Accuracy Evaluation for Neural World Models.

Features:
- Multi-step autoregressive rollout
- Trajectory RMSE and MAE
- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Position and velocity accuracy
- Horizon-wise error analysis
- Trajectory drift
- Teacher-forced transition accuracy
- Rollout / teacher-forced disagreement
- Uncertainty-error correlation
- Uncertainty threshold diagnostics
- Weighted trajectory score
- Model comparison
- JSON / CSV report export
- Reproducible synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.trajectory_accuracy

World-model interface:
    model.name
    model.state_dim
    model.action_dim
    model.predict_transition(state, action)

Expected transition:
    transition.next_state
    transition.uncertainty (optional)

Dataset:
    initial_states: (N, S)
    actions: (N, H, A)
    target_states: (N, H, S)

Important:
    Reference trajectories must be independently
    generated or collected.

    The demonstration uses deterministic synthetic
    dynamics, not a trained neural world model.
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


logger = logging.getLogger(
    "neuroworld.evaluation.trajectory_accuracy"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TrajectoryAccuracyError(Exception):
    pass


class TrajectoryValidationError(TrajectoryAccuracyError):
    pass


class TrajectoryModelError(TrajectoryAccuracyError):
    pass


# ============================================================
# WORLD MODEL INTERFACE
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
class TrajectoryAccuracyConfig:
    """
    State indices are configurable because different
    world models use different state representations.

    Default state:
        [x, y, vx, vy]

    For other representations, specify position_indices
    and velocity_indices explicitly.

    When velocity_indices=None, velocity accuracy
    is excluded from the overall score.
    """

    position_indices: tuple[int, ...] = (0, 1)
    velocity_indices: tuple[int, ...] | None = (2, 3)

    max_samples: int = 10000
    max_horizon: int = 256

    state_error_scale: float = 1.0
    position_error_scale: float = 1.0
    velocity_error_scale: float = 1.0
    drift_error_scale: float = 1.0
    consistency_error_scale: float = 1.0

    uncertainty_threshold: float = 1.0

    pass_threshold: float = 0.70

    epsilon: float = 1e-8

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "trajectory_accuracy": 0.25,
            "position_accuracy": 0.20,
            "velocity_accuracy": 0.10,
            "final_position_accuracy": 0.15,
            "drift_score": 0.10,
            "teacher_forced_accuracy": 0.10,
            "rollout_consistency": 0.10,
        }
    )

    def __post_init__(self):
        if self.max_samples < 1:
            raise ValueError(
                "max_samples must be positive"
            )

        if self.max_horizon < 1:
            raise ValueError(
                "max_horizon must be positive"
            )

        for name in (
            "state_error_scale",
            "position_error_scale",
            "velocity_error_scale",
            "drift_error_scale",
            "consistency_error_scale",
            "epsilon",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError(
                "uncertainty_threshold must be in [0, 1]"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        if not self.position_indices:
            raise ValueError(
                "At least one position index is required"
            )

        if (
            len(set(self.position_indices))
            != len(self.position_indices)
            or min(self.position_indices) < 0
        ):
            raise ValueError(
                "Invalid position indices"
            )

        if self.velocity_indices is not None:
            if (
                not self.velocity_indices
                or len(set(self.velocity_indices))
                != len(self.velocity_indices)
                or min(self.velocity_indices) < 0
            ):
                raise ValueError(
                    "Invalid velocity indices"
                )

        allowed = {
            "trajectory_accuracy",
            "position_accuracy",
            "velocity_accuracy",
            "final_position_accuracy",
            "drift_score",
            "teacher_forced_accuracy",
            "rollout_consistency",
        }

        if (
            not self.weights
            or set(self.weights) - allowed
            or any(
                not math.isfinite(weight) or weight < 0
                for weight in self.weights.values()
            )
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Invalid trajectory metric weights"
            )

    def validate_state_layout(
        self,
        state_dim: int,
    ) -> None:
        if max(self.position_indices) >= state_dim:
            raise TrajectoryValidationError(
                "Position index exceeds state dimension"
            )

        if self.velocity_indices is not None:
            if max(self.velocity_indices) >= state_dim:
                raise TrajectoryValidationError(
                    "Velocity index exceeds state dimension"
                )


# ============================================================
# DATASET
# ============================================================

@dataclass
class TrajectoryAccuracyDataset:
    """
    initial_states:
        (N, state_dim)

    actions:
        (N, horizon, action_dim)

    target_states:
        (N, horizon, state_dim)

    target_states[:, 0] is the reference state
    after executing actions[:, 0].

    The target states are expected to be aligned
    with the action timestamps.
    """

    initial_states: np.ndarray
    actions: np.ndarray
    target_states: np.ndarray

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        state_dim: int,
        action_dim: int,
        *,
        max_samples: int,
        max_horizon: int,
    ) -> None:
        try:
            states = np.asarray(
                self.initial_states,
                dtype=np.float64,
            )

            actions = np.asarray(
                self.actions,
                dtype=np.float64,
            )

            targets = np.asarray(
                self.target_states,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise TrajectoryValidationError(
                "Dataset arrays must be numeric"
            ) from exc

        if states.ndim != 2:
            raise TrajectoryValidationError(
                "initial_states must be 2D"
            )

        if actions.ndim != 3:
            raise TrajectoryValidationError(
                "actions must be 3D"
            )

        if targets.ndim != 3:
            raise TrajectoryValidationError(
                "target_states must be 3D"
            )

        n = states.shape[0]
        horizon = actions.shape[1]

        if not 1 <= n <= max_samples:
            raise TrajectoryValidationError(
                "Invalid sample count"
            )

        if not 1 <= horizon <= max_horizon:
            raise TrajectoryValidationError(
                "Invalid prediction horizon"
            )

        if states.shape != (n, state_dim):
            raise TrajectoryValidationError(
                "Invalid initial state shape"
            )

        if actions.shape != (
            n,
            horizon,
            action_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid action shape"
            )

        if targets.shape != (
            n,
            horizon,
            state_dim,
        ):
            raise TrajectoryValidationError(
                "Invalid target state shape"
            )

        if not all(
            np.all(np.isfinite(array))
            for array in (
                states,
                actions,
                targets,
            )
        ):
            raise TrajectoryValidationError(
                "Dataset contains non-finite values"
            )

        self.initial_states = states.copy()
        self.actions = actions.copy()
        self.target_states = targets.copy()


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class HorizonAccuracyMetrics:
    horizon_step: int

    state_rmse: float
    state_mae: float

    position_ade: float
    position_rmse: float

    velocity_rmse: float | None

    accuracy_score: float

    mean_uncertainty: float | None


@dataclass
class TrajectoryAccuracyMetrics:
    trajectory_accuracy: float

    position_accuracy: float
    velocity_accuracy: float | None

    final_position_accuracy: float

    drift_score: float

    teacher_forced_accuracy: float
    rollout_consistency: float

    uncertainty_error_correlation: float | None
    uncertainty_acceptance_rate: float | None

    overall_score: float


@dataclass
class TrajectoryAccuracyResult:
    evaluation_id: str
    model_name: str

    sample_count: int
    horizon: int

    metrics: TrajectoryAccuracyMetrics

    horizon_metrics: list[HorizonAccuracyMetrics]

    passed: bool
    created_at: str

    raw_metrics: dict[str, float]

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPERS
# ============================================================

def calculate_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    if predictions.shape != targets.shape:
        raise TrajectoryValidationError(
            "Prediction and target shapes differ"
        )

    return float(
        np.sqrt(
            np.mean(
                np.square(
                    predictions - targets
                )
            )
        )
    )


def calculate_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    if predictions.shape != targets.shape:
        raise TrajectoryValidationError(
            "Prediction and target shapes differ"
        )

    return float(
        np.mean(
            np.abs(
                predictions - targets
            )
        )
    )


def error_to_score(
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


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float,
) -> float | None:
    x = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)

    if x.shape != y.shape:
        raise TrajectoryValidationError(
            "Correlation input shapes differ"
        )

    if (
        len(x) < 2
        or np.std(x) <= epsilon
        or np.std(y) <= epsilon
    ):
        return None

    correlation = float(
        np.corrcoef(x, y)[0, 1]
    )

    return (
        correlation
        if math.isfinite(correlation)
        else None
    )


# ============================================================
# EVALUATOR
# ============================================================

class TrajectoryAccuracyEvaluator:
    """
    Evaluate predicted trajectories against
    independently supplied reference trajectories.

    Supports autoregressive rollouts and
    teacher-forced transition predictions.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: TrajectoryAccuracyConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or TrajectoryAccuracyConfig()
        )

        self.config.validate_state_layout(
            model.state_dim
        )

        self._results: dict[
            str,
            TrajectoryAccuracyResult,
        ] = {}

    # --------------------------------------------------------
    # SINGLE TRANSITION
    # --------------------------------------------------------

    def _predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float | None]:
        try:
            transition = (
                self.model.predict_transition(
                    state.copy(),
                    action.copy(),
                )
            )

            predicted = np.asarray(
                transition.next_state,
                dtype=np.float64,
            )

            raw_uncertainty = getattr(
                transition,
                "uncertainty",
                None,
            )

            uncertainty = (
                float(raw_uncertainty)
                if raw_uncertainty is not None
                else None
            )

        except Exception as exc:
            raise TrajectoryModelError(
                f"World-model prediction failed: {exc}"
            ) from exc

        if predicted.shape != (
            self.model.state_dim,
        ):
            raise TrajectoryModelError(
                "Invalid predicted state dimensions"
            )

        if not np.all(np.isfinite(predicted)):
            raise TrajectoryModelError(
                "Predicted state contains non-finite values"
            )

        if uncertainty is not None and (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise TrajectoryModelError(
                "Invalid uncertainty value"
            )

        return predicted.copy(), uncertainty

    # --------------------------------------------------------
    # AUTOREGRESSIVE ROLLOUT
    # --------------------------------------------------------

    def predict_rollout(
        self,
        dataset: TrajectoryAccuracyDataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = []
        uncertainties = []

        horizon = dataset.actions.shape[1]

        for sample_index, initial_state in enumerate(
            dataset.initial_states
        ):
            state = initial_state.copy()

            sample_predictions = []
            sample_uncertainties = []

            for step in range(horizon):
                action = (
                    dataset.actions[
                        sample_index,
                        step,
                    ]
                )

                state, uncertainty = self._predict(
                    state,
                    action,
                )

                sample_predictions.append(
                    state.copy()
                )

                sample_uncertainties.append(
                    np.nan
                    if uncertainty is None
                    else uncertainty
                )

            predictions.append(
                sample_predictions
            )

            uncertainties.append(
                sample_uncertainties
            )

        return (
            np.asarray(
                predictions,
                dtype=np.float64,
            ),
            np.asarray(
                uncertainties,
                dtype=np.float64,
            ),
        )

    # --------------------------------------------------------
    # TEACHER-FORCED PREDICTION
    # --------------------------------------------------------

    def predict_teacher_forced(
        self,
        dataset: TrajectoryAccuracyDataset,
    ) -> np.ndarray:
        predictions = []

        horizon = dataset.actions.shape[1]

        for sample_index, initial_state in enumerate(
            dataset.initial_states
        ):
            sample_predictions = []

            for step in range(horizon):
                if step == 0:
                    input_state = initial_state
                else:
                    input_state = (
                        dataset.target_states[
                            sample_index,
                            step - 1,
                        ]
                    )

                action = (
                    dataset.actions[
                        sample_index,
                        step,
                    ]
                )

                predicted, _ = self._predict(
                    input_state,
                    action,
                )

                sample_predictions.append(
                    predicted
                )

            predictions.append(
                sample_predictions
            )

        return np.asarray(
            predictions,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # HORIZON-WISE METRICS
    # --------------------------------------------------------

    def evaluate_horizons(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
    ) -> list[HorizonAccuracyMetrics]:
        results = []

        position_indices = (
            self.config.position_indices
        )

        velocity_indices = (
            self.config.velocity_indices
        )

        horizon = predictions.shape[1]

        for step in range(horizon):
            predicted = predictions[:, step, :]
            actual = targets[:, step, :]

            state_rmse = calculate_rmse(
                predicted,
                actual,
            )

            state_mae = calculate_mae(
                predicted,
                actual,
            )

            predicted_positions = predicted[
                :,
                position_indices,
            ]

            target_positions = actual[
                :,
                position_indices,
            ]

            position_errors = np.linalg.norm(
                predicted_positions - target_positions,
                axis=1,
            )

            position_ade = float(
                np.mean(position_errors)
            )

            position_rmse = calculate_rmse(
                predicted_positions,
                target_positions,
            )

            velocity_rmse = None

            if velocity_indices is not None:
                velocity_rmse = calculate_rmse(
                    predicted[:, velocity_indices],
                    actual[:, velocity_indices],
                )

            step_uncertainties = (
                uncertainties[:, step]
            )

            valid_uncertainties = np.isfinite(
                step_uncertainties
            )

            mean_uncertainty = (
                float(
                    np.mean(
                        step_uncertainties[
                            valid_uncertainties
                        ]
                    )
                )
                if np.any(valid_uncertainties)
                else None
            )

            results.append(
                HorizonAccuracyMetrics(
                    horizon_step=step + 1,
                    state_rmse=state_rmse,
                    state_mae=state_mae,
                    position_ade=position_ade,
                    position_rmse=position_rmse,
                    velocity_rmse=velocity_rmse,
                    accuracy_score=error_to_score(
                        state_rmse,
                        self.config.state_error_scale,
                    ),
                    mean_uncertainty=mean_uncertainty,
                )
            )

        return results

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: TrajectoryAccuracyDataset,
    ) -> TrajectoryAccuracyResult:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        predictions, uncertainties = (
            self.predict_rollout(
                dataset
            )
        )

        teacher_predictions = (
            self.predict_teacher_forced(
                dataset
            )
        )

        targets = dataset.target_states

        # ----------------------------------------------------
        # State trajectory accuracy
        # ----------------------------------------------------

        trajectory_rmse = calculate_rmse(
            predictions,
            targets,
        )

        trajectory_mae = calculate_mae(
            predictions,
            targets,
        )

        trajectory_score = error_to_score(
            trajectory_rmse,
            self.config.state_error_scale,
        )

        # ----------------------------------------------------
        # Position accuracy
        # ----------------------------------------------------

        position_indices = (
            self.config.position_indices
        )

        predicted_positions = predictions[
            :,
            :,
            position_indices,
        ]

        target_positions = targets[
            :,
            :,
            position_indices,
        ]

        position_errors = np.linalg.norm(
            predicted_positions - target_positions,
            axis=2,
        )

        average_displacement_error = float(
            np.mean(position_errors)
        )

        final_displacement_error = float(
            np.mean(
                position_errors[:, -1]
            )
        )

        position_score = error_to_score(
            average_displacement_error,
            self.config.position_error_scale,
        )

        final_position_score = error_to_score(
            final_displacement_error,
            self.config.position_error_scale,
        )

        # ----------------------------------------------------
        # Velocity accuracy
        # ----------------------------------------------------

        velocity_rmse = None
        velocity_score = None

        if self.config.velocity_indices is not None:
            indices = self.config.velocity_indices

            velocity_rmse = calculate_rmse(
                predictions[:, :, indices],
                targets[:, :, indices],
            )

            velocity_score = error_to_score(
                velocity_rmse,
                self.config.velocity_error_scale,
            )

        # ----------------------------------------------------
        # Drift
        # ----------------------------------------------------

        initial_position_error = float(
            np.mean(
                position_errors[:, 0]
            )
        )

        final_position_error = float(
            np.mean(
                position_errors[:, -1]
            )
        )

        drift_error = max(
            0.0,
            final_position_error
            - initial_position_error,
        )

        drift_score = error_to_score(
            drift_error,
            self.config.drift_error_scale,
        )

        # ----------------------------------------------------
        # Teacher-forced accuracy
        # ----------------------------------------------------

        teacher_rmse = calculate_rmse(
            teacher_predictions,
            targets,
        )

        teacher_score = error_to_score(
            teacher_rmse,
            self.config.state_error_scale,
        )

        # ----------------------------------------------------
        # Rollout consistency
        # ----------------------------------------------------

        consistency_rmse = calculate_rmse(
            predictions,
            teacher_predictions,
        )

        consistency_score = error_to_score(
            consistency_rmse,
            self.config.consistency_error_scale,
        )

        # ----------------------------------------------------
        # Uncertainty diagnostics
        # ----------------------------------------------------

        state_errors = np.linalg.norm(
            predictions - targets,
            axis=2,
        )

        valid_uncertainties = np.isfinite(
            uncertainties
        )

        uncertainty_correlation = None
        uncertainty_acceptance_rate = None

        if np.any(valid_uncertainties):
            uncertainty_correlation = safe_correlation(
                uncertainties[valid_uncertainties],
                state_errors[valid_uncertainties],
                self.config.epsilon,
            )

            uncertainty_acceptance_rate = float(
                np.mean(
                    uncertainties[valid_uncertainties]
                    <= self.config.uncertainty_threshold
                )
            )

        # ----------------------------------------------------
        # Horizon-wise metrics
        # ----------------------------------------------------

        horizon_metrics = (
            self.evaluate_horizons(
                predictions,
                targets,
                uncertainties,
            )
        )

        # ----------------------------------------------------
        # Overall score
        # ----------------------------------------------------

        scores: dict[str, float | None] = {
            "trajectory_accuracy": trajectory_score,
            "position_accuracy": position_score,
            "velocity_accuracy": velocity_score,
            "final_position_accuracy": (
                final_position_score
            ),
            "drift_score": drift_score,
            "teacher_forced_accuracy": teacher_score,
            "rollout_consistency": consistency_score,
        }

        active_weights = {
            name: weight
            for name, weight in self.config.weights.items()
            if scores[name] is not None
        }

        total_weight = sum(
            active_weights.values()
        )

        if total_weight <= 0:
            raise TrajectoryValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = TrajectoryAccuracyMetrics(
            trajectory_accuracy=trajectory_score,
            position_accuracy=position_score,
            velocity_accuracy=velocity_score,
            final_position_accuracy=(
                final_position_score
            ),
            drift_score=drift_score,
            teacher_forced_accuracy=teacher_score,
            rollout_consistency=consistency_score,
            uncertainty_error_correlation=(
                uncertainty_correlation
            ),
            uncertainty_acceptance_rate=(
                uncertainty_acceptance_rate
            ),
            overall_score=float(overall_score),
        )

        raw_metrics = {
            "trajectory_rmse": trajectory_rmse,
            "trajectory_mae": trajectory_mae,
            "average_displacement_error": (
                average_displacement_error
            ),
            "final_displacement_error": (
                final_displacement_error
            ),
            "initial_position_error": (
                initial_position_error
            ),
            "drift_error": drift_error,
            "teacher_forced_rmse": teacher_rmse,
            "rollout_teacher_disagreement_rmse": (
                consistency_rmse
            ),
        }

        if velocity_rmse is not None:
            raw_metrics[
                "velocity_rmse"
            ] = velocity_rmse

        result = TrajectoryAccuracyResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            sample_count=len(
                dataset.initial_states
            ),
            horizon=dataset.actions.shape[1],
            metrics=metrics,
            horizon_metrics=horizon_metrics,
            passed=(
                overall_score
                >= self.config.pass_threshold
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            raw_metrics=raw_metrics,
            details={
                "weights": active_weights,
                "position_indices": list(
                    self.config.position_indices
                ),
                "velocity_indices": (
                    list(self.config.velocity_indices)
                    if self.config.velocity_indices is not None
                    else None
                ),
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "pass_threshold": (
                    self.config.pass_threshold
                ),
                "interpretation": (
                    "Trajectory accuracy against "
                    "reference state sequences"
                ),
                "limitations": (
                    "State and position error scores "
                    "depend on units and normalization. "
                    "Prediction accuracy does not "
                    "establish collision safety."
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Trajectory evaluation: "
            "model=%s ADE=%.4f FDE=%.4f score=%.4f",
            self.model.name,
            average_displacement_error,
            final_displacement_error,
            overall_score,
        )

        return result

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> TrajectoryAccuracyResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise TrajectoryAccuracyError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[TrajectoryAccuracyResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[TrajectoryAccuracyResult],
    ) -> list[TrajectoryAccuracyResult]:
        return sorted(
            results,
            key=lambda result: (
                result.metrics.overall_score
            ),
            reverse=True,
        )

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_json(
        self,
        evaluation_id: str,
        output_path: str | Path,
    ) -> Path:
        result = self.get_result(
            evaluation_id
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(result),
                file,
                indent=2,
                allow_nan=False,
            )

        return path

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    @staticmethod
    def export_csv(
        results: list[TrajectoryAccuracyResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise TrajectoryValidationError(
                "No results to export"
            )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        for result in results:
            row = {
                "evaluation_id": result.evaluation_id,
                "model_name": result.model_name,
                "sample_count": result.sample_count,
                "horizon": result.horizon,
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

        fieldnames = list(
            dict.fromkeys(
                key
                for row in rows
                for key in row
            )
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_trajectory_accuracy_evaluator(
    model: WorldModelProtocol,
    *,
    config: TrajectoryAccuracyConfig | None = None,
) -> TrajectoryAccuracyEvaluator:
    return TrajectoryAccuracyEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_trajectory_dataset(
    *,
    sample_count: int = 128,
    horizon: int = 12,
    seed: int = 42,
    dynamics_scale: float = 0.5,
) -> TrajectoryAccuracyDataset:
    """
    Synthetic 2D state:

        [x, y, vx, vy]

    Synthetic transition:

        next_state = state + 0.5 * action

    This is a numerical demonstration rather
    than a physical motion model.
    """

    if sample_count < 1 or horizon < 1:
        raise ValueError(
            "Sample count and horizon must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    initial_states = rng.normal(
        0.0,
        1.0,
        size=(sample_count, 4),
    )

    actions = rng.uniform(
        -1.0,
        1.0,
        size=(
            sample_count,
            horizon,
            4,
        ),
    )

    targets = np.empty(
        (
            sample_count,
            horizon,
            4,
        ),
        dtype=np.float64,
    )

    current = initial_states.copy()

    for step in range(horizon):
        current = (
            current
            + dynamics_scale * actions[:, step, :]
        )

        targets[:, step, :] = current

    return TrajectoryAccuracyDataset(
        initial_states=initial_states,
        actions=actions,
        target_states=targets,
        metadata={
            "source": "synthetic_linear_dynamics",
            "seed": seed,
            "dynamics_scale": dynamics_scale,
        },
    )


# ============================================================
# DEMONSTRATION WORLD MODEL
# ============================================================

class DemoWorldModel:
    name = "demo_trajectory_world_model"

    state_dim = 4
    action_dim = 4

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            next_state=state + 0.5 * action,
            uncertainty=0.05,
            terminal=False,
            reward=0.0,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    dataset = generate_synthetic_trajectory_dataset(
        sample_count=128,
        horizon=12,
        seed=42,
    )

    model = DemoWorldModel()

    evaluator = (
        create_trajectory_accuracy_evaluator(
            model
        )
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — TRAJECTORY ACCURACY")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/trajectory_accuracy.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/trajectory_accuracy.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
