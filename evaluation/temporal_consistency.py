
"""
NeuroWorld AI
evaluation/temporal_consistency.py

Temporal Consistency Evaluation for Neural World Models.

Features:
- One-step temporal prediction accuracy
- Autoregressive rollout accuracy
- Teacher-forced transition accuracy
- Temporal smoothness
- Prediction drift analysis
- Rollout vs teacher-forced consistency
- Prediction repeatability
- Horizon-wise error metrics
- Uncertainty-error correlation
- Model comparison
- JSON / CSV report export
- Reproducible synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.temporal_consistency

World-model interface:
    model.name
    model.state_dim
    model.action_dim
    model.predict_transition(state, action)

Transition result:
    transition.next_state
    transition.uncertainty (optional)

Dataset:
    initial_states: (N, state_dim)
    actions: (N, horizon, action_dim)
    target_states: (N, horizon, state_dim)

Important:
    Reference future states must come from a
    trusted simulator or independently collected
    observations.

    Temporal smoothness alone does not imply
    physical correctness. Rapid legitimate motion
    must not be penalized as inconsistency.
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
    "neuroworld.evaluation.temporal_consistency"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TemporalConsistencyError(Exception):
    pass


class TemporalValidationError(TemporalConsistencyError):
    pass


class TemporalModelError(TemporalConsistencyError):
    pass


# ============================================================
# WORLD MODEL PROTOCOL
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
class TemporalConsistencyConfig:
    max_samples: int = 10000
    max_horizon: int = 256

    prediction_error_scale: float = 1.0
    smoothness_error_scale: float = 1.0
    drift_error_scale: float = 1.0
    consistency_error_scale: float = 1.0

    repeatability_trials: int = 3

    uncertainty_threshold: float = 1.0

    pass_threshold: float = 0.70

    random_seed: int = 42
    epsilon: float = 1e-8

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "one_step_accuracy": 0.15,
            "rollout_accuracy": 0.25,
            "teacher_forced_accuracy": 0.10,
            "temporal_smoothness": 0.10,
            "drift_score": 0.15,
            "rollout_consistency": 0.15,
            "repeatability": 0.10,
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

        if self.repeatability_trials < 2:
            raise ValueError(
                "repeatability_trials must be >= 2"
            )

        for name in (
            "prediction_error_scale",
            "smoothness_error_scale",
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

        allowed = {
            "one_step_accuracy",
            "rollout_accuracy",
            "teacher_forced_accuracy",
            "temporal_smoothness",
            "drift_score",
            "rollout_consistency",
            "repeatability",
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
                "Invalid temporal metric weights"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class TemporalConsistencyDataset:
    """
    initial_states:
        (N, state_dim)

    actions:
        (N, horizon, action_dim)

    target_states:
        (N, horizon, state_dim)

    target_states[:, 0] is the reference state
    after applying actions[:, 0].
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
            raise TemporalValidationError(
                "Dataset must contain numeric arrays"
            ) from exc

        if states.ndim != 2:
            raise TemporalValidationError(
                "initial_states must be 2D"
            )

        if actions.ndim != 3:
            raise TemporalValidationError(
                "actions must be 3D"
            )

        if targets.ndim != 3:
            raise TemporalValidationError(
                "target_states must be 3D"
            )

        n = states.shape[0]
        horizon = actions.shape[1]

        if not 1 <= n <= max_samples:
            raise TemporalValidationError(
                "Invalid sample count"
            )

        if not 1 <= horizon <= max_horizon:
            raise TemporalValidationError(
                "Invalid prediction horizon"
            )

        if states.shape != (n, state_dim):
            raise TemporalValidationError(
                "Invalid initial state shape"
            )

        if actions.shape != (
            n,
            horizon,
            action_dim,
        ):
            raise TemporalValidationError(
                "Invalid action shape"
            )

        if targets.shape != (
            n,
            horizon,
            state_dim,
        ):
            raise TemporalValidationError(
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
            raise TemporalValidationError(
                "Dataset contains non-finite values"
            )

        self.initial_states = states.copy()
        self.actions = actions.copy()
        self.target_states = targets.copy()


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class TemporalHorizonMetrics:
    horizon_step: int

    rmse: float
    mae: float

    accuracy_score: float

    mean_displacement_error: float
    mean_uncertainty: float | None


@dataclass
class TemporalConsistencyMetrics:
    one_step_accuracy: float
    rollout_accuracy: float
    teacher_forced_accuracy: float

    temporal_smoothness: float | None
    drift_score: float

    rollout_consistency: float
    repeatability: float

    uncertainty_error_correlation: float | None
    uncertainty_acceptance_rate: float | None

    overall_score: float


@dataclass
class TemporalConsistencyResult:
    evaluation_id: str
    model_name: str

    sample_count: int
    horizon: int

    metrics: TemporalConsistencyMetrics

    horizon_metrics: list[TemporalHorizonMetrics]

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
        raise TemporalValidationError(
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
        raise TemporalValidationError(
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
        raise TemporalValidationError(
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

class TemporalConsistencyEvaluator:
    """
    Evaluates temporal consistency of neural
    world-model state predictions.

    Rollout:
        Each predicted state becomes the next
        model input.

    Teacher forcing:
        Each prediction starts from the
        previous ground-truth state.

    The distinction helps reveal error
    accumulation across time.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: TemporalConsistencyConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or TemporalConsistencyConfig()
        )

        self._results: dict[
            str,
            TemporalConsistencyResult,
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
            raise TemporalModelError(
                f"World-model prediction failed: {exc}"
            ) from exc

        if predicted.shape != (
            self.model.state_dim,
        ):
            raise TemporalModelError(
                "Invalid predicted state dimensions"
            )

        if not np.all(np.isfinite(predicted)):
            raise TemporalModelError(
                "Predicted state contains non-finite values"
            )

        if uncertainty is not None and (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise TemporalModelError(
                "Invalid uncertainty value"
            )

        return predicted.copy(), uncertainty

    # --------------------------------------------------------
    # AUTOREGRESSIVE ROLLOUT
    # --------------------------------------------------------

    def predict_rollout(
        self,
        dataset: TemporalConsistencyDataset,
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
        dataset: TemporalConsistencyDataset,
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
    # TEMPORAL SMOOTHNESS
    # --------------------------------------------------------

    def evaluate_temporal_smoothness(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """
        Compares predicted and reference
        changes in velocity-like state increments.

        Uses second finite differences.

        A high score means predicted temporal
        changes resemble reference changes.

        This does not require motion to be slow.
        """

        if predictions.shape[1] < 3:
            return None, None

        predicted_second_difference = (
            predictions[:, 2:, :]
            - 2.0 * predictions[:, 1:-1, :]
            + predictions[:, :-2, :]
        )

        target_second_difference = (
            targets[:, 2:, :]
            - 2.0 * targets[:, 1:-1, :]
            + targets[:, :-2, :]
        )

        error = calculate_rmse(
            predicted_second_difference,
            target_second_difference,
        )

        return (
            error_to_score(
                error,
                self.config.smoothness_error_scale,
            ),
            error,
        )

    # --------------------------------------------------------
    # ROLLOUT DRIFT
    # --------------------------------------------------------

    def evaluate_drift(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, float]:
        errors = np.linalg.norm(
            predictions - targets,
            axis=2,
        )

        first_error = float(
            np.mean(
                errors[:, 0]
            )
        )

        last_error = float(
            np.mean(
                errors[:, -1]
            )
        )

        drift = max(
            0.0,
            last_error - first_error,
        )

        score = error_to_score(
            drift,
            self.config.drift_error_scale,
        )

        return score, drift

    # --------------------------------------------------------
    # ROLLOUT CONSISTENCY
    # --------------------------------------------------------

    def evaluate_rollout_consistency(
        self,
        rollout_predictions: np.ndarray,
        teacher_predictions: np.ndarray,
    ) -> tuple[float, float]:
        disagreement = calculate_rmse(
            rollout_predictions,
            teacher_predictions,
        )

        score = error_to_score(
            disagreement,
            self.config.consistency_error_scale,
        )

        return score, disagreement

    # --------------------------------------------------------
    # REPEATABILITY
    # --------------------------------------------------------

    def evaluate_repeatability(
        self,
        dataset: TemporalConsistencyDataset,
        reference_rollout: np.ndarray,
    ) -> tuple[float, float]:
        """
        Measures repeatability of predictions
        for identical inputs.

        Stochastic models may legitimately produce
        different samples. For such models, use a
        fixed sampling seed or evaluate predicted
        distributions separately.
        """

        errors = []

        for _ in range(
            self.config.repeatability_trials - 1
        ):
            repeated_predictions, _ = (
                self.predict_rollout(
                    dataset
                )
            )

            errors.append(
                calculate_rmse(
                    repeated_predictions,
                    reference_rollout,
                )
            )

        mean_error = float(
            np.mean(errors)
        )

        score = error_to_score(
            mean_error,
            self.config.consistency_error_scale,
        )

        return score, mean_error

    # --------------------------------------------------------
    # HORIZON METRICS
    # --------------------------------------------------------

    def evaluate_horizons(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
    ) -> list[TemporalHorizonMetrics]:
        results = []

        horizon = predictions.shape[1]

        for step in range(horizon):
            predicted = predictions[:, step, :]
            actual = targets[:, step, :]

            rmse = calculate_rmse(
                predicted,
                actual,
            )

            mae = calculate_mae(
                predicted,
                actual,
            )

            displacement = float(
                np.mean(
                    np.linalg.norm(
                        predicted - actual,
                        axis=1,
                    )
                )
            )

            uncertainty_values = (
                uncertainties[:, step]
            )

            valid = np.isfinite(
                uncertainty_values
            )

            mean_uncertainty = (
                float(
                    np.mean(
                        uncertainty_values[valid]
                    )
                )
                if np.any(valid)
                else None
            )

            results.append(
                TemporalHorizonMetrics(
                    horizon_step=step + 1,
                    rmse=rmse,
                    mae=mae,
                    accuracy_score=error_to_score(
                        rmse,
                        self.config.prediction_error_scale,
                    ),
                    mean_displacement_error=displacement,
                    mean_uncertainty=mean_uncertainty,
                )
            )

        return results

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: TemporalConsistencyDataset,
    ) -> TemporalConsistencyResult:
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

        horizon_metrics = (
            self.evaluate_horizons(
                predictions,
                targets,
                uncertainties,
            )
        )

        # ----------------------------------------------------
        # Prediction errors
        # ----------------------------------------------------

        one_step_rmse = calculate_rmse(
            predictions[:, 0, :],
            targets[:, 0, :],
        )

        rollout_rmse = calculate_rmse(
            predictions,
            targets,
        )

        teacher_rmse = calculate_rmse(
            teacher_predictions,
            targets,
        )

        one_step_score = error_to_score(
            one_step_rmse,
            self.config.prediction_error_scale,
        )

        rollout_score = error_to_score(
            rollout_rmse,
            self.config.prediction_error_scale,
        )

        teacher_score = error_to_score(
            teacher_rmse,
            self.config.prediction_error_scale,
        )

        # ----------------------------------------------------
        # Temporal smoothness
        # ----------------------------------------------------

        smoothness_score, smoothness_error = (
            self.evaluate_temporal_smoothness(
                predictions,
                targets,
            )
        )

        # ----------------------------------------------------
        # Prediction drift
        # ----------------------------------------------------

        drift_score, drift_error = (
            self.evaluate_drift(
                predictions,
                targets,
            )
        )

        # ----------------------------------------------------
        # Rollout vs teacher forcing
        # ----------------------------------------------------

        consistency_score, consistency_error = (
            self.evaluate_rollout_consistency(
                predictions,
                teacher_predictions,
            )
        )

        # ----------------------------------------------------
        # Repeatability
        # ----------------------------------------------------

        repeatability_score, repeatability_error = (
            self.evaluate_repeatability(
                dataset,
                predictions,
            )
        )

        # ----------------------------------------------------
        # Uncertainty diagnostics
        # ----------------------------------------------------

        per_step_errors = np.linalg.norm(
            predictions - targets,
            axis=2,
        )

        valid_uncertainty = np.isfinite(
            uncertainties
        )

        uncertainty_correlation = None
        uncertainty_acceptance_rate = None

        if np.any(valid_uncertainty):
            uncertainty_correlation = safe_correlation(
                uncertainties[valid_uncertainty],
                per_step_errors[valid_uncertainty],
                self.config.epsilon,
            )

            uncertainty_acceptance_rate = float(
                np.mean(
                    uncertainties[valid_uncertainty]
                    <= self.config.uncertainty_threshold
                )
            )

        # ----------------------------------------------------
        # Weighted score
        # ----------------------------------------------------

        scores: dict[str, float | None] = {
            "one_step_accuracy": one_step_score,
            "rollout_accuracy": rollout_score,
            "teacher_forced_accuracy": teacher_score,
            "temporal_smoothness": smoothness_score,
            "drift_score": drift_score,
            "rollout_consistency": consistency_score,
            "repeatability": repeatability_score,
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
            raise TemporalValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = TemporalConsistencyMetrics(
            one_step_accuracy=one_step_score,
            rollout_accuracy=rollout_score,
            teacher_forced_accuracy=teacher_score,
            temporal_smoothness=smoothness_score,
            drift_score=drift_score,
            rollout_consistency=consistency_score,
            repeatability=repeatability_score,
            uncertainty_error_correlation=(
                uncertainty_correlation
            ),
            uncertainty_acceptance_rate=(
                uncertainty_acceptance_rate
            ),
            overall_score=float(overall_score),
        )

        raw_metrics = {
            "one_step_rmse": one_step_rmse,
            "rollout_rmse": rollout_rmse,
            "teacher_forced_rmse": teacher_rmse,
            "drift_error": drift_error,
            "rollout_teacher_disagreement": (
                consistency_error
            ),
            "repeatability_rmse": (
                repeatability_error
            ),
        }

        if smoothness_error is not None:
            raw_metrics[
                "temporal_second_difference_rmse"
            ] = smoothness_error

        result = TemporalConsistencyResult(
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
                "pass_threshold": (
                    self.config.pass_threshold
                ),
                "repeatability_trials": (
                    self.config.repeatability_trials
                ),
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "interpretation": (
                    "Temporal prediction agreement "
                    "with supplied reference trajectories"
                ),
                "limitations": (
                    "Repeatability is not a distributional "
                    "calibration metric. Rollout consistency "
                    "is not a substitute for prediction accuracy."
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Temporal consistency evaluation: "
            "model=%s score=%.4f passed=%s",
            self.model.name,
            overall_score,
            result.passed,
        )

        return result

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> TemporalConsistencyResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise TemporalConsistencyError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[TemporalConsistencyResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[TemporalConsistencyResult],
    ) -> list[TemporalConsistencyResult]:
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
        results: list[TemporalConsistencyResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise TemporalValidationError(
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

def create_temporal_consistency_evaluator(
    model: WorldModelProtocol,
    *,
    config: TemporalConsistencyConfig | None = None,
) -> TemporalConsistencyEvaluator:
    return TemporalConsistencyEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_temporal_dataset(
    *,
    state_dim: int = 4,
    action_dim: int = 4,
    sample_count: int = 128,
    horizon: int = 12,
    seed: int = 42,
    dynamics_scale: float = 0.5,
) -> TemporalConsistencyDataset:
    """
    Synthetic linear dynamics:

        next_state = state + B @ action

    The reference trajectory is generated
    independently of the evaluator.
    """

    if min(
        state_dim,
        action_dim,
        sample_count,
        horizon,
    ) < 1:
        raise TemporalValidationError(
            "Dimensions and counts must be positive"
        )

    rng = np.random.default_rng(seed)

    transition_matrix = np.zeros(
        (state_dim, action_dim),
        dtype=np.float64,
    )

    for index in range(
        min(state_dim, action_dim)
    ):
        transition_matrix[
            index,
            index,
        ] = dynamics_scale

    initial_states = rng.normal(
        0.0,
        1.0,
        size=(sample_count, state_dim),
    )

    actions = rng.uniform(
        -1.0,
        1.0,
        size=(
            sample_count,
            horizon,
            action_dim,
        ),
    )

    targets = np.empty(
        (
            sample_count,
            horizon,
            state_dim,
        ),
        dtype=np.float64,
    )

    current = initial_states.copy()

    for step in range(horizon):
        current = (
            current
            + actions[:, step, :]
            @ transition_matrix.T
        )

        targets[:, step, :] = current

    return TemporalConsistencyDataset(
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
    name = "demo_temporal_world_model"

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
    dataset = generate_synthetic_temporal_dataset(
        state_dim=4,
        action_dim=4,
        sample_count=128,
        horizon=12,
        seed=42,
    )

    model = DemoWorldModel()

    evaluator = (
        create_temporal_consistency_evaluator(
            model
        )
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — TEMPORAL CONSISTENCY")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/temporal_consistency.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/temporal_consistency.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
