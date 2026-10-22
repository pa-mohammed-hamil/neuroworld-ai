
"""
NeuroWorld AI
evaluation/world_consistency.py

World Consistency Evaluation for Neural World Models.

Features:
- One-step state prediction accuracy
- Multi-step world-state consistency
- Action-conditioned transition accuracy
- Spatial relationship preservation
- Object permanence / identity consistency
- Temporal drift detection
- Teacher-forced vs rollout consistency
- Uncertainty-error correlation
- Horizon-wise diagnostics
- Weighted consistency scoring
- JSON and CSV report export
- Synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.world_consistency

Model interface:
    model.name
    model.state_dim
    model.action_dim
    model.predict_transition(state, action)

Transition result:
    transition.next_state
    transition.uncertainty (optional)

Dataset:
    initial_states: (N, S)
    actions: (N, H, A)
    target_states: (N, H, S)

Optional object positions:
    predicted_object_positions: (N, H, O, D)
    target_object_positions: (N, H, O, D)
    object_visibility: (N, H, O)
    object_existence: (N, H, O)

IMPORTANT:
    Ground-truth trajectories must be independent
    of the evaluated world model.

    Object slots must correspond to consistent
    object identities across frames.

    Scores are diagnostic and do not establish
    physical correctness or deployment safety.
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
    "neuroworld.evaluation.world_consistency"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class WorldConsistencyError(Exception):
    pass


class WorldConsistencyValidationError(
    WorldConsistencyError
):
    pass


class WorldConsistencyModelError(
    WorldConsistencyError
):
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
class WorldConsistencyConfig:
    max_samples: int = 10000
    max_horizon: int = 256
    max_objects: int = 512

    state_error_scale: float = 1.0
    transition_error_scale: float = 1.0
    spatial_error_scale: float = 1.0
    object_error_scale: float = 1.0
    drift_error_scale: float = 1.0
    consistency_error_scale: float = 1.0

    object_position_tolerance: float = 0.5
    uncertainty_threshold: float = 1.0

    pass_threshold: float = 0.70
    epsilon: float = 1e-8

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "state_consistency": 0.20,
            "transition_consistency": 0.15,
            "rollout_consistency": 0.15,
            "temporal_consistency": 0.10,
            "spatial_consistency": 0.15,
            "object_permanence": 0.15,
            "drift_score": 0.10,
        }
    )

    def __post_init__(self):
        if min(
            self.max_samples,
            self.max_horizon,
            self.max_objects,
        ) < 1:
            raise ValueError(
                "Evaluation limits must be positive"
            )

        for name in (
            "state_error_scale",
            "transition_error_scale",
            "spatial_error_scale",
            "object_error_scale",
            "drift_error_scale",
            "consistency_error_scale",
            "object_position_tolerance",
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
            "state_consistency",
            "transition_consistency",
            "rollout_consistency",
            "temporal_consistency",
            "spatial_consistency",
            "object_permanence",
            "drift_score",
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
                "Invalid consistency metric weights"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class WorldConsistencyDataset:
    initial_states: np.ndarray
    actions: np.ndarray
    target_states: np.ndarray

    predicted_object_positions: np.ndarray | None = None
    target_object_positions: np.ndarray | None = None

    object_visibility: np.ndarray | None = None
    object_existence: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        state_dim: int,
        action_dim: int,
        config: WorldConsistencyConfig,
    ) -> None:
        try:
            initial = np.asarray(
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
            raise WorldConsistencyValidationError(
                "State and action arrays must be numeric"
            ) from exc

        if initial.ndim != 2:
            raise WorldConsistencyValidationError(
                "initial_states must be 2D"
            )

        if actions.ndim != 3:
            raise WorldConsistencyValidationError(
                "actions must be 3D"
            )

        if targets.ndim != 3:
            raise WorldConsistencyValidationError(
                "target_states must be 3D"
            )

        n = initial.shape[0]
        horizon = actions.shape[1]

        if not 1 <= n <= config.max_samples:
            raise WorldConsistencyValidationError(
                "Invalid sample count"
            )

        if not 1 <= horizon <= config.max_horizon:
            raise WorldConsistencyValidationError(
                "Invalid horizon"
            )

        if initial.shape != (n, state_dim):
            raise WorldConsistencyValidationError(
                "Invalid initial state shape"
            )

        if actions.shape != (
            n,
            horizon,
            action_dim,
        ):
            raise WorldConsistencyValidationError(
                "Invalid action shape"
            )

        if targets.shape != (
            n,
            horizon,
            state_dim,
        ):
            raise WorldConsistencyValidationError(
                "Invalid target state shape"
            )

        if not all(
            np.all(np.isfinite(array))
            for array in (
                initial,
                actions,
                targets,
            )
        ):
            raise WorldConsistencyValidationError(
                "Non-finite state or action values"
            )

        self.initial_states = initial.copy()
        self.actions = actions.copy()
        self.target_states = targets.copy()

        object_arrays = (
            self.predicted_object_positions,
            self.target_object_positions,
        )

        if any(
            value is not None
            for value in object_arrays
        ):
            if not all(
                value is not None
                for value in object_arrays
            ):
                raise WorldConsistencyValidationError(
                    "Both object-position arrays are required"
                )

            predicted_objects = np.asarray(
                self.predicted_object_positions,
                dtype=np.float64,
            )

            target_objects = np.asarray(
                self.target_object_positions,
                dtype=np.float64,
            )

            if (
                predicted_objects.ndim != 4
                or predicted_objects.shape
                != target_objects.shape
            ):
                raise WorldConsistencyValidationError(
                    "Invalid object-position arrays"
                )

            if (
                predicted_objects.shape[0] != n
                or predicted_objects.shape[1] != horizon
                or not 1 <= predicted_objects.shape[2]
                <= config.max_objects
                or predicted_objects.shape[3] not in (2, 3)
            ):
                raise WorldConsistencyValidationError(
                    "Object positions must have shape "
                    "(N, H, O, 2|3)"
                )

            if not (
                np.all(np.isfinite(predicted_objects))
                and np.all(np.isfinite(target_objects))
            ):
                raise WorldConsistencyValidationError(
                    "Object positions must be finite"
                )

            self.predicted_object_positions = (
                predicted_objects.copy()
            )

            self.target_object_positions = (
                target_objects.copy()
            )

            object_shape = (
                predicted_objects.shape[:3]
            )

            for name in (
                "object_visibility",
                "object_existence",
            ):
                value = getattr(self, name)

                if value is not None:
                    mask = np.asarray(value)

                    if (
                        mask.shape != object_shape
                        or not np.issubdtype(
                            mask.dtype,
                            np.bool_,
                        )
                    ):
                        raise WorldConsistencyValidationError(
                            f"{name} must be boolean "
                            f"with shape {object_shape}"
                        )

                    setattr(
                        self,
                        name,
                        mask.copy(),
                    )

        elif (
            self.object_visibility is not None
            or self.object_existence is not None
        ):
            raise WorldConsistencyValidationError(
                "Object masks require object-position arrays"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class WorldHorizonMetrics:
    horizon_step: int

    state_rmse: float
    state_mae: float

    state_accuracy: float

    mean_uncertainty: float | None


@dataclass
class WorldConsistencyMetrics:
    state_consistency: float

    transition_consistency: float

    rollout_consistency: float

    temporal_consistency: float | None

    spatial_consistency: float | None

    object_permanence: float | None

    drift_score: float

    uncertainty_error_correlation: float | None

    uncertainty_acceptance_rate: float | None

    overall_score: float


@dataclass
class WorldConsistencyResult:
    evaluation_id: str
    model_name: str

    sample_count: int
    horizon: int

    metrics: WorldConsistencyMetrics

    horizon_metrics: list[WorldHorizonMetrics]

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
        raise WorldConsistencyValidationError(
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
        raise WorldConsistencyValidationError(
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
        raise WorldConsistencyValidationError(
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

class WorldConsistencyEvaluator:
    """
    Evaluate world-model predictions across time.

    Object metrics are optional and require
    externally supplied object-position predictions.

    The core state metrics are computed from
    predict_transition() outputs.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: WorldConsistencyConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or WorldConsistencyConfig()
        )

        self._results: dict[
            str,
            WorldConsistencyResult,
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
            raise WorldConsistencyModelError(
                f"World-model prediction failed: {exc}"
            ) from exc

        if predicted.shape != (
            self.model.state_dim,
        ):
            raise WorldConsistencyModelError(
                "Invalid predicted state dimension"
            )

        if not np.all(np.isfinite(predicted)):
            raise WorldConsistencyModelError(
                "Predicted state contains non-finite values"
            )

        if uncertainty is not None and (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise WorldConsistencyModelError(
                "Invalid uncertainty value"
            )

        return predicted.copy(), uncertainty

    # --------------------------------------------------------
    # AUTOREGRESSIVE ROLLOUT
    # --------------------------------------------------------

    def predict_rollout(
        self,
        dataset: WorldConsistencyDataset,
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
    # TEACHER-FORCED TRANSITIONS
    # --------------------------------------------------------

    def predict_teacher_forced(
        self,
        dataset: WorldConsistencyDataset,
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
    # TEMPORAL CONSISTENCY
    # --------------------------------------------------------

    def evaluate_temporal_consistency(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float | None, float | None]:
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

        score = error_to_score(
            error,
            self.config.consistency_error_scale,
        )

        return score, error

    # --------------------------------------------------------
    # SPATIAL CONSISTENCY
    # --------------------------------------------------------

    def evaluate_spatial_consistency(
        self,
        dataset: WorldConsistencyDataset,
    ) -> tuple[float | None, float | None]:
        if dataset.predicted_object_positions is None:
            return None, None

        predicted = (
            dataset.predicted_object_positions
        )

        targets = (
            dataset.target_object_positions
        )

        n, horizon, object_count, _ = (
            predicted.shape
        )

        if object_count < 2:
            return None, None

        existence = (
            dataset.object_existence
            if dataset.object_existence is not None
            else np.ones(
                (n, horizon, object_count),
                dtype=bool,
            )
        )

        errors = []

        # Compare pairwise object distances.
        # Pairwise distances are invariant to
        # global translation and rotation.
        for first in range(object_count):
            for second in range(
                first + 1,
                object_count,
            ):
                valid = (
                    existence[:, :, first]
                    & existence[:, :, second]
                )

                if not np.any(valid):
                    continue

                predicted_distances = np.linalg.norm(
                    predicted[:, :, first, :]
                    - predicted[:, :, second, :],
                    axis=2,
                )

                target_distances = np.linalg.norm(
                    targets[:, :, first, :]
                    - targets[:, :, second, :],
                    axis=2,
                )

                errors.extend(
                    np.abs(
                        predicted_distances[valid]
                        - target_distances[valid]
                    ).tolist()
                )

        if not errors:
            return None, None

        mean_error = float(
            np.mean(errors)
        )

        return (
            error_to_score(
                mean_error,
                self.config.spatial_error_scale,
            ),
            mean_error,
        )

    # --------------------------------------------------------
    # OBJECT PERMANENCE
    # --------------------------------------------------------

    def evaluate_object_permanence(
        self,
        dataset: WorldConsistencyDataset,
    ) -> tuple[float | None, float | None]:
        if dataset.predicted_object_positions is None:
            return None, None

        predicted = (
            dataset.predicted_object_positions
        )

        targets = (
            dataset.target_object_positions
        )

        n, horizon, object_count, _ = (
            predicted.shape
        )

        existence = (
            dataset.object_existence
            if dataset.object_existence is not None
            else np.ones(
                (n, horizon, object_count),
                dtype=bool,
            )
        )

        visibility = (
            dataset.object_visibility
            if dataset.object_visibility is not None
            else np.ones(
                (n, horizon, object_count),
                dtype=bool,
            )
        )

        # Object permanence is specifically
        # evaluated while an object exists but
        # is not visible.
        hidden_mask = (
            existence & ~visibility
        )

        if not np.any(hidden_mask):
            return None, None

        errors = np.linalg.norm(
            predicted - targets,
            axis=3,
        )

        hidden_errors = errors[
            hidden_mask
        ]

        mean_error = float(
            np.mean(hidden_errors)
        )

        correct_fraction = float(
            np.mean(
                hidden_errors
                <= self.config.object_position_tolerance
            )
        )

        # Combine continuous position accuracy
        # with a tolerance-based hit rate.
        score = (
            0.5 * error_to_score(
                mean_error,
                self.config.object_error_scale,
            )
            + 0.5 * correct_fraction
        )

        return float(score), mean_error

    # --------------------------------------------------------
    # DRIFT
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

        return (
            error_to_score(
                drift,
                self.config.drift_error_scale,
            ),
            drift,
        )

    # --------------------------------------------------------
    # HORIZON METRICS
    # --------------------------------------------------------

    def evaluate_horizons(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
    ) -> list[WorldHorizonMetrics]:
        results = []

        for step in range(
            predictions.shape[1]
        ):
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
                WorldHorizonMetrics(
                    horizon_step=step + 1,
                    state_rmse=rmse,
                    state_mae=mae,
                    state_accuracy=error_to_score(
                        rmse,
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
        dataset: WorldConsistencyDataset,
    ) -> WorldConsistencyResult:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            self.config,
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
        # State consistency
        # ----------------------------------------------------

        state_rmse = calculate_rmse(
            predictions,
            targets,
        )

        state_mae = calculate_mae(
            predictions,
            targets,
        )

        state_score = error_to_score(
            state_rmse,
            self.config.state_error_scale,
        )

        # ----------------------------------------------------
        # Action-conditioned transition consistency
        # ----------------------------------------------------

        transition_rmse = calculate_rmse(
            teacher_predictions,
            targets,
        )

        transition_score = error_to_score(
            transition_rmse,
            self.config.transition_error_scale,
        )

        # ----------------------------------------------------
        # Rollout / teacher-forced consistency
        # ----------------------------------------------------

        rollout_disagreement = calculate_rmse(
            predictions,
            teacher_predictions,
        )

        rollout_score = error_to_score(
            rollout_disagreement,
            self.config.consistency_error_scale,
        )

        # ----------------------------------------------------
        # Temporal consistency
        # ----------------------------------------------------

        temporal_score, temporal_error = (
            self.evaluate_temporal_consistency(
                predictions,
                targets,
            )
        )

        # ----------------------------------------------------
        # Spatial consistency
        # ----------------------------------------------------

        spatial_score, spatial_error = (
            self.evaluate_spatial_consistency(
                dataset
            )
        )

        # ----------------------------------------------------
        # Object permanence
        # ----------------------------------------------------

        permanence_score, permanence_error = (
            self.evaluate_object_permanence(
                dataset
            )
        )

        # ----------------------------------------------------
        # Temporal drift
        # ----------------------------------------------------

        drift_score, drift_error = (
            self.evaluate_drift(
                predictions,
                targets,
            )
        )

        # ----------------------------------------------------
        # Uncertainty diagnostics
        # ----------------------------------------------------

        prediction_errors = np.linalg.norm(
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
                prediction_errors[valid_uncertainties],
                self.config.epsilon,
            )

            uncertainty_acceptance_rate = float(
                np.mean(
                    uncertainties[valid_uncertainties]
                    <= self.config.uncertainty_threshold
                )
            )

        # ----------------------------------------------------
        # Horizon metrics
        # ----------------------------------------------------

        horizon_metrics = (
            self.evaluate_horizons(
                predictions,
                targets,
                uncertainties,
            )
        )

        # ----------------------------------------------------
        # Weighted score
        # ----------------------------------------------------

        scores: dict[str, float | None] = {
            "state_consistency": state_score,
            "transition_consistency": transition_score,
            "rollout_consistency": rollout_score,
            "temporal_consistency": temporal_score,
            "spatial_consistency": spatial_score,
            "object_permanence": permanence_score,
            "drift_score": drift_score,
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
            raise WorldConsistencyValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = WorldConsistencyMetrics(
            state_consistency=state_score,
            transition_consistency=transition_score,
            rollout_consistency=rollout_score,
            temporal_consistency=temporal_score,
            spatial_consistency=spatial_score,
            object_permanence=permanence_score,
            drift_score=drift_score,
            uncertainty_error_correlation=(
                uncertainty_correlation
            ),
            uncertainty_acceptance_rate=(
                uncertainty_acceptance_rate
            ),
            overall_score=float(overall_score),
        )

        raw_metrics = {
            "state_rmse": state_rmse,
            "state_mae": state_mae,
            "transition_rmse": transition_rmse,
            "rollout_teacher_disagreement_rmse": (
                rollout_disagreement
            ),
            "drift_error": drift_error,
        }

        optional_errors = {
            "temporal_second_difference_rmse": (
                temporal_error
            ),
            "spatial_pairwise_distance_error": (
                spatial_error
            ),
            "hidden_object_position_error": (
                permanence_error
            ),
        }

        raw_metrics.update(
            {
                name: value
                for name, value in optional_errors.items()
                if value is not None
            }
        )

        result = WorldConsistencyResult(
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
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "object_metrics_available": (
                    dataset.predicted_object_positions
                    is not None
                ),
                "interpretation": (
                    "Agreement of predicted world states "
                    "with supplied reference trajectories "
                    "and optional object-level references"
                ),
                "limitations": (
                    "Object metrics assume aligned identities. "
                    "Spatial distances alone do not verify "
                    "absolute coordinates or object orientation. "
                    "State consistency does not establish "
                    "physical-law compliance."
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "World consistency evaluated: "
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
    ) -> WorldConsistencyResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise WorldConsistencyError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[WorldConsistencyResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[WorldConsistencyResult],
    ) -> list[WorldConsistencyResult]:
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
        results: list[WorldConsistencyResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise WorldConsistencyValidationError(
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

def create_world_consistency_evaluator(
    model: WorldModelProtocol,
    *,
    config: WorldConsistencyConfig | None = None,
) -> WorldConsistencyEvaluator:
    return WorldConsistencyEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_world_dataset(
    *,
    sample_count: int = 128,
    horizon: int = 12,
    seed: int = 42,
    dynamics_scale: float = 0.5,
) -> WorldConsistencyDataset:
    """
    Synthetic linear dynamics:

        next_state = state + 0.5 * action

    State and action dimensions are both 4.

    This is an evaluator sanity check, not
    a realistic world simulation.
    """

    if sample_count < 1 or horizon < 1:
        raise ValueError(
            "Sample count and horizon must be positive"
        )

    rng = np.random.default_rng(seed)

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

    return WorldConsistencyDataset(
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
    name = "demo_world_consistency_model"

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
    dataset = generate_synthetic_world_dataset(
        sample_count=128,
        horizon=12,
        seed=42,
    )

    model = DemoWorldModel()

    evaluator = (
        create_world_consistency_evaluator(
            model
        )
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — WORLD CONSISTENCY")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/world_consistency.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/world_consistency.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
