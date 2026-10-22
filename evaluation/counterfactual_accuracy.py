
"""
NeuroWorld AI
evaluation/counterfactual_accuracy.py

Counterfactual Accuracy Evaluation.

Evaluates action-conditioned world-model predictions
against independently supplied reference outcomes.

Features:
- Factual next-state prediction accuracy
- Counterfactual next-state prediction accuracy
- Intervention effect accuracy
- Action sensitivity evaluation
- Direction alignment
- Counterfactual ranking accuracy
- Multi-step counterfactual rollouts
- Uncertainty-error correlation
- Model comparison
- Weighted evaluation scores
- JSON / CSV report export
- Reproducible synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.counterfactual_accuracy

IMPORTANT:
    Ground-truth counterfactual outcomes are required
    for genuine counterfactual accuracy evaluation.

    If counterfactual targets are generated from a
    simulator, results measure simulator agreement,
    not necessarily real-world causal accuracy.
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
    "neuroworld.evaluation.counterfactual_accuracy"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class CounterfactualEvaluationError(Exception):
    pass


class CounterfactualValidationError(
    CounterfactualEvaluationError
):
    pass


class CounterfactualModelError(
    CounterfactualEvaluationError
):
    pass


# ============================================================
# MODEL INTERFACE
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
class CounterfactualConfig:
    prediction_error_scale: float = 1.0
    intervention_error_scale: float = 1.0
    trajectory_error_scale: float = 1.0

    sensitivity_threshold: float = 0.01
    ranking_tolerance: float = 1e-6

    max_samples: int = 10000
    max_horizon: int = 256

    uncertainty_threshold: float = 1.0

    random_seed: int = 42
    epsilon: float = 1e-8

    pass_threshold: float = 0.70

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "factual_accuracy": 0.10,
            "counterfactual_accuracy": 0.25,
            "intervention_accuracy": 0.25,
            "direction_alignment": 0.10,
            "action_sensitivity": 0.10,
            "ranking_accuracy": 0.10,
            "trajectory_accuracy": 0.10,
        }
    )

    def __post_init__(self):
        for name in (
            "prediction_error_scale",
            "intervention_error_scale",
            "trajectory_error_scale",
            "epsilon",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if self.sensitivity_threshold < 0:
            raise ValueError(
                "sensitivity_threshold must be nonnegative"
            )

        if self.ranking_tolerance < 0:
            raise ValueError(
                "ranking_tolerance must be nonnegative"
            )

        if self.max_samples < 1:
            raise ValueError(
                "max_samples must be positive"
            )

        if self.max_horizon < 1:
            raise ValueError(
                "max_horizon must be positive"
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
            "factual_accuracy",
            "counterfactual_accuracy",
            "intervention_accuracy",
            "direction_alignment",
            "action_sensitivity",
            "ranking_accuracy",
            "trajectory_accuracy",
        }

        if not self.weights or set(self.weights) - allowed:
            raise ValueError(
                "Invalid evaluation metric weights"
            )

        if (
            any(
                not math.isfinite(value) or value < 0
                for value in self.weights.values()
            )
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Weights must be nonnegative with "
                "a positive total"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class CounterfactualDataset:
    """
    Required:
        states:                  (N, S)
        factual_actions:         (N, A)
        counterfactual_actions:  (N, A)
        factual_targets:         (N, S)
        counterfactual_targets:  (N, S)

    Optional:
        trajectory_actions:      (N, H, A)
        trajectory_targets:      (N, H, S)

    Factual and counterfactual targets must
    correspond to the same initial state.
    """

    states: np.ndarray

    factual_actions: np.ndarray
    counterfactual_actions: np.ndarray

    factual_targets: np.ndarray
    counterfactual_targets: np.ndarray

    trajectory_actions: np.ndarray | None = None
    trajectory_targets: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        state_dim: int,
        action_dim: int,
        *,
        max_samples: int = 10000,
        max_horizon: int = 256,
    ) -> None:
        required = {
            "states": self.states,
            "factual_actions": self.factual_actions,
            "counterfactual_actions": (
                self.counterfactual_actions
            ),
            "factual_targets": self.factual_targets,
            "counterfactual_targets": (
                self.counterfactual_targets
            ),
        }

        converted = {}

        for name, values in required.items():
            try:
                array = np.asarray(
                    values,
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise CounterfactualValidationError(
                    f"{name} must be numeric"
                ) from exc

            if not np.all(np.isfinite(array)):
                raise CounterfactualValidationError(
                    f"{name} contains non-finite values"
                )

            converted[name] = array

        n = len(converted["states"])

        if not 1 <= n <= max_samples:
            raise CounterfactualValidationError(
                "Invalid sample count"
            )

        shapes = {
            "states": (n, state_dim),
            "factual_actions": (n, action_dim),
            "counterfactual_actions": (
                n, action_dim
            ),
            "factual_targets": (n, state_dim),
            "counterfactual_targets": (
                n, state_dim
            ),
        }

        for name, expected in shapes.items():
            if converted[name].shape != expected:
                raise CounterfactualValidationError(
                    f"{name} must have shape {expected}"
                )

        for name, array in converted.items():
            setattr(self, name, array.copy())

        has_actions = (
            self.trajectory_actions is not None
        )

        has_targets = (
            self.trajectory_targets is not None
        )

        if has_actions != has_targets:
            raise CounterfactualValidationError(
                "Both trajectory arrays are required"
            )

        if has_actions:
            actions = np.asarray(
                self.trajectory_actions,
                dtype=np.float64,
            )

            targets = np.asarray(
                self.trajectory_targets,
                dtype=np.float64,
            )

            if actions.ndim != 3:
                raise CounterfactualValidationError(
                    "trajectory_actions must be 3D"
                )

            horizon = actions.shape[1]

            if not 1 <= horizon <= max_horizon:
                raise CounterfactualValidationError(
                    "Invalid trajectory horizon"
                )

            if actions.shape != (
                n, horizon, action_dim
            ):
                raise CounterfactualValidationError(
                    "Invalid trajectory_actions shape"
                )

            if targets.shape != (
                n, horizon, state_dim
            ):
                raise CounterfactualValidationError(
                    "Invalid trajectory_targets shape"
                )

            if not (
                np.all(np.isfinite(actions))
                and np.all(np.isfinite(targets))
            ):
                raise CounterfactualValidationError(
                    "Trajectory data must be finite"
                )

            self.trajectory_actions = actions.copy()
            self.trajectory_targets = targets.copy()


# ============================================================
# METRICS
# ============================================================

@dataclass
class CounterfactualMetrics:
    factual_accuracy: float
    counterfactual_accuracy: float
    intervention_accuracy: float

    direction_alignment: float
    action_sensitivity: float
    ranking_accuracy: float

    trajectory_accuracy: float | None

    uncertainty_error_correlation: float
    uncertainty_acceptance_rate: float

    overall_score: float


@dataclass
class CounterfactualResult:
    evaluation_id: str
    model_name: str

    sample_count: int

    metrics: CounterfactualMetrics

    passed: bool
    created_at: str

    raw_metrics: dict[str, float]

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    if predictions.shape != targets.shape:
        raise CounterfactualValidationError(
            "Prediction and target shapes must match"
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


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    denominator = (
        np.linalg.norm(first)
        * np.linalg.norm(second)
    )

    if denominator <= epsilon:
        return 0.0

    return float(
        np.clip(
            np.dot(first, second) / denominator,
            -1.0,
            1.0,
        )
    )


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    x = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)

    y = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)

    if x.shape != y.shape:
        raise CounterfactualValidationError(
            "Correlation inputs must match"
        )

    if (
        len(x) < 2
        or np.std(x) <= epsilon
        or np.std(y) <= epsilon
    ):
        return 0.0

    value = float(
        np.corrcoef(x, y)[0, 1]
    )

    return (
        value
        if math.isfinite(value)
        else 0.0
    )


# ============================================================
# EVALUATOR
# ============================================================

class CounterfactualAccuracyEvaluator:
    """
    Evaluates factual and counterfactual predictions.

    Accuracy requires independently supplied
    reference outcomes.

    The model must implement:
        predict_transition(state, action)

    Evaluation does not issue real-world actions.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: CounterfactualConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or CounterfactualConfig()
        )

        self._results: dict[
            str,
            CounterfactualResult,
        ] = {}

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    def _predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float]:
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

            uncertainty = float(
                getattr(
                    transition,
                    "uncertainty",
                    0.0,
                )
            )

        except Exception as exc:
            raise CounterfactualModelError(
                f"Prediction failed: {exc}"
            ) from exc

        if predicted.shape != (
            self.model.state_dim,
        ):
            raise CounterfactualModelError(
                "Invalid predicted state shape"
            )

        if not np.all(np.isfinite(predicted)):
            raise CounterfactualModelError(
                "Predicted state contains invalid values"
            )

        if (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise CounterfactualModelError(
                "Invalid predicted uncertainty"
            )

        return predicted, uncertainty

    # --------------------------------------------------------
    # BATCH PREDICTION
    # --------------------------------------------------------

    def _predict_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        predictions = []
        uncertainties = []

        for state, action in zip(
            states,
            actions,
        ):
            predicted, uncertainty = self._predict(
                state,
                action,
            )

            predictions.append(predicted)
            uncertainties.append(uncertainty)

        return (
            np.stack(predictions),
            np.asarray(
                uncertainties,
                dtype=np.float64,
            ),
        )

    # --------------------------------------------------------
    # TRAJECTORY EVALUATION
    # --------------------------------------------------------

    def evaluate_trajectory(
        self,
        dataset: CounterfactualDataset,
    ) -> tuple[float, float] | None:
        if dataset.trajectory_actions is None:
            return None

        predictions = []

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
                    state,
                    action,
                )

                predicted_steps.append(
                    state.copy()
                )

            predictions.append(
                predicted_steps
            )

        predicted_array = np.asarray(
            predictions,
            dtype=np.float64,
        )

        trajectory_rmse = rmse(
            predicted_array,
            dataset.trajectory_targets,
        )

        score = error_to_score(
            trajectory_rmse,
            self.config.trajectory_error_scale,
        )

        return score, trajectory_rmse

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: CounterfactualDataset,
    ) -> CounterfactualResult:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        factual_predictions, factual_uncertainties = (
            self._predict_batch(
                dataset.states,
                dataset.factual_actions,
            )
        )

        counterfactual_predictions, cf_uncertainties = (
            self._predict_batch(
                dataset.states,
                dataset.counterfactual_actions,
            )
        )

        factual_rmse = rmse(
            factual_predictions,
            dataset.factual_targets,
        )

        counterfactual_rmse = rmse(
            counterfactual_predictions,
            dataset.counterfactual_targets,
        )

        factual_score = error_to_score(
            factual_rmse,
            self.config.prediction_error_scale,
        )

        counterfactual_score = error_to_score(
            counterfactual_rmse,
            self.config.prediction_error_scale,
        )

        # Compare predicted action effects against
        # reference action effects.
        predicted_effects = (
            counterfactual_predictions
            - factual_predictions
        )

        reference_effects = (
            dataset.counterfactual_targets
            - dataset.factual_targets
        )

        intervention_rmse = rmse(
            predicted_effects,
            reference_effects,
        )

        intervention_score = error_to_score(
            intervention_rmse,
            self.config.intervention_error_scale,
        )

        # Direction alignment of intervention effects.
        alignments = []

        for predicted, reference in zip(
            predicted_effects,
            reference_effects,
        ):
            predicted_norm = np.linalg.norm(
                predicted
            )

            reference_norm = np.linalg.norm(
                reference
            )

            if (
                predicted_norm <= self.config.epsilon
                and reference_norm <= self.config.epsilon
            ):
                alignment = 1.0

            elif (
                predicted_norm <= self.config.epsilon
                or reference_norm <= self.config.epsilon
            ):
                alignment = 0.0

            else:
                alignment = (
                    cosine_similarity(
                        predicted,
                        reference,
                        self.config.epsilon,
                    )
                    + 1.0
                ) / 2.0

            alignments.append(alignment)

        direction_alignment = float(
            np.mean(alignments)
        )

        # Compare intervention sensitivity labels.
        predicted_magnitudes = np.linalg.norm(
            predicted_effects,
            axis=1,
        )

        reference_magnitudes = np.linalg.norm(
            reference_effects,
            axis=1,
        )

        predicted_sensitive = (
            predicted_magnitudes
            >= self.config.sensitivity_threshold
        )

        reference_sensitive = (
            reference_magnitudes
            >= self.config.sensitivity_threshold
        )

        sensitivity_accuracy = float(
            np.mean(
                predicted_sensitive
                == reference_sensitive
            )
        )

        # Ranking: does the model identify which
        # action produces the larger state change?
        predicted_factual_movement = np.linalg.norm(
            factual_predictions - dataset.states,
            axis=1,
        )

        predicted_cf_movement = np.linalg.norm(
            counterfactual_predictions - dataset.states,
            axis=1,
        )

        reference_factual_movement = np.linalg.norm(
            dataset.factual_targets - dataset.states,
            axis=1,
        )

        reference_cf_movement = np.linalg.norm(
            dataset.counterfactual_targets - dataset.states,
            axis=1,
        )

        predicted_difference = (
            predicted_cf_movement
            - predicted_factual_movement
        )

        reference_difference = (
            reference_cf_movement
            - reference_factual_movement
        )

        tolerance = (
            self.config.ranking_tolerance
        )

        predicted_ranks = np.where(
            predicted_difference > tolerance,
            1,
            np.where(
                predicted_difference < -tolerance,
                -1,
                0,
            ),
        )

        reference_ranks = np.where(
            reference_difference > tolerance,
            1,
            np.where(
                reference_difference < -tolerance,
                -1,
                0,
            ),
        )

        ranking_accuracy = float(
            np.mean(
                predicted_ranks == reference_ranks
            )
        )

        # Uncertainty analysis.
        per_sample_errors = np.linalg.norm(
            counterfactual_predictions
            - dataset.counterfactual_targets,
            axis=1,
        )

        uncertainty_correlation = safe_correlation(
            cf_uncertainties,
            per_sample_errors,
            self.config.epsilon,
        )

        uncertainty_acceptance_rate = float(
            np.mean(
                cf_uncertainties
                <= self.config.uncertainty_threshold
            )
        )

        trajectory_result = (
            self.evaluate_trajectory(dataset)
        )

        trajectory_score = (
            trajectory_result[0]
            if trajectory_result is not None
            else None
        )

        trajectory_rmse = (
            trajectory_result[1]
            if trajectory_result is not None
            else None
        )

        scores: dict[str, float | None] = {
            "factual_accuracy": factual_score,
            "counterfactual_accuracy": counterfactual_score,
            "intervention_accuracy": intervention_score,
            "direction_alignment": direction_alignment,
            "action_sensitivity": sensitivity_accuracy,
            "ranking_accuracy": ranking_accuracy,
            "trajectory_accuracy": trajectory_score,
        }

        # Optional metrics are excluded when no
        # corresponding ground-truth data exists.
        active_weights = {
            name: weight
            for name, weight in self.config.weights.items()
            if scores[name] is not None
        }

        total_weight = sum(
            active_weights.values()
        )

        if total_weight <= 0:
            raise CounterfactualValidationError(
                "No weighted metrics are available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = CounterfactualMetrics(
            factual_accuracy=factual_score,
            counterfactual_accuracy=counterfactual_score,
            intervention_accuracy=intervention_score,
            direction_alignment=direction_alignment,
            action_sensitivity=sensitivity_accuracy,
            ranking_accuracy=ranking_accuracy,
            trajectory_accuracy=trajectory_score,
            uncertainty_error_correlation=(
                uncertainty_correlation
            ),
            uncertainty_acceptance_rate=(
                uncertainty_acceptance_rate
            ),
            overall_score=float(overall_score),
        )

        result = CounterfactualResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            sample_count=len(dataset.states),
            metrics=metrics,
            passed=(
                overall_score
                >= self.config.pass_threshold
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            raw_metrics={
                "factual_rmse": factual_rmse,
                "counterfactual_rmse": counterfactual_rmse,
                "intervention_rmse": intervention_rmse,
                **(
                    {
                        "trajectory_rmse": trajectory_rmse
                    }
                    if trajectory_rmse is not None
                    else {}
                ),
            },
            details={
                "weights": active_weights,
                "pass_threshold": (
                    self.config.pass_threshold
                ),
                "uncertainty_threshold": (
                    self.config.uncertainty_threshold
                ),
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "interpretation": (
                    "Agreement with supplied factual and "
                    "counterfactual reference outcomes"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Counterfactual evaluation: model=%s "
            "score=%.4f passed=%s",
            self.model.name,
            overall_score,
            result.passed,
        )

        return result

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> CounterfactualResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise CounterfactualEvaluationError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[CounterfactualResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[CounterfactualResult],
    ) -> list[CounterfactualResult]:
        return sorted(
            results,
            key=lambda item: (
                item.metrics.overall_score
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
        results: list[CounterfactualResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise CounterfactualValidationError(
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

def create_counterfactual_evaluator(
    model: WorldModelProtocol,
    *,
    config: CounterfactualConfig | None = None,
) -> CounterfactualAccuracyEvaluator:
    return CounterfactualAccuracyEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_counterfactual_dataset(
    *,
    state_dim: int = 4,
    action_dim: int = 4,
    sample_count: int = 128,
    horizon: int = 8,
    seed: int = 42,
    dynamics_scale: float = 0.5,
) -> CounterfactualDataset:
    """
    Synthetic linear environment:

        next_state = state + B @ action

    Both factual and counterfactual outcomes
    come from the same known simulator.

    Intended for testing evaluator behavior.
    """

    if min(
        state_dim,
        action_dim,
        sample_count,
        horizon,
    ) < 1:
        raise CounterfactualValidationError(
            "Dimensions and counts must be positive"
        )

    rng = np.random.default_rng(seed)

    matrix = np.zeros(
        (state_dim, action_dim),
        dtype=np.float64,
    )

    for index in range(
        min(state_dim, action_dim)
    ):
        matrix[index, index] = dynamics_scale

    states = rng.normal(
        0.0,
        1.0,
        size=(sample_count, state_dim),
    )

    factual_actions = rng.uniform(
        -1.0,
        1.0,
        size=(sample_count, action_dim),
    )

    counterfactual_actions = rng.uniform(
        -1.0,
        1.0,
        size=(sample_count, action_dim),
    )

    factual_targets = (
        states
        + factual_actions @ matrix.T
    )

    counterfactual_targets = (
        states
        + counterfactual_actions @ matrix.T
    )

    trajectory_actions = rng.uniform(
        -1.0,
        1.0,
        size=(sample_count, horizon, action_dim),
    )

    trajectory_targets = np.empty(
        (sample_count, horizon, state_dim),
        dtype=np.float64,
    )

    current_states = states.copy()

    for step in range(horizon):
        current_states = (
            current_states
            + trajectory_actions[:, step, :] @ matrix.T
        )

        trajectory_targets[:, step, :] = (
            current_states
        )

    return CounterfactualDataset(
        states=states,
        factual_actions=factual_actions,
        counterfactual_actions=counterfactual_actions,
        factual_targets=factual_targets,
        counterfactual_targets=counterfactual_targets,
        trajectory_actions=trajectory_actions,
        trajectory_targets=trajectory_targets,
        metadata={
            "source": "synthetic_linear_simulator",
            "seed": seed,
            "dynamics_scale": dynamics_scale,
        },
    )


# ============================================================
# DEMONSTRATION MODEL
# ============================================================

class DemoWorldModel:
    name = "demo_counterfactual_world_model"

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
            reward=0.0,
            terminal=False,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    dataset = (
        generate_synthetic_counterfactual_dataset(
            state_dim=4,
            action_dim=4,
            sample_count=128,
            horizon=8,
            seed=42,
        )
    )

    model = DemoWorldModel()

    evaluator = create_counterfactual_evaluator(
        model
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — COUNTERFACTUAL ACCURACY")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/counterfactual_accuracy.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/counterfactual_accuracy.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
