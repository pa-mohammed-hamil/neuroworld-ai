
"""
NeuroWorld AI
evaluation/future_prediction.py

Future State Prediction Evaluation.

Features:
- One-step prediction accuracy
- Multi-step future prediction accuracy
- Autoregressive world-model rollouts
- Teacher-forced prediction evaluation
- Horizon-wise RMSE and MAE
- Final-state prediction accuracy
- Accumulated prediction drift
- Trajectory displacement errors
- Uncertainty-error correlation
- Uncertainty threshold coverage
- Terminal-event prediction accuracy (optional)
- Model comparison
- Weighted evaluation scoring
- JSON and CSV report export
- Reproducible synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.future_prediction

Model interface:
    model.name
    model.state_dim
    model.action_dim
    model.predict_transition(state, action)

Expected transition:
    transition.next_state
    transition.uncertainty (optional)
    transition.terminal (optional)

Important:
    Evaluation requires independently supplied
    ground-truth future states.

    Synthetic demonstration scores are not evidence
    of real-world predictive performance.
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
    "neuroworld.evaluation.future_prediction"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class FuturePredictionError(Exception):
    pass


class FuturePredictionValidationError(
    FuturePredictionError
):
    pass


class FuturePredictionModelError(
    FuturePredictionError
):
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
class FuturePredictionConfig:
    max_samples: int = 10000
    max_horizon: int = 256

    prediction_error_scale: float = 1.0
    trajectory_error_scale: float = 1.0
    drift_error_scale: float = 1.0

    uncertainty_threshold: float = 1.0

    pass_threshold: float = 0.70

    random_seed: int = 42
    epsilon: float = 1e-8

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "one_step_accuracy": 0.15,
            "rollout_accuracy": 0.25,
            "final_state_accuracy": 0.15,
            "teacher_forced_accuracy": 0.10,
            "trajectory_accuracy": 0.15,
            "drift_score": 0.10,
            "uncertainty_alignment": 0.05,
            "terminal_accuracy": 0.05,
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
            "prediction_error_scale",
            "trajectory_error_scale",
            "drift_error_scale",
            "epsilon",
        ):
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        if not (
            0 <= self.uncertainty_threshold <= 1
        ):
            raise ValueError(
                "uncertainty_threshold must be in [0, 1]"
            )

        if not (
            0 <= self.pass_threshold <= 1
        ):
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        allowed = {
            "one_step_accuracy",
            "rollout_accuracy",
            "final_state_accuracy",
            "teacher_forced_accuracy",
            "trajectory_accuracy",
            "drift_score",
            "uncertainty_alignment",
            "terminal_accuracy",
        }

        if (
            not self.weights
            or set(self.weights) - allowed
        ):
            raise ValueError(
                "Invalid metric weights"
            )

        if (
            any(
                not math.isfinite(weight)
                or weight < 0
                for weight in self.weights.values()
            )
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Weights must have a positive total"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class FuturePredictionDataset:
    """
    initial_states:
        (N, state_dim)

    actions:
        (N, horizon, action_dim)

    target_states:
        (N, horizon, state_dim)

    terminal_targets:
        Optional (N, horizon) boolean array.

    Target states correspond to states AFTER
    applying each action.
    """

    initial_states: np.ndarray
    actions: np.ndarray
    target_states: np.ndarray

    terminal_targets: np.ndarray | None = None

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
            raise FuturePredictionValidationError(
                "Dataset must contain numeric arrays"
            ) from exc

        if states.ndim != 2:
            raise FuturePredictionValidationError(
                "initial_states must be 2D"
            )

        if actions.ndim != 3:
            raise FuturePredictionValidationError(
                "actions must be 3D"
            )

        if targets.ndim != 3:
            raise FuturePredictionValidationError(
                "target_states must be 3D"
            )

        n = states.shape[0]
        horizon = actions.shape[1]

        if not 1 <= n <= max_samples:
            raise FuturePredictionValidationError(
                "Invalid sample count"
            )

        if not 1 <= horizon <= max_horizon:
            raise FuturePredictionValidationError(
                "Invalid prediction horizon"
            )

        if states.shape != (
            n,
            state_dim,
        ):
            raise FuturePredictionValidationError(
                "Invalid initial state dimensions"
            )

        if actions.shape != (
            n,
            horizon,
            action_dim,
        ):
            raise FuturePredictionValidationError(
                "Invalid action dimensions"
            )

        if targets.shape != (
            n,
            horizon,
            state_dim,
        ):
            raise FuturePredictionValidationError(
                "Invalid target state dimensions"
            )

        if not all(
            np.all(np.isfinite(array))
            for array in (
                states,
                actions,
                targets,
            )
        ):
            raise FuturePredictionValidationError(
                "Dataset contains non-finite values"
            )

        self.initial_states = states.copy()
        self.actions = actions.copy()
        self.target_states = targets.copy()

        if self.terminal_targets is not None:
            terminals = np.asarray(
                self.terminal_targets
            )

            if terminals.shape != (
                n,
                horizon,
            ):
                raise FuturePredictionValidationError(
                    "Invalid terminal_targets dimensions"
                )

            if not np.issubdtype(
                terminals.dtype,
                np.bool_,
            ):
                raise FuturePredictionValidationError(
                    "terminal_targets must be boolean"
                )

            self.terminal_targets = terminals.copy()


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class HorizonMetrics:
    horizon_step: int

    rmse: float
    mae: float

    mean_displacement_error: float
    accuracy_score: float

    mean_uncertainty: float


@dataclass
class FuturePredictionMetrics:
    one_step_accuracy: float
    rollout_accuracy: float

    final_state_accuracy: float
    teacher_forced_accuracy: float

    trajectory_accuracy: float
    drift_score: float

    uncertainty_alignment: float
    uncertainty_acceptance_rate: float

    terminal_accuracy: float | None

    overall_score: float


@dataclass
class FuturePredictionResult:
    evaluation_id: str
    model_name: str

    sample_count: int
    horizon: int

    metrics: FuturePredictionMetrics

    horizon_metrics: list[HorizonMetrics]

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
        raise FuturePredictionValidationError(
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
        raise FuturePredictionValidationError(
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
        raise FuturePredictionValidationError(
            "Correlation arrays must match"
        )

    if (
        len(x) < 2
        or np.std(x) <= epsilon
        or np.std(y) <= epsilon
    ):
        return 0.0

    correlation = float(
        np.corrcoef(x, y)[0, 1]
    )

    return (
        correlation
        if math.isfinite(correlation)
        else 0.0
    )


# ============================================================
# EVALUATOR
# ============================================================

class FuturePredictionEvaluator:
    """
    Evaluates neural world-model future predictions.

    Supports:
        - Autoregressive prediction
        - Teacher-forced prediction
        - Horizon-wise error analysis
        - Prediction drift analysis
        - Uncertainty diagnostics

    This module evaluates predictions only.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: FuturePredictionConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or FuturePredictionConfig()
        )

        self._results: dict[
            str,
            FuturePredictionResult,
        ] = {}

    # --------------------------------------------------------
    # SINGLE TRANSITION
    # --------------------------------------------------------

    def _predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool]:
        try:
            transition = (
                self.model.predict_transition(
                    state.copy(),
                    action.copy(),
                )
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

            terminal = bool(
                getattr(
                    transition,
                    "terminal",
                    False,
                )
            )

        except Exception as exc:
            raise FuturePredictionModelError(
                f"Prediction failed: {exc}"
            ) from exc

        if next_state.shape != (
            self.model.state_dim,
        ):
            raise FuturePredictionModelError(
                "Invalid predicted state shape"
            )

        if not np.all(np.isfinite(next_state)):
            raise FuturePredictionModelError(
                "Predicted state contains non-finite values"
            )

        if (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise FuturePredictionModelError(
                "Invalid uncertainty value"
            )

        return (
            next_state,
            uncertainty,
            terminal,
        )

    # --------------------------------------------------------
    # AUTOREGRESSIVE ROLLOUT
    # --------------------------------------------------------

    def predict_rollout(
        self,
        dataset: FuturePredictionDataset,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        predictions = []
        uncertainties = []
        terminals = []

        horizon = dataset.actions.shape[1]

        for sample_index, initial_state in enumerate(
            dataset.initial_states
        ):
            state = initial_state.copy()

            predicted_states = []
            predicted_uncertainties = []
            predicted_terminals = []

            for step in range(horizon):
                action = (
                    dataset.actions[
                        sample_index,
                        step,
                    ]
                )

                state, uncertainty, terminal = (
                    self._predict(
                        state,
                        action,
                    )
                )

                predicted_states.append(
                    state.copy()
                )

                predicted_uncertainties.append(
                    uncertainty
                )

                predicted_terminals.append(
                    terminal
                )

            predictions.append(
                predicted_states
            )

            uncertainties.append(
                predicted_uncertainties
            )

            terminals.append(
                predicted_terminals
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
            np.asarray(
                terminals,
                dtype=np.bool_,
            ),
        )

    # --------------------------------------------------------
    # TEACHER-FORCED PREDICTION
    # --------------------------------------------------------

    def predict_teacher_forced(
        self,
        dataset: FuturePredictionDataset,
    ) -> np.ndarray:
        """
        Each step uses the reference state
        from the previous timestep.
        """

        predictions = []

        horizon = dataset.actions.shape[1]

        for sample_index, initial_state in enumerate(
            dataset.initial_states
        ):
            predicted_steps = []

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

                predicted, _, _ = self._predict(
                    input_state,
                    action,
                )

                predicted_steps.append(
                    predicted
                )

            predictions.append(
                predicted_steps
            )

        return np.asarray(
            predictions,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # HORIZON METRICS
    # --------------------------------------------------------

    def evaluate_horizons(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray,
    ) -> list[HorizonMetrics]:
        horizon = predictions.shape[1]

        metrics = []

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

            metrics.append(
                HorizonMetrics(
                    horizon_step=step + 1,
                    rmse=rmse,
                    mae=mae,
                    mean_displacement_error=displacement,
                    accuracy_score=error_to_score(
                        rmse,
                        self.config.prediction_error_scale,
                    ),
                    mean_uncertainty=float(
                        np.mean(
                            uncertainties[:, step]
                        )
                    ),
                )
            )

        return metrics

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: FuturePredictionDataset,
    ) -> FuturePredictionResult:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        predictions, uncertainties, terminals = (
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

        one_step_rmse = calculate_rmse(
            predictions[:, 0, :],
            targets[:, 0, :],
        )

        rollout_rmse = calculate_rmse(
            predictions,
            targets,
        )

        final_state_rmse = calculate_rmse(
            predictions[:, -1, :],
            targets[:, -1, :],
        )

        teacher_rmse = calculate_rmse(
            teacher_predictions,
            targets,
        )

        # Mean Euclidean displacement across
        # all samples and prediction horizons.
        displacement_errors = np.linalg.norm(
            predictions - targets,
            axis=2,
        )

        mean_displacement_error = float(
            np.mean(
                displacement_errors
            )
        )

        # Drift measures growth in mean prediction
        # error between the first and last horizons.
        mean_first_error = float(
            np.mean(
                displacement_errors[:, 0]
            )
        )

        mean_last_error = float(
            np.mean(
                displacement_errors[:, -1]
            )
        )

        drift = max(
            0.0,
            mean_last_error - mean_first_error,
        )

        # Uncertainty-error correlation.
        uncertainty_correlation = safe_correlation(
            uncertainties,
            displacement_errors,
            self.config.epsilon,
        )

        # Map correlation [-1, 1] to [0, 1].
        # A higher value indicates stronger
        # positive association between uncertainty
        # and observed prediction error.
        uncertainty_alignment = float(
            np.clip(
                (uncertainty_correlation + 1.0) / 2.0,
                0.0,
                1.0,
            )
        )

        uncertainty_acceptance_rate = float(
            np.mean(
                uncertainties
                <= self.config.uncertainty_threshold
            )
        )

        terminal_accuracy = None

        if dataset.terminal_targets is not None:
            terminal_accuracy = float(
                np.mean(
                    terminals
                    == dataset.terminal_targets
                )
            )

        scores: dict[str, float | None] = {
            "one_step_accuracy": error_to_score(
                one_step_rmse,
                self.config.prediction_error_scale,
            ),
            "rollout_accuracy": error_to_score(
                rollout_rmse,
                self.config.prediction_error_scale,
            ),
            "final_state_accuracy": error_to_score(
                final_state_rmse,
                self.config.prediction_error_scale,
            ),
            "teacher_forced_accuracy": error_to_score(
                teacher_rmse,
                self.config.prediction_error_scale,
            ),
            "trajectory_accuracy": error_to_score(
                mean_displacement_error,
                self.config.trajectory_error_scale,
            ),
            "drift_score": error_to_score(
                drift,
                self.config.drift_error_scale,
            ),
            "uncertainty_alignment": (
                uncertainty_alignment
            ),
            "terminal_accuracy": terminal_accuracy,
        }

        # Exclude optional metrics without labels.
        active_weights = {
            name: weight
            for name, weight in self.config.weights.items()
            if scores[name] is not None
        }

        total_weight = sum(
            active_weights.values()
        )

        if total_weight <= 0:
            raise FuturePredictionValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = FuturePredictionMetrics(
            one_step_accuracy=float(
                scores["one_step_accuracy"]
            ),
            rollout_accuracy=float(
                scores["rollout_accuracy"]
            ),
            final_state_accuracy=float(
                scores["final_state_accuracy"]
            ),
            teacher_forced_accuracy=float(
                scores["teacher_forced_accuracy"]
            ),
            trajectory_accuracy=float(
                scores["trajectory_accuracy"]
            ),
            drift_score=float(
                scores["drift_score"]
            ),
            uncertainty_alignment=(
                uncertainty_alignment
            ),
            uncertainty_acceptance_rate=(
                uncertainty_acceptance_rate
            ),
            terminal_accuracy=terminal_accuracy,
            overall_score=float(overall_score),
        )

        result = FuturePredictionResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            sample_count=len(dataset.initial_states),
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
            raw_metrics={
                "one_step_rmse": one_step_rmse,
                "rollout_rmse": rollout_rmse,
                "final_state_rmse": final_state_rmse,
                "teacher_forced_rmse": teacher_rmse,
                "mean_displacement_error": (
                    mean_displacement_error
                ),
                "drift": drift,
                "uncertainty_error_correlation": (
                    uncertainty_correlation
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
                    "Agreement between predicted future "
                    "states and supplied reference states"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Future prediction evaluation completed: "
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
    ) -> FuturePredictionResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise FuturePredictionError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[FuturePredictionResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[FuturePredictionResult],
    ) -> list[FuturePredictionResult]:
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

        path = Path(
            output_path
        )

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
        results: list[FuturePredictionResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise FuturePredictionValidationError(
                "No results to export"
            )

        path = Path(
            output_path
        )

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

def create_future_prediction_evaluator(
    model: WorldModelProtocol,
    *,
    config: FuturePredictionConfig | None = None,
) -> FuturePredictionEvaluator:
    return FuturePredictionEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DATASET GENERATOR
# ============================================================

def generate_synthetic_future_dataset(
    *,
    state_dim: int = 4,
    action_dim: int = 4,
    sample_count: int = 128,
    horizon: int = 12,
    seed: int = 42,
    dynamics_scale: float = 0.5,
) -> FuturePredictionDataset:
    """
    Deterministic synthetic environment:

        next_state = state + B @ action

    Used to verify evaluator functionality.
    """

    if min(
        state_dim,
        action_dim,
        sample_count,
        horizon,
    ) < 1:
        raise FuturePredictionValidationError(
            "Dimensions and counts must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

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

    target_states = np.empty(
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

        target_states[:, step, :] = current

    return FuturePredictionDataset(
        initial_states=initial_states,
        actions=actions,
        target_states=target_states,
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
    name = "demo_future_prediction_world_model"

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
# DEMONSTRATION
# ============================================================

def main() -> None:
    dataset = generate_synthetic_future_dataset(
        state_dim=4,
        action_dim=4,
        sample_count=128,
        horizon=12,
        seed=42,
    )

    model = DemoWorldModel()

    evaluator = create_future_prediction_evaluator(
        model
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — FUTURE PREDICTION")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/future_prediction.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/future_prediction.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
