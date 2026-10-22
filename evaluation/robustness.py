
"""
NeuroWorld AI
evaluation/robustness.py

Neural World Model Robustness Evaluation.

Features:
- Clean prediction accuracy
- Gaussian observation noise
- Sensor / feature dropout
- Action perturbation
- State distribution shift
- Multi-step rollout robustness
- Prediction degradation analysis
- Uncertainty-error correlation
- Per-perturbation evaluation
- Model comparison
- JSON / CSV reporting
- Synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.robustness

Model interface:
    model.name
    model.state_dim
    model.action_dim
    model.predict_transition(state, action)

Transition:
    transition.next_state
    transition.uncertainty (optional)

Important:
    Robustness depends on the evaluation contract.

    For noisy observations, the reference target is
    the clean underlying state's true next state.

    For action perturbations, the reference target
    must reflect the actual perturbed action.

    For distribution shifts, ground truth must be
    generated or collected for the shifted state.

    A synthetic benchmark is not evidence of
    real-world model robustness.
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
from typing import Any, Callable, Protocol

import numpy as np


logger = logging.getLogger(
    "neuroworld.evaluation.robustness"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class RobustnessError(Exception):
    pass


class RobustnessValidationError(RobustnessError):
    pass


class RobustnessModelError(RobustnessError):
    pass


# ============================================================
# MODEL INTERFACES
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


class ReferenceDynamicsProtocol(Protocol):
    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        ...


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobustnessConfig:
    gaussian_noise_levels: tuple[float, ...] = (
        0.01,
        0.05,
        0.10,
        0.20,
    )

    dropout_rates: tuple[float, ...] = (
        0.05,
        0.10,
        0.25,
    )

    action_noise_levels: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
    )

    state_shift_levels: tuple[float, ...] = (
        0.25,
        0.50,
        1.00,
    )

    prediction_error_scale: float = 1.0
    trajectory_error_scale: float = 1.0

    max_samples: int = 10000
    max_horizon: int = 256

    random_seed: int = 42

    pass_threshold: float = 0.70
    max_accuracy_degradation: float = 0.25

    epsilon: float = 1e-8

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "clean_accuracy": 0.15,
            "noise_robustness": 0.20,
            "dropout_robustness": 0.20,
            "action_robustness": 0.15,
            "shift_robustness": 0.15,
            "trajectory_robustness": 0.15,
        }
    )

    def __post_init__(self):
        if (
            self.prediction_error_scale <= 0
            or self.trajectory_error_scale <= 0
        ):
            raise ValueError(
                "Error scales must be positive"
            )

        if self.max_samples < 1 or self.max_horizon < 1:
            raise ValueError(
                "Evaluation limits must be positive"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        if not 0 <= self.max_accuracy_degradation <= 1:
            raise ValueError(
                "max_accuracy_degradation must be in [0, 1]"
            )

        for name in (
            "gaussian_noise_levels",
            "action_noise_levels",
            "state_shift_levels",
        ):
            levels = getattr(self, name)

            if any(
                not math.isfinite(value) or value < 0
                for value in levels
            ):
                raise ValueError(
                    f"Invalid {name}"
                )

        if any(
            not math.isfinite(rate)
            or not 0 <= rate <= 1
            for rate in self.dropout_rates
        ):
            raise ValueError(
                "Dropout rates must be in [0, 1]"
            )

        allowed = {
            "clean_accuracy",
            "noise_robustness",
            "dropout_robustness",
            "action_robustness",
            "shift_robustness",
            "trajectory_robustness",
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
                "Invalid metric weights"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class RobustnessDataset:
    states: np.ndarray
    actions: np.ndarray
    target_next_states: np.ndarray

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
        max_samples: int,
        max_horizon: int,
    ) -> None:
        try:
            states = np.asarray(
                self.states,
                dtype=np.float64,
            )

            actions = np.asarray(
                self.actions,
                dtype=np.float64,
            )

            targets = np.asarray(
                self.target_next_states,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise RobustnessValidationError(
                "Dataset arrays must be numeric"
            ) from exc

        if states.ndim != 2:
            raise RobustnessValidationError(
                "states must be 2D"
            )

        n = states.shape[0]

        if not 1 <= n <= max_samples:
            raise RobustnessValidationError(
                "Invalid sample count"
            )

        if states.shape != (n, state_dim):
            raise RobustnessValidationError(
                "Invalid state shape"
            )

        if actions.shape != (n, action_dim):
            raise RobustnessValidationError(
                "Invalid action shape"
            )

        if targets.shape != (n, state_dim):
            raise RobustnessValidationError(
                "Invalid target shape"
            )

        if not all(
            np.all(np.isfinite(array))
            for array in (
                states,
                actions,
                targets,
            )
        ):
            raise RobustnessValidationError(
                "Dataset contains non-finite values"
            )

        self.states = states.copy()
        self.actions = actions.copy()
        self.target_next_states = targets.copy()

        has_actions = (
            self.trajectory_actions is not None
        )

        has_targets = (
            self.trajectory_targets is not None
        )

        if has_actions != has_targets:
            raise RobustnessValidationError(
                "Both trajectory arrays are required"
            )

        if has_actions:
            trajectory_actions = np.asarray(
                self.trajectory_actions,
                dtype=np.float64,
            )

            trajectory_targets = np.asarray(
                self.trajectory_targets,
                dtype=np.float64,
            )

            if trajectory_actions.ndim != 3:
                raise RobustnessValidationError(
                    "trajectory_actions must be 3D"
                )

            horizon = trajectory_actions.shape[1]

            if not 1 <= horizon <= max_horizon:
                raise RobustnessValidationError(
                    "Invalid trajectory horizon"
                )

            if trajectory_actions.shape != (
                n,
                horizon,
                action_dim,
            ):
                raise RobustnessValidationError(
                    "Invalid trajectory action shape"
                )

            if trajectory_targets.shape != (
                n,
                horizon,
                state_dim,
            ):
                raise RobustnessValidationError(
                    "Invalid trajectory target shape"
                )

            if not (
                np.all(np.isfinite(trajectory_actions))
                and np.all(np.isfinite(trajectory_targets))
            ):
                raise RobustnessValidationError(
                    "Trajectory data must be finite"
                )

            self.trajectory_actions = (
                trajectory_actions.copy()
            )

            self.trajectory_targets = (
                trajectory_targets.copy()
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class PerturbationMetrics:
    perturbation_type: str
    severity: float

    sample_count: int

    prediction_rmse: float
    prediction_accuracy: float

    clean_accuracy: float
    accuracy_degradation: float

    mean_uncertainty: float | None
    uncertainty_error_correlation: float | None


@dataclass
class RobustnessMetrics:
    clean_accuracy: float

    noise_robustness: float | None
    dropout_robustness: float | None
    action_robustness: float | None
    shift_robustness: float | None

    trajectory_robustness: float | None

    worst_case_accuracy: float
    max_accuracy_degradation: float

    overall_score: float


@dataclass
class RobustnessResult:
    evaluation_id: str
    model_name: str

    sample_count: int

    metrics: RobustnessMetrics

    perturbation_metrics: list[
        PerturbationMetrics
    ]

    passed: bool
    created_at: str

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
        raise RobustnessValidationError(
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


def safe_mean(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(
        np.mean(values)
    )


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float = 1e-8,
) -> float | None:
    if len(first) < 2:
        return None

    if (
        np.std(first) <= epsilon
        or np.std(second) <= epsilon
    ):
        return None

    correlation = float(
        np.corrcoef(first, second)[0, 1]
    )

    return (
        correlation
        if math.isfinite(correlation)
        else None
    )


# ============================================================
# ROBUSTNESS EVALUATOR
# ============================================================

class RobustnessEvaluator:
    """
    Evaluate world-model robustness under
    controlled perturbations.

    Reference dynamics are required for
    action perturbation and state-shift tests.

    Without reference dynamics, those tests
    are omitted rather than scored against
    incorrect clean-state targets.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: RobustnessConfig | None = None,
        reference_dynamics: (
            ReferenceDynamicsProtocol | None
        ) = None,
    ) -> None:
        self.model = model

        self.config = (
            config or RobustnessConfig()
        )

        self.reference_dynamics = (
            reference_dynamics
        )

        self._results: dict[
            str,
            RobustnessResult,
        ] = {}

    # --------------------------------------------------------
    # PREDICTION
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
            raise RobustnessModelError(
                f"Prediction failed: {exc}"
            ) from exc

        if predicted.shape != (
            self.model.state_dim,
        ):
            raise RobustnessModelError(
                "Invalid predicted state dimensions"
            )

        if not np.all(np.isfinite(predicted)):
            raise RobustnessModelError(
                "Prediction contains non-finite values"
            )

        if uncertainty is not None and (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise RobustnessModelError(
                "Invalid uncertainty value"
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

            predictions.append(
                predicted
            )

            uncertainties.append(
                np.nan
                if uncertainty is None
                else uncertainty
            )

        return (
            np.stack(predictions),
            np.asarray(
                uncertainties,
                dtype=np.float64,
            ),
        )

    # --------------------------------------------------------
    # REFERENCE DYNAMICS
    # --------------------------------------------------------

    def _reference_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        if self.reference_dynamics is None:
            raise RobustnessValidationError(
                "Reference dynamics are required"
            )

        targets = []

        for state, action in zip(
            states,
            actions,
        ):
            target = np.asarray(
                self.reference_dynamics(
                    state.copy(),
                    action.copy(),
                ),
                dtype=np.float64,
            )

            if target.shape != (
                self.model.state_dim,
            ):
                raise RobustnessValidationError(
                    "Invalid reference dynamics output"
                )

            if not np.all(np.isfinite(target)):
                raise RobustnessValidationError(
                    "Reference dynamics returned non-finite values"
                )

            targets.append(target)

        return np.stack(
            targets
        )

    # --------------------------------------------------------
    # PERTURBATION SCORING
    # --------------------------------------------------------

    def _evaluate_perturbation(
        self,
        *,
        perturbation_type: str,
        severity: float,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        clean_accuracy: float,
    ) -> PerturbationMetrics:
        predictions, uncertainties = (
            self._predict_batch(
                states,
                actions,
            )
        )

        rmse = calculate_rmse(
            predictions,
            targets,
        )

        accuracy = error_to_score(
            rmse,
            self.config.prediction_error_scale,
        )

        degradation = max(
            0.0,
            clean_accuracy - accuracy,
        )

        valid_uncertainties = (
            np.isfinite(uncertainties)
        )

        mean_uncertainty = None
        correlation = None

        if np.any(valid_uncertainties):
            mean_uncertainty = float(
                np.mean(
                    uncertainties[valid_uncertainties]
                )
            )

            per_sample_errors = np.linalg.norm(
                predictions - targets,
                axis=1,
            )

            correlation = safe_correlation(
                uncertainties[valid_uncertainties],
                per_sample_errors[valid_uncertainties],
                self.config.epsilon,
            )

        return PerturbationMetrics(
            perturbation_type=perturbation_type,
            severity=float(severity),
            sample_count=len(states),
            prediction_rmse=rmse,
            prediction_accuracy=accuracy,
            clean_accuracy=clean_accuracy,
            accuracy_degradation=degradation,
            mean_uncertainty=mean_uncertainty,
            uncertainty_error_correlation=correlation,
        )

    # --------------------------------------------------------
    # GAUSSIAN NOISE
    # --------------------------------------------------------

    def evaluate_gaussian_noise(
        self,
        dataset: RobustnessDataset,
        clean_accuracy: float,
    ) -> list[PerturbationMetrics]:
        results = []

        rng = np.random.default_rng(
            self.config.random_seed + 101
        )

        for severity in (
            self.config.gaussian_noise_levels
        ):
            noisy_states = (
                dataset.states
                + rng.normal(
                    0.0,
                    severity,
                    size=dataset.states.shape,
                )
            )

            results.append(
                self._evaluate_perturbation(
                    perturbation_type="gaussian_noise",
                    severity=severity,
                    states=noisy_states,
                    actions=dataset.actions,
                    targets=dataset.target_next_states,
                    clean_accuracy=clean_accuracy,
                )
            )

        return results

    # --------------------------------------------------------
    # SENSOR DROPOUT
    # --------------------------------------------------------

    def evaluate_sensor_dropout(
        self,
        dataset: RobustnessDataset,
        clean_accuracy: float,
    ) -> list[PerturbationMetrics]:
        results = []

        rng = np.random.default_rng(
            self.config.random_seed + 202
        )

        for severity in (
            self.config.dropout_rates
        ):
            mask = (
                rng.random(
                    dataset.states.shape
                ) >= severity
            )

            dropped_states = (
                dataset.states * mask
            )

            results.append(
                self._evaluate_perturbation(
                    perturbation_type="sensor_dropout",
                    severity=severity,
                    states=dropped_states,
                    actions=dataset.actions,
                    targets=dataset.target_next_states,
                    clean_accuracy=clean_accuracy,
                )
            )

        return results

    # --------------------------------------------------------
    # ACTION PERTURBATION
    # --------------------------------------------------------

    def evaluate_action_perturbation(
        self,
        dataset: RobustnessDataset,
        clean_accuracy: float,
    ) -> list[PerturbationMetrics]:
        if self.reference_dynamics is None:
            return []

        results = []

        rng = np.random.default_rng(
            self.config.random_seed + 303
        )

        for severity in (
            self.config.action_noise_levels
        ):
            perturbed_actions = (
                dataset.actions
                + rng.normal(
                    0.0,
                    severity,
                    size=dataset.actions.shape,
                )
            )

            # Ground truth must correspond to
            # the perturbed action.
            perturbed_targets = (
                self._reference_batch(
                    dataset.states,
                    perturbed_actions,
                )
            )

            results.append(
                self._evaluate_perturbation(
                    perturbation_type="action_perturbation",
                    severity=severity,
                    states=dataset.states,
                    actions=perturbed_actions,
                    targets=perturbed_targets,
                    clean_accuracy=clean_accuracy,
                )
            )

        return results

    # --------------------------------------------------------
    # STATE DISTRIBUTION SHIFT
    # --------------------------------------------------------

    def evaluate_state_shift(
        self,
        dataset: RobustnessDataset,
        clean_accuracy: float,
    ) -> list[PerturbationMetrics]:
        if self.reference_dynamics is None:
            return []

        results = []

        rng = np.random.default_rng(
            self.config.random_seed + 404
        )

        direction = rng.normal(
            0.0,
            1.0,
            size=(self.model.state_dim,),
        )

        norm = np.linalg.norm(
            direction
        )

        direction = (
            direction
            / max(
                norm,
                self.config.epsilon,
            )
        )

        for severity in (
            self.config.state_shift_levels
        ):
            shifted_states = (
                dataset.states
                + severity * direction
            )

            shifted_targets = (
                self._reference_batch(
                    shifted_states,
                    dataset.actions,
                )
            )

            results.append(
                self._evaluate_perturbation(
                    perturbation_type="state_shift",
                    severity=severity,
                    states=shifted_states,
                    actions=dataset.actions,
                    targets=shifted_targets,
                    clean_accuracy=clean_accuracy,
                )
            )

        return results

    # --------------------------------------------------------
    # MULTI-STEP ROBUSTNESS
    # --------------------------------------------------------

    def evaluate_trajectory_robustness(
        self,
        dataset: RobustnessDataset,
    ) -> float | None:
        if dataset.trajectory_actions is None:
            return None

        rng = np.random.default_rng(
            self.config.random_seed + 505
        )

        # Noisy initial observation, clean
        # reference trajectory.
        noisy_initial_states = (
            dataset.states
            + rng.normal(
                0.0,
                0.05,
                size=dataset.states.shape,
            )
        )

        predictions = []

        horizon = (
            dataset.trajectory_actions.shape[1]
        )

        for sample_index in range(
            len(dataset.states)
        ):
            state = (
                noisy_initial_states[
                    sample_index
                ].copy()
            )

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

        predicted_trajectories = np.asarray(
            predictions,
            dtype=np.float64,
        )

        rmse = calculate_rmse(
            predicted_trajectories,
            dataset.trajectory_targets,
        )

        return error_to_score(
            rmse,
            self.config.trajectory_error_scale,
        )

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: RobustnessDataset,
    ) -> RobustnessResult:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        clean_predictions, _ = (
            self._predict_batch(
                dataset.states,
                dataset.actions,
            )
        )

        clean_rmse = calculate_rmse(
            clean_predictions,
            dataset.target_next_states,
        )

        clean_accuracy = error_to_score(
            clean_rmse,
            self.config.prediction_error_scale,
        )

        perturbations = []

        perturbations.extend(
            self.evaluate_gaussian_noise(
                dataset,
                clean_accuracy,
            )
        )

        perturbations.extend(
            self.evaluate_sensor_dropout(
                dataset,
                clean_accuracy,
            )
        )

        perturbations.extend(
            self.evaluate_action_perturbation(
                dataset,
                clean_accuracy,
            )
        )

        perturbations.extend(
            self.evaluate_state_shift(
                dataset,
                clean_accuracy,
            )
        )

        def group_score(
            name: str,
        ) -> float | None:
            return safe_mean(
                [
                    item.prediction_accuracy
                    for item in perturbations
                    if item.perturbation_type == name
                ]
            )

        noise_score = group_score(
            "gaussian_noise"
        )

        dropout_score = group_score(
            "sensor_dropout"
        )

        action_score = group_score(
            "action_perturbation"
        )

        shift_score = group_score(
            "state_shift"
        )

        trajectory_score = (
            self.evaluate_trajectory_robustness(
                dataset
            )
        )

        worst_case_accuracy = min(
            [
                clean_accuracy,
                *[
                    item.prediction_accuracy
                    for item in perturbations
                ],
            ]
        )

        max_degradation = max(
            [
                0.0,
                *[
                    item.accuracy_degradation
                    for item in perturbations
                ],
            ]
        )

        scores: dict[str, float | None] = {
            "clean_accuracy": clean_accuracy,
            "noise_robustness": noise_score,
            "dropout_robustness": dropout_score,
            "action_robustness": action_score,
            "shift_robustness": shift_score,
            "trajectory_robustness": trajectory_score,
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
            raise RobustnessValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = RobustnessMetrics(
            clean_accuracy=clean_accuracy,
            noise_robustness=noise_score,
            dropout_robustness=dropout_score,
            action_robustness=action_score,
            shift_robustness=shift_score,
            trajectory_robustness=trajectory_score,
            worst_case_accuracy=worst_case_accuracy,
            max_accuracy_degradation=max_degradation,
            overall_score=float(overall_score),
        )

        result = RobustnessResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            sample_count=len(dataset.states),
            metrics=metrics,
            perturbation_metrics=perturbations,
            passed=(
                overall_score
                >= self.config.pass_threshold
                and max_degradation
                <= self.config.max_accuracy_degradation
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            details={
                "weights": active_weights,
                "clean_rmse": clean_rmse,
                "pass_threshold": (
                    self.config.pass_threshold
                ),
                "max_allowed_degradation": (
                    self.config.max_accuracy_degradation
                ),
                "reference_dynamics_available": (
                    self.reference_dynamics is not None
                ),
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "interpretation": (
                    "Robustness under configured "
                    "observation and dynamics perturbations"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Robustness evaluation: "
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
    ) -> RobustnessResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise RobustnessError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[RobustnessResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[RobustnessResult],
    ) -> list[RobustnessResult]:
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
        results: list[RobustnessResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise RobustnessValidationError(
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
                "passed": result.passed,
                "created_at": result.created_at,
            }

            row.update(
                asdict(result.metrics)
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

def create_robustness_evaluator(
    model: WorldModelProtocol,
    *,
    config: RobustnessConfig | None = None,
    reference_dynamics: (
        ReferenceDynamicsProtocol | None
    ) = None,
) -> RobustnessEvaluator:
    return RobustnessEvaluator(
        model=model,
        config=config,
        reference_dynamics=reference_dynamics,
    )


# ============================================================
# DEMONSTRATION WORLD MODEL
# ============================================================

class DemoWorldModel:
    name = "demo_robustness_world_model"

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
        )


# ============================================================
# REFERENCE DYNAMICS
# ============================================================

def demo_reference_dynamics(
    state: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    return state + 0.5 * action


# ============================================================
# SYNTHETIC DATASET
# ============================================================

def generate_synthetic_robustness_dataset(
    *,
    sample_count: int = 128,
    horizon: int = 8,
    seed: int = 42,
) -> RobustnessDataset:
    if sample_count < 1 or horizon < 1:
        raise ValueError(
            "Sample count and horizon must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    states = rng.normal(
        0.0,
        1.0,
        size=(sample_count, 4),
    )

    actions = rng.uniform(
        -1.0,
        1.0,
        size=(sample_count, 4),
    )

    targets = (
        states + 0.5 * actions
    )

    trajectory_actions = rng.uniform(
        -1.0,
        1.0,
        size=(
            sample_count,
            horizon,
            4,
        ),
    )

    trajectory_targets = np.empty(
        (
            sample_count,
            horizon,
            4,
        ),
        dtype=np.float64,
    )

    current_states = states.copy()

    for step in range(horizon):
        current_states = (
            current_states
            + 0.5 * trajectory_actions[:, step, :]
        )

        trajectory_targets[:, step, :] = (
            current_states
        )

    return RobustnessDataset(
        states=states,
        actions=actions,
        target_next_states=targets,
        trajectory_actions=trajectory_actions,
        trajectory_targets=trajectory_targets,
        metadata={
            "source": "synthetic_linear_dynamics",
            "seed": seed,
        },
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    dataset = generate_synthetic_robustness_dataset(
        sample_count=128,
        horizon=8,
        seed=42,
    )

    model = DemoWorldModel()

    evaluator = create_robustness_evaluator(
        model=model,
        reference_dynamics=demo_reference_dynamics,
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — ROBUSTNESS EVALUATION")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/robustness.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/robustness.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
