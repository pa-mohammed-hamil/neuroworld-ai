
"""
NeuroWorld AI
evaluation/generalization.py

Neural World Model Generalization Evaluation.

Features:
- In-distribution (ID) evaluation
- Out-of-distribution (OOD) evaluation
- Unseen state generalization
- Unseen action generalization
- Action magnitude extrapolation
- Multi-step trajectory generalization
- Noisy observation robustness
- Generalization gap measurement
- Per-domain evaluation
- Uncertainty-error association
- Model comparison
- JSON / CSV report export
- Synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.generalization

Model interface:
    model.name
    model.state_dim
    model.action_dim
    model.predict_transition(state, action)

Expected transition:
    transition.next_state
    transition.uncertainty (optional)

IMPORTANT:
    A meaningful generalization benchmark requires
    held-out environments, states, actions, and
    independently generated reference outcomes.

    Synthetic demonstrations are implementation tests,
    not evidence of real-world generalization.
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
    "neuroworld.evaluation.generalization"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class GeneralizationError(Exception):
    pass


class GeneralizationValidationError(
    GeneralizationError
):
    pass


class GeneralizationModelError(
    GeneralizationError
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
class GeneralizationConfig:
    max_samples: int = 10000
    max_horizon: int = 256

    prediction_error_scale: float = 1.0
    trajectory_error_scale: float = 1.0

    noise_std: float = 0.05

    pass_threshold: float = 0.70
    max_generalization_gap: float = 0.20

    uncertainty_threshold: float = 1.0

    epsilon: float = 1e-8
    random_seed: int = 42

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "ood_prediction_accuracy": 0.35,
            "ood_trajectory_accuracy": 0.25,
            "robustness": 0.15,
            "gap_score": 0.15,
            "uncertainty_alignment": 0.10,
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

        if (
            self.prediction_error_scale <= 0
            or self.trajectory_error_scale <= 0
        ):
            raise ValueError(
                "Error scales must be positive"
            )

        if self.noise_std < 0:
            raise ValueError(
                "noise_std cannot be negative"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        if self.max_generalization_gap < 0:
            raise ValueError(
                "max_generalization_gap must be nonnegative"
            )

        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError(
                "uncertainty_threshold must be in [0, 1]"
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        allowed = {
            "ood_prediction_accuracy",
            "ood_trajectory_accuracy",
            "robustness",
            "gap_score",
            "uncertainty_alignment",
        }

        if not self.weights or set(self.weights) - allowed:
            raise ValueError(
                "Invalid metric weights"
            )

        if (
            any(
                not math.isfinite(value) or value < 0
                for value in self.weights.values()
            )
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Metric weights must have a positive total"
            )


# ============================================================
# DATASET
# ============================================================

@dataclass
class GeneralizationDataset:
    """
    Dataset for one evaluation domain.

    states:
        (N, state_dim)

    actions:
        (N, action_dim)

    next_states:
        (N, state_dim)

    trajectory_actions:
        Optional (N, H, action_dim)

    trajectory_targets:
        Optional (N, H, state_dim)

    Domain examples:
        in_distribution
        unseen_states
        unseen_actions
        unseen_environment
        action_extrapolation
    """

    name: str
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray

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
        if not self.name.strip():
            raise GeneralizationValidationError(
                "Dataset name cannot be empty"
            )

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
                self.next_states,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:
            raise GeneralizationValidationError(
                "Dataset must contain numeric arrays"
            ) from exc

        if states.ndim != 2:
            raise GeneralizationValidationError(
                "states must be 2D"
            )

        n = states.shape[0]

        if not 1 <= n <= max_samples:
            raise GeneralizationValidationError(
                "Invalid sample count"
            )

        if states.shape != (n, state_dim):
            raise GeneralizationValidationError(
                "Invalid state dimensions"
            )

        if actions.shape != (n, action_dim):
            raise GeneralizationValidationError(
                "Invalid action dimensions"
            )

        if targets.shape != (n, state_dim):
            raise GeneralizationValidationError(
                "Invalid target dimensions"
            )

        if not all(
            np.all(np.isfinite(array))
            for array in (
                states,
                actions,
                targets,
            )
        ):
            raise GeneralizationValidationError(
                "Dataset contains non-finite values"
            )

        self.states = states.copy()
        self.actions = actions.copy()
        self.next_states = targets.copy()

        has_actions = (
            self.trajectory_actions is not None
        )

        has_targets = (
            self.trajectory_targets is not None
        )

        if has_actions != has_targets:
            raise GeneralizationValidationError(
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
                raise GeneralizationValidationError(
                    "trajectory_actions must be 3D"
                )

            horizon = trajectory_actions.shape[1]

            if not 1 <= horizon <= max_horizon:
                raise GeneralizationValidationError(
                    "Invalid trajectory horizon"
                )

            if trajectory_actions.shape != (
                n,
                horizon,
                action_dim,
            ):
                raise GeneralizationValidationError(
                    "Invalid trajectory_actions shape"
                )

            if trajectory_targets.shape != (
                n,
                horizon,
                state_dim,
            ):
                raise GeneralizationValidationError(
                    "Invalid trajectory_targets shape"
                )

            if not (
                np.all(np.isfinite(trajectory_actions))
                and np.all(np.isfinite(trajectory_targets))
            ):
                raise GeneralizationValidationError(
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
class DomainMetrics:
    domain_name: str
    sample_count: int

    prediction_rmse: float
    prediction_mae: float
    prediction_accuracy: float

    trajectory_rmse: float | None
    trajectory_accuracy: float | None

    mean_uncertainty: float
    uncertainty_acceptance_rate: float
    uncertainty_error_correlation: float

    robustness_score: float
    robustness_rmse: float


@dataclass
class GeneralizationMetrics:
    id_prediction_accuracy: float
    ood_prediction_accuracy: float

    ood_trajectory_accuracy: float | None

    robustness: float
    gap_score: float

    uncertainty_alignment: float

    generalization_gap: float
    overall_score: float


@dataclass
class GeneralizationResult:
    evaluation_id: str
    model_name: str

    id_domain: str
    ood_domains: list[str]

    metrics: GeneralizationMetrics
    domain_metrics: list[DomainMetrics]

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
        raise GeneralizationValidationError(
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
        raise GeneralizationValidationError(
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

    if (
        x.shape != y.shape
        or len(x) < 2
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
# GENERALIZATION EVALUATOR
# ============================================================

class GeneralizationEvaluator:
    """
    Evaluate performance across known and
    previously unseen data distributions.

    The evaluator does not infer whether a
    dataset is genuinely out-of-distribution.

    OOD status must be established by the
    dataset collection and split methodology.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: GeneralizationConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or GeneralizationConfig()
        )

        self._results: dict[
            str,
            GeneralizationResult,
        ] = {}

    # --------------------------------------------------------
    # PREDICTION
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
            raise GeneralizationModelError(
                f"Model prediction failed: {exc}"
            ) from exc

        if next_state.shape != (
            self.model.state_dim,
        ):
            raise GeneralizationModelError(
                "Invalid predicted state shape"
            )

        if not np.all(np.isfinite(next_state)):
            raise GeneralizationModelError(
                "Predicted state contains invalid values"
            )

        if (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise GeneralizationModelError(
                "Invalid uncertainty value"
            )

        return next_state, uncertainty

    # --------------------------------------------------------
    # DOMAIN EVALUATION
    # --------------------------------------------------------

    def evaluate_domain(
        self,
        dataset: GeneralizationDataset,
    ) -> DomainMetrics:
        dataset.validate(
            self.model.state_dim,
            self.model.action_dim,
            max_samples=self.config.max_samples,
            max_horizon=self.config.max_horizon,
        )

        predictions = []
        uncertainties = []

        for state, action in zip(
            dataset.states,
            dataset.actions,
        ):
            predicted, uncertainty = self._predict(
                state,
                action,
            )

            predictions.append(predicted)
            uncertainties.append(uncertainty)

        predicted_array = np.stack(
            predictions
        )

        uncertainty_array = np.asarray(
            uncertainties,
            dtype=np.float64,
        )

        prediction_rmse = calculate_rmse(
            predicted_array,
            dataset.next_states,
        )

        prediction_mae = calculate_mae(
            predicted_array,
            dataset.next_states,
        )

        prediction_accuracy = error_to_score(
            prediction_rmse,
            self.config.prediction_error_scale,
        )

        per_sample_errors = np.linalg.norm(
            predicted_array - dataset.next_states,
            axis=1,
        )

        uncertainty_correlation = safe_correlation(
            uncertainty_array,
            per_sample_errors,
            self.config.epsilon,
        )

        acceptance_rate = float(
            np.mean(
                uncertainty_array
                <= self.config.uncertainty_threshold
            )
        )

        # ----------------------------------------------------
        # Trajectory evaluation
        # ----------------------------------------------------

        trajectory_rmse = None
        trajectory_accuracy = None

        if dataset.trajectory_actions is not None:
            horizon = (
                dataset.trajectory_actions.shape[1]
            )

            predicted_trajectories = []

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

                predicted_trajectories.append(
                    predicted_steps
                )

            trajectory_predictions = np.asarray(
                predicted_trajectories,
                dtype=np.float64,
            )

            trajectory_rmse = calculate_rmse(
                trajectory_predictions,
                dataset.trajectory_targets,
            )

            trajectory_accuracy = error_to_score(
                trajectory_rmse,
                self.config.trajectory_error_scale,
            )

        # ----------------------------------------------------
        # Noise robustness
        # ----------------------------------------------------

        rng = np.random.default_rng(
            self.config.random_seed
        )

        noisy_predictions = []

        for state, action in zip(
            dataset.states,
            dataset.actions,
        ):
            noisy_state = (
                state
                + rng.normal(
                    0.0,
                    self.config.noise_std,
                    size=state.shape,
                )
            )

            predicted, _ = self._predict(
                noisy_state,
                action,
            )

            noisy_predictions.append(
                predicted
            )

        noisy_predictions = np.stack(
            noisy_predictions
        )

        # Measures stability relative to clean
        # predictions, not ground-truth accuracy.
        robustness_rmse = calculate_rmse(
            noisy_predictions,
            predicted_array,
        )

        robustness_score = error_to_score(
            robustness_rmse,
            self.config.prediction_error_scale,
        )

        return DomainMetrics(
            domain_name=dataset.name,
            sample_count=len(dataset.states),
            prediction_rmse=prediction_rmse,
            prediction_mae=prediction_mae,
            prediction_accuracy=prediction_accuracy,
            trajectory_rmse=trajectory_rmse,
            trajectory_accuracy=trajectory_accuracy,
            mean_uncertainty=float(
                np.mean(uncertainty_array)
            ),
            uncertainty_acceptance_rate=acceptance_rate,
            uncertainty_error_correlation=(
                uncertainty_correlation
            ),
            robustness_score=robustness_score,
            robustness_rmse=robustness_rmse,
        )

    # --------------------------------------------------------
    # COMPLETE GENERALIZATION EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        in_distribution: GeneralizationDataset,
        out_of_distribution: list[
            GeneralizationDataset
        ],
    ) -> GeneralizationResult:
        if not out_of_distribution:
            raise GeneralizationValidationError(
                "At least one OOD dataset is required"
            )

        names = [
            in_distribution.name,
            *[
                dataset.name
                for dataset in out_of_distribution
            ],
        ]

        if len(names) != len(set(names)):
            raise GeneralizationValidationError(
                "Dataset names must be unique"
            )

        id_metrics = self.evaluate_domain(
            in_distribution
        )

        ood_metrics = [
            self.evaluate_domain(dataset)
            for dataset in out_of_distribution
        ]

        ood_prediction_accuracy = float(
            np.mean(
                [
                    metric.prediction_accuracy
                    for metric in ood_metrics
                ]
            )
        )

        trajectory_values = [
            metric.trajectory_accuracy
            for metric in ood_metrics
            if metric.trajectory_accuracy is not None
        ]

        ood_trajectory_accuracy = (
            float(np.mean(trajectory_values))
            if trajectory_values
            else None
        )

        robustness = float(
            np.mean(
                [
                    metric.robustness_score
                    for metric in ood_metrics
                ]
            )
        )

        correlations = [
            metric.uncertainty_error_correlation
            for metric in ood_metrics
        ]

        mean_correlation = float(
            np.mean(correlations)
        )

        uncertainty_alignment = float(
            np.clip(
                (mean_correlation + 1.0) / 2.0,
                0.0,
                1.0,
            )
        )

        # Positive gap means OOD performance
        # is worse than ID performance.
        generalization_gap = max(
            0.0,
            id_metrics.prediction_accuracy
            - ood_prediction_accuracy,
        )

        gap_score = error_to_score(
            generalization_gap,
            max(
                self.config.max_generalization_gap,
                self.config.epsilon,
            ),
        )

        scores: dict[str, float | None] = {
            "ood_prediction_accuracy": (
                ood_prediction_accuracy
            ),
            "ood_trajectory_accuracy": (
                ood_trajectory_accuracy
            ),
            "robustness": robustness,
            "gap_score": gap_score,
            "uncertainty_alignment": (
                uncertainty_alignment
            ),
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
            raise GeneralizationValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = GeneralizationMetrics(
            id_prediction_accuracy=(
                id_metrics.prediction_accuracy
            ),
            ood_prediction_accuracy=(
                ood_prediction_accuracy
            ),
            ood_trajectory_accuracy=(
                ood_trajectory_accuracy
            ),
            robustness=robustness,
            gap_score=gap_score,
            uncertainty_alignment=(
                uncertainty_alignment
            ),
            generalization_gap=(
                generalization_gap
            ),
            overall_score=float(overall_score),
        )

        result = GeneralizationResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            id_domain=in_distribution.name,
            ood_domains=[
                dataset.name
                for dataset in out_of_distribution
            ],
            metrics=metrics,
            domain_metrics=[
                id_metrics,
                *ood_metrics,
            ],
            passed=(
                overall_score
                >= self.config.pass_threshold
                and generalization_gap
                <= self.config.max_generalization_gap
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            details={
                "weights": active_weights,
                "pass_threshold": (
                    self.config.pass_threshold
                ),
                "max_generalization_gap": (
                    self.config.max_generalization_gap
                ),
                "random_seed": (
                    self.config.random_seed
                ),
                "dataset_sources": {
                    dataset.name: dataset.metadata.get(
                        "source",
                        "unspecified",
                    )
                    for dataset in [
                        in_distribution,
                        *out_of_distribution,
                    ]
                },
                "interpretation": (
                    "Comparison of predictive performance "
                    "across supplied ID and OOD datasets"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Generalization evaluation: "
            "model=%s score=%.4f gap=%.4f passed=%s",
            self.model.name,
            overall_score,
            generalization_gap,
            result.passed,
        )

        return result

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> GeneralizationResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise GeneralizationError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[GeneralizationResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[GeneralizationResult],
    ) -> list[GeneralizationResult]:
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
        results: list[GeneralizationResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise GeneralizationValidationError(
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
                "id_domain": result.id_domain,
                "ood_domains": ",".join(
                    result.ood_domains
                ),
                "passed": result.passed,
                "created_at": result.created_at,
            }

            row.update(
                asdict(result.metrics)
            )

            rows.append(row)

        with path.open(
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

        return path


# ============================================================
# FACTORY
# ============================================================

def create_generalization_evaluator(
    model: WorldModelProtocol,
    *,
    config: GeneralizationConfig | None = None,
) -> GeneralizationEvaluator:
    return GeneralizationEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DATASET GENERATOR
# ============================================================

def generate_synthetic_generalization_datasets(
    *,
    state_dim: int = 4,
    action_dim: int = 4,
    sample_count: int = 128,
    horizon: int = 8,
    seed: int = 42,
    dynamics_scale: float = 0.5,
) -> tuple[
    GeneralizationDataset,
    list[GeneralizationDataset],
]:
    """
    Generate ID and OOD datasets using the
    same known linear dynamics.

    OOD conditions:
        - Shifted initial states
        - Larger action magnitudes
        - Higher-variance states

    These synthetic shifts do not represent
    real environmental generalization.
    """

    if min(
        state_dim,
        action_dim,
        sample_count,
        horizon,
    ) < 1:
        raise GeneralizationValidationError(
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

    def make_dataset(
        name: str,
        *,
        state_mean: float,
        state_std: float,
        action_range: float,
    ) -> GeneralizationDataset:
        states = rng.normal(
            state_mean,
            state_std,
            size=(sample_count, state_dim),
        )

        actions = rng.uniform(
            -action_range,
            action_range,
            size=(sample_count, action_dim),
        )

        next_states = (
            states
            + actions @ transition_matrix.T
        )

        trajectory_actions = rng.uniform(
            -action_range,
            action_range,
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

            trajectory_targets[:, step, :] = (
                current
            )

        return GeneralizationDataset(
            name=name,
            states=states,
            actions=actions,
            next_states=next_states,
            trajectory_actions=trajectory_actions,
            trajectory_targets=trajectory_targets,
            metadata={
                "source": "synthetic_linear_simulator",
                "state_mean": state_mean,
                "state_std": state_std,
                "action_range": action_range,
            },
        )

    in_distribution = make_dataset(
        "in_distribution",
        state_mean=0.0,
        state_std=1.0,
        action_range=1.0,
    )

    out_of_distribution = [
        make_dataset(
            "unseen_states",
            state_mean=3.0,
            state_std=1.0,
            action_range=1.0,
        ),
        make_dataset(
            "action_extrapolation",
            state_mean=0.0,
            state_std=1.0,
            action_range=3.0,
        ),
        make_dataset(
            "high_variance_states",
            state_mean=0.0,
            state_std=3.0,
            action_range=1.0,
        ),
    ]

    return (
        in_distribution,
        out_of_distribution,
    )


# ============================================================
# DEMONSTRATION MODEL
# ============================================================

class DemoWorldModel:
    name = "demo_generalization_world_model"

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
    in_distribution, out_of_distribution = (
        generate_synthetic_generalization_datasets(
            state_dim=4,
            action_dim=4,
            sample_count=128,
            horizon=8,
            seed=42,
        )
    )

    model = DemoWorldModel()

    evaluator = create_generalization_evaluator(
        model
    )

    result = evaluator.evaluate(
        in_distribution,
        out_of_distribution,
    )

    print("=" * 65)
    print("NEUROWORLD AI — GENERALIZATION EVALUATION")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/generalization.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/generalization.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
