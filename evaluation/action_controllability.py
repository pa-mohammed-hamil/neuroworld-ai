
"""
NeuroWorld AI
evaluation/action_controllability.py

Action Controllability Evaluation.

Evaluates whether actions produce meaningful,
consistent, and goal-directed changes in the
predicted states of a neural world model.

Features:
- Action sensitivity
- Action influence measurement
- Action-to-state responsiveness
- Direction alignment
- Goal-directed controllability
- Counterfactual action comparison
- Action consistency
- Action diversity
- Controllability score
- Batch evaluation
- JSON report export

Dependencies:
    pip install numpy

Model interface:
    model.predict_transition(state, action)

Expected transition:
    transition.next_state
    transition.uncertainty (optional)

This module evaluates simulated/model-predicted
transitions. It does not control physical systems.
"""

from __future__ import annotations

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
    "neuroworld.evaluation.action_controllability"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ControllabilityError(Exception):
    pass


class ControllabilityValidationError(
    ControllabilityError
):
    pass


class ControllabilityModelError(
    ControllabilityError
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
class ControllabilityConfig:
    sensitivity_threshold: float = 0.01

    consistency_tolerance: float = 0.1

    goal_success_threshold: float = 0.25

    uncertainty_threshold: float = 1.0

    action_scale: float = 1.0

    epsilon: float = 1e-8

    max_samples: int = 10000

    random_seed: int = 42

    def __post_init__(self):
        if self.sensitivity_threshold < 0:
            raise ValueError(
                "sensitivity_threshold must be nonnegative"
            )

        if self.consistency_tolerance < 0:
            raise ValueError(
                "consistency_tolerance must be nonnegative"
            )

        if self.goal_success_threshold < 0:
            raise ValueError(
                "goal_success_threshold must be nonnegative"
            )

        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError(
                "uncertainty_threshold must be in [0, 1]"
            )

        if self.action_scale <= 0:
            raise ValueError(
                "action_scale must be positive"
            )

        if self.max_samples < 1:
            raise ValueError(
                "max_samples must be positive"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class ActionControllabilityMetrics:
    action_sensitivity: float

    action_influence: float

    action_responsiveness: float

    direction_alignment: float

    goal_controllability: float

    action_consistency: float

    action_diversity: float

    causal_influence: float

    uncertainty_score: float

    controllability_score: float


@dataclass
class ActionControllabilityResult:
    evaluation_id: str

    model_name: str

    sample_count: int

    metrics: ActionControllabilityMetrics

    passed: bool

    created_at: str

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_matrix(
    values: Any,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ControllabilityValidationError(
            f"{name} must be numeric"
        ) from exc

    if array.ndim != 2:
        raise ControllabilityValidationError(
            f"{name} must be a 2D array"
        )

    if array.shape[1] != dimension:
        raise ControllabilityValidationError(
            f"{name} must have {dimension} features"
        )

    if array.shape[0] < 1:
        raise ControllabilityValidationError(
            f"{name} cannot be empty"
        )

    if not np.all(np.isfinite(array)):
        raise ControllabilityValidationError(
            f"{name} contains non-finite values"
        )

    return array.copy()


def safe_cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator <= epsilon:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b) / denominator,
            -1.0,
            1.0,
        )
    )


def normalize_score(
    value: float,
    scale: float = 1.0,
) -> float:
    if not math.isfinite(value):
        return 0.0

    return float(
        np.clip(
            value / max(scale, 1e-8),
            0.0,
            1.0,
        )
    )


# ============================================================
# ACTION CONTROLLABILITY EVALUATOR
# ============================================================

class ActionControllabilityEvaluator:
    """
    Evaluate action-conditioned world-model behavior.

    Tests whether:
        1. Actions affect predicted states.
        2. Different actions produce different effects.
        3. Predictions are consistent.
        4. Actions move states toward goals.
        5. Counterfactual actions change outcomes.

    Scores are heuristic diagnostic indicators.
    They do not establish real-world controllability.
    """

    def __init__(
        self,
        model: WorldModelProtocol,
        config: ControllabilityConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or ControllabilityConfig()
        )

        self.rng = np.random.default_rng(
            self.config.random_seed
        )

        self._results: dict[
            str,
            ActionControllabilityResult,
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
            raise ControllabilityModelError(
                f"World model prediction failed: {exc}"
            ) from exc

        if next_state.shape != (
            self.model.state_dim,
        ):
            raise ControllabilityModelError(
                "Invalid predicted state dimension"
            )

        if not np.all(np.isfinite(next_state)):
            raise ControllabilityModelError(
                "Model produced invalid state values"
            )

        if (
            not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 1
        ):
            raise ControllabilityModelError(
                "Model produced invalid uncertainty"
            )

        return next_state, uncertainty

    # --------------------------------------------------------
    # ACTION SENSITIVITY
    # --------------------------------------------------------

    def evaluate_action_sensitivity(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Measures action-induced state changes
        relative to the same state's zero-action
        baseline.
        """

        zero_action = np.zeros(
            self.model.action_dim,
            dtype=np.float64,
        )

        scores = []

        for state, action in zip(
            states,
            actions,
        ):
            predicted, _ = self._predict(
                state,
                action,
            )

            baseline, _ = self._predict(
                state,
                zero_action,
            )

            difference = float(
                np.linalg.norm(
                    predicted - baseline
                )
            )

            scores.append(
                normalize_score(
                    difference,
                    self.config.action_scale,
                )
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # ACTION INFLUENCE
    # --------------------------------------------------------

    def evaluate_action_influence(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Measures the strength of action effects
        relative to action magnitude.
        """

        scores = []

        zero_action = np.zeros(
            self.model.action_dim
        )

        for state, action in zip(
            states,
            actions,
        ):
            predicted, _ = self._predict(
                state,
                action,
            )

            baseline, _ = self._predict(
                state,
                zero_action,
            )

            effect = np.linalg.norm(
                predicted - baseline
            )

            action_norm = np.linalg.norm(
                action
            )

            influence = (
                effect
                / (
                    action_norm
                    + self.config.epsilon
                )
            )

            scores.append(
                normalize_score(influence)
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # ACTION RESPONSIVENESS
    # --------------------------------------------------------

    def evaluate_responsiveness(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Fraction of nonzero actions producing
        a measurable effect relative to zero action.
        """

        zero_action = np.zeros(
            self.model.action_dim
        )

        responsive = []

        for state, action in zip(
            states,
            actions,
        ):
            if np.linalg.norm(action) <= self.config.epsilon:
                continue

            predicted, _ = self._predict(
                state,
                action,
            )

            baseline, _ = self._predict(
                state,
                zero_action,
            )

            change = np.linalg.norm(
                predicted - baseline
            )

            responsive.append(
                float(
                    change
                    >= self.config.sensitivity_threshold
                )
            )

        return (
            float(np.mean(responsive))
            if responsive
            else 0.0
        )

    # --------------------------------------------------------
    # DIRECTION ALIGNMENT
    # --------------------------------------------------------

    def evaluate_direction_alignment(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        goals: np.ndarray,
    ) -> float:
        """
        Measures whether action-conditioned
        predictions move toward specified goals.
        """

        scores = []

        for state, action, goal in zip(
            states,
            actions,
            goals,
        ):
            predicted, _ = self._predict(
                state,
                action,
            )

            movement = predicted - state
            goal_direction = goal - state

            alignment = safe_cosine_similarity(
                movement,
                goal_direction,
                self.config.epsilon,
            )

            scores.append(
                (alignment + 1.0) / 2.0
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # GOAL CONTROLLABILITY
    # --------------------------------------------------------

    def evaluate_goal_controllability(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        goals: np.ndarray,
    ) -> float:
        """
        Measures how often an action reduces
        distance to the goal.

        Success also requires the improvement
        to exceed the configured threshold.
        """

        scores = []

        for state, action, goal in zip(
            states,
            actions,
            goals,
        ):
            predicted, _ = self._predict(
                state,
                action,
            )

            before = float(
                np.linalg.norm(
                    goal - state
                )
            )

            after = float(
                np.linalg.norm(
                    goal - predicted
                )
            )

            improvement = before - after

            scores.append(
                float(
                    improvement
                    >= self.config.goal_success_threshold
                )
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # ACTION CONSISTENCY
    # --------------------------------------------------------

    def evaluate_action_consistency(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        repeats: int = 3,
    ) -> float:
        """
        Repeats identical state/action predictions.

        Lower output variance means higher
        repeatability.

        This does not establish prediction accuracy.
        """

        if repeats < 2:
            raise ControllabilityValidationError(
                "repeats must be at least 2"
            )

        scores = []

        for state, action in zip(
            states,
            actions,
        ):
            predictions = []

            for _ in range(repeats):
                predicted, _ = self._predict(
                    state,
                    action,
                )

                predictions.append(
                    predicted
                )

            predictions = np.stack(
                predictions
            )

            variance = float(
                np.mean(
                    np.var(
                        predictions,
                        axis=0,
                    )
                )
            )

            score = math.exp(
                -variance
                / max(
                    self.config.consistency_tolerance,
                    self.config.epsilon,
                )
            )

            scores.append(score)

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # ACTION DIVERSITY
    # --------------------------------------------------------

    def evaluate_action_diversity(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Compares outcomes from the original
        action and an alternative action.

        The alternative is the negated action.
        """

        scores = []

        for state, action in zip(
            states,
            actions,
        ):
            original, _ = self._predict(
                state,
                action,
            )

            alternative, _ = self._predict(
                state,
                -action,
            )

            difference = float(
                np.linalg.norm(
                    original - alternative
                )
            )

            scores.append(
                normalize_score(
                    difference,
                    self.config.action_scale,
                )
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # CAUSAL INFLUENCE
    # --------------------------------------------------------

    def evaluate_causal_influence(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Compares predicted outcomes under
        action intervention versus zero action.

        This is an interventional sensitivity
        proxy, not proof of causal correctness.
        """

        zero_action = np.zeros(
            self.model.action_dim
        )

        scores = []

        for state, action in zip(
            states,
            actions,
        ):
            intervention, _ = self._predict(
                state,
                action,
            )

            control, _ = self._predict(
                state,
                zero_action,
            )

            effect = float(
                np.linalg.norm(
                    intervention - control
                )
            )

            scores.append(
                normalize_score(
                    effect,
                    self.config.action_scale,
                )
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    def evaluate_uncertainty(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> float:
        """
        Fraction of predictions with reported
        uncertainty below the configured threshold.

        This measures reported confidence only,
        not uncertainty calibration.
        """

        scores = []

        for state, action in zip(
            states,
            actions,
        ):
            _, uncertainty = self._predict(
                state,
                action,
            )

            scores.append(
                float(
                    uncertainty
                    <= self.config.uncertainty_threshold
                )
            )

        return float(
            np.mean(scores)
        )

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        states: Any,
        actions: Any,
        goals: Any,
    ) -> ActionControllabilityResult:
        states = validate_matrix(
            states,
            dimension=self.model.state_dim,
            name="states",
        )

        actions = validate_matrix(
            actions,
            dimension=self.model.action_dim,
            name="actions",
        )

        goals = validate_matrix(
            goals,
            dimension=self.model.state_dim,
            name="goals",
        )

        sample_count = len(states)

        if not (
            len(actions)
            == len(goals)
            == sample_count
        ):
            raise ControllabilityValidationError(
                "states, actions and goals must have "
                "the same number of samples"
            )

        if sample_count > self.config.max_samples:
            raise ControllabilityValidationError(
                "Sample count exceeds configured limit"
            )

        sensitivity = (
            self.evaluate_action_sensitivity(
                states,
                actions,
            )
        )

        influence = (
            self.evaluate_action_influence(
                states,
                actions,
            )
        )

        responsiveness = (
            self.evaluate_responsiveness(
                states,
                actions,
            )
        )

        alignment = (
            self.evaluate_direction_alignment(
                states,
                actions,
                goals,
            )
        )

        goal_score = (
            self.evaluate_goal_controllability(
                states,
                actions,
                goals,
            )
        )

        consistency = (
            self.evaluate_action_consistency(
                states,
                actions,
            )
        )

        diversity = (
            self.evaluate_action_diversity(
                states,
                actions,
            )
        )

        causal = (
            self.evaluate_causal_influence(
                states,
                actions,
            )
        )

        uncertainty = (
            self.evaluate_uncertainty(
                states,
                actions,
            )
        )

        metrics = ActionControllabilityMetrics(
            action_sensitivity=sensitivity,
            action_influence=influence,
            action_responsiveness=responsiveness,
            direction_alignment=alignment,
            goal_controllability=goal_score,
            action_consistency=consistency,
            action_diversity=diversity,
            causal_influence=causal,
            uncertainty_score=uncertainty,
            controllability_score=0.0,
        )

        # Heuristic weighted composite.
        weights = {
            "action_sensitivity": 0.10,
            "action_influence": 0.10,
            "action_responsiveness": 0.15,
            "direction_alignment": 0.15,
            "goal_controllability": 0.20,
            "action_consistency": 0.10,
            "action_diversity": 0.05,
            "causal_influence": 0.10,
            "uncertainty_score": 0.05,
        }

        composite = sum(
            getattr(metrics, name) * weight
            for name, weight in weights.items()
        )

        metrics.controllability_score = float(
            np.clip(
                composite,
                0.0,
                1.0,
            )
        )

        passed = (
            metrics.controllability_score >= 0.70
            and metrics.goal_controllability >= 0.50
            and metrics.action_responsiveness >= 0.50
        )

        evaluation_id = uuid.uuid4().hex

        result = ActionControllabilityResult(
            evaluation_id=evaluation_id,
            model_name=self.model.name,
            sample_count=sample_count,
            metrics=metrics,
            passed=passed,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            details={
                "weights": weights,
                "thresholds": {
                    "composite": 0.70,
                    "goal_controllability": 0.50,
                    "responsiveness": 0.50,
                },
                "interpretation": (
                    "Heuristic model-based controllability "
                    "assessment; not physical validation"
                ),
            },
        )

        self._results[
            evaluation_id
        ] = result

        logger.info(
            "Action controllability evaluated: "
            "model=%s score=%.4f passed=%s",
            self.model.name,
            metrics.controllability_score,
            passed,
        )

        return result

    # --------------------------------------------------------
    # RESULT RETRIEVAL
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> ActionControllabilityResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise ControllabilityError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[ActionControllabilityResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # REPORT EXPORT
    # --------------------------------------------------------

    def export_report(
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


# ============================================================
# FACTORY
# ============================================================

def create_action_controllability_evaluator(
    model: WorldModelProtocol,
    *,
    config: ControllabilityConfig | None = None,
) -> ActionControllabilityEvaluator:
    return ActionControllabilityEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

class DemoWorldModel:
    """
    Synthetic linear world model.

    next_state = state + action * 0.5

    This is not a trained neural world model.
    """

    name = "demo_controllable_world_model"
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
    model = DemoWorldModel()

    evaluator = (
        create_action_controllability_evaluator(
            model
        )
    )

    states = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
        ]
    )

    actions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    goals = states + actions

    result = evaluator.evaluate(
        states=states,
        actions=actions,
        goals=goals,
    )

    print("=" * 65)
    print("NEUROWORLD AI — ACTION CONTROLLABILITY")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    report_path = evaluator.export_report(
        result.evaluation_id,
        "experiments/reports/action_controllability.json",
    )

    print(
        "\nReport saved:",
        report_path,
    )


if __name__ == "__main__":
    main()
