
"""
NeuroWorld AI
evaluation/object_permanence.py

Object Permanence Evaluation.

Evaluates whether an object-tracking or neural
world-model prediction system maintains object
identity and estimates object positions during
temporary occlusion.

Features:
- Object identity preservation
- Hidden-object position prediction
- Occlusion recovery accuracy
- Reappearance matching
- Visible-object tracking accuracy
- Trajectory RMSE
- False disappearance / hallucination detection
- Confidence calibration proxy
- Per-object and aggregate metrics
- Multiple-model comparison
- JSON and CSV export
- Reproducible synthetic demonstration

Dependencies:
    pip install numpy

Run:
    python -m evaluation.object_permanence

Model interface:
    model.name
    model.predict_sequence(observations)

Expected output:
    List of predicted object states for every frame.

    Each predicted object contains:
        object_id
        position
        confidence
        exists

Important:
    Ground-truth object identities and positions
    must be independently annotated or generated.

    The synthetic demonstration tests evaluation
    functionality, not real-world object permanence.
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
    "neuroworld.evaluation.object_permanence"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectPermanenceError(Exception):
    pass


class ObjectPermanenceValidationError(
    ObjectPermanenceError
):
    pass


class ObjectPermanenceModelError(
    ObjectPermanenceError
):
    pass


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ObjectState:
    object_id: str
    position: list[float]

    visible: bool = True
    exists: bool = True

    confidence: float = 1.0

    def __post_init__(self):
        if not self.object_id:
            raise ObjectPermanenceValidationError(
                "object_id cannot be empty"
            )

        position = np.asarray(
            self.position,
            dtype=np.float64,
        )

        if (
            position.ndim != 1
            or position.size not in (2, 3)
            or not np.all(np.isfinite(position))
        ):
            raise ObjectPermanenceValidationError(
                "position must be finite 2D or 3D coordinates"
            )

        if not 0 <= self.confidence <= 1:
            raise ObjectPermanenceValidationError(
                "confidence must be in [0, 1]"
            )


@dataclass
class ObjectFrame:
    frame_index: int
    objects: list[ObjectState]


@dataclass
class ObjectPermanenceDataset:
    observations: list[ObjectFrame]
    ground_truth: list[ObjectFrame]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.observations:
            raise ObjectPermanenceValidationError(
                "Dataset cannot be empty"
            )

        if len(self.observations) != len(
            self.ground_truth
        ):
            raise ObjectPermanenceValidationError(
                "Observation and ground-truth lengths differ"
            )

        for index, (observation, truth) in enumerate(
            zip(
                self.observations,
                self.ground_truth,
            )
        ):
            if (
                observation.frame_index != index
                or truth.frame_index != index
            ):
                raise ObjectPermanenceValidationError(
                    "Frame indices must be sequential"
                )

            for frame in (observation, truth):
                ids = [
                    obj.object_id
                    for obj in frame.objects
                ]

                if len(ids) != len(set(ids)):
                    raise ObjectPermanenceValidationError(
                        "Duplicate object IDs in frame"
                    )


# ============================================================
# MODEL PROTOCOL
# ============================================================

class ObjectTrackingModelProtocol(Protocol):
    name: str

    def predict_sequence(
        self,
        observations: list[ObjectFrame],
    ) -> list[ObjectFrame]:
        ...


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObjectPermanenceConfig:
    position_tolerance: float = 0.5
    reappearance_tolerance: float = 0.75

    confidence_threshold: float = 0.30

    error_scale: float = 1.0

    max_frames: int = 10000

    pass_threshold: float = 0.70

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "identity_preservation": 0.20,
            "hidden_position_accuracy": 0.20,
            "occlusion_recovery": 0.20,
            "visible_tracking_accuracy": 0.15,
            "trajectory_accuracy": 0.10,
            "existence_accuracy": 0.10,
            "confidence_alignment": 0.05,
        }
    )

    def __post_init__(self):
        if (
            self.position_tolerance <= 0
            or self.reappearance_tolerance <= 0
            or self.error_scale <= 0
        ):
            raise ValueError(
                "Tolerances and scales must be positive"
            )

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                "confidence_threshold must be in [0, 1]"
            )

        if self.max_frames < 1:
            raise ValueError(
                "max_frames must be positive"
            )

        if not 0 <= self.pass_threshold <= 1:
            raise ValueError(
                "pass_threshold must be in [0, 1]"
            )

        allowed = {
            "identity_preservation",
            "hidden_position_accuracy",
            "occlusion_recovery",
            "visible_tracking_accuracy",
            "trajectory_accuracy",
            "existence_accuracy",
            "confidence_alignment",
        }

        if (
            not self.weights
            or set(self.weights) - allowed
            or any(
                not math.isfinite(value) or value < 0
                for value in self.weights.values()
            )
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Invalid metric weights"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class ObjectMetrics:
    object_id: str

    visible_frames: int
    occluded_frames: int

    identity_preservation: float
    hidden_position_accuracy: float | None

    occlusion_recovery: float | None

    visible_tracking_accuracy: float

    trajectory_rmse: float | None

    existence_accuracy: float

    confidence_alignment: float


@dataclass
class ObjectPermanenceMetrics:
    identity_preservation: float
    hidden_position_accuracy: float | None

    occlusion_recovery: float | None

    visible_tracking_accuracy: float
    trajectory_accuracy: float | None

    existence_accuracy: float

    confidence_alignment: float

    overall_score: float


@dataclass
class ObjectPermanenceResult:
    evaluation_id: str
    model_name: str

    frame_count: int
    object_count: int

    metrics: ObjectPermanenceMetrics
    object_metrics: list[ObjectMetrics]

    passed: bool
    created_at: str

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# HELPERS
# ============================================================

def position_distance(
    first: ObjectState,
    second: ObjectState,
) -> float:
    first_position = np.asarray(
        first.position,
        dtype=np.float64,
    )

    second_position = np.asarray(
        second.position,
        dtype=np.float64,
    )

    if first_position.shape != second_position.shape:
        raise ObjectPermanenceValidationError(
            "Object position dimensions differ"
        )

    return float(
        np.linalg.norm(
            first_position - second_position
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


def frame_to_map(
    frame: ObjectFrame,
) -> dict[str, ObjectState]:
    return {
        obj.object_id: obj
        for obj in frame.objects
    }


# ============================================================
# OBJECT PERMANENCE EVALUATOR
# ============================================================

class ObjectPermanenceEvaluator:
    """
    Evaluate object tracking through occlusion.

    Ground truth includes object positions even
    when objects are temporarily not visible.

    A model is rewarded for maintaining correct
    object identities and hidden-state estimates.
    """

    def __init__(
        self,
        model: ObjectTrackingModelProtocol,
        config: ObjectPermanenceConfig | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or ObjectPermanenceConfig()
        )

        self._results: dict[
            str,
            ObjectPermanenceResult,
        ] = {}

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    def _predict(
        self,
        observations: list[ObjectFrame],
    ) -> list[ObjectFrame]:
        try:
            predictions = (
                self.model.predict_sequence(
                    observations
                )
            )
        except Exception as exc:
            raise ObjectPermanenceModelError(
                f"Object prediction failed: {exc}"
            ) from exc

        if len(predictions) != len(observations):
            raise ObjectPermanenceModelError(
                "Prediction frame count mismatch"
            )

        for index, frame in enumerate(predictions):
            if frame.frame_index != index:
                raise ObjectPermanenceModelError(
                    "Invalid prediction frame index"
                )

            ids = [
                obj.object_id
                for obj in frame.objects
            ]

            if len(ids) != len(set(ids)):
                raise ObjectPermanenceModelError(
                    "Duplicate predicted object IDs"
                )

        return predictions

    # --------------------------------------------------------
    # OBJECT EVALUATION
    # --------------------------------------------------------

    def evaluate_object(
        self,
        object_id: str,
        ground_truth: list[ObjectFrame],
        predictions: list[ObjectFrame],
    ) -> ObjectMetrics:
        visible_scores = []
        hidden_scores = []

        identity_scores = []
        existence_scores = []
        confidence_scores = []

        trajectory_errors = []

        visible_frames = 0
        occluded_frames = 0

        recovery_scores = []

        previous_occluded = False

        for truth_frame, prediction_frame in zip(
            ground_truth,
            predictions,
        ):
            truth_map = frame_to_map(
                truth_frame
            )

            prediction_map = frame_to_map(
                prediction_frame
            )

            truth = truth_map.get(
                object_id
            )

            predicted = prediction_map.get(
                object_id
            )

            if truth is None:
                # This object is absent from ground truth.
                existence_scores.append(
                    float(
                        predicted is None
                        or not predicted.exists
                    )
                )

                previous_occluded = False
                continue

            # ------------------------------------------------
            # Existence accuracy
            # ------------------------------------------------

            correct_existence = (
                predicted is not None
                and predicted.exists == truth.exists
            )

            existence_scores.append(
                float(correct_existence)
            )

            if not truth.exists:
                previous_occluded = False
                continue

            # ------------------------------------------------
            # Identity preservation
            # ------------------------------------------------

            identity_scores.append(
                float(
                    predicted is not None
                    and predicted.exists
                )
            )

            if predicted is None or not predicted.exists:
                if truth.visible:
                    visible_frames += 1
                    visible_scores.append(0.0)
                else:
                    occluded_frames += 1
                    hidden_scores.append(0.0)

                if previous_occluded and truth.visible:
                    recovery_scores.append(0.0)

                previous_occluded = not truth.visible
                continue

            distance = position_distance(
                truth,
                predicted,
            )

            trajectory_errors.append(
                distance ** 2
            )

            # Confidence alignment is a simple
            # correctness-event confidence proxy.
            correct_position = float(
                distance <= self.config.position_tolerance
            )

            confidence_scores.append(
                1.0
                - abs(
                    predicted.confidence
                    - correct_position
                )
            )

            # ------------------------------------------------
            # Visible tracking
            # ------------------------------------------------

            if truth.visible:
                visible_frames += 1

                visible_scores.append(
                    correct_position
                )

                if previous_occluded:
                    recovery_scores.append(
                        float(
                            distance
                            <= self.config.reappearance_tolerance
                        )
                    )

                previous_occluded = False

            # ------------------------------------------------
            # Hidden object prediction
            # ------------------------------------------------

            else:
                occluded_frames += 1

                hidden_scores.append(
                    error_to_score(
                        distance,
                        self.config.error_scale,
                    )
                )

                previous_occluded = True

        trajectory_rmse = (
            float(
                np.sqrt(
                    np.mean(trajectory_errors)
                )
            )
            if trajectory_errors
            else None
        )

        return ObjectMetrics(
            object_id=object_id,
            visible_frames=visible_frames,
            occluded_frames=occluded_frames,
            identity_preservation=(
                safe_mean(identity_scores) or 0.0
            ),
            hidden_position_accuracy=(
                safe_mean(hidden_scores)
            ),
            occlusion_recovery=(
                safe_mean(recovery_scores)
            ),
            visible_tracking_accuracy=(
                safe_mean(visible_scores) or 0.0
            ),
            trajectory_rmse=trajectory_rmse,
            existence_accuracy=(
                safe_mean(existence_scores) or 0.0
            ),
            confidence_alignment=(
                safe_mean(confidence_scores) or 0.0
            ),
        )

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        dataset: ObjectPermanenceDataset,
    ) -> ObjectPermanenceResult:
        dataset.validate()

        if len(dataset.observations) > self.config.max_frames:
            raise ObjectPermanenceValidationError(
                "Frame count exceeds configured limit"
            )

        predictions = self._predict(
            dataset.observations
        )

        object_ids = sorted(
            {
                obj.object_id
                for frame in dataset.ground_truth
                for obj in frame.objects
            }
        )

        if not object_ids:
            raise ObjectPermanenceValidationError(
                "Ground truth contains no objects"
            )

        object_metrics = [
            self.evaluate_object(
                object_id,
                dataset.ground_truth,
                predictions,
            )
            for object_id in object_ids
        ]

        identity = float(
            np.mean(
                [
                    item.identity_preservation
                    for item in object_metrics
                ]
            )
        )

        hidden_values = [
            item.hidden_position_accuracy
            for item in object_metrics
            if item.hidden_position_accuracy is not None
        ]

        recovery_values = [
            item.occlusion_recovery
            for item in object_metrics
            if item.occlusion_recovery is not None
        ]

        trajectory_values = [
            item.trajectory_rmse
            for item in object_metrics
            if item.trajectory_rmse is not None
        ]

        hidden_score = safe_mean(
            hidden_values
        )

        recovery_score = safe_mean(
            recovery_values
        )

        trajectory_rmse = safe_mean(
            trajectory_values
        )

        trajectory_score = (
            error_to_score(
                trajectory_rmse,
                self.config.error_scale,
            )
            if trajectory_rmse is not None
            else None
        )

        visible_score = float(
            np.mean(
                [
                    item.visible_tracking_accuracy
                    for item in object_metrics
                ]
            )
        )

        existence_score = float(
            np.mean(
                [
                    item.existence_accuracy
                    for item in object_metrics
                ]
            )
        )

        confidence_score = float(
            np.mean(
                [
                    item.confidence_alignment
                    for item in object_metrics
                ]
            )
        )

        scores: dict[str, float | None] = {
            "identity_preservation": identity,
            "hidden_position_accuracy": hidden_score,
            "occlusion_recovery": recovery_score,
            "visible_tracking_accuracy": visible_score,
            "trajectory_accuracy": trajectory_score,
            "existence_accuracy": existence_score,
            "confidence_alignment": confidence_score,
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
            raise ObjectPermanenceValidationError(
                "No weighted metrics available"
            )

        overall_score = sum(
            float(scores[name]) * weight
            for name, weight in active_weights.items()
        ) / total_weight

        metrics = ObjectPermanenceMetrics(
            identity_preservation=identity,
            hidden_position_accuracy=hidden_score,
            occlusion_recovery=recovery_score,
            visible_tracking_accuracy=visible_score,
            trajectory_accuracy=trajectory_score,
            existence_accuracy=existence_score,
            confidence_alignment=confidence_score,
            overall_score=float(overall_score),
        )

        result = ObjectPermanenceResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=self.model.name,
            frame_count=len(
                dataset.observations
            ),
            object_count=len(
                object_ids
            ),
            metrics=metrics,
            object_metrics=object_metrics,
            passed=(
                overall_score
                >= self.config.pass_threshold
            ),
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            details={
                "weights": active_weights,
                "pass_threshold": (
                    self.config.pass_threshold
                ),
                "position_tolerance": (
                    self.config.position_tolerance
                ),
                "reappearance_tolerance": (
                    self.config.reappearance_tolerance
                ),
                "reference_source": dataset.metadata.get(
                    "source",
                    "unspecified",
                ),
                "interpretation": (
                    "Object identity and position agreement "
                    "with supplied reference trajectories"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        logger.info(
            "Object permanence evaluated: "
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
    ) -> ObjectPermanenceResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise ObjectPermanenceError(
                "Evaluation result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[ObjectPermanenceResult]:
        return list(
            self._results.values()
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_results(
        results: list[ObjectPermanenceResult],
    ) -> list[ObjectPermanenceResult]:
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
        results: list[ObjectPermanenceResult],
        output_path: str | Path,
    ) -> Path:
        if not results:
            raise ObjectPermanenceValidationError(
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
                "frame_count": result.frame_count,
                "object_count": result.object_count,
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

def create_object_permanence_evaluator(
    model: ObjectTrackingModelProtocol,
    *,
    config: ObjectPermanenceConfig | None = None,
) -> ObjectPermanenceEvaluator:
    return ObjectPermanenceEvaluator(
        model=model,
        config=config,
    )


# ============================================================
# SYNTHETIC DEMONSTRATION
# ============================================================

def generate_synthetic_dataset(
    *,
    frame_count: int = 30,
) -> ObjectPermanenceDataset:
    """
    Single object moving with constant velocity.

    The object is occluded for frames 10-17.
    Ground truth retains its hidden position.
    """

    if frame_count < 20:
        raise ValueError(
            "frame_count must be at least 20"
        )

    observations = []
    ground_truth = []

    for frame_index in range(frame_count):
        position = [
            0.25 * frame_index,
            0.10 * frame_index,
        ]

        visible = not (
            10 <= frame_index <= 17
        )

        truth_object = ObjectState(
            object_id="object_001",
            position=position,
            visible=visible,
            exists=True,
            confidence=1.0,
        )

        ground_truth.append(
            ObjectFrame(
                frame_index=frame_index,
                objects=[truth_object],
            )
        )

        observed_objects = (
            [
                ObjectState(
                    object_id="object_001",
                    position=position,
                    visible=True,
                    exists=True,
                    confidence=1.0,
                )
            ]
            if visible
            else []
        )

        observations.append(
            ObjectFrame(
                frame_index=frame_index,
                objects=observed_objects,
            )
        )

    return ObjectPermanenceDataset(
        observations=observations,
        ground_truth=ground_truth,
        metadata={
            "source": "synthetic_constant_velocity",
            "occlusion_frames": [10, 17],
        },
    )


class DemoObjectTrackingModel:
    """
    Constant-velocity object tracker.

    Demonstrates simple hidden-state prediction.

    Not a trained neural world model.
    """

    name = "demo_object_permanence_tracker"

    def predict_sequence(
        self,
        observations: list[ObjectFrame],
    ) -> list[ObjectFrame]:
        predictions = []

        last_position = None
        velocity = np.zeros(
            2,
            dtype=np.float64,
        )

        for frame in observations:
            observed = (
                frame.objects[0]
                if frame.objects
                else None
            )

            if observed is not None:
                current_position = np.asarray(
                    observed.position,
                    dtype=np.float64,
                )

                if last_position is not None:
                    velocity = (
                        current_position
                        - last_position
                    )

                last_position = current_position

                predicted_position = (
                    current_position
                )

                confidence = 0.99

            elif last_position is not None:
                predicted_position = (
                    last_position + velocity
                )

                last_position = (
                    predicted_position
                )

                confidence = 0.75

            else:
                predictions.append(
                    ObjectFrame(
                        frame_index=frame.frame_index,
                        objects=[],
                    )
                )
                continue

            predictions.append(
                ObjectFrame(
                    frame_index=frame.frame_index,
                    objects=[
                        ObjectState(
                            object_id="object_001",
                            position=(
                                predicted_position.tolist()
                            ),
                            visible=(
                                observed is not None
                            ),
                            exists=True,
                            confidence=confidence,
                        )
                    ],
                )
            )

        return predictions


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    dataset = generate_synthetic_dataset(
        frame_count=30
    )

    model = DemoObjectTrackingModel()

    evaluator = (
        create_object_permanence_evaluator(
            model
        )
    )

    result = evaluator.evaluate(
        dataset
    )

    print("=" * 65)
    print("NEUROWORLD AI — OBJECT PERMANENCE")
    print("=" * 65)

    print(
        json.dumps(
            asdict(result),
            indent=2,
        )
    )

    json_path = evaluator.export_json(
        result.evaluation_id,
        "experiments/reports/object_permanence.json",
    )

    csv_path = evaluator.export_csv(
        [result],
        "experiments/reports/object_permanence.csv",
    )

    print("\nJSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()
