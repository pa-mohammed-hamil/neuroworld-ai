
"""
NeuroWorld AI
evaluation/calibration.py

Uncertainty Calibration Evaluation for Neural World Models.

Features:
- Regression uncertainty calibration
- Classification confidence calibration
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score
- Negative Log-Likelihood (NLL)
- Gaussian prediction interval coverage
- Prediction interval calibration error
- Uncertainty-error correlation
- Reliability diagram data
- Batch evaluation
- JSON report export
- Reproducible demonstration

Dependencies:
    pip install numpy

Optional visualization:
    pip install matplotlib

Notes:
- Regression uncertainties must represent standard
  deviations, not arbitrary confidence scores.
- Classification confidence must be in [0, 1].
- Calibration does not establish model correctness,
  safety, or out-of-distribution reliability.
"""

from __future__ import annotations

import json
import logging
import math
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np


logger = logging.getLogger(
    "neuroworld.evaluation.calibration"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class CalibrationError(Exception):
    pass


class CalibrationValidationError(CalibrationError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CalibrationConfig:
    num_bins: int = 15

    confidence_levels: tuple[float, ...] = (
        0.50,
        0.68,
        0.80,
        0.90,
        0.95,
        0.99,
    )

    epsilon: float = 1e-8

    min_std: float = 1e-6

    max_samples: int = 100000

    random_seed: int = 42

    def __post_init__(self):
        if self.num_bins < 2:
            raise ValueError(
                "num_bins must be at least 2"
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if self.min_std <= 0:
            raise ValueError(
                "min_std must be positive"
            )

        if self.max_samples < 1:
            raise ValueError(
                "max_samples must be positive"
            )

        if not self.confidence_levels:
            raise ValueError(
                "confidence_levels cannot be empty"
            )

        if any(
            not math.isfinite(level)
            or not 0 < level < 1
            for level in self.confidence_levels
        ):
            raise ValueError(
                "Confidence levels must be between 0 and 1"
            )


# ============================================================
# RESULT MODELS
# ============================================================

@dataclass
class ReliabilityBin:
    lower_bound: float
    upper_bound: float

    sample_count: int

    mean_confidence: float
    observed_accuracy: float

    calibration_gap: float


@dataclass
class PredictionIntervalResult:
    nominal_coverage: float
    observed_coverage: float

    absolute_calibration_error: float

    mean_interval_width: float


@dataclass
class ClassificationCalibrationMetrics:
    expected_calibration_error: float
    maximum_calibration_error: float

    brier_score: float
    negative_log_likelihood: float

    accuracy: float
    mean_confidence: float

    sample_count: int


@dataclass
class RegressionCalibrationMetrics:
    mean_squared_error: float
    root_mean_squared_error: float
    mean_absolute_error: float

    gaussian_negative_log_likelihood: float

    mean_predicted_std: float

    uncertainty_error_correlation: float

    interval_calibration_error: float

    sample_count: int


@dataclass
class CalibrationResult:
    evaluation_id: str

    model_name: str
    task_type: str

    created_at: str

    metrics: dict[str, Any]

    reliability_bins: list[ReliabilityBin] = field(
        default_factory=list
    )

    prediction_intervals: list[
        PredictionIntervalResult
    ] = field(
        default_factory=list
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_numeric_array(
    values: Any,
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationValidationError(
            f"{name} must be numeric"
        ) from exc

    if array.size == 0:
        raise CalibrationValidationError(
            f"{name} cannot be empty"
        )

    if not np.all(np.isfinite(array)):
        raise CalibrationValidationError(
            f"{name} contains non-finite values"
        )

    return array


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    x = first.reshape(-1)
    y = second.reshape(-1)

    if x.shape != y.shape:
        raise CalibrationValidationError(
            "Correlation arrays must have matching shapes"
        )

    if x.size < 2:
        return 0.0

    if (
        np.std(x) <= epsilon
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
# CALIBRATION EVALUATOR
# ============================================================

class CalibrationEvaluator:
    """
    Evaluate uncertainty calibration.

    Supports:
        Classification:
            confidence vs observed correctness

        Regression:
            Gaussian predictive mean and std
            vs observed continuous targets

    Results are diagnostic measurements, not
    guarantees of reliable deployment behavior.
    """

    def __init__(
        self,
        config: CalibrationConfig | None = None,
    ) -> None:
        self.config = (
            config or CalibrationConfig()
        )

        self._results: dict[
            str,
            CalibrationResult,
        ] = {}

    # --------------------------------------------------------
    # CLASSIFICATION CALIBRATION
    # --------------------------------------------------------

    def evaluate_classification(
        self,
        *,
        model_name: str,
        confidences: Any,
        predictions: Any,
        targets: Any,
    ) -> CalibrationResult:
        """
        Evaluate confidence assigned to predicted
        classes.

        confidences:
            Shape (N,), confidence of predicted class.

        predictions:
            Shape (N,), predicted class labels.

        targets:
            Shape (N,), true class labels.

        The Brier score here is the binary
        correctness-event Brier score:
            (confidence - correctness)^2

        It is not the multiclass probability
        vector Brier score.
        """

        confidence = validate_numeric_array(
            confidences,
            name="confidences",
        )

        predicted = np.asarray(
            predictions
        )

        target = np.asarray(
            targets
        )

        if (
            confidence.ndim != 1
            or predicted.ndim != 1
            or target.ndim != 1
        ):
            raise CalibrationValidationError(
                "Classification inputs must be 1D"
            )

        n = len(confidence)

        if not (
            len(predicted) == len(target) == n
        ):
            raise CalibrationValidationError(
                "Classification array lengths must match"
            )

        if n > self.config.max_samples:
            raise CalibrationValidationError(
                "Sample count exceeds configured limit"
            )

        if np.any(
            (confidence < 0)
            | (confidence > 1)
        ):
            raise CalibrationValidationError(
                "Confidence values must be in [0, 1]"
            )

        correctness = (
            predicted == target
        ).astype(np.float64)

        accuracy = float(
            np.mean(correctness)
        )

        mean_confidence = float(
            np.mean(confidence)
        )

        bins = self._build_reliability_bins(
            confidence,
            correctness,
        )

        ece = sum(
            (
                item.sample_count / n
            ) * item.calibration_gap
            for item in bins
        )

        mce = max(
            (
                item.calibration_gap
                for item in bins
                if item.sample_count > 0
            ),
            default=0.0,
        )

        brier = float(
            np.mean(
                np.square(
                    confidence - correctness
                )
            )
        )

        clipped = np.clip(
            confidence,
            self.config.epsilon,
            1.0 - self.config.epsilon,
        )

        nll = float(
            -np.mean(
                correctness * np.log(clipped)
                + (1.0 - correctness)
                * np.log(1.0 - clipped)
            )
        )

        metrics = ClassificationCalibrationMetrics(
            expected_calibration_error=float(ece),
            maximum_calibration_error=float(mce),
            brier_score=brier,
            negative_log_likelihood=nll,
            accuracy=accuracy,
            mean_confidence=mean_confidence,
            sample_count=n,
        )

        result = CalibrationResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=model_name,
            task_type="classification",
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            metrics=asdict(metrics),
            reliability_bins=bins,
            details={
                "num_bins": self.config.num_bins,
                "brier_definition": (
                    "predicted-class correctness event"
                ),
                "nll_definition": (
                    "binary correctness-event log loss"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        return result

    # --------------------------------------------------------
    # RELIABILITY BINS
    # --------------------------------------------------------

    def _build_reliability_bins(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> list[ReliabilityBin]:
        edges = np.linspace(
            0.0,
            1.0,
            self.config.num_bins + 1,
        )

        bins = []

        for index in range(
            self.config.num_bins
        ):
            lower = float(
                edges[index]
            )

            upper = float(
                edges[index + 1]
            )

            if index == self.config.num_bins - 1:
                mask = (
                    (confidences >= lower)
                    & (confidences <= upper)
                )
            else:
                mask = (
                    (confidences >= lower)
                    & (confidences < upper)
                )

            count = int(
                np.sum(mask)
            )

            if count > 0:
                mean_confidence = float(
                    np.mean(
                        confidences[mask]
                    )
                )

                observed_accuracy = float(
                    np.mean(
                        correctness[mask]
                    )
                )

                gap = abs(
                    mean_confidence
                    - observed_accuracy
                )
            else:
                mean_confidence = 0.0
                observed_accuracy = 0.0
                gap = 0.0

            bins.append(
                ReliabilityBin(
                    lower_bound=lower,
                    upper_bound=upper,
                    sample_count=count,
                    mean_confidence=mean_confidence,
                    observed_accuracy=observed_accuracy,
                    calibration_gap=float(gap),
                )
            )

        return bins

    # --------------------------------------------------------
    # REGRESSION CALIBRATION
    # --------------------------------------------------------

    def evaluate_regression(
        self,
        *,
        model_name: str,
        predicted_means: Any,
        predicted_stds: Any,
        targets: Any,
    ) -> CalibrationResult:
        """
        Evaluate Gaussian predictive distributions.

        predicted_means:
            Shape (N,) or (N, D).

        predicted_stds:
            Same shape as predicted_means.
            Values represent predictive standard
            deviations.

        targets:
            Same shape as predicted_means.

        Gaussian NLL assumes independent
        predictive dimensions.
        """

        means = validate_numeric_array(
            predicted_means,
            name="predicted_means",
        )

        stds = validate_numeric_array(
            predicted_stds,
            name="predicted_stds",
        )

        actual = validate_numeric_array(
            targets,
            name="targets",
        )

        if not (
            means.shape
            == stds.shape
            == actual.shape
        ):
            raise CalibrationValidationError(
                "Regression arrays must have identical shapes"
            )

        if means.ndim not in (1, 2):
            raise CalibrationValidationError(
                "Regression arrays must be 1D or 2D"
            )

        if means.shape[0] > self.config.max_samples:
            raise CalibrationValidationError(
                "Sample count exceeds configured limit"
            )

        if np.any(stds <= 0):
            raise CalibrationValidationError(
                "Predictive standard deviations must be positive"
            )

        stds = np.maximum(
            stds,
            self.config.min_std,
        )

        residual = (
            actual - means
        )

        squared_error = np.square(
            residual
        )

        mse = float(
            np.mean(squared_error)
        )

        rmse = float(
            np.sqrt(mse)
        )

        mae = float(
            np.mean(
                np.abs(residual)
            )
        )

        variance = np.square(
            stds
        )

        gaussian_nll = float(
            np.mean(
                0.5 * np.log(
                    2.0 * math.pi * variance
                )
                + squared_error
                / (2.0 * variance)
            )
        )

        intervals = (
            self._evaluate_prediction_intervals(
                means,
                stds,
                actual,
            )
        )

        interval_calibration_error = float(
            np.mean(
                [
                    item.absolute_calibration_error
                    for item in intervals
                ]
            )
        )

        correlation = safe_correlation(
            stds,
            np.abs(residual),
            self.config.epsilon,
        )

        metrics = RegressionCalibrationMetrics(
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            mean_absolute_error=mae,
            gaussian_negative_log_likelihood=gaussian_nll,
            mean_predicted_std=float(
                np.mean(stds)
            ),
            uncertainty_error_correlation=correlation,
            interval_calibration_error=(
                interval_calibration_error
            ),
            sample_count=means.shape[0],
        )

        result = CalibrationResult(
            evaluation_id=uuid.uuid4().hex,
            model_name=model_name,
            task_type="regression",
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
            metrics=asdict(metrics),
            prediction_intervals=intervals,
            details={
                "distribution": "gaussian",
                "independent_dimensions": True,
                "confidence_levels": list(
                    self.config.confidence_levels
                ),
                "coverage_aggregation": (
                    "elementwise across samples and dimensions"
                ),
            },
        )

        self._results[
            result.evaluation_id
        ] = result

        return result

    # --------------------------------------------------------
    # PREDICTION INTERVALS
    # --------------------------------------------------------

    def _evaluate_prediction_intervals(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        targets: np.ndarray,
    ) -> list[PredictionIntervalResult]:
        intervals = []

        normal = NormalDist()

        for level in self.config.confidence_levels:
            z_score = normal.inv_cdf(
                (1.0 + level) / 2.0
            )

            lower = (
                means - z_score * stds
            )

            upper = (
                means + z_score * stds
            )

            covered = (
                (targets >= lower)
                & (targets <= upper)
            )

            observed_coverage = float(
                np.mean(
                    covered
                )
            )

            width = float(
                np.mean(
                    upper - lower
                )
            )

            intervals.append(
                PredictionIntervalResult(
                    nominal_coverage=float(level),
                    observed_coverage=observed_coverage,
                    absolute_calibration_error=abs(
                        observed_coverage - level
                    ),
                    mean_interval_width=width,
                )
            )

        return intervals

    # --------------------------------------------------------
    # UNCERTAINTY ERROR ANALYSIS
    # --------------------------------------------------------

    def uncertainty_error_analysis(
        self,
        *,
        predicted_means: Any,
        predicted_stds: Any,
        targets: Any,
    ) -> dict[str, float]:
        means = validate_numeric_array(
            predicted_means,
            name="predicted_means",
        )

        stds = validate_numeric_array(
            predicted_stds,
            name="predicted_stds",
        )

        actual = validate_numeric_array(
            targets,
            name="targets",
        )

        if not (
            means.shape
            == stds.shape
            == actual.shape
        ):
            raise CalibrationValidationError(
                "Array shapes must match"
            )

        if np.any(stds <= 0):
            raise CalibrationValidationError(
                "Standard deviations must be positive"
            )

        errors = np.abs(
            actual - means
        )

        correlation = safe_correlation(
            stds,
            errors,
            self.config.epsilon,
        )

        standardized_errors = (
            errors
            / np.maximum(
                stds,
                self.config.min_std,
            )
        )

        return {
            "uncertainty_error_correlation": (
                correlation
            ),
            "mean_absolute_error": float(
                np.mean(errors)
            ),
            "mean_predicted_std": float(
                np.mean(stds)
            ),
            "mean_standardized_absolute_error": float(
                np.mean(standardized_errors)
            ),
        }

    # --------------------------------------------------------
    # RESULT MANAGEMENT
    # --------------------------------------------------------

    def get_result(
        self,
        evaluation_id: str,
    ) -> CalibrationResult:
        result = self._results.get(
            evaluation_id
        )

        if result is None:
            raise CalibrationError(
                "Calibration result not found"
            )

        return result

    def list_results(
        self,
    ) -> list[CalibrationResult]:
        return list(
            self._results.values()
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
    # RELIABILITY PLOT
    # --------------------------------------------------------

    def plot_reliability(
        self,
        evaluation_id: str,
        output_path: str | Path,
    ) -> Path:
        result = self.get_result(
            evaluation_id
        )

        if result.task_type != "classification":
            raise CalibrationValidationError(
                "Reliability diagram requires classification results"
            )

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise CalibrationError(
                "Install matplotlib to create plots"
            ) from exc

        populated = [
            item
            for item in result.reliability_bins
            if item.sample_count > 0
        ]

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )

        ax.plot(
            [
                item.mean_confidence
                for item in populated
            ],
            [
                item.observed_accuracy
                for item in populated
            ],
            marker="o",
            label="Model calibration",
        )

        ax.set_xlabel(
            "Mean predicted confidence"
        )

        ax.set_ylabel(
            "Observed accuracy"
        )

        ax.set_title(
            f"Reliability Diagram — {result.model_name}"
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        path = Path(
            output_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.tight_layout()
        fig.savefig(
            path,
            dpi=150,
        )

        plt.close(fig)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_calibration_evaluator(
    config: CalibrationConfig | None = None,
) -> CalibrationEvaluator:
    return CalibrationEvaluator(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    rng = np.random.default_rng(
        42
    )

    evaluator = create_calibration_evaluator()

    # --------------------------------------------------------
    # Regression example
    # --------------------------------------------------------

    sample_count = 1000

    predicted_means = rng.normal(
        0.0,
        1.0,
        sample_count,
    )

    predicted_stds = rng.uniform(
        0.1,
        0.5,
        sample_count,
    )

    targets = (
        predicted_means
        + rng.normal(
            0.0,
            predicted_stds,
        )
    )

    regression_result = (
        evaluator.evaluate_regression(
            model_name="demo_world_model",
            predicted_means=predicted_means,
            predicted_stds=predicted_stds,
            targets=targets,
        )
    )

    # --------------------------------------------------------
    # Classification example
    # --------------------------------------------------------

    confidences = rng.uniform(
        0.5,
        1.0,
        sample_count,
    )

    correctness = (
        rng.random(sample_count)
        < confidences
    )

    predictions = np.ones(
        sample_count,
        dtype=np.int64,
    )

    class_targets = np.where(
        correctness,
        1,
        0,
    )

    classification_result = (
        evaluator.evaluate_classification(
            model_name="demo_event_predictor",
            confidences=confidences,
            predictions=predictions,
            targets=class_targets,
        )
    )

    print("=" * 65)
    print("NEUROWORLD AI — UNCERTAINTY CALIBRATION")
    print("=" * 65)

    print("\nREGRESSION CALIBRATION:")
    print(
        json.dumps(
            regression_result.metrics,
            indent=2,
        )
    )

    print("\nCLASSIFICATION CALIBRATION:")
    print(
        json.dumps(
            classification_result.metrics,
            indent=2,
        )
    )

    evaluator.export_json(
        regression_result.evaluation_id,
        "experiments/reports/regression_calibration.json",
    )

    evaluator.export_json(
        classification_result.evaluation_id,
        "experiments/reports/classification_calibration.json",
    )


if __name__ == "__main__":
    main()
