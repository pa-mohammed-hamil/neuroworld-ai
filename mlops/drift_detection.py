
"""
NeuroWorld AI
backend/app/database/drift_detection.py

Data, prediction and model-performance drift monitoring.

Features:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) statistic
- Wasserstein-1 distance
- Jensen-Shannon divergence
- Numerical feature drift
- Prediction distribution drift
- Performance metric drift
- Configurable alert thresholds
- Persistent drift reports and alerts
- Historical monitoring
- JSON report exports

Dependencies:
    pip install numpy sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module detects statistical changes.
It does not automatically retrain or deploy models.
"""

from __future__ import annotations

import json
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    select,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from backend.app.database.session import (
    Base,
    session_scope,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class DriftDetectionError(Exception):
    """Base drift detection exception."""


class DriftValidationError(DriftDetectionError):
    """Invalid monitoring input."""


class DriftReportNotFoundError(
    DriftDetectionError
):
    """Requested drift report does not exist."""


# ============================================================
# ENUMS
# ============================================================

class DriftType(str, Enum):
    DATA = "data"
    PREDICTION = "prediction"
    PERFORMANCE = "performance"


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DriftStatus(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DriftConfig:
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    ks_warning: float = 0.10
    ks_critical: float = 0.20

    js_warning: float = 0.05
    js_critical: float = 0.15

    wasserstein_warning: float = 0.20
    wasserstein_critical: float = 0.50

    performance_warning: float = 0.10
    performance_critical: float = 0.25

    histogram_bins: int = 10
    minimum_samples: int = 30

    def __post_init__(self) -> None:
        for prefix in (
            "psi",
            "ks",
            "js",
            "wasserstein",
            "performance",
        ):
            warning = getattr(
                self,
                f"{prefix}_warning",
            )

            critical = getattr(
                self,
                f"{prefix}_critical",
            )

            if (
                not math.isfinite(warning)
                or not math.isfinite(critical)
                or warning < 0
                or critical <= warning
            ):
                raise DriftValidationError(
                    f"Invalid {prefix} thresholds"
                )

        if (
            isinstance(self.histogram_bins, bool)
            or not isinstance(
                self.histogram_bins,
                int,
            )
            or self.histogram_bins < 2
        ):
            raise DriftValidationError(
                "histogram_bins must be >= 2"
            )

        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(
                self.minimum_samples,
                int,
            )
            or self.minimum_samples < 2
        ):
            raise DriftValidationError(
                "minimum_samples must be >= 2"
            )


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_datetime(
    value: datetime | None,
) -> str | None:
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def validate_string(
    value: str,
    field: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str):
        raise DriftValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise DriftValidationError(
            f"Invalid {field}"
        )

    return value


def numeric_array(
    values: Sequence[float] | np.ndarray,
    name: str,
    minimum: int = 2,
) -> np.ndarray:
    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise DriftValidationError(
            f"{name} must contain numbers"
        ) from exc

    if array.ndim != 1:
        raise DriftValidationError(
            f"{name} must be one-dimensional"
        )

    if len(array) < minimum:
        raise DriftValidationError(
            f"{name} requires at least "
            f"{minimum} samples"
        )

    if not np.all(np.isfinite(array)):
        raise DriftValidationError(
            f"{name} contains non-finite values"
        )

    return array


def safe_json(
    value: Any,
) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return iso_datetime(value)

    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())

    if isinstance(value, np.generic):
        return safe_json(value.item())

    if isinstance(value, dict):
        return {
            str(key): safe_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            safe_json(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise DriftValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise DriftValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def histogram_probabilities(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build shared histogram bins using reference
    quantiles, falling back to equal-width bins.

    Smoothing prevents log(0) in PSI and JS.
    """

    if bins < 2:
        raise DriftValidationError(
            "bins must be >= 2"
        )

    edges = np.quantile(
        reference,
        np.linspace(
            0.0,
            1.0,
            bins + 1,
        ),
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        minimum = min(
            float(reference.min()),
            float(current.min()),
        )

        maximum = max(
            float(reference.max()),
            float(current.max()),
        )

        if minimum == maximum:
            return (
                np.array([1.0]),
                np.array([1.0]),
            )

        edges = np.linspace(
            minimum,
            maximum,
            bins + 1,
        )

    # Include values outside reference range
    # in the outermost histogram bins.
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    cur_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    ref_prob = (
        ref_counts.astype(float)
        + epsilon
    )

    cur_prob = (
        cur_counts.astype(float)
        + epsilon
    )

    ref_prob /= ref_prob.sum()
    cur_prob /= cur_prob.sum()

    return ref_prob, cur_prob


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    p, q = histogram_probabilities(
        reference,
        current,
        bins=bins,
    )

    return float(
        np.sum(
            (q - p) * np.log(q / p)
        )
    )


def jensen_shannon_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    p, q = histogram_probabilities(
        reference,
        current,
        bins=bins,
    )

    midpoint = 0.5 * (p + q)

    divergence = 0.5 * (
        np.sum(
            p * np.log(p / midpoint)
        )
        + np.sum(
            q * np.log(q / midpoint)
        )
    )

    return float(
        max(divergence, 0.0)
    )


def kolmogorov_smirnov_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Two-sample KS statistic.

    This returns the D statistic, not a p-value.
    """

    reference = np.sort(reference)
    current = np.sort(current)

    points = np.unique(
        np.concatenate(
            [reference, current]
        )
    )

    reference_cdf = (
        np.searchsorted(
            reference,
            points,
            side="right",
        )
        / len(reference)
    )

    current_cdf = (
        np.searchsorted(
            current,
            points,
            side="right",
        )
        / len(current)
    )

    return float(
        np.max(
            np.abs(
                reference_cdf - current_cdf
            )
        )
    )


def wasserstein_distance_1d(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Empirical 1D Wasserstein-1 distance.

    Integrates the absolute difference between
    the two empirical cumulative distributions.
    """

    combined = np.sort(
        np.concatenate(
            [reference, current]
        )
    )

    intervals = np.diff(combined)

    if len(intervals) == 0:
        return 0.0

    left_points = combined[:-1]

    ref_cdf = (
        np.searchsorted(
            np.sort(reference),
            left_points,
            side="right",
        )
        / len(reference)
    )

    cur_cdf = (
        np.searchsorted(
            np.sort(current),
            left_points,
            side="right",
        )
        / len(current)
    )

    return float(
        np.sum(
            np.abs(ref_cdf - cur_cdf)
            * intervals
        )
    )


# ============================================================
# ORM MODELS
# ============================================================

class DriftReportORM(Base):
    __tablename__ = "ai_drift_reports"

    __table_args__ = (
        Index(
            "ix_ai_drift_model_type_time",
            "model_name",
            "drift_type",
            "created_at",
        ),
    )

    report_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[
        str | None
    ] = mapped_column(
        String(128),
        nullable=True,
    )

    dataset_version_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
    )

    drift_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    reference_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    current_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    metrics_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "metrics",
        JSON,
        nullable=False,
    )

    configuration_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "configuration",
        JSON,
        nullable=False,
    )

    metadata_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    alerts: Mapped[
        list["DriftAlertORM"]
    ] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
    )


class DriftAlertORM(Base):
    __tablename__ = "ai_drift_alerts"

    alert_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    report_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_drift_reports.report_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    feature_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    metric_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    report: Mapped[DriftReportORM] = relationship(
        back_populates="alerts",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class FeatureDriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    wasserstein_distance: float
    js_divergence: float
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    status: str


@dataclass(frozen=True)
class DriftAlertRecord:
    alert_id: str
    report_id: str
    feature_name: str
    metric_name: str
    metric_value: float
    threshold: float
    severity: str
    message: str
    created_at: str


@dataclass(frozen=True)
class DriftReportRecord:
    report_id: str
    model_name: str
    model_version: str | None
    dataset_version_id: str | None
    drift_type: str
    status: str
    severity: str
    reference_count: int
    current_count: int
    metrics: dict[str, Any]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str


# ============================================================
# DRIFT DETECTOR
# ============================================================

class DriftDetector:
    """
    Statistical drift detector.

    Thresholds are heuristic monitoring rules.
    Calibrate them against historical data,
    domain-specific risk and false-alert rates.
    """

    def __init__(
        self,
        config: DriftConfig | None = None,
    ) -> None:
        self.config = (
            config or DriftConfig()
        )

    def _metric_status(
        self,
        value: float,
        warning: float,
        critical: float,
    ) -> DriftStatus:
        if value >= critical:
            return DriftStatus.CRITICAL

        if value >= warning:
            return DriftStatus.WARNING

        return DriftStatus.NORMAL

    def detect_feature(
        self,
        feature_name: str,
        reference: Sequence[float],
        current: Sequence[float],
    ) -> FeatureDriftResult:
        feature_name = validate_string(
            feature_name,
            "feature_name",
        )

        ref = numeric_array(
            reference,
            "reference",
            self.config.minimum_samples,
        )

        cur = numeric_array(
            current,
            "current",
            self.config.minimum_samples,
        )

        psi = population_stability_index(
            ref,
            cur,
            bins=self.config.histogram_bins,
        )

        ks = kolmogorov_smirnov_statistic(
            ref,
            cur,
        )

        wasserstein = wasserstein_distance_1d(
            ref,
            cur,
        )

        js = jensen_shannon_divergence(
            ref,
            cur,
            bins=self.config.histogram_bins,
        )

        # Normalize Wasserstein by reference
        # standard deviation to obtain a
        # dimensionless monitoring score.
        reference_std = float(
            np.std(ref)
        )

        scale = max(
            reference_std,
            1e-8,
        )

        normalized_wasserstein = (
            wasserstein / scale
        )

        statuses = [
            self._metric_status(
                psi,
                self.config.psi_warning,
                self.config.psi_critical,
            ),
            self._metric_status(
                ks,
                self.config.ks_warning,
                self.config.ks_critical,
            ),
            self._metric_status(
                js,
                self.config.js_warning,
                self.config.js_critical,
            ),
            self._metric_status(
                normalized_wasserstein,
                self.config.wasserstein_warning,
                self.config.wasserstein_critical,
            ),
        ]

        if DriftStatus.CRITICAL in statuses:
            status = DriftStatus.CRITICAL
        elif DriftStatus.WARNING in statuses:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.NORMAL

        return FeatureDriftResult(
            feature_name=feature_name,
            psi=psi,
            ks_statistic=ks,
            wasserstein_distance=wasserstein,
            js_divergence=js,
            reference_mean=float(
                np.mean(ref)
            ),
            current_mean=float(
                np.mean(cur)
            ),
            reference_std=reference_std,
            current_std=float(
                np.std(cur)
            ),
            status=status.value,
        )

    def detect_features(
        self,
        reference: Mapping[
            str,
            Sequence[float],
        ],
        current: Mapping[
            str,
            Sequence[float],
        ],
    ) -> list[FeatureDriftResult]:
        if not reference or not current:
            raise DriftValidationError(
                "Feature mappings cannot be empty"
            )

        if set(reference) != set(current):
            raise DriftValidationError(
                "Reference/current feature names differ"
            )

        return [
            self.detect_feature(
                feature_name,
                reference[feature_name],
                current[feature_name],
            )
            for feature_name in sorted(reference)
        ]

    def detect_prediction_drift(
        self,
        reference_predictions: Sequence[float],
        current_predictions: Sequence[float],
    ) -> FeatureDriftResult:
        return self.detect_feature(
            "model_predictions",
            reference_predictions,
            current_predictions,
        )

    def detect_performance_drift(
        self,
        *,
        baseline_metric: float,
        current_metric: float,
        lower_is_better: bool = True,
    ) -> dict[str, Any]:
        baseline = float(
            baseline_metric
        )

        current = float(
            current_metric
        )

        if not all(
            math.isfinite(value)
            for value in (baseline, current)
        ):
            raise DriftValidationError(
                "Performance metrics must be finite"
            )

        if lower_is_better:
            degradation = (
                current - baseline
            )
        else:
            degradation = (
                baseline - current
            )

        relative_degradation = max(
            degradation,
            0.0,
        ) / max(
            abs(baseline),
            1e-8,
        )

        status = self._metric_status(
            relative_degradation,
            self.config.performance_warning,
            self.config.performance_critical,
        )

        return {
            "baseline_metric": baseline,
            "current_metric": current,
            "lower_is_better": lower_is_better,
            "absolute_degradation": (
                max(degradation, 0.0)
            ),
            "relative_degradation": (
                relative_degradation
            ),
            "status": status.value,
        }


# ============================================================
# PERSISTENT DRIFT REPOSITORY
# ============================================================

class DriftDetectionRepository:
    """
    Stores drift reports and alerts.

    Uses the existing SQLAlchemy Base and
    session_scope from the database package.
    """

    def __init__(
        self,
        config: DriftConfig | None = None,
    ) -> None:
        self.config = (
            config or DriftConfig()
        )

        self.detector = DriftDetector(
            self.config
        )

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _report_record(
        row: DriftReportORM,
    ) -> DriftReportRecord:
        return DriftReportRecord(
            report_id=row.report_id,
            model_name=row.model_name,
            model_version=row.model_version,
            dataset_version_id=(
                row.dataset_version_id
            ),
            drift_type=row.drift_type,
            status=row.status,
            severity=row.severity,
            reference_count=row.reference_count,
            current_count=row.current_count,
            metrics=safe_json(
                row.metrics_data
            ),
            configuration=safe_json(
                row.configuration_data
            ),
            metadata=safe_json(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _alert_record(
        row: DriftAlertORM,
    ) -> DriftAlertRecord:
        return DriftAlertRecord(
            alert_id=row.alert_id,
            report_id=row.report_id,
            feature_name=row.feature_name,
            metric_name=row.metric_name,
            metric_value=row.metric_value,
            threshold=row.threshold,
            severity=row.severity,
            message=row.message,
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _severity(
        status: DriftStatus,
    ) -> DriftSeverity:
        if status == DriftStatus.CRITICAL:
            return DriftSeverity.CRITICAL

        if status == DriftStatus.WARNING:
            return DriftSeverity.MODERATE

        return DriftSeverity.NONE

    def _create_alerts(
        self,
        session: Session,
        report_id: str,
        feature: FeatureDriftResult,
    ) -> None:
        checks = [
            (
                "psi",
                feature.psi,
                self.config.psi_warning,
                self.config.psi_critical,
            ),
            (
                "ks_statistic",
                feature.ks_statistic,
                self.config.ks_warning,
                self.config.ks_critical,
            ),
            (
                "js_divergence",
                feature.js_divergence,
                self.config.js_warning,
                self.config.js_critical,
            ),
            (
                "normalized_wasserstein",
                (
                    feature.wasserstein_distance
                    / max(
                        feature.reference_std,
                        1e-8,
                    )
                ),
                self.config.wasserstein_warning,
                self.config.wasserstein_critical,
            ),
        ]

        for (
            metric_name,
            value,
            warning,
            critical,
        ) in checks:
            if value >= critical:
                severity = (
                    DriftSeverity.CRITICAL
                )

                threshold = critical

            elif value >= warning:
                severity = (
                    DriftSeverity.MODERATE
                )

                threshold = warning

            else:
                continue

            session.add(
                DriftAlertORM(
                    alert_id=uuid.uuid4().hex,
                    report_id=report_id,
                    feature_name=(
                        feature.feature_name
                    ),
                    metric_name=metric_name,
                    metric_value=float(value),
                    threshold=float(threshold),
                    severity=severity.value,
                    message=(
                        f"{feature.feature_name}: "
                        f"{metric_name}={value:.4f} "
                        f"exceeds {threshold:.4f}"
                    ),
                    created_at=utc_now(),
                )
            )

    # --------------------------------------------------------
    # MONITOR DATA DRIFT
    # --------------------------------------------------------

    def monitor_data_drift(
        self,
        *,
        model_name: str,
        reference: Mapping[
            str,
            Sequence[float],
        ],
        current: Mapping[
            str,
            Sequence[float],
        ],
        model_version: str | None = None,
        dataset_version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> DriftReportRecord:
        model_name = validate_string(
            model_name,
            "model_name",
        )

        features = self.detector.detect_features(
            reference,
            current,
        )

        statuses = {
            feature.status
            for feature in features
        }

        if DriftStatus.CRITICAL.value in statuses:
            status = DriftStatus.CRITICAL
        elif DriftStatus.WARNING.value in statuses:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.NORMAL

        # All features must represent the same
        # monitoring window.
        reference_sizes = {
            len(values)
            for values in reference.values()
        }

        current_sizes = {
            len(values)
            for values in current.values()
        }

        if (
            len(reference_sizes) != 1
            or len(current_sizes) != 1
        ):
            raise DriftValidationError(
                "All features must have equal "
                "sample counts per window"
            )

        row = DriftReportORM(
            report_id=uuid.uuid4().hex,
            model_name=model_name,
            model_version=model_version,
            dataset_version_id=(
                dataset_version_id
            ),
            drift_type=DriftType.DATA.value,
            status=status.value,
            severity=self._severity(
                status
            ).value,
            reference_count=(
                next(iter(reference_sizes))
            ),
            current_count=(
                next(iter(current_sizes))
            ),
            metrics_data={
                "features": [
                    asdict(feature)
                    for feature in features
                ],
                "drifted_feature_count": sum(
                    feature.status
                    != DriftStatus.NORMAL.value
                    for feature in features
                ),
            },
            configuration_data=asdict(
                self.config
            ),
            metadata_data=safe_json(
                metadata or {}
            ),
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            for feature in features:
                self._create_alerts(
                    session,
                    row.report_id,
                    feature,
                )

            session.flush()

            return self._report_record(row)

    # --------------------------------------------------------
    # MONITOR PREDICTION DRIFT
    # --------------------------------------------------------

    def monitor_prediction_drift(
        self,
        *,
        model_name: str,
        reference_predictions: Sequence[float],
        current_predictions: Sequence[float],
        model_version: str | None = None,
        db: Session | None = None,
    ) -> DriftReportRecord:
        model_name = validate_string(
            model_name,
            "model_name",
        )

        feature = (
            self.detector.detect_prediction_drift(
                reference_predictions,
                current_predictions,
            )
        )

        status = DriftStatus(
            feature.status
        )

        row = DriftReportORM(
            report_id=uuid.uuid4().hex,
            model_name=model_name,
            model_version=model_version,
            drift_type=(
                DriftType.PREDICTION.value
            ),
            status=status.value,
            severity=self._severity(
                status
            ).value,
            reference_count=len(
                reference_predictions
            ),
            current_count=len(
                current_predictions
            ),
            metrics_data={
                "prediction": asdict(feature)
            },
            configuration_data=asdict(
                self.config
            ),
            metadata_data={},
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            self._create_alerts(
                session,
                row.report_id,
                feature,
            )

            session.flush()

            return self._report_record(row)

    # --------------------------------------------------------
    # MONITOR PERFORMANCE DRIFT
    # --------------------------------------------------------

    def monitor_performance_drift(
        self,
        *,
        model_name: str,
        metric_name: str,
        baseline_metric: float,
        current_metric: float,
        lower_is_better: bool = True,
        model_version: str | None = None,
        db: Session | None = None,
    ) -> DriftReportRecord:
        model_name = validate_string(
            model_name,
            "model_name",
        )

        metric_name = validate_string(
            metric_name,
            "metric_name",
        )

        result = (
            self.detector.detect_performance_drift(
                baseline_metric=baseline_metric,
                current_metric=current_metric,
                lower_is_better=lower_is_better,
            )
        )

        status = DriftStatus(
            result["status"]
        )

        row = DriftReportORM(
            report_id=uuid.uuid4().hex,
            model_name=model_name,
            model_version=model_version,
            drift_type=(
                DriftType.PERFORMANCE.value
            ),
            status=status.value,
            severity=self._severity(
                status
            ).value,
            reference_count=1,
            current_count=1,
            metrics_data={
                "metric_name": metric_name,
                **result,
            },
            configuration_data=asdict(
                self.config
            ),
            metadata_data={},
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            if status != DriftStatus.NORMAL:
                threshold = (
                    self.config.performance_critical
                    if status == DriftStatus.CRITICAL
                    else self.config.performance_warning
                )

                session.add(
                    DriftAlertORM(
                        alert_id=uuid.uuid4().hex,
                        report_id=row.report_id,
                        feature_name=metric_name,
                        metric_name=(
                            "relative_degradation"
                        ),
                        metric_value=(
                            result[
                                "relative_degradation"
                            ]
                        ),
                        threshold=threshold,
                        severity=self._severity(
                            status
                        ).value,
                        message=(
                            f"{metric_name} performance "
                            f"degraded by "
                            f"{result['relative_degradation']:.1%}"
                        ),
                        created_at=utc_now(),
                    )
                )

            session.flush()

            return self._report_record(row)

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    def get_report(
        self,
        report_id: str,
        *,
        db: Session | None = None,
    ) -> DriftReportRecord:
        with self._session(db) as session:
            row = session.get(
                DriftReportORM,
                report_id,
            )

            if row is None:
                raise DriftReportNotFoundError(
                    "Drift report not found"
                )

            return self._report_record(row)

    # --------------------------------------------------------
    # LIST REPORTS
    # --------------------------------------------------------

    def list_reports(
        self,
        *,
        model_name: str | None = None,
        drift_type: DriftType | None = None,
        status: DriftStatus | None = None,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[DriftReportRecord]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise DriftValidationError(
                "Invalid limit"
            )

        statement = select(
            DriftReportORM
        )

        if model_name is not None:
            statement = statement.where(
                DriftReportORM.model_name
                == model_name
            )

        if drift_type is not None:
            statement = statement.where(
                DriftReportORM.drift_type
                == drift_type.value
            )

        if status is not None:
            statement = statement.where(
                DriftReportORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                DriftReportORM.created_at.desc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._report_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST ALERTS
    # --------------------------------------------------------

    def list_alerts(
        self,
        report_id: str,
        *,
        db: Session | None = None,
    ) -> list[DriftAlertRecord]:
        statement = (
            select(DriftAlertORM)
            .where(
                DriftAlertORM.report_id
                == report_id
            )
            .order_by(
                DriftAlertORM.created_at.asc()
            )
        )

        with self._session(db) as session:
            return [
                self._alert_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_json(
        self,
        report_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            report = self.get_report(
                report_id,
                db=session,
            )

            alerts = self.list_alerts(
                report_id,
                db=session,
            )

        payload = {
            "report": asdict(report),
            "alerts": [
                asdict(alert)
                for alert in alerts
            ],
        }

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
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return path


# ============================================================
# FACTORY
# ============================================================

def create_drift_detection_repository(
    config: DriftConfig | None = None,
) -> DriftDetectionRepository:
    return DriftDetectionRepository(
        config=config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    create_database_tables()

    rng = np.random.default_rng(
        42
    )

    reference = {
        "position_x": rng.normal(
            0.0, 1.0, 1000
        ),
        "velocity": rng.normal(
            5.0, 1.0, 1000
        ),
        "temperature": rng.normal(
            25.0, 2.0, 1000
        ),
    }

    current = {
        "position_x": rng.normal(
            0.8, 1.0, 1000
        ),
        "velocity": rng.normal(
            5.2, 1.0, 1000
        ),
        "temperature": rng.normal(
            30.0, 2.0, 1000
        ),
    }

    repository = (
        create_drift_detection_repository()
    )

    report = repository.monitor_data_drift(
        model_name="neuroworld_transformer",
        model_version="v1",
        reference=reference,
        current=current,
        metadata={
            "environment": "synthetic_physics",
            "demo": True,
        },
    )

    alerts = repository.list_alerts(
        report.report_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - DRIFT DETECTION")
    print("=" * 60)

    print(
        "Report ID:",
        report.report_id,
    )

    print(
        "Status:",
        report.status,
    )

    print(
        "Severity:",
        report.severity,
    )

    print(
        "Drifted features:",
        report.metrics[
            "drifted_feature_count"
        ],
    )

    print(
        "Alerts:",
        len(alerts),
    )

    for feature in report.metrics["features"]:
        print(
            feature["feature_name"],
            "PSI:",
            round(feature["psi"], 4),
            "KS:",
            round(
                feature["ks_statistic"],
                4,
            ),
            "Status:",
            feature["status"],
        )

    output = repository.export_json(
        report.report_id,
        "experiments/reports/"
        "drift_detection_report.json",
    )

    print(
        "Report exported:",
        output,
    )

    close_database()


if __name__ == "__main__":
    main()
