
"""
NeuroWorld AI
backend/app/database/inference_monitor.py

Persistent inference monitoring for AI/ML systems.

Features:
- Model inference logging
- Latency and throughput tracking
- P50/P95/P99 latency
- Success and error rates
- Prediction confidence monitoring
- Input/output token usage
- Estimated inference costs
- Model/version filtering
- Configurable latency/error alerts
- Historical performance reports
- JSON/CSV exports
- FastAPI integration support

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module records inference telemetry.
It does not execute model inference.
"""

from __future__ import annotations

import csv
import json
import math
import time
import uuid

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
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

class InferenceMonitorError(Exception):
    """Base monitoring exception."""


class InferenceValidationError(InferenceMonitorError):
    """Invalid inference telemetry."""


class InferenceNotFoundError(InferenceMonitorError):
    """Inference record not found."""


# ============================================================
# ENUMS
# ============================================================

class InferenceStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class InferenceType(str, Enum):
    WORLD_MODEL = "world_model"
    NEXT_STATE = "next_state"
    FUTURE_PREDICTION = "future_prediction"
    TRAJECTORY = "trajectory"
    IMAGINATION = "imagination"
    PLANNING = "planning"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    LLM = "llm"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class InferenceMonitorConfig:
    latency_warning_ms: float = 500.0
    latency_critical_ms: float = 2000.0

    confidence_warning: float = 0.50
    confidence_critical: float = 0.20

    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.20

    minimum_requests_for_rate_alert: int = 20

    def __post_init__(self) -> None:
        if not (
            0 < self.latency_warning_ms
            < self.latency_critical_ms
        ):
            raise InferenceValidationError(
                "Invalid latency thresholds"
            )

        if not (
            0 <= self.confidence_critical
            < self.confidence_warning <= 1
        ):
            raise InferenceValidationError(
                "Invalid confidence thresholds"
            )

        if not (
            0 <= self.error_rate_warning
            < self.error_rate_critical <= 1
        ):
            raise InferenceValidationError(
                "Invalid error-rate thresholds"
            )

        if (
            isinstance(
                self.minimum_requests_for_rate_alert,
                bool,
            )
            or self.minimum_requests_for_rate_alert < 1
        ):
            raise InferenceValidationError(
                "Invalid minimum request count"
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
        raise InferenceValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise InferenceValidationError(
            f"Invalid {field}"
        )

    return value


def finite_number(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise InferenceValidationError(
            f"{field} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise InferenceValidationError(
            f"{field} must be finite"
        )

    if minimum is not None and result < minimum:
        raise InferenceValidationError(
            f"{field} is below minimum"
        )

    if maximum is not None and result > maximum:
        raise InferenceValidationError(
            f"{field} exceeds maximum"
        )

    return result


def nonnegative_int(
    value: int,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise InferenceValidationError(
            f"{field} must be a nonnegative integer"
        )

    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return iso_datetime(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise InferenceValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise InferenceValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


def percentile(
    values: list[float],
    percent: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * percent / 100.0

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


# ============================================================
# ORM MODELS
# ============================================================

class InferenceLogORM(Base):
    __tablename__ = "ai_inference_logs"

    __table_args__ = (
        Index(
            "ix_ai_inference_model_time",
            "model_name",
            "created_at",
        ),
        Index(
            "ix_ai_inference_model_version",
            "model_name",
            "model_version",
        ),
    )

    inference_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    request_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
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

    inference_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    confidence: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    output_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    estimated_cost_usd: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    batch_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    device: Mapped[
        str | None
    ] = mapped_column(
        String(128),
        nullable=True,
    )

    error_type: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    error_message: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
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
        index=True,
    )

    alerts: Mapped[
        list["InferenceAlertORM"]
    ] = relationship(
        back_populates="inference",
        cascade="all, delete-orphan",
    )


class InferenceAlertORM(Base):
    __tablename__ = "ai_inference_alerts"

    alert_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    inference_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_inference_logs.inference_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    alert_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    metric_value: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    threshold: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
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

    inference: Mapped[
        InferenceLogORM
    ] = relationship(
        back_populates="alerts",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class InferenceRecord:
    inference_id: str
    request_id: str
    model_name: str
    model_version: str | None
    inference_type: str
    status: str
    latency_ms: float
    confidence: float | None
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    batch_size: int
    device: str | None
    error_type: str | None
    error_message: str | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class InferenceAlertRecord:
    alert_id: str
    inference_id: str
    alert_type: str
    severity: str
    metric_value: float | None
    threshold: float | None
    message: str
    created_at: str


@dataclass(frozen=True)
class InferenceStatistics:
    model_name: str
    model_version: str | None
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    error_rate: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    minimum_latency_ms: float
    maximum_latency_ms: float
    average_confidence: float | None
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    throughput_requests_per_second: float
    window_seconds: float


# ============================================================
# INFERENCE MONITOR
# ============================================================

class InferenceMonitor:
    """
    SQLAlchemy-backed inference monitoring.

    Avoid storing prompts, raw images, credentials,
    personal data or other sensitive payloads
    inside telemetry metadata.
    """

    def __init__(
        self,
        config: InferenceMonitorConfig | None = None,
    ) -> None:
        self.config = (
            config or InferenceMonitorConfig()
        )

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _record(
        row: InferenceLogORM,
    ) -> InferenceRecord:
        return InferenceRecord(
            inference_id=row.inference_id,
            request_id=row.request_id,
            model_name=row.model_name,
            model_version=row.model_version,
            inference_type=row.inference_type,
            status=row.status,
            latency_ms=row.latency_ms,
            confidence=row.confidence,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            estimated_cost_usd=(
                row.estimated_cost_usd
            ),
            batch_size=row.batch_size,
            device=row.device,
            error_type=row.error_type,
            error_message=row.error_message,
            metadata=json_safe(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _alert_record(
        row: InferenceAlertORM,
    ) -> InferenceAlertRecord:
        return InferenceAlertRecord(
            alert_id=row.alert_id,
            inference_id=row.inference_id,
            alert_type=row.alert_type,
            severity=row.severity,
            metric_value=row.metric_value,
            threshold=row.threshold,
            message=row.message,
            created_at=iso_datetime(
                row.created_at
            ),
        )

    # --------------------------------------------------------
    # CREATE ALERT
    # --------------------------------------------------------

    @staticmethod
    def _add_alert(
        session: Session,
        *,
        inference_id: str,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        metric_value: float | None = None,
        threshold: float | None = None,
    ) -> None:
        session.add(
            InferenceAlertORM(
                alert_id=uuid.uuid4().hex,
                inference_id=inference_id,
                alert_type=alert_type,
                severity=severity.value,
                metric_value=metric_value,
                threshold=threshold,
                message=message,
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # EVALUATE INFERENCE ALERTS
    # --------------------------------------------------------

    def _evaluate_alerts(
        self,
        session: Session,
        row: InferenceLogORM,
    ) -> None:
        if (
            row.latency_ms
            >= self.config.latency_critical_ms
        ):
            self._add_alert(
                session,
                inference_id=row.inference_id,
                alert_type="high_latency",
                severity=AlertSeverity.CRITICAL,
                metric_value=row.latency_ms,
                threshold=(
                    self.config.latency_critical_ms
                ),
                message=(
                    "Inference latency exceeded "
                    "critical threshold"
                ),
            )

        elif (
            row.latency_ms
            >= self.config.latency_warning_ms
        ):
            self._add_alert(
                session,
                inference_id=row.inference_id,
                alert_type="high_latency",
                severity=AlertSeverity.WARNING,
                metric_value=row.latency_ms,
                threshold=(
                    self.config.latency_warning_ms
                ),
                message=(
                    "Inference latency exceeded "
                    "warning threshold"
                ),
            )

        if row.confidence is not None:
            if (
                row.confidence
                <= self.config.confidence_critical
            ):
                self._add_alert(
                    session,
                    inference_id=row.inference_id,
                    alert_type="low_confidence",
                    severity=AlertSeverity.CRITICAL,
                    metric_value=row.confidence,
                    threshold=(
                        self.config.confidence_critical
                    ),
                    message=(
                        "Prediction confidence "
                        "below critical threshold"
                    ),
                )

            elif (
                row.confidence
                <= self.config.confidence_warning
            ):
                self._add_alert(
                    session,
                    inference_id=row.inference_id,
                    alert_type="low_confidence",
                    severity=AlertSeverity.WARNING,
                    metric_value=row.confidence,
                    threshold=(
                        self.config.confidence_warning
                    ),
                    message=(
                        "Prediction confidence "
                        "below warning threshold"
                    ),
                )

        if row.status != InferenceStatus.SUCCESS.value:
            self._add_alert(
                session,
                inference_id=row.inference_id,
                alert_type="inference_failure",
                severity=(
                    AlertSeverity.CRITICAL
                    if row.status
                    == InferenceStatus.TIMEOUT.value
                    else AlertSeverity.WARNING
                ),
                message=(
                    f"Inference finished with "
                    f"status: {row.status}"
                ),
            )

    # --------------------------------------------------------
    # LOG INFERENCE
    # --------------------------------------------------------

    def log_inference(
        self,
        *,
        model_name: str,
        inference_type: InferenceType,
        status: InferenceStatus,
        latency_ms: float,
        model_version: str | None = None,
        request_id: str | None = None,
        confidence: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        estimated_cost_usd: float = 0.0,
        batch_size: int = 1,
        device: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> InferenceRecord:
        model_name = validate_string(
            model_name,
            "model_name",
        )

        if not isinstance(
            inference_type,
            InferenceType,
        ):
            raise InferenceValidationError(
                "Invalid inference_type"
            )

        if not isinstance(
            status,
            InferenceStatus,
        ):
            raise InferenceValidationError(
                "Invalid status"
            )

        latency_ms = finite_number(
            latency_ms,
            "latency_ms",
            minimum=0.0,
        )

        if confidence is not None:
            confidence = finite_number(
                confidence,
                "confidence",
                minimum=0.0,
                maximum=1.0,
            )

        input_tokens = nonnegative_int(
            input_tokens,
            "input_tokens",
        )

        output_tokens = nonnegative_int(
            output_tokens,
            "output_tokens",
        )

        batch_size = nonnegative_int(
            batch_size,
            "batch_size",
        )

        if batch_size == 0:
            raise InferenceValidationError(
                "batch_size must be positive"
            )

        estimated_cost_usd = finite_number(
            estimated_cost_usd,
            "estimated_cost_usd",
            minimum=0.0,
        )

        if request_id is None:
            request_id = uuid.uuid4().hex

        request_id = validate_string(
            request_id,
            "request_id",
            128,
        )

        if model_version is not None:
            model_version = validate_string(
                model_version,
                "model_version",
                128,
            )

        if device is not None:
            device = validate_string(
                device,
                "device",
                128,
            )

        if error_type is not None:
            error_type = validate_string(
                error_type,
                "error_type",
            )

        if error_message is not None:
            error_message = validate_string(
                error_message,
                "error_message",
                4000,
            )

        metadata_data = json_safe(
            metadata or {}
        )

        if not isinstance(metadata_data, dict):
            raise InferenceValidationError(
                "metadata must be a dictionary"
            )

        row = InferenceLogORM(
            inference_id=uuid.uuid4().hex,
            request_id=request_id,
            model_name=model_name,
            model_version=model_version,
            inference_type=inference_type.value,
            status=status.value,
            latency_ms=latency_ms,
            confidence=confidence,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            batch_size=batch_size,
            device=device,
            error_type=error_type,
            error_message=error_message,
            metadata_data=metadata_data,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            self._evaluate_alerts(
                session,
                row,
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET INFERENCE
    # --------------------------------------------------------

    def get_inference(
        self,
        inference_id: str,
        *,
        db: Session | None = None,
    ) -> InferenceRecord:
        with self._session(db) as session:
            row = session.get(
                InferenceLogORM,
                inference_id,
            )

            if row is None:
                raise InferenceNotFoundError(
                    "Inference record not found"
                )

            return self._record(row)

    # --------------------------------------------------------
    # LIST INFERENCES
    # --------------------------------------------------------

    def list_inferences(
        self,
        *,
        model_name: str | None = None,
        model_version: str | None = None,
        status: InferenceStatus | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[InferenceRecord]:
        if (
            isinstance(limit, bool)
            or isinstance(offset, bool)
            or not isinstance(limit, int)
            or not isinstance(offset, int)
            or not 1 <= limit <= 1000
            or offset < 0
        ):
            raise InferenceValidationError(
                "Invalid pagination"
            )

        statement = select(
            InferenceLogORM
        )

        if model_name is not None:
            statement = statement.where(
                InferenceLogORM.model_name
                == model_name
            )

        if model_version is not None:
            statement = statement.where(
                InferenceLogORM.model_version
                == model_version
            )

        if status is not None:
            statement = statement.where(
                InferenceLogORM.status
                == status.value
            )

        if since is not None:
            statement = statement.where(
                InferenceLogORM.created_at
                >= since
            )

        statement = (
            statement
            .order_by(
                InferenceLogORM.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST ALERTS
    # --------------------------------------------------------

    def list_alerts(
        self,
        inference_id: str,
        *,
        db: Session | None = None,
    ) -> list[InferenceAlertRecord]:
        statement = (
            select(InferenceAlertORM)
            .where(
                InferenceAlertORM.inference_id
                == inference_id
            )
            .order_by(
                InferenceAlertORM.created_at.asc()
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
    # STATISTICS
    # --------------------------------------------------------

    def calculate_statistics(
        self,
        *,
        model_name: str,
        model_version: str | None = None,
        window_minutes: int = 60,
        db: Session | None = None,
    ) -> InferenceStatistics:
        model_name = validate_string(
            model_name,
            "model_name",
        )

        if (
            isinstance(window_minutes, bool)
            or not isinstance(window_minutes, int)
            or window_minutes < 1
        ):
            raise InferenceValidationError(
                "Invalid window_minutes"
            )

        end_time = utc_now()

        start_time = (
            end_time
            - timedelta(minutes=window_minutes)
        )

        statement = (
            select(InferenceLogORM)
            .where(
                InferenceLogORM.model_name
                == model_name,
                InferenceLogORM.created_at
                >= start_time,
                InferenceLogORM.created_at
                <= end_time,
            )
        )

        if model_version is not None:
            statement = statement.where(
                InferenceLogORM.model_version
                == model_version
            )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

        total = len(rows)

        successful = sum(
            row.status
            == InferenceStatus.SUCCESS.value
            for row in rows
        )

        failed = total - successful

        latencies = [
            row.latency_ms
            for row in rows
        ]

        confidences = [
            row.confidence
            for row in rows
            if row.confidence is not None
        ]

        window_seconds = (
            window_minutes * 60.0
        )

        return InferenceStatistics(
            model_name=model_name,
            model_version=model_version,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            success_rate=(
                successful / total
                if total else 0.0
            ),
            error_rate=(
                failed / total
                if total else 0.0
            ),
            average_latency_ms=(
                sum(latencies) / total
                if total else 0.0
            ),
            p50_latency_ms=percentile(
                latencies, 50
            ),
            p95_latency_ms=percentile(
                latencies, 95
            ),
            p99_latency_ms=percentile(
                latencies, 99
            ),
            minimum_latency_ms=(
                min(latencies)
                if latencies else 0.0
            ),
            maximum_latency_ms=(
                max(latencies)
                if latencies else 0.0
            ),
            average_confidence=(
                sum(confidences)
                / len(confidences)
                if confidences else None
            ),
            total_input_tokens=sum(
                row.input_tokens
                for row in rows
            ),
            total_output_tokens=sum(
                row.output_tokens
                for row in rows
            ),
            total_estimated_cost_usd=sum(
                row.estimated_cost_usd
                for row in rows
            ),
            throughput_requests_per_second=(
                total / window_seconds
            ),
            window_seconds=window_seconds,
        )

    # --------------------------------------------------------
    # ERROR RATE ALERT
    # --------------------------------------------------------

    def check_error_rate(
        self,
        *,
        model_name: str,
        model_version: str | None = None,
        window_minutes: int = 60,
    ) -> dict[str, Any]:
        statistics = self.calculate_statistics(
            model_name=model_name,
            model_version=model_version,
            window_minutes=window_minutes,
        )

        if (
            statistics.total_requests
            < self.config.minimum_requests_for_rate_alert
        ):
            return {
                "status": "insufficient_data",
                "error_rate": statistics.error_rate,
                "total_requests": (
                    statistics.total_requests
                ),
            }

        if (
            statistics.error_rate
            >= self.config.error_rate_critical
        ):
            status = "critical"

        elif (
            statistics.error_rate
            >= self.config.error_rate_warning
        ):
            status = "warning"

        else:
            status = "normal"

        return {
            "status": status,
            "error_rate": statistics.error_rate,
            "total_requests": (
                statistics.total_requests
            ),
            "model_name": model_name,
            "model_version": model_version,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        *,
        model_name: str,
        output_path: str | Path,
        model_version: str | None = None,
        window_minutes: int = 60,
    ) -> Path:
        statistics = self.calculate_statistics(
            model_name=model_name,
            model_version=model_version,
            window_minutes=window_minutes,
        )

        payload = {
            "statistics": asdict(statistics),
            "error_rate_alert": (
                self.check_error_rate(
                    model_name=model_name,
                    model_version=model_version,
                    window_minutes=window_minutes,
                )
            ),
            "generated_at": iso_datetime(
                utc_now()
            ),
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

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        *,
        model_name: str,
        output_path: str | Path,
        model_version: str | None = None,
        limit: int = 1000,
    ) -> Path:
        records = self.list_inferences(
            model_name=model_name,
            model_version=model_version,
            limit=limit,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not records:
            path.write_text(
                "",
                encoding="utf-8",
            )
            return path

        fields = list(
            asdict(records[0]).keys()
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()

            for record in records:
                data = asdict(record)

                data["metadata"] = json.dumps(
                    data["metadata"],
                    ensure_ascii=False,
                )

                writer.writerow(data)

        return path

    # --------------------------------------------------------
    # INFERENCE TIMER
    # --------------------------------------------------------

    @contextmanager
    def track_inference(
        self,
        *,
        model_name: str,
        inference_type: InferenceType,
        model_version: str | None = None,
        request_id: str | None = None,
        device: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Context manager for synchronous inference.

        Example:
            with monitor.track_inference(
                model_name="world_model",
                inference_type=InferenceType.NEXT_STATE,
            ) as telemetry:
                prediction = model.predict(state)
                telemetry["confidence"] = 0.94

        If inference raises an exception,
        the exception is re-raised after logging.
        """

        telemetry: dict[str, Any] = {
            "confidence": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "batch_size": 1,
            "metadata": {},
        }

        start = time.perf_counter()

        status = InferenceStatus.SUCCESS
        error_type = None
        error_message = None

        try:
            yield telemetry

        except Exception as exc:
            status = InferenceStatus.ERROR
            error_type = type(exc).__name__

            # Do not persist exception messages:
            # they may contain sensitive payloads.
            error_message = (
                "Inference execution failed"
            )

            raise

        finally:
            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            self.log_inference(
                model_name=model_name,
                model_version=model_version,
                inference_type=inference_type,
                status=status,
                latency_ms=latency_ms,
                request_id=request_id,
                confidence=telemetry.get(
                    "confidence"
                ),
                input_tokens=telemetry.get(
                    "input_tokens", 0
                ),
                output_tokens=telemetry.get(
                    "output_tokens", 0
                ),
                estimated_cost_usd=telemetry.get(
                    "estimated_cost_usd", 0.0
                ),
                batch_size=telemetry.get(
                    "batch_size", 1
                ),
                device=device,
                error_type=error_type,
                error_message=error_message,
                metadata=telemetry.get(
                    "metadata", {}
                ),
            )


# ============================================================
# FACTORY
# ============================================================

def create_inference_monitor(
    config: InferenceMonitorConfig | None = None,
) -> InferenceMonitor:
    return InferenceMonitor(
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

    monitor = create_inference_monitor()

    model_name = "neuroworld_transformer"

    for index in range(30):
        monitor.log_inference(
            model_name=model_name,
            model_version="v1",
            inference_type=(
                InferenceType.NEXT_STATE
            ),
            status=(
                InferenceStatus.SUCCESS
                if index % 10 != 0
                else InferenceStatus.ERROR
            ),
            latency_ms=(
                100.0 + index * 20
            ),
            confidence=(
                max(0.1, 0.95 - index * 0.02)
            ),
            batch_size=1,
            device="cpu",
            metadata={
                "environment": (
                    "synthetic_physics"
                ),
                "demo": True,
            },
        )

    statistics = monitor.calculate_statistics(
        model_name=model_name,
        model_version="v1",
        window_minutes=60,
    )

    print("=" * 60)
    print("NEUROWORLD AI - INFERENCE MONITOR")
    print("=" * 60)

    print(
        "Total requests:",
        statistics.total_requests,
    )

    print(
        "Success rate:",
        f"{statistics.success_rate:.2%}",
    )

    print(
        "Error rate:",
        f"{statistics.error_rate:.2%}",
    )

    print(
        "Average latency:",
        round(
            statistics.average_latency_ms, 2
        ),
        "ms",
    )

    print(
        "P95 latency:",
        round(
            statistics.p95_latency_ms, 2
        ),
        "ms",
    )

    print(
        "P99 latency:",
        round(
            statistics.p99_latency_ms, 2
        ),
        "ms",
    )

    print(
        "Throughput:",
        round(
            statistics.throughput_requests_per_second,
            4,
        ),
        "requests/sec",
    )

    print(
        "Error-rate alert:",
        monitor.check_error_rate(
            model_name=model_name,
            model_version="v1",
        ),
    )

    output = monitor.export_json(
        model_name=model_name,
        model_version="v1",
        output_path=(
            "experiments/reports/"
            "inference_monitor_report.json"
        ),
    )

    print(
        "Report:",
        output,
    )

    close_database()


if __name__ == "__main__":
    main()
