
"""
NeuroWorld AI
backend/app/database/evaluation.py

Persistent Evaluation and Benchmark Repository.

Features:
- Create evaluation runs
- Store model and dataset identifiers
- Record evaluation metrics
- Save per-horizon metrics
- Track evaluation lifecycle
- Store benchmark scores
- Compare model evaluation results
- Persist experiment references
- Record evaluation events
- Export results to JSON and CSV
- SQLite and PostgreSQL support

Dependencies:
    pip install sqlalchemy

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module stores metrics. It does not
calculate or independently verify them.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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
    update,
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


logger = logging.getLogger(
    "neuroworld.database.evaluation"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class EvaluationDatabaseError(Exception):
    """Base evaluation database error."""


class EvaluationValidationError(
    EvaluationDatabaseError
):
    """Invalid evaluation data."""


class EvaluationNotFoundError(
    EvaluationDatabaseError
):
    """Evaluation or benchmark not found."""


class EvaluationStateError(
    EvaluationDatabaseError
):
    """Invalid evaluation lifecycle transition."""


class EvaluationVersionConflict(
    EvaluationDatabaseError
):
    """Concurrent evaluation update conflict."""


# ============================================================
# ENUMS
# ============================================================

class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationType(str, Enum):
    WORLD_CONSISTENCY = "world_consistency"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    PHYSICAL_CONSISTENCY = "physical_consistency"
    OBJECT_PERMANENCE = "object_permanence"
    ACTION_CONTROLLABILITY = "action_controllability"
    TRAJECTORY_ACCURACY = "trajectory_accuracy"
    FUTURE_PREDICTION = "future_prediction"
    COUNTERFACTUAL_ACCURACY = "counterfactual_accuracy"
    PLANNING_SUCCESS = "planning_success"
    CALIBRATION = "calibration"
    ROBUSTNESS = "robustness"
    GENERALIZATION = "generalization"
    BENCHMARK = "benchmark"
    CUSTOM = "custom"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def datetime_string(
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


def validate_identifier(
    value: str,
    name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise EvaluationValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise EvaluationValidationError(
            f"Invalid {name}"
        )

    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return datetime_string(value)

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
            raise EvaluationValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise EvaluationValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise EvaluationValidationError(
            f"{name} must be a dictionary"
        )

    return result


def validate_score(
    value: Any,
    name: str = "score",
) -> float:
    if isinstance(value, bool):
        raise EvaluationValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise EvaluationValidationError(
            f"{name} must be finite"
        )

    if not 0.0 <= number <= 1.0:
        raise EvaluationValidationError(
            f"{name} must be between 0 and 1"
        )

    return number


def validate_metrics(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Accept structured metric values.

    Examples:
        {
            "rmse": 0.12,
            "accuracy": 0.91,
            "horizon_errors": [0.1, 0.2]
        }

    Metric values are not forced into [0, 1].
    Only normalized overall scores are.
    """

    return validate_json_dict(
        metrics,
        "metrics",
    )


# ============================================================
# ORM MODELS
# ============================================================

class EvaluationORM(Base):
    __tablename__ = "model_evaluations"

    __table_args__ = (
        Index(
            "ix_model_evaluations_model_type",
            "model_name",
            "evaluation_type",
        ),
        Index(
            "ix_model_evaluations_status_created",
            "status",
            "created_at",
        ),
    )

    evaluation_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    experiment_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    evaluation_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    metrics_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "metrics",
        JSON,
        nullable=False,
        default=dict,
    )

    configuration_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "configuration",
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    overall_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    passed: Mapped[
        bool | None
    ] = mapped_column(
        nullable=True,
    )

    sample_count: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    horizon: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    error_message: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    horizon_metrics: Mapped[
        list["EvaluationHorizonORM"]
    ] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["EvaluationEventORM"]
    ] = relationship(
        back_populates="evaluation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EvaluationHorizonORM(Base):
    __tablename__ = "evaluation_horizon_metrics"

    __table_args__ = (
        Index(
            "ix_eval_horizon_eval_step",
            "evaluation_id",
            "horizon_step",
            unique=True,
        ),
    )

    horizon_metric_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    evaluation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "model_evaluations.evaluation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    horizon_step: Mapped[int] = mapped_column(
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

    evaluation: Mapped[EvaluationORM] = relationship(
        back_populates="horizon_metrics",
    )


class EvaluationEventORM(Base):
    __tablename__ = "evaluation_events"

    __table_args__ = (
        Index(
            "ix_evaluation_events_eval_created",
            "evaluation_id",
            "created_at",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    evaluation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "model_evaluations.evaluation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    details_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "details",
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    evaluation: Mapped[EvaluationORM] = relationship(
        back_populates="events",
    )


class BenchmarkORM(Base):
    __tablename__ = "model_benchmarks"

    __table_args__ = (
        Index(
            "ix_model_benchmarks_model_created",
            "model_name",
            "created_at",
        ),
    )

    benchmark_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    benchmark_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    evaluation_ids_data: Mapped[
        list[str]
    ] = mapped_column(
        "evaluation_ids",
        JSON,
        nullable=False,
        default=list,
    )

    category_scores_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "category_scores",
        JSON,
        nullable=False,
        default=dict,
    )

    overall_score: Mapped[float] = mapped_column(
        Float,
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


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    experiment_id: str | None
    model_name: str
    model_version: str
    dataset_name: str
    evaluation_type: str
    status: str
    metrics: dict[str, Any]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    overall_score: float | None
    passed: bool | None
    sample_count: int | None
    horizon: int | None
    error_message: str | None
    version: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class HorizonMetricRecord:
    horizon_metric_id: str
    evaluation_id: str
    horizon_step: int
    metrics: dict[str, Any]


@dataclass(frozen=True)
class EvaluationEventRecord:
    event_id: str
    evaluation_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_id: str
    benchmark_name: str
    model_name: str
    model_version: str
    evaluation_ids: list[str]
    category_scores: dict[str, Any]
    overall_score: float
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ModelComparison:
    evaluation_type: str
    dataset_name: str
    rankings: list[dict[str, Any]]
    best_evaluation_id: str | None


# ============================================================
# REPOSITORY
# ============================================================

class EvaluationRepository:
    """
    Stores results produced by evaluation modules.

    The caller owns the transaction when passing
    a SQLAlchemy Session.

    Otherwise session_scope() commits or rolls back.
    """

    @staticmethod
    def _session(
        db: Session | None,
    ):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _to_record(
        row: EvaluationORM,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            evaluation_id=row.evaluation_id,
            experiment_id=row.experiment_id,
            model_name=row.model_name,
            model_version=row.model_version,
            dataset_name=row.dataset_name,
            evaluation_type=row.evaluation_type,
            status=row.status,
            metrics=copy.deepcopy(
                row.metrics_data
            ),
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            overall_score=row.overall_score,
            passed=row.passed,
            sample_count=row.sample_count,
            horizon=row.horizon,
            error_message=row.error_message,
            version=row.version,
            created_at=datetime_string(
                row.created_at
            ),
            started_at=datetime_string(
                row.started_at
            ),
            completed_at=datetime_string(
                row.completed_at
            ),
        )

    @staticmethod
    def _to_horizon(
        row: EvaluationHorizonORM,
    ) -> HorizonMetricRecord:
        return HorizonMetricRecord(
            horizon_metric_id=row.horizon_metric_id,
            evaluation_id=row.evaluation_id,
            horizon_step=row.horizon_step,
            metrics=copy.deepcopy(
                row.metrics_data
            ),
        )

    @staticmethod
    def _to_event(
        row: EvaluationEventORM,
    ) -> EvaluationEventRecord:
        return EvaluationEventRecord(
            event_id=row.event_id,
            evaluation_id=row.evaluation_id,
            event_type=row.event_type,
            details=copy.deepcopy(
                row.details_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    @staticmethod
    def _to_benchmark(
        row: BenchmarkORM,
    ) -> BenchmarkRecord:
        return BenchmarkRecord(
            benchmark_id=row.benchmark_id,
            benchmark_name=row.benchmark_name,
            model_name=row.model_name,
            model_version=row.model_version,
            evaluation_ids=copy.deepcopy(
                row.evaluation_ids_data
            ),
            category_scores=copy.deepcopy(
                row.category_scores_data
            ),
            overall_score=row.overall_score,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    @staticmethod
    def _get_evaluation(
        session: Session,
        evaluation_id: str,
    ) -> EvaluationORM:
        row = session.get(
            EvaluationORM,
            evaluation_id,
        )

        if row is None:
            raise EvaluationNotFoundError(
                "Evaluation not found"
            )

        return row

    @staticmethod
    def _add_event(
        session: Session,
        *,
        evaluation_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            EvaluationEventORM(
                event_id=uuid.uuid4().hex,
                evaluation_id=evaluation_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE EVALUATION
    # --------------------------------------------------------

    def create_evaluation(
        self,
        *,
        model_name: str,
        dataset_name: str,
        evaluation_type: EvaluationType,
        model_version: str = "unknown",
        experiment_id: str | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> EvaluationRecord:
        model_name = validate_identifier(
            model_name,
            "model_name",
            255,
        )

        dataset_name = validate_identifier(
            dataset_name,
            "dataset_name",
            255,
        )

        model_version = validate_identifier(
            model_version,
            "model_version",
        )

        if not isinstance(
            evaluation_type,
            EvaluationType,
        ):
            raise EvaluationValidationError(
                "Invalid evaluation type"
            )

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        if experiment_id is not None:
            experiment_id = validate_identifier(
                experiment_id,
                "experiment_id",
                64,
            )

        evaluation_id = uuid.uuid4().hex

        row = EvaluationORM(
            evaluation_id=evaluation_id,
            experiment_id=experiment_id,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            evaluation_type=evaluation_type.value,
            status=EvaluationStatus.PENDING.value,
            metrics_data={},
            configuration_data=configuration,
            metadata_data=metadata,
            version=1,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)

            self._add_event(
                session,
                evaluation_id=evaluation_id,
                event_type="evaluation_created",
                details={
                    "evaluation_type": (
                        evaluation_type.value
                    ),
                },
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # GET EVALUATION
    # --------------------------------------------------------

    def get_evaluation(
        self,
        evaluation_id: str,
        *,
        db: Session | None = None,
    ) -> EvaluationRecord:
        with self._session(db) as session:
            return self._to_record(
                self._get_evaluation(
                    session,
                    evaluation_id,
                )
            )

    # --------------------------------------------------------
    # VERSIONED STATUS UPDATE
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        evaluation_id: str,
        expected_version: int,
        required_status: EvaluationStatus,
        new_status: EvaluationStatus,
        values: dict[str, Any] | None = None,
        event_details: dict[str, Any] | None = None,
    ) -> EvaluationRecord:
        result = session.execute(
            update(EvaluationORM)
            .where(
                EvaluationORM.evaluation_id
                == evaluation_id,
                EvaluationORM.version
                == expected_version,
                EvaluationORM.status
                == required_status.value,
            )
            .values(
                **(values or {}),
                status=new_status.value,
                version=expected_version + 1,
            )
        )

        if result.rowcount != 1:
            raise EvaluationVersionConflict(
                "Evaluation version or status changed"
            )

        self._add_event(
            session,
            evaluation_id=evaluation_id,
            event_type=(
                f"evaluation_{new_status.value}"
            ),
            details=event_details,
        )

        session.flush()
        session.expire_all()

        return self._to_record(
            self._get_evaluation(
                session,
                evaluation_id,
            )
        )

    # --------------------------------------------------------
    # START EVALUATION
    # --------------------------------------------------------

    def start_evaluation(
        self,
        evaluation_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> EvaluationRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                evaluation_id=evaluation_id,
                expected_version=expected_version,
                required_status=EvaluationStatus.PENDING,
                new_status=EvaluationStatus.RUNNING,
                values={
                    "started_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # COMPLETE EVALUATION
    # --------------------------------------------------------

    def complete_evaluation(
        self,
        evaluation_id: str,
        *,
        expected_version: int,
        metrics: dict[str, Any],
        overall_score: float,
        passed: bool,
        sample_count: int | None = None,
        horizon: int | None = None,
        horizon_metrics: list[
            dict[str, Any]
        ] | None = None,
        db: Session | None = None,
    ) -> EvaluationRecord:
        metrics = validate_metrics(
            metrics
        )

        overall_score = validate_score(
            overall_score,
            "overall_score",
        )

        if not isinstance(passed, bool):
            raise EvaluationValidationError(
                "passed must be boolean"
            )

        if sample_count is not None and sample_count < 1:
            raise EvaluationValidationError(
                "sample_count must be positive"
            )

        if horizon is not None and horizon < 1:
            raise EvaluationValidationError(
                "horizon must be positive"
            )

        prepared_horizons = []
        seen_steps = set()

        for item in horizon_metrics or []:
            if not isinstance(item, dict):
                raise EvaluationValidationError(
                    "Each horizon entry must be a dict"
                )

            step = item.get(
                "horizon_step"
            )

            if (
                not isinstance(step, int)
                or isinstance(step, bool)
                or step < 1
                or step in seen_steps
            ):
                raise EvaluationValidationError(
                    "Invalid or duplicate horizon step"
                )

            if horizon is not None and step > horizon:
                raise EvaluationValidationError(
                    "Horizon step exceeds horizon"
                )

            seen_steps.add(step)

            prepared_horizons.append(
                (
                    step,
                    validate_metrics(
                        item.get("metrics", {})
                    ),
                )
            )

        with self._session(db) as session:
            record = self._transition(
                session,
                evaluation_id=evaluation_id,
                expected_version=expected_version,
                required_status=EvaluationStatus.RUNNING,
                new_status=EvaluationStatus.COMPLETED,
                values={
                    "metrics_data": metrics,
                    "overall_score": overall_score,
                    "passed": passed,
                    "sample_count": sample_count,
                    "horizon": horizon,
                    "completed_at": utc_now(),
                },
                event_details={
                    "overall_score": overall_score,
                    "passed": passed,
                },
            )

            for step, step_metrics in prepared_horizons:
                session.add(
                    EvaluationHorizonORM(
                        horizon_metric_id=(
                            uuid.uuid4().hex
                        ),
                        evaluation_id=evaluation_id,
                        horizon_step=step,
                        metrics_data=step_metrics,
                    )
                )

            session.flush()

            return record

    # --------------------------------------------------------
    # FAIL EVALUATION
    # --------------------------------------------------------

    def fail_evaluation(
        self,
        evaluation_id: str,
        *,
        expected_version: int,
        error_message: str,
        db: Session | None = None,
    ) -> EvaluationRecord:
        if not isinstance(
            error_message,
            str,
        ) or not error_message.strip():
            raise EvaluationValidationError(
                "error_message is required"
            )

        with self._session(db) as session:
            return self._transition(
                session,
                evaluation_id=evaluation_id,
                expected_version=expected_version,
                required_status=EvaluationStatus.RUNNING,
                new_status=EvaluationStatus.FAILED,
                values={
                    "error_message": error_message,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # CANCEL PENDING EVALUATION
    # --------------------------------------------------------

    def cancel_evaluation(
        self,
        evaluation_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> EvaluationRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                evaluation_id=evaluation_id,
                expected_version=expected_version,
                required_status=EvaluationStatus.PENDING,
                new_status=EvaluationStatus.CANCELLED,
                values={
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # LIST EVALUATIONS
    # --------------------------------------------------------

    def list_evaluations(
        self,
        *,
        model_name: str | None = None,
        evaluation_type: EvaluationType | None = None,
        status: EvaluationStatus | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[EvaluationRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise EvaluationValidationError(
                "Invalid pagination"
            )

        statement = select(
            EvaluationORM
        )

        if model_name is not None:
            statement = statement.where(
                EvaluationORM.model_name == model_name
            )

        if evaluation_type is not None:
            statement = statement.where(
                EvaluationORM.evaluation_type
                == evaluation_type.value
            )

        if status is not None:
            statement = statement.where(
                EvaluationORM.status == status.value
            )

        if experiment_id is not None:
            statement = statement.where(
                EvaluationORM.experiment_id
                == experiment_id
            )

        statement = (
            statement
            .order_by(
                EvaluationORM.created_at.desc(),
                EvaluationORM.evaluation_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._to_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # GET HORIZON METRICS
    # --------------------------------------------------------

    def get_horizon_metrics(
        self,
        evaluation_id: str,
        *,
        db: Session | None = None,
    ) -> list[HorizonMetricRecord]:
        statement = (
            select(EvaluationHorizonORM)
            .where(
                EvaluationHorizonORM.evaluation_id
                == evaluation_id
            )
            .order_by(
                EvaluationHorizonORM.horizon_step.asc()
            )
        )

        with self._session(db) as session:
            return [
                self._to_horizon(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # GET EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        evaluation_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[EvaluationEventRecord]:
        if not 1 <= limit <= 1000:
            raise EvaluationValidationError(
                "Invalid limit"
            )

        statement = (
            select(EvaluationEventORM)
            .where(
                EvaluationEventORM.evaluation_id
                == evaluation_id
            )
            .order_by(
                EvaluationEventORM.created_at.asc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._to_event(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # SAVE BENCHMARK
    # --------------------------------------------------------

    def create_benchmark(
        self,
        *,
        benchmark_name: str,
        model_name: str,
        model_version: str,
        evaluation_ids: list[str],
        category_scores: dict[str, float],
        overall_score: float,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> BenchmarkRecord:
        benchmark_name = validate_identifier(
            benchmark_name,
            "benchmark_name",
            255,
        )

        model_name = validate_identifier(
            model_name,
            "model_name",
            255,
        )

        model_version = validate_identifier(
            model_version,
            "model_version",
        )

        overall_score = validate_score(
            overall_score,
        )

        if not isinstance(evaluation_ids, list):
            raise EvaluationValidationError(
                "evaluation_ids must be a list"
            )

        ids = [
            validate_identifier(
                item,
                "evaluation_id",
                64,
            )
            for item in evaluation_ids
        ]

        if len(set(ids)) != len(ids):
            raise EvaluationValidationError(
                "Duplicate evaluation IDs"
            )

        if not isinstance(category_scores, dict):
            raise EvaluationValidationError(
                "category_scores must be a dictionary"
            )

        scores = {
            validate_identifier(
                name,
                "category name",
            ): validate_score(
                score,
                name,
            )
            for name, score in category_scores.items()
        }

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            if ids:
                evaluations = session.scalars(
                    select(EvaluationORM).where(
                        EvaluationORM.evaluation_id.in_(ids)
                    )
                ).all()

                found = {
                    row.evaluation_id: row
                    for row in evaluations
                }

                if len(found) != len(ids):
                    raise EvaluationNotFoundError(
                        "One or more evaluations not found"
                    )

                for evaluation in evaluations:
                    if (
                        evaluation.status
                        != EvaluationStatus.COMPLETED.value
                        or evaluation.model_name
                        != model_name
                        or evaluation.model_version
                        != model_version
                    ):
                        raise EvaluationValidationError(
                            "Benchmark evaluations must "
                            "be completed and belong to "
                            "the selected model/version"
                        )

            row = BenchmarkORM(
                benchmark_id=uuid.uuid4().hex,
                benchmark_name=benchmark_name,
                model_name=model_name,
                model_version=model_version,
                evaluation_ids_data=ids,
                category_scores_data=scores,
                overall_score=overall_score,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._to_benchmark(row)

    # --------------------------------------------------------
    # GET BENCHMARK
    # --------------------------------------------------------

    def get_benchmark(
        self,
        benchmark_id: str,
        *,
        db: Session | None = None,
    ) -> BenchmarkRecord:
        with self._session(db) as session:
            row = session.get(
                BenchmarkORM,
                benchmark_id,
            )

            if row is None:
                raise EvaluationNotFoundError(
                    "Benchmark not found"
                )

            return self._to_benchmark(row)

    # --------------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------------

    def compare_models(
        self,
        *,
        evaluation_type: EvaluationType,
        dataset_name: str,
        model_names: list[str] | None = None,
        db: Session | None = None,
    ) -> ModelComparison:
        """
        Rank completed evaluation runs.

        Scores are comparable only when
        datasets, evaluation configuration,
        and metric definitions are compatible.
        """

        statement = (
            select(EvaluationORM)
            .where(
                EvaluationORM.evaluation_type
                == evaluation_type.value,
                EvaluationORM.dataset_name
                == dataset_name,
                EvaluationORM.status
                == EvaluationStatus.COMPLETED.value,
                EvaluationORM.overall_score.is_not(None),
            )
        )

        if model_names:
            statement = statement.where(
                EvaluationORM.model_name.in_(
                    model_names
                )
            )

        statement = statement.order_by(
            EvaluationORM.overall_score.desc(),
            EvaluationORM.created_at.desc(),
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            rankings = [
                {
                    "rank": index,
                    "evaluation_id": row.evaluation_id,
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "overall_score": row.overall_score,
                    "passed": row.passed,
                }
                for index, row in enumerate(
                    rows,
                    start=1,
                )
            ]

            return ModelComparison(
                evaluation_type=evaluation_type.value,
                dataset_name=dataset_name,
                rankings=rankings,
                best_evaluation_id=(
                    rows[0].evaluation_id
                    if rows
                    else None
                ),
            )

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        evaluation_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        payload = {
            "evaluation": asdict(
                self.get_evaluation(
                    evaluation_id,
                    db=db,
                )
            ),
            "horizon_metrics": [
                asdict(item)
                for item in self.get_horizon_metrics(
                    evaluation_id,
                    db=db,
                )
            ],
            "events": [
                asdict(item)
                for item in self.list_events(
                    evaluation_id,
                    limit=1000,
                    db=db,
                )
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

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        evaluation_ids: list[str],
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        records = [
            self.get_evaluation(
                evaluation_id,
                db=db,
            )
            for evaluation_id in evaluation_ids
        ]

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            EvaluationRecord.__dataclass_fields__.keys()
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

            for record in records:
                row = asdict(record)

                for field in (
                    "metrics",
                    "configuration",
                    "metadata",
                ):
                    row[field] = json.dumps(
                        row[field],
                        allow_nan=False,
                    )

                writer.writerow(row)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_evaluation_repository() -> (
    EvaluationRepository
):
    return EvaluationRepository()


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    logging.basicConfig(
        level=logging.INFO
    )

    create_database_tables()

    repository = create_evaluation_repository()

    evaluation = repository.create_evaluation(
        model_name="neuroworld_transformer",
        model_version="v1.0",
        dataset_name="synthetic_navigation",
        evaluation_type=(
            EvaluationType.TRAJECTORY_ACCURACY
        ),
        configuration={
            "horizon": 12,
            "sample_count": 128,
        },
    )

    evaluation = repository.start_evaluation(
        evaluation.evaluation_id,
        expected_version=evaluation.version,
    )

    # Illustrative metrics only.
    # Real metrics should be supplied by
    # evaluation/trajectory_accuracy.py.
    evaluation = repository.complete_evaluation(
        evaluation.evaluation_id,
        expected_version=evaluation.version,
        metrics={
            "trajectory_rmse": 0.12,
            "position_ade": 0.09,
            "position_fde": 0.15,
            "drift_score": 0.93,
        },
        overall_score=0.91,
        passed=True,
        sample_count=128,
        horizon=12,
        horizon_metrics=[
            {
                "horizon_step": 1,
                "metrics": {
                    "rmse": 0.03,
                    "accuracy": 0.97,
                },
            },
            {
                "horizon_step": 12,
                "metrics": {
                    "rmse": 0.12,
                    "accuracy": 0.89,
                },
            },
        ],
    )

    benchmark = repository.create_benchmark(
        benchmark_name="NeuroWorld Synthetic Benchmark",
        model_name="neuroworld_transformer",
        model_version="v1.0",
        evaluation_ids=[
            evaluation.evaluation_id
        ],
        category_scores={
            "trajectory_accuracy": 0.91,
        },
        overall_score=0.91,
    )

    print("=" * 60)
    print("NEUROWORLD AI - EVALUATION DATABASE")
    print("=" * 60)

    print(
        "Evaluation ID:",
        evaluation.evaluation_id,
    )

    print(
        "Evaluation status:",
        evaluation.status,
    )

    print(
        "Overall score:",
        evaluation.overall_score,
    )

    print(
        "Benchmark ID:",
        benchmark.benchmark_id,
    )

    report = repository.export_json(
        evaluation.evaluation_id,
        "experiments/reports/evaluation_record.json",
    )

    print(
        "Report:",
        report,
    )

    close_database()


if __name__ == "__main__":
    main()
