
"""
NeuroWorld AI
backend/app/database/experiments.py

Persistent experiment tracking for neural world models.

Features
--------
- Register experiments
- Track experiment lifecycle
- Save model and dataset configurations
- Record metrics at training/evaluation steps
- Register checkpoint and report artifacts
- Track reproducibility metadata
- Compare completed experiments
- Store experiment audit events
- Export experiment results to JSON/CSV
- SQLite / PostgreSQL via SQLAlchemy 2.x

This module records experiments. It does not
execute training, evaluate models or run agents.

Dependencies:
    pip install sqlalchemy

Run:
    python -m backend.app.database.experiments
"""

from __future__ import annotations

import copy
import csv
import json
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
    UniqueConstraint,
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


# ============================================================
# ERRORS
# ============================================================

class ExperimentDatabaseError(Exception):
    """Base experiment database error."""


class ExperimentValidationError(
    ExperimentDatabaseError
):
    """Invalid experiment input."""


class ExperimentNotFoundError(
    ExperimentDatabaseError
):
    """Experiment was not found."""


class ExperimentStateError(
    ExperimentDatabaseError
):
    """Invalid experiment lifecycle operation."""


class ExperimentConflictError(
    ExperimentDatabaseError
):
    """Concurrent update or duplicate metric."""


# ============================================================
# ENUMS
# ============================================================

class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentType(str, Enum):
    WORLD_MODEL = "world_model"
    DYNAMICS = "dynamics"
    TOKENIZER = "tokenizer"
    VIDEO_PREDICTION = "video_prediction"
    IMAGINATION = "imagination"
    PLANNING = "planning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"
    CUSTOM = "custom"


class ArtifactType(str, Enum):
    CHECKPOINT = "checkpoint"
    MODEL = "model"
    DATASET = "dataset"
    REPORT = "report"
    PLOT = "plot"
    LOG = "log"
    VIDEO = "video"
    TRAJECTORY = "trajectory"
    OTHER = "other"


# ============================================================
# VALIDATION
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


def identifier(
    value: str,
    field: str,
    limit: int = 255,
) -> str:
    if not isinstance(value, str):
        raise ExperimentValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > limit:
        raise ExperimentValidationError(
            f"Invalid {field}"
        )

    return value


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return iso_datetime(value)

    if isinstance(value, dict):
        return {
            str(key): json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_value(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentValidationError(
                "JSON cannot contain NaN or infinity"
            )
        return value

    if value is None or isinstance(
        value,
        (str, bool, int),
    ):
        return value

    raise ExperimentValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def json_object(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_value(value)

    if not isinstance(result, dict):
        raise ExperimentValidationError(
            f"{field} must be a JSON object"
        )

    return result


def finite_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise ExperimentValidationError(
            f"{field} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ExperimentValidationError(
            f"{field} must be finite"
        )

    return number


# ============================================================
# SQLALCHEMY MODELS
# ============================================================

class ExperimentORM(Base):
    __tablename__ = "research_experiments"

    __table_args__ = (
        Index(
            "ix_research_experiments_model_status",
            "model_name",
            "status",
        ),
    )

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    experiment_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
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
    )

    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    config_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "config",
        JSON,
        nullable=False,
        default=dict,
    )

    reproducibility_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "reproducibility",
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

    summary_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "summary",
        JSON,
        nullable=False,
        default=dict,
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

    metrics: Mapped[
        list["ExperimentMetricORM"]
    ] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    artifacts: Mapped[
        list["ExperimentArtifactORM"]
    ] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["ExperimentEventORM"]
    ] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ExperimentMetricORM(Base):
    __tablename__ = "experiment_metrics"

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "metric_name",
            "step",
            name="uq_experiment_metric_step",
        ),
        Index(
            "ix_experiment_metrics_name_step",
            "metric_name",
            "step",
        ),
    )

    metric_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "research_experiments.experiment_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    metric_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    step: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    experiment: Mapped[ExperimentORM] = relationship(
        back_populates="metrics",
    )


class ExperimentArtifactORM(Base):
    __tablename__ = "experiment_artifacts"

    artifact_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "research_experiments.experiment_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    uri: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    checksum_sha256: Mapped[
        str | None
    ] = mapped_column(
        String(64),
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
    )

    experiment: Mapped[ExperimentORM] = relationship(
        back_populates="artifacts",
    )


class ExperimentEventORM(Base):
    __tablename__ = "experiment_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "research_experiments.experiment_id",
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

    experiment: Mapped[ExperimentORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    name: str
    experiment_type: str
    status: str
    model_name: str
    model_version: str
    dataset_name: str
    config: dict[str, Any]
    reproducibility: dict[str, Any]
    metadata: dict[str, Any]
    summary: dict[str, Any]
    error_message: str | None
    version: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class ExperimentMetric:
    metric_id: str
    experiment_id: str
    metric_name: str
    step: int
    value: float
    recorded_at: str


@dataclass(frozen=True)
class ExperimentArtifact:
    artifact_id: str
    experiment_id: str
    artifact_type: str
    name: str
    uri: str
    checksum_sha256: str | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ExperimentEvent:
    event_id: str
    experiment_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


# ============================================================
# EXPERIMENT REPOSITORY
# ============================================================

class ExperimentRepository:
    """
    Database-backed experiment tracking.

    Passing db=Session leaves transaction
    management to the caller.

    Without db, session_scope() manages
    commit, rollback and close.
    """

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _get(
        session: Session,
        experiment_id: str,
    ) -> ExperimentORM:
        row = session.get(
            ExperimentORM,
            experiment_id,
        )

        if row is None:
            raise ExperimentNotFoundError(
                "Experiment not found"
            )

        return row

    @staticmethod
    def _record(
        row: ExperimentORM,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row.experiment_id,
            name=row.name,
            experiment_type=row.experiment_type,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            dataset_name=row.dataset_name,
            config=copy.deepcopy(row.config_data),
            reproducibility=copy.deepcopy(
                row.reproducibility_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            summary=copy.deepcopy(
                row.summary_data
            ),
            error_message=row.error_message,
            version=row.version,
            created_at=iso_datetime(
                row.created_at
            ),
            started_at=iso_datetime(
                row.started_at
            ),
            completed_at=iso_datetime(
                row.completed_at
            ),
        )

    @staticmethod
    def _metric(
        row: ExperimentMetricORM,
    ) -> ExperimentMetric:
        return ExperimentMetric(
            metric_id=row.metric_id,
            experiment_id=row.experiment_id,
            metric_name=row.metric_name,
            step=row.step,
            value=row.value,
            recorded_at=iso_datetime(
                row.recorded_at
            ),
        )

    @staticmethod
    def _artifact(
        row: ExperimentArtifactORM,
    ) -> ExperimentArtifact:
        return ExperimentArtifact(
            artifact_id=row.artifact_id,
            experiment_id=row.experiment_id,
            artifact_type=row.artifact_type,
            name=row.name,
            uri=row.uri,
            checksum_sha256=row.checksum_sha256,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _event(
        row: ExperimentEventORM,
    ) -> ExperimentEvent:
        return ExperimentEvent(
            event_id=row.event_id,
            experiment_id=row.experiment_id,
            event_type=row.event_type,
            details=copy.deepcopy(
                row.details_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _audit(
        session: Session,
        experiment_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            ExperimentEventORM(
                event_id=uuid.uuid4().hex,
                experiment_id=experiment_id,
                event_type=event_type,
                details_data=json_object(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE EXPERIMENT
    # --------------------------------------------------------

    def create_experiment(
        self,
        *,
        name: str,
        experiment_type: ExperimentType,
        model_name: str,
        dataset_name: str,
        model_version: str = "v1",
        config: dict[str, Any] | None = None,
        reproducibility: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ExperimentRecord:
        name = identifier(
            name, "name"
        )

        model_name = identifier(
            model_name, "model_name"
        )

        dataset_name = identifier(
            dataset_name, "dataset_name"
        )

        model_version = identifier(
            model_version, "model_version", 128
        )

        if not isinstance(
            experiment_type,
            ExperimentType,
        ):
            raise ExperimentValidationError(
                "Invalid experiment type"
            )

        config = json_object(
            config or {}, "config"
        )

        reproducibility = json_object(
            reproducibility or {},
            "reproducibility",
        )

        metadata = json_object(
            metadata or {}, "metadata"
        )

        experiment_id = uuid.uuid4().hex

        row = ExperimentORM(
            experiment_id=experiment_id,
            name=name,
            experiment_type=experiment_type.value,
            status=ExperimentStatus.CREATED.value,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            config_data=config,
            reproducibility_data=reproducibility,
            metadata_data=metadata,
            summary_data={},
            version=1,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)

            self._audit(
                session,
                experiment_id,
                "experiment_created",
                {
                    "name": name,
                    "type": experiment_type.value,
                },
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # READ EXPERIMENT
    # --------------------------------------------------------

    def get_experiment(
        self,
        experiment_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentRecord:
        with self._session(db) as session:
            return self._record(
                self._get(
                    session,
                    experiment_id,
                )
            )

    # --------------------------------------------------------
    # LIFECYCLE TRANSITIONS
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        experiment_id: str,
        expected_version: int,
        required_status: ExperimentStatus,
        new_status: ExperimentStatus,
        values: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        result = session.execute(
            update(ExperimentORM)
            .where(
                ExperimentORM.experiment_id
                == experiment_id,
                ExperimentORM.version
                == expected_version,
                ExperimentORM.status
                == required_status.value,
            )
            .values(
                **(values or {}),
                status=new_status.value,
                version=expected_version + 1,
            )
        )

        if result.rowcount != 1:
            raise ExperimentConflictError(
                "Experiment version or status changed"
            )

        self._audit(
            session,
            experiment_id,
            f"experiment_{new_status.value}",
        )

        session.flush()
        session.expire_all()

        return self._record(
            self._get(
                session,
                experiment_id,
            )
        )

    def start_experiment(
        self,
        experiment_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> ExperimentRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                experiment_id=experiment_id,
                expected_version=expected_version,
                required_status=ExperimentStatus.CREATED,
                new_status=ExperimentStatus.RUNNING,
                values={
                    "started_at": utc_now(),
                },
            )

    def complete_experiment(
        self,
        experiment_id: str,
        *,
        expected_version: int,
        summary: dict[str, Any],
        db: Session | None = None,
    ) -> ExperimentRecord:
        summary = json_object(
            summary, "summary"
        )

        with self._session(db) as session:
            return self._transition(
                session,
                experiment_id=experiment_id,
                expected_version=expected_version,
                required_status=ExperimentStatus.RUNNING,
                new_status=ExperimentStatus.COMPLETED,
                values={
                    "summary_data": summary,
                    "completed_at": utc_now(),
                },
            )

    def fail_experiment(
        self,
        experiment_id: str,
        *,
        expected_version: int,
        error_message: str,
        db: Session | None = None,
    ) -> ExperimentRecord:
        error_message = identifier(
            error_message,
            "error_message",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                experiment_id=experiment_id,
                expected_version=expected_version,
                required_status=ExperimentStatus.RUNNING,
                new_status=ExperimentStatus.FAILED,
                values={
                    "error_message": error_message,
                    "completed_at": utc_now(),
                },
            )

    def cancel_experiment(
        self,
        experiment_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> ExperimentRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                experiment_id=experiment_id,
                expected_version=expected_version,
                required_status=ExperimentStatus.CREATED,
                new_status=ExperimentStatus.CANCELLED,
                values={
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # LOG METRIC
    # --------------------------------------------------------

    def log_metric(
        self,
        experiment_id: str,
        *,
        metric_name: str,
        value: float,
        step: int,
        db: Session | None = None,
    ) -> ExperimentMetric:
        metric_name = identifier(
            metric_name,
            "metric_name",
            128,
        )

        value = finite_number(
            value, "metric value"
        )

        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
        ):
            raise ExperimentValidationError(
                "step must be a nonnegative integer"
            )

        with self._session(db) as session:
            experiment = self._get(
                session,
                experiment_id,
            )

            if experiment.status != (
                ExperimentStatus.RUNNING.value
            ):
                raise ExperimentStateError(
                    "Metrics can only be logged "
                    "while an experiment is running"
                )

            existing = session.scalar(
                select(ExperimentMetricORM.metric_id)
                .where(
                    ExperimentMetricORM.experiment_id
                    == experiment_id,
                    ExperimentMetricORM.metric_name
                    == metric_name,
                    ExperimentMetricORM.step
                    == step,
                )
            )

            if existing is not None:
                raise ExperimentConflictError(
                    "Metric already exists at this step"
                )

            row = ExperimentMetricORM(
                metric_id=uuid.uuid4().hex,
                experiment_id=experiment_id,
                metric_name=metric_name,
                step=step,
                value=value,
                recorded_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._metric(row)

    # --------------------------------------------------------
    # LOG MULTIPLE METRICS
    # --------------------------------------------------------

    def log_metrics(
        self,
        experiment_id: str,
        *,
        metrics: dict[str, float],
        step: int,
        db: Session | None = None,
    ) -> list[ExperimentMetric]:
        if not isinstance(metrics, dict):
            raise ExperimentValidationError(
                "metrics must be a dictionary"
            )

        # All writes use the same transaction.
        with self._session(db) as session:
            return [
                self.log_metric(
                    experiment_id,
                    metric_name=name,
                    value=value,
                    step=step,
                    db=session,
                )
                for name, value in metrics.items()
            ]

    # --------------------------------------------------------
    # LIST METRICS
    # --------------------------------------------------------

    def list_metrics(
        self,
        experiment_id: str,
        *,
        metric_name: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[ExperimentMetric]:
        if not 1 <= limit <= 10000 or offset < 0:
            raise ExperimentValidationError(
                "Invalid pagination"
            )

        statement = select(
            ExperimentMetricORM
        ).where(
            ExperimentMetricORM.experiment_id
            == experiment_id
        )

        if metric_name is not None:
            statement = statement.where(
                ExperimentMetricORM.metric_name
                == metric_name
            )

        statement = (
            statement
            .order_by(
                ExperimentMetricORM.step.asc(),
                ExperimentMetricORM.metric_name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._metric(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # REGISTER ARTIFACT
    # --------------------------------------------------------

    def register_artifact(
        self,
        experiment_id: str,
        *,
        artifact_type: ArtifactType,
        name: str,
        uri: str,
        checksum_sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ExperimentArtifact:
        if not isinstance(
            artifact_type,
            ArtifactType,
        ):
            raise ExperimentValidationError(
                "Invalid artifact type"
            )

        name = identifier(
            name, "artifact name"
        )

        uri = identifier(
            uri, "artifact URI", 4096
        )

        if checksum_sha256 is not None:
            checksum_sha256 = (
                checksum_sha256.lower()
            )

            if (
                len(checksum_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in checksum_sha256
                )
            ):
                raise ExperimentValidationError(
                    "Invalid SHA256 checksum"
                )

        metadata = json_object(
            metadata or {},
            "artifact metadata",
        )

        with self._session(db) as session:
            experiment = self._get(
                session,
                experiment_id,
            )

            if experiment.status not in {
                ExperimentStatus.CREATED.value,
                ExperimentStatus.RUNNING.value,
                ExperimentStatus.COMPLETED.value,
            }:
                raise ExperimentStateError(
                    "Cannot register artifacts "
                    "for this experiment status"
                )

            row = ExperimentArtifactORM(
                artifact_id=uuid.uuid4().hex,
                experiment_id=experiment_id,
                artifact_type=artifact_type.value,
                name=name,
                uri=uri,
                checksum_sha256=checksum_sha256,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                experiment_id,
                "artifact_registered",
                {
                    "artifact_id": row.artifact_id,
                    "type": artifact_type.value,
                },
            )

            session.flush()

            return self._artifact(row)

    # --------------------------------------------------------
    # LIST ARTIFACTS
    # --------------------------------------------------------

    def list_artifacts(
        self,
        experiment_id: str,
        *,
        limit: int = 1000,
        db: Session | None = None,
    ) -> list[ExperimentArtifact]:
        if not 1 <= limit <= 10000:
            raise ExperimentValidationError(
                "Invalid limit"
            )

        statement = (
            select(ExperimentArtifactORM)
            .where(
                ExperimentArtifactORM.experiment_id
                == experiment_id
            )
            .order_by(
                ExperimentArtifactORM.created_at.asc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._artifact(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST EXPERIMENTS
    # --------------------------------------------------------

    def list_experiments(
        self,
        *,
        model_name: str | None = None,
        experiment_type: ExperimentType | None = None,
        status: ExperimentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[ExperimentRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ExperimentValidationError(
                "Invalid pagination"
            )

        statement = select(
            ExperimentORM
        )

        if model_name is not None:
            statement = statement.where(
                ExperimentORM.model_name
                == model_name
            )

        if experiment_type is not None:
            statement = statement.where(
                ExperimentORM.experiment_type
                == experiment_type.value
            )

        if status is not None:
            statement = statement.where(
                ExperimentORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                ExperimentORM.created_at.desc(),
                ExperimentORM.experiment_id.desc(),
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
    # LATEST METRIC
    # --------------------------------------------------------

    def get_latest_metric(
        self,
        experiment_id: str,
        metric_name: str,
        *,
        db: Session | None = None,
    ) -> ExperimentMetric | None:
        statement = (
            select(ExperimentMetricORM)
            .where(
                ExperimentMetricORM.experiment_id
                == experiment_id,
                ExperimentMetricORM.metric_name
                == metric_name,
            )
            .order_by(
                ExperimentMetricORM.step.desc()
            )
            .limit(1)
        )

        with self._session(db) as session:
            row = session.scalar(
                statement
            )

            return (
                self._metric(row)
                if row is not None
                else None
            )

    # --------------------------------------------------------
    # COMPARE EXPERIMENTS
    # --------------------------------------------------------

    def compare_experiments(
        self,
        experiment_ids: list[str],
        *,
        metric_name: str,
        higher_is_better: bool = True,
        db: Session | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compare the latest recorded value
        of a metric across completed experiments.

        Dataset, metric definition, and
        evaluation protocol must be comparable.
        """

        comparisons = []

        with self._session(db) as session:
            for experiment_id in experiment_ids:
                experiment = self._get(
                    session,
                    experiment_id,
                )

                if experiment.status != (
                    ExperimentStatus.COMPLETED.value
                ):
                    continue

                metric = self.get_latest_metric(
                    experiment_id,
                    metric_name,
                    db=session,
                )

                if metric is None:
                    continue

                comparisons.append(
                    {
                        "experiment_id": experiment_id,
                        "experiment_name": (
                            experiment.name
                        ),
                        "model_name": (
                            experiment.model_name
                        ),
                        "dataset_name": (
                            experiment.dataset_name
                        ),
                        "metric_name": metric_name,
                        "metric_value": metric.value,
                        "step": metric.step,
                    }
                )

        comparisons.sort(
            key=lambda item: item["metric_value"],
            reverse=higher_is_better,
        )

        for rank, item in enumerate(
            comparisons,
            start=1,
        ):
            item["rank"] = rank

        return comparisons

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def list_events(
        self,
        experiment_id: str,
        *,
        limit: int = 1000,
        db: Session | None = None,
    ) -> list[ExperimentEvent]:
        if not 1 <= limit <= 10000:
            raise ExperimentValidationError(
                "Invalid limit"
            )

        statement = (
            select(ExperimentEventORM)
            .where(
                ExperimentEventORM.experiment_id
                == experiment_id
            )
            .order_by(
                ExperimentEventORM.created_at.asc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._event(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        experiment_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        payload = {
            "experiment": asdict(
                self.get_experiment(
                    experiment_id,
                    db=db,
                )
            ),
            "metrics": [
                asdict(metric)
                for metric in self.list_metrics(
                    experiment_id,
                    limit=10000,
                    db=db,
                )
            ],
            "artifacts": [
                asdict(artifact)
                for artifact in self.list_artifacts(
                    experiment_id,
                    db=db,
                )
            ],
            "events": [
                asdict(event)
                for event in self.list_events(
                    experiment_id,
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
    # EXPORT METRICS CSV
    # --------------------------------------------------------

    def export_metrics_csv(
        self,
        experiment_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        metrics = self.list_metrics(
            experiment_id,
            limit=10000,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            ExperimentMetric.__dataclass_fields__.keys()
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

            for metric in metrics:
                writer.writerow(
                    asdict(metric)
                )

        return path


# ============================================================
# FACTORY
# ============================================================

def create_experiment_repository() -> (
    ExperimentRepository
):
    return ExperimentRepository()


# ============================================================
# DEMO
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    create_database_tables()

    repository = create_experiment_repository()

    experiment = repository.create_experiment(
        name="Neural World Model Training Demo",
        experiment_type=ExperimentType.WORLD_MODEL,
        model_name="transformer_world_model",
        model_version="v1.0",
        dataset_name="synthetic_navigation",
        config={
            "epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.001,
            "latent_dimension": 256,
        },
        reproducibility={
            "seed": 42,
            "framework": "pytorch",
            "dataset_version": "synthetic-v1",
        },
    )

    experiment = repository.start_experiment(
        experiment.experiment_id,
        expected_version=experiment.version,
    )

    # Illustrative metrics, not actual training.
    demo_metrics = [
        (1, 0.42, 0.38),
        (2, 0.31, 0.29),
        (3, 0.24, 0.23),
    ]

    for epoch, train_loss, val_loss in demo_metrics:
        repository.log_metrics(
            experiment.experiment_id,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            step=epoch,
        )

    repository.register_artifact(
        experiment.experiment_id,
        artifact_type=ArtifactType.REPORT,
        name="demo_training_report",
        uri="experiments/reports/demo_training.json",
        metadata={
            "illustrative": True,
        },
    )

    experiment = repository.complete_experiment(
        experiment.experiment_id,
        expected_version=experiment.version,
        summary={
            "epochs": 3,
            "final_train_loss": 0.24,
            "final_val_loss": 0.23,
            "illustrative": True,
        },
    )

    report_path = repository.export_json(
        experiment.experiment_id,
        "experiments/reports/experiment_history.json",
    )

    csv_path = repository.export_metrics_csv(
        experiment.experiment_id,
        "experiments/reports/experiment_metrics.csv",
    )

    print("Experiment:", experiment.experiment_id)
    print("Status:", experiment.status)
    print("JSON:", report_path)
    print("CSV:", csv_path)

    close_database()


if __name__ == "__main__":
    main()
