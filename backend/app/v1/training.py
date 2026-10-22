
"""
NeuroWorld AI
backend/app/database/training.py

Persistent AI training job management.

Features:
- Create and manage model training jobs
- Track training status and configuration
- Record training and validation metrics
- Track epochs, learning rates, and losses
- Register model checkpoints
- Record training artifacts
- Track early stopping and failures
- Optimistic concurrency control
- Training audit history
- JSON/CSV report exports

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Training execution belongs in training_service.py.
This module persists training metadata only.
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
# EXCEPTIONS
# ============================================================

class TrainingDatabaseError(Exception):
    """Base training database exception."""


class TrainingValidationError(
    TrainingDatabaseError
):
    """Invalid training input."""


class TrainingNotFoundError(
    TrainingDatabaseError
):
    """Training job was not found."""


class TrainingStateError(
    TrainingDatabaseError
):
    """Invalid training state transition."""


class TrainingConflictError(
    TrainingDatabaseError
):
    """Concurrent modification or duplicate data."""


# ============================================================
# ENUMS
# ============================================================

class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    EARLY_STOPPED = "early_stopped"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingType(str, Enum):
    WORLD_MODEL = "world_model"
    TOKENIZER = "tokenizer"
    DYNAMICS = "dynamics"
    VIDEO_PREDICTION = "video_prediction"
    REWARD_MODEL = "reward_model"
    POLICY = "policy"
    JOINT = "joint"
    FINE_TUNING = "fine_tuning"
    CUSTOM = "custom"


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


def validate_string(
    value: str,
    field: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str):
        raise TrainingValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise TrainingValidationError(
            f"Invalid {field}"
        )

    return value


def finite_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise TrainingValidationError(
            f"{field} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise TrainingValidationError(
            f"{field} must be finite"
        )

    return result


def nonnegative_number(
    value: Any,
    field: str,
) -> float:
    result = finite_number(value, field)

    if result < 0:
        raise TrainingValidationError(
            f"{field} cannot be negative"
        )

    return result


def validate_positive_int(
    value: Any,
    field: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise TrainingValidationError(
            f"{field} must be a positive integer"
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
            raise TrainingValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise TrainingValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise TrainingValidationError(
            f"{field} must be a dictionary"
        )

    return result


def validate_metrics(
    metrics: dict[str, float],
) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise TrainingValidationError(
            "metrics must be a dictionary"
        )

    validated = {}

    for key, value in metrics.items():
        key = validate_string(
            key,
            "metric name",
            128,
        )

        validated[key] = finite_number(
            value,
            f"metric {key}",
        )

    return validated


# ============================================================
# ORM MODELS
# ============================================================

class TrainingJobORM(Base):
    __tablename__ = "model_training_jobs"

    __table_args__ = (
        Index(
            "ix_model_training_model_status",
            "model_name",
            "status",
        ),
    )

    job_id: Mapped[str] = mapped_column(
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

    job_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    training_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    dataset_version: Mapped[
        str | None
    ] = mapped_column(
        String(128),
        nullable=True,
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

    total_epochs: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    current_epoch: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    best_epoch: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    best_validation_loss: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    latest_metrics_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "latest_metrics",
        JSON,
        nullable=False,
        default=dict,
    )

    best_checkpoint_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
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

    epochs: Mapped[
        list["TrainingEpochORM"]
    ] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    checkpoints: Mapped[
        list["TrainingCheckpointORM"]
    ] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["TrainingEventORM"]
    ] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TrainingEpochORM(Base):
    __tablename__ = "model_training_epochs"

    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "epoch",
            name="uq_model_training_epoch",
        ),
    )

    epoch_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "model_training_jobs.job_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    epoch: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    training_loss: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    validation_loss: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    learning_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    duration_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    metrics_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "metrics",
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    job: Mapped[TrainingJobORM] = relationship(
        back_populates="epochs",
    )


class TrainingCheckpointORM(Base):
    __tablename__ = "model_training_checkpoints"

    checkpoint_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "model_training_jobs.job_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    epoch: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    checkpoint_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    validation_loss: Mapped[
        float | None
    ] = mapped_column(
        Float,
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

    job: Mapped[TrainingJobORM] = relationship(
        back_populates="checkpoints",
    )


class TrainingEventORM(Base):
    __tablename__ = "model_training_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "model_training_jobs.job_id",
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

    job: Mapped[TrainingJobORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class TrainingJobRecord:
    job_id: str
    experiment_id: str | None
    job_name: str
    training_type: str
    status: str
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str | None
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    total_epochs: int
    current_epoch: int
    best_epoch: int | None
    best_validation_loss: float | None
    latest_metrics: dict[str, Any]
    best_checkpoint_id: str | None
    error_message: str | None
    version: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class TrainingEpochRecord:
    epoch_id: str
    job_id: str
    epoch: int
    training_loss: float
    validation_loss: float | None
    learning_rate: float
    duration_seconds: float
    metrics: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TrainingCheckpointRecord:
    checkpoint_id: str
    job_id: str
    epoch: int
    checkpoint_path: str
    validation_loss: float | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TrainingEventRecord:
    event_id: str
    job_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


# ============================================================
# TRAINING REPOSITORY
# ============================================================

class TrainingRepository:
    """
    Persistent training job repository.

    The training service performs PyTorch
    model execution and sends metrics here.

    Supports caller-managed SQLAlchemy
    sessions and session_scope().
    """

    def __init__(
        self,
        *,
        max_epochs: int = 100000,
    ) -> None:
        self.max_epochs = validate_positive_int(
            max_epochs,
            "max_epochs",
        )

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
        job_id: str,
    ) -> TrainingJobORM:
        row = session.get(
            TrainingJobORM,
            job_id,
        )

        if row is None:
            raise TrainingNotFoundError(
                "Training job not found"
            )

        return row

    @staticmethod
    def _job_record(
        row: TrainingJobORM,
    ) -> TrainingJobRecord:
        return TrainingJobRecord(
            job_id=row.job_id,
            experiment_id=row.experiment_id,
            job_name=row.job_name,
            training_type=row.training_type,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            dataset_name=row.dataset_name,
            dataset_version=row.dataset_version,
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            total_epochs=row.total_epochs,
            current_epoch=row.current_epoch,
            best_epoch=row.best_epoch,
            best_validation_loss=row.best_validation_loss,
            latest_metrics=copy.deepcopy(
                row.latest_metrics_data
            ),
            best_checkpoint_id=row.best_checkpoint_id,
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
    def _epoch_record(
        row: TrainingEpochORM,
    ) -> TrainingEpochRecord:
        return TrainingEpochRecord(
            epoch_id=row.epoch_id,
            job_id=row.job_id,
            epoch=row.epoch,
            training_loss=row.training_loss,
            validation_loss=row.validation_loss,
            learning_rate=row.learning_rate,
            duration_seconds=row.duration_seconds,
            metrics=copy.deepcopy(
                row.metrics_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _checkpoint_record(
        row: TrainingCheckpointORM,
    ) -> TrainingCheckpointRecord:
        return TrainingCheckpointRecord(
            checkpoint_id=row.checkpoint_id,
            job_id=row.job_id,
            epoch=row.epoch,
            checkpoint_path=row.checkpoint_path,
            validation_loss=row.validation_loss,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _event_record(
        row: TrainingEventORM,
    ) -> TrainingEventRecord:
        return TrainingEventRecord(
            event_id=row.event_id,
            job_id=row.job_id,
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
        job_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            TrainingEventORM(
                event_id=uuid.uuid4().hex,
                job_id=job_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE JOB
    # --------------------------------------------------------

    def create_job(
        self,
        *,
        job_name: str,
        training_type: TrainingType,
        model_name: str,
        dataset_name: str,
        total_epochs: int,
        model_version: str = "v1",
        dataset_version: str | None = None,
        experiment_id: str | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> TrainingJobRecord:
        job_name = validate_string(
            job_name,
            "job_name",
        )

        model_name = validate_string(
            model_name,
            "model_name",
        )

        dataset_name = validate_string(
            dataset_name,
            "dataset_name",
        )

        model_version = validate_string(
            model_version,
            "model_version",
            128,
        )

        if not isinstance(
            training_type,
            TrainingType,
        ):
            raise TrainingValidationError(
                "Invalid training type"
            )

        total_epochs = validate_positive_int(
            total_epochs,
            "total_epochs",
        )

        if total_epochs > self.max_epochs:
            raise TrainingValidationError(
                "total_epochs exceeds limit"
            )

        if dataset_version is not None:
            dataset_version = validate_string(
                dataset_version,
                "dataset_version",
                128,
            )

        if experiment_id is not None:
            experiment_id = validate_string(
                experiment_id,
                "experiment_id",
                64,
            )

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        row = TrainingJobORM(
            job_id=uuid.uuid4().hex,
            experiment_id=experiment_id,
            job_name=job_name,
            training_type=training_type.value,
            status=TrainingStatus.QUEUED.value,
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            configuration_data=configuration,
            metadata_data=metadata,
            total_epochs=total_epochs,
            current_epoch=0,
            latest_metrics_data={},
            version=1,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)

            self._audit(
                session,
                row.job_id,
                "training_queued",
                {
                    "model_name": model_name,
                    "total_epochs": total_epochs,
                },
            )

            session.flush()

            return self._job_record(row)

    # --------------------------------------------------------
    # GET JOB
    # --------------------------------------------------------

    def get_job(
        self,
        job_id: str,
        *,
        db: Session | None = None,
    ) -> TrainingJobRecord:
        with self._session(db) as session:
            return self._job_record(
                self._get(
                    session,
                    job_id,
                )
            )

    # --------------------------------------------------------
    # STATUS TRANSITION
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        job_id: str,
        expected_version: int,
        allowed_from: set[TrainingStatus],
        target: TrainingStatus,
        values: dict[str, Any] | None = None,
    ) -> TrainingJobRecord:
        result = session.execute(
            update(TrainingJobORM)
            .where(
                TrainingJobORM.job_id == job_id,
                TrainingJobORM.version
                == expected_version,
                TrainingJobORM.status.in_(
                    [
                        status.value
                        for status in allowed_from
                    ]
                ),
            )
            .values(
                **(values or {}),
                status=target.value,
                version=expected_version + 1,
            )
        )

        if result.rowcount != 1:
            raise TrainingConflictError(
                "Training status/version changed"
            )

        self._audit(
            session,
            job_id,
            f"training_{target.value}",
        )

        session.flush()
        session.expire_all()

        return self._job_record(
            self._get(
                session,
                job_id,
            )
        )

    def start_job(
        self,
        job_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> TrainingJobRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                job_id=job_id,
                expected_version=expected_version,
                allowed_from={TrainingStatus.QUEUED},
                target=TrainingStatus.RUNNING,
                values={
                    "started_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # LOG EPOCH
    # --------------------------------------------------------

    def log_epoch(
        self,
        job_id: str,
        *,
        epoch: int,
        training_loss: float,
        validation_loss: float | None,
        learning_rate: float,
        duration_seconds: float,
        metrics: dict[str, float] | None = None,
        db: Session | None = None,
    ) -> TrainingEpochRecord:
        epoch = validate_positive_int(
            epoch,
            "epoch",
        )

        training_loss = nonnegative_number(
            training_loss,
            "training_loss",
        )

        if validation_loss is not None:
            validation_loss = nonnegative_number(
                validation_loss,
                "validation_loss",
            )

        learning_rate = nonnegative_number(
            learning_rate,
            "learning_rate",
        )

        duration_seconds = nonnegative_number(
            duration_seconds,
            "duration_seconds",
        )

        metrics = validate_metrics(
            metrics or {}
        )

        with self._session(db) as session:
            job = self._get(
                session,
                job_id,
            )

            if job.status != TrainingStatus.RUNNING.value:
                raise TrainingStateError(
                    "Training job must be running"
                )

            if epoch != job.current_epoch + 1:
                raise TrainingValidationError(
                    "Epochs must be logged sequentially"
                )

            if epoch > job.total_epochs:
                raise TrainingValidationError(
                    "Epoch exceeds configured total"
                )

            best_loss = job.best_validation_loss
            best_epoch = job.best_epoch

            if (
                validation_loss is not None
                and (
                    best_loss is None
                    or validation_loss < best_loss
                )
            ):
                best_loss = validation_loss
                best_epoch = epoch

            latest_metrics = {
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
                "duration_seconds": duration_seconds,
                **metrics,
            }

            result = session.execute(
                update(TrainingJobORM)
                .where(
                    TrainingJobORM.job_id == job_id,
                    TrainingJobORM.status
                    == TrainingStatus.RUNNING.value,
                    TrainingJobORM.current_epoch
                    == epoch - 1,
                    TrainingJobORM.version
                    == job.version,
                )
                .values(
                    current_epoch=epoch,
                    best_validation_loss=best_loss,
                    best_epoch=best_epoch,
                    latest_metrics_data=latest_metrics,
                    version=job.version + 1,
                )
            )

            if result.rowcount != 1:
                raise TrainingConflictError(
                    "Training epoch changed concurrently"
                )

            row = TrainingEpochORM(
                epoch_id=uuid.uuid4().hex,
                job_id=job_id,
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=validation_loss,
                learning_rate=learning_rate,
                duration_seconds=duration_seconds,
                metrics_data=metrics,
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._epoch_record(row)

    # --------------------------------------------------------
    # REGISTER CHECKPOINT
    # --------------------------------------------------------

    def register_checkpoint(
        self,
        job_id: str,
        *,
        epoch: int,
        checkpoint_path: str,
        validation_loss: float | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> TrainingCheckpointRecord:
        epoch = validate_positive_int(
            epoch,
            "epoch",
        )

        checkpoint_path = validate_string(
            checkpoint_path,
            "checkpoint_path",
            4000,
        )

        if validation_loss is not None:
            validation_loss = nonnegative_number(
                validation_loss,
                "validation_loss",
            )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            job = self._get(
                session,
                job_id,
            )

            if job.status not in {
                TrainingStatus.RUNNING.value,
                TrainingStatus.COMPLETED.value,
                TrainingStatus.EARLY_STOPPED.value,
            }:
                raise TrainingStateError(
                    "Cannot register checkpoint "
                    "for this job status"
                )

            if epoch > job.current_epoch:
                raise TrainingValidationError(
                    "Checkpoint epoch exceeds progress"
                )

            row = TrainingCheckpointORM(
                checkpoint_id=uuid.uuid4().hex,
                job_id=job_id,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
                validation_loss=validation_loss,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)

            # Prefer the checkpoint corresponding
            # to the best validation epoch.
            if (
                job.best_epoch == epoch
                and validation_loss is not None
            ):
                job.best_checkpoint_id = (
                    row.checkpoint_id
                )
                job.version += 1

            self._audit(
                session,
                job_id,
                "checkpoint_registered",
                {
                    "checkpoint_id": row.checkpoint_id,
                    "epoch": epoch,
                },
            )

            session.flush()

            return self._checkpoint_record(row)

    # --------------------------------------------------------
    # COMPLETE JOB
    # --------------------------------------------------------

    def complete_job(
        self,
        job_id: str,
        *,
        expected_version: int,
        early_stopped: bool = False,
        db: Session | None = None,
    ) -> TrainingJobRecord:
        if not isinstance(early_stopped, bool):
            raise TrainingValidationError(
                "early_stopped must be boolean"
            )

        with self._session(db) as session:
            job = self._get(
                session,
                job_id,
            )

            if job.current_epoch == 0:
                raise TrainingStateError(
                    "Cannot complete job without epochs"
                )

            if (
                not early_stopped
                and job.current_epoch != job.total_epochs
            ):
                raise TrainingStateError(
                    "Not all configured epochs completed"
                )

            target = (
                TrainingStatus.EARLY_STOPPED
                if early_stopped
                else TrainingStatus.COMPLETED
            )

            return self._transition(
                session,
                job_id=job_id,
                expected_version=expected_version,
                allowed_from={TrainingStatus.RUNNING},
                target=target,
                values={
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # FAIL JOB
    # --------------------------------------------------------

    def fail_job(
        self,
        job_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> TrainingJobRecord:
        reason = validate_string(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                job_id=job_id,
                expected_version=expected_version,
                allowed_from={
                    TrainingStatus.QUEUED,
                    TrainingStatus.RUNNING,
                },
                target=TrainingStatus.FAILED,
                values={
                    "error_message": reason,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # CANCEL JOB
    # --------------------------------------------------------

    def cancel_job(
        self,
        job_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> TrainingJobRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                job_id=job_id,
                expected_version=expected_version,
                allowed_from={
                    TrainingStatus.QUEUED,
                    TrainingStatus.RUNNING,
                },
                target=TrainingStatus.CANCELLED,
                values={
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # LIST EPOCHS
    # --------------------------------------------------------

    def list_epochs(
        self,
        job_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[TrainingEpochRecord]:
        if not 1 <= limit <= 10000 or offset < 0:
            raise TrainingValidationError(
                "Invalid pagination"
            )

        statement = (
            select(TrainingEpochORM)
            .where(
                TrainingEpochORM.job_id == job_id
            )
            .order_by(
                TrainingEpochORM.epoch.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._epoch_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST CHECKPOINTS
    # --------------------------------------------------------

    def list_checkpoints(
        self,
        job_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[TrainingCheckpointRecord]:
        if not 1 <= limit <= 1000:
            raise TrainingValidationError(
                "Invalid limit"
            )

        statement = (
            select(TrainingCheckpointORM)
            .where(
                TrainingCheckpointORM.job_id == job_id
            )
            .order_by(
                TrainingCheckpointORM.epoch.desc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._checkpoint_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST JOBS
    # --------------------------------------------------------

    def list_jobs(
        self,
        *,
        model_name: str | None = None,
        status: TrainingStatus | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[TrainingJobRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise TrainingValidationError(
                "Invalid pagination"
            )

        statement = select(
            TrainingJobORM
        )

        if model_name is not None:
            statement = statement.where(
                TrainingJobORM.model_name
                == model_name
            )

        if status is not None:
            statement = statement.where(
                TrainingJobORM.status
                == status.value
            )

        if experiment_id is not None:
            statement = statement.where(
                TrainingJobORM.experiment_id
                == experiment_id
            )

        statement = (
            statement
            .order_by(
                TrainingJobORM.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._job_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # AUDIT EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        job_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[TrainingEventRecord]:
        if not 1 <= limit <= 1000:
            raise TrainingValidationError(
                "Invalid limit"
            )

        statement = (
            select(TrainingEventORM)
            .where(
                TrainingEventORM.job_id == job_id
            )
            .order_by(
                TrainingEventORM.created_at.asc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._event_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        job_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            job = self.get_job(
                job_id,
                db=session,
            )

            epochs = []

            for offset in range(
                0,
                job.current_epoch,
                1000,
            ):
                epochs.extend(
                    self.list_epochs(
                        job_id,
                        limit=1000,
                        offset=offset,
                        db=session,
                    )
                )

            payload = {
                "job": asdict(job),
                "epochs": [
                    asdict(epoch)
                    for epoch in epochs
                ],
                "checkpoints": [
                    asdict(checkpoint)
                    for checkpoint in self.list_checkpoints(
                        job_id,
                        limit=1000,
                        db=session,
                    )
                ],
                "events": [
                    asdict(event)
                    for event in self.list_events(
                        job_id,
                        limit=1000,
                        db=session,
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

    def export_epochs_csv(
        self,
        job_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fields = [
            "epoch_id",
            "job_id",
            "epoch",
            "training_loss",
            "validation_loss",
            "learning_rate",
            "duration_seconds",
            "metrics",
            "created_at",
        ]

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

            offset = 0

            while True:
                epochs = self.list_epochs(
                    job_id,
                    limit=1000,
                    offset=offset,
                    db=db,
                )

                if not epochs:
                    break

                for epoch in epochs:
                    data = asdict(epoch)

                    data["metrics"] = json.dumps(
                        data["metrics"]
                    )

                    writer.writerow(
                        {
                            key: data.get(key)
                            for key in fields
                        }
                    )

                offset += len(epochs)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_training_repository(
    **kwargs: Any,
) -> TrainingRepository:
    return TrainingRepository(
        **kwargs
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

    repository = create_training_repository()

    job = repository.create_job(
        job_name="Demo World Model Training",
        training_type=TrainingType.WORLD_MODEL,
        model_name="neuroworld_transformer",
        model_version="v1",
        dataset_name="synthetic_physics",
        dataset_version="v1",
        total_epochs=3,
        configuration={
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "device": "cpu",
        },
        metadata={
            "illustrative": True,
        },
    )

    job = repository.start_job(
        job.job_id,
        expected_version=job.version,
    )

    losses = [
        (0.80, 0.75),
        (0.55, 0.50),
        (0.35, 0.32),
    ]

    for epoch_index, (
        train_loss,
        val_loss,
    ) in enumerate(losses, start=1):
        repository.log_epoch(
            job.job_id,
            epoch=epoch_index,
            training_loss=train_loss,
            validation_loss=val_loss,
            learning_rate=0.001,
            duration_seconds=1.5,
            metrics={
                "prediction_accuracy": (
                    0.65 + epoch_index * 0.08
                ),
            },
        )

    job = repository.get_job(
        job.job_id
    )

    # Register a demonstration path only.
    # No checkpoint weights are created here.
    repository.register_checkpoint(
        job.job_id,
        epoch=3,
        checkpoint_path=(
            "models/checkpoints/"
            "demo_world_model_epoch_3.pt"
        ),
        validation_loss=0.32,
        metadata={
            "illustrative": True,
        },
    )

    job = repository.get_job(
        job.job_id
    )

    job = repository.complete_job(
        job.job_id,
        expected_version=job.version,
    )

    print("=" * 60)
    print("NEUROWORLD AI - TRAINING DATABASE")
    print("=" * 60)

    print("Job ID:", job.job_id)
    print("Status:", job.status)
    print("Epochs:", job.current_epoch)
    print("Best epoch:", job.best_epoch)
    print(
        "Best validation loss:",
        job.best_validation_loss,
    )

    json_path = repository.export_json(
        job.job_id,
        "experiments/reports/training_result.json",
    )

    csv_path = repository.export_epochs_csv(
        job.job_id,
        "experiments/reports/training_epochs.csv",
    )

    print("JSON:", json_path)
    print("CSV:", csv_path)

    close_database()


if __name__ == "__main__":
    main()
