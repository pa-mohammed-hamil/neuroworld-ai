
"""
NeuroWorld AI
backend/app/database/experiment.py

Persistent AI Experiment Tracking.

Features:
- Create and manage experiments
- Start, complete, fail and cancel runs
- Track model and dataset versions
- Store training hyperparameters
- Record metrics by training step
- Attach checkpoint and evaluation IDs
- Store artifact references
- Experiment history and filtering
- Compare completed runs
- Experiment summaries
- JSON / CSV export
- PostgreSQL and SQLite support
- Transaction-safe database operations

Dependencies:
    pip install sqlalchemy numpy

Run:
    python -m backend.app.database.experiment

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Notes:
    This module stores metadata and metric history.
    It does not train models or execute experiments.

    Artifacts should be stored in an approved
    filesystem or object store. The database
    stores their references.

    Authorization and tenant isolation must
    be enforced by the API/service layer.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
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
    "neuroworld.database.experiment"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ExperimentDatabaseError(Exception):
    """Base experiment database exception."""


class ExperimentValidationError(ExperimentDatabaseError):
    """Invalid experiment input."""


class ExperimentNotFoundError(ExperimentDatabaseError):
    """Experiment or run not found."""


class ExperimentStateError(ExperimentDatabaseError):
    """Invalid experiment state transition."""


# ============================================================
# STATUS ENUM
# ============================================================

class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
}


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


def datetime_string(
    value: datetime | None,
) -> str | None:
    normalized = normalize_datetime(value)

    return (
        normalized.isoformat()
        if normalized is not None
        else None
    )


def validate_identifier(
    value: str,
    name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise ExperimentValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise ExperimentValidationError(
            f"Invalid {name}"
        )

    return value


def json_safe(value: Any) -> Any:
    """
    Convert supported values to JSON.

    Reject NaN, infinity and unsupported types.
    """

    if is_dataclass(value) and not isinstance(
        value,
        type,
    ):
        return json_safe(
            asdict(value)
        )

    if isinstance(value, np.ndarray):
        return json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return json_safe(
            value.item()
        )

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
            raise ExperimentValidationError(
                "Non-finite numbers are not allowed"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ExperimentValidationError(
        f"Unsupported value type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    normalized = json_safe(value)

    if not isinstance(normalized, dict):
        raise ExperimentValidationError(
            f"{name} must be a dictionary"
        )

    return normalized


def validate_metric_value(
    value: float,
) -> float:
    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError(
            "Metric value must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ExperimentValidationError(
            "Metric value must be finite"
        )

    return number


# ============================================================
# ORM MODELS
# ============================================================

class ExperimentORM(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ExperimentStatus.DRAFT.value,
        index=True,
    )

    project_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="neuroworld-ai",
    )

    tags_data: Mapped[list[str]] = mapped_column(
        "tags",
        JSON,
        nullable=False,
        default=list,
    )

    metadata_data: Mapped[dict[str, Any]] = mapped_column(
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

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    runs: Mapped[list["ExperimentRunORM"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ExperimentRunORM(Base):
    __tablename__ = "experiment_runs"

    __table_args__ = (
        Index(
            "ix_experiment_runs_experiment_created",
            "experiment_id",
            "created_at",
        ),
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "experiments.experiment_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    run_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RunStatus.QUEUED.value,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
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

    dataset_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    parameters_data: Mapped[dict[str, Any]] = mapped_column(
        "parameters",
        JSON,
        nullable=False,
        default=dict,
    )

    summary_metrics_data: Mapped[dict[str, Any]] = mapped_column(
        "summary_metrics",
        JSON,
        nullable=False,
        default=dict,
    )

    artifacts_data: Mapped[list[dict[str, Any]]] = mapped_column(
        "artifacts",
        JSON,
        nullable=False,
        default=list,
    )

    checkpoint_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    evaluation_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    experiment: Mapped[ExperimentORM] = relationship(
        back_populates="runs",
    )

    metrics: Mapped[list["ExperimentMetricORM"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ExperimentMetricORM(Base):
    __tablename__ = "experiment_metrics"

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "metric_name",
            "step",
            name="uq_experiment_metric_run_name_step",
        ),
    )

    metric_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "experiment_runs.run_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    metric_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    step: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    run: Mapped[ExperimentRunORM] = relationship(
        back_populates="metrics",
    )


# ============================================================
# PUBLIC MODELS
# ============================================================

@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    name: str
    description: str
    status: str
    project_name: str
    tags: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ExperimentRunRecord:
    run_id: str
    experiment_id: str
    run_name: str
    status: str
    model_name: str
    model_version: str
    dataset_name: str
    dataset_version: str
    parameters: dict[str, Any]
    summary_metrics: dict[str, Any]
    artifacts: list[dict[str, Any]]
    checkpoint_id: str | None
    evaluation_id: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class ExperimentMetricRecord:
    metric_id: str
    run_id: str
    metric_name: str
    metric_value: float
    step: int
    recorded_at: str


@dataclass(frozen=True)
class ExperimentSummary:
    experiment_id: str
    total_runs: int
    completed_runs: int
    failed_runs: int
    running_runs: int
    cancelled_runs: int


# ============================================================
# REPOSITORY
# ============================================================

class ExperimentRepository:
    """
    Database-backed experiment repository.

    Caller-owned SQLAlchemy sessions are not
    committed or closed by this repository.

    When db=None, operations use session_scope().
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
    def _experiment_record(
        row: ExperimentORM,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row.experiment_id,
            name=row.name,
            description=row.description,
            status=row.status,
            project_name=row.project_name,
            tags=list(row.tags_data),
            metadata=dict(row.metadata_data),
            created_at=datetime_string(
                row.created_at
            ),
            updated_at=datetime_string(
                row.updated_at
            ),
        )

    @staticmethod
    def _run_record(
        row: ExperimentRunORM,
    ) -> ExperimentRunRecord:
        return ExperimentRunRecord(
            run_id=row.run_id,
            experiment_id=row.experiment_id,
            run_name=row.run_name,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            dataset_name=row.dataset_name,
            dataset_version=row.dataset_version,
            parameters=dict(row.parameters_data),
            summary_metrics=dict(
                row.summary_metrics_data
            ),
            artifacts=list(row.artifacts_data),
            checkpoint_id=row.checkpoint_id,
            evaluation_id=row.evaluation_id,
            error_message=row.error_message,
            created_at=datetime_string(
                row.created_at
            ),
            started_at=datetime_string(
                row.started_at
            ),
            finished_at=datetime_string(
                row.finished_at
            ),
        )

    @staticmethod
    def _metric_record(
        row: ExperimentMetricORM,
    ) -> ExperimentMetricRecord:
        return ExperimentMetricRecord(
            metric_id=row.metric_id,
            run_id=row.run_id,
            metric_name=row.metric_name,
            metric_value=row.metric_value,
            step=row.step,
            recorded_at=datetime_string(
                row.recorded_at
            ),
        )

    # --------------------------------------------------------
    # CREATE EXPERIMENT
    # --------------------------------------------------------

    def create_experiment(
        self,
        *,
        name: str,
        description: str = "",
        project_name: str = "neuroworld-ai",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ExperimentRecord:
        name = validate_identifier(
            name,
            "name",
            255,
        )

        project_name = validate_identifier(
            project_name,
            "project_name",
        )

        tags = tags or []

        if not isinstance(tags, list) or any(
            not isinstance(tag, str)
            for tag in tags
        ):
            raise ExperimentValidationError(
                "tags must be a list of strings"
            )

        tags = [
            validate_identifier(
                tag,
                "tag",
                128,
            )
            for tag in tags
        ]

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        now = utc_now()

        row = ExperimentORM(
            experiment_id=uuid.uuid4().hex,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT.value,
            project_name=project_name,
            tags_data=tags,
            metadata_data=metadata,
            created_at=now,
            updated_at=now,
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            return self._experiment_record(row)

    # --------------------------------------------------------
    # GET EXPERIMENT
    # --------------------------------------------------------

    def get_experiment(
        self,
        experiment_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentRecord:
        with self._session(db) as session:
            row = session.get(
                ExperimentORM,
                experiment_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Experiment not found"
                )

            return self._experiment_record(row)

    # --------------------------------------------------------
    # LIST EXPERIMENTS
    # --------------------------------------------------------

    def list_experiments(
        self,
        *,
        status: ExperimentStatus | None = None,
        project_name: str | None = None,
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

        if status is not None:
            statement = statement.where(
                ExperimentORM.status
                == status.value
            )

        if project_name is not None:
            statement = statement.where(
                ExperimentORM.project_name
                == project_name
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
            rows = session.scalars(
                statement
            ).all()

            return [
                self._experiment_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # UPDATE EXPERIMENT STATUS
    # --------------------------------------------------------

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        db: Session | None = None,
    ) -> ExperimentRecord:
        if not isinstance(status, ExperimentStatus):
            raise ExperimentValidationError(
                "Invalid experiment status"
            )

        with self._session(db) as session:
            row = session.get(
                ExperimentORM,
                experiment_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Experiment not found"
                )

            row.status = status.value
            row.updated_at = utc_now()

            session.flush()

            return self._experiment_record(row)

    # --------------------------------------------------------
    # CREATE RUN
    # --------------------------------------------------------

    def create_run(
        self,
        *,
        experiment_id: str,
        model_name: str,
        dataset_name: str,
        run_name: str | None = None,
        model_version: str = "unknown",
        dataset_version: str = "unknown",
        parameters: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ExperimentRunRecord:
        model_name = validate_identifier(
            model_name,
            "model_name",
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

        dataset_version = validate_identifier(
            dataset_version,
            "dataset_version",
        )

        parameters = validate_json_dict(
            parameters or {},
            "parameters",
        )

        with self._session(db) as session:
            experiment = session.get(
                ExperimentORM,
                experiment_id,
            )

            if experiment is None:
                raise ExperimentNotFoundError(
                    "Experiment not found"
                )

            if experiment.status == (
                ExperimentStatus.ARCHIVED.value
            ):
                raise ExperimentStateError(
                    "Cannot create runs in archived experiments"
                )

            run_id = uuid.uuid4().hex

            row = ExperimentRunORM(
                run_id=run_id,
                experiment_id=experiment_id,
                run_name=(
                    run_name or f"run-{run_id[:8]}"
                ),
                status=RunStatus.QUEUED.value,
                model_name=model_name,
                model_version=model_version,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                parameters_data=parameters,
                summary_metrics_data={},
                artifacts_data=[],
                created_at=utc_now(),
            )

            session.add(row)

            experiment.status = (
                ExperimentStatus.ACTIVE.value
            )

            experiment.updated_at = utc_now()

            session.flush()

            return self._run_record(row)

    # --------------------------------------------------------
    # GET RUN
    # --------------------------------------------------------

    def get_run(
        self,
        run_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentRunRecord:
        with self._session(db) as session:
            row = session.get(
                ExperimentRunORM,
                run_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Experiment run not found"
                )

            return self._run_record(row)

    # --------------------------------------------------------
    # RUN STATE TRANSITION
    # --------------------------------------------------------

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_message: str | None = None,
        db: Session | None = None,
    ) -> ExperimentRunRecord:
        if not isinstance(status, RunStatus):
            raise ExperimentValidationError(
                "Invalid run status"
            )

        allowed = {
            RunStatus.QUEUED: {
                RunStatus.RUNNING,
                RunStatus.CANCELLED,
            },
            RunStatus.RUNNING: {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            },
            RunStatus.COMPLETED: set(),
            RunStatus.FAILED: set(),
            RunStatus.CANCELLED: set(),
        }

        with self._session(db) as session:
            row = session.get(
                ExperimentRunORM,
                run_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Experiment run not found"
                )

            current = RunStatus(
                row.status
            )

            if status not in allowed[current]:
                raise ExperimentStateError(
                    f"Invalid run transition: "
                    f"{current.value} -> {status.value}"
                )

            now = utc_now()

            row.status = status.value

            if status == RunStatus.RUNNING:
                row.started_at = now

            if status in TERMINAL_RUN_STATUSES:
                row.finished_at = now

            if status == RunStatus.FAILED:
                row.error_message = (
                    error_message or "Unknown error"
                )

            session.flush()

            return self._run_record(row)

    # --------------------------------------------------------
    # RECORD METRIC
    # --------------------------------------------------------

    def log_metric(
        self,
        run_id: str,
        metric_name: str,
        value: float,
        *,
        step: int,
        db: Session | None = None,
    ) -> ExperimentMetricRecord:
        metric_name = validate_identifier(
            metric_name,
            "metric_name",
        )

        metric_value = validate_metric_value(
            value
        )

        if not isinstance(step, int) or step < 0:
            raise ExperimentValidationError(
                "step must be a nonnegative integer"
            )

        with self._session(db) as session:
            run = session.get(
                ExperimentRunORM,
                run_id,
            )

            if run is None:
                raise ExperimentNotFoundError(
                    "Experiment run not found"
                )

            if run.status != RunStatus.RUNNING.value:
                raise ExperimentStateError(
                    "Metrics can only be logged "
                    "for running experiments"
                )

            statement = select(
                ExperimentMetricORM
            ).where(
                ExperimentMetricORM.run_id == run_id,
                ExperimentMetricORM.metric_name
                == metric_name,
                ExperimentMetricORM.step == step,
            )

            row = session.scalar(
                statement
            )

            if row is None:
                row = ExperimentMetricORM(
                    metric_id=uuid.uuid4().hex,
                    run_id=run_id,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    step=step,
                    recorded_at=utc_now(),
                )

                session.add(row)

            else:
                row.metric_value = metric_value
                row.recorded_at = utc_now()

            summary = dict(
                run.summary_metrics_data
            )

            summary[metric_name] = metric_value

            run.summary_metrics_data = summary

            session.flush()

            return self._metric_record(row)

    # --------------------------------------------------------
    # METRIC HISTORY
    # --------------------------------------------------------

    def get_metric_history(
        self,
        run_id: str,
        *,
        metric_name: str | None = None,
        db: Session | None = None,
    ) -> list[ExperimentMetricRecord]:
        statement = select(
            ExperimentMetricORM
        ).where(
            ExperimentMetricORM.run_id == run_id
        )

        if metric_name is not None:
            statement = statement.where(
                ExperimentMetricORM.metric_name
                == metric_name
            )

        statement = statement.order_by(
            ExperimentMetricORM.step.asc(),
            ExperimentMetricORM.metric_name.asc(),
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._metric_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # ATTACH CHECKPOINT
    # --------------------------------------------------------

    def attach_checkpoint(
        self,
        run_id: str,
        checkpoint_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentRunRecord:
        checkpoint_id = validate_identifier(
            checkpoint_id,
            "checkpoint_id",
            64,
        )

        with self._session(db) as session:
            row = session.get(
                ExperimentRunORM,
                run_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Run not found"
                )

            row.checkpoint_id = checkpoint_id

            session.flush()

            return self._run_record(row)

    # --------------------------------------------------------
    # ATTACH EVALUATION
    # --------------------------------------------------------

    def attach_evaluation(
        self,
        run_id: str,
        evaluation_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentRunRecord:
        evaluation_id = validate_identifier(
            evaluation_id,
            "evaluation_id",
            64,
        )

        with self._session(db) as session:
            row = session.get(
                ExperimentRunORM,
                run_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Run not found"
                )

            row.evaluation_id = evaluation_id

            session.flush()

            return self._run_record(row)

    # --------------------------------------------------------
    # ADD ARTIFACT REFERENCE
    # --------------------------------------------------------

    def add_artifact(
        self,
        run_id: str,
        *,
        name: str,
        uri: str,
        artifact_type: str = "file",
        db: Session | None = None,
    ) -> ExperimentRunRecord:
        name = validate_identifier(
            name,
            "artifact_name",
            255,
        )

        uri = validate_identifier(
            uri,
            "artifact_uri",
            2048,
        )

        artifact_type = validate_identifier(
            artifact_type,
            "artifact_type",
        )

        with self._session(db) as session:
            row = session.get(
                ExperimentRunORM,
                run_id,
            )

            if row is None:
                raise ExperimentNotFoundError(
                    "Run not found"
                )

            artifacts = list(
                row.artifacts_data
            )

            artifacts.append(
                {
                    "name": name,
                    "uri": uri,
                    "type": artifact_type,
                    "created_at": datetime_string(
                        utc_now()
                    ),
                }
            )

            row.artifacts_data = artifacts

            session.flush()

            return self._run_record(row)

    # --------------------------------------------------------
    # LIST RUNS
    # --------------------------------------------------------

    def list_runs(
        self,
        experiment_id: str,
        *,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[ExperimentRunRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ExperimentValidationError(
                "Invalid pagination"
            )

        statement = select(
            ExperimentRunORM
        ).where(
            ExperimentRunORM.experiment_id
            == experiment_id
        )

        if status is not None:
            statement = statement.where(
                ExperimentRunORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                ExperimentRunORM.created_at.desc(),
                ExperimentRunORM.run_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._run_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # EXPERIMENT SUMMARY
    # --------------------------------------------------------

    def get_experiment_summary(
        self,
        experiment_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentSummary:
        with self._session(db) as session:
            experiment = session.get(
                ExperimentORM,
                experiment_id,
            )

            if experiment is None:
                raise ExperimentNotFoundError(
                    "Experiment not found"
                )

            rows = session.execute(
                select(
                    ExperimentRunORM.status,
                    func.count(
                        ExperimentRunORM.run_id
                    ),
                )
                .where(
                    ExperimentRunORM.experiment_id
                    == experiment_id
                )
                .group_by(
                    ExperimentRunORM.status
                )
            ).all()

        counts = {
            status: int(count)
            for status, count in rows
        }

        return ExperimentSummary(
            experiment_id=experiment_id,
            total_runs=sum(counts.values()),
            completed_runs=counts.get(
                RunStatus.COMPLETED.value,
                0,
            ),
            failed_runs=counts.get(
                RunStatus.FAILED.value,
                0,
            ),
            running_runs=counts.get(
                RunStatus.RUNNING.value,
                0,
            ),
            cancelled_runs=counts.get(
                RunStatus.CANCELLED.value,
                0,
            ),
        )

    # --------------------------------------------------------
    # COMPARE RUNS
    # --------------------------------------------------------

    def compare_runs(
        self,
        experiment_id: str,
        *,
        metric_name: str,
        higher_is_better: bool = True,
        db: Session | None = None,
    ) -> list[ExperimentRunRecord]:
        """
        Compare completed runs using a metric.

        Runs should share compatible datasets
        and evaluation configurations.
        """

        runs = self.list_runs(
            experiment_id,
            status=RunStatus.COMPLETED,
            limit=1000,
            db=db,
        )

        runs = [
            run
            for run in runs
            if metric_name in run.summary_metrics
        ]

        return sorted(
            runs,
            key=lambda run: float(
                run.summary_metrics[metric_name]
            ),
            reverse=higher_is_better,
        )

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
        experiment = self.get_experiment(
            experiment_id,
            db=db,
        )

        runs = self.list_runs(
            experiment_id,
            limit=1000,
            db=db,
        )

        payload = {
            "experiment": asdict(experiment),
            "runs": [],
        }

        for run in runs:
            payload["runs"].append(
                {
                    **asdict(run),
                    "metric_history": [
                        asdict(metric)
                        for metric in self.get_metric_history(
                            run.run_id,
                            db=db,
                        )
                    ],
                }
            )

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
        experiment_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        runs = self.list_runs(
            experiment_id,
            limit=1000,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "run_id",
            "experiment_id",
            "run_name",
            "status",
            "model_name",
            "model_version",
            "dataset_name",
            "dataset_version",
            "parameters",
            "summary_metrics",
            "checkpoint_id",
            "evaluation_id",
            "created_at",
            "started_at",
            "finished_at",
        ]

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

            for run in runs:
                data = asdict(run)

                row = {
                    key: data.get(key)
                    for key in fieldnames
                }

                row["parameters"] = json.dumps(
                    run.parameters,
                    allow_nan=False,
                )

                row["summary_metrics"] = json.dumps(
                    run.summary_metrics,
                    allow_nan=False,
                )

                writer.writerow(row)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_experiment_repository() -> ExperimentRepository:
    return ExperimentRepository()


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

    repository = create_experiment_repository()

    experiment = repository.create_experiment(
        name="World Model Prediction Experiment",
        description=(
            "Evaluate multi-step prediction accuracy "
            "for NeuroWorld AI"
        ),
        tags=[
            "world-model",
            "prediction",
            "evaluation",
        ],
        metadata={
            "environment": "synthetic_navigation",
        },
    )

    run = repository.create_run(
        experiment_id=experiment.experiment_id,
        model_name="transformer_world_model",
        model_version="v1.0",
        dataset_name="synthetic_navigation",
        dataset_version="v1",
        parameters={
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "prediction_horizon": 12,
        },
    )

    repository.update_run_status(
        run.run_id,
        RunStatus.RUNNING,
    )

    for step in range(1, 6):
        repository.log_metric(
            run.run_id,
            "training_loss",
            1.0 / step,
            step=step,
        )

        repository.log_metric(
            run.run_id,
            "validation_accuracy",
            min(0.99, 0.5 + 0.08 * step),
            step=step,
        )

    repository.update_run_status(
        run.run_id,
        RunStatus.COMPLETED,
    )

    summary = repository.get_experiment_summary(
        experiment.experiment_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - EXPERIMENT TRACKING")
    print("=" * 60)

    print(
        "Experiment ID:",
        experiment.experiment_id,
    )

    print(
        "Run ID:",
        run.run_id,
    )

    print(
        "Summary:",
        asdict(summary),
    )

    json_path = repository.export_json(
        experiment.experiment_id,
        "experiments/reports/experiment_history.json",
    )

    csv_path = repository.export_csv(
        experiment.experiment_id,
        "experiments/reports/experiment_runs.csv",
    )

    print(
        "JSON report:",
        json_path,
    )

    print(
        "CSV report:",
        csv_path,
    )

    close_database()


if __name__ == "__main__":
    main()
