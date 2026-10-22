work-orders
"""
NeuroWorld AI
backend/app/database/experiment_tracker.py

Persistent AI/ML experiment tracking.

Features:
- Experiment registration
- Run lifecycle management
- Hyperparameter tracking
- Training and validation metrics
- Artifact and checkpoint references
- Reproducibility metadata
- Run comparison
- Best-run selection
- Run audit events
- JSON and CSV export

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module stores metadata and artifact references.
It does not execute model training or upload files.
"""

from __future__ import annotations

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


# ============================================================
# EXCEPTIONS
# ============================================================

class ExperimentTrackerError(Exception):
    """Base experiment tracking error."""


class TrackerValidationError(ExperimentTrackerError):
    """Invalid tracking input."""


class TrackerNotFoundError(ExperimentTrackerError):
    """Experiment or run not found."""


class TrackerStateError(ExperimentTrackerError):
    """Invalid run state transition."""


class TrackerConflictError(ExperimentTrackerError):
    """Concurrent modification or duplicate data."""


# ============================================================
# ENUMS
# ============================================================

class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


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
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(
        timezone.utc
    ).isoformat()


def validate_string(
    value: str,
    field: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str):
        raise TrackerValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise TrackerValidationError(
            f"Invalid {field}"
        )

    return value


def finite_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise TrackerValidationError(
            f"{field} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrackerValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise TrackerValidationError(
            f"{field} must be finite"
        )

    return result


def validate_step(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise TrackerValidationError(
            "step must be a non-negative integer"
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
            raise TrackerValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise TrackerValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise TrackerValidationError(
            f"{field} must be a dictionary"
        )

    return result


def validate_limit(
    limit: int,
    offset: int = 0,
) -> None:
    if (
        isinstance(limit, bool)
        or isinstance(offset, bool)
        or not isinstance(limit, int)
        or not isinstance(offset, int)
        or not 1 <= limit <= 1000
        or offset < 0
    ):
        raise TrackerValidationError(
            "Invalid pagination"
        )


# ============================================================
# ORM MODELS
# ============================================================

class TrackedExperimentORM(Base):
    __tablename__ = "tracker_experiments"

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    description: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    project_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
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

    runs: Mapped[
        list["TrackedRunORM"]
    ] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
    )


class TrackedRunORM(Base):
    __tablename__ = "tracker_runs"

    __table_args__ = (
        Index(
            "ix_tracker_runs_experiment_status",
            "experiment_id",
            "status",
        ),
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    experiment_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "tracker_experiments.experiment_id",
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
        index=True,
    )

    model_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
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

    git_commit: Mapped[
        str | None
    ] = mapped_column(
        String(128),
        nullable=True,
    )

    random_seed: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    parameters_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "parameters",
        JSON,
        nullable=False,
        default=dict,
    )

    environment_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "environment",
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

    experiment: Mapped[
        TrackedExperimentORM
    ] = relationship(
        back_populates="runs",
    )

    metrics: Mapped[
        list["TrackedMetricORM"]
    ] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )

    artifacts: Mapped[
        list["TrackedArtifactORM"]
    ] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )

    events: Mapped[
        list["TrackedEventORM"]
    ] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class TrackedMetricORM(Base):
    __tablename__ = "tracker_metrics"

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "metric_name",
            "step",
            name="uq_tracker_metric_step",
        ),
        Index(
            "ix_tracker_metrics_run_name",
            "run_id",
            "metric_name",
        ),
    )

    metric_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "tracker_runs.run_id",
            ondelete="CASCADE",
        ),
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

    step: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    run: Mapped[TrackedRunORM] = relationship(
        back_populates="metrics",
    )


class TrackedArtifactORM(Base):
    __tablename__ = "tracker_artifacts"

    artifact_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "tracker_runs.run_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    artifact_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    artifact_uri: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    sha256: Mapped[
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

    run: Mapped[TrackedRunORM] = relationship(
        back_populates="artifacts",
    )


class TrackedEventORM(Base):
    __tablename__ = "tracker_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "tracker_runs.run_id",
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

    run: Mapped[TrackedRunORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    name: str
    description: str | None
    project_name: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment_id: str
    run_name: str
    status: str
    model_name: str | None
    model_version: str | None
    dataset_version_id: str | None
    git_commit: str | None
    random_seed: int | None
    parameters: dict[str, Any]
    environment: dict[str, Any]
    metadata: dict[str, Any]
    error_message: str | None
    version: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class MetricRecord:
    metric_id: str
    run_id: str
    metric_name: str
    metric_value: float
    step: int
    created_at: str


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    run_id: str
    artifact_name: str
    artifact_type: str
    artifact_uri: str
    sha256: str | None
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RunComparison:
    experiment_id: str
    metric_name: str
    direction: str
    ranked_runs: list[dict[str, Any]]


# ============================================================
# EXPERIMENT TRACKER
# ============================================================

class ExperimentTracker:
    """
    Persistent experiment tracking repository.

    Training services should:
      1. create_experiment()
      2. create_run()
      3. start_run()
      4. log_metric()
      5. log_artifact()
      6. complete_run()
    """

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _get_experiment(
        session: Session,
        experiment_id: str,
    ) -> TrackedExperimentORM:
        row = session.get(
            TrackedExperimentORM,
            experiment_id,
        )

        if row is None:
            raise TrackerNotFoundError(
                "Experiment not found"
            )

        return row

    @staticmethod
    def _get_run(
        session: Session,
        run_id: str,
    ) -> TrackedRunORM:
        row = session.get(
            TrackedRunORM,
            run_id,
        )

        if row is None:
            raise TrackerNotFoundError(
                "Run not found"
            )

        return row

    @staticmethod
    def _experiment_record(
        row: TrackedExperimentORM,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=row.experiment_id,
            name=row.name,
            description=row.description,
            project_name=row.project_name,
            metadata=json_safe(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _run_record(
        row: TrackedRunORM,
    ) -> RunRecord:
        return RunRecord(
            run_id=row.run_id,
            experiment_id=row.experiment_id,
            run_name=row.run_name,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            dataset_version_id=(
                row.dataset_version_id
            ),
            git_commit=row.git_commit,
            random_seed=row.random_seed,
            parameters=json_safe(
                row.parameters_data
            ),
            environment=json_safe(
                row.environment_data
            ),
            metadata=json_safe(
                row.metadata_data
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
    def _metric_record(
        row: TrackedMetricORM,
    ) -> MetricRecord:
        return MetricRecord(
            metric_id=row.metric_id,
            run_id=row.run_id,
            metric_name=row.metric_name,
            metric_value=row.metric_value,
            step=row.step,
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _artifact_record(
        row: TrackedArtifactORM,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row.artifact_id,
            run_id=row.run_id,
            artifact_name=row.artifact_name,
            artifact_type=row.artifact_type,
            artifact_uri=row.artifact_uri,
            sha256=row.sha256,
            metadata=json_safe(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _audit(
        session: Session,
        run_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            TrackedEventORM(
                event_id=uuid.uuid4().hex,
                run_id=run_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "details",
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
        project_name: str = "NeuroWorld AI",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ExperimentRecord:
        name = validate_string(
            name,
            "name",
        )

        project_name = validate_string(
            project_name,
            "project_name",
        )

        if description is not None:
            description = validate_string(
                description,
                "description",
                4000,
            )

        row = TrackedExperimentORM(
            experiment_id=uuid.uuid4().hex,
            name=name,
            project_name=project_name,
            description=description,
            metadata_data=validate_json_dict(
                metadata or {},
                "metadata",
            ),
            created_at=utc_now(),
        )

        with self._session(db) as session:
            existing = session.scalar(
                select(TrackedExperimentORM)
                .where(
                    TrackedExperimentORM.name
                    == name
                )
            )

            if existing is not None:
                raise TrackerConflictError(
                    "Experiment name already exists"
                )

            session.add(row)
            session.flush()

            return self._experiment_record(row)

    # --------------------------------------------------------
    # CREATE RUN
    # --------------------------------------------------------

    def create_run(
        self,
        experiment_id: str,
        *,
        run_name: str,
        model_name: str | None = None,
        model_version: str | None = None,
        dataset_version_id: str | None = None,
        git_commit: str | None = None,
        random_seed: int | None = None,
        parameters: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> RunRecord:
        run_name = validate_string(
            run_name,
            "run_name",
        )

        if model_name is not None:
            model_name = validate_string(
                model_name,
                "model_name",
            )

        if model_version is not None:
            model_version = validate_string(
                model_version,
                "model_version",
                128,
            )

        if dataset_version_id is not None:
            dataset_version_id = validate_string(
                dataset_version_id,
                "dataset_version_id",
                64,
            )

        if git_commit is not None:
            git_commit = validate_string(
                git_commit,
                "git_commit",
                128,
            )

        if random_seed is not None:
            if (
                isinstance(random_seed, bool)
                or not isinstance(random_seed, int)
                or random_seed < 0
            ):
                raise TrackerValidationError(
                    "Invalid random_seed"
                )

        row = TrackedRunORM(
            run_id=uuid.uuid4().hex,
            experiment_id=experiment_id,
            run_name=run_name,
            status=RunStatus.CREATED.value,
            model_name=model_name,
            model_version=model_version,
            dataset_version_id=(
                dataset_version_id
            ),
            git_commit=git_commit,
            random_seed=random_seed,
            parameters_data=validate_json_dict(
                parameters or {},
                "parameters",
            ),
            environment_data=validate_json_dict(
                environment or {},
                "environment",
            ),
            metadata_data=validate_json_dict(
                metadata or {},
                "metadata",
            ),
            version=1,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            self._get_experiment(
                session,
                experiment_id,
            )

            session.add(row)

            self._audit(
                session,
                row.run_id,
                "run_created",
                {
                    "run_name": run_name,
                },
            )

            session.flush()

            return self._run_record(row)

    # --------------------------------------------------------
    # GET EXPERIMENT / RUN
    # --------------------------------------------------------

    def get_experiment(
        self,
        experiment_id: str,
        *,
        db: Session | None = None,
    ) -> ExperimentRecord:
        with self._session(db) as session:
            return self._experiment_record(
                self._get_experiment(
                    session,
                    experiment_id,
                )
            )

    def get_run(
        self,
        run_id: str,
        *,
        db: Session | None = None,
    ) -> RunRecord:
        with self._session(db) as session:
            return self._run_record(
                self._get_run(
                    session,
                    run_id,
                )
            )

    # --------------------------------------------------------
    # RUN LIFECYCLE
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        run_id: str,
        expected_version: int,
        allowed_from: set[RunStatus],
        target: RunStatus,
        values: dict[str, Any] | None = None,
    ) -> RunRecord:
        result = session.execute(
            update(TrackedRunORM)
            .where(
                TrackedRunORM.run_id == run_id,
                TrackedRunORM.version
                == expected_version,
                TrackedRunORM.status.in_(
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
            raise TrackerConflictError(
                "Run status/version changed"
            )

        self._audit(
            session,
            run_id,
            f"run_{target.value}",
        )

        session.flush()
        session.expire_all()

        return self._run_record(
            self._get_run(
                session,
                run_id,
            )
        )

    def start_run(
        self,
        run_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> RunRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                run_id=run_id,
                expected_version=expected_version,
                allowed_from={RunStatus.CREATED},
                target=RunStatus.RUNNING,
                values={
                    "started_at": utc_now(),
                },
            )

    def complete_run(
        self,
        run_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> RunRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                run_id=run_id,
                expected_version=expected_version,
                allowed_from={RunStatus.RUNNING},
                target=RunStatus.COMPLETED,
                values={
                    "completed_at": utc_now(),
                },
            )

    def fail_run(
        self,
        run_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> RunRecord:
        reason = validate_string(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                run_id=run_id,
                expected_version=expected_version,
                allowed_from={
                    RunStatus.CREATED,
                    RunStatus.RUNNING,
                },
                target=RunStatus.FAILED,
                values={
                    "error_message": reason,
                    "completed_at": utc_now(),
                },
            )

    def cancel_run(
        self,
        run_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> RunRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                run_id=run_id,
                expected_version=expected_version,
                allowed_from={
                    RunStatus.CREATED,
                    RunStatus.RUNNING,
                },
                target=RunStatus.CANCELLED,
                values={
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # LOG METRIC
    # --------------------------------------------------------

    def log_metric(
        self,
        run_id: str,
        *,
        metric_name: str,
        metric_value: float,
        step: int,
        db: Session | None = None,
    ) -> MetricRecord:
        metric_name = validate_string(
            metric_name,
            "metric_name",
            128,
        )

        metric_value = finite_number(
            metric_value,
            "metric_value",
        )

        step = validate_step(
            step
        )

        with self._session(db) as session:
            run = self._get_run(
                session,
                run_id,
            )

            if run.status != RunStatus.RUNNING.value:
                raise TrackerStateError(
                    "Run must be running "
                    "to log metrics"
                )

            existing = session.scalar(
                select(TrackedMetricORM)
                .where(
                    TrackedMetricORM.run_id == run_id,
                    TrackedMetricORM.metric_name
                    == metric_name,
                    TrackedMetricORM.step == step,
                )
            )

            if existing is not None:
                raise TrackerConflictError(
                    "Metric already logged at this step"
                )

            row = TrackedMetricORM(
                metric_id=uuid.uuid4().hex,
                run_id=run_id,
                metric_name=metric_name,
                metric_value=metric_value,
                step=step,
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._metric_record(row)

    # --------------------------------------------------------
    # LOG MULTIPLE METRICS
    # --------------------------------------------------------

    def log_metrics(
        self,
        run_id: str,
        *,
        metrics: dict[str, float],
        step: int,
        db: Session | None = None,
    ) -> list[MetricRecord]:
        if not isinstance(metrics, dict) or not metrics:
            raise TrackerValidationError(
                "metrics must be a nonempty dictionary"
            )

        with self._session(db) as session:
            return [
                self.log_metric(
                    run_id,
                    metric_name=name,
                    metric_value=value,
                    step=step,
                    db=session,
                )
                for name, value in metrics.items()
            ]

    # --------------------------------------------------------
    # LOG ARTIFACT
    # --------------------------------------------------------

    def log_artifact(
        self,
        run_id: str,
        *,
        artifact_name: str,
        artifact_type: ArtifactType,
        artifact_uri: str,
        sha256: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ArtifactRecord:
        artifact_name = validate_string(
            artifact_name,
            "artifact_name",
        )

        artifact_uri = validate_string(
            artifact_uri,
            "artifact_uri",
            4000,
        )

        if not isinstance(
            artifact_type,
            ArtifactType,
        ):
            raise TrackerValidationError(
                "Invalid artifact type"
            )

        if sha256 is not None:
            sha256 = validate_string(
                sha256,
                "sha256",
                64,
            ).lower()

            if (
                len(sha256) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in sha256
                )
            ):
                raise TrackerValidationError(
                    "Invalid SHA-256 digest"
                )

        with self._session(db) as session:
            run = self._get_run(
                session,
                run_id,
            )

            if run.status not in {
                RunStatus.RUNNING.value,
                RunStatus.COMPLETED.value,
            }:
                raise TrackerStateError(
                    "Cannot log artifact "
                    "for this run status"
                )

            row = TrackedArtifactORM(
                artifact_id=uuid.uuid4().hex,
                run_id=run_id,
                artifact_name=artifact_name,
                artifact_type=artifact_type.value,
                artifact_uri=artifact_uri,
                sha256=sha256,
                metadata_data=validate_json_dict(
                    metadata or {},
                    "metadata",
                ),
                created_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                run_id,
                "artifact_logged",
                {
                    "artifact_id": row.artifact_id,
                    "artifact_name": artifact_name,
                },
            )

            session.flush()

            return self._artifact_record(row)

    # --------------------------------------------------------
    # LIST METRICS
    # --------------------------------------------------------

    def list_metrics(
        self,
        run_id: str,
        *,
        metric_name: str | None = None,
        limit: int = 1000,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[MetricRecord]:
        validate_limit(
            limit,
            offset,
        )

        statement = (
            select(TrackedMetricORM)
            .where(
                TrackedMetricORM.run_id == run_id
            )
        )

        if metric_name is not None:
            statement = statement.where(
                TrackedMetricORM.metric_name
                == metric_name
            )

        statement = (
            statement
            .order_by(
                TrackedMetricORM.step.asc(),
                TrackedMetricORM.metric_name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._metric_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST ARTIFACTS
    # --------------------------------------------------------

    def list_artifacts(
        self,
        run_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[ArtifactRecord]:
        validate_limit(
            limit,
            offset,
        )

        statement = (
            select(TrackedArtifactORM)
            .where(
                TrackedArtifactORM.run_id == run_id
            )
            .order_by(
                TrackedArtifactORM.created_at.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._artifact_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

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
    ) -> list[RunRecord]:
        validate_limit(
            limit,
            offset,
        )

        statement = (
            select(TrackedRunORM)
            .where(
                TrackedRunORM.experiment_id
                == experiment_id
            )
        )

        if status is not None:
            statement = statement.where(
                TrackedRunORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                TrackedRunORM.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._run_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # COMPARE RUNS
    # --------------------------------------------------------

    def compare_runs(
        self,
        experiment_id: str,
        *,
        metric_name: str,
        direction: MetricDirection = (
            MetricDirection.MINIMIZE
        ),
        db: Session | None = None,
    ) -> RunComparison:
        metric_name = validate_string(
            metric_name,
            "metric_name",
            128,
        )

        if not isinstance(
            direction,
            MetricDirection,
        ):
            raise TrackerValidationError(
                "Invalid metric direction"
            )

        with self._session(db) as session:
            self._get_experiment(
                session,
                experiment_id,
            )

            runs = session.scalars(
                select(TrackedRunORM)
                .where(
                    TrackedRunORM.experiment_id
                    == experiment_id,
                    TrackedRunORM.status
                    == RunStatus.COMPLETED.value,
                )
            ).all()

            ranked = []

            for run in runs:
                latest_metric = session.scalar(
                    select(TrackedMetricORM)
                    .where(
                        TrackedMetricORM.run_id
                        == run.run_id,
                        TrackedMetricORM.metric_name
                        == metric_name,
                    )
                    .order_by(
                        TrackedMetricORM.step.desc()
                    )
                    .limit(1)
                )

                if latest_metric is None:
                    continue

                ranked.append(
                    {
                        "run_id": run.run_id,
                        "run_name": run.run_name,
                        "model_name": run.model_name,
                        "metric_value": (
                            latest_metric.metric_value
                        ),
                        "step": latest_metric.step,
                        "parameters": json_safe(
                            run.parameters_data
                        ),
                    }
                )

        ranked.sort(
            key=lambda item: item["metric_value"],
            reverse=(
                direction
                == MetricDirection.MAXIMIZE
            ),
        )

        return RunComparison(
            experiment_id=experiment_id,
            metric_name=metric_name,
            direction=direction.value,
            ranked_runs=ranked,
        )

    # --------------------------------------------------------
    # BEST RUN
    # --------------------------------------------------------

    def get_best_run(
        self,
        experiment_id: str,
        *,
        metric_name: str,
        direction: MetricDirection = (
            MetricDirection.MINIMIZE
        ),
        db: Session | None = None,
    ) -> dict[str, Any] | None:
        comparison = self.compare_runs(
            experiment_id,
            metric_name=metric_name,
            direction=direction,
            db=db,
        )

        if not comparison.ranked_runs:
            return None

        return comparison.ranked_runs[0]

    # --------------------------------------------------------
    # EXPORT RUN JSON
    # --------------------------------------------------------

    def export_run_json(
        self,
        run_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            run = self.get_run(
                run_id,
                db=session,
            )

            metrics = []

            offset = 0

            while True:
                page = self.list_metrics(
                    run_id,
                    limit=1000,
                    offset=offset,
                    db=session,
                )

                if not page:
                    break

                metrics.extend(page)
                offset += len(page)

            artifacts = []
            offset = 0

            while True:
                page = self.list_artifacts(
                    run_id,
                    limit=1000,
                    offset=offset,
                    db=session,
                )

                if not page:
                    break

                artifacts.extend(page)
                offset += len(page)

            payload = {
                "run": asdict(run),
                "metrics": [
                    asdict(metric)
                    for metric in metrics
                ],
                "artifacts": [
                    asdict(artifact)
                    for artifact in artifacts
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
        run_id: str,
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
            "metric_id",
            "run_id",
            "metric_name",
            "metric_value",
            "step",
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
                metrics = self.list_metrics(
                    run_id,
                    limit=1000,
                    offset=offset,
                    db=db,
                )

                if not metrics:
                    break

                for metric in metrics:
                    writer.writerow(
                        asdict(metric)
                    )

                offset += len(metrics)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_experiment_tracker() -> ExperimentTracker:
    return ExperimentTracker()


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    create_database_tables()

    tracker = create_experiment_tracker()

    experiment = tracker.create_experiment(
        name=(
            "neuroworld_world_model_"
            + uuid.uuid4().hex[:8]
        ),
        project_name="NeuroWorld AI",
        description=(
            "Synthetic world-model training "
            "experiment"
        ),
    )

    run = tracker.create_run(
        experiment.experiment_id,
        run_name="transformer_baseline_v1",
        model_name="neuroworld_transformer",
        model_version="v1",
        dataset_version_id="synthetic_dataset_v1",
        random_seed=42,
        parameters={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 3,
            "optimizer": "adamw",
        },
        environment={
            "framework": "pytorch",
            "device": "cpu",
        },
    )

    run = tracker.start_run(
        run.run_id,
        expected_version=run.version,
    )

    for step, (train_loss, val_loss) in enumerate(
        [
            (0.80, 0.75),
            (0.55, 0.50),
            (0.35, 0.32),
        ],
        start=1,
    ):
        tracker.log_metrics(
            run.run_id,
            step=step,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )

    tracker.log_artifact(
        run.run_id,
        artifact_name="world_model_checkpoint",
        artifact_type=ArtifactType.CHECKPOINT,
        artifact_uri=(
            "models/checkpoints/"
            "demo_world_model.pt"
        ),
        metadata={
            "illustrative": True,
        },
    )

    run = tracker.complete_run(
        run.run_id,
        expected_version=run.version,
    )

    best = tracker.get_best_run(
        experiment.experiment_id,
        metric_name="val_loss",
        direction=MetricDirection.MINIMIZE,
    )

    print("=" * 60)
    print("NEUROWORLD AI - EXPERIMENT TRACKER")
    print("=" * 60)

    print("Experiment:", experiment.name)
    print("Run:", run.run_name)
    print("Status:", run.status)
    print("Best run:", best)

    json_path = tracker.export_run_json(
        run.run_id,
        "experiments/reports/"
        "experiment_tracker_run.json",
    )

    csv_path = tracker.export_metrics_csv(
        run.run_id,
        "experiments/reports/"
        "experiment_tracker_metrics.csv",
    )

    print("JSON:", json_path)
    print("CSV:", csv_path)

    close_database()


if __name__ == "__main__":
    main()
