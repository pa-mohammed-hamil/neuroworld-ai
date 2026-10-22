
"""
NeuroWorld AI
backend/app/database/trajectory.py

Persistent Trajectory and Simulation History Storage.

Features:
- Create trajectory records
- Record action/state transitions
- Store observed and predicted states
- Store rewards, uncertainty and terminal flags
- Store simulation episode metadata
- Track trajectory status
- Retrieve and replay trajectories
- Calculate trajectory statistics
- Compare predicted and observed states
- Filter and paginate trajectory history
- Export trajectories to JSON and CSV
- PostgreSQL and SQLite support

Dependencies:
    pip install sqlalchemy numpy

Run:
    python -m backend.app.database.trajectory

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Important:
    This module stores trajectories; it does not
    execute model inference or physical actions.

    A trajectory represents one ordered sequence
    of recorded state transitions.

    Production deployments should use migrations,
    access controls and bounded retention.
"""

from __future__ import annotations

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

import numpy as np

from sqlalchemy import (
    Boolean,
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
    "neuroworld.database.trajectory"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TrajectoryDatabaseError(Exception):
    """Base trajectory database exception."""


class TrajectoryValidationError(TrajectoryDatabaseError):
    """Invalid trajectory data."""


class TrajectoryNotFoundError(TrajectoryDatabaseError):
    """Trajectory or step does not exist."""


class TrajectoryStateError(TrajectoryDatabaseError):
    """Invalid trajectory state transition."""


# ============================================================
# ENUMS
# ============================================================

class TrajectoryStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrajectorySource(str, Enum):
    SIMULATION = "simulation"
    PREDICTION = "prediction"
    IMAGINATION = "imagination"
    PLANNING = "planning"
    OBSERVATION = "observation"


TERMINAL_STATUSES = {
    TrajectoryStatus.COMPLETED,
    TrajectoryStatus.FAILED,
    TrajectoryStatus.CANCELLED,
}


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
    field_name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise TrajectoryValidationError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise TrajectoryValidationError(
            f"Invalid {field_name}"
        )

    return value


def json_safe(value: Any) -> Any:
    """
    Convert NumPy values to JSON-compatible data.

    Reject NaN, infinity and unsupported objects.
    """

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, Enum):
        return value.value

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
            raise TrajectoryValidationError(
                "Non-finite numbers are not allowed"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise TrajectoryValidationError(
        f"Unsupported data type: "
        f"{type(value).__name__}"
    )


def validate_vector(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
) -> list[float] | None:
    if value is None and allow_none:
        return None

    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )

    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(
            f"{name} must be numeric"
        ) from exc

    if array.ndim != 1 or array.size < 1:
        raise TrajectoryValidationError(
            f"{name} must be a nonempty 1D vector"
        )

    if not np.all(np.isfinite(array)):
        raise TrajectoryValidationError(
            f"{name} must contain finite values"
        )

    return array.tolist()


def validate_optional_number(
    value: float | None,
    name: str,
) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)

    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise TrajectoryValidationError(
            f"{name} must be finite"
        )

    return number


# ============================================================
# ORM MODELS
# ============================================================

class TrajectoryORM(Base):
    __tablename__ = "trajectories"

    __table_args__ = (
        Index(
            "ix_trajectories_world_created",
            "world_id",
            "created_at",
        ),
    )

    trajectory_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    trajectory_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=TrajectoryStatus.CREATED.value,
        index=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    initial_state_data: Mapped[list[float]] = mapped_column(
        "initial_state",
        JSON,
        nullable=False,
    )

    metadata_data: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    total_reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
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

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    steps: Mapped[list["TrajectoryStepORM"]] = relationship(
        back_populates="trajectory",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TrajectoryStepORM.step_index",
    )


class TrajectoryStepORM(Base):
    __tablename__ = "trajectory_steps"

    __table_args__ = (
        UniqueConstraint(
            "trajectory_id",
            "step_index",
            name="uq_trajectory_step_index",
        ),
    )

    step_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    trajectory_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "trajectories.trajectory_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    state_data: Mapped[list[float]] = mapped_column(
        "state",
        JSON,
        nullable=False,
    )

    action_data: Mapped[list[float]] = mapped_column(
        "action",
        JSON,
        nullable=False,
    )

    predicted_next_state_data: Mapped[
        list[float] | None
    ] = mapped_column(
        "predicted_next_state",
        JSON,
        nullable=True,
    )

    observed_next_state_data: Mapped[
        list[float] | None
    ] = mapped_column(
        "observed_next_state",
        JSON,
        nullable=True,
    )

    reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    uncertainty: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    terminal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    collision: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    trajectory: Mapped[TrajectoryORM] = relationship(
        back_populates="steps",
    )


# ============================================================
# PUBLIC DATA MODELS
# ============================================================

@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    world_id: str
    trajectory_name: str

    source: str
    status: str

    model_name: str
    model_version: str

    initial_state: list[float]
    metadata: dict[str, Any]

    total_reward: float
    step_count: int

    created_at: str
    started_at: str | None
    finished_at: str | None

    error_message: str | None


@dataclass(frozen=True)
class TrajectoryStepRecord:
    step_id: str
    trajectory_id: str
    step_index: int

    state: list[float]
    action: list[float]

    predicted_next_state: list[float] | None
    observed_next_state: list[float] | None

    reward: float
    uncertainty: float | None

    terminal: bool
    collision: bool

    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TrajectoryStatistics:
    trajectory_id: str

    step_count: int
    total_reward: float
    mean_reward: float

    mean_uncertainty: float | None

    collision_count: int
    terminal_reached: bool

    prediction_rmse: float | None
    prediction_mae: float | None


@dataclass(frozen=True)
class TrajectoryReplay:
    trajectory: TrajectoryRecord
    steps: list[TrajectoryStepRecord]


# ============================================================
# REPOSITORY
# ============================================================

class TrajectoryRepository:
    """
    Persistent trajectory repository.

    Supports caller-managed or automatically
    managed SQLAlchemy sessions.

    Caller-owned sessions are never committed
    or closed by this repository.
    """

    def __init__(
        self,
        *,
        max_steps_per_trajectory: int = 10000,
        max_state_dimension: int = 4096,
        max_action_dimension: int = 1024,
    ) -> None:
        if min(
            max_steps_per_trajectory,
            max_state_dimension,
            max_action_dimension,
        ) < 1:
            raise ValueError(
                "Repository limits must be positive"
            )

        self.max_steps_per_trajectory = (
            max_steps_per_trajectory
        )

        self.max_state_dimension = (
            max_state_dimension
        )

        self.max_action_dimension = (
            max_action_dimension
        )

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
    def _to_trajectory_record(
        row: TrajectoryORM,
    ) -> TrajectoryRecord:
        return TrajectoryRecord(
            trajectory_id=row.trajectory_id,
            world_id=row.world_id,
            trajectory_name=row.trajectory_name,
            source=row.source,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            initial_state=list(
                row.initial_state_data
            ),
            metadata=dict(
                row.metadata_data
            ),
            total_reward=row.total_reward,
            step_count=row.step_count,
            created_at=datetime_string(
                row.created_at
            ),
            started_at=datetime_string(
                row.started_at
            ),
            finished_at=datetime_string(
                row.finished_at
            ),
            error_message=row.error_message,
        )

    @staticmethod
    def _to_step_record(
        row: TrajectoryStepORM,
    ) -> TrajectoryStepRecord:
        return TrajectoryStepRecord(
            step_id=row.step_id,
            trajectory_id=row.trajectory_id,
            step_index=row.step_index,
            state=list(row.state_data),
            action=list(row.action_data),
            predicted_next_state=(
                list(row.predicted_next_state_data)
                if row.predicted_next_state_data is not None
                else None
            ),
            observed_next_state=(
                list(row.observed_next_state_data)
                if row.observed_next_state_data is not None
                else None
            ),
            reward=row.reward,
            uncertainty=row.uncertainty,
            terminal=row.terminal,
            collision=row.collision,
            metadata=dict(
                row.metadata_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    # --------------------------------------------------------
    # CREATE TRAJECTORY
    # --------------------------------------------------------

    def create_trajectory(
        self,
        *,
        world_id: str,
        initial_state: Any,
        source: TrajectorySource = (
            TrajectorySource.SIMULATION
        ),
        trajectory_name: str | None = None,
        model_name: str = "unknown",
        model_version: str = "unknown",
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> TrajectoryRecord:
        world_id = validate_identifier(
            world_id,
            "world_id",
        )

        model_name = validate_identifier(
            model_name,
            "model_name",
        )

        model_version = validate_identifier(
            model_version,
            "model_version",
        )

        if not isinstance(source, TrajectorySource):
            raise TrajectoryValidationError(
                "Invalid trajectory source"
            )

        initial = validate_vector(
            initial_state,
            "initial_state",
        )

        if len(initial) > self.max_state_dimension:
            raise TrajectoryValidationError(
                "Initial state dimension exceeds limit"
            )

        metadata = json_safe(
            metadata or {}
        )

        if not isinstance(metadata, dict):
            raise TrajectoryValidationError(
                "metadata must be a dictionary"
            )

        trajectory_id = uuid.uuid4().hex

        row = TrajectoryORM(
            trajectory_id=trajectory_id,
            world_id=world_id,
            trajectory_name=(
                trajectory_name
                or f"trajectory-{trajectory_id[:8]}"
            ),
            source=source.value,
            status=TrajectoryStatus.CREATED.value,
            model_name=model_name,
            model_version=model_version,
            initial_state_data=initial,
            metadata_data=metadata,
            total_reward=0.0,
            step_count=0,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            return self._to_trajectory_record(row)

    # --------------------------------------------------------
    # GET TRAJECTORY
    # --------------------------------------------------------

    def get_trajectory(
        self,
        trajectory_id: str,
        *,
        world_id: str | None = None,
        db: Session | None = None,
    ) -> TrajectoryRecord:
        with self._session(db) as session:
            row = session.get(
                TrajectoryORM,
                trajectory_id,
            )

            if row is None:
                raise TrajectoryNotFoundError(
                    "Trajectory not found"
                )

            if (
                world_id is not None
                and row.world_id != world_id
            ):
                raise TrajectoryNotFoundError(
                    "Trajectory not found"
                )

            return self._to_trajectory_record(row)

    # --------------------------------------------------------
    # UPDATE STATUS
    # --------------------------------------------------------

    def update_status(
        self,
        trajectory_id: str,
        status: TrajectoryStatus,
        *,
        error_message: str | None = None,
        db: Session | None = None,
    ) -> TrajectoryRecord:
        if not isinstance(status, TrajectoryStatus):
            raise TrajectoryValidationError(
                "Invalid trajectory status"
            )

        allowed = {
            TrajectoryStatus.CREATED: {
                TrajectoryStatus.RUNNING,
                TrajectoryStatus.CANCELLED,
            },
            TrajectoryStatus.RUNNING: {
                TrajectoryStatus.COMPLETED,
                TrajectoryStatus.FAILED,
                TrajectoryStatus.CANCELLED,
            },
            TrajectoryStatus.COMPLETED: set(),
            TrajectoryStatus.FAILED: set(),
            TrajectoryStatus.CANCELLED: set(),
        }

        with self._session(db) as session:
            row = session.get(
                TrajectoryORM,
                trajectory_id,
            )

            if row is None:
                raise TrajectoryNotFoundError(
                    "Trajectory not found"
                )

            current = TrajectoryStatus(
                row.status
            )

            if status not in allowed[current]:
                raise TrajectoryStateError(
                    f"Invalid transition: "
                    f"{current.value} -> {status.value}"
                )

            row.status = status.value

            now = utc_now()

            if status == TrajectoryStatus.RUNNING:
                row.started_at = now

            if status in TERMINAL_STATUSES:
                row.finished_at = now

            if status == TrajectoryStatus.FAILED:
                row.error_message = (
                    error_message or "Unknown error"
                )

            session.flush()

            return self._to_trajectory_record(row)

    # --------------------------------------------------------
    # ADD TRAJECTORY STEP
    # --------------------------------------------------------

    def add_step(
        self,
        trajectory_id: str,
        *,
        state: Any,
        action: Any,
        predicted_next_state: Any = None,
        observed_next_state: Any = None,
        reward: float = 0.0,
        uncertainty: float | None = None,
        terminal: bool = False,
        collision: bool = False,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> TrajectoryStepRecord:
        state_vector = validate_vector(
            state,
            "state",
        )

        action_vector = validate_vector(
            action,
            "action",
        )

        predicted_vector = validate_vector(
            predicted_next_state,
            "predicted_next_state",
            allow_none=True,
        )

        observed_vector = validate_vector(
            observed_next_state,
            "observed_next_state",
            allow_none=True,
        )

        if len(state_vector) > self.max_state_dimension:
            raise TrajectoryValidationError(
                "State dimension exceeds limit"
            )

        if len(action_vector) > self.max_action_dimension:
            raise TrajectoryValidationError(
                "Action dimension exceeds limit"
            )

        for name, vector in (
            ("predicted_next_state", predicted_vector),
            ("observed_next_state", observed_vector),
        ):
            if (
                vector is not None
                and len(vector) != len(state_vector)
            ):
                raise TrajectoryValidationError(
                    f"{name} dimension mismatch"
                )

        reward = validate_optional_number(
            reward,
            "reward",
        )

        uncertainty = validate_optional_number(
            uncertainty,
            "uncertainty",
        )

        if uncertainty is not None and not (
            0 <= uncertainty <= 1
        ):
            raise TrajectoryValidationError(
                "uncertainty must be in [0, 1]"
            )

        if not isinstance(terminal, bool):
            raise TrajectoryValidationError(
                "terminal must be boolean"
            )

        if not isinstance(collision, bool):
            raise TrajectoryValidationError(
                "collision must be boolean"
            )

        metadata = json_safe(
            metadata or {}
        )

        if not isinstance(metadata, dict):
            raise TrajectoryValidationError(
                "metadata must be a dictionary"
            )

        with self._session(db) as session:
            row = session.get(
                TrajectoryORM,
                trajectory_id,
            )

            if row is None:
                raise TrajectoryNotFoundError(
                    "Trajectory not found"
                )

            if row.status != (
                TrajectoryStatus.RUNNING.value
            ):
                raise TrajectoryStateError(
                    "Steps can only be added to "
                    "running trajectories"
                )

            if row.step_count >= (
                self.max_steps_per_trajectory
            ):
                raise TrajectoryValidationError(
                    "Maximum trajectory length reached"
                )

            # Validate state continuity.
            if row.step_count == 0:
                expected_state = np.asarray(
                    row.initial_state_data,
                    dtype=np.float64,
                )

            else:
                previous = session.scalar(
                    select(TrajectoryStepORM)
                    .where(
                        TrajectoryStepORM.trajectory_id
                        == trajectory_id,
                        TrajectoryStepORM.step_index
                        == row.step_count - 1,
                    )
                )

                if previous is None:
                    raise TrajectoryStateError(
                        "Previous trajectory step missing"
                    )

                expected_state = np.asarray(
                    previous.observed_next_state_data
                    if previous.observed_next_state_data
                    is not None
                    else previous.predicted_next_state_data,
                    dtype=np.float64,
                ) if (
                    previous.observed_next_state_data
                    is not None
                    or previous.predicted_next_state_data
                    is not None
                ) else None

            if expected_state is not None:
                if (
                    expected_state.shape
                    != (len(state_vector),)
                    or not np.allclose(
                        expected_state,
                        state_vector,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                ):
                    raise TrajectoryValidationError(
                        "State is not continuous with "
                        "the previous trajectory step"
                    )

            step = TrajectoryStepORM(
                step_id=uuid.uuid4().hex,
                trajectory_id=trajectory_id,
                step_index=row.step_count,
                state_data=state_vector,
                action_data=action_vector,
                predicted_next_state_data=predicted_vector,
                observed_next_state_data=observed_vector,
                reward=reward,
                uncertainty=uncertainty,
                terminal=terminal,
                collision=collision,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(step)

            row.step_count += 1
            row.total_reward += reward

            session.flush()

            return self._to_step_record(step)

    # --------------------------------------------------------
    # GET STEPS
    # --------------------------------------------------------

    def get_steps(
        self,
        trajectory_id: str,
        *,
        limit: int = 10000,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[TrajectoryStepRecord]:
        if not 1 <= limit <= 100000 or offset < 0:
            raise TrajectoryValidationError(
                "Invalid pagination"
            )

        with self._session(db) as session:
            statement = (
                select(TrajectoryStepORM)
                .where(
                    TrajectoryStepORM.trajectory_id
                    == trajectory_id
                )
                .order_by(
                    TrajectoryStepORM.step_index.asc()
                )
                .offset(offset)
                .limit(limit)
            )

            rows = session.scalars(
                statement
            ).all()

            return [
                self._to_step_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # REPLAY
    # --------------------------------------------------------

    def replay_trajectory(
        self,
        trajectory_id: str,
        *,
        db: Session | None = None,
    ) -> TrajectoryReplay:
        trajectory = self.get_trajectory(
            trajectory_id,
            db=db,
        )

        if trajectory.step_count > (
            self.max_steps_per_trajectory
        ):
            raise TrajectoryValidationError(
                "Trajectory exceeds replay limit"
            )

        steps = self.get_steps(
            trajectory_id,
            limit=self.max_steps_per_trajectory,
            db=db,
        )

        if len(steps) != trajectory.step_count:
            raise TrajectoryStateError(
                "Trajectory history is incomplete"
            )

        return TrajectoryReplay(
            trajectory=trajectory,
            steps=steps,
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def calculate_statistics(
        self,
        trajectory_id: str,
        *,
        db: Session | None = None,
    ) -> TrajectoryStatistics:
        replay = self.replay_trajectory(
            trajectory_id,
            db=db,
        )

        steps = replay.steps

        rewards = [
            step.reward
            for step in steps
        ]

        uncertainties = [
            step.uncertainty
            for step in steps
            if step.uncertainty is not None
        ]

        prediction_errors = []

        for step in steps:
            if (
                step.predicted_next_state is not None
                and step.observed_next_state is not None
            ):
                predicted = np.asarray(
                    step.predicted_next_state,
                    dtype=np.float64,
                )

                observed = np.asarray(
                    step.observed_next_state,
                    dtype=np.float64,
                )

                prediction_errors.extend(
                    (predicted - observed).tolist()
                )

        prediction_rmse = None
        prediction_mae = None

        if prediction_errors:
            errors = np.asarray(
                prediction_errors,
                dtype=np.float64,
            )

            prediction_rmse = float(
                np.sqrt(
                    np.mean(
                        errors ** 2
                    )
                )
            )

            prediction_mae = float(
                np.mean(
                    np.abs(errors)
                )
            )

        return TrajectoryStatistics(
            trajectory_id=trajectory_id,
            step_count=len(steps),
            total_reward=float(
                sum(rewards)
            ),
            mean_reward=(
                float(np.mean(rewards))
                if rewards
                else 0.0
            ),
            mean_uncertainty=(
                float(np.mean(uncertainties))
                if uncertainties
                else None
            ),
            collision_count=sum(
                step.collision
                for step in steps
            ),
            terminal_reached=any(
                step.terminal
                for step in steps
            ),
            prediction_rmse=prediction_rmse,
            prediction_mae=prediction_mae,
        )

    # --------------------------------------------------------
    # LIST TRAJECTORIES
    # --------------------------------------------------------

    def list_trajectories(
        self,
        *,
        world_id: str | None = None,
        source: TrajectorySource | None = None,
        status: TrajectoryStatus | None = None,
        model_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[TrajectoryRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise TrajectoryValidationError(
                "Invalid pagination"
            )

        statement = select(
            TrajectoryORM
        )

        if world_id is not None:
            statement = statement.where(
                TrajectoryORM.world_id == world_id
            )

        if source is not None:
            statement = statement.where(
                TrajectoryORM.source == source.value
            )

        if status is not None:
            statement = statement.where(
                TrajectoryORM.status == status.value
            )

        if model_name is not None:
            statement = statement.where(
                TrajectoryORM.model_name == model_name
            )

        statement = (
            statement
            .order_by(
                TrajectoryORM.created_at.desc(),
                TrajectoryORM.trajectory_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._to_trajectory_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # DELETE TRAJECTORY
    # --------------------------------------------------------

    def delete_trajectory(
        self,
        trajectory_id: str,
        *,
        db: Session | None = None,
    ) -> bool:
        with self._session(db) as session:
            row = session.get(
                TrajectoryORM,
                trajectory_id,
            )

            if row is None:
                return False

            session.delete(row)
            session.flush()

            return True

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        trajectory_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        replay = self.replay_trajectory(
            trajectory_id,
            db=db,
        )

        payload = {
            "trajectory": asdict(
                replay.trajectory
            ),
            "steps": [
                asdict(step)
                for step in replay.steps
            ],
            "statistics": asdict(
                self.calculate_statistics(
                    trajectory_id,
                    db=db,
                )
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
        trajectory_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        steps = self.get_steps(
            trajectory_id,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "step_id",
            "trajectory_id",
            "step_index",
            "state",
            "action",
            "predicted_next_state",
            "observed_next_state",
            "reward",
            "uncertainty",
            "terminal",
            "collision",
            "metadata",
            "created_at",
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

            for step in steps:
                row = asdict(step)

                for key in (
                    "state",
                    "action",
                    "predicted_next_state",
                    "observed_next_state",
                    "metadata",
                ):
                    row[key] = json.dumps(
                        row[key],
                        allow_nan=False,
                    )

                writer.writerow(row)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_trajectory_repository(
    *,
    max_steps_per_trajectory: int = 10000,
    max_state_dimension: int = 4096,
    max_action_dimension: int = 1024,
) -> TrajectoryRepository:
    return TrajectoryRepository(
        max_steps_per_trajectory=(
            max_steps_per_trajectory
        ),
        max_state_dimension=max_state_dimension,
        max_action_dimension=max_action_dimension,
    )


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

    repository = create_trajectory_repository()

    trajectory = repository.create_trajectory(
        world_id="demo_world_001",
        initial_state=[0.0, 0.0, 0.0, 0.0],
        source=TrajectorySource.SIMULATION,
        trajectory_name="navigation_demo",
        model_name="demo_world_model",
        model_version="v1.0",
        metadata={
            "environment": "synthetic_navigation",
        },
    )

    repository.update_status(
        trajectory.trajectory_id,
        TrajectoryStatus.RUNNING,
    )

    state = np.zeros(
        4,
        dtype=np.float64,
    )

    for step_index in range(10):
        action = np.array(
            [0.1, 0.2, 0.0, 0.0],
            dtype=np.float64,
        )

        next_state = (
            state + 0.5 * action
        )

        repository.add_step(
            trajectory.trajectory_id,
            state=state,
            action=action,
            predicted_next_state=next_state,
            observed_next_state=next_state,
            reward=1.0,
            uncertainty=0.05,
            terminal=(step_index == 9),
            collision=False,
        )

        state = next_state

    repository.update_status(
        trajectory.trajectory_id,
        TrajectoryStatus.COMPLETED,
    )

    statistics = repository.calculate_statistics(
        trajectory.trajectory_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - TRAJECTORY DATABASE")
    print("=" * 60)

    print(
        "Trajectory ID:",
        trajectory.trajectory_id,
    )

    print(
        "Statistics:",
        asdict(statistics),
    )

    json_path = repository.export_json(
        trajectory.trajectory_id,
        "experiments/reports/trajectory_history.json",
    )

    csv_path = repository.export_csv(
        trajectory.trajectory_id,
        "experiments/reports/trajectory_history.csv",
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
