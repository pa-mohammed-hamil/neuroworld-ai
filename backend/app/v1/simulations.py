
"""
NeuroWorld AI
backend/app/database/simulations.py

Persistent simulation management.

Features:
    - Simulation session management
    - Environment and world associations
    - Simulation lifecycle tracking
    - State/action/reward histories
    - Predicted and observed state storage
    - Collision and safety-event tracking
    - Episode termination management
    - Simulation statistics
    - Optimistic concurrency control
    - JSON and CSV exports

Dependencies:
    pip install sqlalchemy numpy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This is a database repository, not a physics
engine or real-world hardware controller.
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
from typing import Any, Sequence

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

class SimulationDatabaseError(Exception):
    """Base simulation database exception."""


class SimulationValidationError(
    SimulationDatabaseError
):
    """Invalid simulation input."""


class SimulationNotFoundError(
    SimulationDatabaseError
):
    """Simulation record was not found."""


class SimulationStateError(
    SimulationDatabaseError
):
    """Invalid simulation state transition."""


class SimulationConflictError(
    SimulationDatabaseError
):
    """Concurrent modification detected."""


# ============================================================
# ENUMS
# ============================================================

class SimulationStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimulationEnvironmentType(str, Enum):
    SYNTHETIC_PHYSICS = "synthetic_physics"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    DRIVING = "driving"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    CUSTOM = "custom"


class SimulationTermination(str, Enum):
    GOAL_REACHED = "goal_reached"
    MAX_STEPS = "max_steps"
    COLLISION = "collision"
    SAFETY_STOP = "safety_stop"
    TERMINAL_STATE = "terminal_state"
    MANUAL_STOP = "manual_stop"


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
        raise SimulationValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise SimulationValidationError(
            f"Invalid {field}"
        )

    return value


def finite_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise SimulationValidationError(
            f"{field} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise SimulationValidationError(
            f"{field} must be finite"
        )

    return number


def validate_vector(
    values: Sequence[float],
    field: str,
    maximum_dimension: int = 4096,
) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise SimulationValidationError(
            f"{field} must be a list or tuple"
        )

    if not 1 <= len(values) <= maximum_dimension:
        raise SimulationValidationError(
            f"Invalid {field} dimension"
        )

    return [
        finite_number(value, field)
        for value in values
    ]


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return iso_datetime(value)

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.generic):
        return json_safe(value.item())

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
            raise SimulationValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise SimulationValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise SimulationValidationError(
            f"{field} must be a dictionary"
        )

    return result


# ============================================================
# ORM MODELS
# ============================================================

class SimulationORM(Base):
    __tablename__ = "world_simulations"

    __table_args__ = (
        Index(
            "ix_world_simulations_world_status",
            "world_id",
            "status",
        ),
    )

    simulation_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    environment_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    environment_type: Mapped[str] = mapped_column(
        String(64),
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

    initial_state_data: Mapped[
        list[float]
    ] = mapped_column(
        "initial_state",
        JSON,
        nullable=False,
    )

    current_state_data: Mapped[
        list[float]
    ] = mapped_column(
        "current_state",
        JSON,
        nullable=False,
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

    max_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    total_reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    collision_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    safety_event_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    termination_reason: Mapped[
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

    steps: Mapped[
        list["SimulationStepORM"]
    ] = relationship(
        back_populates="simulation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["SimulationEventORM"]
    ] = relationship(
        back_populates="simulation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SimulationStepORM(Base):
    __tablename__ = "world_simulation_steps"

    __table_args__ = (
        UniqueConstraint(
            "simulation_id",
            "step_index",
            name="uq_world_simulation_step",
        ),
    )

    step_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    simulation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "world_simulations.simulation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    state_data: Mapped[
        list[float]
    ] = mapped_column(
        "state",
        JSON,
        nullable=False,
    )

    action_data: Mapped[
        list[float]
    ] = mapped_column(
        "action",
        JSON,
        nullable=False,
    )

    next_state_data: Mapped[
        list[float]
    ] = mapped_column(
        "next_state",
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

    reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    collision: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    safety_event: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    terminal: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    simulation: Mapped[SimulationORM] = relationship(
        back_populates="steps",
    )


class SimulationEventORM(Base):
    __tablename__ = "world_simulation_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    simulation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "world_simulations.simulation_id",
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

    simulation: Mapped[SimulationORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class SimulationRecord:
    simulation_id: str
    world_id: str
    name: str
    environment_name: str
    environment_type: str
    status: str
    model_name: str | None
    model_version: str | None
    initial_state: list[float]
    current_state: list[float]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    max_steps: int
    step_count: int
    total_reward: float
    collision_count: int
    safety_event_count: int
    termination_reason: str | None
    error_message: str | None
    version: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class SimulationStepRecord:
    step_id: str
    simulation_id: str
    step_index: int
    state: list[float]
    action: list[float]
    next_state: list[float]
    predicted_next_state: list[float] | None
    reward: float
    collision: bool
    safety_event: bool
    terminal: bool
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class SimulationEventRecord:
    event_id: str
    simulation_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class SimulationStatistics:
    simulation_id: str
    total_steps: int
    total_reward: float
    average_reward: float
    collision_count: int
    safety_event_count: int
    prediction_mae: float | None
    prediction_rmse: float | None
    terminal: bool


# ============================================================
# SIMULATION REPOSITORY
# ============================================================

class SimulationRepository:
    """
    Stores simulator-generated trajectories.

    It does not calculate physics transitions.
    Pass the next state returned by an
    authorized simulation environment.
    """

    def __init__(
        self,
        *,
        max_steps_per_simulation: int = 10000,
        max_state_dimension: int = 4096,
        max_action_dimension: int = 1024,
    ) -> None:
        for name, value in {
            "max_steps_per_simulation":
                max_steps_per_simulation,
            "max_state_dimension":
                max_state_dimension,
            "max_action_dimension":
                max_action_dimension,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise SimulationValidationError(
                    f"{name} must be positive"
                )

        self.max_steps_per_simulation = (
            max_steps_per_simulation
        )
        self.max_state_dimension = (
            max_state_dimension
        )
        self.max_action_dimension = (
            max_action_dimension
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
        simulation_id: str,
    ) -> SimulationORM:
        row = session.get(
            SimulationORM,
            simulation_id,
        )

        if row is None:
            raise SimulationNotFoundError(
                "Simulation not found"
            )

        return row

    @staticmethod
    def _record(
        row: SimulationORM,
    ) -> SimulationRecord:
        return SimulationRecord(
            simulation_id=row.simulation_id,
            world_id=row.world_id,
            name=row.name,
            environment_name=row.environment_name,
            environment_type=row.environment_type,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            initial_state=copy.deepcopy(
                row.initial_state_data
            ),
            current_state=copy.deepcopy(
                row.current_state_data
            ),
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            max_steps=row.max_steps,
            step_count=row.step_count,
            total_reward=row.total_reward,
            collision_count=row.collision_count,
            safety_event_count=row.safety_event_count,
            termination_reason=row.termination_reason,
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
    def _step_record(
        row: SimulationStepORM,
    ) -> SimulationStepRecord:
        return SimulationStepRecord(
            step_id=row.step_id,
            simulation_id=row.simulation_id,
            step_index=row.step_index,
            state=copy.deepcopy(
                row.state_data
            ),
            action=copy.deepcopy(
                row.action_data
            ),
            next_state=copy.deepcopy(
                row.next_state_data
            ),
            predicted_next_state=copy.deepcopy(
                row.predicted_next_state_data
            ),
            reward=row.reward,
            collision=row.collision,
            safety_event=row.safety_event,
            terminal=row.terminal,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _event_record(
        row: SimulationEventORM,
    ) -> SimulationEventRecord:
        return SimulationEventRecord(
            event_id=row.event_id,
            simulation_id=row.simulation_id,
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
        simulation_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            SimulationEventORM(
                event_id=uuid.uuid4().hex,
                simulation_id=simulation_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE SIMULATION
    # --------------------------------------------------------

    def create_simulation(
        self,
        *,
        world_id: str,
        name: str,
        environment_name: str,
        environment_type: SimulationEnvironmentType,
        initial_state: Sequence[float],
        max_steps: int = 200,
        model_name: str | None = None,
        model_version: str | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> SimulationRecord:
        world_id = validate_string(
            world_id,
            "world_id",
            64,
        )

        name = validate_string(
            name,
            "name",
        )

        environment_name = validate_string(
            environment_name,
            "environment_name",
        )

        if not isinstance(
            environment_type,
            SimulationEnvironmentType,
        ):
            raise SimulationValidationError(
                "Invalid environment type"
            )

        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or not 1 <= max_steps
            <= self.max_steps_per_simulation
        ):
            raise SimulationValidationError(
                "Invalid max_steps"
            )

        initial_state = validate_vector(
            initial_state,
            "initial_state",
            self.max_state_dimension,
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

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        row = SimulationORM(
            simulation_id=uuid.uuid4().hex,
            world_id=world_id,
            name=name,
            environment_name=environment_name,
            environment_type=environment_type.value,
            status=SimulationStatus.CREATED.value,
            model_name=model_name,
            model_version=model_version,
            initial_state_data=initial_state,
            current_state_data=copy.deepcopy(
                initial_state
            ),
            configuration_data=configuration,
            metadata_data=metadata,
            max_steps=max_steps,
            step_count=0,
            total_reward=0.0,
            collision_count=0,
            safety_event_count=0,
            version=1,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)

            self._audit(
                session,
                row.simulation_id,
                "simulation_created",
                {
                    "world_id": world_id,
                    "environment": environment_name,
                },
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET SIMULATION
    # --------------------------------------------------------

    def get_simulation(
        self,
        simulation_id: str,
        *,
        db: Session | None = None,
    ) -> SimulationRecord:
        with self._session(db) as session:
            return self._record(
                self._get(
                    session,
                    simulation_id,
                )
            )

    # --------------------------------------------------------
    # STATUS TRANSITION
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        simulation_id: str,
        expected_version: int,
        allowed_from: set[SimulationStatus],
        target: SimulationStatus,
        values: dict[str, Any] | None = None,
    ) -> SimulationRecord:
        result = session.execute(
            update(SimulationORM)
            .where(
                SimulationORM.simulation_id
                == simulation_id,
                SimulationORM.version
                == expected_version,
                SimulationORM.status.in_(
                    [
                        item.value
                        for item in allowed_from
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
            raise SimulationConflictError(
                "Simulation status/version changed"
            )

        self._audit(
            session,
            simulation_id,
            f"simulation_{target.value}",
        )

        session.flush()
        session.expire_all()

        return self._record(
            self._get(
                session,
                simulation_id,
            )
        )

    def start_simulation(
        self,
        simulation_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> SimulationRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                simulation_id=simulation_id,
                expected_version=expected_version,
                allowed_from={SimulationStatus.CREATED},
                target=SimulationStatus.RUNNING,
                values={
                    "started_at": utc_now(),
                },
            )

    def pause_simulation(
        self,
        simulation_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> SimulationRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                simulation_id=simulation_id,
                expected_version=expected_version,
                allowed_from={SimulationStatus.RUNNING},
                target=SimulationStatus.PAUSED,
            )

    def resume_simulation(
        self,
        simulation_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> SimulationRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                simulation_id=simulation_id,
                expected_version=expected_version,
                allowed_from={SimulationStatus.PAUSED},
                target=SimulationStatus.RUNNING,
            )

    # --------------------------------------------------------
    # ADD SIMULATION STEP
    # --------------------------------------------------------

    def add_step(
        self,
        simulation_id: str,
        *,
        state: Sequence[float],
        action: Sequence[float],
        next_state: Sequence[float],
        reward: float,
        predicted_next_state: Sequence[float] | None = None,
        collision: bool = False,
        safety_event: bool = False,
        terminal: bool = False,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> SimulationStepRecord:
        state = validate_vector(
            state,
            "state",
            self.max_state_dimension,
        )

        action = validate_vector(
            action,
            "action",
            self.max_action_dimension,
        )

        next_state = validate_vector(
            next_state,
            "next_state",
            self.max_state_dimension,
        )

        if len(state) != len(next_state):
            raise SimulationValidationError(
                "State dimensions must match"
            )

        if predicted_next_state is not None:
            predicted_next_state = validate_vector(
                predicted_next_state,
                "predicted_next_state",
                self.max_state_dimension,
            )

            if len(predicted_next_state) != len(state):
                raise SimulationValidationError(
                    "Predicted state dimension mismatch"
                )

        reward = finite_number(
            reward,
            "reward",
        )

        for name, value in {
            "collision": collision,
            "safety_event": safety_event,
            "terminal": terminal,
        }.items():
            if not isinstance(value, bool):
                raise SimulationValidationError(
                    f"{name} must be boolean"
                )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            simulation = self._get(
                session,
                simulation_id,
            )

            if simulation.status != (
                SimulationStatus.RUNNING.value
            ):
                raise SimulationStateError(
                    "Simulation must be running"
                )

            if simulation.step_count >= simulation.max_steps:
                raise SimulationStateError(
                    "Maximum simulation steps reached"
                )

            if not np.allclose(
                state,
                simulation.current_state_data,
                atol=1e-6,
                rtol=1e-6,
            ):
                raise SimulationValidationError(
                    "State does not match current state"
                )

            if len(state) != len(
                simulation.initial_state_data
            ):
                raise SimulationValidationError(
                    "State dimension mismatch"
                )

            step_index = simulation.step_count

            claim = session.execute(
                update(SimulationORM)
                .where(
                    SimulationORM.simulation_id
                    == simulation_id,
                    SimulationORM.status
                    == SimulationStatus.RUNNING.value,
                    SimulationORM.step_count
                    == step_index,
                )
                .values(
                    current_state_data=next_state,
                    step_count=step_index + 1,
                    total_reward=(
                        SimulationORM.total_reward
                        + reward
                    ),
                    collision_count=(
                        SimulationORM.collision_count
                        + int(collision)
                    ),
                    safety_event_count=(
                        SimulationORM.safety_event_count
                        + int(safety_event)
                    ),
                    version=SimulationORM.version + 1,
                )
            )

            if claim.rowcount != 1:
                raise SimulationConflictError(
                    "Simulation changed concurrently"
                )

            row = SimulationStepORM(
                step_id=uuid.uuid4().hex,
                simulation_id=simulation_id,
                step_index=step_index,
                state_data=state,
                action_data=action,
                next_state_data=next_state,
                predicted_next_state_data=(
                    predicted_next_state
                ),
                reward=reward,
                collision=collision,
                safety_event=safety_event,
                terminal=terminal,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)

            if collision:
                self._audit(
                    session,
                    simulation_id,
                    "collision_detected",
                    {"step_index": step_index},
                )

            if safety_event:
                self._audit(
                    session,
                    simulation_id,
                    "safety_event",
                    {"step_index": step_index},
                )

            session.flush()

            return self._step_record(row)

    # --------------------------------------------------------
    # COMPLETE SIMULATION
    # --------------------------------------------------------

    def complete_simulation(
        self,
        simulation_id: str,
        *,
        expected_version: int,
        termination_reason: SimulationTermination,
        db: Session | None = None,
    ) -> SimulationRecord:
        if not isinstance(
            termination_reason,
            SimulationTermination,
        ):
            raise SimulationValidationError(
                "Invalid termination reason"
            )

        with self._session(db) as session:
            simulation = self._get(
                session,
                simulation_id,
            )

            if simulation.step_count == 0:
                raise SimulationStateError(
                    "Cannot complete empty simulation"
                )

            return self._transition(
                session,
                simulation_id=simulation_id,
                expected_version=expected_version,
                allowed_from={
                    SimulationStatus.RUNNING,
                    SimulationStatus.PAUSED,
                },
                target=SimulationStatus.COMPLETED,
                values={
                    "termination_reason":
                        termination_reason.value,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # FAIL SIMULATION
    # --------------------------------------------------------

    def fail_simulation(
        self,
        simulation_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> SimulationRecord:
        reason = validate_string(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                simulation_id=simulation_id,
                expected_version=expected_version,
                allowed_from={
                    SimulationStatus.RUNNING,
                    SimulationStatus.PAUSED,
                },
                target=SimulationStatus.FAILED,
                values={
                    "error_message": reason,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # GET SIMULATION STEPS
    # --------------------------------------------------------

    def get_steps(
        self,
        simulation_id: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[SimulationStepRecord]:
        if not 1 <= limit <= 10000 or offset < 0:
            raise SimulationValidationError(
                "Invalid pagination"
            )

        statement = (
            select(SimulationStepORM)
            .where(
                SimulationStepORM.simulation_id
                == simulation_id
            )
            .order_by(
                SimulationStepORM.step_index.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._step_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # SIMULATION STATISTICS
    # --------------------------------------------------------

    def calculate_statistics(
        self,
        simulation_id: str,
        *,
        db: Session | None = None,
    ) -> SimulationStatistics:
        with self._session(db) as session:
            simulation = self._get(
                session,
                simulation_id,
            )

            rows = session.scalars(
                select(SimulationStepORM)
                .where(
                    SimulationStepORM.simulation_id
                    == simulation_id
                )
                .order_by(
                    SimulationStepORM.step_index.asc()
                )
            ).all()

            prediction_errors = []

            for row in rows:
                if row.predicted_next_state_data is None:
                    continue

                predicted = np.asarray(
                    row.predicted_next_state_data,
                    dtype=np.float64,
                )

                actual = np.asarray(
                    row.next_state_data,
                    dtype=np.float64,
                )

                prediction_errors.extend(
                    (predicted - actual).tolist()
                )

            if prediction_errors:
                errors = np.asarray(
                    prediction_errors,
                    dtype=np.float64,
                )

                mae = float(
                    np.mean(
                        np.abs(errors)
                    )
                )

                rmse = float(
                    np.sqrt(
                        np.mean(errors ** 2)
                    )
                )
            else:
                mae = None
                rmse = None

            return SimulationStatistics(
                simulation_id=simulation_id,
                total_steps=simulation.step_count,
                total_reward=simulation.total_reward,
                average_reward=(
                    simulation.total_reward
                    / simulation.step_count
                    if simulation.step_count
                    else 0.0
                ),
                collision_count=(
                    simulation.collision_count
                ),
                safety_event_count=(
                    simulation.safety_event_count
                ),
                prediction_mae=mae,
                prediction_rmse=rmse,
                terminal=(
                    simulation.status
                    in {
                        SimulationStatus.COMPLETED.value,
                        SimulationStatus.FAILED.value,
                        SimulationStatus.CANCELLED.value,
                    }
                ),
            )

    # --------------------------------------------------------
    # LIST SIMULATIONS
    # --------------------------------------------------------

    def list_simulations(
        self,
        *,
        world_id: str | None = None,
        status: SimulationStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[SimulationRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise SimulationValidationError(
                "Invalid pagination"
            )

        statement = select(
            SimulationORM
        )

        if world_id is not None:
            statement = statement.where(
                SimulationORM.world_id
                == world_id
            )

        if status is not None:
            statement = statement.where(
                SimulationORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                SimulationORM.created_at.desc()
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
    # AUDIT EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        simulation_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[SimulationEventRecord]:
        if not 1 <= limit <= 1000:
            raise SimulationValidationError(
                "Invalid limit"
            )

        statement = (
            select(SimulationEventORM)
            .where(
                SimulationEventORM.simulation_id
                == simulation_id
            )
            .order_by(
                SimulationEventORM.created_at.asc()
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
        simulation_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            simulation = self.get_simulation(
                simulation_id,
                db=session,
            )

            steps = []

            for offset in range(
                0,
                simulation.step_count,
                1000,
            ):
                steps.extend(
                    self.get_steps(
                        simulation_id,
                        limit=1000,
                        offset=offset,
                        db=session,
                    )
                )

            payload = {
                "simulation": asdict(simulation),
                "statistics": asdict(
                    self.calculate_statistics(
                        simulation_id,
                        db=session,
                    )
                ),
                "steps": [
                    asdict(step)
                    for step in steps
                ],
                "events": [
                    asdict(event)
                    for event in self.list_events(
                        simulation_id,
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

    def export_csv(
        self,
        simulation_id: str,
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
            "step_id",
            "simulation_id",
            "step_index",
            "state",
            "action",
            "next_state",
            "predicted_next_state",
            "reward",
            "collision",
            "safety_event",
            "terminal",
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
                steps = self.get_steps(
                    simulation_id,
                    limit=1000,
                    offset=offset,
                    db=db,
                )

                if not steps:
                    break

                for step in steps:
                    data = asdict(step)

                    for key in (
                        "state",
                        "action",
                        "next_state",
                        "predicted_next_state",
                    ):
                        data[key] = json.dumps(
                            data[key]
                        )

                    writer.writerow(
                        {
                            key: data.get(key)
                            for key in fields
                        }
                    )

                offset += len(steps)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_simulation_repository(
    **kwargs: Any,
) -> SimulationRepository:
    return SimulationRepository(
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

    repository = create_simulation_repository()

    simulation = repository.create_simulation(
        world_id="demo_navigation_world",
        name="Synthetic Navigation Simulation",
        environment_name="synthetic_physics",
        environment_type=(
            SimulationEnvironmentType.SYNTHETIC_PHYSICS
        ),
        initial_state=[
            0.0, 0.0, 0.0, 0.0
        ],
        max_steps=3,
        model_name="demo_world_model",
        model_version="v1",
        configuration={
            "dt": 0.1,
            "simulation_only": True,
        },
    )

    simulation = repository.start_simulation(
        simulation.simulation_id,
        expected_version=simulation.version,
    )

    states = [
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1],
        [0.3, 0.3, 0.2, 0.2],
    ]

    next_states = [
        [0.1, 0.1, 0.1, 0.1],
        [0.3, 0.3, 0.2, 0.2],
        [0.6, 0.6, 0.3, 0.3],
    ]

    for index in range(3):
        repository.add_step(
            simulation.simulation_id,
            state=states[index],
            action=[0.2, 0.2],
            next_state=next_states[index],
            predicted_next_state=[
                value + 0.01
                for value in next_states[index]
            ],
            reward=-0.5,
            collision=False,
            safety_event=False,
            terminal=(index == 2),
        )

    simulation = repository.get_simulation(
        simulation.simulation_id
    )

    simulation = repository.complete_simulation(
        simulation.simulation_id,
        expected_version=simulation.version,
        termination_reason=(
            SimulationTermination.TERMINAL_STATE
        ),
    )

    statistics = repository.calculate_statistics(
        simulation.simulation_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - SIMULATIONS")
    print("=" * 60)

    print(
        "Simulation ID:",
        simulation.simulation_id,
    )

    print(
        "Status:",
        simulation.status,
    )

    print(
        "Steps:",
        statistics.total_steps,
    )

    print(
        "Total reward:",
        statistics.total_reward,
    )

    print(
        "Prediction RMSE:",
        statistics.prediction_rmse,
    )

    json_path = repository.export_json(
        simulation.simulation_id,
        "experiments/reports/simulation_result.json",
    )

    csv_path = repository.export_csv(
        simulation.simulation_id,
        "experiments/reports/simulation_steps.csv",
    )

    print("JSON:", json_path)
    print("CSV:", csv_path)

    close_database()


if __name__ == "__main__":
    main()
