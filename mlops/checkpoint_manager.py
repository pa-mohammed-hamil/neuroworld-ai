
"""
NeuroWorld AI
backend/app/database/worlds.py

Persistent simulated world management.

Features:
- Create and manage virtual worlds
- Store world configurations and metadata
- Track current state and state dimensions
- Apply simulated state transitions
- Pause, resume and archive worlds
- Save and restore world snapshots
- Maintain audit events
- Optimistic concurrency control
- List and filter worlds
- Export world histories as JSON

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Important:
    This repository stores simulated world states.
    It does not control physical robots or vehicles.
"""

from __future__ import annotations

import copy
import json
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
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

class WorldDatabaseError(Exception):
    """Base database error."""


class WorldValidationError(WorldDatabaseError):
    """Invalid world input."""


class WorldNotFoundError(WorldDatabaseError):
    """World or snapshot not found."""


class WorldStateError(WorldDatabaseError):
    """Invalid world lifecycle operation."""


class WorldConflictError(WorldDatabaseError):
    """Concurrent world modification."""


# ============================================================
# ENUMS
# ============================================================

class WorldStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class WorldType(str, Enum):
    SYNTHETIC = "synthetic"
    NAVIGATION = "navigation"
    ROBOTICS = "robotics"
    DRIVING = "driving"
    PHYSICS = "physics"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    CUSTOM = "custom"


# ============================================================
# VALIDATION HELPERS
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
        raise WorldValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise WorldValidationError(
            f"Invalid {field}"
        )

    return value


def validate_integer(
    value: int,
    field: str,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise WorldValidationError(
            f"{field} must be an integer >= {minimum}"
        )

    return value


def validate_vector(
    value: Sequence[float],
    field: str,
    maximum_dimension: int = 4096,
) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise WorldValidationError(
            f"{field} must be a list or tuple"
        )

    if not 1 <= len(value) <= maximum_dimension:
        raise WorldValidationError(
            f"Invalid {field} dimension"
        )

    output = []

    for item in value:
        if isinstance(item, bool):
            raise WorldValidationError(
                f"{field} must contain numbers"
            )

        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise WorldValidationError(
                f"{field} contains invalid numbers"
            ) from exc

        if not math.isfinite(number):
            raise WorldValidationError(
                f"{field} contains non-finite numbers"
            )

        output.append(number)

    return output


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
            raise WorldValidationError(
                "Non-finite JSON value"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise WorldValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise WorldValidationError(
            f"{field} must be a dictionary"
        )

    return result


# ============================================================
# ORM MODELS
# ============================================================

class WorldORM(Base):
    __tablename__ = "simulation_worlds"

    __table_args__ = (
        Index(
            "ix_simulation_worlds_type_status",
            "world_type",
            "status",
        ),
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    world_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    environment_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    state_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
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

    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
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

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    archived_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    snapshots: Mapped[
        list["WorldSnapshotORM"]
    ] = relationship(
        back_populates="world",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["WorldEventORM"]
    ] = relationship(
        back_populates="world",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WorldSnapshotORM(Base):
    __tablename__ = "simulation_world_snapshots"

    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "simulation_worlds.world_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    label: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    state_data: Mapped[
        list[float]
    ] = mapped_column(
        "state",
        JSON,
        nullable=False,
    )

    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    source_version: Mapped[int] = mapped_column(
        Integer,
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

    world: Mapped[WorldORM] = relationship(
        back_populates="snapshots",
    )


class WorldEventORM(Base):
    __tablename__ = "simulation_world_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "simulation_worlds.world_id",
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

    world: Mapped[WorldORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class WorldRecord:
    world_id: str
    name: str
    description: str | None
    world_type: str
    status: str
    environment_name: str
    model_name: str | None
    state_dimension: int
    initial_state: list[float]
    current_state: list[float]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    step_count: int
    version: int
    created_at: str
    updated_at: str
    archived_at: str | None


@dataclass(frozen=True)
class WorldSnapshotRecord:
    snapshot_id: str
    world_id: str
    label: str
    state: list[float]
    step_count: int
    source_version: int
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class WorldEventRecord:
    event_id: str
    world_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


# ============================================================
# WORLD REPOSITORY
# ============================================================

class WorldRepository:
    """
    Persistent world state and snapshot repository.

    Use expected_version for updates to prevent
    concurrent agents overwriting world state.
    """

    def __init__(
        self,
        *,
        max_state_dimension: int = 4096,
    ) -> None:
        self.max_state_dimension = validate_integer(
            max_state_dimension,
            "max_state_dimension",
            1,
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
        world_id: str,
    ) -> WorldORM:
        row = session.get(
            WorldORM,
            world_id,
        )

        if row is None:
            raise WorldNotFoundError(
                "World not found"
            )

        return row

    @staticmethod
    def _record(
        row: WorldORM,
    ) -> WorldRecord:
        return WorldRecord(
            world_id=row.world_id,
            name=row.name,
            description=row.description,
            world_type=row.world_type,
            status=row.status,
            environment_name=row.environment_name,
            model_name=row.model_name,
            state_dimension=row.state_dimension,
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
            step_count=row.step_count,
            version=row.version,
            created_at=iso_datetime(
                row.created_at
            ),
            updated_at=iso_datetime(
                row.updated_at
            ),
            archived_at=iso_datetime(
                row.archived_at
            ),
        )

    @staticmethod
    def _snapshot_record(
        row: WorldSnapshotORM,
    ) -> WorldSnapshotRecord:
        return WorldSnapshotRecord(
            snapshot_id=row.snapshot_id,
            world_id=row.world_id,
            label=row.label,
            state=copy.deepcopy(
                row.state_data
            ),
            step_count=row.step_count,
            source_version=row.source_version,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _event_record(
        row: WorldEventORM,
    ) -> WorldEventRecord:
        return WorldEventRecord(
            event_id=row.event_id,
            world_id=row.world_id,
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
        world_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            WorldEventORM(
                event_id=uuid.uuid4().hex,
                world_id=world_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE WORLD
    # --------------------------------------------------------

    def create_world(
        self,
        *,
        name: str,
        world_type: WorldType,
        environment_name: str,
        initial_state: Sequence[float],
        description: str | None = None,
        model_name: str | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> WorldRecord:
        name = validate_string(
            name,
            "name",
        )

        environment_name = validate_string(
            environment_name,
            "environment_name",
        )

        if not isinstance(world_type, WorldType):
            raise WorldValidationError(
                "Invalid world type"
            )

        initial_state = validate_vector(
            initial_state,
            "initial_state",
            self.max_state_dimension,
        )

        if description is not None:
            description = validate_string(
                description,
                "description",
                4000,
            )

        if model_name is not None:
            model_name = validate_string(
                model_name,
                "model_name",
            )

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        now = utc_now()

        row = WorldORM(
            world_id=uuid.uuid4().hex,
            name=name,
            description=description,
            world_type=world_type.value,
            status=WorldStatus.CREATED.value,
            environment_name=environment_name,
            model_name=model_name,
            state_dimension=len(initial_state),
            initial_state_data=initial_state,
            current_state_data=copy.deepcopy(
                initial_state
            ),
            configuration_data=configuration,
            metadata_data=metadata,
            step_count=0,
            version=1,
            created_at=now,
            updated_at=now,
        )

        with self._session(db) as session:
            session.add(row)

            self._audit(
                session,
                row.world_id,
                "world_created",
                {
                    "world_type": world_type.value,
                    "environment": environment_name,
                },
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET WORLD
    # --------------------------------------------------------

    def get_world(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._record(
                self._get(
                    session,
                    world_id,
                )
            )

    # --------------------------------------------------------
    # STATUS TRANSITIONS
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        world_id: str,
        expected_version: int,
        allowed_from: set[WorldStatus],
        target: WorldStatus,
        values: dict[str, Any] | None = None,
    ) -> WorldRecord:
        result = session.execute(
            update(WorldORM)
            .where(
                WorldORM.world_id == world_id,
                WorldORM.version
                == expected_version,
                WorldORM.status.in_(
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
                updated_at=utc_now(),
            )
        )

        if result.rowcount != 1:
            raise WorldConflictError(
                "World status/version changed"
            )

        self._audit(
            session,
            world_id,
            f"world_{target.value}",
        )

        session.flush()
        session.expire_all()

        return self._record(
            self._get(
                session,
                world_id,
            )
        )

    def activate_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                world_id=world_id,
                expected_version=expected_version,
                allowed_from={WorldStatus.CREATED},
                target=WorldStatus.ACTIVE,
            )

    def pause_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                world_id=world_id,
                expected_version=expected_version,
                allowed_from={WorldStatus.ACTIVE},
                target=WorldStatus.PAUSED,
            )

    def resume_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                world_id=world_id,
                expected_version=expected_version,
                allowed_from={WorldStatus.PAUSED},
                target=WorldStatus.ACTIVE,
            )

    def archive_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                world_id=world_id,
                expected_version=expected_version,
                allowed_from={
                    WorldStatus.CREATED,
                    WorldStatus.ACTIVE,
                    WorldStatus.PAUSED,
                },
                target=WorldStatus.ARCHIVED,
                values={
                    "archived_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # APPLY SIMULATED STATE TRANSITION
    # --------------------------------------------------------

    def apply_transition(
        self,
        world_id: str,
        *,
        expected_version: int,
        next_state: Sequence[float],
        action: Sequence[float] | None = None,
        reward: float | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> WorldRecord:
        next_state = validate_vector(
            next_state,
            "next_state",
            self.max_state_dimension,
        )

        if action is not None:
            action = validate_vector(
                action,
                "action",
            )

        if reward is not None:
            if isinstance(reward, bool):
                raise WorldValidationError(
                    "reward must be numeric"
                )

            reward = float(reward)

            if not math.isfinite(reward):
                raise WorldValidationError(
                    "reward must be finite"
                )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            world = self._get(
                session,
                world_id,
            )

            if len(next_state) != world.state_dimension:
                raise WorldValidationError(
                    "Next-state dimension mismatch"
                )

            result = session.execute(
                update(WorldORM)
                .where(
                    WorldORM.world_id == world_id,
                    WorldORM.status
                    == WorldStatus.ACTIVE.value,
                    WorldORM.version
                    == expected_version,
                )
                .values(
                    current_state_data=next_state,
                    step_count=WorldORM.step_count + 1,
                    version=expected_version + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise WorldConflictError(
                    "World is not active or "
                    "version changed"
                )

            self._audit(
                session,
                world_id,
                "world_transition",
                {
                    "action": action,
                    "reward": reward,
                    "metadata": metadata,
                },
            )

            session.flush()
            session.expire_all()

            return self._record(
                self._get(
                    session,
                    world_id,
                )
            )

    # --------------------------------------------------------
    # CREATE SNAPSHOT
    # --------------------------------------------------------

    def create_snapshot(
        self,
        world_id: str,
        *,
        label: str,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> WorldSnapshotRecord:
        label = validate_string(
            label,
            "label",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            world = self._get(
                session,
                world_id,
            )

            if world.status == WorldStatus.ARCHIVED.value:
                raise WorldStateError(
                    "Cannot snapshot archived world"
                )

            row = WorldSnapshotORM(
                snapshot_id=uuid.uuid4().hex,
                world_id=world_id,
                label=label,
                state_data=copy.deepcopy(
                    world.current_state_data
                ),
                step_count=world.step_count,
                source_version=world.version,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                world_id,
                "snapshot_created",
                {
                    "snapshot_id": row.snapshot_id,
                    "label": label,
                },
            )

            session.flush()

            return self._snapshot_record(row)

    # --------------------------------------------------------
    # RESTORE SNAPSHOT
    # --------------------------------------------------------

    def restore_snapshot(
        self,
        world_id: str,
        snapshot_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            snapshot = session.get(
                WorldSnapshotORM,
                snapshot_id,
            )

            if (
                snapshot is None
                or snapshot.world_id != world_id
            ):
                raise WorldNotFoundError(
                    "Snapshot not found"
                )

            world = self._get(
                session,
                world_id,
            )

            if world.status not in {
                WorldStatus.PAUSED.value,
                WorldStatus.CREATED.value,
            }:
                raise WorldStateError(
                    "Pause the world before restoration"
                )

            result = session.execute(
                update(WorldORM)
                .where(
                    WorldORM.world_id == world_id,
                    WorldORM.version
                    == expected_version,
                    WorldORM.status.in_(
                        [
                            WorldStatus.PAUSED.value,
                            WorldStatus.CREATED.value,
                        ]
                    ),
                )
                .values(
                    current_state_data=copy.deepcopy(
                        snapshot.state_data
                    ),
                    step_count=snapshot.step_count,
                    version=expected_version + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise WorldConflictError(
                    "World changed during restoration"
                )

            self._audit(
                session,
                world_id,
                "snapshot_restored",
                {
                    "snapshot_id": snapshot_id,
                    "source_version":
                        snapshot.source_version,
                },
            )

            session.flush()
            session.expire_all()

            return self._record(
                self._get(
                    session,
                    world_id,
                )
            )

    # --------------------------------------------------------
    # LIST SNAPSHOTS
    # --------------------------------------------------------

    def list_snapshots(
        self,
        world_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[WorldSnapshotRecord]:
        validate_integer(
            limit,
            "limit",
            1,
        )

        if limit > 1000:
            raise WorldValidationError(
                "Limit exceeds 1000"
            )

        statement = (
            select(WorldSnapshotORM)
            .where(
                WorldSnapshotORM.world_id
                == world_id
            )
            .order_by(
                WorldSnapshotORM.created_at.desc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._snapshot_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST WORLDS
    # --------------------------------------------------------

    def list_worlds(
        self,
        *,
        world_type: WorldType | None = None,
        status: WorldStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[WorldRecord]:
        validate_integer(
            limit,
            "limit",
            1,
        )

        validate_integer(
            offset,
            "offset",
        )

        if limit > 1000:
            raise WorldValidationError(
                "Limit exceeds 1000"
            )

        statement = select(WorldORM)

        if world_type is not None:
            statement = statement.where(
                WorldORM.world_type
                == world_type.value
            )

        if status is not None:
            statement = statement.where(
                WorldORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                WorldORM.created_at.desc()
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
    # LIST EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        world_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[WorldEventRecord]:
        validate_integer(
            limit,
            "limit",
            1,
        )

        if limit > 1000:
            raise WorldValidationError(
                "Limit exceeds 1000"
            )

        statement = (
            select(WorldEventORM)
            .where(
                WorldEventORM.world_id
                == world_id
            )
            .order_by(
                WorldEventORM.created_at.asc()
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
    # DELETE ARCHIVED WORLD
    # --------------------------------------------------------

    def delete_archived_world(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> None:
        with self._session(db) as session:
            world = self._get(
                session,
                world_id,
            )

            if world.status != WorldStatus.ARCHIVED.value:
                raise WorldStateError(
                    "Only archived worlds can be deleted"
                )

            session.delete(world)
            session.flush()

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        world_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            payload = {
                "world": asdict(
                    self.get_world(
                        world_id,
                        db=session,
                    )
                ),
                "snapshots": [
                    asdict(snapshot)
                    for snapshot in self.list_snapshots(
                        world_id,
                        limit=1000,
                        db=session,
                    )
                ],
                "events": [
                    asdict(event)
                    for event in self.list_events(
                        world_id,
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


# ============================================================
# FACTORY
# ============================================================

def create_world_repository(
    **kwargs: Any,
) -> WorldRepository:
    return WorldRepository(
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

    repository = create_world_repository()

    world = repository.create_world(
        name="Demo Navigation World",
        world_type=WorldType.NAVIGATION,
        environment_name="synthetic_physics",
        initial_state=[
            0.0, 0.0, 0.0, 0.0
        ],
        model_name="neuroworld_transformer",
        configuration={
            "simulation_only": True,
            "dt": 0.1,
        },
    )

    world = repository.activate_world(
        world.world_id,
        expected_version=world.version,
    )

    world = repository.apply_transition(
        world.world_id,
        expected_version=world.version,
        next_state=[
            0.1, 0.1, 0.1, 0.1
        ],
        action=[0.2, 0.2],
        reward=-0.5,
    )

    snapshot = repository.create_snapshot(
        world.world_id,
        label="checkpoint_step_1",
    )

    world = repository.apply_transition(
        world.world_id,
        expected_version=world.version,
        next_state=[
            0.3, 0.3, 0.2, 0.2
        ],
        action=[0.2, 0.2],
        reward=-0.3,
    )

    world = repository.pause_world(
        world.world_id,
        expected_version=world.version,
    )

    world = repository.restore_snapshot(
        world.world_id,
        snapshot.snapshot_id,
        expected_version=world.version,
    )

    print("=" * 60)
    print("NEUROWORLD AI - WORLDS")
    print("=" * 60)

    print("World ID:", world.world_id)
    print("Status:", world.status)
    print("Version:", world.version)
    print("Steps:", world.step_count)
    print("Current state:", world.current_state)

    output_path = repository.export_json(
        world.world_id,
        "experiments/reports/world_history.json",
    )

    print("Export:", output_path)

    close_database()


if __name__ == "__main__":
    main()
