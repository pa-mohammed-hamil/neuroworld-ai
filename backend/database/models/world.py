
"""
NeuroWorld AI
backend/app/database/world.py

Persistent World State and History Management

Features:
- Create and manage virtual worlds
- Store world configuration and metadata
- Persist current world state
- Optimistic concurrency control
- Pause, resume and archive worlds
- Store world event history
- Create persistent world snapshots
- Restore previous world snapshots
- Record action and observation references
- Paginated world and event listings
- JSON-safe NumPy serialization
- PostgreSQL and SQLite support

Dependencies:
    pip install sqlalchemy numpy

Run:
    python -m backend.app.database.world

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Important:
    This module persists virtual world state.
    It does not execute physical robot actions.

    World access authorization and tenant
    isolation belong in the API/service layer.

    Production databases should use Alembic
    migrations rather than create_all().
"""

from __future__ import annotations

import copy
import json
import logging
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    delete,
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
    "neuroworld.database.world"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class WorldDatabaseError(Exception):
    """Base world database exception."""


class WorldValidationError(WorldDatabaseError):
    """Invalid world data."""


class WorldNotFoundError(WorldDatabaseError):
    """World or snapshot not found."""


class WorldStateError(WorldDatabaseError):
    """Invalid world lifecycle operation."""


class WorldVersionConflict(WorldDatabaseError):
    """World state was modified concurrently."""


# ============================================================
# ENUMS
# ============================================================

class WorldStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class WorldEventType(str, Enum):
    CREATED = "created"
    STATE_UPDATED = "state_updated"
    ACTION_APPLIED = "action_applied"
    OBSERVATION_ATTACHED = "observation_attached"
    PAUSED = "paused"
    RESUMED = "resumed"
    ARCHIVED = "archived"
    RESTORED = "restored"
    ROLLED_BACK = "rolled_back"


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
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(
        timezone.utc
    ).isoformat()


def validate_identifier(
    value: str,
    name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise WorldValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise WorldValidationError(
            f"Invalid {name}"
        )

    return value


def json_safe(value: Any) -> Any:
    """
    Convert NumPy values and nested structures
    into JSON-compatible Python values.
    """

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.generic):
        return json_safe(value.item())

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
            raise WorldValidationError(
                "Non-finite numbers are not allowed"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise WorldValidationError(
        f"Unsupported value: {type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    normalized = json_safe(value)

    if not isinstance(normalized, dict):
        raise WorldValidationError(
            f"{name} must be a dictionary"
        )

    return normalized


def validate_state(
    value: Any,
) -> dict[str, Any]:
    """
    World state is represented as a JSON object.

    Example:
        {
            "position": [1.0, 2.0],
            "velocity": [0.1, 0.0],
            "objects": []
        }
    """

    return validate_json_dict(
        value,
        "state",
    )


def deep_copy_json(
    value: Any,
) -> Any:
    return copy.deepcopy(value)


# ============================================================
# ORM MODELS
# ============================================================

class WorldORM(Base):
    __tablename__ = "worlds"

    __table_args__ = (
        Index(
            "ix_worlds_status_created",
            "status",
            "created_at",
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

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    environment_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="synthetic",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=WorldStatus.ACTIVE.value,
        index=True,
    )

    state_data: Mapped[dict[str, Any]] = mapped_column(
        "state",
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

    events: Mapped[list["WorldEventORM"]] = relationship(
        back_populates="world",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    snapshots: Mapped[list["WorldSnapshotORM"]] = relationship(
        back_populates="world",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WorldEventORM(Base):
    __tablename__ = "world_events"

    __table_args__ = (
        Index(
            "ix_world_events_world_created",
            "world_id",
            "created_at",
        ),
        UniqueConstraint(
            "world_id",
            "world_version",
            name="uq_world_event_version",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "worlds.world_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    world_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    payload_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "payload",
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


class WorldSnapshotORM(Base):
    __tablename__ = "world_snapshots"

    __table_args__ = (
        Index(
            "ix_world_snapshots_world_created",
            "world_id",
            "created_at",
        ),
    )

    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "worlds.world_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    snapshot_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    world_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    state_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "state",
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

    world: Mapped[WorldORM] = relationship(
        back_populates="snapshots",
    )


# ============================================================
# PUBLIC DATA MODELS
# ============================================================

@dataclass(frozen=True)
class WorldRecord:
    world_id: str
    name: str
    description: str
    environment_type: str
    status: str
    state: dict[str, Any]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    version: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class WorldEventRecord:
    event_id: str
    world_id: str
    event_type: str
    world_version: int
    payload: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class WorldSnapshotRecord:
    snapshot_id: str
    world_id: str
    snapshot_name: str
    world_version: int
    state: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str


# ============================================================
# REPOSITORY
# ============================================================

class WorldRepository:
    """
    Database-backed virtual world repository.

    State mutations use optimistic concurrency:
    UPDATE ... WHERE version = expected_version

    A failed version check raises
    WorldVersionConflict.

    A supplied SQLAlchemy Session remains
    owned by the caller.
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
    def _world_record(
        row: WorldORM,
    ) -> WorldRecord:
        return WorldRecord(
            world_id=row.world_id,
            name=row.name,
            description=row.description,
            environment_type=row.environment_type,
            status=row.status,
            state=deep_copy_json(
                row.state_data
            ),
            configuration=deep_copy_json(
                row.configuration_data
            ),
            metadata=deep_copy_json(
                row.metadata_data
            ),
            version=row.version,
            created_at=datetime_string(
                row.created_at
            ),
            updated_at=datetime_string(
                row.updated_at
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
            world_version=row.world_version,
            payload=deep_copy_json(
                row.payload_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    @staticmethod
    def _snapshot_record(
        row: WorldSnapshotORM,
    ) -> WorldSnapshotRecord:
        return WorldSnapshotRecord(
            snapshot_id=row.snapshot_id,
            world_id=row.world_id,
            snapshot_name=row.snapshot_name,
            world_version=row.world_version,
            state=deep_copy_json(
                row.state_data
            ),
            metadata=deep_copy_json(
                row.metadata_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    @staticmethod
    def _get_world_row(
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
    def _record_event(
        session: Session,
        *,
        world_id: str,
        event_type: WorldEventType,
        version: int,
        payload: dict[str, Any],
    ) -> None:
        session.add(
            WorldEventORM(
                event_id=uuid.uuid4().hex,
                world_id=world_id,
                event_type=event_type.value,
                world_version=version,
                payload_data=json_safe(payload),
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
        initial_state: dict[str, Any],
        description: str = "",
        environment_type: str = "synthetic",
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> WorldRecord:
        name = validate_identifier(
            name,
            "name",
            255,
        )

        environment_type = validate_identifier(
            environment_type,
            "environment_type",
        )

        state = validate_state(
            initial_state
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

        world_id = uuid.uuid4().hex

        row = WorldORM(
            world_id=world_id,
            name=name,
            description=description,
            environment_type=environment_type,
            status=WorldStatus.ACTIVE.value,
            state_data=state,
            configuration_data=configuration,
            metadata_data=metadata,
            version=1,
            created_at=now,
            updated_at=now,
        )

        with self._session(db) as session:
            session.add(row)

            self._record_event(
                session,
                world_id=world_id,
                event_type=WorldEventType.CREATED,
                version=1,
                payload={
                    "name": name,
                    "environment_type": environment_type,
                },
            )

            session.flush()

            return self._world_record(row)

    # --------------------------------------------------------
    # GET WORLD
    # --------------------------------------------------------

    def get_world(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> WorldRecord:
        world_id = validate_identifier(
            world_id,
            "world_id",
        )

        with self._session(db) as session:
            row = self._get_world_row(
                session,
                world_id,
            )

            return self._world_record(row)

    # --------------------------------------------------------
    # LIST WORLDS
    # --------------------------------------------------------

    def list_worlds(
        self,
        *,
        status: WorldStatus | None = None,
        environment_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[WorldRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise WorldValidationError(
                "Invalid pagination"
            )

        statement = select(
            WorldORM
        )

        if status is not None:
            statement = statement.where(
                WorldORM.status == status.value
            )

        if environment_type is not None:
            statement = statement.where(
                WorldORM.environment_type
                == environment_type
            )

        statement = (
            statement
            .order_by(
                WorldORM.created_at.desc(),
                WorldORM.world_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._world_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # VERSIONED UPDATE
    # --------------------------------------------------------

    def _versioned_update(
        self,
        session: Session,
        *,
        world_id: str,
        expected_version: int,
        values: dict[str, Any],
        event_type: WorldEventType,
        event_payload: dict[str, Any],
        required_status: str | None = None,
    ) -> WorldRecord:
        """
        Atomically update the world if its
        version matches expected_version.
        """

        new_version = expected_version + 1

        statement = update(
            WorldORM
        ).where(
            WorldORM.world_id == world_id,
            WorldORM.version == expected_version,
        )

        if required_status is not None:
            statement = statement.where(
                WorldORM.status == required_status
            )

        result = session.execute(
            statement.values(
                **values,
                version=new_version,
                updated_at=utc_now(),
            )
        )

        if result.rowcount != 1:
            raise WorldVersionConflict(
                "World version or status changed"
            )

        self._record_event(
            session,
            world_id=world_id,
            event_type=event_type,
            version=new_version,
            payload=event_payload,
        )

        session.flush()

        # Expire any previously loaded instance
        # so the next query sees current values.
        session.expire_all()

        row = self._get_world_row(
            session,
            world_id,
        )

        return self._world_record(row)

    # --------------------------------------------------------
    # UPDATE WORLD STATE
    # --------------------------------------------------------

    def update_world_state(
        self,
        world_id: str,
        new_state: dict[str, Any],
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        state = validate_state(
            new_state
        )

        if expected_version < 1:
            raise WorldValidationError(
                "expected_version must be positive"
            )

        with self._session(db) as session:
            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={
                    "state_data": state,
                },
                event_type=WorldEventType.STATE_UPDATED,
                event_payload={
                    "state": state,
                },
                required_status=WorldStatus.ACTIVE.value,
            )

    # --------------------------------------------------------
    # RECORD ACTION TRANSITION
    # --------------------------------------------------------

    def record_action_transition(
        self,
        world_id: str,
        *,
        action: Any,
        next_state: dict[str, Any],
        expected_version: int,
        model_name: str = "unknown",
        db: Session | None = None,
    ) -> WorldRecord:
        """
        Persist a transition already computed
        by a trusted simulation/world service.

        This method does not execute actions.
        """

        action_data = json_safe(
            action
        )

        state = validate_state(
            next_state
        )

        return self._apply_transition(
            world_id=world_id,
            expected_version=expected_version,
            state=state,
            event_type=WorldEventType.ACTION_APPLIED,
            event_payload={
                "action": action_data,
                "model_name": model_name,
                "next_state": state,
            },
            db=db,
        )

    def _apply_transition(
        self,
        *,
        world_id: str,
        expected_version: int,
        state: dict[str, Any],
        event_type: WorldEventType,
        event_payload: dict[str, Any],
        db: Session | None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={
                    "state_data": state,
                },
                event_type=event_type,
                event_payload=event_payload,
                required_status=WorldStatus.ACTIVE.value,
            )

    # --------------------------------------------------------
    # ATTACH OBSERVATION REFERENCE
    # --------------------------------------------------------

    def attach_observation(
        self,
        world_id: str,
        *,
        observation_id: str,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        observation_id = validate_identifier(
            observation_id,
            "observation_id",
        )

        with self._session(db) as session:
            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={},
                event_type=(
                    WorldEventType.OBSERVATION_ATTACHED
                ),
                event_payload={
                    "observation_id": observation_id,
                },
                required_status=WorldStatus.ACTIVE.value,
            )

    # --------------------------------------------------------
    # LIFECYCLE OPERATIONS
    # --------------------------------------------------------

    def pause_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={
                    "status": WorldStatus.PAUSED.value,
                },
                event_type=WorldEventType.PAUSED,
                event_payload={},
                required_status=WorldStatus.ACTIVE.value,
            )

    def resume_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={
                    "status": WorldStatus.ACTIVE.value,
                },
                event_type=WorldEventType.RESUMED,
                event_payload={},
                required_status=WorldStatus.PAUSED.value,
            )

    def archive_world(
        self,
        world_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> WorldRecord:
        with self._session(db) as session:
            row = self._get_world_row(
                session,
                world_id,
            )

            if row.status == WorldStatus.ARCHIVED.value:
                raise WorldStateError(
                    "World is already archived"
                )

            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={
                    "status": WorldStatus.ARCHIVED.value,
                },
                event_type=WorldEventType.ARCHIVED,
                event_payload={},
                required_status=row.status,
            )

    # --------------------------------------------------------
    # CREATE SNAPSHOT
    # --------------------------------------------------------

    def create_snapshot(
        self,
        world_id: str,
        *,
        snapshot_name: str = "manual_snapshot",
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> WorldSnapshotRecord:
        snapshot_name = validate_identifier(
            snapshot_name,
            "snapshot_name",
            255,
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            world = self._get_world_row(
                session,
                world_id,
            )

            snapshot = WorldSnapshotORM(
                snapshot_id=uuid.uuid4().hex,
                world_id=world_id,
                snapshot_name=snapshot_name,
                world_version=world.version,
                state_data=deep_copy_json(
                    world.state_data
                ),
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(snapshot)
            session.flush()

            return self._snapshot_record(snapshot)

    # --------------------------------------------------------
    # GET SNAPSHOT
    # --------------------------------------------------------

    def get_snapshot(
        self,
        snapshot_id: str,
        *,
        world_id: str | None = None,
        db: Session | None = None,
    ) -> WorldSnapshotRecord:
        with self._session(db) as session:
            snapshot = session.get(
                WorldSnapshotORM,
                snapshot_id,
            )

            if snapshot is None:
                raise WorldNotFoundError(
                    "Snapshot not found"
                )

            if (
                world_id is not None
                and snapshot.world_id != world_id
            ):
                raise WorldNotFoundError(
                    "Snapshot not found"
                )

            return self._snapshot_record(snapshot)

    # --------------------------------------------------------
    # LIST SNAPSHOTS
    # --------------------------------------------------------

    def list_snapshots(
        self,
        world_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[WorldSnapshotRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise WorldValidationError(
                "Invalid pagination"
            )

        statement = (
            select(WorldSnapshotORM)
            .where(
                WorldSnapshotORM.world_id == world_id
            )
            .order_by(
                WorldSnapshotORM.created_at.desc(),
                WorldSnapshotORM.snapshot_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._snapshot_record(row)
                for row in rows
            ]

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
        """
        Restore a stored world state.

        The world must be paused.

        Restoration creates a new world version
        and does not erase historical events.

        A running simulation must be stopped
        separately before this operation.
        """

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

            restored_state = deep_copy_json(
                snapshot.state_data
            )

            return self._versioned_update(
                session,
                world_id=world_id,
                expected_version=expected_version,
                values={
                    "state_data": restored_state,
                },
                event_type=WorldEventType.ROLLED_BACK,
                event_payload={
                    "snapshot_id": snapshot_id,
                    "snapshot_version": (
                        snapshot.world_version
                    ),
                },
                required_status=WorldStatus.PAUSED.value,
            )

    # --------------------------------------------------------
    # EVENT HISTORY
    # --------------------------------------------------------

    def list_events(
        self,
        world_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[WorldEventRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise WorldValidationError(
                "Invalid pagination"
            )

        statement = (
            select(WorldEventORM)
            .where(
                WorldEventORM.world_id == world_id
            )
            .order_by(
                WorldEventORM.world_version.asc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._event_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # DELETE WORLD
    # --------------------------------------------------------

    def delete_world(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> bool:
        """
        Permanently delete an archived world.

        Related events and snapshots are
        deleted through ORM cascades.

        Does not delete external trajectory,
        checkpoint or artifact records.
        """

        with self._session(db) as session:
            row = session.get(
                WorldORM,
                world_id,
            )

            if row is None:
                return False

            if row.status != WorldStatus.ARCHIVED.value:
                raise WorldStateError(
                    "World must be archived before deletion"
                )

            session.delete(row)
            session.flush()

            return True


# ============================================================
# FACTORY
# ============================================================

def create_world_repository() -> WorldRepository:
    return WorldRepository()


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

    repository = create_world_repository()

    world = repository.create_world(
        name="NeuroWorld Navigation Demo",
        description="Synthetic navigation environment",
        environment_type="synthetic_physics",
        initial_state={
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "objects": [],
            "simulation_step": 0,
        },
        configuration={
            "time_step": 0.1,
            "max_steps": 200,
        },
    )

    print("=" * 60)
    print("NEUROWORLD AI - WORLD DATABASE")
    print("=" * 60)

    print(
        "World created:",
        world.world_id,
    )

    world = repository.update_world_state(
        world.world_id,
        {
            "position": [1.0, 2.0],
            "velocity": [0.1, 0.2],
            "objects": [],
            "simulation_step": 1,
        },
        expected_version=world.version,
    )

    snapshot = repository.create_snapshot(
        world.world_id,
        snapshot_name="navigation_checkpoint",
    )

    world = repository.update_world_state(
        world.world_id,
        {
            "position": [5.0, 8.0],
            "velocity": [0.5, 0.3],
            "objects": [],
            "simulation_step": 10,
        },
        expected_version=world.version,
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

    print(
        "Restored world state:",
        json.dumps(
            world.state,
            indent=2,
        ),
    )

    print(
        "Current world version:",
        world.version,
    )

    events = repository.list_events(
        world.world_id
    )

    print(
        "Recorded events:",
        len(events),
    )

    close_database()


if __name__ == "__main__":
    main()
