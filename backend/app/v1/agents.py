
"""
NeuroWorld AI
backend/app/database/agents.py

Persistent Agent Registry, Tasks and Execution History.

Features:
- Register AI agents
- Track agent lifecycle
- Store agent capabilities and configurations
- Assign simulation and research tasks
- Record task execution results
- Store agent heartbeats
- Detect stale agents
- Track agent errors
- Record audit events
- List agents and tasks with pagination
- Export agent activity
- SQLite and PostgreSQL support

Dependencies:
    pip install sqlalchemy

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Safety:
    This is a persistence layer only.
    It does not execute agent tools, issue physical
    commands or authorize real-world actuation.

    Authentication, authorization, agent policy,
    and tenant isolation belong to the service
    and API layers.
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
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime,
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
    DeclarativeBase,
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
    "neuroworld.database.agents"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class AgentDatabaseError(Exception):
    """Base database exception."""


class AgentValidationError(AgentDatabaseError):
    """Invalid agent or task input."""


class AgentNotFoundError(AgentDatabaseError):
    """Agent or task was not found."""


class AgentStateError(AgentDatabaseError):
    """Invalid lifecycle transition."""


class AgentVersionConflict(AgentDatabaseError):
    """Concurrent update conflict."""


# ============================================================
# ENUMS
# ============================================================

class AgentType(str, Enum):
    WORLD = "world"
    EXPLORATION = "exploration"
    PLANNING = "planning"
    IMAGINATION = "imagination"
    NAVIGATION = "navigation"
    CURIOSITY = "curiosity"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    REGISTERED = "registered"
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    OFFLINE = "offline"
    DISABLED = "disabled"


class AgentTaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATUSES = {
    AgentTaskStatus.COMPLETED,
    AgentTaskStatus.FAILED,
    AgentTaskStatus.CANCELLED,
}


# ============================================================
# VALIDATION UTILITIES
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


def normalized_datetime(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


def validate_identifier(
    value: str,
    name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise AgentValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise AgentValidationError(
            f"Invalid {name}"
        )

    return value


def json_safe(value: Any) -> Any:
    """
    Convert supported objects into JSON data.

    Non-finite numeric values are rejected.
    """

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
            raise AgentValidationError(
                "Non-finite numbers are not allowed"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise AgentValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise AgentValidationError(
            f"{name} must be a dictionary"
        )

    return result


def validate_capabilities(
    capabilities: list[str] | None,
) -> list[str]:
    if capabilities is None:
        return []

    if not isinstance(capabilities, list):
        raise AgentValidationError(
            "capabilities must be a list"
        )

    result = []

    for capability in capabilities:
        item = validate_identifier(
            capability,
            "capability",
        )

        if item not in result:
            result.append(item)

    if len(result) > 100:
        raise AgentValidationError(
            "Too many capabilities"
        )

    return result


# ============================================================
# DATABASE MODELS
# ============================================================

class AgentORM(Base):
    __tablename__ = "ai_agents"

    __table_args__ = (
        Index(
            "ix_ai_agents_type_status",
            "agent_type",
            "status",
        ),
    )

    agent_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    agent_type: Mapped[str] = mapped_column(
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
        String(128),
        nullable=False,
        default="unknown",
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    capabilities_data: Mapped[list[str]] = mapped_column(
        "capabilities",
        JSON,
        nullable=False,
        default=list,
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

    last_heartbeat_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    tasks: Mapped[list["AgentTaskORM"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[list["AgentEventORM"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class AgentTaskORM(Base):
    __tablename__ = "ai_agent_tasks"

    __table_args__ = (
        Index(
            "ix_ai_agent_tasks_agent_status",
            "agent_id",
            "status",
        ),
    )

    task_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    agent_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_agents.agent_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    world_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    task_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=5,
    )

    input_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "input",
        JSON,
        nullable=False,
        default=dict,
    )

    result_data: Mapped[
        dict[str, Any] | None
    ] = mapped_column(
        "result",
        JSON,
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

    agent: Mapped[AgentORM] = relationship(
        back_populates="tasks",
    )


class AgentEventORM(Base):
    __tablename__ = "ai_agent_events"

    __table_args__ = (
        Index(
            "ix_ai_agent_events_agent_created",
            "agent_id",
            "created_at",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    agent_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_agents.agent_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    task_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
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

    agent: Mapped[AgentORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class AgentRecord:
    agent_id: str
    name: str
    agent_type: str
    status: str
    model_name: str
    model_version: str
    capabilities: list[str]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    version: int
    last_heartbeat_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AgentTaskRecord:
    task_id: str
    agent_id: str
    world_id: str | None
    task_type: str
    status: str
    priority: int
    input: dict[str, Any]
    result: dict[str, Any] | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True)
class AgentEventRecord:
    event_id: str
    agent_id: str
    event_type: str
    task_id: str | None
    details: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class AgentStatistics:
    agent_id: str
    total_tasks: int
    queued_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    completion_rate: float


# ============================================================
# AGENT REPOSITORY
# ============================================================

class AgentRepository:
    """
    Persistent agent registry and task history.

    The repository does not execute agent tasks.

    The caller controls transactions when
    passing an existing Session.
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
    def _to_agent(
        row: AgentORM,
    ) -> AgentRecord:
        return AgentRecord(
            agent_id=row.agent_id,
            name=row.name,
            agent_type=row.agent_type,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            capabilities=copy.deepcopy(
                row.capabilities_data
            ),
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            version=row.version,
            last_heartbeat_at=datetime_string(
                row.last_heartbeat_at
            ),
            created_at=datetime_string(
                row.created_at
            ),
            updated_at=datetime_string(
                row.updated_at
            ),
        )

    @staticmethod
    def _to_task(
        row: AgentTaskORM,
    ) -> AgentTaskRecord:
        return AgentTaskRecord(
            task_id=row.task_id,
            agent_id=row.agent_id,
            world_id=row.world_id,
            task_type=row.task_type,
            status=row.status,
            priority=row.priority,
            input=copy.deepcopy(
                row.input_data
            ),
            result=copy.deepcopy(
                row.result_data
            ),
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
    def _to_event(
        row: AgentEventORM,
    ) -> AgentEventRecord:
        return AgentEventRecord(
            event_id=row.event_id,
            agent_id=row.agent_id,
            event_type=row.event_type,
            task_id=row.task_id,
            details=copy.deepcopy(
                row.details_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    @staticmethod
    def _get_agent(
        session: Session,
        agent_id: str,
    ) -> AgentORM:
        row = session.get(
            AgentORM,
            agent_id,
        )

        if row is None:
            raise AgentNotFoundError(
                "Agent not found"
            )

        return row

    @staticmethod
    def _get_task(
        session: Session,
        task_id: str,
    ) -> AgentTaskORM:
        row = session.get(
            AgentTaskORM,
            task_id,
        )

        if row is None:
            raise AgentNotFoundError(
                "Agent task not found"
            )

        return row

    @staticmethod
    def _add_event(
        session: Session,
        *,
        agent_id: str,
        event_type: str,
        task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            AgentEventORM(
                event_id=uuid.uuid4().hex,
                agent_id=agent_id,
                event_type=event_type,
                task_id=task_id,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # REGISTER AGENT
    # --------------------------------------------------------

    def register_agent(
        self,
        *,
        name: str,
        agent_type: AgentType,
        model_name: str = "unknown",
        model_version: str = "unknown",
        capabilities: list[str] | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> AgentRecord:
        name = validate_identifier(
            name,
            "name",
            255,
        )

        if not isinstance(agent_type, AgentType):
            raise AgentValidationError(
                "Invalid agent_type"
            )

        model_name = validate_identifier(
            model_name,
            "model_name",
        )

        model_version = validate_identifier(
            model_version,
            "model_version",
        )

        capabilities = validate_capabilities(
            capabilities
        )

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        agent_id = uuid.uuid4().hex
        now = utc_now()

        row = AgentORM(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type.value,
            status=AgentStatus.REGISTERED.value,
            model_name=model_name,
            model_version=model_version,
            capabilities_data=capabilities,
            configuration_data=configuration,
            metadata_data=metadata,
            version=1,
            created_at=now,
            updated_at=now,
        )

        with self._session(db) as session:
            session.add(row)

            self._add_event(
                session,
                agent_id=agent_id,
                event_type="agent_registered",
                details={
                    "name": name,
                    "type": agent_type.value,
                },
            )

            session.flush()

            return self._to_agent(row)

    # --------------------------------------------------------
    # GET AGENT
    # --------------------------------------------------------

    def get_agent(
        self,
        agent_id: str,
        *,
        db: Session | None = None,
    ) -> AgentRecord:
        agent_id = validate_identifier(
            agent_id,
            "agent_id",
            64,
        )

        with self._session(db) as session:
            return self._to_agent(
                self._get_agent(
                    session,
                    agent_id,
                )
            )

    # --------------------------------------------------------
    # UPDATE AGENT STATUS
    # --------------------------------------------------------

    def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> AgentRecord:
        if not isinstance(status, AgentStatus):
            raise AgentValidationError(
                "Invalid agent status"
            )

        allowed = {
            AgentStatus.REGISTERED: {
                AgentStatus.IDLE,
                AgentStatus.DISABLED,
            },
            AgentStatus.IDLE: {
                AgentStatus.BUSY,
                AgentStatus.PAUSED,
                AgentStatus.OFFLINE,
                AgentStatus.DISABLED,
            },
            AgentStatus.BUSY: {
                AgentStatus.IDLE,
                AgentStatus.PAUSED,
                AgentStatus.OFFLINE,
                AgentStatus.DISABLED,
            },
            AgentStatus.PAUSED: {
                AgentStatus.IDLE,
                AgentStatus.OFFLINE,
                AgentStatus.DISABLED,
            },
            AgentStatus.OFFLINE: {
                AgentStatus.IDLE,
                AgentStatus.DISABLED,
            },
            AgentStatus.DISABLED: set(),
        }

        with self._session(db) as session:
            row = self._get_agent(
                session,
                agent_id,
            )

            current = AgentStatus(
                row.status
            )

            if status not in allowed[current]:
                raise AgentStateError(
                    f"Invalid transition: "
                    f"{current.value} -> {status.value}"
                )

            result = session.execute(
                update(AgentORM)
                .where(
                    AgentORM.agent_id == agent_id,
                    AgentORM.version == expected_version,
                    AgentORM.status == current.value,
                )
                .values(
                    status=status.value,
                    version=expected_version + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise AgentVersionConflict(
                    "Agent changed concurrently"
                )

            self._add_event(
                session,
                agent_id=agent_id,
                event_type="status_changed",
                details={
                    "from": current.value,
                    "to": status.value,
                },
            )

            session.flush()
            session.expire_all()

            return self._to_agent(
                self._get_agent(
                    session,
                    agent_id,
                )
            )

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def heartbeat(
        self,
        agent_id: str,
        *,
        db: Session | None = None,
    ) -> AgentRecord:
        with self._session(db) as session:
            row = self._get_agent(
                session,
                agent_id,
            )

            if row.status == AgentStatus.DISABLED.value:
                raise AgentStateError(
                    "Disabled agents cannot heartbeat"
                )

            row.last_heartbeat_at = utc_now()
            row.updated_at = utc_now()

            session.flush()

            return self._to_agent(row)

    # --------------------------------------------------------
    # STALE AGENT DETECTION
    # --------------------------------------------------------

    def list_stale_agents(
        self,
        *,
        timeout_seconds: int = 120,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[AgentRecord]:
        if timeout_seconds < 1:
            raise AgentValidationError(
                "timeout_seconds must be positive"
            )

        if not 1 <= limit <= 1000:
            raise AgentValidationError(
                "Invalid limit"
            )

        cutoff = utc_now() - timedelta(
            seconds=timeout_seconds
        )

        statement = (
            select(AgentORM)
            .where(
                AgentORM.status.in_(
                    [
                        AgentStatus.IDLE.value,
                        AgentStatus.BUSY.value,
                    ]
                ),
                (
                    AgentORM.last_heartbeat_at
                    < cutoff
                )
                | (
                    AgentORM.last_heartbeat_at.is_(None)
                ),
            )
            .order_by(
                AgentORM.updated_at.asc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._to_agent(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # CREATE TASK
    # --------------------------------------------------------

    def create_task(
        self,
        *,
        agent_id: str,
        task_type: str,
        input_data: dict[str, Any],
        world_id: str | None = None,
        priority: int = 5,
        db: Session | None = None,
    ) -> AgentTaskRecord:
        task_type = validate_identifier(
            task_type,
            "task_type",
        )

        input_data = validate_json_dict(
            input_data,
            "input_data",
        )

        if not isinstance(priority, int) or not (
            0 <= priority <= 10
        ):
            raise AgentValidationError(
                "priority must be between 0 and 10"
            )

        if world_id is not None:
            world_id = validate_identifier(
                world_id,
                "world_id",
            )

        with self._session(db) as session:
            agent = self._get_agent(
                session,
                agent_id,
            )

            if agent.status not in {
                AgentStatus.IDLE.value,
                AgentStatus.BUSY.value,
            }:
                raise AgentStateError(
                    "Agent is not accepting tasks"
                )

            task_id = uuid.uuid4().hex

            row = AgentTaskORM(
                task_id=task_id,
                agent_id=agent_id,
                world_id=world_id,
                task_type=task_type,
                status=AgentTaskStatus.QUEUED.value,
                priority=priority,
                input_data=input_data,
                created_at=utc_now(),
            )

            session.add(row)

            self._add_event(
                session,
                agent_id=agent_id,
                task_id=task_id,
                event_type="task_created",
                details={
                    "task_type": task_type,
                    "priority": priority,
                },
            )

            session.flush()

            return self._to_task(row)

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    def start_task(
        self,
        task_id: str,
        *,
        db: Session | None = None,
    ) -> AgentTaskRecord:
        with self._session(db) as session:
            task = self._get_task(
                session,
                task_id,
            )

            if task.status != (
                AgentTaskStatus.QUEUED.value
            ):
                raise AgentStateError(
                    "Only queued tasks can start"
                )

            agent = self._get_agent(
                session,
                task.agent_id,
            )

            if agent.status != AgentStatus.IDLE.value:
                raise AgentStateError(
                    "Agent must be idle to start a task"
                )

            task.status = AgentTaskStatus.RUNNING.value
            task.started_at = utc_now()

            agent.status = AgentStatus.BUSY.value
            agent.version += 1
            agent.updated_at = utc_now()

            self._add_event(
                session,
                agent_id=agent.agent_id,
                task_id=task_id,
                event_type="task_started",
            )

            session.flush()

            return self._to_task(task)

    # --------------------------------------------------------
    # FINISH TASK
    # --------------------------------------------------------

    def finish_task(
        self,
        task_id: str,
        *,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
        status: AgentTaskStatus = (
            AgentTaskStatus.COMPLETED
        ),
        db: Session | None = None,
    ) -> AgentTaskRecord:
        if status not in {
            AgentTaskStatus.COMPLETED,
            AgentTaskStatus.FAILED,
        }:
            raise AgentValidationError(
                "finish_task supports completed or failed"
            )

        if status == AgentTaskStatus.COMPLETED:
            result = validate_json_dict(
                result or {},
                "result",
            )

        with self._session(db) as session:
            task = self._get_task(
                session,
                task_id,
            )

            if task.status != (
                AgentTaskStatus.RUNNING.value
            ):
                raise AgentStateError(
                    "Only running tasks can finish"
                )

            task.status = status.value
            task.finished_at = utc_now()

            if status == AgentTaskStatus.COMPLETED:
                task.result_data = result
            else:
                task.error_message = (
                    error_message or "Task failed"
                )

            agent = self._get_agent(
                session,
                task.agent_id,
            )

            # A disabled/paused/offline agent
            # must not be reactivated by a
            # delayed task completion.
            if agent.status == AgentStatus.BUSY.value:
                agent.status = AgentStatus.IDLE.value
                agent.version += 1
                agent.updated_at = utc_now()

            self._add_event(
                session,
                agent_id=agent.agent_id,
                task_id=task_id,
                event_type=f"task_{status.value}",
                details={
                    "error": task.error_message,
                },
            )

            session.flush()

            return self._to_task(task)

    # --------------------------------------------------------
    # CANCEL QUEUED TASK
    # --------------------------------------------------------

    def cancel_task(
        self,
        task_id: str,
        *,
        reason: str = "Cancelled",
        db: Session | None = None,
    ) -> AgentTaskRecord:
        with self._session(db) as session:
            task = self._get_task(
                session,
                task_id,
            )

            if task.status != (
                AgentTaskStatus.QUEUED.value
            ):
                raise AgentStateError(
                    "Only queued tasks can be cancelled"
                )

            task.status = AgentTaskStatus.CANCELLED.value
            task.finished_at = utc_now()
            task.error_message = reason

            self._add_event(
                session,
                agent_id=task.agent_id,
                task_id=task_id,
                event_type="task_cancelled",
                details={"reason": reason},
            )

            session.flush()

            return self._to_task(task)

    # --------------------------------------------------------
    # GET TASK
    # --------------------------------------------------------

    def get_task(
        self,
        task_id: str,
        *,
        db: Session | None = None,
    ) -> AgentTaskRecord:
        with self._session(db) as session:
            return self._to_task(
                self._get_task(
                    session,
                    task_id,
                )
            )

    # --------------------------------------------------------
    # LIST AGENTS
    # --------------------------------------------------------

    def list_agents(
        self,
        *,
        agent_type: AgentType | None = None,
        status: AgentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[AgentRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise AgentValidationError(
                "Invalid pagination"
            )

        statement = select(
            AgentORM
        )

        if agent_type is not None:
            statement = statement.where(
                AgentORM.agent_type
                == agent_type.value
            )

        if status is not None:
            statement = statement.where(
                AgentORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                AgentORM.created_at.desc(),
                AgentORM.agent_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._to_agent(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST TASKS
    # --------------------------------------------------------

    def list_tasks(
        self,
        *,
        agent_id: str | None = None,
        world_id: str | None = None,
        status: AgentTaskStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[AgentTaskRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise AgentValidationError(
                "Invalid pagination"
            )

        statement = select(
            AgentTaskORM
        )

        if agent_id is not None:
            statement = statement.where(
                AgentTaskORM.agent_id == agent_id
            )

        if world_id is not None:
            statement = statement.where(
                AgentTaskORM.world_id == world_id
            )

        if status is not None:
            statement = statement.where(
                AgentTaskORM.status == status.value
            )

        statement = (
            statement
            .order_by(
                AgentTaskORM.priority.desc(),
                AgentTaskORM.created_at.asc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._to_task(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def list_events(
        self,
        agent_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[AgentEventRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise AgentValidationError(
                "Invalid pagination"
            )

        statement = (
            select(AgentEventORM)
            .where(
                AgentEventORM.agent_id == agent_id
            )
            .order_by(
                AgentEventORM.created_at.asc(),
                AgentEventORM.event_id.asc(),
            )
            .offset(offset)
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
    # AGENT STATISTICS
    # --------------------------------------------------------

    def get_agent_statistics(
        self,
        agent_id: str,
        *,
        db: Session | None = None,
    ) -> AgentStatistics:
        with self._session(db) as session:
            self._get_agent(
                session,
                agent_id,
            )

            rows = session.execute(
                select(
                    AgentTaskORM.status,
                    func.count(
                        AgentTaskORM.task_id
                    ),
                )
                .where(
                    AgentTaskORM.agent_id == agent_id
                )
                .group_by(
                    AgentTaskORM.status
                )
            ).all()

        counts = {
            status: int(count)
            for status, count in rows
        }

        total = sum(counts.values())

        completed = counts.get(
            AgentTaskStatus.COMPLETED.value,
            0,
        )

        return AgentStatistics(
            agent_id=agent_id,
            total_tasks=total,
            queued_tasks=counts.get(
                AgentTaskStatus.QUEUED.value,
                0,
            ),
            running_tasks=counts.get(
                AgentTaskStatus.RUNNING.value,
                0,
            ),
            completed_tasks=completed,
            failed_tasks=counts.get(
                AgentTaskStatus.FAILED.value,
                0,
            ),
            cancelled_tasks=counts.get(
                AgentTaskStatus.CANCELLED.value,
                0,
            ),
            completion_rate=(
                completed / total
                if total
                else 0.0
            ),
        )

    # --------------------------------------------------------
    # EXPORT AGENT HISTORY
    # --------------------------------------------------------

    def export_json(
        self,
        agent_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        payload = {
            "agent": asdict(
                self.get_agent(
                    agent_id,
                    db=db,
                )
            ),
            "tasks": [
                asdict(task)
                for task in self.list_tasks(
                    agent_id=agent_id,
                    limit=1000,
                    db=db,
                )
            ],
            "events": [
                asdict(event)
                for event in self.list_events(
                    agent_id,
                    limit=1000,
                    db=db,
                )
            ],
            "statistics": asdict(
                self.get_agent_statistics(
                    agent_id,
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

    def export_tasks_csv(
        self,
        agent_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        tasks = self.list_tasks(
            agent_id=agent_id,
            limit=1000,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            AgentTaskRecord.__dataclass_fields__.keys()
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

            for task in tasks:
                row = asdict(task)

                row["input"] = json.dumps(
                    row["input"],
                    allow_nan=False,
                )

                row["result"] = json.dumps(
                    row["result"],
                    allow_nan=False,
                )

                writer.writerow(row)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_agent_repository() -> AgentRepository:
    return AgentRepository()


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

    repository = create_agent_repository()

    agent = repository.register_agent(
        name="NeuroWorld Planning Agent",
        agent_type=AgentType.PLANNING,
        model_name="transformer_world_model",
        model_version="v1.0",
        capabilities=[
            "trajectory_planning",
            "goal_conditioned_planning",
            "simulation",
            "uncertainty_estimation",
        ],
        configuration={
            "planner": "cem",
            "horizon": 12,
            "simulation_only": True,
        },
    )

    agent = repository.update_agent_status(
        agent.agent_id,
        AgentStatus.IDLE,
        expected_version=agent.version,
    )

    repository.heartbeat(
        agent.agent_id
    )

    task = repository.create_task(
        agent_id=agent.agent_id,
        task_type="navigation_planning",
        input_data={
            "initial_state": [0.0, 0.0, 0.0, 0.0],
            "goal_state": [5.0, 5.0, 0.0, 0.0],
            "horizon": 12,
        },
        priority=8,
    )

    task = repository.start_task(
        task.task_id
    )

    # Illustrative result: a real planner must
    # compute and validate its own output.
    task = repository.finish_task(
        task.task_id,
        result={
            "status": "success",
            "planned_actions": [
                [0.5, 0.5],
                [0.4, 0.4],
            ],
            "simulated_only": True,
        },
    )

    statistics = repository.get_agent_statistics(
        agent.agent_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - AGENT DATABASE")
    print("=" * 60)

    print(
        "Agent ID:",
        agent.agent_id,
    )

    print(
        "Task ID:",
        task.task_id,
    )

    print(
        "Task status:",
        task.status,
    )

    print(
        "Statistics:",
        asdict(statistics),
    )

    json_path = repository.export_json(
        agent.agent_id,
        "experiments/reports/agent_history.json",
    )

    csv_path = repository.export_tasks_csv(
        agent.agent_id,
        "experiments/reports/agent_tasks.csv",
    )

    print(
        "JSON:",
        json_path,
    )

    print(
        "CSV:",
        csv_path,
    )

    close_database()


if __name__ == "__main__":
    main()
