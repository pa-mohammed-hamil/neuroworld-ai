
"""
NeuroWorld AI
backend/app/database/actions.py

Persistent World Action Management

Features:
- Register simulated actions
- Store action vectors and parameters
- Validate action dimensions and bounds
- Track action execution lifecycle
- Record predicted and observed outcomes
- Store model uncertainty and rewards
- Record safety validation decisions
- Store action timestamps and audit history
- Filter actions by world, trajectory and status
- Export action history to JSON and CSV
- PostgreSQL / SQLite support

Dependencies:
    pip install sqlalchemy numpy

Run:
    python -m backend.app.database.actions

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Safety:
    This module only records simulated actions.
    It does not send commands to physical robots,
    vehicles or industrial devices.

    Database records are not authorization to
    execute real-world actions.
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
    select,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    mapped_column,
)

from backend.app.database.session import (
    Base,
    session_scope,
)


logger = logging.getLogger(
    "neuroworld.database.actions"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ActionDatabaseError(Exception):
    """Base action database exception."""


class ActionValidationError(ActionDatabaseError):
    """Invalid action data."""


class ActionNotFoundError(ActionDatabaseError):
    """Action does not exist."""


class ActionStateError(ActionDatabaseError):
    """Invalid action lifecycle transition."""


# ============================================================
# ENUMS
# ============================================================

class ActionStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionSource(str, Enum):
    MANUAL = "manual"
    PLANNER = "planner"
    AGENT = "agent"
    POLICY = "policy"
    SIMULATION = "simulation"


TERMINAL_STATUSES = {
    ActionStatus.REJECTED,
    ActionStatus.COMPLETED,
    ActionStatus.FAILED,
    ActionStatus.CANCELLED,
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
    name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise ActionValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise ActionValidationError(
            f"Invalid {name}"
        )

    return value


def validate_vector(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
    max_dimension: int = 1024,
) -> list[float] | None:
    if value is None and allow_none:
        return None

    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ActionValidationError(
            f"{name} must be numeric"
        ) from exc

    if (
        array.ndim != 1
        or array.size < 1
        or array.size > max_dimension
    ):
        raise ActionValidationError(
            f"{name} must be a nonempty 1D vector "
            f"with at most {max_dimension} elements"
        )

    if not np.all(np.isfinite(array)):
        raise ActionValidationError(
            f"{name} contains non-finite values"
        )

    return array.tolist()


def validate_number(
    value: Any,
    name: str,
    *,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ActionValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ActionValidationError(
            f"{name} must be finite"
        )

    return number


def json_safe(value: Any) -> Any:
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
            raise ActionValidationError(
                "Non-finite JSON number"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ActionValidationError(
        f"Unsupported type: {type(value).__name__}"
    )


def validate_metadata(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = json_safe(
        value or {}
    )

    if not isinstance(normalized, dict):
        raise ActionValidationError(
            "metadata must be a dictionary"
        )

    return normalized


# ============================================================
# ORM MODELS
# ============================================================

class ActionORM(Base):
    __tablename__ = "world_actions"

    __table_args__ = (
        Index(
            "ix_world_actions_world_created",
            "world_id",
            "created_at",
        ),
    )

    action_id: Mapped[str] = mapped_column(
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

    trajectory_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    action_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ActionStatus.PENDING.value,
        index=True,
    )

    action_vector_data: Mapped[
        list[float]
    ] = mapped_column(
        "action_vector",
        JSON,
        nullable=False,
    )

    parameters_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "parameters",
        JSON,
        nullable=False,
        default=dict,
    )

    initial_state_data: Mapped[
        list[float] | None
    ] = mapped_column(
        "initial_state",
        JSON,
        nullable=True,
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

    reward: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    uncertainty: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    safety_approved: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    safety_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
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

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ActionAuditORM(Base):
    __tablename__ = "world_action_audit"

    audit_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    action_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "world_actions.action_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(64),
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


# ============================================================
# PUBLIC DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    world_id: str
    trajectory_id: str | None
    action_name: str
    source: str
    status: str

    action_vector: list[float]
    parameters: dict[str, Any]

    initial_state: list[float] | None
    predicted_next_state: list[float] | None
    observed_next_state: list[float] | None

    reward: float | None
    uncertainty: float | None

    safety_approved: bool
    safety_reason: str | None
    error_message: str | None

    metadata: dict[str, Any]

    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class ActionAuditRecord:
    audit_id: str
    action_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ActionStatistics:
    world_id: str
    total_actions: int
    completed_actions: int
    failed_actions: int
    rejected_actions: int
    pending_actions: int
    mean_reward: float | None
    mean_uncertainty: float | None


# ============================================================
# REPOSITORY
# ============================================================

class ActionRepository:
    """
    Persistent action history repository.

    The repository manages database records only.

    It does not execute actions or provide
    real-world device authorization.
    """

    def __init__(
        self,
        *,
        max_action_dimension: int = 1024,
        max_state_dimension: int = 4096,
        action_min: float = -1.0,
        action_max: float = 1.0,
    ) -> None:
        if max_action_dimension < 1:
            raise ValueError(
                "max_action_dimension must be positive"
            )

        if max_state_dimension < 1:
            raise ValueError(
                "max_state_dimension must be positive"
            )

        if not (
            math.isfinite(action_min)
            and math.isfinite(action_max)
            and action_min < action_max
        ):
            raise ValueError(
                "Invalid action bounds"
            )

        self.max_action_dimension = (
            max_action_dimension
        )
        self.max_state_dimension = (
            max_state_dimension
        )
        self.action_min = action_min
        self.action_max = action_max

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
        row: ActionORM,
    ) -> ActionRecord:
        return ActionRecord(
            action_id=row.action_id,
            world_id=row.world_id,
            trajectory_id=row.trajectory_id,
            action_name=row.action_name,
            source=row.source,
            status=row.status,
            action_vector=list(
                row.action_vector_data
            ),
            parameters=dict(
                row.parameters_data
            ),
            initial_state=(
                list(row.initial_state_data)
                if row.initial_state_data is not None
                else None
            ),
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
            safety_approved=row.safety_approved,
            safety_reason=row.safety_reason,
            error_message=row.error_message,
            metadata=dict(
                row.metadata_data
            ),
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
    def _to_audit_record(
        row: ActionAuditORM,
    ) -> ActionAuditRecord:
        return ActionAuditRecord(
            audit_id=row.audit_id,
            action_id=row.action_id,
            event_type=row.event_type,
            details=dict(
                row.details_data
            ),
            created_at=datetime_string(
                row.created_at
            ),
        )

    @staticmethod
    def _audit(
        session: Session,
        *,
        action_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            ActionAuditORM(
                audit_id=uuid.uuid4().hex,
                action_id=action_id,
                event_type=event_type,
                details_data=json_safe(
                    details or {}
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE ACTION
    # --------------------------------------------------------

    def create_action(
        self,
        *,
        world_id: str,
        action_vector: Any,
        action_name: str = "world_action",
        source: ActionSource = ActionSource.SIMULATION,
        trajectory_id: str | None = None,
        initial_state: Any = None,
        parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ActionRecord:
        world_id = validate_identifier(
            world_id,
            "world_id",
            64,
        )

        action_name = validate_identifier(
            action_name,
            "action_name",
            255,
        )

        if not isinstance(source, ActionSource):
            raise ActionValidationError(
                "Invalid action source"
            )

        vector = validate_vector(
            action_vector,
            "action_vector",
            max_dimension=self.max_action_dimension,
        )

        if not all(
            self.action_min <= value <= self.action_max
            for value in vector
        ):
            raise ActionValidationError(
                "Action exceeds configured bounds"
            )

        state = validate_vector(
            initial_state,
            "initial_state",
            allow_none=True,
            max_dimension=self.max_state_dimension,
        )

        parameters = validate_metadata(
            parameters
        )

        metadata = validate_metadata(
            metadata
        )

        action_id = uuid.uuid4().hex

        with self._session(db) as session:
            # Verify world exists without requiring
            # its full ORM definition here.
            from backend.app.database.world import WorldORM

            world = session.get(
                WorldORM,
                world_id,
            )

            if world is None:
                raise ActionValidationError(
                    "World does not exist"
                )

            if world.status != "active":
                raise ActionStateError(
                    "Actions require an active world"
                )

            row = ActionORM(
                action_id=action_id,
                world_id=world_id,
                trajectory_id=trajectory_id,
                action_name=action_name,
                source=source.value,
                status=ActionStatus.PENDING.value,
                action_vector_data=vector,
                parameters_data=parameters,
                initial_state_data=state,
                metadata_data=metadata,
                safety_approved=False,
                created_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                action_id=action_id,
                event_type="action_created",
                details={
                    "source": source.value,
                    "action_name": action_name,
                },
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # GET ACTION
    # --------------------------------------------------------

    def get_action(
        self,
        action_id: str,
        *,
        db: Session | None = None,
    ) -> ActionRecord:
        with self._session(db) as session:
            row = session.get(
                ActionORM,
                action_id,
            )

            if row is None:
                raise ActionNotFoundError(
                    "Action not found"
                )

            return self._to_record(row)

    # --------------------------------------------------------
    # SAFETY VALIDATION
    # --------------------------------------------------------

    def validate_action(
        self,
        action_id: str,
        *,
        approved: bool,
        reason: str,
        db: Session | None = None,
    ) -> ActionRecord:
        """
        Store a simulation safety decision.

        Approval must originate from a trusted
        safety policy or authorized reviewer.

        This method does not perform independent
        collision or physical safety analysis.
        """

        if not isinstance(approved, bool):
            raise ActionValidationError(
                "approved must be boolean"
            )

        if not isinstance(reason, str) or not reason.strip():
            raise ActionValidationError(
                "Safety reason is required"
            )

        with self._session(db) as session:
            row = session.get(
                ActionORM,
                action_id,
            )

            if row is None:
                raise ActionNotFoundError(
                    "Action not found"
                )

            if row.status != ActionStatus.PENDING.value:
                raise ActionStateError(
                    "Only pending actions can be validated"
                )

            row.safety_approved = approved
            row.safety_reason = reason

            row.status = (
                ActionStatus.VALIDATED.value
                if approved
                else ActionStatus.REJECTED.value
            )

            if not approved:
                row.completed_at = utc_now()

            self._audit(
                session,
                action_id=action_id,
                event_type="safety_validation",
                details={
                    "approved": approved,
                    "reason": reason,
                },
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # START SIMULATED ACTION
    # --------------------------------------------------------

    def start_action(
        self,
        action_id: str,
        *,
        db: Session | None = None,
    ) -> ActionRecord:
        with self._session(db) as session:
            row = session.get(
                ActionORM,
                action_id,
            )

            if row is None:
                raise ActionNotFoundError(
                    "Action not found"
                )

            if (
                row.status != ActionStatus.VALIDATED.value
                or not row.safety_approved
            ):
                raise ActionStateError(
                    "Action must be safety-approved "
                    "before execution tracking"
                )

            from backend.app.database.world import WorldORM

            world = session.get(
                WorldORM,
                row.world_id,
            )

            if world is None or world.status != "active":
                raise ActionStateError(
                    "World must be active"
                )

            row.status = ActionStatus.EXECUTING.value
            row.started_at = utc_now()

            self._audit(
                session,
                action_id=action_id,
                event_type="execution_started",
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # COMPLETE ACTION
    # --------------------------------------------------------

    def complete_action(
        self,
        action_id: str,
        *,
        observed_next_state: Any,
        predicted_next_state: Any = None,
        reward: float | None = None,
        uncertainty: float | None = None,
        db: Session | None = None,
    ) -> ActionRecord:
        observed = validate_vector(
            observed_next_state,
            "observed_next_state",
            max_dimension=self.max_state_dimension,
        )

        predicted = validate_vector(
            predicted_next_state,
            "predicted_next_state",
            allow_none=True,
            max_dimension=self.max_state_dimension,
        )

        if predicted is not None and (
            len(predicted) != len(observed)
        ):
            raise ActionValidationError(
                "Predicted and observed state "
                "dimensions must match"
            )

        reward = validate_number(
            reward,
            "reward",
            allow_none=True,
        )

        uncertainty = validate_number(
            uncertainty,
            "uncertainty",
            allow_none=True,
        )

        if uncertainty is not None and not (
            0 <= uncertainty <= 1
        ):
            raise ActionValidationError(
                "uncertainty must be in [0, 1]"
            )

        with self._session(db) as session:
            row = session.get(
                ActionORM,
                action_id,
            )

            if row is None:
                raise ActionNotFoundError(
                    "Action not found"
                )

            if row.status != ActionStatus.EXECUTING.value:
                raise ActionStateError(
                    "Only executing actions can complete"
                )

            if (
                row.initial_state_data is not None
                and len(row.initial_state_data)
                != len(observed)
            ):
                raise ActionValidationError(
                    "Observed state dimension mismatch"
                )

            row.observed_next_state_data = observed
            row.predicted_next_state_data = predicted
            row.reward = reward
            row.uncertainty = uncertainty

            row.status = ActionStatus.COMPLETED.value
            row.completed_at = utc_now()

            self._audit(
                session,
                action_id=action_id,
                event_type="execution_completed",
                details={
                    "reward": reward,
                    "uncertainty": uncertainty,
                },
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # FAIL ACTION
    # --------------------------------------------------------

    def fail_action(
        self,
        action_id: str,
        *,
        error_message: str,
        db: Session | None = None,
    ) -> ActionRecord:
        if not error_message.strip():
            raise ActionValidationError(
                "error_message is required"
            )

        with self._session(db) as session:
            row = session.get(
                ActionORM,
                action_id,
            )

            if row is None:
                raise ActionNotFoundError(
                    "Action not found"
                )

            if row.status != ActionStatus.EXECUTING.value:
                raise ActionStateError(
                    "Only executing actions can fail"
                )

            row.status = ActionStatus.FAILED.value
            row.error_message = error_message
            row.completed_at = utc_now()

            self._audit(
                session,
                action_id=action_id,
                event_type="execution_failed",
                details={
                    "error": error_message,
                },
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # CANCEL ACTION
    # --------------------------------------------------------

    def cancel_action(
        self,
        action_id: str,
        *,
        reason: str = "Cancelled",
        db: Session | None = None,
    ) -> ActionRecord:
        with self._session(db) as session:
            row = session.get(
                ActionORM,
                action_id,
            )

            if row is None:
                raise ActionNotFoundError(
                    "Action not found"
                )

            if row.status not in {
                ActionStatus.PENDING.value,
                ActionStatus.VALIDATED.value,
            }:
                raise ActionStateError(
                    "Only pending or validated "
                    "actions can be cancelled"
                )

            row.status = ActionStatus.CANCELLED.value
            row.completed_at = utc_now()

            self._audit(
                session,
                action_id=action_id,
                event_type="action_cancelled",
                details={
                    "reason": reason,
                },
            )

            session.flush()

            return self._to_record(row)

    # --------------------------------------------------------
    # ACTION AUDIT HISTORY
    # --------------------------------------------------------

    def get_audit_history(
        self,
        action_id: str,
        *,
        limit: int = 1000,
        db: Session | None = None,
    ) -> list[ActionAuditRecord]:
        if not 1 <= limit <= 10000:
            raise ActionValidationError(
                "Invalid limit"
            )

        statement = (
            select(ActionAuditORM)
            .where(
                ActionAuditORM.action_id == action_id
            )
            .order_by(
                ActionAuditORM.created_at.asc(),
                ActionAuditORM.audit_id.asc(),
            )
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._to_audit_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # LIST ACTIONS
    # --------------------------------------------------------

    def list_actions(
        self,
        *,
        world_id: str | None = None,
        trajectory_id: str | None = None,
        status: ActionStatus | None = None,
        source: ActionSource | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[ActionRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ActionValidationError(
                "Invalid pagination"
            )

        statement = select(
            ActionORM
        )

        if world_id is not None:
            statement = statement.where(
                ActionORM.world_id == world_id
            )

        if trajectory_id is not None:
            statement = statement.where(
                ActionORM.trajectory_id == trajectory_id
            )

        if status is not None:
            statement = statement.where(
                ActionORM.status == status.value
            )

        if source is not None:
            statement = statement.where(
                ActionORM.source == source.value
            )

        statement = (
            statement
            .order_by(
                ActionORM.created_at.desc(),
                ActionORM.action_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._to_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # WORLD ACTION STATISTICS
    # --------------------------------------------------------

    def get_world_statistics(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> ActionStatistics:
        """
        Aggregate all action records for one world.

        For very large datasets, replace this
        with SQL aggregate queries.
        """

        with self._session(db) as session:
            rows = session.scalars(
                select(ActionORM).where(
                    ActionORM.world_id == world_id
                )
            ).all()

            rewards = [
                row.reward
                for row in rows
                if row.reward is not None
            ]

            uncertainties = [
                row.uncertainty
                for row in rows
                if row.uncertainty is not None
            ]

            return ActionStatistics(
                world_id=world_id,
                total_actions=len(rows),
                completed_actions=sum(
                    row.status
                    == ActionStatus.COMPLETED.value
                    for row in rows
                ),
                failed_actions=sum(
                    row.status
                    == ActionStatus.FAILED.value
                    for row in rows
                ),
                rejected_actions=sum(
                    row.status
                    == ActionStatus.REJECTED.value
                    for row in rows
                ),
                pending_actions=sum(
                    row.status
                    == ActionStatus.PENDING.value
                    for row in rows
                ),
                mean_reward=(
                    float(np.mean(rewards))
                    if rewards
                    else None
                ),
                mean_uncertainty=(
                    float(np.mean(uncertainties))
                    if uncertainties
                    else None
                ),
            )

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
        actions = self.list_actions(
            world_id=world_id,
            limit=1000,
            db=db,
        )

        payload = {
            "world_id": world_id,
            "actions": [
                asdict(action)
                for action in actions
            ],
            "statistics": asdict(
                self.get_world_statistics(
                    world_id,
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
        world_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        actions = self.list_actions(
            world_id=world_id,
            limit=1000,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = list(
            ActionRecord.__dataclass_fields__.keys()
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

            for action in actions:
                row = asdict(action)

                for key in (
                    "action_vector",
                    "parameters",
                    "initial_state",
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

def create_action_repository(
    *,
    max_action_dimension: int = 1024,
    max_state_dimension: int = 4096,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> ActionRepository:
    return ActionRepository(
        max_action_dimension=max_action_dimension,
        max_state_dimension=max_state_dimension,
        action_min=action_min,
        action_max=action_max,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    from backend.app.database.world import (
        create_world_repository,
    )

    logging.basicConfig(
        level=logging.INFO
    )

    create_database_tables()

    world_repository = create_world_repository()

    world = world_repository.create_world(
        name="Action Demo World",
        initial_state={
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
        },
    )

    repository = create_action_repository()

    action = repository.create_action(
        world_id=world.world_id,
        action_name="move_forward",
        action_vector=[0.5, 0.0],
        source=ActionSource.PLANNER,
        initial_state=[0.0, 0.0, 0.0, 0.0],
        parameters={
            "duration": 0.1,
            "simulation_only": True,
        },
    )

    action = repository.validate_action(
        action.action_id,
        approved=True,
        reason="Approved for synthetic simulation",
    )

    action = repository.start_action(
        action.action_id
    )

    action = repository.complete_action(
        action.action_id,
        predicted_next_state=[
            0.05, 0.0, 0.5, 0.0
        ],
        observed_next_state=[
            0.05, 0.0, 0.5, 0.0
        ],
        reward=1.0,
        uncertainty=0.05,
    )

    print("=" * 60)
    print("NEUROWORLD AI - ACTION DATABASE")
    print("=" * 60)

    print(
        "Action ID:",
        action.action_id,
    )

    print(
        "Action status:",
        action.status,
    )

    print(
        "World statistics:",
        asdict(
            repository.get_world_statistics(
                world.world_id
            )
        ),
    )

    json_path = repository.export_json(
        world.world_id,
        "experiments/reports/action_history.json",
    )

    csv_path = repository.export_csv(
        world.world_id,
        "experiments/reports/action_history.csv",
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
