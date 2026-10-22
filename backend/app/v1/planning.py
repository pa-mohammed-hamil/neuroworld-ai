
"""
NeuroWorld AI
backend/app/database/planning.py

Persistent planning database.

Features:
- Create and manage AI plans
- Store initial states and goals
- Store planning algorithms and configurations
- Record planned action sequences
- Track predicted states and uncertainty
- Register and rank planning candidates
- Store simulated trajectory results
- Approve or reject plans
- Track simulated plan execution
- Optimistic concurrency for plan transitions
- Plan audit history
- JSON/CSV exports

Dependencies:
    pip install sqlalchemy

This module stores planning results.
It does not perform neural inference, run
a planner, or control physical hardware.

Requires:
    backend.app.database.session
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

class PlanningDatabaseError(Exception):
    """Base planning database exception."""


class PlanningValidationError(PlanningDatabaseError):
    """Invalid planning data."""


class PlanNotFoundError(PlanningDatabaseError):
    """Plan was not found."""


class PlanningStateError(PlanningDatabaseError):
    """Invalid plan state transition."""


class PlanningConflictError(PlanningDatabaseError):
    """Concurrent modification or duplicate data."""


# ============================================================
# ENUMS
# ============================================================

class PlanningAlgorithm(str, Enum):
    CEM = "cem"
    RANDOM_SHOOTING = "random_shooting"
    MPC = "mpc"
    LATENT_PLANNING = "latent_planning"
    TREE_SEARCH = "tree_search"
    GOAL_CONDITIONED = "goal_conditioned"
    CUSTOM = "custom"


class PlanStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    READY = "ready"
    APPROVED = "approved"
    REJECTED = "rejected"
    SIMULATING = "simulating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CandidateStatus(str, Enum):
    GENERATED = "generated"
    EVALUATED = "evaluated"
    SELECTED = "selected"
    REJECTED = "rejected"


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
        raise PlanningValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise PlanningValidationError(
            f"Invalid {field}"
        )

    return value


def finite_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise PlanningValidationError(
            f"{field} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanningValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise PlanningValidationError(
            f"{field} must be finite"
        )

    return number


def validate_uncertainty(
    value: Any,
) -> float:
    number = finite_number(
        value,
        "uncertainty",
    )

    if not 0 <= number <= 1:
        raise PlanningValidationError(
            "uncertainty must be between 0 and 1"
        )

    return number


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
            raise PlanningValidationError(
                "JSON contains a non-finite number"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise PlanningValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise PlanningValidationError(
            f"{field} must be a JSON dictionary"
        )

    return result


def validate_vector(
    value: Sequence[float],
    field: str,
    *,
    maximum: int = 4096,
) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise PlanningValidationError(
            f"{field} must be a list or tuple"
        )

    if not 1 <= len(value) <= maximum:
        raise PlanningValidationError(
            f"{field} dimension must be 1..{maximum}"
        )

    return [
        finite_number(item, field)
        for item in value
    ]


# ============================================================
# ORM MODELS
# ============================================================

class PlanORM(Base):
    __tablename__ = "ai_plans"

    __table_args__ = (
        Index(
            "ix_ai_plans_world_status",
            "world_id",
            "status",
        ),
    )

    plan_id: Mapped[str] = mapped_column(
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

    algorithm: Mapped[str] = mapped_column(
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

    initial_state_data: Mapped[
        list[float]
    ] = mapped_column(
        "initial_state",
        JSON,
        nullable=False,
    )

    goal_state_data: Mapped[
        list[float]
    ] = mapped_column(
        "goal_state",
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

    best_candidate_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
    )

    best_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    goal_distance: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    approved_by: Mapped[
        str | None
    ] = mapped_column(
        String(255),
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

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    completed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    candidates: Mapped[
        list["PlanCandidateORM"]
    ] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["PlanEventORM"]
    ] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PlanCandidateORM(Base):
    __tablename__ = "ai_plan_candidates"

    __table_args__ = (
        UniqueConstraint(
            "plan_id",
            "candidate_index",
            name="uq_ai_plan_candidate_index",
        ),
    )

    candidate_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    plan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_plans.plan_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    candidate_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    actions_data: Mapped[
        list[list[float]]
    ] = mapped_column(
        "actions",
        JSON,
        nullable=False,
    )

    predicted_states_data: Mapped[
        list[list[float]]
    ] = mapped_column(
        "predicted_states",
        JSON,
        nullable=False,
    )

    uncertainties_data: Mapped[
        list[float]
    ] = mapped_column(
        "uncertainties",
        JSON,
        nullable=False,
        default=list,
    )

    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    goal_distance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    predicted_reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
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

    plan: Mapped[PlanORM] = relationship(
        back_populates="candidates",
    )


class PlanEventORM(Base):
    __tablename__ = "ai_plan_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    plan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_plans.plan_id",
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

    plan: Mapped[PlanORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class PlanRecord:
    plan_id: str
    world_id: str
    name: str
    algorithm: str
    status: str
    model_name: str
    model_version: str
    initial_state: list[float]
    goal_state: list[float]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    best_candidate_id: str | None
    best_score: float | None
    goal_distance: float | None
    approved_by: str | None
    error_message: str | None
    version: int
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True)
class PlanCandidateRecord:
    candidate_id: str
    plan_id: str
    candidate_index: int
    status: str
    actions: list[list[float]]
    predicted_states: list[list[float]]
    uncertainties: list[float]
    score: float
    goal_distance: float
    predicted_reward: float
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class PlanEventRecord:
    event_id: str
    plan_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


# ============================================================
# PLANNING REPOSITORY
# ============================================================

class PlanningRepository:
    """
    Persistent planning history and lifecycle.

    The repository records candidate trajectories
    supplied by a separate planning service.

    No real-world actuation is performed.
    """

    def __init__(
        self,
        *,
        max_horizon: int = 128,
        max_candidates: int = 4096,
        max_state_dimension: int = 4096,
        max_action_dimension: int = 1024,
    ) -> None:
        for name, value in {
            "max_horizon": max_horizon,
            "max_candidates": max_candidates,
            "max_state_dimension": max_state_dimension,
            "max_action_dimension": max_action_dimension,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise PlanningValidationError(
                    f"{name} must be a positive integer"
                )

        self.max_horizon = max_horizon
        self.max_candidates = max_candidates
        self.max_state_dimension = max_state_dimension
        self.max_action_dimension = max_action_dimension

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
        plan_id: str,
    ) -> PlanORM:
        row = session.get(
            PlanORM,
            plan_id,
        )

        if row is None:
            raise PlanNotFoundError(
                "Plan not found"
            )

        return row

    @staticmethod
    def _record(row: PlanORM) -> PlanRecord:
        return PlanRecord(
            plan_id=row.plan_id,
            world_id=row.world_id,
            name=row.name,
            algorithm=row.algorithm,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            initial_state=copy.deepcopy(
                row.initial_state_data
            ),
            goal_state=copy.deepcopy(
                row.goal_state_data
            ),
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            best_candidate_id=row.best_candidate_id,
            best_score=row.best_score,
            goal_distance=row.goal_distance,
            approved_by=row.approved_by,
            error_message=row.error_message,
            version=row.version,
            created_at=iso_datetime(
                row.created_at
            ),
            updated_at=iso_datetime(
                row.updated_at
            ),
            completed_at=iso_datetime(
                row.completed_at
            ),
        )

    @staticmethod
    def _candidate_record(
        row: PlanCandidateORM,
    ) -> PlanCandidateRecord:
        return PlanCandidateRecord(
            candidate_id=row.candidate_id,
            plan_id=row.plan_id,
            candidate_index=row.candidate_index,
            status=row.status,
            actions=copy.deepcopy(
                row.actions_data
            ),
            predicted_states=copy.deepcopy(
                row.predicted_states_data
            ),
            uncertainties=copy.deepcopy(
                row.uncertainties_data
            ),
            score=row.score,
            goal_distance=row.goal_distance,
            predicted_reward=row.predicted_reward,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _event_record(
        row: PlanEventORM,
    ) -> PlanEventRecord:
        return PlanEventRecord(
            event_id=row.event_id,
            plan_id=row.plan_id,
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
        plan_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            PlanEventORM(
                event_id=uuid.uuid4().hex,
                plan_id=plan_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE PLAN
    # --------------------------------------------------------

    def create_plan(
        self,
        *,
        world_id: str,
        name: str,
        algorithm: PlanningAlgorithm,
        model_name: str,
        initial_state: Sequence[float],
        goal_state: Sequence[float],
        model_version: str = "v1",
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> PlanRecord:
        world_id = validate_string(
            world_id, "world_id", 64
        )

        name = validate_string(
            name, "name"
        )

        model_name = validate_string(
            model_name, "model_name"
        )

        model_version = validate_string(
            model_version, "model_version", 128
        )

        if not isinstance(
            algorithm,
            PlanningAlgorithm,
        ):
            raise PlanningValidationError(
                "Invalid planning algorithm"
            )

        initial = validate_vector(
            initial_state,
            "initial_state",
            maximum=self.max_state_dimension,
        )

        goal = validate_vector(
            goal_state,
            "goal_state",
            maximum=self.max_state_dimension,
        )

        if len(initial) != len(goal):
            raise PlanningValidationError(
                "Initial and goal dimensions must match"
            )

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        row = PlanORM(
            plan_id=uuid.uuid4().hex,
            world_id=world_id,
            name=name,
            algorithm=algorithm.value,
            status=PlanStatus.CREATED.value,
            model_name=model_name,
            model_version=model_version,
            initial_state_data=initial,
            goal_state_data=goal,
            configuration_data=configuration,
            metadata_data=metadata,
            version=1,
            created_at=utc_now(),
            updated_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)

            self._audit(
                session,
                row.plan_id,
                "plan_created",
                {
                    "algorithm": algorithm.value,
                    "world_id": world_id,
                },
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET PLAN
    # --------------------------------------------------------

    def get_plan(
        self,
        plan_id: str,
        *,
        db: Session | None = None,
    ) -> PlanRecord:
        with self._session(db) as session:
            return self._record(
                self._get(
                    session,
                    plan_id,
                )
            )

    # --------------------------------------------------------
    # STATE TRANSITIONS
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        plan_id: str,
        expected_version: int,
        allowed_from: set[PlanStatus],
        target: PlanStatus,
        values: dict[str, Any] | None = None,
    ) -> PlanRecord:
        result = session.execute(
            update(PlanORM)
            .where(
                PlanORM.plan_id == plan_id,
                PlanORM.version == expected_version,
                PlanORM.status.in_(
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
            raise PlanningConflictError(
                "Plan status or version changed"
            )

        self._audit(
            session,
            plan_id,
            f"plan_{target.value}",
        )

        session.flush()
        session.expire_all()

        return self._record(
            self._get(
                session,
                plan_id,
            )
        )

    def start_planning(
        self,
        plan_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> PlanRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={PlanStatus.CREATED},
                target=PlanStatus.PLANNING,
            )

    # --------------------------------------------------------
    # ADD CANDIDATE
    # --------------------------------------------------------

    def add_candidate(
        self,
        plan_id: str,
        *,
        candidate_index: int,
        actions: Sequence[Sequence[float]],
        predicted_states: Sequence[Sequence[float]],
        uncertainties: Sequence[float],
        score: float,
        goal_distance: float,
        predicted_reward: float = 0.0,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> PlanCandidateRecord:
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
            or candidate_index >= self.max_candidates
        ):
            raise PlanningValidationError(
                "Invalid candidate_index"
            )

        if not isinstance(actions, (list, tuple)):
            raise PlanningValidationError(
                "actions must be a sequence"
            )

        if not 1 <= len(actions) <= self.max_horizon:
            raise PlanningValidationError(
                "Invalid action horizon"
            )

        validated_actions = [
            validate_vector(
                action,
                "action",
                maximum=self.max_action_dimension,
            )
            for action in actions
        ]

        action_dimension = len(
            validated_actions[0]
        )

        if any(
            len(action) != action_dimension
            for action in validated_actions
        ):
            raise PlanningValidationError(
                "Inconsistent action dimensions"
            )

        if not isinstance(
            predicted_states,
            (list, tuple),
        ):
            raise PlanningValidationError(
                "predicted_states must be a sequence"
            )

        if len(predicted_states) != len(actions):
            raise PlanningValidationError(
                "Expected one predicted state "
                "per action"
            )

        states = [
            validate_vector(
                state,
                "predicted state",
                maximum=self.max_state_dimension,
            )
            for state in predicted_states
        ]

        if len(uncertainties) != len(actions):
            raise PlanningValidationError(
                "Expected one uncertainty "
                "per action"
            )

        uncertainties = [
            validate_uncertainty(item)
            for item in uncertainties
        ]

        score = finite_number(
            score, "score"
        )

        goal_distance = finite_number(
            goal_distance,
            "goal_distance",
        )

        if goal_distance < 0:
            raise PlanningValidationError(
                "goal_distance cannot be negative"
            )

        predicted_reward = finite_number(
            predicted_reward,
            "predicted_reward",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            plan = self._get(
                session,
                plan_id,
            )

            if plan.status != PlanStatus.PLANNING.value:
                raise PlanningStateError(
                    "Candidates can only be added "
                    "during planning"
                )

            expected_dimension = len(
                plan.initial_state_data
            )

            if any(
                len(state) != expected_dimension
                for state in states
            ):
                raise PlanningValidationError(
                    "Predicted state dimension mismatch"
                )

            existing = session.scalar(
                select(PlanCandidateORM.candidate_id)
                .where(
                    PlanCandidateORM.plan_id
                    == plan_id,
                    PlanCandidateORM.candidate_index
                    == candidate_index,
                )
            )

            if existing is not None:
                raise PlanningConflictError(
                    "Candidate index already exists"
                )

            row = PlanCandidateORM(
                candidate_id=uuid.uuid4().hex,
                plan_id=plan_id,
                candidate_index=candidate_index,
                status=CandidateStatus.EVALUATED.value,
                actions_data=validated_actions,
                predicted_states_data=states,
                uncertainties_data=uncertainties,
                score=score,
                goal_distance=goal_distance,
                predicted_reward=predicted_reward,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._candidate_record(row)

    # --------------------------------------------------------
    # FINALIZE PLANNING
    # --------------------------------------------------------

    def finalize_plan(
        self,
        plan_id: str,
        *,
        expected_version: int,
        candidate_id: str,
        db: Session | None = None,
    ) -> PlanRecord:
        with self._session(db) as session:
            candidate = session.get(
                PlanCandidateORM,
                candidate_id,
            )

            if (
                candidate is None
                or candidate.plan_id != plan_id
            ):
                raise PlanningValidationError(
                    "Candidate does not belong to plan"
                )

            result = self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={PlanStatus.PLANNING},
                target=PlanStatus.READY,
                values={
                    "best_candidate_id": candidate_id,
                    "best_score": candidate.score,
                    "goal_distance": candidate.goal_distance,
                },
            )

            candidate.status = (
                CandidateStatus.SELECTED.value
            )

            session.flush()

            return result

    # --------------------------------------------------------
    # APPROVE PLAN
    # --------------------------------------------------------

    def approve_plan(
        self,
        plan_id: str,
        *,
        expected_version: int,
        approved_by: str,
        db: Session | None = None,
    ) -> PlanRecord:
        approved_by = validate_string(
            approved_by,
            "approved_by",
        )

        # The caller must verify the approver's
        # identity and permissions.
        with self._session(db) as session:
            return self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={PlanStatus.READY},
                target=PlanStatus.APPROVED,
                values={
                    "approved_by": approved_by,
                },
            )

    # --------------------------------------------------------
    # REJECT PLAN
    # --------------------------------------------------------

    def reject_plan(
        self,
        plan_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> PlanRecord:
        reason = validate_string(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={PlanStatus.READY},
                target=PlanStatus.REJECTED,
                values={
                    "error_message": reason,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # SIMULATION EXECUTION
    # --------------------------------------------------------

    def start_simulation(
        self,
        plan_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> PlanRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={PlanStatus.APPROVED},
                target=PlanStatus.SIMULATING,
            )

    def complete_simulation(
        self,
        plan_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> PlanRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={PlanStatus.SIMULATING},
                target=PlanStatus.COMPLETED,
                values={
                    "completed_at": utc_now(),
                },
            )

    def fail_plan(
        self,
        plan_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> PlanRecord:
        reason = validate_string(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                plan_id=plan_id,
                expected_version=expected_version,
                allowed_from={
                    PlanStatus.PLANNING,
                    PlanStatus.SIMULATING,
                },
                target=PlanStatus.FAILED,
                values={
                    "error_message": reason,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # LIST CANDIDATES
    # --------------------------------------------------------

    def list_candidates(
        self,
        plan_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[PlanCandidateRecord]:
        if not 1 <= limit <= self.max_candidates:
            raise PlanningValidationError(
                "Invalid limit"
            )

        statement = (
            select(PlanCandidateORM)
            .where(
                PlanCandidateORM.plan_id == plan_id
            )
            .order_by(
                PlanCandidateORM.score.desc(),
                PlanCandidateORM.candidate_index.asc(),
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._candidate_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST PLANS
    # --------------------------------------------------------

    def list_plans(
        self,
        *,
        world_id: str | None = None,
        status: PlanStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[PlanRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise PlanningValidationError(
                "Invalid pagination"
            )

        statement = select(
            PlanORM
        )

        if world_id is not None:
            statement = statement.where(
                PlanORM.world_id == world_id
            )

        if status is not None:
            statement = statement.where(
                PlanORM.status == status.value
            )

        statement = (
            statement
            .order_by(
                PlanORM.created_at.desc()
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
    # AUDIT HISTORY
    # --------------------------------------------------------

    def list_events(
        self,
        plan_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[PlanEventRecord]:
        if not 1 <= limit <= 1000:
            raise PlanningValidationError(
                "Invalid limit"
            )

        statement = (
            select(PlanEventORM)
            .where(
                PlanEventORM.plan_id == plan_id
            )
            .order_by(
                PlanEventORM.created_at.asc()
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
        plan_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            payload = {
                "plan": asdict(
                    self.get_plan(
                        plan_id,
                        db=session,
                    )
                ),
                "candidates": [
                    asdict(item)
                    for item in self.list_candidates(
                        plan_id,
                        limit=self.max_candidates,
                        db=session,
                    )
                ],
                "events": [
                    asdict(item)
                    for item in self.list_events(
                        plan_id,
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

    def export_candidates_csv(
        self,
        plan_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        candidates = self.list_candidates(
            plan_id,
            limit=self.max_candidates,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fields = [
            "candidate_id",
            "plan_id",
            "candidate_index",
            "status",
            "score",
            "goal_distance",
            "predicted_reward",
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

            for candidate in candidates:
                data = asdict(candidate)

                writer.writerow(
                    {
                        key: data[key]
                        for key in fields
                    }
                )

        return path


# ============================================================
# FACTORY
# ============================================================

def create_planning_repository(
    **kwargs: Any,
) -> PlanningRepository:
    return PlanningRepository(
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

    repository = create_planning_repository()

    plan = repository.create_plan(
        world_id="demo_navigation_world",
        name="Synthetic Navigation Planning",
        algorithm=PlanningAlgorithm.CEM,
        model_name="demo_world_model",
        model_version="v1",
        initial_state=[
            0.0, 0.0, 0.0, 0.0
        ],
        goal_state=[
            5.0, 5.0, 0.0, 0.0
        ],
        configuration={
            "horizon": 3,
            "candidate_count": 2,
            "simulation_only": True,
        },
    )

    plan = repository.start_planning(
        plan.plan_id,
        expected_version=plan.version,
    )

    # Illustrative planner output.
    candidate = repository.add_candidate(
        plan.plan_id,
        candidate_index=0,
        actions=[
            [0.2, 0.2],
            [0.4, 0.4],
            [0.6, 0.6],
        ],
        predicted_states=[
            [0.1, 0.1, 0.1, 0.1],
            [0.3, 0.3, 0.2, 0.2],
            [0.6, 0.6, 0.3, 0.3],
        ],
        uncertainties=[
            0.03, 0.04, 0.05
        ],
        score=-6.3,
        goal_distance=6.22,
        predicted_reward=-6.0,
        metadata={
            "illustrative": True,
        },
    )

    plan = repository.finalize_plan(
        plan.plan_id,
        expected_version=plan.version,
        candidate_id=candidate.candidate_id,
    )

    plan = repository.approve_plan(
        plan.plan_id,
        expected_version=plan.version,
        approved_by="demo_operator",
    )

    plan = repository.start_simulation(
        plan.plan_id,
        expected_version=plan.version,
    )

    # Illustrative lifecycle transition only.
    plan = repository.complete_simulation(
        plan.plan_id,
        expected_version=plan.version,
    )

    print("=" * 60)
    print("NEUROWORLD AI - PLANNING DATABASE")
    print("=" * 60)

    print("Plan ID:", plan.plan_id)
    print("Status:", plan.status)
    print("Algorithm:", plan.algorithm)
    print("Best score:", plan.best_score)

    report = repository.export_json(
        plan.plan_id,
        "experiments/reports/planning_result.json",
    )

    print("Report:", report)

    close_database()


if __name__ == "__main__":
    main()
