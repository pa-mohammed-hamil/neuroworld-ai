
"""
NeuroWorld AI
backend/app/database/predictions.py

Persistent neural world-model prediction tracking.

Features:
- Next-state and future-state prediction records
- Multi-step prediction trajectories
- Action-conditioned predictions
- Uncertainty-aware prediction storage
- Prediction status lifecycle
- Predicted vs observed state evaluation
- MAE and RMSE calculations
- Model and world associations
- Optimistic concurrency
- Prediction audit events
- JSON/CSV report exports

Dependencies:
    pip install sqlalchemy numpy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module stores predictions and evaluates
recorded outputs. Neural inference is performed
by a separate prediction service.
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

class PredictionDatabaseError(Exception):
    """Base prediction database error."""


class PredictionValidationError(
    PredictionDatabaseError
):
    """Invalid prediction input."""


class PredictionNotFoundError(
    PredictionDatabaseError
):
    """Prediction was not found."""


class PredictionStateError(
    PredictionDatabaseError
):
    """Invalid prediction state transition."""


class PredictionConflictError(
    PredictionDatabaseError
):
    """Concurrent modification or duplicate step."""


# ============================================================
# ENUMS
# ============================================================

class PredictionType(str, Enum):
    NEXT_STATE = "next_state"
    FUTURE_STATE = "future_state"
    TRAJECTORY = "trajectory"
    VIDEO_FRAME = "video_frame"
    LATENT_STATE = "latent_state"
    EVENT = "event"
    COUNTERFACTUAL = "counterfactual"
    CUSTOM = "custom"


class PredictionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
        raise PredictionValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise PredictionValidationError(
            f"Invalid {field}"
        )

    return value


def finite_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise PredictionValidationError(
            f"{field} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise PredictionValidationError(
            f"{field} must be finite"
        )

    return number


def validate_uncertainty(
    value: float,
) -> float:
    result = finite_number(
        value,
        "uncertainty",
    )

    if not 0 <= result <= 1:
        raise PredictionValidationError(
            "uncertainty must be between 0 and 1"
        )

    return result


def validate_vector(
    value: Sequence[float],
    field: str,
    maximum_dimension: int = 4096,
) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise PredictionValidationError(
            f"{field} must be a list or tuple"
        )

    if not 1 <= len(value) <= maximum_dimension:
        raise PredictionValidationError(
            f"Invalid {field} dimension"
        )

    return [
        finite_number(item, field)
        for item in value
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
            raise PredictionValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise PredictionValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise PredictionValidationError(
            f"{field} must be a dictionary"
        )

    return result


# ============================================================
# ORM MODELS
# ============================================================

class PredictionORM(Base):
    __tablename__ = "world_predictions"

    __table_args__ = (
        Index(
            "ix_world_predictions_world_status",
            "world_id",
            "status",
        ),
    )

    prediction_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    prediction_type: Mapped[str] = mapped_column(
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
        index=True,
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

    horizon: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    step_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    mean_uncertainty: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    maximum_uncertainty: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    evaluation_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "evaluation",
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

    steps: Mapped[
        list["PredictionStepORM"]
    ] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    events: Mapped[
        list["PredictionEventORM"]
    ] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PredictionStepORM(Base):
    __tablename__ = "world_prediction_steps"

    __table_args__ = (
        UniqueConstraint(
            "prediction_id",
            "step_index",
            name="uq_world_prediction_step",
        ),
    )

    step_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    prediction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "world_predictions.prediction_id",
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

    predicted_next_state_data: Mapped[
        list[float]
    ] = mapped_column(
        "predicted_next_state",
        JSON,
        nullable=False,
    )

    observed_next_state_data: Mapped[
        list[float] | None
    ] = mapped_column(
        "observed_next_state",
        JSON,
        nullable=True,
    )

    uncertainty: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    reward: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    terminal: Mapped[bool] = mapped_column(
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

    prediction: Mapped[PredictionORM] = relationship(
        back_populates="steps",
    )


class PredictionEventORM(Base):
    __tablename__ = "world_prediction_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    prediction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "world_predictions.prediction_id",
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

    prediction: Mapped[PredictionORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    world_id: str
    prediction_type: str
    status: str
    model_name: str
    model_version: str
    initial_state: list[float]
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    horizon: int
    step_count: int
    mean_uncertainty: float | None
    maximum_uncertainty: float | None
    evaluation: dict[str, Any]
    error_message: str | None
    version: int
    created_at: str
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class PredictionStepRecord:
    step_id: str
    prediction_id: str
    step_index: int
    state: list[float]
    action: list[float]
    predicted_next_state: list[float]
    observed_next_state: list[float] | None
    uncertainty: float
    reward: float
    terminal: bool
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class PredictionEventRecord:
    event_id: str
    prediction_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


# ============================================================
# PREDICTION REPOSITORY
# ============================================================

class PredictionRepository:
    """
    Persistent neural world-model prediction history.

    Prediction outputs must be generated by
    a separate inference or prediction service.

    The repository supports caller-managed
    SQLAlchemy sessions or session_scope().
    """

    def __init__(
        self,
        *,
        max_horizon: int = 1024,
        max_state_dimension: int = 4096,
        max_action_dimension: int = 1024,
    ) -> None:
        for name, value in {
            "max_horizon": max_horizon,
            "max_state_dimension": max_state_dimension,
            "max_action_dimension": max_action_dimension,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise PredictionValidationError(
                    f"{name} must be a positive integer"
                )

        self.max_horizon = max_horizon
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
        prediction_id: str,
    ) -> PredictionORM:
        row = session.get(
            PredictionORM,
            prediction_id,
        )

        if row is None:
            raise PredictionNotFoundError(
                "Prediction not found"
            )

        return row

    @staticmethod
    def _record(
        row: PredictionORM,
    ) -> PredictionRecord:
        return PredictionRecord(
            prediction_id=row.prediction_id,
            world_id=row.world_id,
            prediction_type=row.prediction_type,
            status=row.status,
            model_name=row.model_name,
            model_version=row.model_version,
            initial_state=copy.deepcopy(
                row.initial_state_data
            ),
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            horizon=row.horizon,
            step_count=row.step_count,
            mean_uncertainty=row.mean_uncertainty,
            maximum_uncertainty=row.maximum_uncertainty,
            evaluation=copy.deepcopy(
                row.evaluation_data
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
    def _step_record(
        row: PredictionStepORM,
    ) -> PredictionStepRecord:
        return PredictionStepRecord(
            step_id=row.step_id,
            prediction_id=row.prediction_id,
            step_index=row.step_index,
            state=copy.deepcopy(
                row.state_data
            ),
            action=copy.deepcopy(
                row.action_data
            ),
            predicted_next_state=copy.deepcopy(
                row.predicted_next_state_data
            ),
            observed_next_state=copy.deepcopy(
                row.observed_next_state_data
            ),
            uncertainty=row.uncertainty,
            reward=row.reward,
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
        row: PredictionEventORM,
    ) -> PredictionEventRecord:
        return PredictionEventRecord(
            event_id=row.event_id,
            prediction_id=row.prediction_id,
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
        prediction_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            PredictionEventORM(
                event_id=uuid.uuid4().hex,
                prediction_id=prediction_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE PREDICTION
    # --------------------------------------------------------

    def create_prediction(
        self,
        *,
        world_id: str,
        prediction_type: PredictionType,
        model_name: str,
        initial_state: Sequence[float],
        horizon: int,
        model_version: str = "v1",
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> PredictionRecord:
        world_id = validate_string(
            world_id,
            "world_id",
            64,
        )

        model_name = validate_string(
            model_name,
            "model_name",
        )

        model_version = validate_string(
            model_version,
            "model_version",
            128,
        )

        if not isinstance(
            prediction_type,
            PredictionType,
        ):
            raise PredictionValidationError(
                "Invalid prediction type"
            )

        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 1 <= horizon <= self.max_horizon
        ):
            raise PredictionValidationError(
                "Invalid prediction horizon"
            )

        initial_state = validate_vector(
            initial_state,
            "initial_state",
            self.max_state_dimension,
        )

        configuration = validate_json_dict(
            configuration or {},
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        row = PredictionORM(
            prediction_id=uuid.uuid4().hex,
            world_id=world_id,
            prediction_type=prediction_type.value,
            status=PredictionStatus.CREATED.value,
            model_name=model_name,
            model_version=model_version,
            initial_state_data=initial_state,
            configuration_data=configuration,
            metadata_data=metadata,
            horizon=horizon,
            step_count=0,
            version=1,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)

            self._audit(
                session,
                row.prediction_id,
                "prediction_created",
                {
                    "world_id": world_id,
                    "type": prediction_type.value,
                    "horizon": horizon,
                },
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET PREDICTION
    # --------------------------------------------------------

    def get_prediction(
        self,
        prediction_id: str,
        *,
        db: Session | None = None,
    ) -> PredictionRecord:
        with self._session(db) as session:
            return self._record(
                self._get(
                    session,
                    prediction_id,
                )
            )

    # --------------------------------------------------------
    # STATUS TRANSITION
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        prediction_id: str,
        expected_version: int,
        required_status: PredictionStatus,
        new_status: PredictionStatus,
        values: dict[str, Any] | None = None,
    ) -> PredictionRecord:
        result = session.execute(
            update(PredictionORM)
            .where(
                PredictionORM.prediction_id
                == prediction_id,
                PredictionORM.version
                == expected_version,
                PredictionORM.status
                == required_status.value,
            )
            .values(
                **(values or {}),
                status=new_status.value,
                version=expected_version + 1,
            )
        )

        if result.rowcount != 1:
            raise PredictionConflictError(
                "Prediction version or status changed"
            )

        self._audit(
            session,
            prediction_id,
            f"prediction_{new_status.value}",
        )

        session.flush()
        session.expire_all()

        return self._record(
            self._get(
                session,
                prediction_id,
            )
        )

    def start_prediction(
        self,
        prediction_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> PredictionRecord:
        with self._session(db) as session:
            return self._transition(
                session,
                prediction_id=prediction_id,
                expected_version=expected_version,
                required_status=PredictionStatus.CREATED,
                new_status=PredictionStatus.RUNNING,
                values={
                    "started_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # ADD PREDICTION STEP
    # --------------------------------------------------------

    def add_step(
        self,
        prediction_id: str,
        *,
        state: Sequence[float],
        action: Sequence[float],
        predicted_next_state: Sequence[float],
        uncertainty: float = 0.0,
        reward: float = 0.0,
        terminal: bool = False,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> PredictionStepRecord:
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

        predicted_next_state = validate_vector(
            predicted_next_state,
            "predicted_next_state",
            self.max_state_dimension,
        )

        if len(state) != len(predicted_next_state):
            raise PredictionValidationError(
                "State dimensions must match"
            )

        uncertainty = validate_uncertainty(
            uncertainty
        )

        reward = finite_number(
            reward,
            "reward",
        )

        if not isinstance(terminal, bool):
            raise PredictionValidationError(
                "terminal must be boolean"
            )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            prediction = self._get(
                session,
                prediction_id,
            )

            if prediction.status != (
                PredictionStatus.RUNNING.value
            ):
                raise PredictionStateError(
                    "Prediction must be running"
                )

            if prediction.step_count >= prediction.horizon:
                raise PredictionStateError(
                    "Prediction horizon reached"
                )

            if len(state) != len(
                prediction.initial_state_data
            ):
                raise PredictionValidationError(
                    "State dimension mismatch"
                )

            previous = session.scalar(
                select(PredictionStepORM)
                .where(
                    PredictionStepORM.prediction_id
                    == prediction_id
                )
                .order_by(
                    PredictionStepORM.step_index.desc()
                )
                .limit(1)
            )

            if previous is not None:
                if previous.terminal:
                    raise PredictionStateError(
                        "Cannot append after terminal step"
                    )

                if not np.allclose(
                    state,
                    previous.predicted_next_state_data,
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise PredictionValidationError(
                        "State does not match previous "
                        "predicted next state"
                    )

            elif not np.allclose(
                state,
                prediction.initial_state_data,
                atol=1e-6,
                rtol=1e-6,
            ):
                raise PredictionValidationError(
                    "First state must match initial state"
                )

            step_index = prediction.step_count

            # Compare-and-swap protects the step index
            # against concurrent writers.
            claim = session.execute(
                update(PredictionORM)
                .where(
                    PredictionORM.prediction_id
                    == prediction_id,
                    PredictionORM.status
                    == PredictionStatus.RUNNING.value,
                    PredictionORM.step_count
                    == step_index,
                )
                .values(
                    step_count=step_index + 1,
                    version=PredictionORM.version + 1,
                )
            )

            if claim.rowcount != 1:
                raise PredictionConflictError(
                    "Prediction changed concurrently"
                )

            row = PredictionStepORM(
                step_id=uuid.uuid4().hex,
                prediction_id=prediction_id,
                step_index=step_index,
                state_data=state,
                action_data=action,
                predicted_next_state_data=(
                    predicted_next_state
                ),
                uncertainty=uncertainty,
                reward=reward,
                terminal=terminal,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._step_record(row)

    # --------------------------------------------------------
    # GET STEPS
    # --------------------------------------------------------

    def get_steps(
        self,
        prediction_id: str,
        *,
        limit: int = 1024,
        db: Session | None = None,
    ) -> list[PredictionStepRecord]:
        if not 1 <= limit <= self.max_horizon:
            raise PredictionValidationError(
                "Invalid step limit"
            )

        statement = (
            select(PredictionStepORM)
            .where(
                PredictionStepORM.prediction_id
                == prediction_id
            )
            .order_by(
                PredictionStepORM.step_index.asc()
            )
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
    # COMPLETE PREDICTION
    # --------------------------------------------------------

    def complete_prediction(
        self,
        prediction_id: str,
        *,
        expected_version: int,
        db: Session | None = None,
    ) -> PredictionRecord:
        with self._session(db) as session:
            prediction = self._get(
                session,
                prediction_id,
            )

            if prediction.step_count == 0:
                raise PredictionStateError(
                    "Cannot complete an empty prediction"
                )

            uncertainties = session.scalars(
                select(PredictionStepORM.uncertainty)
                .where(
                    PredictionStepORM.prediction_id
                    == prediction_id
                )
            ).all()

            mean_uncertainty = float(
                np.mean(uncertainties)
            )

            maximum_uncertainty = float(
                np.max(uncertainties)
            )

            return self._transition(
                session,
                prediction_id=prediction_id,
                expected_version=expected_version,
                required_status=PredictionStatus.RUNNING,
                new_status=PredictionStatus.COMPLETED,
                values={
                    "mean_uncertainty": mean_uncertainty,
                    "maximum_uncertainty": maximum_uncertainty,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # FAIL PREDICTION
    # --------------------------------------------------------

    def fail_prediction(
        self,
        prediction_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> PredictionRecord:
        reason = validate_string(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            return self._transition(
                session,
                prediction_id=prediction_id,
                expected_version=expected_version,
                required_status=PredictionStatus.RUNNING,
                new_status=PredictionStatus.FAILED,
                values={
                    "error_message": reason,
                    "completed_at": utc_now(),
                },
            )

    # --------------------------------------------------------
    # ATTACH OBSERVED STATE
    # --------------------------------------------------------

    def attach_observed_state(
        self,
        step_id: str,
        *,
        observed_next_state: Sequence[float],
        db: Session | None = None,
    ) -> PredictionStepRecord:
        observed = validate_vector(
            observed_next_state,
            "observed_next_state",
            self.max_state_dimension,
        )

        with self._session(db) as session:
            row = session.get(
                PredictionStepORM,
                step_id,
            )

            if row is None:
                raise PredictionNotFoundError(
                    "Prediction step not found"
                )

            if len(observed) != len(
                row.predicted_next_state_data
            ):
                raise PredictionValidationError(
                    "Observed state dimension mismatch"
                )

            if row.observed_next_state_data is not None:
                raise PredictionConflictError(
                    "Observed state already attached"
                )

            # Conditional update prevents duplicate
            # attachment by concurrent workers.
            result = session.execute(
                update(PredictionStepORM)
                .where(
                    PredictionStepORM.step_id == step_id,
                    PredictionStepORM.observed_next_state_data
                    .is_(None),
                )
                .values(
                    observed_next_state_data=observed
                )
            )

            if result.rowcount != 1:
                raise PredictionConflictError(
                    "Observed state changed concurrently"
                )

            session.flush()
            session.expire_all()

            return self._step_record(
                session.get(
                    PredictionStepORM,
                    step_id,
                )
            )

    # --------------------------------------------------------
    # EVALUATE PREDICTION
    # --------------------------------------------------------

    def evaluate_prediction(
        self,
        prediction_id: str,
        *,
        db: Session | None = None,
    ) -> dict[str, Any]:
        with self._session(db) as session:
            prediction = self._get(
                session,
                prediction_id,
            )

            if prediction.status != (
                PredictionStatus.COMPLETED.value
            ):
                raise PredictionStateError(
                    "Only completed predictions "
                    "can be evaluated"
                )

            steps = session.scalars(
                select(PredictionStepORM)
                .where(
                    PredictionStepORM.prediction_id
                    == prediction_id,
                    PredictionStepORM.observed_next_state_data
                    .is_not(None),
                )
                .order_by(
                    PredictionStepORM.step_index.asc()
                )
            ).all()

            if not steps:
                raise PredictionValidationError(
                    "No observed states available"
                )

            predicted = np.asarray(
                [
                    step.predicted_next_state_data
                    for step in steps
                ],
                dtype=np.float64,
            )

            observed = np.asarray(
                [
                    step.observed_next_state_data
                    for step in steps
                ],
                dtype=np.float64,
            )

            errors = predicted - observed

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

            evaluation = {
                "mae": mae,
                "rmse": rmse,
                "evaluated_steps": len(steps),
                "total_steps": prediction.step_count,
                "coverage": (
                    len(steps)
                    / prediction.step_count
                ),
                "evaluated_at": iso_datetime(
                    utc_now()
                ),
            }

            prediction.evaluation_data = (
                evaluation
            )

            session.flush()

            return copy.deepcopy(evaluation)

    # --------------------------------------------------------
    # LIST PREDICTIONS
    # --------------------------------------------------------

    def list_predictions(
        self,
        *,
        world_id: str | None = None,
        model_name: str | None = None,
        status: PredictionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[PredictionRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise PredictionValidationError(
                "Invalid pagination"
            )

        statement = select(
            PredictionORM
        )

        if world_id is not None:
            statement = statement.where(
                PredictionORM.world_id
                == world_id
            )

        if model_name is not None:
            statement = statement.where(
                PredictionORM.model_name
                == model_name
            )

        if status is not None:
            statement = statement.where(
                PredictionORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                PredictionORM.created_at.desc()
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
        prediction_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[PredictionEventRecord]:
        if not 1 <= limit <= 1000:
            raise PredictionValidationError(
                "Invalid limit"
            )

        statement = (
            select(PredictionEventORM)
            .where(
                PredictionEventORM.prediction_id
                == prediction_id
            )
            .order_by(
                PredictionEventORM.created_at.asc()
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
        prediction_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            payload = {
                "prediction": asdict(
                    self.get_prediction(
                        prediction_id,
                        db=session,
                    )
                ),
                "steps": [
                    asdict(step)
                    for step in self.get_steps(
                        prediction_id,
                        limit=self.max_horizon,
                        db=session,
                    )
                ],
                "events": [
                    asdict(event)
                    for event in self.list_events(
                        prediction_id,
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
        prediction_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        steps = self.get_steps(
            prediction_id,
            limit=self.max_horizon,
            db=db,
        )

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fields = [
            "step_id",
            "prediction_id",
            "step_index",
            "state",
            "action",
            "predicted_next_state",
            "observed_next_state",
            "uncertainty",
            "reward",
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

            for step in steps:
                data = asdict(step)

                for key in (
                    "state",
                    "action",
                    "predicted_next_state",
                    "observed_next_state",
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

        return path


# ============================================================
# FACTORY
# ============================================================

def create_prediction_repository(
    **kwargs: Any,
) -> PredictionRepository:
    return PredictionRepository(
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

    repository = create_prediction_repository()

    prediction = repository.create_prediction(
        world_id="demo_navigation_world",
        prediction_type=PredictionType.FUTURE_STATE,
        model_name="demo_world_model",
        model_version="v1",
        initial_state=[
            0.0, 0.0, 0.0, 0.0
        ],
        horizon=3,
        configuration={
            "simulation_only": True,
        },
    )

    prediction = repository.start_prediction(
        prediction.prediction_id,
        expected_version=prediction.version,
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

    step_ids = []

    for index in range(3):
        step = repository.add_step(
            prediction.prediction_id,
            state=states[index],
            action=[0.2, 0.2],
            predicted_next_state=next_states[index],
            uncertainty=0.02 + index * 0.01,
            reward=-1.0,
            terminal=(index == 2),
        )

        step_ids.append(
            step.step_id
        )

    prediction = repository.get_prediction(
        prediction.prediction_id
    )

    prediction = repository.complete_prediction(
        prediction.prediction_id,
        expected_version=prediction.version,
    )

    # Illustrative observed states.
    for index, step_id in enumerate(step_ids):
        observed = [
            value + 0.01
            for value in next_states[index]
        ]

        repository.attach_observed_state(
            step_id,
            observed_next_state=observed,
        )

    evaluation = repository.evaluate_prediction(
        prediction.prediction_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - PREDICTION DATABASE")
    print("=" * 60)

    print(
        "Prediction ID:",
        prediction.prediction_id,
    )

    print(
        "Status:",
        prediction.status,
    )

    print(
        "Steps:",
        prediction.step_count,
    )

    print(
        "Evaluation:",
        evaluation,
    )

    json_path = repository.export_json(
        prediction.prediction_id,
        "experiments/reports/prediction_result.json",
    )

    csv_path = repository.export_csv(
        prediction.prediction_id,
        "experiments/reports/prediction_steps.csv",
    )

    print("JSON:", json_path)
    print("CSV:", csv_path)

    close_database()


if __name__ == "__main__":
    main()
