
"""
NeuroWorld AI
backend/app/database/observations.py

Persistent observation storage.

Features
--------
- Sensor, image, video, latent, multimodal observations
- Observation timestamps and source tracking
- World and agent associations
- JSON sensor measurements
- Latent and world-state vectors
- Quality and anomaly scores
- Media artifact references
- Observation lifecycle
- Optimistic concurrency for status changes
- Paginated observation history
- Observation statistics
- JSON and CSV export

Requires:
    pip install sqlalchemy

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Important:
    Media files are not stored in the database.
    Store validated, access-controlled media
    references instead.

    This module records observations.
    It does not run computer vision inference,
    anomaly detection or world-model prediction.
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
    func,
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

# Register the worlds table and its mappings.
from backend.app.database.world import WorldORM


# ============================================================
# EXCEPTIONS
# ============================================================

class ObservationDatabaseError(Exception):
    """Base observation database error."""


class ObservationValidationError(
    ObservationDatabaseError
):
    """Invalid observation input."""


class ObservationNotFoundError(
    ObservationDatabaseError
):
    """Observation does not exist."""


class ObservationStateError(
    ObservationDatabaseError
):
    """Invalid observation state transition."""


class ObservationConflictError(
    ObservationDatabaseError
):
    """Concurrent update or duplicate observation."""


# ============================================================
# ENUMS
# ============================================================

class ObservationType(str, Enum):
    SENSOR = "sensor"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    LATENT = "latent"
    WORLD_STATE = "world_state"
    MULTIMODAL = "multimodal"
    SIMULATION = "simulation"


class ObservationStatus(str, Enum):
    RECEIVED = "received"
    VALIDATED = "validated"
    PROCESSED = "processed"
    REJECTED = "rejected"


class ObservationSource(str, Enum):
    SENSOR = "sensor"
    CAMERA = "camera"
    VIDEO_UPLOAD = "video_upload"
    AUDIO_UPLOAD = "audio_upload"
    SIMULATOR = "simulator"
    WORLD_MODEL = "world_model"
    AGENT = "agent"
    API = "api"


# ============================================================
# UTILITIES
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


def validate_identifier(
    value: str,
    name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str):
        raise ObservationValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise ObservationValidationError(
            f"Invalid {name}"
        )

    return value


def validate_score(
    value: float | None,
    name: str,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ObservationValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ObservationValidationError(
            f"{name} must be finite"
        )

    if not 0.0 <= result <= 1.0:
        raise ObservationValidationError(
            f"{name} must be between 0 and 1"
        )

    return result


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
            raise ObservationValidationError(
                "JSON contains a non-finite value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ObservationValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise ObservationValidationError(
            f"{name} must be a dictionary"
        )

    return result


def validate_vector(
    values: Sequence[float] | None,
    *,
    name: str,
    maximum_dimension: int = 4096,
) -> list[float] | None:
    if values is None:
        return None

    if (
        not isinstance(values, (list, tuple))
        or not 1 <= len(values) <= maximum_dimension
    ):
        raise ObservationValidationError(
            f"{name} must contain between 1 "
            f"and {maximum_dimension} values"
        )

    result = []

    for item in values:
        if isinstance(item, bool):
            raise ObservationValidationError(
                f"{name} must contain numeric values"
            )

        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ObservationValidationError(
                f"{name} must contain numeric values"
            ) from exc

        if not math.isfinite(number):
            raise ObservationValidationError(
                f"{name} contains NaN or infinity"
            )

        result.append(number)

    return result


def validate_timestamp(
    value: datetime | None,
) -> datetime:
    if value is None:
        return utc_now()

    if not isinstance(value, datetime):
        raise ObservationValidationError(
            "observed_at must be a datetime"
        )

    if value.tzinfo is None:
        raise ObservationValidationError(
            "observed_at must include timezone"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# ORM MODELS
# ============================================================

class ObservationORM(Base):
    __tablename__ = "world_observations"

    __table_args__ = (
        Index(
            "ix_world_observations_world_time",
            "world_id",
            "observed_at",
        ),
        Index(
            "ix_world_observations_type_status",
            "observation_type",
            "status",
        ),
    )

    observation_id: Mapped[str] = mapped_column(
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

    agent_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    observation_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    sensor_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    state_vector: Mapped[
        list[float] | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    latent_vector: Mapped[
        list[float] | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    media_uri: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    media_sha256: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
    )

    quality_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    anomaly_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
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

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    processed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    events: Mapped[
        list["ObservationEventORM"]
    ] = relationship(
        back_populates="observation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ObservationEventORM(Base):
    __tablename__ = "observation_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    observation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "world_observations.observation_id",
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

    observation: Mapped[ObservationORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class ObservationRecord:
    observation_id: str
    world_id: str
    agent_id: str | None
    observation_type: str
    source: str
    status: str
    sensor_data: dict[str, Any]
    state_vector: list[float] | None
    latent_vector: list[float] | None
    media_uri: str | None
    media_sha256: str | None
    quality_score: float | None
    anomaly_score: float | None
    metadata: dict[str, Any]
    error_message: str | None
    version: int
    observed_at: str
    created_at: str
    processed_at: str | None


@dataclass(frozen=True)
class ObservationEvent:
    event_id: str
    observation_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ObservationStatistics:
    world_id: str
    total_observations: int
    received: int
    validated: int
    processed: int
    rejected: int
    average_quality: float | None
    average_anomaly: float | None
    by_type: dict[str, int]


# ============================================================
# OBSERVATION REPOSITORY
# ============================================================

class ObservationRepository:
    """
    Persistent repository for world observations.

    If a Session is supplied, the caller manages
    transaction commit and rollback.

    Otherwise session_scope() manages transactions.
    """

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
        observation_id: str,
    ) -> ObservationORM:
        row = session.get(
            ObservationORM,
            observation_id,
        )

        if row is None:
            raise ObservationNotFoundError(
                "Observation not found"
            )

        return row

    @staticmethod
    def _record(
        row: ObservationORM,
    ) -> ObservationRecord:
        return ObservationRecord(
            observation_id=row.observation_id,
            world_id=row.world_id,
            agent_id=row.agent_id,
            observation_type=row.observation_type,
            source=row.source,
            status=row.status,
            sensor_data=copy.deepcopy(
                row.sensor_data
            ),
            state_vector=copy.deepcopy(
                row.state_vector
            ),
            latent_vector=copy.deepcopy(
                row.latent_vector
            ),
            media_uri=row.media_uri,
            media_sha256=row.media_sha256,
            quality_score=row.quality_score,
            anomaly_score=row.anomaly_score,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            error_message=row.error_message,
            version=row.version,
            observed_at=iso_datetime(
                row.observed_at
            ),
            created_at=iso_datetime(
                row.created_at
            ),
            processed_at=iso_datetime(
                row.processed_at
            ),
        )

    @staticmethod
    def _event_record(
        row: ObservationEventORM,
    ) -> ObservationEvent:
        return ObservationEvent(
            event_id=row.event_id,
            observation_id=row.observation_id,
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
        observation_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            ObservationEventORM(
                event_id=uuid.uuid4().hex,
                observation_id=observation_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CREATE OBSERVATION
    # --------------------------------------------------------

    def create_observation(
        self,
        *,
        world_id: str,
        observation_type: ObservationType,
        source: ObservationSource,
        agent_id: str | None = None,
        sensor_data: dict[str, Any] | None = None,
        state_vector: Sequence[float] | None = None,
        latent_vector: Sequence[float] | None = None,
        media_uri: str | None = None,
        media_sha256: str | None = None,
        observed_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ObservationRecord:
        world_id = validate_identifier(
            world_id,
            "world_id",
            64,
        )

        if not isinstance(
            observation_type,
            ObservationType,
        ):
            raise ObservationValidationError(
                "Invalid observation type"
            )

        if not isinstance(
            source,
            ObservationSource,
        ):
            raise ObservationValidationError(
                "Invalid observation source"
            )

        if agent_id is not None:
            agent_id = validate_identifier(
                agent_id,
                "agent_id",
                64,
            )

        sensor_data = validate_json_dict(
            sensor_data or {},
            "sensor_data",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        state_vector = validate_vector(
            state_vector,
            name="state_vector",
        )

        latent_vector = validate_vector(
            latent_vector,
            name="latent_vector",
        )

        if media_uri is not None:
            media_uri = validate_identifier(
                media_uri,
                "media_uri",
                4096,
            )

        if media_sha256 is not None:
            import re

            media_sha256 = media_sha256.lower()

            if not re.fullmatch(
                r"[0-9a-f]{64}",
                media_sha256,
            ):
                raise ObservationValidationError(
                    "Invalid media SHA256"
                )

        observed_at = validate_timestamp(
            observed_at
        )

        with self._session(db) as session:
            world = session.get(
                WorldORM,
                world_id,
            )

            if world is None:
                raise ObservationValidationError(
                    "World does not exist"
                )

            if world.status == "archived":
                raise ObservationStateError(
                    "Cannot add observations "
                    "to an archived world"
                )

            observation_id = uuid.uuid4().hex

            row = ObservationORM(
                observation_id=observation_id,
                world_id=world_id,
                agent_id=agent_id,
                observation_type=(
                    observation_type.value
                ),
                source=source.value,
                status=(
                    ObservationStatus.RECEIVED.value
                ),
                sensor_data=sensor_data,
                state_vector=state_vector,
                latent_vector=latent_vector,
                media_uri=media_uri,
                media_sha256=media_sha256,
                metadata_data=metadata,
                version=1,
                observed_at=observed_at,
                created_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                observation_id,
                "observation_received",
                {
                    "world_id": world_id,
                    "type": observation_type.value,
                    "source": source.value,
                },
            )

            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET OBSERVATION
    # --------------------------------------------------------

    def get_observation(
        self,
        observation_id: str,
        *,
        db: Session | None = None,
    ) -> ObservationRecord:
        with self._session(db) as session:
            return self._record(
                self._get(
                    session,
                    observation_id,
                )
            )

    # --------------------------------------------------------
    # STATE TRANSITION
    # --------------------------------------------------------

    def _transition(
        self,
        session: Session,
        *,
        observation_id: str,
        expected_version: int,
        required_status: ObservationStatus,
        new_status: ObservationStatus,
        values: dict[str, Any] | None = None,
    ) -> ObservationRecord:
        result = session.execute(
            update(ObservationORM)
            .where(
                ObservationORM.observation_id
                == observation_id,
                ObservationORM.version
                == expected_version,
                ObservationORM.status
                == required_status.value,
            )
            .values(
                **(values or {}),
                status=new_status.value,
                version=expected_version + 1,
            )
        )

        if result.rowcount != 1:
            raise ObservationConflictError(
                "Observation version or status changed"
            )

        self._audit(
            session,
            observation_id,
            f"observation_{new_status.value}",
        )

        session.flush()
        session.expire_all()

        return self._record(
            self._get(
                session,
                observation_id,
            )
        )

    # --------------------------------------------------------
    # VALIDATE OBSERVATION
    # --------------------------------------------------------

    def validate_observation(
        self,
        observation_id: str,
        *,
        expected_version: int,
        quality_score: float,
        db: Session | None = None,
    ) -> ObservationRecord:
        quality_score = validate_score(
            quality_score,
            "quality_score",
        )

        with self._session(db) as session:
            return self._transition(
                session,
                observation_id=observation_id,
                expected_version=expected_version,
                required_status=ObservationStatus.RECEIVED,
                new_status=ObservationStatus.VALIDATED,
                values={
                    "quality_score": quality_score,
                },
            )

    # --------------------------------------------------------
    # PROCESS OBSERVATION
    # --------------------------------------------------------

    def process_observation(
        self,
        observation_id: str,
        *,
        expected_version: int,
        anomaly_score: float | None = None,
        state_vector: Sequence[float] | None = None,
        latent_vector: Sequence[float] | None = None,
        db: Session | None = None,
    ) -> ObservationRecord:
        anomaly_score = validate_score(
            anomaly_score,
            "anomaly_score",
        )

        values: dict[str, Any] = {
            "processed_at": utc_now(),
        }

        if anomaly_score is not None:
            values["anomaly_score"] = (
                anomaly_score
            )

        if state_vector is not None:
            values["state_vector"] = (
                validate_vector(
                    state_vector,
                    name="state_vector",
                )
            )

        if latent_vector is not None:
            values["latent_vector"] = (
                validate_vector(
                    latent_vector,
                    name="latent_vector",
                )
            )

        with self._session(db) as session:
            return self._transition(
                session,
                observation_id=observation_id,
                expected_version=expected_version,
                required_status=ObservationStatus.VALIDATED,
                new_status=ObservationStatus.PROCESSED,
                values=values,
            )

    # --------------------------------------------------------
    # REJECT OBSERVATION
    # --------------------------------------------------------

    def reject_observation(
        self,
        observation_id: str,
        *,
        expected_version: int,
        reason: str,
        db: Session | None = None,
    ) -> ObservationRecord:
        reason = validate_identifier(
            reason,
            "reason",
            4000,
        )

        with self._session(db) as session:
            row = self._get(
                session,
                observation_id,
            )

            if row.status not in {
                ObservationStatus.RECEIVED.value,
                ObservationStatus.VALIDATED.value,
            }:
                raise ObservationStateError(
                    "Only received or validated "
                    "observations can be rejected"
                )

            result = session.execute(
                update(ObservationORM)
                .where(
                    ObservationORM.observation_id
                    == observation_id,
                    ObservationORM.version
                    == expected_version,
                    ObservationORM.status
                    == row.status,
                )
                .values(
                    status=ObservationStatus.REJECTED.value,
                    error_message=reason,
                    processed_at=utc_now(),
                    version=expected_version + 1,
                )
            )

            if result.rowcount != 1:
                raise ObservationConflictError(
                    "Observation changed concurrently"
                )

            self._audit(
                session,
                observation_id,
                "observation_rejected",
                {"reason": reason},
            )

            session.flush()
            session.expire_all()

            return self._record(
                self._get(
                    session,
                    observation_id,
                )
            )

    # --------------------------------------------------------
    # LIST OBSERVATIONS
    # --------------------------------------------------------

    def list_observations(
        self,
        *,
        world_id: str | None = None,
        observation_type: ObservationType | None = None,
        status: ObservationStatus | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[ObservationRecord]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ObservationValidationError(
                "Invalid pagination"
            )

        statement = select(
            ObservationORM
        )

        if world_id is not None:
            statement = statement.where(
                ObservationORM.world_id
                == world_id
            )

        if observation_type is not None:
            statement = statement.where(
                ObservationORM.observation_type
                == observation_type.value
            )

        if status is not None:
            statement = statement.where(
                ObservationORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                ObservationORM.observed_at.desc(),
                ObservationORM.observation_id.desc(),
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
    # OBSERVATION EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        observation_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[ObservationEvent]:
        if not 1 <= limit <= 1000:
            raise ObservationValidationError(
                "Invalid limit"
            )

        statement = (
            select(ObservationEventORM)
            .where(
                ObservationEventORM.observation_id
                == observation_id
            )
            .order_by(
                ObservationEventORM.created_at.asc(),
                ObservationEventORM.event_id.asc(),
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
    # WORLD STATISTICS
    # --------------------------------------------------------

    def get_world_statistics(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> ObservationStatistics:
        with self._session(db) as session:
            self._ensure_world_exists(
                session,
                world_id,
            )

            rows = session.execute(
                select(
                    ObservationORM.status,
                    ObservationORM.observation_type,
                    ObservationORM.quality_score,
                    ObservationORM.anomaly_score,
                ).where(
                    ObservationORM.world_id
                    == world_id
                )
            ).all()

        statuses = {
            item.value: 0
            for item in ObservationStatus
        }

        by_type: dict[str, int] = {}
        quality_values: list[float] = []
        anomaly_values: list[float] = []

        for row in rows:
            statuses[row.status] = (
                statuses.get(row.status, 0) + 1
            )

            by_type[row.observation_type] = (
                by_type.get(
                    row.observation_type,
                    0,
                ) + 1
            )

            if row.quality_score is not None:
                quality_values.append(
                    row.quality_score
                )

            if row.anomaly_score is not None:
                anomaly_values.append(
                    row.anomaly_score
                )

        return ObservationStatistics(
            world_id=world_id,
            total_observations=len(rows),
            received=statuses["received"],
            validated=statuses["validated"],
            processed=statuses["processed"],
            rejected=statuses["rejected"],
            average_quality=(
                sum(quality_values)
                / len(quality_values)
                if quality_values
                else None
            ),
            average_anomaly=(
                sum(anomaly_values)
                / len(anomaly_values)
                if anomaly_values
                else None
            ),
            by_type=by_type,
        )

    @staticmethod
    def _ensure_world_exists(
        session: Session,
        world_id: str,
    ) -> None:
        if session.get(WorldORM, world_id) is None:
            raise ObservationValidationError(
                "World does not exist"
            )

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        world_id: str,
        output_path: str | Path,
        *,
        limit: int = 1000,
        db: Session | None = None,
    ) -> Path:
        observations = self.list_observations(
            world_id=world_id,
            limit=limit,
            db=db,
        )

        payload = {
            "world_id": world_id,
            "exported_at": iso_datetime(
                utc_now()
            ),
            "observations": [
                asdict(item)
                for item in observations
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
        world_id: str,
        output_path: str | Path,
        *,
        limit: int = 1000,
        db: Session | None = None,
    ) -> Path:
        observations = self.list_observations(
            world_id=world_id,
            limit=limit,
            db=db,
        )

        path = Path(output_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "observation_id",
            "world_id",
            "agent_id",
            "observation_type",
            "source",
            "status",
            "quality_score",
            "anomaly_score",
            "media_uri",
            "observed_at",
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

            for observation in observations:
                row = asdict(observation)

                writer.writerow(
                    {
                        key: row.get(key)
                        for key in fieldnames
                    }
                )

        return path


# ============================================================
# FACTORY
# ============================================================

def create_observation_repository() -> (
    ObservationRepository
):
    return ObservationRepository()


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

    create_database_tables()

    worlds = create_world_repository()
    repository = create_observation_repository()

    world = worlds.create_world(
        name="Observation Demo World",
        environment_type="synthetic_navigation",
        initial_state={
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
        },
    )

    observation = repository.create_observation(
        world_id=world.world_id,
        observation_type=ObservationType.SENSOR,
        source=ObservationSource.SIMULATOR,
        sensor_data={
            "position_x": 1.2,
            "position_y": 0.8,
            "velocity_x": 0.1,
            "velocity_y": 0.2,
        },
        state_vector=[
            1.2,
            0.8,
            0.1,
            0.2,
        ],
        metadata={
            "simulation_only": True,
            "sensor_name": "demo_navigation_sensor",
        },
    )

    observation = repository.validate_observation(
        observation.observation_id,
        expected_version=observation.version,
        quality_score=0.98,
    )

    observation = repository.process_observation(
        observation.observation_id,
        expected_version=observation.version,
        anomaly_score=0.05,
        latent_vector=[
            0.15,
            0.42,
            0.71,
            0.33,
        ],
    )

    statistics = repository.get_world_statistics(
        world.world_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - OBSERVATIONS")
    print("=" * 60)

    print(
        "Observation ID:",
        observation.observation_id,
    )

    print(
        "World ID:",
        observation.world_id,
    )

    print(
        "Status:",
        observation.status,
    )

    print(
        "Quality:",
        observation.quality_score,
    )

    print(
        "Anomaly:",
        observation.anomaly_score,
    )

    print(
        "Statistics:",
        asdict(statistics),
    )

    json_path = repository.export_json(
        world.world_id,
        "experiments/reports/observations.json",
    )

    csv_path = repository.export_csv(
        world.world_id,
        "experiments/reports/observations.csv",
    )

    print("JSON:", json_path)
    print("CSV:", csv_path)

    close_database()


if __name__ == "__main__":
    main()
