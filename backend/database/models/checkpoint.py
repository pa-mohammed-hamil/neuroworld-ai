
"""
NeuroWorld AI
backend/app/database/checkpoint.py

Persistent World-State Checkpoint Management

Features:
- Save world-state checkpoints
- Restore checkpoints
- List checkpoint history
- Roll back to a previous state
- Store state, metadata and model version
- SHA-256 integrity verification
- Optional checkpoint expiration
- Per-world checkpoint retention
- Transaction-safe database operations
- PostgreSQL and SQLite support

Dependencies:
    pip install sqlalchemy numpy

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module stores structured world-state data.

For large PyTorch model weights, use a separate
artifact/checkpoint storage system rather than
placing large binary tensors in database rows.

Important:
    Restoring a checkpoint returns a verified
    snapshot. Applying it to an active world
    requires an explicit world-service operation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import numpy as np

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    delete,
    func,
    select,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.app.database.session import (
    Base,
    session_scope,
)


logger = logging.getLogger(
    "neuroworld.database.checkpoint"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class CheckpointError(Exception):
    """Base checkpoint exception."""


class CheckpointValidationError(CheckpointError):
    """Invalid checkpoint input."""


class CheckpointNotFoundError(CheckpointError):
    """Checkpoint does not exist."""


class CheckpointIntegrityError(CheckpointError):
    """Checkpoint integrity verification failed."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None

    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


def to_json_compatible(
    value: Any,
) -> Any:
    """
    Convert supported NumPy values into
    JSON-compatible Python objects.

    Reject non-finite numbers.
    """

    if isinstance(value, np.ndarray):
        return to_json_compatible(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return to_json_compatible(
            value.item()
        )

    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, float):
        if not np.isfinite(value):
            raise CheckpointValidationError(
                "Non-finite numbers are not allowed"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise CheckpointValidationError(
        f"Unsupported checkpoint value: "
        f"{type(value).__name__}"
    )


def canonical_json(
    value: Any,
) -> str:
    normalized = to_json_compatible(value)

    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(
            "Unable to serialize checkpoint"
        ) from exc


def calculate_checksum(
    payload: Any,
) -> str:
    serialized = canonical_json(
        payload
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise CheckpointValidationError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise CheckpointValidationError(
            f"Invalid {field_name}"
        )

    return value


# ============================================================
# ORM MODEL
# ============================================================

class WorldCheckpoint(Base):
    """
    Persistent world-state snapshot.

    This table is independent of a particular
    World ORM table to avoid circular imports.

    world_id is a logical identifier.
    """

    __tablename__ = "world_checkpoints"

    __table_args__ = (
        Index(
            "ix_world_checkpoints_world_created",
            "world_id",
            "created_at",
        ),
    )

    checkpoint_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    world_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    checkpoint_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    state_data: Mapped[Any] = mapped_column(
        JSON,
        nullable=False,
    )

    metadata_data: Mapped[Any] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    checksum: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# ============================================================
# PUBLIC RESULT MODEL
# ============================================================

@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    world_id: str

    checkpoint_name: str
    model_version: str

    state: Any
    metadata: dict[str, Any]

    checksum: str

    created_at: str
    expires_at: str | None


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """
    Database-backed world-state checkpoint manager.

    Each public operation may receive an existing
    SQLAlchemy Session.

    If no Session is provided, a managed transaction
    is created automatically.

    Caller-owned sessions are never committed
    or closed by this manager.
    """

    def __init__(
        self,
        *,
        max_checkpoints_per_world: int = 100,
        max_payload_bytes: int = 2_000_000,
    ) -> None:
        if max_checkpoints_per_world < 1:
            raise ValueError(
                "max_checkpoints_per_world must be positive"
            )

        if max_payload_bytes < 1:
            raise ValueError(
                "max_payload_bytes must be positive"
            )

        self.max_checkpoints_per_world = (
            max_checkpoints_per_world
        )

        self.max_payload_bytes = (
            max_payload_bytes
        )

    # --------------------------------------------------------
    # SESSION HANDLING
    # --------------------------------------------------------

    def _session(
        self,
        db: Session | None,
    ):
        """
        Return a context manager that uses
        the provided session without closing it.
        """

        if db is None:
            return session_scope()

        from contextlib import nullcontext

        return nullcontext(db)

    # --------------------------------------------------------
    # VALIDATE PAYLOAD
    # --------------------------------------------------------

    def _prepare_payload(
        self,
        state: Any,
        metadata: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], str]:
        normalized_state = (
            to_json_compatible(state)
        )

        normalized_metadata = (
            to_json_compatible(metadata)
        )

        if not isinstance(
            normalized_metadata,
            dict,
        ):
            raise CheckpointValidationError(
                "metadata must be a dictionary"
            )

        payload = {
            "state": normalized_state,
            "metadata": normalized_metadata,
        }

        serialized = canonical_json(
            payload
        )

        if len(
            serialized.encode("utf-8")
        ) > self.max_payload_bytes:
            raise CheckpointValidationError(
                "Checkpoint payload exceeds size limit"
            )

        checksum = calculate_checksum(
            payload
        )

        return (
            normalized_state,
            normalized_metadata,
            checksum,
        )

    # --------------------------------------------------------
    # ORM TO RESULT
    # --------------------------------------------------------

    @staticmethod
    def _to_record(
        checkpoint: WorldCheckpoint,
    ) -> CheckpointRecord:
        created = normalize_datetime(
            checkpoint.created_at
        )

        expires = normalize_datetime(
            checkpoint.expires_at
        )

        return CheckpointRecord(
            checkpoint_id=checkpoint.checkpoint_id,
            world_id=checkpoint.world_id,
            checkpoint_name=checkpoint.checkpoint_name,
            model_version=checkpoint.model_version,
            state=checkpoint.state_data,
            metadata=checkpoint.metadata_data,
            checksum=checkpoint.checksum,
            created_at=created.isoformat(),
            expires_at=(
                expires.isoformat()
                if expires is not None
                else None
            ),
        )

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    def save_checkpoint(
        self,
        *,
        world_id: str,
        state: Any,
        checkpoint_name: str = "automatic",
        model_version: str = "unknown",
        metadata: dict[str, Any] | None = None,
        expires_in_seconds: int | None = None,
        db: Session | None = None,
    ) -> CheckpointRecord:
        world_id = validate_identifier(
            world_id,
            "world_id",
        )

        checkpoint_name = (
            checkpoint_name.strip()
        )

        if not checkpoint_name:
            raise CheckpointValidationError(
                "checkpoint_name cannot be empty"
            )

        if len(checkpoint_name) > 255:
            raise CheckpointValidationError(
                "checkpoint_name is too long"
            )

        model_version = validate_identifier(
            model_version,
            "model_version",
        )

        if (
            expires_in_seconds is not None
            and expires_in_seconds <= 0
        ):
            raise CheckpointValidationError(
                "Expiration must be positive"
            )

        normalized_state, normalized_metadata, checksum = (
            self._prepare_payload(
                state,
                metadata or {},
            )
        )

        now = utc_now()

        expires_at = (
            now + timedelta(
                seconds=expires_in_seconds
            )
            if expires_in_seconds is not None
            else None
        )

        checkpoint = WorldCheckpoint(
            checkpoint_id=uuid.uuid4().hex,
            world_id=world_id,
            checkpoint_name=checkpoint_name,
            model_version=model_version,
            state_data=normalized_state,
            metadata_data=normalized_metadata,
            checksum=checksum,
            created_at=now,
            expires_at=expires_at,
        )

        with self._session(db) as session:
            session.add(checkpoint)
            session.flush()

            self._enforce_retention(
                session,
                world_id,
            )

            record = self._to_record(
                checkpoint
            )

        logger.info(
            "Checkpoint saved: world=%s checkpoint=%s",
            world_id,
            record.checkpoint_id,
        )

        return record

    # --------------------------------------------------------
    # GET CHECKPOINT
    # --------------------------------------------------------

    def get_checkpoint(
        self,
        checkpoint_id: str,
        *,
        world_id: str | None = None,
        include_expired: bool = False,
        db: Session | None = None,
    ) -> CheckpointRecord:
        checkpoint_id = validate_identifier(
            checkpoint_id,
            "checkpoint_id",
        )

        with self._session(db) as session:
            statement = select(
                WorldCheckpoint
            ).where(
                WorldCheckpoint.checkpoint_id
                == checkpoint_id
            )

            if world_id is not None:
                statement = statement.where(
                    WorldCheckpoint.world_id
                    == validate_identifier(
                        world_id,
                        "world_id",
                    )
                )

            checkpoint = session.scalar(
                statement
            )

            if checkpoint is None:
                raise CheckpointNotFoundError(
                    "Checkpoint not found"
                )

            expires = normalize_datetime(
                checkpoint.expires_at
            )

            if (
                not include_expired
                and expires is not None
                and expires <= utc_now()
            ):
                raise CheckpointNotFoundError(
                    "Checkpoint has expired"
                )

            self._verify_integrity(
                checkpoint
            )

            return self._to_record(
                checkpoint
            )

    # --------------------------------------------------------
    # VERIFY INTEGRITY
    # --------------------------------------------------------

    @staticmethod
    def _verify_integrity(
        checkpoint: WorldCheckpoint,
    ) -> None:
        payload = {
            "state": checkpoint.state_data,
            "metadata": checkpoint.metadata_data,
        }

        actual_checksum = calculate_checksum(
            payload
        )

        if actual_checksum != checkpoint.checksum:
            raise CheckpointIntegrityError(
                "Checkpoint checksum mismatch"
            )

    def verify_checkpoint(
        self,
        checkpoint_id: str,
        *,
        world_id: str | None = None,
        db: Session | None = None,
    ) -> bool:
        self.get_checkpoint(
            checkpoint_id,
            world_id=world_id,
            db=db,
        )

        return True

    # --------------------------------------------------------
    # RESTORE CHECKPOINT
    # --------------------------------------------------------

    def restore_checkpoint(
        self,
        checkpoint_id: str,
        *,
        world_id: str | None = None,
        db: Session | None = None,
    ) -> Any:
        """
        Return a verified copy of stored state.

        Does not mutate the live world.
        """

        record = self.get_checkpoint(
            checkpoint_id,
            world_id=world_id,
            db=db,
        )

        # JSON round-trip prevents callers from
        # mutating the ORM-owned object.
        return json.loads(
            canonical_json(
                record.state
            )
        )

    # --------------------------------------------------------
    # LIST CHECKPOINTS
    # --------------------------------------------------------

    def list_checkpoints(
        self,
        world_id: str,
        *,
        limit: int = 100,
        include_expired: bool = False,
        db: Session | None = None,
    ) -> list[CheckpointRecord]:
        world_id = validate_identifier(
            world_id,
            "world_id",
        )

        if not 1 <= limit <= 1000:
            raise CheckpointValidationError(
                "limit must be between 1 and 1000"
            )

        with self._session(db) as session:
            statement = (
                select(WorldCheckpoint)
                .where(
                    WorldCheckpoint.world_id
                    == world_id
                )
                .order_by(
                    WorldCheckpoint.created_at.desc(),
                    WorldCheckpoint.checkpoint_id.desc(),
                )
                .limit(limit)
            )

            checkpoints = session.scalars(
                statement
            ).all()

            records = []

            for checkpoint in checkpoints:
                expires = normalize_datetime(
                    checkpoint.expires_at
                )

                if (
                    not include_expired
                    and expires is not None
                    and expires <= utc_now()
                ):
                    continue

                records.append(
                    self._to_record(
                        checkpoint
                    )
                )

            return records

    # --------------------------------------------------------
    # LATEST CHECKPOINT
    # --------------------------------------------------------

    def get_latest_checkpoint(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> CheckpointRecord | None:
        records = self.list_checkpoints(
            world_id,
            limit=self.max_checkpoints_per_world,
            db=db,
        )

        if not records:
            return None

        latest = records[0]

        return self.get_checkpoint(
            latest.checkpoint_id,
            world_id=world_id,
            db=db,
        )

    # --------------------------------------------------------
    # DELETE CHECKPOINT
    # --------------------------------------------------------

    def delete_checkpoint(
        self,
        checkpoint_id: str,
        *,
        world_id: str | None = None,
        db: Session | None = None,
    ) -> bool:
        checkpoint_id = validate_identifier(
            checkpoint_id,
            "checkpoint_id",
        )

        with self._session(db) as session:
            statement = delete(
                WorldCheckpoint
            ).where(
                WorldCheckpoint.checkpoint_id
                == checkpoint_id
            )

            if world_id is not None:
                statement = statement.where(
                    WorldCheckpoint.world_id
                    == validate_identifier(
                        world_id,
                        "world_id",
                    )
                )

            result = session.execute(
                statement
            )

            return result.rowcount > 0

    # --------------------------------------------------------
    # RETENTION
    # --------------------------------------------------------

    def _enforce_retention(
        self,
        db: Session,
        world_id: str,
    ) -> None:
        statement = (
            select(
                WorldCheckpoint.checkpoint_id
            )
            .where(
                WorldCheckpoint.world_id
                == world_id
            )
            .order_by(
                WorldCheckpoint.created_at.desc(),
                WorldCheckpoint.checkpoint_id.desc(),
            )
        )

        ids = list(
            db.scalars(
                statement
            ).all()
        )

        obsolete_ids = ids[
            self.max_checkpoints_per_world:
        ]

        if obsolete_ids:
            db.execute(
                delete(WorldCheckpoint).where(
                    WorldCheckpoint.checkpoint_id.in_(
                        obsolete_ids
                    )
                )
            )

    # --------------------------------------------------------
    # CLEAN EXPIRED CHECKPOINTS
    # --------------------------------------------------------

    def cleanup_expired(
        self,
        *,
        db: Session | None = None,
    ) -> int:
        now = utc_now()

        with self._session(db) as session:
            result = session.execute(
                delete(WorldCheckpoint).where(
                    WorldCheckpoint.expires_at.is_not(None),
                    WorldCheckpoint.expires_at <= now,
                )
            )

            return int(
                result.rowcount or 0
            )

    # --------------------------------------------------------
    # COUNT CHECKPOINTS
    # --------------------------------------------------------

    def count_checkpoints(
        self,
        world_id: str,
        *,
        db: Session | None = None,
    ) -> int:
        world_id = validate_identifier(
            world_id,
            "world_id",
        )

        with self._session(db) as session:
            result = session.scalar(
                select(
                    func.count(
                        WorldCheckpoint.checkpoint_id
                    )
                ).where(
                    WorldCheckpoint.world_id
                    == world_id
                )
            )

            return int(
                result or 0
            )


# ============================================================
# FACTORY
# ============================================================

def create_checkpoint_manager(
    *,
    max_checkpoints_per_world: int = 100,
    max_payload_bytes: int = 2_000_000,
) -> CheckpointManager:
    return CheckpointManager(
        max_checkpoints_per_world=(
            max_checkpoints_per_world
        ),
        max_payload_bytes=max_payload_bytes,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    """
    Development-only demonstration.

    The checkpoint table must exist before
    saving records.
    """

    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    logging.basicConfig(
        level=logging.INFO
    )

    create_database_tables()

    manager = create_checkpoint_manager()

    world_id = "demo_world_001"

    state = {
        "position": [1.5, 2.0],
        "velocity": [0.2, -0.1],
        "objects": [
            {
                "object_id": "robot_001",
                "position": [1.5, 2.0],
                "visible": True,
            }
        ],
        "simulation_step": 42,
    }

    checkpoint = manager.save_checkpoint(
        world_id=world_id,
        state=state,
        checkpoint_name="before_simulation",
        model_version="world_model_v1",
        metadata={
            "experiment": "navigation_test",
            "description": (
                "Checkpoint before next simulation"
            ),
        },
    )

    print("=" * 60)
    print("NEUROWORLD AI - CHECKPOINT MANAGER")
    print("=" * 60)

    print(
        "Checkpoint ID:",
        checkpoint.checkpoint_id,
    )

    print(
        "Integrity:",
        manager.verify_checkpoint(
            checkpoint.checkpoint_id,
            world_id=world_id,
        ),
    )

    restored_state = (
        manager.restore_checkpoint(
            checkpoint.checkpoint_id,
            world_id=world_id,
        )
    )

    print(
        "Restored state:",
        json.dumps(
            restored_state,
            indent=2,
        ),
    )

    print(
        "Checkpoint count:",
        manager.count_checkpoints(
            world_id
        ),
    )

    close_database()


if __name__ == "__main__":
    main()
