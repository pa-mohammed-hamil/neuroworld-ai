
"""
NeuroWorld AI
backend/app/database/models.py

Persistent AI Model Registry.

Features:
- Register neural world models
- Store model versions and configurations
- Track model lifecycle and deployment stages
- Register checkpoint artifact references
- Attach evaluation references
- Track training experiment references
- Store model metadata and architecture details
- Promote models between stages
- Record audit history
- List and search registered models
- Export model registry reports

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Important:
    This module stores model metadata and artifact
    references. It does not load neural network
    weights, train models or deploy services.
"""

from __future__ import annotations

import copy
import json
import math
import re
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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

class ModelRegistryError(Exception):
    """Base model registry exception."""


class ModelValidationError(ModelRegistryError):
    """Invalid model registration data."""


class ModelNotFoundError(ModelRegistryError):
    """Model or version not found."""


class ModelStateError(ModelRegistryError):
    """Invalid model lifecycle transition."""


class ModelConflictError(ModelRegistryError):
    """Duplicate model or concurrent modification."""


# ============================================================
# ENUMS
# ============================================================

class ModelType(str, Enum):
    TRANSFORMER_WORLD_MODEL = (
        "transformer_world_model"
    )
    DIFFUSION_WORLD_MODEL = (
        "diffusion_world_model"
    )
    AUTOREGRESSIVE_WORLD_MODEL = (
        "autoregressive_world_model"
    )
    ACTION_CONDITIONED_MODEL = (
        "action_conditioned_model"
    )
    DYNAMICS_MODEL = "dynamics_model"
    VIDEO_PREDICTION = "video_prediction"
    TOKENIZER = "tokenizer"
    VQVAE = "vqvae"
    REWARD_MODEL = "reward_model"
    POLICY = "policy"
    VALUE_MODEL = "value_model"
    PERCEPTION_MODEL = "perception_model"
    CUSTOM = "custom"


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(str, Enum):
    REGISTERED = "registered"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    DISABLED = "disabled"


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
    name: str,
    maximum: int = 255,
) -> str:
    if not isinstance(value, str):
        raise ModelValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise ModelValidationError(
            f"Invalid {name}"
        )

    return value


def validate_version(value: str) -> str:
    value = validate_string(
        value,
        "version",
        128,
    )

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._+\-]*",
        value,
    ):
        raise ModelValidationError(
            "Invalid version format"
        )

    return value


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
            raise ModelValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ModelValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


def validate_json_dict(
    value: Any,
    name: str,
) -> dict[str, Any]:
    result = json_safe(value)

    if not isinstance(result, dict):
        raise ModelValidationError(
            f"{name} must be a dictionary"
        )

    return result


def validate_sha256(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    value = value.lower()

    if not re.fullmatch(
        r"[0-9a-f]{64}",
        value,
    ):
        raise ModelValidationError(
            "Invalid SHA256 checksum"
        )

    return value


# ============================================================
# ORM MODELS
# ============================================================

class RegisteredModelORM(Base):
    __tablename__ = "registered_ai_models"

    __table_args__ = (
        UniqueConstraint(
            "name",
            name="uq_registered_ai_model_name",
        ),
    )

    model_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
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

    versions: Mapped[
        list["ModelVersionORM"]
    ] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ModelVersionORM(Base):
    __tablename__ = "ai_model_versions"

    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "version",
            name="uq_ai_model_version",
        ),
        Index(
            "ix_ai_model_versions_stage_status",
            "stage",
            "status",
        ),
    )

    model_version_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    model_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "registered_ai_models.model_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    architecture_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "architecture",
        JSON,
        nullable=False,
        default=dict,
    )

    configuration_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "configuration",
        JSON,
        nullable=False,
        default=dict,
    )

    metrics_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "metrics",
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

    checkpoint_uri: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    checkpoint_sha256: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
    )

    experiment_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    evaluation_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    revision: Mapped[int] = mapped_column(
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

    model: Mapped[RegisteredModelORM] = relationship(
        back_populates="versions",
    )

    events: Mapped[
        list["ModelRegistryEventORM"]
    ] = relationship(
        back_populates="model_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ModelRegistryEventORM(Base):
    __tablename__ = "ai_model_registry_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    model_version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_model_versions.model_version_id",
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

    model_version: Mapped[ModelVersionORM] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class RegisteredModel:
    model_id: str
    name: str
    model_type: str
    description: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ModelVersion:
    model_version_id: str
    model_id: str
    version: str
    stage: str
    status: str
    architecture: dict[str, Any]
    configuration: dict[str, Any]
    metrics: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint_uri: str | None
    checkpoint_sha256: str | None
    experiment_id: str | None
    evaluation_id: str | None
    revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ModelRegistryEvent:
    event_id: str
    model_version_id: str
    event_type: str
    details: dict[str, Any]
    created_at: str


# ============================================================
# MODEL REGISTRY REPOSITORY
# ============================================================

class ModelRegistryRepository:
    """
    Persistent registry for model metadata.

    A deployment stage is a registry label.
    Promoting a version does not automatically
    deploy it to an inference server.

    The service layer must enforce authorization,
    artifact verification and deployment approval.
    """

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _model_record(
        row: RegisteredModelORM,
    ) -> RegisteredModel:
        return RegisteredModel(
            model_id=row.model_id,
            name=row.name,
            model_type=row.model_type,
            description=row.description,
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _version_record(
        row: ModelVersionORM,
    ) -> ModelVersion:
        return ModelVersion(
            model_version_id=row.model_version_id,
            model_id=row.model_id,
            version=row.version,
            stage=row.stage,
            status=row.status,
            architecture=copy.deepcopy(
                row.architecture_data
            ),
            configuration=copy.deepcopy(
                row.configuration_data
            ),
            metrics=copy.deepcopy(
                row.metrics_data
            ),
            metadata=copy.deepcopy(
                row.metadata_data
            ),
            checkpoint_uri=row.checkpoint_uri,
            checkpoint_sha256=row.checkpoint_sha256,
            experiment_id=row.experiment_id,
            evaluation_id=row.evaluation_id,
            revision=row.revision,
            created_at=iso_datetime(
                row.created_at
            ),
            updated_at=iso_datetime(
                row.updated_at
            ),
        )

    @staticmethod
    def _event_record(
        row: ModelRegistryEventORM,
    ) -> ModelRegistryEvent:
        return ModelRegistryEvent(
            event_id=row.event_id,
            model_version_id=row.model_version_id,
            event_type=row.event_type,
            details=copy.deepcopy(
                row.details_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _get_model(
        session: Session,
        model_id: str,
    ) -> RegisteredModelORM:
        row = session.get(
            RegisteredModelORM,
            model_id,
        )

        if row is None:
            raise ModelNotFoundError(
                "Model not found"
            )

        return row

    @staticmethod
    def _get_version(
        session: Session,
        model_version_id: str,
    ) -> ModelVersionORM:
        row = session.get(
            ModelVersionORM,
            model_version_id,
        )

        if row is None:
            raise ModelNotFoundError(
                "Model version not found"
            )

        return row

    @staticmethod
    def _audit(
        session: Session,
        *,
        model_version_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            ModelRegistryEventORM(
                event_id=uuid.uuid4().hex,
                model_version_id=model_version_id,
                event_type=event_type,
                details_data=validate_json_dict(
                    details or {},
                    "event details",
                ),
                created_at=utc_now(),
            )
        )

    # --------------------------------------------------------
    # REGISTER MODEL
    # --------------------------------------------------------

    def register_model(
        self,
        *,
        name: str,
        model_type: ModelType,
        description: str = "",
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> RegisteredModel:
        name = validate_string(
            name,
            "model name",
        )

        if not isinstance(model_type, ModelType):
            raise ModelValidationError(
                "Invalid model type"
            )

        if not isinstance(description, str):
            raise ModelValidationError(
                "description must be a string"
            )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            existing = session.scalar(
                select(RegisteredModelORM.model_id)
                .where(
                    RegisteredModelORM.name == name
                )
            )

            if existing is not None:
                raise ModelConflictError(
                    "Model name already registered"
                )

            row = RegisteredModelORM(
                model_id=uuid.uuid4().hex,
                name=name,
                model_type=model_type.value,
                description=description,
                metadata_data=metadata,
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._model_record(row)

    # --------------------------------------------------------
    # GET MODEL
    # --------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        *,
        db: Session | None = None,
    ) -> RegisteredModel:
        with self._session(db) as session:
            return self._model_record(
                self._get_model(
                    session,
                    model_id,
                )
            )

    # --------------------------------------------------------
    # REGISTER VERSION
    # --------------------------------------------------------

    def register_version(
        self,
        *,
        model_id: str,
        version: str,
        architecture: dict[str, Any],
        configuration: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ModelVersion:
        version = validate_version(
            version
        )

        architecture = validate_json_dict(
            architecture,
            "architecture",
        )

        configuration = validate_json_dict(
            configuration,
            "configuration",
        )

        metadata = validate_json_dict(
            metadata or {},
            "metadata",
        )

        with self._session(db) as session:
            self._get_model(
                session,
                model_id,
            )

            existing = session.scalar(
                select(ModelVersionORM.model_version_id)
                .where(
                    ModelVersionORM.model_id == model_id,
                    ModelVersionORM.version == version,
                )
            )

            if existing is not None:
                raise ModelConflictError(
                    "Model version already exists"
                )

            row = ModelVersionORM(
                model_version_id=uuid.uuid4().hex,
                model_id=model_id,
                version=version,
                stage=ModelStage.DEVELOPMENT.value,
                status=ModelStatus.REGISTERED.value,
                architecture_data=architecture,
                configuration_data=configuration,
                metrics_data={},
                metadata_data=metadata,
                revision=1,
                created_at=utc_now(),
                updated_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                model_version_id=row.model_version_id,
                event_type="version_registered",
                details={
                    "version": version,
                },
            )

            session.flush()

            return self._version_record(row)

    # --------------------------------------------------------
    # GET VERSION
    # --------------------------------------------------------

    def get_version(
        self,
        model_version_id: str,
        *,
        db: Session | None = None,
    ) -> ModelVersion:
        with self._session(db) as session:
            return self._version_record(
                self._get_version(
                    session,
                    model_version_id,
                )
            )

    # --------------------------------------------------------
    # UPDATE VERSION STATUS
    # --------------------------------------------------------

    def update_status(
        self,
        model_version_id: str,
        *,
        status: ModelStatus,
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersion:
        if not isinstance(status, ModelStatus):
            raise ModelValidationError(
                "Invalid model status"
            )

        allowed = {
            ModelStatus.REGISTERED: {
                ModelStatus.TRAINING,
                ModelStatus.DISABLED,
            },
            ModelStatus.TRAINING: {
                ModelStatus.READY,
                ModelStatus.FAILED,
                ModelStatus.DISABLED,
            },
            ModelStatus.READY: {
                ModelStatus.TRAINING,
                ModelStatus.DISABLED,
            },
            ModelStatus.FAILED: {
                ModelStatus.TRAINING,
                ModelStatus.DISABLED,
            },
            ModelStatus.DISABLED: set(),
        }

        with self._session(db) as session:
            row = self._get_version(
                session,
                model_version_id,
            )

            current = ModelStatus(
                row.status
            )

            if status not in allowed[current]:
                raise ModelStateError(
                    f"Invalid status transition: "
                    f"{current.value} -> {status.value}"
                )

            result = session.execute(
                update(ModelVersionORM)
                .where(
                    ModelVersionORM.model_version_id
                    == model_version_id,
                    ModelVersionORM.revision
                    == expected_revision,
                    ModelVersionORM.status
                    == current.value,
                )
                .values(
                    status=status.value,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model version changed concurrently"
                )

            self._audit(
                session,
                model_version_id=model_version_id,
                event_type="status_changed",
                details={
                    "from": current.value,
                    "to": status.value,
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    model_version_id,
                )
            )

    # --------------------------------------------------------
    # ATTACH CHECKPOINT
    # --------------------------------------------------------

    def attach_checkpoint(
        self,
        model_version_id: str,
        *,
        checkpoint_uri: str,
        checkpoint_sha256: str | None = None,
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersion:
        checkpoint_uri = validate_string(
            checkpoint_uri,
            "checkpoint_uri",
            4096,
        )

        checkpoint_sha256 = validate_sha256(
            checkpoint_sha256
        )

        with self._session(db) as session:
            result = session.execute(
                update(ModelVersionORM)
                .where(
                    ModelVersionORM.model_version_id
                    == model_version_id,
                    ModelVersionORM.revision
                    == expected_revision,
                    ModelVersionORM.status
                    != ModelStatus.DISABLED.value,
                )
                .values(
                    checkpoint_uri=checkpoint_uri,
                    checkpoint_sha256=checkpoint_sha256,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model version missing, disabled "
                    "or changed concurrently"
                )

            self._audit(
                session,
                model_version_id=model_version_id,
                event_type="checkpoint_attached",
                details={
                    "checkpoint_uri": checkpoint_uri,
                    "checksum_provided": (
                        checkpoint_sha256 is not None
                    ),
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    model_version_id,
                )
            )

    # --------------------------------------------------------
    # ATTACH EVALUATION
    # --------------------------------------------------------

    def attach_evaluation(
        self,
        model_version_id: str,
        *,
        evaluation_id: str,
        metrics: dict[str, Any],
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersion:
        evaluation_id = validate_string(
            evaluation_id,
            "evaluation_id",
            64,
        )

        metrics = validate_json_dict(
            metrics,
            "metrics",
        )

        with self._session(db) as session:
            result = session.execute(
                update(ModelVersionORM)
                .where(
                    ModelVersionORM.model_version_id
                    == model_version_id,
                    ModelVersionORM.revision
                    == expected_revision,
                    ModelVersionORM.status
                    != ModelStatus.DISABLED.value,
                )
                .values(
                    evaluation_id=evaluation_id,
                    metrics_data=metrics,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model version missing, disabled "
                    "or changed concurrently"
                )

            self._audit(
                session,
                model_version_id=model_version_id,
                event_type="evaluation_attached",
                details={
                    "evaluation_id": evaluation_id,
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    model_version_id,
                )
            )

    # --------------------------------------------------------
    # PROMOTE MODEL
    # --------------------------------------------------------

    def promote_version(
        self,
        model_version_id: str,
        *,
        stage: ModelStage,
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersion:
        """
        Change registry stage.

        Does not deploy the model.

        A production deployment service should
        additionally require:
        - authenticated operator approval
        - verified checkpoint checksum
        - successful benchmark/evaluation
        - compatible inference runtime
        - rollback capability
        """

        if not isinstance(stage, ModelStage):
            raise ModelValidationError(
                "Invalid model stage"
            )

        allowed = {
            ModelStage.DEVELOPMENT: {
                ModelStage.STAGING,
                ModelStage.ARCHIVED,
            },
            ModelStage.STAGING: {
                ModelStage.DEVELOPMENT,
                ModelStage.PRODUCTION,
                ModelStage.ARCHIVED,
            },
            ModelStage.PRODUCTION: {
                ModelStage.STAGING,
                ModelStage.ARCHIVED,
            },
            ModelStage.ARCHIVED: set(),
        }

        with self._session(db) as session:
            row = self._get_version(
                session,
                model_version_id,
            )

            current = ModelStage(
                row.stage
            )

            if stage not in allowed[current]:
                raise ModelStateError(
                    f"Invalid stage transition: "
                    f"{current.value} -> {stage.value}"
                )

            if (
                stage == ModelStage.PRODUCTION
                and row.status
                != ModelStatus.READY.value
            ):
                raise ModelStateError(
                    "Only ready models can be promoted "
                    "to production"
                )

            result = session.execute(
                update(ModelVersionORM)
                .where(
                    ModelVersionORM.model_version_id
                    == model_version_id,
                    ModelVersionORM.revision
                    == expected_revision,
                    ModelVersionORM.stage
                    == current.value,
                )
                .values(
                    stage=stage.value,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model version changed concurrently"
                )

            self._audit(
                session,
                model_version_id=model_version_id,
                event_type="stage_changed",
                details={
                    "from": current.value,
                    "to": stage.value,
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    model_version_id,
                )
            )

    # --------------------------------------------------------
    # LIST MODELS
    # --------------------------------------------------------

    def list_models(
        self,
        *,
        model_type: ModelType | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[RegisteredModel]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise ModelValidationError(
                "Invalid pagination"
            )

        statement = select(
            RegisteredModelORM
        )

        if model_type is not None:
            statement = statement.where(
                RegisteredModelORM.model_type
                == model_type.value
            )

        statement = (
            statement
            .order_by(
                RegisteredModelORM.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._model_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST MODEL VERSIONS
    # --------------------------------------------------------

    def list_versions(
        self,
        model_id: str,
        *,
        stage: ModelStage | None = None,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[ModelVersion]:
        if not 1 <= limit <= 1000:
            raise ModelValidationError(
                "Invalid limit"
            )

        statement = (
            select(ModelVersionORM)
            .where(
                ModelVersionORM.model_id == model_id
            )
        )

        if stage is not None:
            statement = statement.where(
                ModelVersionORM.stage == stage.value
            )

        statement = (
            statement
            .order_by(
                ModelVersionORM.created_at.desc()
            )
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._version_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # GET AUDIT EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        model_version_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[ModelRegistryEvent]:
        if not 1 <= limit <= 1000:
            raise ModelValidationError(
                "Invalid limit"
            )

        statement = (
            select(ModelRegistryEventORM)
            .where(
                ModelRegistryEventORM.model_version_id
                == model_version_id
            )
            .order_by(
                ModelRegistryEventORM.created_at.asc()
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
    # EXPORT MODEL REGISTRY
    # --------------------------------------------------------

    def export_json(
        self,
        model_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        with self._session(db) as session:
            model = self.get_model(
                model_id,
                db=session,
            )

            versions = self.list_versions(
                model_id,
                limit=1000,
                db=session,
            )

            payload = {
                "model": asdict(model),
                "versions": [],
            }

            for version in versions:
                payload["versions"].append(
                    {
                        **asdict(version),
                        "events": [
                            asdict(event)
                            for event in self.list_events(
                                version.model_version_id,
                                limit=1000,
                                db=session,
                            )
                        ],
                    }
                )

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

def create_model_registry_repository() -> (
    ModelRegistryRepository
):
    return ModelRegistryRepository()


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    create_database_tables()

    registry = create_model_registry_repository()

    model = registry.register_model(
        name="NeuroWorld Transformer Demo",
        model_type=ModelType.TRANSFORMER_WORLD_MODEL,
        description=(
            "Action-conditioned neural world model "
            "registry demonstration"
        ),
        metadata={
            "project": "NeuroWorld AI",
            "simulation_only": True,
        },
    )

    version = registry.register_version(
        model_id=model.model_id,
        version="v1.0",
        architecture={
            "architecture": "transformer",
            "latent_dimension": 256,
            "num_layers": 8,
            "attention_heads": 8,
        },
        configuration={
            "learning_rate": 0.0001,
            "batch_size": 32,
            "prediction_horizon": 12,
        },
    )

    version = registry.update_status(
        version.model_version_id,
        status=ModelStatus.TRAINING,
        expected_revision=version.revision,
    )

    # Demo only: this does not train a model.
    version = registry.update_status(
        version.model_version_id,
        status=ModelStatus.READY,
        expected_revision=version.revision,
    )

    version = registry.attach_checkpoint(
        version.model_version_id,
        checkpoint_uri=(
            "models/checkpoints/"
            "transformer_world_model_v1.pt"
        ),
        expected_revision=version.revision,
    )

    version = registry.promote_version(
        version.model_version_id,
        stage=ModelStage.STAGING,
        expected_revision=version.revision,
    )

    print("=" * 60)
    print("NEUROWORLD AI - MODEL REGISTRY")
    print("=" * 60)

    print("Model ID:", model.model_id)
    print("Model name:", model.name)
    print("Version:", version.version)
    print("Status:", version.status)
    print("Stage:", version.stage)

    report = registry.export_json(
        model.model_id,
        "experiments/reports/model_registry.json",
    )

    print("Report:", report)

    close_database()


if __name__ == "__main__":
    main()
