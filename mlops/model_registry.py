
"""
NeuroWorld AI
backend/app/database/model_registry.py

Persistent AI/ML model registry.

Features:
- Register AI models
- Register immutable model versions
- SHA-256 artifact verification
- Model evaluation metrics
- Model lifecycle management
- Development/staging/production stages
- Production promotion and rollback
- Model version comparison
- Model audit history
- Model registry JSON export

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

This module manages model metadata.
It does not execute inference or deploy models.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

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

class ModelRegistryError(Exception):
    """Base model registry exception."""


class ModelValidationError(ModelRegistryError):
    """Invalid model registry input."""


class ModelNotFoundError(ModelRegistryError):
    """Model or version does not exist."""


class ModelConflictError(ModelRegistryError):
    """Duplicate or concurrent modification."""


class ModelStateError(ModelRegistryError):
    """Invalid model lifecycle operation."""


class ModelIntegrityError(ModelRegistryError):
    """Model artifact checksum mismatch."""


# ============================================================
# ENUMS
# ============================================================

class ModelType(str, Enum):
    WORLD_MODEL = "world_model"
    TRANSFORMER_WORLD_MODEL = (
        "transformer_world_model"
    )
    DIFFUSION_WORLD_MODEL = (
        "diffusion_world_model"
    )
    DYNAMICS = "dynamics"
    TOKENIZER = "tokenizer"
    VIDEO_PREDICTION = "video_prediction"
    TRAJECTORY_PREDICTION = (
        "trajectory_prediction"
    )
    POLICY = "policy"
    REWARD_MODEL = "reward_model"
    VISION = "vision"
    MULTIMODAL = "multimodal"
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


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


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
        raise ModelValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise ModelValidationError(
            f"Invalid {field}"
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
                "Non-finite JSON number"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ModelValidationError(
        f"Unsupported JSON type: "
        f"{type(value).__name__}"
    )


def validate_metadata(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    result = json_safe(
        value or {}
    )

    if not isinstance(result, dict):
        raise ModelValidationError(
            "metadata must be a dictionary"
        )

    return result


def validate_metric(
    value: float,
) -> float:
    if isinstance(value, bool):
        raise ModelValidationError(
            "Metric must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelValidationError(
            "Invalid metric value"
        ) from exc

    if not math.isfinite(result):
        raise ModelValidationError(
            "Metric must be finite"
        )

    return result


def sha256_file(
    path: str | Path,
) -> str:
    path = Path(path)

    if not path.is_file():
        raise ModelIntegrityError(
            f"Artifact not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_sha256(
    value: str,
) -> str:
    value = validate_string(
        value,
        "sha256",
        64,
    ).lower()

    if (
        len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ModelValidationError(
            "Invalid SHA-256 checksum"
        )

    return value


# ============================================================
# ORM MODELS
# ============================================================

class RegistryModelORM(Base):
    __tablename__ = "registry_models"

    model_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    model_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    description: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    framework: Mapped[str] = mapped_column(
        String(128),
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

    versions: Mapped[
        list["RegistryVersionORM"]
    ] = relationship(
        back_populates="model",
        cascade="all, delete-orphan",
    )


class RegistryVersionORM(Base):
    __tablename__ = "registry_model_versions"

    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "version",
            name="uq_registry_model_version",
        ),
        Index(
            "ix_registry_model_stage",
            "model_id",
            "stage",
        ),
    )

    version_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    model_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "registry_models.model_id",
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
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )

    artifact_uri: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    artifact_sha256: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
    )

    dataset_version_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
    )

    training_run_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
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

    model: Mapped[RegistryModelORM] = relationship(
        back_populates="versions",
    )

    metrics: Mapped[
        list["RegistryMetricORM"]
    ] = relationship(
        back_populates="version_record",
        cascade="all, delete-orphan",
    )

    events: Mapped[
        list["RegistryEventORM"]
    ] = relationship(
        back_populates="version_record",
        cascade="all, delete-orphan",
    )


class RegistryMetricORM(Base):
    __tablename__ = "registry_model_metrics"

    __table_args__ = (
        UniqueConstraint(
            "version_id",
            "metric_name",
            name="uq_registry_version_metric",
        ),
    )

    metric_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "registry_model_versions.version_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    metric_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    dataset_name: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    version_record: Mapped[
        RegistryVersionORM
    ] = relationship(
        back_populates="metrics",
    )


class RegistryEventORM(Base):
    __tablename__ = "registry_model_events"

    event_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "registry_model_versions.version_id",
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

    version_record: Mapped[
        RegistryVersionORM
    ] = relationship(
        back_populates="events",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    name: str
    model_type: str
    description: str | None
    framework: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ModelVersionRecord:
    version_id: str
    model_id: str
    version: str
    stage: str
    status: str
    artifact_uri: str | None
    artifact_sha256: str | None
    dataset_version_id: str | None
    training_run_id: str | None
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ModelMetricRecord:
    metric_id: str
    version_id: str
    metric_name: str
    metric_value: float
    dataset_name: str | None
    created_at: str


@dataclass(frozen=True)
class ModelComparison:
    model_id: str
    metric_name: str
    direction: str
    ranked_versions: list[dict[str, Any]]


# ============================================================
# MODEL REGISTRY REPOSITORY
# ============================================================

class ModelRegistry:
    """
    Persistent model registry.

    Register models, attach model versions,
    record evaluation results and manage
    promotion stages.

    Production promotion updates registry
    metadata only. It does not deploy models.
    """

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _get_model(
        session: Session,
        model_id: str,
    ) -> RegistryModelORM:
        row = session.get(
            RegistryModelORM,
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
        version_id: str,
    ) -> RegistryVersionORM:
        row = session.get(
            RegistryVersionORM,
            version_id,
        )

        if row is None:
            raise ModelNotFoundError(
                "Model version not found"
            )

        return row

    @staticmethod
    def _model_record(
        row: RegistryModelORM,
    ) -> ModelRecord:
        return ModelRecord(
            model_id=row.model_id,
            name=row.name,
            model_type=row.model_type,
            description=row.description,
            framework=row.framework,
            metadata=json_safe(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _version_record(
        row: RegistryVersionORM,
    ) -> ModelVersionRecord:
        return ModelVersionRecord(
            version_id=row.version_id,
            model_id=row.model_id,
            version=row.version,
            stage=row.stage,
            status=row.status,
            artifact_uri=row.artifact_uri,
            artifact_sha256=(
                row.artifact_sha256
            ),
            dataset_version_id=(
                row.dataset_version_id
            ),
            training_run_id=row.training_run_id,
            configuration=json_safe(
                row.configuration_data
            ),
            metadata=json_safe(
                row.metadata_data
            ),
            revision=row.revision,
            created_at=iso_datetime(
                row.created_at
            ),
            updated_at=iso_datetime(
                row.updated_at
            ),
        )

    @staticmethod
    def _metric_record(
        row: RegistryMetricORM,
    ) -> ModelMetricRecord:
        return ModelMetricRecord(
            metric_id=row.metric_id,
            version_id=row.version_id,
            metric_name=row.metric_name,
            metric_value=row.metric_value,
            dataset_name=row.dataset_name,
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _audit(
        session: Session,
        version_id: str,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            RegistryEventORM(
                event_id=uuid.uuid4().hex,
                version_id=version_id,
                event_type=event_type,
                details_data=validate_metadata(
                    details
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
        framework: str = "pytorch",
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ModelRecord:
        name = validate_string(
            name,
            "name",
        )

        framework = validate_string(
            framework,
            "framework",
            128,
        )

        if not isinstance(model_type, ModelType):
            raise ModelValidationError(
                "Invalid model_type"
            )

        if description is not None:
            description = validate_string(
                description,
                "description",
                4000,
            )

        row = RegistryModelORM(
            model_id=uuid.uuid4().hex,
            name=name,
            model_type=model_type.value,
            framework=framework,
            description=description,
            metadata_data=validate_metadata(
                metadata
            ),
            created_at=utc_now(),
        )

        with self._session(db) as session:
            existing = session.scalar(
                select(RegistryModelORM)
                .where(
                    RegistryModelORM.name == name
                )
            )

            if existing is not None:
                raise ModelConflictError(
                    "Model name already registered"
                )

            session.add(row)
            session.flush()

            return self._model_record(row)

    # --------------------------------------------------------
    # REGISTER VERSION
    # --------------------------------------------------------

    def register_version(
        self,
        model_id: str,
        *,
        version: str,
        artifact_uri: str | None = None,
        artifact_sha256: str | None = None,
        dataset_version_id: str | None = None,
        training_run_id: str | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ModelVersionRecord:
        version = validate_string(
            version,
            "version",
            128,
        )

        if artifact_uri is not None:
            artifact_uri = validate_string(
                artifact_uri,
                "artifact_uri",
                4000,
            )

        if artifact_sha256 is not None:
            artifact_sha256 = validate_sha256(
                artifact_sha256
            )

        if (
            artifact_sha256 is not None
            and artifact_uri is None
        ):
            raise ModelValidationError(
                "artifact_uri is required "
                "when a checksum is provided"
            )

        row = RegistryVersionORM(
            version_id=uuid.uuid4().hex,
            model_id=model_id,
            version=version,
            stage=ModelStage.DEVELOPMENT.value,
            status=ModelStatus.REGISTERED.value,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
            dataset_version_id=(
                dataset_version_id
            ),
            training_run_id=training_run_id,
            configuration_data=validate_metadata(
                configuration
            ),
            metadata_data=validate_metadata(
                metadata
            ),
            revision=1,
            created_at=utc_now(),
            updated_at=utc_now(),
        )

        with self._session(db) as session:
            self._get_model(
                session,
                model_id,
            )

            existing = session.scalar(
                select(RegistryVersionORM)
                .where(
                    RegistryVersionORM.model_id
                    == model_id,
                    RegistryVersionORM.version
                    == version,
                )
            )

            if existing is not None:
                raise ModelConflictError(
                    "Model version already exists"
                )

            session.add(row)

            self._audit(
                session,
                row.version_id,
                "version_registered",
                {
                    "version": version,
                },
            )

            session.flush()

            return self._version_record(row)

    # --------------------------------------------------------
    # GET MODEL / VERSION
    # --------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        *,
        db: Session | None = None,
    ) -> ModelRecord:
        with self._session(db) as session:
            return self._model_record(
                self._get_model(
                    session,
                    model_id,
                )
            )

    def get_version(
        self,
        version_id: str,
        *,
        db: Session | None = None,
    ) -> ModelVersionRecord:
        with self._session(db) as session:
            return self._version_record(
                self._get_version(
                    session,
                    version_id,
                )
            )

    # --------------------------------------------------------
    # UPDATE STATUS
    # --------------------------------------------------------

    def update_status(
        self,
        version_id: str,
        *,
        expected_revision: int,
        status: ModelStatus,
        db: Session | None = None,
    ) -> ModelVersionRecord:
        if not isinstance(status, ModelStatus):
            raise ModelValidationError(
                "Invalid model status"
            )

        with self._session(db) as session:
            row = self._get_version(
                session,
                version_id,
            )

            if (
                row.stage
                == ModelStage.ARCHIVED.value
            ):
                raise ModelStateError(
                    "Archived model cannot change status"
                )

            result = session.execute(
                update(RegistryVersionORM)
                .where(
                    RegistryVersionORM.version_id
                    == version_id,
                    RegistryVersionORM.revision
                    == expected_revision,
                )
                .values(
                    status=status.value,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model revision conflict"
                )

            self._audit(
                session,
                version_id,
                "status_updated",
                {
                    "status": status.value,
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    version_id,
                )
            )

    # --------------------------------------------------------
    # ATTACH MODEL ARTIFACT
    # --------------------------------------------------------

    def attach_artifact(
        self,
        version_id: str,
        *,
        artifact_uri: str,
        artifact_sha256: str,
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersionRecord:
        artifact_uri = validate_string(
            artifact_uri,
            "artifact_uri",
            4000,
        )

        artifact_sha256 = validate_sha256(
            artifact_sha256
        )

        with self._session(db) as session:
            row = self._get_version(
                session,
                version_id,
            )

            if row.stage in {
                ModelStage.PRODUCTION.value,
                ModelStage.ARCHIVED.value,
            }:
                raise ModelStateError(
                    "Cannot replace artifact "
                    "in production or archived stage"
                )

            result = session.execute(
                update(RegistryVersionORM)
                .where(
                    RegistryVersionORM.version_id
                    == version_id,
                    RegistryVersionORM.revision
                    == expected_revision,
                )
                .values(
                    artifact_uri=artifact_uri,
                    artifact_sha256=artifact_sha256,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model revision conflict"
                )

            self._audit(
                session,
                version_id,
                "artifact_attached",
                {
                    "artifact_uri": artifact_uri,
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    version_id,
                )
            )

    # --------------------------------------------------------
    # VERIFY ARTIFACT
    # --------------------------------------------------------

    def verify_artifact(
        self,
        version_id: str,
        *,
        db: Session | None = None,
    ) -> bool:
        version = self.get_version(
            version_id,
            db=db,
        )

        if (
            version.artifact_uri is None
            or version.artifact_sha256 is None
        ):
            return False

        # Only local paths are verified here.
        # Cloud/object-storage URIs require
        # their corresponding storage adapter.
        if "://" in version.artifact_uri:
            return False

        try:
            actual_hash = sha256_file(
                version.artifact_uri
            )
        except ModelIntegrityError:
            return False

        return (
            actual_hash
            == version.artifact_sha256
        )

    # --------------------------------------------------------
    # LOG EVALUATION METRIC
    # --------------------------------------------------------

    def log_metric(
        self,
        version_id: str,
        *,
        metric_name: str,
        metric_value: float,
        dataset_name: str | None = None,
        db: Session | None = None,
    ) -> ModelMetricRecord:
        metric_name = validate_string(
            metric_name,
            "metric_name",
            128,
        )

        metric_value = validate_metric(
            metric_value
        )

        if dataset_name is not None:
            dataset_name = validate_string(
                dataset_name,
                "dataset_name",
            )

        with self._session(db) as session:
            version = self._get_version(
                session,
                version_id,
            )

            if version.stage == ModelStage.ARCHIVED.value:
                raise ModelStateError(
                    "Cannot modify archived model metrics"
                )

            existing = session.scalar(
                select(RegistryMetricORM)
                .where(
                    RegistryMetricORM.version_id
                    == version_id,
                    RegistryMetricORM.metric_name
                    == metric_name,
                )
            )

            if existing is not None:
                raise ModelConflictError(
                    "Metric already exists"
                )

            row = RegistryMetricORM(
                metric_id=uuid.uuid4().hex,
                version_id=version_id,
                metric_name=metric_name,
                metric_value=metric_value,
                dataset_name=dataset_name,
                created_at=utc_now(),
            )

            session.add(row)

            self._audit(
                session,
                version_id,
                "metric_logged",
                {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                },
            )

            session.flush()

            return self._metric_record(row)

    # --------------------------------------------------------
    # PROMOTE MODEL
    # --------------------------------------------------------

    def promote_version(
        self,
        version_id: str,
        *,
        target_stage: ModelStage,
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersionRecord:
        if not isinstance(target_stage, ModelStage):
            raise ModelValidationError(
                "Invalid target stage"
            )

        with self._session(db) as session:
            version = self._get_version(
                session,
                version_id,
            )

            if version.status != ModelStatus.READY.value:
                raise ModelStateError(
                    "Only READY versions "
                    "can be promoted"
                )

            if version.stage == ModelStage.ARCHIVED.value:
                raise ModelStateError(
                    "Archived versions cannot be promoted"
                )

            allowed = {
                ModelStage.DEVELOPMENT.value: {
                    ModelStage.STAGING,
                },
                ModelStage.STAGING.value: {
                    ModelStage.PRODUCTION,
                },
                ModelStage.PRODUCTION.value: {
                    ModelStage.STAGING,
                },
            }

            if target_stage not in allowed.get(
                version.stage,
                set(),
            ):
                raise ModelStateError(
                    "Invalid stage transition"
                )

            if target_stage == ModelStage.PRODUCTION:
                if not (
                    version.artifact_uri
                    and version.artifact_sha256
                ):
                    raise ModelStateError(
                        "Production requires "
                        "a registered artifact"
                    )

                # Deployment policy and external
                # artifact verification should be
                # enforced by the calling service.

                current_production = session.scalars(
                    select(RegistryVersionORM)
                    .where(
                        RegistryVersionORM.model_id
                        == version.model_id,
                        RegistryVersionORM.stage
                        == ModelStage.PRODUCTION.value,
                        RegistryVersionORM.version_id
                        != version_id,
                    )
                ).all()

                for previous in current_production:
                    previous.stage = (
                        ModelStage.STAGING.value
                    )
                    previous.revision += 1
                    previous.updated_at = utc_now()

                    self._audit(
                        session,
                        previous.version_id,
                        "production_replaced",
                        {
                            "replacement_version_id": (
                                version_id
                            ),
                        },
                    )

            result = session.execute(
                update(RegistryVersionORM)
                .where(
                    RegistryVersionORM.version_id
                    == version_id,
                    RegistryVersionORM.revision
                    == expected_revision,
                )
                .values(
                    stage=target_stage.value,
                    revision=expected_revision + 1,
                    updated_at=utc_now(),
                )
            )

            if result.rowcount != 1:
                raise ModelConflictError(
                    "Model revision conflict"
                )

            self._audit(
                session,
                version_id,
                "stage_promoted",
                {
                    "target_stage": (
                        target_stage.value
                    ),
                },
            )

            session.flush()
            session.expire_all()

            return self._version_record(
                self._get_version(
                    session,
                    version_id,
                )
            )

    # --------------------------------------------------------
    # ROLLBACK TO PREVIOUS VERSION
    # --------------------------------------------------------

    def rollback_to_version(
        self,
        model_id: str,
        *,
        target_version_id: str,
        expected_revision: int,
        db: Session | None = None,
    ) -> ModelVersionRecord:
        with self._session(db) as session:
            self._get_model(
                session,
                model_id,
            )

            target = self._get_version(
                session,
                target_version_id,
            )

            if target.model_id != model_id:
                raise ModelValidationError(
                    "Version belongs to another model"
                )

            if target.status != ModelStatus.READY.value:
                raise ModelStateError(
                    "Rollback target must be READY"
                )

            if target.stage != ModelStage.STAGING.value:
                raise ModelStateError(
                    "Rollback target must be in staging"
                )

            return self.promote_version(
                target_version_id,
                target_stage=ModelStage.PRODUCTION,
                expected_revision=expected_revision,
                db=session,
            )

    # --------------------------------------------------------
    # GET PRODUCTION VERSION
    # --------------------------------------------------------

    def get_production_version(
        self,
        model_id: str,
        *,
        db: Session | None = None,
    ) -> ModelVersionRecord | None:
        with self._session(db) as session:
            row = session.scalar(
                select(RegistryVersionORM)
                .where(
                    RegistryVersionORM.model_id
                    == model_id,
                    RegistryVersionORM.stage
                    == ModelStage.PRODUCTION.value,
                )
                .order_by(
                    RegistryVersionORM.updated_at.desc()
                )
                .limit(1)
            )

            return (
                self._version_record(row)
                if row is not None
                else None
            )

    # --------------------------------------------------------
    # LIST VERSIONS
    # --------------------------------------------------------

    def list_versions(
        self,
        model_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[ModelVersionRecord]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ModelValidationError(
                "Invalid limit"
            )

        with self._session(db) as session:
            rows = session.scalars(
                select(RegistryVersionORM)
                .where(
                    RegistryVersionORM.model_id
                    == model_id
                )
                .order_by(
                    RegistryVersionORM.created_at.desc()
                )
                .limit(limit)
            ).all()

            return [
                self._version_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # COMPARE MODEL VERSIONS
    # --------------------------------------------------------

    def compare_versions(
        self,
        model_id: str,
        *,
        metric_name: str,
        direction: MetricDirection = (
            MetricDirection.MINIMIZE
        ),
        db: Session | None = None,
    ) -> ModelComparison:
        metric_name = validate_string(
            metric_name,
            "metric_name",
            128,
        )

        if not isinstance(
            direction,
            MetricDirection,
        ):
            raise ModelValidationError(
                "Invalid metric direction"
            )

        with self._session(db) as session:
            versions = session.scalars(
                select(RegistryVersionORM)
                .where(
                    RegistryVersionORM.model_id
                    == model_id
                )
            ).all()

            ranked = []

            for version in versions:
                metric = session.scalar(
                    select(RegistryMetricORM)
                    .where(
                        RegistryMetricORM.version_id
                        == version.version_id,
                        RegistryMetricORM.metric_name
                        == metric_name,
                    )
                )

                if metric is None:
                    continue

                ranked.append(
                    {
                        "version_id": (
                            version.version_id
                        ),
                        "version": version.version,
                        "stage": version.stage,
                        "status": version.status,
                        "metric_value": (
                            metric.metric_value
                        ),
                        "dataset_name": (
                            metric.dataset_name
                        ),
                    }
                )

        ranked.sort(
            key=lambda item: item["metric_value"],
            reverse=(
                direction
                == MetricDirection.MAXIMIZE
            ),
        )

        return ModelComparison(
            model_id=model_id,
            metric_name=metric_name,
            direction=direction.value,
            ranked_versions=ranked,
        )

    # --------------------------------------------------------
    # EXPORT REGISTRY
    # --------------------------------------------------------

    def export_model_json(
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

            versions = session.scalars(
                select(RegistryVersionORM)
                .where(
                    RegistryVersionORM.model_id
                    == model_id
                )
            ).all()

            version_payload = []

            for version in versions:
                metrics = session.scalars(
                    select(RegistryMetricORM)
                    .where(
                        RegistryMetricORM.version_id
                        == version.version_id
                    )
                ).all()

                version_payload.append(
                    {
                        "version": asdict(
                            self._version_record(
                                version
                            )
                        ),
                        "metrics": [
                            asdict(
                                self._metric_record(
                                    metric
                                )
                            )
                            for metric in metrics
                        ],
                    }
                )

        payload = {
            "model": asdict(model),
            "versions": version_payload,
            "exported_at": iso_datetime(
                utc_now()
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


# ============================================================
# FACTORY
# ============================================================

def create_model_registry() -> ModelRegistry:
    return ModelRegistry()


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    create_database_tables()

    registry = create_model_registry()

    model = registry.register_model(
        name=(
            "neuroworld_transformer_"
            + uuid.uuid4().hex[:8]
        ),
        model_type=(
            ModelType.TRANSFORMER_WORLD_MODEL
        ),
        framework="pytorch",
        description=(
            "Transformer-based latent world model"
        ),
    )

    version = registry.register_version(
        model.model_id,
        version="v1.0.0",
        dataset_version_id="dataset_v1",
        training_run_id="run_v1",
        configuration={
            "latent_dim": 256,
            "hidden_dim": 512,
            "num_layers": 6,
        },
    )

    registry.log_metric(
        version.version_id,
        metric_name="validation_loss",
        metric_value=0.125,
        dataset_name="synthetic_validation",
    )

    version = registry.update_status(
        version.version_id,
        expected_revision=version.revision,
        status=ModelStatus.READY,
    )

    version = registry.promote_version(
        version.version_id,
        target_stage=ModelStage.STAGING,
        expected_revision=version.revision,
    )

    print("=" * 60)
    print("NEUROWORLD AI - MODEL REGISTRY")
    print("=" * 60)

    print("Model:", model.name)
    print("Version:", version.version)
    print("Stage:", version.stage)
    print("Status:", version.status)

    comparison = registry.compare_versions(
        model.model_id,
        metric_name="validation_loss",
    )

    print(
        "Ranked versions:",
        comparison.ranked_versions,
    )

    output = registry.export_model_json(
        model.model_id,
        "experiments/reports/"
        "model_registry_export.json",
    )

    print("Export:", output)

    close_database()


if __name__ == "__main__":
    main()
