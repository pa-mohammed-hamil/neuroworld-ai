
"""
NeuroWorld AI
backend/app/database/model_versioning.py

Immutable model artifact version management.

Features:
- Model artifact version registration
- SHA-256 integrity verification
- Model artifact manifests
- Parent-version lineage
- Version comparison
- Version aliases
- Training run and dataset lineage
- Model configuration snapshots
- Reproducibility metadata
- JSON manifest export
- Historical version listing

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Integration:
    backend.app.database.model_registry

Important:
    This module records immutable artifact metadata.
    It does not copy model files, run inference,
    or deploy model versions.

    Artifact paths are restricted to artifact_root.
    Use immutable object storage for production.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
)
from sqlalchemy.exc import IntegrityError
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

from backend.app.database.model_registry import (
    RegistryModelORM,
    RegistryVersionORM,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ModelVersioningError(Exception):
    """Base model versioning exception."""


class ModelVersionValidationError(
    ModelVersioningError
):
    """Invalid model version input."""


class ModelVersionNotFoundError(
    ModelVersioningError
):
    """Model artifact version not found."""


class ModelVersionConflictError(
    ModelVersioningError
):
    """Duplicate or conflicting version."""


class ModelArtifactIntegrityError(
    ModelVersioningError
):
    """Model artifact integrity failure."""


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
        raise ModelVersionValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise ModelVersionValidationError(
            f"Invalid {field}"
        )

    return value


def json_safe(value: Any) -> Any:
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
            raise ModelVersionValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise ModelVersionValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


def validate_dict(
    value: dict[str, Any] | None,
    field: str,
) -> dict[str, Any]:
    result = json_safe(value or {})

    if not isinstance(result, dict):
        raise ModelVersionValidationError(
            f"{field} must be a dictionary"
        )

    return result


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value)
    ).hexdigest()


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    path = Path(path)

    if not path.is_file():
        raise ModelArtifactIntegrityError(
            f"Artifact not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=".model_manifest_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)

            file.write(
                canonical_json(payload)
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


# ============================================================
# ORM MODELS
# ============================================================

class ModelArtifactVersionORM(Base):
    __tablename__ = "model_artifact_versions"

    __table_args__ = (
        UniqueConstraint(
            "registry_version_id",
            name="uq_model_artifact_registry_version",
        ),
        Index(
            "ix_model_artifact_model_created",
            "model_id",
            "created_at",
        ),
    )

    artifact_version_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    model_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "registry_models.model_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    registry_version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "registry_model_versions.version_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    parent_artifact_version_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        ForeignKey(
            "model_artifact_versions.artifact_version_id",
            ondelete="RESTRICT",
        ),
        nullable=True,
    )

    artifact_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    artifact_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    artifact_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    manifest_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "manifest",
        JSON,
        nullable=False,
    )

    manifest_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
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

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class ModelVersionAliasORM(Base):
    __tablename__ = "model_version_aliases"

    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "alias",
            name="uq_model_version_alias",
        ),
    )

    alias_id: Mapped[str] = mapped_column(
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

    artifact_version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "model_artifact_versions.artifact_version_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )

    alias: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class ModelArtifactVersionRecord:
    artifact_version_id: str
    model_id: str
    registry_version_id: str
    parent_artifact_version_id: str | None
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    manifest: dict[str, Any]
    manifest_sha256: str
    dataset_version_id: str | None
    training_run_id: str | None
    created_at: str


@dataclass(frozen=True)
class ModelVersionVerification:
    artifact_version_id: str
    valid: bool
    artifact_exists: bool
    artifact_hash_valid: bool
    artifact_size_valid: bool
    manifest_hash_valid: bool
    registry_consistent: bool


@dataclass(frozen=True)
class ModelVersionComparison:
    source_version_id: str
    target_version_id: str
    same_artifact: bool
    same_dataset: bool
    same_training_run: bool
    configuration_changes: dict[str, Any]
    source_size_bytes: int
    target_size_bytes: int


# ============================================================
# MODEL VERSION MANAGER
# ============================================================

class ModelVersionManager:
    """
    Manages immutable model artifact metadata.

    Each artifact version is linked to one
    version from model_registry.py.

    Artifacts are referenced by relative paths
    beneath artifact_root.
    """

    def __init__(
        self,
        *,
        artifact_root: str | Path = "models",
    ) -> None:
        self.artifact_root = Path(
            artifact_root
        ).resolve()

        self.artifact_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    def _resolve_artifact(
        self,
        artifact_path: str | Path,
    ) -> tuple[Path, str]:
        path = Path(artifact_path)

        if not path.is_absolute():
            path = self.artifact_root / path

        path = path.resolve()

        try:
            relative = path.relative_to(
                self.artifact_root
            )
        except ValueError as exc:
            raise ModelVersionValidationError(
                "Artifact must be inside "
                "the configured artifact root"
            ) from exc

        if not path.is_file():
            raise ModelArtifactIntegrityError(
                f"Artifact does not exist: {path}"
            )

        return path, relative.as_posix()

    @staticmethod
    def _get_version(
        session: Session,
        artifact_version_id: str,
    ) -> ModelArtifactVersionORM:
        row = session.get(
            ModelArtifactVersionORM,
            artifact_version_id,
        )

        if row is None:
            raise ModelVersionNotFoundError(
                "Model artifact version not found"
            )

        return row

    @staticmethod
    def _record(
        row: ModelArtifactVersionORM,
    ) -> ModelArtifactVersionRecord:
        return ModelArtifactVersionRecord(
            artifact_version_id=(
                row.artifact_version_id
            ),
            model_id=row.model_id,
            registry_version_id=(
                row.registry_version_id
            ),
            parent_artifact_version_id=(
                row.parent_artifact_version_id
            ),
            artifact_path=row.artifact_path,
            artifact_sha256=row.artifact_sha256,
            artifact_size_bytes=(
                row.artifact_size_bytes
            ),
            manifest=json_safe(
                row.manifest_data
            ),
            manifest_sha256=(
                row.manifest_sha256
            ),
            dataset_version_id=(
                row.dataset_version_id
            ),
            training_run_id=(
                row.training_run_id
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    # --------------------------------------------------------
    # REGISTER ARTIFACT VERSION
    # --------------------------------------------------------

    def register_artifact_version(
        self,
        *,
        model_id: str,
        registry_version_id: str,
        artifact_path: str | Path,
        parent_artifact_version_id: str | None = None,
        configuration: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> ModelArtifactVersionRecord:
        model_id = validate_string(
            model_id,
            "model_id",
            64,
        )

        registry_version_id = validate_string(
            registry_version_id,
            "registry_version_id",
            64,
        )

        path, relative_path = self._resolve_artifact(
            artifact_path
        )

        artifact_hash = sha256_file(
            path
        )

        artifact_size = path.stat().st_size

        with self._session(db) as session:
            model = session.get(
                RegistryModelORM,
                model_id,
            )

            if model is None:
                raise ModelVersionNotFoundError(
                    "Registered model not found"
                )

            registry_version = session.get(
                RegistryVersionORM,
                registry_version_id,
            )

            if registry_version is None:
                raise ModelVersionNotFoundError(
                    "Registry version not found"
                )

            if registry_version.model_id != model_id:
                raise ModelVersionValidationError(
                    "Registry version belongs "
                    "to another model"
                )

            existing = session.scalar(
                select(ModelArtifactVersionORM)
                .where(
                    ModelArtifactVersionORM.registry_version_id
                    == registry_version_id
                )
            )

            if existing is not None:
                raise ModelVersionConflictError(
                    "Artifact version already registered"
                )

            if parent_artifact_version_id is not None:
                parent = self._get_version(
                    session,
                    parent_artifact_version_id,
                )

                if parent.model_id != model_id:
                    raise ModelVersionValidationError(
                        "Parent belongs to another model"
                    )

            # The registry is the canonical source
            # for dataset and training-run lineage.
            dataset_version_id = (
                registry_version.dataset_version_id
            )

            training_run_id = (
                registry_version.training_run_id
            )

            manifest = {
                "schema_version": 1,
                "model_id": model_id,
                "registry_version_id": (
                    registry_version_id
                ),
                "registry_version": (
                    registry_version.version
                ),
                "parent_artifact_version_id": (
                    parent_artifact_version_id
                ),
                "artifact": {
                    "path": relative_path,
                    "sha256": artifact_hash,
                    "size_bytes": artifact_size,
                },
                "dataset_version_id": (
                    dataset_version_id
                ),
                "training_run_id": (
                    training_run_id
                ),
                "configuration": validate_dict(
                    configuration
                    if configuration is not None
                    else registry_version.configuration_data,
                    "configuration",
                ),
                "metadata": validate_dict(
                    metadata,
                    "metadata",
                ),
            }

            row = ModelArtifactVersionORM(
                artifact_version_id=uuid.uuid4().hex,
                model_id=model_id,
                registry_version_id=(
                    registry_version_id
                ),
                parent_artifact_version_id=(
                    parent_artifact_version_id
                ),
                artifact_path=relative_path,
                artifact_sha256=artifact_hash,
                artifact_size_bytes=artifact_size,
                manifest_data=manifest,
                manifest_sha256=sha256_json(
                    manifest
                ),
                dataset_version_id=(
                    dataset_version_id
                ),
                training_run_id=(
                    training_run_id
                ),
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._record(row)

    # --------------------------------------------------------
    # GET ARTIFACT VERSION
    # --------------------------------------------------------

    def get_artifact_version(
        self,
        artifact_version_id: str,
        *,
        db: Session | None = None,
    ) -> ModelArtifactVersionRecord:
        with self._session(db) as session:
            return self._record(
                self._get_version(
                    session,
                    artifact_version_id,
                )
            )

    # --------------------------------------------------------
    # VERIFY ARTIFACT VERSION
    # --------------------------------------------------------

    def verify_artifact_version(
        self,
        artifact_version_id: str,
        *,
        db: Session | None = None,
    ) -> ModelVersionVerification:
        with self._session(db) as session:
            row = self._get_version(
                session,
                artifact_version_id,
            )

            record = self._record(row)

            registry = session.get(
                RegistryVersionORM,
                record.registry_version_id,
            )

            registry_consistent = (
                registry is not None
                and registry.model_id
                == record.model_id
                and registry.dataset_version_id
                == record.dataset_version_id
                and registry.training_run_id
                == record.training_run_id
            )

        manifest_valid = (
            sha256_json(record.manifest)
            == record.manifest_sha256
        )

        artifact_exists = False
        artifact_hash_valid = False
        artifact_size_valid = False

        try:
            path, relative_path = self._resolve_artifact(
                record.artifact_path
            )

            artifact_exists = True

            artifact_size_valid = (
                path.stat().st_size
                == record.artifact_size_bytes
            )

            artifact_hash_valid = (
                sha256_file(path)
                == record.artifact_sha256
            )

        except (
            ModelVersionValidationError,
            ModelArtifactIntegrityError,
        ):
            pass

        manifest_artifact = (
            record.manifest.get("artifact", {})
        )

        manifest_consistent = (
            manifest_artifact.get("path")
            == record.artifact_path
            and manifest_artifact.get("sha256")
            == record.artifact_sha256
            and manifest_artifact.get("size_bytes")
            == record.artifact_size_bytes
        )

        manifest_valid = (
            manifest_valid
            and manifest_consistent
        )

        valid = all(
            [
                artifact_exists,
                artifact_hash_valid,
                artifact_size_valid,
                manifest_valid,
                registry_consistent,
            ]
        )

        return ModelVersionVerification(
            artifact_version_id=artifact_version_id,
            valid=valid,
            artifact_exists=artifact_exists,
            artifact_hash_valid=artifact_hash_valid,
            artifact_size_valid=artifact_size_valid,
            manifest_hash_valid=manifest_valid,
            registry_consistent=registry_consistent,
        )

    # --------------------------------------------------------
    # COMPARE ARTIFACT VERSIONS
    # --------------------------------------------------------

    def compare_versions(
        self,
        source_version_id: str,
        target_version_id: str,
        *,
        db: Session | None = None,
    ) -> ModelVersionComparison:
        with self._session(db) as session:
            source = self.get_artifact_version(
                source_version_id,
                db=session,
            )

            target = self.get_artifact_version(
                target_version_id,
                db=session,
            )

        if source.model_id != target.model_id:
            raise ModelVersionValidationError(
                "Cannot compare different models"
            )

        source_config = source.manifest.get(
            "configuration",
            {},
        )

        target_config = target.manifest.get(
            "configuration",
            {},
        )

        configuration_changes = {}

        for key in sorted(
            set(source_config) | set(target_config)
        ):
            old = source_config.get(key)
            new = target_config.get(key)

            if old != new:
                configuration_changes[key] = {
                    "source": old,
                    "target": new,
                }

        return ModelVersionComparison(
            source_version_id=source_version_id,
            target_version_id=target_version_id,
            same_artifact=(
                source.artifact_sha256
                == target.artifact_sha256
            ),
            same_dataset=(
                source.dataset_version_id
                == target.dataset_version_id
            ),
            same_training_run=(
                source.training_run_id
                == target.training_run_id
            ),
            configuration_changes=(
                configuration_changes
            ),
            source_size_bytes=(
                source.artifact_size_bytes
            ),
            target_size_bytes=(
                target.artifact_size_bytes
            ),
        )

    # --------------------------------------------------------
    # CREATE OR UPDATE ALIAS
    # --------------------------------------------------------

    def set_alias(
        self,
        artifact_version_id: str,
        *,
        alias: str,
        db: Session | None = None,
    ) -> None:
        alias = validate_string(
            alias,
            "alias",
            128,
        )

        # Production/staging deployment aliases
        # should be managed by model_registry.py.
        if alias.lower() in {
            "production",
            "staging",
            "prod",
        }:
            raise ModelVersionValidationError(
                "Deployment stage aliases "
                "are managed by model_registry"
            )

        with self._session(db) as session:
            version = self._get_version(
                session,
                artifact_version_id,
            )

            existing = session.scalar(
                select(ModelVersionAliasORM)
                .where(
                    ModelVersionAliasORM.model_id
                    == version.model_id,
                    ModelVersionAliasORM.alias
                    == alias,
                )
            )

            if existing is None:
                session.add(
                    ModelVersionAliasORM(
                        alias_id=uuid.uuid4().hex,
                        model_id=version.model_id,
                        artifact_version_id=(
                            artifact_version_id
                        ),
                        alias=alias,
                        updated_at=utc_now(),
                    )
                )
            else:
                existing.artifact_version_id = (
                    artifact_version_id
                )
                existing.updated_at = utc_now()

            session.flush()

    # --------------------------------------------------------
    # RESOLVE ALIAS
    # --------------------------------------------------------

    def resolve_alias(
        self,
        model_id: str,
        alias: str,
        *,
        db: Session | None = None,
    ) -> ModelArtifactVersionRecord:
        with self._session(db) as session:
            row = session.scalar(
                select(ModelVersionAliasORM)
                .where(
                    ModelVersionAliasORM.model_id
                    == model_id,
                    ModelVersionAliasORM.alias
                    == alias,
                )
            )

            if row is None:
                raise ModelVersionNotFoundError(
                    "Model version alias not found"
                )

            return self._record(
                self._get_version(
                    session,
                    row.artifact_version_id,
                )
            )

    # --------------------------------------------------------
    # LIST MODEL ARTIFACT VERSIONS
    # --------------------------------------------------------

    def list_versions(
        self,
        model_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[ModelArtifactVersionRecord]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ModelVersionValidationError(
                "Invalid limit"
            )

        statement = (
            select(ModelArtifactVersionORM)
            .where(
                ModelArtifactVersionORM.model_id
                == model_id
            )
            .order_by(
                ModelArtifactVersionORM.created_at.desc()
            )
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
    # GET VERSION LINEAGE
    # --------------------------------------------------------

    def get_lineage(
        self,
        artifact_version_id: str,
        *,
        db: Session | None = None,
    ) -> list[ModelArtifactVersionRecord]:
        lineage = []
        visited = set()

        with self._session(db) as session:
            current_id = artifact_version_id

            while current_id is not None:
                if current_id in visited:
                    raise ModelVersionConflictError(
                        "Cycle detected in model lineage"
                    )

                visited.add(current_id)

                version = self._get_version(
                    session,
                    current_id,
                )

                lineage.append(
                    self._record(version)
                )

                current_id = (
                    version.parent_artifact_version_id
                )

        return lineage

    # --------------------------------------------------------
    # EXPORT MANIFEST
    # --------------------------------------------------------

    def export_manifest(
        self,
        artifact_version_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        version = self.get_artifact_version(
            artifact_version_id,
            db=db,
        )

        payload = {
            "artifact_version_id": (
                version.artifact_version_id
            ),
            "manifest": version.manifest,
            "manifest_sha256": (
                version.manifest_sha256
            ),
            "created_at": version.created_at,
        }

        path = Path(output_path)

        atomic_write_json(
            path,
            payload,
        )

        return path


# ============================================================
# FACTORY
# ============================================================

def create_model_version_manager(
    *,
    artifact_root: str | Path = "models",
) -> ModelVersionManager:
    return ModelVersionManager(
        artifact_root=artifact_root
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    from backend.app.database.model_registry import (
        ModelRegistry,
        ModelType,
    )

    create_database_tables()

    root = Path("models").resolve()

    demo_dir = root / "demo_versioning"

    demo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Demo artifact only.
    # Real checkpoints should be produced by
    # your training service.
    artifact = demo_dir / "demo_weights.bin"

    artifact.write_bytes(
        b"NEUROWORLD_DEMO_MODEL_WEIGHTS"
    )

    registry = ModelRegistry()

    model = registry.register_model(
        name=(
            "neuroworld_versioned_"
            + uuid.uuid4().hex[:8]
        ),
        model_type=ModelType.WORLD_MODEL,
        framework="pytorch",
    )

    version = registry.register_version(
        model.model_id,
        version="v1.0.0",
        dataset_version_id="demo_dataset_v1",
        training_run_id="demo_training_run_v1",
        configuration={
            "latent_dim": 256,
            "hidden_dim": 512,
            "layers": 6,
        },
    )

    manager = create_model_version_manager(
        artifact_root=root
    )

    artifact_version = (
        manager.register_artifact_version(
            model_id=model.model_id,
            registry_version_id=(
                version.version_id
            ),
            artifact_path=artifact,
            metadata={
                "demo": True,
            },
        )
    )

    verification = (
        manager.verify_artifact_version(
            artifact_version.artifact_version_id
        )
    )

    manager.set_alias(
        artifact_version.artifact_version_id,
        alias="baseline",
    )

    print("=" * 60)
    print("NEUROWORLD AI - MODEL VERSIONING")
    print("=" * 60)

    print("Model:", model.name)
    print("Version:", version.version)

    print(
        "Artifact SHA-256:",
        artifact_version.artifact_sha256,
    )

    print(
        "Integrity valid:",
        verification.valid,
    )

    print(
        "Lineage length:",
        len(
            manager.get_lineage(
                artifact_version.artifact_version_id
            )
        ),
    )

    output = manager.export_manifest(
        artifact_version.artifact_version_id,
        "experiments/reports/"
        "model_version_manifest.json",
    )

    print("Manifest:", output)

    close_database()


if __name__ == "__main__":
    main()
