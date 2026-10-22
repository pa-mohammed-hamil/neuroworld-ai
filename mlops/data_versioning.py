
"""
NeuroWorld AI
backend/app/database/data_versioning.py

Dataset and training-data version management.

Features:
    - Dataset registration
    - Immutable dataset versions
    - SHA-256 file integrity
    - Dataset manifests
    - Parent-version lineage
    - Dataset comparison
    - Version tags
    - Dataset verification
    - Reproducible training references
    - JSON manifest export
    - Dataset and version listing

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Important:
    This module stores dataset metadata and
    references to existing files. It does not
    upload files or copy dataset contents.
"""

from __future__ import annotations

import hashlib
import json
import math
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


# ============================================================
# EXCEPTIONS
# ============================================================

class DataVersioningError(Exception):
    """Base data versioning exception."""


class DataValidationError(DataVersioningError):
    """Invalid dataset input."""


class DatasetNotFoundError(DataVersioningError):
    """Dataset or version not found."""


class DatasetConflictError(DataVersioningError):
    """Duplicate dataset or version."""


class DatasetIntegrityError(DataVersioningError):
    """Dataset file verification failed."""


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
        raise DataValidationError(
            f"{field} must be a string"
        )

    value = value.strip()

    if not value or len(value) > maximum:
        raise DataValidationError(
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
            raise DataValidationError(
                "Non-finite JSON value"
            )
        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise DataValidationError(
        f"Unsupported JSON value: "
        f"{type(value).__name__}"
    )


def validate_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    result = json_safe(
        metadata or {}
    )

    if not isinstance(result, dict):
        raise DataValidationError(
            "metadata must be a dictionary"
        )

    return result


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    path = Path(path)

    if not path.is_file():
        raise DataValidationError(
            f"File not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(
    payload: Any,
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ============================================================
# ORM MODELS
# ============================================================

class DatasetORM(Base):
    __tablename__ = "ai_datasets"

    dataset_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    description: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    dataset_type: Mapped[str] = mapped_column(
        String(64),
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
        list["DatasetVersionORM"]
    ] = relationship(
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class DatasetVersionORM(Base):
    __tablename__ = "ai_dataset_versions"

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "version",
            name="uq_ai_dataset_version",
        ),
        Index(
            "ix_ai_dataset_versions_dataset_created",
            "dataset_id",
            "created_at",
        ),
    )

    version_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    dataset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_datasets.dataset_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    parent_version_id: Mapped[
        str | None
    ] = mapped_column(
        String(64),
        nullable=True,
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

    file_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    total_size_bytes: Mapped[int] = mapped_column(
        Integer,
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

    dataset: Mapped[DatasetORM] = relationship(
        back_populates="versions",
    )

    tags: Mapped[
        list["DatasetTagORM"]
    ] = relationship(
        back_populates="version_record",
        cascade="all, delete-orphan",
    )


class DatasetTagORM(Base):
    __tablename__ = "ai_dataset_tags"

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "tag",
            name="uq_ai_dataset_tag",
        ),
    )

    tag_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    dataset_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_datasets.dataset_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    version_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey(
            "ai_dataset_versions.version_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    tag: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    version_record: Mapped[
        DatasetVersionORM
    ] = relationship(
        back_populates="tags",
    )


# ============================================================
# PUBLIC RECORDS
# ============================================================

@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    description: str | None
    dataset_type: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class DatasetVersionRecord:
    version_id: str
    dataset_id: str
    version: str
    parent_version_id: str | None
    manifest: dict[str, Any]
    manifest_sha256: str
    file_count: int
    total_size_bytes: int
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class DatasetComparison:
    source_version: str
    target_version: str
    added_files: list[str]
    removed_files: list[str]
    modified_files: list[str]
    unchanged_files: list[str]


@dataclass(frozen=True)
class DatasetVerification:
    version_id: str
    valid: bool
    verified_files: int
    missing_files: list[str]
    corrupted_files: list[str]
    manifest_valid: bool


# ============================================================
# DATA VERSIONING REPOSITORY
# ============================================================

class DataVersioningRepository:
    """
    Manages immutable dataset-version records.

    A dataset version stores:
        - file paths
        - file sizes
        - SHA-256 checksums
        - manifest checksum
        - parent version reference

    Dataset files remain outside the database.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path = "data",
        max_files_per_version: int = 100000,
    ) -> None:
        self.dataset_root = Path(
            dataset_root
        ).resolve()

        self.dataset_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if (
            isinstance(max_files_per_version, bool)
            or not isinstance(
                max_files_per_version,
                int,
            )
            or max_files_per_version < 1
        ):
            raise DataValidationError(
                "Invalid max_files_per_version"
            )

        self.max_files_per_version = (
            max_files_per_version
        )

    @staticmethod
    def _session(db: Session | None):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _dataset(
        session: Session,
        dataset_id: str,
    ) -> DatasetORM:
        row = session.get(
            DatasetORM,
            dataset_id,
        )

        if row is None:
            raise DatasetNotFoundError(
                "Dataset not found"
            )

        return row

    @staticmethod
    def _version(
        session: Session,
        version_id: str,
    ) -> DatasetVersionORM:
        row = session.get(
            DatasetVersionORM,
            version_id,
        )

        if row is None:
            raise DatasetNotFoundError(
                "Dataset version not found"
            )

        return row

    @staticmethod
    def _dataset_record(
        row: DatasetORM,
    ) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=row.dataset_id,
            name=row.name,
            description=row.description,
            dataset_type=row.dataset_type,
            metadata=dict(row.metadata_data),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    @staticmethod
    def _version_record(
        row: DatasetVersionORM,
    ) -> DatasetVersionRecord:
        return DatasetVersionRecord(
            version_id=row.version_id,
            dataset_id=row.dataset_id,
            version=row.version,
            parent_version_id=(
                row.parent_version_id
            ),
            manifest=json_safe(
                row.manifest_data
            ),
            manifest_sha256=(
                row.manifest_sha256
            ),
            file_count=row.file_count,
            total_size_bytes=(
                row.total_size_bytes
            ),
            metadata=json_safe(
                row.metadata_data
            ),
            created_at=iso_datetime(
                row.created_at
            ),
        )

    # --------------------------------------------------------
    # REGISTER DATASET
    # --------------------------------------------------------

    def register_dataset(
        self,
        *,
        name: str,
        dataset_type: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> DatasetRecord:
        name = validate_string(
            name,
            "name",
        )

        dataset_type = validate_string(
            dataset_type,
            "dataset_type",
            64,
        )

        if description is not None:
            description = validate_string(
                description,
                "description",
                4000,
            )

        row = DatasetORM(
            dataset_id=uuid.uuid4().hex,
            name=name,
            description=description,
            dataset_type=dataset_type,
            metadata_data=validate_metadata(
                metadata
            ),
            created_at=utc_now(),
        )

        with self._session(db) as session:
            existing = session.scalar(
                select(DatasetORM)
                .where(
                    DatasetORM.name == name
                )
            )

            if existing is not None:
                raise DatasetConflictError(
                    "Dataset name already exists"
                )

            session.add(row)
            session.flush()

            return self._dataset_record(row)

    # --------------------------------------------------------
    # GET DATASET
    # --------------------------------------------------------

    def get_dataset(
        self,
        dataset_id: str,
        *,
        db: Session | None = None,
    ) -> DatasetRecord:
        with self._session(db) as session:
            return self._dataset_record(
                self._dataset(
                    session,
                    dataset_id,
                )
            )

    # --------------------------------------------------------
    # BUILD FILE MANIFEST
    # --------------------------------------------------------

    def build_manifest(
        self,
        file_paths: list[str | Path],
    ) -> dict[str, Any]:
        if not isinstance(file_paths, list):
            raise DataValidationError(
                "file_paths must be a list"
            )

        if not 1 <= len(file_paths) <= (
            self.max_files_per_version
        ):
            raise DataValidationError(
                "Invalid number of files"
            )

        entries = []
        seen_paths = set()

        for input_path in file_paths:
            path = Path(input_path).resolve()

            try:
                relative_path = path.relative_to(
                    self.dataset_root
                )
            except ValueError as exc:
                raise DataValidationError(
                    "Dataset files must be inside "
                    "the configured dataset root"
                ) from exc

            if not path.is_file():
                raise DataValidationError(
                    f"Dataset file not found: {path}"
                )

            relative_name = (
                relative_path.as_posix()
            )

            if relative_name in seen_paths:
                raise DataValidationError(
                    "Duplicate dataset file"
                )

            seen_paths.add(
                relative_name
            )

            entries.append(
                {
                    "path": relative_name,
                    "size_bytes": (
                        path.stat().st_size
                    ),
                    "sha256": sha256_file(
                        path
                    ),
                }
            )

        entries.sort(
            key=lambda item: item["path"]
        )

        return {
            "schema_version": 1,
            "files": entries,
            "file_count": len(entries),
            "total_size_bytes": sum(
                item["size_bytes"]
                for item in entries
            ),
        }

    # --------------------------------------------------------
    # CREATE DATASET VERSION
    # --------------------------------------------------------

    def create_version(
        self,
        dataset_id: str,
        *,
        version: str,
        file_paths: list[str | Path],
        parent_version_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        db: Session | None = None,
    ) -> DatasetVersionRecord:
        version = validate_string(
            version,
            "version",
            128,
        )

        manifest = self.build_manifest(
            file_paths
        )

        checksum = sha256_json(
            manifest
        )

        with self._session(db) as session:
            self._dataset(
                session,
                dataset_id,
            )

            existing = session.scalar(
                select(DatasetVersionORM)
                .where(
                    DatasetVersionORM.dataset_id
                    == dataset_id,
                    DatasetVersionORM.version
                    == version,
                )
            )

            if existing is not None:
                raise DatasetConflictError(
                    "Dataset version already exists"
                )

            if parent_version_id is not None:
                parent = self._version(
                    session,
                    parent_version_id,
                )

                if parent.dataset_id != dataset_id:
                    raise DataValidationError(
                        "Parent version belongs "
                        "to another dataset"
                    )

            row = DatasetVersionORM(
                version_id=uuid.uuid4().hex,
                dataset_id=dataset_id,
                version=version,
                parent_version_id=(
                    parent_version_id
                ),
                manifest_data=manifest,
                manifest_sha256=checksum,
                file_count=manifest["file_count"],
                total_size_bytes=(
                    manifest["total_size_bytes"]
                ),
                metadata_data=validate_metadata(
                    metadata
                ),
                created_at=utc_now(),
            )

            session.add(row)
            session.flush()

            return self._version_record(row)

    # --------------------------------------------------------
    # GET VERSION
    # --------------------------------------------------------

    def get_version(
        self,
        version_id: str,
        *,
        db: Session | None = None,
    ) -> DatasetVersionRecord:
        with self._session(db) as session:
            return self._version_record(
                self._version(
                    session,
                    version_id,
                )
            )

    # --------------------------------------------------------
    # VERIFY VERSION
    # --------------------------------------------------------

    def verify_version(
        self,
        version_id: str,
        *,
        db: Session | None = None,
    ) -> DatasetVerification:
        version = self.get_version(
            version_id,
            db=db,
        )

        manifest_valid = (
            sha256_json(version.manifest)
            == version.manifest_sha256
        )

        missing_files = []
        corrupted_files = []
        verified_files = 0

        for entry in version.manifest["files"]:
            path = (
                self.dataset_root
                / entry["path"]
            ).resolve()

            try:
                path.relative_to(
                    self.dataset_root
                )
            except ValueError:
                corrupted_files.append(
                    entry["path"]
                )
                continue

            if not path.is_file():
                missing_files.append(
                    entry["path"]
                )
                continue

            actual_size = path.stat().st_size
            actual_hash = sha256_file(
                path
            )

            if (
                actual_size != entry["size_bytes"]
                or actual_hash != entry["sha256"]
            ):
                corrupted_files.append(
                    entry["path"]
                )
            else:
                verified_files += 1

        return DatasetVerification(
            version_id=version_id,
            valid=(
                manifest_valid
                and not missing_files
                and not corrupted_files
            ),
            verified_files=verified_files,
            missing_files=missing_files,
            corrupted_files=corrupted_files,
            manifest_valid=manifest_valid,
        )

    # --------------------------------------------------------
    # COMPARE DATASET VERSIONS
    # --------------------------------------------------------

    def compare_versions(
        self,
        source_version_id: str,
        target_version_id: str,
        *,
        db: Session | None = None,
    ) -> DatasetComparison:
        with self._session(db) as session:
            source = self.get_version(
                source_version_id,
                db=session,
            )

            target = self.get_version(
                target_version_id,
                db=session,
            )

        if source.dataset_id != target.dataset_id:
            raise DataValidationError(
                "Versions belong to different datasets"
            )

        source_files = {
            item["path"]: item["sha256"]
            for item in source.manifest["files"]
        }

        target_files = {
            item["path"]: item["sha256"]
            for item in target.manifest["files"]
        }

        source_paths = set(
            source_files
        )

        target_paths = set(
            target_files
        )

        common = (
            source_paths & target_paths
        )

        return DatasetComparison(
            source_version=source.version,
            target_version=target.version,
            added_files=sorted(
                target_paths - source_paths
            ),
            removed_files=sorted(
                source_paths - target_paths
            ),
            modified_files=sorted(
                path
                for path in common
                if source_files[path]
                != target_files[path]
            ),
            unchanged_files=sorted(
                path
                for path in common
                if source_files[path]
                == target_files[path]
            ),
        )

    # --------------------------------------------------------
    # TAG DATASET VERSION
    # --------------------------------------------------------

    def tag_version(
        self,
        version_id: str,
        *,
        tag: str,
        db: Session | None = None,
    ) -> None:
        tag = validate_string(
            tag,
            "tag",
            128,
        )

        with self._session(db) as session:
            version = self._version(
                session,
                version_id,
            )

            existing = session.scalar(
                select(DatasetTagORM)
                .where(
                    DatasetTagORM.dataset_id
                    == version.dataset_id,
                    DatasetTagORM.tag == tag,
                )
            )

            if existing is not None:
                raise DatasetConflictError(
                    "Dataset tag already exists"
                )

            session.add(
                DatasetTagORM(
                    tag_id=uuid.uuid4().hex,
                    dataset_id=version.dataset_id,
                    version_id=version_id,
                    tag=tag,
                    created_at=utc_now(),
                )
            )

            session.flush()

    # --------------------------------------------------------
    # RESOLVE TAG
    # --------------------------------------------------------

    def resolve_tag(
        self,
        dataset_id: str,
        tag: str,
        *,
        db: Session | None = None,
    ) -> DatasetVersionRecord:
        with self._session(db) as session:
            row = session.scalar(
                select(DatasetTagORM)
                .where(
                    DatasetTagORM.dataset_id
                    == dataset_id,
                    DatasetTagORM.tag == tag,
                )
            )

            if row is None:
                raise DatasetNotFoundError(
                    "Dataset tag not found"
                )

            return self._version_record(
                self._version(
                    session,
                    row.version_id,
                )
            )

    # --------------------------------------------------------
    # LIST DATASETS
    # --------------------------------------------------------

    def list_datasets(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[DatasetRecord]:
        if (
            isinstance(limit, bool)
            or isinstance(offset, bool)
            or not isinstance(limit, int)
            or not isinstance(offset, int)
            or not 1 <= limit <= 1000
            or offset < 0
        ):
            raise DataValidationError(
                "Invalid pagination"
            )

        statement = (
            select(DatasetORM)
            .order_by(
                DatasetORM.created_at.desc()
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._dataset_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    # --------------------------------------------------------
    # LIST VERSIONS
    # --------------------------------------------------------

    def list_versions(
        self,
        dataset_id: str,
        *,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[DatasetVersionRecord]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise DataValidationError(
                "Invalid limit"
            )

        statement = (
            select(DatasetVersionORM)
            .where(
                DatasetVersionORM.dataset_id
                == dataset_id
            )
            .order_by(
                DatasetVersionORM.created_at.desc()
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
    # EXPORT VERSION MANIFEST
    # --------------------------------------------------------

    def export_manifest(
        self,
        version_id: str,
        output_path: str | Path,
        *,
        db: Session | None = None,
    ) -> Path:
        version = self.get_version(
            version_id,
            db=db,
        )

        payload = {
            "dataset_id": version.dataset_id,
            "version_id": version.version_id,
            "version": version.version,
            "parent_version_id": (
                version.parent_version_id
            ),
            "manifest": version.manifest,
            "manifest_sha256": (
                version.manifest_sha256
            ),
            "metadata": version.metadata,
            "created_at": version.created_at,
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

def create_data_versioning_repository(
    *,
    dataset_root: str | Path = "data",
    max_files_per_version: int = 100000,
) -> DataVersioningRepository:
    return DataVersioningRepository(
        dataset_root=dataset_root,
        max_files_per_version=(
            max_files_per_version
        ),
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

    root = Path(
        "data/demo_versioning"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_file = root / "observations.json"

    sample_file.write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "state": [0.0, 0.0],
                        "action": [0.1, 0.1],
                        "next_state": [0.1, 0.1],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    repository = (
        create_data_versioning_repository(
            dataset_root="data"
        )
    )

    dataset = repository.register_dataset(
        name=(
            "neuroworld_demo_"
            + uuid.uuid4().hex[:8]
        ),
        dataset_type="world_model",
        description=(
            "Synthetic world-model observations"
        ),
    )

    version = repository.create_version(
        dataset.dataset_id,
        version="v1.0.0",
        file_paths=[
            sample_file
        ],
        metadata={
            "split": "train",
            "synthetic": True,
        },
    )

    repository.tag_version(
        version.version_id,
        tag="baseline",
    )

    verification = repository.verify_version(
        version.version_id
    )

    print("=" * 60)
    print("NEUROWORLD AI - DATA VERSIONING")
    print("=" * 60)

    print(
        "Dataset:",
        dataset.name,
    )

    print(
        "Version:",
        version.version,
    )

    print(
        "Files:",
        version.file_count,
    )

    print(
        "Manifest SHA-256:",
        version.manifest_sha256,
    )

    print(
        "Integrity valid:",
        verification.valid,
    )

    output = repository.export_manifest(
        version.version_id,
        "experiments/reports/"
        "dataset_manifest.json",
    )

    print(
        "Manifest exported:",
        output,
    )

    close_database()


if __name__ == "__main__":
    main()
