
```python
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/download_datasets.py

Dataset Download and Management Pipeline

Features:
- Download datasets from HTTPS URLs
- Dataset registry using JSON
- Resumable downloads
- Retry and timeout handling
- SHA-256 integrity verification
- ZIP / TAR / TAR.GZ extraction
- Archive path traversal protection
- Dataset version tracking
- Download progress reporting
- Dataset inventory
- CLI commands

Dependencies:
    pip install requests

Usage:
    python scripts/download_datasets.py list

    python scripts/download_datasets.py download \
        --name sample_navigation

    python scripts/download_datasets.py download \
        --all

    python scripts/download_datasets.py verify \
        --name sample_navigation

    python scripts/download_datasets.py inventory

Dataset registry:
    configs/datasets.json

Downloads:
    data/raw/

Important:
    Register only trusted dataset sources.
    Review licenses before downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import zipfile

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests


LOGGER = logging.getLogger(
    "neuroworld.dataset_downloader"
)

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "configs"
    / "datasets.json"
)

DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

CHUNK_SIZE = 1024 * 1024

SUPPORTED_ARCHIVES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
)


@dataclass
class DatasetDefinition:
    name: str
    url: str

    version: str = "1.0"
    filename: Optional[str] = None

    sha256: Optional[str] = None

    archive_type: Optional[str] = None
    extract: bool = False

    license: str = "unknown"
    description: str = ""

    category: str = "general"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class DownloadConfig:
    data_dir: str = str(
        DEFAULT_DATA_DIR
    )

    timeout: int = 60
    retries: int = 3
    retry_delay: float = 2.0

    chunk_size: int = CHUNK_SIZE

    resume: bool = True
    verify_checksum: bool = True

    max_download_bytes: int = (
        20 * 1024**3
    )

    max_extract_bytes: int = (
        50 * 1024**3
    )

    max_archive_members: int = 100000


@dataclass
class DownloadResult:
    dataset_name: str
    version: str

    success: bool

    file_path: Optional[str] = None
    extracted_path: Optional[str] = None

    bytes_downloaded: int = 0
    sha256: Optional[str] = None

    elapsed_seconds: float = 0.0

    message: str = ""


class DatasetRegistry:
    """Load and validate dataset definitions."""

    def __init__(
        self,
        registry_path: str | Path = DEFAULT_REGISTRY,
    ):
        self.registry_path = Path(
            registry_path
        )

        self.datasets: dict[
            str, DatasetDefinition
        ] = {}

        self.load()

    @staticmethod
    def _safe_component(value: str) -> str:
        if (
            not value
            or value in (".", "..")
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError(
                f"Invalid dataset name/version: {value!r}"
            )

        return value

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme != "https":
            raise ValueError(
                "Only HTTPS dataset URLs are supported"
            )

        if not parsed.hostname:
            raise ValueError(
                "Dataset URL must have a hostname"
            )

        if parsed.username or parsed.password:
            raise ValueError(
                "Credentials in dataset URLs "
                "are not supported"
            )

    def load(self) -> None:
        if not self.registry_path.exists():
            LOGGER.warning(
                "Dataset registry not found: %s",
                self.registry_path,
            )
            return

        payload = json.loads(
            self.registry_path.read_text(
                encoding="utf-8"
            )
        )

        definitions = payload.get(
            "datasets", []
        )

        for item in definitions:
            definition = DatasetDefinition(
                **item
            )

            self._safe_component(
                definition.name
            )

            self._safe_component(
                definition.version
            )

            self._validate_url(
                definition.url
            )

            if definition.filename is not None:
                self._safe_component(
                    definition.filename
                )

            if definition.sha256 is not None:
                checksum = (
                    definition.sha256.lower()
                )

                if (
                    len(checksum) != 64
                    or any(
                        character
                        not in "0123456789abcdef"
                        for character in checksum
                    )
                ):
                    raise ValueError(
                        "Invalid SHA-256 checksum"
                    )

            if definition.name in self.datasets:
                raise ValueError(
                    "Duplicate dataset name: "
                    f"{definition.name}"
                )

            self.datasets[
                definition.name
            ] = definition

    def get(
        self,
        name: str,
    ) -> DatasetDefinition:
        if name not in self.datasets:
            raise KeyError(
                f"Dataset not registered: {name}"
            )

        return self.datasets[name]

    def list(
        self,
    ) -> list[DatasetDefinition]:
        return list(
            self.datasets.values()
        )


class DatasetDownloader:
    """
    Download and extract registered datasets.

    Network downloads require explicit
    dataset URLs in the registry.

    This module does not automatically
    scrape websites or bypass access controls.
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        config: Optional[
            DownloadConfig
        ] = None,
    ):
        self.registry = registry

        self.config = (
            config or DownloadConfig()
        )

        self.data_dir = Path(
            self.config.data_dir
        ).resolve()

        self.data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "User-Agent": (
                    "NeuroWorldAI-DatasetManager/1.0"
                )
            }
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    @staticmethod
    def calculate_sha256(
        path: str | Path,
    ) -> str:
        digest = hashlib.sha256()

        with open(path, "rb") as stream:
            while True:
                chunk = stream.read(
                    CHUNK_SIZE
                )

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _archive_type(
        definition: DatasetDefinition,
        filename: str,
    ) -> Optional[str]:
        if definition.archive_type:
            return definition.archive_type.lower()

        lower = filename.lower()

        if lower.endswith(".tar.gz"):
            return "tar.gz"

        if lower.endswith(".tgz"):
            return "tgz"

        if lower.endswith(".tar"):
            return "tar"

        if lower.endswith(".zip"):
            return "zip"

        return None

    def _dataset_dir(
        self,
        definition: DatasetDefinition,
    ) -> Path:
        directory = (
            self.data_dir
            / definition.name
            / definition.version
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @staticmethod
    def _filename(
        definition: DatasetDefinition,
    ) -> str:
        if definition.filename:
            return definition.filename

        name = Path(
            urlparse(
                definition.url
            ).path
        ).name

        if not name:
            raise ValueError(
                "Specify filename for URL "
                "without a file basename"
            )

        DatasetRegistry._safe_component(
            name
        )

        return name

    def _download_once(
        self,
        definition: DatasetDefinition,
        destination: Path,
    ) -> int:
        """
        Download to a .part file.

        Resume is attempted only when
        the server confirms a valid
        HTTP 206 Content-Range response.
        """

        partial = destination.with_name(
            destination.name + ".part"
        )

        existing = (
            partial.stat().st_size
            if (
                self.config.resume
                and partial.exists()
            )
            else 0
        )

        if existing > self.config.max_download_bytes:
            raise RuntimeError(
                "Partial file exceeds size limit"
            )

        headers = {}

        if existing:
            headers["Range"] = (
                f"bytes={existing}-"
            )

        response = self.session.get(
            definition.url,
            headers=headers,
            stream=True,
            timeout=self.config.timeout,
            allow_redirects=False,
        )

        try:
            if 300 <= response.status_code < 400:
                raise RuntimeError(
                    "Redirects are disabled. "
                    "Register the final trusted HTTPS URL."
                )

            if response.status_code == 416:
                raise RuntimeError(
                    "Server rejected resume range"
                )

            response.raise_for_status()

            append = False

            if existing and response.status_code == 206:
                content_range = response.headers.get(
                    "Content-Range", ""
                )

                if not content_range.startswith(
                    f"bytes {existing}-"
                ):
                    raise RuntimeError(
                        "Invalid resume Content-Range"
                    )

                append = True

            elif response.status_code == 206:
                raise RuntimeError(
                    "Unexpected partial response"
                )

            mode = (
                "ab" if append else "wb"
            )

            total_bytes = (
                existing if append else 0
            )

            with open(partial, mode) as stream:
                for chunk in response.iter_content(
                    chunk_size=(
                        self.config.chunk_size
                    )
                ):
                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if (
                        total_bytes
                        > self.config.max_download_bytes
                    ):
                        raise RuntimeError(
                            "Download exceeds "
                            "configured size limit"
                        )

                    stream.write(chunk)

            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length is not None:
                expected = int(
                    content_length
                )

                actual = (
                    total_bytes
                    - existing
                    if append
                    else total_bytes
                )

                if actual != expected:
                    raise RuntimeError(
                        "Incomplete HTTP response"
                    )

            os.replace(
                partial,
                destination,
            )

            return total_bytes

        finally:
            response.close()

    def _download_with_retries(
        self,
        definition: DatasetDefinition,
        destination: Path,
    ) -> int:
        last_error: Optional[
            Exception
        ] = None

        for attempt in range(
            self.config.retries + 1
        ):
            try:
                return self._download_once(
                    definition,
                    destination,
                )

            except (
                requests.RequestException,
                OSError,
                RuntimeError,
            ) as exc:
                last_error = exc

                LOGGER.warning(
                    "Download attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )

                if attempt >= self.config.retries:
                    break

                time.sleep(
                    self.config.retry_delay
                    * (2 ** attempt)
                )

        raise RuntimeError(
            "Dataset download failed"
        ) from last_error

    @staticmethod
    def _safe_member_path(
        destination: Path,
        member_name: str,
    ) -> Path:
        """
        Reject absolute paths and paths
        escaping the extraction directory.
        """

        normalized = member_name.replace(
            "\\", "/"
        )

        if (
            normalized.startswith("/")
            or "\x00" in normalized
            or (
                len(normalized) >= 2
                and normalized[1] == ":"
            )
        ):
            raise ValueError(
                "Unsafe archive member path"
            )

        target = (
            destination / normalized
        ).resolve()

        if not target.is_relative_to(
            destination.resolve()
        ):
            raise ValueError(
                "Archive path traversal detected"
            )

        return target

    def _extract_zip(
        self,
        archive: Path,
        destination: Path,
    ) -> None:
        with zipfile.ZipFile(
            archive
        ) as source:
            members = source.infolist()

            if (
                len(members)
                > self.config.max_archive_members
            ):
                raise RuntimeError(
                    "Archive contains too many members"
                )

            total_size = sum(
                member.file_size
                for member in members
            )

            if (
                total_size
                > self.config.max_extract_bytes
            ):
                raise RuntimeError(
                    "Archive exceeds extraction limit"
                )

            for member in members:
                target = self._safe_member_path(
                    destination,
                    member.filename,
                )

                # Reject Unix symbolic links.
                unix_mode = (
                    member.external_attr >> 16
                )

                if (
                    (unix_mode & 0o170000)
                    == 0o120000
                ):
                    raise ValueError(
                        "ZIP symbolic links "
                        "are not permitted"
                    )

                if member.is_dir():
                    target.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    continue

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with source.open(member) as input_file:
                    with open(
                        target, "wb"
                    ) as output_file:
                        shutil.copyfileobj(
                            input_file,
                            output_file,
                        )

    def _extract_tar(
        self,
        archive: Path,
        destination: Path,
    ) -> None:
        with tarfile.open(
            archive,
            mode="r:*",
        ) as source:
            members = source.getmembers()

            if (
                len(members)
                > self.config.max_archive_members
            ):
                raise RuntimeError(
                    "Archive contains too many members"
                )

            total_size = sum(
                member.size
                for member in members
                if member.isfile()
            )

            if (
                total_size
                > self.config.max_extract_bytes
            ):
                raise RuntimeError(
                    "Archive exceeds extraction limit"
                )

            for member in members:
                target = self._safe_member_path(
                    destination,
                    member.name,
                )

                if member.isdir():
                    target.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    continue

                if not member.isfile():
                    raise ValueError(
                        "Archive links and special "
                        "files are not permitted"
                    )

                target.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                input_file = source.extractfile(
                    member
                )

                if input_file is None:
                    raise RuntimeError(
                        "Could not read archive member"
                    )

                with input_file:
                    with open(
                        target, "wb"
                    ) as output_file:
                        shutil.copyfileobj(
                            input_file,
                            output_file,
                        )

    def extract(
        self,
        archive: Path,
        destination: Path,
        archive_type: str,
    ) -> None:
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        if archive_type == "zip":
            self._extract_zip(
                archive,
                destination,
            )

        elif archive_type in (
            "tar",
            "tar.gz",
            "tgz",
        ):
            self._extract_tar(
                archive,
                destination,
            )

        else:
            raise ValueError(
                "Unsupported archive type: "
                f"{archive_type}"
            )

    def download(
        self,
        name: str,
        force: bool = False,
    ) -> DownloadResult:
        definition = self.registry.get(
            name
        )

        started = time.monotonic()

        directory = self._dataset_dir(
            definition
        )

        filename = self._filename(
            definition
        )

        destination = (
            directory / filename
        )

        extracted_path = None

        try:
            if force:
                destination.unlink(
                    missing_ok=True
                )

                destination.with_name(
                    destination.name + ".part"
                ).unlink(
                    missing_ok=True
                )

            if destination.exists():
                LOGGER.info(
                    "Dataset file already exists: %s",
                    destination,
                )

                downloaded_bytes = (
                    destination.stat().st_size
                )
            else:
                downloaded_bytes = (
                    self._download_with_retries(
                        definition,
                        destination,
                    )
                )

            checksum = self.calculate_sha256(
                destination
            )

            if (
                self.config.verify_checksum
                and definition.sha256
                and checksum.lower()
                != definition.sha256.lower()
            ):
                raise RuntimeError(
                    "SHA-256 verification failed"
                )

            if definition.extract:
                archive_type = (
                    self._archive_type(
                        definition,
                        filename,
                    )
                )

                if archive_type is None:
                    raise ValueError(
                        "Extraction requested but "
                        "archive type is unknown"
                    )

                extract_directory = (
                    directory / "extracted"
                )

                # Extract into a fresh staging
                # directory before publishing.
                staging = (
                    directory / ".extracting"
                )

                if staging.exists():
                    shutil.rmtree(staging)

                staging.mkdir(
                    parents=True
                )

                try:
                    self.extract(
                        destination,
                        staging,
                        archive_type,
                    )

                    if extract_directory.exists():
                        shutil.rmtree(
                            extract_directory
                        )

                    staging.rename(
                        extract_directory
                    )

                finally:
                    if staging.exists():
                        shutil.rmtree(staging)

                extracted_path = str(
                    extract_directory
                )

            manifest = {
                "name": definition.name,
                "version": definition.version,
                "source_url": definition.url,
                "license": definition.license,
                "category": definition.category,
                "filename": filename,
                "sha256": checksum,
                "file_size_bytes": (
                    destination.stat().st_size
                ),
                "extracted_path": extracted_path,
                "downloaded_at": time.time(),
            }

            (
                directory / "manifest.json"
            ).write_text(
                json.dumps(
                    manifest,
                    indent=2,
                ),
                encoding="utf-8",
            )

            return DownloadResult(
                dataset_name=name,
                version=definition.version,
                success=True,
                file_path=str(destination),
                extracted_path=extracted_path,
                bytes_downloaded=downloaded_bytes,
                sha256=checksum,
                elapsed_seconds=(
                    time.monotonic() - started
                ),
                message="Dataset ready",
            )

        except Exception as exc:
            LOGGER.exception(
                "Dataset %s failed",
                name,
            )

            return DownloadResult(
                dataset_name=name,
                version=definition.version,
                success=False,
                elapsed_seconds=(
                    time.monotonic() - started
                ),
                message=str(exc),
            )

    def download_all(
        self,
        force: bool = False,
    ) -> list[DownloadResult]:
        return [
            self.download(
                definition.name,
                force=force,
            )
            for definition
            in self.registry.list()
        ]

    def verify(
        self,
        name: str,
    ) -> dict[str, Any]:
        definition = self.registry.get(
            name
        )

        directory = self._dataset_dir(
            definition
        )

        destination = (
            directory
            / self._filename(definition)
        )

        if not destination.exists():
            return {
                "dataset": name,
                "exists": False,
                "verified": False,
                "message": "File not found",
            }

        checksum = self.calculate_sha256(
            destination
        )

        expected = definition.sha256

        return {
            "dataset": name,
            "exists": True,
            "verified": (
                checksum.lower()
                == expected.lower()
                if expected
                else None
            ),
            "sha256": checksum,
            "expected_sha256": expected,
            "size_bytes": (
                destination.stat().st_size
            ),
            "message": (
                "Checksum not registered"
                if expected is None
                else (
                    "Checksum matches"
                    if checksum.lower()
                    == expected.lower()
                    else "Checksum mismatch"
                )
            ),
        }

    def inventory(
        self,
    ) -> list[dict[str, Any]]:
        results = []

        for definition in self.registry.list():
            directory = (
                self._dataset_dir(
                    definition
                )
            )

            file_path = (
                directory
                / self._filename(
                    definition
                )
            )

            results.append(
                {
                    "name": definition.name,
                    "version": definition.version,
                    "category": definition.category,
                    "installed": (
                        file_path.exists()
                    ),
                    "file_path": (
                        str(file_path)
                        if file_path.exists()
                        else None
                    ),
                    "size_bytes": (
                        file_path.stat().st_size
                        if file_path.exists()
                        else 0
                    ),
                }
            )

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI dataset downloader"
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "list",
            "download",
            "verify",
            "inventory",
        ],
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--registry",
        type=str,
        default=str(
            DEFAULT_REGISTRY
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_DATA_DIR
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    registry = DatasetRegistry(
        args.registry
    )

    with DatasetDownloader(
        registry=registry,
        config=DownloadConfig(
            data_dir=args.output
        ),
    ) as downloader:

        if args.command == "list":
            for dataset in registry.list():
                print(
                    f"{dataset.name:25} "
                    f"{dataset.version:10} "
                    f"{dataset.category:15} "
                    f"{dataset.license}"
                )

            return 0

        if args.command == "download":
            if args.all:
                results = (
                    downloader.download_all(
                        force=args.force
                    )
                )

            elif args.name:
                results = [
                    downloader.download(
                        args.name,
                        force=args.force,
                    )
                ]

            else:
                raise SystemExit(
                    "Specify --name or --all"
                )

            for result in results:
                print(
                    json.dumps(
                        asdict(result),
                        indent=2,
                    )
                )

            return (
                0
                if all(
                    result.success
                    for result in results
                )
                else 1
            )

        if args.command == "verify":
            if not args.name:
                raise SystemExit(
                    "Specify --name"
                )

            result = downloader.verify(
                args.name
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                )
            )

            return (
                0
                if result["verified"] is True
                else 1
            )

        if args.command == "inventory":
            print(
                json.dumps(
                    downloader.inventory(),
                    indent=2,
                )
            )

            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
