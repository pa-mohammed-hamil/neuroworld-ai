
"""
NeuroWorld AI
backend/app/services/observation_service.py

Multimodal observation ingestion and preprocessing service.

Features:
- Numerical sensor observation validation
- Image observation ingestion
- Video metadata ingestion
- Temporal observation sequences
- Observation history and retrieval
- Numerical feature normalization
- Observation quality assessment
- Sensor anomaly detection
- Optional perception encoder integration
- Async-compatible service methods
- Thread-safe bounded in-memory storage

Dependencies:
    pip install numpy pydantic pillow

Notes:
- Image files are read from trusted server-side paths.
- Video ingestion stores validated metadata only.
- Raw image/video content is not persisted here.
- Neural embeddings require a registered encoder.
- Database persistence must be integrated separately.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import threading
import uuid

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from ..core.logging import get_logger


logger = get_logger("services.observation")


# ============================================================
# EXCEPTIONS
# ============================================================

class ObservationError(Exception):
    """Base observation service error."""


class ObservationValidationError(ObservationError):
    """Invalid observation data."""


class ObservationNotFoundError(ObservationError):
    """Observation record not found."""


class ObservationEncoderError(ObservationError):
    """Perception encoder failure."""


# ============================================================
# DATA MODELS
# ============================================================

class SensorObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    sensor_name: str = Field(
        min_length=1,
        max_length=128,
    )

    values: list[float] = Field(
        min_length=1,
        max_length=100000,
    )

    timestamp: datetime | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, values):
        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                "Sensor values must be finite"
            )

        return values


class ImageObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    image_path: str = Field(
        min_length=1,
        max_length=2048,
    )

    timestamp: datetime | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class VideoObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    video_id: str = Field(
        min_length=1,
        max_length=128,
    )

    frame_count: int = Field(ge=0)

    fps: float = Field(gt=0, le=1000)

    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)

    duration_seconds: float = Field(ge=0)

    timestamp: datetime | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class ObservationRecord(BaseModel):
    observation_id: str
    world_id: str
    observation_type: str
    timestamp: datetime
    created_at: datetime

    sensor_name: str | None = None
    values: list[float] | None = None

    image_sha256: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_mode: str | None = None

    video_id: str | None = None
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None

    embedding: list[float] | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class ObservationQuality(BaseModel):
    observation_id: str
    valid: bool
    completeness: float
    finite_ratio: float
    mean: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    warnings: list[str] = Field(
        default_factory=list,
    )


class SensorAnomalyResult(BaseModel):
    observation_id: str
    sensor_name: str
    anomaly_detected: bool
    anomaly_indices: list[int]
    z_scores: list[float]
    threshold: float


class ObservationSequence(BaseModel):
    world_id: str
    observations: list[ObservationRecord]
    count: int
    start_time: datetime | None = None
    end_time: datetime | None = None


# ============================================================
# PERCEPTION ENCODER INTERFACE
# ============================================================

class ObservationEncoder(Protocol):
    """
    Adapter for an actual trained perception model.

    Implement encode_sensor() and/or encode_image().
    """

    def encode_sensor(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        ...

    def encode_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        ...


# ============================================================
# NUMERICAL PROCESSING
# ============================================================

def normalize_sensor_values(
    values: list[float] | np.ndarray,
    *,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize a numerical sensor vector.

    Methods:
        zscore
        minmax
        l2

    For ML inference, prefer normalization parameters
    fitted on training data to prevent data leakage.
    """

    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1 or array.size == 0:
        raise ObservationValidationError(
            "Sensor values must be a nonempty vector"
        )

    if not np.all(np.isfinite(array)):
        raise ObservationValidationError(
            "Sensor values must be finite"
        )

    if method == "zscore":
        mean = np.mean(array)
        std = np.std(array)

        if std < 1e-12:
            return np.zeros_like(array)

        return (array - mean) / std

    if method == "minmax":
        minimum = np.min(array)
        maximum = np.max(array)

        difference = maximum - minimum

        if difference < 1e-12:
            return np.zeros_like(array)

        return (array - minimum) / difference

    if method == "l2":
        norm = np.linalg.norm(array)

        if norm < 1e-12:
            return np.zeros_like(array)

        return array / norm

    raise ObservationValidationError(
        f"Unsupported normalization method: {method}"
    )


def calculate_observation_quality(
    record: ObservationRecord,
) -> ObservationQuality:

    if record.values is None:
        return ObservationQuality(
            observation_id=record.observation_id,
            valid=True,
            completeness=1.0,
            finite_ratio=1.0,
            warnings=[
                "Numerical quality metrics are unavailable "
                "for this observation type"
            ],
        )

    array = np.asarray(
        record.values,
        dtype=np.float64,
    )

    finite_mask = np.isfinite(array)

    finite_ratio = float(
        np.mean(finite_mask)
    )

    valid_values = array[finite_mask]

    warnings = []

    if valid_values.size == 0:
        warnings.append(
            "No finite sensor values"
        )

        return ObservationQuality(
            observation_id=record.observation_id,
            valid=False,
            completeness=0.0,
            finite_ratio=0.0,
            warnings=warnings,
        )

    std = float(
        np.std(valid_values)
    )

    if std < 1e-12:
        warnings.append(
            "Sensor values have near-zero variance"
        )

    return ObservationQuality(
        observation_id=record.observation_id,
        valid=finite_ratio == 1.0,
        completeness=finite_ratio,
        finite_ratio=finite_ratio,
        mean=float(np.mean(valid_values)),
        standard_deviation=std,
        minimum=float(np.min(valid_values)),
        maximum=float(np.max(valid_values)),
        warnings=warnings,
    )


def detect_sensor_anomalies(
    record: ObservationRecord,
    *,
    threshold: float = 3.0,
) -> SensorAnomalyResult:
    """
    Basic within-vector z-score anomaly detection.

    This is a statistical heuristic, not a trained
    time-series anomaly detector.
    """

    if record.values is None:
        raise ObservationValidationError(
            "Anomaly detection requires sensor values"
        )

    if (
        not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ObservationValidationError(
            "Threshold must be positive"
        )

    array = np.asarray(
        record.values,
        dtype=np.float64,
    )

    mean = np.mean(array)
    std = np.std(array)

    if std < 1e-12:
        scores = np.zeros_like(array)
    else:
        scores = np.abs(
            (array - mean) / std
        )

    indices = np.flatnonzero(
        scores > threshold
    ).tolist()

    return SensorAnomalyResult(
        observation_id=record.observation_id,
        sensor_name=record.sensor_name or "unknown",
        anomaly_detected=bool(indices),
        anomaly_indices=indices,
        z_scores=scores.tolist(),
        threshold=threshold,
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def load_image_securely(
    image_path: str,
    *,
    allowed_directory: Path,
    max_pixels: int = 40_000_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load an image from a restricted server-side directory.

    Avoid accepting arbitrary filesystem paths from
    untrusted API callers.
    """

    from PIL import Image, ImageOps, UnidentifiedImageError

    root = allowed_directory.resolve()
    path = Path(image_path).resolve()

    if not path.is_relative_to(root):
        raise ObservationValidationError(
            "Image path is outside the allowed directory"
        )

    if not path.is_file():
        raise ObservationValidationError(
            "Image file not found"
        )

    if path.suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        raise ObservationValidationError(
            "Unsupported image format"
        )

    if path.stat().st_size > 50 * 1024 * 1024:
        raise ObservationValidationError(
            "Image file exceeds 50 MB"
        )

    Image.MAX_IMAGE_PIXELS = max_pixels

    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)

            if image.width * image.height > max_pixels:
                raise ObservationValidationError(
                    "Image resolution exceeds limit"
                )

            rgb = image.convert("RGB")
            pixels = np.asarray(rgb).copy()

            info = {
                "width": rgb.width,
                "height": rgb.height,
                "mode": rgb.mode,
            }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise ObservationValidationError(
            "Image could not be decoded safely"
        ) from exc

    return pixels, info


def image_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(
        image.tobytes()
    ).hexdigest()


# ============================================================
# OBSERVATION SERVICE
# ============================================================

class ObservationService:
    """
    Thread-safe bounded observation store.

    Observation records are held in process memory.

    Multi-user deployments must add world ownership
    authorization and persistent storage.
    """

    def __init__(
        self,
        *,
        encoder: ObservationEncoder | None = None,
        allowed_image_directory: str | Path = "data/uploads",
        max_observations_per_world: int = 1000,
    ) -> None:

        if max_observations_per_world < 1:
            raise ValueError(
                "max_observations_per_world must be positive"
            )

        self.encoder = encoder

        self.allowed_image_directory = Path(
            allowed_image_directory
        ).resolve()

        self.max_observations_per_world = (
            max_observations_per_world
        )

        self._observations: dict[
            str,
            ObservationRecord,
        ] = {}

        self._world_history: dict[
            str,
            deque[str],
        ] = defaultdict(deque)

        self._lock = threading.RLock()

    def register_encoder(
        self,
        encoder: ObservationEncoder,
    ) -> None:
        self.encoder = encoder

        logger.info(
            "Observation encoder registered"
        )

    def _encode_sensor(
        self,
        values: np.ndarray,
    ) -> list[float] | None:

        if self.encoder is None:
            return None

        try:
            encoded = np.asarray(
                self.encoder.encode_sensor(values),
                dtype=np.float64,
            )

            if (
                encoded.ndim != 1
                or encoded.size == 0
                or not np.all(np.isfinite(encoded))
            ):
                raise ObservationEncoderError(
                    "Encoder returned invalid embedding"
                )

            return encoded.tolist()

        except ObservationEncoderError:
            raise

        except Exception as exc:
            logger.exception(
                "Sensor encoding failed"
            )

            raise ObservationEncoderError(
                "Sensor encoder failed"
            ) from exc

    def _encode_image(
        self,
        image: np.ndarray,
    ) -> list[float] | None:

        if self.encoder is None:
            return None

        try:
            encoded = np.asarray(
                self.encoder.encode_image(image),
                dtype=np.float64,
            )

            if (
                encoded.ndim != 1
                or encoded.size == 0
                or not np.all(np.isfinite(encoded))
            ):
                raise ObservationEncoderError(
                    "Image encoder returned invalid embedding"
                )

            return encoded.tolist()

        except ObservationEncoderError:
            raise

        except Exception as exc:
            logger.exception(
                "Image encoding failed"
            )

            raise ObservationEncoderError(
                "Image encoder failed"
            ) from exc

    def _store(
        self,
        record: ObservationRecord,
    ) -> ObservationRecord:

        with self._lock:
            history = self._world_history[
                record.world_id
            ]

            while (
                len(history)
                >= self.max_observations_per_world
            ):
                oldest_id = history.popleft()

                self._observations.pop(
                    oldest_id,
                    None,
                )

            self._observations[
                record.observation_id
            ] = record

            history.append(
                record.observation_id
            )

        return record

    def ingest_sensor(
        self,
        request: SensorObservationRequest,
    ) -> ObservationRecord:

        values = np.asarray(
            request.values,
            dtype=np.float64,
        )

        embedding = self._encode_sensor(
            values
        )

        now = datetime.now(timezone.utc)

        record = ObservationRecord(
            observation_id=uuid.uuid4().hex,
            world_id=request.world_id,
            observation_type="sensor",
            timestamp=request.timestamp or now,
            created_at=now,
            sensor_name=request.sensor_name,
            values=values.tolist(),
            embedding=embedding,
            metadata=request.metadata,
        )

        self._store(record)

        logger.info(
            "Sensor observation ingested: %s",
            record.observation_id,
        )

        return record

    def ingest_image(
        self,
        request: ImageObservationRequest,
    ) -> ObservationRecord:

        pixels, image_info = load_image_securely(
            request.image_path,
            allowed_directory=(
                self.allowed_image_directory
            ),
        )

        embedding = self._encode_image(
            pixels
        )

        now = datetime.now(timezone.utc)

        record = ObservationRecord(
            observation_id=uuid.uuid4().hex,
            world_id=request.world_id,
            observation_type="image",
            timestamp=request.timestamp or now,
            created_at=now,
            image_sha256=image_sha256(pixels),
            image_width=image_info["width"],
            image_height=image_info["height"],
            image_mode=image_info["mode"],
            embedding=embedding,
            metadata=request.metadata,
        )

        self._store(record)

        logger.info(
            "Image observation ingested: %s",
            record.observation_id,
        )

        return record

    def ingest_video_metadata(
        self,
        request: VideoObservationRequest,
    ) -> ObservationRecord:
        """
        Record video metadata only.

        Actual frame extraction belongs in the
        video perception pipeline.
        """

        expected_duration = (
            request.frame_count / request.fps
        )

        tolerance = max(
            0.5,
            2.0 / request.fps,
        )

        if abs(
            request.duration_seconds
            - expected_duration
        ) > tolerance:
            raise ObservationValidationError(
                "Video duration is inconsistent with "
                "frame count and FPS"
            )

        now = datetime.now(timezone.utc)

        record = ObservationRecord(
            observation_id=uuid.uuid4().hex,
            world_id=request.world_id,
            observation_type="video",
            timestamp=request.timestamp or now,
            created_at=now,
            video_id=request.video_id,
            frame_count=request.frame_count,
            fps=request.fps,
            duration_seconds=request.duration_seconds,
            image_width=request.width,
            image_height=request.height,
            metadata=request.metadata,
        )

        self._store(record)

        logger.info(
            "Video metadata ingested: %s",
            record.observation_id,
        )

        return record

    def get_observation(
        self,
        observation_id: str,
    ) -> ObservationRecord:

        with self._lock:
            record = self._observations.get(
                observation_id
            )

        if record is None:
            raise ObservationNotFoundError(
                "Observation not found"
            )

        return record

    def list_observations(
        self,
        *,
        world_id: str,
        observation_type: str | None = None,
        limit: int = 100,
    ) -> list[ObservationRecord]:

        if not 1 <= limit <= 1000:
            raise ObservationValidationError(
                "limit must be between 1 and 1000"
            )

        with self._lock:
            ids = list(
                self._world_history.get(
                    world_id,
                    (),
                )
            )

            records = [
                self._observations[observation_id]
                for observation_id in ids
                if observation_id in self._observations
            ]

        if observation_type is not None:
            records = [
                record
                for record in records
                if record.observation_type
                == observation_type
            ]

        records.sort(
            key=lambda item: item.timestamp,
            reverse=True,
        )

        return records[:limit]

    def get_sequence(
        self,
        *,
        world_id: str,
        limit: int = 100,
    ) -> ObservationSequence:

        records = self.list_observations(
            world_id=world_id,
            limit=limit,
        )

        records.sort(
            key=lambda item: item.timestamp
        )

        return ObservationSequence(
            world_id=world_id,
            observations=records,
            count=len(records),
            start_time=(
                records[0].timestamp
                if records
                else None
            ),
            end_time=(
                records[-1].timestamp
                if records
                else None
            ),
        )

    def normalize_observation(
        self,
        observation_id: str,
        *,
        method: str = "zscore",
    ) -> list[float]:

        record = self.get_observation(
            observation_id
        )

        if record.values is None:
            raise ObservationValidationError(
                "Observation has no numerical values"
            )

        normalized = normalize_sensor_values(
            record.values,
            method=method,
        )

        return normalized.tolist()

    def assess_quality(
        self,
        observation_id: str,
    ) -> ObservationQuality:

        record = self.get_observation(
            observation_id
        )

        return calculate_observation_quality(
            record
        )

    def detect_anomalies(
        self,
        observation_id: str,
        *,
        threshold: float = 3.0,
    ) -> SensorAnomalyResult:

        record = self.get_observation(
            observation_id
        )

        return detect_sensor_anomalies(
            record,
            threshold=threshold,
        )

    async def ingest_sensor_async(
        self,
        request: SensorObservationRequest,
    ) -> ObservationRecord:

        return await asyncio.to_thread(
            self.ingest_sensor,
            request,
        )

    async def ingest_image_async(
        self,
        request: ImageObservationRequest,
    ) -> ObservationRecord:

        return await asyncio.to_thread(
            self.ingest_image,
            request,
        )

    def status(self) -> dict[str, Any]:

        with self._lock:
            observation_count = len(
                self._observations
            )

            world_count = len(
                self._world_history
            )

        return {
            "service": "observation",
            "status": "ready",
            "encoder_registered": (
                self.encoder is not None
            ),
            "observation_count": observation_count,
            "world_count": world_count,
            "max_observations_per_world": (
                self.max_observations_per_world
            ),
            "storage": "in_memory",
        }


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_observation_service(
    *,
    encoder: ObservationEncoder | None = None,
) -> ObservationService:

    project_root = (
        Path(__file__).resolve().parents[3]
    )

    return ObservationService(
        encoder=encoder,
        allowed_image_directory=(
            project_root / "data" / "uploads"
        ),
        max_observations_per_world=1000,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:

    service = create_observation_service()

    first = service.ingest_sensor(
        SensorObservationRequest(
            world_id="demo-world-001",
            sensor_name="position",
            values=[0.0, 1.0, 2.0, 3.0],
        )
    )

    second = service.ingest_sensor(
        SensorObservationRequest(
            world_id="demo-world-001",
            sensor_name="velocity",
            values=[0.1, 0.2, 0.3, 0.4],
        )
    )

    print("=" * 65)
    print("NEUROWORLD AI — OBSERVATION SERVICE")
    print("=" * 65)

    print()
    print("First observation:")
    print(first.model_dump_json(indent=2))

    print()
    print("Second observation:")
    print(second.model_dump_json(indent=2))

    print()
    print("Normalized values:")
    print(
        service.normalize_observation(
            first.observation_id
        )
    )

    print()
    print("Quality assessment:")
    print(
        service.assess_quality(
            first.observation_id
        ).model_dump_json(indent=2)
    )

    print()
    print("Temporal sequence:")
    print(
        service.get_sequence(
            world_id="demo-world-001"
        ).model_dump_json(indent=2)
    )

    print()
    print("Service status:")
    print(service.status())

    print()
    print(
        "Demonstration completed using synthetic "
        "sensor observations."
    )


if __name__ == "__main__":
    main()
