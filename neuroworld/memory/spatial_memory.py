
"""
NeuroWorld AI
File: neuroworld/memory/spatial_memory.py

Spatial Memory & World-State Mapping Engine

Features:
    - 2D and 3D spatial memory
    - Object position and velocity tracking
    - Timestamped observations
    - Confidence-weighted updates
    - Spatial nearest-neighbor search
    - Radius-based object retrieval
    - Semantic category filtering
    - Object movement history
    - Object permanence / stale tracking
    - Grid occupancy mapping
    - Occupancy probability updates
    - Memory capacity management
    - Spatial change detection
    - Scene snapshots
    - JSON serialization
    - Thread-safe memory operations
    - PyTorch feature export
    - Agent and planner integration

Dependencies:
    pip install numpy torch

Run:
    python -m neuroworld.memory.spatial_memory
"""

from __future__ import annotations

import json
import math
import threading
import time

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor


EPSILON = 1e-9


# ============================================================
# EXCEPTIONS
# ============================================================

class SpatialMemoryError(Exception):
    """Base spatial memory exception."""


class SpatialValidationError(SpatialMemoryError):
    """Invalid spatial data."""


# ============================================================
# ENUMERATIONS
# ============================================================

class SpatialObjectStatus(str, Enum):
    ACTIVE = "active"
    STALE = "stale"
    LOST = "lost"


class SpatialChangeType(str, Enum):
    ADDED = "added"
    MOVED = "moved"
    UPDATED = "updated"
    REMOVED = "removed"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SpatialMemoryConfig:
    spatial_dim: int = 2

    max_objects: int = 10000
    max_history_per_object: int = 100

    confidence_threshold: float = 0.1

    stale_after_seconds: float = 30.0
    lost_after_seconds: float = 300.0

    movement_threshold: float = 0.05

    position_smoothing: float = 0.6

    occupancy_resolution: float = 0.5
    occupancy_width: int = 200
    occupancy_height: int = 200

    occupancy_origin_x: float = -50.0
    occupancy_origin_y: float = -50.0

    occupancy_hit_probability: float = 0.7
    occupancy_miss_probability: float = 0.3

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if (
            self.max_objects <= 0
            or self.max_history_per_object <= 0
        ):
            raise ValueError(
                "Memory capacities must be positive"
            )

        if not (
            0 <= self.confidence_threshold <= 1
        ):
            raise ValueError(
                "confidence_threshold must be in [0,1]"
            )

        if not (
            0 < self.stale_after_seconds
            < self.lost_after_seconds
        ):
            raise ValueError(
                "Invalid stale/lost time thresholds"
            )

        if self.movement_threshold < 0:
            raise ValueError(
                "movement_threshold must be nonnegative"
            )

        if not (
            0 <= self.position_smoothing <= 1
        ):
            raise ValueError(
                "position_smoothing must be in [0,1]"
            )

        if self.occupancy_resolution <= 0:
            raise ValueError(
                "occupancy_resolution must be positive"
            )

        if (
            self.occupancy_width <= 0
            or self.occupancy_height <= 0
        ):
            raise ValueError(
                "Occupancy dimensions must be positive"
            )

        if not (
            0 < self.occupancy_hit_probability < 1
            and 0 < self.occupancy_miss_probability < 1
        ):
            raise ValueError(
                "Occupancy probabilities must be in (0,1)"
            )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def as_vector(
    value: Any,
    dimension: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=np.float64,
    )

    if vector.shape != (dimension,):
        raise SpatialValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(vector).all():
        raise SpatialValidationError(
            f"{name} contains non-finite values"
        )

    return vector


def validate_confidence(
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value) or not 0 <= value <= 1:
        raise SpatialValidationError(
            "Confidence must be in [0,1]"
        )

    return value


def validate_timestamp(
    value: float | None,
) -> float:
    timestamp = (
        time.time()
        if value is None
        else float(value)
    )

    if not math.isfinite(timestamp):
        raise SpatialValidationError(
            "Timestamp must be finite"
        )

    return timestamp


def sigmoid(
    value: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(
        value,
        -60.0,
        60.0,
    )

    return 1.0 / (
        1.0 + np.exp(-clipped)
    )


def logit(
    probability: float,
) -> float:
    probability = float(
        np.clip(
            probability,
            1e-6,
            1 - 1e-6,
        )
    )

    return math.log(
        probability / (1 - probability)
    )


# ============================================================
# SPATIAL OBSERVATION
# ============================================================

@dataclass
class SpatialObservation:
    object_id: str

    position: np.ndarray | list[float]

    velocity: np.ndarray | list[float] | None = None

    category: str = "unknown"

    confidence: float = 1.0

    timestamp: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.object_id:
            raise SpatialValidationError(
                "object_id cannot be empty"
            )

        self.position = np.asarray(
            self.position,
            dtype=np.float64,
        )

        if self.position.shape not in (
            (2,),
            (3,),
        ):
            raise SpatialValidationError(
                "Position must be 2D or 3D"
            )

        if not np.isfinite(
            self.position
        ).all():
            raise SpatialValidationError(
                "Position must be finite"
            )

        if self.velocity is not None:
            self.velocity = as_vector(
                self.velocity,
                len(self.position),
                "velocity",
            )

        self.confidence = validate_confidence(
            self.confidence
        )

        self.timestamp = validate_timestamp(
            self.timestamp
        )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise SpatialValidationError(
                "metadata must be a dictionary"
            )


# ============================================================
# SPATIAL HISTORY ENTRY
# ============================================================

@dataclass
class SpatialHistoryEntry:
    position: np.ndarray
    velocity: np.ndarray

    timestamp: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


# ============================================================
# SPATIAL MEMORY OBJECT
# ============================================================

@dataclass
class SpatialMemoryObject:
    object_id: str

    position: np.ndarray
    velocity: np.ndarray

    category: str

    confidence: float

    first_seen: float
    last_seen: float

    status: SpatialObjectStatus = (
        SpatialObjectStatus.ACTIVE
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    history: deque[SpatialHistoryEntry] = field(
        default_factory=deque
    )

    def distance_to(
        self,
        position: np.ndarray,
    ) -> float:
        return float(
            np.linalg.norm(
                self.position - position
            )
        )

    def predict_position(
        self,
        timestamp: float,
    ) -> np.ndarray:
        """
        Constant-velocity position estimate.

        This is a prediction, not an
        observed position.
        """
        dt = max(
            0.0,
            timestamp - self.last_seen,
        )

        return (
            self.position
            + self.velocity * dt
        )

    def to_dict(
        self,
        include_history: bool = True,
    ) -> dict[str, Any]:
        data = {
            "object_id": self.object_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "category": self.category,
            "confidence": self.confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "status": self.status.value,
            "metadata": self.metadata,
        }

        if include_history:
            data["history"] = [
                entry.to_dict()
                for entry in self.history
            ]

        return data


# ============================================================
# SPATIAL CHANGE EVENT
# ============================================================

@dataclass
class SpatialChangeEvent:
    object_id: str
    change_type: SpatialChangeType

    timestamp: float

    previous_position: np.ndarray | None = None
    current_position: np.ndarray | None = None

    distance_moved: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp,
            "previous_position": (
                None
                if self.previous_position is None
                else self.previous_position.tolist()
            ),
            "current_position": (
                None
                if self.current_position is None
                else self.current_position.tolist()
            ),
            "distance_moved": self.distance_moved,
        }


# ============================================================
# OCCUPANCY GRID
# ============================================================

class OccupancyGrid:
    """
    2D occupancy probability grid.

    Uses log-odds updates.

    Unknown cells start with
    probability 0.5.

    This map stores planar occupancy.
    In 3D mode, only x/y are used.
    """

    def __init__(
        self,
        config: SpatialMemoryConfig,
    ) -> None:
        self.config = config

        self.log_odds = np.zeros(
            (
                config.occupancy_height,
                config.occupancy_width,
            ),
            dtype=np.float32,
        )

        self.hit_log_odds = logit(
            config.occupancy_hit_probability
        )

        self.miss_log_odds = logit(
            config.occupancy_miss_probability
        )

        self._lock = threading.RLock()

    def world_to_grid(
        self,
        position: np.ndarray | list[float],
    ) -> tuple[int, int] | None:
        position = np.asarray(
            position,
            dtype=np.float64,
        )

        if position.ndim != 1 or len(position) < 2:
            raise SpatialValidationError(
                "Position must have x and y"
            )

        x = int(
            math.floor(
                (
                    position[0]
                    - self.config.occupancy_origin_x
                )
                / self.config.occupancy_resolution
            )
        )

        y = int(
            math.floor(
                (
                    position[1]
                    - self.config.occupancy_origin_y
                )
                / self.config.occupancy_resolution
            )
        )

        if (
            0 <= x < self.config.occupancy_width
            and 0 <= y < self.config.occupancy_height
        ):
            return y, x

        return None

    def update_cell(
        self,
        position: np.ndarray | list[float],
        occupied: bool,
    ) -> bool:
        index = self.world_to_grid(
            position
        )

        if index is None:
            return False

        increment = (
            self.hit_log_odds
            if occupied
            else self.miss_log_odds
        )

        with self._lock:
            self.log_odds[index] = np.clip(
                self.log_odds[index]
                + increment,
                -10.0,
                10.0,
            )

        return True

    def update_ray(
        self,
        sensor_position: np.ndarray,
        endpoint: np.ndarray,
        *,
        endpoint_occupied: bool = True,
    ) -> None:
        """
        Ray update with discrete samples.

        Free cells are updated along the ray.
        The endpoint is marked occupied
        when endpoint_occupied=True.

        This is a basic mapping approximation,
        not a full sensor inverse model.
        """
        start = np.asarray(
            sensor_position,
            dtype=np.float64,
        )[:2]

        end = np.asarray(
            endpoint,
            dtype=np.float64,
        )[:2]

        if (
            start.shape != (2,)
            or end.shape != (2,)
            or not np.isfinite(start).all()
            or not np.isfinite(end).all()
        ):
            raise SpatialValidationError(
                "Invalid ray endpoints"
            )

        distance = float(
            np.linalg.norm(
                end - start
            )
        )

        sample_count = max(
            1,
            int(
                math.ceil(
                    distance
                    / (
                        self.config.occupancy_resolution
                        * 0.5
                    )
                )
            ),
        )

        visited = set()

        for index in range(sample_count):
            fraction = (
                index / sample_count
            )

            point = (
                start
                + fraction * (end - start)
            )

            cell = self.world_to_grid(
                point
            )

            if cell is not None and cell not in visited:
                visited.add(cell)

                self.update_cell(
                    point,
                    occupied=False,
                )

        if endpoint_occupied:
            self.update_cell(
                end,
                occupied=True,
            )

    def probability_map(
        self,
    ) -> np.ndarray:
        with self._lock:
            return sigmoid(
                self.log_odds.copy()
            )

    def get_probability(
        self,
        position: np.ndarray | list[float],
    ) -> float | None:
        index = self.world_to_grid(
            position
        )

        if index is None:
            return None

        with self._lock:
            value = float(
                self.log_odds[index]
            )

        return float(
            sigmoid(
                np.array(value)
            )
        )

    def reset(self) -> None:
        with self._lock:
            self.log_odds.fill(
                0.0
            )


# ============================================================
# SPATIAL MEMORY ENGINE
# ============================================================

class SpatialMemory:
    """
    Thread-safe object-centric spatial memory.

    Stores:
        - latest object position
        - estimated velocity
        - category and confidence
        - observation timestamps
        - bounded movement history
        - change events

    Objects are keyed by object_id.

    Coordinates must share one reference
    frame. Transform sensor coordinates
    before inserting observations.
    """

    def __init__(
        self,
        config: SpatialMemoryConfig | None = None,
    ) -> None:
        self.config = (
            config or SpatialMemoryConfig()
        )

        self.objects: dict[
            str,
            SpatialMemoryObject,
        ] = {}

        self.events: deque[
            SpatialChangeEvent
        ] = deque(
            maxlen=10000
        )

        self.occupancy_grid = OccupancyGrid(
            self.config
        )

        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(
                self.objects
            )

    def _validate_observation(
        self,
        observation: SpatialObservation,
    ) -> None:
        if len(observation.position) != (
            self.config.spatial_dim
        ):
            raise SpatialValidationError(
                "Observation dimension mismatch"
            )

        if observation.confidence < (
            self.config.confidence_threshold
        ):
            raise SpatialValidationError(
                "Observation confidence too low"
            )

    def _evict_if_needed(self) -> None:
        if len(self.objects) < (
            self.config.max_objects
        ):
            return

        oldest_id = min(
            self.objects,
            key=lambda object_id: (
                self.objects[
                    object_id
                ].last_seen
            ),
        )

        self.objects.pop(
            oldest_id
        )

    def observe(
        self,
        observation: SpatialObservation,
    ) -> SpatialMemoryObject:
        self._validate_observation(
            observation
        )

        with self._lock:
            existing = self.objects.get(
                observation.object_id
            )

            if existing is None:
                self._evict_if_needed()

                velocity = (
                    observation.velocity.copy()
                    if observation.velocity is not None
                    else np.zeros(
                        self.config.spatial_dim,
                        dtype=np.float64,
                    )
                )

                obj = SpatialMemoryObject(
                    object_id=observation.object_id,
                    position=(
                        observation.position.copy()
                    ),
                    velocity=velocity,
                    category=observation.category,
                    confidence=observation.confidence,
                    first_seen=observation.timestamp,
                    last_seen=observation.timestamp,
                    metadata=dict(
                        observation.metadata
                    ),
                    history=deque(
                        maxlen=(
                            self.config.max_history_per_object
                        )
                    ),
                )

                obj.history.append(
                    SpatialHistoryEntry(
                        position=obj.position.copy(),
                        velocity=obj.velocity.copy(),
                        timestamp=observation.timestamp,
                        confidence=observation.confidence,
                    )
                )

                self.objects[
                    observation.object_id
                ] = obj

                self.events.append(
                    SpatialChangeEvent(
                        object_id=obj.object_id,
                        change_type=(
                            SpatialChangeType.ADDED
                        ),
                        timestamp=observation.timestamp,
                        current_position=(
                            obj.position.copy()
                        ),
                    )
                )

                return obj

            if observation.timestamp < existing.last_seen:
                raise SpatialValidationError(
                    "Out-of-order observation rejected"
                )

            previous_position = (
                existing.position.copy()
            )

            dt = (
                observation.timestamp
                - existing.last_seen
            )

            alpha = (
                self.config.position_smoothing
                * observation.confidence
            )

            alpha = float(
                np.clip(
                    alpha,
                    0.0,
                    1.0,
                )
            )

            updated_position = (
                (1 - alpha)
                * existing.position
                + alpha
                * observation.position
            )

            if observation.velocity is not None:
                updated_velocity = (
                    (1 - alpha)
                    * existing.velocity
                    + alpha
                    * observation.velocity
                )

            elif dt > EPSILON:
                measured_velocity = (
                    observation.position
                    - existing.position
                ) / dt

                updated_velocity = (
                    (1 - alpha)
                    * existing.velocity
                    + alpha
                    * measured_velocity
                )

            else:
                updated_velocity = (
                    existing.velocity.copy()
                )

            distance_moved = float(
                np.linalg.norm(
                    updated_position
                    - previous_position
                )
            )

            existing.position = (
                updated_position
            )

            existing.velocity = (
                updated_velocity
            )

            existing.confidence = (
                observation.confidence
            )

            existing.category = (
                observation.category
            )

            existing.last_seen = (
                observation.timestamp
            )

            existing.status = (
                SpatialObjectStatus.ACTIVE
            )

            existing.metadata.update(
                observation.metadata
            )

            existing.history.append(
                SpatialHistoryEntry(
                    position=(
                        updated_position.copy()
                    ),
                    velocity=(
                        updated_velocity.copy()
                    ),
                    timestamp=observation.timestamp,
                    confidence=observation.confidence,
                )
            )

            change_type = (
                SpatialChangeType.MOVED
                if distance_moved
                >= self.config.movement_threshold
                else SpatialChangeType.UPDATED
            )

            self.events.append(
                SpatialChangeEvent(
                    object_id=existing.object_id,
                    change_type=change_type,
                    timestamp=observation.timestamp,
                    previous_position=previous_position,
                    current_position=(
                        updated_position.copy()
                    ),
                    distance_moved=distance_moved,
                )
            )

            return existing

    def observe_many(
        self,
        observations: Iterable[
            SpatialObservation
        ],
    ) -> list[SpatialMemoryObject]:
        return [
            self.observe(observation)
            for observation in observations
        ]

    def get(
        self,
        object_id: str,
    ) -> SpatialMemoryObject | None:
        with self._lock:
            return self.objects.get(
                object_id
            )

    def remove(
        self,
        object_id: str,
        *,
        timestamp: float | None = None,
    ) -> bool:
        with self._lock:
            obj = self.objects.pop(
                object_id,
                None,
            )

            if obj is None:
                return False

            self.events.append(
                SpatialChangeEvent(
                    object_id=object_id,
                    change_type=(
                        SpatialChangeType.REMOVED
                    ),
                    timestamp=validate_timestamp(
                        timestamp
                    ),
                    previous_position=(
                        obj.position.copy()
                    ),
                )
            )

            return True

    def update_statuses(
        self,
        timestamp: float | None = None,
    ) -> dict[str, int]:
        timestamp = validate_timestamp(
            timestamp
        )

        counts = {
            "active": 0,
            "stale": 0,
            "lost": 0,
        }

        with self._lock:
            for obj in self.objects.values():
                age = max(
                    0.0,
                    timestamp - obj.last_seen,
                )

                if age >= (
                    self.config.lost_after_seconds
                ):
                    obj.status = (
                        SpatialObjectStatus.LOST
                    )

                elif age >= (
                    self.config.stale_after_seconds
                ):
                    obj.status = (
                        SpatialObjectStatus.STALE
                    )

                else:
                    obj.status = (
                        SpatialObjectStatus.ACTIVE
                    )

                counts[
                    obj.status.value
                ] += 1

        return counts

    def nearest(
        self,
        position: np.ndarray | list[float],
        *,
        k: int = 5,
        category: str | None = None,
        include_lost: bool = False,
    ) -> list[SpatialMemoryObject]:
        if k <= 0:
            raise SpatialValidationError(
                "k must be positive"
            )

        position = as_vector(
            position,
            self.config.spatial_dim,
            "position",
        )

        with self._lock:
            candidates = [
                obj
                for obj in self.objects.values()
                if (
                    category is None
                    or obj.category == category
                )
                and (
                    include_lost
                    or obj.status
                    != SpatialObjectStatus.LOST
                )
            ]

            candidates.sort(
                key=lambda obj: (
                    obj.distance_to(
                        position
                    )
                )
            )

            return candidates[:k]

    def within_radius(
        self,
        position: np.ndarray | list[float],
        radius: float,
        *,
        category: str | None = None,
    ) -> list[SpatialMemoryObject]:
        if radius < 0:
            raise SpatialValidationError(
                "radius must be nonnegative"
            )

        position = as_vector(
            position,
            self.config.spatial_dim,
            "position",
        )

        with self._lock:
            matches = [
                obj
                for obj in self.objects.values()
                if obj.distance_to(
                    position
                ) <= radius
                and (
                    category is None
                    or obj.category == category
                )
                and obj.status
                != SpatialObjectStatus.LOST
            ]

            matches.sort(
                key=lambda obj: (
                    obj.distance_to(
                        position
                    )
                )
            )

            return matches

    def predict_object_position(
        self,
        object_id: str,
        timestamp: float,
    ) -> np.ndarray:
        with self._lock:
            obj = self.objects.get(
                object_id
            )

            if obj is None:
                raise KeyError(
                    f"Unknown object: {object_id}"
                )

            return obj.predict_position(
                timestamp
            )

    def get_history(
        self,
        object_id: str,
    ) -> list[SpatialHistoryEntry]:
        with self._lock:
            obj = self.objects.get(
                object_id
            )

            if obj is None:
                raise KeyError(
                    f"Unknown object: {object_id}"
                )

            return list(
                obj.history
            )

    def recent_events(
        self,
        limit: int = 100,
    ) -> list[SpatialChangeEvent]:
        if limit < 0:
            raise SpatialValidationError(
                "limit must be nonnegative"
            )

        with self._lock:
            if limit == 0:
                return []

            return list(
                self.events
            )[-limit:]

    def snapshot(
        self,
        *,
        include_history: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "spatial_dim": (
                    self.config.spatial_dim
                ),
                "object_count": (
                    len(self.objects)
                ),
                "objects": {
                    object_id: obj.to_dict(
                        include_history=(
                            include_history
                        )
                    )
                    for object_id, obj
                    in self.objects.items()
                },
            }

    def clear(self) -> None:
        with self._lock:
            self.objects.clear()
            self.events.clear()
            self.occupancy_grid.reset()


# ============================================================
# SPATIAL FEATURE EXPORT
# ============================================================

class SpatialMemoryEncoder:
    """
    Converts object memory into tensors.

    Feature layout:
        position [D]
        velocity [D]
        confidence [1]
        age [1]
        status [1]

    Feature dimension:
        2*D + 3

    Returns:
        features [N,F]
        mask [N]
        object_ids list[str]

    Rows are ordered by object_id.
    """

    def __init__(
        self,
        config: SpatialMemoryConfig,
    ) -> None:
        self.config = config

    @property
    def feature_dim(self) -> int:
        return (
            2 * self.config.spatial_dim
            + 3
        )

    def encode(
        self,
        memory: SpatialMemory,
        *,
        timestamp: float | None = None,
        max_objects: int | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        list[str],
    ]:
        timestamp = validate_timestamp(
            timestamp
        )

        with memory._lock:
            objects = sorted(
                memory.objects.values(),
                key=lambda obj: obj.object_id,
            )

            if max_objects is not None:
                if max_objects < 0:
                    raise SpatialValidationError(
                        "max_objects must be nonnegative"
                    )

                objects = objects[
                    :max_objects
                ]

            object_ids = [
                obj.object_id
                for obj in objects
            ]

            features = []

            for obj in objects:
                age = max(
                    0.0,
                    timestamp - obj.last_seen,
                )

                status_value = {
                    SpatialObjectStatus.ACTIVE: 0.0,
                    SpatialObjectStatus.STALE: 0.5,
                    SpatialObjectStatus.LOST: 1.0,
                }[obj.status]

                feature = np.concatenate(
                    [
                        obj.position,
                        obj.velocity,
                        np.array(
                            [
                                obj.confidence,
                                age,
                                status_value,
                            ],
                            dtype=np.float64,
                        ),
                    ]
                )

                features.append(
                    feature
                )

        if features:
            feature_array = np.stack(
                features
            )
        else:
            feature_array = np.empty(
                (
                    0,
                    self.feature_dim,
                ),
                dtype=np.float64,
            )

        tensor = torch.as_tensor(
            feature_array,
            dtype=torch.float32,
            device=self.config.device,
        )

        mask = torch.ones(
            len(objects),
            dtype=torch.bool,
            device=self.config.device,
        )

        return (
            tensor,
            mask,
            object_ids,
        )


# ============================================================
# SPATIAL MEMORY PERSISTENCE
# ============================================================

class SpatialMemoryPersistence:
    """
    Save and load object memory.

    JSON supports the object state and
    observation history.

    Occupancy log-odds are stored
    separately as a NumPy array.
    """

    @staticmethod
    def save(
        memory: SpatialMemory,
        directory: str | Path,
    ) -> None:
        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot = memory.snapshot(
            include_history=True
        )

        with (
            directory / "spatial_memory.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                snapshot,
                file,
                indent=2,
            )

        np.save(
            directory / "occupancy_grid.npy",
            memory.occupancy_grid.log_odds,
        )

    @staticmethod
    def load(
        directory: str | Path,
        config: SpatialMemoryConfig,
    ) -> SpatialMemory:
        directory = Path(
            directory
        )

        with (
            directory / "spatial_memory.json"
        ).open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

        if data["spatial_dim"] != (
            config.spatial_dim
        ):
            raise SpatialValidationError(
                "Saved spatial dimension mismatch"
            )

        memory = SpatialMemory(
            config
        )

        for object_id, item in (
            data["objects"].items()
        ):
            obj = SpatialMemoryObject(
                object_id=object_id,
                position=np.asarray(
                    item["position"],
                    dtype=np.float64,
                ),
                velocity=np.asarray(
                    item["velocity"],
                    dtype=np.float64,
                ),
                category=item["category"],
                confidence=item["confidence"],
                first_seen=item["first_seen"],
                last_seen=item["last_seen"],
                status=SpatialObjectStatus(
                    item["status"]
                ),
                metadata=item.get(
                    "metadata",
                    {},
                ),
                history=deque(
                    maxlen=(
                        config.max_history_per_object
                    )
                ),
            )

            for entry in item.get(
                "history",
                [],
            ):
                obj.history.append(
                    SpatialHistoryEntry(
                        position=np.asarray(
                            entry["position"],
                            dtype=np.float64,
                        ),
                        velocity=np.asarray(
                            entry["velocity"],
                            dtype=np.float64,
                        ),
                        timestamp=entry["timestamp"],
                        confidence=entry["confidence"],
                    )
                )

            memory.objects[
                object_id
            ] = obj

        grid_path = (
            directory / "occupancy_grid.npy"
        )

        if grid_path.exists():
            grid = np.load(
                grid_path,
                allow_pickle=False,
            )

            if grid.shape != (
                memory.occupancy_grid.log_odds.shape
            ):
                raise SpatialValidationError(
                    "Saved occupancy grid shape mismatch"
                )

            memory.occupancy_grid.log_odds[:] = (
                grid
            )

        return memory


# ============================================================
# AGENT ADAPTER
# ============================================================

class SpatialMemoryAgentAdapter:
    """
    Lightweight API for world-model agents.

    Supports:
        observe_object
        get_nearby_objects
        predict_position
        get_scene_context
    """

    def __init__(
        self,
        memory: SpatialMemory,
    ) -> None:
        self.memory = memory

    def observe_object(
        self,
        object_id: str,
        position: list[float],
        *,
        velocity: list[float] | None = None,
        category: str = "unknown",
        confidence: float = 1.0,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        obj = self.memory.observe(
            SpatialObservation(
                object_id=object_id,
                position=position,
                velocity=velocity,
                category=category,
                confidence=confidence,
                timestamp=timestamp,
            )
        )

        return obj.to_dict(
            include_history=False
        )

    def get_nearby_objects(
        self,
        position: list[float],
        radius: float,
    ) -> list[dict[str, Any]]:
        return [
            obj.to_dict(
                include_history=False
            )
            for obj in self.memory.within_radius(
                position,
                radius,
            )
        ]

    def predict_position(
        self,
        object_id: str,
        timestamp: float,
    ) -> list[float]:
        return (
            self.memory.predict_object_position(
                object_id,
                timestamp,
            ).tolist()
        )

    def get_scene_context(
        self,
    ) -> dict[str, Any]:
        return self.memory.snapshot(
            include_history=False
        )


# ============================================================
# FACTORY
# ============================================================

def create_spatial_memory(
    config: SpatialMemoryConfig | None = None,
) -> SpatialMemory:
    return SpatialMemory(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - SPATIAL MEMORY"
    )
    print("=" * 65)

    config = SpatialMemoryConfig(
        spatial_dim=2,
        max_objects=1000,
        max_history_per_object=50,
        occupancy_resolution=0.25,
    )

    memory = SpatialMemory(
        config
    )

    timestamp = time.time()

    observations = [
        SpatialObservation(
            object_id="robot_01",
            position=[0.0, 0.0],
            velocity=[0.5, 0.0],
            category="robot",
            timestamp=timestamp,
        ),
        SpatialObservation(
            object_id="box_01",
            position=[2.0, 1.0],
            category="box",
            timestamp=timestamp,
        ),
        SpatialObservation(
            object_id="obstacle_01",
            position=[3.0, 2.0],
            category="obstacle",
            timestamp=timestamp,
        ),
    ]

    memory.observe_many(
        observations
    )

    print(
        "Objects stored:",
        len(memory),
    )

    print("\nNearest objects")

    nearest = memory.nearest(
        [0.0, 0.0],
        k=3,
    )

    for obj in nearest:
        print(
            obj.object_id,
            obj.position,
        )

    print("\nUpdating robot position")

    memory.observe(
        SpatialObservation(
            object_id="robot_01",
            position=[1.0, 0.0],
            velocity=[0.5, 0.0],
            category="robot",
            timestamp=timestamp + 1.0,
        )
    )

    print(
        "Predicted robot position:",
        memory.predict_object_position(
            "robot_01",
            timestamp + 3.0,
        ),
    )

    print("\nOccupancy mapping")

    memory.occupancy_grid.update_ray(
        sensor_position=[0.0, 0.0],
        endpoint=[2.0, 1.0],
        endpoint_occupied=True,
    )

    probability = (
        memory.occupancy_grid.get_probability(
            [2.0, 1.0]
        )
    )

    print(
        "Occupancy probability:",
        probability,
    )

    print("\nSpatial feature encoding")

    encoder = SpatialMemoryEncoder(
        config
    )

    features, mask, object_ids = (
        encoder.encode(
            memory,
            timestamp=timestamp + 1.0,
        )
    )

    print(
        "Feature shape:",
        features.shape,
    )

    print(
        "Object IDs:",
        object_ids,
    )

    print(
        "Mask:",
        mask,
    )

    print("\nRecent changes")

    for event in memory.recent_events(
        limit=5
    ):
        print(
            event.to_dict()
        )

    print(
        "\nSpatial memory demonstration completed"
    )


if __name__ == "__main__":
    main()
