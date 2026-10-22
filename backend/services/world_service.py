
"""
NeuroWorld AI
backend/app/services/world_service.py

World lifecycle and state-management service.

Features:
- Create and manage virtual worlds
- Initialize and validate latent states
- Update world states
- Apply action-conditioned model transitions
- Optimistic state-version checks
- Pause, resume, and archive worlds
- Snapshot and restore world states
- Roll back to previous state versions
- Track world events
- Track state history
- Attach observation references
- Compare world states
- Thread-safe bounded in-memory storage
- Async-compatible service methods

Dependencies:
    pip install numpy pydantic

World states are numerical latent vectors.
This service does not execute physical hardware actions.
"""

from __future__ import annotations

import asyncio
import math
import threading
import uuid

from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class WorldServiceError(Exception):
    """Base world service exception."""


class WorldNotFound(WorldServiceError):
    """World does not exist."""


class WorldAlreadyExists(WorldServiceError):
    """World ID already exists."""


class WorldValidationError(WorldServiceError):
    """Invalid world data."""


class WorldStateConflict(WorldServiceError):
    """World version mismatch."""


class WorldLifecycleError(WorldServiceError):
    """Invalid world lifecycle operation."""


class WorldCapacityError(WorldServiceError):
    """World or history capacity exceeded."""


class WorldModelTransitionError(WorldServiceError):
    """World-model transition failed."""


# ============================================================
# ENUMS
# ============================================================

class WorldStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class WorldEventType(str, Enum):
    CREATED = "created"
    STATE_UPDATED = "state_updated"
    ACTION_APPLIED = "action_applied"
    OBSERVATION_ATTACHED = "observation_attached"
    PAUSED = "paused"
    RESUMED = "resumed"
    ARCHIVED = "archived"
    RESTORED = "restored"
    ROLLED_BACK = "rolled_back"


# ============================================================
# REQUEST MODELS
# ============================================================

class CreateWorldRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=128,
    )

    description: str = Field(
        default="",
        max_length=2000,
    )

    state_dimension: int = Field(
        ge=1,
        le=16384,
    )

    initial_state: list[float] | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_initial_state(self):
        if self.initial_state is not None:
            if len(self.initial_state) != self.state_dimension:
                raise ValueError(
                    "initial_state length must match "
                    "state_dimension"
                )

            if not all(
                math.isfinite(value)
                for value in self.initial_state
            ):
                raise ValueError(
                    "initial_state must contain finite values"
                )

        return self


class UpdateWorldStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    expected_version: int | None = Field(
        default=None,
        ge=0,
    )

    source: str = Field(
        default="manual",
        min_length=1,
        max_length=128,
    )


class ApplyWorldActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    expected_version: int | None = Field(
        default=None,
        ge=0,
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )


class AttachObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(
        min_length=1,
        max_length=128,
    )

    expected_version: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class WorldState(BaseModel):
    version: int
    vector: list[float]
    updated_at: datetime
    source: str


class WorldEvent(BaseModel):
    event_id: str
    world_id: str
    event_type: WorldEventType
    timestamp: datetime
    version: int
    details: dict[str, Any] = Field(
        default_factory=dict,
    )


class WorldRecord(BaseModel):
    world_id: str
    name: str
    description: str

    status: WorldStatus

    state_dimension: int
    state: WorldState

    created_at: datetime
    updated_at: datetime

    observation_ids: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class WorldSnapshot(BaseModel):
    snapshot_id: str
    world_id: str
    name: str
    created_at: datetime

    state: WorldState
    observation_ids: list[str]

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class WorldComparison(BaseModel):
    world_a: str
    world_b: str

    dimension: int

    euclidean_distance: float
    mean_absolute_difference: float
    maximum_absolute_difference: float

    cosine_similarity: float | None


class WorldTransitionResult(BaseModel):
    world_id: str
    previous_version: int
    new_version: int

    action: list[float]
    next_state: list[float]

    predicted_reward: float
    uncertainty: float
    terminal: bool


# ============================================================
# WORLD-MODEL INTERFACE
# ============================================================

class WorldTransitionModel(Protocol):
    name: str
    state_dim: int
    action_dim: int

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        ...


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_vector(
    values: Any,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    try:
        vector = np.asarray(
            values,
            dtype=np.float64,
        )
    except (ValueError, TypeError) as exc:
        raise WorldValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if vector.shape != (dimension,):
        raise WorldValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.all(np.isfinite(vector)):
        raise WorldValidationError(
            f"{name} must contain finite values"
        )

    return vector.copy()


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    norm_a = float(np.linalg.norm(first))
    norm_b = float(np.linalg.norm(second))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return None

    return float(
        np.dot(first, second)
        / (norm_a * norm_b)
    )


# ============================================================
# INTERNAL WORLD RUNTIME
# ============================================================

class _WorldRuntime:
    def __init__(
        self,
        record: WorldRecord,
        *,
        max_history: int,
        max_events: int,
        max_snapshots: int,
    ) -> None:
        self.record = record

        self.history: deque[WorldState] = deque(
            [record.state.model_copy(deep=True)],
            maxlen=max_history,
        )

        self.events: deque[WorldEvent] = deque(
            maxlen=max_events
        )

        self.snapshots: dict[
            str,
            WorldSnapshot,
        ] = {}

        self.max_snapshots = max_snapshots

        self.lock = threading.RLock()


# ============================================================
# WORLD SERVICE
# ============================================================

class WorldService:
    """
    Thread-safe in-memory world-state manager.

    Each world owns its state, event log, snapshots,
    and observation references.

    State changes are versioned.

    For production:
    - Persist records in PostgreSQL
    - Use transactional version checks
    - Enforce tenant/world ownership
    - Store large states in object storage
    """

    def __init__(
        self,
        *,
        max_worlds: int = 100,
        max_history_per_world: int = 500,
        max_events_per_world: int = 1000,
        max_snapshots_per_world: int = 50,
        max_observations_per_world: int = 10000,
    ) -> None:
        limits = (
            max_worlds,
            max_history_per_world,
            max_events_per_world,
            max_snapshots_per_world,
            max_observations_per_world,
        )

        if any(value < 1 for value in limits):
            raise ValueError(
                "World service limits must be positive"
            )

        self.max_worlds = max_worlds
        self.max_history_per_world = (
            max_history_per_world
        )
        self.max_events_per_world = (
            max_events_per_world
        )
        self.max_snapshots_per_world = (
            max_snapshots_per_world
        )
        self.max_observations_per_world = (
            max_observations_per_world
        )

        self._worlds: dict[
            str,
            _WorldRuntime,
        ] = {}

        self._lock = threading.RLock()

    def _get_runtime(
        self,
        world_id: str,
    ) -> _WorldRuntime:
        with self._lock:
            runtime = self._worlds.get(
                world_id
            )

        if runtime is None:
            raise WorldNotFound(
                f"World not found: {world_id}"
            )

        return runtime

    def _record_event(
        self,
        runtime: _WorldRuntime,
        event_type: WorldEventType,
        details: dict[str, Any] | None = None,
    ) -> WorldEvent:
        record = runtime.record

        event = WorldEvent(
            event_id=uuid.uuid4().hex,
            world_id=record.world_id,
            event_type=event_type,
            timestamp=utc_now(),
            version=record.state.version,
            details=details or {},
        )

        runtime.events.append(event)

        return event

    @staticmethod
    def _check_version(
        record: WorldRecord,
        expected_version: int | None,
    ) -> None:
        if (
            expected_version is not None
            and record.state.version != expected_version
        ):
            raise WorldStateConflict(
                "World state version mismatch: "
                f"expected={expected_version}, "
                f"actual={record.state.version}"
            )

    @staticmethod
    def _require_active(
        record: WorldRecord,
    ) -> None:
        if record.status != WorldStatus.ACTIVE:
            raise WorldLifecycleError(
                "World must be active"
            )

    def _commit_state(
        self,
        runtime: _WorldRuntime,
        *,
        state: np.ndarray,
        source: str,
        event_type: WorldEventType,
        details: dict[str, Any] | None = None,
    ) -> WorldState:
        record = runtime.record
        now = utc_now()

        new_state = WorldState(
            version=record.state.version + 1,
            vector=state.tolist(),
            updated_at=now,
            source=source,
        )

        record.state = new_state
        record.updated_at = now

        runtime.history.append(
            new_state.model_copy(deep=True)
        )

        self._record_event(
            runtime,
            event_type,
            details,
        )

        return new_state.model_copy(deep=True)

    # --------------------------------------------------------
    # WORLD CREATION
    # --------------------------------------------------------

    def create_world(
        self,
        request: CreateWorldRequest,
    ) -> WorldRecord:
        if request.initial_state is None:
            initial_vector = np.zeros(
                request.state_dimension,
                dtype=np.float64,
            )
        else:
            initial_vector = validate_vector(
                request.initial_state,
                dimension=request.state_dimension,
                name="initial_state",
            )

        now = utc_now()

        state = WorldState(
            version=0,
            vector=initial_vector.tolist(),
            updated_at=now,
            source="initialization",
        )

        world_id = uuid.uuid4().hex

        record = WorldRecord(
            world_id=world_id,
            name=request.name,
            description=request.description,
            status=WorldStatus.ACTIVE,
            state_dimension=request.state_dimension,
            state=state,
            created_at=now,
            updated_at=now,
            metadata=request.metadata,
        )

        runtime = _WorldRuntime(
            record,
            max_history=self.max_history_per_world,
            max_events=self.max_events_per_world,
            max_snapshots=self.max_snapshots_per_world,
        )

        with self._lock:
            if len(self._worlds) >= self.max_worlds:
                raise WorldCapacityError(
                    "Maximum world count reached"
                )

            self._worlds[world_id] = runtime

        with runtime.lock:
            self._record_event(
                runtime,
                WorldEventType.CREATED,
                {
                    "name": request.name,
                    "state_dimension": (
                        request.state_dimension
                    ),
                },
            )

        return record.model_copy(deep=True)

    # --------------------------------------------------------
    # WORLD RETRIEVAL
    # --------------------------------------------------------

    def get_world(
        self,
        world_id: str,
    ) -> WorldRecord:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            return runtime.record.model_copy(
                deep=True
            )

    def list_worlds(
        self,
        *,
        status: WorldStatus | None = None,
        limit: int = 100,
    ) -> list[WorldRecord]:
        if not 1 <= limit <= self.max_worlds:
            raise WorldValidationError(
                "Invalid world listing limit"
            )

        with self._lock:
            runtimes = list(
                self._worlds.values()
            )

        records = []

        for runtime in runtimes:
            with runtime.lock:
                record = runtime.record.model_copy(
                    deep=True
                )

            if (
                status is None
                or record.status == status
            ):
                records.append(record)

        records.sort(
            key=lambda item: item.created_at,
            reverse=True,
        )

        return records[:limit]

    # --------------------------------------------------------
    # STATE UPDATES
    # --------------------------------------------------------

    def update_state(
        self,
        world_id: str,
        request: UpdateWorldStateRequest,
    ) -> WorldState:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            self._require_active(record)

            self._check_version(
                record,
                request.expected_version,
            )

            vector = validate_vector(
                request.state,
                dimension=record.state_dimension,
                name="state",
            )

            return self._commit_state(
                runtime,
                state=vector,
                source=request.source,
                event_type=(
                    WorldEventType.STATE_UPDATED
                ),
                details={
                    "source": request.source,
                },
            )

    def apply_action(
        self,
        world_id: str,
        request: ApplyWorldActionRequest,
        *,
        model: WorldTransitionModel,
    ) -> WorldTransitionResult:
        """
        Apply one action-conditioned model transition.

        This updates the virtual world's latent state.
        It does not execute the action in the real world.
        """
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            self._require_active(record)

            self._check_version(
                record,
                request.expected_version,
            )

            if model.state_dim != record.state_dimension:
                raise WorldValidationError(
                    "World/model state dimension mismatch"
                )

            action = validate_vector(
                request.action,
                dimension=model.action_dim,
                name="action",
            )

            previous_state = np.asarray(
                record.state.vector,
                dtype=np.float64,
            )

            try:
                transition = model.predict_transition(
                    previous_state.copy(),
                    action.copy(),
                )
            except Exception as exc:
                raise WorldModelTransitionError(
                    "World-model transition failed"
                ) from exc

            next_state = validate_vector(
                transition.next_state,
                dimension=record.state_dimension,
                name="next_state",
            )

            uncertainty = float(
                transition.uncertainty
            )

            reward = float(
                transition.reward
            )

            if (
                not math.isfinite(uncertainty)
                or not 0 <= uncertainty <= 1
            ):
                raise WorldModelTransitionError(
                    "Model returned invalid uncertainty"
                )

            if not math.isfinite(reward):
                raise WorldModelTransitionError(
                    "Model returned invalid reward"
                )

            if (
                uncertainty
                > request.uncertainty_threshold
            ):
                raise WorldModelTransitionError(
                    "Model uncertainty exceeds threshold"
                )

            previous_version = (
                record.state.version
            )

            new_state = self._commit_state(
                runtime,
                state=next_state,
                source=f"world_model:{model.name}",
                event_type=(
                    WorldEventType.ACTION_APPLIED
                ),
                details={
                    "model": model.name,
                    "uncertainty": uncertainty,
                    "reward": reward,
                    "terminal": bool(
                        transition.terminal
                    ),
                },
            )

            return WorldTransitionResult(
                world_id=world_id,
                previous_version=previous_version,
                new_version=new_state.version,
                action=action.tolist(),
                next_state=next_state.tolist(),
                predicted_reward=reward,
                uncertainty=uncertainty,
                terminal=bool(
                    transition.terminal
                ),
            )

    # --------------------------------------------------------
    # OBSERVATION REFERENCES
    # --------------------------------------------------------

    def attach_observation(
        self,
        world_id: str,
        request: AttachObservationRequest,
    ) -> WorldRecord:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            self._require_active(record)

            self._check_version(
                record,
                request.expected_version,
            )

            if (
                request.observation_id
                in record.observation_ids
            ):
                return record.model_copy(deep=True)

            if (
                len(record.observation_ids)
                >= self.max_observations_per_world
            ):
                raise WorldCapacityError(
                    "World observation limit reached"
                )

            record.observation_ids.append(
                request.observation_id
            )

            record.updated_at = utc_now()

            self._record_event(
                runtime,
                WorldEventType.OBSERVATION_ATTACHED,
                {
                    "observation_id": (
                        request.observation_id
                    ),
                },
            )

            return record.model_copy(deep=True)

    # --------------------------------------------------------
    # WORLD LIFECYCLE
    # --------------------------------------------------------

    def pause_world(
        self,
        world_id: str,
    ) -> WorldRecord:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            self._require_active(record)

            record.status = WorldStatus.PAUSED
            record.updated_at = utc_now()

            self._record_event(
                runtime,
                WorldEventType.PAUSED,
            )

            return record.model_copy(deep=True)

    def resume_world(
        self,
        world_id: str,
    ) -> WorldRecord:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            if record.status != WorldStatus.PAUSED:
                raise WorldLifecycleError(
                    "Only paused worlds can be resumed"
                )

            record.status = WorldStatus.ACTIVE
            record.updated_at = utc_now()

            self._record_event(
                runtime,
                WorldEventType.RESUMED,
            )

            return record.model_copy(deep=True)

    def archive_world(
        self,
        world_id: str,
    ) -> WorldRecord:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            if record.status == WorldStatus.ARCHIVED:
                raise WorldLifecycleError(
                    "World is already archived"
                )

            record.status = WorldStatus.ARCHIVED
            record.updated_at = utc_now()

            self._record_event(
                runtime,
                WorldEventType.ARCHIVED,
            )

            return record.model_copy(deep=True)

    # --------------------------------------------------------
    # SNAPSHOTS
    # --------------------------------------------------------

    def create_snapshot(
        self,
        world_id: str,
        *,
        name: str = "manual-snapshot",
    ) -> WorldSnapshot:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            if (
                len(runtime.snapshots)
                >= runtime.max_snapshots
            ):
                raise WorldCapacityError(
                    "Snapshot limit reached"
                )

            record = runtime.record

            snapshot = WorldSnapshot(
                snapshot_id=uuid.uuid4().hex,
                world_id=world_id,
                name=name,
                created_at=utc_now(),
                state=record.state.model_copy(
                    deep=True
                ),
                observation_ids=list(
                    record.observation_ids
                ),
                metadata=dict(
                    record.metadata
                ),
            )

            runtime.snapshots[
                snapshot.snapshot_id
            ] = snapshot

            return snapshot.model_copy(deep=True)

    def list_snapshots(
        self,
        world_id: str,
    ) -> list[WorldSnapshot]:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            return [
                snapshot.model_copy(deep=True)
                for snapshot
                in runtime.snapshots.values()
            ]

    def restore_snapshot(
        self,
        world_id: str,
        snapshot_id: str,
        *,
        expected_version: int | None = None,
    ) -> WorldRecord:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            self._require_active(record)

            self._check_version(
                record,
                expected_version,
            )

            snapshot = runtime.snapshots.get(
                snapshot_id
            )

            if snapshot is None:
                raise WorldValidationError(
                    "Snapshot not found"
                )

            vector = validate_vector(
                snapshot.state.vector,
                dimension=record.state_dimension,
                name="snapshot_state",
            )

            self._commit_state(
                runtime,
                state=vector,
                source=f"snapshot:{snapshot_id}",
                event_type=WorldEventType.RESTORED,
                details={
                    "snapshot_id": snapshot_id,
                    "restored_from_version": (
                        snapshot.state.version
                    ),
                },
            )

            record.observation_ids = list(
                snapshot.observation_ids
            )

            record.metadata = dict(
                snapshot.metadata
            )

            return record.model_copy(deep=True)

    # --------------------------------------------------------
    # ROLLBACK
    # --------------------------------------------------------

    def rollback_state(
        self,
        world_id: str,
        *,
        target_version: int,
        expected_version: int | None = None,
    ) -> WorldState:
        runtime = self._get_runtime(world_id)

        with runtime.lock:
            record = runtime.record

            self._require_active(record)

            self._check_version(
                record,
                expected_version,
            )

            target = next(
                (
                    state
                    for state in runtime.history
                    if state.version == target_version
                ),
                None,
            )

            if target is None:
                raise WorldValidationError(
                    "Target state version not available"
                )

            vector = validate_vector(
                target.vector,
                dimension=record.state_dimension,
                name="rollback_state",
            )

            # Rollback creates a NEW version.
            # Version numbers remain monotonically increasing.
            return self._commit_state(
                runtime,
                state=vector,
                source=f"rollback:{target_version}",
                event_type=(
                    WorldEventType.ROLLED_BACK
                ),
                details={
                    "target_version": target_version,
                },
            )

    # --------------------------------------------------------
    # HISTORY AND EVENTS
    # --------------------------------------------------------

    def get_state_history(
        self,
        world_id: str,
        *,
        limit: int = 100,
    ) -> list[WorldState]:
        runtime = self._get_runtime(world_id)

        if not 1 <= limit <= self.max_history_per_world:
            raise WorldValidationError(
                "Invalid history limit"
            )

        with runtime.lock:
            return [
                item.model_copy(deep=True)
                for item in list(
                    runtime.history
                )[-limit:]
            ]

    def get_events(
        self,
        world_id: str,
        *,
        limit: int = 100,
    ) -> list[WorldEvent]:
        runtime = self._get_runtime(world_id)

        if not 1 <= limit <= self.max_events_per_world:
            raise WorldValidationError(
                "Invalid event limit"
            )

        with runtime.lock:
            return [
                event.model_copy(deep=True)
                for event in list(
                    runtime.events
                )[-limit:]
            ]

    # --------------------------------------------------------
    # WORLD COMPARISON
    # --------------------------------------------------------

    def compare_worlds(
        self,
        world_a: str,
        world_b: str,
    ) -> WorldComparison:
        first = self.get_world(world_a)
        second = self.get_world(world_b)

        if (
            first.state_dimension
            != second.state_dimension
        ):
            raise WorldValidationError(
                "World dimensions must match"
            )

        vector_a = np.asarray(
            first.state.vector,
            dtype=np.float64,
        )

        vector_b = np.asarray(
            second.state.vector,
            dtype=np.float64,
        )

        difference = vector_a - vector_b

        return WorldComparison(
            world_a=world_a,
            world_b=world_b,
            dimension=first.state_dimension,
            euclidean_distance=float(
                np.linalg.norm(difference)
            ),
            mean_absolute_difference=float(
                np.mean(np.abs(difference))
            ),
            maximum_absolute_difference=float(
                np.max(np.abs(difference))
            ),
            cosine_similarity=cosine_similarity(
                vector_a,
                vector_b,
            ),
        )

    # --------------------------------------------------------
    # DELETION
    # --------------------------------------------------------

    def delete_world(
        self,
        world_id: str,
    ) -> bool:
        with self._lock:
            runtime = self._worlds.pop(
                world_id,
                None,
            )

        return runtime is not None

    # --------------------------------------------------------
    # ASYNC METHODS
    # --------------------------------------------------------

    async def create_world_async(
        self,
        request: CreateWorldRequest,
    ) -> WorldRecord:
        return await asyncio.to_thread(
            self.create_world,
            request,
        )

    async def update_state_async(
        self,
        world_id: str,
        request: UpdateWorldStateRequest,
    ) -> WorldState:
        return await asyncio.to_thread(
            self.update_state,
            world_id,
            request,
        )

    async def apply_action_async(
        self,
        world_id: str,
        request: ApplyWorldActionRequest,
        *,
        model: WorldTransitionModel,
    ) -> WorldTransitionResult:
        return await asyncio.to_thread(
            self.apply_action,
            world_id,
            request,
            model=model,
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            runtimes = list(
                self._worlds.values()
            )

        status_counts = {
            status.value: 0
            for status in WorldStatus
        }

        for runtime in runtimes:
            with runtime.lock:
                status_counts[
                    runtime.record.status.value
                ] += 1

        return {
            "service": "world",
            "status": "ready",
            "world_count": len(runtimes),
            "worlds_by_status": status_counts,
            "max_worlds": self.max_worlds,
            "max_history_per_world": (
                self.max_history_per_world
            ),
            "max_snapshots_per_world": (
                self.max_snapshots_per_world
            ),
            "storage": "in_memory",
            "hardware_control_enabled": False,
        }


# ============================================================
# FACTORY
# ============================================================

def create_world_service() -> WorldService:
    return WorldService(
        max_worlds=100,
        max_history_per_world=500,
        max_events_per_world=1000,
        max_snapshots_per_world=50,
        max_observations_per_world=10000,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = create_world_service()

    world = service.create_world(
        CreateWorldRequest(
            name="NeuroWorld Demo",
            description=(
                "Synthetic latent-state environment"
            ),
            state_dimension=4,
            initial_state=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        )
    )

    print("=" * 60)
    print("NEUROWORLD AI — WORLD SERVICE")
    print("=" * 60)

    print("\nCREATED WORLD:")
    print(world.model_dump_json(indent=2))

    updated = service.update_state(
        world.world_id,
        UpdateWorldStateRequest(
            state=[1.0, 2.0, 0.1, 0.2],
            expected_version=0,
            source="demo_observation",
        ),
    )

    print("\nUPDATED STATE:")
    print(updated.model_dump_json(indent=2))

    snapshot = service.create_snapshot(
        world.world_id,
        name="checkpoint-1",
    )

    print("\nSNAPSHOT:")
    print(snapshot.model_dump_json(indent=2))

    service.update_state(
        world.world_id,
        UpdateWorldStateRequest(
            state=[3.0, 4.0, 0.5, 0.6],
            expected_version=1,
        ),
    )

    restored = service.restore_snapshot(
        world.world_id,
        snapshot.snapshot_id,
        expected_version=2,
    )

    print("\nRESTORED WORLD:")
    print(restored.model_dump_json(indent=2))

    print("\nSTATE HISTORY:")
    for state in service.get_state_history(
        world.world_id
    ):
        print(state.model_dump_json())

    print("\nEVENT HISTORY:")
    for event in service.get_events(
        world.world_id
    ):
        print(event.model_dump_json())

    print("\nSERVICE STATUS:")
    print(service.status())


if __name__ == "__main__":
    main()
