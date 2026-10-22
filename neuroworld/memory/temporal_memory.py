
"""
NeuroWorld AI
File: neuroworld/memory/temporal_memory.py

Temporal Memory & Time-Aware Retrieval Engine

Features:
    - Timestamped memory records
    - Event and observation history
    - State/action/reward sequences
    - Temporal sliding windows
    - Recency-weighted retrieval
    - Embedding similarity search
    - Event-type filtering
    - Time-range filtering
    - Sequence retrieval
    - Temporal pattern detection
    - Trajectory reconstruction
    - Memory capacity management
    - TTL-based retention
    - JSON persistence
    - Thread-safe operations
    - World-model training sequences
    - Agent memory integration

Dependencies:
    pip install numpy

Run:
    python -m neuroworld.memory.temporal_memory
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class TemporalMemoryError(Exception):
    pass


class TemporalMemoryValidationError(
    TemporalMemoryError
):
    pass


# ============================================================
# ENUMERATIONS
# ============================================================

class MemoryEventType(str, Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    STATE = "state"
    REWARD = "reward"
    PREDICTION = "prediction"
    COLLISION = "collision"
    INTERACTION = "interaction"
    GOAL = "goal"
    ERROR = "error"
    CUSTOM = "custom"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TemporalMemoryConfig:
    capacity: int = 10000

    embedding_dim: int = 128

    recency_half_life_seconds: float = 3600.0

    similarity_weight: float = 0.7
    recency_weight: float = 0.3

    ttl_seconds: float | None = None

    default_sequence_length: int = 16

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if self.recency_half_life_seconds <= 0:
            raise ValueError(
                "recency_half_life_seconds must be positive"
            )

        if (
            self.similarity_weight < 0
            or self.recency_weight < 0
        ):
            raise ValueError(
                "Retrieval weights must be nonnegative"
            )

        if (
            self.similarity_weight
            + self.recency_weight
            <= 0
        ):
            raise ValueError(
                "At least one retrieval weight "
                "must be positive"
            )

        if (
            self.ttl_seconds is not None
            and self.ttl_seconds <= 0
        ):
            raise ValueError(
                "ttl_seconds must be positive"
            )

        if self.default_sequence_length <= 0:
            raise ValueError(
                "default_sequence_length must be positive"
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_vector(
    value: Any,
    name: str,
    expected_dim: int | None = None,
) -> np.ndarray:
    result = np.asarray(
        value,
        dtype=np.float32,
    )

    if result.ndim != 1:
        raise TemporalMemoryValidationError(
            f"{name} must be a 1D vector"
        )

    if (
        expected_dim is not None
        and result.shape[0] != expected_dim
    ):
        raise TemporalMemoryValidationError(
            f"{name} must have {expected_dim} values"
        )

    if not np.isfinite(result).all():
        raise TemporalMemoryValidationError(
            f"{name} contains non-finite values"
        )

    return result.copy()


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    denominator = (
        float(np.linalg.norm(a))
        * float(np.linalg.norm(b))
    )

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def utc_timestamp() -> float:
    return time.time()


# ============================================================
# MEMORY RECORD
# ============================================================

@dataclass
class TemporalMemoryRecord:
    memory_id: str

    timestamp: float
    event_type: MemoryEventType

    content: dict[str, Any]

    embedding: np.ndarray | None = None

    episode_id: str | None = None
    source: str | None = None

    importance: float = 1.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise TemporalMemoryValidationError(
                "timestamp must be finite"
            )

        if not (
            0.0 <= self.importance <= 1.0
        ):
            raise TemporalMemoryValidationError(
                "importance must be in [0,1]"
            )

        if self.embedding is not None:
            self.embedding = validate_vector(
                self.embedding,
                "embedding",
            )

    @property
    def datetime_utc(self) -> str:
        return datetime.fromtimestamp(
            self.timestamp,
            tz=timezone.utc,
        ).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "content": self.content,
            "embedding": (
                self.embedding.tolist()
                if self.embedding is not None
                else None
            ),
            "episode_id": self.episode_id,
            "source": self.source,
            "importance": self.importance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> TemporalMemoryRecord:
        return cls(
            memory_id=data["memory_id"],
            timestamp=float(data["timestamp"]),
            event_type=MemoryEventType(
                data["event_type"]
            ),
            content=data["content"],
            embedding=(
                np.asarray(
                    data["embedding"],
                    dtype=np.float32,
                )
                if data.get("embedding") is not None
                else None
            ),
            episode_id=data.get("episode_id"),
            source=data.get("source"),
            importance=float(
                data.get("importance", 1.0)
            ),
            metadata=data.get(
                "metadata", {}
            ),
        )


# ============================================================
# RETRIEVAL RESULT
# ============================================================

@dataclass
class TemporalSearchResult:
    record: TemporalMemoryRecord

    score: float
    similarity: float
    recency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "score": self.score,
            "similarity": self.similarity,
            "recency": self.recency,
        }


# ============================================================
# TEMPORAL MEMORY STORE
# ============================================================

class TemporalMemory:
    """
    Thread-safe temporal memory store.

    Records are maintained in chronological
    order by insertion timestamp.

    Out-of-order timestamps are supported:
    time-range and sequence queries sort
    matching records by timestamp.
    """

    def __init__(
        self,
        config: TemporalMemoryConfig | None = None,
    ) -> None:
        self.config = (
            config or TemporalMemoryConfig()
        )

        self._records: deque[
            TemporalMemoryRecord
        ] = deque(
            maxlen=self.config.capacity
        )

        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            self._prune_expired()
            return len(self._records)

    def _prune_expired(
        self,
        now: float | None = None,
    ) -> int:
        """
        Caller must hold self._lock.

        Expired records can occur anywhere
        because timestamps may be inserted
        out of order.
        """
        if self.config.ttl_seconds is None:
            return 0

        now = (
            utc_timestamp()
            if now is None
            else now
        )

        before = len(self._records)

        valid = [
            record
            for record in self._records
            if (
                now - record.timestamp
                <= self.config.ttl_seconds
            )
        ]

        self._records = deque(
            valid,
            maxlen=self.config.capacity,
        )

        return before - len(self._records)

    def add(
        self,
        event_type: MemoryEventType | str,
        content: dict[str, Any],
        *,
        embedding: np.ndarray | None = None,
        timestamp: float | None = None,
        episode_id: str | None = None,
        source: str | None = None,
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalMemoryRecord:
        event_type = MemoryEventType(
            event_type
        )

        if not isinstance(content, dict):
            raise TemporalMemoryValidationError(
                "content must be a dictionary"
            )

        if embedding is not None:
            embedding = validate_vector(
                embedding,
                "embedding",
                self.config.embedding_dim,
            )

        record = TemporalMemoryRecord(
            memory_id=str(
                uuid.uuid4()
            ),
            timestamp=(
                utc_timestamp()
                if timestamp is None
                else float(timestamp)
            ),
            event_type=event_type,
            content=content.copy(),
            embedding=embedding,
            episode_id=episode_id,
            source=source,
            importance=importance,
            metadata=(
                metadata.copy()
                if metadata is not None
                else {}
            ),
        )

        with self._lock:
            self._prune_expired()
            self._records.append(record)

        return record

    def get(
        self,
        memory_id: str,
    ) -> TemporalMemoryRecord | None:
        with self._lock:
            self._prune_expired()

            for record in self._records:
                if record.memory_id == memory_id:
                    return record

        return None

    def recent(
        self,
        limit: int = 20,
        *,
        event_type: (
            MemoryEventType | str | None
        ) = None,
        episode_id: str | None = None,
    ) -> list[TemporalMemoryRecord]:
        if limit < 0:
            raise ValueError(
                "limit must be nonnegative"
            )

        with self._lock:
            self._prune_expired()
            records = list(
                self._records
            )

        if event_type is not None:
            event_type = MemoryEventType(
                event_type
            )

            records = [
                record
                for record in records
                if record.event_type == event_type
            ]

        if episode_id is not None:
            records = [
                record
                for record in records
                if record.episode_id == episode_id
            ]

        records.sort(
            key=lambda record: (
                record.timestamp,
                record.memory_id,
            ),
            reverse=True,
        )

        return records[:limit]

    def time_range(
        self,
        start_time: float,
        end_time: float,
        *,
        event_type: (
            MemoryEventType | str | None
        ) = None,
        episode_id: str | None = None,
    ) -> list[TemporalMemoryRecord]:
        if start_time > end_time:
            raise ValueError(
                "start_time exceeds end_time"
            )

        with self._lock:
            self._prune_expired()

            records = [
                record
                for record in self._records
                if (
                    start_time
                    <= record.timestamp
                    <= end_time
                )
            ]

        if event_type is not None:
            event_type = MemoryEventType(
                event_type
            )

            records = [
                record
                for record in records
                if record.event_type == event_type
            ]

        if episode_id is not None:
            records = [
                record
                for record in records
                if record.episode_id == episode_id
            ]

        return sorted(
            records,
            key=lambda record: (
                record.timestamp,
                record.memory_id,
            ),
        )

    def sequence(
        self,
        length: int | None = None,
        *,
        episode_id: str | None = None,
        end_time: float | None = None,
    ) -> list[TemporalMemoryRecord]:
        length = (
            self.config.default_sequence_length
            if length is None
            else length
        )

        if length <= 0:
            raise ValueError(
                "length must be positive"
            )

        with self._lock:
            self._prune_expired()
            records = list(
                self._records
            )

        if episode_id is not None:
            records = [
                record
                for record in records
                if record.episode_id == episode_id
            ]

        if end_time is not None:
            records = [
                record
                for record in records
                if record.timestamp <= end_time
            ]

        records.sort(
            key=lambda record: (
                record.timestamp,
                record.memory_id,
            )
        )

        return records[-length:]

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 10,
        event_type: (
            MemoryEventType | str | None
        ) = None,
        episode_id: str | None = None,
        now: float | None = None,
    ) -> list[TemporalSearchResult]:
        query = validate_vector(
            query_embedding,
            "query_embedding",
            self.config.embedding_dim,
        )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        now = (
            utc_timestamp()
            if now is None
            else float(now)
        )

        if not math.isfinite(now):
            raise ValueError(
                "now must be finite"
            )

        if event_type is not None:
            event_type = MemoryEventType(
                event_type
            )

        with self._lock:
            self._prune_expired(now)
            records = list(
                self._records
            )

        weight_total = (
            self.config.similarity_weight
            + self.config.recency_weight
        )

        results = []

        for record in records:
            if record.embedding is None:
                continue

            if (
                event_type is not None
                and record.event_type != event_type
            ):
                continue

            if (
                episode_id is not None
                and record.episode_id != episode_id
            ):
                continue

            similarity = cosine_similarity(
                query,
                record.embedding,
            )

            # Convert cosine similarity from
            # [-1,1] to [0,1].
            normalized_similarity = (
                similarity + 1.0
            ) / 2.0

            age = max(
                0.0,
                now - record.timestamp,
            )

            recency = math.exp(
                -math.log(2)
                * age
                / self.config.recency_half_life_seconds
            )

            score = (
                self.config.similarity_weight
                * normalized_similarity
                + self.config.recency_weight
                * recency
            ) / weight_total

            score *= record.importance

            results.append(
                TemporalSearchResult(
                    record=record,
                    score=float(score),
                    similarity=similarity,
                    recency=recency,
                )
            )

        return sorted(
            results,
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    def event_counts(
        self,
        *,
        episode_id: str | None = None,
    ) -> dict[str, int]:
        with self._lock:
            self._prune_expired()
            records = list(
                self._records
            )

        if episode_id is not None:
            records = [
                record
                for record in records
                if record.episode_id == episode_id
            ]

        counts = Counter(
            record.event_type.value
            for record in records
        )

        return dict(counts)

    def clear(
        self,
        episode_id: str | None = None,
    ) -> int:
        with self._lock:
            if episode_id is None:
                removed = len(
                    self._records
                )

                self._records.clear()
                return removed

            before = len(
                self._records
            )

            self._records = deque(
                (
                    record
                    for record in self._records
                    if record.episode_id != episode_id
                ),
                maxlen=self.config.capacity,
            )

            return (
                before - len(self._records)
            )

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        with self._lock:
            self._prune_expired()

            payload = {
                "version": 1,
                "records": [
                    record.to_dict()
                    for record in self._records
                ],
            }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_name(
            path.name + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    def load_json(
        self,
        path: str | Path,
        *,
        replace: bool = True,
    ) -> int:
        path = Path(path)

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if payload.get("version") != 1:
            raise TemporalMemoryValidationError(
                "Unsupported memory file version"
            )

        records = [
            TemporalMemoryRecord.from_dict(
                item
            )
            for item in payload["records"]
        ]

        for record in records:
            if (
                record.embedding is not None
                and len(record.embedding)
                != self.config.embedding_dim
            ):
                raise TemporalMemoryValidationError(
                    "Saved embedding dimension mismatch"
                )

        with self._lock:
            if replace:
                self._records.clear()

            for record in records:
                self._records.append(
                    record
                )

            self._prune_expired()

        return len(records)


# ============================================================
# TEMPORAL WINDOW MANAGER
# ============================================================

class TemporalWindowManager:
    """
    Extracts fixed-duration windows
    from timestamped memory.
    """

    def __init__(
        self,
        memory: TemporalMemory,
    ) -> None:
        self.memory = memory

    def latest_window(
        self,
        duration_seconds: float,
        *,
        now: float | None = None,
        episode_id: str | None = None,
    ) -> list[TemporalMemoryRecord]:
        if duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be positive"
            )

        now = (
            utc_timestamp()
            if now is None
            else float(now)
        )

        return self.memory.time_range(
            now - duration_seconds,
            now,
            episode_id=episode_id,
        )

    def sliding_windows(
        self,
        *,
        start_time: float,
        end_time: float,
        window_seconds: float,
        stride_seconds: float,
        episode_id: str | None = None,
    ) -> list[list[TemporalMemoryRecord]]:
        if (
            window_seconds <= 0
            or stride_seconds <= 0
        ):
            raise ValueError(
                "Window and stride must be positive"
            )

        if start_time > end_time:
            raise ValueError(
                "Invalid time range"
            )

        windows = []
        current = start_time

        while (
            current + window_seconds
            <= end_time + 1e-9
        ):
            window = self.memory.time_range(
                current,
                current + window_seconds,
                episode_id=episode_id,
            )

            windows.append(
                window
            )

            current += stride_seconds

        return windows


# ============================================================
# TRAJECTORY MEMORY
# ============================================================

@dataclass
class TemporalTransition:
    timestamp: float

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    reward: float
    done: bool

    episode_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "state": self.state.tolist(),
            "action": self.action.tolist(),
            "next_state": (
                self.next_state.tolist()
            ),
            "reward": self.reward,
            "done": self.done,
            "episode_id": self.episode_id,
        }


class TemporalTrajectoryMemory:
    """
    Stores world-model transitions.

    Each transition contains:
        state_t
        action_t
        state_t+1
        reward_t
        done_t

    Supports sequence extraction for
    dynamics and world-model training.
    """

    def __init__(
        self,
        capacity: int = 100000,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        self.transitions: deque[
            TemporalTransition
        ] = deque(
            maxlen=capacity
        )

        self._lock = threading.RLock()

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool,
        *,
        episode_id: str,
        timestamp: float | None = None,
    ) -> TemporalTransition:
        state = validate_vector(
            state,
            "state",
        )

        action = validate_vector(
            action,
            "action",
        )

        next_state = validate_vector(
            next_state,
            "next_state",
            len(state),
        )

        if not math.isfinite(
            float(reward)
        ):
            raise TemporalMemoryValidationError(
                "Reward must be finite"
            )

        record = TemporalTransition(
            timestamp=(
                utc_timestamp()
                if timestamp is None
                else float(timestamp)
            ),
            state=state,
            action=action,
            next_state=next_state,
            reward=float(reward),
            done=bool(done),
            episode_id=episode_id,
        )

        with self._lock:
            self.transitions.append(
                record
            )

        return record

    def episode(
        self,
        episode_id: str,
    ) -> list[TemporalTransition]:
        with self._lock:
            records = [
                item
                for item in self.transitions
                if item.episode_id == episode_id
            ]

        return sorted(
            records,
            key=lambda item: item.timestamp,
        )

    def training_sequences(
        self,
        sequence_length: int,
        *,
        episode_id: str | None = None,
        stride: int = 1,
    ) -> list[dict[str, np.ndarray]]:
        if sequence_length <= 0 or stride <= 0:
            raise ValueError(
                "Sequence length and stride "
                "must be positive"
            )

        with self._lock:
            records = list(
                self.transitions
            )

        episodes: dict[
            str,
            list[TemporalTransition],
        ] = {}

        for record in records:
            if (
                episode_id is not None
                and record.episode_id != episode_id
            ):
                continue

            episodes.setdefault(
                record.episode_id,
                [],
            ).append(record)

        sequences = []

        for episode_records in episodes.values():
            episode_records.sort(
                key=lambda item: item.timestamp
            )

            for start in range(
                0,
                len(episode_records)
                - sequence_length + 1,
                stride,
            ):
                window = episode_records[
                    start:
                    start + sequence_length
                ]

                # Do not cross a terminal
                # transition within a sequence.
                if any(
                    item.done
                    for item in window[:-1]
                ):
                    continue

                sequences.append(
                    {
                        "states": np.stack(
                            [
                                item.state
                                for item in window
                            ]
                        ),
                        "actions": np.stack(
                            [
                                item.action
                                for item in window
                            ]
                        ),
                        "next_states": np.stack(
                            [
                                item.next_state
                                for item in window
                            ]
                        ),
                        "rewards": np.asarray(
                            [
                                item.reward
                                for item in window
                            ],
                            dtype=np.float32,
                        ),
                        "dones": np.asarray(
                            [
                                item.done
                                for item in window
                            ],
                            dtype=np.bool_,
                        ),
                    }
                )

        return sequences


# ============================================================
# TEMPORAL PATTERN ANALYZER
# ============================================================

class TemporalPatternAnalyzer:
    """
    Basic temporal statistics.

    Useful for:
        - Event frequencies
        - Event intervals
        - Repeated patterns
        - Agent behavior analysis
    """

    @staticmethod
    def event_intervals(
        records: Iterable[TemporalMemoryRecord],
        event_type: MemoryEventType | str,
    ) -> np.ndarray:
        event_type = MemoryEventType(
            event_type
        )

        timestamps = sorted(
            record.timestamp
            for record in records
            if record.event_type == event_type
        )

        if len(timestamps) < 2:
            return np.empty(
                (0,),
                dtype=np.float64,
            )

        return np.diff(
            np.asarray(
                timestamps,
                dtype=np.float64,
            )
        )

    @staticmethod
    def event_frequency(
        records: Iterable[TemporalMemoryRecord],
        event_type: MemoryEventType | str,
        duration_seconds: float,
    ) -> float:
        if duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be positive"
            )

        event_type = MemoryEventType(
            event_type
        )

        count = sum(
            1
            for record in records
            if record.event_type == event_type
        )

        return (
            count / duration_seconds
        )

    @staticmethod
    def transition_counts(
        records: Iterable[TemporalMemoryRecord],
    ) -> dict[str, int]:
        ordered = sorted(
            records,
            key=lambda record: record.timestamp,
        )

        transitions = Counter()

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):
            key = (
                f"{previous.event_type.value}"
                f" -> "
                f"{current.event_type.value}"
            )

            transitions[key] += 1

        return dict(transitions)


# ============================================================
# MAIN DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - TEMPORAL MEMORY"
    )
    print("=" * 65)

    config = TemporalMemoryConfig(
        capacity=1000,
        embedding_dim=8,
        default_sequence_length=4,
    )

    memory = TemporalMemory(
        config
    )

    episode_id = "episode_001"

    base_time = utc_timestamp()

    print("\n1. Recording temporal events")

    for step in range(10):
        event_type = (
            MemoryEventType.OBSERVATION
            if step % 2 == 0
            else MemoryEventType.ACTION
        )

        embedding = np.zeros(
            8,
            dtype=np.float32,
        )

        embedding[
            step % 8
        ] = 1.0

        memory.add(
            event_type=event_type,
            content={
                "step": step,
                "position": [
                    step * 0.1,
                    0.0,
                ],
            },
            embedding=embedding,
            timestamp=(
                base_time + step
            ),
            episode_id=episode_id,
            source="simulation",
            importance=0.8,
        )

    print(
        "Total records:",
        len(memory),
    )

    print("\n2. Recent events")

    for record in memory.recent(
        limit=3
    ):
        print(
            record.event_type.value,
            record.content,
        )

    print("\n3. Similarity retrieval")

    query = np.zeros(
        8,
        dtype=np.float32,
    )

    query[0] = 1.0

    results = memory.search(
        query,
        top_k=3,
        now=base_time + 10,
    )

    for result in results:
        print(
            "Score:",
            round(result.score, 4),
            "Event:",
            result.record.content,
        )

    print("\n4. Temporal sequence")

    sequence = memory.sequence(
        length=4,
        episode_id=episode_id,
    )

    print(
        [
            record.content["step"]
            for record in sequence
        ]
    )

    print("\n5. Trajectory transitions")

    trajectory_memory = (
        TemporalTrajectoryMemory(
            capacity=100
        )
    )

    for step in range(8):
        state = np.array(
            [
                step * 0.1,
                0.0,
                0.1,
                0.0,
            ],
            dtype=np.float32,
        )

        next_state = state.copy()
        next_state[0] += 0.1

        trajectory_memory.add(
            state=state,
            action=np.array(
                [0.1, 0.0],
                dtype=np.float32,
            ),
            next_state=next_state,
            reward=1.0,
            done=step == 7,
            episode_id=episode_id,
            timestamp=base_time + step,
        )

    sequences = (
        trajectory_memory.training_sequences(
            sequence_length=4,
            episode_id=episode_id,
        )
    )

    print(
        "Training sequences:",
        len(sequences),
    )

    if sequences:
        print(
            "States shape:",
            sequences[0]["states"].shape,
        )

        print(
            "Actions shape:",
            sequences[0]["actions"].shape,
        )

    print("\n6. Event statistics")

    print(
        memory.event_counts()
    )

    print(
        "\nTemporal memory demonstration completed"
    )


if __name__ == "__main__":
    main()
