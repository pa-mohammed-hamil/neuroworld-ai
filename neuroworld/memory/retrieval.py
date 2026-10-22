
"""
NeuroWorld AI
File: neuroworld/memory/retrieval.py

World Memory Retrieval Engine

Features:
    - Episodic memory retrieval
    - Spatial memory retrieval
    - Temporal memory retrieval
    - Object memory retrieval
    - Trajectory memory retrieval
    - Vector similarity search
    - Keyword search
    - Hybrid memory ranking
    - Metadata filtering
    - Time-decay relevance
    - Spatial proximity ranking
    - Trajectory similarity
    - Memory importance weighting
    - Memory deduplication
    - Persistent JSON storage
    - Thread-safe memory access
    - Agent/world-model integration

Dependencies:
    pip install numpy

Run:
    python -m neuroworld.memory.retrieval

Notes:
    This is an exact-search reference
    implementation for small/medium
    research memory collections.

    For millions of memories, use a
    vector index such as FAISS or Qdrant.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


EPS = 1e-12


# ============================================================
# EXCEPTIONS
# ============================================================

class MemoryRetrievalError(Exception):
    pass


class MemoryValidationError(
    MemoryRetrievalError
):
    pass


# ============================================================
# ENUMERATIONS
# ============================================================

class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OBJECT = "object"
    TRAJECTORY = "trajectory"
    GENERAL = "general"


class RetrievalMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RetrievalConfig:
    embedding_dim: int = 128

    default_top_k: int = 10

    semantic_weight: float = 0.55
    keyword_weight: float = 0.15
    temporal_weight: float = 0.10
    spatial_weight: float = 0.10
    importance_weight: float = 0.10

    time_decay_hours: float = 168.0
    spatial_decay_distance: float = 10.0

    duplicate_similarity_threshold: float = 0.995

    max_memories: int = 100_000

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if self.default_top_k <= 0:
            raise ValueError(
                "default_top_k must be positive"
            )

        if self.max_memories <= 0:
            raise ValueError(
                "max_memories must be positive"
            )

        weights = [
            self.semantic_weight,
            self.keyword_weight,
            self.temporal_weight,
            self.spatial_weight,
            self.importance_weight,
        ]

        if any(
            not math.isfinite(w) or w < 0
            for w in weights
        ):
            raise ValueError(
                "Retrieval weights must be nonnegative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one weight must be positive"
            )

        if (
            self.time_decay_hours <= 0
            or self.spatial_decay_distance <= 0
        ):
            raise ValueError(
                "Decay scales must be positive"
            )

        if not (
            0 <= self.duplicate_similarity_threshold <= 1
        ):
            raise ValueError(
                "Duplicate threshold must be in [0,1]"
            )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def as_embedding(
    value: Any,
    dimension: int,
) -> np.ndarray:
    embedding = np.asarray(
        value,
        dtype=np.float32,
    )

    if embedding.shape != (
        dimension,
    ):
        raise MemoryValidationError(
            f"Embedding must be [{dimension}]"
        )

    if not np.isfinite(
        embedding
    ).all():
        raise MemoryValidationError(
            "Embedding contains non-finite values"
        )

    return embedding


def as_position(
    value: Any,
) -> np.ndarray:
    position = np.asarray(
        value,
        dtype=np.float64,
    )

    if position.shape not in (
        (2,),
        (3,),
    ):
        raise MemoryValidationError(
            "Position must be 2D or 3D"
        )

    if not np.isfinite(
        position
    ).all():
        raise MemoryValidationError(
            "Position contains non-finite values"
        )

    return position


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def tokenize(
    text: str,
) -> set[str]:
    return set(
        re.findall(
            r"\b\w+\b",
            text.lower(),
        )
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    denominator = (
        float(np.linalg.norm(a))
        * float(np.linalg.norm(b))
    )

    if denominator <= EPS:
        return 0.0

    similarity = float(
        np.dot(a, b)
        / denominator
    )

    return float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )


# ============================================================
# MEMORY RECORD
# ============================================================

@dataclass
class MemoryRecord:
    content: str

    embedding: np.ndarray

    memory_type: MemoryType = (
        MemoryType.GENERAL
    )

    memory_id: str = field(
        default_factory=lambda: str(
            uuid.uuid4()
        )
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    importance: float = 0.5

    position: np.ndarray | None = None

    trajectory: np.ndarray | None = None

    object_id: str | None = None
    episode_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.content,
            str,
        ):
            raise MemoryValidationError(
                "content must be a string"
            )

        self.embedding = np.asarray(
            self.embedding,
            dtype=np.float32,
        )

        if (
            self.embedding.ndim != 1
            or not np.isfinite(
                self.embedding
            ).all()
        ):
            raise MemoryValidationError(
                "Invalid embedding"
            )

        if not (
            0 <= self.importance <= 1
        ):
            raise MemoryValidationError(
                "importance must be in [0,1]"
            )

        if self.timestamp.tzinfo is None:
            raise MemoryValidationError(
                "timestamp must be timezone-aware"
            )

        self.timestamp = (
            self.timestamp.astimezone(
                timezone.utc
            )
        )

        if self.position is not None:
            self.position = as_position(
                self.position
            )

        if self.trajectory is not None:
            self.trajectory = np.asarray(
                self.trajectory,
                dtype=np.float64,
            )

            if (
                self.trajectory.ndim != 2
                or self.trajectory.shape[1]
                not in (2, 3)
                or self.trajectory.shape[0] == 0
                or not np.isfinite(
                    self.trajectory
                ).all()
            ):
                raise MemoryValidationError(
                    "Trajectory must be [T,2/3]"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "embedding": (
                self.embedding.tolist()
            ),
            "memory_type": (
                self.memory_type.value
            ),
            "timestamp": (
                self.timestamp.isoformat()
            ),
            "importance": self.importance,
            "position": (
                self.position.tolist()
                if self.position is not None
                else None
            ),
            "trajectory": (
                self.trajectory.tolist()
                if self.trajectory is not None
                else None
            ),
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> MemoryRecord:
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            embedding=np.asarray(
                data["embedding"],
                dtype=np.float32,
            ),
            memory_type=MemoryType(
                data["memory_type"]
            ),
            timestamp=datetime.fromisoformat(
                data["timestamp"]
            ),
            importance=float(
                data["importance"]
            ),
            position=data.get(
                "position"
            ),
            trajectory=data.get(
                "trajectory"
            ),
            object_id=data.get(
                "object_id"
            ),
            episode_id=data.get(
                "episode_id"
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )


# ============================================================
# RETRIEVAL QUERY
# ============================================================

@dataclass
class MemoryQuery:
    text: str = ""

    embedding: np.ndarray | None = None

    mode: RetrievalMode = (
        RetrievalMode.HYBRID
    )

    memory_types: set[MemoryType] | None = None

    position: np.ndarray | None = None

    trajectory: np.ndarray | None = None

    object_id: str | None = None
    episode_id: str | None = None

    metadata_filters: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    start_time: datetime | None = None
    end_time: datetime | None = None

    minimum_importance: float = 0.0

    top_k: int | None = None

    def __post_init__(self) -> None:
        if self.position is not None:
            self.position = as_position(
                self.position
            )

        if self.trajectory is not None:
            self.trajectory = np.asarray(
                self.trajectory,
                dtype=np.float64,
            )

            if (
                self.trajectory.ndim != 2
                or self.trajectory.shape[0] == 0
                or self.trajectory.shape[1]
                not in (2, 3)
                or not np.isfinite(
                    self.trajectory
                ).all()
            ):
                raise MemoryValidationError(
                    "Invalid query trajectory"
                )

        if not (
            0 <= self.minimum_importance <= 1
        ):
            raise MemoryValidationError(
                "minimum_importance must be in [0,1]"
            )

        if self.top_k is not None and self.top_k <= 0:
            raise MemoryValidationError(
                "top_k must be positive"
            )

        for name in (
            "start_time",
            "end_time",
        ):
            value = getattr(
                self,
                name,
            )

            if (
                value is not None
                and value.tzinfo is None
            ):
                raise MemoryValidationError(
                    f"{name} must be timezone-aware"
                )


# ============================================================
# RETRIEVAL RESULT
# ============================================================

@dataclass
class MemorySearchHit:
    record: MemoryRecord

    score: float

    semantic_score: float
    keyword_score: float

    temporal_score: float
    spatial_score: float

    importance_score: float
    trajectory_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": (
                self.record.memory_id
            ),
            "content": (
                self.record.content
            ),
            "memory_type": (
                self.record.memory_type.value
            ),
            "score": self.score,
            "semantic_score": (
                self.semantic_score
            ),
            "keyword_score": (
                self.keyword_score
            ),
            "temporal_score": (
                self.temporal_score
            ),
            "spatial_score": (
                self.spatial_score
            ),
            "importance_score": (
                self.importance_score
            ),
            "trajectory_score": (
                self.trajectory_score
            ),
            "metadata": (
                self.record.metadata
            ),
        }


# ============================================================
# MEMORY STORE
# ============================================================

class MemoryStore:
    """
    Thread-safe in-memory record store.

    Persistence:
        JSON file

    Note:
        JSON persistence is convenient for
        development, not large production
        memory collections.
    """

    def __init__(
        self,
        config: RetrievalConfig,
    ) -> None:
        self.config = config

        self._records: dict[
            str,
            MemoryRecord,
        ] = {}

        self._lock = (
            threading.RLock()
        )

    def add(
        self,
        record: MemoryRecord,
        *,
        deduplicate: bool = False,
    ) -> str:
        record.embedding = as_embedding(
            record.embedding,
            self.config.embedding_dim,
        )

        with self._lock:
            if deduplicate:
                duplicate = self.find_duplicate(
                    record
                )

                if duplicate is not None:
                    return duplicate.memory_id

            if (
                record.memory_id
                not in self._records
                and len(self._records)
                >= self.config.max_memories
            ):
                raise MemoryRetrievalError(
                    "Memory store capacity reached"
                )

            self._records[
                record.memory_id
            ] = record

        return record.memory_id

    def get(
        self,
        memory_id: str,
    ) -> MemoryRecord | None:
        with self._lock:
            return self._records.get(
                memory_id
            )

    def delete(
        self,
        memory_id: str,
    ) -> bool:
        with self._lock:
            return (
                self._records.pop(
                    memory_id,
                    None,
                )
                is not None
            )

    def list_all(
        self,
    ) -> list[MemoryRecord]:
        with self._lock:
            return list(
                self._records.values()
            )

    def count(self) -> int:
        with self._lock:
            return len(
                self._records
            )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def find_duplicate(
        self,
        candidate: MemoryRecord,
    ) -> MemoryRecord | None:
        with self._lock:
            for record in self._records.values():
                if (
                    record.memory_type
                    != candidate.memory_type
                ):
                    continue

                if (
                    record.object_id
                    != candidate.object_id
                ):
                    continue

                if (
                    record.episode_id
                    != candidate.episode_id
                ):
                    continue

                similarity = cosine_similarity(
                    record.embedding,
                    candidate.embedding,
                )

                if (
                    similarity
                    >= self.config.duplicate_similarity_threshold
                ):
                    return record

        return None

    def save(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            data = {
                "version": 1,
                "embedding_dim": (
                    self.config.embedding_dim
                ),
                "records": [
                    record.to_dict()
                    for record
                    in self._records.values()
                ],
            }

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary.replace(
            path
        )

    def load(
        self,
        path: str | Path,
        *,
        replace: bool = True,
    ) -> int:
        path = Path(path)

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            data["embedding_dim"]
            != self.config.embedding_dim
        ):
            raise MemoryValidationError(
                "Stored embedding dimension mismatch"
            )

        records = [
            MemoryRecord.from_dict(
                item
            )
            for item in data["records"]
        ]

        for record in records:
            as_embedding(
                record.embedding,
                self.config.embedding_dim,
            )

        with self._lock:
            merged = (
                {}
                if replace
                else dict(self._records)
            )

            for record in records:
                merged[
                    record.memory_id
                ] = record

            if (
                len(merged)
                > self.config.max_memories
            ):
                raise MemoryRetrievalError(
                    "Loaded memories exceed capacity"
                )

            self._records = merged

        return len(records)


# ============================================================
# TRAJECTORY SIMILARITY
# ============================================================

class TrajectorySimilarity:
    """
    Compare trajectories with different
    numbers of timesteps.

    Uses normalized progress resampling,
    followed by mean Euclidean distance.

    Assumes trajectories use the same
    coordinate frame and distance units.
    """

    @staticmethod
    def resample(
        trajectory: np.ndarray,
        samples: int = 32,
    ) -> np.ndarray:
        if samples <= 1:
            raise ValueError(
                "samples must exceed 1"
            )

        trajectory = np.asarray(
            trajectory,
            dtype=np.float64,
        )

        if (
            trajectory.ndim != 2
            or trajectory.shape[0] == 0
            or trajectory.shape[1]
            not in (2, 3)
            or not np.isfinite(
                trajectory
            ).all()
        ):
            raise MemoryValidationError(
                "Invalid trajectory"
            )

        if len(trajectory) == 1:
            return np.repeat(
                trajectory,
                samples,
                axis=0,
            )

        old_times = np.linspace(
            0.0,
            1.0,
            len(trajectory),
        )

        new_times = np.linspace(
            0.0,
            1.0,
            samples,
        )

        return np.stack(
            [
                np.interp(
                    new_times,
                    old_times,
                    trajectory[:, axis],
                )
                for axis in range(
                    trajectory.shape[1]
                )
            ],
            axis=-1,
        )

    @classmethod
    def similarity(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        *,
        scale: float = 5.0,
    ) -> float:
        if scale <= 0:
            raise ValueError(
                "scale must be positive"
            )

        if a.shape[1] != b.shape[1]:
            return 0.0

        a_resampled = cls.resample(
            a
        )

        b_resampled = cls.resample(
            b
        )

        mean_distance = float(
            np.linalg.norm(
                a_resampled
                - b_resampled,
                axis=-1,
            ).mean()
        )

        return math.exp(
            -mean_distance / scale
        )


# ============================================================
# MEMORY RANKER
# ============================================================

class MemoryRanker:
    """
    Hybrid ranking components:

        semantic similarity
        keyword overlap
        temporal relevance
        spatial proximity
        importance

    Trajectory similarity is included
    when the query supplies a trajectory.
    """

    def __init__(
        self,
        config: RetrievalConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def keyword_score(
        query_text: str,
        content: str,
    ) -> float:
        query_tokens = tokenize(
            query_text
        )

        if not query_tokens:
            return 0.0

        content_tokens = tokenize(
            content
        )

        return (
            len(
                query_tokens
                & content_tokens
            )
            / len(query_tokens)
        )

    def temporal_score(
        self,
        timestamp: datetime,
        now: datetime,
    ) -> float:
        age_hours = max(
            0.0,
            (
                now - timestamp
            ).total_seconds()
            / 3600.0,
        )

        return math.exp(
            -age_hours
            / self.config.time_decay_hours
        )

    def spatial_score(
        self,
        query_position: np.ndarray | None,
        record_position: np.ndarray | None,
    ) -> float:
        if (
            query_position is None
            or record_position is None
        ):
            return 0.0

        if (
            len(query_position)
            != len(record_position)
        ):
            return 0.0

        distance = float(
            np.linalg.norm(
                query_position
                - record_position
            )
        )

        return math.exp(
            -distance
            / self.config.spatial_decay_distance
        )

    def score(
        self,
        query: MemoryQuery,
        record: MemoryRecord,
        now: datetime,
    ) -> MemorySearchHit:
        semantic = 0.0

        if query.embedding is not None:
            semantic = (
                cosine_similarity(
                    query.embedding,
                    record.embedding,
                )
                + 1.0
            ) / 2.0

        keyword = self.keyword_score(
            query.text,
            record.content,
        )

        temporal = self.temporal_score(
            record.timestamp,
            now,
        )

        spatial = self.spatial_score(
            query.position,
            record.position,
        )

        importance = (
            record.importance
        )

        trajectory = 0.0

        if (
            query.trajectory is not None
            and record.trajectory is not None
        ):
            trajectory = (
                TrajectorySimilarity.similarity(
                    query.trajectory,
                    record.trajectory,
                )
            )

        if query.mode == RetrievalMode.SEMANTIC:
            total = semantic

        elif query.mode == RetrievalMode.KEYWORD:
            total = keyword

        else:
            components = [
                (
                    semantic,
                    self.config.semantic_weight,
                    query.embedding is not None,
                ),
                (
                    keyword,
                    self.config.keyword_weight,
                    bool(query.text.strip()),
                ),
                (
                    temporal,
                    self.config.temporal_weight,
                    True,
                ),
                (
                    spatial,
                    self.config.spatial_weight,
                    query.position is not None
                    and record.position is not None,
                ),
                (
                    importance,
                    self.config.importance_weight,
                    True,
                ),
            ]

            weighted_sum = sum(
                score * weight
                for score, weight, enabled
                in components
                if enabled
            )

            active_weight = sum(
                weight
                for _, weight, enabled
                in components
                if enabled
            )

            total = (
                weighted_sum / active_weight
                if active_weight > 0
                else 0.0
            )

            if query.trajectory is not None:
                if record.trajectory is not None:
                    total = (
                        0.8 * total
                        + 0.2 * trajectory
                    )
                else:
                    total *= 0.8

        return MemorySearchHit(
            record=record,
            score=float(total),
            semantic_score=semantic,
            keyword_score=keyword,
            temporal_score=temporal,
            spatial_score=spatial,
            importance_score=importance,
            trajectory_score=trajectory,
        )


# ============================================================
# MEMORY RETRIEVER
# ============================================================

class MemoryRetriever:
    """
    Main retrieval engine.

    Supports:
        semantic
        keyword
        hybrid
        memory type filtering
        metadata filtering
        time filtering
        spatial filtering
        trajectory similarity
    """

    def __init__(
        self,
        store: MemoryStore,
        config: RetrievalConfig,
        *,
        embedding_fn: (
            Callable[[str], np.ndarray]
            | None
        ) = None,
    ) -> None:
        self.store = store
        self.config = config

        self.embedding_fn = (
            embedding_fn
        )

        self.ranker = MemoryRanker(
            config
        )

    def _prepare_query(
        self,
        query: MemoryQuery,
    ) -> MemoryQuery:
        if query.embedding is None:
            if (
                query.text
                and self.embedding_fn is not None
            ):
                query.embedding = (
                    self.embedding_fn(
                        query.text
                    )
                )

        if query.embedding is not None:
            query.embedding = as_embedding(
                query.embedding,
                self.config.embedding_dim,
            )

        if (
            query.mode == RetrievalMode.SEMANTIC
            and query.embedding is None
        ):
            raise MemoryValidationError(
                "Semantic search requires an embedding"
            )

        return query

    @staticmethod
    def _matches(
        record: MemoryRecord,
        query: MemoryQuery,
    ) -> bool:
        if (
            query.memory_types is not None
            and record.memory_type
            not in query.memory_types
        ):
            return False

        if (
            query.object_id is not None
            and record.object_id
            != query.object_id
        ):
            return False

        if (
            query.episode_id is not None
            and record.episode_id
            != query.episode_id
        ):
            return False

        if (
            record.importance
            < query.minimum_importance
        ):
            return False

        if (
            query.start_time is not None
            and record.timestamp
            < query.start_time
        ):
            return False

        if (
            query.end_time is not None
            and record.timestamp
            > query.end_time
        ):
            return False

        for key, expected in (
            query.metadata_filters.items()
        ):
            if (
                record.metadata.get(key)
                != expected
            ):
                return False

        return True

    def search(
        self,
        query: MemoryQuery,
    ) -> list[MemorySearchHit]:
        query = self._prepare_query(
            query
        )

        now = utc_now()

        records = self.store.list_all()

        hits = []

        for record in records:
            if not self._matches(
                record,
                query,
            ):
                continue

            hit = self.ranker.score(
                query,
                record,
                now,
            )

            hits.append(
                hit
            )

        hits.sort(
            key=lambda hit: (
                -hit.score,
                -hit.record.importance,
                hit.record.memory_id,
            )
        )

        top_k = (
            query.top_k
            or self.config.default_top_k
        )

        return hits[:top_k]

    def semantic_search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[MemorySearchHit]:
        return self.search(
            MemoryQuery(
                embedding=embedding,
                mode=RetrievalMode.SEMANTIC,
                top_k=top_k,
            )
        )

    def keyword_search(
        self,
        text: str,
        top_k: int = 10,
    ) -> list[MemorySearchHit]:
        return self.search(
            MemoryQuery(
                text=text,
                mode=RetrievalMode.KEYWORD,
                top_k=top_k,
            )
        )

    def retrieve_nearby(
        self,
        position: np.ndarray,
        *,
        memory_types: (
            set[MemoryType] | None
        ) = None,
        top_k: int = 10,
    ) -> list[MemorySearchHit]:
        return self.search(
            MemoryQuery(
                position=position,
                memory_types=memory_types,
                top_k=top_k,
            )
        )

    def retrieve_episode(
        self,
        episode_id: str,
        top_k: int = 100,
    ) -> list[MemorySearchHit]:
        return self.search(
            MemoryQuery(
                episode_id=episode_id,
                top_k=top_k,
            )
        )

    def retrieve_object(
        self,
        object_id: str,
        top_k: int = 20,
    ) -> list[MemorySearchHit]:
        return self.search(
            MemoryQuery(
                object_id=object_id,
                top_k=top_k,
            )
        )

    def retrieve_trajectory(
        self,
        trajectory: np.ndarray,
        top_k: int = 10,
    ) -> list[MemorySearchHit]:
        return self.search(
            MemoryQuery(
                trajectory=trajectory,
                memory_types={
                    MemoryType.TRAJECTORY
                },
                top_k=top_k,
            )
        )


# ============================================================
# MEMORY CONTEXT BUILDER
# ============================================================

class MemoryContextBuilder:
    """
    Converts retrieved memories into
    structured agent context.

    Retrieved content is untrusted data,
    not executable instructions.
    """

    @staticmethod
    def build(
        hits: Iterable[MemorySearchHit],
        *,
        max_items: int = 10,
    ) -> dict[str, Any]:
        selected = list(
            hits
        )[:max_items]

        return {
            "memory_count": len(
                selected
            ),
            "memories": [
                {
                    "id": hit.record.memory_id,
                    "type": (
                        hit.record.memory_type.value
                    ),
                    "content": (
                        hit.record.content
                    ),
                    "score": round(
                        hit.score,
                        4,
                    ),
                    "timestamp": (
                        hit.record.timestamp.isoformat()
                    ),
                    "metadata": (
                        hit.record.metadata
                    ),
                }
                for hit in selected
            ],
        }


# ============================================================
# HIGH-LEVEL MEMORY ENGINE
# ============================================================

class WorldMemoryRetrievalEngine:
    """
    Unified API for NeuroWorld AI.

    Components:
        MemoryStore
        MemoryRetriever
        MemoryRanker
        MemoryContextBuilder
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        embedding_fn: (
            Callable[[str], np.ndarray]
            | None
        ) = None,
    ) -> None:
        self.config = (
            config or RetrievalConfig()
        )

        self.store = MemoryStore(
            self.config
        )

        self.retriever = MemoryRetriever(
            self.store,
            self.config,
            embedding_fn=embedding_fn,
        )

        self.context_builder = (
            MemoryContextBuilder()
        )

    def remember(
        self,
        content: str,
        embedding: np.ndarray,
        *,
        memory_type: MemoryType = (
            MemoryType.GENERAL
        ),
        importance: float = 0.5,
        position: np.ndarray | None = None,
        trajectory: np.ndarray | None = None,
        object_id: str | None = None,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        deduplicate: bool = False,
    ) -> str:
        record = MemoryRecord(
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            importance=importance,
            position=position,
            trajectory=trajectory,
            object_id=object_id,
            episode_id=episode_id,
            metadata=metadata or {},
        )

        return self.store.add(
            record,
            deduplicate=deduplicate,
        )

    def recall(
        self,
        query: MemoryQuery,
    ) -> list[MemorySearchHit]:
        return self.retriever.search(
            query
        )

    def build_context(
        self,
        query: MemoryQuery,
    ) -> dict[str, Any]:
        hits = self.recall(
            query
        )

        return self.context_builder.build(
            hits
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        self.store.save(
            path
        )

    def load(
        self,
        path: str | Path,
    ) -> int:
        return self.store.load(
            path
        )


# ============================================================
# FACTORY
# ============================================================

def create_memory_retriever(
    config: RetrievalConfig | None = None,
    *,
    embedding_fn: (
        Callable[[str], np.ndarray]
        | None
    ) = None,
) -> WorldMemoryRetrievalEngine:
    return WorldMemoryRetrievalEngine(
        config,
        embedding_fn=embedding_fn,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - MEMORY RETRIEVAL"
    )
    print("=" * 65)

    config = RetrievalConfig(
        embedding_dim=8,
        default_top_k=5,
    )

    engine = WorldMemoryRetrievalEngine(
        config
    )

    engine.remember(
        content=(
            "Robot navigated around an obstacle"
        ),
        embedding=np.array(
            [1, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        ),
        memory_type=MemoryType.EPISODIC,
        importance=0.9,
        position=np.array(
            [2.0, 3.0]
        ),
        episode_id="episode_001",
        metadata={
            "environment": "indoor"
        },
    )

    engine.remember(
        content=(
            "A moving object was detected "
            "near the robot"
        ),
        embedding=np.array(
            [0.9, 0.1, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        ),
        memory_type=MemoryType.OBJECT,
        importance=0.8,
        position=np.array(
            [2.5, 3.2]
        ),
        object_id="object_42",
    )

    engine.remember(
        content=(
            "Successful obstacle avoidance "
            "trajectory"
        ),
        embedding=np.array(
            [0.8, 0.2, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        ),
        memory_type=MemoryType.TRAJECTORY,
        importance=0.95,
        trajectory=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.5],
                [2.0, 1.0],
                [3.0, 2.0],
            ]
        ),
    )

    query = MemoryQuery(
        text="robot obstacle avoidance",
        embedding=np.array(
            [1, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        ),
        mode=RetrievalMode.HYBRID,
        position=np.array(
            [2.0, 3.0]
        ),
        top_k=3,
    )

    hits = engine.recall(
        query
    )

    print(
        "\nRetrieved memories:"
    )

    for hit in hits:
        print(
            hit.to_dict()
        )

    print(
        "\nAgent memory context:"
    )

    context = engine.build_context(
        query
    )

    print(
        json.dumps(
            context,
            indent=2,
        )
    )

    print(
        "\nTrajectory retrieval:"
    )

    trajectory_hits = (
        engine.retriever.retrieve_trajectory(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.5],
                    [2.0, 1.0],
                    [3.0, 2.0],
                ]
            )
        )
    )

    for hit in trajectory_hits:
        print(
            hit.record.content,
            hit.trajectory_score,
        )

    print(
        "\nMemory retrieval demonstration completed"
    )


if __name__ == "__main__":
    main()
