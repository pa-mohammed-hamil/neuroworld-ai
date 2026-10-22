
"""
NeuroWorld AI
File: neuroworld/memory/episodic_memory.py

Episodic Memory & Experience Retrieval Engine

Features:
    - Store agent experiences
    - Organize memories into episodes
    - Observation/action/next-state storage
    - Reward and terminal-state tracking
    - Semantic embedding similarity search
    - Temporal memory retrieval
    - Goal-conditioned memory retrieval
    - Importance-based memory ranking
    - Recency-weighted retrieval
    - Episode summaries
    - Experience replay sampling
    - Memory capacity management
    - JSON persistence
    - NumPy/PyTorch integration
    - Thread-safe memory operations

Dependencies:
    pip install numpy torch

Run:
    python -m neuroworld.memory.episodic_memory

Notes:
    - Embeddings must be supplied by an encoder.
    - No pretrained embedding model is included.
    - Similarity search is exact, not ANN-indexed.
    - JSON storage is intended for research-scale use.
"""

from __future__ import annotations

import json
import math
import random
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


EPSILON = 1e-9


# ============================================================
# EXCEPTIONS
# ============================================================

class EpisodicMemoryError(Exception):
    """Base episodic memory exception."""


class MemoryValidationError(EpisodicMemoryError):
    """Invalid memory input."""


class MemoryNotFoundError(EpisodicMemoryError):
    """Requested memory does not exist."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EpisodicMemoryConfig:
    state_dim: int = 128
    action_dim: int = 8
    embedding_dim: int = 256

    max_memories: int = 10000
    max_episodes: int = 1000

    similarity_weight: float = 0.65
    importance_weight: float = 0.20
    recency_weight: float = 0.15

    recency_half_life_hours: float = 168.0

    default_importance: float = 0.5

    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "embedding_dim",
            "max_memories",
            "max_episodes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        weights = (
            self.similarity_weight,
            self.importance_weight,
            self.recency_weight,
        )

        if any(
            not math.isfinite(weight)
            or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Retrieval weights must be finite "
                "and nonnegative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one retrieval weight "
                "must be positive"
            )

        if self.recency_half_life_hours <= 0:
            raise ValueError(
                "Recency half-life must be positive"
            )

        if not 0 <= self.default_importance <= 1:
            raise ValueError(
                "default_importance must be in [0,1]"
            )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def as_numpy(
    value: np.ndarray | Tensor | list[float],
    expected_dim: int,
    name: str,
) -> np.ndarray:
    if isinstance(value, Tensor):
        value = (
            value.detach()
            .cpu()
            .numpy()
        )

    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.shape != (expected_dim,):
        raise MemoryValidationError(
            f"{name} must have shape "
            f"({expected_dim},)"
        )

    if not np.isfinite(array).all():
        raise MemoryValidationError(
            f"{name} contains non-finite values"
        )

    return array.copy()


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def parse_timestamp(
    value: str,
) -> datetime:
    timestamp = datetime.fromisoformat(
        value
    )

    if timestamp.tzinfo is None:
        raise MemoryValidationError(
            "Timestamp must include timezone"
        )

    return timestamp.astimezone(
        timezone.utc
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    denominator = (
        float(np.linalg.norm(a))
        * float(np.linalg.norm(b))
    )

    if denominator <= EPSILON:
        return 0.0

    return float(
        np.clip(
            np.dot(a, b) / denominator,
            -1.0,
            1.0,
        )
    )


# ============================================================
# EMBEDDING ENCODER PROTOCOL
# ============================================================

class MemoryEncoder(Protocol):
    def encode(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> np.ndarray:
        ...


# ============================================================
# SIMPLE MEMORY ENCODER
# ============================================================

class RandomProjectionMemoryEncoder:
    """
    Deterministic random-projection encoder.

    This is a lightweight research baseline,
    not a pretrained semantic embedding model.

    The same seed and dimensions are required
    to reproduce embeddings.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        input_dim = (
            state_dim * 2
            + action_dim
        )

        rng = np.random.default_rng(
            seed
        )

        self.projection = (
            rng.standard_normal(
                (
                    input_dim,
                    embedding_dim,
                )
            ).astype(np.float32)
            / math.sqrt(input_dim)
        )

    def encode(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        next_observation: np.ndarray,
    ) -> np.ndarray:
        features = np.concatenate(
            [
                observation,
                action,
                next_observation,
            ]
        )

        embedding = (
            features @ self.projection
        )

        norm = float(
            np.linalg.norm(
                embedding
            )
        )

        if norm > EPSILON:
            embedding = (
                embedding / norm
            )

        return embedding.astype(
            np.float32
        )


# ============================================================
# EPISODIC MEMORY RECORD
# ============================================================

@dataclass
class EpisodicMemoryRecord:
    memory_id: str
    episode_id: str

    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray

    reward: float
    done: bool

    embedding: np.ndarray

    timestamp: str

    importance: float = 0.5

    goal_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "episode_id": self.episode_id,
            "observation": (
                self.observation.tolist()
            ),
            "action": self.action.tolist(),
            "next_observation": (
                self.next_observation.tolist()
            ),
            "reward": self.reward,
            "done": self.done,
            "embedding": (
                self.embedding.tolist()
            ),
            "timestamp": self.timestamp,
            "importance": self.importance,
            "goal_id": self.goal_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: EpisodicMemoryConfig,
    ) -> EpisodicMemoryRecord:
        return cls(
            memory_id=str(
                data["memory_id"]
            ),
            episode_id=str(
                data["episode_id"]
            ),
            observation=as_numpy(
                data["observation"],
                config.state_dim,
                "observation",
            ),
            action=as_numpy(
                data["action"],
                config.action_dim,
                "action",
            ),
            next_observation=as_numpy(
                data["next_observation"],
                config.state_dim,
                "next_observation",
            ),
            reward=float(
                data["reward"]
            ),
            done=bool(
                data["done"]
            ),
            embedding=as_numpy(
                data["embedding"],
                config.embedding_dim,
                "embedding",
            ),
            timestamp=parse_timestamp(
                data["timestamp"]
            ).isoformat(),
            importance=float(
                data["importance"]
            ),
            goal_id=data.get(
                "goal_id"
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
            ),
        )


# ============================================================
# EPISODE RECORD
# ============================================================

@dataclass
class Episode:
    episode_id: str

    memory_ids: list[str] = field(
        default_factory=list
    )

    started_at: str = field(
        default_factory=lambda: (
            utc_now().isoformat()
        )
    )

    ended_at: str | None = None

    total_reward: float = 0.0

    completed: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def length(self) -> int:
        return len(
            self.memory_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "memory_ids": self.memory_ids,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_reward": (
                self.total_reward
            ),
            "completed": self.completed,
            "metadata": self.metadata,
        }


# ============================================================
# RETRIEVAL RESULT
# ============================================================

@dataclass
class MemorySearchResult:
    memory: EpisodicMemoryRecord

    score: float
    similarity: float
    recency_score: float

    importance_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": (
                self.memory.memory_id
            ),
            "episode_id": (
                self.memory.episode_id
            ),
            "score": self.score,
            "similarity": self.similarity,
            "recency_score": (
                self.recency_score
            ),
            "importance_score": (
                self.importance_score
            ),
        }


# ============================================================
# EPISODIC MEMORY STORE
# ============================================================

class EpisodicMemory:
    """
    Thread-safe in-memory episodic store.

    Stores transitions:
        observation
        action
        reward
        next_observation
        done

    Supports:
        episode management
        semantic retrieval
        temporal retrieval
        replay sampling
        JSON persistence
    """

    def __init__(
        self,
        config: EpisodicMemoryConfig | None = None,
        encoder: MemoryEncoder | None = None,
    ) -> None:
        self.config = (
            config
            or EpisodicMemoryConfig()
        )

        self.encoder = (
            encoder
            or RandomProjectionMemoryEncoder(
                state_dim=self.config.state_dim,
                action_dim=self.config.action_dim,
                embedding_dim=(
                    self.config.embedding_dim
                ),
                seed=self.config.seed,
            )
        )

        self.memories: dict[
            str,
            EpisodicMemoryRecord,
        ] = {}

        self.episodes: dict[
            str,
            Episode,
        ] = {}

        self.lock = threading.RLock()

        self.rng = random.Random(
            self.config.seed
        )

    # --------------------------------------------------------
    # EPISODE MANAGEMENT
    # --------------------------------------------------------

    def start_episode(
        self,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.lock:
            episode_id = (
                episode_id
                or str(uuid.uuid4())
            )

            if episode_id in self.episodes:
                raise MemoryValidationError(
                    "Episode already exists"
                )

            self.episodes[
                episode_id
            ] = Episode(
                episode_id=episode_id,
                metadata=metadata or {},
            )

            self._enforce_episode_capacity()

            return episode_id

    def end_episode(
        self,
        episode_id: str,
    ) -> Episode:
        with self.lock:
            episode = self.get_episode(
                episode_id
            )

            episode.completed = True
            episode.ended_at = (
                utc_now().isoformat()
            )

            return episode

    def get_episode(
        self,
        episode_id: str,
    ) -> Episode:
        with self.lock:
            if episode_id not in self.episodes:
                raise MemoryNotFoundError(
                    f"Unknown episode: {episode_id}"
                )

            return self.episodes[
                episode_id
            ]

    # --------------------------------------------------------
    # MEMORY INSERTION
    # --------------------------------------------------------

    def add(
        self,
        episode_id: str,
        observation: np.ndarray | Tensor,
        action: np.ndarray | Tensor,
        next_observation: np.ndarray | Tensor,
        reward: float,
        done: bool,
        *,
        embedding: np.ndarray | Tensor | None = None,
        importance: float | None = None,
        goal_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self.lock:
            episode = self.get_episode(
                episode_id
            )

            if episode.completed:
                raise MemoryValidationError(
                    "Cannot add to completed episode"
                )

            observation = as_numpy(
                observation,
                self.config.state_dim,
                "observation",
            )

            action = as_numpy(
                action,
                self.config.action_dim,
                "action",
            )

            next_observation = as_numpy(
                next_observation,
                self.config.state_dim,
                "next_observation",
            )

            reward = float(
                reward
            )

            if not math.isfinite(reward):
                raise MemoryValidationError(
                    "Reward must be finite"
                )

            if importance is None:
                importance = (
                    self.config.default_importance
                )

            importance = float(
                importance
            )

            if not 0 <= importance <= 1:
                raise MemoryValidationError(
                    "Importance must be in [0,1]"
                )

            if embedding is None:
                embedding = self.encoder.encode(
                    observation,
                    action,
                    next_observation,
                )

            embedding = as_numpy(
                embedding,
                self.config.embedding_dim,
                "embedding",
            )

            memory_id = str(
                uuid.uuid4()
            )

            memory = EpisodicMemoryRecord(
                memory_id=memory_id,
                episode_id=episode_id,
                observation=observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                done=bool(done),
                embedding=embedding,
                timestamp=utc_now().isoformat(),
                importance=importance,
                goal_id=goal_id,
                metadata=metadata or {},
            )

            self.memories[
                memory_id
            ] = memory

            episode.memory_ids.append(
                memory_id
            )

            episode.total_reward += (
                reward
            )

            if done:
                episode.completed = True
                episode.ended_at = (
                    utc_now().isoformat()
                )

            self._enforce_memory_capacity()

            return memory_id

    # --------------------------------------------------------
    # MEMORY ACCESS
    # --------------------------------------------------------

    def get(
        self,
        memory_id: str,
    ) -> EpisodicMemoryRecord:
        with self.lock:
            if memory_id not in self.memories:
                raise MemoryNotFoundError(
                    f"Unknown memory: {memory_id}"
                )

            return self.memories[
                memory_id
            ]

    def get_episode_memories(
        self,
        episode_id: str,
    ) -> list[EpisodicMemoryRecord]:
        with self.lock:
            episode = self.get_episode(
                episode_id
            )

            return [
                self.memories[memory_id]
                for memory_id
                in episode.memory_ids
                if memory_id in self.memories
            ]

    # --------------------------------------------------------
    # SIMILARITY RETRIEVAL
    # --------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray | Tensor,
        *,
        top_k: int = 5,
        goal_id: str | None = None,
        episode_id: str | None = None,
        minimum_similarity: float = -1.0,
    ) -> list[MemorySearchResult]:
        if top_k <= 0:
            raise MemoryValidationError(
                "top_k must be positive"
            )

        query = as_numpy(
            query_embedding,
            self.config.embedding_dim,
            "query_embedding",
        )

        now = utc_now()

        weights = np.array(
            [
                self.config.similarity_weight,
                self.config.importance_weight,
                self.config.recency_weight,
            ],
            dtype=np.float64,
        )

        weights = (
            weights / weights.sum()
        )

        results = []

        with self.lock:
            for memory in self.memories.values():
                if (
                    goal_id is not None
                    and memory.goal_id != goal_id
                ):
                    continue

                if (
                    episode_id is not None
                    and memory.episode_id != episode_id
                ):
                    continue

                similarity = cosine_similarity(
                    query,
                    memory.embedding,
                )

                if similarity < minimum_similarity:
                    continue

                timestamp = parse_timestamp(
                    memory.timestamp
                )

                age_hours = max(
                    0.0,
                    (
                        now - timestamp
                    ).total_seconds() / 3600.0,
                )

                recency_score = (
                    0.5 ** (
                        age_hours
                        / self.config.recency_half_life_hours
                    )
                )

                # Convert cosine similarity
                # from [-1,1] into [0,1].
                normalized_similarity = (
                    similarity + 1.0
                ) / 2.0

                score = (
                    weights[0]
                    * normalized_similarity
                    + weights[1]
                    * memory.importance
                    + weights[2]
                    * recency_score
                )

                results.append(
                    MemorySearchResult(
                        memory=memory,
                        score=float(score),
                        similarity=similarity,
                        recency_score=recency_score,
                        importance_score=(
                            memory.importance
                        ),
                    )
                )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return results[
            :top_k
        ]

    # --------------------------------------------------------
    # EXPERIENCE-BASED RETRIEVAL
    # --------------------------------------------------------

    def retrieve_similar_experience(
        self,
        observation: np.ndarray | Tensor,
        action: np.ndarray | Tensor,
        next_observation: np.ndarray | Tensor,
        *,
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        observation = as_numpy(
            observation,
            self.config.state_dim,
            "observation",
        )

        action = as_numpy(
            action,
            self.config.action_dim,
            "action",
        )

        next_observation = as_numpy(
            next_observation,
            self.config.state_dim,
            "next_observation",
        )

        embedding = self.encoder.encode(
            observation,
            action,
            next_observation,
        )

        return self.search(
            embedding,
            top_k=top_k,
        )

    # --------------------------------------------------------
    # TEMPORAL RETRIEVAL
    # --------------------------------------------------------

    def retrieve_recent(
        self,
        limit: int = 10,
    ) -> list[EpisodicMemoryRecord]:
        if limit <= 0:
            raise MemoryValidationError(
                "limit must be positive"
            )

        with self.lock:
            memories = sorted(
                self.memories.values(),
                key=lambda memory: (
                    memory.timestamp
                ),
                reverse=True,
            )

            return memories[
                :limit
            ]

    def retrieve_time_range(
        self,
        start_time: str,
        end_time: str,
    ) -> list[EpisodicMemoryRecord]:
        start = parse_timestamp(
            start_time
        )

        end = parse_timestamp(
            end_time
        )

        if start > end:
            raise MemoryValidationError(
                "start_time exceeds end_time"
            )

        with self.lock:
            return [
                memory
                for memory in self.memories.values()
                if start <= parse_timestamp(
                    memory.timestamp
                ) <= end
            ]

    # --------------------------------------------------------
    # IMPORTANCE RETRIEVAL
    # --------------------------------------------------------

    def retrieve_important(
        self,
        minimum_importance: float = 0.8,
        limit: int = 20,
    ) -> list[EpisodicMemoryRecord]:
        if not 0 <= minimum_importance <= 1:
            raise MemoryValidationError(
                "Importance threshold invalid"
            )

        if limit <= 0:
            raise MemoryValidationError(
                "limit must be positive"
            )

        with self.lock:
            matches = [
                memory
                for memory in self.memories.values()
                if memory.importance
                >= minimum_importance
            ]

            matches.sort(
                key=lambda memory: (
                    memory.importance,
                    memory.timestamp,
                ),
                reverse=True,
            )

            return matches[
                :limit
            ]

    # --------------------------------------------------------
    # EXPERIENCE REPLAY
    # --------------------------------------------------------

    def sample_replay(
        self,
        batch_size: int,
    ) -> dict[str, Tensor]:
        if batch_size <= 0:
            raise MemoryValidationError(
                "batch_size must be positive"
            )

        with self.lock:
            memories = list(
                self.memories.values()
            )

            if not memories:
                raise EpisodicMemoryError(
                    "Memory store is empty"
                )

            selected = self.rng.choices(
                memories,
                k=batch_size,
            )

        observations = np.stack(
            [
                memory.observation
                for memory in selected
            ]
        )

        actions = np.stack(
            [
                memory.action
                for memory in selected
            ]
        )

        next_observations = np.stack(
            [
                memory.next_observation
                for memory in selected
            ]
        )

        rewards = np.array(
            [
                memory.reward
                for memory in selected
            ],
            dtype=np.float32,
        )

        dones = np.array(
            [
                memory.done
                for memory in selected
            ],
            dtype=np.float32,
        )

        return {
            "observations": torch.from_numpy(
                observations
            ),
            "actions": torch.from_numpy(
                actions
            ),
            "next_observations": torch.from_numpy(
                next_observations
            ),
            "rewards": torch.from_numpy(
                rewards
            ),
            "dones": torch.from_numpy(
                dones
            ),
        }

    # --------------------------------------------------------
    # EPISODE SUMMARY
    # --------------------------------------------------------

    def summarize_episode(
        self,
        episode_id: str,
    ) -> dict[str, Any]:
        with self.lock:
            episode = self.get_episode(
                episode_id
            )

            memories = (
                self.get_episode_memories(
                    episode_id
                )
            )

            rewards = [
                memory.reward
                for memory in memories
            ]

            return {
                "episode_id": episode_id,
                "length": len(memories),
                "total_reward": (
                    episode.total_reward
                ),
                "average_reward": (
                    float(np.mean(rewards))
                    if rewards
                    else 0.0
                ),
                "maximum_reward": (
                    max(rewards)
                    if rewards
                    else 0.0
                ),
                "minimum_reward": (
                    min(rewards)
                    if rewards
                    else 0.0
                ),
                "completed": (
                    episode.completed
                ),
                "started_at": (
                    episode.started_at
                ),
                "ended_at": (
                    episode.ended_at
                ),
            }

    # --------------------------------------------------------
    # CAPACITY MANAGEMENT
    # --------------------------------------------------------

    def _remove_memory(
        self,
        memory_id: str,
    ) -> None:
        memory = self.memories.pop(
            memory_id,
            None,
        )

        if memory is None:
            return

        episode = self.episodes.get(
            memory.episode_id
        )

        if episode is not None:
            if memory_id in episode.memory_ids:
                episode.memory_ids.remove(
                    memory_id
                )

            episode.total_reward -= (
                memory.reward
            )

    def _enforce_memory_capacity(
        self,
    ) -> None:
        while len(self.memories) > (
            self.config.max_memories
        ):
            oldest = min(
                self.memories.values(),
                key=lambda memory: (
                    memory.timestamp
                ),
            )

            self._remove_memory(
                oldest.memory_id
            )

    def _enforce_episode_capacity(
        self,
    ) -> None:
        while len(self.episodes) > (
            self.config.max_episodes
        ):
            oldest = min(
                self.episodes.values(),
                key=lambda episode: (
                    episode.started_at
                ),
            )

            for memory_id in list(
                oldest.memory_ids
            ):
                self._remove_memory(
                    memory_id
                )

            self.episodes.pop(
                oldest.episode_id,
                None,
            )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        path = Path(
            path
        )

        with self.lock:
            payload = {
                "version": 1,
                "config": {
                    "state_dim": (
                        self.config.state_dim
                    ),
                    "action_dim": (
                        self.config.action_dim
                    ),
                    "embedding_dim": (
                        self.config.embedding_dim
                    ),
                },
                "episodes": [
                    episode.to_dict()
                    for episode
                    in self.episodes.values()
                ],
                "memories": [
                    memory.to_dict()
                    for memory
                    in self.memories.values()
                ],
            }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = path.with_suffix(
            path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(
            path
        )

    def load(
        self,
        path: str | Path,
    ) -> None:
        path = Path(
            path
        )

        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        stored_config = payload.get(
            "config",
            {},
        )

        for name in (
            "state_dim",
            "action_dim",
            "embedding_dim",
        ):
            if stored_config.get(name) != (
                getattr(
                    self.config,
                    name,
                )
            ):
                raise MemoryValidationError(
                    f"Configuration mismatch: {name}"
                )

        new_episodes = {}

        for data in payload.get(
            "episodes",
            [],
        ):
            episode = Episode(
                episode_id=data["episode_id"],
                memory_ids=list(
                    data["memory_ids"]
                ),
                started_at=data["started_at"],
                ended_at=data.get(
                    "ended_at"
                ),
                total_reward=float(
                    data["total_reward"]
                ),
                completed=bool(
                    data["completed"]
                ),
                metadata=dict(
                    data.get(
                        "metadata",
                        {},
                    )
                ),
            )

            new_episodes[
                episode.episode_id
            ] = episode

        new_memories = {}

        for data in payload.get(
            "memories",
            [],
        ):
            memory = (
                EpisodicMemoryRecord.from_dict(
                    data,
                    self.config,
                )
            )

            new_memories[
                memory.memory_id
            ] = memory

        with self.lock:
            self.episodes = new_episodes
            self.memories = new_memories

            self._enforce_memory_capacity()
            self._enforce_episode_capacity()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self.lock:
            memories = list(
                self.memories.values()
            )

            return {
                "total_memories": (
                    len(memories)
                ),
                "total_episodes": (
                    len(self.episodes)
                ),
                "completed_episodes": sum(
                    episode.completed
                    for episode
                    in self.episodes.values()
                ),
                "average_importance": (
                    float(
                        np.mean(
                            [
                                memory.importance
                                for memory in memories
                            ]
                        )
                    )
                    if memories
                    else 0.0
                ),
                "memory_capacity": (
                    self.config.max_memories
                ),
            }

    def clear(self) -> None:
        with self.lock:
            self.memories.clear()
            self.episodes.clear()


# ============================================================
# FACTORY
# ============================================================

def create_episodic_memory(
    config: EpisodicMemoryConfig | None = None,
    encoder: MemoryEncoder | None = None,
) -> EpisodicMemory:
    return EpisodicMemory(
        config=config,
        encoder=encoder,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - EPISODIC MEMORY"
    )
    print("=" * 65)

    config = EpisodicMemoryConfig(
        state_dim=4,
        action_dim=2,
        embedding_dim=32,
        max_memories=1000,
    )

    memory = EpisodicMemory(
        config
    )

    episode_id = memory.start_episode(
        metadata={
            "environment": "navigation",
            "agent": "world_agent",
        }
    )

    print(
        "Episode:",
        episode_id,
    )

    for step in range(10):
        observation = np.array(
            [
                step * 0.1,
                0.0,
                0.1,
                0.0,
            ],
            dtype=np.float32,
        )

        action = np.array(
            [0.1, 0.0],
            dtype=np.float32,
        )

        next_observation = (
            observation.copy()
        )

        next_observation[0] += 0.1

        memory_id = memory.add(
            episode_id=episode_id,
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=1.0,
            done=(step == 9),
            importance=0.5 + step * 0.05,
            goal_id="reach_target",
        )

        print(
            "Stored memory:",
            memory_id,
        )

    print("\nEpisode summary")

    print(
        memory.summarize_episode(
            episode_id
        )
    )

    print("\nRecent memories")

    for record in memory.retrieve_recent(
        limit=3
    ):
        print(
            record.memory_id,
            record.reward,
        )

    print("\nSimilarity retrieval")

    latest = memory.retrieve_recent(
        limit=1
    )[0]

    results = memory.search(
        latest.embedding,
        top_k=3,
    )

    for result in results:
        print(
            result.memory.memory_id,
            "score:",
            round(
                result.score,
                4,
            ),
        )

    print("\nExperience replay")

    batch = memory.sample_replay(
        batch_size=4
    )

    for key, value in batch.items():
        print(
            key,
            value.shape,
        )

    print("\nSaving memory")

    memory.save(
        "data/episodic_memory.json"
    )

    print("\nStatistics")

    print(
        memory.statistics()
    )

    print(
        "\nEpisodic memory demonstration completed"
    )


if __name__ == "__main__":
    main()
    