
"""
NeuroWorld AI
File: neuroworld/memory/trajectory_memory.py

Trajectory Memory & Experience Retrieval Engine

Features:
    - Store successful and failed trajectories
    - Record states, actions, rewards and uncertainty
    - Goal-conditioned trajectory retrieval
    - State similarity search
    - Success-aware memory ranking
    - Prioritized experience sampling
    - Sequence sampling for world-model training
    - Trajectory statistics
    - Capacity-based memory eviction
    - Thread-safe memory operations
    - Save/load with NPZ and JSON
    - Memory-guided planning warm starts
    - NumPy/PyTorch integration
    - Synthetic demonstration

Dependencies:
    pip install numpy torch

Run:
    python -m neuroworld.memory.trajectory_memory

Security:
    NPZ loading uses allow_pickle=False.
    Memory files should still come from trusted sources.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class TrajectoryMemoryError(Exception):
    pass


class TrajectoryValidationError(
    TrajectoryMemoryError
):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrajectoryMemoryConfig:
    state_dim: int = 4
    action_dim: int = 2

    capacity: int = 10_000

    discount_factor: float = 0.99

    state_similarity_weight: float = 0.45
    goal_similarity_weight: float = 0.35
    success_weight: float = 0.15
    return_weight: float = 0.05

    similarity_temperature: float = 1.0

    priority_alpha: float = 0.6
    priority_epsilon: float = 1e-5

    seed: int = 42

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError(
                "state_dim must be positive"
            )

        if self.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if self.capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        if not 0 < self.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in (0,1]"
            )

        if self.similarity_temperature <= 0:
            raise ValueError(
                "similarity_temperature must be positive"
            )

        if self.priority_alpha < 0:
            raise ValueError(
                "priority_alpha must be nonnegative"
            )

        if self.priority_epsilon <= 0:
            raise ValueError(
                "priority_epsilon must be positive"
            )

        weights = (
            self.state_similarity_weight,
            self.goal_similarity_weight,
            self.success_weight,
            self.return_weight,
        )

        if any(
            not math.isfinite(w) or w < 0
            for w in weights
        ):
            raise ValueError(
                "Ranking weights must be finite "
                "and nonnegative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one ranking weight "
                "must be positive"
            )


# ============================================================
# DATA HELPERS
# ============================================================

def to_numpy(
    value: np.ndarray | Tensor | Sequence,
    name: str,
) -> np.ndarray:
    if isinstance(value, Tensor):
        value = (
            value.detach()
            .cpu()
            .numpy()
        )

    result = np.asarray(
        value,
        dtype=np.float32,
    )

    if not np.isfinite(result).all():
        raise TrajectoryValidationError(
            f"{name} contains non-finite values"
        )

    return result


def json_safe(
    value: Any,
) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

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

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    raise TrajectoryValidationError(
        f"Metadata type not JSON serializable: "
        f"{type(value).__name__}"
    )


def discounted_return(
    rewards: np.ndarray,
    gamma: float,
) -> float:
    total = 0.0
    discount = 1.0

    for reward in rewards:
        total += (
            discount * float(reward)
        )

        discount *= gamma

    return total


# ============================================================
# TRAJECTORY RECORD
# ============================================================

@dataclass
class TrajectoryRecord:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    goal_state: np.ndarray | None = None
    uncertainties: np.ndarray | None = None

    success: bool = False

    priority: float = 1.0

    trajectory_id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def horizon(self) -> int:
        return int(
            self.actions.shape[0]
        )

    @property
    def total_reward(self) -> float:
        return float(
            np.sum(self.rewards)
        )

    @property
    def initial_state(self) -> np.ndarray:
        return self.states[0]

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1]

    @property
    def mean_uncertainty(self) -> float:
        if self.uncertainties is None:
            return 0.0

        if len(self.uncertainties) == 0:
            return 0.0

        return float(
            np.mean(self.uncertainties)
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "horizon": self.horizon,
            "total_reward": self.total_reward,
            "success": self.success,
            "priority": self.priority,
            "mean_uncertainty": (
                self.mean_uncertainty
            ),
            "metadata": json_safe(
                self.metadata
            ),
        }


# ============================================================
# RETRIEVAL RESULT
# ============================================================

@dataclass
class TrajectoryMatch:
    record: TrajectoryRecord

    score: float

    state_similarity: float
    goal_similarity: float

    discounted_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": (
                self.record.trajectory_id
            ),
            "score": self.score,
            "state_similarity": (
                self.state_similarity
            ),
            "goal_similarity": (
                self.goal_similarity
            ),
            "discounted_return": (
                self.discounted_return
            ),
            "success": self.record.success,
            "horizon": self.record.horizon,
        }


# ============================================================
# SEQUENCE BATCH
# ============================================================

@dataclass
class TrajectorySequenceBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray

    next_states: np.ndarray
    uncertainties: np.ndarray

    trajectory_ids: list[str]

    def to_torch(
        self,
        device: str = "cpu",
    ) -> dict[str, Tensor]:
        return {
            "states": torch.as_tensor(
                self.states,
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.as_tensor(
                self.actions,
                dtype=torch.float32,
                device=device,
            ),
            "rewards": torch.as_tensor(
                self.rewards,
                dtype=torch.float32,
                device=device,
            ),
            "next_states": torch.as_tensor(
                self.next_states,
                dtype=torch.float32,
                device=device,
            ),
            "uncertainties": torch.as_tensor(
                self.uncertainties,
                dtype=torch.float32,
                device=device,
            ),
        }


# ============================================================
# TRAJECTORY MEMORY
# ============================================================

class TrajectoryMemory:
    """
    Thread-safe in-memory trajectory store.

    State convention:
        states:  [T+1, state_dim]
        actions: [T, action_dim]
        rewards: [T]

    Goal convention:
        goal_state: [state_dim]

    Uncertainties:
        [T]

    Eviction:
        FIFO when capacity is reached.

    Retrieval:
        Weighted state/goal similarity
        with success and return bonuses.
    """

    def __init__(
        self,
        config: TrajectoryMemoryConfig | None = None,
    ) -> None:
        self.config = (
            config or TrajectoryMemoryConfig()
        )

        self._records: list[
            TrajectoryRecord
        ] = []

        self._index: dict[
            str,
            TrajectoryRecord,
        ] = {}

        self._lock = threading.RLock()

        self._rng = np.random.default_rng(
            self.config.seed
        )

    def __len__(self) -> int:
        with self._lock:
            return len(
                self._records
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def _validate_record(
        self,
        record: TrajectoryRecord,
    ) -> TrajectoryRecord:
        states = to_numpy(
            record.states,
            "states",
        )

        actions = to_numpy(
            record.actions,
            "actions",
        )

        rewards = to_numpy(
            record.rewards,
            "rewards",
        )

        if (
            states.ndim != 2
            or states.shape[1]
            != self.config.state_dim
        ):
            raise TrajectoryValidationError(
                "states must be [T+1,state_dim]"
            )

        if (
            actions.ndim != 2
            or actions.shape[1]
            != self.config.action_dim
        ):
            raise TrajectoryValidationError(
                "actions must be [T,action_dim]"
            )

        horizon = actions.shape[0]

        if horizon <= 0:
            raise TrajectoryValidationError(
                "Trajectory horizon must be positive"
            )

        if states.shape[0] != horizon + 1:
            raise TrajectoryValidationError(
                "states length must equal "
                "actions length + 1"
            )

        if rewards.shape != (horizon,):
            raise TrajectoryValidationError(
                "rewards must be [T]"
            )

        goal = None

        if record.goal_state is not None:
            goal = to_numpy(
                record.goal_state,
                "goal_state",
            )

            if goal.shape != (
                self.config.state_dim,
            ):
                raise TrajectoryValidationError(
                    "goal_state must be [state_dim]"
                )

        uncertainties = None

        if record.uncertainties is not None:
            uncertainties = to_numpy(
                record.uncertainties,
                "uncertainties",
            )

            if uncertainties.shape != (
                horizon,
            ):
                raise TrajectoryValidationError(
                    "uncertainties must be [T]"
                )

            if np.any(uncertainties < 0):
                raise TrajectoryValidationError(
                    "uncertainties must be nonnegative"
                )

        if (
            not math.isfinite(record.priority)
            or record.priority <= 0
        ):
            raise TrajectoryValidationError(
                "priority must be positive"
            )

        if (
            not isinstance(record.trajectory_id, str)
            or not record.trajectory_id
        ):
            raise TrajectoryValidationError(
                "trajectory_id must be nonempty"
            )

        return TrajectoryRecord(
            states=states.copy(),
            actions=actions.copy(),
            rewards=rewards.copy(),
            goal_state=(
                goal.copy()
                if goal is not None
                else None
            ),
            uncertainties=(
                uncertainties.copy()
                if uncertainties is not None
                else None
            ),
            success=bool(record.success),
            priority=float(record.priority),
            trajectory_id=record.trajectory_id,
            metadata=json_safe(
                record.metadata
            ),
        )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    def add(
        self,
        record: TrajectoryRecord,
    ) -> str:
        validated = self._validate_record(
            record
        )

        with self._lock:
            if validated.trajectory_id in self._index:
                raise TrajectoryMemoryError(
                    "Duplicate trajectory ID"
                )

            if (
                len(self._records)
                >= self.config.capacity
            ):
                removed = self._records.pop(
                    0
                )

                self._index.pop(
                    removed.trajectory_id,
                    None,
                )

            self._records.append(
                validated
            )

            self._index[
                validated.trajectory_id
            ] = validated

        return validated.trajectory_id

    def store(
        self,
        states: np.ndarray | Tensor,
        actions: np.ndarray | Tensor,
        rewards: np.ndarray | Tensor,
        *,
        goal_state: (
            np.ndarray | Tensor | None
        ) = None,
        uncertainties: (
            np.ndarray | Tensor | None
        ) = None,
        success: bool = False,
        priority: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.add(
            TrajectoryRecord(
                states=states,
                actions=actions,
                rewards=rewards,
                goal_state=goal_state,
                uncertainties=uncertainties,
                success=success,
                priority=priority,
                metadata=metadata or {},
            )
        )

    def get(
        self,
        trajectory_id: str,
    ) -> TrajectoryRecord | None:
        with self._lock:
            record = self._index.get(
                trajectory_id
            )

            if record is None:
                return None

            return self._validate_record(
                record
            )

    def remove(
        self,
        trajectory_id: str,
    ) -> bool:
        with self._lock:
            record = self._index.pop(
                trajectory_id,
                None,
            )

            if record is None:
                return False

            self._records = [
                item
                for item in self._records
                if item.trajectory_id
                != trajectory_id
            ]

            return True

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._index.clear()

    # --------------------------------------------------------
    # SIMILARITY
    # --------------------------------------------------------

    def _similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        distance = float(
            np.linalg.norm(
                a - b
            )
        )

        return math.exp(
            -distance
            / self.config.similarity_temperature
        )

    def retrieve(
        self,
        current_state: np.ndarray | Tensor,
        goal_state: (
            np.ndarray | Tensor | None
        ) = None,
        *,
        top_k: int = 5,
        successful_only: bool = False,
    ) -> list[TrajectoryMatch]:
        current = to_numpy(
            current_state,
            "current_state",
        )

        if current.shape != (
            self.config.state_dim,
        ):
            raise TrajectoryValidationError(
                "current_state must be [state_dim]"
            )

        goal = None

        if goal_state is not None:
            goal = to_numpy(
                goal_state,
                "goal_state",
            )

            if goal.shape != (
                self.config.state_dim,
            ):
                raise TrajectoryValidationError(
                    "goal_state must be [state_dim]"
                )

        if top_k <= 0:
            raise TrajectoryValidationError(
                "top_k must be positive"
            )

        with self._lock:
            records = [
                self._validate_record(item)
                for item in self._records
                if (
                    not successful_only
                    or item.success
                )
            ]

        if not records:
            return []

        returns = np.array(
            [
                discounted_return(
                    record.rewards,
                    self.config.discount_factor,
                )
                for record in records
            ],
            dtype=np.float64,
        )

        return_min = float(
            returns.min()
        )

        return_max = float(
            returns.max()
        )

        return_range = (
            return_max - return_min
        )

        matches = []

        for record, total_return in zip(
            records,
            returns,
        ):
            state_similarity = (
                self._similarity(
                    current,
                    record.initial_state,
                )
            )

            if (
                goal is not None
                and record.goal_state is not None
            ):
                goal_similarity = (
                    self._similarity(
                        goal,
                        record.goal_state,
                    )
                )
                goal_available = True

            else:
                goal_similarity = 0.0
                goal_available = False

            normalized_return = (
                (
                    float(total_return)
                    - return_min
                ) / return_range
                if return_range > 1e-12
                else 0.5
            )

            weight_state = (
                self.config.state_similarity_weight
            )

            weight_goal = (
                self.config.goal_similarity_weight
                if goal_available
                else 0.0
            )

            weight_success = (
                self.config.success_weight
            )

            weight_return = (
                self.config.return_weight
            )

            denominator = (
                weight_state
                + weight_goal
                + weight_success
                + weight_return
            )

            score = (
                weight_state * state_similarity
                + weight_goal * goal_similarity
                + weight_success
                * float(record.success)
                + weight_return
                * normalized_return
            ) / max(denominator, 1e-12)

            matches.append(
                TrajectoryMatch(
                    record=record,
                    score=float(score),
                    state_similarity=(
                        state_similarity
                    ),
                    goal_similarity=(
                        goal_similarity
                    ),
                    discounted_return=float(
                        total_return
                    ),
                )
            )

        matches.sort(
            key=lambda match: match.score,
            reverse=True,
        )

        return matches[:top_k]

    # --------------------------------------------------------
    # PRIORITIZED SAMPLING
    # --------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        *,
        prioritized: bool = True,
        replace: bool = False,
    ) -> list[TrajectoryRecord]:
        if batch_size <= 0:
            raise TrajectoryValidationError(
                "batch_size must be positive"
            )

        with self._lock:
            count = len(
                self._records
            )

            if count == 0:
                raise TrajectoryMemoryError(
                    "Memory is empty"
                )

            if (
                not replace
                and batch_size > count
            ):
                raise TrajectoryMemoryError(
                    "batch_size exceeds memory size"
                )

            probabilities = None

            if prioritized:
                priorities = np.array(
                    [
                        item.priority
                        for item in self._records
                    ],
                    dtype=np.float64,
                )

                weights = np.power(
                    priorities
                    + self.config.priority_epsilon,
                    self.config.priority_alpha,
                )

                probabilities = (
                    weights / weights.sum()
                )

            indices = self._rng.choice(
                count,
                size=batch_size,
                replace=replace,
                p=probabilities,
            )

            return [
                self._validate_record(
                    self._records[int(index)]
                )
                for index in indices
            ]

    def update_priority(
        self,
        trajectory_id: str,
        priority: float,
    ) -> None:
        if (
            not math.isfinite(priority)
            or priority <= 0
        ):
            raise TrajectoryValidationError(
                "priority must be positive"
            )

        with self._lock:
            record = self._index.get(
                trajectory_id
            )

            if record is None:
                raise KeyError(
                    trajectory_id
                )

            record.priority = float(
                priority
            )

    # --------------------------------------------------------
    # SEQUENCE SAMPLING
    # --------------------------------------------------------

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> TrajectorySequenceBatch:
        """
        Samples contiguous state-action
        sequences for world-model training.

        Output:
            states        [B,L,S]
            actions       [B,L,A]
            rewards       [B,L]
            next_states   [B,L,S]
            uncertainties [B,L]
        """
        if (
            batch_size <= 0
            or sequence_length <= 0
        ):
            raise TrajectoryValidationError(
                "Batch size and sequence length "
                "must be positive"
            )

        with self._lock:
            eligible = [
                item
                for item in self._records
                if item.horizon >= sequence_length
            ]

            if not eligible:
                raise TrajectoryMemoryError(
                    "No trajectories are long enough"
                )

            sampled_states = []
            sampled_actions = []
            sampled_rewards = []
            sampled_next_states = []
            sampled_uncertainties = []
            trajectory_ids = []

            for _ in range(batch_size):
                index = int(
                    self._rng.integers(
                        len(eligible)
                    )
                )

                record = eligible[index]

                max_start = (
                    record.horizon
                    - sequence_length
                )

                start = int(
                    self._rng.integers(
                        max_start + 1
                    )
                )

                end = (
                    start + sequence_length
                )

                sampled_states.append(
                    record.states[
                        start:end
                    ]
                )

                sampled_next_states.append(
                    record.states[
                        start + 1:end + 1
                    ]
                )

                sampled_actions.append(
                    record.actions[
                        start:end
                    ]
                )

                sampled_rewards.append(
                    record.rewards[
                        start:end
                    ]
                )

                if record.uncertainties is None:
                    sampled_uncertainties.append(
                        np.zeros(
                            sequence_length,
                            dtype=np.float32,
                        )
                    )
                else:
                    sampled_uncertainties.append(
                        record.uncertainties[
                            start:end
                        ]
                    )

                trajectory_ids.append(
                    record.trajectory_id
                )

        return TrajectorySequenceBatch(
            states=np.stack(
                sampled_states
            ),
            actions=np.stack(
                sampled_actions
            ),
            rewards=np.stack(
                sampled_rewards
            ),
            next_states=np.stack(
                sampled_next_states
            ),
            uncertainties=np.stack(
                sampled_uncertainties
            ),
            trajectory_ids=trajectory_ids,
        )

    # --------------------------------------------------------
    # PLANNING WARM START
    # --------------------------------------------------------

    def get_warm_start(
        self,
        current_state: np.ndarray | Tensor,
        goal_state: np.ndarray | Tensor,
        *,
        horizon: int,
        successful_only: bool = True,
    ) -> np.ndarray | None:
        """
        Retrieves an action sequence from
        the closest relevant trajectory.

        Short trajectories are zero-padded.
        Long trajectories are truncated.

        IMPORTANT:
            Retrieved actions are suggestions.
            Revalidate them against the current
            world model and safety constraints.
        """
        if horizon <= 0:
            raise TrajectoryValidationError(
                "horizon must be positive"
            )

        matches = self.retrieve(
            current_state,
            goal_state,
            top_k=1,
            successful_only=successful_only,
        )

        if not matches:
            return None

        actions = matches[
            0
        ].record.actions

        result = np.zeros(
            (
                horizon,
                self.config.action_dim,
            ),
            dtype=np.float32,
        )

        length = min(
            horizon,
            len(actions),
        )

        result[:length] = (
            actions[:length]
        )

        return result

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            records = list(
                self._records
            )

        if not records:
            return {
                "count": 0,
                "capacity": self.config.capacity,
                "success_rate": 0.0,
                "average_horizon": 0.0,
                "average_reward": 0.0,
                "average_priority": 0.0,
            }

        return {
            "count": len(records),
            "capacity": self.config.capacity,
            "success_rate": float(
                np.mean(
                    [
                        item.success
                        for item in records
                    ]
                )
            ),
            "average_horizon": float(
                np.mean(
                    [
                        item.horizon
                        for item in records
                    ]
                )
            ),
            "average_reward": float(
                np.mean(
                    [
                        item.total_reward
                        for item in records
                    ]
                )
            ),
            "average_priority": float(
                np.mean(
                    [
                        item.priority
                        for item in records
                    ]
                )
            ),
        }

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """
        Saves each trajectory to NPZ
        and metadata to manifest.json.

        Only numeric arrays are stored
        in NPZ files.
        """
        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            records = [
                self._validate_record(item)
                for item in self._records
            ]

        manifest = {
            "format_version": 1,
            "config": {
                "state_dim": (
                    self.config.state_dim
                ),
                "action_dim": (
                    self.config.action_dim
                ),
            },
            "trajectories": [],
        }

        for index, record in enumerate(
            records
        ):
            filename = (
                f"trajectory_{index:06d}.npz"
            )

            np.savez_compressed(
                directory / filename,
                states=record.states,
                actions=record.actions,
                rewards=record.rewards,
                goal_state=(
                    record.goal_state
                    if record.goal_state is not None
                    else np.zeros(
                        self.config.state_dim,
                        dtype=np.float32,
                    )
                ),
                uncertainties=(
                    record.uncertainties
                    if record.uncertainties is not None
                    else np.zeros(
                        record.horizon,
                        dtype=np.float32,
                    )
                ),
            )

            manifest[
                "trajectories"
            ].append(
                {
                    "filename": filename,
                    "trajectory_id": (
                        record.trajectory_id
                    ),
                    "success": record.success,
                    "priority": record.priority,
                    "has_goal": (
                        record.goal_state
                        is not None
                    ),
                    "has_uncertainties": (
                        record.uncertainties
                        is not None
                    ),
                    "metadata": json_safe(
                        record.metadata
                    ),
                }
            )

        manifest_path = (
            directory / "manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(
        self,
        directory: str | Path,
        *,
        clear_existing: bool = True,
    ) -> int:
        directory = Path(
            directory
        )

        manifest_path = (
            directory / "manifest.json"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                manifest_path
            )

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if manifest.get(
            "format_version"
        ) != 1:
            raise TrajectoryMemoryError(
                "Unsupported memory format"
            )

        saved_config = manifest.get(
            "config",
            {},
        )

        if (
            saved_config.get("state_dim")
            != self.config.state_dim
            or saved_config.get("action_dim")
            != self.config.action_dim
        ):
            raise TrajectoryMemoryError(
                "Saved dimensions do not match "
                "current memory configuration"
            )

        records = []

        for entry in manifest.get(
            "trajectories",
            []
        ):
            filename = entry[
                "filename"
            ]

            # Only permit simple generated
            # filenames, not arbitrary paths.
            if (
                Path(filename).name != filename
                or not filename.endswith(".npz")
            ):
                raise TrajectoryMemoryError(
                    "Invalid trajectory filename"
                )

            with np.load(
                directory / filename,
                allow_pickle=False,
            ) as data:
                record = TrajectoryRecord(
                    states=data[
                        "states"
                    ].copy(),
                    actions=data[
                        "actions"
                    ].copy(),
                    rewards=data[
                        "rewards"
                    ].copy(),
                    goal_state=(
                        data["goal_state"].copy()
                        if entry["has_goal"]
                        else None
                    ),
                    uncertainties=(
                        data["uncertainties"].copy()
                        if entry["has_uncertainties"]
                        else None
                    ),
                    success=bool(
                        entry["success"]
                    ),
                    priority=float(
                        entry["priority"]
                    ),
                    trajectory_id=entry[
                        "trajectory_id"
                    ],
                    metadata=entry.get(
                        "metadata",
                        {},
                    ),
                )

                records.append(
                    self._validate_record(
                        record
                    )
                )

        with self._lock:
            if clear_existing:
                self._records.clear()
                self._index.clear()

            for record in records:
                if record.trajectory_id in self._index:
                    raise TrajectoryMemoryError(
                        "Duplicate trajectory ID "
                        "during loading"
                    )

                if (
                    len(self._records)
                    >= self.config.capacity
                ):
                    removed = self._records.pop(
                        0
                    )

                    self._index.pop(
                        removed.trajectory_id,
                        None,
                    )

                self._records.append(
                    record
                )

                self._index[
                    record.trajectory_id
                ] = record

        return len(records)


# ============================================================
# FACTORY
# ============================================================

def create_trajectory_memory(
    config: TrajectoryMemoryConfig | None = None,
) -> TrajectoryMemory:
    return TrajectoryMemory(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - TRAJECTORY MEMORY"
    )
    print("=" * 65)

    config = TrajectoryMemoryConfig(
        state_dim=4,
        action_dim=2,
        capacity=100,
    )

    memory = TrajectoryMemory(
        config
    )

    rng = np.random.default_rng(
        42
    )

    for index in range(10):
        horizon = 20

        states = np.zeros(
            (horizon + 1, 4),
            dtype=np.float32,
        )

        actions = rng.normal(
            0,
            0.2,
            size=(horizon, 2),
        ).astype(
            np.float32
        )

        rewards = np.zeros(
            horizon,
            dtype=np.float32,
        )

        goal = np.array(
            [2.0, 2.0, 0.0, 0.0],
            dtype=np.float32,
        )

        for step in range(horizon):
            states[step + 1] = (
                states[step]
            )

            states[
                step + 1, :2
            ] += (
                actions[step]
                * 0.1
            )

            rewards[step] = (
                -np.linalg.norm(
                    goal[:2]
                    - states[
                        step + 1, :2
                    ]
                )
            )

        success = (
            index % 2 == 0
        )

        memory.store(
            states=states,
            actions=actions,
            rewards=rewards,
            goal_state=goal,
            uncertainties=np.full(
                horizon,
                0.05,
                dtype=np.float32,
            ),
            success=success,
            priority=(
                2.0 if success else 1.0
            ),
            metadata={
                "environment": "navigation",
                "episode": index,
            },
        )

    print(
        "\nMemory statistics:"
    )

    print(
        memory.statistics()
    )

    print(
        "\nRetrieve similar trajectories:"
    )

    matches = memory.retrieve(
        current_state=np.zeros(
            4,
            dtype=np.float32,
        ),
        goal_state=np.array(
            [2.0, 2.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        top_k=3,
    )

    for match in matches:
        print(
            match.to_dict()
        )

    print(
        "\nSample world-model sequences:"
    )

    batch = memory.sample_sequences(
        batch_size=4,
        sequence_length=8,
    )

    print(
        "States:",
        batch.states.shape,
    )

    print(
        "Actions:",
        batch.actions.shape,
    )

    print(
        "Next states:",
        batch.next_states.shape,
    )

    print(
        "\nRetrieve planning warm start:"
    )

    warm_start = memory.get_warm_start(
        current_state=np.zeros(
            4,
            dtype=np.float32,
        ),
        goal_state=np.array(
            [2.0, 2.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        horizon=15,
    )

    print(
        "Warm-start shape:",
        (
            warm_start.shape
            if warm_start is not None
            else None
        ),
    )

    print(
        "\nTrajectory memory demonstration completed"
    )


if __name__ == "__main__":
    main()
