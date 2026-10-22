
"""
NeuroWorld AI
File: neuroworld/rl/imagined_replay.py

Model-Based Reinforcement Learning:
Imagined Experience Replay Engine

Features
--------
1. Real experience replay
2. World-model imagined experience replay
3. Prioritized experience replay
4. Real/imagined sampling ratios
5. Uncertainty-aware transition filtering
6. Model rollout generation
7. Reward and termination handling
8. N-step return estimation
9. Importance-sampling weights
10. Replay buffer statistics
11. Save/load replay buffers
12. PyTorch batch conversion
13. Reproducible demonstration

Designed to integrate with:
    neuroworld.rl.actor
    neuroworld.rl.critic
    neuroworld.rl.reward_model
    neuroworld.rl.model_based_rl
    neuroworld.world_model.world_model
    neuroworld.imagination.latent_rollout

Dependencies:
    pip install numpy torch

Important:
    Imagined transitions are synthetic.
    They must not be treated as ground truth.
"""

from __future__ import annotations

import copy
import math
import pickle
import random
import threading
import time

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


FloatArray = NDArray[np.float32]


# ============================================================
# EXCEPTIONS
# ============================================================

class ReplayError(Exception):
    """Base replay exception."""


class ReplayValidationError(ReplayError):
    """Invalid replay data or configuration."""


class ReplayEmptyError(ReplayError):
    """No eligible experiences are available."""


class WorldModelError(ReplayError):
    """World-model rollout generation failed."""


# ============================================================
# EXPERIENCE SOURCES
# ============================================================

class ExperienceSource(str, Enum):
    REAL = "real"
    IMAGINED = "imagined"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ImaginedReplayConfig:
    state_dim: int = 128
    action_dim: int = 8

    real_capacity: int = 100_000
    imagined_capacity: int = 200_000

    batch_size: int = 256

    imagined_ratio: float = 0.5

    gamma: float = 0.99

    prioritized: bool = True
    priority_alpha: float = 0.6
    importance_beta: float = 0.4
    priority_epsilon: float = 1e-6

    max_imagination_uncertainty: float = 0.5

    imagination_horizon: int = 5
    max_generated_transitions: int = 10_000

    uncertainty_penalty: float = 0.0

    seed: int = 42

    def __post_init__(self) -> None:
        positive_ints = (
            "state_dim",
            "action_dim",
            "real_capacity",
            "imagined_capacity",
            "batch_size",
            "imagination_horizon",
            "max_generated_transitions",
        )

        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not 0 <= self.imagined_ratio <= 1:
            raise ValueError(
                "imagined_ratio must be in [0,1]"
            )

        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma must be in [0,1]"
            )

        if self.priority_alpha < 0:
            raise ValueError(
                "priority_alpha cannot be negative"
            )

        if not 0 <= self.importance_beta <= 1:
            raise ValueError(
                "importance_beta must be in [0,1]"
            )

        if self.priority_epsilon <= 0:
            raise ValueError(
                "priority_epsilon must be positive"
            )

        if self.max_imagination_uncertainty < 0:
            raise ValueError(
                "max_imagination_uncertainty "
                "cannot be negative"
            )

        if self.uncertainty_penalty < 0:
            raise ValueError(
                "uncertainty_penalty cannot be negative"
            )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def as_float_vector(
    value: Any,
    dimension: int,
    name: str,
) -> FloatArray:
    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.shape != (dimension,):
        raise ReplayValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(array).all():
        raise ReplayValidationError(
            f"{name} contains non-finite values"
        )

    return array.copy()


def finite_scalar(
    value: Any,
    name: str,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ReplayValidationError(
            f"{name} must be finite"
        )

    return result


# ============================================================
# EXPERIENCE TRANSITION
# ============================================================

@dataclass
class ReplayTransition:
    state: FloatArray
    action: FloatArray
    reward: float
    next_state: FloatArray

    terminated: bool = False
    truncated: bool = False

    source: ExperienceSource = ExperienceSource.REAL

    uncertainty: float = 0.0
    priority: float = 1.0

    episode_id: str = ""
    step_index: int = 0

    model_version: str | None = None

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.state = np.asarray(
            self.state,
            dtype=np.float32,
        ).copy()

        self.action = np.asarray(
            self.action,
            dtype=np.float32,
        ).copy()

        self.next_state = np.asarray(
            self.next_state,
            dtype=np.float32,
        ).copy()

        if (
            self.state.ndim != 1
            or self.action.ndim != 1
            or self.next_state.ndim != 1
        ):
            raise ReplayValidationError(
                "State and action must be vectors"
            )

        if (
            self.state.shape
            != self.next_state.shape
        ):
            raise ReplayValidationError(
                "State dimensions do not match"
            )

        for name, array in (
            ("state", self.state),
            ("action", self.action),
            ("next_state", self.next_state),
        ):
            if not np.isfinite(array).all():
                raise ReplayValidationError(
                    f"{name} contains non-finite values"
                )

        self.reward = finite_scalar(
            self.reward,
            "reward",
        )

        self.uncertainty = finite_scalar(
            self.uncertainty,
            "uncertainty",
        )

        self.priority = finite_scalar(
            self.priority,
            "priority",
        )

        if self.uncertainty < 0:
            raise ReplayValidationError(
                "uncertainty cannot be negative"
            )

        if self.priority <= 0:
            raise ReplayValidationError(
                "priority must be positive"
            )

        self.source = ExperienceSource(
            self.source
        )

        if self.step_index < 0:
            raise ReplayValidationError(
                "step_index cannot be negative"
            )

    @property
    def done(self) -> bool:
        return bool(
            self.terminated or self.truncated
        )

    def clone(self) -> "ReplayTransition":
        return copy.deepcopy(self)


# ============================================================
# REPLAY BATCH
# ============================================================

@dataclass
class ReplayBatch:
    states: FloatArray
    actions: FloatArray
    rewards: FloatArray
    next_states: FloatArray

    terminated: FloatArray
    truncated: FloatArray

    uncertainties: FloatArray
    weights: FloatArray

    sources: list[ExperienceSource]

    indices: list[tuple[ExperienceSource, int]]

    probabilities: FloatArray

    def __len__(self) -> int:
        return int(
            self.states.shape[0]
        )

    def to_torch(
        self,
        device: str | torch.device = "cpu",
    ) -> dict[str, Tensor]:
        device = torch.device(device)

        return {
            "states": torch.as_tensor(
                self.states,
                device=device,
            ),
            "actions": torch.as_tensor(
                self.actions,
                device=device,
            ),
            "rewards": torch.as_tensor(
                self.rewards,
                device=device,
            ),
            "next_states": torch.as_tensor(
                self.next_states,
                device=device,
            ),
            "terminated": torch.as_tensor(
                self.terminated,
                device=device,
            ),
            "truncated": torch.as_tensor(
                self.truncated,
                device=device,
            ),
            "uncertainties": torch.as_tensor(
                self.uncertainties,
                device=device,
            ),
            "weights": torch.as_tensor(
                self.weights,
                device=device,
            ),
        }


# ============================================================
# PRIORITIZED RING BUFFER
# ============================================================

class PrioritizedRingBuffer:
    """
    Fixed-capacity transition storage.

    Supports proportional prioritized replay.

    Sampling is with replacement, matching
    the standard prioritized replay setting.
    """

    def __init__(
        self,
        capacity: int,
        *,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        prioritized: bool = True,
        seed: int = 42,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.prioritized = prioritized

        self._storage: list[
            ReplayTransition | None
        ] = [None] * capacity

        self._priorities = np.zeros(
            capacity,
            dtype=np.float64,
        )

        self._size = 0
        self._next_index = 0

        self._rng = np.random.default_rng(
            seed
        )

        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def add(
        self,
        transition: ReplayTransition,
    ) -> int:
        with self._lock:
            index = self._next_index

            self._storage[index] = (
                transition.clone()
            )

            self._priorities[index] = max(
                transition.priority,
                self.epsilon,
            )

            self._next_index = (
                self._next_index + 1
            ) % self.capacity

            self._size = min(
                self._size + 1,
                self.capacity,
            )

            return index

    def get(
        self,
        index: int,
    ) -> ReplayTransition:
        with self._lock:
            if not 0 <= index < self._size:
                raise IndexError(
                    "Replay index out of range"
                )

            transition = (
                self._storage[index]
            )

            if transition is None:
                raise ReplayError(
                    "Missing transition"
                )

            return transition.clone()

    def _sampling_probabilities(
        self,
    ) -> NDArray[np.float64]:
        if self._size == 0:
            raise ReplayEmptyError(
                "Replay buffer is empty"
            )

        if not self.prioritized:
            return np.full(
                self._size,
                1.0 / self._size,
            )

        priorities = (
            self._priorities[:self._size]
        )

        scaled = np.power(
            np.maximum(
                priorities,
                self.epsilon,
            ),
            self.alpha,
        )

        total = scaled.sum()

        if (
            not np.isfinite(total)
            or total <= 0
        ):
            return np.full(
                self._size,
                1.0 / self._size,
            )

        return scaled / total

    def sample(
        self,
        batch_size: int,
        *,
        beta: float = 0.4,
    ) -> tuple[
        list[ReplayTransition],
        NDArray[np.int64],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        if batch_size <= 0:
            raise ReplayValidationError(
                "batch_size must be positive"
            )

        if not 0 <= beta <= 1:
            raise ReplayValidationError(
                "beta must be in [0,1]"
            )

        with self._lock:
            probabilities = (
                self._sampling_probabilities()
            )

            indices = self._rng.choice(
                self._size,
                size=batch_size,
                replace=True,
                p=probabilities,
            )

            selected_probabilities = (
                probabilities[indices]
            )

            weights = np.power(
                self._size
                * selected_probabilities,
                -beta,
            )

            # Normalize by the largest possible
            # importance weight for this buffer.
            minimum_probability = (
                probabilities.min()
            )

            maximum_weight = (
                self._size
                * minimum_probability
            ) ** (-beta)

            weights = (
                weights / maximum_weight
            )

            transitions = [
                self.get(int(index))
                for index in indices
            ]

            return (
                transitions,
                indices.astype(np.int64),
                weights.astype(np.float32),
                selected_probabilities.astype(
                    np.float32
                ),
            )

    def update_priorities(
        self,
        indices: Any,
        priorities: Any,
    ) -> None:
        indices = np.asarray(
            indices,
            dtype=np.int64,
        ).reshape(-1)

        priorities = np.asarray(
            priorities,
            dtype=np.float64,
        ).reshape(-1)

        if indices.shape != priorities.shape:
            raise ReplayValidationError(
                "Priority update shape mismatch"
            )

        if (
            not np.isfinite(priorities).all()
            or np.any(priorities < 0)
        ):
            raise ReplayValidationError(
                "Priorities must be finite "
                "and nonnegative"
            )

        with self._lock:
            for index, priority in zip(
                indices,
                priorities,
            ):
                if not 0 <= index < self._size:
                    raise IndexError(
                        "Priority index out of range"
                    )

                self._priorities[index] = max(
                    float(priority),
                    self.epsilon,
                )

    def clear(self) -> None:
        with self._lock:
            self._storage = (
                [None] * self.capacity
            )

            self._priorities.fill(
                0.0
            )

            self._size = 0
            self._next_index = 0

    def transitions(
        self,
    ) -> list[ReplayTransition]:
        with self._lock:
            return [
                self.get(index)
                for index in range(
                    self._size
                )
            ]

    def state_dict(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "prioritized": self.prioritized,
                "storage": copy.deepcopy(
                    self._storage
                ),
                "priorities": (
                    self._priorities.copy()
                ),
                "size": self._size,
                "next_index": self._next_index,
                "rng_state": copy.deepcopy(
                    self._rng.bit_generator.state
                ),
            }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        if state["capacity"] != self.capacity:
            raise ReplayValidationError(
                "Buffer capacity mismatch"
            )

        with self._lock:
            self._storage = copy.deepcopy(
                state["storage"]
            )

            self._priorities = np.asarray(
                state["priorities"],
                dtype=np.float64,
            ).copy()

            self._size = int(
                state["size"]
            )

            self._next_index = int(
                state["next_index"]
            )

            self._rng.bit_generator.state = (
                copy.deepcopy(
                    state["rng_state"]
                )
            )


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

@dataclass
class ImaginedStep:
    next_state: FloatArray
    reward: float

    terminated: bool = False
    uncertainty: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class WorldModelProtocol(Protocol):
    """
    Adapter contract for imagined rollouts.

    Implement:

        predict_step(state, action) -> ImaginedStep

    The adapter can internally call any
    NeuroWorld world model, reward model,
    or termination predictor.
    """

    def predict_step(
        self,
        state: FloatArray,
        action: FloatArray,
    ) -> ImaginedStep:
        ...


class PolicyProtocol(Protocol):
    """
    Policy adapter contract.

    Implement:

        act(state) -> continuous action vector
    """

    def act(
        self,
        state: FloatArray,
    ) -> FloatArray:
        ...


# ============================================================
# IMAGINED REPLAY MANAGER
# ============================================================

class ImaginedReplayBuffer:
    """
    Maintains separate real and imagined
    experience buffers.

    Real transitions are preserved separately
    from synthetic model predictions.

    Sampling combines both sources according
    to the configured imagined ratio.
    """

    def __init__(
        self,
        config: ImaginedReplayConfig | None = None,
    ) -> None:
        self.config = (
            config or ImaginedReplayConfig()
        )

        self.real_buffer = PrioritizedRingBuffer(
            capacity=self.config.real_capacity,
            alpha=self.config.priority_alpha,
            epsilon=self.config.priority_epsilon,
            prioritized=self.config.prioritized,
            seed=self.config.seed,
        )

        self.imagined_buffer = PrioritizedRingBuffer(
            capacity=(
                self.config.imagined_capacity
            ),
            alpha=self.config.priority_alpha,
            epsilon=self.config.priority_epsilon,
            prioritized=self.config.prioritized,
            seed=self.config.seed + 1,
        )

        self._rng = np.random.default_rng(
            self.config.seed + 2
        )

        self._lock = threading.RLock()

        self.total_real_added = 0
        self.total_imagined_added = 0

        self.total_imagined_rejected = 0

    # --------------------------------------------------------
    # LENGTH
    # --------------------------------------------------------

    def __len__(self) -> int:
        return (
            len(self.real_buffer)
            + len(self.imagined_buffer)
        )

    # --------------------------------------------------------
    # TRANSITION VALIDATION
    # --------------------------------------------------------

    def _validate_transition(
        self,
        transition: ReplayTransition,
    ) -> None:
        if transition.state.shape != (
            self.config.state_dim,
        ):
            raise ReplayValidationError(
                "Invalid state dimension"
            )

        if transition.next_state.shape != (
            self.config.state_dim,
        ):
            raise ReplayValidationError(
                "Invalid next_state dimension"
            )

        if transition.action.shape != (
            self.config.action_dim,
        ):
            raise ReplayValidationError(
                "Invalid action dimension"
            )

    # --------------------------------------------------------
    # ADD REAL EXPERIENCE
    # --------------------------------------------------------

    def add_real(
        self,
        *,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        terminated: bool = False,
        truncated: bool = False,
        priority: float = 1.0,
        episode_id: str = "",
        step_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        transition = ReplayTransition(
            state=as_float_vector(
                state,
                self.config.state_dim,
                "state",
            ),
            action=as_float_vector(
                action,
                self.config.action_dim,
                "action",
            ),
            reward=reward,
            next_state=as_float_vector(
                next_state,
                self.config.state_dim,
                "next_state",
            ),
            terminated=terminated,
            truncated=truncated,
            source=ExperienceSource.REAL,
            priority=priority,
            episode_id=episode_id,
            step_index=step_index,
            metadata=metadata or {},
        )

        with self._lock:
            index = self.real_buffer.add(
                transition
            )

            self.total_real_added += 1

            return index

    # --------------------------------------------------------
    # ADD IMAGINED EXPERIENCE
    # --------------------------------------------------------

    def add_imagined(
        self,
        *,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        uncertainty: float,
        terminated: bool = False,
        priority: float = 1.0,
        episode_id: str = "",
        step_index: int = 0,
        model_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int | None:
        transition = ReplayTransition(
            state=as_float_vector(
                state,
                self.config.state_dim,
                "state",
            ),
            action=as_float_vector(
                action,
                self.config.action_dim,
                "action",
            ),
            reward=reward,
            next_state=as_float_vector(
                next_state,
                self.config.state_dim,
                "next_state",
            ),
            terminated=terminated,
            source=ExperienceSource.IMAGINED,
            uncertainty=uncertainty,
            priority=priority,
            episode_id=episode_id,
            step_index=step_index,
            model_version=model_version,
            metadata=metadata or {},
        )

        self._validate_transition(
            transition
        )

        with self._lock:
            if (
                transition.uncertainty
                > self.config.max_imagination_uncertainty
            ):
                self.total_imagined_rejected += 1

                return None

            index = self.imagined_buffer.add(
                transition
            )

            self.total_imagined_added += 1

            return index

    # --------------------------------------------------------
    # MIXED BATCH SAMPLING
    # --------------------------------------------------------

    def sample(
        self,
        batch_size: int | None = None,
        *,
        imagined_ratio: float | None = None,
        beta: float | None = None,
    ) -> ReplayBatch:
        if batch_size is None:
            batch_size = (
                self.config.batch_size
            )

        if imagined_ratio is None:
            imagined_ratio = (
                self.config.imagined_ratio
            )

        if beta is None:
            beta = (
                self.config.importance_beta
            )

        if batch_size <= 0:
            raise ReplayValidationError(
                "batch_size must be positive"
            )

        if not 0 <= imagined_ratio <= 1:
            raise ReplayValidationError(
                "imagined_ratio must be in [0,1]"
            )

        with self._lock:
            real_available = (
                len(self.real_buffer) > 0
            )

            imagined_available = (
                len(self.imagined_buffer) > 0
            )

            if (
                not real_available
                and not imagined_available
            ):
                raise ReplayEmptyError(
                    "Both replay buffers are empty"
                )

            imagined_count = int(
                round(
                    batch_size * imagined_ratio
                )
            )

            real_count = (
                batch_size - imagined_count
            )

            # Fallback when one source is empty.
            if not real_available:
                imagined_count = batch_size
                real_count = 0

            elif not imagined_available:
                real_count = batch_size
                imagined_count = 0

            all_transitions: list[
                ReplayTransition
            ] = []

            all_indices: list[
                tuple[ExperienceSource, int]
            ] = []

            all_weights: list[float] = []
            all_probabilities: list[float] = []

            source_counts = (
                (
                    ExperienceSource.REAL,
                    self.real_buffer,
                    real_count,
                ),
                (
                    ExperienceSource.IMAGINED,
                    self.imagined_buffer,
                    imagined_count,
                ),
            )

            for source, buffer, count in source_counts:
                if count == 0:
                    continue

                (
                    transitions,
                    indices,
                    weights,
                    probabilities,
                ) = buffer.sample(
                    count,
                    beta=beta,
                )

                source_probability = (
                    count / batch_size
                )

                # Joint sampling probability:
                # P(source) * P(index | source)
                joint_probabilities = (
                    source_probability
                    * probabilities
                )

                all_transitions.extend(
                    transitions
                )

                all_indices.extend(
                    [
                        (
                            source,
                            int(index),
                        )
                        for index in indices
                    ]
                )

                # Correct source-specific PER weights
                # by accounting for the mixture.
                #
                # This yields unnormalized joint
                # importance weights.
                source_size = len(buffer)

                corrected_weights = (
                    (
                        source_size
                        * joint_probabilities
                    )
                    ** (-beta)
                )

                all_weights.extend(
                    corrected_weights.tolist()
                )

                all_probabilities.extend(
                    joint_probabilities.tolist()
                )

            permutation = self._rng.permutation(
                len(all_transitions)
            )

            transitions = [
                all_transitions[i]
                for i in permutation
            ]

            indices = [
                all_indices[i]
                for i in permutation
            ]

            weights = np.asarray(
                [
                    all_weights[i]
                    for i in permutation
                ],
                dtype=np.float32,
            )

            weights /= max(
                float(weights.max()),
                1e-8,
            )

            probabilities = np.asarray(
                [
                    all_probabilities[i]
                    for i in permutation
                ],
                dtype=np.float32,
            )

            return ReplayBatch(
                states=np.stack(
                    [
                        item.state
                        for item in transitions
                    ]
                ),
                actions=np.stack(
                    [
                        item.action
                        for item in transitions
                    ]
                ),
                rewards=np.asarray(
                    [
                        item.reward
                        for item in transitions
                    ],
                    dtype=np.float32,
                ),
                next_states=np.stack(
                    [
                        item.next_state
                        for item in transitions
                    ]
                ),
                terminated=np.asarray(
                    [
                        float(item.terminated)
                        for item in transitions
                    ],
                    dtype=np.float32,
                ),
                truncated=np.asarray(
                    [
                        float(item.truncated)
                        for item in transitions
                    ],
                    dtype=np.float32,
                ),
                uncertainties=np.asarray(
                    [
                        item.uncertainty
                        for item in transitions
                    ],
                    dtype=np.float32,
                ),
                weights=weights,
                sources=[
                    item.source
                    for item in transitions
                ],
                indices=indices,
                probabilities=probabilities,
            )

    # --------------------------------------------------------
    # UPDATE PRIORITIES
    # --------------------------------------------------------

    def update_priorities(
        self,
        indices: list[
            tuple[ExperienceSource, int]
        ],
        td_errors: Any,
    ) -> None:
        errors = np.asarray(
            td_errors,
            dtype=np.float64,
        ).reshape(-1)

        if len(indices) != len(errors):
            raise ReplayValidationError(
                "Priority update length mismatch"
            )

        if not np.isfinite(errors).all():
            raise ReplayValidationError(
                "Non-finite TD errors"
            )

        real_indices = []
        real_priorities = []

        imagined_indices = []
        imagined_priorities = []

        for (
            source,
            index,
        ), error in zip(
            indices,
            errors,
        ):
            priority = (
                abs(float(error))
                + self.config.priority_epsilon
            )

            if source == ExperienceSource.REAL:
                real_indices.append(
                    index
                )

                real_priorities.append(
                    priority
                )

            elif source == ExperienceSource.IMAGINED:
                imagined_indices.append(
                    index
                )

                imagined_priorities.append(
                    priority
                )

            else:
                raise ReplayValidationError(
                    "Unknown experience source"
                )

        with self._lock:
            if real_indices:
                self.real_buffer.update_priorities(
                    real_indices,
                    real_priorities,
                )

            if imagined_indices:
                self.imagined_buffer.update_priorities(
                    imagined_indices,
                    imagined_priorities,
                )

    # --------------------------------------------------------
    # GENERATE IMAGINED ROLLOUTS
    # --------------------------------------------------------

    def generate_imagined_rollouts(
        self,
        *,
        world_model: WorldModelProtocol,
        policy: PolicyProtocol,
        initial_states: Any,
        horizon: int | None = None,
        model_version: str | None = None,
    ) -> dict[str, int]:
        """
        Generate synthetic trajectories.

        world_model.predict_step(state, action)
            -> ImaginedStep

        policy.act(state)
            -> continuous action vector

        Generation stops on:
            - model-predicted termination
            - excessive uncertainty
            - rollout horizon
            - generation limit

        This function assumes model and
        policy adapters handle their own
        inference/device management.
        """

        states = np.asarray(
            initial_states,
            dtype=np.float32,
        )

        if states.ndim == 1:
            states = states[None, :]

        if (
            states.ndim != 2
            or states.shape[1]
            != self.config.state_dim
        ):
            raise ReplayValidationError(
                "initial_states must have shape "
                f"[N,{self.config.state_dim}]"
            )

        if not np.isfinite(states).all():
            raise ReplayValidationError(
                "initial_states are non-finite"
            )

        if horizon is None:
            horizon = (
                self.config.imagination_horizon
            )

        if horizon <= 0:
            raise ReplayValidationError(
                "horizon must be positive"
            )

        accepted = 0
        rejected = 0
        predicted_terminal = 0
        generated = 0

        for rollout_index, initial_state in enumerate(
            states
        ):
            if (
                generated
                >= self.config.max_generated_transitions
            ):
                break

            current_state = (
                initial_state.copy()
            )

            episode_id = (
                f"imagined_{int(time.time() * 1000)}"
                f"_{rollout_index}"
            )

            for step in range(horizon):
                if (
                    generated
                    >= self.config.max_generated_transitions
                ):
                    break

                try:
                    action = as_float_vector(
                        policy.act(
                            current_state.copy()
                        ),
                        self.config.action_dim,
                        "policy action",
                    )

                    prediction = (
                        world_model.predict_step(
                            current_state.copy(),
                            action.copy(),
                        )
                    )

                    next_state = as_float_vector(
                        prediction.next_state,
                        self.config.state_dim,
                        "predicted next_state",
                    )

                    uncertainty = finite_scalar(
                        prediction.uncertainty,
                        "prediction uncertainty",
                    )

                    reward = finite_scalar(
                        prediction.reward,
                        "predicted reward",
                    )

                except Exception as exc:
                    raise WorldModelError(
                        "Imagined rollout failed at "
                        f"rollout={rollout_index}, "
                        f"step={step}"
                    ) from exc

                generated += 1

                if (
                    uncertainty
                    > self.config.max_imagination_uncertainty
                ):
                    rejected += 1

                    self.total_imagined_rejected += 1

                    # Stop this rollout to avoid
                    # compounding model error.
                    break

                penalized_reward = (
                    reward
                    - self.config.uncertainty_penalty
                    * uncertainty
                )

                result = self.add_imagined(
                    state=current_state,
                    action=action,
                    reward=penalized_reward,
                    next_state=next_state,
                    uncertainty=uncertainty,
                    terminated=prediction.terminated,
                    episode_id=episode_id,
                    step_index=step,
                    model_version=model_version,
                    metadata={
                        **prediction.metadata,
                        "raw_model_reward": reward,
                        "rollout_index": rollout_index,
                    },
                )

                if result is not None:
                    accepted += 1

                if prediction.terminated:
                    predicted_terminal += 1
                    break

                current_state = next_state

        return {
            "generated": generated,
            "accepted": accepted,
            "rejected": rejected,
            "predicted_terminal": (
                predicted_terminal
            ),
        }

    # --------------------------------------------------------
    # START STATES FROM REAL EXPERIENCE
    # --------------------------------------------------------

    def sample_real_start_states(
        self,
        count: int,
    ) -> FloatArray:
        if count <= 0:
            raise ReplayValidationError(
                "count must be positive"
            )

        transitions, _, _, _ = (
            self.real_buffer.sample(
                count,
                beta=0.0,
            )
        )

        return np.stack(
            [
                transition.state
                for transition in transitions
            ]
        )

    # --------------------------------------------------------
    # MODEL VERSION MANAGEMENT
    # --------------------------------------------------------

    def discard_stale_imagined(
        self,
        current_model_version: str,
    ) -> int:
        """
        Remove imagined transitions produced
        by a different world-model version.

        Rebuilds the imagined buffer.
        """

        with self._lock:
            transitions = (
                self.imagined_buffer.transitions()
            )

            retained = [
                transition
                for transition in transitions
                if transition.model_version
                == current_model_version
            ]

            removed = (
                len(transitions)
                - len(retained)
            )

            self.imagined_buffer.clear()

            for transition in retained:
                self.imagined_buffer.add(
                    transition
                )

            return removed

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            real_count = len(
                self.real_buffer
            )

            imagined_count = len(
                self.imagined_buffer
            )

            imagined_transitions = (
                self.imagined_buffer.transitions()
            )

            if imagined_transitions:
                mean_uncertainty = float(
                    np.mean(
                        [
                            transition.uncertainty
                            for transition
                            in imagined_transitions
                        ]
                    )
                )

            else:
                mean_uncertainty = 0.0

            return {
                "real_count": real_count,
                "imagined_count": imagined_count,
                "total_count": (
                    real_count + imagined_count
                ),
                "real_capacity": (
                    self.config.real_capacity
                ),
                "imagined_capacity": (
                    self.config.imagined_capacity
                ),
                "total_real_added": (
                    self.total_real_added
                ),
                "total_imagined_added": (
                    self.total_imagined_added
                ),
                "total_imagined_rejected": (
                    self.total_imagined_rejected
                ),
                "mean_imagined_uncertainty": (
                    mean_uncertainty
                ),
                "configured_imagined_ratio": (
                    self.config.imagined_ratio
                ),
            }

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> Path:
        """
        Pickle is used for local trusted
        checkpoints only.

        Never load untrusted pickle files.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "config": self.config,
                "real_buffer": (
                    self.real_buffer.state_dict()
                ),
                "imagined_buffer": (
                    self.imagined_buffer.state_dict()
                ),
                "rng_state": copy.deepcopy(
                    self._rng.bit_generator.state
                ),
                "total_real_added": (
                    self.total_real_added
                ),
                "total_imagined_added": (
                    self.total_imagined_added
                ),
                "total_imagined_rejected": (
                    self.total_imagined_rejected
                ),
            }

        with path.open("wb") as file:
            pickle.dump(
                payload,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ImaginedReplayBuffer":
        """
        Load a trusted local checkpoint.
        """

        with Path(path).open("rb") as file:
            payload = pickle.load(
                file
            )

        replay = cls(
            payload["config"]
        )

        replay.real_buffer.load_state_dict(
            payload["real_buffer"]
        )

        replay.imagined_buffer.load_state_dict(
            payload["imagined_buffer"]
        )

        replay._rng.bit_generator.state = (
            copy.deepcopy(
                payload["rng_state"]
            )
        )

        replay.total_real_added = (
            payload["total_real_added"]
        )

        replay.total_imagined_added = (
            payload["total_imagined_added"]
        )

        replay.total_imagined_rejected = (
            payload["total_imagined_rejected"]
        )

        return replay


# ============================================================
# N-STEP RETURN ESTIMATION
# ============================================================

def calculate_n_step_return(
    rewards: Any,
    *,
    gamma: float = 0.99,
    bootstrap_value: float = 0.0,
    terminated: Any | None = None,
) -> float:
    """
    Discounted n-step return:

        G = r0 + gamma*r1 + ... +
            gamma^n * V(s_n)

    Stops bootstrapping at true terminal
    transitions.

    Input rewards must be an ordered
    contiguous trajectory segment.
    """

    rewards = np.asarray(
        rewards,
        dtype=np.float64,
    ).reshape(-1)

    if rewards.size == 0:
        raise ReplayValidationError(
            "rewards cannot be empty"
        )

    if not np.isfinite(rewards).all():
        raise ReplayValidationError(
            "rewards contain non-finite values"
        )

    if not 0 <= gamma <= 1:
        raise ReplayValidationError(
            "gamma must be in [0,1]"
        )

    bootstrap_value = finite_scalar(
        bootstrap_value,
        "bootstrap_value",
    )

    if terminated is None:
        terminated = np.zeros(
            rewards.size,
            dtype=np.bool_,
        )

    terminated = np.asarray(
        terminated,
        dtype=np.bool_,
    ).reshape(-1)

    if terminated.shape != rewards.shape:
        raise ReplayValidationError(
            "terminated shape mismatch"
        )

    result = 0.0
    discount = 1.0

    for reward, done in zip(
        rewards,
        terminated,
    ):
        result += (
            discount * float(reward)
        )

        if done:
            return result

        discount *= gamma

    return (
        result
        + discount * bootstrap_value
    )


# ============================================================
# PYTORCH POLICY ADAPTER
# ============================================================

class TorchActorAdapter:
    """
    Connects neuroworld.rl.actor to
    the imagined rollout interface.

    Supports continuous-action actors
    and the continuous branch of a
    hybrid actor.

    For hybrid environments, implement
    a custom adapter that preserves
    discrete actions as well.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        deterministic: bool = False,
    ) -> None:
        self.actor = actor
        self.device = torch.device(
            device
        )

        self.deterministic = deterministic

        self.actor.to(
            self.device
        )

    @torch.no_grad()
    def act(
        self,
        state: FloatArray,
    ) -> FloatArray:
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.actor.eval()

        output = self.actor(
            state_tensor,
            deterministic=self.deterministic,
        )

        action = getattr(
            output,
            "continuous_action",
            None,
        )

        if action is None:
            raise ReplayValidationError(
                "Actor has no continuous action"
            )

        return (
            action.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


# ============================================================
# WORLD MODEL FUNCTION ADAPTER
# ============================================================

class CallableWorldModelAdapter:
    """
    Wraps a function as a world-model
    rollout adapter.

    Function signature:

        fn(state, action) -> ImaginedStep

    This adapter makes integration possible
    without assuming one specific world
    model's prediction API.
    """

    def __init__(
        self,
        predict_fn: Callable[
            [FloatArray, FloatArray],
            ImaginedStep,
        ],
    ) -> None:
        self.predict_fn = predict_fn

    def predict_step(
        self,
        state: FloatArray,
        action: FloatArray,
    ) -> ImaginedStep:
        result = self.predict_fn(
            state,
            action,
        )

        if not isinstance(
            result,
            ImaginedStep,
        ):
            raise WorldModelError(
                "World model must return ImaginedStep"
            )

        return result


# ============================================================
# FACTORY
# ============================================================

def create_imagined_replay(
    config: ImaginedReplayConfig | None = None,
) -> ImaginedReplayBuffer:
    return ImaginedReplayBuffer(
        config
    )


# ============================================================
# DEMO WORLD MODEL
# ============================================================

class DemoWorldModel:
    """
    Simple synthetic dynamics for testing.

    This is not a trained neural world model.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

    def predict_step(
        self,
        state: FloatArray,
        action: FloatArray,
    ) -> ImaginedStep:
        next_state = state.copy()

        affected_dimensions = min(
            self.state_dim,
            self.action_dim,
        )

        next_state[:affected_dimensions] += (
            0.1
            * action[:affected_dimensions]
        )

        reward = -float(
            np.linalg.norm(
                next_state
            )
        )

        uncertainty = (
            0.05
            + 0.01
            * float(
                np.linalg.norm(action)
            )
        )

        terminated = bool(
            np.linalg.norm(next_state) < 0.1
        )

        return ImaginedStep(
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            uncertainty=uncertainty,
        )


# ============================================================
# DEMO POLICY
# ============================================================

class DemoPolicy:
    """
    Deterministic controller used only
    to demonstrate replay generation.
    """

    def __init__(
        self,
        action_dim: int,
    ) -> None:
        self.action_dim = action_dim

    def act(
        self,
        state: FloatArray,
    ) -> FloatArray:
        action = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        count = min(
            self.action_dim,
            state.size,
        )

        action[:count] = np.clip(
            -state[:count],
            -1.0,
            1.0,
        )

        return action


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - IMAGINED EXPERIENCE REPLAY"
    )
    print("=" * 65)

    config = ImaginedReplayConfig(
        state_dim=16,
        action_dim=4,
        real_capacity=1000,
        imagined_capacity=2000,
        batch_size=32,
        imagined_ratio=0.5,
        imagination_horizon=5,
        max_imagination_uncertainty=0.3,
    )

    replay = create_imagined_replay(
        config
    )

    rng = np.random.default_rng(
        42
    )

    # --------------------------------------------------------
    # ADD REAL EXPERIENCES
    # --------------------------------------------------------

    for step in range(100):
        state = rng.normal(
            size=config.state_dim
        ).astype(np.float32)

        action = rng.uniform(
            -1,
            1,
            size=config.action_dim,
        ).astype(np.float32)

        next_state = state.copy()

        next_state[:config.action_dim] += (
            action * 0.1
        )

        reward = -float(
            np.linalg.norm(next_state)
        )

        replay.add_real(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=False,
            episode_id="demo_real",
            step_index=step,
        )

    # --------------------------------------------------------
    # GENERATE IMAGINED EXPERIENCES
    # --------------------------------------------------------

    world_model = DemoWorldModel(
        config.state_dim,
        config.action_dim,
    )

    policy = DemoPolicy(
        config.action_dim
    )

    initial_states = (
        replay.sample_real_start_states(
            20
        )
    )

    generation = (
        replay.generate_imagined_rollouts(
            world_model=world_model,
            policy=policy,
            initial_states=initial_states,
            horizon=5,
            model_version="demo_v1",
        )
    )

    print(
        "Generation statistics:",
        generation,
    )

    # --------------------------------------------------------
    # SAMPLE MIXED BATCH
    # --------------------------------------------------------

    batch = replay.sample(
        batch_size=32
    )

    print(
        "Batch states:",
        batch.states.shape,
    )

    print(
        "Batch actions:",
        batch.actions.shape,
    )

    print(
        "Real samples:",
        sum(
            source == ExperienceSource.REAL
            for source in batch.sources
        ),
    )

    print(
        "Imagined samples:",
        sum(
            source == ExperienceSource.IMAGINED
            for source in batch.sources
        ),
    )

    # --------------------------------------------------------
    # PYTORCH CONVERSION
    # --------------------------------------------------------

    torch_batch = batch.to_torch(
        "cpu"
    )

    print(
        "Torch state tensor:",
        torch_batch["states"].shape,
    )

    # --------------------------------------------------------
    # UPDATE PRIORITIES
    # --------------------------------------------------------

    td_errors = rng.normal(
        size=len(batch)
    )

    replay.update_priorities(
        batch.indices,
        td_errors,
    )

    # --------------------------------------------------------
    # N-STEP RETURNS
    # --------------------------------------------------------

    estimated_return = (
        calculate_n_step_return(
            rewards=[1.0, 0.5, 0.25],
            gamma=0.99,
            bootstrap_value=2.0,
            terminated=[
                False,
                False,
                False,
            ],
        )
    )

    print(
        "N-step return:",
        round(
            estimated_return,
            4,
        ),
    )

    # --------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------

    checkpoint_path = replay.save(
        "models/imagined_replay_demo.pkl"
    )

    restored = ImaginedReplayBuffer.load(
        checkpoint_path
    )

    print(
        "Restored statistics:",
        restored.statistics(),
    )

    print(
        "Imagined replay demonstration completed"
    )


if __name__ == "__main__":
    main()
