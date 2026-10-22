
"""
NeuroWorld AI
File: training/datasets/trajectory_dataset.py

Trajectory dataset for world-model and RL training.

Features:
- NPZ trajectory loading
- State/action/next-state sequences
- Reward and discounted-return targets
- Episode-aware windows
- Terminal/truncation boundaries
- Optional sequence padding
- Optional observations and timestamps
- Optional train-fitted normalization
- DataLoader integration

Required NPZ keys:
    states:       [N, state_dim]
    actions:      [N, action_dim]
    next_states:  [N, state_dim]

Optional keys:
    rewards:      [N]
    terminated:   [N]
    truncated:    [N]
    episode_ids:  [N]
    robot_ids:    [N]
    steps:        [N]
    observations: [N, observation_dim]
    timestamps:   [N]

Dependencies:
    numpy
    torch
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrajectoryDatasetConfig:
    path: str

    sequence_length: int = 16
    stride: int = 1
    pad_short_sequences: bool = False

    gamma: float = 0.99

    normalize_states: bool = False
    normalize_actions: bool = False

    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None

    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1"
            )

        if self.stride < 1:
            raise ValueError(
                "stride must be >= 1"
            )

        if not 0 <= self.gamma <= 1:
            raise ValueError(
                "gamma must be in [0, 1]"
            )

        if self.dtype not in (
            "float32",
            "float64",
        ):
            raise ValueError(
                "dtype must be float32 or float64"
            )


class TrajectoryDataset(Dataset):
    """
    Episode-aware trajectory dataset.

    Output shapes:
        states:      [T, state_dim]
        actions:     [T, action_dim]
        next_states: [T, state_dim]
        rewards:     [T]
        returns:     [T]
        terminated:  [T]
        truncated:   [T]
        episode_ids: [T]
        robot_ids:   [T]
        steps:       [T]
        mask:        [T]

    Each window stays within one episode
    and never crosses a terminal boundary.

    Returns are computed over the full
    available episode segment, not only
    the sampled window.

    Truncation is treated as a return
    boundary with zero bootstrap value.
    For bootstrapped RL targets, compute
    returns separately using a value model.
    """

    def __init__(
        self,
        config: TrajectoryDatasetConfig | str,
    ) -> None:
        if isinstance(config, str):
            config = TrajectoryDatasetConfig(
                path=config
            )

        self.config = config
        self.path = Path(config.path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Trajectory file not found: "
                f"{self.path}"
            )

        self.np_dtype = (
            np.float32
            if config.dtype == "float32"
            else np.float64
        )

        self.torch_dtype = (
            torch.float32
            if config.dtype == "float32"
            else torch.float64
        )

        self._load_data()
        self._validate_data()
        self._build_segments()
        self._compute_returns()
        self._normalize_data()
        self._build_windows()

    def _load_data(self) -> None:
        with np.load(
            self.path,
            allow_pickle=False,
        ) as data:
            required = (
                "states",
                "actions",
                "next_states",
            )

            missing = [
                key
                for key in required
                if key not in data.files
            ]

            if missing:
                raise ValueError(
                    f"Missing NPZ keys: {missing}"
                )

            self.states = np.asarray(
                data["states"],
                dtype=self.np_dtype,
            ).copy()

            self.actions = np.asarray(
                data["actions"],
                dtype=self.np_dtype,
            ).copy()

            self.next_states = np.asarray(
                data["next_states"],
                dtype=self.np_dtype,
            ).copy()

            n = len(self.states)

            def optional(
                name: str,
                default: np.ndarray,
                dtype: Any,
            ) -> np.ndarray:
                if name not in data.files:
                    return default

                return np.asarray(
                    data[name],
                    dtype=dtype,
                ).copy()

            self.rewards = optional(
                "rewards",
                np.zeros(n, dtype=self.np_dtype),
                self.np_dtype,
            )

            self.terminated = optional(
                "terminated",
                np.zeros(n, dtype=np.bool_),
                np.bool_,
            )

            self.truncated = optional(
                "truncated",
                np.zeros(n, dtype=np.bool_),
                np.bool_,
            )

            self.episode_ids = optional(
                "episode_ids",
                np.zeros(n, dtype=np.int64),
                np.int64,
            )

            self.robot_ids = optional(
                "robot_ids",
                np.zeros(n, dtype=np.int64),
                np.int64,
            )

            self.steps = optional(
                "steps",
                np.arange(n, dtype=np.int64),
                np.int64,
            )

            self.timestamps = (
                np.asarray(
                    data["timestamps"],
                    dtype=self.np_dtype,
                ).copy()
                if "timestamps" in data.files
                else None
            )

            self.observations = (
                np.asarray(
                    data["observations"],
                    dtype=self.np_dtype,
                ).copy()
                if "observations" in data.files
                else None
            )

    def _validate_data(self) -> None:
        n = len(self.states)

        if n == 0:
            raise ValueError(
                "Trajectory dataset is empty"
            )

        for name in (
            "states",
            "actions",
            "next_states",
        ):
            array = getattr(self, name)

            if (
                array.ndim != 2
                or len(array) != n
                or array.shape[1] < 1
            ):
                raise ValueError(
                    f"{name} must have shape [N, D]"
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

        if (
            self.states.shape[1]
            != self.next_states.shape[1]
        ):
            raise ValueError(
                "State dimensions do not match"
            )

        for name in (
            "rewards",
            "terminated",
            "truncated",
            "episode_ids",
            "robot_ids",
            "steps",
        ):
            array = getattr(self, name)

            if array.ndim != 1 or len(array) != n:
                raise ValueError(
                    f"{name} must have shape [{n}]"
                )

        if not np.isfinite(
            self.rewards
        ).all():
            raise ValueError(
                "Rewards contain NaN or Inf"
            )

        for name in (
            "timestamps",
            "observations",
        ):
            array = getattr(self, name)

            if array is None:
                continue

            expected_ndim = (
                1 if name == "timestamps" else 2
            )

            if (
                array.ndim != expected_ndim
                or len(array) != n
            ):
                raise ValueError(
                    f"Invalid {name} shape"
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def _build_segments(self) -> None:
        """
        Split the transition stream into
        contiguous episode segments.

        A boundary occurs after:
        - A terminal transition
        - A truncated transition
        - An episode ID change
        - A robot ID change
        - The final transition
        """
        n = len(self.states)

        self.segments: list[
            tuple[int, int]
        ] = []

        start = 0

        for index in range(n):
            last = index == n - 1

            changed_episode = (
                not last
                and self.episode_ids[index]
                != self.episode_ids[index + 1]
            )

            changed_robot = (
                not last
                and self.robot_ids[index]
                != self.robot_ids[index + 1]
            )

            boundary = (
                last
                or changed_episode
                or changed_robot
                or bool(self.terminated[index])
                or bool(self.truncated[index])
            )

            if boundary:
                self.segments.append(
                    (start, index + 1)
                )

                start = index + 1

    def _compute_returns(self) -> None:
        """
        Compute discounted Monte Carlo
        returns independently per segment.

        G[t] = r[t] + gamma * G[t+1]

        The final segment value is zero
        beyond the recorded boundary.
        """
        self.returns = np.zeros_like(
            self.rewards
        )

        gamma = self.config.gamma

        for start, end in self.segments:
            running_return = 0.0

            for index in range(
                end - 1,
                start - 1,
                -1,
            ):
                running_return = (
                    float(self.rewards[index])
                    + gamma * running_return
                )

                self.returns[index] = (
                    running_return
                )

    def _normalize_data(self) -> None:
        """
        Apply statistics fitted on training
        data only.

        No automatic fitting occurs here.
        """
        self.state_mean = None
        self.state_std = None

        self.action_mean = None
        self.action_std = None

        if self.config.normalize_states:
            if (
                self.config.state_mean is None
                or self.config.state_std is None
            ):
                raise ValueError(
                    "State normalization requires "
                    "state_mean and state_std"
                )

            self.state_mean = np.asarray(
                self.config.state_mean,
                dtype=self.np_dtype,
            )

            self.state_std = np.asarray(
                self.config.state_std,
                dtype=self.np_dtype,
            )

            if (
                self.state_mean.shape
                != (self.states.shape[1],)
                or self.state_std.shape
                != (self.states.shape[1],)
            ):
                raise ValueError(
                    "Invalid state statistics shape"
                )

            if (
                not np.isfinite(
                    self.state_mean
                ).all()
                or not np.isfinite(
                    self.state_std
                ).all()
                or np.any(self.state_std <= 0)
            ):
                raise ValueError(
                    "Invalid state statistics"
                )

            self.state_std = np.maximum(
                self.state_std,
                1e-6,
            )

            self.states = (
                self.states - self.state_mean
            ) / self.state_std

            self.next_states = (
                self.next_states - self.state_mean
            ) / self.state_std

        if self.config.normalize_actions:
            if (
                self.config.action_mean is None
                or self.config.action_std is None
            ):
                raise ValueError(
                    "Action normalization requires "
                    "action_mean and action_std"
                )

            self.action_mean = np.asarray(
                self.config.action_mean,
                dtype=self.np_dtype,
            )

            self.action_std = np.asarray(
                self.config.action_std,
                dtype=self.np_dtype,
            )

            if (
                self.action_mean.shape
                != (self.actions.shape[1],)
                or self.action_std.shape
                != (self.actions.shape[1],)
            ):
                raise ValueError(
                    "Invalid action statistics shape"
                )

            if (
                not np.isfinite(
                    self.action_mean
                ).all()
                or not np.isfinite(
                    self.action_std
                ).all()
                or np.any(self.action_std <= 0)
            ):
                raise ValueError(
                    "Invalid action statistics"
                )

            self.action_std = np.maximum(
                self.action_std,
                1e-6,
            )

            self.actions = (
                self.actions - self.action_mean
            ) / self.action_std

    def _build_windows(self) -> None:
        self.windows: list[
            tuple[int, int]
        ] = []

        length = self.config.sequence_length
        stride = self.config.stride

        for segment_start, segment_end in (
            self.segments
        ):
            size = segment_end - segment_start

            if size >= length:
                for start in range(
                    segment_start,
                    segment_end - length + 1,
                    stride,
                ):
                    self.windows.append(
                        (start, start + length)
                    )

            elif self.config.pad_short_sequences:
                self.windows.append(
                    (segment_start, segment_end)
                )

        if not self.windows:
            raise ValueError(
                "No valid trajectory windows. "
                "Reduce sequence_length or "
                "enable pad_short_sequences."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _tensor(
        self,
        value: np.ndarray,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return torch.as_tensor(
            np.array(value, copy=True),
            dtype=dtype or self.torch_dtype,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        start, end = self.windows[index]

        length = end - start

        sample = {
            "states": self._tensor(
                self.states[start:end]
            ),
            "actions": self._tensor(
                self.actions[start:end]
            ),
            "next_states": self._tensor(
                self.next_states[start:end]
            ),
            "rewards": self._tensor(
                self.rewards[start:end]
            ),
            "returns": self._tensor(
                self.returns[start:end]
            ),
            "terminated": self._tensor(
                self.terminated[start:end],
                torch.bool,
            ),
            "truncated": self._tensor(
                self.truncated[start:end],
                torch.bool,
            ),
            "episode_ids": self._tensor(
                self.episode_ids[start:end],
                torch.long,
            ),
            "robot_ids": self._tensor(
                self.robot_ids[start:end],
                torch.long,
            ),
            "steps": self._tensor(
                self.steps[start:end],
                torch.long,
            ),
            "mask": torch.ones(
                length,
                dtype=torch.bool,
            ),
        }

        if self.timestamps is not None:
            sample["timestamps"] = self._tensor(
                self.timestamps[start:end]
            )

        if self.observations is not None:
            sample["observations"] = self._tensor(
                self.observations[start:end]
            )

        target_length = (
            self.config.sequence_length
        )

        if length < target_length:
            padding = target_length - length

            for key, value in sample.items():
                zeros = torch.zeros(
                    (
                        padding,
                        *value.shape[1:],
                    ),
                    dtype=value.dtype,
                )

                sample[key] = torch.cat(
                    (value, zeros),
                    dim=0,
                )

        return sample

    @property
    def state_dim(self) -> int:
        return int(
            self.states.shape[1]
        )

    @property
    def action_dim(self) -> int:
        return int(
            self.actions.shape[1]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "transitions": len(self.states),
            "sequences": len(self),
            "segments": len(self.segments),
            "episodes": int(
                np.unique(
                    self.episode_ids
                ).size
            ),
            "robots": int(
                np.unique(
                    self.robot_ids
                ).size
            ),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "sequence_length": (
                self.config.sequence_length
            ),
            "gamma": self.config.gamma,
            "has_observations": (
                self.observations is not None
            ),
        }


def create_trajectory_dataloader(
    config: TrajectoryDatasetConfig,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = TrajectoryDataset(config)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    num_episodes = 10
    steps_per_episode = 30

    n = num_episodes * steps_per_episode

    state_dim = 128
    action_dim = 32

    states = rng.normal(
        size=(n, state_dim)
    ).astype(np.float32)

    actions = rng.normal(
        size=(n, action_dim)
    ).astype(np.float32)

    next_states = (
        states
        + rng.normal(
            scale=0.01,
            size=(n, state_dim),
        )
    ).astype(np.float32)

    rewards = rng.normal(
        size=n
    ).astype(np.float32)

    episode_ids = np.repeat(
        np.arange(num_episodes),
        steps_per_episode,
    )

    robot_ids = np.repeat(
        np.arange(2),
        n // 2,
    )

    steps = np.tile(
        np.arange(steps_per_episode),
        num_episodes,
    )

    terminated = np.zeros(
        n,
        dtype=np.bool_,
    )

    terminated[
        steps_per_episode - 1::steps_per_episode
    ] = True

    path = Path(
        "data/processed/trajectories_demo.npz"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        episode_ids=episode_ids,
        robot_ids=robot_ids,
        steps=steps,
        terminated=terminated,
    )

    config = TrajectoryDatasetConfig(
        path=str(path),
        sequence_length=16,
        stride=2,
        gamma=0.99,
    )

    loader = create_trajectory_dataloader(
        config,
        batch_size=8,
    )

    print(
        "Dataset:",
        loader.dataset.summary(),
    )

    batch = next(iter(loader))

    for key, value in batch.items():
        print(
            key,
            tuple(value.shape),
        )
