
"""
NeuroWorld AI
File: training/datasets/simulation_dataset.py

Simulation dataset for world-model training.

Features:
- NPZ simulation dataset loading
- State/action/next-state transitions
- Rewards and terminal flags
- Episode-aware sequence sampling
- Optional robot observations
- Optional physics information
- Optional timestamps
- Sequence padding and masks
- Normalization with supplied statistics
- PyTorch DataLoader compatibility
- Synthetic demonstration dataset

Required NPZ arrays:
    states:       [N, state_dim]
    actions:      [N, action_dim]
    next_states:  [N, state_dim]

Optional NPZ arrays:
    rewards:      [N]
    terminated:   [N]
    truncated:    [N]
    episode_ids:  [N]
    timestamps:   [N]
    observations: [N, observation_dim]
    physics:      [N, physics_dim]

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
from torch.utils.data import Dataset, DataLoader


@dataclass
class SimulationDatasetConfig:
    path: str

    sequence_length: int = 16
    stride: int = 1

    pad_short_sequences: bool = False

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

        if self.dtype not in (
            "float32",
            "float64",
        ):
            raise ValueError(
                "Unsupported dtype"
            )


class SimulationDataset(Dataset):
    """
    Episode-aware simulation dataset.

    Output:
        states
        actions
        next_states
        rewards
        terminated
        truncated
        episode_ids
        timestamps
        mask

    Optional:
        observations
        physics

    Shapes:
        states: [T, state_dim]
        actions: [T, action_dim]
        mask: [T]

    Windows never cross episode boundaries
    or interior terminal transitions.
    """

    def __init__(
        self,
        config: SimulationDatasetConfig | str,
    ) -> None:
        if isinstance(config, str):
            config = SimulationDatasetConfig(
                path=config
            )

        self.config = config
        self.path = Path(config.path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Simulation dataset not found: "
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

        self._load()
        self._validate()
        self._normalize()
        self._build_windows()

    def _load(self) -> None:
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
                    f"Missing keys: {missing}"
                )

            self.states = np.asarray(
                data["states"],
                dtype=self.np_dtype,
            )

            self.actions = np.asarray(
                data["actions"],
                dtype=self.np_dtype,
            )

            self.next_states = np.asarray(
                data["next_states"],
                dtype=self.np_dtype,
            )

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
                )

            self.rewards = optional(
                "rewards",
                np.zeros(
                    n,
                    dtype=self.np_dtype,
                ),
                self.np_dtype,
            )

            self.terminated = optional(
                "terminated",
                np.zeros(
                    n,
                    dtype=np.bool_,
                ),
                np.bool_,
            )

            self.truncated = optional(
                "truncated",
                np.zeros(
                    n,
                    dtype=np.bool_,
                ),
                np.bool_,
            )

            self.episode_ids = optional(
                "episode_ids",
                np.zeros(
                    n,
                    dtype=np.int64,
                ),
                np.int64,
            )

            self.timestamps = optional(
                "timestamps",
                np.arange(
                    n,
                    dtype=self.np_dtype,
                ),
                self.np_dtype,
            )

            self.observations = (
                np.asarray(
                    data["observations"],
                    dtype=self.np_dtype,
                )
                if "observations" in data.files
                else None
            )

            self.physics = (
                np.asarray(
                    data["physics"],
                    dtype=self.np_dtype,
                )
                if "physics" in data.files
                else None
            )

    def _validate(self) -> None:
        n = len(self.states)

        if n == 0:
            raise ValueError(
                "Simulation dataset is empty"
            )

        for name in (
            "states",
            "actions",
            "next_states",
        ):
            array = getattr(
                self,
                name,
            )

            if array.ndim != 2:
                raise ValueError(
                    f"{name} must have shape [N, D]"
                )

            if len(array) != n:
                raise ValueError(
                    f"{name} length mismatch"
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
                "State dimension mismatch"
            )

        for name in (
            "rewards",
            "terminated",
            "truncated",
            "episode_ids",
            "timestamps",
        ):
            array = getattr(
                self,
                name,
            )

            if (
                array.ndim != 1
                or len(array) != n
            ):
                raise ValueError(
                    f"{name} must have shape [{n}]"
                )

        for name in (
            "observations",
            "physics",
        ):
            array = getattr(
                self,
                name,
            )

            if array is None:
                continue

            if (
                array.ndim != 2
                or len(array) != n
            ):
                raise ValueError(
                    f"{name} must have shape [N, D]"
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

        if not np.isfinite(
            self.rewards
        ).all():
            raise ValueError(
                "Rewards contain NaN or Inf"
            )

        if not np.isfinite(
            self.timestamps
        ).all():
            raise ValueError(
                "Timestamps contain NaN or Inf"
            )

    def _normalize(self) -> None:
        """
        Normalize using externally supplied
        training-set statistics.

        Prevents fitting normalization on
        validation/test data.
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

            self.state_std = np.maximum(
                np.asarray(
                    self.config.state_std,
                    dtype=self.np_dtype,
                ),
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

            self.action_std = np.maximum(
                np.asarray(
                    self.config.action_std,
                    dtype=self.np_dtype,
                ),
                1e-6,
            )

            self.actions = (
                self.actions - self.action_mean
            ) / self.action_std

    def _build_windows(self) -> None:
        """
        Build contiguous windows per episode.

        Each terminal/truncated transition
        ends a segment, even if episode_ids
        are incorrectly reused.
        """
        n = len(self.states)
        length = self.config.sequence_length

        self.windows: list[
            tuple[int, int]
        ] = []

        segment_start = 0

        for index in range(n):
            last = index == n - 1

            episode_change = (
                not last
                and self.episode_ids[index]
                != self.episode_ids[index + 1]
            )

            boundary = (
                last
                or episode_change
                or bool(self.terminated[index])
                or bool(self.truncated[index])
            )

            if not boundary:
                continue

            segment_end = index + 1

            segment_size = (
                segment_end - segment_start
            )

            if segment_size >= length:
                for start in range(
                    segment_start,
                    segment_end - length + 1,
                    self.config.stride,
                ):
                    self.windows.append(
                        (start, start + length)
                    )

            elif self.config.pad_short_sequences:
                self.windows.append(
                    (segment_start, segment_end)
                )

            segment_start = segment_end

        if not self.windows:
            raise ValueError(
                "No valid simulation sequences. "
                "Reduce sequence_length or enable "
                "pad_short_sequences."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _tensor(
        self,
        array: np.ndarray,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return torch.as_tensor(
            np.array(
                array,
                copy=True,
            ),
            dtype=dtype or self.torch_dtype,
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        start, end = self.windows[index]

        valid_length = end - start

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
            "terminated": self._tensor(
                self.terminated[start:end],
                dtype=torch.bool,
            ),
            "truncated": self._tensor(
                self.truncated[start:end],
                dtype=torch.bool,
            ),
            "episode_ids": self._tensor(
                self.episode_ids[start:end],
                dtype=torch.long,
            ),
            "timestamps": self._tensor(
                self.timestamps[start:end]
            ),
            "mask": torch.ones(
                valid_length,
                dtype=torch.bool,
            ),
        }

        if self.observations is not None:
            sample["observations"] = (
                self._tensor(
                    self.observations[start:end]
                )
            )

        if self.physics is not None:
            sample["physics"] = (
                self._tensor(
                    self.physics[start:end]
                )
            )

        target_length = (
            self.config.sequence_length
        )

        if valid_length < target_length:
            padding = (
                target_length - valid_length
            )

            for key, value in sample.items():
                pad_shape = (
                    padding,
                    *value.shape[1:],
                )

                zeros = torch.zeros(
                    pad_shape,
                    dtype=value.dtype,
                    device=value.device,
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
            "episodes": int(
                np.unique(
                    self.episode_ids
                ).size
            ),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "sequence_length": (
                self.config.sequence_length
            ),
            "has_observations": (
                self.observations is not None
            ),
            "has_physics": (
                self.physics is not None
            ),
        }


def create_simulation_dataloader(
    config: SimulationDatasetConfig,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SimulationDataset(
        config
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    num_episodes = 10
    steps_per_episode = 30

    n = (
        num_episodes * steps_per_episode
    )

    state_dim = 128
    action_dim = 32

    states = rng.normal(
        size=(n, state_dim)
    ).astype(np.float32)

    actions = rng.normal(
        size=(n, action_dim)
    ).astype(np.float32)

    # Synthetic demonstration transitions.
    # Not a realistic physics simulator.
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

    terminated = np.zeros(
        n,
        dtype=np.bool_,
    )

    terminated[
        steps_per_episode - 1::steps_per_episode
    ] = True

    timestamps = np.tile(
        np.arange(steps_per_episode),
        num_episodes,
    ).astype(np.float32)

    physics = rng.normal(
        size=(n, 12)
    ).astype(np.float32)

    path = Path(
        "data/processed/simulation_demo.npz"
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
        terminated=terminated,
        timestamps=timestamps,
        physics=physics,
    )

    config = SimulationDatasetConfig(
        path=str(path),
        sequence_length=16,
        stride=2,
    )

    loader = create_simulation_dataloader(
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
