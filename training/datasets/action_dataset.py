
"""
NeuroWorld AI
File: training/datasets/action_dataset.py

PyTorch dataset for action-conditioned world models.

Supported:
- NPZ datasets
- State-action transitions
- Next-state and reward targets
- Episode-aware sequences
- Optional normalization
- Optional action clipping
- Dataset validation
- Train/validation splitting
- PyTorch DataLoaders

Expected NPZ keys:
    states:       [N, state_dim]
    actions:      [N, action_dim]
    next_states:  [N, state_dim]

Optional:
    rewards:      [N]
    terminated:   [N]
    truncated:    [N]
    episode_ids:  [N]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)


@dataclass
class ActionDatasetConfig:
    path: str

    sequence_length: int = 1
    stride: int = 1

    normalize_states: bool = False
    normalize_actions: bool = False

    clip_actions: bool = False
    action_min: float = -1.0
    action_max: float = 1.0

    dtype: Literal["float32", "float64"] = "float32"

    allow_pickle: bool = False

    def __post_init__(self) -> None:
        if self.sequence_length < 1:
            raise ValueError(
                "sequence_length must be >= 1"
            )

        if self.stride < 1:
            raise ValueError(
                "stride must be >= 1"
            )

        if self.action_min > self.action_max:
            raise ValueError(
                "action_min must not exceed action_max"
            )


class ActionDataset(Dataset):
    """
    Dataset of state-action transitions.

    For sequence_length=1:
        states:      [state_dim]
        actions:     [action_dim]
        next_states: [state_dim]

    For sequence_length>1:
        states:      [T, state_dim]
        actions:     [T, action_dim]
        next_states: [T, state_dim]

    Other returned fields:
        rewards
        terminated
        truncated
        episode_ids
        mask

    Sequence windows never cross episode
    boundaries or terminal transitions.
    """

    def __init__(
        self,
        config: ActionDatasetConfig | str,
    ) -> None:
        if isinstance(config, str):
            config = ActionDatasetConfig(
                path=config
            )

        self.config = config

        self.path = Path(config.path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {self.path}"
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
        self._prepare_data()
        self._build_windows()

    def _load_data(self) -> None:
        with np.load(
            self.path,
            allow_pickle=self.config.allow_pickle,
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
                    f"Missing dataset keys: {missing}"
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

            self.rewards = (
                np.asarray(
                    data["rewards"],
                    dtype=self.np_dtype,
                ).reshape(-1)
                if "rewards" in data.files
                else np.zeros(
                    n,
                    dtype=self.np_dtype,
                )
            )

            self.terminated = (
                np.asarray(
                    data["terminated"],
                    dtype=np.bool_,
                ).reshape(-1)
                if "terminated" in data.files
                else np.zeros(
                    n,
                    dtype=np.bool_,
                )
            )

            self.truncated = (
                np.asarray(
                    data["truncated"],
                    dtype=np.bool_,
                ).reshape(-1)
                if "truncated" in data.files
                else np.zeros(
                    n,
                    dtype=np.bool_,
                )
            )

            self.episode_ids = (
                np.asarray(
                    data["episode_ids"],
                    dtype=np.int64,
                ).reshape(-1)
                if "episode_ids" in data.files
                else np.zeros(
                    n,
                    dtype=np.int64,
                )
            )

    def _validate_data(self) -> None:
        n = len(self.states)

        if n == 0:
            raise ValueError(
                "Dataset cannot be empty"
            )

        if (
            self.states.ndim != 2
            or self.actions.ndim != 2
            or self.next_states.ndim != 2
        ):
            raise ValueError(
                "States, actions and next_states "
                "must be 2D arrays"
            )

        arrays = {
            "actions": self.actions,
            "next_states": self.next_states,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_ids": self.episode_ids,
        }

        for name, array in arrays.items():
            if len(array) != n:
                raise ValueError(
                    f"{name} length mismatch"
                )

        if (
            self.states.shape[1]
            != self.next_states.shape[1]
        ):
            raise ValueError(
                "State dimensions do not match"
            )

        for name, array in (
            ("states", self.states),
            ("actions", self.actions),
            ("next_states", self.next_states),
            ("rewards", self.rewards),
        ):
            if not np.isfinite(array).all():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    @staticmethod
    def _normalize(
        array: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        mean = array.mean(
            axis=0,
            keepdims=True,
        )

        std = array.std(
            axis=0,
            keepdims=True,
        )

        std = np.maximum(
            std,
            1e-6,
        )

        normalized = (
            array - mean
        ) / std

        return normalized, mean, std

    def _prepare_data(self) -> None:
        self.state_mean = None
        self.state_std = None

        self.action_mean = None
        self.action_std = None

        if self.config.clip_actions:
            self.actions = np.clip(
                self.actions,
                self.config.action_min,
                self.config.action_max,
            )

        if self.config.normalize_states:
            (
                self.states,
                self.state_mean,
                self.state_std,
            ) = self._normalize(
                self.states
            )

            self.next_states = (
                self.next_states
                - self.state_mean
            ) / self.state_std

        if self.config.normalize_actions:
            (
                self.actions,
                self.action_mean,
                self.action_std,
            ) = self._normalize(
                self.actions
            )

    def _build_windows(self) -> None:
        self.windows: list[
            tuple[int, int]
        ] = []

        n = len(self.states)
        length = self.config.sequence_length

        for start in range(
            0,
            n - length + 1,
            self.config.stride,
        ):
            end = start + length

            episode_slice = (
                self.episode_ids[start:end]
            )

            if not np.all(
                episode_slice == episode_slice[0]
            ):
                continue

            # A terminal transition may be the
            # last element of a window, but not
            # an interior element.
            if length > 1:
                if np.any(
                    self.terminated[start:end - 1]
                ):
                    continue

                if np.any(
                    self.truncated[start:end - 1]
                ):
                    continue

            self.windows.append(
                (start, end)
            )

        if not self.windows:
            raise ValueError(
                "No valid action sequences found"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        start, end = self.windows[index]

        def tensor(
            value: np.ndarray,
        ) -> Tensor:
            return torch.as_tensor(
                np.array(
                    value,
                    copy=True,
                ),
                dtype=self.torch_dtype,
            )

        sample = {
            "states": tensor(
                self.states[start:end]
            ),
            "actions": tensor(
                self.actions[start:end]
            ),
            "next_states": tensor(
                self.next_states[start:end]
            ),
            "rewards": tensor(
                self.rewards[start:end]
            ),
            "terminated": torch.as_tensor(
                self.terminated[start:end].copy(),
                dtype=torch.bool,
            ),
            "truncated": torch.as_tensor(
                self.truncated[start:end].copy(),
                dtype=torch.bool,
            ),
            "episode_ids": torch.as_tensor(
                self.episode_ids[start:end].copy(),
                dtype=torch.long,
            ),
            "mask": torch.ones(
                end - start,
                dtype=torch.bool,
            ),
        }

        if self.config.sequence_length == 1:
            return {
                key: value[0]
                for key, value in sample.items()
            }

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
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "sequence_length": (
                self.config.sequence_length
            ),
            "episodes": int(
                np.unique(
                    self.episode_ids
                ).size
            ),
            "normalized_states": (
                self.config.normalize_states
            ),
            "normalized_actions": (
                self.config.normalize_actions
            ),
        }


def create_action_dataloaders(
    config: ActionDatasetConfig,
    *,
    batch_size: int = 32,
    validation_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[
    DataLoader,
    DataLoader,
]:
    """
    Split by episode to prevent overlapping
    trajectory windows leaking between sets.

    Requires at least two distinct episode IDs.
    """
    if not 0 < validation_split < 1:
        raise ValueError(
            "validation_split must be between 0 and 1"
        )

    dataset = ActionDataset(config)

    episodes = np.unique(
        dataset.episode_ids
    )

    if len(episodes) < 2:
        raise ValueError(
            "Episode-based splitting requires "
            "at least two episodes"
        )

    rng = np.random.default_rng(seed)

    episodes = rng.permutation(
        episodes
    )

    n_val = max(
        1,
        min(
            len(episodes) - 1,
            round(
                len(episodes)
                * validation_split
            ),
        ),
    )

    val_episodes = set(
        episodes[:n_val].tolist()
    )

    train_indices = []
    val_indices = []

    for index, (start, _) in enumerate(
        dataset.windows
    ):
        episode = int(
            dataset.episode_ids[start]
        )

        if episode in val_episodes:
            val_indices.append(index)
        else:
            train_indices.append(index)

    if not train_indices or not val_indices:
        raise ValueError(
            "Train or validation split is empty"
        )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Generate a small demonstration dataset.
    rng = np.random.default_rng(42)

    n = 200
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
        + 0.01 * rng.normal(
            size=(n, state_dim)
        )
    ).astype(np.float32)

    rewards = rng.normal(
        size=n
    ).astype(np.float32)

    episode_ids = np.repeat(
        np.arange(10),
        20,
    )

    terminated = np.zeros(
        n,
        dtype=np.bool_,
    )

    terminated[19::20] = True

    path = Path(
        "data/processed/actions_demo.npz"
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
    )

    config = ActionDatasetConfig(
        path=str(path),
        sequence_length=8,
        stride=2,
    )

    train_loader, val_loader = (
        create_action_dataloaders(
            config,
            batch_size=8,
        )
    )

    print(
        "Dataset:",
        train_loader.dataset.dataset.summary(),
    )

    batch = next(
        iter(train_loader)
    )

    print(
        "States:",
        batch["states"].shape,
    )

    print(
        "Actions:",
        batch["actions"].shape,
    )

    print(
        "Next states:",
        batch["next_states"].shape,
    )
