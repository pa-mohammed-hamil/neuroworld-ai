
"""
NeuroWorld AI
File: training/datasets/dataset_splitter.py

Leakage-aware dataset splitting for:
- World model training
- Dynamics prediction
- Action-conditioned learning
- Reward model training
- Video prediction
- Robotics and simulation trajectories

Features:
- Random splitting
- Episode/group-based splitting
- Chronological splitting
- Reproducible seeds
- PyTorch Subset support
- Train/validation/test partitions
- Split integrity validation
- JSON manifest export/import

Important:
For overlapping temporal windows, use episode
or recording-level groups instead of random
sample-level splitting.
"""

from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from torch.utils.data import Dataset, Subset


SplitStrategy = Literal[
    "random",
    "episode",
    "group",
    "chronological",
]


@dataclass
class DatasetSplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    strategy: SplitStrategy = "episode"

    seed: int = 42
    shuffle_groups: bool = True

    min_train_samples: int = 1
    min_val_samples: int = 0
    min_test_samples: int = 0

    def __post_init__(self) -> None:
        ratios = (
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )

        if any(
            not np.isfinite(value) or value < 0
            for value in ratios
        ):
            raise ValueError(
                "Split ratios must be finite "
                "and nonnegative"
            )

        if not np.isclose(
            sum(ratios),
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Split ratios must sum to 1.0"
            )

        if self.train_ratio <= 0:
            raise ValueError(
                "train_ratio must be positive"
            )

        if self.strategy not in (
            "random",
            "episode",
            "group",
            "chronological",
        ):
            raise ValueError(
                "Unsupported split strategy"
            )

        for name in (
            "min_train_samples",
            "min_val_samples",
            "min_test_samples",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


@dataclass
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]

    def as_dict(
        self,
    ) -> dict[str, list[int]]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }

    def sizes(
        self,
    ) -> dict[str, int]:
        return {
            name: len(indices)
            for name, indices
            in self.as_dict().items()
        }


class DatasetSplitter:
    """
    Reproducible dataset splitting.

    Episode/group splits keep all windows
    belonging to one group in the same
    partition.

    Chronological splits assume the supplied
    sample order is already chronological.
    """

    def __init__(
        self,
        config: DatasetSplitConfig | None = None,
    ) -> None:
        self.config = (
            config or DatasetSplitConfig()
        )

    @staticmethod
    def _allocate_counts(
        total: int,
        ratios: Sequence[float],
    ) -> list[int]:
        """
        Largest-remainder allocation.

        Produces counts whose sum equals total.
        """
        raw = np.asarray(
            ratios,
            dtype=np.float64,
        ) * total

        counts = np.floor(
            raw
        ).astype(int)

        remaining = (
            total - int(counts.sum())
        )

        remainders = raw - counts

        order = np.argsort(
            -remainders,
            kind="stable",
        )

        for index in order[:remaining]:
            counts[index] += 1

        return counts.tolist()

    def _random_split(
        self,
        total: int,
    ) -> SplitIndices:
        rng = np.random.default_rng(
            self.config.seed
        )

        indices = rng.permutation(
            total
        ).tolist()

        n_train, n_val, _ = (
            self._allocate_counts(
                total,
                (
                    self.config.train_ratio,
                    self.config.val_ratio,
                    self.config.test_ratio,
                ),
            )
        )

        return SplitIndices(
            train=indices[:n_train],
            val=indices[
                n_train:n_train + n_val
            ],
            test=indices[
                n_train + n_val:
            ],
        )

    def _chronological_split(
        self,
        total: int,
    ) -> SplitIndices:
        n_train, n_val, _ = (
            self._allocate_counts(
                total,
                (
                    self.config.train_ratio,
                    self.config.val_ratio,
                    self.config.test_ratio,
                ),
            )
        )

        return SplitIndices(
            train=list(
                range(0, n_train)
            ),
            val=list(
                range(
                    n_train,
                    n_train + n_val,
                )
            ),
            test=list(
                range(
                    n_train + n_val,
                    total,
                )
            ),
        )

    def _group_split(
        self,
        groups: Sequence[Any],
    ) -> SplitIndices:
        """
        Allocate complete groups.

        Ratios are approximated by sample count
        rather than simply counting groups.
        """
        group_map: dict[
            Any,
            list[int],
        ] = {}

        for index, group in enumerate(groups):
            # Convert NumPy scalars to Python
            # scalars for stable dictionary keys.
            if isinstance(group, np.generic):
                group = group.item()

            try:
                group_map.setdefault(
                    group,
                    [],
                ).append(index)
            except TypeError as exc:
                raise ValueError(
                    "Group IDs must be hashable"
                ) from exc

        group_keys = list(
            group_map.keys()
        )

        if self.config.shuffle_groups:
            rng = np.random.default_rng(
                self.config.seed
            )

            rng.shuffle(group_keys)

        ratios = (
            self.config.train_ratio,
            self.config.val_ratio,
            self.config.test_ratio,
        )

        targets = np.asarray(
            ratios
        ) * len(groups)

        partitions: dict[
            str,
            list[int],
        ] = {
            "train": [],
            "val": [],
            "test": [],
        }

        names = (
            "train",
            "val",
            "test",
        )

        # Greedily allocate groups based on
        # remaining target sample capacity.
        for group in group_keys:
            indices = group_map[group]

            deficits = [
                targets[i]
                - len(partitions[name])
                if ratios[i] > 0
                else float("-inf")
                for i, name in enumerate(names)
            ]

            selected = names[
                int(np.argmax(deficits))
            ]

            partitions[selected].extend(
                indices
            )

        return SplitIndices(
            train=partitions["train"],
            val=partitions["val"],
            test=partitions["test"],
        )

    @staticmethod
    def infer_episode_groups(
        dataset: Dataset,
    ) -> list[Any]:
        """
        Infer episode ID for each dataset item.

        Compatible with ActionDataset:
            dataset.windows
            dataset.episode_ids

        Also supports a dataset exposing
        sample-level episode_ids.
        """
        if not hasattr(
            dataset,
            "episode_ids",
        ):
            raise ValueError(
                "Dataset has no episode_ids. "
                "Supply groups explicitly."
            )

        episode_ids = getattr(
            dataset,
            "episode_ids",
        )

        if hasattr(
            dataset,
            "windows",
        ):
            windows = getattr(
                dataset,
                "windows",
            )

            return [
                episode_ids[int(start)]
                for start, _ in windows
            ]

        if len(episode_ids) == len(dataset):
            return list(episode_ids)

        raise ValueError(
            "Cannot map episode_ids to "
            "dataset samples"
        )

    def split_indices(
        self,
        total: int,
        *,
        groups: Sequence[Any] | None = None,
    ) -> SplitIndices:
        if total < 1:
            raise ValueError(
                "Dataset must contain samples"
            )

        strategy = self.config.strategy

        if strategy == "random":
            result = self._random_split(
                total
            )

        elif strategy == "chronological":
            result = (
                self._chronological_split(
                    total
                )
            )

        else:
            if groups is None:
                raise ValueError(
                    f"{strategy} splitting "
                    "requires groups"
                )

            if len(groups) != total:
                raise ValueError(
                    "groups length must match "
                    "dataset length"
                )

            result = self._group_split(
                groups
            )

        self.validate_split(
            result,
            total,
        )

        return result

    def split_dataset(
        self,
        dataset: Dataset,
        *,
        groups: Sequence[Any] | None = None,
    ) -> dict[str, Subset]:
        """
        Return PyTorch Subsets.

        If strategy='episode', episode groups
        are inferred when not provided.
        """
        if (
            self.config.strategy == "episode"
            and groups is None
        ):
            groups = (
                self.infer_episode_groups(
                    dataset
                )
            )

        indices = self.split_indices(
            len(dataset),
            groups=groups,
        )

        return {
            "train": Subset(
                dataset,
                indices.train,
            ),
            "val": Subset(
                dataset,
                indices.val,
            ),
            "test": Subset(
                dataset,
                indices.test,
            ),
        }

    def validate_split(
        self,
        split: SplitIndices,
        total: int,
    ) -> None:
        partitions = split.as_dict()

        seen: set[int] = set()

        for name, indices in partitions.items():
            for index in indices:
                if not isinstance(
                    index,
                    (int, np.integer),
                ):
                    raise ValueError(
                        f"Invalid index in {name}"
                    )

                index = int(index)

                if not 0 <= index < total:
                    raise ValueError(
                        f"Out-of-range index: {index}"
                    )

                if index in seen:
                    raise ValueError(
                        f"Duplicate index: {index}"
                    )

                seen.add(index)

        if len(seen) != total:
            raise ValueError(
                "Split does not cover all samples"
            )

        minimums = {
            "train": (
                self.config.min_train_samples
            ),
            "val": (
                self.config.min_val_samples
            ),
            "test": (
                self.config.min_test_samples
            ),
        }

        for name, minimum in minimums.items():
            if len(partitions[name]) < minimum:
                raise ValueError(
                    f"{name} split has fewer "
                    f"than {minimum} samples"
                )

    @staticmethod
    def validate_group_isolation(
        split: SplitIndices,
        groups: Sequence[Any],
    ) -> bool:
        """
        Ensure no group appears in more
        than one partition.
        """
        group_sets = {
            name: {
                groups[index]
                for index in indices
            }
            for name, indices
            in split.as_dict().items()
        }

        names = list(group_sets)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if (
                    group_sets[names[i]]
                    & group_sets[names[j]]
                ):
                    raise ValueError(
                        "Group leakage detected "
                        f"between {names[i]} "
                        f"and {names[j]}"
                    )

        return True

    def save_manifest(
        self,
        split: SplitIndices,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save split indices and configuration.
        """
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": 1,
            "config": asdict(
                self.config
            ),
            "splits": split.as_dict(),
            "metadata": metadata or {},
        }

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        return destination

    @staticmethod
    def load_manifest(
        path: str | Path,
        *,
        dataset_size: int | None = None,
    ) -> SplitIndices:
        source = Path(path)

        with source.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        data = payload["splits"]

        split = SplitIndices(
            train=[
                int(i)
                for i in data["train"]
            ],
            val=[
                int(i)
                for i in data["val"]
            ],
            test=[
                int(i)
                for i in data["test"]
            ],
        )

        if dataset_size is not None:
            config = DatasetSplitConfig(
                **payload.get(
                    "config",
                    {},
                )
            )

            DatasetSplitter(
                config
            ).validate_split(
                split,
                dataset_size,
            )

        return split

    @staticmethod
    def summary(
        split: SplitIndices,
    ) -> dict[str, Any]:
        sizes = split.sizes()

        total = sum(
            sizes.values()
        )

        return {
            "total_samples": total,
            "train_samples": sizes["train"],
            "val_samples": sizes["val"],
            "test_samples": sizes["test"],
            "train_percentage": round(
                100 * sizes["train"] / total,
                2,
            ) if total else 0.0,
            "val_percentage": round(
                100 * sizes["val"] / total,
                2,
            ) if total else 0.0,
            "test_percentage": round(
                100 * sizes["test"] / total,
                2,
            ) if total else 0.0,
        }


def split_dataset(
    dataset: Dataset,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    strategy: SplitStrategy = "episode",
    seed: int = 42,
    groups: Sequence[Any] | None = None,
) -> dict[str, Subset]:
    """
    Convenience function.
    """
    config = DatasetSplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        strategy=strategy,
        seed=seed,
    )

    splitter = DatasetSplitter(
        config
    )

    return splitter.split_dataset(
        dataset,
        groups=groups,
    )


if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    import torch

    # Example: 10 episodes, 20 samples each.
    states = torch.randn(
        200,
        128,
    )

    actions = torch.randn(
        200,
        32,
    )

    dataset = TensorDataset(
        states,
        actions,
    )

    episode_groups = np.repeat(
        np.arange(10),
        20,
    ).tolist()

    config = DatasetSplitConfig(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        strategy="episode",
        seed=42,
    )

    splitter = DatasetSplitter(
        config
    )

    indices = splitter.split_indices(
        len(dataset),
        groups=episode_groups,
    )

    splitter.validate_group_isolation(
        indices,
        episode_groups,
    )

    print(
        "Split summary:",
        splitter.summary(indices),
    )

    splitter.save_manifest(
        indices,
        "data/splits/action_split.json",
    )

    subsets = {
        name: Subset(
            dataset,
            partition,
        )
        for name, partition
        in indices.as_dict().items()
    }

    print(
        "Training samples:",
        len(subsets["train"]),
    )
