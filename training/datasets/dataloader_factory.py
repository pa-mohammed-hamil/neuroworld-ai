
"""
NeuroWorld AI
File: training/datasets/dataloader_factory.py

Centralized PyTorch DataLoader factory.

Features:
- Action, trajectory, video and simulation datasets
- Dataset registry
- Train / validation / test DataLoaders
- Separate dataset paths or dataset splitting
- Episode-aware splitting for transition datasets
- DistributedSampler for DDP
- Reproducible worker seeding
- Configurable DataLoader performance settings
- Optional custom collate functions
- Distributed sampler epoch updates

Dependencies:
    torch
    numpy

Note:
    Episode-aware splitting requires valid episode IDs.
    Random splitting is unsuitable for overlapping
    time-series windows when leakage matters.
"""

from __future__ import annotations

import importlib
import inspect
import random

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import (
    DataLoader,
    Dataset,
    DistributedSampler,
    Subset,
)


@dataclass
class DataLoaderConfig:
    dataset_type: str = "action"

    train_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None

    batch_size: int = 32
    eval_batch_size: int | None = None

    num_workers: int = 0

    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    shuffle_train: bool = True
    drop_last_train: bool = False

    validation_split: float = 0.1
    test_split: float = 0.1

    split_by_episode: bool = True

    distributed: bool = False
    rank: int | None = None
    world_size: int | None = None

    seed: int = 42

    dataset_kwargs: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        if (
            self.eval_batch_size is not None
            and self.eval_batch_size < 1
        ):
            raise ValueError(
                "eval_batch_size must be >= 1"
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers cannot be negative"
            )

        if (
            self.prefetch_factor is not None
            and self.prefetch_factor < 1
        ):
            raise ValueError(
                "prefetch_factor must be >= 1"
            )

        if not (
            0 <= self.validation_split < 1
            and 0 <= self.test_split < 1
        ):
            raise ValueError(
                "Split fractions must be in [0, 1)"
            )

        if (
            self.validation_split
            + self.test_split
            >= 1
        ):
            raise ValueError(
                "Validation and test fractions "
                "must sum to less than 1"
            )


class DataLoaderFactory:
    """
    Factory for building dataset DataLoaders.

    Register a custom dataset:
        factory.register_dataset(
            "custom",
            CustomDataset,
        )

    Dataset classes may accept either:
        Dataset(path=..., **kwargs)
        Dataset(config=...)
        Dataset(path, **kwargs)

    The built-in action dataset is constructed
    through ActionDatasetConfig.
    """

    DEFAULT_DATASETS = {
        "action": (
            "training.datasets.action_dataset",
            "ActionDataset",
        ),
        "trajectory": (
            "training.datasets.trajectory_dataset",
            "TrajectoryDataset",
        ),
        "video": (
            "training.datasets.video_dataset",
            "VideoDataset",
        ),
        "simulation": (
            "training.datasets.simulation_dataset",
            "SimulationDataset",
        ),
    }

    def __init__(
        self,
        config: DataLoaderConfig | None = None,
    ) -> None:
        self.config = (
            config or DataLoaderConfig()
        )

        self.registry: dict[
            str,
            type[Dataset],
        ] = {}

        self.samplers: dict[
            str,
            DistributedSampler,
        ] = {}

    def register_dataset(
        self,
        name: str,
        dataset_class: type[Dataset],
    ) -> None:
        if not issubclass(
            dataset_class,
            Dataset,
        ):
            raise TypeError(
                "dataset_class must extend Dataset"
            )

        self.registry[name.lower()] = (
            dataset_class
        )

    def _resolve_dataset_class(
        self,
        dataset_type: str,
    ) -> type[Dataset]:
        name = dataset_type.lower()

        if name in self.registry:
            return self.registry[name]

        if name not in self.DEFAULT_DATASETS:
            raise ValueError(
                f"Unknown dataset type: {name}"
            )

        module_name, class_name = (
            self.DEFAULT_DATASETS[name]
        )

        module = importlib.import_module(
            module_name
        )

        dataset_class = getattr(
            module,
            class_name,
        )

        return dataset_class

    def create_dataset(
        self,
        path: str,
        *,
        dataset_type: str | None = None,
        **overrides: Any,
    ) -> Dataset:
        name = (
            dataset_type
            or self.config.dataset_type
        ).lower()

        dataset_class = (
            self._resolve_dataset_class(name)
        )

        kwargs = {
            **self.config.dataset_kwargs,
            **overrides,
        }

        if name == "action":
            from training.datasets.action_dataset import (
                ActionDatasetConfig,
            )

            config = ActionDatasetConfig(
                path=path,
                **kwargs,
            )

            return dataset_class(config)

        # For other datasets, support common
        # path-based constructors.
        signature = inspect.signature(
            dataset_class
        )

        if "path" in signature.parameters:
            return dataset_class(
                path=path,
                **kwargs,
            )

        if "data_path" in signature.parameters:
            return dataset_class(
                data_path=path,
                **kwargs,
            )

        if "root_dir" in signature.parameters:
            return dataset_class(
                root_dir=path,
                **kwargs,
            )

        # A custom dataset may accept its path
        # as the first positional argument.
        return dataset_class(
            path,
            **kwargs,
        )

    @staticmethod
    def _seed_worker(
        worker_id: int,
    ) -> None:
        worker_seed = (
            torch.initial_seed() % (2**32)
        )

        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _distributed_info(
        self,
    ) -> tuple[int, int]:
        if not self.config.distributed:
            return 0, 1

        if (
            dist.is_available()
            and dist.is_initialized()
        ):
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        else:
            rank = self.config.rank
            world_size = self.config.world_size

            if rank is None or world_size is None:
                raise RuntimeError(
                    "Distributed mode requires an "
                    "initialized process group or "
                    "explicit rank/world_size"
                )

        if not 0 <= rank < world_size:
            raise ValueError(
                "Invalid distributed rank/world_size"
            )

        return rank, world_size

    def create_dataloader(
        self,
        dataset: Dataset,
        *,
        split: str = "train",
        collate_fn: Callable | None = None,
    ) -> DataLoader:
        if split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                "split must be train, val or test"
            )

        is_train = split == "train"

        sampler = None

        if self.config.distributed:
            rank, world_size = (
                self._distributed_info()
            )

            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=(
                    self.config.shuffle_train
                    if is_train
                    else False
                ),
                seed=self.config.seed,
                drop_last=(
                    self.config.drop_last_train
                    if is_train
                    else False
                ),
            )

            self.samplers[split] = sampler

        generator = torch.Generator()

        generator.manual_seed(
            self.config.seed
        )

        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": (
                self.config.batch_size
                if is_train
                else (
                    self.config.eval_batch_size
                    or self.config.batch_size
                )
            ),
            "shuffle": (
                self.config.shuffle_train
                if is_train and sampler is None
                else False
            ),
            "sampler": sampler,
            "num_workers": (
                self.config.num_workers
            ),
            "pin_memory": (
                self.config.pin_memory
            ),
            "drop_last": (
                self.config.drop_last_train
                if is_train
                else False
            ),
            "worker_init_fn": (
                self._seed_worker
            ),
            "generator": generator,
            "collate_fn": collate_fn,
        }

        if self.config.num_workers > 0:
            kwargs["persistent_workers"] = (
                self.config.persistent_workers
            )

            if (
                self.config.prefetch_factor
                is not None
            ):
                kwargs["prefetch_factor"] = (
                    self.config.prefetch_factor
                )

        return DataLoader(**kwargs)

    def _split_indices(
        self,
        dataset: Dataset,
    ) -> tuple[
        list[int],
        list[int],
        list[int],
    ]:
        n = len(dataset)

        if n < 1:
            raise ValueError(
                "Dataset cannot be empty"
            )

        rng = np.random.default_rng(
            self.config.seed
        )

        if self.config.split_by_episode:
            if not (
                hasattr(dataset, "windows")
                and hasattr(dataset, "episode_ids")
            ):
                raise ValueError(
                    "Episode-aware splitting requires "
                    "dataset.windows and "
                    "dataset.episode_ids"
                )

            groups: dict[
                int,
                list[int],
            ] = {}

            for index, window in enumerate(
                dataset.windows
            ):
                start = int(window[0])

                episode = int(
                    dataset.episode_ids[start]
                )

                groups.setdefault(
                    episode,
                    [],
                ).append(index)

            episode_keys = list(
                groups.keys()
            )

            rng.shuffle(episode_keys)

            n_groups = len(episode_keys)

            n_val = round(
                n_groups
                * self.config.validation_split
            )

            n_test = round(
                n_groups
                * self.config.test_split
            )

            if (
                n_val + n_test
                >= n_groups
            ):
                raise ValueError(
                    "Not enough episodes for splits"
                )

            test_groups = episode_keys[
                :n_test
            ]

            val_groups = episode_keys[
                n_test:n_test + n_val
            ]

            train_groups = episode_keys[
                n_test + n_val:
            ]

            def flatten(
                keys: list[int],
            ) -> list[int]:
                return [
                    index
                    for key in keys
                    for index in groups[key]
                ]

            return (
                flatten(train_groups),
                flatten(val_groups),
                flatten(test_groups),
            )

        indices = rng.permutation(
            n
        ).tolist()

        n_val = round(
            n * self.config.validation_split
        )

        n_test = round(
            n * self.config.test_split
        )

        train_indices = indices[
            n_val + n_test:
        ]

        val_indices = indices[
            :n_val
        ]

        test_indices = indices[
            n_val:n_val + n_test
        ]

        return (
            train_indices,
            val_indices,
            test_indices,
        )

    def create_loaders(
        self,
        *,
        collate_fn: Callable | None = None,
    ) -> dict[str, DataLoader]:
        """
        Create loaders using either:
        1. Explicit train/val/test paths
        2. A single training dataset split

        If explicit val/test paths are supplied,
        no automatic splitting is performed.
        """
        train_path = (
            self.config.train_path
        )

        if not train_path:
            raise ValueError(
                "train_path is required"
            )

        train_dataset = (
            self.create_dataset(
                train_path
            )
        )

        datasets: dict[
            str,
            Dataset,
        ] = {}

        if (
            self.config.val_path is not None
            or self.config.test_path is not None
        ):
            datasets["train"] = (
                train_dataset
            )

            if self.config.val_path:
                datasets["val"] = (
                    self.create_dataset(
                        self.config.val_path
                    )
                )

            if self.config.test_path:
                datasets["test"] = (
                    self.create_dataset(
                        self.config.test_path
                    )
                )

        else:
            (
                train_indices,
                val_indices,
                test_indices,
            ) = self._split_indices(
                train_dataset
            )

            datasets["train"] = Subset(
                train_dataset,
                train_indices,
            )

            if val_indices:
                datasets["val"] = Subset(
                    train_dataset,
                    val_indices,
                )

            if test_indices:
                datasets["test"] = Subset(
                    train_dataset,
                    test_indices,
                )

        return {
            split: self.create_dataloader(
                dataset,
                split=split,
                collate_fn=collate_fn,
            )
            for split, dataset in datasets.items()
        }

    def set_epoch(
        self,
        epoch: int,
    ) -> None:
        """
        Update distributed sampler epoch.

        Call once before every training epoch.
        """
        for sampler in self.samplers.values():
            sampler.set_epoch(epoch)


def create_dataloaders(
    config: DataLoaderConfig,
    *,
    collate_fn: Callable | None = None,
) -> dict[str, DataLoader]:
    factory = DataLoaderFactory(
        config
    )

    return factory.create_loaders(
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    config = DataLoaderConfig(
        dataset_type="action",
        train_path="data/processed/actions_demo.npz",
        batch_size=16,
        validation_split=0.2,
        test_split=0.1,
        split_by_episode=True,
        dataset_kwargs={
            "sequence_length": 8,
            "stride": 1,
        },
    )

    factory = DataLoaderFactory(
        config
    )

    loaders = factory.create_loaders()

    for split, loader in loaders.items():
        print(
            split,
            "samples:",
            len(loader.dataset),
            "batches:",
            len(loader),
        )

        if len(loader) > 0:
            batch = next(iter(loader))

            print(
                "States:",
                batch["states"].shape,
            )

            print(
                "Actions:",
                batch["actions"].shape,
            )
