
"""
NeuroWorld AI
File: training/utils/seed.py

Reproducibility and random seed management
for PyTorch, NumPy, Python, CUDA, DDP and FSDP.

Features:
- Global random seed initialization
- Python, NumPy and PyTorch seeding
- CUDA GPU seeding
- Deterministic training configuration
- cuDNN deterministic settings
- DataLoader worker seeding
- Distributed rank-aware seeds
- Epoch-aware seed generation
- Reproducible random generators
- RNG state capture and restoration
- RNG checkpoint save/load
- Temporary seed context manager
- Reproducibility environment reporting

Dependencies:
    pip install torch numpy

Usage:
    seed_everything(
        42,
        deterministic=True,
    )

    generator = create_generator(42)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )

Notes:
    Deterministic execution may reduce
    training performance.

    Exact reproducibility is not guaranteed
    across different PyTorch versions,
    hardware, drivers or platforms.

    Set deterministic configuration before
    starting training whenever possible.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist


MAX_SEED = 2**32 - 1


@dataclass
class SeedConfig:
    seed: int = 42

    deterministic: bool = True
    warn_only: bool = False

    benchmark: bool = False

    seed_cuda: bool = True

    rank_offset: bool = True
    worker_seed_offset: int = 1000

    cublas_workspace_config: str = ":4096:8"

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= MAX_SEED:
            raise ValueError(
                f"seed must be in [0, {MAX_SEED}]"
            )

        if self.worker_seed_offset < 0:
            raise ValueError(
                "worker_seed_offset must be >= 0"
            )

        if self.cublas_workspace_config not in {
            ":4096:8",
            ":16:8",
        }:
            raise ValueError(
                "Invalid CUBLAS_WORKSPACE_CONFIG"
            )


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(
        seed,
        int,
    ):
        raise TypeError(
            "seed must be an integer"
        )

    if not 0 <= seed <= MAX_SEED:
        raise ValueError(
            f"seed must be in [0, {MAX_SEED}]"
        )

    return seed


def get_distributed_rank() -> int:
    """
    Return the current distributed rank.

    Falls back to the RANK environment
    variable when the process group has
    not yet been initialized.
    """
    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        return dist.get_rank()

    return int(
        os.environ.get("RANK", "0")
    )


def get_world_size() -> int:
    if (
        dist.is_available()
        and dist.is_initialized()
    ):
        return dist.get_world_size()

    return int(
        os.environ.get("WORLD_SIZE", "1")
    )


def derive_seed(
    base_seed: int,
    *,
    rank: int = 0,
    worker_id: int = 0,
    epoch: int = 0,
    stream: int = 0,
) -> int:
    """
    Generate a reproducible seed from
    a base seed and execution identifiers.

    Useful for distributed training,
    DataLoader workers and epochs.
    """
    validate_seed(base_seed)

    for name, value in {
        "rank": rank,
        "worker_id": worker_id,
        "epoch": epoch,
        "stream": stream,
    }.items():
        if value < 0:
            raise ValueError(
                f"{name} must be nonnegative"
            )

    # Stable integer mixing; no reliance
    # on Python's randomized hash().
    value = base_seed

    for component in (
        rank,
        worker_id,
        epoch,
        stream,
    ):
        value = (
            value * 1664525
            + 1013904223
            + component * 2654435761
        ) % (MAX_SEED + 1)

    return int(value)


def seed_everything(
    seed: int = 42,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
    benchmark: bool = False,
    seed_cuda: bool = True,
    rank: int | None = None,
    rank_offset: bool = False,
    cublas_workspace_config: str = ":4096:8",
) -> int:
    """
    Initialize Python, NumPy and PyTorch RNGs.

    Args:
        seed:
            Base random seed.

        deterministic:
            Enable deterministic PyTorch
            algorithms and cuDNN behavior.

        warn_only:
            Warn instead of raising when
            a deterministic implementation
            is unavailable.

        benchmark:
            Enable cuDNN benchmarking only
            when deterministic=False.

        rank:
            Distributed process rank.

        rank_offset:
            Derive a different random seed
            for each distributed rank.

    Returns:
        Effective seed used.
    """
    validate_seed(seed)

    if rank is None:
        rank = get_distributed_rank()

    if rank_offset:
        effective_seed = derive_seed(
            seed,
            rank=rank,
        )
    else:
        effective_seed = seed

    if deterministic:
        if cublas_workspace_config not in {
            ":4096:8",
            ":16:8",
        }:
            raise ValueError(
                "Invalid CUBLAS_WORKSPACE_CONFIG"
            )

        # Must be configured before CUDA
        # creates its cuBLAS workspace.
        os.environ[
            "CUBLAS_WORKSPACE_CONFIG"
        ] = cublas_workspace_config

    random.seed(
        effective_seed
    )

    np.random.seed(
        effective_seed
    )

    torch.manual_seed(
        effective_seed
    )

    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            effective_seed
        )

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = (
            deterministic
        )

        torch.backends.cudnn.benchmark = (
            benchmark
            if not deterministic
            else False
        )

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=warn_only,
    )

    return effective_seed


def seed_from_config(
    config: SeedConfig | Mapping[str, Any],
    *,
    rank: int | None = None,
) -> int:
    if isinstance(config, Mapping):
        config = SeedConfig(
            **dict(config)
        )

    return seed_everything(
        seed=config.seed,
        deterministic=config.deterministic,
        warn_only=config.warn_only,
        benchmark=config.benchmark,
        seed_cuda=config.seed_cuda,
        rank=rank,
        rank_offset=config.rank_offset,
        cublas_workspace_config=(
            config.cublas_workspace_config
        ),
    )


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function.

    PyTorch assigns each worker a seed.
    Use that seed to initialize NumPy
    and Python's random module.

    Example:
        DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=create_generator(42),
        )
    """
    worker_seed = (
        torch.initial_seed()
        % (MAX_SEED + 1)
    )

    np.random.seed(
        worker_seed
    )

    random.seed(
        worker_seed
    )


def create_generator(
    seed: int = 42,
    *,
    device: str | torch.device = "cpu",
    rank: int | None = None,
    worker_id: int = 0,
    epoch: int = 0,
    stream: int = 0,
) -> torch.Generator:
    """
    Create an independently seeded
    PyTorch random generator.
    """
    validate_seed(seed)

    if rank is None:
        rank = 0

    effective_seed = derive_seed(
        seed,
        rank=rank,
        worker_id=worker_id,
        epoch=epoch,
        stream=stream,
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        effective_seed
    )

    return generator


def create_numpy_generator(
    seed: int = 42,
    *,
    rank: int = 0,
    worker_id: int = 0,
    epoch: int = 0,
    stream: int = 0,
) -> np.random.Generator:
    """
    Create a local NumPy generator
    without modifying global RNG state.
    """
    effective_seed = derive_seed(
        seed,
        rank=rank,
        worker_id=worker_id,
        epoch=epoch,
        stream=stream,
    )

    return np.random.default_rng(
        effective_seed
    )


def create_python_generator(
    seed: int = 42,
    *,
    rank: int = 0,
    worker_id: int = 0,
    epoch: int = 0,
    stream: int = 0,
) -> random.Random:
    """
    Create an independent Python RNG.
    """
    effective_seed = derive_seed(
        seed,
        rank=rank,
        worker_id=worker_id,
        epoch=epoch,
        stream=stream,
    )

    return random.Random(
        effective_seed
    )


def seed_for_epoch(
    base_seed: int,
    epoch: int,
    *,
    rank: int | None = None,
) -> int:
    """
    Derive an epoch-specific seed.

    For DistributedSampler, also call:
        sampler.set_epoch(epoch)

    This function does not replace
    DistributedSampler.set_epoch().
    """
    if rank is None:
        rank = get_distributed_rank()

    return derive_seed(
        base_seed,
        rank=rank,
        epoch=epoch,
    )


def seed_for_worker(
    base_seed: int,
    worker_id: int,
    *,
    epoch: int = 0,
    rank: int | None = None,
) -> int:
    if rank is None:
        rank = get_distributed_rank()

    return derive_seed(
        base_seed,
        rank=rank,
        worker_id=worker_id,
        epoch=epoch,
    )


def capture_rng_state(
    *,
    include_cuda: bool = True,
) -> dict[str, Any]:
    """
    Capture Python, NumPy and PyTorch RNG states.

    NumPy's legacy RNG state is converted
    into a weights_only-compatible format.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    state: dict[str, Any] = {
        "python": {
            "version": int(
                python_state[0]
            ),
            "state": list(
                python_state[1]
            ),
            "gauss_next": (
                python_state[2]
            ),
        },
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(
                numpy_state[1].copy()
            ),
            "position": int(
                numpy_state[2]
            ),
            "has_gauss": int(
                numpy_state[3]
            ),
            "cached_gaussian": float(
                numpy_state[4]
            ),
        },
        "torch_cpu": (
            torch.get_rng_state().clone()
        ),
    }

    if (
        include_cuda
        and torch.cuda.is_available()
    ):
        state["torch_cuda"] = [
            value.cpu().clone()
            for value in (
                torch.cuda.get_rng_state_all()
            )
        ]

    return state


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    restore_cuda: bool = True,
    strict_cuda_devices: bool = False,
) -> None:
    """
    Restore RNG states previously captured
    with capture_rng_state().

    For exact continuation, restore after
    model/optimizer construction and before
    generating the next training batch.
    """
    if "python" in state:
        python_state = state["python"]

        random.setstate(
            (
                int(
                    python_state["version"]
                ),
                tuple(
                    python_state["state"]
                ),
                python_state.get(
                    "gauss_next"
                ),
            )
        )

    if "numpy" in state:
        numpy_state = state["numpy"]

        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.uint32,
                    copy=False,
                ),
                int(
                    numpy_state["position"]
                ),
                int(
                    numpy_state["has_gauss"]
                ),
                float(
                    numpy_state["cached_gaussian"]
                ),
            )
        )

    if "torch_cpu" in state:
        torch.set_rng_state(
            state["torch_cpu"].cpu()
        )

    if (
        restore_cuda
        and torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        cuda_states = state["torch_cuda"]

        device_count = (
            torch.cuda.device_count()
        )

        if len(cuda_states) != device_count:
            if strict_cuda_devices:
                raise RuntimeError(
                    "CUDA device count does not "
                    "match saved RNG state"
                )

            # Restore matching devices only.
            for device_index, cuda_state in enumerate(
                cuda_states[:device_count]
            ):
                torch.cuda.set_rng_state(
                    cuda_state.cpu(),
                    device=device_index,
                )

        else:
            torch.cuda.set_rng_state_all(
                [
                    cuda_state.cpu()
                    for cuda_state in cuda_states
                ]
            )


def save_rng_checkpoint(
    path: str | Path,
    *,
    include_cuda: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """
    Save RNG states to a PyTorch checkpoint.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "rng_state": capture_rng_state(
            include_cuda=include_cuda
        ),
        "metadata": dict(
            metadata or {}
        ),
    }

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_rng_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    restore_cuda: bool = True,
) -> dict[str, Any]:
    """
    Load a trusted RNG checkpoint and
    restore its random generator states.
    """
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    restore_rng_state(
        checkpoint["rng_state"],
        restore_cuda=restore_cuda,
    )

    return dict(
        checkpoint.get(
            "metadata",
            {},
        )
    )


@contextmanager
def temporary_seed(
    seed: int,
    *,
    include_cuda: bool = True,
    deterministic: bool | None = None,
) -> Iterator[None]:
    """
    Temporarily set a random seed.

    All captured RNG states are restored
    when leaving the context.

    Example:
        with temporary_seed(42):
            sample = torch.randn(10)

    Note:
        This context is not thread-safe.
    """
    validate_seed(seed)

    original_state = capture_rng_state(
        include_cuda=include_cuda
    )

    original_deterministic = (
        torch.are_deterministic_algorithms_enabled()
    )

    original_warn_only = (
        torch.is_deterministic_algorithms_warn_only_enabled()
    )

    original_cudnn_deterministic = (
        torch.backends.cudnn.deterministic
    )

    original_cudnn_benchmark = (
        torch.backends.cudnn.benchmark
    )

    original_cublas_config = os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    )

    try:
        if deterministic is None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if (
                include_cuda
                and torch.cuda.is_available()
            ):
                torch.cuda.manual_seed_all(
                    seed
                )

        else:
            seed_everything(
                seed,
                deterministic=deterministic,
                seed_cuda=include_cuda,
            )

        yield

    finally:
        restore_rng_state(
            original_state,
            restore_cuda=include_cuda,
        )

        torch.use_deterministic_algorithms(
            original_deterministic,
            warn_only=original_warn_only,
        )

        torch.backends.cudnn.deterministic = (
            original_cudnn_deterministic
        )

        torch.backends.cudnn.benchmark = (
            original_cudnn_benchmark
        )

        if original_cublas_config is None:
            os.environ.pop(
                "CUBLAS_WORKSPACE_CONFIG",
                None,
            )
        else:
            os.environ[
                "CUBLAS_WORKSPACE_CONFIG"
            ] = original_cublas_config


def get_reproducibility_info() -> dict[str, Any]:
    """
    Return reproducibility configuration
    and runtime environment information.
    """
    information = {
        "torch_version": str(
            torch.__version__
        ),
        "numpy_version": str(
            np.__version__
        ),
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cuda_version": (
            torch.version.cuda
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),
        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),
        "cublas_workspace_config": (
            os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            )
        ),
        "distributed_rank": (
            get_distributed_rank()
        ),
        "world_size": (
            get_world_size()
        ),
    }

    if torch.cuda.is_available():
        information["gpu_count"] = (
            torch.cuda.device_count()
        )

        information["gpu_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(
                torch.cuda.device_count()
            )
        ]

    return information


class SeedManager:
    """
    Centralized reproducibility manager.

    Example:
        manager = SeedManager(
            SeedConfig(
                seed=42,
                deterministic=True,
            )
        )

        manager.initialize()

        generator = manager.generator(
            epoch=0,
        )

        state = manager.state_dict()
        manager.load_state_dict(state)
    """

    def __init__(
        self,
        config: (
            SeedConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = SeedConfig()

        elif isinstance(config, Mapping):
            config = SeedConfig(
                **dict(config)
            )

        self.config = config

        self.rank = get_distributed_rank()
        self.world_size = get_world_size()

        self.effective_seed: int | None = None
        self.current_epoch = 0

    def initialize(self) -> int:
        self.rank = get_distributed_rank()
        self.world_size = get_world_size()

        self.effective_seed = (
            seed_from_config(
                self.config,
                rank=self.rank,
            )
        )

        return self.effective_seed

    def set_epoch(
        self,
        epoch: int,
        *,
        reseed_global: bool = False,
    ) -> int:
        if epoch < 0:
            raise ValueError(
                "epoch must be nonnegative"
            )

        self.current_epoch = epoch

        epoch_seed = derive_seed(
            self.config.seed,
            rank=(
                self.rank
                if self.config.rank_offset
                else 0
            ),
            epoch=epoch,
        )

        if reseed_global:
            seed_everything(
                epoch_seed,
                deterministic=(
                    self.config.deterministic
                ),
                warn_only=(
                    self.config.warn_only
                ),
                benchmark=(
                    self.config.benchmark
                ),
                seed_cuda=(
                    self.config.seed_cuda
                ),
                rank_offset=False,
                cublas_workspace_config=(
                    self.config.cublas_workspace_config
                ),
            )

        return epoch_seed

    def generator(
        self,
        *,
        device: str | torch.device = "cpu",
        epoch: int | None = None,
        worker_id: int = 0,
        stream: int = 0,
    ) -> torch.Generator:
        return create_generator(
            self.config.seed,
            device=device,
            rank=(
                self.rank
                if self.config.rank_offset
                else 0
            ),
            worker_id=worker_id,
            epoch=(
                self.current_epoch
                if epoch is None
                else epoch
            ),
            stream=stream,
        )

    def numpy_generator(
        self,
        *,
        epoch: int | None = None,
        worker_id: int = 0,
        stream: int = 0,
    ) -> np.random.Generator:
        return create_numpy_generator(
            self.config.seed,
            rank=(
                self.rank
                if self.config.rank_offset
                else 0
            ),
            worker_id=worker_id,
            epoch=(
                self.current_epoch
                if epoch is None
                else epoch
            ),
            stream=stream,
        )

    def worker_seed(
        self,
        worker_id: int,
        *,
        epoch: int | None = None,
    ) -> int:
        return seed_for_worker(
            self.config.seed,
            worker_id,
            rank=(
                self.rank
                if self.config.rank_offset
                else 0
            ),
            epoch=(
                self.current_epoch
                if epoch is None
                else epoch
            ),
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "config": asdict(
                self.config
            ),
            "rank": self.rank,
            "world_size": self.world_size,
            "effective_seed": (
                self.effective_seed
            ),
            "current_epoch": (
                self.current_epoch
            ),
            "rng_state": capture_rng_state(
                include_cuda=(
                    self.config.seed_cuda
                )
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        restore_rng: bool = True,
    ) -> None:
        self.rank = int(
            state.get(
                "rank",
                self.rank,
            )
        )

        self.world_size = int(
            state.get(
                "world_size",
                self.world_size,
            )
        )

        self.effective_seed = state.get(
            "effective_seed"
        )

        self.current_epoch = int(
            state.get(
                "current_epoch",
                0,
            )
        )

        if (
            restore_rng
            and "rng_state" in state
        ):
            restore_rng_state(
                state["rng_state"],
                restore_cuda=(
                    self.config.seed_cuda
                ),
            )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> Path:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )

        return path

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        restore_rng: bool = True,
    ) -> None:
        state = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        self.load_state_dict(
            state,
            restore_rng=restore_rng,
        )

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "base_seed": self.config.seed,
            "effective_seed": (
                self.effective_seed
            ),
            "rank": self.rank,
            "world_size": self.world_size,
            "current_epoch": (
                self.current_epoch
            ),
            "deterministic": (
                self.config.deterministic
            ),
            "cuda_seeded": (
                self.config.seed_cuda
            ),
        }


if __name__ == "__main__":
    print("NeuroWorld AI Seed Manager")

    manager = SeedManager(
        SeedConfig(
            seed=42,
            deterministic=True,
            warn_only=True,
            benchmark=False,
            seed_cuda=True,
            rank_offset=True,
        )
    )

    effective_seed = manager.initialize()

    print(
        "Effective seed:",
        effective_seed,
    )

    print("\nPyTorch random values:")
    print(
        torch.randn(5)
    )

    print("\nNumPy random values:")
    print(
        np.random.rand(5)
    )

    print("\nPython random value:")
    print(
        random.random()
    )

    print("\nReproducible generator test:")

    generator_a = manager.generator(
        stream=1
    )

    generator_b = manager.generator(
        stream=1
    )

    values_a = torch.randn(
        5,
        generator=generator_a,
    )

    values_b = torch.randn(
        5,
        generator=generator_b,
    )

    print("Generator A:", values_a)
    print("Generator B:", values_b)

    print(
        "Identical:",
        torch.equal(
            values_a,
            values_b,
        ),
    )

    print("\nTemporary seed test:")

    with temporary_seed(123):
        temporary_values = torch.randn(3)

    with temporary_seed(123):
        repeated_values = torch.randn(3)

    print(
        "Temporary seed reproducible:",
        torch.equal(
            temporary_values,
            repeated_values,
        ),
    )

    print("\nEpoch seeds:")

    for epoch in range(3):
        print(
            f"Epoch {epoch}: "
            f"{manager.set_epoch(epoch)}"
        )

    print("\nManager statistics:")
    print(
        manager.get_statistics()
    )

    print("\nReproducibility information:")
    print(
        get_reproducibility_info()
    )
