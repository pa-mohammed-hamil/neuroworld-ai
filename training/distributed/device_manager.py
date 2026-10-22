
"""
NeuroWorld AI
File: training/distributed/device_manager.py

Device management for distributed AI training.

Features:
- Automatic CPU / CUDA / MPS selection
- torchrun LOCAL_RANK support
- GPU capability detection
- BF16 / FP16 mixed precision
- Recursive batch-to-device transfer
- Model and optimizer device handling
- CUDA memory monitoring
- GPU cache cleanup
- Reproducible random seeds
- Optional TF32 acceleration
- Device information reporting

Dependencies:
    torch
    numpy
"""

from __future__ import annotations

import contextlib
import os
import random

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from torch import Tensor, nn


@dataclass
class DeviceConfig:
    device: str = "auto"

    local_rank: int | None = None

    mixed_precision: str = "auto"

    allow_tf32: bool = True

    deterministic: bool = False

    seed: int = 42

    empty_cache_on_cleanup: bool = True

    def __post_init__(self) -> None:
        valid_devices = (
            "auto",
            "cpu",
            "cuda",
            "mps",
        )

        if (
            self.device not in valid_devices
            and not self.device.startswith("cuda:")
        ):
            raise ValueError(
                f"Unsupported device: {self.device}"
            )

        if self.mixed_precision not in (
            "auto",
            "none",
            "fp16",
            "bf16",
        ):
            raise ValueError(
                "mixed_precision must be "
                "auto, none, fp16, or bf16"
            )


class DeviceManager:
    """
    Hardware/device manager for NeuroWorld AI.

    Compatible with:
    - Standard PyTorch training
    - DistributedDataParallel
    - World-model training
    - Video prediction
    - Robotics policy learning
    """

    def __init__(
        self,
        config: DeviceConfig | None = None,
    ) -> None:
        self.config = config or DeviceConfig()

        self.local_rank = (
            self.config.local_rank
            if self.config.local_rank is not None
            else int(
                os.environ.get(
                    "LOCAL_RANK",
                    "0",
                )
            )
        )

        self.rank = int(
            os.environ.get(
                "RANK",
                "0",
            )
        )

        self.world_size = int(
            os.environ.get(
                "WORLD_SIZE",
                "1",
            )
        )

        self.device = self._select_device()

        self._configure_backend()

        self.precision = self._select_precision()

        self.use_amp = (
            self.precision != "none"
        )

        self.amp_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "none": torch.float32,
        }[self.precision]

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.device.type == "cuda"
                and self.precision == "fp16"
            ),
        )

        self.seed_everything(
            self.config.seed + self.rank
        )

    def _select_device(
        self,
    ) -> torch.device:
        requested = self.config.device

        if requested == "cpu":
            return torch.device("cpu")

        if requested == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "Apple MPS is unavailable"
                )

            if self.world_size > 1:
                raise RuntimeError(
                    "Multi-process MPS training "
                    "is not supported by this manager"
                )

            return torch.device("mps")

        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but "
                    "is unavailable"
                )

            if requested == "cuda":
                index = (
                    self.local_rank
                    if self.world_size > 1
                    else 0
                )
            else:
                index = int(
                    requested.split(":", 1)[1]
                )

            return self._cuda_device(index)

        # Automatic selection
        if torch.cuda.is_available():
            index = (
                self.local_rank
                if self.world_size > 1
                else 0
            )

            return self._cuda_device(index)

        if (
            self.world_size == 1
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    @staticmethod
    def _cuda_device(
        index: int,
    ) -> torch.device:
        count = torch.cuda.device_count()

        if not 0 <= index < count:
            raise ValueError(
                f"Invalid CUDA device {index}. "
                f"Available GPUs: {count}"
            )

        torch.cuda.set_device(index)

        return torch.device(
            "cuda",
            index,
        )

    def _configure_backend(
        self,
    ) -> None:
        torch.backends.cudnn.benchmark = (
            not self.config.deterministic
        )

        torch.backends.cudnn.deterministic = (
            self.config.deterministic
        )

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = (
                self.config.allow_tf32
            )

            torch.backends.cudnn.allow_tf32 = (
                self.config.allow_tf32
            )

        if self.config.deterministic:
            torch.use_deterministic_algorithms(
                True
            )

    def _select_precision(
        self,
    ) -> str:
        requested = (
            self.config.mixed_precision
        )

        if requested == "none":
            return "none"

        if self.device.type == "cpu":
            if requested == "fp16":
                raise RuntimeError(
                    "CPU FP16 is unsupported "
                    "by this manager"
                )

            return (
                "bf16"
                if requested == "bf16"
                else "none"
            )

        if self.device.type == "mps":
            if requested == "bf16":
                raise RuntimeError(
                    "MPS BF16 is not enabled "
                    "by this manager"
                )

            return (
                "fp16"
                if requested == "fp16"
                else "none"
            )

        # CUDA
        bf16_supported = (
            torch.cuda.is_bf16_supported()
        )

        if requested == "bf16":
            if not bf16_supported:
                raise RuntimeError(
                    "CUDA BF16 is unavailable"
                )

            return "bf16"

        if requested == "fp16":
            return "fp16"

        return (
            "bf16"
            if bf16_supported
            else "fp16"
        )

    def seed_everything(
        self,
        seed: int,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def move_model(
        self,
        model: nn.Module,
    ) -> nn.Module:
        return model.to(
            self.device
        )

    def move_tensor(
        self,
        tensor: Tensor,
        *,
        non_blocking: bool = True,
    ) -> Tensor:
        return tensor.to(
            self.device,
            non_blocking=non_blocking,
        )

    def move_batch(
        self,
        batch: Any,
    ) -> Any:
        """
        Recursively transfer tensors.

        Supports:
        - Tensor
        - Dictionary
        - List
        - Tuple
        - Other values (unchanged)
        """
        if isinstance(batch, Tensor):
            return self.move_tensor(
                batch
            )

        if isinstance(batch, Mapping):
            return {
                key: self.move_batch(value)
                for key, value in batch.items()
            }

        if isinstance(batch, tuple):
            return tuple(
                self.move_batch(value)
                for value in batch
            )

        if isinstance(batch, list):
            return [
                self.move_batch(value)
                for value in batch
            ]

        return batch

    def move_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Move existing optimizer state tensors
        to the selected device.

        Useful after loading a checkpoint.
        """
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(
                        self.device
                    )

    def autocast(
        self,
    ):
        if not self.use_amp:
            return contextlib.nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )

    def backward(
        self,
        loss: Tensor,
    ) -> None:
        self.scaler.scale(
            loss
        ).backward()

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        model: nn.Module | None = None,
        max_grad_norm: float | None = None,
    ) -> bool:
        """
        Perform an AMP-aware optimizer step.

        Returns:
            True if the optimizer stepped.
            False if GradScaler skipped the
            update due to non-finite gradients.
        """
        if max_grad_norm is not None:
            if model is None:
                raise ValueError(
                    "model is required for "
                    "gradient clipping"
                )

            self.scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

        previous_scale = (
            self.scaler.get_scale()
        )

        self.scaler.step(
            optimizer
        )

        self.scaler.update()

        return (
            not self.scaler.is_enabled()
            or self.scaler.get_scale()
            >= previous_scale
        )

    def memory_info(
        self,
    ) -> dict[str, float]:
        """
        Return CUDA memory statistics in MB.

        CPU/MPS returns an empty dictionary.
        """
        if self.device.type != "cuda":
            return {}

        index = self.device.index

        allocated = torch.cuda.memory_allocated(
            index
        )

        reserved = torch.cuda.memory_reserved(
            index
        )

        peak = torch.cuda.max_memory_allocated(
            index
        )

        free, total = torch.cuda.mem_get_info(
            index
        )

        mb = 1024 ** 2

        return {
            "allocated_mb": round(
                allocated / mb,
                2,
            ),
            "reserved_mb": round(
                reserved / mb,
                2,
            ),
            "peak_allocated_mb": round(
                peak / mb,
                2,
            ),
            "free_mb": round(
                free / mb,
                2,
            ),
            "total_mb": round(
                total / mb,
                2,
            ),
        }

    def reset_peak_memory(
        self,
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(
                self.device
            )

    def synchronize(
        self,
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        elif self.device.type == "mps":
            torch.mps.synchronize()

    def empty_cache(
        self,
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def device_info(
        self,
    ) -> dict[str, Any]:
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "precision": self.precision,
            "amp_enabled": self.use_amp,
            "torch_version": torch.__version__,
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "mps_available": (
                torch.backends.mps.is_available()
            ),
        }

        if self.device.type == "cuda":
            index = self.device.index

            properties = (
                torch.cuda.get_device_properties(
                    index
                )
            )

            info.update({
                "gpu_name": properties.name,
                "gpu_count": (
                    torch.cuda.device_count()
                ),
                "gpu_memory_gb": round(
                    properties.total_memory
                    / (1024 ** 3),
                    2,
                ),
                "compute_capability": (
                    f"{properties.major}."
                    f"{properties.minor}"
                ),
                "bf16_supported": (
                    torch.cuda.is_bf16_supported()
                ),
            })

        return info

    def print_info(
        self,
    ) -> None:
        if self.rank != 0:
            return

        print(
            "\nNeuroWorld AI Device Information"
        )

        for key, value in (
            self.device_info().items()
        ):
            print(
                f"{key}: {value}"
            )

        if self.device.type == "cuda":
            print(
                "\nCUDA Memory:"
            )

            for key, value in (
                self.memory_info().items()
            ):
                print(
                    f"{key}: {value}"
                )

    def cleanup(
        self,
    ) -> None:
        if self.config.empty_cache_on_cleanup:
            self.empty_cache()

    def __enter__(
        self,
    ) -> "DeviceManager":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.cleanup()


def get_device(
    device: str = "auto",
) -> torch.device:
    """
    Convenience device selection.
    """
    return DeviceManager(
        DeviceConfig(
            device=device,
            mixed_precision="none",
        )
    ).device


if __name__ == "__main__":
    config = DeviceConfig(
        device="auto",
        mixed_precision="auto",
        seed=42,
    )

    with DeviceManager(config) as manager:
        manager.print_info()

        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        model = manager.move_model(
            model
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
        )

        inputs = torch.randn(
            16,
            128,
        )

        targets = torch.randn(
            16,
            128,
        )

        inputs = manager.move_tensor(
            inputs
        )

        targets = manager.move_tensor(
            targets
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with manager.autocast():
            predictions = model(inputs)

            loss = torch.nn.functional.mse_loss(
                predictions,
                targets,
            )

        manager.backward(loss)

        stepped = manager.optimizer_step(
            optimizer,
            model=model,
            max_grad_norm=1.0,
        )

        print(
            "\nTraining loss:",
            loss.item(),
        )

        print(
            "Optimizer stepped:",
            stepped,
        )
