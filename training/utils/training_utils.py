
"""
NeuroWorld AI
File: training/utils/training_utils.py

Shared PyTorch training utilities for world models,
video prediction, dynamics, robotics, and RL.

Features:
- Recursive batch-to-device transfer
- Model parameter statistics
- Gradient norm calculation and clipping
- Gradient accumulation
- Automatic mixed precision (AMP)
- Safe optimizer updates
- Non-finite loss/gradient detection
- Learning-rate utilities
- Model freezing/unfreezing
- Training and validation steps
- Training progress and ETA
- Throughput monitoring
- Batch splitting
- GPU memory statistics
- Checkpoint-compatible state management

Dependencies:
    pip install torch

Compatible with:
    training/utils/seed.py
    training/utils/checkpoint_io.py
    training/utils/metric_tracker.py
    training/utils/experiment_logger.py
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class TrainingConfig:
    device: str = "auto"

    mixed_precision: bool = True
    precision: str = "float16"

    accumulation_steps: int = 1

    gradient_clip_norm: float | None = 1.0
    gradient_clip_value: float | None = None

    detect_nonfinite: bool = True
    skip_nonfinite_steps: bool = True

    zero_grad_set_to_none: bool = True

    channels_last: bool = False

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be >= 1"
            )

        if self.precision not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError(
                "Unsupported precision"
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive"
            )

        if (
            self.gradient_clip_value is not None
            and self.gradient_clip_value <= 0
        ):
            raise ValueError(
                "gradient_clip_value must be positive"
            )


@dataclass
class TrainingStepResult:
    loss: float
    optimizer_updated: bool
    gradient_norm: float | None
    learning_rates: list[float]
    global_step: int
    optimizer_step: int
    skipped_steps: int
    accumulation_progress: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(
    device: str | torch.device = "auto",
) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


def resolve_dtype(
    precision: str,
) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if precision not in mapping:
        raise ValueError(
            f"Unsupported precision: {precision}"
        )

    return mapping[precision]


def move_to_device(
    batch: Any,
    device: str | torch.device,
    *,
    non_blocking: bool = True,
    dtype: torch.dtype | None = None,
) -> Any:
    """
    Recursively move tensors to a device.

    Supports:
        Tensor
        dict
        list
        tuple
        namedtuple

    Floating-point tensors can optionally
    be converted to a target dtype.
    Integer and boolean tensors retain
    their original dtype.
    """
    device = resolve_device(device)

    if isinstance(batch, Tensor):
        target_dtype = (
            dtype
            if dtype is not None
            and batch.is_floating_point()
            else batch.dtype
        )

        return batch.to(
            device=device,
            dtype=target_dtype,
            non_blocking=non_blocking,
        )

    if isinstance(batch, Mapping):
        return {
            key: move_to_device(
                value,
                device,
                non_blocking=non_blocking,
                dtype=dtype,
            )
            for key, value in batch.items()
        }

    if isinstance(batch, tuple):
        values = [
            move_to_device(
                item,
                device,
                non_blocking=non_blocking,
                dtype=dtype,
            )
            for item in batch
        ]

        if hasattr(batch, "_fields"):
            return type(batch)(*values)

        return tuple(values)

    if isinstance(batch, list):
        return [
            move_to_device(
                item,
                device,
                non_blocking=non_blocking,
                dtype=dtype,
            )
            for item in batch
        ]

    return batch


def detach_to_cpu(
    value: Any,
) -> Any:
    """
    Recursively detach tensors and
    transfer them to CPU.
    """
    if isinstance(value, Tensor):
        return value.detach().cpu()

    if isinstance(value, Mapping):
        return {
            key: detach_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        values = [
            detach_to_cpu(item)
            for item in value
        ]

        if hasattr(value, "_fields"):
            return type(value)(*values)

        return tuple(values)

    if isinstance(value, list):
        return [
            detach_to_cpu(item)
            for item in value
        ]

    return value


def infer_batch_size(
    batch: Any,
) -> int:
    """
    Infer batch size from the first
    tensor with at least one dimension.
    """
    if isinstance(batch, Tensor):
        if batch.ndim == 0:
            raise ValueError(
                "Cannot infer batch size "
                "from a scalar tensor"
            )

        return int(batch.shape[0])

    if isinstance(batch, Mapping):
        for value in batch.values():
            try:
                return infer_batch_size(value)
            except ValueError:
                continue

    if isinstance(batch, (list, tuple)):
        for value in batch:
            try:
                return infer_batch_size(value)
            except ValueError:
                continue

    raise ValueError(
        "Unable to infer batch size"
    )


def split_batch(
    batch: Any,
    microbatch_size: int,
) -> list[Any]:
    """
    Split a nested batch into microbatches.

    Tensors must share the same leading
    batch dimension. Non-tensor metadata
    is replicated across microbatches.
    """
    if microbatch_size < 1:
        raise ValueError(
            "microbatch_size must be >= 1"
        )

    batch_size = infer_batch_size(batch)

    def slice_value(
        value: Any,
        start: int,
        end: int,
    ) -> Any:
        if isinstance(value, Tensor):
            if (
                value.ndim > 0
                and value.shape[0] == batch_size
            ):
                return value[start:end]

            return value

        if isinstance(value, Mapping):
            return {
                key: slice_value(
                    item,
                    start,
                    end,
                )
                for key, item in value.items()
            }

        if isinstance(value, tuple):
            values = [
                slice_value(item, start, end)
                for item in value
            ]

            if hasattr(value, "_fields"):
                return type(value)(*values)

            return tuple(values)

        if isinstance(value, list):
            return [
                slice_value(item, start, end)
                for item in value
            ]

        return value

    return [
        slice_value(
            batch,
            start,
            min(
                start + microbatch_size,
                batch_size,
            ),
        )
        for start in range(
            0,
            batch_size,
            microbatch_size,
        )
    ]


def count_parameters(
    model: nn.Module,
) -> dict[str, int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def get_trainable_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def freeze_model(
    model: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )


def unfreeze_model(
    model: nn.Module,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(
            True
        )


def freeze_layers(
    model: nn.Module,
    prefixes: Sequence[str],
) -> int:
    frozen = 0

    for name, parameter in model.named_parameters():
        if any(
            name.startswith(prefix)
            for prefix in prefixes
        ):
            parameter.requires_grad_(
                False
            )
            frozen += parameter.numel()

    return frozen


def unfreeze_layers(
    model: nn.Module,
    prefixes: Sequence[str],
) -> int:
    unfrozen = 0

    for name, parameter in model.named_parameters():
        if any(
            name.startswith(prefix)
            for prefix in prefixes
        ):
            parameter.requires_grad_(
                True
            )
            unfrozen += parameter.numel()

    return unfrozen


def get_optimizer_parameters(
    optimizer: Optimizer,
) -> list[nn.Parameter]:
    """
    Return unique optimizer parameters.
    """
    parameters = []
    seen = set()

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identifier = id(parameter)

            if identifier not in seen:
                parameters.append(
                    parameter
                )
                seen.add(identifier)

    return parameters


def calculate_gradient_norm(
    parameters: Sequence[nn.Parameter],
    *,
    norm_type: float = 2.0,
) -> float:
    """
    Calculate global gradient norm.
    """
    gradients = []

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        if gradient.is_sparse:
            gradient = (
                gradient.coalesce().values()
            )

        gradients.append(gradient)

    if not gradients:
        return 0.0

    if norm_type == float("inf"):
        return max(
            float(
                gradient.abs().max().item()
            )
            for gradient in gradients
        )

    if norm_type <= 0:
        raise ValueError(
            "norm_type must be positive"
        )

    total = sum(
        float(
            gradient.float()
            .abs()
            .pow(norm_type)
            .sum()
            .item()
        )
        for gradient in gradients
    )

    return total ** (
        1.0 / norm_type
    )


def gradients_are_finite(
    parameters: Sequence[nn.Parameter],
) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad

        if gradient.is_sparse:
            gradient = (
                gradient.coalesce().values()
            )

        if not bool(
            torch.isfinite(gradient).all().item()
        ):
            return False

    return True


def clip_gradients(
    parameters: Sequence[nn.Parameter],
    *,
    max_norm: float | None = 1.0,
    max_value: float | None = None,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradients and return the
    pre-clipping global gradient norm.
    """
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not parameters:
        return 0.0

    gradient_norm = (
        calculate_gradient_norm(
            parameters,
            norm_type=norm_type,
        )
    )

    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=True,
        )

    if max_value is not None:
        torch.nn.utils.clip_grad_value_(
            parameters,
            clip_value=max_value,
        )

    return gradient_norm


def get_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def set_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
    *,
    group_index: int | None = None,
) -> None:
    if learning_rate < 0:
        raise ValueError(
            "learning_rate must be nonnegative"
        )

    if group_index is not None:
        optimizer.param_groups[
            group_index
        ]["lr"] = learning_rate
        return

    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def get_gpu_memory_statistics(
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }

    device = torch.device(device)

    return {
        "allocated_mb": (
            torch.cuda.memory_allocated(device)
            / 1024**2
        ),
        "reserved_mb": (
            torch.cuda.memory_reserved(device)
            / 1024**2
        ),
        "max_allocated_mb": (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        ),
        "max_reserved_mb": (
            torch.cuda.max_memory_reserved(device)
            / 1024**2
        ),
    }


def synchronize_device(
    device: str | torch.device,
) -> None:
    device = resolve_device(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        torch.mps.synchronize()


def calculate_effective_batch_size(
    batch_size: int,
    accumulation_steps: int = 1,
    world_size: int = 1,
) -> int:
    if min(
        batch_size,
        accumulation_steps,
        world_size,
    ) < 1:
        raise ValueError(
            "All batch-size factors "
            "must be positive"
        )

    return (
        batch_size
        * accumulation_steps
        * world_size
    )


def calculate_optimizer_steps(
    *,
    num_batches: int,
    accumulation_steps: int = 1,
    flush_last: bool = True,
) -> int:
    if num_batches < 0:
        raise ValueError(
            "num_batches must be nonnegative"
        )

    if accumulation_steps < 1:
        raise ValueError(
            "accumulation_steps must be >= 1"
        )

    if flush_last:
        return math.ceil(
            num_batches / accumulation_steps
        )

    return (
        num_batches // accumulation_steps
    )


class TrainingTimer:
    """
    Track elapsed time, throughput and ETA.
    """

    def __init__(
        self,
        total_steps: int | None = None,
    ) -> None:
        self.total_steps = total_steps
        self.start_time = time.perf_counter()
        self.last_time = self.start_time

        self.completed_steps = 0
        self.processed_samples = 0

    def update(
        self,
        *,
        steps: int = 1,
        samples: int = 0,
    ) -> None:
        self.completed_steps += steps
        self.processed_samples += samples
        self.last_time = time.perf_counter()

    def elapsed(self) -> float:
        return (
            time.perf_counter()
            - self.start_time
        )

    def steps_per_second(self) -> float:
        elapsed = self.elapsed()

        return (
            self.completed_steps / elapsed
            if elapsed > 0
            else 0.0
        )

    def samples_per_second(self) -> float:
        elapsed = self.elapsed()

        return (
            self.processed_samples / elapsed
            if elapsed > 0
            else 0.0
        )

    def estimated_remaining(self) -> float | None:
        if (
            self.total_steps is None
            or self.completed_steps == 0
        ):
            return None

        remaining_steps = max(
            0,
            self.total_steps
            - self.completed_steps,
        )

        rate = self.steps_per_second()

        if rate <= 0:
            return None

        return remaining_steps / rate

    def summary(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed(),
            "completed_steps": (
                self.completed_steps
            ),
            "processed_samples": (
                self.processed_samples
            ),
            "steps_per_second": (
                self.steps_per_second()
            ),
            "samples_per_second": (
                self.samples_per_second()
            ),
            "estimated_remaining_seconds": (
                self.estimated_remaining()
            ),
        }


class TrainingStepManager:
    """
    Centralized optimizer update manager.

    Supports:
        - Gradient accumulation
        - AMP autocast
        - CUDA GradScaler
        - Gradient clipping
        - Non-finite detection
        - Safe optimizer stepping
        - Learning-rate scheduler updates
        - Checkpoint state management

    Example:
        manager = TrainingStepManager(
            optimizer,
            TrainingConfig(
                accumulation_steps=4,
                mixed_precision=True,
            ),
            scheduler=scheduler,
        )

        for batch in dataloader:
            with manager.autocast():
                loss = model_loss(batch)

            result = manager.backward_step(loss)

        manager.flush()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            TrainingConfig
            | Mapping[str, Any]
            | None
        ) = None,
        *,
        scheduler: Any = None,
    ) -> None:
        if config is None:
            config = TrainingConfig()

        elif isinstance(config, Mapping):
            config = TrainingConfig(
                **dict(config)
            )

        self.optimizer = optimizer
        self.config = config
        self.scheduler = scheduler

        self.device = resolve_device(
            config.device
        )

        self.dtype = resolve_dtype(
            config.precision
        )

        self.amp_enabled = (
            config.mixed_precision
            and config.precision != "float32"
            and self.device.type in {
                "cuda",
                "cpu",
            }
        )

        if (
            self.device.type == "cpu"
            and self.dtype == torch.float16
        ):
            self.amp_enabled = False

        self.scaler_enabled = (
            self.amp_enabled
            and self.device.type == "cuda"
            and self.dtype == torch.float16
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.scaler_enabled,
        )

        self.parameters = (
            get_optimizer_parameters(
                optimizer
            )
        )

        self.global_step = 0
        self.optimizer_step = 0
        self.skipped_steps = 0

        self.accumulated_steps = 0
        self.last_gradient_norm: float | None = None

        self.optimizer.zero_grad(
            set_to_none=(
                self.config.zero_grad_set_to_none
            )
        )

    def autocast(self) -> Any:
        if not self.amp_enabled:
            return nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=True,
        )

    def backward(
        self,
        loss: Tensor,
    ) -> None:
        if loss.ndim != 0:
            raise ValueError(
                "Loss must be a scalar tensor"
            )

        if (
            self.config.detect_nonfinite
            and not bool(
                torch.isfinite(
                    loss.detach()
                ).item()
            )
        ):
            raise FloatingPointError(
                "Non-finite training loss"
            )

        scaled_loss = (
            loss
            / self.config.accumulation_steps
        )

        if self.scaler_enabled:
            self.scaler.scale(
                scaled_loss
            ).backward()
        else:
            scaled_loss.backward()

        self.accumulated_steps += 1
        self.global_step += 1

    def _scheduler_step(self) -> None:
        if self.scheduler is None:
            return

        # ReduceLROnPlateau requires a metric
        # and should be stepped after validation.
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            return

        self.scheduler.step()

    def _optimizer_update(
        self,
        *,
        partial: bool = False,
    ) -> bool:
        if self.accumulated_steps == 0:
            return False

        if (
            not partial
            and self.accumulated_steps
            < self.config.accumulation_steps
        ):
            return False

        if self.scaler_enabled:
            self.scaler.unscale_(
                self.optimizer
            )

        # If flushing an incomplete accumulation
        # group, correct the gradient scale.
        if (
            partial
            and self.accumulated_steps
            < self.config.accumulation_steps
        ):
            correction = (
                self.config.accumulation_steps
                / self.accumulated_steps
            )

            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(
                        correction
                    )

        finite_gradients = (
            gradients_are_finite(
                self.parameters
            )
        )

        if (
            not finite_gradients
            and self.config.detect_nonfinite
        ):
            self.skipped_steps += 1

            if self.scaler_enabled:
                # GradScaler.step detects overflow
                # and skips the optimizer update.
                self.scaler.step(
                    self.optimizer
                )
                self.scaler.update()

            self.optimizer.zero_grad(
                set_to_none=(
                    self.config.zero_grad_set_to_none
                )
            )

            self.accumulated_steps = 0

            if self.config.skip_nonfinite_steps:
                return False

            raise FloatingPointError(
                "Non-finite gradients detected"
            )

        if self.config.gradient_clip_norm is not None:
            self.last_gradient_norm = (
                clip_gradients(
                    self.parameters,
                    max_norm=(
                        self.config.gradient_clip_norm
                    ),
                    max_value=(
                        self.config.gradient_clip_value
                    ),
                )
            )

        elif self.config.gradient_clip_value is not None:
            self.last_gradient_norm = (
                clip_gradients(
                    self.parameters,
                    max_norm=None,
                    max_value=(
                        self.config.gradient_clip_value
                    ),
                )
            )

        else:
            self.last_gradient_norm = (
                calculate_gradient_norm(
                    self.parameters
                )
            )

        updated = True

        if self.scaler_enabled:
            old_scale = (
                self.scaler.get_scale()
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            new_scale = (
                self.scaler.get_scale()
            )

            # With standard GradScaler behavior,
            # a reduced scale indicates that
            # the optimizer step was skipped.
            updated = (
                new_scale >= old_scale
            )

        else:
            self.optimizer.step()

        if updated:
            self.optimizer_step += 1
            self._scheduler_step()
        else:
            self.skipped_steps += 1

        self.optimizer.zero_grad(
            set_to_none=(
                self.config.zero_grad_set_to_none
            )
        )

        self.accumulated_steps = 0

        return updated

    def backward_step(
        self,
        loss: Tensor,
    ) -> TrainingStepResult:
        loss_value = float(
            loss.detach().item()
        )

        self.backward(
            loss
        )

        updated = self._optimizer_update(
            partial=False
        )

        return TrainingStepResult(
            loss=loss_value,
            optimizer_updated=updated,
            gradient_norm=(
                self.last_gradient_norm
            ),
            learning_rates=(
                get_learning_rates(
                    self.optimizer
                )
            ),
            global_step=self.global_step,
            optimizer_step=self.optimizer_step,
            skipped_steps=self.skipped_steps,
            accumulation_progress=(
                self.accumulated_steps
            ),
        )

    def flush(self) -> bool:
        """
        Apply remaining accumulated gradients
        at the end of an epoch.
        """
        return self._optimizer_update(
            partial=True
        )

    def validation_step(
        self,
        forward_fn: Callable[[], Tensor],
    ) -> float:
        with torch.no_grad():
            with self.autocast():
                loss = forward_fn()

        return float(
            loss.detach().item()
        )

    def step_plateau_scheduler(
        self,
        metric: float,
    ) -> None:
        if self.scheduler is None:
            return

        if not isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            raise TypeError(
                "Scheduler is not ReduceLROnPlateau"
            )

        self.scheduler.step(
            metric
        )

    def state_dict(self) -> dict[str, Any]:
        if self.accumulated_steps != 0:
            raise RuntimeError(
                "Save TrainingStepManager state "
                "only at an optimizer-step boundary. "
                "Pending gradients are not serialized."
            )

        return {
            "config": asdict(
                self.config
            ),
            "global_step": (
                self.global_step
            ),
            "optimizer_step": (
                self.optimizer_step
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "accumulated_steps": (
                self.accumulated_steps
            ),
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
            "scaler": (
                self.scaler.state_dict()
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if int(
            state.get(
                "accumulated_steps",
                0,
            )
        ) != 0:
            raise ValueError(
                "Cannot resume a checkpoint "
                "with pending gradients"
            )

        self.global_step = int(
            state.get(
                "global_step",
                0,
            )
        )

        self.optimizer_step = int(
            state.get(
                "optimizer_step",
                0,
            )
        )

        self.skipped_steps = int(
            state.get(
                "skipped_steps",
                0,
            )
        )

        self.accumulated_steps = 0

        self.last_gradient_norm = state.get(
            "last_gradient_norm"
        )

        if "scaler" in state:
            self.scaler.load_state_dict(
                state["scaler"]
            )

        self.optimizer.zero_grad(
            set_to_none=(
                self.config.zero_grad_set_to_none
            )
        )

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "device": str(
                self.device
            ),
            "precision": (
                self.config.precision
            ),
            "mixed_precision": (
                self.amp_enabled
            ),
            "gradient_scaler": (
                self.scaler_enabled
            ),
            "accumulation_steps": (
                self.config.accumulation_steps
            ),
            "accumulation_progress": (
                self.accumulated_steps
            ),
            "global_step": (
                self.global_step
            ),
            "optimizer_step": (
                self.optimizer_step
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
            "learning_rates": (
                get_learning_rates(
                    self.optimizer
                )
            ),
        }


def create_training_step_manager(
    optimizer: Optimizer,
    *,
    scheduler: Any = None,
    device: str = "auto",
    mixed_precision: bool = True,
    precision: str = "float16",
    accumulation_steps: int = 1,
    gradient_clip_norm: float | None = 1.0,
) -> TrainingStepManager:
    config = TrainingConfig(
        device=device,
        mixed_precision=mixed_precision,
        precision=precision,
        accumulation_steps=accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
    )

    return TrainingStepManager(
        optimizer,
        config,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    device = resolve_device(
        "auto"
    )

    class DemoWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )

            self.dynamics = nn.Sequential(
                nn.Linear(256 + 32, 256),
                nn.GELU(),
                nn.Linear(256, 128),
            )

        def forward(
            self,
            states: Tensor,
            actions: Tensor,
        ) -> Tensor:
            latent = self.encoder(
                states
            )

            features = torch.cat(
                [latent, actions],
                dim=-1,
            )

            return self.dynamics(
                features
            )

    model = DemoWorldModel().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=25,
            eta_min=1e-6,
        )
    )

    manager = TrainingStepManager(
        optimizer,
        TrainingConfig(
            device=str(device),
            mixed_precision=True,
            precision=(
                "float16"
                if device.type == "cuda"
                else "bfloat16"
            ),
            accumulation_steps=4,
            gradient_clip_norm=1.0,
        ),
        scheduler=scheduler,
    )

    timer = TrainingTimer(
        total_steps=100
    )

    model.train()

    for step in range(100):
        states = torch.randn(
            16,
            128,
            device=device,
        )

        actions = torch.randn(
            16,
            32,
            device=device,
        )

        targets = (
            states
            + 0.05
            * torch.randn_like(states)
        )

        with manager.autocast():
            predictions = model(
                states,
                actions,
            )

            loss = torch.nn.functional.mse_loss(
                predictions.float(),
                targets.float(),
            )

        result = manager.backward_step(
            loss
        )

        timer.update(
            samples=16
        )

        if (
            step == 0
            or (step + 1) % 10 == 0
        ):
            print(
                f"Step {step + 1:03d} | "
                f"Loss: {result.loss:.6f} | "
                f"LR: {result.learning_rates[0]:.8f} | "
                f"Optimizer Steps: "
                f"{result.optimizer_step} | "
                f"Updated: "
                f"{result.optimizer_updated}"
            )

    manager.flush()

    print("\nModel parameters:")
    print(
        count_parameters(model)
    )

    print("\nTraining statistics:")
    print(
        manager.get_statistics()
    )

    print("\nTimer statistics:")
    print(
        timer.summary()
    )

    if device.type == "cuda":
        print("\nGPU memory:")
        print(
            get_gpu_memory_statistics(
                device
            )
        )
