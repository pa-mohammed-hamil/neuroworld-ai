
"""
NeuroWorld AI
File: training/distributed/gradient_accumulation.py

Reusable gradient accumulation manager.

Features:
- Gradient accumulation across microbatches
- Correct scaling for incomplete final groups
- FP32, FP16, and BF16 training
- Automatic mixed precision (AMP)
- GradScaler integration
- Gradient clipping
- DDP no_sync() support
- FSDP no_sync() support
- Scheduler stepping
- Non-finite gradient detection
- Optimizer-step tracking
- Gradient norm monitoring
- State saving and restoration
- CPU and CUDA support

Requirements:
    torch >= 2.1
"""

from __future__ import annotations

import contextlib
import math

from dataclasses import dataclass
from typing import Any, Iterator

import torch

from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class GradientAccumulationConfig:
    accumulation_steps: int = 4

    mixed_precision: str = "none"

    max_grad_norm: float | None = 1.0

    use_no_sync: bool = True

    detect_nonfinite: bool = True

    scheduler_step_on_update: bool = True

    zero_grad_set_to_none: bool = True

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be >= 1"
            )

        if self.mixed_precision not in (
            "none",
            "fp16",
            "bf16",
        ):
            raise ValueError(
                "mixed_precision must be "
                "none, fp16, or bf16"
            )

        if (
            self.max_grad_norm is not None
            and self.max_grad_norm <= 0
        ):
            raise ValueError(
                "max_grad_norm must be positive"
            )


@dataclass
class AccumulationStepResult:
    optimizer_stepped: bool

    gradients_finite: bool

    gradient_norm: float | None

    global_step: int

    optimizer_steps: int

    skipped_steps: int

    loss_scale: float


class GradientAccumulator:
    """
    Manage microbatch gradient accumulation.

    Usage:
        accumulator = GradientAccumulator(
            model,
            optimizer,
            config=config,
        )

        for epoch in range(epochs):
            accumulator.start_epoch(
                len(train_loader)
            )

            for batch in train_loader:
                with accumulator.accumulate():
                    with accumulator.autocast():
                        loss = ...

                    accumulator.backward(loss)

                result = accumulator.last_result

    Important:
        The number of microbatches must
        be known at the start of each epoch.

        The optimizer must be created after
        FSDP wrapping.

        All distributed ranks must execute
        the same accumulation schedule.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        config: (
            GradientAccumulationConfig | None
        ) = None,
        scheduler: Any | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = (
            config or GradientAccumulationConfig()
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        if device is None:
            try:
                device = next(
                    model.parameters()
                ).device
            except StopIteration:
                device = torch.device("cpu")

        self.device = torch.device(device)

        self._configure_precision()

        self.global_step = 0
        self.optimizer_steps = 0
        self.skipped_steps = 0

        self.epoch_microbatches = 0
        self.microbatch_index = 0

        self._group_size = 1
        self._should_step = False

        self._epoch_active = False
        self._inside_context = False
        self._backward_called = False

        self.last_result: (
            AccumulationStepResult | None
        ) = None

    def _configure_precision(self) -> None:
        precision = self.config.mixed_precision

        if (
            precision == "fp16"
            and self.device.type != "cuda"
        ):
            raise RuntimeError(
                "FP16 gradient scaling requires "
                "CUDA in this implementation"
            )

        if (
            precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 is unsupported on this GPU"
            )

        if (
            precision == "bf16"
            and self.device.type not in (
                "cpu",
                "cuda",
            )
        ):
            raise RuntimeError(
                "BF16 is supported here only "
                "on CPU or CUDA"
            )

        self.use_amp = (
            precision != "none"
        )

        self.amp_dtype = {
            "none": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                precision == "fp16"
                and self.device.type == "cuda"
            ),
        )

    def start_epoch(
        self,
        num_microbatches: int,
    ) -> None:
        """
        Initialize a new accumulation epoch.

        Must be called before processing
        any microbatches.
        """
        if num_microbatches < 1:
            raise ValueError(
                "num_microbatches must be >= 1"
            )

        if self._inside_context:
            raise RuntimeError(
                "Cannot start an epoch inside "
                "an accumulation context"
            )

        self.epoch_microbatches = (
            num_microbatches
        )

        self.microbatch_index = 0
        self._epoch_active = True

        self.last_result = None

        self.optimizer.zero_grad(
            set_to_none=(
                self.config.zero_grad_set_to_none
            )
        )

    def _current_group_size(self) -> int:
        """
        Return the actual number of
        microbatches in the current group.

        Correctly handles a final group
        smaller than accumulation_steps.
        """
        steps = (
            self.config.accumulation_steps
        )

        group_start = (
            self.microbatch_index // steps
        ) * steps

        remaining = (
            self.epoch_microbatches
            - group_start
        )

        return min(
            steps,
            remaining,
        )

    def _is_update_boundary(self) -> bool:
        next_index = (
            self.microbatch_index + 1
        )

        return (
            next_index
            % self.config.accumulation_steps
            == 0
            or next_index
            == self.epoch_microbatches
        )

    def _synchronization_context(self):
        """
        Skip DDP/FSDP gradient synchronization
        on non-update microbatches.

        FSDP no_sync() can consume additional
        GPU memory because unsharded gradients
        are retained until synchronization.
        """
        if (
            self.config.use_no_sync
            and not self._should_step
            and hasattr(
                self.model,
                "no_sync",
            )
        ):
            return self.model.no_sync()

        return contextlib.nullcontext()

    @contextlib.contextmanager
    def accumulate(
        self,
    ) -> Iterator[None]:
        """
        Accumulation context.

        Example:
            with accumulator.accumulate():
                with accumulator.autocast():
                    loss = model_loss(...)

                accumulator.backward(loss)

        The optimizer is stepped automatically
        when the accumulation boundary is
        reached.
        """
        if not self._epoch_active:
            raise RuntimeError(
                "Call start_epoch() first"
            )

        if self._inside_context:
            raise RuntimeError(
                "Nested accumulation contexts "
                "are not supported"
            )

        if (
            self.microbatch_index
            >= self.epoch_microbatches
        ):
            raise RuntimeError(
                "Epoch already completed"
            )

        self._group_size = (
            self._current_group_size()
        )

        self._should_step = (
            self._is_update_boundary()
        )

        self._inside_context = True
        self._backward_called = False

        try:
            with self._synchronization_context():
                yield

                if not self._backward_called:
                    raise RuntimeError(
                        "backward() must be called "
                        "inside accumulate()"
                    )

            if self._should_step:
                self.last_result = (
                    self._optimizer_step()
                )
            else:
                self.last_result = None

            self.microbatch_index += 1
            self.global_step += 1

            if (
                self.microbatch_index
                == self.epoch_microbatches
            ):
                self._epoch_active = False

        finally:
            self._inside_context = False

    def autocast(self):
        """
        Return an automatic mixed-precision
        context manager.
        """
        if not self.use_amp:
            return contextlib.nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )

    def backward(
        self,
        loss: Tensor,
    ) -> Tensor:
        """
        Scale loss by the actual accumulation
        group size, then run backward().
        """
        if not self._inside_context:
            raise RuntimeError(
                "backward() must be called "
                "inside accumulate()"
            )

        if self._backward_called:
            raise RuntimeError(
                "backward() was already called "
                "for this microbatch"
            )

        if not isinstance(loss, Tensor):
            raise TypeError(
                "Loss must be a Tensor"
            )

        if loss.ndim != 0:
            raise ValueError(
                "Loss must be scalar"
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "Non-finite loss detected"
            )

        scaled_loss = (
            loss / self._group_size
        )

        self.scaler.scale(
            scaled_loss
        ).backward()

        self._backward_called = True

        return scaled_loss.detach()

    def _gradients_finite(self) -> bool:
        """
        Check gradients for NaN and Inf.

        This performs device synchronization
        when extracting the result.
        """
        for parameter in self.model.parameters():
            gradient = parameter.grad

            if gradient is None:
                continue

            if gradient.is_sparse:
                values = gradient.coalesce().values()
            else:
                values = gradient

            if not torch.isfinite(
                values
            ).all().item():
                return False

        return True

    def _compute_grad_norm(
        self,
    ) -> float | None:
        """
        Calculate gradient norm.

        FSDP must use its own distributed
        clip_grad_norm_ implementation.
        """
        max_norm = (
            self.config.max_grad_norm
        )

        if max_norm is None:
            return None

        if hasattr(
            self.model,
            "clip_grad_norm_",
        ):
            norm = self.model.clip_grad_norm_(
                max_norm
            )
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm,
            )

        return float(
            norm.detach().item()
            if isinstance(norm, Tensor)
            else norm
        )

    def _optimizer_step(
        self,
    ) -> AccumulationStepResult:
        """
        Unscale gradients, optionally clip,
        and update the optimizer.

        GradScaler may skip an update if
        overflow is detected.
        """
        if self.scaler.is_enabled():
            self.scaler.unscale_(
                self.optimizer
            )

        gradients_finite = True

        if self.config.detect_nonfinite:
            gradients_finite = (
                self._gradients_finite()
            )

        gradient_norm = None
        optimizer_stepped = False

        if gradients_finite:
            gradient_norm = (
                self._compute_grad_norm()
            )

            if (
                gradient_norm is not None
                and not math.isfinite(
                    gradient_norm
                )
            ):
                gradients_finite = False

        if gradients_finite:
            if self.scaler.is_enabled():
                previous_scale = (
                    self.scaler.get_scale()
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                optimizer_stepped = (
                    self.scaler.get_scale()
                    >= previous_scale
                )
            else:
                self.optimizer.step()
                optimizer_stepped = True

        else:
            if self.scaler.is_enabled():
                # Let GradScaler inspect the
                # non-finite gradients and
                # adjust its scale.
                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

        if optimizer_stepped:
            self.optimizer_steps += 1

            if (
                self.scheduler is not None
                and self.config.scheduler_step_on_update
            ):
                self.scheduler.step()
        else:
            self.skipped_steps += 1

        self.optimizer.zero_grad(
            set_to_none=(
                self.config.zero_grad_set_to_none
            )
        )

        return AccumulationStepResult(
            optimizer_stepped=optimizer_stepped,
            gradients_finite=gradients_finite,
            gradient_norm=gradient_norm,
            global_step=self.global_step + 1,
            optimizer_steps=self.optimizer_steps,
            skipped_steps=self.skipped_steps,
            loss_scale=float(
                self.scaler.get_scale()
            ),
        )

    def state_dict(
        self,
    ) -> dict[str, Any]:
        """
        Save accumulation metadata.

        Only save at optimizer-step boundaries.
        Partial accumulated gradients are
        not included.
        """
        if self._inside_context:
            raise RuntimeError(
                "Cannot save state during "
                "an accumulation context"
            )

        if (
            self._epoch_active
            and self.microbatch_index
            % self.config.accumulation_steps
            != 0
            and self.microbatch_index
            != self.epoch_microbatches
        ):
            raise RuntimeError(
                "Save only at accumulation "
                "boundaries"
            )

        return {
            "global_step": self.global_step,
            "optimizer_steps": (
                self.optimizer_steps
            ),
            "skipped_steps": self.skipped_steps,
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        """
        Restore counters and AMP scaler.

        Resume at an epoch boundary or
        a correctly reconstructed sampler
        position.
        """
        if self._inside_context:
            raise RuntimeError(
                "Cannot load state inside "
                "an accumulation context"
            )

        self.global_step = int(
            state.get(
                "global_step",
                0,
            )
        )

        self.optimizer_steps = int(
            state.get(
                "optimizer_steps",
                0,
            )
        )

        self.skipped_steps = int(
            state.get(
                "skipped_steps",
                0,
            )
        )

        if "scaler" in state:
            self.scaler.load_state_dict(
                state["scaler"]
            )

    def summary(
        self,
    ) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "mixed_precision": (
                self.config.mixed_precision
            ),
            "accumulation_steps": (
                self.config.accumulation_steps
            ),
            "global_microbatches": (
                self.global_step
            ),
            "optimizer_steps": (
                self.optimizer_steps
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "loss_scale": float(
                self.scaler.get_scale()
            ),
        }


def calculate_effective_batch_size(
    microbatch_size: int,
    accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """
    Effective global batch size when all
    microbatches have equal sample counts.
    """
    if min(
        microbatch_size,
        accumulation_steps,
        world_size,
    ) < 1:
        raise ValueError(
            "All batch dimensions must be >= 1"
        )

    return (
        microbatch_size
        * accumulation_steps
        * world_size
    )


if __name__ == "__main__":
    from torch.utils.data import (
        DataLoader,
        TensorDataset,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    dataset = TensorDataset(
        torch.randn(100, 128),
        torch.randn(100, 128),
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
    )

    precision = (
        "bf16"
        if device.type == "cuda"
        and torch.cuda.is_bf16_supported()
        else "none"
    )

    config = GradientAccumulationConfig(
        accumulation_steps=4,
        mixed_precision=precision,
        max_grad_norm=1.0,
    )

    accumulator = GradientAccumulator(
        model,
        optimizer,
        config=config,
        device=device,
    )

    for epoch in range(3):
        model.train()

        accumulator.start_epoch(
            len(loader)
        )

        total_loss = 0.0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with accumulator.accumulate():
                with accumulator.autocast():
                    predictions = model(inputs)

                    loss = (
                        nn.functional.mse_loss(
                            predictions,
                            targets,
                        )
                    )

                accumulator.backward(loss)

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {total_loss / len(loader):.6f}"
        )

    print(
        accumulator.summary()
    )
