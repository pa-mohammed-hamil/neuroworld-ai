
"""
NeuroWorld AI
File: training/schedulers/cosine_scheduler.py

Cosine learning-rate schedulers for:
- World-model training
- Video prediction
- Transformer and tokenizer training
- Dynamics and reward models
- Reinforcement learning
- Robotics and Physical AI
- Distributed training

Features:
- Cosine annealing
- Linear warmup + cosine decay
- Cosine annealing with warm restarts
- Minimum learning-rate support
- Multiple optimizer parameter groups
- Step-based and epoch-based scheduling
- Scheduler checkpointing
- Learning-rate inspection
- Schedule visualization data
- Scheduler factory

Requires:
    PyTorch >= 2.1

Usage:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = create_cosine_scheduler(
        optimizer,
        total_steps=10000,
        warmup_steps=500,
        min_lr=1e-6,
    )

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

Important:
    Call scheduler.step() after a successful
    optimizer.step(). If AMP skips the optimizer
    update, do not advance the scheduler.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


SchedulerType = Literal[
    "cosine",
    "warmup_cosine",
    "cosine_restarts",
]


@dataclass
class CosineSchedulerConfig:
    scheduler_type: SchedulerType = "warmup_cosine"

    total_steps: int = 10000
    warmup_steps: int = 500

    min_lr: float = 1e-6
    min_lr_ratio: float | None = None

    warmup_start_factor: float = 0.01

    restart_period: int = 1000
    restart_multiplier: float = 2.0
    restart_decay: float = 1.0

    num_cycles: float = 0.5

    step_unit: Literal["step", "epoch"] = "step"

    def __post_init__(self) -> None:
        if self.scheduler_type not in {
            "cosine",
            "warmup_cosine",
            "cosine_restarts",
        }:
            raise ValueError(
                "Unsupported cosine scheduler type"
            )

        if self.total_steps < 1:
            raise ValueError(
                "total_steps must be >= 1"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be >= 0"
            )

        if self.warmup_steps >= self.total_steps:
            if self.scheduler_type == "warmup_cosine":
                raise ValueError(
                    "warmup_steps must be smaller "
                    "than total_steps"
                )

        if (
            not math.isfinite(self.min_lr)
            or self.min_lr < 0
        ):
            raise ValueError(
                "min_lr must be nonnegative"
            )

        if self.min_lr_ratio is not None:
            if not 0 <= self.min_lr_ratio <= 1:
                raise ValueError(
                    "min_lr_ratio must be in [0,1]"
                )

        if not 0 <= self.warmup_start_factor <= 1:
            raise ValueError(
                "warmup_start_factor must be in [0,1]"
            )

        if self.restart_period < 1:
            raise ValueError(
                "restart_period must be >= 1"
            )

        if self.restart_multiplier < 1:
            raise ValueError(
                "restart_multiplier must be >= 1"
            )

        if not 0 < self.restart_decay <= 1:
            raise ValueError(
                "restart_decay must be in (0,1]"
            )

        if (
            not math.isfinite(self.num_cycles)
            or self.num_cycles <= 0
        ):
            raise ValueError(
                "num_cycles must be positive"
            )


def cosine_decay_factor(
    progress: float,
    *,
    num_cycles: float = 0.5,
) -> float:
    """
    Cosine decay factor.

    Args:
        progress:
            Normalized progress in [0,1].

        num_cycles:
            0.5 means half a cosine cycle:
            learning rate decreases from
            maximum to minimum.

    Returns:
        Decay factor in [0,1].
    """
    progress = min(
        max(float(progress), 0.0),
        1.0,
    )

    if progress >= 1.0:
        return 0.0

    value = math.cos(
        math.pi * 2.0 * num_cycles * progress
    )

    return max(
        0.0,
        value,
    )


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    """
    Standard cosine annealing from base_lr
    to min_lr over total_steps.
    """
    if total_steps < 1:
        raise ValueError(
            "total_steps must be >= 1"
        )

    if not 0 <= min_lr <= base_lr:
        raise ValueError(
            "Require 0 <= min_lr <= base_lr"
        )

    progress = min(
        max(step / total_steps, 0.0),
        1.0,
    )

    cosine = (
        1.0 + math.cos(math.pi * progress)
    ) / 2.0

    return (
        min_lr
        + (base_lr - min_lr) * cosine
    )


def warmup_cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
    warmup_start_factor: float = 0.01,
) -> float:
    """
    Linear warmup followed by cosine decay.
    """
    if total_steps < 1:
        raise ValueError(
            "total_steps must be >= 1"
        )

    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "Require 0 <= warmup_steps < total_steps"
        )

    if not 0 <= min_lr <= base_lr:
        raise ValueError(
            "Require 0 <= min_lr <= base_lr"
        )

    step = max(0, int(step))

    if warmup_steps > 0 and step < warmup_steps:
        progress = step / warmup_steps

        factor = (
            warmup_start_factor
            + (
                1.0 - warmup_start_factor
            ) * progress
        )

        return base_lr * factor

    decay_steps = (
        total_steps - warmup_steps
    )

    decay_progress = min(
        max(
            (step - warmup_steps)
            / decay_steps,
            0.0,
        ),
        1.0,
    )

    cosine = (
        1.0
        + math.cos(
            math.pi * decay_progress
        )
    ) / 2.0

    return (
        min_lr
        + (base_lr - min_lr) * cosine
    )


class CosineScheduler(LRScheduler):
    """
    Cosine annealing scheduler.

    Supports:
        - Per-group base learning rates
        - Absolute minimum LR
        - Minimum LR ratio
        - State serialization
        - Step or epoch scheduling

    The scheduler reaches min_lr after
    total_steps successful updates.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        *,
        min_lr: float = 0.0,
        min_lr_ratio: float | None = None,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        if total_steps < 1:
            raise ValueError(
                "total_steps must be >= 1"
            )

        if min_lr < 0:
            raise ValueError(
                "min_lr must be nonnegative"
            )

        if min_lr_ratio is not None:
            if not 0 <= min_lr_ratio <= 1:
                raise ValueError(
                    "min_lr_ratio must be in [0,1]"
                )

        if num_cycles <= 0:
            raise ValueError(
                "num_cycles must be positive"
            )

        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = float(num_cycles)

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
        )

    def _minimum_lr(
        self,
        base_lr: float,
    ) -> float:
        if self.min_lr_ratio is not None:
            return (
                base_lr * self.min_lr_ratio
            )

        return min(
            self.min_lr,
            base_lr,
        )

    def get_lr(self) -> list[float]:
        progress = min(
            max(
                self.last_epoch
                / self.total_steps,
                0.0,
            ),
            1.0,
        )

        factor = cosine_decay_factor(
            progress,
            num_cycles=self.num_cycles,
        )

        rates = []

        for base_lr in self.base_lrs:
            minimum = self._minimum_lr(
                base_lr
            )

            lr = (
                minimum
                + (base_lr - minimum) * factor
            )

            rates.append(lr)

        return rates


class WarmupCosineScheduler(LRScheduler):
    """
    Linear warmup followed by cosine decay.

    Example:
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=10000,
            warmup_steps=500,
            min_lr=1e-6,
        )

    The warmup starts at:
        base_lr * warmup_start_factor

    It reaches base_lr at warmup_steps,
    then decays to min_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        *,
        warmup_steps: int = 500,
        min_lr: float = 0.0,
        min_lr_ratio: float | None = None,
        warmup_start_factor: float = 0.01,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        if total_steps < 1:
            raise ValueError(
                "total_steps must be >= 1"
            )

        if not 0 <= warmup_steps < total_steps:
            raise ValueError(
                "Require 0 <= warmup_steps < total_steps"
            )

        if not 0 <= warmup_start_factor <= 1:
            raise ValueError(
                "warmup_start_factor must be in [0,1]"
            )

        if min_lr < 0:
            raise ValueError(
                "min_lr must be nonnegative"
            )

        if min_lr_ratio is not None:
            if not 0 <= min_lr_ratio <= 1:
                raise ValueError(
                    "min_lr_ratio must be in [0,1]"
                )

        if num_cycles <= 0:
            raise ValueError(
                "num_cycles must be positive"
            )

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)

        self.min_lr = float(min_lr)
        self.min_lr_ratio = min_lr_ratio

        self.warmup_start_factor = float(
            warmup_start_factor
        )

        self.num_cycles = float(num_cycles)

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
        )

    def _minimum_lr(
        self,
        base_lr: float,
    ) -> float:
        if self.min_lr_ratio is not None:
            return (
                base_lr * self.min_lr_ratio
            )

        return min(
            self.min_lr,
            base_lr,
        )

    def get_lr(self) -> list[float]:
        step = max(
            self.last_epoch,
            0,
        )

        rates = []

        for base_lr in self.base_lrs:
            minimum = self._minimum_lr(
                base_lr
            )

            if (
                self.warmup_steps > 0
                and step < self.warmup_steps
            ):
                progress = (
                    step / self.warmup_steps
                )

                factor = (
                    self.warmup_start_factor
                    + (
                        1.0
                        - self.warmup_start_factor
                    ) * progress
                )

                lr = base_lr * factor

            else:
                decay_steps = (
                    self.total_steps
                    - self.warmup_steps
                )

                progress = min(
                    max(
                        (
                            step
                            - self.warmup_steps
                        ) / decay_steps,
                        0.0,
                    ),
                    1.0,
                )

                factor = cosine_decay_factor(
                    progress,
                    num_cycles=self.num_cycles,
                )

                lr = (
                    minimum
                    + (
                        base_lr - minimum
                    ) * factor
                )

            rates.append(lr)

        return rates


class CosineWarmRestartsScheduler(LRScheduler):
    """
    Cosine annealing with warm restarts.

    Features:
        - Configurable initial restart period
        - Period multiplication
        - Maximum LR decay after restarts
        - Per-group minimum LR

    Example:
        scheduler = CosineWarmRestartsScheduler(
            optimizer,
            restart_period=1000,
            restart_multiplier=2.0,
            restart_decay=0.9,
            min_lr=1e-6,
        )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        restart_period: int = 1000,
        restart_multiplier: float = 2.0,
        restart_decay: float = 1.0,
        min_lr: float = 0.0,
        min_lr_ratio: float | None = None,
        last_epoch: int = -1,
    ) -> None:
        if restart_period < 1:
            raise ValueError(
                "restart_period must be >= 1"
            )

        if restart_multiplier < 1:
            raise ValueError(
                "restart_multiplier must be >= 1"
            )

        if not 0 < restart_decay <= 1:
            raise ValueError(
                "restart_decay must be in (0,1]"
            )

        if min_lr < 0:
            raise ValueError(
                "min_lr must be nonnegative"
            )

        if min_lr_ratio is not None:
            if not 0 <= min_lr_ratio <= 1:
                raise ValueError(
                    "min_lr_ratio must be in [0,1]"
                )

        self.restart_period = int(
            restart_period
        )

        self.restart_multiplier = float(
            restart_multiplier
        )

        self.restart_decay = float(
            restart_decay
        )

        self.min_lr = float(min_lr)
        self.min_lr_ratio = min_lr_ratio

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
        )

    def _cycle_info(
        self,
        step: int,
    ) -> tuple[int, int, int]:
        cycle = 0
        period = self.restart_period
        position = max(0, int(step))

        while position >= period:
            position -= period
            cycle += 1

            period = max(
                1,
                int(
                    round(
                        period
                        * self.restart_multiplier
                    )
                ),
            )

        return cycle, position, period

    def get_lr(self) -> list[float]:
        cycle, position, period = (
            self._cycle_info(
                self.last_epoch
            )
        )

        progress = (
            position / period
        )

        cosine = (
            1.0
            + math.cos(
                math.pi * progress
            )
        ) / 2.0

        rates = []

        for base_lr in self.base_lrs:
            minimum = (
                base_lr * self.min_lr_ratio
                if self.min_lr_ratio is not None
                else min(
                    self.min_lr,
                    base_lr,
                )
            )

            peak_lr = (
                base_lr
                * self.restart_decay**cycle
            )

            peak_lr = max(
                minimum,
                peak_lr,
            )

            lr = (
                minimum
                + (
                    peak_lr - minimum
                ) * cosine
            )

            rates.append(lr)

        return rates


def create_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int = 10000,
    warmup_steps: int = 500,
    min_lr: float = 1e-6,
    min_lr_ratio: float | None = None,
    warmup_start_factor: float = 0.01,
    num_cycles: float = 0.5,
) -> WarmupCosineScheduler:
    """
    Default scheduler for NeuroWorld AI.

    Linear warmup + cosine learning-rate decay.
    """
    return WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        min_lr_ratio=min_lr_ratio,
        warmup_start_factor=(
            warmup_start_factor
        ),
        num_cycles=num_cycles,
    )


def create_scheduler(
    optimizer: Optimizer,
    config: (
        CosineSchedulerConfig
        | Mapping[str, Any]
        | None
    ) = None,
) -> LRScheduler:
    """
    General cosine scheduler factory.
    """
    if config is None:
        config = CosineSchedulerConfig()

    elif isinstance(config, Mapping):
        config = CosineSchedulerConfig(
            **dict(config)
        )

    if config.scheduler_type == "cosine":
        return CosineScheduler(
            optimizer,
            total_steps=config.total_steps,
            min_lr=config.min_lr,
            min_lr_ratio=config.min_lr_ratio,
            num_cycles=config.num_cycles,
        )

    if config.scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
            min_lr_ratio=config.min_lr_ratio,
            warmup_start_factor=(
                config.warmup_start_factor
            ),
            num_cycles=config.num_cycles,
        )

    if config.scheduler_type == "cosine_restarts":
        return CosineWarmRestartsScheduler(
            optimizer,
            restart_period=config.restart_period,
            restart_multiplier=(
                config.restart_multiplier
            ),
            restart_decay=config.restart_decay,
            min_lr=config.min_lr,
            min_lr_ratio=config.min_lr_ratio,
        )

    raise ValueError(
        "Unsupported scheduler type"
    )


def get_current_learning_rates(
    scheduler: LRScheduler,
) -> list[float]:
    return [
        float(lr)
        for lr in scheduler.get_last_lr()
    ]


def get_scheduler_statistics(
    scheduler: LRScheduler,
) -> dict[str, Any]:
    return {
        "scheduler": (
            scheduler.__class__.__name__
        ),
        "last_epoch": scheduler.last_epoch,
        "learning_rates": (
            get_current_learning_rates(
                scheduler
            )
        ),
        "base_learning_rates": [
            float(lr)
            for lr in scheduler.base_lrs
        ],
        "parameter_groups": len(
            scheduler.optimizer.param_groups
        ),
    }


def generate_schedule_values(
    scheduler: LRScheduler,
    *,
    steps: int,
) -> list[dict[str, Any]]:
    """
    Generate LR values without modifying
    the supplied scheduler or optimizer.

    The temporary optimizer/scheduler are
    independent of the originals.
    """
    if steps < 1:
        raise ValueError(
            "steps must be >= 1"
        )

    original_lrs = [
        float(lr)
        for lr in scheduler.base_lrs
    ]

    # Use independent dummy parameters.
    dummy_parameters = [
        torch.nn.Parameter(
            torch.zeros(1)
        )
        for _ in original_lrs
    ]

    dummy_optimizer = torch.optim.SGD(
        [
            {
                "params": [parameter],
                "lr": lr,
            }
            for parameter, lr in zip(
                dummy_parameters,
                original_lrs,
            )
        ],
        lr=1.0,
    )

    if isinstance(
        scheduler,
        WarmupCosineScheduler,
    ):
        clone = WarmupCosineScheduler(
            dummy_optimizer,
            total_steps=scheduler.total_steps,
            warmup_steps=scheduler.warmup_steps,
            min_lr=scheduler.min_lr,
            min_lr_ratio=scheduler.min_lr_ratio,
            warmup_start_factor=(
                scheduler.warmup_start_factor
            ),
            num_cycles=scheduler.num_cycles,
        )

    elif isinstance(
        scheduler,
        CosineWarmRestartsScheduler,
    ):
        clone = CosineWarmRestartsScheduler(
            dummy_optimizer,
            restart_period=scheduler.restart_period,
            restart_multiplier=(
                scheduler.restart_multiplier
            ),
            restart_decay=scheduler.restart_decay,
            min_lr=scheduler.min_lr,
            min_lr_ratio=scheduler.min_lr_ratio,
        )

    elif isinstance(
        scheduler,
        CosineScheduler,
    ):
        clone = CosineScheduler(
            dummy_optimizer,
            total_steps=scheduler.total_steps,
            min_lr=scheduler.min_lr,
            min_lr_ratio=scheduler.min_lr_ratio,
            num_cycles=scheduler.num_cycles,
        )

    else:
        raise TypeError(
            "Unsupported scheduler for preview"
        )

    values = []

    for step in range(steps + 1):
        if step > 0:
            dummy_optimizer.step()
            clone.step()

        values.append(
            {
                "step": step,
                "learning_rates": [
                    float(lr)
                    for lr in clone.get_last_lr()
                ],
            }
        )

    return values


def save_scheduler_checkpoint(
    scheduler: LRScheduler,
    path: str | Path,
    *,
    step: int = 0,
    epoch: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "scheduler": scheduler.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "extra": dict(extra or {}),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_scheduler_checkpoint(
    scheduler: LRScheduler,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    # Synchronize the optimizer LR values
    # with the restored scheduler state.
    for group, lr in zip(
        scheduler.optimizer.param_groups,
        scheduler.get_last_lr(),
    ):
        group["lr"] = lr

    return {
        "step": int(
            checkpoint.get("step", 0)
        ),
        "epoch": int(
            checkpoint.get("epoch", 0)
        ),
        "extra": dict(
            checkpoint.get("extra", {})
        ),
    }


class CosineSchedulerManager:
    """
    Wrapper for cosine scheduler lifecycle.

    Example:
        manager = CosineSchedulerManager(
            optimizer,
            CosineSchedulerConfig(
                scheduler_type="warmup_cosine",
                total_steps=10000,
                warmup_steps=500,
            ),
        )

        optimizer.step()
        manager.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            CosineSchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = CosineSchedulerConfig()

        elif isinstance(config, Mapping):
            config = CosineSchedulerConfig(
                **dict(config)
            )

        self.optimizer = optimizer
        self.config = config

        self.scheduler = create_scheduler(
            optimizer,
            config,
        )

        self.completed_steps = 0

    def step(
        self,
        *,
        optimizer_updated: bool = True,
    ) -> list[float]:
        """
        Advance only after a successful
        optimizer update.
        """
        if optimizer_updated:
            self.scheduler.step()
            self.completed_steps += 1

        return self.get_learning_rates()

    def get_learning_rates(
        self,
    ) -> list[float]:
        return get_current_learning_rates(
            self.scheduler
        )

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            **get_scheduler_statistics(
                self.scheduler
            ),
            "completed_steps": (
                self.completed_steps
            ),
            "total_steps": (
                self.config.total_steps
            ),
            "warmup_steps": (
                self.config.warmup_steps
            ),
            "scheduler_type": (
                self.config.scheduler_type
            ),
        }

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "scheduler": (
                self.scheduler.state_dict()
            ),
            "completed_steps": (
                self.completed_steps
            ),
            "config": asdict(self.config),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.scheduler.load_state_dict(
            state["scheduler"]
        )

        self.completed_steps = int(
            state.get(
                "completed_steps",
                0,
            )
        )

        for group, lr in zip(
            self.optimizer.param_groups,
            self.scheduler.get_last_lr(),
        ):
            group["lr"] = lr

    def preview(
        self,
        steps: int | None = None,
    ) -> list[dict[str, Any]]:
        return generate_schedule_values(
            self.scheduler,
            steps=(
                steps
                if steps is not None
                else self.config.total_steps
            ),
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.LayerNorm(256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    manager = CosineSchedulerManager(
        optimizer,
        CosineSchedulerConfig(
            scheduler_type="warmup_cosine",
            total_steps=100,
            warmup_steps=10,
            min_lr=1e-6,
            warmup_start_factor=0.01,
        ),
    )

    model.train()

    for step in range(100):
        inputs = torch.randn(
            16,
            128,
            device=device,
        )

        targets = torch.randn(
            16,
            128,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(inputs)

        loss = torch.nn.functional.mse_loss(
            predictions,
            targets,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        manager.step()

        if (
            step == 0
            or (step + 1) % 10 == 0
        ):
            print(
                f"Step {step + 1:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"LR: "
                f"{manager.get_learning_rates()[0]:.8f}"
            )

    print("\nScheduler statistics:")
    print(manager.get_statistics())

    print("\nSchedule preview:")
    for item in manager.preview(steps=5):
        print(item)
