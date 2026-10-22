from torch.optim.lr_scheduler import LinearLR

def build_warmup_scheduler(optimizer,steps=100,start_factor=0.01):
    return LinearLR(optimizer,start_factor=start_factor,total_iters=steps)

"""
NeuroWorld AI
File: training/schedulers/warmup_scheduler.py

Learning-rate warmup schedulers for:
- World models
- Transformers
- Video prediction
- Dynamics models
- Reward models
- Policy networks
- Robotics and Physical AI

Features:
- Linear warmup
- Cosine warmup
- Exponential warmup
- Constant warmup
- Polynomial warmup
- Inverse-square-root scheduling
- Noam Transformer scheduling
- Warmup followed by another scheduler
- Multiple optimizer parameter groups
- Learning-rate monitoring
- Scheduler state serialization
- Checkpoint save/load
- Gradient accumulation step calculation

Requires:
    PyTorch >= 2.1

Usage:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = create_warmup_scheduler(
        optimizer,
        warmup_steps=500,
        warmup_type="linear",
    )

    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

Call scheduler.step() only after a
successful optimizer update.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


WarmupType = Literal[
    "linear",
    "cosine",
    "exponential",
    "constant",
    "polynomial",
    "inverse_sqrt",
    "noam",
]


@dataclass
class WarmupSchedulerConfig:
    warmup_type: WarmupType = "linear"

    warmup_steps: int = 500
    start_factor: float = 0.01

    constant_factor: float = 0.1
    exponential_power: float = 5.0
    polynomial_power: float = 2.0

    min_lr: float = 0.0
    min_lr_ratio: float | None = None

    model_dim: int = 512
    noam_factor: float = 1.0

    total_steps: int | None = None

    def __post_init__(self) -> None:
        if self.warmup_type not in {
            "linear",
            "cosine",
            "exponential",
            "constant",
            "polynomial",
            "inverse_sqrt",
            "noam",
        }:
            raise ValueError(
                "Unsupported warmup type"
            )

        if self.warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be >= 1"
            )

        if not 0 <= self.start_factor <= 1:
            raise ValueError(
                "start_factor must be in [0,1]"
            )

        if not 0 < self.constant_factor <= 1:
            raise ValueError(
                "constant_factor must be in (0,1]"
            )

        if self.exponential_power <= 0:
            raise ValueError(
                "exponential_power must be positive"
            )

        if self.polynomial_power <= 0:
            raise ValueError(
                "polynomial_power must be positive"
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

        if self.model_dim < 1:
            raise ValueError(
                "model_dim must be >= 1"
            )

        if self.noam_factor <= 0:
            raise ValueError(
                "noam_factor must be positive"
            )

        if (
            self.total_steps is not None
            and self.total_steps < 1
        ):
            raise ValueError(
                "total_steps must be >= 1"
            )


def linear_warmup_factor(
    step: int,
    warmup_steps: int,
    start_factor: float = 0.01,
) -> float:
    if warmup_steps < 1:
        raise ValueError(
            "warmup_steps must be >= 1"
        )

    progress = min(
        max(step / warmup_steps, 0.0),
        1.0,
    )

    return (
        start_factor
        + (1.0 - start_factor) * progress
    )


def cosine_warmup_factor(
    step: int,
    warmup_steps: int,
    start_factor: float = 0.01,
) -> float:
    progress = min(
        max(step / warmup_steps, 0.0),
        1.0,
    )

    cosine = (
        1.0 - math.cos(math.pi * progress)
    ) / 2.0

    return (
        start_factor
        + (1.0 - start_factor) * cosine
    )


def exponential_warmup_factor(
    step: int,
    warmup_steps: int,
    start_factor: float = 0.01,
    power: float = 5.0,
) -> float:
    if not 0 < start_factor <= 1:
        raise ValueError(
            "Exponential warmup requires "
            "0 < start_factor <= 1"
        )

    progress = min(
        max(step / warmup_steps, 0.0),
        1.0,
    )

    # Exponential interpolation reaches
    # exactly 1.0 at warmup completion.
    return start_factor ** (
        (1.0 - progress) ** power
    )


def polynomial_warmup_factor(
    step: int,
    warmup_steps: int,
    start_factor: float = 0.01,
    power: float = 2.0,
) -> float:
    progress = min(
        max(step / warmup_steps, 0.0),
        1.0,
    )

    return (
        start_factor
        + (1.0 - start_factor)
        * progress**power
    )


def inverse_sqrt_factor(
    step: int,
    warmup_steps: int,
    start_factor: float = 0.01,
) -> float:
    """
    Linear warmup, then inverse-square-root
    learning-rate decay.
    """
    if step <= warmup_steps:
        return linear_warmup_factor(
            step,
            warmup_steps,
            start_factor,
        )

    return math.sqrt(
        warmup_steps / max(step, 1)
    )


def noam_learning_rate(
    step: int,
    *,
    model_dim: int = 512,
    warmup_steps: int = 4000,
    factor: float = 1.0,
) -> float:
    """
    Original Transformer Noam schedule:

    lr = factor * model_dim^(-0.5) *
         min(step^(-0.5),
             step * warmup_steps^(-1.5))

    Step is clamped to at least 1.
    """
    if model_dim < 1:
        raise ValueError(
            "model_dim must be positive"
        )

    if warmup_steps < 1:
        raise ValueError(
            "warmup_steps must be positive"
        )

    step = max(
        int(step),
        1,
    )

    return (
        factor
        * model_dim**-0.5
        * min(
            step**-0.5,
            step * warmup_steps**-1.5,
        )
    )


class WarmupScheduler(LRScheduler):
    """
    General-purpose learning-rate warmup.

    For finite warmup types:
        LR starts at base_lr * start_factor
        and reaches base_lr at warmup_steps.

    For inverse_sqrt:
        LR warms up and then decays.

    For noam:
        LR follows Transformer Noam scaling.
        The optimizer's initial LR acts as
        a multiplicative group-specific scale.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            WarmupSchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
        *,
        last_epoch: int = -1,
    ) -> None:
        if config is None:
            config = WarmupSchedulerConfig()

        elif isinstance(config, Mapping):
            config = WarmupSchedulerConfig(
                **dict(config)
            )

        self.config = config

        super().__init__(
            optimizer,
            last_epoch=last_epoch,
        )

    def _factor(
        self,
        step: int,
    ) -> float:
        config = self.config

        if config.warmup_type == "linear":
            return linear_warmup_factor(
                step,
                config.warmup_steps,
                config.start_factor,
            )

        if config.warmup_type == "cosine":
            return cosine_warmup_factor(
                step,
                config.warmup_steps,
                config.start_factor,
            )

        if config.warmup_type == "exponential":
            return exponential_warmup_factor(
                step,
                config.warmup_steps,
                max(config.start_factor, 1e-12),
                config.exponential_power,
            )

        if config.warmup_type == "constant":
            if step < config.warmup_steps:
                return config.constant_factor

            return 1.0

        if config.warmup_type == "polynomial":
            return polynomial_warmup_factor(
                step,
                config.warmup_steps,
                config.start_factor,
                config.polynomial_power,
            )

        if config.warmup_type == "inverse_sqrt":
            return inverse_sqrt_factor(
                step,
                config.warmup_steps,
                config.start_factor,
            )

        if config.warmup_type == "noam":
            # Normalize the Noam schedule
            # to 1.0 at warmup completion.
            step = max(step, 1)
            warmup = config.warmup_steps

            return (
                min(
                    step**-0.5,
                    step * warmup**-1.5,
                )
                * math.sqrt(warmup)
                * config.noam_factor
            )

        raise ValueError(
            "Unknown warmup type"
        )

    def get_lr(self) -> list[float]:
        step = max(
            self.last_epoch,
            0,
        )

        factor = self._factor(
            step
        )

        rates = []

        for base_lr in self.base_lrs:
            if self.config.min_lr_ratio is not None:
                minimum = (
                    base_lr
                    * self.config.min_lr_ratio
                )
            else:
                minimum = min(
                    self.config.min_lr,
                    base_lr,
                )

            lr = max(
                minimum,
                base_lr * factor,
            )

            rates.append(lr)

        return rates


class SequentialWarmupScheduler:
    """
    Warmup followed by another LR scheduler.

    Unlike a standalone LRScheduler, this
    wrapper manages two schedulers.

    The downstream scheduler is created
    lazily at the warmup boundary to preserve
    its intended starting learning rates.

    Example:
        scheduler = SequentialWarmupScheduler(
            optimizer,
            warmup_steps=500,
            after_scheduler_factory=lambda opt: (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=9500,
                    eta_min=1e-6,
                )
            ),
        )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        start_factor: float = 0.01,
        warmup_type: str = "linear",
        after_scheduler_factory: Any = None,
    ) -> None:
        if warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be >= 1"
            )

        if after_scheduler_factory is None:
            raise ValueError(
                "after_scheduler_factory is required"
            )

        if not callable(
            after_scheduler_factory
        ):
            raise TypeError(
                "after_scheduler_factory must be callable"
            )

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.after_scheduler_factory = (
            after_scheduler_factory
        )

        self.base_lrs = [
            float(group["lr"])
            for group in optimizer.param_groups
        ]

        self.warmup_scheduler = WarmupScheduler(
            optimizer,
            WarmupSchedulerConfig(
                warmup_type=warmup_type,
                warmup_steps=warmup_steps,
                start_factor=start_factor,
            ),
        )

        self.after_scheduler: (
            LRScheduler | None
        ) = None

        self.completed_steps = 0

    def step(self) -> None:
        if self.completed_steps < self.warmup_steps:
            self.warmup_scheduler.step()
            self.completed_steps += 1

            if self.completed_steps == self.warmup_steps:
                # The warmup scheduler has now
                # restored the base learning rates.
                self.after_scheduler = (
                    self.after_scheduler_factory(
                        self.optimizer
                    )
                )

        else:
            assert self.after_scheduler is not None

            self.after_scheduler.step()
            self.completed_steps += 1

    def get_last_lr(self) -> list[float]:
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_scheduler": (
                self.warmup_scheduler.state_dict()
            ),
            "after_scheduler": (
                self.after_scheduler.state_dict()
                if self.after_scheduler is not None
                else None
            ),
            "completed_steps": (
                self.completed_steps
            ),
            "warmup_steps": (
                self.warmup_steps
            ),
            "base_lrs": list(
                self.base_lrs
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.warmup_scheduler.load_state_dict(
            state["warmup_scheduler"]
        )

        self.completed_steps = int(
            state["completed_steps"]
        )

        self.base_lrs = list(
            state.get(
                "base_lrs",
                self.base_lrs,
            )
        )

        after_state = state.get(
            "after_scheduler"
        )

        if after_state is not None:
            # Construct the downstream scheduler
            # before restoring its state.
            for group, lr in zip(
                self.optimizer.param_groups,
                self.base_lrs,
            ):
                group["lr"] = lr

            self.after_scheduler = (
                self.after_scheduler_factory(
                    self.optimizer
                )
            )

            self.after_scheduler.load_state_dict(
                after_state
            )

            for group, lr in zip(
                self.optimizer.param_groups,
                self.after_scheduler.get_last_lr(),
            ):
                group["lr"] = lr

        else:
            self.after_scheduler = None

            for group, lr in zip(
                self.optimizer.param_groups,
                self.warmup_scheduler.get_last_lr(),
            ):
                group["lr"] = lr


def create_warmup_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int = 500,
    warmup_type: WarmupType = "linear",
    start_factor: float = 0.01,
    min_lr: float = 0.0,
    min_lr_ratio: float | None = None,
    model_dim: int = 512,
    noam_factor: float = 1.0,
    polynomial_power: float = 2.0,
    exponential_power: float = 5.0,
    constant_factor: float = 0.1,
) -> WarmupScheduler:
    config = WarmupSchedulerConfig(
        warmup_type=warmup_type,
        warmup_steps=warmup_steps,
        start_factor=start_factor,
        min_lr=min_lr,
        min_lr_ratio=min_lr_ratio,
        model_dim=model_dim,
        noam_factor=noam_factor,
        polynomial_power=polynomial_power,
        exponential_power=exponential_power,
        constant_factor=constant_factor,
    )

    return WarmupScheduler(
        optimizer,
        config,
    )


def create_noam_scheduler(
    optimizer: Optimizer,
    *,
    model_dim: int = 512,
    warmup_steps: int = 4000,
    factor: float = 1.0,
) -> WarmupScheduler:
    return create_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_type="noam",
        model_dim=model_dim,
        noam_factor=factor,
    )


def create_inverse_sqrt_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int = 4000,
    start_factor: float = 0.01,
    min_lr: float = 0.0,
) -> WarmupScheduler:
    return create_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_type="inverse_sqrt",
        start_factor=start_factor,
        min_lr=min_lr,
    )


def get_scheduler_statistics(
    scheduler: WarmupScheduler,
) -> dict[str, Any]:
    return {
        "scheduler": "WarmupScheduler",
        "warmup_type": (
            scheduler.config.warmup_type
        ),
        "warmup_steps": (
            scheduler.config.warmup_steps
        ),
        "current_step": (
            scheduler.last_epoch
        ),
        "warmup_complete": (
            scheduler.last_epoch
            >= scheduler.config.warmup_steps
        ),
        "learning_rates": [
            float(lr)
            for lr in scheduler.get_last_lr()
        ],
        "base_learning_rates": [
            float(lr)
            for lr in scheduler.base_lrs
        ],
    }


def generate_warmup_schedule(
    *,
    base_lr: float = 3e-4,
    warmup_steps: int = 500,
    total_steps: int = 1000,
    warmup_type: WarmupType = "linear",
    start_factor: float = 0.01,
) -> list[dict[str, float | int]]:
    """
    Generate learning-rate values without
    modifying a real training optimizer.
    """
    if total_steps < 1:
        raise ValueError(
            "total_steps must be >= 1"
        )

    parameter = torch.nn.Parameter(
        torch.zeros(1)
    )

    optimizer = torch.optim.SGD(
        [parameter],
        lr=base_lr,
    )

    scheduler = create_warmup_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_type=warmup_type,
        start_factor=start_factor,
    )

    results = []

    for step in range(total_steps + 1):
        if step > 0:
            optimizer.step()
            scheduler.step()

        results.append(
            {
                "step": step,
                "learning_rate": float(
                    scheduler.get_last_lr()[0]
                ),
            }
        )

    return results


def save_warmup_checkpoint(
    scheduler: WarmupScheduler,
    path: str | Path,
    *,
    optimizer: Optimizer | None = None,
    global_step: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: dict[str, Any] = {
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "config": asdict(scheduler.config),
        "extra": dict(extra or {}),
    }

    if optimizer is not None:
        checkpoint["optimizer"] = (
            optimizer.state_dict()
        )

    torch.save(
        checkpoint,
        path,
    )


def load_warmup_checkpoint(
    scheduler: WarmupScheduler,
    path: str | Path,
    *,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if (
        optimizer is not None
        and "optimizer" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    target_optimizer = (
        optimizer
        if optimizer is not None
        else scheduler.optimizer
    )

    for group, lr in zip(
        target_optimizer.param_groups,
        scheduler.get_last_lr(),
    ):
        group["lr"] = lr

    return {
        "global_step": int(
            checkpoint.get(
                "global_step",
                0,
            )
        ),
        "extra": dict(
            checkpoint.get(
                "extra",
                {},
            )
        ),
    }


class WarmupSchedulerManager:
    """
    Warmup scheduler manager.

    Features:
        - Successful-update tracking
        - AMP-aware step skipping
        - LR monitoring
        - State serialization
        - Checkpointing
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            WarmupSchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = WarmupSchedulerConfig()

        elif isinstance(config, Mapping):
            config = WarmupSchedulerConfig(
                **dict(config)
            )

        self.optimizer = optimizer
        self.config = config

        self.scheduler = WarmupScheduler(
            optimizer,
            config,
        )

        self.completed_steps = 0
        self.skipped_steps = 0

    def step(
        self,
        *,
        optimizer_updated: bool = True,
    ) -> list[float]:
        if not optimizer_updated:
            self.skipped_steps += 1
            return self.get_learning_rates()

        self.scheduler.step()
        self.completed_steps += 1

        return self.get_learning_rates()

    def get_learning_rates(
        self,
    ) -> list[float]:
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def get_progress(self) -> float:
        return min(
            1.0,
            self.completed_steps
            / self.config.warmup_steps,
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
            "skipped_steps": (
                self.skipped_steps
            ),
            "warmup_progress": (
                self.get_progress()
            ),
        }

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "scheduler": (
                self.scheduler.state_dict()
            ),
            "config": asdict(
                self.config
            ),
            "completed_steps": (
                self.completed_steps
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
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

        self.skipped_steps = int(
            state.get(
                "skipped_steps",
                0,
            )
        )

        for group, lr in zip(
            self.optimizer.param_groups,
            self.scheduler.get_last_lr(),
        ):
            group["lr"] = lr

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_optimizer: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint: dict[str, Any] = {
            "manager": self.state_dict(),
            "extra": dict(extra or {}),
        }

        if include_optimizer:
            checkpoint["optimizer"] = (
                self.optimizer.state_dict()
            )

        torch.save(
            checkpoint,
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        if load_optimizer:
            if "optimizer" not in checkpoint:
                raise KeyError(
                    "Checkpoint has no optimizer state"
                )

            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        self.load_state_dict(
            checkpoint["manager"]
        )

        return dict(
            checkpoint.get(
                "extra",
                {},
            )
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

    manager = WarmupSchedulerManager(
        optimizer,
        WarmupSchedulerConfig(
            warmup_type="linear",
            warmup_steps=20,
            start_factor=0.01,
        ),
    )

    model.train()

    for step in range(50):
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

        manager.step(
            optimizer_updated=True
        )

        if (
            step == 0
            or (step + 1) % 5 == 0
        ):
            print(
                f"Step {step + 1:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"LR: "
                f"{manager.get_learning_rates()[0]:.8f} | "
                f"Warmup: "
                f"{manager.get_progress():.0%}"
            )

    print("\nWarmup statistics:")
    print(
        manager.get_statistics()
    )

    print("\nNoam learning rate example:")
    for step in (1, 100, 1000, 4000, 8000):
        print(
            f"Step {step}: "
            f"{noam_learning_rate(step):.8f}"
        )
