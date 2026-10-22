
"""
NeuroWorld AI
File: training/schedulers/one_cycle_scheduler.py

One-Cycle Learning Rate Scheduler for:
- World-model training
- Video prediction
- Transformer training
- Dynamics models
- Reward models
- Policy optimization
- Robotics and Physical AI

Features:
- One-Cycle learning-rate scheduling
- Cosine and linear annealing
- Two-phase and three-phase schedules
- Learning-rate warmup
- Optional momentum cycling
- Multiple optimizer parameter groups
- Epoch-based configuration
- Step-based configuration
- Scheduler checkpointing
- Schedule preview
- Learning-rate and momentum monitoring
- AMP-aware optimizer update handling

Requires:
    PyTorch >= 2.1

Usage:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = create_one_cycle_scheduler(
        optimizer,
        max_lr=3e-4,
        epochs=50,
        steps_per_epoch=200,
    )

    for epoch in range(50):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = compute_loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

Important:
    Call scheduler.step() once after each
    successful optimizer update.

    If using gradient accumulation, configure
    total_steps to match optimizer updates,
    not raw dataloader batches.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


AnnealStrategy = Literal[
    "cos",
    "linear",
]


@dataclass
class OneCycleSchedulerConfig:
    max_lr: float = 3e-4

    total_steps: int | None = None
    epochs: int | None = 50
    steps_per_epoch: int | None = 200

    pct_start: float = 0.3

    anneal_strategy: AnnealStrategy = "cos"

    cycle_momentum: bool = False
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    div_factor: float = 25.0
    final_div_factor: float = 10000.0

    three_phase: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_lr)
            or self.max_lr <= 0
        ):
            raise ValueError(
                "max_lr must be positive"
            )

        if self.total_steps is not None:
            if self.total_steps < 1:
                raise ValueError(
                    "total_steps must be >= 1"
                )

        else:
            if (
                self.epochs is None
                or self.epochs < 1
            ):
                raise ValueError(
                    "epochs must be >= 1"
                )

            if (
                self.steps_per_epoch is None
                or self.steps_per_epoch < 1
            ):
                raise ValueError(
                    "steps_per_epoch must be >= 1"
                )

        if not 0 < self.pct_start < 1:
            raise ValueError(
                "pct_start must be in (0,1)"
            )

        if self.anneal_strategy not in {
            "cos",
            "linear",
        }:
            raise ValueError(
                "anneal_strategy must be cos or linear"
            )

        if self.div_factor <= 1:
            raise ValueError(
                "div_factor must be > 1"
            )

        if self.final_div_factor <= 1:
            raise ValueError(
                "final_div_factor must be > 1"
            )

        if not (
            0 <= self.base_momentum
            <= self.max_momentum < 1
        ):
            raise ValueError(
                "Momentum values must satisfy "
                "0 <= base_momentum <= "
                "max_momentum < 1"
            )

    def resolved_total_steps(self) -> int:
        if self.total_steps is not None:
            return self.total_steps

        assert self.epochs is not None
        assert self.steps_per_epoch is not None

        return (
            self.epochs
            * self.steps_per_epoch
        )


def calculate_optimizer_steps(
    *,
    epochs: int,
    batches_per_epoch: int,
    accumulation_steps: int = 1,
) -> int:
    """
    Calculate optimizer updates when each
    epoch flushes its remaining gradients.

    Example:
        100 batches, accumulation=4
        -> 25 optimizer steps per epoch.
    """
    if epochs < 1:
        raise ValueError(
            "epochs must be >= 1"
        )

    if batches_per_epoch < 1:
        raise ValueError(
            "batches_per_epoch must be >= 1"
        )

    if accumulation_steps < 1:
        raise ValueError(
            "accumulation_steps must be >= 1"
        )

    steps_per_epoch = math.ceil(
        batches_per_epoch
        / accumulation_steps
    )

    return epochs * steps_per_epoch


def create_one_cycle_scheduler(
    optimizer: Optimizer,
    *,
    max_lr: float | Sequence[float] = 3e-4,
    total_steps: int | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    pct_start: float = 0.3,
    anneal_strategy: AnnealStrategy = "cos",
    cycle_momentum: bool = False,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
    three_phase: bool = False,
    last_epoch: int = -1,
) -> OneCycleLR:
    """
    Create a PyTorch OneCycleLR scheduler.

    If total_steps is supplied, it takes
    precedence over epochs and steps_per_epoch.
    """
    if total_steps is None:
        if (
            epochs is None
            or steps_per_epoch is None
        ):
            raise ValueError(
                "Provide total_steps or both "
                "epochs and steps_per_epoch"
            )

        total_steps = (
            epochs * steps_per_epoch
        )

    if total_steps < 2:
        raise ValueError(
            "OneCycleLR requires at least "
            "2 optimizer steps"
        )

    if isinstance(max_lr, (list, tuple)):
        if len(max_lr) != len(
            optimizer.param_groups
        ):
            raise ValueError(
                "max_lr list length must match "
                "optimizer parameter groups"
            )

        if any(
            not math.isfinite(lr) or lr <= 0
            for lr in max_lr
        ):
            raise ValueError(
                "All maximum learning rates "
                "must be positive"
            )

    else:
        if (
            not math.isfinite(max_lr)
            or max_lr <= 0
        ):
            raise ValueError(
                "max_lr must be positive"
            )

    if not 0 < pct_start < 1:
        raise ValueError(
            "pct_start must be in (0,1)"
        )

    if anneal_strategy not in {
        "cos",
        "linear",
    }:
        raise ValueError(
            "Invalid anneal strategy"
        )

    if div_factor <= 1:
        raise ValueError(
            "div_factor must be > 1"
        )

    if final_div_factor <= 1:
        raise ValueError(
            "final_div_factor must be > 1"
        )

    # PyTorch's OneCycleLR represents phase
    # boundaries with fractional endpoints.
    # Require sufficiently long phases to
    # avoid degenerate interpolation ranges.
    first_phase_steps = (
        pct_start * total_steps
    )

    if first_phase_steps <= 1:
        raise ValueError(
            "Increase pct_start or total_steps: "
            "warmup phase is too short"
        )

    if three_phase:
        if (
            2 * first_phase_steps
            >= total_steps + 1
        ):
            raise ValueError(
                "Three-phase schedule requires "
                "a longer final phase"
            )

    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        base_momentum=base_momentum,
        max_momentum=max_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
        last_epoch=last_epoch,
    )


def create_scheduler_from_config(
    optimizer: Optimizer,
    config: (
        OneCycleSchedulerConfig
        | Mapping[str, Any]
    ),
) -> OneCycleLR:
    if isinstance(config, Mapping):
        config = OneCycleSchedulerConfig(
            **dict(config)
        )

    return create_one_cycle_scheduler(
        optimizer,
        max_lr=config.max_lr,
        total_steps=(
            config.resolved_total_steps()
        ),
        pct_start=config.pct_start,
        anneal_strategy=config.anneal_strategy,
        cycle_momentum=config.cycle_momentum,
        base_momentum=config.base_momentum,
        max_momentum=config.max_momentum,
        div_factor=config.div_factor,
        final_div_factor=config.final_div_factor,
        three_phase=config.three_phase,
    )


def get_learning_rates(
    scheduler: OneCycleLR,
) -> list[float]:
    return [
        float(lr)
        for lr in scheduler.get_last_lr()
    ]


def get_optimizer_momentums(
    optimizer: Optimizer,
) -> list[float | None]:
    momentums: list[float | None] = []

    for group in optimizer.param_groups:
        if "momentum" in group:
            momentums.append(
                float(group["momentum"])
            )

        elif "betas" in group:
            momentums.append(
                float(group["betas"][0])
            )

        else:
            momentums.append(None)

    return momentums


def get_scheduler_statistics(
    scheduler: OneCycleLR,
) -> dict[str, Any]:
    total_steps = int(
        scheduler.total_steps
    )

    completed_steps = max(
        0,
        scheduler.last_epoch,
    )

    return {
        "scheduler": "OneCycleLR",
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "remaining_steps": max(
            0,
            total_steps - completed_steps,
        ),
        "progress": min(
            1.0,
            completed_steps / total_steps,
        ),
        "learning_rates": get_learning_rates(
            scheduler
        ),
        "momentums": get_optimizer_momentums(
            scheduler.optimizer
        ),
        "parameter_groups": len(
            scheduler.optimizer.param_groups
        ),
    }


def generate_one_cycle_schedule(
    *,
    max_lr: float = 3e-4,
    total_steps: int = 1000,
    pct_start: float = 0.3,
    anneal_strategy: AnnealStrategy = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
    three_phase: bool = False,
    cycle_momentum: bool = False,
) -> list[dict[str, float | int]]:
    """
    Generate a standalone LR schedule without
    modifying the real training optimizer.

    Returns:
        [
            {"step": 0, "lr": ...},
            {"step": 1, "lr": ...},
            ...
        ]
    """
    dummy_parameter = torch.nn.Parameter(
        torch.zeros(1)
    )

    dummy_optimizer = torch.optim.SGD(
        [dummy_parameter],
        lr=max_lr,
        momentum=0.9,
    )

    scheduler = create_one_cycle_scheduler(
        dummy_optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
        cycle_momentum=cycle_momentum,
    )

    schedule: list[
        dict[str, float | int]
    ] = []

    # At initialization, OneCycleLR has
    # already set the starting learning rate.
    for step in range(total_steps):
        schedule.append(
            {
                "step": step,
                "lr": float(
                    scheduler.get_last_lr()[0]
                ),
            }
        )

        if step < total_steps - 1:
            dummy_optimizer.step()
            scheduler.step()

    return schedule


def save_scheduler_checkpoint(
    scheduler: OneCycleLR,
    path: str | Path,
    *,
    optimizer: Optimizer | None = None,
    global_step: int = 0,
    epoch: int = 0,
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
        "epoch": int(epoch),
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


def load_scheduler_checkpoint(
    scheduler: OneCycleLR,
    path: str | Path,
    *,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Restore a trusted scheduler checkpoint.

    For exact training continuation, restore
    the optimizer and scheduler together.
    """
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
        "epoch": int(
            checkpoint.get(
                "epoch",
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


class OneCycleSchedulerManager:
    """
    High-level One-Cycle scheduler manager.

    Supports:
        - Step tracking
        - AMP-aware stepping
        - Scheduler statistics
        - State serialization
        - Checkpointing

    Example:
        manager = OneCycleSchedulerManager(
            optimizer,
            OneCycleSchedulerConfig(
                total_steps=10000,
                epochs=None,
                steps_per_epoch=None,
            ),
        )

        optimizer.step()
        manager.step(optimizer_updated=True)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            OneCycleSchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = OneCycleSchedulerConfig()

        elif isinstance(config, Mapping):
            config = OneCycleSchedulerConfig(
                **dict(config)
            )

        self.optimizer = optimizer
        self.config = config

        self.total_steps = (
            config.resolved_total_steps()
        )

        self.scheduler = (
            create_scheduler_from_config(
                optimizer,
                config,
            )
        )

        self.completed_steps = 0
        self.skipped_steps = 0

    def step(
        self,
        *,
        optimizer_updated: bool = True,
    ) -> list[float]:
        """
        Advance the schedule only when the
        optimizer actually updates parameters.

        optimizer_updated=False is useful
        when AMP GradScaler skips a step.
        """
        if not optimizer_updated:
            self.skipped_steps += 1
            return self.get_learning_rates()

        if self.completed_steps >= self.total_steps:
            raise RuntimeError(
                "One-Cycle schedule exhausted. "
                "Increase total_steps for "
                "additional optimizer updates."
            )

        self.scheduler.step()
        self.completed_steps += 1

        return self.get_learning_rates()

    def get_learning_rates(
        self,
    ) -> list[float]:
        return get_learning_rates(
            self.scheduler
        )

    def get_momentums(
        self,
    ) -> list[float | None]:
        return get_optimizer_momentums(
            self.optimizer
        )

    def get_progress(self) -> float:
        return min(
            1.0,
            self.completed_steps
            / self.total_steps,
        )

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "scheduler": "OneCycleLR",
            "total_steps": self.total_steps,
            "completed_steps": (
                self.completed_steps
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "remaining_steps": max(
                0,
                self.total_steps
                - self.completed_steps,
            ),
            "progress": self.get_progress(),
            "learning_rates": (
                self.get_learning_rates()
            ),
            "momentums": (
                self.get_momentums()
            ),
            "anneal_strategy": (
                self.config.anneal_strategy
            ),
            "three_phase": (
                self.config.three_phase
            ),
            "cycle_momentum": (
                self.config.cycle_momentum
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
            "skipped_steps": (
                self.skipped_steps
            ),
            "config": asdict(
                self.config
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
                self.scheduler.last_epoch,
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

    def preview(
        self,
        *,
        max_lr: float | None = None,
    ) -> list[dict[str, float | int]]:
        return generate_one_cycle_schedule(
            max_lr=(
                max_lr
                if max_lr is not None
                else self.config.max_lr
            ),
            total_steps=self.total_steps,
            pct_start=self.config.pct_start,
            anneal_strategy=(
                self.config.anneal_strategy
            ),
            div_factor=self.config.div_factor,
            final_div_factor=(
                self.config.final_div_factor
            ),
            three_phase=self.config.three_phase,
            cycle_momentum=(
                self.config.cycle_momentum
            ),
        )


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    class DemoWorldModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.LayerNorm(256),
                torch.nn.GELU(),
            )

            self.dynamics = torch.nn.Sequential(
                torch.nn.Linear(256 + 32, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
            )

        def forward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
        ) -> torch.Tensor:
            latent = self.encoder(states)

            features = torch.cat(
                [latent, actions],
                dim=-1,
            )

            return self.dynamics(features)

    model = DemoWorldModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    manager = OneCycleSchedulerManager(
        optimizer,
        OneCycleSchedulerConfig(
            max_lr=3e-4,
            total_steps=100,
            epochs=None,
            steps_per_epoch=None,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
        ),
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

        target_states = (
            states
            + 0.05 * torch.randn_like(states)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            states,
            actions,
        )

        loss = torch.nn.functional.mse_loss(
            predictions,
            target_states,
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
            or (step + 1) % 10 == 0
        ):
            print(
                f"Step {step + 1:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"LR: "
                f"{manager.get_learning_rates()[0]:.8f} | "
                f"Momentum: "
                f"{manager.get_momentums()[0]}"
            )

    print("\nScheduler statistics:")
    print(manager.get_statistics())

    print("\nFirst five schedule values:")
    for entry in manager.preview()[:5]:
        print(entry)
