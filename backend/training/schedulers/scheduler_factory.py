
"""
NeuroWorld AI
File: training/schedulers/scheduler_factory.py

Unified learning-rate scheduler factory.

Supported schedulers:
    - constant
    - step
    - multistep
    - exponential
    - linear
    - cosine
    - warmup_cosine
    - cosine_restarts
    - warmup
    - one_cycle
    - plateau
    - polynomial
    - cyclic

Features:
    - Scheduler creation from Python dictionaries
    - YAML-compatible configuration
    - Step-based and epoch-based scheduling
    - Gradient accumulation awareness
    - Metric-based plateau scheduling
    - Multiple optimizer parameter groups
    - AMP-aware scheduler stepping
    - Learning-rate monitoring
    - Scheduler checkpoint save/load
    - Training progress tracking

Requires:
    PyTorch >= 2.1

Example:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    factory = SchedulerFactory(
        SchedulerConfig(
            name="warmup_cosine",
            total_steps=10000,
            warmup_steps=500,
            min_lr=1e-6,
        )
    )

    scheduler = factory.create(optimizer)

    # After a successful optimizer update:
    scheduler.step()

    # For ReduceLROnPlateau:
    scheduler.step(validation_loss)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

try:
    from .cosine_scheduler import (
        CosineScheduler,
        WarmupCosineScheduler,
        CosineWarmRestartsScheduler,
    )
    from .one_cycle_scheduler import (
        create_one_cycle_scheduler,
    )
    from .plateau_scheduler import (
        create_plateau_scheduler,
    )
except ImportError:
    from cosine_scheduler import (
        CosineScheduler,
        WarmupCosineScheduler,
        CosineWarmRestartsScheduler,
    )
    from one_cycle_scheduler import (
        create_one_cycle_scheduler,
    )
    from plateau_scheduler import (
        create_plateau_scheduler,
    )


SUPPORTED_SCHEDULERS = (
    "constant",
    "step",
    "multistep",
    "exponential",
    "linear",
    "cosine",
    "warmup_cosine",
    "cosine_restarts",
    "warmup",
    "one_cycle",
    "plateau",
    "polynomial",
    "cyclic",
)


SCHEDULER_ALIASES = {
    "cosine_annealing": "cosine",
    "cosineannealing": "cosine",
    "cosine_with_warmup": "warmup_cosine",
    "cosine_warmup": "warmup_cosine",
    "cosine_annealing_warm_restarts": "cosine_restarts",
    "warm_restarts": "cosine_restarts",
    "onecycle": "one_cycle",
    "onecyclelr": "one_cycle",
    "reduce_on_plateau": "plateau",
    "reduce_lr_on_plateau": "plateau",
    "reducelronplateau": "plateau",
    "multi_step": "multistep",
    "constant_lr": "constant",
}


@dataclass
class SchedulerConfig:
    name: str = "warmup_cosine"

    # General scheduling
    total_steps: int | None = 10000
    epochs: int | None = None
    steps_per_epoch: int | None = None
    accumulation_steps: int = 1

    step_unit: str = "step"

    # Cosine scheduling
    warmup_steps: int = 500
    min_lr: float = 1e-6
    min_lr_ratio: float | None = None
    warmup_start_factor: float = 0.01
    num_cycles: float = 0.5

    # Step / multistep
    step_size: int = 10
    gamma: float = 0.1
    milestones: tuple[int, ...] = (30, 60, 90)

    # Linear / constant
    start_factor: float = 1.0
    end_factor: float = 0.01

    # Cosine restarts
    restart_period: int = 1000
    restart_multiplier: float = 2.0
    restart_decay: float = 1.0

    # One-Cycle
    max_lr: float | list[float] = 3e-4
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = False
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    three_phase: bool = False

    # Plateau
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    threshold_mode: str = "rel"
    cooldown: int = 0
    eps: float = 1e-8
    monitor: str = "val_loss"

    # Polynomial
    power: float = 1.0

    # Cyclic
    base_lr: float = 1e-5
    cycle_step_size_up: int = 2000
    cycle_step_size_down: int | None = None
    cyclic_mode: str = "triangular2"

    def __post_init__(self) -> None:
        self.name = normalize_scheduler_name(
            self.name
        )

        if self.name not in SUPPORTED_SCHEDULERS:
            raise ValueError(
                f"Unsupported scheduler: {self.name}. "
                f"Supported: {SUPPORTED_SCHEDULERS}"
            )

        if self.step_unit not in {
            "step",
            "epoch",
        }:
            raise ValueError(
                "step_unit must be step or epoch"
            )

        if (
            self.total_steps is not None
            and self.total_steps < 1
        ):
            raise ValueError(
                "total_steps must be >= 1"
            )

        if self.accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be >= 1"
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be nonnegative"
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

        if self.step_size < 1:
            raise ValueError(
                "step_size must be >= 1"
            )

        if self.gamma <= 0:
            raise ValueError(
                "gamma must be positive"
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

        if self.power <= 0:
            raise ValueError(
                "power must be positive"
            )

        if self.patience < 0:
            raise ValueError(
                "patience must be nonnegative"
            )

        if not 0 < self.factor < 1:
            raise ValueError(
                "factor must be in (0,1)"
            )

        if self.cycle_step_size_up < 1:
            raise ValueError(
                "cycle_step_size_up must be >= 1"
            )

        if (
            self.cycle_step_size_down is not None
            and self.cycle_step_size_down < 1
        ):
            raise ValueError(
                "cycle_step_size_down must be >= 1"
            )


def normalize_scheduler_name(
    name: str,
) -> str:
    name = name.lower().strip().replace(
        "-",
        "_",
    )

    return SCHEDULER_ALIASES.get(
        name,
        name,
    )


def get_supported_schedulers() -> list[str]:
    return list(
        SUPPORTED_SCHEDULERS
    )


def calculate_total_steps(
    *,
    epochs: int,
    batches_per_epoch: int,
    accumulation_steps: int = 1,
    flush_last_batch: bool = True,
) -> int:
    """
    Calculate optimizer updates.

    If flush_last_batch=True, the incomplete
    accumulation group at the end of each
    epoch produces an optimizer update.
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

    if flush_last_batch:
        steps_per_epoch = math.ceil(
            batches_per_epoch
            / accumulation_steps
        )

        return epochs * steps_per_epoch

    return (
        epochs * batches_per_epoch
    ) // accumulation_steps


def resolve_total_steps(
    config: SchedulerConfig,
) -> int:
    """
    Resolve scheduler length.

    Explicit total_steps takes precedence.

    Otherwise, epochs and steps_per_epoch
    are used. steps_per_epoch is interpreted
    as dataloader batches when step_unit='step'.
    """
    if config.total_steps is not None:
        return config.total_steps

    if (
        config.epochs is None
        or config.steps_per_epoch is None
    ):
        raise ValueError(
            "Provide total_steps or both "
            "epochs and steps_per_epoch"
        )

    if config.step_unit == "epoch":
        return config.epochs

    return calculate_total_steps(
        epochs=config.epochs,
        batches_per_epoch=(
            config.steps_per_epoch
        ),
        accumulation_steps=(
            config.accumulation_steps
        ),
    )


def scheduler_config_from_dict(
    values: Mapping[str, Any],
) -> SchedulerConfig:
    """
    Parse a YAML-compatible configuration.

    Supported:
        scheduler:
          name: warmup_cosine
          total_steps: 10000
          warmup_steps: 500
          min_lr: 0.000001
    """
    data = dict(values)

    if (
        "scheduler" in data
        and isinstance(
            data["scheduler"],
            Mapping,
        )
    ):
        data = dict(
            data["scheduler"]
        )

    aliases = {
        "type": "name",
        "scheduler_type": "name",
        "t_max": "total_steps",
        "warmup": "warmup_steps",
        "eta_min": "min_lr",
    }

    for source, destination in aliases.items():
        if source in data:
            data.setdefault(
                destination,
                data[source],
            )

            del data[source]

    valid = {
        field.name
        for field in fields(
            SchedulerConfig
        )
    }

    filtered = {
        key: value
        for key, value in data.items()
        if key in valid
    }

    return SchedulerConfig(
        **filtered
    )


def create_scheduler(
    optimizer: Optimizer,
    config: (
        SchedulerConfig
        | Mapping[str, Any]
        | None
    ) = None,
) -> LRScheduler | ReduceLROnPlateau:
    """
    Unified scheduler factory.
    """
    if config is None:
        config = SchedulerConfig()

    elif isinstance(config, Mapping):
        config = scheduler_config_from_dict(
            config
        )

    if not isinstance(
        config,
        SchedulerConfig,
    ):
        raise TypeError(
            "config must be SchedulerConfig "
            "or mapping"
        )

    name = config.name

    if name == "constant":
        return ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=1,
        )

    if name == "step":
        return StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )

    if name == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=list(
                config.milestones
            ),
            gamma=config.gamma,
        )

    if name == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=config.gamma,
        )

    if name == "linear":
        total_steps = resolve_total_steps(
            config
        )

        return LinearLR(
            optimizer,
            start_factor=config.start_factor,
            end_factor=config.end_factor,
            total_iters=total_steps,
        )

    if name == "cosine":
        return CosineScheduler(
            optimizer,
            total_steps=resolve_total_steps(
                config
            ),
            min_lr=config.min_lr,
            min_lr_ratio=config.min_lr_ratio,
            num_cycles=config.num_cycles,
        )

    if name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            total_steps=resolve_total_steps(
                config
            ),
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
            min_lr_ratio=config.min_lr_ratio,
            warmup_start_factor=(
                config.warmup_start_factor
            ),
            num_cycles=config.num_cycles,
        )

    if name == "cosine_restarts":
        return CosineWarmRestartsScheduler(
            optimizer,
            restart_period=(
                config.restart_period
            ),
            restart_multiplier=(
                config.restart_multiplier
            ),
            restart_decay=(
                config.restart_decay
            ),
            min_lr=config.min_lr,
            min_lr_ratio=config.min_lr_ratio,
        )

    if name == "warmup":
        if config.warmup_steps < 1:
            raise ValueError(
                "warmup_steps must be >= 1"
            )

        def warmup_lambda(
            step: int,
        ) -> float:
            progress = min(
                max(
                    step / config.warmup_steps,
                    0.0,
                ),
                1.0,
            )

            return (
                config.warmup_start_factor
                + (
                    1.0
                    - config.warmup_start_factor
                ) * progress
            )

        return LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda,
        )

    if name == "one_cycle":
        return create_one_cycle_scheduler(
            optimizer,
            max_lr=config.max_lr,
            total_steps=resolve_total_steps(
                config
            ),
            pct_start=config.pct_start,
            anneal_strategy=(
                config.anneal_strategy
            ),
            cycle_momentum=(
                config.cycle_momentum
            ),
            base_momentum=(
                config.base_momentum
            ),
            max_momentum=(
                config.max_momentum
            ),
            div_factor=config.div_factor,
            final_div_factor=(
                config.final_div_factor
            ),
            three_phase=config.three_phase,
        )

    if name == "plateau":
        return create_plateau_scheduler(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=(
                config.threshold_mode
            ),
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps,
        )

    if name == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=resolve_total_steps(
                config
            ),
            power=config.power,
        )

    if name == "cyclic":
        return CyclicLR(
            optimizer,
            base_lr=config.base_lr,
            max_lr=config.max_lr,
            step_size_up=(
                config.cycle_step_size_up
            ),
            step_size_down=(
                config.cycle_step_size_down
            ),
            mode=config.cyclic_mode,
            cycle_momentum=(
                config.cycle_momentum
            ),
        )

    raise ValueError(
        f"Unsupported scheduler: {name}"
    )


def is_metric_scheduler(
    scheduler: Any,
) -> bool:
    return isinstance(
        scheduler,
        ReduceLROnPlateau,
    )


def get_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def get_scheduler_statistics(
    scheduler: LRScheduler | ReduceLROnPlateau,
) -> dict[str, Any]:
    optimizer = scheduler.optimizer

    statistics = {
        "scheduler": (
            scheduler.__class__.__name__
        ),
        "last_epoch": int(
            scheduler.last_epoch
        ),
        "learning_rates": get_learning_rates(
            optimizer
        ),
        "parameter_groups": len(
            optimizer.param_groups
        ),
        "metric_based": is_metric_scheduler(
            scheduler
        ),
    }

    if isinstance(
        scheduler,
        ReduceLROnPlateau,
    ):
        statistics.update(
            {
                "best_metric": float(
                    scheduler.best
                ),
                "bad_epochs": int(
                    scheduler.num_bad_epochs
                ),
                "cooldown_counter": int(
                    scheduler.cooldown_counter
                ),
            }
        )

    return statistics


class SchedulerManager:
    """
    Unified scheduler lifecycle manager.

    Features:
        - Metric-based and regular scheduling
        - AMP-aware step skipping
        - Learning-rate tracking
        - Checkpoint save/load
        - Scheduler progress
        - State serialization

    Example:
        manager = SchedulerManager(
            optimizer,
            SchedulerConfig(
                name="warmup_cosine",
                total_steps=10000,
                warmup_steps=500,
            ),
        )

        optimizer.step()
        manager.step(optimizer_updated=True)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            SchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = SchedulerConfig()

        elif isinstance(config, Mapping):
            config = scheduler_config_from_dict(
                config
            )

        self.optimizer = optimizer
        self.config = config

        self.scheduler = create_scheduler(
            optimizer,
            config,
        )

        self.metric_based = (
            is_metric_scheduler(
                self.scheduler
            )
        )

        self.completed_steps = 0
        self.skipped_steps = 0
        self.last_metric: float | None = None

        self.lr_history: list[
            dict[str, Any]
        ] = []

    def _extract_metric(
        self,
        metric: (
            float
            | torch.Tensor
            | Mapping[str, Any]
            | None
        ),
    ) -> float:
        if metric is None:
            raise ValueError(
                f"Scheduler '{self.config.name}' "
                "requires a validation metric"
            )

        if isinstance(metric, Mapping):
            if self.config.monitor not in metric:
                raise KeyError(
                    f"Missing metric: "
                    f"{self.config.monitor}"
                )

            metric = metric[
                self.config.monitor
            ]

        if isinstance(
            metric,
            torch.Tensor,
        ):
            if metric.numel() != 1:
                raise ValueError(
                    "Metric tensor must be scalar"
                )

            metric = float(
                metric.detach().item()
            )

        metric = float(metric)

        if not math.isfinite(metric):
            raise FloatingPointError(
                "Scheduler metric is not finite"
            )

        return metric

    def step(
        self,
        metric: (
            float
            | torch.Tensor
            | Mapping[str, Any]
            | None
        ) = None,
        *,
        optimizer_updated: bool = True,
    ) -> list[float]:
        """
        Advance scheduler.

        Regular scheduler:
            step(optimizer_updated=True)

        Plateau scheduler:
            step({"val_loss": 0.25})

        optimizer_updated=False:
            skip scheduling, useful for AMP
            overflow or skipped updates.
        """
        if not optimizer_updated:
            self.skipped_steps += 1
            return self.get_learning_rates()

        if self.metric_based:
            metric_value = self._extract_metric(
                metric
            )

            self.scheduler.step(
                metric_value
            )

            self.last_metric = metric_value

        else:
            if (
                self.config.name == "one_cycle"
                and self.completed_steps
                >= resolve_total_steps(
                    self.config
                )
            ):
                raise RuntimeError(
                    "One-Cycle schedule exhausted"
                )

            self.scheduler.step()

        self.completed_steps += 1

        learning_rates = (
            self.get_learning_rates()
        )

        self.lr_history.append(
            {
                "step": self.completed_steps,
                "learning_rates": (
                    learning_rates.copy()
                ),
                "metric": self.last_metric,
            }
        )

        return learning_rates

    def get_learning_rates(
        self,
    ) -> list[float]:
        return get_learning_rates(
            self.optimizer
        )

    def get_progress(
        self,
    ) -> float | None:
        if self.metric_based:
            return None

        if self.config.name in {
            "constant",
            "step",
            "multistep",
            "exponential",
            "cosine_restarts",
            "cyclic",
        }:
            return None

        total_steps = resolve_total_steps(
            self.config
        )

        return min(
            1.0,
            self.completed_steps
            / total_steps,
        )

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            **get_scheduler_statistics(
                self.scheduler
            ),
            "configured_name": (
                self.config.name
            ),
            "completed_steps": (
                self.completed_steps
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "last_metric": (
                self.last_metric
            ),
            "progress": self.get_progress(),
            "history_length": len(
                self.lr_history
            ),
        }

    def get_history(
        self,
    ) -> list[dict[str, Any]]:
        return list(
            self.lr_history
        )

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
            "last_metric": (
                self.last_metric
            ),
            "lr_history": list(
                self.lr_history
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

        self.last_metric = state.get(
            "last_metric"
        )

        self.lr_history = list(
            state.get(
                "lr_history",
                [],
            )
        )

        # Regular schedulers store their
        # last LR values. Synchronize them
        # after restoring scheduler state.
        if not self.metric_based:
            last_lrs = (
                self.scheduler.get_last_lr()
            )

            for group, lr in zip(
                self.optimizer.param_groups,
                last_lrs,
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
            "scheduler_manager": (
                self.state_dict()
            ),
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
            checkpoint["scheduler_manager"]
        )

        return dict(
            checkpoint.get(
                "extra",
                {},
            )
        )


class SchedulerFactory:
    """
    Object-oriented scheduler factory.

    Example:
        factory = SchedulerFactory(
            SchedulerConfig(
                name="warmup_cosine",
                total_steps=10000,
                warmup_steps=500,
            )
        )

        scheduler = factory.create(
            optimizer
        )

        manager = factory.create_manager(
            optimizer
        )
    """

    def __init__(
        self,
        config: (
            SchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = SchedulerConfig()

        elif isinstance(config, Mapping):
            config = scheduler_config_from_dict(
                config
            )

        self.config = config

    def create(
        self,
        optimizer: Optimizer,
    ) -> LRScheduler | ReduceLROnPlateau:
        return create_scheduler(
            optimizer,
            self.config,
        )

    def create_manager(
        self,
        optimizer: Optimizer,
    ) -> SchedulerManager:
        return SchedulerManager(
            optimizer,
            self.config,
        )

    def update_config(
        self,
        **kwargs: Any,
    ) -> None:
        data = asdict(
            self.config
        )

        data.update(
            kwargs
        )

        self.config = SchedulerConfig(
            **data
        )

    def get_config(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self.config
        )

    @staticmethod
    def available() -> list[str]:
        return get_supported_schedulers()


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

    factory = SchedulerFactory(
        SchedulerConfig(
            name="warmup_cosine",
            total_steps=100,
            warmup_steps=10,
            min_lr=1e-6,
            warmup_start_factor=0.01,
        )
    )

    manager = factory.create_manager(
        optimizer
    )

    print(
        "Supported schedulers:",
        factory.available(),
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
            or (step + 1) % 10 == 0
        ):
            lr = (
                manager.get_learning_rates()[0]
            )

            print(
                f"Step {step + 1:03d} | "
                f"Loss: {loss.item():.6f} | "
                f"LR: {lr:.8f}"
            )

    print("\nScheduler statistics:")
    print(
        manager.get_statistics()
    )

    print("\nScheduler configuration:")
    print(
        factory.get_config()
    )
