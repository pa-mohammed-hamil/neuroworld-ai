
"""
NeuroWorld AI
File: training/schedulers/plateau_scheduler.py

Reduce learning rate when a monitored metric
stops improving.

Features:
- ReduceLROnPlateau scheduling
- Validation loss / accuracy monitoring
- Minimize or maximize metrics
- Patience and improvement thresholds
- Cooldown periods
- Per-parameter-group minimum learning rates
- Early-stopping integration
- NaN/Inf metric handling
- Learning-rate change tracking
- Scheduler checkpointing
- Training statistics
- Multi-model optimizer support

Requires:
    PyTorch >= 2.1

Usage:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = create_plateau_scheduler(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    for epoch in range(epochs):
        train_one_epoch(...)
        validation_loss = validate(...)
        scheduler.step(validation_loss)

Important:
    ReduceLROnPlateau is metric-based.
    Call step(metric) after validation,
    not after every optimizer update.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


PlateauMode = Literal["min", "max"]
ThresholdMode = Literal["rel", "abs"]


@dataclass
class PlateauSchedulerConfig:
    mode: PlateauMode = "min"

    factor: float = 0.5
    patience: int = 5

    threshold: float = 1e-4
    threshold_mode: ThresholdMode = "rel"

    cooldown: int = 0

    min_lr: float = 1e-6
    eps: float = 1e-8

    monitor: str = "val_loss"

    skip_nonfinite_metrics: bool = True
    track_history: bool = True

    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError(
                "mode must be 'min' or 'max'"
            )

        if not 0 < self.factor < 1:
            raise ValueError(
                "factor must be in (0,1)"
            )

        if self.patience < 0:
            raise ValueError(
                "patience must be nonnegative"
            )

        if (
            not math.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise ValueError(
                "threshold must be nonnegative"
            )

        if self.threshold_mode not in {
            "rel",
            "abs",
        }:
            raise ValueError(
                "threshold_mode must be rel or abs"
            )

        if self.cooldown < 0:
            raise ValueError(
                "cooldown must be nonnegative"
            )

        if (
            not math.isfinite(self.min_lr)
            or self.min_lr < 0
        ):
            raise ValueError(
                "min_lr must be nonnegative"
            )

        if (
            not math.isfinite(self.eps)
            or self.eps < 0
        ):
            raise ValueError(
                "eps must be nonnegative"
            )

        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be >= 1"
            )

        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "early_stopping_min_delta must be "
                "nonnegative"
            )


@dataclass
class PlateauStepResult:
    epoch: int
    metric: float

    learning_rates_before: list[float]
    learning_rates_after: list[float]

    learning_rate_reduced: bool
    improvement: bool

    best_metric: float
    bad_epochs: int
    cooldown_counter: int

    should_stop: bool = False
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric_to_float(
    metric: float | int | Tensor,
) -> float:
    if isinstance(metric, Tensor):
        if metric.numel() != 1:
            raise ValueError(
                "Metric tensor must contain one value"
            )

        return float(
            metric.detach().item()
        )

    return float(metric)


def get_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def create_plateau_scheduler(
    optimizer: Optimizer,
    *,
    mode: PlateauMode = "min",
    factor: float = 0.5,
    patience: int = 5,
    threshold: float = 1e-4,
    threshold_mode: ThresholdMode = "rel",
    cooldown: int = 0,
    min_lr: float | Sequence[float] = 1e-6,
    eps: float = 1e-8,
) -> ReduceLROnPlateau:
    """
    Create a PyTorch ReduceLROnPlateau scheduler.

    Example:
        scheduler = create_plateau_scheduler(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        scheduler.step(validation_loss)
    """
    if mode not in {"min", "max"}:
        raise ValueError(
            "Invalid plateau mode"
        )

    if not 0 < factor < 1:
        raise ValueError(
            "factor must be in (0,1)"
        )

    if patience < 0:
        raise ValueError(
            "patience must be nonnegative"
        )

    if cooldown < 0:
        raise ValueError(
            "cooldown must be nonnegative"
        )

    if isinstance(min_lr, (list, tuple)):
        if len(min_lr) != len(
            optimizer.param_groups
        ):
            raise ValueError(
                "min_lr length must match "
                "optimizer parameter groups"
            )

        if any(
            lr < 0 or not math.isfinite(lr)
            for lr in min_lr
        ):
            raise ValueError(
                "All minimum learning rates "
                "must be finite and nonnegative"
            )

    else:
        if min_lr < 0 or not math.isfinite(min_lr):
            raise ValueError(
                "min_lr must be finite "
                "and nonnegative"
            )

    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode=threshold_mode,
        cooldown=cooldown,
        min_lr=min_lr,
        eps=eps,
    )


def create_scheduler_from_config(
    optimizer: Optimizer,
    config: (
        PlateauSchedulerConfig
        | Mapping[str, Any]
    ),
) -> ReduceLROnPlateau:
    if isinstance(config, Mapping):
        config = PlateauSchedulerConfig(
            **dict(config)
        )

    return create_plateau_scheduler(
        optimizer,
        mode=config.mode,
        factor=config.factor,
        patience=config.patience,
        threshold=config.threshold,
        threshold_mode=config.threshold_mode,
        cooldown=config.cooldown,
        min_lr=config.min_lr,
        eps=config.eps,
    )


class PlateauSchedulerManager:
    """
    High-level ReduceLROnPlateau manager.

    Tracks:
        - Validation metrics
        - Best metric
        - Learning-rate reductions
        - Plateau duration
        - Cooldown
        - Optional early stopping
        - Scheduler state

    Example:
        manager = PlateauSchedulerManager(
            optimizer,
            PlateauSchedulerConfig(
                mode="min",
                factor=0.5,
                patience=5,
                monitor="val_loss",
            ),
        )

        result = manager.step(
            {"val_loss": 0.25},
            epoch=10,
        )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: (
            PlateauSchedulerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = PlateauSchedulerConfig()

        elif isinstance(config, Mapping):
            config = PlateauSchedulerConfig(
                **dict(config)
            )

        self.optimizer = optimizer
        self.config = config

        self.scheduler = (
            create_scheduler_from_config(
                optimizer,
                config,
            )
        )

        self.current_epoch = 0
        self.best_metric = (
            float("inf")
            if config.mode == "min"
            else float("-inf")
        )

        self.early_stopping_best = (
            float("inf")
            if config.mode == "min"
            else float("-inf")
        )

        self.early_stopping_bad_epochs = 0

        self.lr_reduction_count = 0
        self.skipped_metric_count = 0

        self.should_stop = False

        self.history: list[
            dict[str, Any]
        ] = []

        self.last_result: (
            PlateauStepResult | None
        ) = None

    def _extract_metric(
        self,
        metrics: (
            float
            | int
            | Tensor
            | Mapping[str, Any]
        ),
    ) -> float:
        if isinstance(metrics, Mapping):
            if self.config.monitor not in metrics:
                raise KeyError(
                    f"Missing monitored metric: "
                    f"{self.config.monitor}"
                )

            return _metric_to_float(
                metrics[self.config.monitor]
            )

        return _metric_to_float(
            metrics
        )

    def _is_improvement(
        self,
        metric: float,
        best: float,
        *,
        min_delta: float = 0.0,
    ) -> bool:
        if self.config.mode == "min":
            return metric < (
                best - min_delta
            )

        return metric > (
            best + min_delta
        )

    def _update_early_stopping(
        self,
        metric: float,
    ) -> None:
        patience = (
            self.config.early_stopping_patience
        )

        if patience is None:
            return

        if self._is_improvement(
            metric,
            self.early_stopping_best,
            min_delta=(
                self.config.early_stopping_min_delta
            ),
        ):
            self.early_stopping_best = metric
            self.early_stopping_bad_epochs = 0

        else:
            self.early_stopping_bad_epochs += 1

        if (
            self.early_stopping_bad_epochs
            >= patience
        ):
            self.should_stop = True

    def step(
        self,
        metrics: (
            float
            | int
            | Tensor
            | Mapping[str, Any]
        ),
        *,
        epoch: int | None = None,
    ) -> PlateauStepResult:
        """
        Process one validation result.

        Args:
            metrics:
                Scalar metric or dictionary.

            epoch:
                Optional epoch label. When omitted,
                epochs are numbered from 1.

        Returns:
            PlateauStepResult
        """
        metric = self._extract_metric(
            metrics
        )

        if epoch is None:
            epoch = self.current_epoch + 1

        if epoch < 0:
            raise ValueError(
                "epoch must be nonnegative"
            )

        if epoch <= self.current_epoch:
            raise ValueError(
                "Epoch must increase monotonically"
            )

        self.current_epoch = epoch

        before = get_learning_rates(
            self.optimizer
        )

        if not math.isfinite(metric):
            self.skipped_metric_count += 1

            if not self.config.skip_nonfinite_metrics:
                raise FloatingPointError(
                    "Non-finite validation metric"
                )

            result = PlateauStepResult(
                epoch=epoch,
                metric=metric,
                learning_rates_before=before,
                learning_rates_after=before.copy(),
                learning_rate_reduced=False,
                improvement=False,
                best_metric=self.best_metric,
                bad_epochs=int(
                    self.scheduler.num_bad_epochs
                ),
                cooldown_counter=int(
                    self.scheduler.cooldown_counter
                ),
                should_stop=self.should_stop,
                skipped=True,
            )

            self._record_result(result)
            return result

        improvement = self._is_improvement(
            metric,
            self.best_metric,
        )

        if improvement:
            self.best_metric = metric

        self.scheduler.step(
            metric
        )

        after = get_learning_rates(
            self.optimizer
        )

        reduced = any(
            new_lr < old_lr - 1e-15
            for old_lr, new_lr in zip(
                before,
                after,
            )
        )

        if reduced:
            self.lr_reduction_count += 1

        self._update_early_stopping(
            metric
        )

        result = PlateauStepResult(
            epoch=epoch,
            metric=metric,
            learning_rates_before=before,
            learning_rates_after=after,
            learning_rate_reduced=reduced,
            improvement=improvement,
            best_metric=self.best_metric,
            bad_epochs=int(
                self.scheduler.num_bad_epochs
            ),
            cooldown_counter=int(
                self.scheduler.cooldown_counter
            ),
            should_stop=self.should_stop,
            skipped=False,
        )

        self._record_result(
            result
        )

        return result

    def _record_result(
        self,
        result: PlateauStepResult,
    ) -> None:
        self.last_result = result

        if self.config.track_history:
            self.history.append(
                result.to_dict()
            )

    def get_learning_rates(
        self,
    ) -> list[float]:
        return get_learning_rates(
            self.optimizer
        )

    def get_best_metric(self) -> float:
        return self.best_metric

    def get_last_result(
        self,
    ) -> dict[str, Any]:
        if self.last_result is None:
            return {}

        return self.last_result.to_dict()

    def get_history(
        self,
    ) -> list[dict[str, Any]]:
        return list(
            self.history
        )

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "scheduler": "ReduceLROnPlateau",
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "current_epoch": (
                self.current_epoch
            ),
            "best_metric": (
                self.best_metric
            ),
            "learning_rates": (
                self.get_learning_rates()
            ),
            "lr_reduction_count": (
                self.lr_reduction_count
            ),
            "skipped_metric_count": (
                self.skipped_metric_count
            ),
            "bad_epochs": int(
                self.scheduler.num_bad_epochs
            ),
            "cooldown_counter": int(
                self.scheduler.cooldown_counter
            ),
            "early_stopping_bad_epochs": (
                self.early_stopping_bad_epochs
            ),
            "should_stop": (
                self.should_stop
            ),
            "history_length": len(
                self.history
            ),
        }

    def reset_early_stopping(self) -> None:
        self.early_stopping_best = (
            float("inf")
            if self.config.mode == "min"
            else float("-inf")
        )

        self.early_stopping_bad_epochs = 0
        self.should_stop = False

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
            "current_epoch": (
                self.current_epoch
            ),
            "best_metric": (
                self.best_metric
            ),
            "early_stopping_best": (
                self.early_stopping_best
            ),
            "early_stopping_bad_epochs": (
                self.early_stopping_bad_epochs
            ),
            "lr_reduction_count": (
                self.lr_reduction_count
            ),
            "skipped_metric_count": (
                self.skipped_metric_count
            ),
            "should_stop": (
                self.should_stop
            ),
            "history": list(
                self.history
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.scheduler.load_state_dict(
            state["scheduler"]
        )

        self.current_epoch = int(
            state.get(
                "current_epoch",
                0,
            )
        )

        self.best_metric = float(
            state.get(
                "best_metric",
                self.scheduler.best,
            )
        )

        self.early_stopping_best = float(
            state.get(
                "early_stopping_best",
                float("inf")
                if self.config.mode == "min"
                else float("-inf"),
            )
        )

        self.early_stopping_bad_epochs = int(
            state.get(
                "early_stopping_bad_epochs",
                0,
            )
        )

        self.lr_reduction_count = int(
            state.get(
                "lr_reduction_count",
                0,
            )
        )

        self.skipped_metric_count = int(
            state.get(
                "skipped_metric_count",
                0,
            )
        )

        self.should_stop = bool(
            state.get(
                "should_stop",
                False,
            )
        )

        self.history = list(
            state.get(
                "history",
                [],
            )
        )

        self.last_result = None

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


def save_scheduler_checkpoint(
    scheduler: ReduceLROnPlateau,
    path: str | Path,
    *,
    optimizer: Optimizer | None = None,
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
    scheduler: ReduceLROnPlateau,
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

    return {
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


if __name__ == "__main__":
    torch.manual_seed(42)

    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    manager = PlateauSchedulerManager(
        optimizer,
        PlateauSchedulerConfig(
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=1,
            min_lr=1e-6,
            monitor="val_loss",
            early_stopping_patience=12,
        ),
    )

    validation_losses = [
        0.90,
        0.75,
        0.62,
        0.55,
        0.54,
        0.54,
        0.55,
        0.54,
        0.56,
        0.55,
        0.54,
        0.53,
        0.53,
        0.54,
        0.55,
        0.56,
        0.57,
        0.58,
        0.59,
        0.60,
    ]

    for epoch, val_loss in enumerate(
        validation_losses,
        start=1,
    ):
        result = manager.step(
            {
                "val_loss": val_loss
            },
            epoch=epoch,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: "
            f"{result.learning_rates_after[0]:.8f} | "
            f"Reduced: "
            f"{result.learning_rate_reduced} | "
            f"Bad Epochs: "
            f"{result.bad_epochs}"
        )

        if result.should_stop:
            print(
                "Early stopping triggered."
            )
            break

    print("\nFinal scheduler statistics:")
    print(
        manager.get_statistics()
    )
