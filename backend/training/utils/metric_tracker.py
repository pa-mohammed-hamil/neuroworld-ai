
"""
NeuroWorld AI
File: training/utils/metric_tracker.py

Metric tracking and aggregation for PyTorch training.

Features:
- Running averages and weighted averages
- Training and validation metric tracking
- Loss component aggregation
- Min/max and standard deviation
- Exponential moving averages (EMA)
- Sliding-window averages
- Distributed metric synchronization (DDP/FSDP)
- NaN/Inf detection and handling
- Gradient norm monitoring
- Learning-rate tracking
- Epoch summaries
- Metric history
- Best metric monitoring
- Early-stopping support
- JSON/CSV export
- Checkpoint state serialization
- Integration with ExperimentLogger

Dependencies:
    pip install torch

Usage:
    tracker = MetricTracker(
        MetricTrackerConfig(
            window_size=100,
            ema_decay=0.98,
        )
    )

    for batch in dataloader:
        loss = train_step(batch)

        tracker.update(
            "train/loss",
            loss,
            weight=batch_size,
        )

    print(tracker.summary())

    tracker.reset()
"""

from __future__ import annotations

import csv
import json
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class MetricTrackerConfig:
    window_size: int = 100
    ema_decay: float = 0.98

    ignore_nonfinite: bool = True
    raise_on_nonfinite: bool = False

    track_history: bool = True
    max_history: int | None = 10000

    distributed_sync: bool = False

    default_mode: str = "min"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(
                "window_size must be >= 1"
            )

        if not 0 <= self.ema_decay < 1:
            raise ValueError(
                "ema_decay must be in [0,1)"
            )

        if (
            self.max_history is not None
            and self.max_history < 1
        ):
            raise ValueError(
                "max_history must be >= 1"
            )

        if self.default_mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "default_mode must be min or max"
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
                "early_stopping_min_delta must be >= 0"
            )


def metric_to_float(
    value: Any,
) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(
                "Metric tensor must contain "
                "exactly one value"
            )

        value = value.detach().item()

    return float(value)


def is_finite_metric(
    value: Any,
) -> bool:
    return math.isfinite(
        metric_to_float(value)
    )


def _distributed_available() -> bool:
    return (
        dist.is_available()
        and dist.is_initialized()
    )


def _distributed_device() -> torch.device:
    if (
        dist.get_backend() == "nccl"
    ):
        return torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    return torch.device("cpu")


@dataclass
class MetricStatistics:
    name: str
    count: int
    weight: float
    average: float
    latest: float
    minimum: float
    maximum: float
    standard_deviation: float
    ema: float
    window_average: float
    total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunningMetric:
    """
    Streaming metric statistics.

    Supports weighted updates, EMA,
    sliding-window averages and variance.

    Weighted variance is computed using
    a numerically stable online algorithm.

    Example:
        metric = RunningMetric(
            "train/loss",
            window_size=100,
        )

        metric.update(0.5, weight=32)
        metric.update(0.4, weight=32)

        print(metric.average)
    """

    def __init__(
        self,
        name: str,
        *,
        window_size: int = 100,
        ema_decay: float = 0.98,
    ) -> None:
        if window_size < 1:
            raise ValueError(
                "window_size must be >= 1"
            )

        if not 0 <= ema_decay < 1:
            raise ValueError(
                "ema_decay must be in [0,1)"
            )

        self.name = name
        self.window_size = window_size
        self.ema_decay = ema_decay

        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.weight = 0.0
        self.total = 0.0

        self.mean = 0.0
        self.m2 = 0.0

        self.latest = float("nan")
        self.minimum = float("inf")
        self.maximum = float("-inf")

        self.ema = float("nan")

        self.window: deque[
            tuple[float, float]
        ] = deque(
            maxlen=self.window_size
        )

    def update(
        self,
        value: Any,
        weight: float = 1.0,
    ) -> None:
        value = metric_to_float(
            value
        )

        weight = float(weight)

        if not math.isfinite(value):
            raise ValueError(
                "Metric value must be finite"
            )

        if (
            not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(
                "Metric weight must be "
                "finite and positive"
            )

        previous_weight = self.weight
        new_weight = previous_weight + weight

        delta = value - self.mean

        self.mean += (
            weight * delta / new_weight
        )

        delta_after = value - self.mean

        self.m2 += (
            weight * delta * delta_after
        )

        self.weight = new_weight
        self.total += value * weight
        self.count += 1

        self.latest = value

        self.minimum = min(
            self.minimum,
            value,
        )

        self.maximum = max(
            self.maximum,
            value,
        )

        if self.count == 1:
            self.ema = value
        else:
            self.ema = (
                self.ema_decay * self.ema
                + (1.0 - self.ema_decay) * value
            )

        self.window.append(
            (value, weight)
        )

    @property
    def average(self) -> float:
        if self.weight == 0:
            return float("nan")

        return self.mean

    @property
    def variance(self) -> float:
        if self.weight == 0:
            return float("nan")

        return max(
            0.0,
            self.m2 / self.weight,
        )

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(
            self.variance
        )

    @property
    def window_average(self) -> float:
        if not self.window:
            return float("nan")

        total_weight = sum(
            weight
            for _, weight in self.window
        )

        return (
            sum(
                value * weight
                for value, weight in self.window
            )
            / total_weight
        )

    def statistics(self) -> MetricStatistics:
        return MetricStatistics(
            name=self.name,
            count=self.count,
            weight=self.weight,
            average=self.average,
            latest=self.latest,
            minimum=(
                self.minimum
                if self.count
                else float("nan")
            ),
            maximum=(
                self.maximum
                if self.count
                else float("nan")
            ),
            standard_deviation=(
                self.standard_deviation
            ),
            ema=self.ema,
            window_average=(
                self.window_average
            ),
            total=self.total,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "window_size": self.window_size,
            "ema_decay": self.ema_decay,
            "count": self.count,
            "weight": self.weight,
            "total": self.total,
            "mean": self.mean,
            "m2": self.m2,
            "latest": self.latest,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "ema": self.ema,
            "window": list(self.window),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.name = str(
            state["name"]
        )

        self.window_size = int(
            state["window_size"]
        )

        self.ema_decay = float(
            state["ema_decay"]
        )

        self.count = int(
            state["count"]
        )

        self.weight = float(
            state["weight"]
        )

        self.total = float(
            state["total"]
        )

        self.mean = float(
            state["mean"]
        )

        self.m2 = float(
            state["m2"]
        )

        self.latest = float(
            state["latest"]
        )

        self.minimum = float(
            state["minimum"]
        )

        self.maximum = float(
            state["maximum"]
        )

        self.ema = float(
            state["ema"]
        )

        self.window = deque(
            [
                (float(value), float(weight))
                for value, weight in state["window"]
            ],
            maxlen=self.window_size,
        )

    def synchronize(
        self,
    ) -> None:
        """
        Synchronize aggregate statistics
        across distributed workers.

        Global average and variance are exact
        for weighted observations.

        Window statistics and EMA remain
        local to the current process.

        This operation is collective:
        every rank must call it.
        """
        if not _distributed_available():
            return

        device = _distributed_device()

        # Raw second moment:
        # sum(w * x^2) = M2 + W * mean^2
        second_moment = (
            self.m2
            + self.weight * self.mean**2
        )

        values = torch.tensor(
            [
                self.total,
                self.weight,
                second_moment,
                float(self.count),
            ],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            values,
            op=dist.ReduceOp.SUM,
        )

        minimum = torch.tensor(
            self.minimum,
            dtype=torch.float64,
            device=device,
        )

        maximum = torch.tensor(
            self.maximum,
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            minimum,
            op=dist.ReduceOp.MIN,
        )

        dist.all_reduce(
            maximum,
            op=dist.ReduceOp.MAX,
        )

        total = float(values[0].item())
        weight = float(values[1].item())
        raw_second = float(values[2].item())

        self.total = total
        self.weight = weight
        self.count = int(
            values[3].item()
        )

        if weight > 0:
            self.mean = total / weight
            self.m2 = max(
                0.0,
                raw_second
                - weight * self.mean**2,
            )

        else:
            self.mean = 0.0
            self.m2 = 0.0

        self.minimum = float(
            minimum.item()
        )

        self.maximum = float(
            maximum.item()
        )


class BestMetricTracker:
    """
    Track the best metric and optionally
    trigger early stopping.
    """

    def __init__(
        self,
        *,
        mode: str = "min",
        min_delta: float = 0.0,
        patience: int | None = None,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(
                "mode must be min or max"
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be >= 0"
            )

        if (
            patience is not None
            and patience < 1
        ):
            raise ValueError(
                "patience must be >= 1"
            )

        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience

        self.reset()

    def reset(self) -> None:
        self.best = (
            float("inf")
            if self.mode == "min"
            else float("-inf")
        )

        self.best_step: int | None = None
        self.bad_steps = 0
        self.should_stop = False

    def update(
        self,
        value: Any,
        *,
        step: int | None = None,
    ) -> bool:
        value = metric_to_float(
            value
        )

        if not math.isfinite(value):
            return False

        if self.mode == "min":
            improved = (
                value < self.best - self.min_delta
            )
        else:
            improved = (
                value > self.best + self.min_delta
            )

        if improved:
            self.best = value
            self.best_step = step
            self.bad_steps = 0

        else:
            self.bad_steps += 1

        if (
            self.patience is not None
            and self.bad_steps >= self.patience
        ):
            self.should_stop = True

        return improved

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "best": self.best,
            "best_step": self.best_step,
            "bad_steps": self.bad_steps,
            "should_stop": self.should_stop,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.mode = str(
            state["mode"]
        )

        self.min_delta = float(
            state["min_delta"]
        )

        self.patience = state["patience"]

        self.best = float(
            state["best"]
        )

        self.best_step = state[
            "best_step"
        ]

        self.bad_steps = int(
            state["bad_steps"]
        )

        self.should_stop = bool(
            state["should_stop"]
        )


class MetricTracker:
    """
    Unified training metric tracker.

    Supports:
        - Multiple named metrics
        - Epoch-level summaries
        - Weighted averages
        - Sliding-window statistics
        - Exponential moving averages
        - Distributed synchronization
        - Metric history
        - Best metric tracking
        - JSON/CSV export
        - Checkpoint serialization
    """

    def __init__(
        self,
        config: (
            MetricTrackerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = MetricTrackerConfig()

        elif isinstance(config, Mapping):
            config = MetricTrackerConfig(
                **dict(config)
            )

        self.config = config

        self.metrics: dict[
            str, RunningMetric
        ] = {}

        self.best_trackers: dict[
            str, BestMetricTracker
        ] = {}

        self.history: list[
            dict[str, Any]
        ] = []

        self.current_epoch = 0
        self.global_step = 0

        self.nonfinite_count = 0

        self._lock = threading.RLock()

    def _get_metric(
        self,
        name: str,
    ) -> RunningMetric:
        if name not in self.metrics:
            self.metrics[name] = RunningMetric(
                name,
                window_size=(
                    self.config.window_size
                ),
                ema_decay=(
                    self.config.ema_decay
                ),
            )

        return self.metrics[name]

    def update(
        self,
        name: str,
        value: Any,
        *,
        weight: float = 1.0,
        step: int | None = None,
        epoch: int | None = None,
    ) -> bool:
        """
        Update one metric.

        Returns True if the value was recorded.
        """
        numeric = metric_to_float(
            value
        )

        if not math.isfinite(numeric):
            self.nonfinite_count += 1

            if self.config.raise_on_nonfinite:
                raise FloatingPointError(
                    f"Non-finite metric: {name}"
                )

            if self.config.ignore_nonfinite:
                return False

            raise ValueError(
                "Non-finite metric cannot be "
                "added to running statistics"
            )

        with self._lock:
            metric = self._get_metric(
                name
            )

            metric.update(
                numeric,
                weight=weight,
            )

            if step is not None:
                self.global_step = max(
                    self.global_step,
                    step,
                )

            if epoch is not None:
                self.current_epoch = epoch

            if self.config.track_history:
                record = {
                    "name": name,
                    "value": numeric,
                    "weight": float(weight),
                    "step": (
                        step
                        if step is not None
                        else self.global_step
                    ),
                    "epoch": (
                        epoch
                        if epoch is not None
                        else self.current_epoch
                    ),
                }

                self.history.append(
                    record
                )

                if (
                    self.config.max_history
                    is not None
                ):
                    overflow = (
                        len(self.history)
                        - self.config.max_history
                    )

                    if overflow > 0:
                        del self.history[:overflow]

        return True

    def update_many(
        self,
        values: Mapping[str, Any],
        *,
        weight: float = 1.0,
        prefix: str = "",
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        for name, value in values.items():
            metric_name = (
                f"{prefix}/{name}"
                if prefix
                else name
            )

            self.update(
                metric_name,
                value,
                weight=weight,
                step=step,
                epoch=epoch,
            )

    def update_losses(
        self,
        losses: Mapping[str, Any],
        *,
        phase: str = "train",
        weight: float = 1.0,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        self.update_many(
            losses,
            weight=weight,
            prefix=f"{phase}/loss",
            step=step,
            epoch=epoch,
        )

    def update_optimizer(
        self,
        optimizer: Optimizer,
        *,
        step: int | None = None,
    ) -> None:
        for index, group in enumerate(
            optimizer.param_groups
        ):
            self.update(
                f"optimizer/lr_group_{index}",
                float(group["lr"]),
                step=step,
            )

            if "weight_decay" in group:
                self.update(
                    f"optimizer/weight_decay_group_{index}",
                    float(
                        group["weight_decay"]
                    ),
                    step=step,
                )

    def update_gradient_norm(
        self,
        model: nn.Module,
        *,
        name: str = "train/gradient_norm",
        step: int | None = None,
    ) -> float:
        """
        Calculate the global L2 gradient norm.
        """
        squared_norm = 0.0

        for parameter in model.parameters():
            gradient = parameter.grad

            if gradient is None:
                continue

            gradient = gradient.detach()

            if gradient.is_sparse:
                gradient = (
                    gradient.coalesce().values()
                )

            squared_norm += float(
                gradient.float()
                .square()
                .sum()
                .item()
            )

        norm = math.sqrt(
            squared_norm
        )

        self.update(
            name,
            norm,
            step=step,
        )

        return norm

    def get(
        self,
        name: str,
        *,
        statistic: str = "average",
        default: float = float("nan"),
    ) -> float:
        metric = self.metrics.get(
            name
        )

        if metric is None:
            return default

        if not hasattr(
            metric,
            statistic,
        ):
            raise ValueError(
                f"Unknown statistic: {statistic}"
            )

        return float(
            getattr(
                metric,
                statistic,
            )
        )

    def average(
        self,
        name: str,
    ) -> float:
        return self.get(
            name,
            statistic="average",
        )

    def latest(
        self,
        name: str,
    ) -> float:
        return self.get(
            name,
            statistic="latest",
        )

    def ema(
        self,
        name: str,
    ) -> float:
        return self.get(
            name,
            statistic="ema",
        )

    def window_average(
        self,
        name: str,
    ) -> float:
        return self.get(
            name,
            statistic="window_average",
        )

    def summary(
        self,
        *,
        detailed: bool = False,
    ) -> dict[str, Any]:
        if detailed:
            return {
                name: metric.statistics().to_dict()
                for name, metric in self.metrics.items()
            }

        return {
            name: metric.average
            for name, metric in self.metrics.items()
        }

    def get_statistics(
        self,
        name: str,
    ) -> dict[str, Any]:
        if name not in self.metrics:
            raise KeyError(
                f"Unknown metric: {name}"
            )

        return self.metrics[
            name
        ].statistics().to_dict()

    def register_best_metric(
        self,
        name: str,
        *,
        mode: str | None = None,
        min_delta: float | None = None,
        patience: int | None = None,
    ) -> None:
        self.best_trackers[name] = (
            BestMetricTracker(
                mode=(
                    mode
                    or self.config.default_mode
                ),
                min_delta=(
                    min_delta
                    if min_delta is not None
                    else self.config.early_stopping_min_delta
                ),
                patience=(
                    patience
                    if patience is not None
                    else self.config.early_stopping_patience
                ),
            )
        )

    def update_best_metric(
        self,
        name: str,
        value: Any,
        *,
        step: int | None = None,
    ) -> bool:
        if name not in self.best_trackers:
            self.register_best_metric(
                name
            )

        return self.best_trackers[
            name
        ].update(
            value,
            step=step,
        )

    def get_best_metrics(
        self,
    ) -> dict[str, float]:
        return {
            name: tracker.best
            for name, tracker in self.best_trackers.items()
        }

    def should_stop(
        self,
        name: str | None = None,
    ) -> bool:
        if name is not None:
            tracker = self.best_trackers.get(
                name
            )

            return (
                tracker.should_stop
                if tracker is not None
                else False
            )

        return any(
            tracker.should_stop
            for tracker in self.best_trackers.values()
        )

    def synchronize_distributed(self) -> None:
        """
        Synchronize running aggregate statistics.

        All distributed ranks must call this
        function with the same metric names.

        Important:
            Synchronization modifies the
            current aggregates. Do not call
            repeatedly without resetting
            or adding a new aggregation stage.
        """
        if not _distributed_available():
            return

        for name in sorted(self.metrics):
            self.metrics[name].synchronize()

    def reset(
        self,
        names: Sequence[str] | None = None,
        *,
        clear_history: bool = False,
    ) -> None:
        with self._lock:
            if names is None:
                for metric in self.metrics.values():
                    metric.reset()

            else:
                for name in names:
                    if name in self.metrics:
                        self.metrics[name].reset()

            if clear_history:
                self.history.clear()

    def start_epoch(
        self,
        epoch: int,
        *,
        reset_metrics: bool = True,
    ) -> None:
        self.current_epoch = epoch

        if reset_metrics:
            self.reset()

    def end_epoch(
        self,
        *,
        synchronize: bool | None = None,
        reset_metrics: bool = False,
    ) -> dict[str, float]:
        if synchronize is None:
            synchronize = (
                self.config.distributed_sync
            )

        if synchronize:
            self.synchronize_distributed()

        result = self.summary()

        if reset_metrics:
            self.reset()

        return result

    def get_history(
        self,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        if name is None:
            return list(
                self.history
            )

        return [
            record
            for record in self.history
            if record["name"] == name
        ]

    def log_to_experiment(
        self,
        experiment_logger: Any,
        *,
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        """
        Integration with experiment_logger.py.

        The logger must expose:
            log_metrics(metrics, step, epoch, phase)
        """
        experiment_logger.log_metrics(
            self.summary(),
            step=step,
            epoch=epoch,
            phase=phase,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(
                self.config
            ),
            "metrics": {
                name: metric.state_dict()
                for name, metric in self.metrics.items()
            },
            "best_trackers": {
                name: tracker.state_dict()
                for name, tracker in self.best_trackers.items()
            },
            "history": list(
                self.history
            ),
            "current_epoch": (
                self.current_epoch
            ),
            "global_step": (
                self.global_step
            ),
            "nonfinite_count": (
                self.nonfinite_count
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.metrics.clear()

        for name, metric_state in state.get(
            "metrics",
            {},
        ).items():
            metric = RunningMetric(
                name,
                window_size=(
                    metric_state["window_size"]
                ),
                ema_decay=(
                    metric_state["ema_decay"]
                ),
            )

            metric.load_state_dict(
                metric_state
            )

            self.metrics[name] = metric

        self.best_trackers.clear()

        for name, tracker_state in state.get(
            "best_trackers",
            {},
        ).items():
            tracker = BestMetricTracker()

            tracker.load_state_dict(
                tracker_state
            )

            self.best_trackers[name] = tracker

        self.history = list(
            state.get(
                "history",
                [],
            )
        )

        self.current_epoch = int(
            state.get(
                "current_epoch",
                0,
            )
        )

        self.global_step = int(
            state.get(
                "global_step",
                0,
            )
        )

        self.nonfinite_count = int(
            state.get(
                "nonfinite_count",
                0,
            )
        )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> None:
        state = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        self.load_state_dict(
            state
        )

    def export_json(
        self,
        path: str | Path,
        *,
        detailed: bool = True,
    ) -> Path:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "nonfinite_count": (
                self.nonfinite_count
            ),
            "metrics": self.summary(
                detailed=detailed
            ),
            "best_metrics": (
                self.get_best_metrics()
            ),
        }

        def sanitize(
            value: Any,
        ) -> Any:
            if isinstance(value, float):
                if not math.isfinite(value):
                    return None

                return value

            if isinstance(value, dict):
                return {
                    key: sanitize(item)
                    for key, item in value.items()
                }

            if isinstance(value, list):
                return [
                    sanitize(item)
                    for item in value
                ]

            return value

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                sanitize(data),
                file,
                indent=2,
                allow_nan=False,
            )

        return path

    def export_csv(
        self,
        path: str | Path,
    ) -> Path:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "name",
                    "count",
                    "weight",
                    "average",
                    "latest",
                    "minimum",
                    "maximum",
                    "standard_deviation",
                    "ema",
                    "window_average",
                    "total",
                ],
            )

            writer.writeheader()

            for metric in self.metrics.values():
                writer.writerow(
                    metric.statistics().to_dict()
                )

        return path


def create_metric_tracker(
    *,
    window_size: int = 100,
    ema_decay: float = 0.98,
    distributed_sync: bool = False,
    track_history: bool = True,
) -> MetricTracker:
    return MetricTracker(
        MetricTrackerConfig(
            window_size=window_size,
            ema_decay=ema_decay,
            distributed_sync=distributed_sync,
            track_history=track_history,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    tracker = MetricTracker(
        MetricTrackerConfig(
            window_size=20,
            ema_decay=0.95,
            track_history=True,
            early_stopping_patience=5,
        )
    )

    tracker.register_best_metric(
        "val/loss",
        mode="min",
        patience=5,
    )

    global_step = 0

    for epoch in range(1, 6):
        tracker.start_epoch(
            epoch
        )

        model.train()

        for batch_index in range(20):
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

            predictions = model(
                inputs
            )

            loss = torch.nn.functional.mse_loss(
                predictions,
                targets,
            )

            loss.backward()

            gradient_norm = (
                tracker.update_gradient_norm(
                    model,
                    step=global_step,
                )
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            tracker.update(
                "train/loss",
                loss.item(),
                weight=16,
                step=global_step,
                epoch=epoch,
            )

            tracker.update_optimizer(
                optimizer,
                step=global_step,
            )

            global_step += 1

        train_loss = tracker.average(
            "train/loss"
        )

        # Demonstration-only validation value.
        # Replace with a real validation loop.
        val_loss = (
            train_loss * 0.95
        )

        tracker.update(
            "val/loss",
            val_loss,
            epoch=epoch,
            step=global_step,
        )

        improved = tracker.update_best_metric(
            "val/loss",
            val_loss,
            step=epoch,
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Grad Norm: {gradient_norm:.4f} | "
            f"Improved: {improved}"
        )

        print(
            "Epoch summary:",
            tracker.end_epoch(),
        )

        if tracker.should_stop(
            "val/loss"
        ):
            print(
                "Early stopping triggered."
            )
            break

    tracker.export_json(
        "runs/metrics_summary.json"
    )

    tracker.export_csv(
        "runs/metrics_summary.csv"
    )

    tracker.save_checkpoint(
        "runs/metric_tracker.pt"
    )

    print("\nBest metrics:")
    print(
        tracker.get_best_metrics()
    )

    print("\nFinal detailed statistics:")
    print(
        tracker.summary(
            detailed=True
        )
    )
