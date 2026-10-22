
"""
NeuroWorld AI
File: training/callbacks/logging_callback.py

Training logging callback for PyTorch.

Features:
- Batch and epoch logging
- Training and validation metrics
- JSONL and CSV metric exports
- TensorBoard integration
- CPU, RAM and GPU memory monitoring
- Learning-rate tracking
- Gradient norm tracking
- Training duration and throughput
- Best-metric tracking
- Checkpoint and early-stopping event logs
- Exception logging
- Thread-safe file writing
- Distributed rank-zero logging
- Resume-compatible state

Dependencies:
    torch

Optional:
    tensorboard
    psutil

Python 3.11+
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import threading
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.optim import Optimizer


try:
    import psutil
except ImportError:
    psutil = None


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def to_float(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None

        value = value.detach().item()

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return (
        result
        if math.isfinite(result)
        else None
    )


@dataclass
class LoggingConfig:
    log_dir: str = "logs/training"

    experiment_name: str = "neuroworld"

    log_every_n_steps: int = 10

    log_every_n_epochs: int = 1

    console_logging: bool = True

    jsonl_logging: bool = True

    csv_logging: bool = True

    tensorboard_logging: bool = True

    log_learning_rate: bool = True

    log_gradient_norm: bool = False

    log_system_metrics: bool = True

    log_gpu_memory: bool = True

    rank_zero_only: bool = True

    monitor: str = "val_loss"

    mode: Literal["min", "max"] = "min"

    append_existing: bool = True

    flush_every_write: bool = True

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError(
                "experiment_name is required"
            )

        if self.log_every_n_steps < 1:
            raise ValueError(
                "log_every_n_steps must be >= 1"
            )

        if self.log_every_n_epochs < 1:
            raise ValueError(
                "log_every_n_epochs must be >= 1"
            )

        if self.mode not in {"min", "max"}:
            raise ValueError(
                "mode must be 'min' or 'max'"
            )


class LoggingCallback:
    STATE_VERSION = 1

    def __init__(
        self,
        config: LoggingConfig | None = None,
    ) -> None:
        self.config = config or LoggingConfig()

        self.log_dir = Path(
            self.config.log_dir
        ) / self.config.experiment_name

        self.enabled = self._is_main_process()

        self.lock = threading.RLock()

        self.global_step = 0
        self.current_epoch = 0

        self.best_score: float | None = None
        self.best_epoch: int | None = None

        self.training_started_at: float | None = None
        self.epoch_started_at: float | None = None

        self.total_samples = 0
        self.epoch_samples = 0

        self.history: list[dict[str, Any]] = []

        self.writer = None

        if self.enabled:
            self.log_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if (
                self.config.tensorboard_logging
                and SummaryWriter is not None
            ):
                self.writer = SummaryWriter(
                    log_dir=str(
                        self.log_dir / "tensorboard"
                    )
                )

            elif self.config.tensorboard_logging:
                logger.warning(
                    "TensorBoard unavailable. "
                    "Install tensorboard to enable it."
                )

        self.jsonl_path = (
            self.log_dir / "metrics.jsonl"
        )

        self.csv_path = (
            self.log_dir / "epoch_metrics.csv"
        )

        self.event_path = (
            self.log_dir / "events.jsonl"
        )

    @staticmethod
    def _is_main_process() -> bool:
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return (
                torch.distributed.get_rank() == 0
            )

        return True

    def _should_log(self) -> bool:
        if not self.config.rank_zero_only:
            return True

        return self.enabled

    @staticmethod
    def _normalize_metrics(
        metrics: Mapping[str, Any],
    ) -> dict[str, float]:
        normalized = {}

        for key, value in metrics.items():
            numeric = to_float(value)

            if numeric is not None:
                normalized[str(key)] = numeric

        return normalized

    def _write_jsonl(
        self,
        path: Path,
        record: Mapping[str, Any],
    ) -> None:
        if not self._should_log():
            return

        with self.lock:
            with path.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

                if self.config.flush_every_write:
                    file.flush()

    def _write_csv(
        self,
        record: Mapping[str, Any],
    ) -> None:
        if not self._should_log():
            return

        # Fixed CSV schema. Additional metrics
        # remain available in JSONL.
        fields = [
            "timestamp",
            "epoch",
            "global_step",
            "train_loss",
            "val_loss",
            "learning_rate",
            "epoch_duration_seconds",
            "samples_per_second",
            "best_score",
        ]

        row = {
            field: record.get(field)
            for field in fields
        }

        with self.lock:
            write_header = (
                not self.csv_path.exists()
                or self.csv_path.stat().st_size == 0
            )

            with self.csv_path.open(
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fields,
                )

                if write_header:
                    writer.writeheader()

                writer.writerow(row)

    @staticmethod
    def get_learning_rates(
        optimizer: Optimizer | None,
    ) -> dict[str, float]:
        if optimizer is None:
            return {}

        return {
            f"learning_rate/group_{index}": float(
                group["lr"]
            )
            for index, group in enumerate(
                optimizer.param_groups
            )
        }

    @staticmethod
    def get_gradient_norm(
        model: nn.Module,
    ) -> float:
        total_squared = 0.0

        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach()

            norm = torch.linalg.vector_norm(
                grad.float()
            )

            total_squared += float(
                norm.item()
            ) ** 2

        return math.sqrt(
            total_squared
        )

    def get_system_metrics(
        self,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if not self.config.log_system_metrics:
            return metrics

        if psutil is not None:
            process = psutil.Process(
                os.getpid()
            )

            metrics["system/process_ram_mb"] = (
                process.memory_info().rss
                / (1024 ** 2)
            )

            metrics["system/ram_percent"] = (
                psutil.virtual_memory().percent
            )

            metrics["system/cpu_percent"] = (
                psutil.cpu_percent(
                    interval=None
                )
            )

        if (
            self.config.log_gpu_memory
            and torch.cuda.is_available()
        ):
            device_index = (
                torch.cuda.current_device()
            )

            metrics["system/gpu_allocated_mb"] = (
                torch.cuda.memory_allocated(
                    device_index
                )
                / (1024 ** 2)
            )

            metrics["system/gpu_reserved_mb"] = (
                torch.cuda.memory_reserved(
                    device_index
                )
                / (1024 ** 2)
            )

            metrics["system/gpu_peak_mb"] = (
                torch.cuda.max_memory_allocated(
                    device_index
                )
                / (1024 ** 2)
            )

        return metrics

    def _log_scalars(
        self,
        prefix: str,
        metrics: Mapping[str, float],
        step: int,
    ) -> None:
        if self.writer is None:
            return

        for name, value in metrics.items():
            self.writer.add_scalar(
                f"{prefix}/{name}",
                value,
                step,
            )

    def log_event(
        self,
        event_type: str,
        *,
        message: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": utc_now(),
            "event_type": event_type,
            "message": message,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "details": dict(
                details or {}
            ),
        }

        if self._should_log():
            self._write_jsonl(
                self.event_path,
                record,
            )

            if self.config.console_logging:
                logger.info(
                    "[%s] %s",
                    event_type,
                    message,
                )

    def on_train_begin(
        self,
        *,
        start_epoch: int = 0,
        global_step: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.current_epoch = start_epoch
        self.global_step = global_step

        self.training_started_at = (
            time.perf_counter()
        )

        self.log_event(
            "training_started",
            message="Training run started",
            details={
                "start_epoch": start_epoch,
                "global_step": global_step,
                "metadata": dict(
                    metadata or {}
                ),
            },
        )

    def on_epoch_begin(
        self,
        epoch: int,
    ) -> None:
        self.current_epoch = epoch

        self.epoch_started_at = (
            time.perf_counter()
        )

        self.epoch_samples = 0

    def on_batch_end(
        self,
        *,
        batch_index: int,
        metrics: Mapping[str, Any],
        batch_size: int,
        optimizer: Optimizer | None = None,
        model: nn.Module | None = None,
        global_step: int | None = None,
    ) -> None:
        if batch_size < 0:
            raise ValueError(
                "batch_size must be nonnegative"
            )

        if global_step is None:
            self.global_step += 1
        else:
            self.global_step = global_step

        self.total_samples += batch_size
        self.epoch_samples += batch_size

        if (
            self.global_step
            % self.config.log_every_n_steps
            != 0
        ):
            return

        values = self._normalize_metrics(
            metrics
        )

        if self.config.log_learning_rate:
            values.update(
                self.get_learning_rates(
                    optimizer
                )
            )

        if (
            self.config.log_gradient_norm
            and model is not None
        ):
            values["gradient_norm"] = (
                self.get_gradient_norm(
                    model
                )
            )

        values.update(
            self.get_system_metrics()
        )

        record = {
            "timestamp": utc_now(),
            "type": "batch",
            "epoch": self.current_epoch,
            "batch_index": batch_index,
            "global_step": self.global_step,
            "batch_size": batch_size,
            "metrics": values,
        }

        if self.config.jsonl_logging:
            self._write_jsonl(
                self.jsonl_path,
                record,
            )

        if self._should_log():
            self._log_scalars(
                "batch",
                values,
                self.global_step,
            )

            if self.config.console_logging:
                logger.info(
                    "Epoch %d | Batch %d | "
                    "Step %d | Metrics: %s",
                    self.current_epoch + 1,
                    batch_index + 1,
                    self.global_step,
                    values,
                )

    def _update_best(
        self,
        metrics: Mapping[str, float],
        epoch: int,
    ) -> bool:
        score = metrics.get(
            self.config.monitor
        )

        if score is None:
            return False

        if self.best_score is None:
            improved = True

        elif self.config.mode == "min":
            improved = (
                score < self.best_score
            )

        else:
            improved = (
                score > self.best_score
            )

        if improved:
            self.best_score = score
            self.best_epoch = epoch

        return improved

    def on_epoch_end(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, Any],
        optimizer: Optimizer | None = None,
    ) -> dict[str, Any]:
        self.current_epoch = epoch

        values = self._normalize_metrics(
            metrics
        )

        if self.config.log_learning_rate:
            values.update(
                self.get_learning_rates(
                    optimizer
                )
            )

        duration = 0.0

        if self.epoch_started_at is not None:
            duration = (
                time.perf_counter()
                - self.epoch_started_at
            )

        throughput = (
            self.epoch_samples / duration
            if duration > 0
            else 0.0
        )

        improved = self._update_best(
            values,
            epoch,
        )

        record = {
            "timestamp": utc_now(),
            "type": "epoch",
            "epoch": epoch,
            "global_step": self.global_step,
            "train_loss": values.get(
                "train_loss"
            ),
            "val_loss": values.get(
                "val_loss"
            ),
            "learning_rate": (
                next(
                    (
                        value
                        for key, value in values.items()
                        if key.startswith(
                            "learning_rate/"
                        )
                    ),
                    None,
                )
            ),
            "epoch_duration_seconds": duration,
            "samples_per_second": throughput,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "improved": improved,
            "metrics": values,
        }

        self.history.append(record)

        if (
            (epoch + 1)
            % self.config.log_every_n_epochs
            == 0
        ):
            if self.config.jsonl_logging:
                self._write_jsonl(
                    self.jsonl_path,
                    record,
                )

            if self.config.csv_logging:
                self._write_csv(record)

            if self._should_log():
                self._log_scalars(
                    "epoch",
                    values,
                    epoch + 1,
                )

                if self.writer is not None:
                    self.writer.add_scalar(
                        "epoch/samples_per_second",
                        throughput,
                        epoch + 1,
                    )

                if self.config.console_logging:
                    logger.info(
                        "Epoch %d completed | "
                        "train_loss=%s | val_loss=%s | "
                        "duration=%.2fs | best=%s",
                        epoch + 1,
                        values.get("train_loss"),
                        values.get("val_loss"),
                        duration,
                        self.best_score,
                    )

        return record

    def on_checkpoint_saved(
        self,
        *,
        path: str,
        epoch: int,
        is_best: bool = False,
    ) -> None:
        self.log_event(
            "checkpoint_saved",
            message=f"Checkpoint saved: {path}",
            details={
                "path": path,
                "epoch": epoch,
                "is_best": is_best,
            },
        )

    def on_early_stopping(
        self,
        *,
        epoch: int,
        reason: str,
        best_score: float | None = None,
    ) -> None:
        self.log_event(
            "early_stopping",
            message=reason,
            details={
                "epoch": epoch,
                "best_score": best_score,
            },
        )

    def on_exception(
        self,
        error: BaseException,
    ) -> None:
        self.log_event(
            "training_exception",
            message=str(error),
            details={
                "error_type": type(error).__name__,
            },
        )

        logger.exception(
            "Training failed: %s",
            error,
            exc_info=(
                type(error),
                error,
                error.__traceback__,
            ),
        )

    def on_train_end(
        self,
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        duration = 0.0

        if self.training_started_at is not None:
            duration = (
                time.perf_counter()
                - self.training_started_at
            )

        summary = {
            "experiment_name": (
                self.config.experiment_name
            ),
            "status": status,
            "completed_at": utc_now(),
            "total_epochs_logged": len(
                self.history
            ),
            "global_step": self.global_step,
            "total_samples": self.total_samples,
            "duration_seconds": duration,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
        }

        self.log_event(
            "training_finished",
            message=f"Training {status}",
            details=summary,
        )

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

        return summary

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "config": asdict(self.config),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "total_samples": self.total_samples,
            "history": list(self.history),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError(
                "Unsupported logging state version"
            )

        self.global_step = int(
            state.get("global_step", 0)
        )

        self.current_epoch = int(
            state.get("current_epoch", 0)
        )

        self.best_score = state.get(
            "best_score"
        )

        self.best_epoch = state.get(
            "best_epoch"
        )

        self.total_samples = int(
            state.get("total_samples", 0)
        )

        self.history = list(
            state.get("history", [])
        )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "LoggingCallback":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    callback = LoggingCallback(
        LoggingConfig(
            experiment_name="world_model_demo",
            log_every_n_steps=2,
            tensorboard_logging=False,
        )
    )

    callback.on_train_begin(
        metadata={
            "model": "NeuroWorld",
            "dataset": "simulation",
        }
    )

    for epoch in range(3):
        callback.on_epoch_begin(epoch)

        for batch in range(5):
            callback.on_batch_end(
                batch_index=batch,
                batch_size=16,
                metrics={
                    "loss": 1.0 / (
                        1 + epoch * 5 + batch
                    )
                },
            )

        callback.on_epoch_end(
            epoch=epoch,
            metrics={
                "train_loss": 0.5 / (epoch + 1),
                "val_loss": 0.6 / (epoch + 1),
            },
        )

    print(
        callback.on_train_end()
    )
