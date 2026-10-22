
"""
NeuroWorld AI
File: training/utils/experiment_logger.py

Experiment tracking and logging for PyTorch training.

Features:
- Experiment creation and management
- Training and validation metric logging
- Loss component tracking
- Learning-rate and gradient monitoring
- JSONL metric logging
- CSV metric export
- Experiment configuration snapshots
- Hyperparameter logging
- System and GPU monitoring
- Model and dataset metadata
- Checkpoint event tracking
- Distributed rank-aware logging
- Optional TensorBoard integration
- Optional MLflow integration
- Experiment comparison
- Resume existing experiments
- Thread-safe file writing
- Context-manager support

Optional dependencies:
    pip install tensorboard mlflow psutil

Core dependencies:
    pip install torch numpy

Usage:
    logger = ExperimentLogger(
        ExperimentLoggerConfig(
            experiment_name="world_model_training",
            log_dir="runs",
            use_tensorboard=True,
        )
    )

    logger.log_config({
        "learning_rate": 3e-4,
        "batch_size": 32,
    })

    for step in range(100):
        logger.log_metrics(
            {
                "train/loss": 0.25,
                "train/learning_rate": 3e-4,
            },
            step=step,
        )

    logger.close()
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import socket
import sys
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer


LOGGER = logging.getLogger("neuroworld.experiment")


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _safe_name(value: str) -> str:
    """
    Convert experiment names to safe
    filesystem directory names.
    """
    result = "".join(
        character
        if character.isalnum()
        or character in "-_"
        else "_"
        for character in value
    )

    return result.strip("_") or "experiment"


def _to_serializable(
    value: Any,
) -> Any:
    """
    Convert common Python, NumPy and PyTorch
    objects into JSON-compatible values.
    """
    if isinstance(value, Tensor):
        detached = value.detach().cpu()

        if detached.numel() == 1:
            return detached.item()

        return detached.tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_serializable(item)
            for item in value
        ]

    if hasattr(value, "__dataclass_fields__"):
        return _to_serializable(
            asdict(value)
        )

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return str(value)


def _metric_to_float(
    value: Any,
) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(
                "Metrics must be scalar tensors"
            )

        value = value.detach().item()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return float(value)

    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            "Metric values must be numeric"
        )

    return float(value)


def _flatten_dict(
    data: Mapping[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    flattened = {}

    for key, value in data.items():
        name = (
            f"{prefix}/{key}"
            if prefix
            else str(key)
        )

        if isinstance(value, Mapping):
            flattened.update(
                _flatten_dict(
                    value,
                    name,
                )
            )
        else:
            flattened[name] = value

    return flattened


def _atomic_json_write(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    """
    Atomically replace a JSON file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                _to_serializable(data),
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary,
            path,
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


def _get_distributed_rank() -> int:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return torch.distributed.get_rank()

    return int(
        os.environ.get(
            "RANK",
            "0",
        )
    )


def _get_world_size() -> int:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        return torch.distributed.get_world_size()

    return int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )


@dataclass
class ExperimentLoggerConfig:
    experiment_name: str = "neuroworld_experiment"
    run_name: str | None = None

    log_dir: str = "runs"
    run_id: str | None = None

    use_jsonl: bool = True
    use_csv: bool = True

    use_tensorboard: bool = False
    use_mlflow: bool = False

    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None

    log_every_n_steps: int = 1
    flush_every_n_steps: int = 10

    log_system_info: bool = True
    log_gpu_metrics: bool = True

    rank_zero_only: bool = True

    allow_nonfinite_metrics: bool = False

    resume: bool = False

    tags: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError(
                "experiment_name cannot be empty"
            )

        if self.log_every_n_steps < 1:
            raise ValueError(
                "log_every_n_steps must be >= 1"
            )

        if self.flush_every_n_steps < 1:
            raise ValueError(
                "flush_every_n_steps must be >= 1"
            )

        if self.resume and not self.run_id:
            raise ValueError(
                "resume=True requires run_id"
            )


@dataclass
class MetricRecord:
    timestamp: str
    step: int
    epoch: int | None
    phase: str
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricAccumulator:
    """
    Running metric statistics.

    Useful for aggregating training losses
    across minibatches.
    """

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.weights: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(
        self,
        metrics: Mapping[str, Any],
        weight: float = 1.0,
    ) -> None:
        if weight <= 0:
            raise ValueError(
                "weight must be positive"
            )

        for name, value in metrics.items():
            numeric = _metric_to_float(
                value
            )

            if not math.isfinite(numeric):
                continue

            self.totals[name] = (
                self.totals.get(name, 0.0)
                + numeric * weight
            )

            self.weights[name] = (
                self.weights.get(name, 0.0)
                + weight
            )

            self.counts[name] = (
                self.counts.get(name, 0)
                + 1
            )

    def compute(self) -> dict[str, float]:
        return {
            name: (
                total
                / self.weights[name]
            )
            for name, total in self.totals.items()
            if self.weights[name] > 0
        }

    def reset(self) -> None:
        self.totals.clear()
        self.weights.clear()
        self.counts.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "totals": dict(self.totals),
            "weights": dict(self.weights),
            "counts": dict(self.counts),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.totals = dict(
            state.get("totals", {})
        )

        self.weights = dict(
            state.get("weights", {})
        )

        self.counts = dict(
            state.get("counts", {})
        )


class ExperimentLogger:
    """
    Unified experiment logging manager.

    Each experiment has a directory:

        runs/
          experiment_name/
            run_id/
              metadata.json
              config.json
              metrics.jsonl
              metrics.csv
              events.jsonl
              summary.json
              tensorboard/

    In distributed training, only rank zero
    writes by default.
    """

    def __init__(
        self,
        config: (
            ExperimentLoggerConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = ExperimentLoggerConfig()

        elif isinstance(config, Mapping):
            config = ExperimentLoggerConfig(
                **dict(config)
            )

        self.config = config

        self.rank = _get_distributed_rank()
        self.world_size = _get_world_size()

        self.enabled = (
            not config.rank_zero_only
            or self.rank == 0
        )

        self.run_id = (
            config.run_id
            or (
                datetime.now(timezone.utc)
                .strftime("%Y%m%d_%H%M%S")
                + "_"
                + uuid.uuid4().hex[:8]
            )
        )

        self.experiment_dir = (
            Path(config.log_dir)
            / _safe_name(
                config.experiment_name
            )
        )

        self.run_dir = (
            self.experiment_dir
            / _safe_name(self.run_id)
        )

        self.metadata_path = (
            self.run_dir / "metadata.json"
        )

        self.config_path = (
            self.run_dir / "config.json"
        )

        self.metrics_path = (
            self.run_dir / "metrics.jsonl"
        )

        self.csv_path = (
            self.run_dir / "metrics.csv"
        )

        self.events_path = (
            self.run_dir / "events.jsonl"
        )

        self.summary_path = (
            self.run_dir / "summary.json"
        )

        self._lock = threading.RLock()

        self._closed = False
        self._metric_log_count = 0
        self._event_count = 0

        self._start_time = time.time()

        self._last_step = -1

        self._latest_metrics: dict[
            str, float
        ] = {}

        self._best_metrics: dict[
            str, float
        ] = {}

        self._metric_history: list[
            dict[str, Any]
        ] = []

        self._writer = None
        self._mlflow = None
        self._mlflow_run = None

        self._metrics_file = None
        self._events_file = None

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        if (
            self.run_dir.exists()
            and not self.config.resume
            and any(self.run_dir.iterdir())
        ):
            raise FileExistsError(
                f"Run directory already exists: "
                f"{self.run_dir}"
            )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.config.resume:
            self._restore_existing_run()

        if self.config.use_jsonl:
            self._metrics_file = (
                self.metrics_path.open(
                    "a",
                    encoding="utf-8",
                )
            )

        self._events_file = (
            self.events_path.open(
                "a",
                encoding="utf-8",
            )
        )

        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import (
                    SummaryWriter,
                )
            except ImportError as exc:
                raise ImportError(
                    "Install TensorBoard with: "
                    "pip install tensorboard"
                ) from exc

            self._writer = SummaryWriter(
                log_dir=str(
                    self.run_dir / "tensorboard"
                )
            )

        if self.config.use_mlflow:
            try:
                import mlflow
            except ImportError as exc:
                raise ImportError(
                    "Install MLflow with: "
                    "pip install mlflow"
                ) from exc

            self._mlflow = mlflow

            if self.config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(
                    self.config.mlflow_tracking_uri
                )

            mlflow.set_experiment(
                self.config.mlflow_experiment_name
                or self.config.experiment_name
            )

            self._mlflow_run = mlflow.start_run(
                run_name=(
                    self.config.run_name
                    or self.run_id
                ),
                tags=self.config.tags,
            )

        metadata = {
            "experiment_name": (
                self.config.experiment_name
            ),
            "run_name": (
                self.config.run_name
                or self.run_id
            ),
            "run_id": self.run_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "created_at": utc_now(),
            "tags": self.config.tags,
        }

        if self.config.log_system_info:
            metadata["system"] = (
                self.get_system_info()
            )

        if (
            not self.config.resume
            or not self.metadata_path.exists()
        ):
            _atomic_json_write(
                self.metadata_path,
                metadata,
            )

        if not self.config.resume:
            self.log_config(
                asdict(self.config)
            )

    def _restore_existing_run(self) -> None:
        if not self.run_dir.exists():
            raise FileNotFoundError(
                f"Run not found: {self.run_dir}"
            )

        if not self.metrics_path.exists():
            return

        with self.metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                self._metric_history.append(
                    record
                )

                self._metric_log_count += 1

                self._last_step = max(
                    self._last_step,
                    int(
                        record.get("step", -1)
                    ),
                )

                self._latest_metrics.update(
                    record.get(
                        "metrics",
                        {},
                    )
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "ExperimentLogger is closed"
            )

    def log_config(
        self,
        config: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._ensure_open()

            serializable = _to_serializable(
                config
            )

            _atomic_json_write(
                self.config_path,
                serializable,
            )

            if self._mlflow is not None:
                flattened = _flatten_dict(
                    serializable
                )

                for key, value in flattened.items():
                    self._mlflow.log_param(
                        key,
                        str(value),
                    )

    def log_hyperparameters(
        self,
        parameters: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return

        self.log_event(
            "hyperparameters",
            dict(parameters),
        )

        if self._mlflow is not None:
            flattened = _flatten_dict(
                parameters
            )

            for key, value in flattened.items():
                self._mlflow.log_param(
                    key,
                    str(value),
                )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: int,
        epoch: int | None = None,
        phase: str = "train",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return

        if step < 0:
            raise ValueError(
                "step must be nonnegative"
            )

        if (
            not force
            and step
            % self.config.log_every_n_steps
            != 0
        ):
            return

        with self._lock:
            self._ensure_open()

            flattened = _flatten_dict(
                metrics
            )

            numeric_metrics = {}

            for name, value in flattened.items():
                numeric = _metric_to_float(
                    value
                )

                if not math.isfinite(numeric):
                    if (
                        not self.config
                        .allow_nonfinite_metrics
                    ):
                        LOGGER.warning(
                            "Skipping non-finite metric: %s",
                            name,
                        )
                        continue

                    # JSON does not support NaN/Inf
                    # as standard numeric values.
                    continue

                numeric_metrics[name] = numeric

            if not numeric_metrics:
                return

            record = MetricRecord(
                timestamp=utc_now(),
                step=step,
                epoch=epoch,
                phase=phase,
                metrics=numeric_metrics,
            ).to_dict()

            self._metric_history.append(
                record
            )

            self._metric_log_count += 1

            self._last_step = max(
                self._last_step,
                step,
            )

            self._latest_metrics.update(
                numeric_metrics
            )

            if self._metrics_file is not None:
                self._metrics_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

            if self._writer is not None:
                for name, value in numeric_metrics.items():
                    self._writer.add_scalar(
                        name,
                        value,
                        global_step=step,
                    )

            if self._mlflow is not None:
                self._mlflow.log_metrics(
                    numeric_metrics,
                    step=step,
                )

            if (
                self._metric_log_count
                % self.config.flush_every_n_steps
                == 0
            ):
                self.flush()

    def log_scalar(
        self,
        name: str,
        value: Any,
        *,
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        self.log_metrics(
            {name: value},
            step=step,
            epoch=epoch,
            phase=phase,
        )

    def log_losses(
        self,
        losses: Mapping[str, Any],
        *,
        step: int,
        epoch: int | None = None,
        phase: str = "train",
    ) -> None:
        self.log_metrics(
            {
                f"{phase}/loss/{name}": value
                for name, value in losses.items()
            },
            step=step,
            epoch=epoch,
            phase=phase,
        )

    def log_optimizer(
        self,
        optimizer: Optimizer,
        *,
        step: int,
        epoch: int | None = None,
    ) -> None:
        metrics: dict[str, float] = {}

        for index, group in enumerate(
            optimizer.param_groups
        ):
            metrics[
                f"optimizer/lr_group_{index}"
            ] = float(
                group["lr"]
            )

            metrics[
                f"optimizer/weight_decay_group_{index}"
            ] = float(
                group.get(
                    "weight_decay",
                    0.0,
                )
            )

        self.log_metrics(
            metrics,
            step=step,
            epoch=epoch,
            phase="optimizer",
        )

    def log_gradient_norm(
        self,
        model: nn.Module,
        *,
        step: int,
        epoch: int | None = None,
    ) -> float:
        gradients = []

        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach()

                if gradient.is_sparse:
                    gradient = (
                        gradient.coalesce().values()
                    )

                gradients.append(
                    gradient.float()
                )

        if not gradients:
            norm = 0.0
        else:
            squared_norm = sum(
                float(
                    torch.sum(
                        gradient.square()
                    ).item()
                )
                for gradient in gradients
            )

            norm = math.sqrt(
                squared_norm
            )

        self.log_scalar(
            "train/gradient_norm",
            norm,
            step=step,
            epoch=epoch,
        )

        return norm

    def log_model_info(
        self,
        model: nn.Module,
        *,
        name: str = "model",
    ) -> dict[str, Any]:
        parameters = list(
            model.parameters()
        )

        total = sum(
            parameter.numel()
            for parameter in parameters
        )

        trainable = sum(
            parameter.numel()
            for parameter in parameters
            if parameter.requires_grad
        )

        information = {
            "name": name,
            "class": (
                model.__class__.__name__
            ),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": (
                total - trainable
            ),
            "parameter_tensors": len(
                parameters
            ),
        }

        self.log_event(
            "model_info",
            information,
        )

        return information

    def log_dataset_info(
        self,
        *,
        name: str,
        size: int,
        batch_size: int,
        features: Mapping[str, Any] | None = None,
    ) -> None:
        self.log_event(
            "dataset_info",
            {
                "name": name,
                "size": size,
                "batch_size": batch_size,
                "features": dict(
                    features or {}
                ),
            },
        )

    def log_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        step: int,
        metric: float | None = None,
        is_best: bool = False,
    ) -> None:
        self.log_event(
            "checkpoint",
            {
                "path": str(path),
                "epoch": epoch,
                "step": step,
                "metric": metric,
                "is_best": is_best,
            },
        )

    def log_event(
        self,
        event_type: str,
        data: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._ensure_open()

            event = {
                "timestamp": utc_now(),
                "type": event_type,
                "data": _to_serializable(
                    data or {}
                ),
            }

            self._event_count += 1

            if self._events_file is not None:
                self._events_file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

    def log_system_metrics(
        self,
        *,
        step: int,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        try:
            import psutil

            process = psutil.Process()

            metrics["system/cpu_percent"] = float(
                psutil.cpu_percent(
                    interval=None
                )
            )

            metrics["system/memory_percent"] = float(
                psutil.virtual_memory().percent
            )

            metrics["system/process_memory_mb"] = (
                process.memory_info().rss
                / (1024**2)
            )

        except ImportError:
            pass

        if (
            self.config.log_gpu_metrics
            and torch.cuda.is_available()
        ):
            device = torch.cuda.current_device()

            metrics["gpu/allocated_mb"] = (
                torch.cuda.memory_allocated(
                    device
                )
                / (1024**2)
            )

            metrics["gpu/reserved_mb"] = (
                torch.cuda.memory_reserved(
                    device
                )
                / (1024**2)
            )

            metrics["gpu/max_allocated_mb"] = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024**2)
            )

        if metrics:
            self.log_metrics(
                metrics,
                step=step,
                phase="system",
            )

        return metrics

    @staticmethod
    def get_system_info() -> dict[str, Any]:
        information = {
            "python_version": (
                sys.version.split()[0]
            ),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "pytorch_version": torch.__version__,
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "cuda_version": (
                torch.version.cuda
            ),
            "gpu_count": (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
        }

        if torch.cuda.is_available():
            information["gpus"] = [
                torch.cuda.get_device_name(index)
                for index in range(
                    torch.cuda.device_count()
                )
            ]

        return information

    def update_best_metric(
        self,
        name: str,
        value: float,
        *,
        mode: str = "min",
    ) -> bool:
        if mode not in {"min", "max"}:
            raise ValueError(
                "mode must be min or max"
            )

        if not math.isfinite(value):
            return False

        previous = self._best_metrics.get(
            name
        )

        improved = (
            previous is None
            or (
                value < previous
                if mode == "min"
                else value > previous
            )
        )

        if improved:
            self._best_metrics[name] = value

            self.log_event(
                "best_metric",
                {
                    "name": name,
                    "value": value,
                    "mode": mode,
                },
            )

        return improved

    def get_latest_metrics(
        self,
    ) -> dict[str, float]:
        return dict(
            self._latest_metrics
        )

    def get_best_metrics(
        self,
    ) -> dict[str, float]:
        return dict(
            self._best_metrics
        )

    def get_metric_history(
        self,
        metric_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if metric_name is None:
            return list(
                self._metric_history
            )

        return [
            {
                "step": record["step"],
                "epoch": record["epoch"],
                "value": record["metrics"][
                    metric_name
                ],
            }
            for record in self._metric_history
            if metric_name in record["metrics"]
        ]

    def export_csv(
        self,
        path: str | Path | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        with self._lock:
            if path is None:
                path = self.csv_path

            path = Path(path)

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            records = self._metric_history

            metric_names = sorted(
                {
                    name
                    for record in records
                    for name in record["metrics"]
                }
            )

            fieldnames = [
                "timestamp",
                "step",
                "epoch",
                "phase",
                *metric_names,
            ]

            with path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )

                writer.writeheader()

                for record in records:
                    row = {
                        "timestamp": record["timestamp"],
                        "step": record["step"],
                        "epoch": record["epoch"],
                        "phase": record["phase"],
                    }

                    row.update(
                        record["metrics"]
                    )

                    writer.writerow(
                        row
                    )

            return path

    def get_summary(self) -> dict[str, Any]:
        return {
            "experiment_name": (
                self.config.experiment_name
            ),
            "run_id": self.run_id,
            "run_directory": str(
                self.run_dir
            ),
            "rank": self.rank,
            "world_size": self.world_size,
            "last_step": self._last_step,
            "metric_log_count": (
                self._metric_log_count
            ),
            "event_count": (
                self._event_count
            ),
            "elapsed_seconds": (
                time.time()
                - self._start_time
            ),
            "latest_metrics": (
                self.get_latest_metrics()
            ),
            "best_metrics": (
                self.get_best_metrics()
            ),
            "timestamp": utc_now(),
        }

    def flush(self) -> None:
        if not self.enabled or self._closed:
            return

        with self._lock:
            if self._metrics_file is not None:
                self._metrics_file.flush()

            if self._events_file is not None:
                self._events_file.flush()

            if self._writer is not None:
                self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return

        with self._lock:
            if self.enabled:
                self.flush()

                if self.config.use_csv:
                    self.export_csv()

                _atomic_json_write(
                    self.summary_path,
                    self.get_summary(),
                )

                if self._writer is not None:
                    self._writer.close()
                    self._writer = None

                if self._mlflow is not None:
                    self._mlflow.end_run()
                    self._mlflow_run = None

                if self._metrics_file is not None:
                    self._metrics_file.close()
                    self._metrics_file = None

                if self._events_file is not None:
                    self._events_file.close()
                    self._events_file = None

            self._closed = True

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        if exc_type is not None:
            self.log_event(
                "experiment_error",
                {
                    "error_type": str(
                        exc_type
                    ),
                    "message": str(
                        exc_value
                    ),
                },
            )

        self.close()


def compare_experiments(
    run_directories: Sequence[
        str | Path
    ],
) -> list[dict[str, Any]]:
    """
    Compare saved experiment summaries.
    """
    results = []

    for directory in run_directories:
        path = (
            Path(directory)
            / "summary.json"
        )

        if not path.exists():
            continue

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            summary = json.load(file)

        results.append(
            summary
        )

    return results


def create_experiment_logger(
    experiment_name: str,
    *,
    log_dir: str = "runs",
    run_name: str | None = None,
    use_tensorboard: bool = False,
    use_mlflow: bool = False,
    log_every_n_steps: int = 1,
) -> ExperimentLogger:
    return ExperimentLogger(
        ExperimentLoggerConfig(
            experiment_name=experiment_name,
            run_name=run_name,
            log_dir=log_dir,
            use_tensorboard=use_tensorboard,
            use_mlflow=use_mlflow,
            log_every_n_steps=(
                log_every_n_steps
            ),
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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
            state: Tensor,
            action: Tensor,
        ) -> Tensor:
            latent = self.encoder(
                state
            )

            features = torch.cat(
                [latent, action],
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
            T_max=20,
            eta_min=1e-6,
        )
    )

    logger = ExperimentLogger(
        ExperimentLoggerConfig(
            experiment_name=(
                "neuroworld_world_model"
            ),
            run_name="demo_training",
            log_dir="runs",
            use_tensorboard=False,
            use_mlflow=False,
            log_every_n_steps=1,
            flush_every_n_steps=5,
        )
    )

    logger.log_config(
        {
            "model": {
                "state_dim": 128,
                "action_dim": 32,
                "hidden_dim": 256,
            },
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 3e-4,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
            },
        }
    )

    logger.log_model_info(
        model,
        name="world_model",
    )

    global_step = 0

    for epoch in range(1, 4):
        model.train()

        accumulator = MetricAccumulator()

        for batch_index in range(10):
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

            gradient_norm = (
                logger.log_gradient_norm(
                    model,
                    step=global_step,
                    epoch=epoch,
                )
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            accumulator.update(
                {
                    "loss": loss.item(),
                    "gradient_norm": gradient_norm,
                }
            )

            logger.log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": (
                        optimizer.param_groups[0]["lr"]
                    ),
                },
                step=global_step,
                epoch=epoch,
            )

            logger.log_optimizer(
                optimizer,
                step=global_step,
                epoch=epoch,
            )

            global_step += 1

        epoch_metrics = (
            accumulator.compute()
        )

        validation_loss = (
            epoch_metrics["loss"] * 0.95
        )

        logger.log_metrics(
            {
                "train/epoch_loss": (
                    epoch_metrics["loss"]
                ),
                "val/loss": validation_loss,
            },
            step=global_step,
            epoch=epoch,
            phase="validation",
            force=True,
        )

        improved = logger.update_best_metric(
            "val/loss",
            validation_loss,
            mode="min",
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss: "
            f"{epoch_metrics['loss']:.6f} | "
            f"Val Loss: "
            f"{validation_loss:.6f} | "
            f"Best: {improved}"
        )

    print("\nExperiment summary:")
    print(
        logger.get_summary()
    )

    logger.close()

    print(
        "\nLogs saved to:",
        logger.run_dir,
    )
