
"""
NeuroWorld AI
File: training/callbacks/validation_callback.py

PyTorch validation callback.

Features:
- Automatic epoch-based validation
- Optional step-based validation
- Validation under torch.inference_mode()
- Automatic mixed precision (AMP)
- Device-aware batch transfer
- Deterministic and probabilistic predictions
- Custom validation step functions
- Custom metrics
- Weighted metric aggregation
- Distributed metric synchronization (DDP)
- Validation history and JSON reports
- Checkpoint and early-stopping integration
- Optional logging callback integration
- NaN/Inf validation detection
- Save and restore callback state

Dependencies:
    torch

Python 3.11+
"""

from __future__ import annotations

import json
import logging
import math
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
)

import torch
import torch.distributed as dist

from torch import Tensor, nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


MetricFunction = Callable[
    [Any, Any],
    Tensor | float,
]

ValidationStep = Callable[
    [nn.Module, Any],
    Mapping[str, Any],
]


@dataclass
class ValidationConfig:
    validate_every_n_epochs: int = 1
    validate_every_n_steps: int | None = None

    max_validation_batches: int | None = None

    device: str = "auto"

    use_amp: bool = True

    amp_dtype: Literal[
        "float16",
        "bfloat16",
    ] = "float16"

    monitor: str = "val_loss"

    mode: Literal["min", "max"] = "min"

    sync_distributed: bool = True

    fail_on_non_finite: bool = True

    save_reports: bool = True

    report_dir: str = "reports/validation"

    verbose: bool = True

    def __post_init__(self) -> None:
        if self.validate_every_n_epochs < 1:
            raise ValueError(
                "validate_every_n_epochs must be >= 1"
            )

        if (
            self.validate_every_n_steps is not None
            and self.validate_every_n_steps < 1
        ):
            raise ValueError(
                "validate_every_n_steps must be >= 1"
            )

        if (
            self.max_validation_batches is not None
            and self.max_validation_batches < 1
        ):
            raise ValueError(
                "max_validation_batches must be >= 1"
            )

        if self.mode not in {"min", "max"}:
            raise ValueError(
                "mode must be min or max"
            )


@dataclass
class ValidationResult:
    epoch: int
    global_step: int

    metrics: dict[str, float]

    batches: int
    samples: int

    duration_seconds: float

    improved: bool
    best_score: float | None
    best_epoch: int | None

    timestamp: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationCallback:
    """
    Validation callback for NeuroWorld AI.

    Default batch format:
        {
            "states": Tensor[B, D],
            "actions": Tensor[B, A],
            "next_states": Tensor[B, D]
        }

    Default model:
        model(states, actions)

    Supported outputs:
        Tensor
        (mean, logvar)
        {"prediction": Tensor}
        {"mean": Tensor, "logvar": Tensor}

    Custom validation_step_fn can override
    these assumptions.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        config: ValidationConfig | None = None,
        *,
        loss_fn: Callable[
            [Tensor, Tensor],
            Tensor,
        ] | None = None,
        metric_fns: Mapping[
            str,
            MetricFunction,
        ] | None = None,
        validation_step_fn: ValidationStep | None = None,
        logging_callback: Any | None = None,
        checkpoint_callback: Any | None = None,
        early_stopping: Any | None = None,
    ) -> None:
        self.config = (
            config or ValidationConfig()
        )

        self.loss_fn = (
            loss_fn or nn.MSELoss()
        )

        self.metric_fns = dict(
            metric_fns or {}
        )

        self.validation_step_fn = (
            validation_step_fn
        )

        self.logging_callback = (
            logging_callback
        )

        self.checkpoint_callback = (
            checkpoint_callback
        )

        self.early_stopping = (
            early_stopping
        )

        self.device = self._resolve_device(
            self.config.device
        )

        self.history: list[
            ValidationResult
        ] = []

        self.best_score: float | None = None
        self.best_epoch: int | None = None

        self.last_validation_step: int | None = None

        self.report_dir = Path(
            self.config.report_dir
        )

        if self.config.save_reports:
            self.report_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _resolve_device(
        device: str,
    ) -> torch.device:
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    @staticmethod
    def _is_main_process() -> bool:
        if (
            dist.is_available()
            and dist.is_initialized()
        ):
            return dist.get_rank() == 0

        return True

    def _move_to_device(
        self,
        value: Any,
    ) -> Any:
        if isinstance(value, Tensor):
            return value.to(
                self.device,
                non_blocking=True,
            )

        if isinstance(value, dict):
            return {
                key: self._move_to_device(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._move_to_device(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._move_to_device(item)
                for item in value
            )

        return value

    @staticmethod
    def _batch_size(
        batch: Any,
    ) -> int:
        if isinstance(batch, Tensor):
            return int(batch.shape[0])

        if isinstance(batch, Mapping):
            for value in batch.values():
                if (
                    isinstance(value, Tensor)
                    and value.ndim > 0
                ):
                    return int(
                        value.shape[0]
                    )

        if isinstance(batch, (list, tuple)):
            for value in batch:
                if (
                    isinstance(value, Tensor)
                    and value.ndim > 0
                ):
                    return int(
                        value.shape[0]
                    )

        return 1

    @staticmethod
    def _extract_prediction(
        output: Any,
    ) -> Tensor:
        if isinstance(output, Tensor):
            return output

        if isinstance(output, Mapping):
            for key in (
                "prediction",
                "predictions",
                "mean",
                "next_state",
                "reconstruction",
            ):
                value = output.get(key)

                if isinstance(value, Tensor):
                    return value

        if isinstance(output, (tuple, list)):
            if (
                len(output) > 0
                and isinstance(output[0], Tensor)
            ):
                return output[0]

        raise TypeError(
            "Unable to extract prediction "
            "tensor from model output"
        )

    def _default_validation_step(
        self,
        model: nn.Module,
        batch: Any,
    ) -> Mapping[str, Any]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "Default validation requires "
                "a dictionary batch"
            )

        states = batch["states"]
        actions = batch["actions"]
        targets = batch["next_states"]

        output = model(
            states,
            actions,
        )

        predictions = (
            self._extract_prediction(output)
        )

        if predictions.shape != targets.shape:
            raise ValueError(
                "Prediction and target shapes "
                "must match"
            )

        loss = self.loss_fn(
            predictions,
            targets,
        )

        results: dict[str, Any] = {
            "val_loss": loss,
        }

        for name, metric_fn in (
            self.metric_fns.items()
        ):
            results[name] = metric_fn(
                predictions,
                targets,
            )

        return results

    def _autocast_context(self):
        enabled = (
            self.config.use_amp
            and self.device.type in {
                "cuda",
                "cpu",
            }
        )

        dtype = (
            torch.bfloat16
            if self.config.amp_dtype
            == "bfloat16"
            else torch.float16
        )

        if (
            self.device.type == "cpu"
            and dtype == torch.float16
        ):
            enabled = False

        return torch.autocast(
            device_type=(
                self.device.type
                if self.device.type in {
                    "cuda",
                    "cpu",
                }
                else "cpu"
            ),
            dtype=dtype,
            enabled=enabled,
        )

    def should_validate(
        self,
        *,
        epoch: int,
        global_step: int | None = None,
    ) -> bool:
        epoch_due = (
            (epoch + 1)
            % self.config.validate_every_n_epochs
            == 0
        )

        if (
            global_step is None
            or self.config.validate_every_n_steps
            is None
        ):
            return epoch_due

        step_due = (
            global_step > 0
            and global_step
            % self.config.validate_every_n_steps
            == 0
            and global_step
            != self.last_validation_step
        )

        return epoch_due or step_due

    @staticmethod
    def _scalar(
        value: Any,
        metric_name: str,
    ) -> float:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{metric_name} must be scalar"
                )

            value = value.detach().item()

        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{metric_name} must be numeric"
            ) from exc

    def _aggregate_distributed(
        self,
        sums: dict[str, float],
        counts: dict[str, float],
        batches: int,
        samples: int,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        int,
        int,
    ]:
        if not (
            self.config.sync_distributed
            and dist.is_available()
            and dist.is_initialized()
        ):
            return sums, counts, batches, samples

        # Every rank must produce the same metric
        # names for distributed aggregation.
        local_names = sorted(sums)

        gathered_names: list[Any] = [
            None
        ] * dist.get_world_size()

        dist.all_gather_object(
            gathered_names,
            local_names,
        )

        names = sorted(
            set().union(
                *[
                    set(items)
                    for items in gathered_names
                ]
            )
        )

        device = (
            self.device
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )

        values = []

        for name in names:
            values.extend(
                [
                    sums.get(name, 0.0),
                    counts.get(name, 0.0),
                ]
            )

        values.extend(
            [float(batches), float(samples)]
        )

        tensor = torch.tensor(
            values,
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        reduced = tensor.cpu().tolist()

        merged_sums = {}
        merged_counts = {}

        for index, name in enumerate(names):
            merged_sums[name] = reduced[
                index * 2
            ]

            merged_counts[name] = reduced[
                index * 2 + 1
            ]

        return (
            merged_sums,
            merged_counts,
            int(reduced[-2]),
            int(reduced[-1]),
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

        if not math.isfinite(score):
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

    def validate(
        self,
        *,
        model: nn.Module,
        dataloader: DataLoader,
        epoch: int,
        global_step: int = 0,
    ) -> ValidationResult:
        """
        Run validation and return aggregated metrics.

        Custom step functions should return
        scalar metrics averaged per sample.
        """
        started_at = time.perf_counter()

        was_training = model.training

        model.eval()

        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, float] = {}

        batch_count = 0
        sample_count = 0

        step_fn = (
            self.validation_step_fn
            or self._default_validation_step
        )

        try:
            with torch.inference_mode():
                for batch in dataloader:
                    if (
                        self.config.max_validation_batches
                        is not None
                        and batch_count
                        >= self.config.max_validation_batches
                    ):
                        break

                    batch = self._move_to_device(
                        batch
                    )

                    batch_size = self._batch_size(
                        batch
                    )

                    with self._autocast_context():
                        output = step_fn(
                            model,
                            batch,
                        )

                    if not isinstance(
                        output,
                        Mapping,
                    ):
                        raise TypeError(
                            "Validation step must "
                            "return a mapping"
                        )

                    for name, raw_value in (
                        output.items()
                    ):
                        value = self._scalar(
                            raw_value,
                            name,
                        )

                        if not math.isfinite(value):
                            if (
                                self.config
                                .fail_on_non_finite
                            ):
                                raise FloatingPointError(
                                    f"Non-finite validation "
                                    f"metric: {name}={value}"
                                )

                            continue

                        metric_sums[name] = (
                            metric_sums.get(
                                name,
                                0.0,
                            )
                            + value * batch_size
                        )

                        metric_counts[name] = (
                            metric_counts.get(
                                name,
                                0.0,
                            )
                            + batch_size
                        )

                    batch_count += 1
                    sample_count += batch_size

        finally:
            model.train(was_training)

        (
            metric_sums,
            metric_counts,
            batch_count,
            sample_count,
        ) = self._aggregate_distributed(
            metric_sums,
            metric_counts,
            batch_count,
            sample_count,
        )

        metrics = {
            name: (
                total
                / metric_counts[name]
            )
            for name, total in metric_sums.items()
            if metric_counts.get(name, 0) > 0
        }

        duration = (
            time.perf_counter()
            - started_at
        )

        improved = self._update_best(
            metrics,
            epoch,
        )

        result = ValidationResult(
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
            batches=batch_count,
            samples=sample_count,
            duration_seconds=duration,
            improved=improved,
            best_score=self.best_score,
            best_epoch=self.best_epoch,
        )

        self.history.append(result)

        self.last_validation_step = (
            global_step
        )

        if self.config.verbose and (
            self._is_main_process()
        ):
            logger.info(
                "Validation | epoch=%d | "
                "step=%d | metrics=%s | "
                "duration=%.2fs",
                epoch + 1,
                global_step,
                metrics,
                duration,
            )

        if (
            self.config.save_reports
            and self._is_main_process()
        ):
            self._save_report(result)

        return result

    def _save_report(
        self,
        result: ValidationResult,
    ) -> Path:
        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = self.report_dir / (
            f"validation_epoch_"
            f"{result.epoch + 1:06d}_"
            f"step_{result.global_step:09d}.json"
        )

        path.write_text(
            json.dumps(
                result.to_dict(),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return path

    def on_epoch_end(
        self,
        *,
        epoch: int,
        model: nn.Module,
        dataloader: DataLoader,
        global_step: int = 0,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        train_metrics: Mapping[
            str,
            float,
        ] | None = None,
    ) -> ValidationResult | None:
        """
        Run validation when due, then notify
        checkpoint and early-stopping callbacks.

        Avoid calling the same callbacks again
        separately in the training loop.
        """
        if not self.should_validate(
            epoch=epoch,
            global_step=global_step,
        ):
            return None

        result = self.validate(
            model=model,
            dataloader=dataloader,
            epoch=epoch,
            global_step=global_step,
        )

        combined_metrics = dict(
            train_metrics or {}
        )

        combined_metrics.update(
            result.metrics
        )

        if self.checkpoint_callback is not None:
            checkpoint_result = (
                self.checkpoint_callback.on_epoch_end(
                    epoch=epoch,
                    model=model,
                    metrics=combined_metrics,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                )
            )

            if self.logging_callback is not None:
                for path in checkpoint_result.get(
                    "paths",
                    [],
                ):
                    self.logging_callback.on_checkpoint_saved(
                        path=path,
                        epoch=epoch,
                        is_best=(
                            path.endswith("_best.pt")
                        ),
                    )

        if self.early_stopping is not None:
            stop_result = (
                self.early_stopping.on_epoch_end(
                    epoch=epoch,
                    metrics=combined_metrics,
                )
            )

            if (
                stop_result.should_stop
                and self.logging_callback is not None
            ):
                self.logging_callback.on_early_stopping(
                    epoch=epoch,
                    reason=stop_result.reason.value,
                    best_score=stop_result.best_score,
                )

        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "config": asdict(self.config),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "last_validation_step": (
                self.last_validation_step
            ),
            "history": [
                item.to_dict()
                for item in self.history
            ],
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        if (
            state.get("state_version")
            != self.STATE_VERSION
        ):
            raise ValueError(
                "Unsupported validation "
                "state version"
            )

        self.best_score = state.get(
            "best_score"
        )

        self.best_epoch = state.get(
            "best_epoch"
        )

        self.last_validation_step = state.get(
            "last_validation_step"
        )

        self.history = [
            ValidationResult(**item)
            for item in state.get(
                "history",
                [],
            )
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "validations": len(self.history),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "latest_metrics": (
                self.history[-1].metrics
                if self.history
                else {}
            ),
        }
