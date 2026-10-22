
"""
NeuroWorld AI
File: training/evaluation/validation_runner.py

Validation engine for:
- World models
- Dynamics prediction
- Video prediction
- Reconstruction
- Reward prediction
- Custom PyTorch models

Features:
- CPU and CUDA validation
- FP32, FP16, BF16 mixed precision
- Masked sequence metrics
- MSE, MAE, RMSE, R2, PSNR
- Custom forward/loss/metric functions
- Distributed metric synchronization
- Best-model tracking
- Early stopping
- Learning-rate scheduler integration
- Rank-zero checkpoint saving
- JSON and CSV logging
- Validation history and summaries

Requires:
    torch >= 2.1
"""

from __future__ import annotations

import csv
import json
import math
import os

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


ForwardFunction = Callable[
    [nn.Module, Mapping[str, Any]],
    Any,
]

LossFunction = Callable[
    [Any, Mapping[str, Any]],
    Tensor,
]

MetricFunction = Callable[
    [Tensor, Tensor],
    Tensor | float,
]


@dataclass
class ValidationRunnerConfig:
    device: str = "auto"
    precision: str = "auto"

    max_batches: int | None = None
    validation_interval: int = 1

    monitor: str = "val_loss"
    monitor_mode: str = "min"
    min_delta: float = 0.0

    early_stopping_patience: int | None = None

    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True

    checkpoint_directory: str = (
        "checkpoints/validation"
    )

    report_directory: str = (
        "outputs/validation"
    )

    save_json: bool = True
    save_csv: bool = True

    distributed: bool = False

    pixel_max_value: float = 1.0

    calculate_mse: bool = True
    calculate_mae: bool = True
    calculate_rmse: bool = True
    calculate_r2: bool = True
    calculate_psnr: bool = False

    def __post_init__(self) -> None:
        if self.precision not in {
            "auto", "fp32", "fp16", "bf16"
        }:
            raise ValueError(
                "Invalid precision"
            )

        if self.validation_interval < 1:
            raise ValueError(
                "validation_interval must be >= 1"
            )

        if (
            self.max_batches is not None
            and self.max_batches < 1
        ):
            raise ValueError(
                "max_batches must be >= 1"
            )

        if self.monitor_mode not in {
            "min", "max"
        }:
            raise ValueError(
                "monitor_mode must be min or max"
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta must be >= 0"
            )

        if (
            self.early_stopping_patience
            is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be >= 1"
            )

        if self.pixel_max_value <= 0:
            raise ValueError(
                "pixel_max_value must be positive"
            )


@dataclass
class ValidationResult:
    epoch: int
    metrics: dict[str, float]

    sample_count: int
    batch_count: int

    improved: bool = False
    should_stop: bool = False

    best_metric: float | None = None
    bad_epochs: int = 0

    timestamp: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationRunner:
    """
    Reusable model validation runner.

    Example:
        runner = ValidationRunner(
            ValidationRunnerConfig(
                monitor="val_loss",
                early_stopping_patience=10,
            )
        )

        result = runner.on_epoch_end(
            epoch=1,
            model=model,
            val_loader=val_loader,
            optimizer=optimizer,
        )

        if result and result.should_stop:
            break

    Notes:
        All distributed ranks must call
        validation methods together.

        For exact distributed validation,
        use a sampler that does not duplicate
        samples during partitioning.
    """

    def __init__(
        self,
        config: ValidationRunnerConfig | None = None,
        *,
        forward_fn: ForwardFunction | None = None,
        loss_fn: LossFunction | None = None,
    ) -> None:
        self.config = (
            config or ValidationRunnerConfig()
        )

        self.device = self._resolve_device(
            self.config.device
        )

        self.precision = self._resolve_precision(
            self.config.precision
        )

        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]

        self.forward_fn = (
            forward_fn or self._default_forward
        )

        self.loss_fn = (
            loss_fn or self._default_loss
        )

        self.custom_metrics: dict[
            str, MetricFunction
        ] = {}

        self.distributed = (
            self.config.distributed
        )

        if self.distributed:
            if (
                not dist.is_available()
                or not dist.is_initialized()
            ):
                raise RuntimeError(
                    "Initialize the distributed "
                    "process group first"
                )

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

            if (
                dist.get_backend() == "nccl"
                and self.device.type != "cuda"
            ):
                raise RuntimeError(
                    "NCCL requires CUDA"
                )

        else:
            self.rank = 0
            self.world_size = 1

        self.is_main_process = (
            self.rank == 0
        )

        self.best_metric: float | None = None
        self.best_epoch: int | None = None

        self.bad_epochs = 0
        self.should_stop = False

        self.history: list[
            ValidationResult
        ] = []

        self.checkpoint_directory = Path(
            self.config.checkpoint_directory
        )

        self.report_directory = Path(
            self.config.report_directory
        )

        if self.is_main_process:
            self.checkpoint_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.report_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    @staticmethod
    def _resolve_device(
        requested: str,
    ) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                local_rank = int(
                    os.environ.get(
                        "LOCAL_RANK",
                        "0",
                    )
                )

                if local_rank >= torch.cuda.device_count():
                    raise RuntimeError(
                        "Invalid LOCAL_RANK"
                    )

                return torch.device(
                    "cuda",
                    local_rank,
                )

            return torch.device("cpu")

        device = torch.device(requested)

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA is unavailable"
            )

        if device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "Only CPU and CUDA are supported"
            )

        return device

    def _resolve_precision(
        self,
        requested: str,
    ) -> str:
        if requested == "auto":
            if self.device.type == "cuda":
                return (
                    "bf16"
                    if torch.cuda.is_bf16_supported()
                    else "fp16"
                )

            return "fp32"

        if (
            requested == "fp16"
            and self.device.type != "cuda"
        ):
            raise RuntimeError(
                "FP16 requires CUDA"
            )

        if (
            requested == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 unsupported on this GPU"
            )

        return requested

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
        )

    def register_metric(
        self,
        name: str,
        function: MetricFunction,
    ) -> None:
        if not name:
            raise ValueError(
                "Metric name cannot be empty"
            )

        if name in {
            "val_loss", "mse", "mae",
            "rmse", "r2", "psnr"
        }:
            raise ValueError(
                "Metric name is reserved"
            )

        self.custom_metrics[name] = function

    def _move_batch(
        self,
        value: Any,
    ) -> Any:
        if isinstance(value, Tensor):
            return value.to(
                self.device,
                non_blocking=True,
            )

        if isinstance(value, Mapping):
            return {
                key: self._move_batch(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._move_batch(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._move_batch(item)
                for item in value
            )

        return value

    @staticmethod
    def _default_forward(
        model: nn.Module,
        batch: Mapping[str, Any],
    ) -> Any:
        if (
            "states" in batch
            and "actions" in batch
        ):
            return model(
                batch["states"],
                batch["actions"],
            )

        if "context_frames" in batch:
            return model(
                batch["context_frames"]
            )

        if "frames" in batch:
            return model(batch["frames"])

        if "images" in batch:
            return model(batch["images"])

        raise ValueError(
            "Unsupported input structure. "
            "Provide forward_fn."
        )

    @staticmethod
    def _extract_prediction(
        output: Any,
    ) -> Tensor:
        if isinstance(output, Tensor):
            return output

        if isinstance(output, Mapping):
            for key in (
                "predictions",
                "next_states",
                "predicted_frames",
                "reconstruction",
                "rewards",
                "output",
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
            "Cannot extract prediction Tensor"
        )

    @staticmethod
    def _extract_target(
        batch: Mapping[str, Any],
    ) -> Tensor:
        for key in (
            "next_states",
            "target_frames",
            "targets",
            "target",
            "reconstruction_target",
            "rewards",
            "frames",
            "images",
        ):
            value = batch.get(key)

            if isinstance(value, Tensor):
                return value

        raise ValueError(
            "No validation target found"
        )

    @staticmethod
    def _extract_mask(
        batch: Mapping[str, Any],
        target: Tensor,
    ) -> Tensor:
        mask = batch.get(
            "mask",
            batch.get("frame_mask"),
        )

        if mask is None:
            return torch.ones_like(
                target,
                dtype=torch.bool,
            )

        if not isinstance(mask, Tensor):
            raise TypeError(
                "Mask must be a Tensor"
            )

        mask = mask.bool()

        if mask.shape[0] != target.shape[0]:
            raise ValueError(
                "Mask batch dimension mismatch"
            )

        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                target.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Mask shape incompatible "
                "with prediction"
            ) from exc

    @classmethod
    def _default_loss(
        cls,
        output: Any,
        batch: Mapping[str, Any],
    ) -> Tensor:
        if isinstance(output, Mapping):
            existing_loss = output.get("loss")

            if isinstance(existing_loss, Tensor):
                if existing_loss.ndim != 0:
                    raise ValueError(
                        "Model loss must be scalar"
                    )

                return existing_loss

        prediction = cls._extract_prediction(
            output
        )

        target = cls._extract_target(
            batch
        )

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ"
            )

        valid = cls._extract_mask(
            batch,
            target,
        )

        squared_error = (
            prediction.float()
            - target.float()
        ).square()

        denominator = (
            valid.sum().clamp_min(1)
        )

        return (
            squared_error[valid].sum()
            / denominator
        )

    @staticmethod
    def _batch_size(
        batch: Mapping[str, Any],
    ) -> int:
        for value in batch.values():
            if (
                isinstance(value, Tensor)
                and value.ndim >= 1
            ):
                return int(value.shape[0])

        raise ValueError(
            "Cannot determine batch size"
        )

    @staticmethod
    def _prediction_statistics(
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> dict[str, float]:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction/target shape mismatch"
            )

        count = int(
            mask.sum().item()
        )

        if count == 0:
            return {
                "element_count": 0.0,
                "squared_error": 0.0,
                "absolute_error": 0.0,
                "target_sum": 0.0,
                "target_squared_sum": 0.0,
            }

        prediction = prediction.float()
        target = target.float()

        error = (
            prediction[mask]
            - target[mask]
        )

        valid_target = target[mask]

        return {
            "element_count": float(count),
            "squared_error": float(
                error.square()
                .sum(dtype=torch.float64)
                .item()
            ),
            "absolute_error": float(
                error.abs()
                .sum(dtype=torch.float64)
                .item()
            ),
            "target_sum": float(
                valid_target
                .sum(dtype=torch.float64)
                .item()
            ),
            "target_squared_sum": float(
                valid_target.square()
                .sum(dtype=torch.float64)
                .item()
            ),
        }

    def _all_reduce(
        self,
        values: Tensor,
    ) -> Tensor:
        if self.distributed:
            dist.all_reduce(
                values,
                op=dist.ReduceOp.SUM,
            )

        return values

    def _finalize_metrics(
        self,
        totals: dict[str, float],
    ) -> dict[str, float]:
        count = totals["element_count"]

        metrics = {
            "val_loss": (
                totals["loss_sum"]
                / totals["loss_weight"]
                if totals["loss_weight"] > 0
                else float("nan")
            )
        }

        if count <= 0:
            return metrics

        mse = (
            totals["squared_error"]
            / count
        )

        mae = (
            totals["absolute_error"]
            / count
        )

        if self.config.calculate_mse:
            metrics["mse"] = mse

        if self.config.calculate_mae:
            metrics["mae"] = mae

        if self.config.calculate_rmse:
            metrics["rmse"] = math.sqrt(
                max(mse, 0.0)
            )

        if self.config.calculate_r2:
            target_mean = (
                totals["target_sum"]
                / count
            )

            total_variance = max(
                totals["target_squared_sum"]
                - count * target_mean**2,
                0.0,
            )

            if total_variance > 0:
                metrics["r2"] = (
                    1.0
                    - totals["squared_error"]
                    / total_variance
                )

            elif totals["squared_error"] == 0:
                metrics["r2"] = 1.0

            else:
                metrics["r2"] = 0.0

        if self.config.calculate_psnr:
            metrics["psnr"] = (
                float("inf")
                if mse == 0
                else (
                    10.0
                    * math.log10(
                        self.config.pixel_max_value**2
                        / mse
                    )
                )
            )

        return metrics

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        *,
        epoch: int = 0,
    ) -> ValidationResult:
        """
        Run validation and return metrics.

        All ranks must participate when
        distributed=True.
        """
        original_training = model.training

        if not self.distributed:
            model = model.to(self.device)

        model.eval()

        totals = {
            "element_count": 0.0,
            "squared_error": 0.0,
            "absolute_error": 0.0,
            "target_sum": 0.0,
            "target_squared_sum": 0.0,
            "loss_sum": 0.0,
            "loss_weight": 0.0,
        }

        custom_sums = {
            name: 0.0
            for name in self.custom_metrics
        }

        custom_weights = {
            name: 0.0
            for name in self.custom_metrics
        }

        local_samples = 0
        local_batches = 0

        try:
            for batch_index, raw_batch in enumerate(
                val_loader
            ):
                if (
                    self.config.max_batches is not None
                    and batch_index
                    >= self.config.max_batches
                ):
                    break

                batch = self._move_batch(
                    raw_batch
                )

                with self._autocast():
                    output = self.forward_fn(
                        model,
                        batch,
                    )

                    loss = self.loss_fn(
                        output,
                        batch,
                    )

                if (
                    not isinstance(loss, Tensor)
                    or loss.ndim != 0
                ):
                    raise ValueError(
                        "Validation loss must "
                        "be a scalar Tensor"
                    )

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        "Non-finite validation loss"
                    )

                batch_size = self._batch_size(
                    batch
                )

                totals["loss_sum"] += (
                    float(loss.item())
                    * batch_size
                )

                totals["loss_weight"] += (
                    batch_size
                )

                local_samples += batch_size
                local_batches += 1

                # A model may return only a loss.
                # Prediction metrics are optional.
                prediction = None

                try:
                    prediction = (
                        self._extract_prediction(
                            output
                        )
                    )

                except TypeError:
                    pass

                if prediction is not None:
                    target = self._extract_target(
                        batch
                    )

                    mask = self._extract_mask(
                        batch,
                        target,
                    )

                    statistics = (
                        self._prediction_statistics(
                            prediction,
                            target,
                            mask,
                        )
                    )

                    for key, value in (
                        statistics.items()
                    ):
                        totals[key] += value

                    for name, function in (
                        self.custom_metrics.items()
                    ):
                        value = function(
                            prediction,
                            target,
                        )

                        if isinstance(value, Tensor):
                            if value.numel() != 1:
                                raise ValueError(
                                    "Custom metric must "
                                    "return a scalar"
                                )

                            value = float(
                                value.item()
                            )

                        else:
                            value = float(value)

                        custom_sums[name] += (
                            value * batch_size
                        )

                        custom_weights[name] += (
                            batch_size
                        )

            keys = list(
                totals.keys()
            )

            values = torch.tensor(
                [
                    totals[key]
                    for key in keys
                ],
                dtype=torch.float64,
                device=self.device,
            )

            self._all_reduce(
                values
            )

            global_totals = {
                key: float(
                    values[index].item()
                )
                for index, key in enumerate(keys)
            }

            metrics = self._finalize_metrics(
                global_totals
            )

            for name in sorted(
                self.custom_metrics
            ):
                pair = torch.tensor(
                    [
                        custom_sums[name],
                        custom_weights[name],
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )

                self._all_reduce(
                    pair
                )

                weight = float(
                    pair[1].item()
                )

                metrics[name] = (
                    float(pair[0].item())
                    / weight
                    if weight > 0
                    else float("nan")
                )

            counts = torch.tensor(
                [
                    local_samples,
                    local_batches,
                ],
                dtype=torch.long,
                device=self.device,
            )

            self._all_reduce(
                counts
            )

            return ValidationResult(
                epoch=epoch,
                metrics=metrics,
                sample_count=int(
                    counts[0].item()
                ),
                batch_count=int(
                    counts[1].item()
                ),
            )

        finally:
            model.train(
                original_training
            )

    def _is_improvement(
        self,
        value: float,
    ) -> bool:
        if not math.isfinite(value):
            return False

        if self.best_metric is None:
            return True

        if self.config.monitor_mode == "min":
            return (
                value
                < self.best_metric
                - self.config.min_delta
            )

        return (
            value
            > self.best_metric
            + self.config.min_delta
        )

    @staticmethod
    def _unwrap_model(
        model: nn.Module,
    ) -> nn.Module:
        if isinstance(
            model,
            nn.parallel.DistributedDataParallel,
        ):
            return model.module

        return model

    def save_checkpoint(
        self,
        filename: str,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """
        Rank-zero checkpoint saving.

        Supports ordinary and DDP models.

        For FSDP, use the FSDP-specific
        checkpoint API instead.
        """
        destination = (
            self.checkpoint_directory
            / filename
        )

        if self.is_main_process:
            self.checkpoint_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": (
                    self._unwrap_model(
                        model
                    ).state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                    if optimizer is not None
                    else None
                ),
                "scheduler_state_dict": (
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
                ),
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "bad_epochs": self.bad_epochs,
                "extra": extra or {},
            }

            temporary = destination.with_suffix(
                destination.suffix + ".tmp"
            )

            torch.save(
                checkpoint,
                temporary,
            )

            os.replace(
                temporary,
                destination,
            )

        if self.distributed:
            dist.barrier()

        return destination

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load a trusted checkpoint.

        Intended for standard/DDP models.
        """
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self._unwrap_model(
            model
        ).load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

        optimizer_state = checkpoint.get(
            "optimizer_state_dict"
        )

        if (
            optimizer is not None
            and optimizer_state is not None
        ):
            optimizer.load_state_dict(
                optimizer_state
            )

        scheduler_state = checkpoint.get(
            "scheduler_state_dict"
        )

        if (
            scheduler is not None
            and scheduler_state is not None
        ):
            scheduler.load_state_dict(
                scheduler_state
            )

        self.best_metric = checkpoint.get(
            "best_metric"
        )

        self.best_epoch = checkpoint.get(
            "best_epoch"
        )

        self.bad_epochs = int(
            checkpoint.get(
                "bad_epochs",
                0,
            )
        )

        if self.distributed:
            dist.barrier()

        return {
            "epoch": checkpoint.get(
                "epoch", 0
            ),
            "extra": checkpoint.get(
                "extra", {}
            ),
        }

    def on_epoch_end(
        self,
        *,
        epoch: int,
        model: nn.Module,
        val_loader: DataLoader,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ValidationResult | None:
        """
        Validate at configured intervals.

        Handles:
        - Best-model tracking
        - Early stopping
        - ReduceLROnPlateau
        - Checkpoint saving
        - Report export
        """
        if (
            epoch
            % self.config.validation_interval
            != 0
        ):
            return None

        result = self.validate(
            model,
            val_loader,
            epoch=epoch,
        )

        if self.config.monitor not in result.metrics:
            raise KeyError(
                f"Monitor '{self.config.monitor}' "
                "not found in validation metrics"
            )

        monitored = result.metrics[
            self.config.monitor
        ]

        improved = self._is_improvement(
            monitored
        )

        if improved:
            self.best_metric = monitored
            self.best_epoch = epoch
            self.bad_epochs = 0

        else:
            self.bad_epochs += 1

        patience = (
            self.config.early_stopping_patience
        )

        self.should_stop = (
            patience is not None
            and self.bad_epochs >= patience
        )

        result.improved = improved
        result.should_stop = self.should_stop
        result.best_metric = self.best_metric
        result.bad_epochs = self.bad_epochs

        self.history.append(
            result
        )

        if (
            scheduler is not None
            and isinstance(
                scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            )
        ):
            scheduler.step(
                monitored
            )

        if (
            improved
            and self.config.save_best_checkpoint
        ):
            self.save_checkpoint(
                "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extra=extra,
            )

        if self.config.save_last_checkpoint:
            self.save_checkpoint(
                "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extra=extra,
            )

        if self.is_main_process:
            self.print_result(
                result
            )

            if self.config.save_json:
                self.save_result_json(
                    result
                )

            if self.config.save_csv:
                self.save_history_csv()

        if self.distributed:
            stop_tensor = torch.tensor(
                int(self.should_stop),
                dtype=torch.int32,
                device=self.device,
            )

            dist.broadcast(
                stop_tensor,
                src=0,
            )

            self.should_stop = bool(
                stop_tensor.item()
            )

            result.should_stop = (
                self.should_stop
            )

        return result

    @staticmethod
    def print_result(
        result: ValidationResult,
    ) -> None:
        print(
            f"\nValidation Epoch {result.epoch}"
        )

        print(
            f"Samples: {result.sample_count}"
        )

        for name, value in (
            result.metrics.items()
        ):
            print(
                f"{name}: {value:.6f}"
            )

        print(
            f"Improved: {result.improved}"
        )

        print(
            f"Bad epochs: {result.bad_epochs}"
        )

        print(
            f"Early stop: {result.should_stop}"
        )

    def save_result_json(
        self,
        result: ValidationResult,
    ) -> Path:
        self.report_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            self.report_directory
            / f"validation_epoch_{result.epoch:04d}.json"
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result.to_dict(),
                file,
                indent=2,
            )

        return destination

    def save_history_csv(
        self,
        filename: str = "validation_history.csv",
    ) -> Path:
        self.report_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            self.report_directory
            / filename
        )

        metric_names = sorted(
            {
                name
                for result in self.history
                for name in result.metrics
            }
        )

        columns = [
            "epoch",
            "sample_count",
            "batch_count",
            "improved",
            "should_stop",
            "best_metric",
            "bad_epochs",
            "timestamp",
        ] + metric_names

        with destination.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=columns,
            )

            writer.writeheader()

            for result in self.history:
                row = result.to_dict()

                metrics = row.pop(
                    "metrics"
                )

                row.update(
                    metrics
                )

                writer.writerow(
                    row
                )

        return destination

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
            "history": [
                result.to_dict()
                for result in self.history
            ],
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.best_metric = state.get(
            "best_metric"
        )

        self.best_epoch = state.get(
            "best_epoch"
        )

        self.bad_epochs = int(
            state.get(
                "bad_epochs", 0
            )
        )

        self.should_stop = bool(
            state.get(
                "should_stop", False
            )
        )

        self.history = [
            ValidationResult(**item)
            for item in state.get(
                "history", []
            )
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
            "validation_runs": len(
                self.history
            ),
            "monitor": self.config.monitor,
            "monitor_mode": (
                self.config.monitor_mode
            ),
        }


def create_validation_runner(
    *,
    device: str = "auto",
    precision: str = "auto",
    monitor: str = "val_loss",
    early_stopping_patience: int | None = 10,
    distributed: bool = False,
) -> ValidationRunner:
    return ValidationRunner(
        ValidationRunnerConfig(
            device=device,
            precision=precision,
            monitor=monitor,
            early_stopping_patience=(
                early_stopping_patience
            ),
            distributed=distributed,
        )
    )


if __name__ == "__main__":
    from torch.utils.data import (
        DataLoader,
        Dataset,
    )

    class DemoDataset(Dataset):
        def __init__(self) -> None:
            self.states = torch.randn(
                128, 128
            )

            self.actions = torch.randn(
                128, 32
            )

            self.next_states = (
                self.states * 0.9
            )

        def __len__(self) -> int:
            return len(
                self.states
            )

        def __getitem__(
            self,
            index: int,
        ) -> dict[str, Tensor]:
            return {
                "states": self.states[index],
                "actions": self.actions[index],
                "next_states": (
                    self.next_states[index]
                ),
            }

    class DemoWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(160, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

        def forward(
            self,
            states: Tensor,
            actions: Tensor,
        ) -> Tensor:
            features = torch.cat(
                [states, actions],
                dim=-1,
            )

            return self.network(
                features
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = DemoWorldModel().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    loader = DataLoader(
        DemoDataset(),
        batch_size=16,
        shuffle=False,
    )

    runner = ValidationRunner(
        ValidationRunnerConfig(
            device=str(device),
            precision="auto",
            monitor="val_loss",
            early_stopping_patience=5,
            save_best_checkpoint=True,
            save_last_checkpoint=True,
        )
    )

    for epoch in range(1, 6):
        result = runner.on_epoch_end(
            epoch=epoch,
            model=model,
            val_loader=loader,
            optimizer=optimizer,
        )

        if result is not None and result.should_stop:
            break

    print(
        json.dumps(
            runner.summary(),
            indent=2,
        )
    )
