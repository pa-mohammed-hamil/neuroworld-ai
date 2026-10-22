
"""
NeuroWorld AI
File: training/evaluation/training_evaluator.py

Training evaluation for world models, dynamics models,
video prediction, reconstruction, and reward models.

Features:
- PyTorch model evaluation
- Validation and test dataset support
- World-model and video-prediction metrics
- MSE, MAE, RMSE, R2, PSNR
- Masked sequence evaluation
- Custom metrics and forward functions
- FP32, FP16 and BF16 inference
- CPU / CUDA support
- Distributed evaluation
- Weighted metric aggregation
- Per-batch evaluation records
- JSON and CSV report export
- Multiple-model comparison
- Best-model selection
- Evaluation history tracking

Requires:
    torch >= 2.1
    numpy
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import Tensor, nn
from torch.utils.data import DataLoader


ForwardFunction = Callable[
    [nn.Module, Mapping[str, Any]],
    Any,
]

MetricFunction = Callable[
    [Tensor, Tensor],
    Tensor | float,
]


@dataclass
class TrainingEvaluatorConfig:
    device: str = "auto"
    precision: str = "auto"

    max_batches: int | None = None

    metrics: tuple[str, ...] = (
        "mse",
        "mae",
        "rmse",
        "r2",
    )

    pixel_max_value: float = 1.0

    distributed: bool = False

    save_reports: bool = True
    save_batch_metrics: bool = False

    output_directory: str = (
        "outputs/training_evaluation"
    )

    monitor: str = "mse"
    monitor_mode: str = "min"

    def __post_init__(self) -> None:
        supported = {
            "mse",
            "mae",
            "rmse",
            "r2",
            "psnr",
        }

        invalid = (
            set(self.metrics) - supported
        )

        if invalid:
            raise ValueError(
                f"Unsupported metrics: {invalid}"
            )

        if self.precision not in {
            "auto",
            "fp32",
            "fp16",
            "bf16",
        }:
            raise ValueError(
                "Unsupported precision"
            )

        if (
            self.max_batches is not None
            and self.max_batches < 1
        ):
            raise ValueError(
                "max_batches must be >= 1"
            )

        if self.pixel_max_value <= 0:
            raise ValueError(
                "pixel_max_value must be positive"
            )

        if self.monitor_mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "monitor_mode must be min or max"
            )


@dataclass
class EvaluationResult:
    model_name: str
    split: str

    epoch: int | None

    sample_count: int
    batch_count: int

    metrics: dict[str, float]

    device: str
    precision: str

    timestamp: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    batch_metrics: list[
        dict[str, float]
    ] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingEvaluator:
    """
    Evaluate NeuroWorld AI models.

    Default batch formats:

    Dynamics:
        states
        actions
        next_states

    Video prediction:
        context_frames
        target_frames

    Reconstruction:
        frames

    Reward prediction:
        states
        actions
        rewards

    Optional sequence masks:
        mask
        frame_mask

    Custom forward functions support
    other model architectures.
    """

    def __init__(
        self,
        config: (
            TrainingEvaluatorConfig | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or TrainingEvaluatorConfig()
        )

        self.device = self._resolve_device(
            self.config.device
        )

        self.precision = (
            self._resolve_precision(
                self.config.precision
            )
        )

        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]

        self.custom_metrics: dict[
            str, MetricFunction
        ] = {}

        self.history: list[
            EvaluationResult
        ] = []

        self.output_directory = Path(
            self.config.output_directory
        )

        self.distributed = (
            self.config.distributed
        )

        if self.distributed:
            if (
                not dist.is_available()
                or not dist.is_initialized()
            ):
                raise RuntimeError(
                    "Distributed process group "
                    "must be initialized"
                )

            self.rank = dist.get_rank()
            self.world_size = (
                dist.get_world_size()
            )

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

        if (
            self.config.save_reports
            and self.is_main_process
        ):
            self.output_directory.mkdir(
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

                if (
                    local_rank
                    >= torch.cuda.device_count()
                ):
                    raise RuntimeError(
                        "Invalid LOCAL_RANK"
                    )

                return torch.device(
                    "cuda",
                    local_rank,
                )

            return torch.device("cpu")

        device = torch.device(
            requested
        )

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA unavailable"
                )

        if device.type not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "Only CPU/CUDA supported"
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
                "BF16 unsupported"
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
            "mse",
            "mae",
            "rmse",
            "r2",
            "psnr",
        }:
            raise ValueError(
                "Cannot override built-in metric"
            )

        self.custom_metrics[name] = (
            function
        )

    def _move_to_device(
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
            return model(
                batch["frames"]
            )

        if "images" in batch:
            return model(
                batch["images"]
            )

        raise ValueError(
            "Unsupported batch structure. "
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
            "Cannot extract model predictions"
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
            "Cannot find evaluation target"
        )

    @staticmethod
    def _extract_mask(
        batch: Mapping[str, Any],
    ) -> Tensor | None:
        mask = batch.get(
            "mask",
            batch.get("frame_mask"),
        )

        if mask is None:
            return None

        if not isinstance(mask, Tensor):
            raise TypeError(
                "Mask must be a Tensor"
            )

        return mask

    @staticmethod
    def _broadcast_mask(
        mask: Tensor | None,
        target: Tensor,
    ) -> Tensor:
        if mask is None:
            return torch.ones_like(
                target,
                dtype=torch.bool,
            )

        mask = mask.bool()

        if mask.shape[0] != target.shape[0]:
            raise ValueError(
                "Mask batch size mismatch"
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
                "Mask cannot broadcast "
                "to target shape"
            ) from exc

    @staticmethod
    def _batch_statistics(
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None,
    ) -> dict[str, float]:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction/target shape mismatch: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        prediction = prediction.float()
        target = target.float()

        valid = (
            TrainingEvaluator._broadcast_mask(
                mask,
                target,
            )
        )

        valid_count = int(
            valid.sum().item()
        )

        if valid_count == 0:
            return {
                "count": 0.0,
                "squared_error": 0.0,
                "absolute_error": 0.0,
                "target_sum": 0.0,
                "target_squared_sum": 0.0,
            }

        error = prediction - target

        selected_error = error[valid]
        selected_target = target[valid]

        return {
            "count": float(valid_count),
            "squared_error": float(
                selected_error.square()
                .sum(dtype=torch.float64)
                .item()
            ),
            "absolute_error": float(
                selected_error.abs()
                .sum(dtype=torch.float64)
                .item()
            ),
            "target_sum": float(
                selected_target
                .sum(dtype=torch.float64)
                .item()
            ),
            "target_squared_sum": float(
                selected_target.square()
                .sum(dtype=torch.float64)
                .item()
            ),
        }

    def _aggregate_statistics(
        self,
        statistics: dict[str, float],
    ) -> dict[str, float]:
        keys = [
            "count",
            "squared_error",
            "absolute_error",
            "target_sum",
            "target_squared_sum",
        ]

        values = torch.tensor(
            [
                statistics.get(key, 0.0)
                for key in keys
            ],
            dtype=torch.float64,
            device=self.device,
        )

        if self.distributed:
            dist.all_reduce(
                values,
                op=dist.ReduceOp.SUM,
            )

        return {
            key: float(values[index].item())
            for index, key in enumerate(keys)
        }

    def _finalize_metrics(
        self,
        statistics: dict[str, float],
    ) -> dict[str, float]:
        count = statistics["count"]

        if count <= 0:
            return {
                metric: float("nan")
                for metric in self.config.metrics
            }

        mse = (
            statistics["squared_error"]
            / count
        )

        mae = (
            statistics["absolute_error"]
            / count
        )

        rmse = math.sqrt(
            max(mse, 0.0)
        )

        target_mean = (
            statistics["target_sum"]
            / count
        )

        target_variance_sum = (
            statistics["target_squared_sum"]
            - count * target_mean**2
        )

        target_variance_sum = max(
            target_variance_sum,
            0.0,
        )

        if target_variance_sum > 0:
            r2 = (
                1.0
                - statistics["squared_error"]
                / target_variance_sum
            )

        elif statistics["squared_error"] == 0:
            r2 = 1.0

        else:
            r2 = 0.0

        psnr = (
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

        available = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "psnr": psnr,
        }

        return {
            name: available[name]
            for name in self.config.metrics
        }

    @staticmethod
    def _batch_size(
        batch: Mapping[str, Any],
    ) -> int:
        for value in batch.values():
            if (
                isinstance(value, Tensor)
                and value.ndim >= 1
            ):
                return int(
                    value.shape[0]
                )

        raise ValueError(
            "Cannot determine batch size"
        )

    def _reduce_sum(
        self,
        value: float,
    ) -> float:
        if not self.distributed:
            return value

        tensor = torch.tensor(
            value,
            dtype=torch.float64,
            device=self.device,
        )

        dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
        )

        return float(
            tensor.item()
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        *,
        model_name: str = "world_model",
        split: str = "validation",
        epoch: int | None = None,
        forward_fn: ForwardFunction | None = None,
    ) -> EvaluationResult:
        """
        Evaluate one model on a dataset.

        In distributed mode, every rank
        must call evaluate().

        The dataloader must be partitioned
        appropriately across ranks.

        Standard DistributedSampler may pad
        and duplicate evaluation examples.
        Use a non-padding evaluation sampler
        for exact dataset-wide metrics.
        """
        forward_fn = (
            forward_fn or self._default_forward
        )

        original_training = model.training

        # Do not move an already-wrapped
        # DDP/FSDP model between devices.
        if not self.distributed:
            model = model.to(
                self.device
            )

        model.eval()

        totals = defaultdict(float)

        sample_count = 0
        batch_count = 0

        batch_records = []

        custom_sums = defaultdict(float)
        custom_weights = defaultdict(float)

        try:
            for batch_index, raw_batch in enumerate(
                dataloader
            ):
                if (
                    self.config.max_batches
                    is not None
                    and batch_index
                    >= self.config.max_batches
                ):
                    break

                batch = self._move_to_device(
                    raw_batch
                )

                with self._autocast():
                    output = forward_fn(
                        model,
                        batch,
                    )

                prediction = (
                    self._extract_prediction(
                        output
                    )
                )

                target = self._extract_target(
                    batch
                )

                mask = self._extract_mask(
                    batch
                )

                statistics = (
                    self._batch_statistics(
                        prediction,
                        target,
                        mask,
                    )
                )

                for key, value in (
                    statistics.items()
                ):
                    totals[key] += value

                current_batch_size = (
                    self._batch_size(
                        batch
                    )
                )

                sample_count += (
                    current_batch_size
                )

                batch_count += 1

                if self.config.save_batch_metrics:
                    record = (
                        self._finalize_metrics(
                            statistics
                        )
                    )

                    record["batch"] = float(
                        batch_index
                    )

                    record["rank"] = float(
                        self.rank
                    )

                    batch_records.append(
                        record
                    )

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
                                f"Metric {name} "
                                "must return a scalar"
                            )

                        value = float(
                            value.item()
                        )

                    else:
                        value = float(value)

                    custom_sums[name] += (
                        value
                        * current_batch_size
                    )

                    custom_weights[name] += (
                        current_batch_size
                    )

            global_statistics = (
                self._aggregate_statistics(
                    totals
                )
            )

            metrics = self._finalize_metrics(
                global_statistics
            )

            for name in sorted(
                self.custom_metrics
            ):
                total_value = (
                    self._reduce_sum(
                        custom_sums[name]
                    )
                )

                total_weight = (
                    self._reduce_sum(
                        custom_weights[name]
                    )
                )

                metrics[name] = (
                    total_value / total_weight
                    if total_weight > 0
                    else float("nan")
                )

            global_samples = int(
                self._reduce_sum(
                    float(sample_count)
                )
            )

            global_batches = int(
                self._reduce_sum(
                    float(batch_count)
                )
            )

            if (
                self.distributed
                and self.config.save_batch_metrics
            ):
                gathered: list[Any] = [
                    None
                    for _ in range(
                        self.world_size
                    )
                ]

                dist.all_gather_object(
                    gathered,
                    batch_records,
                )

                batch_records = [
                    record
                    for records in gathered
                    for record in records
                ]

            result = EvaluationResult(
                model_name=model_name,
                split=split,
                epoch=epoch,
                sample_count=global_samples,
                batch_count=global_batches,
                metrics=metrics,
                device=str(self.device),
                precision=self.precision,
                batch_metrics=batch_records,
            )

            self.history.append(
                result
            )

            if self.is_main_process:
                self.print_result(
                    result
                )

                if self.config.save_reports:
                    self.save_json(
                        result
                    )

                    self.save_history_csv()

            return result

        finally:
            model.train(
                original_training
            )

    def compare_models(
        self,
        models: Mapping[str, nn.Module],
        dataloader: DataLoader,
        *,
        split: str = "test",
        forward_fn: ForwardFunction | None = None,
    ) -> dict[str, EvaluationResult]:
        results = {}

        for name, model in models.items():
            results[name] = self.evaluate(
                model,
                dataloader,
                model_name=name,
                split=split,
                forward_fn=forward_fn,
            )

        return results

    def best_result(
        self,
        *,
        metric: str | None = None,
        mode: str | None = None,
        split: str | None = None,
    ) -> EvaluationResult | None:
        metric = (
            metric or self.config.monitor
        )

        mode = (
            mode or self.config.monitor_mode
        )

        if mode not in {"min", "max"}:
            raise ValueError(
                "mode must be min or max"
            )

        candidates = [
            result
            for result in self.history
            if metric in result.metrics
            and math.isfinite(
                result.metrics[metric]
            )
            and (
                split is None
                or result.split == split
            )
        ]

        if not candidates:
            return None

        return (
            min(
                candidates,
                key=lambda result: (
                    result.metrics[metric]
                ),
            )
            if mode == "min"
            else max(
                candidates,
                key=lambda result: (
                    result.metrics[metric]
                ),
            )
        )

    @staticmethod
    def print_result(
        result: EvaluationResult,
    ) -> None:
        print(
            "\nNeuroWorld AI - Evaluation"
        )

        print(
            f"Model: {result.model_name}"
        )

        print(
            f"Split: {result.split}"
        )

        print(
            f"Epoch: {result.epoch}"
        )

        print(
            f"Samples: {result.sample_count}"
        )

        print(
            f"Device: {result.device}"
        )

        print(
            f"Precision: {result.precision}"
        )

        for name, value in (
            result.metrics.items()
        ):
            print(
                f"{name}: {value:.6f}"
            )

    def save_json(
        self,
        result: EvaluationResult,
    ) -> Path:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        safe_name = "".join(
            character
            if character.isalnum()
            or character in "-_"
            else "_"
            for character in result.model_name
        )

        destination = (
            self.output_directory
            / f"{safe_name}_{timestamp}.json"
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
        filename: str = (
            "evaluation_history.csv"
        ),
    ) -> Path:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            self.output_directory
            / filename
        )

        metric_names = sorted(
            {
                metric
                for result in self.history
                for metric in result.metrics
            }
        )

        columns = [
            "model_name",
            "split",
            "epoch",
            "sample_count",
            "batch_count",
            "device",
            "precision",
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

                row.pop(
                    "batch_metrics",
                    None,
                )

                row.update(metrics)

                writer.writerow(row)

        return destination

    def history_summary(
        self,
    ) -> list[dict[str, Any]]:
        return [
            result.to_dict()
            for result in self.history
        ]


def create_training_evaluator(
    *,
    device: str = "auto",
    precision: str = "auto",
    distributed: bool = False,
    output_directory: str = (
        "outputs/training_evaluation"
    ),
) -> TrainingEvaluator:
    return TrainingEvaluator(
        TrainingEvaluatorConfig(
            device=device,
            precision=precision,
            distributed=distributed,
            output_directory=output_directory,
        )
    )


if __name__ == "__main__":
    from torch.utils.data import (
        DataLoader,
        Dataset,
    )

    class DemoDataset(Dataset):
        def __init__(
            self,
            size: int = 128,
            state_dim: int = 128,
            action_dim: int = 32,
        ) -> None:
            self.states = torch.randn(
                size,
                state_dim,
            )

            self.actions = torch.randn(
                size,
                action_dim,
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
        def __init__(
            self,
            state_dim: int = 128,
            action_dim: int = 32,
        ) -> None:
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(
                    state_dim + action_dim,
                    256,
                ),
                nn.ReLU(),
                nn.Linear(
                    256,
                    state_dim,
                ),
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

    dataset = DemoDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    model = DemoWorldModel()

    evaluator = TrainingEvaluator(
        TrainingEvaluatorConfig(
            device="auto",
            precision="auto",
            metrics=(
                "mse",
                "mae",
                "rmse",
                "r2",
            ),
            save_reports=True,
        )
    )

    result = evaluator.evaluate(
        model,
        dataloader,
        model_name="demo_world_model",
        split="validation",
        epoch=1,
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )
