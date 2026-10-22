# training/evaluation/benchmark_runner.py

"""
NeuroWorld AI - Benchmark Runner

Features:
- PyTorch model benchmarking
- World-model and dynamics evaluation
- Video prediction evaluation
- Reconstruction evaluation
- Custom benchmark functions
- CPU / CUDA support
- FP32 / FP16 / BF16 inference
- Automatic mixed precision
- Inference latency measurement
- Throughput measurement
- GPU peak memory tracking
- MSE, MAE, RMSE, PSNR metrics
- Optional SSIM metric
- Distributed metric aggregation
- Multiple-model comparison
- JSON and CSV report export
- Warmup iterations
- Repeated benchmark runs
- Configurable batch limits

Requires:
    torch >= 2.1
    numpy
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

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
class BenchmarkConfig:
    device: str = "auto"
    precision: str = "auto"

    warmup_iterations: int = 5
    max_batches: int | None = 100
    repetitions: int = 3

    synchronize_cuda: bool = True
    track_memory: bool = True

    calculate_mse: bool = True
    calculate_mae: bool = True
    calculate_rmse: bool = True
    calculate_psnr: bool = True
    calculate_ssim: bool = False

    pixel_max_value: float = 1.0

    distributed: bool = False

    output_directory: str = (
        "outputs/benchmarks"
    )

    save_json: bool = True
    save_csv: bool = True

    def __post_init__(self) -> None:
        if self.precision not in {
            "auto", "fp32", "fp16", "bf16"
        }:
            raise ValueError(
                "Invalid precision"
            )

        if self.warmup_iterations < 0:
            raise ValueError(
                "warmup_iterations must be >= 0"
            )

        if self.repetitions < 1:
            raise ValueError(
                "repetitions must be >= 1"
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


@dataclass
class BenchmarkResult:
    model_name: str
    benchmark_name: str

    device: str
    precision: str

    samples: int
    batches: int
    repetitions: int

    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    throughput_samples_per_second: float

    peak_gpu_memory_mb: float | None

    metrics: dict[str, float] = field(
        default_factory=dict
    )

    timestamp: str = field(
        default_factory=lambda: (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """
    Benchmark PyTorch AI models.

    Supported input conventions:

        World model:
            batch["states"]
            batch["actions"]
            batch["next_states"]

        Video prediction:
            batch["context_frames"]
            batch["target_frames"]

        Reconstruction:
            batch["frames"]

    Custom forward functions and metrics
    can be registered.

    Important:
        The default runner measures inference,
        not training performance.

        Distributed results assume that all
        ranks execute matching collectives.
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.config = (
            config or BenchmarkConfig()
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

        self.custom_metrics: dict[
            str, MetricFunction
        ] = {}

        self.results: list[
            BenchmarkResult
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
                    "must be initialized first"
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
                    "NCCL benchmarking requires CUDA"
                )

        else:
            self.rank = 0
            self.world_size = 1

        self.is_main_process = (
            self.rank == 0
        )

        if self.is_main_process:
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
                import os

                local_rank = int(
                    os.environ.get(
                        "LOCAL_RANK", "0"
                    )
                )

                return torch.device(
                    "cuda",
                    local_rank,
                )

            return torch.device("cpu")

        device = torch.device(
            requested
        )

        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA is unavailable"
            )

        return device

    def _resolve_precision(
        self,
        requested: str,
    ) -> str:
        if requested != "auto":
            precision = requested
        elif self.device.type == "cuda":
            precision = (
                "bf16"
                if torch.cuda.is_bf16_supported()
                else "fp16"
            )
        else:
            precision = "fp32"

        if (
            precision == "fp16"
            and self.device.type != "cuda"
        ):
            raise RuntimeError(
                "FP16 requires CUDA in this runner"
            )

        if (
            precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 unsupported on this GPU"
            )

        if (
            precision == "bf16"
            and self.device.type != "cuda"
            and self.device.type != "cpu"
        ):
            raise RuntimeError(
                "BF16 supported only on CPU/CUDA"
            )

        return precision

    def register_metric(
        self,
        name: str,
        function: MetricFunction,
    ) -> None:
        if not name:
            raise ValueError(
                "Metric name cannot be empty"
            )

        self.custom_metrics[name] = (
            function
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=(
                self.precision != "fp32"
            ),
        )

    def _synchronize(self) -> None:
        if (
            self.device.type == "cuda"
            and self.config.synchronize_cuda
        ):
            torch.cuda.synchronize(
                self.device
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
    ) -> Tensor | None:
        for key in (
            "next_states",
            "target_frames",
            "targets",
            "target",
            "images",
            "frames",
        ):
            value = batch.get(key)

            if isinstance(value, Tensor):
                return value

        return None

    @staticmethod
    def _batch_size(
        batch: Mapping[str, Any],
    ) -> int:
        for value in batch.values():
            if (
                isinstance(value, Tensor)
                and value.ndim > 0
            ):
                return int(
                    value.shape[0]
                )

        raise ValueError(
            "Cannot determine batch size"
        )

    @staticmethod
    def _mse(
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        return torch.mean(
            (
                prediction.float()
                - target.float()
            ).square()
        )

    @staticmethod
    def _mae(
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        return torch.mean(
            torch.abs(
                prediction.float()
                - target.float()
            )
        )

    def _psnr(
        self,
        mse: Tensor,
    ) -> Tensor:
        if mse.item() <= 0:
            return torch.tensor(
                float("inf"),
                device=mse.device,
            )

        return (
            10.0
            * torch.log10(
                torch.tensor(
                    self.config.pixel_max_value ** 2,
                    device=mse.device,
                ) / mse
            )
        )

    @staticmethod
    def _ssim(
        prediction: Tensor,
        target: Tensor,
        max_value: float = 1.0,
    ) -> Tensor:
        """
        Global SSIM approximation.

        This is not windowed SSIM.
        For research comparisons, use
        a standardized SSIM implementation.
        """
        x = prediction.float()
        y = target.float()

        if x.ndim < 3:
            raise ValueError(
                "SSIM expects image-like tensors"
            )

        reduce_dims = tuple(
            range(
                1,
                x.ndim,
            )
        )

        mu_x = x.mean(
            dim=reduce_dims,
            keepdim=True,
        )

        mu_y = y.mean(
            dim=reduce_dims,
            keepdim=True,
        )

        var_x = (
            (x - mu_x).square()
        ).mean(
            dim=reduce_dims,
            keepdim=True,
        )

        var_y = (
            (y - mu_y).square()
        ).mean(
            dim=reduce_dims,
            keepdim=True,
        )

        covariance = (
            (x - mu_x)
            * (y - mu_y)
        ).mean(
            dim=reduce_dims,
            keepdim=True,
        )

        c1 = (
            0.01 * max_value
        ) ** 2

        c2 = (
            0.03 * max_value
        ) ** 2

        numerator = (
            (2 * mu_x * mu_y + c1)
            * (2 * covariance + c2)
        )

        denominator = (
            (
                mu_x.square()
                + mu_y.square()
                + c1
            )
            * (
                var_x
                + var_y
                + c2
            )
        )

        return (
            numerator
            / denominator.clamp_min(1e-12)
        ).mean()

    def _calculate_metrics(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> dict[str, float]:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ: "
                f"{prediction.shape} vs {target.shape}"
            )

        metrics = {}

        mse = self._mse(
            prediction,
            target,
        )

        if self.config.calculate_mse:
            metrics["mse"] = float(
                mse.item()
            )

        if self.config.calculate_mae:
            metrics["mae"] = float(
                self._mae(
                    prediction,
                    target,
                ).item()
            )

        if self.config.calculate_rmse:
            metrics["rmse"] = float(
                torch.sqrt(mse).item()
            )

        if self.config.calculate_psnr:
            metrics["psnr"] = float(
                self._psnr(mse).item()
            )

        if self.config.calculate_ssim:
            metrics["ssim"] = float(
                self._ssim(
                    prediction,
                    target,
                    self.config.pixel_max_value,
                ).item()
            )

        for name, function in (
            self.custom_metrics.items()
        ):
            value = function(
                prediction,
                target,
            )

            metrics[name] = float(
                value.item()
                if isinstance(value, Tensor)
                else value
            )

        return metrics

    def _warmup(
        self,
        model: nn.Module,
        loader: DataLoader,
        forward_fn: ForwardFunction,
    ) -> None:
        if self.config.warmup_iterations == 0:
            return

        iterator = iter(loader)

        for _ in range(
            self.config.warmup_iterations
        ):
            try:
                batch = next(iterator)

            except StopIteration:
                iterator = iter(loader)

                try:
                    batch = next(iterator)
                except StopIteration:
                    raise ValueError(
                        "Benchmark loader is empty"
                    )

            batch = self._move_to_device(
                batch
            )

            with self._autocast():
                forward_fn(
                    model,
                    batch,
                )

        self._synchronize()

    def _distributed_sum(
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

    def _distributed_max(
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
            op=dist.ReduceOp.MAX,
        )

        return float(
            tensor.item()
        )

    def _gather_latencies(
        self,
        latencies: list[float],
    ) -> list[float]:
        if not self.distributed:
            return latencies

        gathered: list[Any] = [
            None
            for _ in range(
                self.world_size
            )
        ]

        dist.all_gather_object(
            gathered,
            latencies,
        )

        return [
            float(value)
            for rank_values in gathered
            for value in rank_values
        ]

    @torch.inference_mode()
    def run(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        *,
        model_name: str = "world_model",
        benchmark_name: str = "default",
        forward_fn: ForwardFunction | None = None,
    ) -> BenchmarkResult:
        forward_fn = (
            forward_fn or self._default_forward
        )

        model = model.to(
            self.device
        )

        original_training = (
            model.training
        )

        model.eval()

        local_latencies: list[float] = []

        metric_sums: dict[
            str, float
        ] = {}

        metric_weights: dict[
            str, float
        ] = {}

        local_samples = 0
        local_batches = 0

        total_execution_time = 0.0

        peak_memory = None

        try:
            self._warmup(
                model,
                dataloader,
                forward_fn,
            )

            if (
                self.device.type == "cuda"
                and self.config.track_memory
            ):
                torch.cuda.reset_peak_memory_stats(
                    self.device
                )

            for repetition in range(
                self.config.repetitions
            ):
                if self.distributed:
                    dist.barrier()

                repetition_time = 0.0

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

                    batch_size = self._batch_size(
                        batch
                    )

                    self._synchronize()

                    start = time.perf_counter()

                    with self._autocast():
                        output = forward_fn(
                            model,
                            batch,
                        )

                    self._synchronize()

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    latency_ms = (
                        elapsed * 1000.0
                    )

                    local_latencies.append(
                        latency_ms
                    )

                    repetition_time += elapsed

                    local_samples += batch_size
                    local_batches += 1

                    target = self._extract_target(
                        batch
                    )

                    if target is not None:
                        prediction = (
                            self._extract_prediction(
                                output
                            )
                        )

                        batch_metrics = (
                            self._calculate_metrics(
                                prediction,
                                target,
                            )
                        )

                        for key, value in (
                            batch_metrics.items()
                        ):
                            metric_sums[key] = (
                                metric_sums.get(
                                    key, 0.0
                                )
                                + value * batch_size
                            )

                            metric_weights[key] = (
                                metric_weights.get(
                                    key, 0.0
                                )
                                + batch_size
                            )

                if local_batches == 0:
                    raise ValueError(
                        "Benchmark produced no batches"
                    )

                # Distributed throughput uses
                # the slowest rank's duration.
                repetition_time = (
                    self._distributed_max(
                        repetition_time
                    )
                )

                total_execution_time += (
                    repetition_time
                )

            if (
                self.device.type == "cuda"
                and self.config.track_memory
            ):
                peak_memory = (
                    torch.cuda.max_memory_allocated(
                        self.device
                    ) / (1024 ** 2)
                )

                peak_memory = (
                    self._distributed_max(
                        peak_memory
                    )
                )

            total_samples = int(
                self._distributed_sum(
                    float(local_samples)
                )
            )

            total_batches = int(
                self._distributed_sum(
                    float(local_batches)
                )
            )

            latencies = (
                self._gather_latencies(
                    local_latencies
                )
            )

            if not latencies:
                raise RuntimeError(
                    "No latency measurements"
                )

            metric_names = sorted(
                metric_sums.keys()
            )

            if self.distributed:
                gathered_names: list[Any] = [
                    None
                    for _ in range(
                        self.world_size
                    )
                ]

                dist.all_gather_object(
                    gathered_names,
                    metric_names,
                )

                metric_names = sorted(
                    {
                        name
                        for names in gathered_names
                        for name in names
                    }
                )

            final_metrics = {}

            for name in metric_names:
                global_sum = (
                    self._distributed_sum(
                        metric_sums.get(
                            name, 0.0
                        )
                    )
                )

                global_weight = (
                    self._distributed_sum(
                        metric_weights.get(
                            name, 0.0
                        )
                    )
                )

                if global_weight > 0:
                    final_metrics[name] = (
                        global_sum
                        / global_weight
                    )

            throughput = (
                total_samples
                / total_execution_time
                if total_execution_time > 0
                else 0.0
            )

            result = BenchmarkResult(
                model_name=model_name,
                benchmark_name=benchmark_name,
                device=str(self.device),
                precision=self.precision,
                samples=total_samples,
                batches=total_batches,
                repetitions=(
                    self.config.repetitions
                ),
                latency_mean_ms=(
                    statistics.mean(latencies)
                ),
                latency_median_ms=(
                    statistics.median(latencies)
                ),
                latency_p95_ms=float(
                    np.percentile(
                        latencies, 95
                    )
                ),
                latency_p99_ms=float(
                    np.percentile(
                        latencies, 99
                    )
                ),
                throughput_samples_per_second=(
                    throughput
                ),
                peak_gpu_memory_mb=peak_memory,
                metrics=final_metrics,
            )

            self.results.append(
                result
            )

            if self.is_main_process:
                self._print_result(
                    result
                )

                if self.config.save_json:
                    self.save_result_json(
                        result
                    )

                if self.config.save_csv:
                    self.save_results_csv()

            return result

        finally:
            model.train(
                original_training
            )

    @staticmethod
    def _print_result(
        result: BenchmarkResult,
    ) -> None:
        print(
            "\nNeuroWorld AI Benchmark"
        )

        print(
            f"Model: {result.model_name}"
        )

        print(
            f"Benchmark: {result.benchmark_name}"
        )

        print(
            f"Device: {result.device}"
        )

        print(
            f"Precision: {result.precision}"
        )

        print(
            f"Samples: {result.samples}"
        )

        print(
            "Latency mean: "
            f"{result.latency_mean_ms:.3f} ms"
        )

        print(
            "Latency P95: "
            f"{result.latency_p95_ms:.3f} ms"
        )

        print(
            "Throughput: "
            f"{result.throughput_samples_per_second:.2f} "
            "samples/sec"
        )

        if (
            result.peak_gpu_memory_mb
            is not None
        ):
            print(
                "Peak GPU memory: "
                f"{result.peak_gpu_memory_mb:.2f} MB"
            )

        for name, value in (
            result.metrics.items()
        ):
            print(
                f"{name}: {value:.6f}"
            )

    def save_result_json(
        self,
        result: BenchmarkResult,
    ) -> Path:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = (
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
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
                allow_nan=True,
            )

        return destination

    def save_results_csv(
        self,
        filename: str = "benchmark_results.csv",
    ) -> Path:
        destination = (
            self.output_directory
            / filename
        )

        metric_names = sorted(
            {
                metric
                for result in self.results
                for metric in result.metrics
            }
        )

        base_fields = [
            "model_name",
            "benchmark_name",
            "device",
            "precision",
            "samples",
            "batches",
            "repetitions",
            "latency_mean_ms",
            "latency_median_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "throughput_samples_per_second",
            "peak_gpu_memory_mb",
            "timestamp",
        ]

        fieldnames = (
            base_fields
            + metric_names
        )

        with destination.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in self.results:
                row = result.to_dict()

                metrics = row.pop(
                    "metrics",
                    {},
                )

                row.update(metrics)

                writer.writerow(row)

        return destination

    def compare_models(
        self,
        models: Mapping[str, nn.Module],
        dataloader: DataLoader,
        *,
        forward_fn: ForwardFunction | None = None,
    ) -> list[BenchmarkResult]:
        comparison = []

        for name, model in models.items():
            result = self.run(
                model,
                dataloader,
                model_name=name,
                benchmark_name=(
                    "model_comparison"
                ),
                forward_fn=forward_fn,
            )

            comparison.append(
                result
            )

        return comparison

    def summary(
        self,
    ) -> list[dict[str, Any]]:
        return [
            result.to_dict()
            for result in self.results
        ]


def create_benchmark_runner(
    *,
    device: str = "auto",
    precision: str = "auto",
    repetitions: int = 3,
    max_batches: int | None = 100,
) -> BenchmarkRunner:
    return BenchmarkRunner(
        BenchmarkConfig(
            device=device,
            precision=precision,
            repetitions=repetitions,
            max_batches=max_batches,
        )
    )


if __name__ == "__main__":
    from torch.utils.data import (
        DataLoader,
        TensorDataset,
    )

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
                nn.Linear(256, 256),
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
            return self.network(
                torch.cat(
                    [states, actions],
                    dim=-1,
                )
            )

    class DemoDataset(
        torch.utils.data.Dataset
    ):
        def __init__(self) -> None:
            self.states = torch.randn(
                256, 128
            )

            self.actions = torch.randn(
                256, 32
            )

            self.next_states = torch.randn(
                256, 128
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

    model = DemoWorldModel()

    loader = DataLoader(
        DemoDataset(),
        batch_size=16,
        shuffle=False,
    )

    runner = BenchmarkRunner(
        BenchmarkConfig(
            device="auto",
            precision="auto",
            warmup_iterations=3,
            max_batches=10,
            repetitions=2,
            calculate_ssim=False,
            save_json=True,
            save_csv=True,
        )
    )

    result = runner.run(
        model,
        loader,
        model_name="demo_world_model",
        benchmark_name="dynamics_prediction",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )