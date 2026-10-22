
"""
NeuroWorld AI
File: training/distributed/ddp_trainer.py

Distributed Data Parallel trainer.

Features:
- Single GPU, CPU, and multi-GPU training
- torchrun / DDP initialization
- NCCL or Gloo backend
- DistributedSampler integration
- Mixed precision (FP16 / BF16)
- Gradient accumulation
- Gradient clipping
- Distributed metric reduction
- Validation across all processes
- Rank-zero checkpoint saving
- Resume model, optimizer, scheduler,
  scaler, and training progress
- Optional early stopping
- Flexible model output/loss handling

Dependencies:
    torch >= 2.1

Launch:
    torchrun --standalone --nproc_per_node=2 \
        training/train_world_model.py
"""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist

from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


LossFunction = Callable[
    [Any, Mapping[str, Any]],
    Tensor,
]


@dataclass
class DDPTrainerConfig:
    epochs: int = 100

    backend: str = "auto"

    learning_rate: float = 3e-4

    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0

    log_interval: int = 10

    checkpoint_dir: str = (
        "checkpoints/distributed"
    )

    checkpoint_interval: int = 5

    monitor: str = "val_loss"
    monitor_mode: str = "min"

    early_stopping_patience: int | None = None

    seed: int = 42
    deterministic: bool = False

    find_unused_parameters: bool = False
    broadcast_buffers: bool = True

    scheduler_step: str = "epoch"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(
                "epochs must be >= 1"
            )

        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps "
                "must be >= 1"
            )

        if self.mixed_precision not in (
            "none",
            "fp16",
            "bf16",
        ):
            raise ValueError(
                "mixed_precision must be "
                "none, fp16, or bf16"
            )

        if self.monitor_mode not in (
            "min",
            "max",
        ):
            raise ValueError(
                "monitor_mode must be min or max"
            )

        if self.scheduler_step not in (
            "epoch",
            "batch",
        ):
            raise ValueError(
                "scheduler_step must be "
                "epoch or batch"
            )

        if self.checkpoint_interval < 1:
            raise ValueError(
                "checkpoint_interval must be >= 1"
            )


class DDPTrainer:
    """
    Generic DDP training engine.

    The loss function receives:
        loss_fn(model_output, batch)

    It must return a scalar Tensor.

    Default batch inputs:
        batch["states"]
        batch["actions"]

    Default targets:
        batch["next_states"]

    Custom forward_fn supports other models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        loss_fn: LossFunction | None = None,
        scheduler: Any | None = None,
        config: DDPTrainerConfig | None = None,
        forward_fn: Callable[
            [nn.Module, Mapping[str, Any]],
            Any,
        ] | None = None,
    ) -> None:
        self.config = (
            config or DDPTrainerConfig()
        )

        self._owns_process_group = False

        self._initialize_distributed()

        self._set_seed(
            self.config.seed + self.rank
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = loss_fn
        self.forward_fn = (
            forward_fn or self._default_forward
        )

        self.raw_model = model.to(
            self.device
        )

        if self.distributed:
            self.model = DistributedDataParallel(
                self.raw_model,
                device_ids=(
                    [self.local_rank]
                    if self.device.type == "cuda"
                    else None
                ),
                output_device=(
                    self.local_rank
                    if self.device.type == "cuda"
                    else None
                ),
                find_unused_parameters=(
                    self.config.find_unused_parameters
                ),
                broadcast_buffers=(
                    self.config.broadcast_buffers
                ),
            )
        else:
            self.model = self.raw_model

        self._configure_precision()

        self.start_epoch = 0
        self.global_step = 0

        self.best_metric = (
            float("inf")
            if self.config.monitor_mode == "min"
            else float("-inf")
        )

        self.bad_epochs = 0

        self.history: list[
            dict[str, float]
        ] = []

        self.checkpoint_dir = Path(
            self.config.checkpoint_dir
        )

        if self.is_main_process:
            self.checkpoint_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        if self.distributed:
            dist.barrier()

        self._validate_sampler(
            self.train_loader,
            "train",
        )

        if self.val_loader is not None:
            self._validate_sampler(
                self.val_loader,
                "validation",
            )

    def _initialize_distributed(
        self,
    ) -> None:
        self.world_size = int(
            os.environ.get(
                "WORLD_SIZE",
                "1",
            )
        )

        self.rank = int(
            os.environ.get(
                "RANK",
                "0",
            )
        )

        self.local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                "0",
            )
        )

        self.distributed = (
            self.world_size > 1
        )

        if torch.cuda.is_available():
            if self.distributed:
                torch.cuda.set_device(
                    self.local_rank
                )

                self.device = torch.device(
                    "cuda",
                    self.local_rank,
                )
            else:
                self.device = torch.device(
                    "cuda",
                    0,
                )
        else:
            self.device = torch.device(
                "cpu"
            )

        if self.distributed:
            backend = self.config.backend

            if backend == "auto":
                backend = (
                    "nccl"
                    if self.device.type == "cuda"
                    else "gloo"
                )

            if not dist.is_initialized():
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                )

                self._owns_process_group = True

            self.rank = dist.get_rank()
            self.world_size = (
                dist.get_world_size()
            )

        self.is_main_process = (
            self.rank == 0
        )

    def _set_seed(
        self,
        seed: int,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            torch.use_deterministic_algorithms(
                True
            )

    def _configure_precision(
        self,
    ) -> None:
        precision = (
            self.config.mixed_precision
        )

        self.use_amp = (
            precision != "none"
            and self.device.type == "cuda"
        )

        if (
            self.use_amp
            and precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 is not supported on this GPU. "
                "Use fp16 or none."
            )

        self.amp_dtype = (
            torch.float16
            if precision == "fp16"
            else torch.bfloat16
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.use_amp
                and precision == "fp16"
            ),
        )

    def _validate_sampler(
        self,
        loader: DataLoader,
        name: str,
    ) -> None:
        if not self.distributed:
            return

        sampler = getattr(
            loader,
            "sampler",
            None,
        )

        if not isinstance(
            sampler,
            DistributedSampler,
        ):
            raise ValueError(
                f"{name} loader must use "
                "DistributedSampler in DDP mode. "
                "All ranks must receive the same "
                "number of training batches."
            )

        if (
            sampler.num_replicas
            != self.world_size
            or sampler.rank != self.rank
        ):
            raise ValueError(
                f"{name} sampler rank/world_size "
                "does not match process group"
            )

    @staticmethod
    def _default_forward(
        model: nn.Module,
        batch: Mapping[str, Any],
    ) -> Any:
        """
        Default for action-conditioned
        dynamics/world models.

        Override for video or policy models.
        """
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

        raise ValueError(
            "No supported default model inputs. "
            "Provide forward_fn."
        )

    @staticmethod
    def _default_loss(
        output: Any,
        batch: Mapping[str, Any],
    ) -> Tensor:
        """
        Default masked MSE loss.

        For stochastic dynamics, custom
        KL/NLL losses should be supplied.
        """
        if "next_states" in batch:
            target = batch[
                "next_states"
            ]
        elif "target_frames" in batch:
            target = batch[
                "target_frames"
            ]
        else:
            raise ValueError(
                "No default target found. "
                "Provide loss_fn."
            )

        if isinstance(output, Mapping):
            if "loss" in output:
                return output["loss"]

            if "predictions" in output:
                prediction = output[
                    "predictions"
                ]
            elif "next_states" in output:
                prediction = output[
                    "next_states"
                ]
            else:
                raise ValueError(
                    "Cannot extract prediction "
                    "from model output"
                )

        elif isinstance(
            output,
            (tuple, list),
        ):
            prediction = output[0]

        else:
            prediction = output

        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes "
                f"differ: {prediction.shape} "
                f"vs {target.shape}"
            )

        squared_error = (
            prediction - target
        ).square()

        mask = batch.get(
            "mask",
            batch.get("frame_mask"),
        )

        if mask is None:
            return squared_error.mean()

        while mask.ndim < squared_error.ndim:
            mask = mask.unsqueeze(-1)

        mask = mask.to(
            dtype=squared_error.dtype
        )

        expanded_mask = mask.expand_as(
            squared_error
        )

        denominator = expanded_mask.sum().clamp_min(
            1.0
        )

        return (
            squared_error * expanded_mask
        ).sum() / denominator

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

        if isinstance(value, tuple):
            return tuple(
                self._move_to_device(item)
                for item in value
            )

        if isinstance(value, list):
            return [
                self._move_to_device(item)
                for item in value
            ]

        return value

    def _autocast(
        self,
    ):
        if not self.use_amp:
            return contextlib.nullcontext()

        return torch.autocast(
            device_type="cuda",
            dtype=self.amp_dtype,
        )

    def _compute_loss(
        self,
        batch: Mapping[str, Any],
    ) -> Tensor:
        output = self.forward_fn(
            self.model,
            batch,
        )

        criterion = (
            self.loss_fn
            or self._default_loss
        )

        loss = criterion(
            output,
            batch,
        )

        if not isinstance(loss, Tensor):
            raise TypeError(
                "Loss function must return "
                "a PyTorch Tensor"
            )

        if loss.ndim != 0:
            raise ValueError(
                "Loss must be a scalar Tensor"
            )

        return loss

    def _reduce_metrics(
        self,
        loss_sum: float,
        count: int,
    ) -> float:
        values = torch.tensor(
            [loss_sum, float(count)],
            dtype=torch.float64,
            device=self.device,
        )

        if self.distributed:
            dist.all_reduce(
                values,
                op=dist.ReduceOp.SUM,
            )

        total_count = values[1].item()

        return (
            values[0].item() / total_count
            if total_count > 0
            else float("nan")
        )

    def _train_epoch(
        self,
        epoch: int,
    ) -> float:
        self.model.train()

        sampler = getattr(
            self.train_loader,
            "sampler",
            None,
        )

        if isinstance(
            sampler,
            DistributedSampler,
        ):
            sampler.set_epoch(epoch)

        accumulation = (
            self.config.gradient_accumulation_steps
        )

        total_batches = len(
            self.train_loader
        )

        if total_batches == 0:
            raise ValueError(
                "Training loader is empty"
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss_sum = 0.0
        batch_count = 0

        for batch_index, raw_batch in enumerate(
            self.train_loader
        ):
            batch = self._move_to_device(
                raw_batch
            )

            group_start = (
                batch_index // accumulation
            ) * accumulation

            group_size = min(
                accumulation,
                total_batches - group_start,
            )

            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == total_batches
            )

            sync_context = (
                contextlib.nullcontext()
                if should_step
                or not self.distributed
                else self.model.no_sync()
            )

            with sync_context:
                with self._autocast():
                    loss = self._compute_loss(
                        batch
                    )

                    scaled_loss = (
                        loss / group_size
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite training loss "
                        f"at batch {batch_index}"
                    )

                self.scaler.scale(
                    scaled_loss
                ).backward()

            loss_sum += float(
                loss.detach().item()
            )

            batch_count += 1

            if should_step:
                if (
                    self.config.gradient_clip_norm
                    is not None
                ):
                    self.scaler.unscale_(
                        self.optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                previous_scale = (
                    self.scaler.get_scale()
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                optimizer_stepped = (
                    not self.scaler.is_enabled()
                    or self.scaler.get_scale()
                    >= previous_scale
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                if optimizer_stepped:
                    self.global_step += 1

                    if (
                        self.scheduler is not None
                        and self.config.scheduler_step
                        == "batch"
                    ):
                        self.scheduler.step()

            if (
                self.is_main_process
                and (
                    batch_index + 1
                ) % self.config.log_interval == 0
            ):
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_index + 1}/"
                    f"{total_batches} | "
                    f"Loss {loss.item():.6f}",
                    flush=True,
                )

        return self._reduce_metrics(
            loss_sum,
            batch_count,
        )

    @torch.no_grad()
    def _validate(
        self,
    ) -> float | None:
        if self.val_loader is None:
            return None

        self.model.eval()

        loss_sum = 0.0
        count = 0

        for raw_batch in self.val_loader:
            batch = self._move_to_device(
                raw_batch
            )

            with self._autocast():
                loss = self._compute_loss(
                    batch
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite validation loss"
                )

            loss_sum += float(
                loss.detach().item()
            )

            count += 1

        return self._reduce_metrics(
            loss_sum,
            count,
        )

    def _is_improvement(
        self,
        value: float,
    ) -> bool:
        if self.config.monitor_mode == "min":
            return value < self.best_metric

        return value > self.best_metric

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
    ) -> None:
        if not self.is_main_process:
            return

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": (
                self.raw_model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "scaler_state_dict": (
                self.scaler.state_dict()
            ),
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": asdict(
                self.config
            ),
        }

        destination = (
            self.checkpoint_dir / filename
        )

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

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """
        Load the same checkpoint on all ranks.

        Call before fit().

        Optimizer and scheduler must be
        constructed before restoring.
        """
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.raw_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler_state = checkpoint.get(
            "scheduler_state_dict"
        )

        if (
            self.scheduler is not None
            and scheduler_state is not None
        ):
            self.scheduler.load_state_dict(
                scheduler_state
            )

        scaler_state = checkpoint.get(
            "scaler_state_dict"
        )

        if scaler_state is not None:
            self.scaler.load_state_dict(
                scaler_state
            )

        self.start_epoch = (
            int(checkpoint["epoch"]) + 1
        )

        self.global_step = int(
            checkpoint.get(
                "global_step",
                0,
            )
        )

        self.best_metric = float(
            checkpoint.get(
                "best_metric",
                self.best_metric,
            )
        )

        self.bad_epochs = int(
            checkpoint.get(
                "bad_epochs",
                0,
            )
        )

        self.history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        if self.distributed:
            dist.barrier()

    def fit(
        self,
    ) -> list[dict[str, float]]:
        if self.is_main_process:
            print(
                f"Training on {self.world_size} "
                f"process(es), device={self.device}",
                flush=True,
            )

        for epoch in range(
            self.start_epoch,
            self.config.epochs,
        ):
            train_loss = self._train_epoch(
                epoch
            )

            val_loss = self._validate()

            if (
                self.scheduler is not None
                and self.config.scheduler_step
                == "epoch"
            ):
                # Supports ReduceLROnPlateau
                # and ordinary epoch schedulers.
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    metric = (
                        val_loss
                        if val_loss is not None
                        else train_loss
                    )

                    self.scheduler.step(
                        metric
                    )
                else:
                    self.scheduler.step()

            metrics = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
            }

            if val_loss is not None:
                metrics["val_loss"] = val_loss

            self.history.append(
                metrics
            )

            monitor_value = metrics.get(
                self.config.monitor
            )

            if monitor_value is None:
                monitor_value = train_loss

            if self._is_improvement(
                monitor_value
            ):
                self.best_metric = (
                    monitor_value
                )

                self.bad_epochs = 0

                self._save_checkpoint(
                    "best.pt",
                    epoch,
                )
            else:
                self.bad_epochs += 1

            self._save_checkpoint(
                "last.pt",
                epoch,
            )

            if (
                (epoch + 1)
                % self.config.checkpoint_interval
                == 0
            ):
                self._save_checkpoint(
                    f"epoch_{epoch + 1:04d}.pt",
                    epoch,
                )

            if self.is_main_process:
                message = (
                    f"Epoch {epoch + 1}/"
                    f"{self.config.epochs} | "
                    f"train_loss={train_loss:.6f}"
                )

                if val_loss is not None:
                    message += (
                        f" | val_loss="
                        f"{val_loss:.6f}"
                    )

                print(
                    message,
                    flush=True,
                )

            should_stop = False

            patience = (
                self.config.early_stopping_patience
            )

            if patience is not None:
                should_stop = (
                    self.bad_epochs >= patience
                )

            if self.distributed:
                stop_tensor = torch.tensor(
                    int(should_stop),
                    device=self.device,
                )

                dist.broadcast(
                    stop_tensor,
                    src=0,
                )

                should_stop = bool(
                    stop_tensor.item()
                )

            if should_stop:
                if self.is_main_process:
                    print(
                        "Early stopping triggered",
                        flush=True,
                    )

                break

        if self.distributed:
            dist.barrier()

        return self.history

    def cleanup(
        self,
    ) -> None:
        """
        Release process group only if
        this trainer created it.
        """
        if (
            self._owns_process_group
            and dist.is_initialized()
        ):
            dist.destroy_process_group()

    def __enter__(
        self,
    ) -> "DDPTrainer":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.cleanup()


def create_distributed_sampler(
    dataset: Dataset,
    *,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
) -> DistributedSampler:
    """
    Create sampler using torchrun environment.

    All ranks must use the same seed.
    """
    return DistributedSampler(
        dataset,
        num_replicas=int(
            os.environ.get(
                "WORLD_SIZE",
                "1",
            )
        ),
        rank=int(
            os.environ.get(
                "RANK",
                "0",
            )
        ),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )
