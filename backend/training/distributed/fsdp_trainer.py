
"""
NeuroWorld AI
File: training/distributed/fsdp_trainer.py

PyTorch Fully Sharded Data Parallel trainer.

Features:
- Multi-GPU FSDP training
- FULL_SHARD / SHARD_GRAD_OP / NO_SHARD
- Transformer auto-wrapping
- Mixed precision FP16 / BF16
- Gradient accumulation
- Activation checkpointing
- DistributedSampler support
- Distributed metric reduction
- Gradient clipping
- Rank-zero full-model checkpoint saving
- Full optimizer-state checkpointing
- Resume training
- Early stopping
- Learning-rate scheduling

Uses the PyTorch FSDP1 API.

Requirements:
    torch >= 2.1
    numpy

Launch:
    torchrun --standalone --nproc_per_node=2 \
        -m training.train_world_model
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist

from torch import Tensor, nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


LossFunction = Callable[
    [Any, Mapping[str, Any]],
    Tensor,
]

ForwardFunction = Callable[
    [nn.Module, Mapping[str, Any]],
    Any,
]


@dataclass
class FSDPTrainerConfig:
    epochs: int = 100

    sharding_strategy: str = "FULL_SHARD"

    mixed_precision: str = "bf16"

    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0

    activation_checkpointing: bool = False

    cpu_offload: bool = False

    sync_module_states: bool = True

    use_orig_params: bool = True

    limit_all_gathers: bool = True

    backward_prefetch: bool = True

    log_interval: int = 10

    checkpoint_dir: str = (
        "checkpoints/fsdp"
    )

    checkpoint_interval: int = 5

    monitor: str = "val_loss"
    monitor_mode: str = "min"

    early_stopping_patience: int | None = None

    scheduler_step: str = "epoch"

    seed: int = 42
    deterministic: bool = False

    timeout_seconds: int = 1800

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

        if self.sharding_strategy not in (
            "FULL_SHARD",
            "SHARD_GRAD_OP",
            "NO_SHARD",
        ):
            raise ValueError(
                "Unsupported sharding strategy"
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

        if self.cpu_offload:
            raise ValueError(
                "CPU offload is not implemented "
                "in this trainer"
            )


class FSDPTrainer:
    """
    FSDP training engine.

    Important:
        Construct the optimizer AFTER
        the model is wrapped by FSDP.

    The trainer accepts an optimizer factory:

        optimizer_factory(model.parameters())

    The factory must return an Optimizer.

    Custom forward and loss functions allow
    training world models, video predictors,
    and other PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_factory: Callable[
            [Any], Optimizer
        ],
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        loss_fn: LossFunction | None = None,
        forward_fn: ForwardFunction | None = None,
        scheduler_factory: Callable[
            [Optimizer], Any
        ] | None = None,
        transformer_layer_classes: (
            set[type[nn.Module]] | None
        ) = None,
        config: FSDPTrainerConfig | None = None,
    ) -> None:
        self.config = (
            config or FSDPTrainerConfig()
        )

        self._owns_process_group = False

        self._initialize_distributed()

        self._seed_everything(
            self.config.seed + self.rank
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = (
            loss_fn or self._default_loss
        )

        self.forward_fn = (
            forward_fn or self._default_forward
        )

        self.transformer_layer_classes = (
            transformer_layer_classes or set()
        )

        self._configure_precision()

        if self.config.activation_checkpointing:
            self._apply_activation_checkpointing(
                model
            )

        self.model = self._wrap_model(
            model
        )

        self.optimizer = optimizer_factory(
            self.model.parameters()
        )

        self.scheduler = (
            scheduler_factory(self.optimizer)
            if scheduler_factory is not None
            else None
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.config.mixed_precision
                == "fp16"
            ),
        )

        self.start_epoch = 0
        self.global_step = 0

        self.best_metric = (
            math.inf
            if self.config.monitor_mode == "min"
            else -math.inf
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

        dist.barrier()

        self._validate_loader(
            self.train_loader,
            "train",
        )

        if self.val_loader is not None:
            self._validate_loader(
                self.val_loader,
                "validation",
            )

    def _initialize_distributed(
        self,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "This FSDP trainer requires CUDA GPUs"
            )

        self.local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                "0",
            )
        )

        if (
            self.local_rank
            >= torch.cuda.device_count()
        ):
            raise RuntimeError(
                "LOCAL_RANK exceeds available GPUs"
            )

        torch.cuda.set_device(
            self.local_rank
        )

        self.device = torch.device(
            "cuda",
            self.local_rank,
        )

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(
                    seconds=(
                        self.config.timeout_seconds
                    )
                ),
            )

            self._owns_process_group = True

        self.rank = dist.get_rank()
        self.world_size = (
            dist.get_world_size()
        )

        self.is_main_process = (
            self.rank == 0
        )

    def _seed_everything(
        self,
        seed: int,
    ) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

        if (
            precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "BF16 unsupported. Use fp16 or none."
            )

        dtype_map = {
            "none": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        self.compute_dtype = dtype_map[
            precision
        ]

        self.mixed_precision_policy = (
            None
            if precision == "none"
            else MixedPrecision(
                param_dtype=self.compute_dtype,
                reduce_dtype=self.compute_dtype,
                buffer_dtype=self.compute_dtype,
            )
        )

    def _apply_activation_checkpointing(
        self,
        model: nn.Module,
    ) -> None:
        if not self.transformer_layer_classes:
            raise ValueError(
                "Activation checkpointing requires "
                "transformer_layer_classes"
            )

        wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=(
                CheckpointImpl.NO_REENTRANT
            ),
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper,
            check_fn=lambda module: type(module)
            in self.transformer_layer_classes,
        )

    def _wrap_model(
        self,
        model: nn.Module,
    ) -> FSDP:
        strategies = {
            "FULL_SHARD": (
                ShardingStrategy.FULL_SHARD
            ),
            "SHARD_GRAD_OP": (
                ShardingStrategy.SHARD_GRAD_OP
            ),
            "NO_SHARD": (
                ShardingStrategy.NO_SHARD
            ),
        }

        auto_wrap_policy = None

        if self.transformer_layer_classes:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=(
                    self.transformer_layer_classes
                ),
            )

        # Model parameters must be initialized
        # consistently across ranks.
        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=(
                self.mixed_precision_policy
            ),
            sharding_strategy=strategies[
                self.config.sharding_strategy
            ],
            device_id=self.device,
            sync_module_states=(
                self.config.sync_module_states
            ),
            use_orig_params=(
                self.config.use_orig_params
            ),
            limit_all_gathers=(
                self.config.limit_all_gathers
            ),
        )

    def _validate_loader(
        self,
        loader: DataLoader,
        name: str,
    ) -> None:
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
                f"{name} DataLoader requires "
                "DistributedSampler"
            )

        if (
            sampler.rank != self.rank
            or sampler.num_replicas
            != self.world_size
        ):
            raise ValueError(
                f"{name} sampler does not match "
                "distributed rank/world size"
            )

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

        raise ValueError(
            "Unsupported default inputs. "
            "Provide forward_fn."
        )

    @staticmethod
    def _default_loss(
        output: Any,
        batch: Mapping[str, Any],
    ) -> Tensor:
        if isinstance(output, Mapping):
            if "loss" in output:
                return output["loss"]

            output = output.get(
                "predictions",
                output.get("next_states"),
            )

        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, Tensor):
            raise TypeError(
                "Model must return Tensor "
                "or supported output mapping"
            )

        if "next_states" in batch:
            target = batch["next_states"]

        elif "target_frames" in batch:
            target = batch["target_frames"]

        else:
            raise ValueError(
                "No target found in batch"
            )

        if output.shape != target.shape:
            raise ValueError(
                "Prediction/target shape mismatch"
            )

        error = (
            output.float()
            - target.float()
        ).square()

        mask = batch.get(
            "mask",
            batch.get("frame_mask"),
        )

        if mask is None:
            return error.mean()

        if (
            mask.ndim != 2
            or mask.shape[:2]
            != error.shape[:2]
        ):
            raise ValueError(
                "Mask must have shape [B, T]"
            )

        while mask.ndim < error.ndim:
            mask = mask.unsqueeze(-1)

        mask = mask.to(
            error.dtype
        ).expand_as(error)

        return (
            error * mask
        ).sum() / mask.sum().clamp_min(1)

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

    def _compute_loss(
        self,
        batch: Mapping[str, Any],
    ) -> Tensor:
        output = self.forward_fn(
            self.model,
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
                "Loss must be a scalar Tensor"
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss detected"
            )

        return loss

    def _reduce_mean(
        self,
        local_sum: float,
        local_count: int,
    ) -> float:
        values = torch.tensor(
            [
                local_sum,
                float(local_count),
            ],
            dtype=torch.float64,
            device=self.device,
        )

        dist.all_reduce(
            values,
            op=dist.ReduceOp.SUM,
        )

        count = values[1].item()

        return (
            values[0].item() / count
            if count > 0
            else float("nan")
        )

    def _train_epoch(
        self,
        epoch: int,
    ) -> float:
        self.model.train()

        sampler = self.train_loader.sampler
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
        count = 0

        for batch_index, raw_batch in enumerate(
            self.train_loader
        ):
            batch = self._move_batch(
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
                (batch_index + 1) % accumulation
                == 0
                or batch_index + 1
                == total_batches
            )

            # FSDP no_sync() accumulates full
            # gradients on each rank and can
            # significantly increase memory use.
            sync_context = (
                contextlib.nullcontext()
                if should_step
                else self.model.no_sync()
            )

            with sync_context:
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.compute_dtype,
                    enabled=(
                        self.config.mixed_precision
                        != "none"
                    ),
                ):
                    loss = self._compute_loss(
                        batch
                    )

                    scaled_loss = (
                        loss / group_size
                    )

                self.scaler.scale(
                    scaled_loss
                ).backward()

            loss_sum += float(
                loss.detach().item()
            )
            count += 1

            if should_step:
                if (
                    self.config.gradient_clip_norm
                    is not None
                ):
                    self.scaler.unscale_(
                        self.optimizer
                    )

                    self.model.clip_grad_norm_(
                        self.config.gradient_clip_norm
                    )

                old_scale = (
                    self.scaler.get_scale()
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                stepped = (
                    not self.scaler.is_enabled()
                    or self.scaler.get_scale()
                    >= old_scale
                )

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                if stepped:
                    self.global_step += 1

                    if (
                        self.scheduler is not None
                        and self.config.scheduler_step
                        == "batch"
                    ):
                        self.scheduler.step()

            if (
                self.is_main_process
                and self.config.log_interval > 0
                and (batch_index + 1)
                % self.config.log_interval == 0
            ):
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {batch_index + 1}/"
                    f"{total_batches} | "
                    f"Loss {loss.item():.6f}",
                    flush=True,
                )

        return self._reduce_mean(
            loss_sum,
            count,
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
            batch = self._move_batch(
                raw_batch
            )

            with torch.autocast(
                device_type="cuda",
                dtype=self.compute_dtype,
                enabled=(
                    self.config.mixed_precision
                    != "none"
                ),
            ):
                loss = self._compute_loss(
                    batch
                )

            loss_sum += float(
                loss.item()
            )
            count += 1

        return self._reduce_mean(
            loss_sum,
            count,
        )

    def _improved(
        self,
        metric: float,
    ) -> bool:
        if self.config.monitor_mode == "min":
            return metric < self.best_metric

        return metric > self.best_metric

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
    ) -> None:
        """
        Save a full FSDP checkpoint.

        All ranks must participate in
        state-dict collection.

        Suitable for moderate-size models.
        Very large models should use
        distributed sharded checkpoints.
        """
        model_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )

        optimizer_config = (
            FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=True,
            )
        )

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            model_config,
            optimizer_config,
        ):
            model_state = (
                self.model.state_dict()
            )

            optimizer_state = (
                FSDP.full_optim_state_dict(
                    self.model,
                    self.optimizer,
                    rank0_only=True,
                )
            )

        if self.is_main_process:
            checkpoint = {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": model_state,
                "optimizer_state_dict": (
                    optimizer_state
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

        dist.barrier()

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """
        Restore full FSDP checkpoint.

        Each rank loads the trusted checkpoint
        and converts optimizer state to the
        current FSDP sharding layout.
        """
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False,
        )

        optimizer_config = (
            FullOptimStateDictConfig(
                offload_to_cpu=True,
                rank0_only=False,
            )
        )

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            model_config,
            optimizer_config,
        ):
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        sharded_optimizer_state = (
            FSDP.scatter_full_optim_state_dict(
                checkpoint["optimizer_state_dict"]
                if self.is_main_process
                else None,
                self.model,
                optim=self.optimizer,
            )
        )

        self.optimizer.load_state_dict(
            sharded_optimizer_state
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

        dist.barrier()

    def fit(
        self,
    ) -> list[dict[str, float]]:
        if self.is_main_process:
            print(
                f"NeuroWorld FSDP training | "
                f"GPUs: {self.world_size}",
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
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(
                        val_loss
                        if val_loss is not None
                        else train_loss
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
                self.config.monitor,
                train_loss,
            )

            if self._improved(
                monitor_value
            ):
                self.best_metric = monitor_value
                self.bad_epochs = 0

                self.save_checkpoint(
                    "best.pt",
                    epoch,
                )

            else:
                self.bad_epochs += 1

            self.save_checkpoint(
                "last.pt",
                epoch,
            )

            if (
                (epoch + 1)
                % self.config.checkpoint_interval
                == 0
            ):
                self.save_checkpoint(
                    f"epoch_{epoch + 1:04d}.pt",
                    epoch,
                )

            if self.is_main_process:
                print(
                    f"Epoch {epoch + 1}/"
                    f"{self.config.epochs} | "
                    f"train={train_loss:.6f} | "
                    f"val={val_loss}",
                    flush=True,
                )

            patience = (
                self.config.early_stopping_patience
            )

            should_stop = (
                patience is not None
                and self.bad_epochs >= patience
            )

            stop_tensor = torch.tensor(
                int(should_stop),
                device=self.device,
            )

            dist.broadcast(
                stop_tensor,
                src=0,
            )

            if bool(stop_tensor.item()):
                if self.is_main_process:
                    print(
                        "Early stopping triggered",
                        flush=True,
                    )

                break

        dist.barrier()

        return self.history

    def cleanup(
        self,
    ) -> None:
        if (
            self._owns_process_group
            and dist.is_initialized()
        ):
            dist.destroy_process_group()

            self._owns_process_group = False

    def __enter__(
        self,
    ) -> "FSDPTrainer":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.cleanup()


def create_fsdp_sampler(
    dataset,
    *,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = False,
) -> DistributedSampler:
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
