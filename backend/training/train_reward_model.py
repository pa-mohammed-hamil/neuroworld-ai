
"""
NeuroWorld AI
File: training/train_reward_model.py

Reward-model training pipeline.

Features:
    - State/action-conditioned reward prediction
    - Optional next-state conditioning
    - Regression: MSE, Huber, Gaussian NLL
    - Binary reward classification
    - Reward uncertainty estimation
    - AMP mixed precision
    - Gradient accumulation
    - Gradient clipping
    - Cosine / plateau LR scheduling
    - Episode-disjoint train/validation split
    - Early stopping
    - Atomic checkpoints
    - Resume training
    - JSONL metric logging
    - Synthetic dataset generation

NPZ dataset:
    states:      [N, state_dim]
    actions:     [N, action_dim]
    rewards:     [N]
    next_states: [N, state_dim] (optional)

Each NPZ file represents an episode.

Examples:
    python -m training.train_reward_model \
        --demo-data --epochs 5

    python -m training.train_reward_model \
        --data-dir data/trajectories \
        --epochs 50

    python -m training.train_reward_model \
        --resume checkpoints/reward_model/latest.pt

Dependencies:
    pip install torch numpy pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import yaml
except ImportError:
    yaml = None


LOGGER = logging.getLogger(
    "neuroworld.train_reward_model"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class RewardTrainingConfig:
    data_dir: str = "data/trajectories"
    demo_data: bool = False

    state_dim: int = 128
    action_dim: int = 32

    use_next_state: bool = True

    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1

    # mse | huber | gaussian_nll | binary
    loss_type: str = "huber"

    huber_delta: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 2.0

    validation_split: float = 0.15

    batch_size: int = 64
    num_workers: int = 0

    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0

    mixed_precision: bool = True
    precision: str = "float16"

    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_learning_rate: float = 1e-6

    validate_every: int = 1
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0

    checkpoint_dir: str = "checkpoints/reward_model"
    log_dir: str = "runs/reward_model"

    save_every: int = 5
    resume: str | None = None

    device: str = "auto"
    seed: int = 42
    log_every: int = 20

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "hidden_dim",
            "num_layers",
            "batch_size",
            "epochs",
            "accumulation_steps",
            "validate_every",
            "save_every",
            "log_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not 0 < self.validation_split < 1:
            raise ValueError(
                "validation_split must be in (0,1)"
            )

        if self.loss_type not in {
            "mse",
            "huber",
            "gaussian_nll",
            "binary",
        }:
            raise ValueError(
                "Unsupported reward loss"
            )

        if self.scheduler not in {
            "cosine",
            "plateau",
            "none",
        }:
            raise ValueError(
                "Unsupported scheduler"
            )

        if self.precision not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError(
                "Unsupported precision"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "min_logvar must be less than max_logvar"
            )


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Dataset
# ============================================================

class RewardDataset(Dataset):
    """
    Episode-based reward dataset.

    Each NPZ file is treated as one episode.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: RewardTrainingConfig,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory missing: {self.data_dir}"
            )

        files = sorted(
            self.data_dir.rglob("*.npz")
        )

        if not files:
            raise FileNotFoundError(
                "No NPZ reward files found"
            )

        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        episode_ids_list = []

        for episode_index, path in enumerate(files):
            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                required = {
                    "states",
                    "actions",
                    "rewards",
                }

                if config.use_next_state:
                    required.add(
                        "next_states"
                    )

                missing = required - set(data.files)

                if missing:
                    raise ValueError(
                        f"{path}: missing {sorted(missing)}"
                    )

                states = np.asarray(
                    data["states"],
                    dtype=np.float32,
                )

                actions = np.asarray(
                    data["actions"],
                    dtype=np.float32,
                )

                rewards = np.asarray(
                    data["rewards"],
                    dtype=np.float32,
                )

                if "next_states" in data.files:
                    next_states = np.asarray(
                        data["next_states"],
                        dtype=np.float32,
                    )
                else:
                    next_states = np.zeros_like(
                        states
                    )

            if states.ndim != 2:
                raise ValueError(
                    f"{path}: states must be 2D"
                )

            n = states.shape[0]

            if n == 0:
                continue

            if (
                states.shape != (
                    n,
                    config.state_dim,
                )
                or actions.shape != (
                    n,
                    config.action_dim,
                )
                or rewards.shape != (n,)
                or next_states.shape != (
                    n,
                    config.state_dim,
                )
            ):
                raise ValueError(
                    f"{path}: incompatible array shapes"
                )

            if not all(
                np.isfinite(array).all()
                for array in (
                    states,
                    actions,
                    rewards,
                    next_states,
                )
            ):
                raise ValueError(
                    f"{path}: non-finite values"
                )

            if config.loss_type == "binary":
                if not np.isin(
                    rewards,
                    [0.0, 1.0],
                ).all():
                    raise ValueError(
                        f"{path}: binary rewards "
                        "must be 0 or 1"
                    )

            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            next_states_list.append(next_states)

            episode_ids_list.append(
                np.full(
                    n,
                    episode_index,
                    dtype=np.int64,
                )
            )

        if not states_list:
            raise ValueError(
                "No valid reward samples"
            )

        def concatenate(
            arrays: list[np.ndarray],
        ) -> Tensor:
            return torch.from_numpy(
                np.ascontiguousarray(
                    np.concatenate(
                        arrays,
                        axis=0,
                    )
                )
            )

        self.states = concatenate(
            states_list
        )

        self.actions = concatenate(
            actions_list
        )

        self.rewards = concatenate(
            rewards_list
        )

        self.next_states = concatenate(
            next_states_list
        )

        self.episode_ids = np.concatenate(
            episode_ids_list
        )

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "rewards": self.rewards[index],
            "next_states": (
                self.next_states[index]
            ),
        }


# ============================================================
# Synthetic demo dataset
# ============================================================

def create_demo_dataset(
    directory: str | Path,
    config: RewardTrainingConfig,
    *,
    num_episodes: int = 20,
    episode_length: int = 128,
) -> None:
    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        config.seed
    )

    action_projection = rng.normal(
        0,
        0.1,
        size=(
            config.action_dim,
            config.state_dim,
        ),
    ).astype(np.float32)

    for episode_index in range(num_episodes):
        states = rng.normal(
            size=(
                episode_length,
                config.state_dim,
            )
        ).astype(np.float32)

        actions = rng.normal(
            size=(
                episode_length,
                config.action_dim,
            )
        ).astype(np.float32)

        next_states = (
            0.95 * states
            + actions @ action_projection
        ).astype(np.float32)

        reward_signal = (
            -0.25
            * np.mean(
                next_states.square(),
                axis=-1,
            )
            - 0.05
            * np.mean(
                actions.square(),
                axis=-1,
            )
        )

        reward_signal += rng.normal(
            0,
            0.02,
            size=episode_length,
        )

        if config.loss_type == "binary":
            rewards = (
                reward_signal
                > np.median(reward_signal)
            ).astype(np.float32)
        else:
            rewards = reward_signal.astype(
                np.float32
            )

        np.savez_compressed(
            directory
            / f"episode_{episode_index:04d}.npz",
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
        )


# ============================================================
# Data loaders
# ============================================================

def create_dataloaders(
    config: RewardTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset = RewardDataset(
        config.data_dir,
        config,
    )

    episodes = np.unique(
        dataset.episode_ids
    )

    if len(episodes) < 2:
        raise ValueError(
            "At least two episodes required "
            "for episode-disjoint validation"
        )

    rng = np.random.default_rng(
        config.seed
    )

    rng.shuffle(
        episodes
    )

    validation_count = max(
        1,
        min(
            len(episodes) - 1,
            round(
                len(episodes)
                * config.validation_split
            ),
        ),
    )

    validation_episodes = set(
        episodes[
            :validation_count
        ].tolist()
    )

    train_indices = []
    validation_indices = []

    for index, episode_id in enumerate(
        dataset.episode_ids
    ):
        if int(episode_id) in validation_episodes:
            validation_indices.append(index)
        else:
            train_indices.append(index)

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=generator,
        **options,
    )

    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        **options,
    )

    LOGGER.info(
        "Reward samples: train=%d validation=%d",
        len(train_indices),
        len(validation_indices),
    )

    return train_loader, validation_loader


# ============================================================
# Reward prediction model
# ============================================================

class RewardPredictionModel(nn.Module):
    """
    Predict rewards from state-action transitions.

    For gaussian_nll:
        Returns mean and log variance.

    For binary:
        Returns raw logits.

    For mse/huber:
        Returns scalar reward predictions.
    """

    def __init__(
        self,
        config: RewardTrainingConfig,
    ) -> None:
        super().__init__()

        self.config = config

        input_dim = (
            config.state_dim
            + config.action_dim
        )

        if config.use_next_state:
            input_dim += config.state_dim

        layers: list[nn.Module] = []

        for index in range(
            config.num_layers
        ):
            layers.extend(
                [
                    nn.Linear(
                        input_dim
                        if index == 0
                        else config.hidden_dim,
                        config.hidden_dim,
                    ),
                    nn.LayerNorm(
                        config.hidden_dim
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        config.dropout
                    ),
                ]
            )

        self.backbone = nn.Sequential(
            *layers
        )

        self.reward_head = nn.Linear(
            config.hidden_dim,
            1,
        )

        self.logvar_head = (
            nn.Linear(
                config.hidden_dim,
                1,
            )
            if config.loss_type == "gaussian_nll"
            else None
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor | None = None,
    ) -> dict[str, Tensor]:
        features = [
            states,
            actions,
        ]

        if self.config.use_next_state:
            if next_states is None:
                raise ValueError(
                    "next_states required by configuration"
                )

            features.append(
                next_states
            )

        hidden = self.backbone(
            torch.cat(
                features,
                dim=-1,
            )
        )

        predictions = self.reward_head(
            hidden
        ).squeeze(-1)

        result = {
            "prediction": predictions,
        }

        if self.logvar_head is not None:
            logvar = self.logvar_head(
                hidden
            ).squeeze(-1)

            result["logvar"] = logvar.clamp(
                min=self.config.min_logvar,
                max=self.config.max_logvar,
            )

        return result

    @torch.no_grad()
    def predict_reward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor | None = None,
    ) -> Tensor:
        result = self(
            states,
            actions,
            next_states,
        )

        predictions = result["prediction"]

        if self.config.loss_type == "binary":
            return torch.sigmoid(
                predictions
            )

        return predictions

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor | None = None,
    ) -> dict[str, Tensor]:
        result = self(
            states,
            actions,
            next_states,
        )

        prediction = result["prediction"]

        if "logvar" in result:
            std = torch.exp(
                0.5 * result["logvar"]
            )

            return {
                "mean": prediction,
                "std": std,
            }

        if self.config.loss_type == "binary":
            probability = torch.sigmoid(
                prediction
            )

            return {
                "probability": probability,
                "std": torch.sqrt(
                    (
                        probability
                        * (1 - probability)
                    ).clamp_min(0)
                ),
            }

        return {
            "mean": prediction,
        }


# ============================================================
# Loss functions
# ============================================================

def compute_reward_loss(
    outputs: Mapping[str, Tensor],
    targets: Tensor,
    config: RewardTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    predictions = (
        outputs["prediction"].float()
    )

    targets = targets.float()

    mse = F.mse_loss(
        predictions,
        targets,
    )

    mae = F.l1_loss(
        predictions,
        targets,
    )

    if config.loss_type == "mse":
        loss = mse

    elif config.loss_type == "huber":
        loss = F.huber_loss(
            predictions,
            targets,
            delta=config.huber_delta,
        )

    elif config.loss_type == "gaussian_nll":
        logvar = (
            outputs["logvar"].float()
        )

        squared_error = (
            targets - predictions
        ).square()

        loss = (
            0.5
            * (
                logvar
                + squared_error
                * torch.exp(-logvar)
                + math.log(2 * math.pi)
            )
        ).mean()

    elif config.loss_type == "binary":
        loss = (
            F.binary_cross_entropy_with_logits(
                predictions,
                targets,
            )
        )

    else:
        raise ValueError(
            "Unsupported loss type"
        )

    return loss, {
        "total": loss,
        "mse": mse,
        "mae": mae,
    }


# ============================================================
# Metrics
# ============================================================

class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(
        self,
        value: float,
        weight: int = 1,
    ) -> None:
        self.total += value * weight
        self.count += weight

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0

        return self.total / self.count


class MetricCollection:
    def __init__(self) -> None:
        self.meters: dict[
            str, AverageMeter
        ] = {}

    def update(
        self,
        metrics: Mapping[str, Tensor],
        weight: int,
    ) -> None:
        for key, value in metrics.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter()

            self.meters[key].update(
                float(
                    value.detach().item()
                ),
                weight,
            )

    def compute(
        self,
        prefix: str,
    ) -> dict[str, float]:
        return {
            f"{prefix}/{key}": meter.average
            for key, meter in self.meters.items()
        }


# ============================================================
# Scheduler
# ============================================================

def create_scheduler(
    optimizer: AdamW,
    config: RewardTrainingConfig,
) -> Any:
    if config.scheduler == "none":
        return None

    if config.scheduler == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=config.min_learning_rate,
            )
        )

    warmup_epochs = min(
        config.warmup_epochs,
        max(config.epochs - 1, 0),
    )

    min_factor = min(
        1.0,
        config.min_learning_rate
        / config.learning_rate,
    )

    def lr_factor(epoch: int) -> float:
        if (
            warmup_epochs > 0
            and epoch < warmup_epochs
        ):
            return max(
                min_factor,
                (epoch + 1) / warmup_epochs,
            )

        progress = (
            (epoch - warmup_epochs)
            / max(
                1,
                config.epochs - warmup_epochs,
            )
        )

        progress = min(
            1.0,
            max(0.0, progress),
        )

        return (
            min_factor
            + (1 - min_factor)
            * 0.5
            * (
                1
                + math.cos(
                    math.pi * progress
                )
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    bad_epochs: int,
    config: RewardTrainingConfig,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": (
            best_validation_loss
        ),
        "bad_epochs": bad_epochs,
        "config": asdict(config),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        },
    }

    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        torch.save(
            checkpoint,
            temporary,
        )

        os.replace(
            temporary,
            path,
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    if (
        scheduler is not None
        and checkpoint.get("scheduler") is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    rng = checkpoint.get(
        "rng",
        {},
    )

    if "torch" in rng:
        torch.set_rng_state(
            rng["torch"].cpu()
        )

    if (
        torch.cuda.is_available()
        and rng.get("cuda")
    ):
        for index, state in enumerate(
            rng["cuda"][
                :torch.cuda.device_count()
            ]
        ):
            torch.cuda.set_rng_state(
                state.cpu(),
                device=index,
            )

    return checkpoint


# ============================================================
# Trainer
# ============================================================

class RewardModelTrainer:
    def __init__(
        self,
        config: RewardTrainingConfig,
    ) -> None:
        self.config = config

        seed_everything(
            config.seed
        )

        self.device = resolve_device(
            config.device
        )

        self.model = RewardPredictionModel(
            config
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = create_scheduler(
            self.optimizer,
            config,
        )

        self.amp_enabled = (
            config.mixed_precision
            and config.precision != "float32"
            and self.device.type in {
                "cuda",
                "cpu",
            }
        )

        if (
            self.device.type == "cpu"
            and config.precision == "float16"
        ):
            self.amp_enabled = False

        self.amp_dtype = (
            torch.float16
            if config.precision == "float16"
            else torch.bfloat16
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                self.amp_enabled
                and self.device.type == "cuda"
                and config.precision == "float16"
            ),
        )

        self.checkpoint_dir = Path(
            config.checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_dir = Path(
            config.log_dir
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.start_epoch = 1
        self.global_step = 0

        self.best_validation_loss = float("inf")
        self.bad_epochs = 0

        if config.resume:
            self.resume(
                config.resume
            )

    def autocast(self) -> Any:
        if not self.amp_enabled:
            return nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
        )

    def move_batch(
        self,
        batch: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        return {
            key: value.to(
                self.device,
                non_blocking=True,
            )
            for key, value in batch.items()
        }

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()

        metrics = MetricCollection()

        accumulation = (
            self.config.accumulation_steps
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        total_batches = len(loader)

        for index, raw_batch in enumerate(
            loader
        ):
            batch = self.move_batch(
                raw_batch
            )

            group_start = (
                index // accumulation
            ) * accumulation

            group_size = min(
                accumulation,
                total_batches - group_start,
            )

            with self.autocast():
                outputs = self.model(
                    batch["states"],
                    batch["actions"],
                    batch["next_states"],
                )

                loss, components = (
                    compute_reward_loss(
                        outputs,
                        batch["rewards"],
                        self.config,
                    )
                )

            if not bool(
                torch.isfinite(
                    loss.detach()
                ).item()
            ):
                raise FloatingPointError(
                    "Non-finite reward loss"
                )

            self.scaler.scale(
                loss / group_size
            ).backward()

            should_step = (
                (index + 1) % accumulation == 0
                or index + 1 == total_batches
            )

            if should_step:
                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

                self.global_step += 1

            metrics.update(
                components,
                weight=len(
                    batch["rewards"]
                ),
            )

            if (
                (index + 1)
                % self.config.log_every == 0
            ):
                LOGGER.info(
                    "Epoch %d | Batch %d/%d | "
                    "Loss %.6f",
                    epoch,
                    index + 1,
                    total_batches,
                    metrics.meters[
                        "total"
                    ].average,
                )

        return metrics.compute(
            "train"
        )

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
    ) -> dict[str, float]:
        self.model.eval()

        metrics = MetricCollection()
        accuracy_meter = AverageMeter()

        for raw_batch in loader:
            batch = self.move_batch(
                raw_batch
            )

            with self.autocast():
                outputs = self.model(
                    batch["states"],
                    batch["actions"],
                    batch["next_states"],
                )

                _, components = (
                    compute_reward_loss(
                        outputs,
                        batch["rewards"],
                        self.config,
                    )
                )

            batch_size = len(
                batch["rewards"]
            )

            metrics.update(
                components,
                weight=batch_size,
            )

            if self.config.loss_type == "binary":
                probabilities = torch.sigmoid(
                    outputs["prediction"].float()
                )

                predicted_labels = (
                    probabilities >= 0.5
                ).float()

                accuracy = (
                    predicted_labels
                    == batch["rewards"]
                ).float().mean()

                accuracy_meter.update(
                    float(accuracy.item()),
                    batch_size,
                )

        results = metrics.compute(
            "val"
        )

        if self.config.loss_type != "binary":
            results["val/rmse"] = math.sqrt(
                results["val/mse"]
            )
        else:
            results["val/accuracy"] = (
                accuracy_meter.average
            )

        return results

    def save(
        self,
        filename: str,
        epoch: int,
    ) -> None:
        save_checkpoint(
            self.checkpoint_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            global_step=self.global_step,
            best_validation_loss=(
                self.best_validation_loss
            ),
            bad_epochs=self.bad_epochs,
            config=self.config,
        )

    def resume(
        self,
        path: str | Path,
    ) -> None:
        checkpoint = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
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

        self.best_validation_loss = float(
            checkpoint.get(
                "best_validation_loss",
                float("inf"),
            )
        )

        self.bad_epochs = int(
            checkpoint.get(
                "bad_epochs",
                0,
            )
        )

        LOGGER.info(
            "Resumed reward training at epoch %d",
            self.start_epoch,
        )

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
    ) -> None:
        path = (
            self.log_dir / "metrics.jsonl"
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        **dict(metrics),
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> None:
        LOGGER.info(
            "Training reward model on %s",
            self.device,
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        LOGGER.info(
            "Model parameters: %d",
            parameter_count,
        )

        for epoch in range(
            self.start_epoch,
            self.config.epochs + 1,
        ):
            start_time = time.perf_counter()

            train_metrics = self.train_epoch(
                train_loader,
                epoch,
            )

            validation_metrics = {}

            improved = False

            should_validate = (
                epoch
                % self.config.validate_every == 0
                or epoch == self.config.epochs
            )

            if should_validate:
                validation_metrics = self.validate(
                    validation_loader
                )

                validation_loss = (
                    validation_metrics["val/total"]
                )

                if not math.isfinite(
                    validation_loss
                ):
                    raise FloatingPointError(
                        "Non-finite validation loss"
                    )

                improved = (
                    validation_loss
                    < self.best_validation_loss
                    - self.config.early_stopping_min_delta
                )

                if improved:
                    self.best_validation_loss = (
                        validation_loss
                    )

                    self.bad_epochs = 0

                else:
                    self.bad_epochs += 1

                if self.config.scheduler == "plateau":
                    self.scheduler.step(
                        validation_loss
                    )

            if self.config.scheduler == "cosine":
                self.scheduler.step()

            metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                "learning_rate": float(
                    self.optimizer.param_groups[0]["lr"]
                ),
                "epoch_seconds": (
                    time.perf_counter()
                    - start_time
                ),
                **train_metrics,
                **validation_metrics,
            }

            self.log_metrics(
                metrics
            )

            LOGGER.info(
                "Epoch %d/%d | "
                "Train %.6f | "
                "Validation %s | "
                "LR %.8f",
                epoch,
                self.config.epochs,
                train_metrics["train/total"],
                (
                    f"{validation_metrics['val/total']:.6f}"
                    if validation_metrics
                    else "not evaluated"
                ),
                metrics["learning_rate"],
            )

            self.save(
                "latest.pt",
                epoch,
            )

            if improved:
                self.save(
                    "best.pt",
                    epoch,
                )

            if (
                epoch
                % self.config.save_every == 0
            ):
                self.save(
                    f"epoch_{epoch:04d}.pt",
                    epoch,
                )

            if (
                should_validate
                and self.bad_epochs
                >= self.config.early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping at epoch %d",
                    epoch,
                )
                break

        LOGGER.info(
            "Reward training complete. "
            "Best validation loss: %.6f",
            self.best_validation_loss,
        )


# ============================================================
# Configuration loading
# ============================================================

def load_config(
    path: str | Path | None,
) -> RewardTrainingConfig:
    if path is None:
        return RewardTrainingConfig()

    if yaml is None:
        raise ImportError(
            "Install PyYAML: pip install pyyaml"
        )

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, Mapping):
        raise ValueError(
            "Configuration must be a mapping"
        )

    if "training" in data:
        data = data["training"]

    valid_fields = set(
        RewardTrainingConfig.__dataclass_fields__
    )

    values = {
        key: value
        for key, value in data.items()
        if key in valid_fields
    }

    return RewardTrainingConfig(
        **values
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroWorld AI reward model"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--demo-data",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--loss-type",
        choices=[
            "mse",
            "huber",
            "gaussian_nll",
            "binary",
        ],
        default=None,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    args = parse_arguments()

    config = load_config(
        args.config
    )

    values = asdict(config)

    overrides = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss_type": args.loss_type,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "resume": args.resume,
    }

    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    if args.demo_data:
        values["demo_data"] = True

    config = RewardTrainingConfig(
        **values
    )

    if config.demo_data:
        create_demo_dataset(
            config.data_dir,
            config,
        )

    train_loader, validation_loader = (
        create_dataloaders(config)
    )

    trainer = RewardModelTrainer(
        config
    )

    trainer.fit(
        train_loader,
        validation_loader,
    )


if __name__ == "__main__":
    main()
