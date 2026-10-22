
"""
NeuroWorld AI
File: training/train_dynamics.py

Train action-conditioned dynamics models.

Supported data:
    NPZ trajectory files containing:
        states:      [N, state_dim]
        actions:     [N, action_dim]
        next_states: [N, state_dim]

Optional:
        episode_ids: [N]

Example:
    python -m training.train_dynamics \
        --data-dir data/trajectories \
        --epochs 50 \
        --batch-size 64

Generate demo data:
    python -m training.train_dynamics \
        --demo-data \
        --epochs 5

Resume:
    python -m training.train_dynamics \
        --data-dir data/trajectories \
        --resume checkpoints/dynamics/latest.pt

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
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

try:
    import yaml
except ImportError:
    yaml = None


LOGGER = logging.getLogger("neuroworld.train_dynamics")


# ============================================================
# Configuration
# ============================================================

@dataclass
class DynamicsTrainingConfig:
    # Data
    data_dir: str = "data/trajectories"
    demo_data: bool = False

    validation_split: float = 0.15

    state_dim: int = 128
    action_dim: int = 32

    batch_size: int = 64
    num_workers: int = 0

    # Model
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1

    probabilistic: bool = True
    predict_residual: bool = True

    min_logvar: float = -10.0
    max_logvar: float = 2.0

    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0

    mixed_precision: bool = True
    precision: str = "float16"

    # Scheduler
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_learning_rate: float = 1e-6

    # Loss
    prediction_weight: float = 1.0
    nll_weight: float = 1.0
    smoothness_weight: float = 0.0

    # Validation
    validate_every: int = 1
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0

    # Checkpoints
    checkpoint_dir: str = "checkpoints/dynamics"
    log_dir: str = "runs/dynamics"

    save_every: int = 5
    resume: str | None = None

    # Runtime
    device: str = "auto"
    seed: int = 42

    log_every: int = 20

    def __post_init__(self) -> None:
        if self.state_dim < 1:
            raise ValueError("state_dim must be positive")

        if self.action_dim < 1:
            raise ValueError("action_dim must be positive")

        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

        if self.epochs < 1:
            raise ValueError("epochs must be positive")

        if self.accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be positive"
            )

        if not 0 < self.validation_split < 1:
            raise ValueError(
                "validation_split must be between 0 and 1"
            )

        if self.scheduler not in {
            "cosine",
            "plateau",
            "none",
        }:
            raise ValueError("Unsupported scheduler")

        if self.precision not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError("Unsupported precision")

        if self.validate_every < 1:
            raise ValueError(
                "validate_every must be positive"
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

class DynamicsDataset(Dataset):
    """
    Loads transition tuples from NPZ files.

    Each NPZ file must contain:
        states
        actions
        next_states

    Data is loaded into CPU memory.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory missing: {self.data_dir}"
            )

        files = sorted(
            self.data_dir.rglob("*.npz")
        )

        if not files:
            raise FileNotFoundError(
                f"No NPZ files found in {self.data_dir}"
            )

        states_list = []
        actions_list = []
        next_states_list = []
        episode_ids_list = []

        episode_offset = 0

        for file_path in files:
            with np.load(
                file_path,
                allow_pickle=False,
            ) as data:
                required = {
                    "states",
                    "actions",
                    "next_states",
                }

                missing = required - set(data.files)

                if missing:
                    raise ValueError(
                        f"{file_path}: missing {sorted(missing)}"
                    )

                states = np.asarray(
                    data["states"],
                    dtype=np.float32,
                )

                actions = np.asarray(
                    data["actions"],
                    dtype=np.float32,
                )

                next_states = np.asarray(
                    data["next_states"],
                    dtype=np.float32,
                )

                if (
                    states.ndim != 2
                    or actions.ndim != 2
                    or next_states.ndim != 2
                ):
                    raise ValueError(
                        f"{file_path}: expected 2D arrays"
                    )

                n = len(states)

                if not (
                    n == len(actions) == len(next_states)
                ):
                    raise ValueError(
                        f"{file_path}: inconsistent lengths"
                    )

                if (
                    states.shape[1] != state_dim
                    or next_states.shape[1] != state_dim
                    or actions.shape[1] != action_dim
                ):
                    raise ValueError(
                        f"{file_path}: dimension mismatch"
                    )

                if n == 0:
                    continue

                if not (
                    np.isfinite(states).all()
                    and np.isfinite(actions).all()
                    and np.isfinite(next_states).all()
                ):
                    raise ValueError(
                        f"{file_path}: non-finite values"
                    )

                if "episode_ids" in data.files:
                    raw_ids = np.asarray(
                        data["episode_ids"]
                    )

                    if len(raw_ids) != n:
                        raise ValueError(
                            f"{file_path}: episode_ids mismatch"
                        )

                    _, ids = np.unique(
                        raw_ids,
                        return_inverse=True,
                    )

                    ids = ids.astype(np.int64)

                else:
                    # Treat each file as one episode
                    # when explicit IDs are unavailable.
                    ids = np.zeros(
                        n,
                        dtype=np.int64,
                    )

                ids += episode_offset

                episode_offset = int(
                    ids.max()
                ) + 1

                states_list.append(states)
                actions_list.append(actions)
                next_states_list.append(next_states)
                episode_ids_list.append(ids)

        if not states_list:
            raise ValueError(
                "Dataset contains no transitions"
            )

        self.states = torch.from_numpy(
            np.concatenate(
                states_list,
                axis=0,
            )
        )

        self.actions = torch.from_numpy(
            np.concatenate(
                actions_list,
                axis=0,
            )
        )

        self.next_states = torch.from_numpy(
            np.concatenate(
                next_states_list,
                axis=0,
            )
        )

        self.episode_ids = np.concatenate(
            episode_ids_list,
            axis=0,
        )

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "next_states": self.next_states[index],
        }


def create_demo_dataset(
    data_dir: str | Path,
    *,
    state_dim: int,
    action_dim: int,
    num_episodes: int = 24,
    episode_length: int = 128,
    seed: int = 42,
) -> None:
    """
    Generate synthetic action-conditioned transitions.

    Intended for smoke testing only.
    """
    directory = Path(data_dir)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(seed)

    action_projection = rng.normal(
        0.0,
        0.1,
        size=(action_dim, state_dim),
    ).astype(np.float32)

    for episode in range(num_episodes):
        state = rng.normal(
            size=state_dim
        ).astype(np.float32)

        states = []
        actions = []
        next_states = []

        for _ in range(episode_length):
            action = rng.normal(
                size=action_dim
            ).astype(np.float32)

            noise = rng.normal(
                0.0,
                0.01,
                size=state_dim,
            ).astype(np.float32)

            next_state = (
                0.96 * state
                + 0.04 * np.tanh(state)
                + action @ action_projection
                + noise
            ).astype(np.float32)

            states.append(state.copy())
            actions.append(action)
            next_states.append(next_state.copy())

            state = next_state

        np.savez_compressed(
            directory / f"episode_{episode:04d}.npz",
            states=np.stack(states),
            actions=np.stack(actions),
            next_states=np.stack(next_states),
        )


def create_dataloaders(
    config: DynamicsTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset = DynamicsDataset(
        config.data_dir,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
    )

    # Split by episode to reduce leakage
    # between training and validation.
    episode_ids = np.unique(
        dataset.episode_ids
    )

    if len(episode_ids) < 2:
        raise ValueError(
            "At least two episodes are required "
            "for an episode-disjoint split"
        )

    rng = np.random.default_rng(
        config.seed
    )

    rng.shuffle(episode_ids)

    validation_count = max(
        1,
        min(
            len(episode_ids) - 1,
            round(
                len(episode_ids)
                * config.validation_split
            ),
        ),
    )

    validation_episodes = set(
        episode_ids[:validation_count].tolist()
    )

    train_indices = [
        index
        for index, episode in enumerate(
            dataset.episode_ids
        )
        if int(episode) not in validation_episodes
    ]

    validation_indices = [
        index
        for index, episode in enumerate(
            dataset.episode_ids
        )
        if int(episode) in validation_episodes
    ]

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=generator,
        **loader_options,
    )

    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        **loader_options,
    )

    LOGGER.info(
        "Transitions: train=%d validation=%d",
        len(train_indices),
        len(validation_indices),
    )

    return train_loader, validation_loader


# ============================================================
# Dynamics model
# ============================================================

class DynamicsModel(nn.Module):
    """
    Action-conditioned transition model.

    Inputs:
        states:  [B, state_dim]
        actions: [B, action_dim]

    Outputs:
        mean:   [B, state_dim]
        logvar: [B, state_dim] or None
    """

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        probabilistic: bool = True,
        predict_residual: bool = True,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.probabilistic = probabilistic
        self.predict_residual = predict_residual

        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        layers: list[nn.Module] = []

        input_dim = state_dim + action_dim

        for layer_index in range(num_layers):
            layers.extend(
                [
                    nn.Linear(
                        input_dim if layer_index == 0
                        else hidden_dim,
                        hidden_dim,
                    ),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        self.backbone = nn.Sequential(
            *layers
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            state_dim,
        )

        self.logvar_head = (
            nn.Linear(
                hidden_dim,
                state_dim,
            )
            if probabilistic
            else None
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                "Expected [batch, features] tensors"
            )

        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                "State/action batch mismatch"
            )

        features = torch.cat(
            [states, actions],
            dim=-1,
        )

        hidden = self.backbone(
            features
        )

        mean = self.mean_head(
            hidden
        )

        if self.predict_residual:
            mean = states + mean

        logvar = None

        if self.logvar_head is not None:
            logvar = self.logvar_head(
                hidden
            ).clamp(
                min=self.min_logvar,
                max=self.max_logvar,
            )

        return mean, logvar

    @torch.no_grad()
    def predict(
        self,
        states: Tensor,
        actions: Tensor,
        *,
        sample: bool = False,
    ) -> Tensor:
        mean, logvar = self(
            states,
            actions,
        )

        if not sample or logvar is None:
            return mean

        noise = torch.randn_like(mean)

        return (
            mean
            + torch.exp(0.5 * logvar) * noise
        )

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """
        Differentiable autoregressive rollout.

        initial_state:
            [B, state_dim]

        actions:
            [B, horizon, action_dim]

        Returns:
            [B, horizon, state_dim]
        """
        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,T,A]"
            )

        state = initial_state
        predictions = []

        for time_index in range(
            actions.shape[1]
        ):
            state, _ = self(
                state,
                actions[:, time_index],
            )

            predictions.append(state)

        return torch.stack(
            predictions,
            dim=1,
        )


def build_model(
    config: DynamicsTrainingConfig,
) -> DynamicsModel:
    return DynamicsModel(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        probabilistic=config.probabilistic,
        predict_residual=config.predict_residual,
        min_logvar=config.min_logvar,
        max_logvar=config.max_logvar,
    )


# ============================================================
# Losses
# ============================================================

def compute_dynamics_loss(
    mean: Tensor,
    logvar: Tensor | None,
    target: Tensor,
    config: DynamicsTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    mean = mean.float()
    target = target.float()

    prediction_loss = F.mse_loss(
        mean,
        target,
    )

    total = (
        config.prediction_weight
        * prediction_loss
    )

    components = {
        "prediction": prediction_loss,
    }

    if logvar is not None:
        logvar = logvar.float()

        inverse_variance = torch.exp(
            -logvar
        )

        nll = 0.5 * (
            logvar
            + (target - mean).square()
            * inverse_variance
            + math.log(2.0 * math.pi)
        )

        nll_loss = nll.mean()

        total = (
            total
            + config.nll_weight * nll_loss
        )

        components["nll"] = nll_loss

    components["total"] = total

    return total, components


# ============================================================
# Metrics
# ============================================================

class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0

    def update(
        self,
        value: float,
        count: int = 1,
    ) -> None:
        self.total += value * count
        self.weight += count

    @property
    def average(self) -> float:
        return (
            self.total / self.weight
            if self.weight
            else 0.0
        )


def calculate_gradient_norm(
    model: nn.Module,
) -> float:
    total = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach()

        total += float(
            gradient.float()
            .square()
            .sum()
            .item()
        )

    return math.sqrt(total)


# ============================================================
# Scheduler
# ============================================================

def create_scheduler(
    optimizer: Optimizer,
    config: DynamicsTrainingConfig,
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

    def learning_rate_factor(epoch: int) -> float:
        if (
            warmup_epochs > 0
            and epoch < warmup_epochs
        ):
            return max(
                1e-3,
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
            max(progress, 0.0),
            1.0,
        )

        min_factor = (
            config.min_learning_rate
            / config.learning_rate
        )

        return (
            min_factor
            + (1.0 - min_factor)
            * 0.5
            * (1.0 + math.cos(math.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )


# ============================================================
# Checkpoint utilities
# ============================================================

def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    bad_epochs: int,
    config: DynamicsTrainingConfig,
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
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
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

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
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

    if (
        scaler is not None
        and checkpoint.get("scaler") is not None
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    rng = checkpoint.get("rng", {})

    if "torch" in rng:
        torch.set_rng_state(
            rng["torch"].cpu()
        )

    if (
        torch.cuda.is_available()
        and rng.get("cuda")
    ):
        saved_cuda_states = rng["cuda"]

        for index, state in enumerate(
            saved_cuda_states[
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

class DynamicsTrainer:
    def __init__(
        self,
        config: DynamicsTrainingConfig,
    ) -> None:
        self.config = config

        seed_everything(
            config.seed
        )

        self.device = resolve_device(
            config.device
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

        self.model = build_model(
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
            and self.device.type in {
                "cuda",
                "cpu",
            }
            and config.precision != "float32"
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

        self.start_epoch = 1
        self.global_step = 0

        self.best_validation_loss = float("inf")
        self.bad_epochs = 0

        self.history: list[
            dict[str, float | int]
        ] = []

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

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        nll_meter = AverageMeter()

        accumulation_steps = (
            self.config.accumulation_steps
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        num_batches = len(loader)

        for batch_index, raw_batch in enumerate(
            loader
        ):
            batch = self.move_batch(
                raw_batch
            )

            states = batch["states"]
            actions = batch["actions"]
            targets = batch["next_states"]

            # Correctly scale the final incomplete
            # accumulation group.
            group_start = (
                batch_index
                // accumulation_steps
            ) * accumulation_steps

            group_size = min(
                accumulation_steps,
                num_batches - group_start,
            )

            with self.autocast():
                mean, logvar = self.model(
                    states,
                    actions,
                )

                loss, components = (
                    compute_dynamics_loss(
                        mean,
                        logvar,
                        targets,
                        self.config,
                    )
                )

            if not bool(
                torch.isfinite(loss.detach()).item()
            ):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, "
                    f"batch={batch_index}"
                )

            scaled_loss = (
                loss / group_size
            )

            self.scaler.scale(
                scaled_loss
            ).backward()

            should_step = (
                (batch_index + 1)
                % accumulation_steps == 0
                or (batch_index + 1) == num_batches
            )

            if should_step:
                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=(
                        self.config.gradient_clip_norm
                    ),
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

            batch_size = states.shape[0]

            loss_meter.update(
                float(loss.detach().item()),
                batch_size,
            )

            mse_meter.update(
                float(
                    components["prediction"]
                    .detach()
                    .item()
                ),
                batch_size,
            )

            if "nll" in components:
                nll_meter.update(
                    float(
                        components["nll"]
                        .detach()
                        .item()
                    ),
                    batch_size,
                )

            if (
                (batch_index + 1)
                % self.config.log_every == 0
            ):
                LOGGER.info(
                    "Epoch %d | Batch %d/%d | "
                    "Loss %.6f | LR %.8f",
                    epoch,
                    batch_index + 1,
                    num_batches,
                    loss_meter.average,
                    self.optimizer.param_groups[0]["lr"],
                )

        return {
            "train_loss": loss_meter.average,
            "train_mse": mse_meter.average,
            "train_nll": nll_meter.average,
        }

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
    ) -> dict[str, float]:
        self.model.eval()

        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()
        nll_meter = AverageMeter()

        for raw_batch in loader:
            batch = self.move_batch(
                raw_batch
            )

            states = batch["states"]
            actions = batch["actions"]
            targets = batch["next_states"]

            with self.autocast():
                mean, logvar = self.model(
                    states,
                    actions,
                )

                loss, components = (
                    compute_dynamics_loss(
                        mean,
                        logvar,
                        targets,
                        self.config,
                    )
                )

            batch_size = states.shape[0]

            loss_meter.update(
                float(loss.item()),
                batch_size,
            )

            mse_meter.update(
                float(
                    components["prediction"].item()
                ),
                batch_size,
            )

            mae_meter.update(
                float(
                    F.l1_loss(
                        mean.float(),
                        targets.float(),
                    ).item()
                ),
                batch_size,
            )

            if "nll" in components:
                nll_meter.update(
                    float(
                        components["nll"].item()
                    ),
                    batch_size,
                )

        return {
            "val_loss": loss_meter.average,
            "val_mse": mse_meter.average,
            "val_mae": mae_meter.average,
            "val_rmse": math.sqrt(
                mse_meter.average
            ),
            "val_nll": nll_meter.average,
        }

    def resume(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint = load_checkpoint(
            checkpoint_path,
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
            "Resumed from %s at epoch %d",
            checkpoint_path,
            self.start_epoch,
        )

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

    def save_history(self) -> None:
        path = (
            self.log_dir / "training_history.json"
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=2,
            )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> list[dict[str, float | int]]:
        LOGGER.info(
            "Training dynamics model on %s",
            self.device,
        )

        LOGGER.info(
            "Parameters: %s",
            sum(
                parameter.numel()
                for parameter in self.model.parameters()
            ),
        )

        for epoch in range(
            self.start_epoch,
            self.config.epochs + 1,
        ):
            epoch_start = time.perf_counter()

            train_metrics = self.train_epoch(
                train_loader,
                epoch,
            )

            validation_metrics: dict[
                str, float
            ] = {}

            improved = False

            if (
                epoch
                % self.config.validate_every == 0
                or epoch == self.config.epochs
            ):
                validation_metrics = self.validate(
                    validation_loader
                )

                validation_loss = (
                    validation_metrics["val_loss"]
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
                    - epoch_start
                ),
                **train_metrics,
                **validation_metrics,
            }

            self.history.append(
                metrics
            )

            LOGGER.info(
                "Epoch %d/%d | "
                "Train %.6f | "
                "Val %s | "
                "LR %.8f | "
                "Time %.1fs",
                epoch,
                self.config.epochs,
                train_metrics["train_loss"],
                (
                    f"{validation_metrics['val_loss']:.6f}"
                    if validation_metrics
                    else "not evaluated"
                ),
                metrics["learning_rate"],
                metrics["epoch_seconds"],
            )

            # Save latest before best to ensure
            # latest contains updated best state.
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
                epoch % self.config.save_every == 0
            ):
                self.save(
                    f"epoch_{epoch:04d}.pt",
                    epoch,
                )

            self.save_history()

            if (
                validation_metrics
                and self.bad_epochs
                >= self.config.early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping at epoch %d",
                    epoch,
                )
                break

        return self.history


# ============================================================
# Configuration loading
# ============================================================

def load_config(
    path: str | Path | None,
) -> DynamicsTrainingConfig:
    if path is None:
        return DynamicsTrainingConfig()

    if yaml is None:
        raise ImportError(
            "PyYAML is required for YAML config files. "
            "Install with: pip install pyyaml"
        )

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, Mapping):
        raise ValueError(
            "YAML config must contain a mapping"
        )

    # Accept flat configuration or
    # a nested 'training' configuration.
    if "training" in data:
        training_section = data["training"]

        if not isinstance(
            training_section,
            Mapping,
        ):
            raise ValueError(
                "training must be a mapping"
            )

        data = training_section

    valid_fields = set(
        DynamicsTrainingConfig.__dataclass_fields__
    )

    values = {
        key: value
        for key, value in data.items()
        if key in valid_fields
    }

    return DynamicsTrainingConfig(
        **values
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroWorld AI dynamics model"
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
        "--learning-rate",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--action-dim",
        type=int,
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

    parser.add_argument(
        "--checkpoint-dir",
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

    overrides = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "device": args.device,
        "resume": args.resume,
        "checkpoint_dir": args.checkpoint_dir,
    }

    values = asdict(config)

    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    if args.demo_data:
        values["demo_data"] = True

    config = DynamicsTrainingConfig(
        **values
    )

    if config.demo_data:
        create_demo_dataset(
            config.data_dir,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            seed=config.seed,
        )

    train_loader, validation_loader = (
        create_dataloaders(config)
    )

    trainer = DynamicsTrainer(
        config
    )

    trainer.fit(
        train_loader,
        validation_loader,
    )

    LOGGER.info(
        "Training completed. "
        "Best validation loss: %.6f",
        trainer.best_validation_loss,
    )

    LOGGER.info(
        "Checkpoints: %s",
        trainer.checkpoint_dir.resolve(),
    )


if __name__ == "__main__":
    main()
