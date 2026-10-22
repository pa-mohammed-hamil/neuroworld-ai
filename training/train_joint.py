
"""
NeuroWorld AI
File: training/train_joint.py

Joint training for:
    1. Visual tokenizer / image autoencoder
    2. Action-conditioned latent dynamics
    3. Future video-frame prediction
    4. Reward prediction
    5. Action policy learning

Features:
    - Joint end-to-end optimization
    - Configurable multi-task loss weights
    - Image reconstruction
    - Latent transition prediction
    - Future-frame prediction
    - Reward prediction
    - Behavioral-cloning policy objective
    - Teacher-forced and autoregressive rollouts
    - AMP mixed precision
    - Gradient accumulation
    - Gradient clipping
    - Cosine / plateau scheduling
    - Episode-disjoint train/validation split
    - Early stopping
    - Atomic checkpoints
    - Resume training
    - JSONL experiment metrics
    - Synthetic smoke-test dataset

Dataset format:
    Each .npz file represents one episode.

    Required:
        frames:  [T+1, C, H, W]
        actions: [T, action_dim]
        rewards: [T]

    Frames may be float32 in [0, 1] or uint8
    in [0, 255].

    A training sample contains:
        frames:  [horizon+1, C, H, W]
        actions: [horizon, action_dim]
        rewards: [horizon]

Example:
    python -m training.train_joint --demo-data --epochs 3

    python -m training.train_joint \
        --data-dir data/joint \
        --epochs 50

    python -m training.train_joint \
        --data-dir data/joint \
        --resume checkpoints/joint/latest.pt

Dependencies:
    pip install torch numpy pyyaml

Notes:
    This file includes compact reference model components
    to make the pipeline runnable independently.

    Replace JointWorldModel submodules with the actual
    NeuroWorld tokenizer, dynamics, video, reward, and
    policy implementations when integrating the full
    repository.

    Policy learning uses behavioral cloning because
    this dataset contains demonstrated actions.
    Actor-critic or model-based RL requires additional
    returns, value targets, and policy objectives.
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


LOGGER = logging.getLogger(
    "neuroworld.train_joint"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class JointTrainingConfig:
    # Data
    data_dir: str = "data/joint"
    demo_data: bool = False

    validation_split: float = 0.15

    image_channels: int = 3
    image_size: int = 64

    action_dim: int = 8
    horizon: int = 4

    batch_size: int = 8
    num_workers: int = 0

    # Model
    latent_dim: int = 128
    hidden_dim: int = 256

    # Training
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    accumulation_steps: int = 2
    gradient_clip_norm: float = 1.0

    mixed_precision: bool = True
    precision: str = "float16"

    # Loss weights
    reconstruction_weight: float = 1.0
    dynamics_weight: float = 1.0
    video_weight: float = 1.0
    reward_weight: float = 0.5
    policy_weight: float = 0.5
    temporal_weight: float = 0.1

    # Rollout
    teacher_forcing_ratio: float = 0.5

    # Optimization
    scheduler: str = "cosine"
    warmup_epochs: int = 3
    min_learning_rate: float = 1e-6

    # Validation
    validate_every: int = 1

    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 0.0

    # Checkpoints and logs
    checkpoint_dir: str = "checkpoints/joint"
    log_dir: str = "runs/joint"

    save_every: int = 5
    resume: str | None = None

    # Runtime
    device: str = "auto"
    seed: int = 42
    log_every: int = 20

    def __post_init__(self) -> None:
        positive_fields = (
            "image_channels",
            "image_size",
            "action_dim",
            "horizon",
            "batch_size",
            "latent_dim",
            "hidden_dim",
            "epochs",
            "accumulation_steps",
            "validate_every",
            "log_every",
            "save_every",
        )

        for name in positive_fields:
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.image_size % 4 != 0:
            raise ValueError(
                "image_size must be divisible by 4"
            )

        if not 0 < self.validation_split < 1:
            raise ValueError(
                "validation_split must be in (0, 1)"
            )

        if not 0 <= self.teacher_forcing_ratio <= 1:
            raise ValueError(
                "teacher_forcing_ratio must be in [0, 1]"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive"
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

        for name in (
            "reconstruction_weight",
            "dynamics_weight",
            "video_weight",
            "reward_weight",
            "policy_weight",
            "temporal_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
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

class JointSequenceDataset(Dataset):
    """
    Episode-aware image/action/reward dataset.

    Every sample stays within a single episode.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        horizon: int,
        image_channels: int,
        image_size: int,
        action_dim: int,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.horizon = horizon

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: "
                f"{self.data_dir}"
            )

        files = sorted(
            self.data_dir.rglob("*.npz")
        )

        if not files:
            raise FileNotFoundError(
                f"No NPZ episodes in {self.data_dir}"
            )

        self.episodes: list[
            dict[str, Tensor]
        ] = []

        self.index: list[
            tuple[int, int]
        ] = []

        for path in files:
            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                required = {
                    "frames",
                    "actions",
                    "rewards",
                }

                missing = required - set(
                    data.files
                )

                if missing:
                    raise ValueError(
                        f"{path}: missing {sorted(missing)}"
                    )

                frames = np.asarray(
                    data["frames"]
                )

                actions = np.asarray(
                    data["actions"],
                    dtype=np.float32,
                )

                rewards = np.asarray(
                    data["rewards"],
                    dtype=np.float32,
                )

            if frames.ndim != 4:
                raise ValueError(
                    f"{path}: frames must be [T,C,H,W]"
                )

            if (
                frames.shape[1:]
                != (
                    image_channels,
                    image_size,
                    image_size,
                )
            ):
                raise ValueError(
                    f"{path}: image dimensions mismatch"
                )

            if (
                actions.ndim != 2
                or actions.shape[1] != action_dim
            ):
                raise ValueError(
                    f"{path}: action dimensions mismatch"
                )

            if rewards.ndim != 1:
                raise ValueError(
                    f"{path}: rewards must be [T]"
                )

            if not (
                len(frames) == len(actions) + 1
                and len(actions) == len(rewards)
            ):
                raise ValueError(
                    f"{path}: inconsistent episode lengths"
                )

            if len(actions) < horizon:
                LOGGER.warning(
                    "Skipping short episode: %s",
                    path,
                )
                continue

            if frames.dtype == np.uint8:
                frames = (
                    frames.astype(np.float32)
                    / 255.0
                )
            else:
                frames = frames.astype(
                    np.float32
                )

            if not (
                np.isfinite(frames).all()
                and np.isfinite(actions).all()
                and np.isfinite(rewards).all()
            ):
                raise ValueError(
                    f"{path}: non-finite values"
                )

            if (
                frames.min() < 0
                or frames.max() > 1
            ):
                raise ValueError(
                    f"{path}: frames must be in [0,1]"
                )

            episode_index = len(
                self.episodes
            )

            self.episodes.append(
                {
                    "frames": torch.from_numpy(
                        np.ascontiguousarray(frames)
                    ),
                    "actions": torch.from_numpy(
                        np.ascontiguousarray(actions)
                    ),
                    "rewards": torch.from_numpy(
                        np.ascontiguousarray(rewards)
                    ),
                }
            )

            for start in range(
                len(actions) - horizon + 1
            ):
                self.index.append(
                    (episode_index, start)
                )

        if not self.index:
            raise ValueError(
                "No valid training sequences found"
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        episode_index, start = (
            self.index[index]
        )

        episode = self.episodes[
            episode_index
        ]

        end = start + self.horizon

        return {
            "frames": (
                episode["frames"][
                    start:end + 1
                ]
            ),
            "actions": (
                episode["actions"][
                    start:end
                ]
            ),
            "rewards": (
                episode["rewards"][
                    start:end
                ]
            ),
        }


def generate_demo_data(
    directory: str | Path,
    config: JointTrainingConfig,
    *,
    num_episodes: int = 12,
    episode_length: int = 24,
) -> None:
    """
    Generate simple moving-object sequences.

    Synthetic data is intended only for
    pipeline validation and smoke tests.
    """
    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        config.seed
    )

    size = config.image_size
    channels = config.image_channels

    for episode_index in range(
        num_episodes
    ):
        frames = np.zeros(
            (
                episode_length + 1,
                channels,
                size,
                size,
            ),
            dtype=np.float32,
        )

        actions = rng.normal(
            0,
            0.5,
            size=(
                episode_length,
                config.action_dim,
            ),
        ).astype(np.float32)

        rewards = np.zeros(
            episode_length,
            dtype=np.float32,
        )

        x = float(
            rng.integers(4, size - 4)
        )

        y = float(
            rng.integers(4, size - 4)
        )

        for t in range(
            episode_length + 1
        ):
            xi = int(
                np.clip(
                    round(x),
                    2,
                    size - 3,
                )
            )

            yi = int(
                np.clip(
                    round(y),
                    2,
                    size - 3,
                )
            )

            frames[
                t,
                :,
                yi - 2:yi + 3,
                xi - 2:xi + 3,
            ] = 1.0

            if t < episode_length:
                action = actions[t]

                x = np.clip(
                    x + action[0],
                    2,
                    size - 3,
                )

                y = np.clip(
                    y + action[1],
                    2,
                    size - 3,
                )

                distance = math.sqrt(
                    (x - size / 2) ** 2
                    + (y - size / 2) ** 2
                )

                rewards[t] = (
                    -distance / size
                )

        np.savez_compressed(
            directory
            / f"episode_{episode_index:04d}.npz",
            frames=frames,
            actions=actions,
            rewards=rewards,
        )


def create_dataloaders(
    config: JointTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset = JointSequenceDataset(
        config.data_dir,
        horizon=config.horizon,
        image_channels=config.image_channels,
        image_size=config.image_size,
        action_dim=config.action_dim,
    )

    episode_count = len(
        dataset.episodes
    )

    if episode_count < 2:
        raise ValueError(
            "At least two episodes are required"
        )

    rng = np.random.default_rng(
        config.seed
    )

    episode_indices = np.arange(
        episode_count
    )

    rng.shuffle(
        episode_indices
    )

    validation_count = max(
        1,
        min(
            episode_count - 1,
            round(
                episode_count
                * config.validation_split
            ),
        ),
    )

    validation_episodes = set(
        episode_indices[
            :validation_count
        ].tolist()
    )

    train_indices = []
    validation_indices = []

    for sample_index, (
        episode_index,
        _,
    ) in enumerate(dataset.index):
        if episode_index in validation_episodes:
            validation_indices.append(
                sample_index
            )
        else:
            train_indices.append(
                sample_index
            )

    generator = torch.Generator()
    generator.manual_seed(
        config.seed
    )

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": (
            torch.cuda.is_available()
        ),
    }

    train_loader = DataLoader(
        Subset(
            dataset,
            train_indices,
        ),
        shuffle=True,
        generator=generator,
        **loader_options,
    )

    validation_loader = DataLoader(
        Subset(
            dataset,
            validation_indices,
        ),
        shuffle=False,
        **loader_options,
    )

    LOGGER.info(
        "Joint dataset: %d training / "
        "%d validation sequences",
        len(train_indices),
        len(validation_indices),
    )

    return train_loader, validation_loader


# ============================================================
# Visual tokenizer
# ============================================================

class VisualTokenizer(nn.Module):
    """
    Compact convolutional image autoencoder.

    Encodes each image into a continuous
    latent vector.

    The production tokenizer can replace
    this implementation with VQ-VAE or
    another learned visual tokenizer.
    """

    def __init__(
        self,
        *,
        channels: int,
        image_size: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Conv2d(
                channels,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),
            nn.Flatten(),
            nn.Linear(
                64 * 4 * 4,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                64 * 4 * 4,
            ),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                32,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def encode(
        self,
        frames: Tensor,
    ) -> Tensor:
        return self.encoder(
            frames
        )

    def decode(
        self,
        latents: Tensor,
    ) -> Tensor:
        batch_size = latents.shape[0]

        hidden = self.decoder_input(
            latents
        )

        hidden = hidden.reshape(
            batch_size,
            64,
            4,
            4,
        )

        frames = self.decoder(
            hidden
        )

        frames = F.interpolate(
            frames,
            size=(
                self.image_size,
                self.image_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        return torch.sigmoid(
            frames
        )

    def forward(
        self,
        frames: Tensor,
    ) -> tuple[Tensor, Tensor]:
        latents = self.encode(
            frames
        )

        reconstruction = self.decode(
            latents
        )

        return reconstruction, latents


# ============================================================
# Latent dynamics
# ============================================================

class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                latent_dim,
            ),
        )

    def forward(
        self,
        latent: Tensor,
        action: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [latent, action],
            dim=-1,
        )

        delta = self.network(
            features
        )

        return latent + delta


# ============================================================
# Reward model
# ============================================================

class RewardPredictionModel(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        latent: Tensor,
        action: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [latent, action],
            dim=-1,
        )

        return self.network(
            features
        ).squeeze(-1)


# ============================================================
# Policy model
# ============================================================

class ActionPolicyModel(nn.Module):
    """
    Continuous-action behavioral-cloning policy.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
        )

    def forward(
        self,
        latent: Tensor,
    ) -> Tensor:
        return self.network(
            latent
        )


# ============================================================
# Joint world model
# ============================================================

class JointWorldModel(nn.Module):
    """
    Unified visual world model.

    Inputs:
        frames:
            [B, T+1, C, H, W]

        actions:
            [B, T, A]

    Outputs:
        reconstructed_frames:
            [B, T+1, C, H, W]

        target_latents:
            [B, T+1, D]

        predicted_latents:
            [B, T, D]

        predicted_frames:
            [B, T, C, H, W]

        predicted_rewards:
            [B, T]

        predicted_actions:
            [B, T, A]
    """

    def __init__(
        self,
        config: JointTrainingConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.tokenizer = VisualTokenizer(
            channels=config.image_channels,
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
        )

        self.dynamics = LatentDynamicsModel(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        self.reward_model = RewardPredictionModel(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        self.policy = ActionPolicyModel(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        *,
        teacher_forcing_ratio: float = 1.0,
    ) -> dict[str, Tensor]:
        if frames.ndim != 5:
            raise ValueError(
                "frames must be [B,T+1,C,H,W]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,T,A]"
            )

        batch_size, frame_count = (
            frames.shape[:2]
        )

        horizon = actions.shape[1]

        if frame_count != horizon + 1:
            raise ValueError(
                "frames must contain T+1 observations"
            )

        flat_frames = frames.reshape(
            -1,
            *frames.shape[2:],
        )

        reconstructed_flat, latent_flat = (
            self.tokenizer(
                flat_frames
            )
        )

        target_latents = latent_flat.reshape(
            batch_size,
            frame_count,
            -1,
        )

        reconstructed_frames = (
            reconstructed_flat.reshape_as(
                frames
            )
        )

        latent = target_latents[:, 0]

        predicted_latents = []
        predicted_frames = []
        predicted_rewards = []
        predicted_actions = []

        for t in range(horizon):
            action = actions[:, t]

            policy_action = self.policy(
                latent
            )

            predicted_reward = self.reward_model(
                latent,
                action,
            )

            next_latent = self.dynamics(
                latent,
                action,
            )

            next_frame = self.tokenizer.decode(
                next_latent
            )

            predicted_latents.append(
                next_latent
            )

            predicted_frames.append(
                next_frame
            )

            predicted_rewards.append(
                predicted_reward
            )

            predicted_actions.append(
                policy_action
            )

            if t + 1 < horizon:
                if self.training:
                    use_teacher = (
                        torch.rand(
                            (),
                            device=frames.device,
                        ).item()
                        < teacher_forcing_ratio
                    )
                else:
                    use_teacher = (
                        teacher_forcing_ratio >= 1.0
                    )

                if use_teacher:
                    latent = target_latents[
                        :, t + 1
                    ]
                else:
                    latent = next_latent

        return {
            "reconstructed_frames": (
                reconstructed_frames
            ),
            "target_latents": (
                target_latents
            ),
            "predicted_latents": torch.stack(
                predicted_latents,
                dim=1,
            ),
            "predicted_frames": torch.stack(
                predicted_frames,
                dim=1,
            ),
            "predicted_rewards": torch.stack(
                predicted_rewards,
                dim=1,
            ),
            "predicted_actions": torch.stack(
                predicted_actions,
                dim=1,
            ),
        }

    @torch.no_grad()
    def imagine(
        self,
        initial_frame: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        """
        Autoregressive future imagination.

        initial_frame:
            [B,C,H,W]

        actions:
            [B,T,A]
        """
        latent = self.tokenizer.encode(
            initial_frame
        )

        frames = []
        latents = []
        rewards = []

        for t in range(
            actions.shape[1]
        ):
            action = actions[:, t]

            reward = self.reward_model(
                latent,
                action,
            )

            latent = self.dynamics(
                latent,
                action,
            )

            frame = self.tokenizer.decode(
                latent
            )

            frames.append(frame)
            latents.append(latent)
            rewards.append(reward)

        return {
            "frames": torch.stack(
                frames,
                dim=1,
            ),
            "latents": torch.stack(
                latents,
                dim=1,
            ),
            "rewards": torch.stack(
                rewards,
                dim=1,
            ),
        }


# ============================================================
# Multi-task losses
# ============================================================

def compute_joint_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    config: JointTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    frames = batch["frames"].float()
    actions = batch["actions"].float()
    rewards = batch["rewards"].float()

    reconstructed = (
        outputs["reconstructed_frames"].float()
    )

    predicted_frames = (
        outputs["predicted_frames"].float()
    )

    predicted_latents = (
        outputs["predicted_latents"].float()
    )

    target_latents = (
        outputs["target_latents"].float()
    )

    predicted_rewards = (
        outputs["predicted_rewards"].float()
    )

    predicted_actions = (
        outputs["predicted_actions"].float()
    )

    reconstruction_loss = F.mse_loss(
        reconstructed,
        frames,
    )

    # Stop gradients through the target
    # representation for the dynamics objective.
    dynamics_loss = F.mse_loss(
        predicted_latents,
        target_latents[:, 1:].detach(),
    )

    video_loss = F.mse_loss(
        predicted_frames,
        frames[:, 1:],
    )

    reward_loss = F.mse_loss(
        predicted_rewards,
        rewards,
    )

    policy_loss = F.mse_loss(
        predicted_actions,
        actions,
    )

    if predicted_latents.shape[1] > 1:
        predicted_delta = (
            predicted_latents[:, 1:]
            - predicted_latents[:, :-1]
        )

        target_delta = (
            target_latents[:, 2:]
            - target_latents[:, 1:-1]
        ).detach()

        temporal_loss = F.mse_loss(
            predicted_delta,
            target_delta,
        )

    else:
        temporal_loss = (
            predicted_latents.sum() * 0.0
        )

    total = (
        config.reconstruction_weight
        * reconstruction_loss
        + config.dynamics_weight
        * dynamics_loss
        + config.video_weight
        * video_loss
        + config.reward_weight
        * reward_loss
        + config.policy_weight
        * policy_loss
        + config.temporal_weight
        * temporal_loss
    )

    components = {
        "total": total,
        "reconstruction": reconstruction_loss,
        "dynamics": dynamics_loss,
        "video": video_loss,
        "reward": reward_loss,
        "policy": policy_loss,
        "temporal": temporal_loss,
    }

    return total, components


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
        weight: int,
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
        components: Mapping[str, Tensor],
        *,
        weight: int,
    ) -> None:
        for name, value in components.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter()

            self.meters[name].update(
                float(value.detach().item()),
                weight,
            )

    def compute(
        self,
        prefix: str,
    ) -> dict[str, float]:
        return {
            f"{prefix}/{name}": meter.average
            for name, meter in self.meters.items()
        }


# ============================================================
# Learning-rate scheduler
# ============================================================

def create_scheduler(
    optimizer: Optimizer,
    config: JointTrainingConfig,
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
        max(
            config.epochs - 1,
            0,
        ),
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
            max(
                0.0,
                progress,
            ),
        )

        cosine = 0.5 * (
            1.0 + math.cos(
                math.pi * progress
            )
        )

        return (
            min_factor
            + (1.0 - min_factor) * cosine
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )


# ============================================================
# Checkpoint management
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
    config: JointTrainingConfig,
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
        "rng_state": {
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

    rng_state = checkpoint.get(
        "rng_state",
        {},
    )

    if "torch" in rng_state:
        torch.set_rng_state(
            rng_state["torch"].cpu()
        )

    if (
        torch.cuda.is_available()
        and rng_state.get("cuda")
    ):
        for index, state in enumerate(
            rng_state["cuda"][
                :torch.cuda.device_count()
            ]
        ):
            torch.cuda.set_rng_state(
                state.cpu(),
                device=index,
            )

    return checkpoint


# ============================================================
# Experiment logging
# ============================================================

class JSONLExperimentLogger:
    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self.directory = Path(
            directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics_path = (
            self.directory / "metrics.jsonl"
        )

        self.config_path = (
            self.directory / "config.json"
        )

    def log_config(
        self,
        config: JointTrainingConfig,
    ) -> None:
        with self.config_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(config),
                file,
                indent=2,
            )

    def log(
        self,
        metrics: Mapping[str, Any],
    ) -> None:
        record = {
            "timestamp": time.time(),
            **dict(metrics),
        }

        with self.metrics_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    allow_nan=False,
                )
                + "\n"
            )


# ============================================================
# Joint trainer
# ============================================================

class JointTrainer:
    def __init__(
        self,
        config: JointTrainingConfig,
    ) -> None:
        self.config = config

        seed_everything(
            config.seed
        )

        self.device = resolve_device(
            config.device
        )

        self.model = JointWorldModel(
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

        self.logger = JSONLExperimentLogger(
            config.log_dir
        )

        self.logger.log_config(
            config
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

        accumulation_steps = (
            self.config.accumulation_steps
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        total_batches = len(loader)

        for batch_index, raw_batch in enumerate(
            loader
        ):
            batch = self.move_batch(
                raw_batch
            )

            # Scale the final partial
            # accumulation group correctly.
            group_start = (
                batch_index
                // accumulation_steps
            ) * accumulation_steps

            group_size = min(
                accumulation_steps,
                total_batches - group_start,
            )

            with self.autocast():
                outputs = self.model(
                    batch["frames"],
                    batch["actions"],
                    teacher_forcing_ratio=(
                        self.config.teacher_forcing_ratio
                    ),
                )

                loss, components = (
                    compute_joint_loss(
                        outputs,
                        batch,
                        self.config,
                    )
                )

            if not bool(
                torch.isfinite(
                    loss.detach()
                ).item()
            ):
                raise FloatingPointError(
                    "Non-finite joint training loss"
                )

            self.scaler.scale(
                loss / group_size
            ).backward()

            should_step = (
                (batch_index + 1)
                % accumulation_steps == 0
                or (batch_index + 1)
                == total_batches
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

            metrics.update(
                components,
                weight=(
                    batch["frames"].shape[0]
                ),
            )

            if (
                (batch_index + 1)
                % self.config.log_every == 0
            ):
                LOGGER.info(
                    "Epoch %d | Batch %d/%d | "
                    "Joint loss %.6f",
                    epoch,
                    batch_index + 1,
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

        frame_mse = AverageMeter()
        frame_mae = AverageMeter()

        for raw_batch in loader:
            batch = self.move_batch(
                raw_batch
            )

            with self.autocast():
                outputs = self.model(
                    batch["frames"],
                    batch["actions"],
                    teacher_forcing_ratio=0.0,
                )

                _, components = (
                    compute_joint_loss(
                        outputs,
                        batch,
                        self.config,
                    )
                )

            batch_size = (
                batch["frames"].shape[0]
            )

            metrics.update(
                components,
                weight=batch_size,
            )

            predicted_frames = (
                outputs["predicted_frames"].float()
            )

            target_frames = (
                batch["frames"][:, 1:].float()
            )

            frame_mse.update(
                float(
                    F.mse_loss(
                        predicted_frames,
                        target_frames,
                    ).item()
                ),
                batch_size,
            )

            frame_mae.update(
                float(
                    F.l1_loss(
                        predicted_frames,
                        target_frames,
                    ).item()
                ),
                batch_size,
            )

        results = metrics.compute(
            "val"
        )

        results["val/frame_mse"] = (
            frame_mse.average
        )

        results["val/frame_mae"] = (
            frame_mae.average
        )

        results["val/frame_rmse"] = math.sqrt(
            frame_mse.average
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
            "Resumed joint training at epoch %d",
            self.start_epoch,
        )

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> None:
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        LOGGER.info(
            "Device: %s | Parameters: %d",
            self.device,
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
                epoch % self.config.validate_every == 0
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

            record = {
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

            self.logger.log(
                record
            )

            LOGGER.info(
                "Epoch %d/%d | "
                "Train %.6f | "
                "Val %s | "
                "LR %.8f | "
                "Time %.1fs",
                epoch,
                self.config.epochs,
                train_metrics["train/total"],
                (
                    f"{validation_metrics['val/total']:.6f}"
                    if validation_metrics
                    else "not evaluated"
                ),
                record["learning_rate"],
                record["epoch_seconds"],
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
                    "Early stopping triggered"
                )
                break

        LOGGER.info(
            "Joint training complete. "
            "Best validation loss: %.6f",
            self.best_validation_loss,
        )


# ============================================================
# Configuration loading
# ============================================================

def load_config(
    path: str | Path | None,
) -> JointTrainingConfig:
    if path is None:
        return JointTrainingConfig()

    if yaml is None:
        raise ImportError(
            "Install PyYAML: pip install pyyaml"
        )

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(
            file
        ) or {}

    if not isinstance(data, Mapping):
        raise ValueError(
            "Config must be a mapping"
        )

    # Accept a flat YAML file or a
    # nested training configuration.
    if "training" in data:
        data = data["training"]

    valid_fields = set(
        JointTrainingConfig.__dataclass_fields__
    )

    values = {
        key: value
        for key, value in data.items()
        if key in valid_fields
    }

    return JointTrainingConfig(
        **values
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroWorld AI joint world model"
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

    values = asdict(config)

    overrides = {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "resume": args.resume,
        "checkpoint_dir": args.checkpoint_dir,
    }

    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    if args.demo_data:
        values["demo_data"] = True

    config = JointTrainingConfig(
        **values
    )

    if config.demo_data:
        generate_demo_data(
            config.data_dir,
            config,
        )

    train_loader, validation_loader = (
        create_dataloaders(
            config
        )
    )

    trainer = JointTrainer(
        config
    )

    trainer.fit(
        train_loader,
        validation_loader,
    )


if __name__ == "__main__":
    main()
