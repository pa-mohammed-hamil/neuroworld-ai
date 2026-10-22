
"""
NeuroWorld AI
File: training/train_world_model.py

Action-conditioned latent world-model training.

Architecture:
    Observation Encoder
          |
          v
    Latent State Encoder
          |
          +---- Action ----+
          |                |
          v                v
    Latent Dynamics / Transition Model
          |
          +---- Next State Decoder
          |
          +---- Reward Predictor
          |
          +---- Termination Predictor

Features:
    - Continuous latent-state world model
    - Action-conditioned transition dynamics
    - Observation reconstruction
    - Next-state prediction
    - Reward prediction
    - Termination prediction
    - Latent consistency loss
    - Multi-step imagined rollout
    - Teacher forcing
    - AMP mixed precision
    - Gradient accumulation
    - Gradient clipping
    - Cosine / plateau scheduler
    - Validation
    - Early stopping
    - Atomic checkpointing
    - Resume training
    - JSONL metrics
    - Synthetic trajectory dataset

Dataset:
    data/trajectories/
        episode_0001.npz
        episode_0002.npz

Each NPZ:
    states:      [T, state_dim]
    actions:     [T, action_dim]
    next_states: [T, state_dim]
    rewards:     [T]
    terminated:  [T]  (optional)

Each file represents one episode.

Examples:
    python -m training.train_world_model --demo-data --epochs 5

    python -m training.train_world_model \
        --data-dir data/trajectories \
        --epochs 50

    python -m training.train_world_model \
        --resume checkpoints/world_model/latest.pt

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
    "neuroworld.train_world_model"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class WorldModelTrainingConfig:
    data_dir: str = "data/trajectories"
    demo_data: bool = False

    state_dim: int = 128
    action_dim: int = 32
    latent_dim: int = 128
    hidden_dim: int = 256

    sequence_length: int = 8
    sequence_stride: int = 1

    num_layers: int = 3
    dropout: float = 0.1

    reconstruction_weight: float = 0.5
    prediction_weight: float = 1.0
    latent_weight: float = 0.25
    reward_weight: float = 0.5
    termination_weight: float = 0.1

    rollout_discount: float = 0.95

    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    teacher_forcing_decay_epochs: int = 40

    validation_split: float = 0.15

    batch_size: int = 32
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

    checkpoint_dir: str = "checkpoints/world_model"
    log_dir: str = "runs/world_model"

    save_every: int = 5
    resume: str | None = None

    device: str = "auto"
    seed: int = 42
    log_every: int = 20

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "state_dim",
            "action_dim",
            "latent_dim",
            "hidden_dim",
            "sequence_length",
            "sequence_stride",
            "num_layers",
            "batch_size",
            "epochs",
            "accumulation_steps",
            "validate_every",
            "save_every",
            "log_every",
            "teacher_forcing_decay_epochs",
        )

        for name in positive_integer_fields:
            if getattr(self, name) < 1:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not 0 < self.validation_split < 1:
            raise ValueError(
                "validation_split must be in (0,1)"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "gradient_clip_norm must be positive"
            )

        if not 0 < self.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in (0,1]"
            )

        for name in (
            "teacher_forcing_start",
            "teacher_forcing_end",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be in [0,1]"
                )

        for name in (
            "reconstruction_weight",
            "prediction_weight",
            "latent_weight",
            "reward_weight",
            "termination_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
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


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(
    name: str,
) -> torch.device:
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
# Trajectory dataset
# ============================================================

class WorldModelDataset(Dataset):
    """
    Episode-safe sequence dataset.

    Returns:
        states:      [L, state_dim]
        actions:     [L, action_dim]
        next_states: [L, state_dim]
        rewards:     [L]
        terminated:  [L]

    Sequence windows never cross episode boundaries.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: WorldModelTrainingConfig,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset: {self.data_dir}"
            )

        self.episodes: list[
            dict[str, Tensor]
        ] = []

        self.index: list[
            tuple[int, int]
        ] = []

        files = sorted(
            self.data_dir.rglob("*.npz")
        )

        if not files:
            raise FileNotFoundError(
                "No NPZ trajectory files found"
            )

        for path in files:
            with np.load(
                path,
                allow_pickle=False,
            ) as data:
                required = {
                    "states",
                    "actions",
                    "next_states",
                    "rewards",
                }

                missing = required - set(
                    data.files
                )

                if missing:
                    raise ValueError(
                        f"{path}: missing {sorted(missing)}"
                    )

                arrays = {
                    key: np.asarray(
                        data[key],
                        dtype=np.float32,
                    )
                    for key in required
                }

                if "terminated" in data.files:
                    terminated = np.asarray(
                        data["terminated"],
                        dtype=np.float32,
                    )
                else:
                    terminated = np.zeros(
                        len(arrays["states"]),
                        dtype=np.float32,
                    )

            states = arrays["states"]
            actions = arrays["actions"]
            next_states = arrays["next_states"]
            rewards = arrays["rewards"]

            n = len(states)

            if n < config.sequence_length:
                continue

            expected_shapes = {
                "states": (
                    n,
                    config.state_dim,
                ),
                "actions": (
                    n,
                    config.action_dim,
                ),
                "next_states": (
                    n,
                    config.state_dim,
                ),
                "rewards": (n,),
                "terminated": (n,),
            }

            arrays["terminated"] = terminated

            for name, shape in expected_shapes.items():
                if arrays[name].shape != shape:
                    raise ValueError(
                        f"{path}: {name} has shape "
                        f"{arrays[name].shape}, "
                        f"expected {shape}"
                    )

                if not np.isfinite(
                    arrays[name]
                ).all():
                    raise ValueError(
                        f"{path}: non-finite {name}"
                    )

            if not np.isin(
                terminated,
                [0.0, 1.0],
            ).all():
                raise ValueError(
                    f"{path}: terminated must be binary"
                )

            episode_index = len(
                self.episodes
            )

            self.episodes.append(
                {
                    name: torch.from_numpy(
                        np.ascontiguousarray(
                            array
                        )
                    )
                    for name, array in arrays.items()
                }
            )

            for start in range(
                0,
                n - config.sequence_length + 1,
                config.sequence_stride,
            ):
                end = (
                    start
                    + config.sequence_length
                )

                # Do not include a terminal transition
                # before the final step in a window.
                if (
                    terminated[
                        start:end - 1
                    ] > 0
                ).any():
                    continue

                self.index.append(
                    (episode_index, start)
                )

        if not self.index:
            raise ValueError(
                "No valid world-model sequences found"
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

        end = (
            start
            + self.config.sequence_length
        )

        episode = self.episodes[
            episode_index
        ]

        return {
            name: tensor[
                start:end
            ]
            for name, tensor in episode.items()
        }


# ============================================================
# Synthetic trajectory generation
# ============================================================

def create_demo_dataset(
    directory: str | Path,
    config: WorldModelTrainingConfig,
    *,
    num_episodes: int = 20,
    episode_length: int = 96,
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
        0.05,
        size=(
            config.action_dim,
            config.state_dim,
        ),
    ).astype(np.float32)

    episode_length = max(
        episode_length,
        config.sequence_length + 1,
    )

    for episode_index in range(
        num_episodes
    ):
        states = np.zeros(
            (
                episode_length,
                config.state_dim,
            ),
            dtype=np.float32,
        )

        actions = rng.normal(
            size=(
                episode_length,
                config.action_dim,
            )
        ).astype(np.float32)

        next_states = np.zeros_like(
            states
        )

        rewards = np.zeros(
            episode_length,
            dtype=np.float32,
        )

        terminated = np.zeros(
            episode_length,
            dtype=np.float32,
        )

        current = rng.normal(
            0,
            0.5,
            size=config.state_dim,
        ).astype(np.float32)

        for timestep in range(
            episode_length
        ):
            states[timestep] = current

            next_state = (
                0.95 * current
                + actions[timestep]
                @ action_projection
            )

            next_state += rng.normal(
                0,
                0.01,
                size=config.state_dim,
            )

            next_state = next_state.astype(
                np.float32
            )

            next_states[timestep] = next_state

            rewards[timestep] = (
                -np.mean(
                    next_state.square()
                )
                - 0.05
                * np.mean(
                    actions[timestep].square()
                )
            )

            current = next_state

        terminated[-1] = 1.0

        np.savez_compressed(
            directory
            / f"episode_{episode_index:04d}.npz",
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminated=terminated,
        )


# ============================================================
# Data loaders
# ============================================================

def create_dataloaders(
    config: WorldModelTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset = WorldModelDataset(
        config.data_dir,
        config,
    )

    episode_count = len(
        dataset.episodes
    )

    if episode_count < 2:
        raise ValueError(
            "At least two independent episodes "
            "are required for validation"
        )

    rng = np.random.default_rng(
        config.seed
    )

    episode_ids = np.arange(
        episode_count
    )

    rng.shuffle(
        episode_ids
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
        episode_ids[
            :validation_count
        ].tolist()
    )

    train_indices = []
    validation_indices = []

    for index, (
        episode_id,
        _,
    ) in enumerate(dataset.index):
        if episode_id in validation_episodes:
            validation_indices.append(index)
        else:
            train_indices.append(index)

    generator = torch.Generator()
    generator.manual_seed(
        config.seed
    )

    options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        Subset(
            dataset,
            train_indices,
        ),
        shuffle=True,
        generator=generator,
        **options,
    )

    validation_loader = DataLoader(
        Subset(
            dataset,
            validation_indices,
        ),
        shuffle=False,
        **options,
    )

    LOGGER.info(
        "World-model sequences: "
        "train=%d validation=%d",
        len(train_indices),
        len(validation_indices),
    )

    return train_loader, validation_loader


# ============================================================
# MLP building block
# ============================================================

def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []

    current_dim = input_dim

    for _ in range(
        num_layers
    ):
        layers.extend(
            [
                nn.Linear(
                    current_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(
                    hidden_dim
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout
                ),
            ]
        )

        current_dim = hidden_dim

    layers.append(
        nn.Linear(
            current_dim,
            output_dim,
        )
    )

    return nn.Sequential(
        *layers
    )


# ============================================================
# World model
# ============================================================

class LatentWorldModel(nn.Module):
    """
    Continuous latent-state world model.

    Supports:
        - Encoding observations
        - Predicting next latent states
        - Decoding observations
        - Predicting rewards
        - Predicting termination probabilities
        - Multi-step imagined rollouts
    """

    def __init__(
        self,
        config: WorldModelTrainingConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.encoder = build_mlp(
            input_dim=config.state_dim,
            output_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.decoder = build_mlp(
            input_dim=config.latent_dim,
            output_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.dynamics = build_mlp(
            input_dim=(
                config.latent_dim
                + config.action_dim
            ),
            output_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        self.reward_head = build_mlp(
            input_dim=(
                config.latent_dim
                + config.action_dim
                + config.latent_dim
            ),
            output_dim=1,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )

        self.termination_head = build_mlp(
            input_dim=(
                config.latent_dim
                + config.action_dim
                + config.latent_dim
            ),
            output_dim=1,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            dropout=config.dropout,
        )

    def encode(
        self,
        states: Tensor,
    ) -> Tensor:
        return self.encoder(
            states
        )

    def decode(
        self,
        latents: Tensor,
    ) -> Tensor:
        return self.decoder(
            latents
        )

    def transition(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [
                latents,
                actions,
            ],
            dim=-1,
        )

        delta = self.dynamics(
            features
        )

        return latents + delta

    def predict_step(
        self,
        latents: Tensor,
        actions: Tensor,
    ) -> dict[str, Tensor]:
        next_latents = self.transition(
            latents,
            actions,
        )

        features = torch.cat(
            [
                latents,
                actions,
                next_latents,
            ],
            dim=-1,
        )

        predicted_reward = (
            self.reward_head(
                features
            ).squeeze(-1)
        )

        termination_logits = (
            self.termination_head(
                features
            ).squeeze(-1)
        )

        predicted_state = self.decode(
            next_latents
        )

        return {
            "next_latents": next_latents,
            "next_states": predicted_state,
            "rewards": predicted_reward,
            "termination_logits": (
                termination_logits
            ),
        }

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        *,
        next_states: Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> dict[str, Tensor]:
        if states.ndim != 3:
            raise ValueError(
                "states must be [B,T,state_dim]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,T,action_dim]"
            )

        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "State/action sequence dimensions mismatch"
            )

        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError(
                "Invalid teacher forcing ratio"
            )

        batch_size, sequence_length, _ = (
            states.shape
        )

        # Encode all observed states for reconstruction.
        observed_latents = self.encode(
            states
        )

        reconstructed_states = self.decode(
            observed_latents
        )

        # Start rollout from the first observed state.
        current_latents = observed_latents[
            :,
            0,
        ]

        predicted_latents = []
        predicted_states = []
        predicted_rewards = []
        termination_logits = []

        for timestep in range(
            sequence_length
        ):
            step = self.predict_step(
                current_latents,
                actions[:, timestep],
            )

            predicted_latents.append(
                step["next_latents"]
            )

            predicted_states.append(
                step["next_states"]
            )

            predicted_rewards.append(
                step["rewards"]
            )

            termination_logits.append(
                step["termination_logits"]
            )

            if timestep == sequence_length - 1:
                break

            current_latents = (
                step["next_latents"]
            )

            if (
                self.training
                and teacher_forcing_ratio > 0
            ):
                teacher_mask = (
                    torch.rand(
                        (
                            batch_size,
                            1,
                        ),
                        device=states.device,
                    )
                    < teacher_forcing_ratio
                )

                # Use the actual next state for
                # the next recurrent rollout step.
                if next_states is not None:
                    teacher_latents = self.encode(
                        next_states[
                            :,
                            timestep,
                        ]
                    )
                else:
                    teacher_latents = observed_latents[
                        :,
                        timestep + 1,
                    ]

                current_latents = torch.where(
                    teacher_mask,
                    teacher_latents,
                    current_latents,
                )

        return {
            "reconstructed_states": (
                reconstructed_states
            ),
            "observed_latents": (
                observed_latents
            ),
            "predicted_latents": torch.stack(
                predicted_latents,
                dim=1,
            ),
            "predicted_states": torch.stack(
                predicted_states,
                dim=1,
            ),
            "predicted_rewards": torch.stack(
                predicted_rewards,
                dim=1,
            ),
            "termination_logits": torch.stack(
                termination_logits,
                dim=1,
            ),
        }

    @torch.no_grad()
    def imagine(
        self,
        initial_states: Tensor,
        action_sequences: Tensor,
    ) -> dict[str, Tensor]:
        """
        Imagine future trajectories.

        initial_states:
            [B, state_dim]

        action_sequences:
            [B, T, action_dim]
        """
        if initial_states.ndim != 2:
            raise ValueError(
                "initial_states must be [B,state_dim]"
            )

        if action_sequences.ndim != 3:
            raise ValueError(
                "action_sequences must be [B,T,action_dim]"
            )

        if (
            initial_states.shape[0]
            != action_sequences.shape[0]
        ):
            raise ValueError(
                "Batch size mismatch"
            )

        was_training = self.training
        self.eval()

        try:
            latents = self.encode(
                initial_states
            )

            states_history = []
            latent_history = []
            rewards_history = []
            termination_history = []

            for timestep in range(
                action_sequences.shape[1]
            ):
                step = self.predict_step(
                    latents,
                    action_sequences[
                        :,
                        timestep,
                    ],
                )

                latents = step[
                    "next_latents"
                ]

                states_history.append(
                    step["next_states"]
                )

                latent_history.append(
                    latents
                )

                rewards_history.append(
                    step["rewards"]
                )

                termination_history.append(
                    torch.sigmoid(
                        step["termination_logits"]
                    )
                )

            return {
                "states": torch.stack(
                    states_history,
                    dim=1,
                ),
                "latents": torch.stack(
                    latent_history,
                    dim=1,
                ),
                "rewards": torch.stack(
                    rewards_history,
                    dim=1,
                ),
                "termination_probabilities": (
                    torch.stack(
                        termination_history,
                        dim=1,
                    )
                ),
            }

        finally:
            self.train(
                was_training
            )


# ============================================================
# World-model losses
# ============================================================

def compute_world_model_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    model: LatentWorldModel,
    config: WorldModelTrainingConfig,
) -> tuple[
    Tensor,
    dict[str, Tensor],
]:
    states = batch["states"].float()
    next_states = batch["next_states"].float()

    rewards = batch["rewards"].float()
    terminated = batch["terminated"].float()

    reconstructed_states = (
        outputs["reconstructed_states"].float()
    )

    predicted_states = (
        outputs["predicted_states"].float()
    )

    predicted_rewards = (
        outputs["predicted_rewards"].float()
    )

    termination_logits = (
        outputs["termination_logits"].float()
    )

    predicted_latents = (
        outputs["predicted_latents"].float()
    )

    reconstruction = F.mse_loss(
        reconstructed_states,
        states,
    )

    # Multi-step rollout weighting.
    sequence_length = states.shape[1]

    discounts = torch.pow(
        torch.tensor(
            config.rollout_discount,
            device=states.device,
            dtype=torch.float32,
        ),
        torch.arange(
            sequence_length,
            device=states.device,
            dtype=torch.float32,
        ),
    )

    discounts = (
        discounts
        / discounts.sum().clamp_min(1e-8)
    )

    prediction_error = (
        predicted_states
        - next_states
    ).square().mean(
        dim=-1
    )

    prediction = (
        prediction_error
        * discounts.unsqueeze(0)
    ).sum(
        dim=1
    ).mean()

    reward_error = (
        predicted_rewards
        - rewards
    ).square()

    reward_loss = (
        reward_error
        * discounts.unsqueeze(0)
    ).sum(
        dim=1
    ).mean()

    termination_error = (
        F.binary_cross_entropy_with_logits(
            termination_logits,
            terminated,
            reduction="none",
        )
    )

    termination_loss = (
        termination_error
        * discounts.unsqueeze(0)
    ).sum(
        dim=1
    ).mean()

    # Stop-gradient target prevents both sides
    # of the consistency objective from moving
    # toward each other trivially.
    with torch.no_grad():
        target_latents = model.encode(
            next_states
        ).float()

    latent_error = (
        predicted_latents
        - target_latents
    ).square().mean(
        dim=-1
    )

    latent_loss = (
        latent_error
        * discounts.unsqueeze(0)
    ).sum(
        dim=1
    ).mean()

    total = (
        config.reconstruction_weight
        * reconstruction
        + config.prediction_weight
        * prediction
        + config.latent_weight
        * latent_loss
        + config.reward_weight
        * reward_loss
        + config.termination_weight
        * termination_loss
    )

    mae = F.l1_loss(
        predicted_states,
        next_states,
    )

    mse = F.mse_loss(
        predicted_states,
        next_states,
    )

    termination_accuracy = (
        (
            (
                torch.sigmoid(
                    termination_logits
                )
                >= 0.5
            ).float()
            == terminated
        ).float().mean()
    )

    return total, {
        "total": total,
        "reconstruction": reconstruction,
        "prediction": prediction,
        "latent": latent_loss,
        "reward": reward_loss,
        "termination": termination_loss,
        "mse": mse,
        "mae": mae,
        "termination_accuracy": (
            termination_accuracy
        ),
    }


# ============================================================
# Teacher forcing schedule
# ============================================================

def get_teacher_forcing_ratio(
    epoch: int,
    config: WorldModelTrainingConfig,
) -> float:
    progress = min(
        1.0,
        max(
            0.0,
            (epoch - 1)
            / max(
                1,
                config.teacher_forcing_decay_epochs,
            ),
        ),
    )

    return (
        config.teacher_forcing_start
        + progress
        * (
            config.teacher_forcing_end
            - config.teacher_forcing_start
        )
    )


# ============================================================
# Metric tracking
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
        return (
            self.total / self.count
            if self.count
            else 0.0
        )


class MetricCollection:
    def __init__(self) -> None:
        self.meters: dict[
            str, AverageMeter
        ] = {}

    def update(
        self,
        values: Mapping[str, Tensor],
        weight: int,
    ) -> None:
        for name, value in values.items():
            if name not in self.meters:
                self.meters[name] = AverageMeter()

            self.meters[name].update(
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
            f"{prefix}/{name}": meter.average
            for name, meter in self.meters.items()
        }


# ============================================================
# Learning-rate scheduler
# ============================================================

def create_scheduler(
    optimizer: AdamW,
    config: WorldModelTrainingConfig,
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
    config: WorldModelTrainingConfig,
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

class WorldModelTrainer:
    def __init__(
        self,
        config: WorldModelTrainingConfig,
    ) -> None:
        self.config = config

        seed_everything(
            config.seed
        )

        self.device = resolve_device(
            config.device
        )

        self.model = LatentWorldModel(
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
            name: value.to(
                self.device,
                non_blocking=True,
            )
            for name, value in batch.items()
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

        forcing_ratio = get_teacher_forcing_ratio(
            epoch,
            self.config,
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
                    next_states=(
                        batch["next_states"]
                    ),
                    teacher_forcing_ratio=(
                        forcing_ratio
                    ),
                )

                loss, components = (
                    compute_world_model_loss(
                        outputs,
                        batch,
                        self.model,
                        self.config,
                    )
                )

            if not bool(
                torch.isfinite(
                    loss.detach()
                ).item()
            ):
                raise FloatingPointError(
                    "Non-finite world-model loss"
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
                weight=batch["states"].shape[0],
            )

            if (
                (index + 1)
                % self.config.log_every == 0
            ):
                LOGGER.info(
                    "Epoch %d | Batch %d/%d | "
                    "Loss %.6f | TF %.3f",
                    epoch,
                    index + 1,
                    total_batches,
                    metrics.meters[
                        "total"
                    ].average,
                    forcing_ratio,
                )

        results = metrics.compute(
            "train"
        )

        results["train/teacher_forcing"] = (
            forcing_ratio
        )

        return results

    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
    ) -> dict[str, float]:
        self.model.eval()

        metrics = MetricCollection()

        for raw_batch in loader:
            batch = self.move_batch(
                raw_batch
            )

            with self.autocast():
                outputs = self.model(
                    batch["states"],
                    batch["actions"],
                    next_states=(
                        batch["next_states"]
                    ),
                    teacher_forcing_ratio=0.0,
                )

                _, components = (
                    compute_world_model_loss(
                        outputs,
                        batch,
                        self.model,
                        self.config,
                    )
                )

            metrics.update(
                components,
                weight=batch["states"].shape[0],
            )

        results = metrics.compute(
            "val"
        )

        results["val/rmse"] = math.sqrt(
            results["val/mse"]
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
            "Resumed world-model training at epoch %d",
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
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        LOGGER.info(
            "World-model training on %s | "
            "Parameters: %d",
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
                "RMSE %s | "
                "LR %.8f",
                epoch,
                self.config.epochs,
                train_metrics["train/total"],
                (
                    f"{validation_metrics['val/total']:.6f}"
                    if validation_metrics
                    else "not evaluated"
                ),
                (
                    f"{validation_metrics['val/rmse']:.6f}"
                    if validation_metrics
                    else "N/A"
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
            "World-model training complete. "
            "Best validation loss: %.6f",
            self.best_validation_loss,
        )


# ============================================================
# Configuration loading
# ============================================================

def load_config(
    path: str | Path | None,
) -> WorldModelTrainingConfig:
    if path is None:
        return WorldModelTrainingConfig()

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
            "Configuration must be a mapping"
        )

    if "training" in data:
        data = data["training"]

    valid_fields = set(
        WorldModelTrainingConfig.__dataclass_fields__
    )

    return WorldModelTrainingConfig(
        **{
            key: value
            for key, value in data.items()
            if key in valid_fields
        }
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroWorld AI latent world model"
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
    }

    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    if args.demo_data:
        values["demo_data"] = True

    config = WorldModelTrainingConfig(
        **values
    )

    if config.demo_data:
        create_demo_dataset(
            config.data_dir,
            config,
        )

    train_loader, validation_loader = (
        create_dataloaders(
            config
        )
    )

    trainer = WorldModelTrainer(
        config
    )

    trainer.fit(
        train_loader,
        validation_loader,
    )


if __name__ == "__main__":
    main()
