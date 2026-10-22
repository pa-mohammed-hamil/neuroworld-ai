
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/train_tokenizer.py

Train a VQ-VAE visual tokenizer on preprocessed videos.

Input:
    data/processed/videos/train/*.npz
    data/processed/videos/val/*.npz

Expected NPZ field:
    frames: [T, H, W, 3], RGB
            uint8 [0, 255] or float [0, 1]

Features:
    - Convolutional VQ-VAE
    - Learnable vector quantization codebook
    - Straight-through estimator
    - Reconstruction and commitment losses
    - Codebook usage/perplexity tracking
    - Video-frame dataset loading
    - Training and validation loops
    - Automatic CPU/CUDA selection
    - CUDA mixed precision
    - Gradient clipping
    - Checkpoint saving/resuming
    - Early stopping
    - JSON training history
    - Discrete token export

Dependencies:
    pip install torch numpy pillow

Usage:
    python scripts/train_tokenizer.py

    python scripts/train_tokenizer.py \
        --epochs 50 \
        --batch-size 32 \
        --embedding-dim 128 \
        --num-embeddings 1024

    python scripts/train_tokenizer.py \
        --resume artifacts/tokenizer/checkpoints/last.pt

    python scripts/train_tokenizer.py \
        --mode export \
        --checkpoint artifacts/tokenizer/checkpoints/best.pt \
        --export-input data/processed/videos/test \
        --export-output data/tokens/test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("neuroworld.tokenizer")


@dataclass
class TokenizerConfig:
    train_dir: str = "data/processed/videos/train"
    val_dir: str = "data/processed/videos/val"

    output_dir: str = "artifacts/tokenizer"

    image_size: int = 128
    in_channels: int = 3

    hidden_dim: int = 128
    embedding_dim: int = 128
    num_embeddings: int = 1024

    commitment_beta: float = 0.25

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5

    num_workers: int = 0
    max_grad_norm: float = 1.0

    validation_fraction: float = 0.1
    early_stopping_patience: int = 10

    seed: int = 42
    device: str = "auto"

    mixed_precision: bool = True

    max_train_frames: Optional[int] = None
    max_val_frames: Optional[int] = None

    save_every: int = 5


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(value)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return device


class VideoFrameDataset(Dataset):
    """
    Dataset of frames stored in sequence NPZ files.

    Each item is one RGB frame [3, H, W].

    NPZ files are opened per sample, avoiding
    loading the entire dataset into memory.

    Note:
        For large datasets, convert NPZ sequences
        to an indexed storage format such as
        WebDataset, Zarr or LMDB.
    """

    def __init__(
        self,
        directory: Path,
        image_size: int,
        max_frames: Optional[int] = None,
    ) -> None:
        self.directory = directory
        self.image_size = image_size

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory missing: {directory}"
            )

        files = sorted(
            directory.rglob("*.npz")
        )

        if not files:
            raise RuntimeError(
                f"No NPZ files in {directory}"
            )

        self.index: list[
            tuple[Path, int]
        ] = []

        for path in files:
            try:
                with np.load(
                    path,
                    allow_pickle=False,
                ) as archive:
                    if "frames" not in archive:
                        LOGGER.warning(
                            "Skipping %s: no frames field",
                            path,
                        )
                        continue

                    shape = archive[
                        "frames"
                    ].shape

                    if (
                        len(shape) != 4
                        or shape[-1] != 3
                    ):
                        LOGGER.warning(
                            "Skipping invalid shape: %s",
                            path,
                        )
                        continue

                    frame_count = shape[0]

                for frame_index in range(
                    frame_count
                ):
                    self.index.append(
                        (path, frame_index)
                    )

                    if (
                        max_frames is not None
                        and len(self.index) >= max_frames
                    ):
                        break

                if (
                    max_frames is not None
                    and len(self.index) >= max_frames
                ):
                    break

            except (
                OSError,
                ValueError,
                KeyError,
            ) as exc:
                LOGGER.warning(
                    "Skipping %s: %s",
                    path,
                    exc,
                )

        if not self.index:
            raise RuntimeError(
                "No valid video frames found"
            )

        LOGGER.info(
            "Indexed %d frames from %s",
            len(self.index),
            directory,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:
        path, frame_index = self.index[index]

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            frame = archive[
                "frames"
            ][frame_index].copy()

        if frame.dtype == np.uint8:
            frame = (
                frame.astype(np.float32)
                / 255.0
            )
        else:
            frame = frame.astype(
                np.float32
            )

            if (
                np.nanmax(frame) > 1.0
                and np.nanmax(frame) <= 255.0
            ):
                frame /= 255.0

        if not np.isfinite(frame).all():
            raise ValueError(
                f"Non-finite frame in {path}"
            )

        frame = np.clip(
            frame,
            0.0,
            1.0,
        )

        tensor = torch.from_numpy(
            frame
        ).permute(
            2, 0, 1
        ).contiguous()

        if tensor.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(
                    self.image_size,
                    self.image_size,
                ),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.network(x)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                3,
                hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                embedding_dim,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                embedding_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim // 2,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(z)


class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE vector quantization.

    Uses:
        - Nearest-neighbor codebook lookup
        - Codebook embedding loss
        - Commitment loss
        - Straight-through gradients

    Perplexity measures codebook utilization.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = (
            commitment_beta
        )

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
        )

        bound = 1.0 / num_embeddings

        nn.init.uniform_(
            self.embedding.weight,
            -bound,
            bound,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, channels, height, width = (
            z.shape
        )

        flattened = z.permute(
            0, 2, 3, 1
        ).reshape(
            -1,
            channels,
        )

        # Compute distances in float32 for
        # stability under mixed precision.
        vectors = flattened.float()

        codebook = (
            self.embedding.weight.float()
        )

        distances = (
            vectors.square().sum(
                dim=1,
                keepdim=True,
            )
            + codebook.square().sum(
                dim=1
            ).unsqueeze(0)
            - 2.0
            * vectors
            @ codebook.t()
        )

        indices = torch.argmin(
            distances,
            dim=1,
        )

        quantized = self.embedding(
            indices
        ).view(
            batch,
            height,
            width,
            channels,
        ).permute(
            0, 3, 1, 2
        ).contiguous()

        quantized = quantized.to(
            dtype=z.dtype
        )

        codebook_loss = F.mse_loss(
            quantized.float(),
            z.detach().float(),
        )

        commitment_loss = F.mse_loss(
            z.float(),
            quantized.detach().float(),
        )

        vq_loss = (
            codebook_loss
            + self.commitment_beta
            * commitment_loss
        )

        straight_through = (
            z
            + (
                quantized - z
            ).detach()
        )

        counts = torch.bincount(
            indices,
            minlength=self.num_embeddings,
        ).float()

        probabilities = (
            counts
            / counts.sum().clamp_min(1)
        )

        perplexity = torch.exp(
            -torch.sum(
                probabilities
                * torch.log(
                    probabilities + 1e-10
                )
            )
        )

        indices = indices.view(
            batch,
            height,
            width,
        )

        return (
            straight_through,
            vq_loss,
            perplexity,
            indices,
        )


class VQVAE(nn.Module):
    """
    Convolutional VQ-VAE visual tokenizer.

    Input:
        [B, 3, H, W]

    Encoded:
        [B, embedding_dim, H/4, W/4]

    Tokens:
        [B, H/4, W/4]

    Reconstruction:
        [B, 3, H, W]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_embeddings: int = 1024,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            hidden_dim,
            embedding_dim,
        )

        self.quantizer = VectorQuantizer(
            num_embeddings,
            embedding_dim,
            commitment_beta,
        )

        self.decoder = Decoder(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(x)

        quantized, vq_loss, perplexity, indices = (
            self.quantizer(z)
        )

        reconstruction = self.decoder(
            quantized
        )

        reconstruction_loss = F.mse_loss(
            reconstruction,
            x,
        )

        total_loss = (
            reconstruction_loss
            + vq_loss
        )

        return {
            "reconstruction": reconstruction,
            "loss": total_loss,
            "reconstruction_loss": (
                reconstruction_loss
            ),
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "indices": indices,
        }

    @torch.no_grad()
    def encode_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encoder(x)

        _, _, _, indices = (
            self.quantizer(z)
        )

        return indices

    @torch.no_grad()
    def decode_tokens(
        self,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError(
                "Expected token indices [B, H, W]"
            )

        embeddings = (
            self.quantizer.embedding(
                indices.long()
            )
        )

        quantized = embeddings.permute(
            0, 3, 1, 2
        ).contiguous()

        return self.decoder(
            quantized
        )


class TokenizerTrainer:
    def __init__(
        self,
        config: TokenizerConfig,
    ) -> None:
        self.config = config

        self._validate()

        set_seed(
            config.seed
        )

        self.device = choose_device(
            config.device
        )

        self.output_dir = resolve_path(
            config.output_dir
        )

        self.checkpoint_dir = (
            self.output_dir / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = VQVAE(
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_embeddings=config.num_embeddings,
            commitment_beta=(
                config.commitment_beta
            ),
        ).to(
            self.device
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.use_amp = (
            config.mixed_precision
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.history: list[
            dict[str, Any]
        ] = []

        self.start_epoch = 0
        self.best_val_loss = math.inf
        self.bad_epochs = 0

    def _validate(self) -> None:
        c = self.config

        if (
            c.image_size <= 0
            or c.image_size % 4 != 0
        ):
            raise ValueError(
                "image_size must be divisible by 4"
            )

        if c.hidden_dim < 2:
            raise ValueError(
                "hidden_dim must be at least 2"
            )

        if (
            c.embedding_dim <= 0
            or c.num_embeddings <= 1
        ):
            raise ValueError(
                "Invalid codebook configuration"
            )

        if (
            c.epochs <= 0
            or c.batch_size <= 0
            or c.learning_rate <= 0
        ):
            raise ValueError(
                "Invalid training parameters"
            )

        if not 0 < c.validation_fraction < 1:
            raise ValueError(
                "validation_fraction must be "
                "between 0 and 1"
            )

        if c.save_every <= 0:
            raise ValueError(
                "save_every must be positive"
            )

    def _build_loaders(
        self,
    ) -> tuple[
        DataLoader,
        DataLoader,
    ]:
        train_dir = resolve_path(
            self.config.train_dir
        )

        val_dir = resolve_path(
            self.config.val_dir
        )

        if not val_dir.is_dir() or not list(
            val_dir.rglob("*.npz")
        ):
            raise RuntimeError(
                "A separate validation dataset "
                "is required. Generate a val split "
                "with preprocess_videos.py. "
                "Do not randomly split overlapping "
                "frames from the same source video."
            )

        train_dataset = VideoFrameDataset(
            train_dir,
            self.config.image_size,
            self.config.max_train_frames,
        )

        val_dataset = VideoFrameDataset(
            val_dir,
            self.config.image_size,
            self.config.max_val_frames,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=(
                self.config.batch_size
            ),
            shuffle=True,
            num_workers=(
                self.config.num_workers
            ),
            pin_memory=(
                self.device.type == "cuda"
            ),
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=(
                self.config.batch_size
            ),
            shuffle=False,
            num_workers=(
                self.config.num_workers
            ),
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

        return (
            train_loader,
            val_loader,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> dict[str, float]:
        self.model.train(
            training
        )

        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "vq_loss": 0.0,
            "perplexity": 0.0,
        }

        sample_count = 0

        for frames in loader:
            frames = frames.to(
                self.device,
                non_blocking=True,
            )

            if training:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(
                training
            ):
                with torch.autocast(
                    device_type=(
                        self.device.type
                    ),
                    enabled=self.use_amp,
                ):
                    output = self.model(
                        frames
                    )

                    loss = output[
                        "loss"
                    ]

                if training:
                    self.scaler.scale(
                        loss
                    ).backward()

                    self.scaler.unscale_(
                        self.optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    self.scaler.step(
                        self.optimizer
                    )

                    self.scaler.update()

            batch_size = len(
                frames
            )

            sample_count += batch_size

            for key in totals:
                totals[key] += (
                    float(
                        output[key]
                        .detach()
                        .item()
                    )
                    * batch_size
                )

        if sample_count == 0:
            raise RuntimeError(
                "Empty data loader"
            )

        return {
            key: value / sample_count
            for key, value in totals.items()
        }

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
    ) -> Path:
        path = (
            self.checkpoint_dir / filename
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                self.model.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "scaler_state_dict": (
                self.scaler.state_dict()
            ),
            "best_val_loss": (
                self.best_val_loss
            ),
            "bad_epochs": (
                self.bad_epochs
            ),
            "config": asdict(
                self.config
            ),
            "history": (
                self.history
            ),
        }

        temporary = path.with_suffix(
            ".tmp"
        )

        torch.save(
            checkpoint,
            temporary,
        )

        temporary.replace(
            path
        )

        return path

    def resume(
        self,
        checkpoint_path: str,
    ) -> None:
        path = resolve_path(
            checkpoint_path
        )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        self.start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        self.best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                math.inf,
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

        LOGGER.info(
            "Resumed from epoch %d",
            self.start_epoch,
        )

    def train(self) -> dict[str, Any]:
        train_loader, val_loader = (
            self._build_loaders()
        )

        started = time.perf_counter()

        for epoch in range(
            self.start_epoch,
            self.config.epochs,
        ):
            train_metrics = (
                self._run_epoch(
                    train_loader,
                    training=True,
                )
            )

            with torch.no_grad():
                val_metrics = (
                    self._run_epoch(
                        val_loader,
                        training=False,
                    )
                )

            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": val_metrics,
            }

            self.history.append(
                record
            )

            val_loss = (
                val_metrics["loss"]
            )

            improved = (
                val_loss
                < self.best_val_loss
            )

            if improved:
                self.best_val_loss = (
                    val_loss
                )

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
                % self.config.save_every
                == 0
            ):
                self.save_checkpoint(
                    f"epoch_{epoch + 1:04d}.pt",
                    epoch,
                )

            LOGGER.info(
                "Epoch %d/%d | "
                "train=%.6f | "
                "val=%.6f | "
                "perplexity=%.2f",
                epoch + 1,
                self.config.epochs,
                train_metrics["loss"],
                val_loss,
                val_metrics["perplexity"],
            )

            self._save_history()

            if (
                self.bad_epochs
                >= self.config.early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping triggered"
                )
                break

        summary = {
            "project": "NeuroWorld AI",
            "model": "VQ-VAE Visual Tokenizer",
            "best_validation_loss": (
                self.best_val_loss
            ),
            "epochs_completed": len(
                self.history
            ),
            "training_seconds": (
                time.perf_counter()
                - started
            ),
            "device": str(
                self.device
            ),
            "codebook_size": (
                self.config.num_embeddings
            ),
            "embedding_dim": (
                self.config.embedding_dim
            ),
            "tokens_per_frame": (
                (
                    self.config.image_size
                    // 4
                ) ** 2
            ),
            "checkpoint": str(
                self.checkpoint_dir
                / "best.pt"
            ),
        }

        (
            self.output_dir
            / "training_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        return summary

    def _save_history(self) -> None:
        path = (
            self.output_dir
            / "training_history.json"
        )

        path.write_text(
            json.dumps(
                self.history,
                indent=2,
            ),
            encoding="utf-8",
        )


@torch.no_grad()
def export_tokens(
    model: VQVAE,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Convert video sequences to discrete tokens.

    Input:
        frames [T, H, W, 3]

    Output:
        tokens [T, H/4, W/4]

    Source sequence directory structure
    is preserved.
    """

    if not input_dir.is_dir():
        raise FileNotFoundError(
            input_dir
        )

    files = sorted(
        input_dir.rglob("*.npz")
    )

    if not files:
        raise RuntimeError(
            "No input sequences found"
        )

    model.eval()

    processed = 0
    token_count = 0

    for path in files:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            if "frames" not in archive:
                continue

            frames = archive[
                "frames"
            ].copy()

            timestamps = (
                archive["timestamps"].copy()
                if "timestamps" in archive
                else np.arange(
                    len(frames),
                    dtype=np.float64,
                )
            )

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Invalid video shape: {path}"
            )

        if frames.dtype == np.uint8:
            frames = (
                frames.astype(np.float32)
                / 255.0
            )
        else:
            frames = frames.astype(
                np.float32
            )

            if frames.size and frames.max() > 1.0:
                frames /= 255.0

        if not np.isfinite(
            frames
        ).all():
            raise ValueError(
                f"Non-finite frames: {path}"
            )

        frames = np.clip(
            frames,
            0.0,
            1.0,
        )

        tokens = []

        for start in range(
            0,
            len(frames),
            batch_size,
        ):
            batch = torch.from_numpy(
                frames[
                    start:start + batch_size
                ]
            ).permute(
                0, 3, 1, 2
            ).to(
                device
            )

            if batch.shape[-2:] != (
                image_size,
                image_size,
            ):
                batch = F.interpolate(
                    batch,
                    size=(
                        image_size,
                        image_size,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

            indices = model.encode_tokens(
                batch
            )

            tokens.append(
                indices.cpu().numpy()
            )

        token_array = np.concatenate(
            tokens,
            axis=0,
        )

        relative = path.relative_to(
            input_dir
        )

        output_path = (
            output_dir / relative
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            output_path,
            tokens=token_array.astype(
                np.int32
            ),
            timestamps=timestamps,
            source_file=np.asarray(
                str(relative)
            ),
        )

        processed += 1
        token_count += (
            token_array.size
        )

    return {
        "processed_sequences": processed,
        "total_tokens": token_count,
        "output_dir": str(
            output_dir
        ),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroWorld VQ-VAE tokenizer"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "export",
        ],
        default="train",
    )

    parser.add_argument(
        "--train-dir",
        default="data/processed/videos/train",
    )

    parser.add_argument(
        "--val-dir",
        default="data/processed/videos/val",
    )

    parser.add_argument(
        "--output",
        default="artifacts/tokenizer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--export-input",
        default="data/processed/videos/test",
    )

    parser.add_argument(
        "--export-output",
        default="data/tokens/test",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = TokenizerConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        output_dir=args.output,
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        num_workers=args.workers,
        mixed_precision=(
            not args.no_amp
        ),
    )

    try:
        trainer = TokenizerTrainer(
            config
        )

        if args.mode == "train":
            if args.resume:
                trainer.resume(
                    args.resume
                )

            summary = trainer.train()

        else:
            if not args.checkpoint:
                raise ValueError(
                    "--checkpoint is required "
                    "for token export"
                )

            checkpoint = torch.load(
                resolve_path(
                    args.checkpoint
                ),
                map_location="cpu",
                weights_only=True,
            )

            saved_config = checkpoint.get(
                "config",
                {},
            )

            # Reconstruct the architecture
            # from the saved training config.
            model = VQVAE(
                hidden_dim=saved_config.get(
                    "hidden_dim",
                    config.hidden_dim,
                ),
                embedding_dim=saved_config.get(
                    "embedding_dim",
                    config.embedding_dim,
                ),
                num_embeddings=saved_config.get(
                    "num_embeddings",
                    config.num_embeddings,
                ),
                commitment_beta=saved_config.get(
                    "commitment_beta",
                    config.commitment_beta,
                ),
            ).to(
                trainer.device
            )

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ],
                strict=True,
            )

            summary = export_tokens(
                model=model,
                input_dir=resolve_path(
                    args.export_input
                ),
                output_dir=resolve_path(
                    args.export_output
                ),
                device=trainer.device,
                image_size=saved_config.get(
                    "image_size",
                    config.image_size,
                ),
                batch_size=args.batch_size,
            )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        return 0

    except Exception:
        LOGGER.exception(
            "Tokenizer operation failed"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
