
"""
NeuroWorld AI
File: training/train_tokenizer.py

Visual tokenizer training for image/video frames.

Architectures:
    - VQ-VAE tokenizer
    - Continuous convolutional autoencoder

Features:
    - Image and video-frame datasets
    - Convolutional encoder and decoder
    - Vector quantization with straight-through estimator
    - Codebook and commitment losses
    - Codebook perplexity and utilization metrics
    - MSE / L1 / Huber reconstruction
    - Optional image-gradient reconstruction
    - Mixed precision
    - Gradient accumulation
    - Gradient clipping
    - Cosine / plateau learning-rate scheduling
    - Validation
    - Early stopping
    - Atomic checkpoints
    - Resume training
    - JSONL experiment logging
    - Synthetic demo data

Supported data:
    data/tokenizer/
        image_001.png
        image_002.jpg
        video_frames.npy
        episode_001.npz

NPY:
    [N, C, H, W] or [N, H, W, C]

NPZ:
    frames: [N, C, H, W]
    or
    frames: [N, H, W, C]

Run:
    python -m training.train_tokenizer --demo-data

    python -m training.train_tokenizer \
        --data-dir data/tokenizer \
        --epochs 50

Dependencies:
    pip install torch numpy pillow pyyaml
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

from PIL import Image
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import yaml
except ImportError:
    yaml = None


LOGGER = logging.getLogger(
    "neuroworld.train_tokenizer"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class TokenizerTrainingConfig:
    data_dir: str = "data/tokenizer"
    demo_data: bool = False

    tokenizer_type: str = "vqvae"

    image_channels: int = 3
    image_size: int = 64

    hidden_dim: int = 128
    latent_dim: int = 64

    codebook_size: int = 512
    commitment_beta: float = 0.25

    reconstruction_loss: str = "mse"

    reconstruction_weight: float = 1.0
    quantization_weight: float = 1.0
    gradient_weight: float = 0.05

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

    checkpoint_dir: str = "checkpoints/tokenizer"
    log_dir: str = "runs/tokenizer"

    save_every: int = 5
    resume: str | None = None

    device: str = "auto"
    seed: int = 42
    log_every: int = 20

    def __post_init__(self) -> None:
        if self.tokenizer_type not in {
            "vqvae",
            "autoencoder",
        }:
            raise ValueError(
                "tokenizer_type must be vqvae or autoencoder"
            )

        if self.reconstruction_loss not in {
            "mse",
            "l1",
            "huber",
        }:
            raise ValueError(
                "Unsupported reconstruction loss"
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
            "image_channels",
            "image_size",
            "hidden_dim",
            "latent_dim",
            "codebook_size",
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

        if self.image_size % 4 != 0:
            raise ValueError(
                "image_size must be divisible by 4"
            )

        if not 0 < self.validation_split < 1:
            raise ValueError(
                "validation_split must be between 0 and 1"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.commitment_beta < 0:
            raise ValueError(
                "commitment_beta must be nonnegative"
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
    device: str,
) -> torch.device:
    if device != "auto":
        return torch.device(device)

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

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


class TokenizerDataset(Dataset):
    """
    Loads image frames from image, NPY, or NPZ files.

    Each file is considered an independent source.
    Train/validation splitting is performed by
    source file to avoid splitting adjacent video
    frames across the two datasets.
    """

    def __init__(
        self,
        data_dir: str | Path,
        config: TokenizerTrainingConfig,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory missing: {self.data_dir}"
            )

        self.sources: list[
            dict[str, Any]
        ] = []

        self.index: list[
            tuple[int, int]
        ] = []

        files = sorted(
            path
            for path in self.data_dir.rglob("*")
            if path.is_file()
            and (
                path.suffix.lower() in IMAGE_EXTENSIONS
                or path.suffix.lower()
                in {".npy", ".npz"}
            )
        )

        for path in files:
            suffix = path.suffix.lower()

            if suffix in IMAGE_EXTENSIONS:
                source_index = len(
                    self.sources
                )

                self.sources.append(
                    {
                        "type": "image",
                        "path": path,
                        "frames": None,
                    }
                )

                self.index.append(
                    (source_index, 0)
                )

                continue

            if suffix == ".npy":
                frames = np.load(
                    path,
                    allow_pickle=False,
                )

            else:
                with np.load(
                    path,
                    allow_pickle=False,
                ) as data:
                    if "frames" not in data.files:
                        continue

                    frames = np.asarray(
                        data["frames"]
                    )

            if frames.ndim != 4:
                raise ValueError(
                    f"{path}: expected 4D frames"
                )

            source_index = len(
                self.sources
            )

            self.sources.append(
                {
                    "type": "array",
                    "path": path,
                    "frames": frames,
                }
            )

            for frame_index in range(
                len(frames)
            ):
                self.index.append(
                    (source_index, frame_index)
                )

        if not self.index:
            raise ValueError(
                "No image frames found"
            )

    def __len__(self) -> int:
        return len(self.index)

    def _prepare_frame(
        self,
        frame: np.ndarray,
    ) -> Tensor:
        channels = (
            self.config.image_channels
        )

        if frame.ndim == 2:
            frame = frame[
                ...,
                None,
            ]

        if frame.ndim != 3:
            raise ValueError(
                "Frame must have 2 or 3 dimensions"
            )

        # Convert CHW to HWC when appropriate.
        if (
            frame.shape[0] == channels
            and frame.shape[-1] != channels
        ):
            frame = np.transpose(
                frame,
                (1, 2, 0),
            )

        if frame.shape[-1] != channels:
            raise ValueError(
                "Frame channel count mismatch"
            )

        if frame.dtype == np.uint8:
            frame = (
                frame.astype(np.float32)
                / 255.0
            )
        else:
            frame = frame.astype(
                np.float32
            )

        if not np.isfinite(frame).all():
            raise ValueError(
                "Frame contains non-finite values"
            )

        if (
            frame.min() < 0
            or frame.max() > 1
        ):
            raise ValueError(
                "Frame values must be in [0,1]"
            )

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                np.transpose(
                    frame,
                    (2, 0, 1),
                )
            )
        )

        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(
                self.config.image_size,
                self.config.image_size,
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return tensor

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        source_index, frame_index = (
            self.index[index]
        )

        source = self.sources[
            source_index
        ]

        if source["type"] == "image":
            mode = (
                "RGB"
                if self.config.image_channels == 3
                else "L"
            )

            if self.config.image_channels not in {
                1,
                3,
            }:
                raise ValueError(
                    "Image files support 1 or 3 channels"
                )

            with Image.open(
                source["path"]
            ) as image:
                frame = np.asarray(
                    image.convert(mode)
                )

        else:
            frame = source["frames"][
                frame_index
            ]

        return self._prepare_frame(
            frame
        )


def create_demo_dataset(
    directory: str | Path,
    config: TokenizerTrainingConfig,
    *,
    num_sources: int = 12,
    frames_per_source: int = 24,
) -> None:
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

    for source_index in range(
        num_sources
    ):
        frames = np.zeros(
            (
                frames_per_source,
                channels,
                size,
                size,
            ),
            dtype=np.float32,
        )

        x = float(
            rng.integers(
                5,
                size - 5,
            )
        )

        y = float(
            rng.integers(
                5,
                size - 5,
            )
        )

        for t in range(
            frames_per_source
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

            x += rng.normal(
                0,
                1.5,
            )

            y += rng.normal(
                0,
                1.5,
            )

            x = np.clip(
                x,
                2,
                size - 3,
            )

            y = np.clip(
                y,
                2,
                size - 3,
            )

        np.savez_compressed(
            directory
            / f"demo_{source_index:04d}.npz",
            frames=frames,
        )


def create_dataloaders(
    config: TokenizerTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    dataset = TokenizerDataset(
        config.data_dir,
        config,
    )

    source_count = len(
        dataset.sources
    )

    if source_count < 2:
        raise ValueError(
            "At least two independent sources "
            "are required for validation"
        )

    rng = np.random.default_rng(
        config.seed
    )

    source_indices = np.arange(
        source_count
    )

    rng.shuffle(
        source_indices
    )

    validation_count = max(
        1,
        min(
            source_count - 1,
            round(
                source_count
                * config.validation_split
            ),
        ),
    )

    validation_sources = set(
        source_indices[
            :validation_count
        ].tolist()
    )

    train_indices = []
    validation_indices = []

    for index, (
        source_index,
        _,
    ) in enumerate(dataset.index):
        if source_index in validation_sources:
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
        "Tokenizer frames: train=%d validation=%d",
        len(train_indices),
        len(validation_indices),
    )

    return train_loader, validation_loader


# ============================================================
# Residual blocks
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.GroupNorm(
                8,
                channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                8,
                channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return x + self.network(x)


# ============================================================
# Encoder
# ============================================================

class VisualEncoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            ResidualBlock(
                hidden_dim
            ),
            ResidualBlock(
                hidden_dim
            ),
            nn.Conv2d(
                hidden_dim,
                latent_dim,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.network(
            x
        )


# ============================================================
# Decoder
# ============================================================

class VisualDecoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                latent_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            ResidualBlock(
                hidden_dim
            ),
            ResidualBlock(
                hidden_dim
            ),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                hidden_dim // 2,
                channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latents: Tensor,
    ) -> Tensor:
        return self.network(
            latents
        )


# ============================================================
# Vector quantization
# ============================================================

class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE codebook.

    Uses straight-through quantization.

    Loss:
        codebook_loss
        + beta * commitment_loss
    """

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        commitment_beta: float = 0.25,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
        )

        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / num_embeddings,
            1.0 / num_embeddings,
        )

    def forward(
        self,
        latents: Tensor,
    ) -> dict[str, Tensor]:
        batch, channels, height, width = (
            latents.shape
        )

        if channels != self.embedding_dim:
            raise ValueError(
                "Latent channel dimension mismatch"
            )

        # Keep distance computation in FP32
        # for numerical stability under AMP.
        latent_fp32 = latents.float()

        flattened = latent_fp32.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            -1,
            channels,
        )

        embeddings = (
            self.embedding.weight.float()
        )

        distances = (
            flattened.square().sum(
                dim=1,
                keepdim=True,
            )
            - 2
            * flattened
            @ embeddings.t()
            + embeddings.square().sum(
                dim=1
            ).unsqueeze(0)
        )

        indices = distances.argmin(
            dim=-1
        )

        quantized = F.embedding(
            indices,
            embeddings,
        )

        quantized = quantized.reshape(
            batch,
            height,
            width,
            channels,
        ).permute(
            0,
            3,
            1,
            2,
        )

        codebook_loss = F.mse_loss(
            quantized,
            latent_fp32.detach(),
        )

        commitment_loss = F.mse_loss(
            latent_fp32,
            quantized.detach(),
        )

        quantization_loss = (
            codebook_loss
            + self.commitment_beta
            * commitment_loss
        )

        # Straight-through estimator:
        # encoder receives reconstruction gradients;
        # codebook receives codebook-loss gradients.
        quantized_st = (
            latent_fp32
            + (
                quantized - latent_fp32
            ).detach()
        )

        quantized_st = quantized_st.to(
            latents.dtype
        )

        counts = torch.bincount(
            indices,
            minlength=self.num_embeddings,
        ).float()

        probabilities = (
            counts
            / counts.sum().clamp_min(1.0)
        )

        entropy = -(
            probabilities
            * probabilities.clamp_min(
                1e-10
            ).log()
        ).sum()

        perplexity = torch.exp(
            entropy
        )

        utilization = (
            (counts > 0).float().mean()
        )

        return {
            "quantized": quantized_st,
            "indices": indices.reshape(
                batch,
                height,
                width,
            ),
            "quantization_loss": (
                quantization_loss
            ),
            "codebook_loss": (
                codebook_loss
            ),
            "commitment_loss": (
                commitment_loss
            ),
            "perplexity": perplexity,
            "utilization": utilization,
        }


# ============================================================
# Visual tokenizer model
# ============================================================

class VisualTokenizer(nn.Module):
    """
    VQ-VAE or continuous image autoencoder.
    """

    def __init__(
        self,
        config: TokenizerTrainingConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.encoder = VisualEncoder(
            channels=config.image_channels,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )

        self.decoder = VisualDecoder(
            channels=config.image_channels,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )

        self.quantizer = (
            VectorQuantizer(
                num_embeddings=(
                    config.codebook_size
                ),
                embedding_dim=(
                    config.latent_dim
                ),
                commitment_beta=(
                    config.commitment_beta
                ),
            )
            if config.tokenizer_type == "vqvae"
            else None
        )

    def encode(
        self,
        images: Tensor,
        *,
        quantize: bool = True,
    ) -> Tensor:
        latents = self.encoder(
            images
        )

        if (
            quantize
            and self.quantizer is not None
        ):
            result = self.quantizer(
                latents
            )

            return result["quantized"]

        return latents

    def decode(
        self,
        latents: Tensor,
    ) -> Tensor:
        return self.decoder(
            latents
        )

    def forward(
        self,
        images: Tensor,
    ) -> dict[str, Tensor]:
        latents = self.encoder(
            images
        )

        if self.quantizer is not None:
            quantization = self.quantizer(
                latents
            )

            decoded_latents = (
                quantization["quantized"]
            )

        else:
            zero = latents.sum() * 0.0

            quantization = {
                "quantization_loss": zero,
                "codebook_loss": zero,
                "commitment_loss": zero,
                "perplexity": zero.detach(),
                "utilization": zero.detach(),
            }

            decoded_latents = latents

        reconstruction = self.decoder(
            decoded_latents
        )

        return {
            "reconstruction": reconstruction,
            "latents": latents,
            "quantized_latents": (
                decoded_latents
            ),
            **quantization,
        }

    @torch.no_grad()
    def tokenize(
        self,
        images: Tensor,
    ) -> Tensor:
        if self.quantizer is None:
            raise RuntimeError(
                "Discrete tokens require VQ-VAE"
            )

        latents = self.encoder(
            images
        )

        result = self.quantizer(
            latents
        )

        return result["indices"]

    @torch.no_grad()
    def decode_tokens(
        self,
        indices: Tensor,
    ) -> Tensor:
        if self.quantizer is None:
            raise RuntimeError(
                "Discrete decoding requires VQ-VAE"
            )

        embeddings = F.embedding(
            indices.long(),
            self.quantizer.embedding.weight,
        )

        latents = embeddings.permute(
            0,
            3,
            1,
            2,
        )

        return self.decoder(
            latents
        )


# ============================================================
# Reconstruction losses
# ============================================================

def image_gradient_loss(
    predictions: Tensor,
    targets: Tensor,
) -> Tensor:
    prediction_dx = (
        predictions[..., :, 1:]
        - predictions[..., :, :-1]
    )

    target_dx = (
        targets[..., :, 1:]
        - targets[..., :, :-1]
    )

    prediction_dy = (
        predictions[..., 1:, :]
        - predictions[..., :-1, :]
    )

    target_dy = (
        targets[..., 1:, :]
        - targets[..., :-1, :]
    )

    return (
        F.l1_loss(
            prediction_dx,
            target_dx,
        )
        + F.l1_loss(
            prediction_dy,
            target_dy,
        )
    ) * 0.5


def compute_tokenizer_loss(
    outputs: Mapping[str, Tensor],
    images: Tensor,
    config: TokenizerTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    predictions = (
        outputs["reconstruction"].float()
    )

    targets = images.float()

    if config.reconstruction_loss == "mse":
        reconstruction = F.mse_loss(
            predictions,
            targets,
        )

    elif config.reconstruction_loss == "l1":
        reconstruction = F.l1_loss(
            predictions,
            targets,
        )

    else:
        reconstruction = F.huber_loss(
            predictions,
            targets,
        )

    gradient_loss = image_gradient_loss(
        predictions,
        targets,
    )

    quantization_loss = (
        outputs["quantization_loss"]
    )

    total = (
        config.reconstruction_weight
        * reconstruction
        + config.quantization_weight
        * quantization_loss
        + config.gradient_weight
        * gradient_loss
    )

    mse = F.mse_loss(
        predictions,
        targets,
    )

    mae = F.l1_loss(
        predictions,
        targets,
    )

    return total, {
        "total": total,
        "reconstruction": reconstruction,
        "quantization": quantization_loss,
        "gradient": gradient_loss,
        "codebook": outputs["codebook_loss"],
        "commitment": outputs["commitment_loss"],
        "mse": mse,
        "mae": mae,
        "perplexity": outputs["perplexity"],
        "utilization": outputs["utilization"],
    }


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
    config: TokenizerTrainingConfig,
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
            + (1.0 - min_factor)
            * 0.5
            * (
                1.0
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
# Checkpoints
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
    config: TokenizerTrainingConfig,
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

class TokenizerTrainer:
    def __init__(
        self,
        config: TokenizerTrainingConfig,
    ) -> None:
        self.config = config

        seed_everything(
            config.seed
        )

        self.device = resolve_device(
            config.device
        )

        self.model = VisualTokenizer(
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

        for index, images in enumerate(
            loader
        ):
            images = images.to(
                self.device,
                non_blocking=True,
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
                    images
                )

                loss, components = (
                    compute_tokenizer_loss(
                        outputs,
                        images,
                        self.config,
                    )
                )

            if not bool(
                torch.isfinite(
                    loss.detach()
                ).item()
            ):
                raise FloatingPointError(
                    "Non-finite tokenizer loss"
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
                weight=images.shape[0],
            )

            if (
                (index + 1)
                % self.config.log_every == 0
            ):
                LOGGER.info(
                    "Epoch %d | Batch %d/%d | "
                    "Loss %.6f | "
                    "Perplexity %.2f",
                    epoch,
                    index + 1,
                    total_batches,
                    metrics.meters[
                        "total"
                    ].average,
                    metrics.meters[
                        "perplexity"
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

        for images in loader:
            images = images.to(
                self.device,
                non_blocking=True,
            )

            with self.autocast():
                outputs = self.model(
                    images
                )

                _, components = (
                    compute_tokenizer_loss(
                        outputs,
                        images,
                        self.config,
                    )
                )

            metrics.update(
                components,
                weight=images.shape[0],
            )

        results = metrics.compute(
            "val"
        )

        mse = results["val/mse"]

        results["val/rmse"] = math.sqrt(
            mse
        )

        results["val/psnr"] = (
            -10.0 * math.log10(
                max(mse, 1e-12)
            )
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
            "Resumed tokenizer training at epoch %d",
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
            "Tokenizer training on %s | "
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
                "Val %s | "
                "PSNR %s | "
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
                    f"{validation_metrics['val/psnr']:.2f}"
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
            "Tokenizer training complete. "
            "Best validation loss: %.6f",
            self.best_validation_loss,
        )


# ============================================================
# Configuration loading
# ============================================================

def load_config(
    path: str | Path | None,
) -> TokenizerTrainingConfig:
    if path is None:
        return TokenizerTrainingConfig()

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

    if "training" in data:
        data = data["training"]

    valid_fields = set(
        TokenizerTrainingConfig.__dataclass_fields__
    )

    values = {
        key: value
        for key, value in data.items()
        if key in valid_fields
    }

    return TokenizerTrainingConfig(
        **values
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train NeuroWorld AI visual tokenizer"
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
        "--tokenizer-type",
        choices=[
            "vqvae",
            "autoencoder",
        ],
        default=None,
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
        "tokenizer_type": args.tokenizer_type,
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

    config = TokenizerTrainingConfig(
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

    trainer = TokenizerTrainer(
        config
    )

    trainer.fit(
        train_loader,
        validation_loader,
    )


if __name__ == "__main__":
    main()
