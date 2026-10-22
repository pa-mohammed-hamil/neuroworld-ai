
"""
NeuroWorld AI
File: neuroworld/perception/video_encoder.py

Spatiotemporal Video Representation Encoder

Features:
    - RGB video tensor encoding
    - CNN spatial feature extraction
    - Transformer temporal modeling
    - Sinusoidal temporal positional encoding
    - Frame-level embeddings
    - Video-level embeddings
    - Latent world-state representations
    - Temporal attention pooling
    - Variable-length video masking
    - Batch-first PyTorch processing
    - Frame sampling utilities
    - Video normalization
    - Optional pretrained image backbone
    - Streaming temporal feature buffer
    - World-model integration adapter

Input:
    [B,T,C,H,W]

Output:
    frame_embeddings: [B,T,D]
    video_embedding:  [B,D]
    latent_states:    [B,T,L]

Install:
    pip install torch torchvision numpy

Run:
    python -m neuroworld.perception.video_encoder
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class VideoEncoderError(Exception):
    pass


class VideoValidationError(VideoEncoderError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class VideoEncoderConfig:
    image_channels: int = 3

    embedding_dim: int = 256
    latent_dim: int = 128

    cnn_channels: tuple[int, ...] = (
        32,
        64,
        128,
        256,
    )

    transformer_layers: int = 4
    attention_heads: int = 8

    feedforward_dim: int = 1024

    dropout: float = 0.1

    max_frames: int = 128

    normalize_inputs: bool = True

    use_pretrained_backbone: bool = False

    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "image_channels",
            "embedding_dim",
            "latent_dim",
            "transformer_layers",
            "attention_heads",
            "feedforward_dim",
            "max_frames",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not self.cnn_channels:
            raise ValueError(
                "cnn_channels cannot be empty"
            )

        if any(
            channel <= 0
            for channel in self.cnn_channels
        ):
            raise ValueError(
                "CNN channels must be positive"
            )

        if (
            self.embedding_dim
            % self.attention_heads != 0
        ):
            raise ValueError(
                "embedding_dim must be divisible "
                "by attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )


# ============================================================
# VIDEO VALIDATION
# ============================================================

def validate_video_tensor(
    video: Tensor,
    config: VideoEncoderConfig,
) -> None:
    if video.ndim != 5:
        raise VideoValidationError(
            "Video must have shape [B,T,C,H,W]"
        )

    batch, frames, channels, height, width = (
        video.shape
    )

    if batch <= 0 or frames <= 0:
        raise VideoValidationError(
            "Batch and frame dimensions must be positive"
        )

    if channels != config.image_channels:
        raise VideoValidationError(
            "Unexpected image channel count"
        )

    if frames > config.max_frames:
        raise VideoValidationError(
            f"Maximum frames: {config.max_frames}"
        )

    if height < 16 or width < 16:
        raise VideoValidationError(
            "Frame dimensions must be >= 16"
        )

    if not torch.isfinite(video).all():
        raise VideoValidationError(
            "Video contains non-finite values"
        )


# ============================================================
# FRAME NORMALIZER
# ============================================================

class VideoNormalizer(nn.Module):
    """
    Normalizes RGB frames.

    Input values are expected in [0,1].

    Uses ImageNet normalization constants.
    """

    def __init__(
        self,
        channels: int = 3,
    ) -> None:
        super().__init__()

        if channels == 3:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.5] * channels
            std = [0.5] * channels

        self.register_buffer(
            "mean",
            torch.tensor(
                mean,
                dtype=torch.float32,
            ).view(1, 1, channels, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                std,
                dtype=torch.float32,
            ).view(1, 1, channels, 1, 1),
        )

    def forward(
        self,
        video: Tensor,
    ) -> Tensor:
        mean = self.mean.to(
            dtype=video.dtype,
            device=video.device,
        )

        std = self.std.to(
            dtype=video.dtype,
            device=video.device,
        )

        return (
            video - mean
        ) / std


# ============================================================
# CNN SPATIAL ENCODER
# ============================================================

class SpatialFrameEncoder(nn.Module):
    """
    Encodes individual video frames.

    Input:
        [B*T,C,H,W]

    Output:
        [B*T,embedding_dim]
    """

    def __init__(
        self,
        config: VideoEncoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        layers = []
        input_channels = (
            config.image_channels
        )

        for output_channels in (
            config.cnn_channels
        ):
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        output_channels
                    ),
                    nn.GELU(),
                ]
            )

            input_channels = (
                output_channels
            )

        self.backbone = nn.Sequential(
            *layers
        )

        self.pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_channels,
                config.embedding_dim,
            ),
            nn.LayerNorm(
                config.embedding_dim
            ),
        )

    def forward(
        self,
        frames: Tensor,
    ) -> Tensor:
        features = self.backbone(
            frames
        )

        features = self.pool(
            features
        )

        return self.projection(
            features
        )


# ============================================================
# PRETRAINED RESNET BACKBONE
# ============================================================

class PretrainedResNetEncoder(nn.Module):
    """
    Optional ResNet-18 frame encoder.

    Downloads pretrained weights on first
    use if not cached locally.

    If offline, disable pretrained backbone
    or pre-cache torchvision weights.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        try:
            from torchvision.models import (
                resnet18,
                ResNet18_Weights,
            )
        except ImportError as exc:
            raise VideoEncoderError(
                "torchvision is required "
                "for pretrained ResNet"
            ) from exc

        backbone = resnet18(
            weights=(
                ResNet18_Weights.DEFAULT
            )
        )

        feature_dim = (
            backbone.fc.in_features
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(
                feature_dim,
                embedding_dim,
            ),
            nn.LayerNorm(
                embedding_dim
            ),
        )

    def forward(
        self,
        frames: Tensor,
    ) -> Tensor:
        features = self.backbone(
            frames
        )

        return self.projection(
            features
        )


# ============================================================
# TEMPORAL POSITIONAL ENCODING
# ============================================================

class TemporalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal position encoding.

    Input:
        [B,T,D]
    """

    def __init__(
        self,
        embedding_dim: int,
        max_frames: int,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            max_frames,
            dtype=torch.float32,
        ).unsqueeze(1)

        even_indices = torch.arange(
            0,
            embedding_dim,
            2,
            dtype=torch.float32,
        )

        div_term = torch.exp(
            even_indices
            * (
                -math.log(10000.0)
                / embedding_dim
            )
        )

        encoding = torch.zeros(
            max_frames,
            embedding_dim,
        )

        encoding[:, 0::2] = torch.sin(
            positions * div_term
        )

        cosine_columns = (
            encoding[:, 1::2].shape[1]
        )

        encoding[:, 1::2] = torch.cos(
            positions
            * div_term[:cosine_columns]
        )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(
        self,
        embeddings: Tensor,
    ) -> Tensor:
        frames = embeddings.shape[1]

        if frames > self.encoding.shape[1]:
            raise VideoValidationError(
                "Sequence exceeds positional capacity"
            )

        return (
            embeddings
            + self.encoding[
                :, :frames
            ].to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        )


# ============================================================
# TEMPORAL TRANSFORMER
# ============================================================

class TemporalTransformer(nn.Module):
    """
    Models relationships across frames.

    Input:
        [B,T,D]

    Padding mask:
        [B,T], True means invalid frame.
    """

    def __init__(
        self,
        config: VideoEncoderConfig,
    ) -> None:
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=(
                config.feedforward_dim
            ),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=(
                config.transformer_layers
            ),
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(
            config.embedding_dim
        )

    def forward(
        self,
        embeddings: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        output = self.encoder(
            embeddings,
            src_key_padding_mask=(
                padding_mask
            ),
        )

        return self.norm(
            output
        )


# ============================================================
# TEMPORAL ATTENTION POOLING
# ============================================================

class TemporalAttentionPooling(nn.Module):
    """
    Learns which frames are most important.

    Input:
        [B,T,D]

    Returns:
        pooled: [B,D]
        weights: [B,T]
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                embedding_dim,
                1,
            ),
        )

    def forward(
        self,
        embeddings: Tensor,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = self.scorer(
            embeddings
        ).squeeze(-1)

        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask,
                torch.finfo(
                    scores.dtype
                ).min,
            )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        pooled = torch.sum(
            embeddings
            * weights.unsqueeze(-1),
            dim=1,
        )

        return pooled, weights


# ============================================================
# VIDEO ENCODER OUTPUT
# ============================================================

@dataclass
class VideoEncoderOutput:
    frame_embeddings: Tensor

    temporal_embeddings: Tensor

    latent_states: Tensor

    video_embedding: Tensor

    attention_weights: Tensor

    valid_frame_mask: Tensor

    def detach_cpu(
        self,
    ) -> dict[str, Tensor]:
        return {
            "frame_embeddings": (
                self.frame_embeddings
                .detach()
                .cpu()
            ),
            "temporal_embeddings": (
                self.temporal_embeddings
                .detach()
                .cpu()
            ),
            "latent_states": (
                self.latent_states
                .detach()
                .cpu()
            ),
            "video_embedding": (
                self.video_embedding
                .detach()
                .cpu()
            ),
            "attention_weights": (
                self.attention_weights
                .detach()
                .cpu()
            ),
            "valid_frame_mask": (
                self.valid_frame_mask
                .detach()
                .cpu()
            ),
        }


# ============================================================
# MAIN VIDEO ENCODER
# ============================================================

class VideoEncoder(nn.Module):
    """
    NeuroWorld video representation encoder.

    Pipeline:
        Video frames
            |
        Spatial CNN
            |
        Frame embeddings
            |
        Positional encoding
            |
        Temporal Transformer
            |
        Latent state projection
            |
        Attention pooling
            |
        Global video embedding

    This is a representation encoder,
    not a generative video model.
    """

    def __init__(
        self,
        config: VideoEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or VideoEncoderConfig()
        )

        if self.config.use_pretrained_backbone:
            if self.config.image_channels != 3:
                raise VideoEncoderError(
                    "Pretrained ResNet requires RGB"
                )

            self.spatial_encoder = (
                PretrainedResNetEncoder(
                    self.config.embedding_dim
                )
            )

        else:
            self.spatial_encoder = (
                SpatialFrameEncoder(
                    self.config
                )
            )

        self.normalizer = (
            VideoNormalizer(
                self.config.image_channels
            )
        )

        self.position_encoding = (
            TemporalPositionalEncoding(
                self.config.embedding_dim,
                self.config.max_frames,
            )
        )

        self.temporal_encoder = (
            TemporalTransformer(
                self.config
            )
        )

        self.latent_projection = nn.Sequential(
            nn.Linear(
                self.config.embedding_dim,
                self.config.latent_dim,
            ),
            nn.LayerNorm(
                self.config.latent_dim
            ),
        )

        self.attention_pool = (
            TemporalAttentionPooling(
                self.config.embedding_dim
            )
        )

        self.global_projection = nn.Sequential(
            nn.Linear(
                self.config.embedding_dim,
                self.config.embedding_dim,
            ),
            nn.LayerNorm(
                self.config.embedding_dim
            ),
        )

    def forward(
        self,
        video: Tensor,
        valid_frame_mask: Tensor | None = None,
    ) -> VideoEncoderOutput:
        validate_video_tensor(
            video,
            self.config,
        )

        batch_size, frame_count, channels, height, width = (
            video.shape
        )

        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=video.device,
            )

        else:
            if valid_frame_mask.shape != (
                batch_size,
                frame_count,
            ):
                raise VideoValidationError(
                    "Mask must be [B,T]"
                )

            valid_frame_mask = (
                valid_frame_mask.to(
                    device=video.device,
                    dtype=torch.bool,
                )
            )

        if not valid_frame_mask.any(dim=1).all():
            raise VideoValidationError(
                "Every video needs at least one valid frame"
            )

        video = video.float()

        if self.config.normalize_inputs:
            video = self.normalizer(
                video
            )

        frames = video.reshape(
            batch_size * frame_count,
            channels,
            height,
            width,
        )

        frame_embeddings = (
            self.spatial_encoder(
                frames
            )
        )

        frame_embeddings = (
            frame_embeddings.reshape(
                batch_size,
                frame_count,
                self.config.embedding_dim,
            )
        )

        positioned = (
            self.position_encoding(
                frame_embeddings
            )
        )

        padding_mask = (
            ~valid_frame_mask
        )

        temporal_embeddings = (
            self.temporal_encoder(
                positioned,
                padding_mask=padding_mask,
            )
        )

        # Remove invalid-frame outputs.
        temporal_embeddings = (
            temporal_embeddings
            * valid_frame_mask
            .unsqueeze(-1)
            .to(
                temporal_embeddings.dtype
            )
        )

        latent_states = (
            self.latent_projection(
                temporal_embeddings
            )
        )

        latent_states = (
            latent_states
            * valid_frame_mask
            .unsqueeze(-1)
            .to(
                latent_states.dtype
            )
        )

        video_embedding, attention_weights = (
            self.attention_pool(
                temporal_embeddings,
                padding_mask=padding_mask,
            )
        )

        video_embedding = (
            self.global_projection(
                video_embedding
            )
        )

        return VideoEncoderOutput(
            frame_embeddings=(
                frame_embeddings
            ),
            temporal_embeddings=(
                temporal_embeddings
            ),
            latent_states=latent_states,
            video_embedding=video_embedding,
            attention_weights=(
                attention_weights
            ),
            valid_frame_mask=(
                valid_frame_mask
            ),
        )

    @torch.no_grad()
    def encode(
        self,
        video: Tensor,
        valid_frame_mask: Tensor | None = None,
    ) -> VideoEncoderOutput:
        was_training = self.training

        self.eval()

        try:
            return self.forward(
                video,
                valid_frame_mask,
            )
        finally:
            self.train(
                was_training
            )


# ============================================================
# VIDEO FRAME SAMPLER
# ============================================================

class VideoFrameSampler:
    """
    Samples frames from a video tensor.

    Input:
        [T,C,H,W]

    Output:
        [sample_count,C,H,W]
    """

    @staticmethod
    def uniform_sample(
        video: Tensor,
        sample_count: int,
    ) -> Tensor:
        if video.ndim != 4:
            raise VideoValidationError(
                "Video must be [T,C,H,W]"
            )

        if sample_count <= 0:
            raise VideoValidationError(
                "sample_count must be positive"
            )

        total_frames = video.shape[0]

        if total_frames <= 0:
            raise VideoValidationError(
                "Video contains no frames"
            )

        indices = torch.linspace(
            0,
            total_frames - 1,
            sample_count,
            device=video.device,
        ).round().long()

        return video.index_select(
            0,
            indices,
        )

    @staticmethod
    def stride_sample(
        video: Tensor,
        stride: int,
    ) -> Tensor:
        if video.ndim != 4:
            raise VideoValidationError(
                "Video must be [T,C,H,W]"
            )

        if stride <= 0:
            raise VideoValidationError(
                "stride must be positive"
            )

        return video[
            ::stride
        ]


# ============================================================
# STREAMING VIDEO ENCODER
# ============================================================

class StreamingVideoEncoder:
    """
    Maintains a sliding frame window.

    Each new frame can update the encoded
    temporal representation.

    This is windowed re-encoding,
    not incremental KV-cache inference.
    """

    def __init__(
        self,
        encoder: VideoEncoder,
        window_size: int = 16,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                "window_size must be positive"
            )

        if window_size > encoder.config.max_frames:
            raise ValueError(
                "window_size exceeds max_frames"
            )

        self.encoder = encoder
        self.window_size = window_size

        self.frames: list[Tensor] = []

    def reset(self) -> None:
        self.frames.clear()

    def add_frame(
        self,
        frame: Tensor,
    ) -> VideoEncoderOutput:
        if frame.ndim != 3:
            raise VideoValidationError(
                "Frame must be [C,H,W]"
            )

        if frame.shape[0] != (
            self.encoder.config.image_channels
        ):
            raise VideoValidationError(
                "Unexpected frame channels"
            )

        self.frames.append(
            frame.detach().clone()
        )

        if len(self.frames) > self.window_size:
            self.frames.pop(0)

        video = torch.stack(
            self.frames,
            dim=0,
        ).unsqueeze(0)

        model_device = next(
            self.encoder.parameters()
        ).device

        video = video.to(
            model_device
        )

        return self.encoder.encode(
            video
        )


# ============================================================
# WORLD-MODEL ADAPTER
# ============================================================

class VideoWorldModelAdapter(nn.Module):
    """
    Provides video-derived latent states
    to downstream neural world models.

    Outputs:
        [B,T,latent_dim]

    The downstream world model must
    accept this latent representation.
    """

    def __init__(
        self,
        encoder: VideoEncoder,
    ) -> None:
        super().__init__()

        self.encoder = encoder

    def forward(
        self,
        video: Tensor,
        valid_frame_mask: Tensor | None = None,
    ) -> Tensor:
        output = self.encoder(
            video,
            valid_frame_mask,
        )

        return output.latent_states

    def encode_global(
        self,
        video: Tensor,
        valid_frame_mask: Tensor | None = None,
    ) -> Tensor:
        output = self.encoder(
            video,
            valid_frame_mask,
        )

        return output.video_embedding


# ============================================================
# FACTORY
# ============================================================

def create_video_encoder(
    config: VideoEncoderConfig | None = None,
) -> VideoEncoder:
    config = (
        config or VideoEncoderConfig()
    )

    return VideoEncoder(
        config
    ).to(
        config.device
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - VIDEO ENCODER"
    )
    print("=" * 65)

    config = VideoEncoderConfig(
        image_channels=3,
        embedding_dim=128,
        latent_dim=64,
        cnn_channels=(
            16,
            32,
            64,
        ),
        transformer_layers=2,
        attention_heads=4,
        feedforward_dim=256,
        max_frames=32,
        use_pretrained_backbone=False,
    )

    encoder = create_video_encoder(
        config
    )

    video = torch.rand(
        2,
        8,
        3,
        64,
        64,
    )

    valid_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    print("\n1. Video encoding")

    output = encoder.encode(
        video,
        valid_mask,
    )

    print(
        "Frame embeddings:",
        output.frame_embeddings.shape,
    )

    print(
        "Temporal embeddings:",
        output.temporal_embeddings.shape,
    )

    print(
        "Latent states:",
        output.latent_states.shape,
    )

    print(
        "Global embedding:",
        output.video_embedding.shape,
    )

    print(
        "Attention weights:",
        output.attention_weights.shape,
    )

    print("\n2. Uniform frame sampling")

    sampled = VideoFrameSampler.uniform_sample(
        video[0],
        sample_count=4,
    )

    print(
        "Sampled frames:",
        sampled.shape,
    )

    print("\n3. Streaming video encoding")

    streaming = StreamingVideoEncoder(
        encoder,
        window_size=4,
    )

    for frame in video[0, :4]:
        stream_output = (
            streaming.add_frame(
                frame
            )
        )

    print(
        "Streaming embedding:",
        stream_output.video_embedding.shape,
    )

    print("\n4. World-model adapter")

    adapter = VideoWorldModelAdapter(
        encoder
    )

    latents = adapter(
        video,
        valid_mask,
    )

    print(
        "World-model latents:",
        latents.shape,
    )

    print(
        "\nVideo encoder demonstration completed"
    )


if __name__ == "__main__":
    main()
