
"""
NeuroWorld AI
File: neuroworld/perception/motion_encoder.py

Motion Encoder for Neural World Models

Input modes:
    video:       [B, T, C, H, W]
    optical_flow:[B, T-1, 2, H, W]
    trajectories:[B, T, N, D]

Outputs:
    global_embedding: [B, E]
    temporal_tokens:  [B, T-1, E]
    motion_logits:    [B, num_motion_classes]
    motion_magnitude: [B, T-1]
    motion_direction: [B, T-1, 2]

Video mode uses frame differences as a
lightweight motion proxy, not optical flow.

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.perception.motion_encoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MotionEncoderConfig:
    input_channels: int = 3
    embedding_dim: int = 256
    hidden_dim: int = 128

    transformer_layers: int = 3
    attention_heads: int = 8
    dropout: float = 0.1

    max_frames: int = 64
    num_motion_classes: int = 6

    trajectory_dim: int = 4

    normalize_embeddings: bool = True

    def __post_init__(self):
        if min(
            self.input_channels,
            self.embedding_dim,
            self.hidden_dim,
            self.transformer_layers,
            self.attention_heads,
            self.max_frames,
            self.num_motion_classes,
            self.trajectory_dim,
        ) <= 0:
            raise ValueError(
                "All dimensions must be positive"
            )

        if (
            self.embedding_dim
            % self.attention_heads
        ) != 0:
            raise ValueError(
                "embedding_dim must be divisible "
                "by attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )


# ============================================================
# OUTPUT
# ============================================================

@dataclass
class MotionEncoderOutput:
    global_embedding: Tensor
    temporal_tokens: Tensor

    motion_logits: Tensor
    motion_probabilities: Tensor

    motion_magnitude: Tensor
    motion_direction: Tensor

    attention_weights: Tensor

    def to_dict(self):
        return {
            "global_embedding": (
                self.global_embedding
            ),
            "temporal_tokens": (
                self.temporal_tokens
            ),
            "motion_logits": (
                self.motion_logits
            ),
            "motion_probabilities": (
                self.motion_probabilities
            ),
            "motion_magnitude": (
                self.motion_magnitude
            ),
            "motion_direction": (
                self.motion_direction
            ),
            "attention_weights": (
                self.attention_weights
            ),
        }


# ============================================================
# FRAME DIFFERENCE
# ============================================================

class FrameDifferenceExtractor(nn.Module):
    """
    Lightweight appearance-change features.

    Frame differences can reflect lighting,
    camera motion, or actual object motion.
    They are not true optical flow.
    """

    def forward(
        self,
        video: Tensor,
    ) -> Tensor:
        if video.ndim != 5:
            raise ValueError(
                "video must be [B,T,C,H,W]"
            )

        if video.shape[1] < 2:
            raise ValueError(
                "At least two frames required"
            )

        if not torch.isfinite(video).all():
            raise ValueError(
                "Video contains non-finite values"
            )

        return (
            video[:, 1:]
            - video[:, :-1]
        )


# ============================================================
# CNN MOTION BACKBONE
# ============================================================

class MotionCNN(nn.Module):
    """
    Encodes each motion frame into a vector.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(
                input_channels,
                hidden_dim // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(
                hidden_dim // 2
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(
                hidden_dim
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(
                hidden_dim,
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
        return self.network(frames)


# ============================================================
# TRAJECTORY MOTION ENCODER
# ============================================================

class TrajectoryMotionEncoder(nn.Module):
    """
    Encodes tracked-object displacement.

    trajectories:
        [B,T,N,D]

    Uses mean object displacement,
    with a validity mask for padded objects.
    """

    def __init__(
        self,
        trajectory_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                trajectory_dim,
                embedding_dim,
            ),
            nn.GELU(),
            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),
            nn.LayerNorm(
                embedding_dim
            ),
        )

    def forward(
        self,
        trajectories: Tensor,
        object_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if trajectories.ndim != 4:
            raise ValueError(
                "trajectories must be [B,T,N,D]"
            )

        if trajectories.shape[1] < 2:
            raise ValueError(
                "At least two timestamps required"
            )

        displacement = (
            trajectories[:, 1:]
            - trajectories[:, :-1]
        )

        features = self.network(
            displacement
        )

        if object_mask is None:
            return features.mean(dim=2)

        if object_mask.shape != (
            trajectories.shape[0],
            trajectories.shape[2],
        ):
            raise ValueError(
                "object_mask must be [B,N]"
            )

        weights = object_mask.to(
            dtype=features.dtype,
            device=features.device,
        )[:, None, :, None]

        weighted_sum = (
            features * weights
        ).sum(dim=2)

        count = weights.sum(
            dim=2
        ).clamp_min(1.0)

        return weighted_sum / count


# ============================================================
# TEMPORAL TRANSFORMER
# ============================================================

class TemporalMotionTransformer(nn.Module):
    def __init__(
        self,
        config: MotionEncoderConfig,
    ):
        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                config.max_frames - 1,
                config.embedding_dim,
            )
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.attention_heads,
            dim_feedforward=(
                config.embedding_dim * 4
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
        tokens: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        sequence_length = tokens.shape[1]

        if sequence_length > (
            self.position_embedding.shape[1]
        ):
            raise ValueError(
                "Sequence exceeds max_frames"
            )

        tokens = (
            tokens
            + self.position_embedding[
                :, :sequence_length
            ]
        )

        tokens = self.encoder(
            tokens,
            src_key_padding_mask=padding_mask,
        )

        return self.norm(tokens)


# ============================================================
# TEMPORAL ATTENTION POOLING
# ============================================================

class TemporalAttentionPooling(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                embedding_dim,
                embedding_dim // 2,
            ),
            nn.Tanh(),
            nn.Linear(
                embedding_dim // 2,
                1,
            ),
        )

    def forward(
        self,
        tokens: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        scores = self.attention(
            tokens
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
            dim=1,
        )

        pooled = (
            tokens
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        return pooled, weights


# ============================================================
# MOTION STATISTICS
# ============================================================

class MotionStatistics(nn.Module):
    """
    Produces basic motion statistics.

    Optical flow:
        Average pixel displacement.

    Video:
        Mean absolute frame change.
        Direction is not inferred reliably
        from raw frame differences.

    Trajectories:
        Average object displacement.
    """

    @staticmethod
    def from_flow(
        flow: Tensor,
    ) -> tuple[Tensor, Tensor]:
        mean_vector = flow.mean(
            dim=(-2, -1)
        )

        magnitude = torch.linalg.vector_norm(
            flow,
            dim=2,
        ).mean(
            dim=(-2, -1)
        )

        direction = F.normalize(
            mean_vector,
            dim=-1,
            eps=1e-8,
        )

        return magnitude, direction

    @staticmethod
    def from_video(
        differences: Tensor,
    ) -> tuple[Tensor, Tensor]:
        magnitude = differences.abs().mean(
            dim=(2, 3, 4)
        )

        direction = torch.zeros(
            *magnitude.shape,
            2,
            device=differences.device,
            dtype=differences.dtype,
        )

        return magnitude, direction

    @staticmethod
    def from_trajectories(
        trajectories: Tensor,
        object_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        displacement = (
            trajectories[:, 1:, :, :2]
            - trajectories[:, :-1, :, :2]
        )

        if object_mask is None:
            mean_vector = displacement.mean(
                dim=2
            )

            magnitude = (
                torch.linalg.vector_norm(
                    displacement,
                    dim=-1,
                ).mean(dim=2)
            )

        else:
            weights = object_mask.to(
                dtype=displacement.dtype,
                device=displacement.device,
            )[:, None, :]

            count = weights.sum(
                dim=2
            ).clamp_min(1.0)

            mean_vector = (
                displacement
                * weights.unsqueeze(-1)
            ).sum(dim=2) / count.unsqueeze(-1)

            magnitude = (
                torch.linalg.vector_norm(
                    displacement,
                    dim=-1,
                ) * weights
            ).sum(dim=2) / count

        direction = F.normalize(
            mean_vector,
            dim=-1,
            eps=1e-8,
        )

        return magnitude, direction


# ============================================================
# MAIN MOTION ENCODER
# ============================================================

class MotionEncoder(nn.Module):
    """
    Multimodal motion encoder.

    Supported input modes:
        video
        optical_flow
        trajectories

    Only one input mode per forward call.

    Outputs temporal and global embeddings
    for future prediction, world modeling,
    planning, and motion recognition.
    """

    def __init__(
        self,
        config: Optional[
            MotionEncoderConfig
        ] = None,
    ):
        super().__init__()

        self.config = (
            config
            or MotionEncoderConfig()
        )

        self.frame_difference = (
            FrameDifferenceExtractor()
        )

        self.video_backbone = MotionCNN(
            input_channels=(
                self.config.input_channels
            ),
            hidden_dim=(
                self.config.hidden_dim
            ),
            embedding_dim=(
                self.config.embedding_dim
            ),
        )

        self.flow_backbone = MotionCNN(
            input_channels=2,
            hidden_dim=(
                self.config.hidden_dim
            ),
            embedding_dim=(
                self.config.embedding_dim
            ),
        )

        self.trajectory_encoder = (
            TrajectoryMotionEncoder(
                trajectory_dim=(
                    self.config.trajectory_dim
                ),
                embedding_dim=(
                    self.config.embedding_dim
                ),
            )
        )

        self.temporal_transformer = (
            TemporalMotionTransformer(
                self.config
            )
        )

        self.attention_pool = (
            TemporalAttentionPooling(
                self.config.embedding_dim
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.config.embedding_dim,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                self.config.dropout
            ),
            nn.Linear(
                self.config.hidden_dim,
                self.config.num_motion_classes,
            ),
        )

        self.statistics = MotionStatistics()

    def _encode_video(
        self,
        video: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        if video.ndim != 5:
            raise ValueError(
                "video must be [B,T,C,H,W]"
            )

        if video.shape[2] != (
            self.config.input_channels
        ):
            raise ValueError(
                "Video channel count mismatch"
            )

        differences = (
            self.frame_difference(video)
        )

        batch, time, channels, height, width = (
            differences.shape
        )

        flat = differences.reshape(
            batch * time,
            channels,
            height,
            width,
        )

        tokens = self.video_backbone(
            flat
        ).reshape(
            batch,
            time,
            -1,
        )

        magnitude, direction = (
            self.statistics.from_video(
                differences
            )
        )

        return tokens, magnitude, direction

    def _encode_flow(
        self,
        flow: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        if (
            flow.ndim != 5
            or flow.shape[2] != 2
        ):
            raise ValueError(
                "optical_flow must be "
                "[B,T-1,2,H,W]"
            )

        batch, time, channels, height, width = (
            flow.shape
        )

        flat = flow.reshape(
            batch * time,
            channels,
            height,
            width,
        )

        tokens = self.flow_backbone(
            flat
        ).reshape(
            batch,
            time,
            -1,
        )

        magnitude, direction = (
            self.statistics.from_flow(
                flow
            )
        )

        return tokens, magnitude, direction

    def _encode_trajectories(
        self,
        trajectories: Tensor,
        object_mask: Optional[Tensor],
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        if trajectories.ndim != 4:
            raise ValueError(
                "trajectories must be [B,T,N,D]"
            )

        if trajectories.shape[-1] != (
            self.config.trajectory_dim
        ):
            raise ValueError(
                "Trajectory feature dimension mismatch"
            )

        if trajectories.shape[-1] < 2:
            raise ValueError(
                "Trajectory coordinates need D>=2"
            )

        tokens = (
            self.trajectory_encoder(
                trajectories,
                object_mask,
            )
        )

        magnitude, direction = (
            self.statistics.from_trajectories(
                trajectories,
                object_mask,
            )
        )

        return tokens, magnitude, direction

    def forward(
        self,
        *,
        video: Optional[Tensor] = None,
        optical_flow: Optional[Tensor] = None,
        trajectories: Optional[Tensor] = None,
        object_mask: Optional[Tensor] = None,
        temporal_padding_mask: Optional[
            Tensor
        ] = None,
    ) -> MotionEncoderOutput:
        supplied = sum(
            item is not None
            for item in (
                video,
                optical_flow,
                trajectories,
            )
        )

        if supplied != 1:
            raise ValueError(
                "Supply exactly one of video, "
                "optical_flow, or trajectories"
            )

        if video is not None:
            tokens, magnitude, direction = (
                self._encode_video(video)
            )

        elif optical_flow is not None:
            tokens, magnitude, direction = (
                self._encode_flow(
                    optical_flow
                )
            )

        else:
            tokens, magnitude, direction = (
                self._encode_trajectories(
                    trajectories,
                    object_mask,
                )
            )

        if not torch.isfinite(tokens).all():
            raise ValueError(
                "Motion features must be finite"
            )

        if tokens.shape[1] < 1:
            raise ValueError(
                "At least one motion step required"
            )

        if temporal_padding_mask is not None:
            if temporal_padding_mask.shape != (
                tokens.shape[0],
                tokens.shape[1],
            ):
                raise ValueError(
                    "temporal_padding_mask must be [B,S]"
                )

            temporal_padding_mask = (
                temporal_padding_mask.bool()
            )

            if temporal_padding_mask.all(
                dim=1
            ).any():
                raise ValueError(
                    "Every sample requires "
                    "at least one valid step"
                )

        temporal_tokens = (
            self.temporal_transformer(
                tokens,
                temporal_padding_mask,
            )
        )

        global_embedding, attention = (
            self.attention_pool(
                temporal_tokens,
                temporal_padding_mask,
            )
        )

        if self.config.normalize_embeddings:
            global_embedding = F.normalize(
                global_embedding,
                dim=-1,
                eps=1e-8,
            )

        logits = self.classifier(
            global_embedding
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        return MotionEncoderOutput(
            global_embedding=(
                global_embedding
            ),
            temporal_tokens=(
                temporal_tokens
            ),
            motion_logits=logits,
            motion_probabilities=(
                probabilities
            ),
            motion_magnitude=magnitude,
            motion_direction=direction,
            attention_weights=attention,
        )

    @torch.no_grad()
    def encode(
        self,
        **kwargs,
    ) -> Tensor:
        """
        Return global motion embeddings.

        Note:
            Does not change model training mode.
            Call model.eval() for inference.
        """
        return self.forward(
            **kwargs
        ).global_embedding


# ============================================================
# TRAINING LOSS
# ============================================================

class MotionClassificationLoss(nn.Module):
    """
    Cross-entropy loss for labeled
    motion categories.
    """

    def forward(
        self,
        output: MotionEncoderOutput,
        labels: Tensor,
    ) -> Tensor:
        if labels.shape != (
            output.motion_logits.shape[0],
        ):
            raise ValueError(
                "labels must be [B]"
            )

        return F.cross_entropy(
            output.motion_logits,
            labels.long(),
        )


# ============================================================
# FACTORY
# ============================================================

def create_motion_encoder(
    config: Optional[
        MotionEncoderConfig
    ] = None,
) -> MotionEncoder:
    return MotionEncoder(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main():
    torch.manual_seed(42)

    config = MotionEncoderConfig(
        input_channels=3,
        embedding_dim=128,
        hidden_dim=64,
        transformer_layers=2,
        attention_heads=4,
        max_frames=16,
        num_motion_classes=6,
        trajectory_dim=4,
    )

    model = MotionEncoder(
        config
    ).eval()

    print("=" * 60)
    print("NEUROWORLD AI - MOTION ENCODER")
    print("=" * 60)

    # ----------------------------------------
    # Video frame differences
    # ----------------------------------------

    video = torch.rand(
        2,
        8,
        3,
        64,
        64,
    )

    with torch.no_grad():
        output = model(
            video=video
        )

    print("\nVIDEO MODE")

    print(
        "Global embedding:",
        output.global_embedding.shape,
    )

    print(
        "Temporal tokens:",
        output.temporal_tokens.shape,
    )

    print(
        "Motion logits:",
        output.motion_logits.shape,
    )

    print(
        "Motion magnitude:",
        output.motion_magnitude.shape,
    )

    # ----------------------------------------
    # Optical flow
    # ----------------------------------------

    flow = torch.randn(
        2,
        7,
        2,
        64,
        64,
    ) * 0.1

    with torch.no_grad():
        flow_output = model(
            optical_flow=flow
        )

    print("\nOPTICAL FLOW MODE")

    print(
        "Global embedding:",
        flow_output.global_embedding.shape,
    )

    print(
        "Motion direction:",
        flow_output.motion_direction.shape,
    )

    # ----------------------------------------
    # Object trajectories
    # ----------------------------------------

    trajectories = torch.randn(
        2,
        8,
        5,
        4,
    )

    object_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        trajectory_output = model(
            trajectories=trajectories,
            object_mask=object_mask,
        )

    print("\nTRAJECTORY MODE")

    print(
        "Global embedding:",
        trajectory_output.global_embedding.shape,
    )

    print(
        "Temporal tokens:",
        trajectory_output.temporal_tokens.shape,
    )

    # ----------------------------------------
    # Training example
    # ----------------------------------------

    model.train()

    labels = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    train_output = model(
        video=video
    )

    criterion = MotionClassificationLoss()

    loss = criterion(
        train_output,
        labels,
    )

    loss.backward()

    print("\nTRAINING")

    print(
        "Classification loss:",
        round(loss.item(), 4),
    )

    print(
        "\nMotion encoder demonstration complete."
    )


if __name__ == "__main__":
    main()
