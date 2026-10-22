
"""
NeuroWorld AI
File: neuroworld/perception/multimodal_encoder.py

Multimodal Perception & Sensor Fusion Encoder

Features:
    - Image feature encoding
    - Video feature encoding
    - Motion feature encoding
    - Spatial feature encoding
    - Audio feature encoding
    - Text feature encoding
    - Robot sensor feature encoding
    - Cross-modal transformer attention
    - Modality embeddings
    - Missing-modality handling
    - Attention-based feature fusion
    - Temporal feature aggregation
    - Unified world-state embedding
    - Batch processing
    - Feature normalization
    - Training-compatible PyTorch modules
    - Modality contribution estimates
    - World-model integration

Install:
    pip install torch numpy

Run:
    python -m neuroworld.perception.multimodal_encoder

Important:
    This encoder expects feature vectors or
    feature sequences from upstream encoders.

    It does not directly process raw images,
    waveforms, or natural-language strings.

    Attention weights are not calibrated
    explanations of modality importance.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class MultimodalEncoderError(Exception):
    pass


class MultimodalValidationError(
    MultimodalEncoderError
):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class MultimodalEncoderConfig:
    image_dim: int = 512
    video_dim: int = 512
    motion_dim: int = 128
    spatial_dim: int = 128
    audio_dim: int = 256
    text_dim: int = 768
    sensor_dim: int = 64

    hidden_dim: int = 256
    latent_dim: int = 128

    attention_heads: int = 8
    transformer_layers: int = 3

    feedforward_multiplier: int = 4

    dropout: float = 0.1

    maximum_video_frames: int = 64

    normalize_output: bool = True

    use_modality_gating: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.image_dim,
            self.video_dim,
            self.motion_dim,
            self.spatial_dim,
            self.audio_dim,
            self.text_dim,
            self.sensor_dim,
            self.hidden_dim,
            self.latent_dim,
            self.attention_heads,
            self.transformer_layers,
            self.feedforward_multiplier,
            self.maximum_video_frames,
        )

        if any(
            dimension <= 0
            for dimension in dimensions
        ):
            raise ValueError(
                "All dimensions must be positive"
            )

        if (
            self.hidden_dim
            % self.attention_heads != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible "
                "by attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )


# ============================================================
# MODALITY REGISTRY
# ============================================================

MODALITIES = (
    "image",
    "video",
    "motion",
    "spatial",
    "audio",
    "text",
    "sensor",
)


def modality_dimensions(
    config: MultimodalEncoderConfig,
) -> dict[str, int]:
    return {
        "image": config.image_dim,
        "video": config.video_dim,
        "motion": config.motion_dim,
        "spatial": config.spatial_dim,
        "audio": config.audio_dim,
        "text": config.text_dim,
        "sensor": config.sensor_dim,
    }


# ============================================================
# INPUT STRUCTURE
# ============================================================

@dataclass
class MultimodalInput:
    """
    Supported shapes:

    image:   [B,image_dim]
    video:   [B,T,video_dim]
    motion:  [B,motion_dim]
    spatial: [B,spatial_dim]
    audio:   [B,audio_dim]
    text:    [B,text_dim]
    sensor:  [B,sensor_dim]

    video_mask:
        [B,T] boolean
        True means valid frame.

    modality_mask:
        Dictionary mapping modality names
        to boolean [B] availability masks.
    """

    image: Tensor | None = None
    video: Tensor | None = None
    motion: Tensor | None = None
    spatial: Tensor | None = None
    audio: Tensor | None = None
    text: Tensor | None = None
    sensor: Tensor | None = None

    video_mask: Tensor | None = None

    modality_mask: dict[
        str,
        Tensor,
    ] = field(default_factory=dict)

    def available_modalities(
        self,
    ) -> list[str]:
        return [
            name
            for name in MODALITIES
            if getattr(self, name) is not None
        ]


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

@dataclass
class MultimodalEncoderOutput:
    latent: Tensor
    fused_features: Tensor

    modality_features: dict[
        str,
        Tensor,
    ]

    modality_weights: Tensor

    availability_mask: Tensor

    attention_weights: Tensor | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def detach_to_numpy(
        self,
    ) -> dict[str, Any]:
        def convert(
            tensor: Tensor,
        ) -> np.ndarray:
            return (
                tensor.detach()
                .cpu()
                .numpy()
            )

        return {
            "latent": convert(
                self.latent
            ),
            "fused_features": convert(
                self.fused_features
            ),
            "modality_features": {
                name: convert(value)
                for name, value
                in self.modality_features.items()
            },
            "modality_weights": convert(
                self.modality_weights
            ),
            "availability_mask": convert(
                self.availability_mask
            ),
            "metadata": self.metadata,
        }


# ============================================================
# FEATURE PROJECTION
# ============================================================

class ModalityProjection(nn.Module):
    """
    Projects input features into
    the shared hidden dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        return self.network(
            features
        )


# ============================================================
# TEMPORAL VIDEO ENCODER
# ============================================================

class TemporalVideoAggregator(nn.Module):
    """
    Converts video features:
        [B,T,video_dim]

    Into:
        [B,hidden_dim]

    Uses temporal positional embeddings,
    transformer self-attention, and
    masked attention pooling.
    """

    def __init__(
        self,
        config: MultimodalEncoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_projection = (
            ModalityProjection(
                config.video_dim,
                config.hidden_dim,
                config.dropout,
            )
        )

        self.position_embedding = (
            nn.Embedding(
                config.maximum_video_frames,
                config.hidden_dim,
            )
        )

        layer = (
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.attention_heads,
                dim_feedforward=(
                    config.hidden_dim
                    * config.feedforward_multiplier
                ),
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=1,
                enable_nested_tensor=False,
            )
        )

        self.pooling_score = nn.Linear(
            config.hidden_dim,
            1,
        )

        self.output_norm = nn.LayerNorm(
            config.hidden_dim
        )

    def forward(
        self,
        video: Tensor,
        video_mask: Tensor | None = None,
    ) -> Tensor:
        if video.ndim != 3:
            raise MultimodalValidationError(
                "Video must be [B,T,D]"
            )

        batch_size, frames, dimension = (
            video.shape
        )

        if dimension != self.config.video_dim:
            raise MultimodalValidationError(
                "Video feature dimension mismatch"
            )

        if frames == 0:
            raise MultimodalValidationError(
                "Video must contain frames"
            )

        if frames > (
            self.config.maximum_video_frames
        ):
            raise MultimodalValidationError(
                "Video exceeds maximum_video_frames"
            )

        if video_mask is None:
            video_mask = torch.ones(
                batch_size,
                frames,
                dtype=torch.bool,
                device=video.device,
            )

        if video_mask.shape != (
            batch_size,
            frames,
        ):
            raise MultimodalValidationError(
                "video_mask must be [B,T]"
            )

        video_mask = video_mask.to(
            device=video.device,
            dtype=torch.bool,
        )

        # A completely masked sequence is
        # not valid transformer input.
        if not video_mask.any(dim=1).all():
            raise MultimodalValidationError(
                "Each video must have a valid frame"
            )

        hidden = self.input_projection(
            video
        )

        indices = torch.arange(
            frames,
            device=video.device,
        )

        hidden = (
            hidden
            + self.position_embedding(
                indices
            ).unsqueeze(0)
        )

        hidden = self.transformer(
            hidden,
            src_key_padding_mask=(
                ~video_mask
            ),
        )

        scores = (
            self.pooling_score(
                hidden
            ).squeeze(-1)
        )

        scores = scores.masked_fill(
            ~video_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        pooled = torch.sum(
            hidden
            * weights.unsqueeze(-1),
            dim=1,
        )

        return self.output_norm(
            pooled
        )


# ============================================================
# MODALITY GATING
# ============================================================

class ModalityGatingNetwork(nn.Module):
    """
    Learns a normalized weight for
    each available modality.

    Input:
        [B,M,H]

    Output:
        [B,M]

    Missing modalities receive zero
    weight after masked softmax.
    """

    def __init__(
        self,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.LayerNorm(
                hidden_dim
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

    def forward(
        self,
        tokens: Tensor,
        availability_mask: Tensor,
    ) -> Tensor:
        scores = (
            self.gate(
                tokens
            ).squeeze(-1)
        )

        scores = scores.masked_fill(
            ~availability_mask,
            torch.finfo(
                scores.dtype
            ).min,
        )

        return torch.softmax(
            scores,
            dim=-1,
        )


# ============================================================
# CROSS-MODAL TRANSFORMER
# ============================================================

class CrossModalTransformer(nn.Module):
    """
    Transformer-based feature fusion.

    A learnable CLS token attends to
    available modality tokens.

    The output CLS representation becomes
    the global multimodal feature.
    """

    def __init__(
        self,
        config: MultimodalEncoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                config.hidden_dim,
            )
        )

        self.modality_embedding = (
            nn.Embedding(
                len(MODALITIES),
                config.hidden_dim,
            )
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=(
                config.hidden_dim
                * config.feedforward_multiplier
            ),
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=(
                    config.transformer_layers
                ),
                enable_nested_tensor=False,
            )
        )

        self.output_norm = nn.LayerNorm(
            config.hidden_dim
        )

    def forward(
        self,
        tokens: Tensor,
        availability_mask: Tensor,
    ) -> Tensor:
        batch_size = tokens.shape[0]

        modality_indices = torch.arange(
            len(MODALITIES),
            device=tokens.device,
        )

        tokens = (
            tokens
            + self.modality_embedding(
                modality_indices
            ).unsqueeze(0)
        )

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        sequence = torch.cat(
            [cls, tokens],
            dim=1,
        )

        cls_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=tokens.device,
        )

        valid_tokens = torch.cat(
            [
                cls_mask,
                availability_mask,
            ],
            dim=1,
        )

        encoded = self.transformer(
            sequence,
            src_key_padding_mask=(
                ~valid_tokens
            ),
        )

        return self.output_norm(
            encoded[:, 0]
        )


# ============================================================
# MULTIMODAL ENCODER
# ============================================================

class MultimodalEncoder(nn.Module):
    """
    Main multimodal world-state encoder.

    Pipeline:
        Modality inputs
             |
        Feature projections
             |
        Modality tokens
             |
        Cross-modal transformer
             |
        Gated feature fusion
             |
        Latent projection
             |
        Unified world-state embedding
    """

    def __init__(
        self,
        config: (
            MultimodalEncoderConfig | None
        ) = None,
    ) -> None:
        super().__init__()

        self.config = (
            config
            or MultimodalEncoderConfig()
        )

        dimensions = modality_dimensions(
            self.config
        )

        self.projections = nn.ModuleDict(
            {
                name: ModalityProjection(
                    input_dim=dimension,
                    hidden_dim=(
                        self.config.hidden_dim
                    ),
                    dropout=(
                        self.config.dropout
                    ),
                )
                for name, dimension
                in dimensions.items()
                if name != "video"
            }
        )

        self.video_encoder = (
            TemporalVideoAggregator(
                self.config
            )
        )

        self.cross_modal = (
            CrossModalTransformer(
                self.config
            )
        )

        self.gating = (
            ModalityGatingNetwork(
                self.config.hidden_dim
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                self.config.hidden_dim * 2,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                self.config.hidden_dim
            ),
        )

        self.latent_head = nn.Sequential(
            nn.Linear(
                self.config.hidden_dim,
                self.config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.config.hidden_dim,
                self.config.latent_dim,
            ),
        )

    def _infer_batch(
        self,
        inputs: MultimodalInput,
    ) -> tuple[int, torch.device]:
        for name in MODALITIES:
            tensor = getattr(
                inputs,
                name
            )

            if tensor is not None:
                return (
                    tensor.shape[0],
                    tensor.device,
                )

        raise MultimodalValidationError(
            "At least one modality is required"
        )

    def _validate_inputs(
        self,
        inputs: MultimodalInput,
        batch_size: int,
        device: torch.device,
    ) -> None:
        dimensions = modality_dimensions(
            self.config
        )

        for name in MODALITIES:
            tensor = getattr(
                inputs,
                name,
            )

            if tensor is None:
                continue

            if not isinstance(
                tensor,
                Tensor,
            ):
                raise MultimodalValidationError(
                    f"{name} must be a torch.Tensor"
                )

            if tensor.device != device:
                raise MultimodalValidationError(
                    "All modalities must share a device"
                )

            if not tensor.is_floating_point():
                raise MultimodalValidationError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(
                tensor
            ).all():
                raise MultimodalValidationError(
                    f"{name} contains non-finite values"
                )

            if name == "video":
                expected = (
                    tensor.ndim == 3
                    and tensor.shape[0]
                    == batch_size
                    and tensor.shape[2]
                    == dimensions[name]
                )

            else:
                expected = (
                    tensor.shape
                    == (
                        batch_size,
                        dimensions[name],
                    )
                )

            if not expected:
                raise MultimodalValidationError(
                    f"Invalid {name} shape"
                )

        for name, mask in (
            inputs.modality_mask.items()
        ):
            if name not in MODALITIES:
                raise MultimodalValidationError(
                    f"Unknown modality: {name}"
                )

            if mask.shape != (
                batch_size,
            ):
                raise MultimodalValidationError(
                    f"{name} mask must be [B]"
                )

    def forward(
        self,
        inputs: MultimodalInput,
    ) -> MultimodalEncoderOutput:
        batch_size, device = (
            self._infer_batch(
                inputs
            )
        )

        self._validate_inputs(
            inputs,
            batch_size,
            device,
        )

        hidden_dim = (
            self.config.hidden_dim
        )

        tokens = []
        availability = []

        projected_features = {}

        for name in MODALITIES:
            features = getattr(
                inputs,
                name
            )

            if features is None:
                projected = torch.zeros(
                    batch_size,
                    hidden_dim,
                    device=device,
                )

                valid = torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                    device=device,
                )

            else:
                if name == "video":
                    projected = (
                        self.video_encoder(
                            features,
                            inputs.video_mask,
                        )
                    )

                else:
                    projected = (
                        self.projections[name](
                            features
                        )
                    )

                valid = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=device,
                )

                if name in inputs.modality_mask:
                    valid = (
                        inputs.modality_mask[
                            name
                        ].to(
                            device=device,
                            dtype=torch.bool,
                        )
                    )

                projected = (
                    projected
                    * valid.unsqueeze(-1)
                )

            tokens.append(
                projected
            )

            availability.append(
                valid
            )

            projected_features[
                name
            ] = projected

        token_tensor = torch.stack(
            tokens,
            dim=1,
        )

        availability_mask = torch.stack(
            availability,
            dim=1,
        )

        if not availability_mask.any(
            dim=1
        ).all():
            raise MultimodalValidationError(
                "Each sample must have at least "
                "one available modality"
            )

        transformer_features = (
            self.cross_modal(
                token_tensor,
                availability_mask,
            )
        )

        if self.config.use_modality_gating:
            modality_weights = (
                self.gating(
                    token_tensor,
                    availability_mask,
                )
            )

        else:
            weights = (
                availability_mask.float()
            )

            modality_weights = (
                weights
                / weights.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1.0)
            )

        gated_features = torch.sum(
            token_tensor
            * modality_weights.unsqueeze(-1),
            dim=1,
        )

        fused_features = self.fusion(
            torch.cat(
                [
                    transformer_features,
                    gated_features,
                ],
                dim=-1,
            )
        )

        latent = self.latent_head(
            fused_features
        )

        if self.config.normalize_output:
            latent = F.normalize(
                latent,
                p=2,
                dim=-1,
            )

        return MultimodalEncoderOutput(
            latent=latent,
            fused_features=(
                fused_features
            ),
            modality_features=(
                projected_features
            ),
            modality_weights=(
                modality_weights
            ),
            availability_mask=(
                availability_mask
            ),
            metadata={
                "modalities": list(
                    MODALITIES
                ),
                "latent_dim": (
                    self.config.latent_dim
                ),
                "hidden_dim": (
                    self.config.hidden_dim
                ),
            },
        )

    def encode(
        self,
        **modalities: Tensor,
    ) -> Tensor:
        """
        Convenience API.

        Example:
            latent = encoder.encode(
                image=image_features,
                motion=motion_features,
            )
        """
        inputs = MultimodalInput(
            **modalities
        )

        return self.forward(
            inputs
        ).latent


# ============================================================
# SENSOR FUSION ADAPTER
# ============================================================

class SensorFusionAdapter:
    """
    Converts sensor feature dictionaries
    into unified latent vectors.

    Designed for backend services and
    neural world-model inference.
    """

    def __init__(
        self,
        encoder: MultimodalEncoder,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(
            device
        )

        self.encoder = encoder.to(
            self.device
        )

        self.encoder.eval()

    @torch.inference_mode()
    def encode(
        self,
        features: Mapping[
            str,
            np.ndarray | Tensor,
        ],
    ) -> np.ndarray:
        prepared = {}

        for name, value in features.items():
            if name not in MODALITIES:
                raise MultimodalValidationError(
                    f"Unknown modality: {name}"
                )

            tensor = torch.as_tensor(
                value,
                dtype=torch.float32,
                device=self.device,
            )

            if name == "video":
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(
                        0
                    )

            else:
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(
                        0
                    )

            prepared[name] = tensor

        latent = self.encoder.encode(
            **prepared
        )

        return (
            latent.detach()
            .cpu()
            .numpy()
        )


# ============================================================
# MULTIMODAL ALIGNMENT LOSS
# ============================================================

class MultimodalAlignmentLoss(nn.Module):
    """
    Encourages paired modality embeddings
    to be similar.

    Uses cosine-distance alignment.

    This is a basic auxiliary loss,
    not a full contrastive objective.
    """

    def forward(
        self,
        features_a: Tensor,
        features_b: Tensor,
    ) -> Tensor:
        if (
            features_a.shape
            != features_b.shape
        ):
            raise MultimodalValidationError(
                "Alignment feature shapes mismatch"
            )

        similarity = F.cosine_similarity(
            features_a,
            features_b,
            dim=-1,
        )

        return (
            1.0 - similarity
        ).mean()


# ============================================================
# LATENT REGULARIZATION
# ============================================================

class MultimodalLatentRegularizer(
    nn.Module
):
    """
    Simple latent regularization.

    Penalizes excessive latent magnitude
    and low batch-level feature variance.
    """

    def __init__(
        self,
        variance_target: float = 0.1,
    ) -> None:
        super().__init__()

        self.variance_target = (
            variance_target
        )

    def forward(
        self,
        latent: Tensor,
    ) -> Tensor:
        magnitude_penalty = (
            latent.square().mean()
        )

        if latent.shape[0] < 2:
            return magnitude_penalty

        variance = latent.var(
            dim=0,
            unbiased=False,
        )

        variance_penalty = (
            F.relu(
                self.variance_target
                - variance
            ).mean()
        )

        return (
            0.01 * magnitude_penalty
            + variance_penalty
        )


# ============================================================
# FACTORY
# ============================================================

def create_multimodal_encoder(
    config: (
        MultimodalEncoderConfig | None
    ) = None,
) -> MultimodalEncoder:
    return MultimodalEncoder(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - MULTIMODAL ENCODER"
    )
    print("=" * 65)

    torch.manual_seed(
        42
    )

    config = MultimodalEncoderConfig(
        image_dim=512,
        video_dim=512,
        motion_dim=128,
        spatial_dim=128,
        audio_dim=256,
        text_dim=768,
        sensor_dim=64,
        hidden_dim=256,
        latent_dim=128,
        attention_heads=8,
        transformer_layers=3,
    )

    encoder = MultimodalEncoder(
        config
    )

    batch_size = 4

    inputs = MultimodalInput(
        image=torch.randn(
            batch_size,
            512,
        ),
        video=torch.randn(
            batch_size,
            12,
            512,
        ),
        motion=torch.randn(
            batch_size,
            128,
        ),
        spatial=torch.randn(
            batch_size,
            128,
        ),
        audio=torch.randn(
            batch_size,
            256,
        ),
        text=torch.randn(
            batch_size,
            768,
        ),
        sensor=torch.randn(
            batch_size,
            64,
        ),
        video_mask=torch.ones(
            batch_size,
            12,
            dtype=torch.bool,
        ),
        modality_mask={
            "audio": torch.tensor(
                [
                    True,
                    True,
                    False,
                    True,
                ]
            ),
            "text": torch.tensor(
                [
                    True,
                    False,
                    True,
                    True,
                ]
            ),
        },
    )

    output = encoder(
        inputs
    )

    print(
        "Latent shape:",
        output.latent.shape,
    )

    print(
        "Fused features:",
        output.fused_features.shape,
    )

    print(
        "Modality weights:",
        output.modality_weights.shape,
    )

    print(
        "Availability mask:",
        output.availability_mask,
    )

    print(
        "Latent vector example:",
        output.latent[0, :8],
    )

    # Demonstrate backpropagation.
    target = torch.randn_like(
        output.latent
    )

    loss = F.mse_loss(
        output.latent,
        target,
    )

    loss.backward()

    print(
        "Training loss:",
        float(loss.item()),
    )

    print(
        "\nMultimodal encoder demonstration completed"
    )


if __name__ == "__main__":
    main()
