
"""
NeuroWorld AI
File: neuroworld/latent/state_representation.py

Neural World State Representation Engine

Features:
- Multimodal state representation
- Visual, spatial, motion and sensor fusion
- Object-level attention
- Temporal sequence encoding
- Action-conditioned state representation
- Latent state projection
- Uncertainty estimation
- State similarity and distance
- Optional integration with state compression
- CPU / GPU compatibility

Dependencies:
    pip install torch
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class StateRepresentationConfig:
    visual_dim: int = 512
    spatial_dim: int = 128
    motion_dim: int = 128
    sensor_dim: int = 64
    action_dim: int = 32
    object_dim: int = 256

    hidden_dim: int = 256
    latent_dim: int = 128

    num_attention_heads: int = 8
    num_temporal_layers: int = 2

    dropout: float = 0.1
    use_temporal_encoding: bool = True
    use_action_conditioning: bool = True
    normalize_output: bool = False

    min_logvar: float = -10.0
    max_logvar: float = 5.0


@dataclass
class WorldState:
    """
    Structured neural world-state output.
    """

    latent: Tensor
    mean: Tensor
    logvar: Tensor

    fused_features: Tensor

    object_features: Optional[Tensor] = None
    object_attention: Optional[Tensor] = None

    compressed: Optional[Tensor] = None

    @property
    def uncertainty(self) -> Tensor:
        """
        Average estimated latent variance.

        This is model-predicted uncertainty,
        not a calibrated confidence score.
        """
        return self.logvar.exp().mean(dim=-1)


class FeatureProjector(nn.Module):
    """
    Project one modality into a shared feature space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class MultimodalFusion(nn.Module):
    """
    Fuse available modalities using learned gates.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int = 4,
    ):
        super().__init__()

        self.gate = nn.Linear(
            hidden_dim,
            1,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.num_modalities = num_modalities

    def forward(
        self,
        features: Dict[str, Tensor],
    ) -> Tensor:

        if not features:
            raise ValueError(
                "At least one modality is required"
            )

        tensors = list(features.values())

        reference_shape = tensors[0].shape

        for tensor in tensors:
            if tensor.shape != reference_shape:
                raise ValueError(
                    "Projected modalities must share shape"
                )

        stacked = torch.stack(
            tensors,
            dim=-2,
        )

        scores = self.gate(stacked)

        weights = torch.softmax(
            scores,
            dim=-2,
        )

        fused = (
            stacked * weights
        ).sum(dim=-2)

        return self.output(fused)


class ObjectAttentionEncoder(nn.Module):
    """
    Encode sets of object features using self-attention.

    Input:
        [B, N, object_dim]
        [B, T, N, object_dim]

    Optional object_mask:
        True means a valid object.
    """

    def __init__(
        self,
        object_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()

        self.projector = FeatureProjector(
            object_dim,
            hidden_dim,
            dropout,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        objects: Tensor,
        object_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:

        if objects.ndim not in (3, 4):
            raise ValueError(
                "Objects must be [B,N,D] or [B,T,N,D]"
            )

        if objects.shape[-2] == 0:
            raise ValueError(
                "Object dimension N cannot be zero"
            )

        original_shape = objects.shape[:-2]
        num_objects = objects.shape[-2]

        flat = objects.reshape(
            -1,
            num_objects,
            objects.shape[-1],
        )

        valid_mask = None

        if object_mask is not None:
            if object_mask.shape != objects.shape[:-1]:
                raise ValueError(
                    "Object mask shape mismatch"
                )

            valid_mask = object_mask.reshape(
                -1,
                num_objects,
            ).bool()

            if not valid_mask.any(dim=-1).all():
                raise ValueError(
                    "Each state requires at least one valid object"
                )

        projected = self.projector(flat)

        key_padding_mask = (
            ~valid_mask
            if valid_mask is not None
            else None
        )

        attended, weights = self.attention(
            projected,
            projected,
            projected,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        attended = self.norm(
            projected + attended
        )

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)

            pooled = (
                attended * mask
            ).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1)

        else:
            pooled = attended.mean(dim=1)

        pooled = pooled.reshape(
            *original_shape,
            attended.shape[-1],
        )

        weights = weights.reshape(
            *original_shape,
            num_objects,
            num_objects,
        )

        return pooled, weights


class TemporalStateEncoder(nn.Module):
    """
    Transformer encoder for sequential world states.

    Uses sinusoidal positional encodings to retain
    temporal ordering information.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    @staticmethod
    def positional_encoding(
        length: int,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:

        positions = torch.arange(
            length,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

        indices = torch.arange(
            0,
            dimension,
            2,
            device=device,
            dtype=torch.float32,
        )

        frequencies = torch.exp(
            -torch.log(
                torch.tensor(
                    10000.0,
                    device=device,
                )
            ) * indices / dimension
        )

        angles = positions * frequencies

        encoding = torch.zeros(
            length,
            dimension,
            device=device,
            dtype=torch.float32,
        )

        encoding[:, 0::2] = torch.sin(angles)

        odd_width = encoding[:, 1::2].shape[-1]

        encoding[:, 1::2] = torch.cos(
            angles[:, :odd_width]
        )

        return encoding.to(dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Temporal input must be [B,T,D]"
            )

        positional = self.positional_encoding(
            x.shape[1],
            x.shape[2],
            x.device,
            x.dtype,
        )

        x = x + positional.unsqueeze(0)

        return self.encoder(x)


class StateRepresentation(nn.Module):
    """
    Unified multimodal neural world-state representation.

    Accepted input shapes:

    Single state:
        visual  [B, visual_dim]
        spatial [B, spatial_dim]
        motion  [B, motion_dim]
        sensors [B, sensor_dim]
        actions [B, action_dim]
        objects [B, N, object_dim]

    Sequence:
        visual  [B, T, visual_dim]
        spatial [B, T, spatial_dim]
        motion  [B, T, motion_dim]
        sensors [B, T, sensor_dim]
        actions [B, T, action_dim]
        objects [B, T, N, object_dim]
    """

    def __init__(
        self,
        config: Optional[
            StateRepresentationConfig
        ] = None,
        compressor: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.config = (
            config or StateRepresentationConfig()
        )

        cfg = self.config

        if cfg.hidden_dim % cfg.num_attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by "
                "num_attention_heads"
            )

        self.projectors = nn.ModuleDict({
            "visual": FeatureProjector(
                cfg.visual_dim,
                cfg.hidden_dim,
                cfg.dropout,
            ),
            "spatial": FeatureProjector(
                cfg.spatial_dim,
                cfg.hidden_dim,
                cfg.dropout,
            ),
            "motion": FeatureProjector(
                cfg.motion_dim,
                cfg.hidden_dim,
                cfg.dropout,
            ),
            "sensors": FeatureProjector(
                cfg.sensor_dim,
                cfg.hidden_dim,
                cfg.dropout,
            ),
        })

        self.fusion = MultimodalFusion(
            cfg.hidden_dim
        )

        self.object_encoder = ObjectAttentionEncoder(
            cfg.object_dim,
            cfg.hidden_dim,
            cfg.num_attention_heads,
            cfg.dropout,
        )

        self.object_fusion = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim * 2,
                cfg.hidden_dim,
            ),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )

        self.action_projector = FeatureProjector(
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.dropout,
        )

        self.action_fusion = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim * 2,
                cfg.hidden_dim,
            ),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )

        self.temporal_encoder = TemporalStateEncoder(
            cfg.hidden_dim,
            cfg.num_attention_heads,
            cfg.num_temporal_layers,
            cfg.dropout,
        )

        self.mean_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim,
        )

        self.logvar_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim,
        )

        self.compressor = compressor

    def forward(
        self,
        visual: Optional[Tensor] = None,
        spatial: Optional[Tensor] = None,
        motion: Optional[Tensor] = None,
        sensors: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        objects: Optional[Tensor] = None,
        object_mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> WorldState:

        modalities = {
            "visual": visual,
            "spatial": spatial,
            "motion": motion,
            "sensors": sensors,
        }

        projected = {}

        for name, value in modalities.items():
            if value is not None:
                projected[name] = (
                    self.projectors[name](value)
                )

        object_features = None
        object_attention = None

        if objects is not None:
            object_features, object_attention = (
                self.object_encoder(
                    objects,
                    object_mask,
                )
            )

        if not projected and object_features is None:
            raise ValueError(
                "At least one input modality is required"
            )

        if projected:
            fused = self.fusion(projected)
        else:
            fused = object_features

        if object_features is not None and projected:
            if fused.shape != object_features.shape:
                raise ValueError(
                    "Object and modality state shapes differ"
                )

            fused = self.object_fusion(
                torch.cat(
                    [fused, object_features],
                    dim=-1,
                )
            )

        if (
            actions is not None
            and self.config.use_action_conditioning
        ):
            action_features = (
                self.action_projector(actions)
            )

            if action_features.shape != fused.shape:
                raise ValueError(
                    "Action and state shapes must match"
                )

            fused = self.action_fusion(
                torch.cat(
                    [fused, action_features],
                    dim=-1,
                )
            )

        if (
            fused.ndim == 3
            and self.config.use_temporal_encoding
        ):
            fused = self.temporal_encoder(fused)

        mean = self.mean_head(fused)

        logvar = self.logvar_head(fused).clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )

        if deterministic:
            latent = mean
        else:
            std = torch.exp(0.5 * logvar)
            latent = mean + (
                torch.randn_like(std) * std
            )

        if self.config.normalize_output:
            latent = F.normalize(
                latent,
                p=2,
                dim=-1,
            )

        compressed = None

        if self.compressor is not None:
            compressed = self.compressor.compress(
                latent
            )

        return WorldState(
            latent=latent,
            mean=mean,
            logvar=logvar,
            fused_features=fused,
            object_features=object_features,
            object_attention=object_attention,
            compressed=compressed,
        )

    @staticmethod
    def state_similarity(
        state_a: Tensor,
        state_b: Tensor,
    ) -> Tensor:

        return F.cosine_similarity(
            state_a,
            state_b,
            dim=-1,
        )

    @staticmethod
    def state_distance(
        state_a: Tensor,
        state_b: Tensor,
    ) -> Tensor:

        return torch.linalg.vector_norm(
            state_a - state_b,
            dim=-1,
        )

    @staticmethod
    def state_transition(
        previous_state: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Latent state displacement.
        """

        return next_state - previous_state

    @staticmethod
    def kl_loss(
        state: WorldState,
    ) -> Tensor:
        """
        KL divergence against N(0, I).
        """

        return -0.5 * (
            1
            + state.logvar
            - state.mean.pow(2)
            - state.logvar.exp()
        ).sum(dim=-1).mean()


if __name__ == "__main__":

    torch.manual_seed(42)

    config = StateRepresentationConfig(
        visual_dim=512,
        spatial_dim=128,
        motion_dim=128,
        sensor_dim=64,
        action_dim=32,
        object_dim=256,
        hidden_dim=256,
        latent_dim=128,
    )

    model = StateRepresentation(config)

    batch_size = 4
    sequence_length = 10
    num_objects = 6

    visual = torch.randn(
        batch_size,
        sequence_length,
        512,
    )

    spatial = torch.randn(
        batch_size,
        sequence_length,
        128,
    )

    motion = torch.randn(
        batch_size,
        sequence_length,
        128,
    )

    sensors = torch.randn(
        batch_size,
        sequence_length,
        64,
    )

    actions = torch.randn(
        batch_size,
        sequence_length,
        32,
    )

    objects = torch.randn(
        batch_size,
        sequence_length,
        num_objects,
        256,
    )

    output = model(
        visual=visual,
        spatial=spatial,
        motion=motion,
        sensors=sensors,
        actions=actions,
        objects=objects,
    )

    print("Latent:", output.latent.shape)
    print("Mean:", output.mean.shape)
    print("Logvar:", output.logvar.shape)
    print("Fused:", output.fused_features.shape)

    print(
        "Uncertainty:",
        output.uncertainty.shape,
    )

    loss = StateRepresentation.kl_loss(output)

    loss.backward()

    print("KL loss:", loss.item())
    print("Backward pass successful")
