
"""
NeuroWorld AI
File: neuroworld/perception/scene_encoder.py

Neural Scene Understanding & Spatial Representation Encoder

Features:
    - Transformer-based scene encoding
    - Visual feature projection
    - Object embedding integration
    - Spatial position encoding
    - Motion feature integration
    - Object confidence masking
    - Multi-head self-attention
    - Global scene representation
    - Object-level contextual embeddings
    - Pairwise spatial relationship features
    - Temporal scene sequence encoding
    - Learnable scene token
    - Scene similarity calculation
    - World-model latent projection
    - Batched inference
    - Optional attention-map extraction
    - Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.perception.scene_encoder

Expected inputs:
    image_features:  [B, image_dim]
    object_features: [B, N, object_dim]
    positions:       [B, N, spatial_dim]
    motion_features: [B, N, motion_dim]
    object_mask:     [B, N]

The object mask uses True for valid objects.

Outputs:
    scene_embedding:   [B, scene_dim]
    object_embeddings: [B, N, scene_dim]
    spatial_relations: [B, N, N, relation_dim]
    world_latent:      [B, latent_dim]

Notes:
    This is a neural representation model.
    It does not automatically detect objects.
    Object features and positions must come
    from upstream perception components.
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

class SceneEncoderError(Exception):
    """Base scene encoder exception."""


class SceneValidationError(SceneEncoderError):
    """Invalid scene input or configuration."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SceneEncoderConfig:
    image_dim: int = 512
    object_dim: int = 256
    motion_dim: int = 128

    spatial_dim: int = 3

    scene_dim: int = 512
    latent_dim: int = 256

    relation_dim: int = 128

    num_attention_heads: int = 8
    num_transformer_layers: int = 4

    feedforward_dim: int = 2048

    max_objects: int = 64
    max_temporal_steps: int = 32

    dropout: float = 0.1

    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        positive_fields = (
            "image_dim",
            "object_dim",
            "motion_dim",
            "spatial_dim",
            "scene_dim",
            "latent_dim",
            "relation_dim",
            "num_attention_heads",
            "num_transformer_layers",
            "feedforward_dim",
            "max_objects",
            "max_temporal_steps",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if (
            self.scene_dim
            % self.num_attention_heads
            != 0
        ):
            raise ValueError(
                "scene_dim must be divisible "
                "by num_attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )


# ============================================================
# INPUT DATA STRUCTURE
# ============================================================

@dataclass
class SceneInput:
    image_features: Tensor

    object_features: Tensor
    positions: Tensor

    motion_features: Tensor | None = None
    object_mask: Tensor | None = None

    def validate(
        self,
        config: SceneEncoderConfig,
    ) -> None:
        if (
            self.image_features.ndim != 2
            or self.image_features.shape[1]
            != config.image_dim
        ):
            raise SceneValidationError(
                "image_features must be [B,image_dim]"
            )

        batch_size = (
            self.image_features.shape[0]
        )

        if (
            self.object_features.ndim != 3
            or self.object_features.shape[0]
            != batch_size
            or self.object_features.shape[2]
            != config.object_dim
        ):
            raise SceneValidationError(
                "object_features must be [B,N,object_dim]"
            )

        object_count = (
            self.object_features.shape[1]
        )

        if object_count > config.max_objects:
            raise SceneValidationError(
                "Too many objects"
            )

        if self.positions.shape != (
            batch_size,
            object_count,
            config.spatial_dim,
        ):
            raise SceneValidationError(
                "positions must be [B,N,spatial_dim]"
            )

        if (
            self.motion_features is not None
            and self.motion_features.shape
            != (
                batch_size,
                object_count,
                config.motion_dim,
            )
        ):
            raise SceneValidationError(
                "motion_features must be [B,N,motion_dim]"
            )

        if (
            self.object_mask is not None
            and self.object_mask.shape
            != (
                batch_size,
                object_count,
            )
        ):
            raise SceneValidationError(
                "object_mask must be [B,N]"
            )

        tensors = [
            self.image_features,
            self.object_features,
            self.positions,
        ]

        if self.motion_features is not None:
            tensors.append(
                self.motion_features
            )

        for tensor in tensors:
            if not torch.isfinite(
                tensor
            ).all():
                raise SceneValidationError(
                    "Scene input contains non-finite values"
                )


# ============================================================
# SCENE OUTPUT
# ============================================================

@dataclass
class SceneEncoderOutput:
    scene_embedding: Tensor

    object_embeddings: Tensor

    spatial_relations: Tensor

    world_latent: Tensor

    object_mask: Tensor

    attention_maps: Tensor | None = None

    def to_numpy(self) -> dict[str, Any]:
        def convert(
            tensor: Tensor | None,
        ) -> np.ndarray | None:
            if tensor is None:
                return None

            return (
                tensor.detach()
                .cpu()
                .numpy()
            )

        return {
            "scene_embedding": convert(
                self.scene_embedding
            ),
            "object_embeddings": convert(
                self.object_embeddings
            ),
            "spatial_relations": convert(
                self.spatial_relations
            ),
            "world_latent": convert(
                self.world_latent
            ),
            "object_mask": convert(
                self.object_mask
            ),
            "attention_maps": convert(
                self.attention_maps
            ),
        }


# ============================================================
# SPATIAL POSITION ENCODER
# ============================================================

class SpatialPositionEncoder(nn.Module):
    """
    Converts object coordinates into
    learnable spatial embeddings.

    Input:
        [B,N,D]

    Output:
        [B,N,scene_dim]
    """

    def __init__(
        self,
        config: SceneEncoderConfig,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.spatial_dim,
                config.scene_dim,
            ),
            nn.LayerNorm(
                config.scene_dim
            ),
            nn.GELU(),
            nn.Linear(
                config.scene_dim,
                config.scene_dim,
            ),
        )

    def forward(
        self,
        positions: Tensor,
    ) -> Tensor:
        return self.network(
            positions
        )


# ============================================================
# OBJECT FEATURE ENCODER
# ============================================================

class ObjectFeatureEncoder(nn.Module):
    """
    Projects object appearance and motion
    features into the shared scene space.
    """

    def __init__(
        self,
        config: SceneEncoderConfig,
    ) -> None:
        super().__init__()

        self.object_projection = nn.Sequential(
            nn.Linear(
                config.object_dim,
                config.scene_dim,
            ),
            nn.LayerNorm(
                config.scene_dim
            ),
            nn.GELU(),
        )

        self.motion_projection = nn.Sequential(
            nn.Linear(
                config.motion_dim,
                config.scene_dim,
            ),
            nn.LayerNorm(
                config.scene_dim
            ),
            nn.GELU(),
        )

        self.position_encoder = (
            SpatialPositionEncoder(
                config
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                config.scene_dim * 3,
                config.scene_dim,
            ),
            nn.LayerNorm(
                config.scene_dim
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
        )

    def forward(
        self,
        object_features: Tensor,
        positions: Tensor,
        motion_features: Tensor | None,
    ) -> Tensor:
        object_embedding = (
            self.object_projection(
                object_features
            )
        )

        position_embedding = (
            self.position_encoder(
                positions
            )
        )

        if motion_features is None:
            motion_embedding = (
                torch.zeros_like(
                    object_embedding
                )
            )
        else:
            motion_embedding = (
                self.motion_projection(
                    motion_features
                )
            )

        combined = torch.cat(
            [
                object_embedding,
                position_embedding,
                motion_embedding,
            ],
            dim=-1,
        )

        return self.fusion(
            combined
        )


# ============================================================
# PAIRWISE SPATIAL RELATION ENCODER
# ============================================================

class SpatialRelationEncoder(nn.Module):
    """
    Computes pairwise spatial features.

    For objects i and j:
        relative_position = p_j - p_i
        distance = ||p_j-p_i||

    Input:
        positions [B,N,D]

    Output:
        relations [B,N,N,relation_dim]
    """

    def __init__(
        self,
        config: SceneEncoderConfig,
    ) -> None:
        super().__init__()

        self.spatial_dim = (
            config.spatial_dim
        )

        self.network = nn.Sequential(
            nn.Linear(
                config.spatial_dim + 1,
                config.relation_dim,
            ),
            nn.GELU(),
            nn.Linear(
                config.relation_dim,
                config.relation_dim,
            ),
        )

    def forward(
        self,
        positions: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        relative = (
            positions[:, None, :, :]
            - positions[:, :, None, :]
        )

        distance = (
            torch.linalg.vector_norm(
                relative,
                dim=-1,
                keepdim=True,
            )
        )

        features = torch.cat(
            [
                relative,
                distance,
            ],
            dim=-1,
        )

        relations = self.network(
            features
        )

        pair_mask = (
            object_mask[:, :, None]
            & object_mask[:, None, :]
        )

        relations = (
            relations
            * pair_mask.unsqueeze(-1)
        )

        return relations


# ============================================================
# TRANSFORMER SCENE LAYER
# ============================================================

class SceneTransformerLayer(nn.Module):
    """
    Self-attention over:
        [scene_token, object_tokens...]

    Optional attention extraction.
    """

    def __init__(
        self,
        config: SceneEncoderConfig,
    ) -> None:
        super().__init__()

        self.attention = (
            nn.MultiheadAttention(
                embed_dim=config.scene_dim,
                num_heads=(
                    config.num_attention_heads
                ),
                dropout=config.dropout,
                batch_first=True,
            )
        )

        self.norm1 = nn.LayerNorm(
            config.scene_dim
        )

        self.norm2 = nn.LayerNorm(
            config.scene_dim
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                config.scene_dim,
                config.feedforward_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.feedforward_dim,
                config.scene_dim,
            ),
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        tokens: Tensor,
        padding_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[
        Tensor,
        Tensor | None,
    ]:
        normalized = self.norm1(
            tokens
        )

        attended, weights = (
            self.attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        )

        tokens = (
            tokens
            + self.dropout(
                attended
            )
        )

        tokens = (
            tokens
            + self.dropout(
                self.feedforward(
                    self.norm2(
                        tokens
                    )
                )
            )
        )

        return (
            tokens,
            weights,
        )


# ============================================================
# MAIN SCENE ENCODER
# ============================================================

class SceneEncoder(nn.Module):
    """
    Transformer-based scene representation.

    Pipeline:
        Image features
              |
        Image projection
              |
        Learnable scene token
              |
        Object + position + motion
              |
        Transformer attention
              |
        Global scene embedding
              |
        World-model latent

    Object padding is excluded from
    attention keys and final embeddings.
    """

    def __init__(
        self,
        config: SceneEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or SceneEncoderConfig()
        )

        self.image_projection = nn.Sequential(
            nn.Linear(
                self.config.image_dim,
                self.config.scene_dim,
            ),
            nn.LayerNorm(
                self.config.scene_dim
            ),
            nn.GELU(),
        )

        self.object_encoder = (
            ObjectFeatureEncoder(
                self.config
            )
        )

        self.relation_encoder = (
            SpatialRelationEncoder(
                self.config
            )
        )

        self.scene_token = nn.Parameter(
            torch.randn(
                1,
                1,
                self.config.scene_dim,
            ) * 0.02
        )

        self.layers = nn.ModuleList(
            [
                SceneTransformerLayer(
                    self.config
                )
                for _ in range(
                    self.config.num_transformer_layers
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            self.config.scene_dim
        )

        self.scene_head = nn.Sequential(
            nn.Linear(
                self.config.scene_dim,
                self.config.scene_dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.config.scene_dim,
                self.config.scene_dim,
            ),
        )

        self.latent_head = nn.Sequential(
            nn.Linear(
                self.config.scene_dim,
                self.config.latent_dim,
            ),
            nn.LayerNorm(
                self.config.latent_dim
            ),
        )

    def forward(
        self,
        scene: SceneInput,
        *,
        return_attention: bool = False,
    ) -> SceneEncoderOutput:
        scene.validate(
            self.config
        )

        batch_size = (
            scene.image_features.shape[0]
        )

        object_count = (
            scene.object_features.shape[1]
        )

        if scene.object_mask is None:
            object_mask = torch.ones(
                batch_size,
                object_count,
                dtype=torch.bool,
                device=(
                    scene.object_features.device
                ),
            )
        else:
            object_mask = (
                scene.object_mask.to(
                    dtype=torch.bool,
                    device=(
                        scene.object_features.device
                    ),
                )
            )

        image_embedding = (
            self.image_projection(
                scene.image_features
            )
        )

        object_embeddings = (
            self.object_encoder(
                scene.object_features,
                scene.positions,
                scene.motion_features,
            )
        )

        object_embeddings = (
            object_embeddings
            * object_mask.unsqueeze(-1)
        )

        learned_scene_token = (
            self.scene_token.expand(
                batch_size,
                -1,
                -1,
            )
        )

        scene_token = (
            learned_scene_token
            + image_embedding.unsqueeze(1)
        )

        tokens = torch.cat(
            [
                scene_token,
                object_embeddings,
            ],
            dim=1,
        )

        scene_valid = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=object_mask.device,
        )

        valid_tokens = torch.cat(
            [
                scene_valid,
                object_mask,
            ],
            dim=1,
        )

        padding_mask = (
            ~valid_tokens
        )

        attention_maps = []

        for layer in self.layers:
            tokens, attention = layer(
                tokens,
                padding_mask,
                return_attention=(
                    return_attention
                ),
            )

            if attention is not None:
                attention_maps.append(
                    attention
                )

        tokens = self.final_norm(
            tokens
        )

        global_embedding = (
            self.scene_head(
                tokens[:, 0]
            )
        )

        contextual_objects = (
            tokens[:, 1:]
        )

        contextual_objects = (
            contextual_objects
            * object_mask.unsqueeze(-1)
        )

        if self.config.normalize_embeddings:
            global_embedding = F.normalize(
                global_embedding,
                dim=-1,
            )

            contextual_objects = F.normalize(
                contextual_objects,
                dim=-1,
            )

            contextual_objects = (
                contextual_objects
                * object_mask.unsqueeze(-1)
            )

        world_latent = self.latent_head(
            global_embedding
        )

        relations = (
            self.relation_encoder(
                scene.positions,
                object_mask,
            )
        )

        stacked_attention = None

        if attention_maps:
            stacked_attention = torch.stack(
                attention_maps,
                dim=1,
            )

        return SceneEncoderOutput(
            scene_embedding=(
                global_embedding
            ),
            object_embeddings=(
                contextual_objects
            ),
            spatial_relations=relations,
            world_latent=world_latent,
            object_mask=object_mask,
            attention_maps=(
                stacked_attention
            ),
        )


# ============================================================
# TEMPORAL SCENE ENCODER
# ============================================================

class TemporalSceneEncoder(nn.Module):
    """
    Encodes sequences of scene embeddings.

    Input:
        [B,T,scene_dim]

    Output:
        [B,latent_dim]

    Uses a causal Transformer so each
    timestep attends only to the past
    and present.

    The final valid timestep is pooled.
    """

    def __init__(
        self,
        config: SceneEncoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                config.max_temporal_steps,
                config.scene_dim,
            ) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.scene_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=(
                config.feedforward_dim
            ),
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=(
                config.num_transformer_layers
            ),
            enable_nested_tensor=False,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(
                config.scene_dim
            ),
            nn.Linear(
                config.scene_dim,
                config.latent_dim,
            ),
        )

    def forward(
        self,
        scene_embeddings: Tensor,
        temporal_mask: Tensor | None = None,
    ) -> Tensor:
        if (
            scene_embeddings.ndim != 3
            or scene_embeddings.shape[-1]
            != self.config.scene_dim
        ):
            raise SceneValidationError(
                "scene_embeddings must be [B,T,scene_dim]"
            )

        batch_size, steps, _ = (
            scene_embeddings.shape
        )

        if (
            steps < 1
            or steps
            > self.config.max_temporal_steps
        ):
            raise SceneValidationError(
                "Invalid temporal sequence length"
            )

        if temporal_mask is None:
            temporal_mask = torch.ones(
                batch_size,
                steps,
                dtype=torch.bool,
                device=scene_embeddings.device,
            )

        if temporal_mask.shape != (
            batch_size,
            steps,
        ):
            raise SceneValidationError(
                "temporal_mask must be [B,T]"
            )

        temporal_mask = temporal_mask.bool()

        if not temporal_mask.any(dim=1).all():
            raise SceneValidationError(
                "Each sequence needs a valid timestep"
            )

        positions = (
            self.position_embedding[
                :, :steps
            ]
        )

        tokens = (
            scene_embeddings
            + positions
        )

        causal_mask = torch.triu(
            torch.ones(
                steps,
                steps,
                dtype=torch.bool,
                device=scene_embeddings.device,
            ),
            diagonal=1,
        )

        encoded = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=(
                ~temporal_mask
            ),
        )

        time_indices = torch.arange(
            steps,
            device=scene_embeddings.device,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
        )

        last_valid = torch.where(
            temporal_mask,
            time_indices,
            -1,
        ).max(dim=1).values

        batch_indices = torch.arange(
            batch_size,
            device=scene_embeddings.device,
        )

        pooled = encoded[
            batch_indices,
            last_valid,
        ]

        return self.output_head(
            pooled
        )


# ============================================================
# SCENE SIMILARITY
# ============================================================

class SceneSimilarity:
    """
    Cosine similarity between scene
    representations.
    """

    @staticmethod
    def cosine(
        first: Tensor,
        second: Tensor,
    ) -> Tensor:
        if first.shape != second.shape:
            raise SceneValidationError(
                "Scene embedding shapes must match"
            )

        return F.cosine_similarity(
            first,
            second,
            dim=-1,
        )

    @staticmethod
    def pairwise(
        embeddings: Tensor,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise SceneValidationError(
                "Embeddings must be [B,D]"
            )

        normalized = F.normalize(
            embeddings,
            dim=-1,
        )

        return (
            normalized
            @ normalized.transpose(
                0,
                1,
            )
        )


# ============================================================
# SCENE ENCODER ADAPTER
# ============================================================

class SceneWorldModelAdapter:
    """
    Adapter exposing scene-to-latent encoding.

    Used by:
        world_model.py
        state_representation.py
        imagination_engine.py
        planning_agent.py
    """

    def __init__(
        self,
        encoder: SceneEncoder,
    ) -> None:
        self.encoder = encoder

    @torch.no_grad()
    def encode_scene(
        self,
        image_features: Tensor,
        object_features: Tensor,
        positions: Tensor,
        motion_features: Tensor | None = None,
        object_mask: Tensor | None = None,
    ) -> Tensor:
        scene = SceneInput(
            image_features=image_features,
            object_features=object_features,
            positions=positions,
            motion_features=motion_features,
            object_mask=object_mask,
        )

        output = self.encoder(
            scene
        )

        return output.world_latent


# ============================================================
# FACTORY
# ============================================================

def create_scene_encoder(
    config: SceneEncoderConfig | None = None,
) -> SceneEncoder:
    return SceneEncoder(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - SCENE ENCODER"
    )
    print("=" * 65)

    config = SceneEncoderConfig(
        image_dim=512,
        object_dim=256,
        motion_dim=128,
        spatial_dim=3,
        scene_dim=512,
        latent_dim=256,
        relation_dim=128,
        max_objects=16,
        num_transformer_layers=3,
    )

    encoder = SceneEncoder(
        config
    )

    encoder.eval()

    batch_size = 2
    object_count = 5

    image_features = torch.randn(
        batch_size,
        config.image_dim,
    )

    object_features = torch.randn(
        batch_size,
        object_count,
        config.object_dim,
    )

    positions = torch.randn(
        batch_size,
        object_count,
        config.spatial_dim,
    )

    motion_features = torch.randn(
        batch_size,
        object_count,
        config.motion_dim,
    )

    object_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )

    scene = SceneInput(
        image_features=image_features,
        object_features=object_features,
        positions=positions,
        motion_features=motion_features,
        object_mask=object_mask,
    )

    with torch.no_grad():
        output = encoder(
            scene,
            return_attention=True,
        )

    print(
        "Scene embedding:",
        output.scene_embedding.shape,
    )

    print(
        "Object embeddings:",
        output.object_embeddings.shape,
    )

    print(
        "Spatial relations:",
        output.spatial_relations.shape,
    )

    print(
        "World latent:",
        output.world_latent.shape,
    )

    if output.attention_maps is not None:
        print(
            "Attention maps:",
            output.attention_maps.shape,
        )

    temporal_encoder = (
        TemporalSceneEncoder(
            config
        )
    )

    temporal_encoder.eval()

    sequence = torch.randn(
        batch_size,
        8,
        config.scene_dim,
    )

    with torch.no_grad():
        temporal_latent = (
            temporal_encoder(
                sequence
            )
        )

    print(
        "Temporal latent:",
        temporal_latent.shape,
    )

    similarity = SceneSimilarity.cosine(
        output.scene_embedding[0:1],
        output.scene_embedding[1:2],
    )

    print(
        "Scene similarity:",
        similarity.item(),
    )

    print(
        "\nScene encoder demonstration completed"
    )


if __name__ == "__main__":
    main()
