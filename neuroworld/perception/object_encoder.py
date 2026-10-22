
"""
NeuroWorld AI
File: neuroworld/perception/object_encoder.py

Object-Centric Visual Representation Encoder

Features:
    - Object visual feature encoding
    - Bounding-box geometry encoding
    - Position and velocity encoding
    - Object class embeddings
    - Detection confidence encoding
    - Transformer-based object interaction
    - Object attention masks
    - Pairwise spatial relation features
    - Scene-level object aggregation
    - Temporal object-state encoding
    - Missing-object masking
    - Batched PyTorch inference
    - Optional image ROI feature extraction
    - World-model integration
    - Object reconstruction head
    - Training-ready differentiable architecture

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.perception.object_encoder

Input:
    visual_features: [B,N,F]
    bounding_boxes:  [B,N,4]
    positions:       [B,N,D]
    velocities:      [B,N,D]
    class_ids:       [B,N]
    confidence:      [B,N]
    object_mask:     [B,N]

Output:
    object_embeddings: [B,N,E]
    scene_embedding:   [B,E]
    spatial_relations: [B,N,N,R]
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
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObjectEncoderConfig:
    visual_feature_dim: int = 256

    spatial_dim: int = 3

    embedding_dim: int = 256
    hidden_dim: int = 512

    num_classes: int = 1000
    max_objects: int = 128

    num_attention_heads: int = 8
    num_transformer_layers: int = 3

    dropout: float = 0.1

    use_velocity: bool = True
    use_class_embeddings: bool = True
    use_confidence: bool = True

    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.visual_feature_dim,
            self.spatial_dim,
            self.embedding_dim,
            self.hidden_dim,
            self.num_classes,
            self.max_objects,
            self.num_attention_heads,
            self.num_transformer_layers,
        )

        if any(value <= 0 for value in dimensions):
            raise ValueError(
                "All dimensions must be positive"
            )

        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if (
            self.embedding_dim
            % self.num_attention_heads != 0
        ):
            raise ValueError(
                "embedding_dim must be divisible "
                "by num_attention_heads"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )


# ============================================================
# INPUT DATA
# ============================================================

@dataclass
class ObjectEncoderInput:
    visual_features: Tensor
    bounding_boxes: Tensor
    positions: Tensor

    velocities: Tensor | None = None
    class_ids: Tensor | None = None
    confidence: Tensor | None = None
    object_mask: Tensor | None = None


@dataclass
class ObjectEncoderOutput:
    object_embeddings: Tensor
    scene_embedding: Tensor

    spatial_relations: Tensor
    object_mask: Tensor

    reconstructed_features: Tensor | None = None

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "object_embeddings": (
                self.object_embeddings
                .detach()
                .cpu()
                .numpy()
            ),
            "scene_embedding": (
                self.scene_embedding
                .detach()
                .cpu()
                .numpy()
            ),
            "spatial_relations": (
                self.spatial_relations
                .detach()
                .cpu()
                .numpy()
            ),
            "object_mask": (
                self.object_mask
                .detach()
                .cpu()
                .numpy()
            ),
        }


# ============================================================
# FEATURE MLP
# ============================================================

class FeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                output_dim,
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
# BOUNDING BOX ENCODER
# ============================================================

class BoundingBoxEncoder(nn.Module):
    """
    Bounding boxes:
        [x1,y1,x2,y2]

    Produces:
        center_x, center_y,
        width, height, area,
        aspect_ratio
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = FeatureMLP(
            input_dim=6,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        boxes: Tensor,
    ) -> Tensor:
        x1, y1, x2, y2 = (
            boxes.unbind(
                dim=-1
            )
        )

        width = (
            x2 - x1
        ).clamp_min(0)

        height = (
            y2 - y1
        ).clamp_min(0)

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        area = (
            width * height
        )

        aspect_ratio = (
            width
            / height.clamp_min(
                1e-6
            )
        )

        features = torch.stack(
            [
                center_x,
                center_y,
                width,
                height,
                area,
                aspect_ratio,
            ],
            dim=-1,
        )

        return self.encoder(
            features
        )


# ============================================================
# SPATIAL STATE ENCODER
# ============================================================

class SpatialStateEncoder(nn.Module):
    """
    Encodes object position and velocity.
    """

    def __init__(
        self,
        config: ObjectEncoderConfig,
    ) -> None:
        super().__init__()

        input_dim = (
            config.spatial_dim
            * (
                2
                if config.use_velocity
                else 1
            )
        )

        self.use_velocity = (
            config.use_velocity
        )

        self.encoder = FeatureMLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor | None,
    ) -> Tensor:
        if self.use_velocity:
            if velocities is None:
                velocities = torch.zeros_like(
                    positions
                )

            features = torch.cat(
                [
                    positions,
                    velocities,
                ],
                dim=-1,
            )
        else:
            features = positions

        return self.encoder(
            features
        )


# ============================================================
# PAIRWISE SPATIAL RELATIONS
# ============================================================

class SpatialRelationEncoder(nn.Module):
    """
    Computes pairwise object relationships.

    For objects i and j:
        relative_position = pos_j - pos_i
        distance = ||relative_position||
        relative_velocity = vel_j - vel_i
        speed_difference = ||relative_velocity||

    Output:
        [B,N,N,E]
    """

    def __init__(
        self,
        config: ObjectEncoderConfig,
    ) -> None:
        super().__init__()

        self.spatial_dim = (
            config.spatial_dim
        )

        input_dim = (
            2 * config.spatial_dim
            + 2
        )

        self.encoder = FeatureMLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor | None,
        object_mask: Tensor,
    ) -> Tensor:
        if velocities is None:
            velocities = torch.zeros_like(
                positions
            )

        relative_position = (
            positions[:, None, :, :]
            - positions[:, :, None, :]
        )

        relative_velocity = (
            velocities[:, None, :, :]
            - velocities[:, :, None, :]
        )

        distance = (
            torch.linalg.vector_norm(
                relative_position,
                dim=-1,
                keepdim=True,
            )
        )

        relative_speed = (
            torch.linalg.vector_norm(
                relative_velocity,
                dim=-1,
                keepdim=True,
            )
        )

        features = torch.cat(
            [
                relative_position,
                relative_velocity,
                distance,
                relative_speed,
            ],
            dim=-1,
        )

        relations = self.encoder(
            features
        )

        pair_mask = (
            object_mask[:, :, None]
            & object_mask[:, None, :]
        )

        return relations * (
            pair_mask.unsqueeze(-1)
        )


# ============================================================
# OBJECT TRANSFORMER
# ============================================================

class ObjectInteractionTransformer(nn.Module):
    """
    Self-attention across detected objects.

    Allows objects to exchange context:
        - relative scene membership
        - semantic context
        - motion interactions
        - multi-object reasoning

    Masked objects do not contribute
    as keys or values.
    """

    def __init__(
        self,
        config: ObjectEncoderConfig,
    ) -> None:
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = (
            nn.TransformerEncoder(
                layer,
                num_layers=(
                    config.num_transformer_layers
                ),
                enable_nested_tensor=False,
            )
        )

    def forward(
        self,
        embeddings: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        # Transformer attention cannot
        # safely operate on batches where
        # every key is masked.
        safe_mask = object_mask.clone()

        empty_rows = ~safe_mask.any(
            dim=1
        )

        if empty_rows.any():
            safe_mask[
                empty_rows,
                0,
            ] = True

        encoded = self.transformer(
            embeddings,
            src_key_padding_mask=(
                ~safe_mask
            ),
        )

        return encoded * (
            object_mask.unsqueeze(-1)
        )


# ============================================================
# SCENE AGGREGATION
# ============================================================

class ObjectSceneAggregator(nn.Module):
    """
    Attention-based scene pooling.

    Object embeddings:
        [B,N,E]

    Scene embedding:
        [B,E]
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.attention = nn.Linear(
            embedding_dim,
            1,
        )

        self.projection = nn.Sequential(
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
        embeddings: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        scores = (
            self.attention(
                embeddings
            ).squeeze(-1)
        )

        safe_mask = object_mask.clone()

        empty_rows = ~safe_mask.any(
            dim=1
        )

        if empty_rows.any():
            safe_mask[
                empty_rows,
                0,
            ] = True

        scores = scores.masked_fill(
            ~safe_mask,
            -1e9,
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        weights = weights * (
            object_mask
        )

        weights = weights / (
            weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-9)
        )

        pooled = torch.sum(
            embeddings
            * weights.unsqueeze(-1),
            dim=1,
        )

        result = self.projection(
            pooled
        )

        # Return an exact zero scene
        # representation when no objects
        # are present.
        return result.masked_fill(
            empty_rows.unsqueeze(-1),
            0.0,
        )


# ============================================================
# MAIN OBJECT ENCODER
# ============================================================

class ObjectEncoder(nn.Module):
    """
    Object-centric multimodal encoder.

    Fuses:
        visual features
        bounding boxes
        positions
        velocities
        class embeddings
        confidence scores

    Then applies:
        Transformer object interaction
        Scene attention pooling
        Optional reconstruction head
    """

    def __init__(
        self,
        config: ObjectEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or ObjectEncoderConfig()
        )

        config = self.config

        self.visual_encoder = FeatureMLP(
            input_dim=config.visual_feature_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )

        self.box_encoder = BoundingBoxEncoder(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        self.spatial_encoder = SpatialStateEncoder(
            config
        )

        if config.use_class_embeddings:
            self.class_embedding = nn.Embedding(
                config.num_classes,
                config.embedding_dim,
            )
        else:
            self.class_embedding = None

        if config.use_confidence:
            self.confidence_encoder = (
                FeatureMLP(
                    input_dim=1,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.embedding_dim,
                    dropout=config.dropout,
                )
            )
        else:
            self.confidence_encoder = None

        self.fusion = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.embedding_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                config.embedding_dim
            ),
        )

        self.object_transformer = (
            ObjectInteractionTransformer(
                config
            )
        )

        self.relation_encoder = (
            SpatialRelationEncoder(
                config
            )
        )

        self.scene_aggregator = (
            ObjectSceneAggregator(
                config.embedding_dim
            )
        )

        self.reconstruction_head = (
            nn.Sequential(
                nn.Linear(
                    config.embedding_dim,
                    config.hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    config.hidden_dim,
                    config.visual_feature_dim,
                ),
            )
        )

    def _validate(
        self,
        data: ObjectEncoderInput,
    ) -> tuple[int, int, Tensor]:
        visual = data.visual_features

        if visual.ndim != 3:
            raise ValueError(
                "visual_features must be [B,N,F]"
            )

        batch_size, object_count, feature_dim = (
            visual.shape
        )

        if object_count == 0:
            raise ValueError(
                "At least one object slot is required"
            )

        if object_count > self.config.max_objects:
            raise ValueError(
                "Object count exceeds max_objects"
            )

        if feature_dim != (
            self.config.visual_feature_dim
        ):
            raise ValueError(
                "Visual feature dimension mismatch"
            )

        if data.bounding_boxes.shape != (
            batch_size,
            object_count,
            4,
        ):
            raise ValueError(
                "bounding_boxes must be [B,N,4]"
            )

        if data.positions.shape != (
            batch_size,
            object_count,
            self.config.spatial_dim,
        ):
            raise ValueError(
                "positions must be [B,N,D]"
            )

        if data.velocities is not None:
            if (
                data.velocities.shape
                != data.positions.shape
            ):
                raise ValueError(
                    "Velocity shape mismatch"
                )

        if data.class_ids is not None:
            if data.class_ids.shape != (
                batch_size,
                object_count,
            ):
                raise ValueError(
                    "class_ids must be [B,N]"
                )

        if data.confidence is not None:
            if data.confidence.shape != (
                batch_size,
                object_count,
            ):
                raise ValueError(
                    "confidence must be [B,N]"
                )

        if data.object_mask is None:
            mask = torch.ones(
                batch_size,
                object_count,
                dtype=torch.bool,
                device=visual.device,
            )
        else:
            if data.object_mask.shape != (
                batch_size,
                object_count,
            ):
                raise ValueError(
                    "object_mask must be [B,N]"
                )

            mask = data.object_mask.to(
                device=visual.device,
                dtype=torch.bool,
            )

        tensors = [
            visual,
            data.bounding_boxes,
            data.positions,
        ]

        if data.velocities is not None:
            tensors.append(
                data.velocities
            )

        if data.confidence is not None:
            tensors.append(
                data.confidence
            )

        for tensor in tensors:
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    "Input contains non-finite values"
                )

        if data.class_ids is not None:
            if (
                (data.class_ids < 0).any()
                or (
                    data.class_ids
                    >= self.config.num_classes
                ).any()
            ):
                raise ValueError(
                    "Class ID out of range"
                )

        return (
            batch_size,
            object_count,
            mask,
        )

    def forward(
        self,
        data: ObjectEncoderInput,
        *,
        reconstruct: bool = False,
    ) -> ObjectEncoderOutput:
        (
            batch_size,
            object_count,
            object_mask,
        ) = self._validate(
            data
        )

        visual_embeddings = (
            self.visual_encoder(
                data.visual_features
            )
        )

        box_embeddings = (
            self.box_encoder(
                data.bounding_boxes
            )
        )

        spatial_embeddings = (
            self.spatial_encoder(
                data.positions,
                data.velocities,
            )
        )

        fused = (
            visual_embeddings
            + box_embeddings
            + spatial_embeddings
        )

        if self.class_embedding is not None:
            if data.class_ids is None:
                class_ids = torch.zeros(
                    batch_size,
                    object_count,
                    dtype=torch.long,
                    device=fused.device,
                )
            else:
                class_ids = data.class_ids.long()

            fused = (
                fused
                + self.class_embedding(
                    class_ids
                )
            )

        if self.confidence_encoder is not None:
            if data.confidence is None:
                confidence = torch.ones(
                    batch_size,
                    object_count,
                    1,
                    device=fused.device,
                    dtype=fused.dtype,
                )
            else:
                confidence = (
                    data.confidence.unsqueeze(
                        -1
                    )
                )

            fused = (
                fused
                + self.confidence_encoder(
                    confidence
                )
            )

        fused = self.fusion(
            fused
        )

        fused = fused * (
            object_mask.unsqueeze(-1)
        )

        object_embeddings = (
            self.object_transformer(
                fused,
                object_mask,
            )
        )

        if self.config.normalize_embeddings:
            object_embeddings = F.normalize(
                object_embeddings,
                p=2,
                dim=-1,
            )

        object_embeddings = (
            object_embeddings
            * object_mask.unsqueeze(-1)
        )

        relations = (
            self.relation_encoder(
                data.positions,
                data.velocities,
                object_mask,
            )
        )

        scene_embedding = (
            self.scene_aggregator(
                object_embeddings,
                object_mask,
            )
        )

        reconstructed = None

        if reconstruct:
            reconstructed = (
                self.reconstruction_head(
                    object_embeddings
                )
            )

            reconstructed = (
                reconstructed
                * object_mask.unsqueeze(-1)
            )

        return ObjectEncoderOutput(
            object_embeddings=(
                object_embeddings
            ),
            scene_embedding=(
                scene_embedding
            ),
            spatial_relations=relations,
            object_mask=object_mask,
            reconstructed_features=(
                reconstructed
            ),
        )


# ============================================================
# TEMPORAL OBJECT ENCODER
# ============================================================

class TemporalObjectEncoder(nn.Module):
    """
    Encodes a tracked object's embedding
    history using a GRU.

    Input:
        [B,T,N,E]

    Mask:
        [B,T,N]

    Output:
        [B,N,E]

    Requires consistent object slot
    identity across frames.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        embeddings: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 4:
            raise ValueError(
                "embeddings must be [B,T,N,E]"
            )

        batch_size, time_steps, object_count, dim = (
            embeddings.shape
        )

        if dim != self.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch"
            )

        if mask is None:
            mask = torch.ones(
                batch_size,
                time_steps,
                object_count,
                dtype=torch.bool,
                device=embeddings.device,
            )

        if mask.shape != (
            batch_size,
            time_steps,
            object_count,
        ):
            raise ValueError(
                "mask must be [B,T,N]"
            )

        # Convert to [B*N,T,E].
        sequences = (
            embeddings
            .permute(0, 2, 1, 3)
            .reshape(
                batch_size * object_count,
                time_steps,
                dim,
            )
        )

        sequence_mask = (
            mask
            .permute(0, 2, 1)
            .reshape(
                batch_size * object_count,
                time_steps,
            )
        )

        sequences = (
            sequences
            * sequence_mask.unsqueeze(-1)
        )

        output, _ = self.gru(
            sequences
        )

        # Select the last valid observation
        # for each tracked object.
        time_indices = torch.arange(
            time_steps,
            device=embeddings.device,
        ).unsqueeze(0)

        valid_indices = torch.where(
            sequence_mask,
            time_indices,
            -1,
        )

        last_indices = valid_indices.max(
            dim=1
        ).values

        valid_objects = (
            last_indices >= 0
        )

        last_indices = last_indices.clamp_min(
            0
        )

        selected = output[
            torch.arange(
                batch_size * object_count,
                device=embeddings.device,
            ),
            last_indices,
        ]

        encoded = self.projection(
            selected
        )

        encoded = encoded * (
            valid_objects.unsqueeze(-1)
        )

        return encoded.reshape(
            batch_size,
            object_count,
            dim,
        )


# ============================================================
# OBJECT RECONSTRUCTION LOSS
# ============================================================

class ObjectReconstructionLoss(nn.Module):
    """
    Masked MSE between reconstructed
    and original visual features.
    """

    def forward(
        self,
        reconstructed: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if reconstructed.shape != target.shape:
            raise ValueError(
                "Reconstruction shape mismatch"
            )

        if mask.shape != target.shape[:2]:
            raise ValueError(
                "Mask shape mismatch"
            )

        squared_error = (
            reconstructed - target
        ).square().mean(
            dim=-1
        )

        weights = mask.to(
            squared_error.dtype
        )

        return (
            (squared_error * weights).sum()
            / weights.sum().clamp_min(1.0)
        )


# ============================================================
# FACTORY
# ============================================================

def create_object_encoder(
    config: ObjectEncoderConfig | None = None,
) -> ObjectEncoder:
    return ObjectEncoder(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - OBJECT ENCODER"
    )
    print("=" * 65)

    torch.manual_seed(
        42
    )

    config = ObjectEncoderConfig(
        visual_feature_dim=128,
        spatial_dim=3,
        embedding_dim=256,
        hidden_dim=512,
        max_objects=32,
        num_attention_heads=8,
        num_transformer_layers=2,
    )

    model = ObjectEncoder(
        config
    )

    batch_size = 2
    object_count = 5

    visual_features = torch.randn(
        batch_size,
        object_count,
        128,
    )

    # Normalized bounding boxes.
    box_start = torch.rand(
        batch_size,
        object_count,
        2,
    ) * 0.5

    box_size = torch.rand(
        batch_size,
        object_count,
        2,
    ) * 0.4

    bounding_boxes = torch.cat(
        [
            box_start,
            box_start + box_size,
        ],
        dim=-1,
    )

    positions = torch.randn(
        batch_size,
        object_count,
        3,
    )

    velocities = torch.randn(
        batch_size,
        object_count,
        3,
    ) * 0.1

    class_ids = torch.randint(
        0,
        config.num_classes,
        (
            batch_size,
            object_count,
        ),
    )

    confidence = torch.rand(
        batch_size,
        object_count,
    )

    object_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    )

    data = ObjectEncoderInput(
        visual_features=visual_features,
        bounding_boxes=bounding_boxes,
        positions=positions,
        velocities=velocities,
        class_ids=class_ids,
        confidence=confidence,
        object_mask=object_mask,
    )

    output = model(
        data,
        reconstruct=True,
    )

    print(
        "Object embeddings:",
        output.object_embeddings.shape,
    )

    print(
        "Scene embedding:",
        output.scene_embedding.shape,
    )

    print(
        "Spatial relations:",
        output.spatial_relations.shape,
    )

    print(
        "Reconstructed features:",
        output.reconstructed_features.shape,
    )

    loss_fn = (
        ObjectReconstructionLoss()
    )

    loss = loss_fn(
        output.reconstructed_features,
        visual_features,
        object_mask,
    )

    print(
        "Reconstruction loss:",
        float(loss.item()),
    )

    loss.backward()

    print(
        "Backward pass successful"
    )

    print(
        "\nObject encoder demonstration completed"
    )


if __name__ == "__main__":
    main()
