
"""
NeuroWorld AI
File: neuroworld/physics/physics_encoder.py

Physics-Aware State & Interaction Encoder

Features:
    - 2D / 3D object physics encoding
    - Position and velocity embeddings
    - Acceleration and force embeddings
    - Mass and radius embeddings
    - Static / dynamic object indicators
    - Gravity-aware features
    - Kinetic energy and momentum
    - Pairwise distance and clearance
    - Relative position and velocity
    - Contact proximity features
    - Multi-head self-attention
    - Permutation-equivariant object encoding
    - Masked object attention
    - Global scene physics embedding
    - Temporal physics sequence encoding
    - Trainable neural feature fusion
    - Physics consistency auxiliary loss
    - World-model integration adapter

Install:
    pip install torch numpy

Run:
    python -m neuroworld.physics.physics_encoder

Input:
    positions      [B, N, D]
    velocities     [B, N, D]
    accelerations  [B, N, D]
    forces         [B, N, D]
    masses         [B, N]
    radii          [B, N]
    static_mask    [B, N]
    object_mask    [B, N]

Output:
    object_embeddings [B, N, H]
    scene_embedding   [B, H]
    interaction       [B, N, N, H]
    contact_scores    [B, N, N]

This is a learned representation model.
It does not replace a physics simulator
or provide safety-certified predictions.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class PhysicsEncoderError(Exception):
    pass


class PhysicsEncoderValidationError(
    PhysicsEncoderError
):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PhysicsEncoderConfig:
    spatial_dim: int = 3

    hidden_dim: int = 128
    embedding_dim: int = 128

    attention_heads: int = 8
    attention_layers: int = 3

    feedforward_dim: int = 256

    dropout: float = 0.1

    max_objects: int = 64

    gravity_magnitude: float = 9.80665

    position_scale: float = 10.0
    velocity_scale: float = 10.0
    acceleration_scale: float = 10.0
    force_scale: float = 100.0

    mass_scale: float = 10.0
    radius_scale: float = 1.0

    interaction_distance_scale: float = 5.0
    contact_temperature: float = 0.1

    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        for name in (
            "hidden_dim",
            "embedding_dim",
            "attention_heads",
            "attention_layers",
            "feedforward_dim",
            "max_objects",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
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

        for name in (
            "position_scale",
            "velocity_scale",
            "acceleration_scale",
            "force_scale",
            "mass_scale",
            "radius_scale",
            "interaction_distance_scale",
            "contact_temperature",
            "layer_norm_eps",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be positive and finite"
                )

        if (
            not math.isfinite(self.gravity_magnitude)
            or self.gravity_magnitude < 0
        ):
            raise ValueError(
                "gravity_magnitude must be nonnegative"
            )


# ============================================================
# PHYSICS INPUT
# ============================================================

@dataclass
class PhysicsBatch:
    positions: Tensor
    velocities: Tensor

    accelerations: Tensor
    forces: Tensor

    masses: Tensor
    radii: Tensor

    static_mask: Tensor
    object_mask: Tensor

    gravity: Tensor | None = None

    def validate(
        self,
        config: PhysicsEncoderConfig,
    ) -> None:
        positions = self.positions

        if (
            positions.ndim != 3
            or positions.shape[-1]
            != config.spatial_dim
        ):
            raise PhysicsEncoderValidationError(
                "positions must be [B,N,D]"
            )

        batch_size, object_count, dimension = (
            positions.shape
        )

        if object_count > config.max_objects:
            raise PhysicsEncoderValidationError(
                "Object count exceeds max_objects"
            )

        for name in (
            "velocities",
            "accelerations",
            "forces",
        ):
            tensor = getattr(self, name)

            if tensor.shape != positions.shape:
                raise PhysicsEncoderValidationError(
                    f"{name} must be [B,N,D]"
                )

        for name in (
            "masses",
            "radii",
            "static_mask",
            "object_mask",
        ):
            tensor = getattr(self, name)

            if tensor.shape != (
                batch_size,
                object_count,
            ):
                raise PhysicsEncoderValidationError(
                    f"{name} must be [B,N]"
                )

        if self.gravity is not None:
            if self.gravity.shape not in (
                (dimension,),
                (batch_size, dimension),
            ):
                raise PhysicsEncoderValidationError(
                    "gravity must be [D] or [B,D]"
                )

        for name in (
            "positions",
            "velocities",
            "accelerations",
            "forces",
            "masses",
            "radii",
        ):
            tensor = getattr(self, name)

            if not torch.isfinite(tensor).all():
                raise PhysicsEncoderValidationError(
                    f"{name} contains non-finite values"
                )

        valid = self.object_mask.bool()

        if not valid.any(dim=1).all():
            raise PhysicsEncoderValidationError(
                "Each scene needs one valid object"
            )

        if (
            (self.masses[valid] <= 0).any()
            or (self.radii[valid] <= 0).any()
        ):
            raise PhysicsEncoderValidationError(
                "Valid objects require positive "
                "mass and radius"
            )


# ============================================================
# PHYSICS FEATURE EXTRACTOR
# ============================================================

class PhysicsFeatureExtractor(nn.Module):
    """
    Produces normalized physical features.

    Per object:
        position                 D
        velocity                 D
        acceleration             D
        force                    D
        momentum                 D
        gravity                  D

        mass                     1
        radius                   1
        speed                    1
        acceleration magnitude   1
        force magnitude          1
        kinetic energy           1
        static indicator         1

    Total:
        6*D + 7
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        gravity = torch.zeros(
            config.spatial_dim,
            dtype=torch.float32,
        )

        vertical_axis = (
            1
            if config.spatial_dim == 2
            else 2
        )

        gravity[vertical_axis] = (
            -config.gravity_magnitude
        )

        self.register_buffer(
            "default_gravity",
            gravity,
        )

        self.feature_dim = (
            6 * config.spatial_dim + 7
        )

    def forward(
        self,
        batch: PhysicsBatch,
    ) -> Tensor:
        config = self.config

        positions = (
            batch.positions
            / config.position_scale
        )

        velocities = (
            batch.velocities
            / config.velocity_scale
        )

        accelerations = (
            batch.accelerations
            / config.acceleration_scale
        )

        forces = (
            batch.forces
            / config.force_scale
        )

        masses = batch.masses.unsqueeze(
            -1
        )

        radii = batch.radii.unsqueeze(
            -1
        )

        speed = torch.linalg.vector_norm(
            batch.velocities,
            dim=-1,
            keepdim=True,
        )

        acceleration_magnitude = (
            torch.linalg.vector_norm(
                batch.accelerations,
                dim=-1,
                keepdim=True,
            )
        )

        force_magnitude = (
            torch.linalg.vector_norm(
                batch.forces,
                dim=-1,
                keepdim=True,
            )
        )

        momentum = (
            batch.velocities
            * masses
        )

        kinetic_energy = (
            0.5
            * masses
            * speed.square()
        )

        static_indicator = (
            batch.static_mask
            .to(batch.positions.dtype)
            .unsqueeze(-1)
        )

        if batch.gravity is None:
            gravity = self.default_gravity.to(
                device=batch.positions.device,
                dtype=batch.positions.dtype,
            )

        else:
            gravity = batch.gravity

        if gravity.ndim == 1:
            gravity = gravity.view(
                1,
                1,
                -1,
            )

        else:
            gravity = gravity.unsqueeze(
                1
            )

        gravity = gravity.expand_as(
            batch.positions
        )

        gravity = (
            gravity
            / config.acceleration_scale
        )

        features = torch.cat(
            [
                positions,
                velocities,
                accelerations,
                forces,
                momentum
                / (
                    config.mass_scale
                    * config.velocity_scale
                ),
                gravity,
                masses / config.mass_scale,
                radii / config.radius_scale,
                speed / config.velocity_scale,
                acceleration_magnitude
                / config.acceleration_scale,
                force_magnitude
                / config.force_scale,
                kinetic_energy
                / (
                    config.mass_scale
                    * config.velocity_scale ** 2
                ),
                static_indicator,
            ],
            dim=-1,
        )

        valid = (
            batch.object_mask
            .unsqueeze(-1)
        )

        return features * valid


# ============================================================
# OBJECT PHYSICS ENCODER
# ============================================================

class ObjectPhysicsEncoder(nn.Module):
    """
    Encodes raw physics features
    into object-level embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        config: PhysicsEncoderConfig,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                config.hidden_dim,
            ),
            nn.LayerNorm(
                config.hidden_dim,
                eps=config.layer_norm_eps,
            ),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
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
# PAIRWISE INTERACTION ENCODER
# ============================================================

class PairwisePhysicsEncoder(nn.Module):
    """
    Encodes pairwise object relationships.

    Pair features:
        relative position       D
        relative velocity       D
        center distance         1
        signed clearance        1
        combined radius         1
        contact probability     1
        relative speed          1

    Total:
        2*D + 5
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.feature_dim = (
            2 * config.spatial_dim + 5
        )

        self.network = nn.Sequential(
            nn.Linear(
                self.feature_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
        )

    def forward(
        self,
        batch: PhysicsBatch,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        positions = (
            batch.positions
        )

        velocities = (
            batch.velocities
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
            )
        )

        relative_speed = (
            torch.linalg.vector_norm(
                relative_velocity,
                dim=-1,
            )
        )

        combined_radius = (
            batch.radii[:, :, None]
            + batch.radii[:, None, :]
        )

        clearance = (
            distance
            - combined_radius
        )

        contact_probability = (
            torch.sigmoid(
                -clearance
                / self.config.contact_temperature
            )
        )

        pair_features = torch.cat(
            [
                relative_position
                / self.config.interaction_distance_scale,
                relative_velocity
                / self.config.velocity_scale,
                (
                    distance
                    / self.config.interaction_distance_scale
                ).unsqueeze(-1),
                (
                    clearance
                    / self.config.interaction_distance_scale
                ).unsqueeze(-1),
                (
                    combined_radius
                    / self.config.radius_scale
                ).unsqueeze(-1),
                contact_probability.unsqueeze(-1),
                (
                    relative_speed
                    / self.config.velocity_scale
                ).unsqueeze(-1),
            ],
            dim=-1,
        )

        interaction_embeddings = (
            self.network(
                pair_features
            )
        )

        valid = (
            batch.object_mask.bool()
        )

        pair_mask = (
            valid[:, :, None]
            & valid[:, None, :]
        )

        object_count = (
            positions.shape[1]
        )

        diagonal = torch.eye(
            object_count,
            dtype=torch.bool,
            device=positions.device,
        ).unsqueeze(0)

        pair_mask = (
            pair_mask & ~diagonal
        )

        interaction_embeddings = (
            interaction_embeddings
            * pair_mask.unsqueeze(-1)
        )

        contact_probability = (
            contact_probability
            * pair_mask
        )

        clearance = clearance.masked_fill(
            ~pair_mask,
            float("inf"),
        )

        return (
            interaction_embeddings,
            contact_probability,
            clearance,
        )


# ============================================================
# INTERACTION MESSAGE PASSING
# ============================================================

class PhysicsInteractionFusion(nn.Module):
    """
    Aggregates messages from other objects.

    Each object receives the mean
    interaction embedding of its valid
    neighbors.

    Permutation equivariant across objects.
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig,
    ) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(
                config.hidden_dim * 2,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                config.hidden_dim,
            ),
        )

        self.norm = nn.LayerNorm(
            config.hidden_dim
        )

    def forward(
        self,
        object_embeddings: Tensor,
        interaction_embeddings: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        valid = object_mask.bool()

        pair_mask = (
            valid[:, :, None]
            & valid[:, None, :]
        )

        count = object_embeddings.shape[1]

        diagonal = torch.eye(
            count,
            dtype=torch.bool,
            device=object_embeddings.device,
        ).unsqueeze(0)

        pair_mask = (
            pair_mask & ~diagonal
        )

        messages = (
            interaction_embeddings
            * pair_mask.unsqueeze(-1)
        )

        neighbor_count = (
            pair_mask.sum(
                dim=2,
                keepdim=True,
            )
            .clamp_min(1)
        )

        aggregated = (
            messages.sum(dim=2)
            / neighbor_count
        )

        fused = self.fusion(
            torch.cat(
                [
                    object_embeddings,
                    aggregated,
                ],
                dim=-1,
            )
        )

        output = self.norm(
            object_embeddings + fused
        )

        return (
            output
            * valid.unsqueeze(-1)
        )


# ============================================================
# OBJECT ATTENTION BLOCK
# ============================================================

class PhysicsAttentionBlock(nn.Module):
    """
    Self-attention over valid objects.

    Attention enables global object
    interaction reasoning.
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig,
    ) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(
            config.hidden_dim
        )

        self.norm2 = nn.LayerNorm(
            config.hidden_dim
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.feedforward_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                config.feedforward_dim,
                config.hidden_dim,
            ),
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        embeddings: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        valid = object_mask.bool()

        attention_output, _ = (
            self.attention(
                embeddings,
                embeddings,
                embeddings,
                key_padding_mask=~valid,
                need_weights=False,
            )
        )

        embeddings = self.norm1(
            embeddings
            + self.dropout(
                attention_output
            )
        )

        embeddings = self.norm2(
            embeddings
            + self.dropout(
                self.feedforward(
                    embeddings
                )
            )
        )

        return (
            embeddings
            * valid.unsqueeze(-1)
        )


# ============================================================
# SCENE PHYSICS POOLING
# ============================================================

class ScenePhysicsPooling(nn.Module):
    """
    Masked attention pooling for
    global scene representation.
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig,
    ) -> None:
        super().__init__()

        self.score_network = nn.Linear(
            config.hidden_dim,
            1,
        )

        self.projection = nn.Sequential(
            nn.Linear(
                config.hidden_dim,
                config.embedding_dim,
            ),
            nn.LayerNorm(
                config.embedding_dim
            ),
        )

    def forward(
        self,
        object_embeddings: Tensor,
        object_mask: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        valid = object_mask.bool()

        scores = (
            self.score_network(
                object_embeddings
            ).squeeze(-1)
        )

        scores = scores.masked_fill(
            ~valid,
            torch.finfo(
                scores.dtype
            ).min,
        )

        weights = torch.softmax(
            scores,
            dim=-1,
        )

        weights = (
            weights
            * valid
        )

        weights = (
            weights
            / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
        )

        pooled = (
            object_embeddings
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        return (
            self.projection(pooled),
            weights,
        )


# ============================================================
# ENCODER OUTPUT
# ============================================================

@dataclass
class PhysicsEncoderOutput:
    object_embeddings: Tensor

    scene_embedding: Tensor

    interaction_embeddings: Tensor

    contact_scores: Tensor
    pairwise_clearance: Tensor

    attention_weights: Tensor

    object_mask: Tensor

    def to_dict(self) -> dict[str, Tensor]:
        return {
            "object_embeddings": (
                self.object_embeddings
            ),
            "scene_embedding": (
                self.scene_embedding
            ),
            "interaction_embeddings": (
                self.interaction_embeddings
            ),
            "contact_scores": (
                self.contact_scores
            ),
            "pairwise_clearance": (
                self.pairwise_clearance
            ),
            "attention_weights": (
                self.attention_weights
            ),
            "object_mask": (
                self.object_mask
            ),
        }


# ============================================================
# MAIN PHYSICS ENCODER
# ============================================================

class PhysicsEncoder(nn.Module):
    """
    End-to-end physics representation model.

    Pipeline:
        Physical features
             |
        Object MLP
             |
        Pairwise interaction MLP
             |
        Message passing
             |
        Multi-head object attention
             |
        Scene attention pooling
             |
        Physics embeddings
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or PhysicsEncoderConfig()
        )

        self.feature_extractor = (
            PhysicsFeatureExtractor(
                self.config
            )
        )

        self.object_encoder = (
            ObjectPhysicsEncoder(
                self.feature_extractor.feature_dim,
                self.config,
            )
        )

        self.interaction_encoder = (
            PairwisePhysicsEncoder(
                self.config
            )
        )

        self.interaction_fusion = (
            PhysicsInteractionFusion(
                self.config
            )
        )

        self.attention_blocks = (
            nn.ModuleList(
                [
                    PhysicsAttentionBlock(
                        self.config
                    )
                    for _ in range(
                        self.config.attention_layers
                    )
                ]
            )
        )

        self.object_projection = (
            nn.Linear(
                self.config.hidden_dim,
                self.config.embedding_dim,
            )
        )

        self.scene_pooling = (
            ScenePhysicsPooling(
                self.config
            )
        )

        self.interaction_projection = (
            nn.Linear(
                self.config.hidden_dim,
                self.config.embedding_dim,
            )
        )

    def forward(
        self,
        batch: PhysicsBatch,
    ) -> PhysicsEncoderOutput:
        batch.validate(
            self.config
        )

        features = (
            self.feature_extractor(
                batch
            )
        )

        object_embeddings = (
            self.object_encoder(
                features
            )
        )

        (
            interaction_embeddings,
            contact_scores,
            pairwise_clearance,
        ) = self.interaction_encoder(
            batch
        )

        object_embeddings = (
            self.interaction_fusion(
                object_embeddings,
                interaction_embeddings,
                batch.object_mask,
            )
        )

        for block in self.attention_blocks:
            object_embeddings = block(
                object_embeddings,
                batch.object_mask,
            )

        scene_embedding, attention_weights = (
            self.scene_pooling(
                object_embeddings,
                batch.object_mask,
            )
        )

        object_embeddings = (
            self.object_projection(
                object_embeddings
            )
        )

        interaction_embeddings = (
            self.interaction_projection(
                interaction_embeddings
            )
        )

        valid = (
            batch.object_mask.bool()
        )

        object_embeddings = (
            object_embeddings
            * valid.unsqueeze(-1)
        )

        return PhysicsEncoderOutput(
            object_embeddings=(
                object_embeddings
            ),
            scene_embedding=(
                scene_embedding
            ),
            interaction_embeddings=(
                interaction_embeddings
            ),
            contact_scores=contact_scores,
            pairwise_clearance=(
                pairwise_clearance
            ),
            attention_weights=(
                attention_weights
            ),
            object_mask=batch.object_mask,
        )


# ============================================================
# TEMPORAL PHYSICS ENCODER
# ============================================================

class TemporalPhysicsEncoder(nn.Module):
    """
    Encodes sequences of scene embeddings.

    Input:
        [B,T,E]

    Output:
        [B,E]

    Uses GRU temporal aggregation.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        scene_sequence: Tensor,
    ) -> Tensor:
        if scene_sequence.ndim != 3:
            raise PhysicsEncoderValidationError(
                "Scene sequence must be [B,T,E]"
            )

        if scene_sequence.shape[1] == 0:
            raise PhysicsEncoderValidationError(
                "Sequence must contain frames"
            )

        _, hidden = self.gru(
            scene_sequence
        )

        return self.output_projection(
            hidden[-1]
        )


# ============================================================
# PHYSICS CONSISTENCY LOSS
# ============================================================

class PhysicsConsistencyLoss(nn.Module):
    """
    Auxiliary training loss.

    Penalizes deviation from:

        x_next = x + v*dt + 0.5*a*dt^2
        v_next = v + a*dt

    This assumes constant acceleration
    over the timestep.

    It does not enforce collision,
    contact, or conservation laws.
    """

    def __init__(
        self,
        timestep: float = 0.1,
        position_weight: float = 1.0,
        velocity_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if timestep <= 0:
            raise ValueError(
                "timestep must be positive"
            )

        if (
            position_weight < 0
            or velocity_weight < 0
        ):
            raise ValueError(
                "Loss weights must be nonnegative"
            )

        self.timestep = timestep
        self.position_weight = (
            position_weight
        )
        self.velocity_weight = (
            velocity_weight
        )

    def forward(
        self,
        current_positions: Tensor,
        current_velocities: Tensor,
        accelerations: Tensor,
        predicted_positions: Tensor,
        predicted_velocities: Tensor,
        object_mask: Tensor | None = None,
    ) -> Tensor:
        dt = self.timestep

        expected_positions = (
            current_positions
            + current_velocities * dt
            + 0.5
            * accelerations
            * dt ** 2
        )

        expected_velocities = (
            current_velocities
            + accelerations * dt
        )

        position_error = (
            predicted_positions
            - expected_positions
        ).square().mean(dim=-1)

        velocity_error = (
            predicted_velocities
            - expected_velocities
        ).square().mean(dim=-1)

        combined = (
            self.position_weight
            * position_error
            + self.velocity_weight
            * velocity_error
        )

        if object_mask is not None:
            mask = object_mask.to(
                combined.dtype
            )

            return (
                (combined * mask).sum()
                / mask.sum().clamp_min(1.0)
            )

        return combined.mean()


# ============================================================
# WORLD-MODEL INTEGRATION ADAPTER
# ============================================================

class PhysicsWorldModelAdapter(nn.Module):
    """
    Encodes physics observations into
    a latent world-model representation.

    Useful for:
        world_model.py
        transformer_world_model.py
        latent_dynamics.py
        imagination_engine.py

    The returned scene embedding can
    be concatenated with visual or
    action-conditioned representations.
    """

    def __init__(
        self,
        config: PhysicsEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.encoder = PhysicsEncoder(
            config
        )

    def forward(
        self,
        batch: PhysicsBatch,
    ) -> Tensor:
        result = self.encoder(
            batch
        )

        return result.scene_embedding

    def encode_objects(
        self,
        batch: PhysicsBatch,
    ) -> Tensor:
        result = self.encoder(
            batch
        )

        return result.object_embeddings

    def encode_full(
        self,
        batch: PhysicsBatch,
    ) -> PhysicsEncoderOutput:
        return self.encoder(
            batch
        )


# ============================================================
# INPUT FACTORY
# ============================================================

def create_physics_batch(
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    radii: Tensor,
    *,
    accelerations: Tensor | None = None,
    forces: Tensor | None = None,
    static_mask: Tensor | None = None,
    object_mask: Tensor | None = None,
    gravity: Tensor | None = None,
) -> PhysicsBatch:
    if accelerations is None:
        accelerations = torch.zeros_like(
            positions
        )

    if forces is None:
        forces = torch.zeros_like(
            positions
        )

    if static_mask is None:
        static_mask = torch.zeros(
            masses.shape,
            dtype=torch.bool,
            device=masses.device,
        )

    if object_mask is None:
        object_mask = torch.ones(
            masses.shape,
            dtype=torch.bool,
            device=masses.device,
        )

    return PhysicsBatch(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        forces=forces,
        masses=masses,
        radii=radii,
        static_mask=static_mask,
        object_mask=object_mask,
        gravity=gravity,
    )


# ============================================================
# FACTORY
# ============================================================

def create_physics_encoder(
    config: PhysicsEncoderConfig | None = None,
) -> PhysicsEncoder:
    return PhysicsEncoder(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - PHYSICS ENCODER"
    )
    print("=" * 65)

    torch.manual_seed(42)

    config = PhysicsEncoderConfig(
        spatial_dim=3,
        hidden_dim=128,
        embedding_dim=128,
        attention_heads=8,
        attention_layers=3,
        max_objects=16,
    )

    encoder = PhysicsEncoder(
        config
    )

    batch_size = 2
    object_count = 5
    spatial_dim = 3

    positions = torch.randn(
        batch_size,
        object_count,
        spatial_dim,
    )

    velocities = torch.randn(
        batch_size,
        object_count,
        spatial_dim,
    )

    accelerations = torch.randn(
        batch_size,
        object_count,
        spatial_dim,
    )

    forces = torch.randn(
        batch_size,
        object_count,
        spatial_dim,
    )

    masses = (
        torch.rand(
            batch_size,
            object_count,
        ) * 5.0 + 0.5
    )

    radii = (
        torch.rand(
            batch_size,
            object_count,
        ) * 0.4 + 0.1
    )

    static_mask = torch.zeros(
        batch_size,
        object_count,
        dtype=torch.bool,
    )

    static_mask[:, -1] = True

    object_mask = torch.ones(
        batch_size,
        object_count,
        dtype=torch.bool,
    )

    object_mask[1, -1] = False

    batch = create_physics_batch(
        positions=positions,
        velocities=velocities,
        masses=masses,
        radii=radii,
        accelerations=accelerations,
        forces=forces,
        static_mask=static_mask,
        object_mask=object_mask,
    )

    output = encoder(
        batch
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
        "Interaction embeddings:",
        output.interaction_embeddings.shape,
    )

    print(
        "Contact scores:",
        output.contact_scores.shape,
    )

    print(
        "Pairwise clearance:",
        output.pairwise_clearance.shape,
    )

    temporal_encoder = TemporalPhysicsEncoder(
        embedding_dim=128,
        hidden_dim=128,
    )

    scene_sequence = torch.randn(
        2,
        10,
        128,
    )

    temporal_embedding = temporal_encoder(
        scene_sequence
    )

    print(
        "Temporal embedding:",
        temporal_embedding.shape,
    )

    print("\nPhysics consistency loss")

    loss_fn = PhysicsConsistencyLoss(
        timestep=0.1
    )

    next_positions = (
        positions
        + velocities * 0.1
        + 0.5
        * accelerations
        * 0.1 ** 2
    )

    next_velocities = (
        velocities
        + accelerations * 0.1
    )

    loss = loss_fn(
        positions,
        velocities,
        accelerations,
        next_positions,
        next_velocities,
        object_mask,
    )

    print(
        "Consistency loss:",
        float(loss.item()),
    )

    print(
        "\nPhysics encoder demonstration completed"
    )


if __name__ == "__main__":
    main()
