
"""
NeuroWorld AI
File: neuroworld/perception/vision_transformer.py

Vision Transformer (ViT) Encoder

Features:
    - Patch embedding
    - Learnable positional embeddings
    - Class token
    - Multi-head self-attention
    - Transformer encoder blocks
    - Feed-forward neural networks
    - Layer normalization
    - Residual connections
    - Attention dropout
    - Stochastic depth
    - Image classification
    - Visual feature extraction
    - Spatial token extraction
    - Attention map extraction
    - World-model feature adapter
    - Configurable image dimensions
    - Checkpoint saving/loading
    - Synthetic demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.perception.vision_transformer

Input:
    Images [B, C, H, W]

Outputs:
    Global embeddings [B, D]
    Patch tokens [B, N, D]
    Attention maps [B, heads, N+1, N+1]

Notes:
    This module defines a trainable architecture.
    Meaningful visual understanding requires
    pretrained weights or task-specific training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class VisionTransformerError(Exception):
    """Base Vision Transformer exception."""


class VisionTransformerValidationError(
    VisionTransformerError
):
    """Invalid input or configuration."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size: int = 224
    patch_size: int = 16

    in_channels: int = 3

    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 6

    mlp_ratio: float = 4.0

    dropout: float = 0.1
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1

    num_classes: int = 1000

    use_class_token: bool = True

    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        integer_fields = (
            "image_size",
            "patch_size",
            "in_channels",
            "embed_dim",
            "depth",
            "num_heads",
        )

        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible "
                "by patch_size"
            )

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible "
                "by num_heads"
            )

        if self.mlp_ratio <= 0:
            raise ValueError(
                "mlp_ratio must be positive"
            )

        for name in (
            "dropout",
            "attention_dropout",
            "drop_path_rate",
        ):
            value = getattr(self, name)

            if not 0 <= value < 1:
                raise ValueError(
                    f"{name} must be in [0,1)"
                )

        if self.num_classes < 0:
            raise ValueError(
                "num_classes cannot be negative"
            )

        if self.layer_norm_eps <= 0:
            raise ValueError(
                "layer_norm_eps must be positive"
            )

    @property
    def grid_size(self) -> int:
        return (
            self.image_size
            // self.patch_size
        )

    @property
    def num_patches(self) -> int:
        return (
            self.grid_size ** 2
        )

    @property
    def num_tokens(self) -> int:
        return (
            self.num_patches
            + int(self.use_class_token)
        )


# ============================================================
# PATCH EMBEDDING
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Converts an image into patch tokens.

    Input:
        [B,C,H,W]

    Output:
        [B,N,D]

    Uses a Conv2d with kernel_size
    equal to patch_size and stride
    equal to patch_size.
    """

    def __init__(
        self,
        config: VisionTransformerConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        if images.ndim != 4:
            raise VisionTransformerValidationError(
                "Images must be [B,C,H,W]"
            )

        batch, channels, height, width = (
            images.shape
        )

        if channels != self.config.in_channels:
            raise VisionTransformerValidationError(
                "Invalid image channel count"
            )

        if (
            height != self.config.image_size
            or width != self.config.image_size
        ):
            raise VisionTransformerValidationError(
                f"Expected images of size "
                f"{self.config.image_size}x"
                f"{self.config.image_size}"
            )

        features = self.projection(
            images
        )

        return features.flatten(
            2
        ).transpose(
            1,
            2,
        )


# ============================================================
# STOCHASTIC DEPTH
# ============================================================

class DropPath(nn.Module):
    """
    Per-sample stochastic depth.

    Active only during training.
    """

    def __init__(
        self,
        drop_probability: float = 0.0,
    ) -> None:
        super().__init__()

        self.drop_probability = (
            drop_probability
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if (
            not self.training
            or self.drop_probability == 0
        ):
            return x

        keep_probability = (
            1.0
            - self.drop_probability
        )

        shape = (
            x.shape[0],
        ) + (
            1,
        ) * (x.ndim - 1)

        mask = (
            torch.rand(
                shape,
                dtype=x.dtype,
                device=x.device,
            )
            < keep_probability
        ).to(x.dtype)

        return (
            x
            * mask
            / keep_probability
        )


# ============================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Standard scaled dot-product attention.

    Q, K, V:
        [B,heads,N,head_dim]

    Attention:
        softmax(Q @ K.T / sqrt(head_dim))

    Output:
        [B,N,D]
    """

    def __init__(
        self,
        config: VisionTransformerConfig,
    ) -> None:
        super().__init__()

        self.embed_dim = (
            config.embed_dim
        )

        self.num_heads = (
            config.num_heads
        )

        self.head_dim = (
            config.embed_dim
            // config.num_heads
        )

        self.scale = (
            self.head_dim ** -0.5
        )

        self.qkv = nn.Linear(
            config.embed_dim,
            config.embed_dim * 3,
        )

        self.attention_dropout = nn.Dropout(
            config.attention_dropout
        )

        self.projection = nn.Linear(
            config.embed_dim,
            config.embed_dim,
        )

        self.output_dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[
        Tensor,
        Tensor | None,
    ]:
        batch_size, token_count, _ = (
            x.shape
        )

        qkv = self.qkv(
            x
        )

        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        query = qkv[0]
        key = qkv[1]
        value = qkv[2]

        attention_logits = (
            query
            @ key.transpose(-2, -1)
        ) * self.scale

        attention = torch.softmax(
            attention_logits,
            dim=-1,
        )

        attention_weights = (
            attention
            if return_attention
            else None
        )

        attention = (
            self.attention_dropout(
                attention
            )
        )

        output = (
            attention @ value
        )

        output = output.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            token_count,
            self.embed_dim,
        )

        output = self.projection(
            output
        )

        output = self.output_dropout(
            output
        )

        return (
            output,
            attention_weights,
        )


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class TransformerMLP(nn.Module):
    """
    Transformer feed-forward block.

    Linear -> GELU -> Dropout
    -> Linear -> Dropout
    """

    def __init__(
        self,
        config: VisionTransformerConfig,
    ) -> None:
        super().__init__()

        hidden_dim = int(
            config.embed_dim
            * config.mlp_ratio
        )

        self.network = nn.Sequential(
            nn.Linear(
                config.embed_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                config.dropout
            ),
            nn.Linear(
                hidden_dim,
                config.embed_dim,
            ),
            nn.Dropout(
                config.dropout
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
# TRANSFORMER ENCODER BLOCK
# ============================================================

class VisionTransformerBlock(nn.Module):
    """
    Pre-normalization Transformer block.

    x = x + Attention(LN(x))
    x = x + MLP(LN(x))
    """

    def __init__(
        self,
        config: VisionTransformerConfig,
        drop_path_probability: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        self.attention = (
            MultiHeadSelfAttention(
                config
            )
        )

        self.norm2 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        self.mlp = TransformerMLP(
            config
        )

        self.drop_path = DropPath(
            drop_path_probability
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[
        Tensor,
        Tensor | None,
    ]:
        attention_output, weights = (
            self.attention(
                self.norm1(x),
                return_attention=(
                    return_attention
                ),
            )
        )

        x = (
            x
            + self.drop_path(
                attention_output
            )
        )

        x = (
            x
            + self.drop_path(
                self.mlp(
                    self.norm2(x)
                )
            )
        )

        return (
            x,
            weights,
        )


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

@dataclass
class VisionTransformerOutput:
    global_embedding: Tensor

    patch_tokens: Tensor
    all_tokens: Tensor

    spatial_features: Tensor

    logits: Tensor | None = None

    attention_maps: list[Tensor] | None = None

    def detach(self) -> VisionTransformerOutput:
        return VisionTransformerOutput(
            global_embedding=(
                self.global_embedding.detach()
            ),
            patch_tokens=(
                self.patch_tokens.detach()
            ),
            all_tokens=(
                self.all_tokens.detach()
            ),
            spatial_features=(
                self.spatial_features.detach()
            ),
            logits=(
                self.logits.detach()
                if self.logits is not None
                else None
            ),
            attention_maps=(
                [
                    attention.detach()
                    for attention
                    in self.attention_maps
                ]
                if self.attention_maps is not None
                else None
            ),
        )


# ============================================================
# VISION TRANSFORMER
# ============================================================

class VisionTransformer(nn.Module):
    """
    Configurable Vision Transformer.

    Supports:
        image classification
        image embeddings
        patch tokens
        spatial feature maps
        attention extraction

    Output:
        VisionTransformerOutput
    """

    def __init__(
        self,
        config: VisionTransformerConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config
            or VisionTransformerConfig()
        )

        self.patch_embedding = (
            PatchEmbedding(
                self.config
            )
        )

        if self.config.use_class_token:
            self.class_token = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    self.config.embed_dim,
                )
            )
        else:
            self.register_parameter(
                "class_token",
                None,
            )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.config.num_tokens,
                self.config.embed_dim,
            )
        )

        self.position_dropout = nn.Dropout(
            self.config.dropout
        )

        drop_rates = torch.linspace(
            0.0,
            self.config.drop_path_rate,
            self.config.depth,
        ).tolist()

        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    self.config,
                    drop_path_probability=(
                        drop_rates[index]
                    ),
                )
                for index in range(
                    self.config.depth
                )
            ]
        )

        self.norm = nn.LayerNorm(
            self.config.embed_dim,
            eps=self.config.layer_norm_eps,
        )

        if self.config.num_classes > 0:
            self.classifier = nn.Linear(
                self.config.embed_dim,
                self.config.num_classes,
            )
        else:
            self.classifier = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        if self.class_token is not None:
            nn.init.trunc_normal_(
                self.class_token,
                std=0.02,
            )

        for module in self.modules():
            if isinstance(
                module,
                nn.Linear,
            ):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.LayerNorm,
            ):
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward_features(
        self,
        images: Tensor,
        *,
        return_attention: bool = False,
    ) -> VisionTransformerOutput:
        tokens = self.patch_embedding(
            images
        )

        batch_size = tokens.shape[0]

        if self.class_token is not None:
            class_tokens = (
                self.class_token.expand(
                    batch_size,
                    -1,
                    -1,
                )
            )

            tokens = torch.cat(
                [
                    class_tokens,
                    tokens,
                ],
                dim=1,
            )

        tokens = (
            tokens
            + self.position_embedding
        )

        tokens = (
            self.position_dropout(
                tokens
            )
        )

        attention_maps = (
            []
            if return_attention
            else None
        )

        for block in self.blocks:
            tokens, attention = block(
                tokens,
                return_attention=(
                    return_attention
                ),
            )

            if (
                return_attention
                and attention is not None
            ):
                attention_maps.append(
                    attention
                )

        tokens = self.norm(
            tokens
        )

        if self.class_token is not None:
            global_embedding = (
                tokens[:, 0]
            )

            patch_tokens = (
                tokens[:, 1:]
            )

        else:
            patch_tokens = tokens

            global_embedding = (
                patch_tokens.mean(
                    dim=1
                )
            )

        grid_size = (
            self.config.grid_size
        )

        spatial_features = (
            patch_tokens.transpose(
                1,
                2,
            ).reshape(
                batch_size,
                self.config.embed_dim,
                grid_size,
                grid_size,
            )
        )

        return VisionTransformerOutput(
            global_embedding=(
                global_embedding
            ),
            patch_tokens=patch_tokens,
            all_tokens=tokens,
            spatial_features=(
                spatial_features
            ),
            attention_maps=(
                attention_maps
            ),
        )

    def forward(
        self,
        images: Tensor,
        *,
        return_attention: bool = False,
    ) -> VisionTransformerOutput:
        output = self.forward_features(
            images,
            return_attention=(
                return_attention
            ),
        )

        if self.config.num_classes > 0:
            output.logits = (
                self.classifier(
                    output.global_embedding
                )
            )

        return output

    @torch.no_grad()
    def encode(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Returns global image embeddings.

        Shape:
            [B,embed_dim]
        """
        was_training = self.training

        self.eval()

        try:
            return self.forward_features(
                images
            ).global_embedding
        finally:
            self.train(
                was_training
            )

    @torch.no_grad()
    def encode_patches(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Returns spatial patch embeddings.

        Shape:
            [B,num_patches,embed_dim]
        """
        was_training = self.training

        self.eval()

        try:
            return self.forward_features(
                images
            ).patch_tokens
        finally:
            self.train(
                was_training
            )

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str = "cpu",
    ) -> VisionTransformer:
        """
        Load checkpoints from trusted sources only.
        """
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        config = VisionTransformerConfig(
            **checkpoint["config"]
        )

        model = cls(
            config
        )

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        return model


# ============================================================
# ATTENTION MAP UTILITIES
# ============================================================

class VisionAttentionAnalyzer:
    """
    Extracts attention-based patch
    relevance maps.

    Attention is a visualization signal,
    not a validated causal explanation.
    """

    @staticmethod
    def class_attention_map(
        output: VisionTransformerOutput,
        grid_size: int,
    ) -> Tensor:
        if not output.attention_maps:
            raise VisionTransformerError(
                "No attention maps available"
            )

        attention = (
            output.attention_maps[-1]
        )

        # Average over attention heads.
        attention = attention.mean(
            dim=1
        )

        # Class token attends to patches.
        patch_attention = (
            attention[:, 0, 1:]
        )

        if patch_attention.shape[1] != (
            grid_size ** 2
        ):
            raise VisionTransformerValidationError(
                "Class token or grid mismatch"
            )

        return patch_attention.reshape(
            -1,
            grid_size,
            grid_size,
        )

    @staticmethod
    def resize_attention_map(
        attention_map: Tensor,
        output_size: tuple[int, int],
    ) -> Tensor:
        if attention_map.ndim != 3:
            raise VisionTransformerValidationError(
                "Attention map must be [B,H,W]"
            )

        resized = F.interpolate(
            attention_map.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        return resized.squeeze(1)


# ============================================================
# WORLD MODEL ADAPTER
# ============================================================

class VisionWorldModelAdapter(nn.Module):
    """
    Converts ViT features into the
    latent dimension expected by a
    neural world model.

    Input:
        images [B,C,H,W]

    Output:
        latent states [B,latent_dim]
    """

    def __init__(
        self,
        vision_transformer: VisionTransformer,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        self.vision_transformer = (
            vision_transformer
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(
                vision_transformer.config.embed_dim
            ),
            nn.Linear(
                vision_transformer.config.embed_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                latent_dim
            ),
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        features = (
            self.vision_transformer.forward_features(
                images
            )
        )

        return self.projection(
            features.global_embedding
        )

    def encode_spatial(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Project each patch token into
        world-model latent space.

        Output:
            [B,N,latent_dim]
        """
        features = (
            self.vision_transformer.forward_features(
                images
            )
        )

        return self.projection(
            features.patch_tokens
        )


# ============================================================
# MODEL FACTORY
# ============================================================

def create_vision_transformer(
    config: VisionTransformerConfig | None = None,
) -> VisionTransformer:
    return VisionTransformer(
        config
    )


def create_small_vit() -> VisionTransformer:
    return VisionTransformer(
        VisionTransformerConfig(
            image_size=224,
            patch_size=16,
            embed_dim=192,
            depth=6,
            num_heads=3,
            num_classes=0,
        )
    )


def create_base_vit() -> VisionTransformer:
    return VisionTransformer(
        VisionTransformerConfig(
            image_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes=0,
        )
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - VISION TRANSFORMER"
    )
    print("=" * 65)

    # Small configuration for a quick
    # architecture smoke test.
    config = VisionTransformerConfig(
        image_size=64,
        patch_size=8,
        in_channels=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        num_classes=10,
        dropout=0.1,
    )

    model = VisionTransformer(
        config
    )

    model.eval()

    images = torch.randn(
        2,
        3,
        64,
        64,
    )

    with torch.no_grad():
        output = model(
            images,
            return_attention=True,
        )

    print(
        "Global embeddings:",
        output.global_embedding.shape,
    )

    print(
        "Patch tokens:",
        output.patch_tokens.shape,
    )

    print(
        "Spatial features:",
        output.spatial_features.shape,
    )

    print(
        "Classification logits:",
        output.logits.shape,
    )

    print(
        "Attention layers:",
        len(output.attention_maps),
    )

    attention_map = (
        VisionAttentionAnalyzer.class_attention_map(
            output,
            grid_size=config.grid_size,
        )
    )

    print(
        "Attention map:",
        attention_map.shape,
    )

    world_adapter = VisionWorldModelAdapter(
        model,
        latent_dim=64,
    )

    with torch.no_grad():
        latent = world_adapter(
            images
        )

    print(
        "World-model latent:",
        latent.shape,
    )

    print(
        "\nVision Transformer demonstration completed"
    )


if __name__ == "__main__":
    main()
