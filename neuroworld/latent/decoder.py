
"""
NeuroWorld AI
File: neuroworld/latent/decoder.py

Neural Latent Decoder & Reconstruction Engine

Features:
    - Vector latent decoding
    - MLP state reconstruction
    - CNN image reconstruction
    - Transformer sequence decoding
    - Multi-head object/state decoding
    - Learned reconstruction uncertainty
    - Latent-to-observation conversion
    - Future-state reconstruction
    - Reconstruction loss calculation
    - Batched inference
    - Model checkpoint save/load
    - Training demonstration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.latent.decoder

Notes:
    - The decoder reconstructs learned estimates.
    - It does not guarantee physically valid states.
    - Image output resolution is configurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class DecoderError(Exception):
    pass


class DecoderValidationError(DecoderError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class DecoderConfig:
    latent_dim: int = 128
    hidden_dim: int = 256

    state_dim: int = 64
    object_dim: int = 32

    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64

    sequence_length: int = 16

    transformer_layers: int = 3
    attention_heads: int = 8

    dropout: float = 0.1

    predict_uncertainty: bool = True

    min_log_variance: float = -10.0
    max_log_variance: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "latent_dim",
            "hidden_dim",
            "state_dim",
            "object_dim",
            "image_channels",
            "image_height",
            "image_width",
            "sequence_length",
            "transformer_layers",
            "attention_heads",
        )

        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if (
            self.hidden_dim
            % self.attention_heads != 0
        ):
            raise ValueError(
                "hidden_dim must be divisible "
                "by attention_heads"
            )

        if (
            self.image_height % 8 != 0
            or self.image_width % 8 != 0
        ):
            raise ValueError(
                "Image dimensions must be "
                "multiples of 8"
            )

        if (
            self.min_log_variance
            >= self.max_log_variance
        ):
            raise ValueError(
                "Invalid log-variance bounds"
            )


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

@dataclass
class DecoderOutput:
    reconstruction: Tensor

    log_variance: Tensor | None = None

    features: Tensor | None = None

    metadata: dict[str, Any] | None = None

    @property
    def uncertainty(self) -> Tensor | None:
        if self.log_variance is None:
            return None

        return torch.exp(
            self.log_variance
        )

    def detach(self) -> DecoderOutput:
        return DecoderOutput(
            reconstruction=(
                self.reconstruction.detach()
            ),
            log_variance=(
                self.log_variance.detach()
                if self.log_variance is not None
                else None
            ),
            features=(
                self.features.detach()
                if self.features is not None
                else None
            ),
            metadata=self.metadata,
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_latent(
    latent: Tensor,
    latent_dim: int,
    *,
    sequence: bool = False,
) -> None:
    expected_rank = (
        3 if sequence else 2
    )

    if latent.ndim != expected_rank:
        raise DecoderValidationError(
            f"Expected rank {expected_rank}, "
            f"received {latent.ndim}"
        )

    if latent.shape[-1] != latent_dim:
        raise DecoderValidationError(
            f"Expected latent_dim={latent_dim}"
        )

    if not torch.isfinite(latent).all():
        raise DecoderValidationError(
            "Latent contains non-finite values"
        )


# ============================================================
# RESIDUAL MLP BLOCK
# ============================================================

class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(
                dimension,
                dimension * 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                dimension * 2,
                dimension,
            ),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return x + self.layers(x)


# ============================================================
# STATE DECODER
# ============================================================

class StateDecoder(nn.Module):
    """
    Latent:
        [B, latent_dim]

    Output:
        [B, state_dim]
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            ResidualMLPBlock(
                config.hidden_dim,
                config.dropout,
            ),
            ResidualMLPBlock(
                config.hidden_dim,
                config.dropout,
            ),
        )

        self.output_head = nn.Linear(
            config.hidden_dim,
            config.state_dim,
        )

        self.uncertainty_head = (
            nn.Linear(
                config.hidden_dim,
                config.state_dim,
            )
            if config.predict_uncertainty
            else None
        )

    def forward(
        self,
        latent: Tensor,
    ) -> DecoderOutput:
        validate_latent(
            latent,
            self.config.latent_dim,
        )

        features = self.blocks(
            self.input_projection(latent)
        )

        reconstruction = (
            self.output_head(features)
        )

        log_variance = None

        if self.uncertainty_head is not None:
            log_variance = (
                self.uncertainty_head(features)
                .clamp(
                    self.config.min_log_variance,
                    self.config.max_log_variance,
                )
            )

        return DecoderOutput(
            reconstruction=reconstruction,
            log_variance=log_variance,
            features=features,
            metadata={
                "decoder": "state",
            },
        )


# ============================================================
# OBJECT DECODER
# ============================================================

class ObjectDecoder(nn.Module):
    """
    Reconstructs object-level features
    from latent embeddings.

    Input:
        [B, latent_dim]

    Output:
        [B, object_dim]
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
            ResidualMLPBlock(
                config.hidden_dim,
                config.dropout,
            ),
            nn.LayerNorm(
                config.hidden_dim
            ),
            nn.Linear(
                config.hidden_dim,
                config.object_dim,
            ),
        )

    def forward(
        self,
        latent: Tensor,
    ) -> Tensor:
        validate_latent(
            latent,
            self.config.latent_dim,
        )

        return self.network(
            latent
        )


# ============================================================
# IMAGE DECODER
# ============================================================

class ImageDecoder(nn.Module):
    """
    CNN image reconstruction.

    Latent:
        [B, latent_dim]

    Image:
        [B, C, H, W]

    Three 2x upsampling stages.

    Pixel values are normalized to [0,1].
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.base_height = (
            config.image_height // 8
        )

        self.base_width = (
            config.image_width // 8
        )

        self.base_channels = (
            config.hidden_dim
        )

        flattened = (
            self.base_channels
            * self.base_height
            * self.base_width
        )

        self.projection = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                flattened,
            ),
        )

        mid_channels = max(
            32,
            config.hidden_dim // 2,
        )

        low_channels = max(
            16,
            config.hidden_dim // 4,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.base_channels,
                mid_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                1,
                mid_channels,
            ),
            nn.GELU(),

            nn.ConvTranspose2d(
                mid_channels,
                low_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                1,
                low_channels,
            ),
            nn.GELU(),

            nn.ConvTranspose2d(
                low_channels,
                low_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                low_channels,
                config.image_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        latent: Tensor,
    ) -> Tensor:
        validate_latent(
            latent,
            self.config.latent_dim,
        )

        batch_size = latent.shape[0]

        features = self.projection(
            latent
        )

        features = features.reshape(
            batch_size,
            self.base_channels,
            self.base_height,
            self.base_width,
        )

        image = self.decoder(
            features
        )

        expected_shape = (
            batch_size,
            self.config.image_channels,
            self.config.image_height,
            self.config.image_width,
        )

        if image.shape != expected_shape:
            raise DecoderError(
                "Image reconstruction shape mismatch"
            )

        return image


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class SinusoidalPositionEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_length: int,
    ) -> None:
        super().__init__()

        positions = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(1)

        indices = torch.arange(
            0,
            hidden_dim,
            2,
            dtype=torch.float32,
        )

        frequency = torch.exp(
            indices
            * (
                -math.log(10000.0)
                / hidden_dim
            )
        )

        encoding = torch.zeros(
            max_length,
            hidden_dim,
        )

        encoding[:, 0::2] = torch.sin(
            positions * frequency
        )

        odd_width = (
            encoding[:, 1::2].shape[1]
        )

        if odd_width > 0:
            encoding[:, 1::2] = torch.cos(
                positions
                * frequency[:odd_width]
            )

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(0),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if x.shape[1] > (
            self.encoding.shape[1]
        ):
            raise DecoderValidationError(
                "Sequence exceeds positional capacity"
            )

        return (
            x
            + self.encoding[
                :, :x.shape[1]
            ].to(
                device=x.device,
                dtype=x.dtype,
            )
        )


# ============================================================
# TRANSFORMER SEQUENCE DECODER
# ============================================================

class SequenceDecoder(nn.Module):
    """
    Decodes a sequence of latent states.

    Input:
        [B,T,latent_dim]

    Output:
        [B,T,state_dim]

    Uses bidirectional self-attention.

    This is a reconstruction decoder,
    not a causal autoregressive decoder.
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Linear(
            config.latent_dim,
            config.hidden_dim,
        )

        self.position_encoding = (
            SinusoidalPositionEncoding(
                config.hidden_dim,
                config.sequence_length,
            )
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=(
                config.hidden_dim * 4
            ),
            dropout=config.dropout,
            activation="gelu",
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

        self.output_head = nn.Linear(
            config.hidden_dim,
            config.state_dim,
        )

    def forward(
        self,
        latents: Tensor,
    ) -> Tensor:
        validate_latent(
            latents,
            self.config.latent_dim,
            sequence=True,
        )

        features = (
            self.input_projection(
                latents
            )
        )

        features = (
            self.position_encoding(
                features
            )
        )

        features = self.transformer(
            features
        )

        return self.output_head(
            features
        )


# ============================================================
# MULTI-HEAD LATENT DECODER
# ============================================================

class MultiHeadDecoder(nn.Module):
    """
    Reconstructs multiple modalities
    from a shared latent vector.

    Outputs:
        state
        objects
        image
        state uncertainty
    """

    def __init__(
        self,
        config: DecoderConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.state_decoder = StateDecoder(
            config
        )

        self.object_decoder = ObjectDecoder(
            config
        )

        self.image_decoder = ImageDecoder(
            config
        )

    def forward(
        self,
        latent: Tensor,
        *,
        decode_image: bool = True,
    ) -> dict[str, Tensor | None]:
        state_output = (
            self.state_decoder(
                latent
            )
        )

        objects = self.object_decoder(
            latent
        )

        image = None

        if decode_image:
            image = self.image_decoder(
                latent
            )

        return {
            "state": (
                state_output.reconstruction
            ),
            "objects": objects,
            "image": image,
            "state_log_variance": (
                state_output.log_variance
            ),
            "state_uncertainty": (
                state_output.uncertainty
            ),
        }


# ============================================================
# RECONSTRUCTION LOSS
# ============================================================

@dataclass(frozen=True)
class ReconstructionLossConfig:
    state_weight: float = 1.0
    object_weight: float = 1.0
    image_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "state_weight",
            "object_weight",
            "image_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


class ReconstructionLoss(nn.Module):
    """
    Weighted multi-modal reconstruction loss.

    Uses:
        State: MSE or Gaussian NLL
        Objects: MSE
        Images: MSE

    All losses are averaged over elements.
    """

    def __init__(
        self,
        config: ReconstructionLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or ReconstructionLossConfig()
        )

    @staticmethod
    def gaussian_nll(
        prediction: Tensor,
        target: Tensor,
        log_variance: Tensor,
    ) -> Tensor:
        if (
            prediction.shape != target.shape
            or log_variance.shape
            != prediction.shape
        ):
            raise DecoderValidationError(
                "Gaussian NLL shape mismatch"
            )

        return 0.5 * (
            log_variance
            + (target - prediction).square()
            * torch.exp(-log_variance)
            + math.log(2.0 * math.pi)
        ).mean()

    def forward(
        self,
        predictions: dict[str, Tensor | None],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        losses: dict[str, Tensor] = {}

        total: Tensor | None = None

        def add_loss(
            name: str,
            value: Tensor,
            weight: float,
        ) -> None:
            nonlocal total

            losses[name] = value

            weighted = value * weight

            total = (
                weighted
                if total is None
                else total + weighted
            )

        state_prediction = predictions.get(
            "state"
        )

        if (
            state_prediction is not None
            and "state" in targets
        ):
            log_variance = predictions.get(
                "state_log_variance"
            )

            if log_variance is not None:
                state_loss = self.gaussian_nll(
                    state_prediction,
                    targets["state"],
                    log_variance,
                )
            else:
                state_loss = F.mse_loss(
                    state_prediction,
                    targets["state"],
                )

            add_loss(
                "state_loss",
                state_loss,
                self.config.state_weight,
            )

        object_prediction = predictions.get(
            "objects"
        )

        if (
            object_prediction is not None
            and "objects" in targets
        ):
            object_loss = F.mse_loss(
                object_prediction,
                targets["objects"],
            )

            add_loss(
                "object_loss",
                object_loss,
                self.config.object_weight,
            )

        image_prediction = predictions.get(
            "image"
        )

        if (
            image_prediction is not None
            and "image" in targets
        ):
            image_loss = F.mse_loss(
                image_prediction,
                targets["image"],
            )

            add_loss(
                "image_loss",
                image_loss,
                self.config.image_weight,
            )

        if total is None:
            raise DecoderValidationError(
                "No matching prediction/target pairs"
            )

        losses["total_loss"] = total

        return losses


# ============================================================
# DECODER CHECKPOINT MANAGER
# ============================================================

class DecoderCheckpointManager:
    """
    Saves decoder weights and configuration.

    Load only checkpoints from trusted
    sources because torch.load uses
    serialization mechanisms that should
    not be applied to untrusted files.
    """

    @staticmethod
    def save(
        model: nn.Module,
        config: DecoderConfig,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "model_state_dict": (
                    model.state_dict()
                ),
                "config": asdict(config),
                "model_class": (
                    type(model).__name__
                ),
            },
            path,
        )

    @staticmethod
    def load(
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> MultiHeadDecoder:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        if (
            checkpoint.get("model_class")
            != "MultiHeadDecoder"
        ):
            raise DecoderValidationError(
                "Checkpoint is not a MultiHeadDecoder"
            )

        config = DecoderConfig(
            **checkpoint["config"]
        )

        model = MultiHeadDecoder(
            config
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(device)
        model.eval()

        return model


# ============================================================
# HIGH-LEVEL LATENT DECODER
# ============================================================

class LatentDecoder(nn.Module):
    """
    Unified decoder interface.

    Supports:
        decode_state()
        decode_objects()
        decode_image()
        decode_sequence()
        forward()
    """

    def __init__(
        self,
        config: DecoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or DecoderConfig()
        )

        self.multi_head = MultiHeadDecoder(
            self.config
        )

        self.sequence_decoder = (
            SequenceDecoder(
                self.config
            )
        )

    def decode_state(
        self,
        latent: Tensor,
    ) -> DecoderOutput:
        return (
            self.multi_head.state_decoder(
                latent
            )
        )

    def decode_objects(
        self,
        latent: Tensor,
    ) -> Tensor:
        return (
            self.multi_head.object_decoder(
                latent
            )
        )

    def decode_image(
        self,
        latent: Tensor,
    ) -> Tensor:
        return (
            self.multi_head.image_decoder(
                latent
            )
        )

    def decode_sequence(
        self,
        latents: Tensor,
    ) -> Tensor:
        return self.sequence_decoder(
            latents
        )

    def forward(
        self,
        latent: Tensor,
        *,
        decode_image: bool = True,
    ) -> dict[str, Tensor | None]:
        return self.multi_head(
            latent,
            decode_image=decode_image,
        )


# ============================================================
# WORLD MODEL DECODER ADAPTER
# ============================================================

class WorldModelDecoderAdapter:
    """
    Converts predicted latent states
    into observation/state predictions.

    Accepts:
        [B,L]
        [B,T,L]

    Returns:
        [B,state_dim]
        [B,T,state_dim]
    """

    def __init__(
        self,
        decoder: LatentDecoder,
    ) -> None:
        self.decoder = decoder

    @torch.no_grad()
    def predict_observation(
        self,
        latents: Tensor,
    ) -> Tensor:
        self.decoder.eval()

        if latents.ndim == 2:
            return (
                self.decoder.decode_state(
                    latents
                ).reconstruction
            )

        if latents.ndim == 3:
            return (
                self.decoder.decode_sequence(
                    latents
                )
            )

        raise DecoderValidationError(
            "Latents must be [B,L] or [B,T,L]"
        )

    @torch.no_grad()
    def predict_frame(
        self,
        latents: Tensor,
    ) -> Tensor:
        self.decoder.eval()

        if latents.ndim != 2:
            raise DecoderValidationError(
                "Frame decoding requires [B,L]"
            )

        return self.decoder.decode_image(
            latents
        )


# ============================================================
# FACTORY
# ============================================================

def create_decoder(
    config: DecoderConfig | None = None,
) -> LatentDecoder:
    return LatentDecoder(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - LATENT DECODER"
    )
    print("=" * 65)

    torch.manual_seed(42)

    config = DecoderConfig(
        latent_dim=128,
        hidden_dim=128,
        state_dim=32,
        object_dim=16,
        image_channels=3,
        image_height=64,
        image_width=64,
        sequence_length=8,
        attention_heads=8,
    )

    decoder = LatentDecoder(
        config
    )

    latent = torch.randn(
        4,
        config.latent_dim,
    )

    print("\n1. State reconstruction")

    state_output = (
        decoder.decode_state(
            latent
        )
    )

    print(
        "State:",
        state_output.reconstruction.shape,
    )

    print(
        "Uncertainty:",
        (
            state_output.uncertainty.shape
            if state_output.uncertainty is not None
            else None
        ),
    )

    print("\n2. Image reconstruction")

    images = decoder.decode_image(
        latent
    )

    print(
        "Images:",
        images.shape,
    )

    print("\n3. Object reconstruction")

    objects = decoder.decode_objects(
        latent
    )

    print(
        "Objects:",
        objects.shape,
    )

    print("\n4. Sequence reconstruction")

    latent_sequence = torch.randn(
        2,
        8,
        config.latent_dim,
    )

    reconstructed_sequence = (
        decoder.decode_sequence(
            latent_sequence
        )
    )

    print(
        "Sequence:",
        reconstructed_sequence.shape,
    )

    print("\n5. Reconstruction training step")

    decoder.train()

    predictions = decoder(
        latent,
        decode_image=True,
    )

    targets = {
        "state": torch.randn(
            4,
            config.state_dim,
        ),
        "objects": torch.randn(
            4,
            config.object_dim,
        ),
        "image": torch.rand(
            4,
            config.image_channels,
            config.image_height,
            config.image_width,
        ),
    }

    loss_function = ReconstructionLoss()

    losses = loss_function(
        predictions,
        targets,
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()

    losses["total_loss"].backward()

    optimizer.step()

    print(
        "Training loss:",
        float(
            losses["total_loss"].item()
        ),
    )

    print("\n6. World-model adapter")

    adapter = WorldModelDecoderAdapter(
        decoder
    )

    predicted_state = (
        adapter.predict_observation(
            latent
        )
    )

    print(
        "Predicted state:",
        predicted_state.shape,
    )

    print(
        "\nLatent decoder demonstration completed"
    )


if __name__ == "__main__":
    main()
