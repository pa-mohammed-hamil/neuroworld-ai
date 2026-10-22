
"""
NeuroWorld AI
File: neuroworld/latent/state_compressor.py

Neural Latent State Compression Engine

Features:
- Learnable neural compression
- Latent dimensionality reduction
- State reconstruction
- Temporal sequence support
- Vector quantization
- Straight-through gradient estimation
- Compression ratio calculation
- Reconstruction quality metrics
- CPU / GPU compatibility
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class StateCompressorConfig:
    input_dim: int = 512
    compressed_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1

    use_quantization: bool = False
    codebook_size: int = 512
    commitment_weight: float = 0.25
    eps: float = 1e-8


@dataclass
class CompressionOutput:
    compressed: Tensor
    reconstructed: Tensor
    reconstruction_loss: Tensor
    quantization_loss: Tensor
    total_loss: Tensor
    code_indices: Optional[Tensor] = None


class VectorQuantizer(nn.Module):
    """
    Learnable vector quantization with a
    straight-through gradient estimator.
    """

    def __init__(
        self,
        dimension: int,
        codebook_size: int,
        commitment_weight: float = 0.25,
    ):
        super().__init__()

        if dimension <= 0 or codebook_size <= 1:
            raise ValueError(
                "Invalid vector quantizer configuration"
            )

        self.dimension = dimension
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight

        self.embedding = nn.Embedding(
            codebook_size,
            dimension,
        )

        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(
        self,
        z: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        if z.shape[-1] != self.dimension:
            raise ValueError(
                "Incorrect quantization dimension"
            )

        flat = z.reshape(-1, self.dimension)

        codebook = self.embedding.weight

        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ codebook.T
            + codebook.pow(2).sum(dim=1)
        )

        indices = distances.argmin(dim=1)

        quantized = self.embedding(indices)
        quantized = quantized.reshape_as(z)

        codebook_loss = F.mse_loss(
            quantized,
            z.detach(),
        )

        commitment_loss = F.mse_loss(
            z,
            quantized.detach(),
        )

        quantization_loss = (
            codebook_loss
            + self.commitment_weight * commitment_loss
        )

        quantized_st = (
            z + (quantized - z).detach()
        )

        indices = indices.reshape(z.shape[:-1])

        return (
            quantized_st,
            quantization_loss,
            indices,
        )


class StateCompressor(nn.Module):
    """
    Neural compression for latent world states.

    Input shapes:
        [batch, input_dim]
        [batch, time, input_dim]

    Compressed shapes:
        [batch, compressed_dim]
        [batch, time, compressed_dim]
    """

    def __init__(
        self,
        config: Optional[StateCompressorConfig] = None,
    ):
        super().__init__()

        self.config = config or StateCompressorConfig()
        cfg = self.config

        if (
            cfg.input_dim <= 0
            or cfg.compressed_dim <= 0
            or cfg.hidden_dim <= 0
        ):
            raise ValueError(
                "All dimensions must be positive"
            )

        if cfg.compressed_dim >= cfg.input_dim:
            raise ValueError(
                "compressed_dim must be smaller than input_dim"
            )

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),

            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),

            nn.Linear(cfg.hidden_dim, cfg.compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.compressed_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),

            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),

            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

        self.quantizer = (
            VectorQuantizer(
                dimension=cfg.compressed_dim,
                codebook_size=cfg.codebook_size,
                commitment_weight=cfg.commitment_weight,
            )
            if cfg.use_quantization
            else None
        )

    def _validate_input(self, x: Tensor) -> None:

        if x.ndim not in (2, 3):
            raise ValueError(
                "Expected [B,D] or [B,T,D]"
            )

        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, "
                f"received {x.shape[-1]}"
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "Input must be a floating-point tensor"
            )

    def compress(
        self,
        x: Tensor,
        quantize: bool = True,
    ) -> Tensor:
        """
        Encode input into compressed latent space.
        """

        self._validate_input(x)

        z = self.encoder(x)

        if quantize and self.quantizer is not None:
            z, _, _ = self.quantizer(z)

        return z

    def decompress(
        self,
        compressed: Tensor,
    ) -> Tensor:
        """
        Reconstruct original latent features.
        """

        if compressed.shape[-1] != (
            self.config.compressed_dim
        ):
            raise ValueError(
                "Invalid compressed dimension"
            )

        return self.decoder(compressed)

    def forward(
        self,
        x: Tensor,
    ) -> CompressionOutput:

        self._validate_input(x)

        compressed = self.encoder(x)

        quantization_loss = x.new_zeros(())
        code_indices = None

        if self.quantizer is not None:
            compressed, quantization_loss, code_indices = (
                self.quantizer(compressed)
            )

        reconstructed = self.decoder(compressed)

        reconstruction_loss = F.mse_loss(
            reconstructed,
            x,
        )

        total_loss = (
            reconstruction_loss + quantization_loss
        )

        return CompressionOutput(
            compressed=compressed,
            reconstructed=reconstructed,
            reconstruction_loss=reconstruction_loss,
            quantization_loss=quantization_loss,
            total_loss=total_loss,
            code_indices=code_indices,
        )

    def compression_ratio(self) -> float:
        """
        Theoretical feature-dimensionality ratio.

        Assumes identical element precision and excludes
        model parameters, codebooks, and storage overhead.
        """

        return (
            self.config.input_dim
            / self.config.compressed_dim
        )

    @torch.no_grad()
    def evaluate(
        self,
        x: Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        """

        was_training = self.training
        self.eval()

        try:
            output = self.forward(x)

            mse = F.mse_loss(
                output.reconstructed,
                x,
            )

            mae = F.l1_loss(
                output.reconstructed,
                x,
            )

            cosine_similarity = F.cosine_similarity(
                output.reconstructed,
                x,
                dim=-1,
            ).mean()

            return {
                "mse": float(mse.item()),
                "mae": float(mae.item()),
                "cosine_similarity": float(
                    cosine_similarity.item()
                ),
                "compression_ratio": (
                    self.compression_ratio()
                ),
            }
        finally:
            self.train(was_training)


if __name__ == "__main__":

    torch.manual_seed(42)

    config = StateCompressorConfig(
        input_dim=512,
        compressed_dim=128,
        hidden_dim=256,
        use_quantization=True,
        codebook_size=512,
    )

    model = StateCompressor(config)

    observations = torch.randn(
        8, 12, 512
    )

    output = model(observations)

    print("Input:", observations.shape)
    print("Compressed:", output.compressed.shape)
    print("Reconstructed:", output.reconstructed.shape)

    print(
        "Compression ratio:",
        model.compression_ratio(),
    )

    print(
        "Total loss:",
        output.total_loss.item(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()
    output.total_loss.backward()
    optimizer.step()

    metrics = model.evaluate(observations)

    print("Evaluation:", metrics)
