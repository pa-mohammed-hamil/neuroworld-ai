
"""
NeuroWorld AI
Module: neuroworld/latent/latent_regularizer.py

Latent-space regularization for neural world models.
Compatible with PyTorch.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LatentRegularizerConfig:
    kl_weight: float = 0.01
    variance_weight: float = 0.1
    covariance_weight: float = 0.01
    temporal_weight: float = 0.05
    norm_weight: float = 0.001

    min_std: float = 1.0
    max_norm: float = 10.0
    eps: float = 1e-6


class LatentRegularizer(nn.Module):
    """
    Regularization for learned latent representations.

    Supports:
    - KL divergence against a standard normal prior
    - Variance floor to discourage representation collapse
    - Covariance decorrelation
    - Temporal smoothness
    - Latent norm constraints

    Input:
        z: [batch, latent_dim]
           or [batch, time, latent_dim]
    """

    def __init__(
        self,
        config: Optional[LatentRegularizerConfig] = None,
    ):
        super().__init__()
        self.config = config or LatentRegularizerConfig()

    def kl_divergence(
        self,
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        if mu.shape != logvar.shape:
            raise ValueError(
                "mu and logvar must have identical shapes"
            )

        logvar = logvar.clamp(-30.0, 20.0)

        kl = -0.5 * (
            1.0 + logvar - mu.pow(2) - logvar.exp()
        )

        return kl.sum(dim=-1).mean()

    def variance_loss(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])

        if flat.shape[0] < 2:
            return flat.new_zeros(())

        std = torch.sqrt(
            flat.var(dim=0, unbiased=False)
            + self.config.eps
        )

        return F.relu(
            self.config.min_std - std
        ).mean()

    def covariance_loss(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        n, d = flat.shape

        if n < 2 or d < 2:
            return flat.new_zeros(())

        centered = flat - flat.mean(
            dim=0, keepdim=True
        )

        covariance = (
            centered.T @ centered
        ) / (n - 1)

        off_diagonal = covariance - torch.diag(
            torch.diagonal(covariance)
        )

        return off_diagonal.pow(2).sum() / (
            d * (d - 1)
        )

    def temporal_loss(self, z: Tensor) -> Tensor:
        if z.ndim != 3 or z.shape[1] < 2:
            return z.new_zeros(())

        differences = z[:, 1:, :] - z[:, :-1, :]

        return differences.pow(2).mean()

    def norm_loss(self, z: Tensor) -> Tensor:
        norms = torch.linalg.vector_norm(
            z, dim=-1
        )

        return F.relu(
            norms - self.config.max_norm
        ).pow(2).mean()

    def forward(
        self,
        z: Tensor,
        mu: Optional[Tensor] = None,
        logvar: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:

        if z.ndim not in (2, 3):
            raise ValueError(
                "z must have shape [B, D] or [B, T, D]"
            )

        if not torch.is_floating_point(z):
            raise TypeError("z must be floating point")

        if not torch.isfinite(z).all().item():
            raise ValueError(
                "z contains NaN or infinite values"
            )

        cfg = self.config

        variance = self.variance_loss(z)
        covariance = self.covariance_loss(z)
        temporal = self.temporal_loss(z)
        norm = self.norm_loss(z)

        kl = z.new_zeros(())

        if (mu is None) != (logvar is None):
            raise ValueError(
                "Provide both mu and logvar, or neither"
            )

        if mu is not None and logvar is not None:
            if mu.shape != z.shape:
                raise ValueError(
                    "mu and logvar must match z shape"
                )
            kl = self.kl_divergence(mu, logvar)

        total = (
            cfg.kl_weight * kl
            + cfg.variance_weight * variance
            + cfg.covariance_weight * covariance
            + cfg.temporal_weight * temporal
            + cfg.norm_weight * norm
        )

        return {
            "total_loss": total,
            "kl_loss": kl,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "temporal_loss": temporal,
            "norm_loss": norm,
        }


if __name__ == "__main__":
    torch.manual_seed(42)

    config = LatentRegularizerConfig()
    regularizer = LatentRegularizer(config)

    batch_size = 8
    sequence_length = 12
    latent_dim = 128

    mu = torch.randn(
        batch_size, sequence_length, latent_dim,
        requires_grad=True,
    )
    logvar = torch.zeros_like(mu)

    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn_like(std)

    losses = regularizer(
        z=z,
        mu=mu,
        logvar=logvar,
    )

    losses["total_loss"].backward()

    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    print("Gradient available:", mu.grad is not None)
