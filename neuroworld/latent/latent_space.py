
"""
NeuroWorld AI
File: neuroworld/latent/latent_space.py

Latent Space Engine for Neural World Models.

Features:
- Latent state representation
- Variational latent sampling
- Latent normalization
- Latent interpolation
- Latent similarity search
- Latent distance calculation
- Latent trajectory management
- Optional latent regularization
- GPU / CPU support
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LatentSpaceConfig:
    input_dim: int = 512
    latent_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    variational: bool = True
    normalize_latent: bool = False
    logvar_min: float = -20.0
    logvar_max: float = 10.0


@dataclass
class LatentState:
    z: Tensor
    mu: Optional[Tensor] = None
    logvar: Optional[Tensor] = None
    reconstruction: Optional[Tensor] = None


class LatentSpace(nn.Module):
    """
    Neural latent-space representation engine.

    Supports:
        [B, input_dim]
        [B, T, input_dim]

    Output:
        [B, latent_dim]
        [B, T, latent_dim]
    """

    def __init__(
        self,
        config: Optional[LatentSpaceConfig] = None,
    ):
        super().__init__()

        self.config = config or LatentSpaceConfig()
        cfg = self.config

        if cfg.input_dim <= 0 or cfg.latent_dim <= 0:
            raise ValueError(
                "Input and latent dimensions must be positive"
            )

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )

        self.mu_head = nn.Linear(
            cfg.hidden_dim, cfg.latent_dim
        )

        self.logvar_head = nn.Linear(
            cfg.hidden_dim, cfg.latent_dim
        )

        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    def encode(
        self,
        x: Tensor,
        deterministic: bool = False,
    ) -> LatentState:
        """
        Encode input observations into latent states.
        """

        if x.ndim not in (2, 3):
            raise ValueError(
                "Input must be [B,D] or [B,T,D]"
            )

        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input dimension "
                f"{self.config.input_dim}, "
                f"received {x.shape[-1]}"
            )

        hidden = self.encoder(x)

        mu = self.mu_head(hidden)

        if self.config.variational:
            logvar = self.logvar_head(hidden).clamp(
                self.config.logvar_min,
                self.config.logvar_max,
            )

            if deterministic:
                z = mu
            else:
                z = self.reparameterize(mu, logvar)
        else:
            logvar = None
            z = mu

        if self.config.normalize_latent:
            z = F.normalize(z, p=2, dim=-1)

        return LatentState(
            z=z,
            mu=mu,
            logvar=logvar,
        )

    @staticmethod
    def reparameterize(
        mu: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        Differentiable Gaussian latent sampling.
        """

        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)

        return mu + noise * std

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent vectors into feature space.
        """

        if z.shape[-1] != self.config.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        return self.decoder(z)

    def forward(
        self,
        x: Tensor,
        deterministic: bool = False,
    ) -> LatentState:

        state = self.encode(
            x,
            deterministic=deterministic,
        )

        state.reconstruction = self.decode(state.z)

        return state

    def sample(
        self,
        batch_size: int,
        sequence_length: Optional[int] = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Sample random latent states from a Gaussian prior.
        """

        if batch_size <= 0 or temperature < 0:
            raise ValueError(
                "Invalid batch size or temperature"
            )

        shape = [batch_size]

        if sequence_length is not None:
            if sequence_length <= 0:
                raise ValueError(
                    "sequence_length must be positive"
                )
            shape.append(sequence_length)

        shape.append(self.config.latent_dim)

        reference = self.mu_head.weight

        z = torch.randn(
            *shape,
            device=reference.device,
            dtype=reference.dtype,
        ) * temperature

        if self.config.normalize_latent:
            z = F.normalize(z, dim=-1)

        return z

    @staticmethod
    def interpolate(
        start: Tensor,
        end: Tensor,
        steps: int = 10,
    ) -> Tensor:
        """
        Linear interpolation between latent states.

        Result shape:
            [steps, *start.shape]
        """

        if start.shape != end.shape:
            raise ValueError(
                "Latent states must have identical shapes"
            )

        if steps < 2:
            raise ValueError(
                "Interpolation requires at least two steps"
            )

        alpha = torch.linspace(
            0.0,
            1.0,
            steps,
            device=start.device,
            dtype=start.dtype,
        )

        view_shape = (
            steps,
            *([1] * start.ndim),
        )

        alpha = alpha.reshape(view_shape)

        return (
            (1.0 - alpha) * start.unsqueeze(0)
            + alpha * end.unsqueeze(0)
        )

    @staticmethod
    def distance(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        """
        Euclidean distance along latent dimensions.
        """

        if z1.shape != z2.shape:
            raise ValueError(
                "Latent states must have matching shapes"
            )

        return torch.linalg.vector_norm(
            z1 - z2,
            dim=-1,
        )

    @staticmethod
    def similarity(
        z1: Tensor,
        z2: Tensor,
    ) -> Tensor:
        """
        Cosine similarity between latent states.
        """

        return F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

    @staticmethod
    def nearest_neighbors(
        query: Tensor,
        memory: Tensor,
        top_k: int = 5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Search stored latent vectors using cosine similarity.

        query:  [Q, D]
        memory: [N, D]

        Returns:
            scores:  [Q, K]
            indices: [Q, K]
        """

        if query.ndim != 2 or memory.ndim != 2:
            raise ValueError(
                "Query and memory must be 2D tensors"
            )

        if query.shape[-1] != memory.shape[-1]:
            raise ValueError(
                "Latent dimensions do not match"
            )

        if memory.shape[0] == 0:
            raise ValueError("Memory cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_normalized = F.normalize(
            query, dim=-1
        )

        memory_normalized = F.normalize(
            memory, dim=-1
        )

        scores = (
            query_normalized @ memory_normalized.T
        )

        k = min(top_k, memory.shape[0])

        return torch.topk(
            scores,
            k=k,
            dim=-1,
        )

    @staticmethod
    def trajectory_velocity(
        trajectory: Tensor,
    ) -> Tensor:
        """
        Calculate changes between consecutive latent states.

        trajectory: [B, T, D]
        """

        if trajectory.ndim != 3:
            raise ValueError(
                "Trajectory must be [B,T,D]"
            )

        return (
            trajectory[:, 1:, :]
            - trajectory[:, :-1, :]
        )

    @staticmethod
    def trajectory_acceleration(
        trajectory: Tensor,
    ) -> Tensor:
        """
        Calculate second-order latent differences.
        """

        if trajectory.ndim != 3:
            raise ValueError(
                "Trajectory must be [B,T,D]"
            )

        velocity = LatentSpace.trajectory_velocity(
            trajectory
        )

        return (
            velocity[:, 1:, :]
            - velocity[:, :-1, :]
        )

    def compute_loss(
        self,
        state: LatentState,
        target: Tensor,
        regularizer: Optional[nn.Module] = None,
        reconstruction_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Compute reconstruction and optional regularization loss.

        Pass LatentRegularizer to include KL divergence,
        variance, covariance, and temporal penalties.
        """

        if state.reconstruction is None:
            raise ValueError(
                "Latent state has no reconstruction"
            )

        reconstruction_loss = F.mse_loss(
            state.reconstruction,
            target,
        )

        total = (
            reconstruction_weight * reconstruction_loss
        )

        results = {
            "reconstruction_loss": reconstruction_loss,
        }

        if regularizer is not None:
            regularization = regularizer(
                z=state.z,
                mu=state.mu,
                logvar=state.logvar,
            )

            total = (
                total + regularization["total_loss"]
            )

            results.update(regularization)

        results["total_loss"] = total

        return results


if __name__ == "__main__":
    torch.manual_seed(42)

    config = LatentSpaceConfig(
        input_dim=512,
        latent_dim=128,
        hidden_dim=256,
        variational=True,
    )

    model = LatentSpace(config)

    observations = torch.randn(8, 12, 512)

    output = model(observations)

    print("Input:", observations.shape)
    print("Latent:", output.z.shape)
    print("Reconstruction:", output.reconstruction.shape)

    losses = model.compute_loss(
        state=output,
        target=observations,
    )

    losses["total_loss"].backward()

    print(
        "Reconstruction loss:",
        losses["reconstruction_loss"].item(),
    )

    random_states = model.sample(
        batch_size=4,
        sequence_length=12,
    )

    print("Sampled latent:", random_states.shape)

    scores, indices = model.nearest_neighbors(
        query=output.z[:, 0, :],
        memory=random_states.reshape(-1, 128),
        top_k=3,
    )

    print("Neighbor scores:", scores.shape)
    print("Neighbor indices:", indices.shape)
