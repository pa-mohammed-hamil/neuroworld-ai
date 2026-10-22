
"""
NeuroWorld AI
File: neuroworld/latent/encoder.py

Neural Latent State Encoder

Features:
    - Vector observation encoding
    - Image feature encoding
    - Motion feature encoding
    - Multimodal feature fusion
    - Deterministic latent encoding
    - Variational latent encoding
    - Reparameterization trick
    - KL divergence regularization
    - Feature normalization
    - Residual neural blocks
    - Latent projection
    - Optional modality masking
    - Batched inference
    - Device management
    - Training-compatible forward pass
    - World-model integration

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.latent.encoder
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class LatentEncoderError(Exception):
    pass


class LatentEncoderValidationError(
    LatentEncoderError
):
    pass


# ============================================================
# ENUMERATIONS
# ============================================================

class EncoderMode(str, Enum):
    DETERMINISTIC = "deterministic"
    VARIATIONAL = "variational"


class FusionMode(str, Enum):
    CONCAT = "concat"
    GATED = "gated"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LatentEncoderConfig:
    observation_dim: int = 128
    image_feature_dim: int = 256
    motion_feature_dim: int = 64

    latent_dim: int = 128
    hidden_dim: int = 256

    modality_dim: int = 128

    num_residual_blocks: int = 2

    dropout: float = 0.1

    mode: EncoderMode = (
        EncoderMode.DETERMINISTIC
    )

    fusion: FusionMode = (
        FusionMode.GATED
    )

    normalize_latent: bool = False

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    kl_weight: float = 0.001

    device: str = "cpu"

    def __post_init__(self) -> None:
        dimensions = (
            self.observation_dim,
            self.image_feature_dim,
            self.motion_feature_dim,
            self.latent_dim,
            self.hidden_dim,
            self.modality_dim,
        )

        if any(value <= 0 for value in dimensions):
            raise ValueError(
                "All dimensions must be positive"
            )

        if self.num_residual_blocks < 0:
            raise ValueError(
                "num_residual_blocks must be nonnegative"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "Invalid log variance bounds"
            )

        if self.kl_weight < 0:
            raise ValueError(
                "kl_weight must be nonnegative"
            )


# ============================================================
# OUTPUT STRUCTURE
# ============================================================

@dataclass
class LatentEncoderOutput:
    latent: Tensor

    mean: Tensor | None = None
    logvar: Tensor | None = None

    kl_loss: Tensor | None = None

    modality_weights: Tensor | None = None

    fused_features: Tensor | None = None

    def to_dict(
        self,
        detach: bool = False,
    ) -> dict[str, Any]:
        result = {
            "latent": self.latent,
            "mean": self.mean,
            "logvar": self.logvar,
            "kl_loss": self.kl_loss,
            "modality_weights": (
                self.modality_weights
            ),
            "fused_features": (
                self.fused_features
            ),
        }

        if not detach:
            return result

        return {
            key: (
                value.detach().cpu()
                if isinstance(value, Tensor)
                else value
            )
            for key, value in result.items()
        }


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualMLPBlock(nn.Module):
    """
    Residual feature transformation.

    Input:
        [B,D]

    Output:
        [B,D]
    """

    def __init__(
        self,
        dimension: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
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
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return x + self.network(x)


# ============================================================
# MODALITY ENCODER
# ============================================================

class ModalityEncoder(nn.Module):
    """
    Projects a single input modality
    into a common embedding dimension.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        features: Tensor,
    ) -> Tensor:
        return self.network(
            features
        )


# ============================================================
# GATED MULTIMODAL FUSION
# ============================================================

class GatedMultimodalFusion(nn.Module):
    """
    Fuses three modality embeddings:

        observation
        image
        motion

    Missing modalities are masked out.

    Gate weights are normalized using
    softmax over available modalities.
    """

    def __init__(
        self,
        modality_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(
                modality_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                3,
            ),
        )

        self.projection = nn.Sequential(
            nn.Linear(
                modality_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        embeddings: Tensor,
        modality_mask: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """
        embeddings:
            [B,3,D]

        modality_mask:
            [B,3] boolean
        """
        batch_size = embeddings.shape[0]

        flattened = embeddings.reshape(
            batch_size,
            -1,
        )

        logits = self.gate(
            flattened
        )

        logits = logits.masked_fill(
            ~modality_mask,
            torch.finfo(
                logits.dtype
            ).min,
        )

        weights = torch.softmax(
            logits,
            dim=-1,
        )

        fused = torch.sum(
            embeddings
            * weights.unsqueeze(-1),
            dim=1,
        )

        return (
            self.projection(fused),
            weights,
        )


# ============================================================
# CONCATENATION FUSION
# ============================================================

class ConcatenationFusion(nn.Module):
    """
    Concatenates masked modality
    embeddings and projects them.
    """

    def __init__(
        self,
        modality_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                modality_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        embeddings: Tensor,
        modality_mask: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        masked = (
            embeddings
            * modality_mask.unsqueeze(-1)
        )

        fused = self.network(
            masked.flatten(start_dim=1)
        )

        weights = modality_mask.to(
            embeddings.dtype
        )

        weights = (
            weights
            / weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
        )

        return fused, weights


# ============================================================
# LATENT ENCODER
# ============================================================

class LatentEncoder(nn.Module):
    """
    Main multimodal latent encoder.

    Accepted inputs:
        observations: [B,observation_dim]
        image_features: [B,image_feature_dim]
        motion_features: [B,motion_feature_dim]

    At least one modality is required.

    Output:
        latent: [B,latent_dim]

    Variational mode additionally returns:
        mean
        logvar
        KL divergence
    """

    MODALITIES = (
        "observations",
        "image_features",
        "motion_features",
    )

    def __init__(
        self,
        config: LatentEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or LatentEncoderConfig()
        )

        cfg = self.config

        self.observation_encoder = (
            ModalityEncoder(
                cfg.observation_dim,
                cfg.modality_dim,
                cfg.hidden_dim,
                cfg.dropout,
            )
        )

        self.image_encoder = (
            ModalityEncoder(
                cfg.image_feature_dim,
                cfg.modality_dim,
                cfg.hidden_dim,
                cfg.dropout,
            )
        )

        self.motion_encoder = (
            ModalityEncoder(
                cfg.motion_feature_dim,
                cfg.modality_dim,
                cfg.hidden_dim,
                cfg.dropout,
            )
        )

        if cfg.fusion == FusionMode.GATED:
            self.fusion = GatedMultimodalFusion(
                cfg.modality_dim,
                cfg.hidden_dim,
            )
        else:
            self.fusion = ConcatenationFusion(
                cfg.modality_dim,
                cfg.hidden_dim,
            )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    cfg.hidden_dim,
                    cfg.dropout,
                )
                for _ in range(
                    cfg.num_residual_blocks
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            cfg.hidden_dim
        )

        if cfg.mode == EncoderMode.VARIATIONAL:
            self.mean_head = nn.Linear(
                cfg.hidden_dim,
                cfg.latent_dim,
            )

            self.logvar_head = nn.Linear(
                cfg.hidden_dim,
                cfg.latent_dim,
            )

            self.latent_head = None

        else:
            self.latent_head = nn.Linear(
                cfg.hidden_dim,
                cfg.latent_dim,
            )

            self.mean_head = None
            self.logvar_head = None

    def _validate_modality(
        self,
        tensor: Tensor | None,
        expected_dim: int,
        name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if tensor is None:
            return torch.zeros(
                batch_size,
                expected_dim,
                device=device,
                dtype=dtype,
            )

        if (
            tensor.ndim != 2
            or tensor.shape != (
                batch_size,
                expected_dim,
            )
        ):
            raise LatentEncoderValidationError(
                f"{name} must be "
                f"[B,{expected_dim}]"
            )

        if tensor.device != device:
            raise LatentEncoderValidationError(
                f"{name} is on the wrong device"
            )

        if not torch.isfinite(tensor).all():
            raise LatentEncoderValidationError(
                f"{name} contains non-finite values"
            )

        return tensor.to(dtype=dtype)

    def _prepare_inputs(
        self,
        observations: Tensor | None,
        image_features: Tensor | None,
        motion_features: Tensor | None,
        modality_mask: Tensor | None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        supplied = [
            tensor
            for tensor in (
                observations,
                image_features,
                motion_features,
            )
            if tensor is not None
        ]

        if not supplied:
            raise LatentEncoderValidationError(
                "At least one modality is required"
            )

        first = supplied[0]

        if first.ndim != 2:
            raise LatentEncoderValidationError(
                "Inputs must be batched [B,D]"
            )

        batch_size = first.shape[0]

        if batch_size == 0:
            raise LatentEncoderValidationError(
                "Empty batch is not supported"
            )

        parameter = next(
            self.parameters()
        )

        device = parameter.device
        dtype = parameter.dtype

        availability = torch.tensor(
            [
                observations is not None,
                image_features is not None,
                motion_features is not None,
            ],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0).expand(
            batch_size,
            -1,
        )

        if modality_mask is None:
            mask = availability
        else:
            if modality_mask.shape != (
                batch_size,
                3,
            ):
                raise LatentEncoderValidationError(
                    "modality_mask must be [B,3]"
                )

            mask = modality_mask.to(
                device=device,
                dtype=torch.bool,
            )

            if (
                mask & ~availability
            ).any():
                raise LatentEncoderValidationError(
                    "Mask enables missing modality"
                )

        if not mask.any(dim=1).all():
            raise LatentEncoderValidationError(
                "Every sample needs a modality"
            )

        observation_tensor = (
            self._validate_modality(
                observations,
                self.config.observation_dim,
                "observations",
                batch_size,
                device,
                dtype,
            )
        )

        image_tensor = (
            self._validate_modality(
                image_features,
                self.config.image_feature_dim,
                "image_features",
                batch_size,
                device,
                dtype,
            )
        )

        motion_tensor = (
            self._validate_modality(
                motion_features,
                self.config.motion_feature_dim,
                "motion_features",
                batch_size,
                device,
                dtype,
            )
        )

        return (
            observation_tensor,
            image_tensor,
            motion_tensor,
            mask,
        )

    @staticmethod
    def reparameterize(
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        std = torch.exp(
            0.5 * logvar
        )

        noise = torch.randn_like(
            std
        )

        return mean + noise * std

    @staticmethod
    def kl_divergence(
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        KL(q(z|x) || N(0,I))

        Mean across the batch,
        sum across latent dimensions.
        """
        return (
            -0.5
            * (
                1
                + logvar
                - mean.square()
                - logvar.exp()
            ).sum(dim=-1).mean()
        )

    def forward(
        self,
        observations: Tensor | None = None,
        image_features: Tensor | None = None,
        motion_features: Tensor | None = None,
        *,
        modality_mask: Tensor | None = None,
        sample: bool | None = None,
    ) -> LatentEncoderOutput:
        (
            observations,
            image_features,
            motion_features,
            mask,
        ) = self._prepare_inputs(
            observations,
            image_features,
            motion_features,
            modality_mask,
        )

        observation_embedding = (
            self.observation_encoder(
                observations
            )
        )

        image_embedding = (
            self.image_encoder(
                image_features
            )
        )

        motion_embedding = (
            self.motion_encoder(
                motion_features
            )
        )

        embeddings = torch.stack(
            [
                observation_embedding,
                image_embedding,
                motion_embedding,
            ],
            dim=1,
        )

        # Mask before fusion so unavailable
        # modalities cannot influence gates.
        embeddings = (
            embeddings
            * mask.unsqueeze(-1)
        )

        fused, modality_weights = (
            self.fusion(
                embeddings,
                mask,
            )
        )

        hidden = self.residual_blocks(
            fused
        )

        hidden = self.final_norm(
            hidden
        )

        if (
            self.config.mode
            == EncoderMode.DETERMINISTIC
        ):
            latent = self.latent_head(
                hidden
            )

            if self.config.normalize_latent:
                latent = F.normalize(
                    latent,
                    p=2,
                    dim=-1,
                )

            return LatentEncoderOutput(
                latent=latent,
                modality_weights=(
                    modality_weights
                ),
                fused_features=hidden,
            )

        mean = self.mean_head(
            hidden
        )

        logvar = self.logvar_head(
            hidden
        ).clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )

        should_sample = (
            self.training
            if sample is None
            else sample
        )

        if should_sample:
            latent = self.reparameterize(
                mean,
                logvar,
            )
        else:
            latent = mean

        if self.config.normalize_latent:
            latent = F.normalize(
                latent,
                p=2,
                dim=-1,
            )

        kl_loss = self.kl_divergence(
            mean,
            logvar,
        )

        return LatentEncoderOutput(
            latent=latent,
            mean=mean,
            logvar=logvar,
            kl_loss=kl_loss,
            modality_weights=(
                modality_weights
            ),
            fused_features=hidden,
        )

    @torch.no_grad()
    def encode(
        self,
        observations: Tensor | None = None,
        image_features: Tensor | None = None,
        motion_features: Tensor | None = None,
        *,
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Deterministic inference interface.

        Variational mode returns the mean.
        """
        was_training = self.training

        try:
            self.eval()

            output = self.forward(
                observations,
                image_features,
                motion_features,
                modality_mask=modality_mask,
                sample=False,
            )

            return output.latent
        finally:
            self.train(
                was_training
            )


# ============================================================
# NUMPY ADAPTER
# ============================================================

class LatentEncoderAdapter:
    """
    Convenience adapter for NumPy inputs.
    """

    def __init__(
        self,
        encoder: LatentEncoder,
    ) -> None:
        self.encoder = encoder

    @staticmethod
    def _convert(
        value: np.ndarray | None,
        device: torch.device,
    ) -> Tensor | None:
        if value is None:
            return None

        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=device,
        )

    def encode(
        self,
        observations: np.ndarray | None = None,
        image_features: np.ndarray | None = None,
        motion_features: np.ndarray | None = None,
    ) -> np.ndarray:
        device = next(
            self.encoder.parameters()
        ).device

        latent = self.encoder.encode(
            observations=self._convert(
                observations,
                device,
            ),
            image_features=self._convert(
                image_features,
                device,
            ),
            motion_features=self._convert(
                motion_features,
                device,
            ),
        )

        return (
            latent.detach()
            .cpu()
            .numpy()
        )


# ============================================================
# TRAINING LOSS
# ============================================================

class LatentEncoderLoss(nn.Module):
    """
    Combines a task-specific reconstruction
    or prediction loss with KL regularization.

    The reconstruction/prediction loss must
    be supplied by the caller.
    """

    def __init__(
        self,
        kl_weight: float = 0.001,
    ) -> None:
        super().__init__()

        if kl_weight < 0:
            raise ValueError(
                "kl_weight must be nonnegative"
            )

        self.kl_weight = kl_weight

    def forward(
        self,
        task_loss: Tensor,
        output: LatentEncoderOutput,
    ) -> dict[str, Tensor]:
        if task_loss.ndim != 0:
            raise LatentEncoderValidationError(
                "task_loss must be scalar"
            )

        kl = (
            output.kl_loss
            if output.kl_loss is not None
            else task_loss.new_zeros(())
        )

        total = (
            task_loss
            + self.kl_weight * kl
        )

        return {
            "total_loss": total,
            "task_loss": task_loss,
            "kl_loss": kl,
        }


# ============================================================
# FACTORY
# ============================================================

def create_latent_encoder(
    config: LatentEncoderConfig | None = None,
) -> LatentEncoder:
    cfg = config or LatentEncoderConfig()

    model = LatentEncoder(
        cfg
    )

    return model.to(
        cfg.device
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - LATENT ENCODER"
    )
    print("=" * 65)

    config = LatentEncoderConfig(
        observation_dim=128,
        image_feature_dim=256,
        motion_feature_dim=64,
        latent_dim=128,
        hidden_dim=256,
        modality_dim=128,
        mode=EncoderMode.VARIATIONAL,
        fusion=FusionMode.GATED,
    )

    encoder = create_latent_encoder(
        config
    )

    batch_size = 4

    observations = torch.randn(
        batch_size,
        config.observation_dim,
    )

    image_features = torch.randn(
        batch_size,
        config.image_feature_dim,
    )

    motion_features = torch.randn(
        batch_size,
        config.motion_feature_dim,
    )

    print("\n1. Multimodal encoding")

    output = encoder(
        observations=observations,
        image_features=image_features,
        motion_features=motion_features,
    )

    print(
        "Latent shape:",
        output.latent.shape,
    )

    print(
        "Mean shape:",
        output.mean.shape,
    )

    print(
        "Logvar shape:",
        output.logvar.shape,
    )

    print(
        "KL loss:",
        output.kl_loss.item(),
    )

    print(
        "Modality weights:",
        output.modality_weights,
    )

    print("\n2. Observation-only encoding")

    latent = encoder.encode(
        observations=observations
    )

    print(
        "Encoded shape:",
        latent.shape,
    )

    print("\n3. Gradient test")

    encoder.train()

    output = encoder(
        observations=observations,
        image_features=image_features,
        motion_features=motion_features,
    )

    task_loss = (
        output.latent.square().mean()
    )

    criterion = LatentEncoderLoss(
        kl_weight=config.kl_weight
    )

    losses = criterion(
        task_loss,
        output,
    )

    losses["total_loss"].backward()

    print(
        "Total loss:",
        losses["total_loss"].item(),
    )

    print(
        "Gradient available:",
        encoder.mean_head.weight.grad
        is not None,
    )

    print(
        "\nLatent encoder demonstration completed"
    )


if __name__ == "__main__":
    main()
