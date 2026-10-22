"""Reconstruction loss."""
import torch

def reconstruction_loss(prediction:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    if prediction.shape!=target.shape: raise ValueError('Shape mismatch')
    if prediction.numel()==0: raise ValueError('Empty tensors')
    return (prediction-target).square().mean()

"""
NeuroWorld AI
File: training/losses/reconstruction.py

Reconstruction losses for:
- World-model training
- Image and video autoencoders
- VAE / VQ-VAE reconstruction
- Latent representation learning
- Robot state reconstruction
- Multimodal feature reconstruction

Supported objectives:
- Mean squared error (MSE)
- Mean absolute error (MAE / L1)
- Huber loss
- Charbonnier loss
- Binary cross-entropy
- Gaussian negative log-likelihood
- Structural similarity (SSIM)
- Multi-scale reconstruction
- Temporal difference reconstruction
- Gradient/edge reconstruction
- Latent feature reconstruction

Features:
- Configurable component weights
- Image/video tensor support
- Sequence and spatial masks
- Discounted temporal reconstruction
- Custom reconstruction losses
- Numerical stability checks
- Per-component training metrics

Requirements:
    PyTorch >= 2.1

Tensor conventions:
    State: [B,D] or [B,T,D]
    Image: [B,C,H,W]
    Video: [B,T,C,H,W]

Image values for SSIM:
    [0, data_range] (normally [0,1])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ReconstructionLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class ReconstructionLossConfig:
    reconstruction_weight: float = 1.0
    ssim_weight: float = 0.0
    multiscale_weight: float = 0.0
    temporal_weight: float = 0.0
    gradient_weight: float = 0.0
    latent_weight: float = 0.0
    gaussian_nll_weight: float = 0.0

    loss_type: str = "mse"

    huber_delta: float = 1.0
    charbonnier_epsilon: float = 1e-3

    data_range: float = 1.0
    ssim_window_size: int = 7

    multiscale_levels: int = 3
    multiscale_decay: float = 0.5

    temporal_discount: float = 1.0

    min_logvar: float = -20.0
    max_logvar: float = 10.0

    detach_latent_target: bool = True
    normalize_weights: bool = False
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        if self.loss_type not in {
            "mse",
            "mae",
            "huber",
            "charbonnier",
            "bce",
        }:
            raise ValueError(
                f"Unsupported loss type: {self.loss_type}"
            )

        weights = (
            self.reconstruction_weight,
            self.ssim_weight,
            self.multiscale_weight,
            self.temporal_weight,
            self.gradient_weight,
            self.latent_weight,
            self.gaussian_nll_weight,
        )

        if any(
            not math.isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Loss weights must be finite "
                "and nonnegative"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if self.charbonnier_epsilon <= 0:
            raise ValueError(
                "charbonnier_epsilon must be positive"
            )

        if self.data_range <= 0:
            raise ValueError(
                "data_range must be positive"
            )

        if (
            self.ssim_window_size < 1
            or self.ssim_window_size % 2 == 0
        ):
            raise ValueError(
                "ssim_window_size must be positive "
                "and odd"
            )

        if self.multiscale_levels < 1:
            raise ValueError(
                "multiscale_levels must be >= 1"
            )

        if not 0 < self.multiscale_decay <= 1:
            raise ValueError(
                "multiscale_decay must be in (0,1]"
            )

        if not 0 < self.temporal_discount <= 1:
            raise ValueError(
                "temporal_discount must be in (0,1]"
            )

        if self.min_logvar >= self.max_logvar:
            raise ValueError(
                "min_logvar must be less than max_logvar"
            )


class ReconstructionLoss(nn.Module):
    """
    Multi-objective reconstruction loss.

    Expected outputs:
        reconstruction
        reconstructed_frames
        reconstructed_images
        reconstructed_states
        reconstruction_mean
        reconstruction_logvar
        reconstructed_latents

    Expected targets:
        reconstruction_target
        frames
        images
        states
        latent_targets
        mask
        sequence_mask
        pixel_mask

    Only available and enabled components
    contribute to the total loss.
    """

    def __init__(
        self,
        config: ReconstructionLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or ReconstructionLossConfig()
        )

        self.weights = {
            "reconstruction": (
                self.config.reconstruction_weight
            ),
            "ssim": self.config.ssim_weight,
            "multiscale": (
                self.config.multiscale_weight
            ),
            "temporal": (
                self.config.temporal_weight
            ),
            "gradient": (
                self.config.gradient_weight
            ),
            "latent": self.config.latent_weight,
            "gaussian_nll": (
                self.config.gaussian_nll_weight
            ),
        }

        self.custom_losses: dict[
            str, ReconstructionLossFunction
        ] = {}

        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _get(
        values: Mapping[str, Any],
        *keys: str,
    ) -> Tensor | None:
        for key in keys:
            value = values.get(key)

            if isinstance(value, Tensor):
                return value

        return None

    def _reconstruction(
        self,
        outputs: Mapping[str, Any],
    ) -> Tensor | None:
        return self._get(
            outputs,
            "reconstruction",
            "reconstructed_frames",
            "reconstructed_images",
            "reconstructed_states",
            "reconstruction_mean",
            "reconstructed",
        )

    def _target(
        self,
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        return self._get(
            targets,
            "reconstruction_target",
            "frames",
            "images",
            "states",
            "targets",
        )

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = ReconstructionLoss._get(
            targets,
            "pixel_mask",
            "mask",
            "sequence_mask",
            "frame_mask",
        )

        if mask is None:
            return None

        mask = mask.to(
            device=reference.device,
            dtype=torch.bool,
        )

        if mask.ndim == 0:
            return mask.expand_as(reference)

        if mask.shape[0] != reference.shape[0]:
            raise ValueError(
                "Mask batch dimension mismatch"
            )

        # For [B,T,C,H,W], allow [B,T,H,W].
        if (
            reference.ndim == 5
            and mask.ndim == 4
            and mask.shape[1] == reference.shape[1]
            and mask.shape[-2:] == reference.shape[-2:]
        ):
            mask = mask.unsqueeze(2)

        # For [B,C,H,W], allow [B,H,W].
        elif (
            reference.ndim == 4
            and mask.ndim == 3
            and mask.shape[-2:] == reference.shape[-2:]
        ):
            mask = mask.unsqueeze(1)

        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                reference.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Reconstruction mask cannot broadcast "
                f"to {tuple(reference.shape)}"
            ) from exc

    @staticmethod
    def _masked_mean(
        values: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is None:
            return values.mean()

        weights = mask.to(
            device=values.device,
            dtype=values.dtype,
        )

        return (
            (values * weights).sum()
            / weights.sum().clamp_min(1.0)
        )

    def _elementwise(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Reconstruction and target shapes differ: "
                f"{tuple(prediction.shape)} vs "
                f"{tuple(target.shape)}"
            )

        prediction = prediction.float()
        target = target.float()

        difference = prediction - target

        if self.config.loss_type == "mse":
            return difference.square()

        if self.config.loss_type == "mae":
            return difference.abs()

        if self.config.loss_type == "huber":
            return F.huber_loss(
                prediction,
                target,
                delta=self.config.huber_delta,
                reduction="none",
            )

        if self.config.loss_type == "charbonnier":
            epsilon = (
                self.config.charbonnier_epsilon
            )

            return torch.sqrt(
                difference.square() + epsilon**2
            ) - epsilon

        if self.config.loss_type == "bce":
            if not torch.all(
                (target >= 0) & (target <= 1)
            ).item():
                raise ValueError(
                    "BCE targets must be in [0,1]"
                )

            return F.binary_cross_entropy_with_logits(
                prediction,
                target,
                reduction="none",
            )

        raise RuntimeError(
            "Unreachable reconstruction loss type"
        )

    def _regression(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self._masked_mean(
            self._elementwise(
                prediction,
                target,
            ),
            mask,
        )

    @staticmethod
    def _flatten_video(
        tensor: Tensor,
    ) -> Tensor:
        if tensor.ndim == 4:
            return tensor

        if tensor.ndim == 5:
            batch, steps, channels, height, width = (
                tensor.shape
            )

            return tensor.reshape(
                batch * steps,
                channels,
                height,
                width,
            )

        raise ValueError(
            "Image/video must have shape "
            "[B,C,H,W] or [B,T,C,H,W]"
        )

    def reconstruction_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._reconstruction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def _ssim_map(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Differentiable local SSIM map.

        Uses average pooling to estimate
        local means and variances.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                "SSIM image shapes must match"
            )

        if prediction.ndim != 4:
            raise ValueError(
                "SSIM requires [N,C,H,W]"
            )

        window = self.config.ssim_window_size

        if min(prediction.shape[-2:]) < window:
            raise ValueError(
                "Image smaller than SSIM window"
            )

        prediction = prediction.float()
        target = target.float()

        padding = window // 2

        def average_pool(x: Tensor) -> Tensor:
            return F.avg_pool2d(
                x,
                kernel_size=window,
                stride=1,
                padding=padding,
                count_include_pad=False,
            )

        mu_x = average_pool(prediction)
        mu_y = average_pool(target)

        mu_x_sq = mu_x.square()
        mu_y_sq = mu_y.square()

        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            average_pool(prediction.square())
            - mu_x_sq
        ).clamp_min(0.0)

        sigma_y_sq = (
            average_pool(target.square())
            - mu_y_sq
        ).clamp_min(0.0)

        sigma_xy = (
            average_pool(prediction * target)
            - mu_xy
        )

        c1 = (
            0.01 * self.config.data_range
        ) ** 2

        c2 = (
            0.03 * self.config.data_range
        ) ** 2

        numerator = (
            (2.0 * mu_xy + c1)
            * (2.0 * sigma_xy + c2)
        )

        denominator = (
            (mu_x_sq + mu_y_sq + c1)
            * (
                sigma_x_sq
                + sigma_y_sq
                + c2
            )
        )

        return numerator / denominator.clamp_min(
            1e-12
        )

    def ssim_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._reconstruction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        if prediction.ndim not in (4, 5):
            return None

        prediction_images = self._flatten_video(
            prediction
        )

        target_images = self._flatten_video(
            target
        )

        ssim_map = self._ssim_map(
            prediction_images,
            target_images,
        )

        loss_map = (
            1.0 - ssim_map
        ).clamp_min(0.0) / 2.0

        mask = self._mask(
            targets,
            prediction,
        )

        if mask is not None:
            mask = self._flatten_video(
                mask.float()
            )

        return self._masked_mean(
            loss_map,
            mask,
        )

    def multiscale_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Multi-resolution reconstruction.

        Each scale uses average pooling.
        """
        prediction = self._reconstruction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        if prediction.ndim not in (4, 5):
            return None

        prediction = self._flatten_video(
            prediction
        ).float()

        target = self._flatten_video(
            target
        ).float()

        original_prediction = (
            self._reconstruction(outputs)
        )

        mask = self._mask(
            targets,
            original_prediction,
        )

        if mask is not None:
            mask = self._flatten_video(
                mask.float()
            )

        total = prediction.new_zeros(())
        total_weight = 0.0

        for level in range(
            self.config.multiscale_levels
        ):
            weight = (
                self.config.multiscale_decay**level
            )

            errors = self._elementwise(
                prediction,
                target,
            )

            total = total + weight * (
                self._masked_mean(
                    errors,
                    mask,
                )
            )

            total_weight += weight

            if level == (
                self.config.multiscale_levels - 1
            ):
                break

            if min(prediction.shape[-2:]) < 2:
                break

            prediction = F.avg_pool2d(
                prediction,
                kernel_size=2,
                stride=2,
            )

            target = F.avg_pool2d(
                target,
                kernel_size=2,
                stride=2,
            )

            if mask is not None:
                mask = (
                    F.avg_pool2d(
                        mask,
                        kernel_size=2,
                        stride=2,
                    ) > 0.999
                ).float()

        return total / total_weight

    def temporal_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Temporal difference reconstruction.

        Supports:
            [B,T,D]
            [B,T,C,H,W]
        """
        prediction = self._reconstruction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        if prediction.shape != target.shape:
            raise ValueError(
                "Temporal reconstruction shapes differ"
            )

        if (
            prediction.ndim not in (3, 5)
            or prediction.shape[1] < 2
        ):
            return None

        predicted_delta = (
            prediction[:, 1:]
            - prediction[:, :-1]
        )

        target_delta = (
            target[:, 1:]
            - target[:, :-1]
        )

        errors = self._elementwise(
            predicted_delta,
            target_delta,
        )

        steps = errors.shape[1]

        discount = torch.pow(
            torch.tensor(
                self.config.temporal_discount,
                device=errors.device,
                dtype=errors.dtype,
            ),
            torch.arange(
                steps,
                device=errors.device,
                dtype=errors.dtype,
            ),
        )

        discount = discount.reshape(
            1,
            steps,
            *([1] * (errors.ndim - 2)),
        ).expand_as(errors)

        mask = self._mask(
            targets,
            prediction,
        )

        if mask is not None:
            pair_mask = (
                mask[:, 1:]
                & mask[:, :-1]
            )

            discount = (
                discount
                * pair_mask.to(discount.dtype)
            )

        return (
            (errors * discount).sum()
            / discount.sum().clamp_min(1e-12)
        )

    @staticmethod
    def _image_gradients(
        images: Tensor,
    ) -> tuple[Tensor, Tensor]:
        horizontal = (
            images[..., :, 1:]
            - images[..., :, :-1]
        )

        vertical = (
            images[..., 1:, :]
            - images[..., :-1, :]
        )

        return horizontal, vertical

    def gradient_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Image gradient reconstruction.

        Encourages preservation of edges
        and fine spatial structure.
        """
        prediction = self._reconstruction(outputs)
        target = self._target(targets)

        if prediction is None or target is None:
            return None

        if prediction.ndim not in (4, 5):
            return None

        prediction_images = self._flatten_video(
            prediction
        )

        target_images = self._flatten_video(
            target
        )

        if min(prediction_images.shape[-2:]) < 2:
            return None

        predicted_dx, predicted_dy = (
            self._image_gradients(
                prediction_images
            )
        )

        target_dx, target_dy = (
            self._image_gradients(
                target_images
            )
        )

        mask = self._mask(
            targets,
            prediction,
        )

        mask_dx = None
        mask_dy = None

        if mask is not None:
            mask = self._flatten_video(
                mask.float()
            ).bool()

            mask_dx = (
                mask[..., :, 1:]
                & mask[..., :, :-1]
            )

            mask_dy = (
                mask[..., 1:, :]
                & mask[..., :-1, :]
            )

        horizontal_loss = self._regression(
            predicted_dx,
            target_dx,
            mask_dx,
        )

        vertical_loss = self._regression(
            predicted_dy,
            target_dy,
            mask_dy,
        )

        return 0.5 * (
            horizontal_loss
            + vertical_loss
        )

    def latent_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        reconstruction = self._get(
            outputs,
            "reconstructed_latents",
            "latent_reconstruction",
        )

        target = self._get(
            targets,
            "latent_targets",
            "latents",
        )

        if target is None:
            target = self._get(
                outputs,
                "target_latents",
            )

        if reconstruction is None or target is None:
            return None

        if self.config.detach_latent_target:
            target = target.detach()

        return self._regression(
            reconstruction,
            target,
            self._mask(
                targets,
                reconstruction,
            ),
        )

    def gaussian_nll_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Gaussian reconstruction NLL.

        The decoder predicts a mean
        and diagonal log variance.
        """
        mean = self._get(
            outputs,
            "reconstruction_mean",
        )

        logvar = self._get(
            outputs,
            "reconstruction_logvar",
        )

        target = self._target(targets)

        if (
            mean is None
            or logvar is None
            or target is None
        ):
            return None

        if not (
            mean.shape
            == logvar.shape
            == target.shape
        ):
            raise ValueError(
                "Gaussian reconstruction shapes differ"
            )

        mean = mean.float()
        target = target.float()

        logvar = logvar.float().clamp(
            min=self.config.min_logvar,
            max=self.config.max_logvar,
        )

        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (
                target - mean
            ).square() * torch.exp(-logvar)
        )

        return self._masked_mean(
            nll,
            self._mask(targets, mean),
        )

    def register_loss(
        self,
        name: str,
        function: ReconstructionLossFunction,
        *,
        weight: float = 1.0,
    ) -> None:
        if not name:
            raise ValueError(
                "Loss name cannot be empty"
            )

        if name in self.weights:
            raise ValueError(
                f"Loss '{name}' already exists"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Weight must be finite "
                "and nonnegative"
            )

        self.custom_losses[name] = function
        self.weights[name] = float(weight)

    def set_weight(
        self,
        name: str,
        weight: float,
    ) -> None:
        if name not in self.weights:
            raise KeyError(
                f"Unknown reconstruction loss: {name}"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Weight must be finite "
                "and nonnegative"
            )

        self.weights[name] = float(weight)

    def get_weight(
        self,
        name: str,
    ) -> float:
        return self.weights[name]

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        return_components: bool = True,
    ) -> (
        Tensor
        | tuple[Tensor, dict[str, Tensor]]
    ):
        functions = {
            "reconstruction": self.reconstruction_loss,
            "ssim": self.ssim_loss,
            "multiscale": self.multiscale_loss,
            "temporal": self.temporal_loss,
            "gradient": self.gradient_loss,
            "latent": self.latent_loss,
            "gaussian_nll": self.gaussian_nll_loss,
        }

        components: dict[str, Tensor] = {}

        for name, function in functions.items():
            if self.weights[name] <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is not None:
                components[name] = loss

        for name, function in self.custom_losses.items():
            if self.weights[name] <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is not None:
                components[name] = loss

        if not components:
            raise ValueError(
                "No active reconstruction loss components. "
                "Check model outputs, targets and weights."
            )

        for name, loss in components.items():
            if not isinstance(loss, Tensor):
                raise TypeError(
                    f"{name} must return a Tensor"
                )

            if loss.ndim != 0:
                raise ValueError(
                    f"{name} loss must be scalar"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Non-finite reconstruction loss: {name}"
                )

        normalizer = 1.0

        if self.config.normalize_weights:
            normalizer = sum(
                self.weights[name]
                for name in components
            )

        weighted = {
            name: (
                loss * self.weights[name]
                / normalizer
            )
            for name, loss in components.items()
        }

        total = torch.stack(
            list(weighted.values())
        ).sum()

        self.last_metrics = {
            "reconstruction_total": float(
                total.detach().item()
            ),
            **{
                f"reconstruction/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"reconstruction_weighted/{name}": float(
                    loss.detach().item()
                )
                for name, loss in weighted.items()
            },
        }

        if return_components:
            return total, {
                "total": total,
                **components,
            }

        return total

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return dict(self.last_metrics)

    def reset_metrics(self) -> None:
        self.last_metrics.clear()


def create_reconstruction_loss(
    *,
    loss_type: str = "mse",
    reconstruction_weight: float = 1.0,
    ssim_weight: float = 0.0,
    multiscale_weight: float = 0.0,
    temporal_weight: float = 0.0,
    gradient_weight: float = 0.0,
    latent_weight: float = 0.0,
    gaussian_nll_weight: float = 0.0,
) -> ReconstructionLoss:
    return ReconstructionLoss(
        ReconstructionLossConfig(
            loss_type=loss_type,
            reconstruction_weight=reconstruction_weight,
            ssim_weight=ssim_weight,
            multiscale_weight=multiscale_weight,
            temporal_weight=temporal_weight,
            gradient_weight=gradient_weight,
            latent_weight=latent_weight,
            gaussian_nll_weight=gaussian_nll_weight,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    sequence_length = 8
    channels = 3
    height = 64
    width = 64
    latent_dim = 128

    class DemoVideoAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(
                    channels,
                    32,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    32,
                    64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(
                    64 * 16 * 16,
                    latent_dim,
                ),
            )

            self.decoder = nn.Sequential(
                nn.Linear(
                    latent_dim,
                    64 * 16 * 16,
                ),
                nn.ReLU(),
                nn.Unflatten(
                    1,
                    (64, 16, 16),
                ),
                nn.ConvTranspose2d(
                    64,
                    32,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    32,
                    channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Sigmoid(),
            )

        def forward(
            self,
            video: Tensor,
        ) -> dict[str, Tensor]:
            batch, steps, channels, h, w = (
                video.shape
            )

            frames = video.reshape(
                batch * steps,
                channels,
                h,
                w,
            )

            latents = self.encoder(
                frames
            )

            reconstructed = self.decoder(
                latents
            )

            reconstructed = reconstructed.reshape(
                batch,
                steps,
                channels,
                h,
                w,
            )

            return {
                "reconstruction": reconstructed,
                "reconstructed_latents": (
                    latents.reshape(
                        batch,
                        steps,
                        latent_dim,
                    )
                ),
            }

    model = DemoVideoAutoencoder()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = ReconstructionLoss(
        ReconstructionLossConfig(
            reconstruction_weight=1.0,
            ssim_weight=0.1,
            multiscale_weight=0.2,
            temporal_weight=0.1,
            gradient_weight=0.05,
            latent_weight=0.0,
            gaussian_nll_weight=0.0,
            loss_type="mse",
        )
    )

    for step in range(3):
        video = torch.rand(
            batch_size,
            sequence_length,
            channels,
            height,
            width,
        )

        outputs = model(video)

        targets = {
            "frames": video,
            "mask": torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
            ),
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss, components = criterion(
            outputs,
            targets,
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )
