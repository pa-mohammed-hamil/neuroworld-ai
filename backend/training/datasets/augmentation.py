
"""
NeuroWorld AI
File: training/datasets/augmentation.py

Data augmentation for world-model training.

Supports:
- Image brightness, contrast and Gaussian noise
- Horizontal flip and random crop
- Video-consistent spatial transformations
- Video temporal cropping and frame dropping
- State Gaussian noise
- Action Gaussian noise and clipping
- Trajectory augmentation
- Deterministic seeded generators
- CPU and CUDA tensors

Dependencies:
    torch

Tensor formats:
    Image:      [C, H, W]
    Images:     [B, C, H, W]
    Video:      [T, C, H, W]
    Video batch:[B, T, C, H, W]

All image/video tensors should be floating point
with pixel values in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from torch import Tensor


@dataclass
class AugmentationConfig:
    enabled: bool = True

    # Image and video
    horizontal_flip_probability: float = 0.5
    brightness: float = 0.15
    contrast: float = 0.15
    image_noise_std: float = 0.01

    random_crop_probability: float = 0.3
    crop_scale_min: float = 0.85

    # State and action
    state_noise_std: float = 0.01
    action_noise_std: float = 0.01

    clip_actions: bool = True
    action_min: float = -1.0
    action_max: float = 1.0

    # Temporal
    temporal_crop_length: int | None = None
    frame_drop_probability: float = 0.0

    # Reproducibility
    seed: int | None = 42

    def __post_init__(self) -> None:
        for name in (
            "horizontal_flip_probability",
            "random_crop_probability",
            "frame_drop_probability",
        ):
            value = getattr(self, name)

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        for name in (
            "brightness",
            "contrast",
            "image_noise_std",
            "state_noise_std",
            "action_noise_std",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )

        if not 0 < self.crop_scale_min <= 1:
            raise ValueError(
                "crop_scale_min must be in (0, 1]"
            )

        if self.action_min > self.action_max:
            raise ValueError(
                "Invalid action bounds"
            )

        if (
            self.temporal_crop_length is not None
            and self.temporal_crop_length < 1
        ):
            raise ValueError(
                "temporal_crop_length must be >= 1"
            )


class DataAugmentation:
    """
    General-purpose augmentation pipeline.

    Spatial transformations are consistent
    across frames in a single video.

    Important:
        Horizontal flipping is suitable only
        when labels, actions and coordinates
        remain valid under that transformation.

        Do not independently flip images in
        action-conditioned physical datasets
        unless corresponding actions/states
        are transformed too.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
    ) -> None:
        self.config = (
            config or AugmentationConfig()
        )

        self._generators: dict[
            str, torch.Generator
        ] = {}

    def _generator(
        self,
        device: torch.device,
    ) -> torch.Generator | None:
        if self.config.seed is None:
            return None

        key = str(device)

        if key not in self._generators:
            generator = torch.Generator(
                device=device
            )

            generator.manual_seed(
                self.config.seed
            )

            self._generators[key] = generator

        return self._generators[key]

    def _rand(
        self,
        device: torch.device,
    ) -> float:
        return float(
            torch.rand(
                (),
                device=device,
                generator=self._generator(device),
            ).item()
        )

    def _uniform(
        self,
        low: float,
        high: float,
        device: torch.device,
    ) -> float:
        return (
            low
            + (high - low)
            * self._rand(device)
        )

    def _randint(
        self,
        low: int,
        high: int,
        device: torch.device,
    ) -> int:
        return int(
            torch.randint(
                low,
                high,
                (),
                device=device,
                generator=self._generator(device),
            ).item()
        )

    def _noise(
        self,
        tensor: Tensor,
        std: float,
    ) -> Tensor:
        if std == 0:
            return tensor

        noise = torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=self._generator(
                tensor.device
            ),
        )

        return tensor + noise * std

    def augment_image(
        self,
        image: Tensor,
        *,
        allow_flip: bool = True,
    ) -> Tensor:
        """
        Augment one image [C, H, W].

        Images should be normalized to [0, 1].
        """
        if not self.config.enabled:
            return image

        if image.ndim != 3:
            raise ValueError(
                "Expected image [C, H, W]"
            )

        if not image.is_floating_point():
            raise TypeError(
                "Image must be floating point"
            )

        output = image.clone()

        device = output.device

        if (
            allow_flip
            and self._rand(device)
            < self.config.horizontal_flip_probability
        ):
            output = torch.flip(
                output,
                dims=(-1,),
            )

        if self.config.brightness > 0:
            delta = self._uniform(
                -self.config.brightness,
                self.config.brightness,
                device,
            )

            output = output + delta

        if self.config.contrast > 0:
            factor = self._uniform(
                1.0 - self.config.contrast,
                1.0 + self.config.contrast,
                device,
            )

            mean = output.mean(
                dim=(-2, -1),
                keepdim=True,
            )

            output = (
                output - mean
            ) * factor + mean

        if (
            self._rand(device)
            < self.config.random_crop_probability
        ):
            output = self._random_resized_crop(
                output
            )

        output = self._noise(
            output,
            self.config.image_noise_std,
        )

        return output.clamp(0.0, 1.0)

    def _random_resized_crop(
        self,
        image: Tensor,
    ) -> Tensor:
        _, height, width = image.shape

        scale = self._uniform(
            self.config.crop_scale_min,
            1.0,
            image.device,
        )

        crop_h = max(
            1,
            round(height * scale),
        )

        crop_w = max(
            1,
            round(width * scale),
        )

        top = self._randint(
            0,
            height - crop_h + 1,
            image.device,
        )

        left = self._randint(
            0,
            width - crop_w + 1,
            image.device,
        )

        cropped = image[
            :,
            top:top + crop_h,
            left:left + crop_w,
        ]

        return F.interpolate(
            cropped.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def augment_video(
        self,
        video: Tensor,
        *,
        allow_flip: bool = True,
    ) -> Tensor:
        """
        Video shape: [T, C, H, W].

        Spatial transformations are sampled
        once and shared across all frames.
        """
        if not self.config.enabled:
            return video

        if video.ndim != 4:
            raise ValueError(
                "Expected video [T, C, H, W]"
            )

        if not video.is_floating_point():
            raise TypeError(
                "Video must be floating point"
            )

        output = video.clone()
        device = output.device

        if (
            allow_flip
            and self._rand(device)
            < self.config.horizontal_flip_probability
        ):
            output = torch.flip(
                output,
                dims=(-1,),
            )

        if self.config.brightness > 0:
            delta = self._uniform(
                -self.config.brightness,
                self.config.brightness,
                device,
            )

            output = output + delta

        if self.config.contrast > 0:
            factor = self._uniform(
                1.0 - self.config.contrast,
                1.0 + self.config.contrast,
                device,
            )

            mean = output.mean(
                dim=(-2, -1),
                keepdim=True,
            )

            output = (
                output - mean
            ) * factor + mean

        if (
            self._rand(device)
            < self.config.random_crop_probability
        ):
            _, _, height, width = output.shape

            scale = self._uniform(
                self.config.crop_scale_min,
                1.0,
                device,
            )

            crop_h = max(
                1,
                round(height * scale),
            )

            crop_w = max(
                1,
                round(width * scale),
            )

            top = self._randint(
                0,
                height - crop_h + 1,
                device,
            )

            left = self._randint(
                0,
                width - crop_w + 1,
                device,
            )

            output = output[
                :,
                :,
                top:top + crop_h,
                left:left + crop_w,
            ]

            output = F.interpolate(
                output,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        # Independent sensor noise per frame.
        output = self._noise(
            output,
            self.config.image_noise_std,
        )

        return output.clamp(0.0, 1.0)

    def augment_states(
        self,
        states: Tensor,
    ) -> Tensor:
        """
        Add sensor noise to state vectors.

        Supports:
            [D], [T, D], [B, T, D]
        """
        if not self.config.enabled:
            return states

        return self._noise(
            states,
            self.config.state_noise_std,
        )

    def augment_actions(
        self,
        actions: Tensor,
    ) -> Tensor:
        """
        Perturb continuous action vectors.

        Use only if labels remain consistent
        with the perturbed actions.
        """
        if not self.config.enabled:
            return actions

        output = self._noise(
            actions,
            self.config.action_noise_std,
        )

        if self.config.clip_actions:
            output = output.clamp(
                self.config.action_min,
                self.config.action_max,
            )

        return output

    def temporal_crop(
        self,
        sequence: Tensor,
    ) -> Tensor:
        """
        Random temporal crop.

        Input shape: [T, ...]
        """
        length = (
            self.config.temporal_crop_length
        )

        if (
            not self.config.enabled
            or length is None
        ):
            return sequence

        total = sequence.shape[0]

        if total <= length:
            return sequence

        start = self._randint(
            0,
            total - length + 1,
            sequence.device,
        )

        return sequence[
            start:start + length
        ]

    def drop_video_frames(
        self,
        video: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Simulate missing video observations.

        Preserves sequence length by replacing
        dropped frames with previous frames.

        Returns:
            augmented_video
            observation_mask

        mask[t] is False for a dropped frame.
        """
        if video.ndim != 4:
            raise ValueError(
                "Expected [T, C, H, W]"
            )

        length = video.shape[0]

        mask = torch.ones(
            length,
            dtype=torch.bool,
            device=video.device,
        )

        if (
            not self.config.enabled
            or self.config.frame_drop_probability == 0
        ):
            return video, mask

        output = video.clone()

        for index in range(1, length):
            if (
                self._rand(video.device)
                < self.config.frame_drop_probability
            ):
                output[index] = output[index - 1]
                mask[index] = False

        return output, mask

    def augment_trajectory(
        self,
        sample: dict[str, Tensor],
        *,
        perturb_actions: bool = False,
    ) -> dict[str, Tensor]:
        """
        Augment a trajectory sample.

        Supported keys:
            states
            actions
            next_states
            rewards
            terminated
            truncated
            mask

        By default, action perturbation is off
        because changing actions without
        recomputing next_states/rewards can
        invalidate supervised training labels.
        """
        if not self.config.enabled:
            return sample

        result = {
            key: (
                value.clone()
                if isinstance(value, Tensor)
                else value
            )
            for key, value in sample.items()
        }

        if "states" in result:
            result["states"] = (
                self.augment_states(
                    result["states"]
                )
            )

        # Keep next_states unchanged as targets.
        # This simulates noisy input observations.

        if (
            perturb_actions
            and "actions" in result
        ):
            result["actions"] = (
                self.augment_actions(
                    result["actions"]
                )
            )

        return result

    def __call__(
        self,
        sample: Tensor | dict[str, Any],
    ) -> Tensor | dict[str, Any]:
        if isinstance(sample, dict):
            if (
                "states" in sample
                and "actions" in sample
            ):
                return self.augment_trajectory(
                    sample
                )

            if "frames" in sample:
                result = dict(sample)

                result["frames"] = (
                    self.augment_video(
                        sample["frames"]
                    )
                )

                return result

            return sample

        if isinstance(sample, Tensor):
            if sample.ndim == 3:
                return self.augment_image(
                    sample
                )

            if sample.ndim == 4:
                return self.augment_video(
                    sample
                )

        raise TypeError(
            "Unsupported augmentation input"
        )


if __name__ == "__main__":
    config = AugmentationConfig(
        horizontal_flip_probability=0.5,
        brightness=0.1,
        contrast=0.1,
        image_noise_std=0.01,
        state_noise_std=0.02,
        action_noise_std=0.01,
        frame_drop_probability=0.1,
        seed=42,
    )

    augmentation = DataAugmentation(
        config
    )

    # Image example
    image = torch.rand(
        3, 128, 128
    )

    augmented_image = (
        augmentation.augment_image(
            image
        )
    )

    print(
        "Image:",
        augmented_image.shape,
    )

    # Video example
    video = torch.rand(
        16, 3, 128, 128
    )

    augmented_video = (
        augmentation.augment_video(
            video
        )
    )

    print(
        "Video:",
        augmented_video.shape,
    )

    # State-action example
    trajectory = {
        "states": torch.randn(
            16, 128
        ),
        "actions": torch.randn(
            16, 32
        ),
        "next_states": torch.randn(
            16, 128
        ),
        "rewards": torch.randn(16),
    }

    augmented_trajectory = (
        augmentation.augment_trajectory(
            trajectory
        )
    )

    print(
        "States:",
        augmented_trajectory["states"].shape,
    )

    print(
        "Actions:",
        augmented_trajectory["actions"].shape,
    )
