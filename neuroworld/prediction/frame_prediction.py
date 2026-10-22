
"""
NeuroWorld AI
File: neuroworld/prediction/frame_prediction.py

Future Video Frame Prediction Engine.

Features:
    1. CNN frame encoder
    2. ConvGRU temporal dynamics
    3. Action-conditioned prediction
    4. Single future-frame prediction
    5. Multi-frame autoregressive rollout
    6. Teacher-forced sequence training
    7. Monte Carlo dropout uncertainty
    8. Pixel reconstruction loss
    9. Image gradient consistency loss
   10. Temporal consistency loss
   11. PSNR and MAE evaluation
   12. Gradient clipping
   13. Checkpoint save/load
   14. Demonstration

Input:
    frames: [B, T, C, H, W]
    actions: [B, T_future, action_dim]

Output:
    predicted_frames: [B, T_future, C, H, W]

Frame values must be normalized to [0, 1].

Dependencies:
    pip install torch numpy

Run:
    python -m neuroworld.prediction.frame_prediction

This is a learned predictor. Generated frames
are hypotheses about future observations,
not ground-truth video.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class FramePredictionError(Exception):
    """Base frame prediction exception."""


class FrameValidationError(FramePredictionError):
    """Invalid frame or action input."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FramePredictionConfig:
    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64

    action_dim: int = 8

    latent_channels: int = 128
    hidden_channels: int = 128

    max_history_frames: int = 16
    max_future_frames: int = 32

    dropout: float = 0.1

    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0

    pixel_loss_weight: float = 1.0
    gradient_loss_weight: float = 0.1
    temporal_loss_weight: float = 0.1

    teacher_forcing_ratio: float = 0.5

    mc_dropout_samples: int = 8

    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "image_channels",
            "image_height",
            "image_width",
            "action_dim",
            "latent_channels",
            "hidden_channels",
            "max_history_frames",
            "max_future_frames",
            "mc_dropout_samples",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if (
            self.image_height % 8 != 0
            or self.image_width % 8 != 0
        ):
            raise ValueError(
                "Image dimensions must be "
                "divisible by 8"
            )

        if not 0 <= self.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if not 0 <= self.teacher_forcing_ratio <= 1:
            raise ValueError(
                "teacher_forcing_ratio must be in [0,1]"
            )

        for name in (
            "learning_rate",
            "max_grad_norm",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        for name in (
            "weight_decay",
            "pixel_loss_weight",
            "gradient_loss_weight",
            "temporal_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# FRAME VALIDATION
# ============================================================

def prepare_frames(
    frames: Tensor | np.ndarray,
    config: FramePredictionConfig,
    device: torch.device,
    *,
    max_frames: int,
) -> Tensor:
    frames = torch.as_tensor(
        frames,
        dtype=torch.float32,
        device=device,
    )

    if frames.ndim == 4:
        frames = frames.unsqueeze(0)

    if frames.ndim != 5:
        raise FrameValidationError(
            "Frames must have shape [B,T,C,H,W]"
        )

    batch_size, sequence_length, channels, height, width = (
        frames.shape
    )

    if batch_size <= 0 or sequence_length <= 0:
        raise FrameValidationError(
            "Frame batch and sequence cannot be empty"
        )

    if sequence_length > max_frames:
        raise FrameValidationError(
            "Too many frames in sequence"
        )

    if (
        channels != config.image_channels
        or height != config.image_height
        or width != config.image_width
    ):
        raise FrameValidationError(
            "Frame dimensions do not match configuration"
        )

    if not torch.isfinite(frames).all():
        raise FrameValidationError(
            "Frames contain non-finite values"
        )

    if (
        (frames < 0).any()
        or (frames > 1).any()
    ):
        raise FrameValidationError(
            "Frame values must be normalized to [0,1]"
        )

    return frames


def prepare_actions(
    actions: Tensor | np.ndarray | None,
    *,
    batch_size: int,
    future_steps: int,
    config: FramePredictionConfig,
    device: torch.device,
) -> Tensor:
    if actions is None:
        return torch.zeros(
            batch_size,
            future_steps,
            config.action_dim,
            dtype=torch.float32,
            device=device,
        )

    actions = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=device,
    )

    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    expected_shape = (
        batch_size,
        future_steps,
        config.action_dim,
    )

    if actions.shape != expected_shape:
        raise FrameValidationError(
            f"Expected actions {expected_shape}, "
            f"got {tuple(actions.shape)}"
        )

    if not torch.isfinite(actions).all():
        raise FrameValidationError(
            "Actions contain non-finite values"
        )

    return actions


# ============================================================
# FRAME ENCODER
# ============================================================

class FrameEncoder(nn.Module):
    """
    CNN encoder.

    Input:
        [B,C,H,W]

    Output:
        [B,latent_channels,H/8,W/8]
    """

    def __init__(
        self,
        config: FramePredictionConfig,
    ) -> None:
        super().__init__()

        channels = config.latent_channels

        self.network = nn.Sequential(
            nn.Conv2d(
                config.image_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Conv2d(
                64,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        )

    def forward(
        self,
        frames: Tensor,
    ) -> Tensor:
        return self.network(
            frames
        )


# ============================================================
# CONVGRU CELL
# ============================================================

class ConvGRUCell(nn.Module):
    """
    Convolutional gated recurrent unit.

    Preserves spatial structure during
    temporal state updates.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        combined_channels = (
            input_channels + hidden_channels
        )

        self.gates = nn.Conv2d(
            combined_channels,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
        )

        self.candidate = nn.Conv2d(
            combined_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        inputs: Tensor,
        hidden: Tensor | None = None,
    ) -> Tensor:
        if hidden is None:
            hidden = torch.zeros(
                inputs.shape[0],
                self.hidden_channels,
                inputs.shape[2],
                inputs.shape[3],
                device=inputs.device,
                dtype=inputs.dtype,
            )

        combined = torch.cat(
            [inputs, hidden],
            dim=1,
        )

        reset_gate, update_gate = (
            self.gates(combined)
            .sigmoid()
            .chunk(2, dim=1)
        )

        candidate_input = torch.cat(
            [
                inputs,
                reset_gate * hidden,
            ],
            dim=1,
        )

        candidate = torch.tanh(
            self.candidate(
                candidate_input
            )
        )

        next_hidden = (
            (1.0 - update_gate)
            * candidate
            + update_gate * hidden
        )

        return next_hidden


# ============================================================
# ACTION CONDITIONING
# ============================================================

class ActionConditioner(nn.Module):
    """
    Converts action vectors into spatial
    latent feature maps.
    """

    def __init__(
        self,
        config: FramePredictionConfig,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.action_dim,
                config.hidden_channels,
            ),
            nn.SiLU(),
            nn.Linear(
                config.hidden_channels,
                config.hidden_channels,
            ),
        )

    def forward(
        self,
        actions: Tensor,
        spatial_size: tuple[int, int],
    ) -> Tensor:
        embedding = self.network(
            actions
        )

        return embedding[
            :, :, None, None
        ].expand(
            -1,
            -1,
            spatial_size[0],
            spatial_size[1],
        )


# ============================================================
# FRAME DECODER
# ============================================================

class FrameDecoder(nn.Module):
    """
    CNN decoder.

    Input:
        [B,hidden_channels,H/8,W/8]

    Output:
        [B,image_channels,H,W]
    """

    def __init__(
        self,
        config: FramePredictionConfig,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                config.hidden_channels,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),

            nn.Conv2d(
                32,
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
        return self.network(
            latent
        )


# ============================================================
# OUTPUT
# ============================================================

@dataclass
class FramePredictionOutput:
    frames: Tensor
    latent_states: Tensor

    uncertainty: Tensor | None = None

    def detach(self) -> "FramePredictionOutput":
        return FramePredictionOutput(
            frames=self.frames.detach(),
            latent_states=(
                self.latent_states.detach()
            ),
            uncertainty=(
                self.uncertainty.detach()
                if self.uncertainty is not None
                else None
            ),
        )


# ============================================================
# MAIN FRAME PREDICTION MODEL
# ============================================================

class FramePredictionModel(nn.Module):
    """
    Action-conditioned autoregressive
    future-frame predictor.

    History:
        CNN encode each observed frame
        ConvGRU update hidden state

    Future:
        Condition hidden state on action
        Predict next latent state
        Decode predicted frame
        Repeat autoregressively
    """

    def __init__(
        self,
        config: FramePredictionConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or FramePredictionConfig()
        )

        self.encoder = FrameEncoder(
            self.config
        )

        self.history_gru = ConvGRUCell(
            input_channels=(
                self.config.latent_channels
            ),
            hidden_channels=(
                self.config.hidden_channels
            ),
        )

        self.action_conditioner = (
            ActionConditioner(
                self.config
            )
        )

        self.future_gru = ConvGRUCell(
            input_channels=(
                self.config.latent_channels
                + self.config.hidden_channels
            ),
            hidden_channels=(
                self.config.hidden_channels
            ),
        )

        self.decoder = FrameDecoder(
            self.config
        )

        self.dropout = nn.Dropout2d(
            self.config.dropout
        )

        self.to(
            torch.device(
                self.config.device
            )
        )

    @property
    def device(self) -> torch.device:
        return next(
            self.parameters()
        ).device

    # --------------------------------------------------------
    # HISTORY ENCODING
    # --------------------------------------------------------

    def encode_history(
        self,
        history_frames: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = None
        last_latent = None

        for time_index in range(
            history_frames.shape[1]
        ):
            frame = history_frames[
                :, time_index
            ]

            latent = self.encoder(
                frame
            )

            hidden = self.history_gru(
                latent,
                hidden,
            )

            last_latent = latent

        return hidden, last_latent

    # --------------------------------------------------------
    # FUTURE STEP
    # --------------------------------------------------------

    def predict_step(
        self,
        previous_latent: Tensor,
        hidden: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        spatial_size = (
            previous_latent.shape[-2:]
        )

        action_features = (
            self.action_conditioner(
                action,
                spatial_size,
            )
        )

        combined = torch.cat(
            [
                previous_latent,
                action_features,
            ],
            dim=1,
        )

        hidden = self.future_gru(
            combined,
            hidden,
        )

        decoded_hidden = self.dropout(
            hidden
        )

        next_frame = self.decoder(
            decoded_hidden
        )

        # Re-encode the generated frame to
        # maintain an autoregressive latent
        # representation.
        next_latent = self.encoder(
            next_frame
        )

        return (
            next_frame,
            next_latent,
            hidden,
        )

    # --------------------------------------------------------
    # FORWARD ROLLOUT
    # --------------------------------------------------------

    def forward(
        self,
        history_frames: Tensor | np.ndarray,
        *,
        future_steps: int,
        actions: Tensor | np.ndarray | None = None,
        target_frames: Tensor | np.ndarray | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> FramePredictionOutput:
        if not (
            1
            <= future_steps
            <= self.config.max_future_frames
        ):
            raise FrameValidationError(
                "Invalid future_steps"
            )

        if not (
            0.0
            <= teacher_forcing_ratio
            <= 1.0
        ):
            raise FrameValidationError(
                "Invalid teacher_forcing_ratio"
            )

        history_frames = prepare_frames(
            history_frames,
            self.config,
            self.device,
            max_frames=(
                self.config.max_history_frames
            ),
        )

        batch_size = (
            history_frames.shape[0]
        )

        actions = prepare_actions(
            actions,
            batch_size=batch_size,
            future_steps=future_steps,
            config=self.config,
            device=self.device,
        )

        targets = None

        if target_frames is not None:
            targets = prepare_frames(
                target_frames,
                self.config,
                self.device,
                max_frames=(
                    self.config.max_future_frames
                ),
            )

            if targets.shape[:2] != (
                batch_size,
                future_steps,
            ):
                raise FrameValidationError(
                    "Target sequence shape mismatch"
                )

        hidden, previous_latent = (
            self.encode_history(
                history_frames
            )
        )

        predicted_frames = []
        predicted_latents = []

        for step in range(
            future_steps
        ):
            action = actions[
                :, step
            ]

            (
                next_frame,
                next_latent,
                hidden,
            ) = self.predict_step(
                previous_latent,
                hidden,
                action,
            )

            predicted_frames.append(
                next_frame
            )

            predicted_latents.append(
                next_latent
            )

            previous_latent = (
                next_latent
            )

            if (
                self.training
                and targets is not None
                and teacher_forcing_ratio > 0
            ):
                teacher_mask = (
                    torch.rand(
                        batch_size,
                        device=self.device,
                    )
                    < teacher_forcing_ratio
                )

                if teacher_mask.any():
                    target_latent = self.encoder(
                        targets[:, step]
                    )

                    previous_latent = torch.where(
                        teacher_mask[
                            :, None, None, None
                        ],
                        target_latent,
                        previous_latent,
                    )

        frames = torch.stack(
            predicted_frames,
            dim=1,
        )

        latents = torch.stack(
            predicted_latents,
            dim=1,
        )

        return FramePredictionOutput(
            frames=frames,
            latent_states=latents,
        )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        history_frames: Tensor | np.ndarray,
        *,
        future_steps: int,
        actions: Tensor | np.ndarray | None = None,
        estimate_uncertainty: bool = False,
        mc_samples: int | None = None,
    ) -> FramePredictionOutput:
        if mc_samples is None:
            mc_samples = (
                self.config.mc_dropout_samples
            )

        if mc_samples <= 0:
            raise FrameValidationError(
                "mc_samples must be positive"
            )

        previous_mode = self.training

        try:
            if not estimate_uncertainty:
                self.eval()

                return self.forward(
                    history_frames,
                    future_steps=future_steps,
                    actions=actions,
                ).detach()

            # Keep model in evaluation mode,
            # enabling only dropout layers.
            self.eval()

            for module in self.modules():
                if isinstance(
                    module,
                    nn.Dropout2d,
                ):
                    module.train()

            outputs = [
                self.forward(
                    history_frames,
                    future_steps=future_steps,
                    actions=actions,
                )
                for _ in range(mc_samples)
            ]

            frame_samples = torch.stack(
                [
                    output.frames
                    for output in outputs
                ],
                dim=0,
            )

            latent_samples = torch.stack(
                [
                    output.latent_states
                    for output in outputs
                ],
                dim=0,
            )

            return FramePredictionOutput(
                frames=frame_samples.mean(
                    dim=0
                ),
                latent_states=(
                    latent_samples.mean(
                        dim=0
                    )
                ),
                uncertainty=(
                    frame_samples.std(
                        dim=0,
                        unbiased=False,
                    )
                ),
            ).detach()

        finally:
            self.train(
                previous_mode
            )


# ============================================================
# FRAME PREDICTION LOSSES
# ============================================================

def image_gradient_loss(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Compare horizontal and vertical
    image gradients.
    """

    predicted_dx = (
        predicted[..., :, 1:]
        - predicted[..., :, :-1]
    )

    target_dx = (
        target[..., :, 1:]
        - target[..., :, :-1]
    )

    predicted_dy = (
        predicted[..., 1:, :]
        - predicted[..., :-1, :]
    )

    target_dy = (
        target[..., 1:, :]
        - target[..., :-1, :]
    )

    return (
        F.l1_loss(
            predicted_dx,
            target_dx,
        )
        + F.l1_loss(
            predicted_dy,
            target_dy,
        )
    )


def temporal_consistency_loss(
    predicted: Tensor,
    target: Tensor,
) -> Tensor:
    """
    Compare frame-to-frame changes.

    Requires at least two future frames.
    """

    if predicted.shape[1] < 2:
        return (
            predicted.sum() * 0.0
        )

    predicted_changes = (
        predicted[:, 1:]
        - predicted[:, :-1]
    )

    target_changes = (
        target[:, 1:]
        - target[:, :-1]
    )

    return F.l1_loss(
        predicted_changes,
        target_changes,
    )


def calculate_frame_prediction_loss(
    predicted: Tensor,
    target: Tensor,
    config: FramePredictionConfig,
) -> dict[str, Tensor]:
    if predicted.shape != target.shape:
        raise FrameValidationError(
            "Predicted and target shapes differ"
        )

    pixel_loss = F.l1_loss(
        predicted,
        target,
    )

    gradient_loss = image_gradient_loss(
        predicted,
        target,
    )

    temporal_loss = temporal_consistency_loss(
        predicted,
        target,
    )

    total_loss = (
        config.pixel_loss_weight
        * pixel_loss
        + config.gradient_loss_weight
        * gradient_loss
        + config.temporal_loss_weight
        * temporal_loss
    )

    return {
        "total_loss": total_loss,
        "pixel_loss": pixel_loss,
        "gradient_loss": gradient_loss,
        "temporal_loss": temporal_loss,
    }


# ============================================================
# TRAINING BATCH
# ============================================================

@dataclass
class FrameTrainingBatch:
    history_frames: Tensor
    target_frames: Tensor
    actions: Tensor | None = None


# ============================================================
# TRAINER
# ============================================================

class FramePredictionTrainer:
    """
    Trainer for supervised future-frame
    reconstruction.
    """

    def __init__(
        self,
        model: FramePredictionModel,
    ) -> None:
        self.model = model
        self.config = model.config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.training_steps = 0

    def _prepare_batch(
        self,
        batch: FrameTrainingBatch,
    ) -> FrameTrainingBatch:
        history = prepare_frames(
            batch.history_frames,
            self.config,
            self.model.device,
            max_frames=(
                self.config.max_history_frames
            ),
        )

        targets = prepare_frames(
            batch.target_frames,
            self.config,
            self.model.device,
            max_frames=(
                self.config.max_future_frames
            ),
        )

        if (
            history.shape[0]
            != targets.shape[0]
        ):
            raise FrameValidationError(
                "History and target batch sizes differ"
            )

        actions = prepare_actions(
            batch.actions,
            batch_size=history.shape[0],
            future_steps=targets.shape[1],
            config=self.config,
            device=self.model.device,
        )

        return FrameTrainingBatch(
            history_frames=history,
            target_frames=targets,
            actions=actions,
        )

    def train_step(
        self,
        batch: FrameTrainingBatch,
    ) -> dict[str, float]:
        self.model.train()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.history_frames,
            future_steps=(
                batch.target_frames.shape[1]
            ),
            actions=batch.actions,
            target_frames=batch.target_frames,
            teacher_forcing_ratio=(
                self.config.teacher_forcing_ratio
            ),
        )

        losses = calculate_frame_prediction_loss(
            output.frames,
            batch.target_frames,
            self.config,
        )

        if not torch.isfinite(
            losses["total_loss"]
        ):
            raise FramePredictionError(
                "Non-finite frame prediction loss"
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        losses["total_loss"].backward()

        gradient_norm = (
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        self.training_steps += 1

        return {
            **{
                name: float(
                    value.detach().item()
                )
                for name, value in losses.items()
            },
            "gradient_norm": float(
                gradient_norm.detach().item()
            ),
            "training_step": (
                self.training_steps
            ),
        }

    @torch.no_grad()
    def evaluate(
        self,
        batch: FrameTrainingBatch,
    ) -> dict[str, float]:
        self.model.eval()

        batch = self._prepare_batch(
            batch
        )

        output = self.model(
            batch.history_frames,
            future_steps=(
                batch.target_frames.shape[1]
            ),
            actions=batch.actions,
            teacher_forcing_ratio=0.0,
        )

        predicted = output.frames
        targets = batch.target_frames

        losses = calculate_frame_prediction_loss(
            predicted,
            targets,
            self.config,
        )

        mse = F.mse_loss(
            predicted,
            targets,
        )

        mae = F.l1_loss(
            predicted,
            targets,
        )

        psnr = (
            -10.0
            * torch.log10(
                mse.clamp_min(1e-10)
            )
        )

        return {
            **{
                name: float(
                    value.item()
                )
                for name, value in losses.items()
            },
            "mse": float(
                mse.item()
            ),
            "mae": float(
                mae.item()
            ),
            "psnr_db": float(
                psnr.item()
            ),
        }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_frame_prediction_model(
    model: FramePredictionModel,
    path: str | Path,
    *,
    trainer: FramePredictionTrainer | None = None,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint: dict[str, Any] = {
        "config": asdict(
            model.config
        ),
        "model": model.state_dict(),
    }

    if trainer is not None:
        checkpoint["optimizer"] = (
            trainer.optimizer.state_dict()
        )

        checkpoint["training_steps"] = (
            trainer.training_steps
        )

    torch.save(
        checkpoint,
        path,
    )

    return path


def load_frame_prediction_model(
    path: str | Path,
    *,
    device: str = "cpu",
    restore_trainer: bool = False,
) -> (
    FramePredictionModel
    | tuple[
        FramePredictionModel,
        FramePredictionTrainer,
    ]
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    config_data = dict(
        checkpoint["config"]
    )

    config_data["device"] = device

    model = FramePredictionModel(
        FramePredictionConfig(
            **config_data
        )
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    if not restore_trainer:
        return model

    trainer = FramePredictionTrainer(
        model
    )

    if "optimizer" in checkpoint:
        trainer.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    trainer.training_steps = int(
        checkpoint.get(
            "training_steps",
            0,
        )
    )

    return model, trainer


# ============================================================
# FACTORY
# ============================================================

def create_frame_predictor(
    config: FramePredictionConfig | None = None,
) -> FramePredictionModel:
    return FramePredictionModel(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - FRAME PREDICTION"
    )
    print("=" * 65)

    config = FramePredictionConfig(
        image_channels=3,
        image_height=32,
        image_width=32,
        action_dim=4,
        latent_channels=32,
        hidden_channels=32,
        max_history_frames=8,
        max_future_frames=8,
        dropout=0.1,
        mc_dropout_samples=4,
    )

    model = FramePredictionModel(
        config
    )

    trainer = FramePredictionTrainer(
        model
    )

    batch_size = 2
    history_length = 4
    future_steps = 3

    history_frames = torch.rand(
        batch_size,
        history_length,
        config.image_channels,
        config.image_height,
        config.image_width,
    )

    target_frames = torch.rand(
        batch_size,
        future_steps,
        config.image_channels,
        config.image_height,
        config.image_width,
    )

    actions = torch.randn(
        batch_size,
        future_steps,
        config.action_dim,
    )

    batch = FrameTrainingBatch(
        history_frames=history_frames,
        target_frames=target_frames,
        actions=actions,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    metrics = trainer.train_step(
        batch
    )

    print(
        "Training metrics:",
        metrics,
    )

    # --------------------------------------------------------
    # FUTURE FRAME PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        history_frames,
        future_steps=future_steps,
        actions=actions,
    )

    print(
        "Predicted frames:",
        tuple(
            prediction.frames.shape
        ),
    )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    uncertain_prediction = model.predict(
        history_frames,
        future_steps=future_steps,
        actions=actions,
        estimate_uncertainty=True,
    )

    print(
        "Frame uncertainty:",
        tuple(
            uncertain_prediction.uncertainty.shape
        ),
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    evaluation = trainer.evaluate(
        batch
    )

    print(
        "Evaluation:",
        evaluation,
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    checkpoint = save_frame_prediction_model(
        model,
        "models/frame_prediction_demo.pt",
        trainer=trainer,
    )

    restored_model = (
        load_frame_prediction_model(
            checkpoint
        )
    )

    print(
        "Restored model:",
        type(restored_model).__name__,
    )

    print(
        "Frame prediction demonstration completed"
    )


if __name__ == "__main__":
    main()
