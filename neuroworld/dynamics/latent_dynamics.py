
"""
NeuroWorld AI
File: neuroworld/dynamics/latent_dynamics.py

Latent Dynamics Prediction Engine

Features:
- Action-conditioned latent state transitions
- Deterministic and stochastic prediction
- Residual latent-state updates
- Gated state-action feature fusion
- Gaussian transition distributions
- Multi-step autoregressive prediction
- Teacher forcing
- Supervised one-step and rollout losses
- Latent transition consistency loss
- Uncertainty estimation
- Training and inference utilities
- PyTorch CPU / CUDA support

Required interface:
    mean, logvar = model(state, action)

Tensor shapes:
    state:  [B, latent_dim]
    action: [B, action_dim]
    mean:   [B, latent_dim]
    logvar: [B, latent_dim]

Dependencies:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class LatentDynamicsConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    num_layers: int = 3
    dropout: float = 0.0

    residual: bool = True
    use_layer_norm: bool = True
    use_action_gating: bool = True

    stochastic: bool = True
    temperature: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    deterministic_logvar: float = -10.0

    rollout_discount: float = 0.99

    prediction_weight: float = 1.0
    consistency_weight: float = 0.1
    smoothness_weight: float = 0.0

    gradient_clip_norm: float = 1.0


@dataclass
class LatentTransitionOutput:
    mean: Tensor
    logvar: Tensor
    next_state: Tensor
    uncertainty: Tensor

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()

    @property
    def std(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)


@dataclass
class LatentDynamicsRollout:
    """
    states:        [B,H+1,D]
    means:         [B,H,D]
    logvars:       [B,H,D]
    uncertainties: [B,H]
    state_deltas:  [B,H,D]
    """

    states: Tensor
    means: Tensor
    logvars: Tensor
    uncertainties: Tensor
    state_deltas: Tensor

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]

    @property
    def horizon(self) -> int:
        return self.means.shape[1]


class LatentResidualBlock(nn.Module):
    """Pre-normalized residual MLP block."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        super().__init__()

        self.norm = (
            nn.LayerNorm(hidden_dim)
            if use_layer_norm
            else nn.Identity()
        )

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.network(self.norm(x))


class LatentDynamics(nn.Module):
    """
    Neural latent transition model.

    Learns:
        p(z[t+1] | z[t], a[t])

    For stochastic mode:
        Predicts a diagonal Gaussian distribution.

    For deterministic mode:
        Predicts a point estimate with a fixed
        placeholder log-variance.

    The standard forward interface is compatible
    with NeuroWorld imagination modules.
    """

    def __init__(
        self,
        config: Optional[LatentDynamicsConfig] = None,
    ):
        super().__init__()

        self.config = config or LatentDynamicsConfig()
        cfg = self.config

        if min(
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_layers,
        ) <= 0:
            raise ValueError(
                "All dimensions and num_layers "
                "must be positive"
            )

        if not 0 <= cfg.dropout < 1:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if cfg.min_logvar > cfg.max_logvar:
            raise ValueError(
                "min_logvar cannot exceed max_logvar"
            )

        if not 0 <= cfg.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in [0,1]"
            )

        for name in (
            "prediction_weight",
            "consistency_weight",
            "smoothness_weight",
        ):
            if getattr(cfg, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative"
                )

        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim)
            if cfg.use_layer_norm
            else nn.Identity(),
            nn.GELU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.GELU(),
        )

        self.action_gate = (
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            if cfg.use_action_gating
            else None
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                cfg.hidden_dim * 2,
                cfg.hidden_dim,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                LatentResidualBlock(
                    hidden_dim=cfg.hidden_dim,
                    dropout=cfg.dropout,
                    use_layer_norm=cfg.use_layer_norm,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.output_norm = (
            nn.LayerNorm(cfg.hidden_dim)
            if cfg.use_layer_norm
            else nn.Identity()
        )

        self.mean_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim,
        )

        self.logvar_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim,
        )

        # Start residual dynamics close to identity.
        nn.init.normal_(
            self.mean_head.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.mean_head.bias)

        # Moderate initial predictive variance.
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(
            self.logvar_head.bias,
            -2.0,
        )

    def _validate_transition(
        self,
        state: Tensor,
        action: Tensor,
    ) -> None:
        cfg = self.config

        if state.ndim != 2:
            raise ValueError(
                "state must be [B,D]"
            )

        if action.ndim != 2:
            raise ValueError(
                "action must be [B,A]"
            )

        if state.shape[0] < 1:
            raise ValueError(
                "Batch size must be positive"
            )

        if state.shape[1] != cfg.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if action.shape != (
            state.shape[0],
            cfg.action_dim,
        ):
            raise ValueError(
                "Invalid action shape"
            )

        if (
            state.device != action.device
            or state.dtype != action.dtype
        ):
            raise ValueError(
                "State and action device/dtype mismatch"
            )

        for name, value in (
            ("state", state),
            ("action", action),
        ):
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(value).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def encode(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Encode a latent state-action pair.

        Returns:
            [B,hidden_dim]
        """

        self._validate_transition(
            state,
            action,
        )

        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)

        if self.action_gate is not None:
            gate = torch.sigmoid(
                self.action_gate(action_features)
            )

            state_features = state_features * gate

        fused = torch.cat(
            [state_features, action_features],
            dim=-1,
        )

        hidden = self.fusion(fused)
        hidden = self.blocks(hidden)

        return self.output_norm(hidden)

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict Gaussian transition parameters.

        Returns:
            mean:   [B,D]
            logvar: [B,D]
        """

        hidden = self.encode(state, action)

        mean = self.mean_head(hidden)

        if self.config.residual:
            mean = state + mean

        if self.config.stochastic:
            logvar = self.logvar_head(hidden).clamp(
                self.config.min_logvar,
                self.config.max_logvar,
            )
        else:
            logvar = torch.full_like(
                mean,
                self.config.deterministic_logvar,
            )

        return mean, logvar

    def sample(
        self,
        mean: Tensor,
        logvar: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Sample next latent state.

        Uses the reparameterization trick.
        """

        if mean.shape != logvar.shape:
            raise ValueError(
                "Mean/logvar shape mismatch"
            )

        if temperature is None:
            temperature = self.config.temperature

        if temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if (
            deterministic
            or not self.config.stochastic
            or temperature == 0
        ):
            return mean

        std = torch.exp(0.5 * logvar)

        return (
            mean
            + temperature
            * std
            * torch.randn_like(mean)
        )

    def predict(
        self,
        state: Tensor,
        action: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> LatentTransitionOutput:
        """
        Predict one latent transition.
        """

        mean, logvar = self.forward(
            state,
            action,
        )

        next_state = self.sample(
            mean,
            logvar,
            deterministic=deterministic,
            temperature=temperature,
        )

        uncertainty = (
            logvar.exp().mean(dim=-1)
        )

        return LatentTransitionOutput(
            mean=mean,
            logvar=logvar,
            next_state=next_state,
            uncertainty=uncertainty,
        )

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        deterministic: bool = False,
        teacher_states: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> LatentDynamicsRollout:
        """
        Generate a multi-step latent trajectory.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        teacher_states:
            Optional [B,H+1,D]

        teacher_forcing_ratio:
            Probability of using ground-truth states
            as inputs for subsequent transitions.

        Predicted states are returned at every step.
        """

        cfg = self.config

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B,D]"
            )

        if initial_state.shape[1] != cfg.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,H,A]"
            )

        if (
            actions.shape[0] != initial_state.shape[0]
            or actions.shape[2] != cfg.action_dim
        ):
            raise ValueError(
                "Invalid action dimensions"
            )

        if actions.shape[1] == 0:
            raise ValueError(
                "Horizon must be positive"
            )

        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError(
                "teacher_forcing_ratio must be in [0,1]"
            )

        if (
            teacher_forcing_ratio > 0
            and teacher_states is None
        ):
            raise ValueError(
                "teacher_states required for "
                "teacher forcing"
            )

        if teacher_states is not None:
            expected = (
                initial_state.shape[0],
                actions.shape[1] + 1,
                cfg.latent_dim,
            )

            if teacher_states.shape != expected:
                raise ValueError(
                    "teacher_states must be [B,H+1,D]"
                )

        for name, tensor in (
            ("initial_state", initial_state),
            ("actions", actions),
            ("teacher_states", teacher_states),
        ):
            if tensor is None:
                continue

            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if (
                tensor.device != initial_state.device
                or tensor.dtype != initial_state.dtype
            ):
                raise ValueError(
                    f"{name} device/dtype mismatch"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

        current = initial_state

        states = [current]
        means = []
        logvars = []
        uncertainties = []
        state_deltas = []

        horizon = actions.shape[1]

        for step in range(horizon):

            prediction = self.predict(
                current,
                actions[:, step, :],
                deterministic=deterministic,
            )

            states.append(
                prediction.next_state
            )

            means.append(prediction.mean)
            logvars.append(prediction.logvar)

            uncertainties.append(
                prediction.uncertainty
            )

            state_deltas.append(
                prediction.next_state - current
            )

            current = prediction.next_state

            if (
                teacher_states is not None
                and teacher_forcing_ratio > 0
                and step < horizon - 1
            ):
                mask = (
                    torch.rand(
                        current.shape[0],
                        1,
                        device=current.device,
                    )
                    < teacher_forcing_ratio
                )

                current = torch.where(
                    mask,
                    teacher_states[:, step + 1, :],
                    current,
                )

        return LatentDynamicsRollout(
            states=torch.stack(states, dim=1),
            means=torch.stack(means, dim=1),
            logvars=torch.stack(logvars, dim=1),
            uncertainties=torch.stack(
                uncertainties,
                dim=1,
            ),
            state_deltas=torch.stack(
                state_deltas,
                dim=1,
            ),
        )

    def gaussian_nll(
        self,
        target: Tensor,
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        Diagonal Gaussian negative log-likelihood.
        """

        if (
            target.shape != mean.shape
            or mean.shape != logvar.shape
        ):
            raise ValueError(
                "Gaussian NLL shape mismatch"
            )

        return (
            0.5 * (
                logvar
                + (target - mean).pow(2)
                * torch.exp(-logvar)
                + 1.8378770664093453
            )
        ).mean()

    def compute_loss(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, Tensor]:
        """
        One-step supervised dynamics loss.

        state:      [B,D]
        action:     [B,A]
        next_state: [B,D]
        """

        mean, logvar = self.forward(
            state,
            action,
        )

        if next_state.shape != mean.shape:
            raise ValueError(
                "next_state shape mismatch"
            )

        if self.config.stochastic:
            prediction_loss = self.gaussian_nll(
                next_state,
                mean,
                logvar,
            )
        else:
            prediction_loss = F.mse_loss(
                mean,
                next_state,
            )

        predicted_delta = mean - state
        target_delta = next_state - state

        consistency_loss = F.mse_loss(
            predicted_delta,
            target_delta,
        )

        total_loss = (
            self.config.prediction_weight
            * prediction_loss
            + self.config.consistency_weight
            * consistency_loss
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "consistency_loss": consistency_loss,
        }

    def compute_rollout_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Multi-step training objective.

        target_states:
            [B,H+1,D]
        """

        output = self.rollout(
            initial_state=initial_state,
            actions=actions,
            deterministic=True,
            teacher_states=target_states,
            teacher_forcing_ratio=(
                teacher_forcing_ratio
            ),
        )

        cfg = self.config

        targets = target_states[:, 1:, :]

        if cfg.stochastic:
            errors = 0.5 * (
                output.logvars
                + (targets - output.means).pow(2)
                * torch.exp(-output.logvars)
                + 1.8378770664093453
            ).mean(dim=-1)
        else:
            errors = (
                output.means - targets
            ).pow(2).mean(dim=-1)

        horizon = actions.shape[1]

        discounts = (
            cfg.rollout_discount
            ** torch.arange(
                horizon,
                device=errors.device,
                dtype=errors.dtype,
            )
        )

        prediction_loss = (
            errors * discounts.unsqueeze(0)
        ).sum(dim=1).mean() / discounts.sum()

        # Consistency compares predicted state
        # transitions against ground-truth targets.
        consistency_loss = F.mse_loss(
            output.means,
            targets,
        )

        if horizon > 1:
            delta_change = (
                output.state_deltas[:, 1:, :]
                - output.state_deltas[:, :-1, :]
            )

            smoothness_loss = (
                delta_change.pow(2).mean()
            )
        else:
            smoothness_loss = (
                prediction_loss.new_zeros(())
            )

        total_loss = (
            cfg.prediction_weight
            * prediction_loss
            + cfg.consistency_weight
            * consistency_loss
            + cfg.smoothness_weight
            * smoothness_loss
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "consistency_loss": consistency_loss,
            "smoothness_loss": smoothness_loss,
        }

    def training_step(
        self,
        optimizer: torch.optim.Optimizer,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, float]:
        """
        Perform one optimizer update.
        """

        self.train()

        optimizer.zero_grad(set_to_none=True)

        losses = self.compute_loss(
            state,
            action,
            next_state,
        )

        losses["loss"].backward()

        if self.config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.gradient_clip_norm,
            )

        optimizer.step()

        return {
            key: float(
                value.detach().item()
            )
            for key, value in losses.items()
        }

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
        deterministic: bool = False,
    ) -> LatentDynamicsRollout:
        """
        Inference-only latent trajectory generation.
        """

        was_training = self.training
        self.eval()

        try:
            return self.rollout(
                initial_state=initial_state,
                actions=actions,
                deterministic=deterministic,
            )
        finally:
            self.train(was_training)


if __name__ == "__main__":

    torch.manual_seed(42)

    config = LatentDynamicsConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=3,
        stochastic=True,
    )

    model = LatentDynamics(config)

    batch_size = 4
    horizon = 10

    state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    action = torch.randn(
        batch_size,
        config.action_dim,
    )

    next_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    mean, logvar = model(
        state,
        action,
    )

    print("Mean:", mean.shape)
    print("Logvar:", logvar.shape)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    metrics = model.training_step(
        optimizer,
        state,
        action,
        next_state,
    )

    print("Training metrics:", metrics)

    actions = torch.randn(
        batch_size,
        horizon,
        config.action_dim,
    )

    output = model.generate(
        initial_state=state,
        actions=actions,
        deterministic=False,
    )

    print(
        "Latent trajectory:",
        output.states.shape,
    )

    print(
        "Uncertainty:",
        output.uncertainties.shape,
    )

    print(
        "Final states:",
        output.final_states.shape,
    )
