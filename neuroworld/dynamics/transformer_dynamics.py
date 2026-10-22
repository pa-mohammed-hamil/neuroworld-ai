
"""
NeuroWorld AI
File: neuroworld/dynamics/transformer_dynamics.py

Causal Transformer Dynamics Model

Features:
- Action-conditioned latent dynamics
- Multi-head causal self-attention
- Historical state-action context
- Learned positional embeddings
- Pre-normalized Transformer blocks
- Residual state transitions
- Gaussian probabilistic prediction
- Deterministic next-state prediction
- Autoregressive multi-step rollouts
- Teacher forcing
- Variable-length training sequences
- Sequence padding masks
- Gaussian NLL and MSE training losses
- Gradient clipping
- Stateless compatibility with WorldRollout
- CPU / CUDA support

Dependencies:
    pip install torch

Standard interface:
    mean, logvar = model(state, action)

Context-aware interface:
    mean, logvar = model.predict_sequence(
        states, actions
    )

History-aware rollout:
    output = model.rollout(
        initial_state,
        future_actions,
        history_states=...,
        history_actions=...,
    )
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class TransformerDynamicsConfig:
    latent_dim: int = 128
    action_dim: int = 32

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    feedforward_multiplier: int = 4
    dropout: float = 0.0

    max_context_length: int = 64

    residual: bool = True
    stochastic: bool = True

    temperature: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0
    deterministic_logvar: float = -10.0

    use_layer_norm: bool = True
    use_action_gating: bool = True

    rollout_discount: float = 0.99
    gradient_clip_norm: float = 1.0


@dataclass
class TransformerTransitionOutput:
    mean: Tensor
    logvar: Tensor
    next_state: Tensor
    uncertainty: Tensor

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()


@dataclass
class TransformerRolloutOutput:
    """
    states:
        [B,H+1,D]

    means:
        [B,H,D]

    logvars:
        [B,H,D]

    uncertainties:
        [B,H]
    """

    states: Tensor
    means: Tensor
    logvars: Tensor
    uncertainties: Tensor

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]

    @property
    def horizon(self) -> int:
        return self.means.shape[1]


class CausalTransformerBlock(nn.Module):
    """
    Pre-normalized causal Transformer block.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_multiplier: int,
        dropout: float,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            hidden_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.feedforward_norm = nn.LayerNorm(
            hidden_dim
        )

        self.feedforward = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
                * feedforward_multiplier,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim
                * feedforward_multiplier,
                hidden_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:

        normalized = self.attention_norm(x)

        attention_output, _ = self.attention(
            query=normalized,
            key=normalized,
            value=normalized,
            attn_mask=attention_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        x = (
            x
            + self.attention_dropout(
                attention_output
            )
        )

        x = (
            x
            + self.feedforward(
                self.feedforward_norm(x)
            )
        )

        return x


class TransformerDynamics(nn.Module):
    """
    Causal Transformer latent dynamics.

    Learns:

        p(z[t+1] | z[0:t], a[0:t])

    Each token represents a state-action pair:

        token_t = encode(z_t, a_t)

    A causal attention mask prevents tokens from
    attending to future state-action pairs.

    forward(state, action) uses one token for
    compatibility with existing world-model APIs.

    predict_sequence(states, actions) uses temporal
    context and is the preferred Transformer API.
    """

    def __init__(
        self,
        config: Optional[
            TransformerDynamicsConfig
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or TransformerDynamicsConfig()
        )

        cfg = self.config

        if min(
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.num_heads,
            cfg.max_context_length,
            cfg.feedforward_multiplier,
        ) <= 0:
            raise ValueError(
                "Dimensions and layer counts "
                "must be positive"
            )

        if cfg.hidden_dim % cfg.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible "
                "by num_heads"
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
                "Invalid log-variance limits"
            )

        if not 0 <= cfg.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in [0,1]"
            )

        self.state_encoder = nn.Linear(
            cfg.latent_dim,
            cfg.hidden_dim,
        )

        self.action_encoder = nn.Linear(
            cfg.action_dim,
            cfg.hidden_dim,
        )

        self.action_gate = (
            nn.Linear(
                cfg.hidden_dim,
                cfg.hidden_dim,
            )
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

        self.position_embedding = nn.Embedding(
            cfg.max_context_length,
            cfg.hidden_dim,
        )

        self.input_dropout = nn.Dropout(
            cfg.dropout
        )

        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    hidden_dim=cfg.hidden_dim,
                    num_heads=cfg.num_heads,
                    feedforward_multiplier=(
                        cfg.feedforward_multiplier
                    ),
                    dropout=cfg.dropout,
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

        nn.init.normal_(
            self.mean_head.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.mean_head.bias
        )

        nn.init.zeros_(
            self.logvar_head.weight
        )

        nn.init.constant_(
            self.logvar_head.bias,
            -2.0,
        )

    def _validate_sequence(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> None:

        cfg = self.config

        if states.ndim != 3:
            raise ValueError(
                "states must be [B,T,D]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,T,A]"
            )

        if states.shape[0] < 1:
            raise ValueError(
                "Batch size must be positive"
            )

        if states.shape[1] < 1:
            raise ValueError(
                "Sequence length must be positive"
            )

        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "State/action sequence mismatch"
            )

        if states.shape[-1] != cfg.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if actions.shape[-1] != cfg.action_dim:
            raise ValueError(
                "Invalid action dimension"
            )

        if states.shape[1] > cfg.max_context_length:
            raise ValueError(
                "Sequence exceeds max_context_length"
            )

        if (
            states.device != actions.device
            or states.dtype != actions.dtype
        ):
            raise ValueError(
                "States/actions must share device/dtype"
            )

        for name, tensor in (
            ("states", states),
            ("actions", actions),
        ):
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def encode_tokens(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """
        Encode state-action tokens.

        Returns:
            [B,T,hidden_dim]
        """

        self._validate_sequence(
            states,
            actions,
        )

        cfg = self.config

        state_features = self.state_encoder(
            states
        )

        action_features = self.action_encoder(
            actions
        )

        if self.action_gate is not None:
            gate = torch.sigmoid(
                self.action_gate(action_features)
            )

            state_features = (
                state_features * gate
            )

        tokens = torch.cat(
            [
                state_features,
                action_features,
            ],
            dim=-1,
        )

        tokens = self.fusion(tokens)

        sequence_length = tokens.shape[1]

        positions = torch.arange(
            sequence_length,
            device=tokens.device,
        )

        position_features = (
            self.position_embedding(positions)
            .unsqueeze(0)
        )

        tokens = tokens + position_features

        return self.input_dropout(tokens)

    @staticmethod
    def causal_mask(
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        """
        Boolean causal mask.

        True indicates attention is blocked.
        """

        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def predict_sequence(
        self,
        states: Tensor,
        actions: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict next-state distributions for
        every timestep.

        states:
            [B,T,D]

        actions:
            [B,T,A]

        padding_mask:
            Optional [B,T] boolean tensor.
            True means padded/ignored token.

        Returns:
            means: [B,T,D]
            logvars: [B,T,D]

        Token t predicts state t+1 using only
        tokens at or before t.
        """

        tokens = self.encode_tokens(
            states,
            actions,
        )

        batch_size, sequence_length, _ = (
            tokens.shape
        )

        if padding_mask is not None:
            if padding_mask.shape != (
                batch_size,
                sequence_length,
            ):
                raise ValueError(
                    "padding_mask must be [B,T]"
                )

            if padding_mask.dtype != torch.bool:
                raise TypeError(
                    "padding_mask must be boolean"
                )

            if padding_mask.device != tokens.device:
                raise ValueError(
                    "padding_mask device mismatch"
                )

            if padding_mask.all(dim=1).any().item():
                raise ValueError(
                    "Each sequence needs at least "
                    "one unmasked token"
                )

            # Right-padded sequences only.
            invalid_order = (
                padding_mask[:, :-1]
                & ~padding_mask[:, 1:]
            )

            if invalid_order.any().item():
                raise ValueError(
                    "padding_mask must use right padding"
                )

        mask = self.causal_mask(
            sequence_length,
            tokens.device,
        )

        hidden = tokens

        for block in self.blocks:
            hidden = block(
                hidden,
                attention_mask=mask,
                padding_mask=padding_mask,
            )

        hidden = self.output_norm(hidden)

        means = self.mean_head(hidden)

        if self.config.residual:
            means = states + means

        if self.config.stochastic:
            logvars = (
                self.logvar_head(hidden)
                .clamp(
                    self.config.min_logvar,
                    self.config.max_logvar,
                )
            )
        else:
            logvars = torch.full_like(
                means,
                self.config.deterministic_logvar,
            )

        return means, logvars

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Stateless world-model compatibility.

        state:
            [B,D]

        action:
            [B,A]

        Returns:
            mean: [B,D]
            logvar: [B,D]

        Important:
            This interface uses a single token.
            It does not use historical context.
        """

        if state.ndim != 2:
            raise ValueError(
                "state must be [B,D]"
            )

        if action.ndim != 2:
            raise ValueError(
                "action must be [B,A]"
            )

        means, logvars = self.predict_sequence(
            states=state.unsqueeze(1),
            actions=action.unsqueeze(1),
        )

        return (
            means[:, -1, :],
            logvars[:, -1, :],
        )

    def sample_next_state(
        self,
        mean: Tensor,
        logvar: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Sample from the predicted Gaussian.
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

        return (
            mean
            + temperature
            * torch.exp(0.5 * logvar)
            * torch.randn_like(mean)
        )

    def predict(
        self,
        state: Tensor,
        action: Tensor,
        deterministic: bool = False,
    ) -> TransformerTransitionOutput:
        """
        Stateless one-step prediction.
        """

        mean, logvar = self.forward(
            state,
            action,
        )

        next_state = self.sample_next_state(
            mean,
            logvar,
            deterministic=deterministic,
        )

        uncertainty = (
            logvar.exp().mean(dim=-1)
        )

        return TransformerTransitionOutput(
            mean=mean,
            logvar=logvar,
            next_state=next_state,
            uncertainty=uncertainty,
        )

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        history_states: Optional[Tensor] = None,
        history_actions: Optional[Tensor] = None,
        deterministic: bool = False,
        teacher_states: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> TransformerRolloutOutput:
        """
        Autoregressive Transformer rollout.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        history_states:
            Optional [B,T,D]

        history_actions:
            Optional [B,T,A]

        History contains completed transitions:
            history_states[t] --history_actions[t]-->
            history_states[t+1]

        The current initial_state is appended as
        the next state-action token.

        Teacher forcing uses ground-truth states
        for selected subsequent predictions.
        """

        cfg = self.config

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B,D]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,H,A]"
            )

        if (
            initial_state.shape[-1]
            != cfg.latent_dim
        ):
            raise ValueError(
                "Invalid latent dimension"
            )

        if (
            actions.shape[0]
            != initial_state.shape[0]
            or actions.shape[-1]
            != cfg.action_dim
        ):
            raise ValueError(
                "Invalid action dimensions"
            )

        if actions.shape[1] < 1:
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

        if (
            history_states is None
        ) != (
            history_actions is None
        ):
            raise ValueError(
                "Both history_states and "
                "history_actions are required"
            )

        for name, tensor in (
            ("initial_state", initial_state),
            ("actions", actions),
            ("teacher_states", teacher_states),
            ("history_states", history_states),
            ("history_actions", history_actions),
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

        if history_states is not None:
            self._validate_sequence(
                history_states,
                history_actions,
            )

            if (
                history_states.shape[0]
                != initial_state.shape[0]
            ):
                raise ValueError(
                    "History batch mismatch"
                )

            context_states = history_states
            context_actions = history_actions
        else:
            context_states = initial_state.new_empty(
                initial_state.shape[0],
                0,
                cfg.latent_dim,
            )

            context_actions = actions.new_empty(
                actions.shape[0],
                0,
                cfg.action_dim,
            )

        current = initial_state

        predicted_states = [current]
        predicted_means = []
        predicted_logvars = []
        predicted_uncertainties = []

        horizon = actions.shape[1]

        for step in range(horizon):

            action = actions[:, step, :]

            # Include current transition query.
            query_states = torch.cat(
                [
                    context_states,
                    current.unsqueeze(1),
                ],
                dim=1,
            )

            query_actions = torch.cat(
                [
                    context_actions,
                    action.unsqueeze(1),
                ],
                dim=1,
            )

            # Sliding attention window.
            query_states = query_states[
                :, -cfg.max_context_length:, :
            ]

            query_actions = query_actions[
                :, -cfg.max_context_length:, :
            ]

            means, logvars = self.predict_sequence(
                query_states,
                query_actions,
            )

            mean = means[:, -1, :]
            logvar = logvars[:, -1, :]

            next_state = self.sample_next_state(
                mean,
                logvar,
                deterministic=deterministic,
            )

            predicted_states.append(next_state)
            predicted_means.append(mean)
            predicted_logvars.append(logvar)

            predicted_uncertainties.append(
                logvar.exp().mean(dim=-1)
            )

            # The completed transition becomes
            # historical context.
            context_states = query_states
            context_actions = query_actions

            current = next_state

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

        return TransformerRolloutOutput(
            states=torch.stack(
                predicted_states,
                dim=1,
            ),
            means=torch.stack(
                predicted_means,
                dim=1,
            ),
            logvars=torch.stack(
                predicted_logvars,
                dim=1,
            ),
            uncertainties=torch.stack(
                predicted_uncertainties,
                dim=1,
            ),
        )

    @staticmethod
    def gaussian_nll(
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
            0.5
            * (
                LOG_2PI
                + logvar
                + (target - mean).pow(2)
                * torch.exp(-logvar)
            )
        ).mean()

    def compute_sequence_loss(
        self,
        states: Tensor,
        actions: Tensor,
        target_next_states: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Parallel teacher-forced sequence training.

        states:
            [B,T,D]

        actions:
            [B,T,A]

        target_next_states:
            [B,T,D]

        padding_mask:
            Optional [B,T] boolean mask.

        Causal attention prevents future-token
        leakage during parallel training.
        """

        means, logvars = self.predict_sequence(
            states,
            actions,
            padding_mask=padding_mask,
        )

        if target_next_states.shape != means.shape:
            raise ValueError(
                "target_next_states shape mismatch"
            )

        if self.config.stochastic:
            errors = (
                0.5
                * (
                    LOG_2PI
                    + logvars
                    + (
                        target_next_states - means
                    ).pow(2)
                    * torch.exp(-logvars)
                )
            ).mean(dim=-1)
        else:
            errors = (
                means - target_next_states
            ).pow(2).mean(dim=-1)

        if padding_mask is not None:
            valid = (~padding_mask).to(
                errors.dtype
            )

            loss = (
                errors * valid
            ).sum() / valid.sum().clamp_min(1.0)
        else:
            loss = errors.mean()

        return {
            "loss": loss,
            "prediction_loss": loss,
        }

    def compute_rollout_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Autoregressive multi-step training loss.

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

        targets = target_states[:, 1:, :]

        if self.config.stochastic:
            errors = (
                0.5
                * (
                    LOG_2PI
                    + output.logvars
                    + (
                        targets - output.means
                    ).pow(2)
                    * torch.exp(-output.logvars)
                )
            ).mean(dim=-1)
        else:
            errors = (
                output.means - targets
            ).pow(2).mean(dim=-1)

        horizon = actions.shape[1]

        discounts = (
            self.config.rollout_discount
            ** torch.arange(
                horizon,
                device=errors.device,
                dtype=errors.dtype,
            )
        )

        loss = (
            errors * discounts.unsqueeze(0)
        ).sum(dim=1).mean() / discounts.sum()

        return {
            "loss": loss,
            "rollout_loss": loss,
        }

    def training_step(
        self,
        optimizer: torch.optim.Optimizer,
        states: Tensor,
        actions: Tensor,
        target_next_states: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """
        One supervised sequence training step.
        """

        self.train()

        optimizer.zero_grad(set_to_none=True)

        losses = self.compute_sequence_loss(
            states=states,
            actions=actions,
            target_next_states=target_next_states,
            padding_mask=padding_mask,
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
        history_states: Optional[Tensor] = None,
        history_actions: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> TransformerRolloutOutput:
        """
        Inference-only future generation.
        """

        was_training = self.training
        self.eval()

        try:
            return self.rollout(
                initial_state=initial_state,
                actions=actions,
                history_states=history_states,
                history_actions=history_actions,
                deterministic=deterministic,
            )
        finally:
            self.train(was_training)


if __name__ == "__main__":

    torch.manual_seed(42)

    config = TransformerDynamicsConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        max_context_length=64,
        stochastic=True,
    )

    model = TransformerDynamics(config)

    batch_size = 4
    sequence_length = 12
    horizon = 10

    states = torch.randn(
        batch_size,
        sequence_length,
        config.latent_dim,
    )

    actions = torch.randn(
        batch_size,
        sequence_length,
        config.action_dim,
    )

    targets = torch.randn(
        batch_size,
        sequence_length,
        config.latent_dim,
    )

    means, logvars = model.predict_sequence(
        states,
        actions,
    )

    print("Sequence means:", means.shape)
    print("Sequence logvars:", logvars.shape)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    metrics = model.training_step(
        optimizer,
        states,
        actions,
        targets,
    )

    print("Training metrics:", metrics)

    initial_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    future_actions = torch.randn(
        batch_size,
        horizon,
        config.action_dim,
    )

    output = model.generate(
        initial_state=initial_state,
        actions=future_actions,
        deterministic=False,
    )

    print(
        "Generated trajectory:",
        output.states.shape,
    )

    print(
        "Uncertainty:",
        output.uncertainties.shape,
    )

    # Stateless compatibility with imagination.
    mean, logvar = model(
        initial_state,
        future_actions[:, 0, :],
    )

    print(
        "One-step mean:",
        mean.shape,
    )
