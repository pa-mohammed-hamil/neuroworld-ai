
"""
NeuroWorld AI
File: neuroworld/dynamics/deterministic_dynamics.py

Deterministic Latent Dynamics Model

Features:
- Deterministic state-transition prediction
- Action-conditioned latent dynamics
- Residual state updates
- Layer normalization and GELU activation
- Optional bounded state deltas
- Multi-step trajectory rollout
- Teacher-forced supervised training
- One-step and multi-step prediction losses
- Optional transition constraints
- Compatible probabilistic adapter
- CPU / CUDA support

Dependencies:
    pip install torch

Core deterministic interface:
    next_state = model(state, action)

Probabilistic adapter interface:
    mean, logvar = adapter(state, action)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


StateConstraint = Callable[[Tensor], Tensor]


@dataclass
class DeterministicDynamicsConfig:
    latent_dim: int = 128
    action_dim: int = 32

    hidden_dim: int = 256
    num_layers: int = 3

    dropout: float = 0.0
    residual: bool = True

    use_layer_norm: bool = True
    use_action_gating: bool = True

    bounded_delta: bool = False
    max_state_delta: float = 1.0

    prediction_loss: str = "mse"
    huber_delta: float = 1.0

    rollout_discount: float = 0.99
    smoothness_weight: float = 0.0

    adapter_logvar: float = -10.0


@dataclass
class DeterministicRolloutOutput:
    """
    states:
        [B,H+1,D]

    actions:
        [B,H,A]

    state_deltas:
        [B,H,D]

    transition_magnitudes:
        [B,H]
    """

    states: Tensor
    actions: Tensor
    state_deltas: Tensor
    transition_magnitudes: Tensor

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]


class ResidualMLPBlock(nn.Module):
    """
    Residual feedforward block.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
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


class DeterministicDynamics(nn.Module):
    """
    Deterministic neural transition model.

    Given:
        state_t: [B,D]
        action_t: [B,A]

    Predicts:
        state_t+1: [B,D]

    If residual=True:

        next_state = state + learned_delta

    Otherwise:

        next_state = learned_state

    This model predicts a point estimate and does
    not intrinsically estimate epistemic or
    aleatoric uncertainty.
    """

    def __init__(
        self,
        config: Optional[
            DeterministicDynamicsConfig
        ] = None,
        state_constraint: Optional[
            StateConstraint
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or DeterministicDynamicsConfig()
        )

        cfg = self.config

        if (
            cfg.latent_dim <= 0
            or cfg.action_dim <= 0
            or cfg.hidden_dim <= 0
            or cfg.num_layers <= 0
        ):
            raise ValueError(
                "Dimensions and num_layers "
                "must be positive"
            )

        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0,1)"
            )

        if cfg.max_state_delta <= 0:
            raise ValueError(
                "max_state_delta must be positive"
            )

        if cfg.prediction_loss not in (
            "mse",
            "huber",
            "mae",
        ):
            raise ValueError(
                "Unsupported prediction loss"
            )

        if cfg.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if not 0.0 <= cfg.rollout_discount <= 1.0:
            raise ValueError(
                "rollout_discount must be in [0,1]"
            )

        if cfg.smoothness_weight < 0:
            raise ValueError(
                "smoothness_weight must be non-negative"
            )

        self.state_constraint = state_constraint

        self.state_encoder = nn.Sequential(
            nn.Linear(
                cfg.latent_dim,
                cfg.hidden_dim,
            ),
            nn.LayerNorm(cfg.hidden_dim)
            if cfg.use_layer_norm
            else nn.Identity(),
            nn.GELU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(
                cfg.action_dim,
                cfg.hidden_dim,
            ),
            nn.GELU(),
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

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    hidden_dim=cfg.hidden_dim,
                    dropout=cfg.dropout,
                    use_layer_norm=(
                        cfg.use_layer_norm
                    ),
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.output_norm = (
            nn.LayerNorm(cfg.hidden_dim)
            if cfg.use_layer_norm
            else nn.Identity()
        )

        self.transition_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim,
        )

        # Small initial residual predictions help
        # avoid excessively large initial jumps.
        nn.init.normal_(
            self.transition_head.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.transition_head.bias
        )

    def _validate_transition(
        self,
        state: Tensor,
        action: Tensor,
    ) -> None:

        cfg = self.config

        if state.ndim != 2:
            raise ValueError(
                "state must have shape [B,D]"
            )

        if action.ndim != 2:
            raise ValueError(
                "action must have shape [B,A]"
            )

        if state.shape[0] == 0:
            raise ValueError(
                "Batch size must be positive"
            )

        if state.shape != (
            action.shape[0],
            cfg.latent_dim,
        ):
            raise ValueError(
                "State batch or latent dimension mismatch"
            )

        if action.shape[-1] != cfg.action_dim:
            raise ValueError(
                "Action dimension mismatch"
            )

        if (
            state.device != action.device
            or state.dtype != action.dtype
        ):
            raise ValueError(
                "State and action must share device/dtype"
            )

        if not (
            torch.is_floating_point(state)
            and torch.is_floating_point(action)
        ):
            raise TypeError(
                "State and action must be floating point"
            )

        if not (
            torch.isfinite(state).all().item()
            and torch.isfinite(action).all().item()
        ):
            raise ValueError(
                "State or action contains NaN/Inf"
            )

    def predict_delta(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Predict latent-state change.

        Returns:
            [B,latent_dim]
        """

        self._validate_transition(
            state,
            action,
        )

        cfg = self.config

        state_features = self.state_encoder(
            state
        )

        action_features = self.action_encoder(
            action
        )

        if self.action_gate is not None:
            gate = torch.sigmoid(
                self.action_gate(action_features)
            )

            state_features = (
                state_features * gate
            )

        fused = torch.cat(
            [
                state_features,
                action_features,
            ],
            dim=-1,
        )

        hidden = self.fusion(fused)

        hidden = self.blocks(hidden)

        hidden = self.output_norm(hidden)

        delta = self.transition_head(hidden)

        if cfg.bounded_delta:
            delta = (
                torch.tanh(delta)
                * cfg.max_state_delta
            )

        return delta

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Predict next latent state.

        Returns:
            next_state: [B,D]
        """

        delta = self.predict_delta(
            state,
            action,
        )

        if self.config.residual:
            next_state = state + delta
        else:
            next_state = delta

        if self.state_constraint is not None:
            next_state = self.state_constraint(
                next_state
            )

            if next_state.shape != state.shape:
                raise ValueError(
                    "State constraint changed tensor shape"
                )

        return next_state

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        teacher_states: Optional[
            Tensor
        ] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> DeterministicRolloutOutput:
        """
        Generate a multi-step latent trajectory.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        teacher_states:
            Optional ground-truth states [B,H+1,D].

        teacher_forcing_ratio:
            Probability of using a ground-truth state
            as input for the following transition.

        The output trajectory contains predictions,
        even when teacher forcing is used.
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

        if initial_state.shape[-1] != cfg.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if actions.shape[0] != initial_state.shape[0]:
            raise ValueError(
                "Batch size mismatch"
            )

        if actions.shape[-1] != cfg.action_dim:
            raise ValueError(
                "Invalid action dimension"
            )

        if actions.shape[1] == 0:
            raise ValueError(
                "Horizon must be positive"
            )

        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError(
                "teacher_forcing_ratio must be in [0,1]"
            )

        if teacher_forcing_ratio > 0:
            if teacher_states is None:
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

        tensors = [
            initial_state,
            actions,
        ]

        if teacher_states is not None:
            tensors.append(teacher_states)

        for tensor in tensors:
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    "Inputs must be floating point"
                )

            if (
                tensor.device != initial_state.device
                or tensor.dtype != initial_state.dtype
            ):
                raise ValueError(
                    "Inputs must share device/dtype"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    "Inputs contain NaN or Inf"
                )

        current = initial_state

        predicted_states = [current]
        deltas = []
        magnitudes = []

        horizon = actions.shape[1]

        for step in range(horizon):

            action = actions[:, step, :]

            predicted_next = self.forward(
                current,
                action,
            )

            # Delta relative to the actual input
            # used for this transition.
            delta = predicted_next - current

            predicted_states.append(
                predicted_next
            )

            deltas.append(delta)

            magnitudes.append(
                torch.linalg.vector_norm(
                    delta,
                    dim=-1,
                )
            )

            current = predicted_next

            if (
                teacher_states is not None
                and teacher_forcing_ratio > 0
                and step < horizon - 1
            ):
                teacher_mask = (
                    torch.rand(
                        initial_state.shape[0],
                        1,
                        device=initial_state.device,
                    )
                    < teacher_forcing_ratio
                )

                current = torch.where(
                    teacher_mask,
                    teacher_states[
                        :, step + 1, :
                    ],
                    current,
                )

        return DeterministicRolloutOutput(
            states=torch.stack(
                predicted_states,
                dim=1,
            ),
            actions=actions,
            state_deltas=torch.stack(
                deltas,
                dim=1,
            ),
            transition_magnitudes=torch.stack(
                magnitudes,
                dim=1,
            ),
        )

    def prediction_loss(
        self,
        predicted: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Supervised next-state prediction loss.
        """

        if predicted.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ"
            )

        if self.config.prediction_loss == "mse":
            return F.mse_loss(
                predicted,
                target,
            )

        if self.config.prediction_loss == "mae":
            return F.l1_loss(
                predicted,
                target,
            )

        return F.huber_loss(
            predicted,
            target,
            delta=self.config.huber_delta,
        )

    def compute_training_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Multi-step supervised training loss.

        target_states:
            [B,H+1,D]

        Includes:
        - Discounted state prediction error
        - Optional transition smoothness penalty

        For pure autoregressive rollout training,
        use teacher_forcing_ratio=0.
        """

        output = self.rollout(
            initial_state=initial_state,
            actions=actions,
            teacher_states=target_states,
            teacher_forcing_ratio=(
                teacher_forcing_ratio
            ),
        )

        predictions = output.states[:, 1:, :]
        targets = target_states[:, 1:, :]

        cfg = self.config

        if cfg.prediction_loss == "mse":
            errors = (
                predictions - targets
            ).pow(2).mean(dim=-1)

        elif cfg.prediction_loss == "mae":
            errors = (
                predictions - targets
            ).abs().mean(dim=-1)

        else:
            errors = F.huber_loss(
                predictions,
                targets,
                delta=cfg.huber_delta,
                reduction="none",
            ).mean(dim=-1)

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

        if horizon > 1:
            delta_difference = (
                output.state_deltas[:, 1:, :]
                - output.state_deltas[:, :-1, :]
            )

            smoothness_loss = (
                delta_difference.pow(2).mean()
            )
        else:
            smoothness_loss = (
                prediction_loss.new_zeros(())
            )

        total_loss = (
            prediction_loss
            + cfg.smoothness_weight
            * smoothness_loss
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "smoothness_loss": smoothness_loss,
        }

    @torch.no_grad()
    def predict(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Inference-only one-step prediction.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(state, action)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> DeterministicRolloutOutput:
        """
        Inference-only multi-step rollout.
        """

        was_training = self.training
        self.eval()

        try:
            return self.rollout(
                initial_state,
                actions,
            )
        finally:
            self.train(was_training)


class DeterministicDynamicsAdapter(nn.Module):
    """
    Adapter for probabilistic rollout interfaces.

    Converts:

        next_state = deterministic_model(state, action)

    Into:

        mean, logvar = adapter(state, action)

    The returned logvar is a fixed placeholder,
    NOT a learned or calibrated uncertainty estimate.

    Compatible with:
        WorldRollout
        LatentRollout
        FutureGenerator
        ScenarioGenerator
        TrajectoryGenerator
    """

    def __init__(
        self,
        model: DeterministicDynamics,
        logvar: Optional[float] = None,
    ):
        super().__init__()

        self.model = model

        self.logvar = (
            model.config.adapter_logvar
            if logvar is None
            else float(logvar)
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        mean = self.model(
            state,
            action,
        )

        logvar = torch.full_like(
            mean,
            self.logvar,
        )

        return mean, logvar


if __name__ == "__main__":

    torch.manual_seed(42)

    config = DeterministicDynamicsConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=3,
        residual=True,
        bounded_delta=True,
        max_state_delta=0.5,
        smoothness_weight=0.01,
    )

    model = DeterministicDynamics(config)

    batch_size = 4
    horizon = 10

    initial_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    actions = torch.randn(
        batch_size,
        horizon,
        config.action_dim,
    ).clamp(-1.0, 1.0)

    target_states = torch.randn(
        batch_size,
        horizon + 1,
        config.latent_dim,
    )

    # One-step prediction
    next_state = model(
        initial_state,
        actions[:, 0, :],
    )

    print(
        "Next state:",
        next_state.shape,
    )

    # Multi-step rollout
    rollout_output = model.generate(
        initial_state,
        actions,
    )

    print(
        "Trajectory:",
        rollout_output.states.shape,
    )

    print(
        "State deltas:",
        rollout_output.state_deltas.shape,
    )

    # Example training step
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    losses = model.compute_training_loss(
        initial_state=initial_state,
        actions=actions,
        target_states=target_states,
        teacher_forcing_ratio=0.5,
    )

    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()

    print(
        "Training loss:",
        losses["loss"].item(),
    )

    # Adapter for imagination modules
    adapter = DeterministicDynamicsAdapter(
        model
    )

    mean, logvar = adapter(
        initial_state,
        actions[:, 0, :],
    )

    print("Adapter mean:", mean.shape)
    print("Adapter logvar:", logvar.shape)
