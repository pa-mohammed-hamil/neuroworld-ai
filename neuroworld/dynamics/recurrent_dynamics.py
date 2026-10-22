
"""
NeuroWorld AI
File: neuroworld/dynamics/recurrent_dynamics.py

Recurrent Neural Dynamics Model

Features:
- GRU and LSTM temporal dynamics
- Action-conditioned latent transitions
- Persistent recurrent hidden state
- Deterministic and stochastic prediction
- Gaussian mean and log-variance outputs
- Residual latent state prediction
- Multi-step autoregressive rollout
- Teacher forcing
- Variable-length sequence training
- Gaussian NLL / MSE losses
- Optional truncated backpropagation
- Gradient clipping
- Stateless compatibility with WorldRollout
- CPU and CUDA support

Dependencies:
    pip install torch

Standard world-model interface:
    mean, logvar = model(state, action)

Stateful recurrent interface:
    mean, logvar, memory = model.step(
        state, action, memory
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


RecurrentMemory = Union[
    Tensor,
    Tuple[Tensor, Tensor],
]


@dataclass
class RecurrentDynamicsConfig:
    latent_dim: int = 128
    action_dim: int = 32

    hidden_dim: int = 256
    num_layers: int = 2

    recurrent_type: str = "gru"
    dropout: float = 0.0

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

    # Detach recurrent memory every N rollout steps.
    # 0 disables truncated backpropagation.
    tbptt_steps: int = 0


@dataclass
class RecurrentTransitionOutput:
    mean: Tensor
    logvar: Tensor
    next_state: Tensor
    uncertainty: Tensor
    memory: RecurrentMemory

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()


@dataclass
class RecurrentRolloutOutput:
    """
    states:
        [B,H+1,D]

    means:
        [B,H,D]

    logvars:
        [B,H,D]

    uncertainties:
        [B,H]

    final_memory:
        GRU:  [L,B,hidden_dim]
        LSTM: tuple of two [L,B,hidden_dim] tensors
    """

    states: Tensor
    means: Tensor
    logvars: Tensor
    uncertainties: Tensor
    final_memory: RecurrentMemory

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]

    @property
    def horizon(self) -> int:
        return self.means.shape[1]


class RecurrentDynamics(nn.Module):
    """
    Action-conditioned recurrent latent dynamics.

    Supports:
        GRU
        LSTM

    Recurrent transition:

        x_t = encode(z_t, a_t)
        h_t = RNN(x_t, h_(t-1))
        mean_t, logvar_t = heads(h_t)

    Stateful predictions require passing memory
    between transitions.

    The standard forward() interface initializes
    fresh memory each call. This makes it compatible
    with existing stateless imagination modules,
    but temporal memory is only preserved through
    step() or rollout().
    """

    def __init__(
        self,
        config: Optional[
            RecurrentDynamicsConfig
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or RecurrentDynamicsConfig()
        )

        cfg = self.config

        if min(
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_layers,
        ) <= 0:
            raise ValueError(
                "Dimensions and num_layers must "
                "be positive"
            )

        if cfg.recurrent_type.lower() not in (
            "gru",
            "lstm",
        ):
            raise ValueError(
                "recurrent_type must be 'gru' or 'lstm'"
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

        if cfg.tbptt_steps < 0:
            raise ValueError(
                "tbptt_steps must be non-negative"
            )

        self.recurrent_type = (
            cfg.recurrent_type.lower()
        )

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

        recurrent_dropout = (
            cfg.dropout
            if cfg.num_layers > 1
            else 0.0
        )

        recurrent_class = (
            nn.GRU
            if self.recurrent_type == "gru"
            else nn.LSTM
        )

        self.recurrent = recurrent_class(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
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

        # Residual mean initially predicts
        # small changes to the current state.
        nn.init.normal_(
            self.mean_head.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.mean_head.bias)

        nn.init.zeros_(
            self.logvar_head.weight
        )
        nn.init.constant_(
            self.logvar_head.bias,
            -2.0,
        )

    def initial_memory(
        self,
        batch_size: int,
        device: Optional[
            torch.device
        ] = None,
        dtype: Optional[
            torch.dtype
        ] = None,
    ) -> RecurrentMemory:
        """
        Create zero-initialized recurrent memory.
        """

        cfg = self.config

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        parameter = next(self.parameters())

        if device is None:
            device = parameter.device

        if dtype is None:
            dtype = parameter.dtype

        shape = (
            cfg.num_layers,
            batch_size,
            cfg.hidden_dim,
        )

        hidden = torch.zeros(
            shape,
            device=device,
            dtype=dtype,
        )

        if self.recurrent_type == "lstm":
            cell = torch.zeros_like(hidden)
            return hidden, cell

        return hidden

    def detach_memory(
        self,
        memory: RecurrentMemory,
    ) -> RecurrentMemory:
        """
        Detach memory from the autograd graph.
        """

        if isinstance(memory, tuple):
            return tuple(
                item.detach()
                for item in memory
            )

        return memory.detach()

    def _validate_inputs(
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

        if state.shape[0] == 0:
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
                "Invalid action dimensions"
            )

        if (
            state.device != action.device
            or state.dtype != action.dtype
        ):
            raise ValueError(
                "State and action device/dtype mismatch"
            )

        for name, tensor in (
            ("state", state),
            ("action", action),
        ):
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def _validate_memory(
        self,
        memory: RecurrentMemory,
        batch_size: int,
        reference: Tensor,
    ) -> None:

        cfg = self.config

        expected_shape = (
            cfg.num_layers,
            batch_size,
            cfg.hidden_dim,
        )

        if self.recurrent_type == "lstm":
            if (
                not isinstance(memory, tuple)
                or len(memory) != 2
            ):
                raise ValueError(
                    "LSTM memory must be (hidden, cell)"
                )

            tensors = memory
        else:
            if not isinstance(memory, Tensor):
                raise ValueError(
                    "GRU memory must be a tensor"
                )

            tensors = (memory,)

        for tensor in tensors:
            if tensor.shape != expected_shape:
                raise ValueError(
                    "Invalid recurrent memory shape"
                )

            if (
                tensor.device != reference.device
                or tensor.dtype != reference.dtype
            ):
                raise ValueError(
                    "Memory device/dtype mismatch"
                )

    def encode_transition(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Encode one state-action pair.

        Returns:
            [B,hidden_dim]
        """

        self._validate_inputs(
            state,
            action,
        )

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

        return self.fusion(fused)

    def step(
        self,
        state: Tensor,
        action: Tensor,
        memory: Optional[
            RecurrentMemory
        ] = None,
    ) -> Tuple[
        Tensor,
        Tensor,
        RecurrentMemory,
    ]:
        """
        Stateful recurrent transition.

        Returns:
            mean: [B,D]
            logvar: [B,D]
            next_memory: GRU tensor or LSTM tuple
        """

        features = self.encode_transition(
            state,
            action,
        )

        if memory is None:
            memory = self.initial_memory(
                batch_size=state.shape[0],
                device=state.device,
                dtype=state.dtype,
            )
        else:
            self._validate_memory(
                memory,
                state.shape[0],
                state,
            )

        recurrent_input = (
            features.unsqueeze(1)
        )

        output, next_memory = self.recurrent(
            recurrent_input,
            memory,
        )

        hidden = self.output_norm(
            output[:, -1, :]
        )

        mean = self.mean_head(hidden)

        if self.config.residual:
            mean = state + mean

        if self.config.stochastic:
            logvar = (
                self.logvar_head(hidden)
                .clamp(
                    self.config.min_logvar,
                    self.config.max_logvar,
                )
            )
        else:
            logvar = torch.full_like(
                mean,
                self.config.deterministic_logvar,
            )

        return mean, logvar, next_memory

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Stateless compatibility interface.

        Initializes fresh recurrent memory on
        each call.

        For true temporal memory, use step()
        or rollout().
        """

        mean, logvar, _ = self.step(
            state,
            action,
            memory=None,
        )

        return mean, logvar

    def sample_next_state(
        self,
        mean: Tensor,
        logvar: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Sample a latent transition.
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

        std = torch.exp(
            0.5 * logvar
        )

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
        memory: Optional[
            RecurrentMemory
        ] = None,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> RecurrentTransitionOutput:
        """
        Predict one transition while updating memory.
        """

        mean, logvar, next_memory = self.step(
            state,
            action,
            memory,
        )

        next_state = self.sample_next_state(
            mean,
            logvar,
            deterministic=deterministic,
            temperature=temperature,
        )

        uncertainty = (
            logvar.exp().mean(dim=-1)
        )

        return RecurrentTransitionOutput(
            mean=mean,
            logvar=logvar,
            next_state=next_state,
            uncertainty=uncertainty,
            memory=next_memory,
        )

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        memory: Optional[
            RecurrentMemory
        ] = None,
        deterministic: bool = False,
        teacher_states: Optional[
            Tensor
        ] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> RecurrentRolloutOutput:
        """
        Generate a recurrent future trajectory.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        teacher_states:
            Optional [B,H+1,D]

        Teacher forcing:
            Ground-truth states may be used as
            inputs to subsequent transitions.

        Recurrent memory is preserved throughout
        the rollout.
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
            initial_state.shape[1]
            != cfg.latent_dim
        ):
            raise ValueError(
                "Invalid latent dimension"
            )

        if (
            actions.shape[0]
            != initial_state.shape[0]
            or actions.shape[2]
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
                "Teacher states required for "
                "teacher forcing"
            )

        if teacher_states is not None:
            expected_shape = (
                initial_state.shape[0],
                actions.shape[1] + 1,
                cfg.latent_dim,
            )

            if teacher_states.shape != expected_shape:
                raise ValueError(
                    "teacher_states must be [B,H+1,D]"
                )

        for tensor in (
            initial_state,
            actions,
            teacher_states,
        ):
            if tensor is None:
                continue

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

        if memory is None:
            memory = self.initial_memory(
                batch_size=initial_state.shape[0],
                device=initial_state.device,
                dtype=initial_state.dtype,
            )
        else:
            self._validate_memory(
                memory,
                initial_state.shape[0],
                initial_state,
            )

        current = initial_state

        states = [current]
        means = []
        logvars = []
        uncertainties = []

        horizon = actions.shape[1]

        for step_index in range(horizon):

            prediction = self.predict(
                state=current,
                action=actions[:, step_index, :],
                memory=memory,
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

            memory = prediction.memory
            current = prediction.next_state

            if (
                teacher_states is not None
                and teacher_forcing_ratio > 0
                and step_index < horizon - 1
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
                    teacher_states[
                        :, step_index + 1, :
                    ],
                    current,
                )

            if (
                cfg.tbptt_steps > 0
                and (step_index + 1)
                % cfg.tbptt_steps == 0
            ):
                memory = self.detach_memory(
                    memory
                )

                # Truncate state feedback as well,
                # avoiding backpropagation through
                # earlier rollout segments.
                current = current.detach()

        return RecurrentRolloutOutput(
            states=torch.stack(
                states,
                dim=1,
            ),
            means=torch.stack(
                means,
                dim=1,
            ),
            logvars=torch.stack(
                logvars,
                dim=1,
            ),
            uncertainties=torch.stack(
                uncertainties,
                dim=1,
            ),
            final_memory=memory,
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
        memory: Optional[
            RecurrentMemory
        ] = None,
    ) -> Dict[str, object]:
        """
        One-step supervised recurrent loss.

        Returns loss tensors and updated memory.

        Preserve the returned memory across
        consecutive transitions when training
        on ordered sequences.
        """

        mean, logvar, next_memory = self.step(
            state,
            action,
            memory,
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

        return {
            "loss": prediction_loss,
            "prediction_loss": prediction_loss,
            "memory": next_memory,
        }

    def compute_rollout_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
        memory: Optional[
            RecurrentMemory
        ] = None,
    ) -> Dict[str, object]:
        """
        Multi-step supervised dynamics loss.

        target_states:
            [B,H+1,D]

        Discounted prediction loss is computed
        over the recurrent rollout.
        """

        output = self.rollout(
            initial_state=initial_state,
            actions=actions,
            memory=memory,
            deterministic=True,
            teacher_states=target_states,
            teacher_forcing_ratio=(
                teacher_forcing_ratio
            ),
        )

        targets = target_states[:, 1:, :]

        if self.config.stochastic:
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
            self.config.rollout_discount
            ** torch.arange(
                horizon,
                device=errors.device,
                dtype=errors.dtype,
            )
        )

        prediction_loss = (
            errors * discounts.unsqueeze(0)
        ).sum(dim=1).mean() / discounts.sum()

        return {
            "loss": prediction_loss,
            "prediction_loss": prediction_loss,
            "final_memory": output.final_memory,
        }

    def training_step(
        self,
        optimizer: torch.optim.Optimizer,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Dict[str, float]:
        """
        Train on a batch of temporal sequences.

        Each batch is treated as a separate
        sequence, starting with fresh memory.
        """

        self.train()

        optimizer.zero_grad(set_to_none=True)

        losses = self.compute_rollout_loss(
            initial_state=initial_state,
            actions=actions,
            target_states=target_states,
            teacher_forcing_ratio=(
                teacher_forcing_ratio
            ),
        )

        losses["loss"].backward()

        if self.config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.gradient_clip_norm,
            )

        optimizer.step()

        return {
            "loss": float(
                losses["loss"].detach().item()
            ),
            "prediction_loss": float(
                losses["prediction_loss"]
                .detach().item()
            ),
        }

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
        memory: Optional[
            RecurrentMemory
        ] = None,
        deterministic: bool = False,
    ) -> RecurrentRolloutOutput:
        """
        Inference-only recurrent rollout.
        """

        was_training = self.training
        self.eval()

        try:
            return self.rollout(
                initial_state=initial_state,
                actions=actions,
                memory=memory,
                deterministic=deterministic,
            )
        finally:
            self.train(was_training)


if __name__ == "__main__":

    torch.manual_seed(42)

    config = RecurrentDynamicsConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=2,
        recurrent_type="gru",
        stochastic=True,
        temperature=0.5,
    )

    model = RecurrentDynamics(config)

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
    )

    target_states = torch.randn(
        batch_size,
        horizon + 1,
        config.latent_dim,
    )

    # Stateless compatibility interface
    mean, logvar = model(
        initial_state,
        actions[:, 0, :],
    )

    print("Mean:", mean.shape)
    print("Logvar:", logvar.shape)

    # Stateful recurrent prediction
    memory = model.initial_memory(
        batch_size
    )

    prediction = model.predict(
        initial_state,
        actions[:, 0, :],
        memory=memory,
    )

    print(
        "Predicted state:",
        prediction.next_state.shape,
    )

    # Multi-step trajectory generation
    output = model.generate(
        initial_state,
        actions,
    )

    print(
        "Trajectory:",
        output.states.shape,
    )

    print(
        "Uncertainty:",
        output.uncertainties.shape,
    )

    # Supervised training example
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    metrics = model.training_step(
        optimizer=optimizer,
        initial_state=initial_state,
        actions=actions,
        target_states=target_states,
        teacher_forcing_ratio=0.5,
    )

    print("Training metrics:", metrics)

    # LSTM example
    lstm_model = RecurrentDynamics(
        RecurrentDynamicsConfig(
            latent_dim=128,
            action_dim=32,
            recurrent_type="lstm",
        )
    )

    lstm_output = lstm_model.generate(
        initial_state,
        actions,
        deterministic=True,
    )

    print(
        "LSTM trajectory:",
        lstm_output.states.shape,
    )
