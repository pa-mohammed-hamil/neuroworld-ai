
```python
"""
NeuroWorld AI
File: neuroworld/dynamics/transition_model.py

Unified Neural State Transition Model

Supported backends:
    deterministic
    stochastic
    latent
    recurrent
    transformer
    neural

Features:
- Unified state-action transition interface
- Deterministic and probabilistic prediction
- Gaussian mean and log-variance outputs
- Model-specific backend selection
- Recurrent hidden-state memory
- Transformer historical context
- Multi-step latent rollouts
- Teacher forcing
- Gaussian NLL / MSE training losses
- Gradient clipping
- Model checkpoint saving and loading
- CPU / CUDA support

Standard interface:
    mean, logvar = model(state, action)

Extended interface:
    prediction = model.predict(state, action)

Rollout:
    output = model.rollout(initial_state, actions)

Dependencies:
    pip install torch
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TransitionModelConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    backend: str = "stochastic"

    num_layers: int = 3
    dropout: float = 0.0

    residual: bool = True
    stochastic: bool = True
    temperature: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    recurrent_type: str = "gru"

    num_heads: int = 8
    max_context_length: int = 64

    distribution: str = "gaussian"
    num_mixtures: int = 5

    rollout_discount: float = 0.99
    gradient_clip_norm: float = 1.0


@dataclass
class TransitionPrediction:
    mean: Tensor
    logvar: Tensor
    next_state: Tensor
    uncertainty: Tensor

    memory: Optional[Any] = None

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()

    @property
    def std(self) -> Tensor:
        return torch.exp(0.5 * self.logvar)


@dataclass
class TransitionRolloutOutput:
    """
    states:        [B,H+1,D]
    means:         [B,H,D]
    logvars:       [B,H,D]
    uncertainties: [B,H]
    actions:       [B,H,A]
    """

    states: Tensor
    means: Tensor
    logvars: Tensor
    uncertainties: Tensor
    actions: Tensor

    final_memory: Optional[Any] = None

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]


class TransitionModel(nn.Module):
    """
    Unified interface for NeuroWorld dynamics.

    Supported backends:

        deterministic:
            DeterministicDynamics

        stochastic:
            StochasticDynamics

        latent:
            LatentDynamics

        recurrent:
            RecurrentDynamics

        transformer:
            TransformerDynamics

        neural:
            DynamicsModel

    Every backend is normalized to:

        mean, logvar = model(state, action)

    Stateful recurrent and context-aware Transformer
    prediction are supported through rollout().
    """

    BACKENDS = {
        "deterministic",
        "stochastic",
        "latent",
        "recurrent",
        "transformer",
        "neural",
    }

    def __init__(
        self,
        config: Optional[TransitionModelConfig] = None,
    ):
        super().__init__()

        self.config = config or TransitionModelConfig()

        self._validate_config()

        self.backend = self.config.backend.lower()

        self.model = self._build_backend()

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.backend.lower() not in self.BACKENDS:
            raise ValueError(
                f"Unsupported backend: {cfg.backend}. "
                f"Choose from {sorted(self.BACKENDS)}"
            )

        if min(
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_layers,
        ) <= 0:
            raise ValueError(
                "Dimensions and num_layers must be positive"
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
                "Invalid log-variance range"
            )

        if not 0 <= cfg.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in [0,1]"
            )

        if cfg.gradient_clip_norm < 0:
            raise ValueError(
                "gradient_clip_norm must be non-negative"
            )

    def _build_backend(self) -> nn.Module:
        """
        Construct the selected dynamics backend.

        Imports are intentionally local to avoid
        unnecessary dependencies and import cycles.
        """

        cfg = self.config

        common = dict(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            residual=cfg.residual,
        )

        if self.backend == "deterministic":
            from neuroworld.dynamics.deterministic_dynamics import (
                DeterministicDynamics,
                DeterministicDynamicsConfig,
            )

            backend_config = DeterministicDynamicsConfig(
                **common,
            )

            return DeterministicDynamics(
                backend_config
            )

        if self.backend == "stochastic":
            from neuroworld.dynamics.stochastic_dynamics import (
                StochasticDynamics,
                StochasticDynamicsConfig,
            )

            backend_config = StochasticDynamicsConfig(
                **common,
                distribution=cfg.distribution,
                num_mixtures=cfg.num_mixtures,
                temperature=cfg.temperature,
                min_logvar=cfg.min_logvar,
                max_logvar=cfg.max_logvar,
            )

            return StochasticDynamics(
                backend_config
            )

        if self.backend == "latent":
            from neuroworld.dynamics.latent_dynamics import (
                LatentDynamics,
                LatentDynamicsConfig,
            )

            backend_config = LatentDynamicsConfig(
                **common,
                stochastic=cfg.stochastic,
                temperature=cfg.temperature,
                min_logvar=cfg.min_logvar,
                max_logvar=cfg.max_logvar,
            )

            return LatentDynamics(
                backend_config
            )

        if self.backend == "recurrent":
            from neuroworld.dynamics.recurrent_dynamics import (
                RecurrentDynamics,
                RecurrentDynamicsConfig,
            )

            backend_config = RecurrentDynamicsConfig(
                **common,
                recurrent_type=cfg.recurrent_type,
                stochastic=cfg.stochastic,
                temperature=cfg.temperature,
                min_logvar=cfg.min_logvar,
                max_logvar=cfg.max_logvar,
            )

            return RecurrentDynamics(
                backend_config
            )

        if self.backend == "transformer":
            from neuroworld.dynamics.transformer_dynamics import (
                TransformerDynamics,
                TransformerDynamicsConfig,
            )

            backend_config = TransformerDynamicsConfig(
                **common,
                num_heads=cfg.num_heads,
                max_context_length=cfg.max_context_length,
                stochastic=cfg.stochastic,
                temperature=cfg.temperature,
                min_logvar=cfg.min_logvar,
                max_logvar=cfg.max_logvar,
            )

            return TransformerDynamics(
                backend_config
            )

        from neuroworld.dynamics.dynamics_model import (
            DynamicsModel,
            DynamicsModelConfig,
        )

        backend_config = DynamicsModelConfig(
            **common,
            stochastic=cfg.stochastic,
            temperature=cfg.temperature,
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )

        return DynamicsModel(
            backend_config
        )

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
                "State/action device or dtype mismatch"
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

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Standard probabilistic dynamics interface.

        Returns:
            mean:   [B,D]
            logvar: [B,D]

        For deterministic backends, logvar is
        a fixed placeholder, not a calibrated
        uncertainty estimate.
        """

        self._validate_inputs(
            state,
            action,
        )

        if self.backend == "deterministic":
            mean = self.model(
                state,
                action,
            )

            logvar = torch.full_like(
                mean,
                self.config.min_logvar,
            )

            return mean, logvar

        return self.model(
            state,
            action,
        )

    def sample_next_state(
        self,
        mean: Tensor,
        logvar: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Sample a Gaussian next-state prediction.

        The stochastic mixture backend uses its
        own sampler through predict().
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
            or self.backend == "deterministic"
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
        deterministic: bool = False,
        memory: Optional[Any] = None,
    ) -> TransitionPrediction:
        """
        Predict one transition.

        For recurrent backends, memory can be
        supplied and is returned after updating.

        For stochastic mixture backends, use
        the native mixture sampler.
        """

        self._validate_inputs(
            state,
            action,
        )

        if self.backend == "recurrent":
            mean, logvar, next_memory = (
                self.model.step(
                    state,
                    action,
                    memory,
                )
            )

            next_state = self.sample_next_state(
                mean,
                logvar,
                deterministic=deterministic,
            )

        elif self.backend == "stochastic":
            prediction = self.model.predict(
                state,
                action,
                deterministic=deterministic,
            )

            mean = prediction.mean
            logvar = prediction.logvar
            next_state = prediction.next_state
            next_memory = None

        else:
            mean, logvar = self.forward(
                state,
                action,
            )

            next_state = self.sample_next_state(
                mean,
                logvar,
                deterministic=deterministic,
            )

            next_memory = None

        uncertainty = (
            logvar.exp().mean(dim=-1)
        )

        return TransitionPrediction(
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
        deterministic: bool = False,
        teacher_states: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        memory: Optional[Any] = None,
        history_states: Optional[Tensor] = None,
        history_actions: Optional[Tensor] = None,
    ) -> TransitionRolloutOutput:
        """
        Multi-step latent transition rollout.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        teacher_states:
            Optional [B,H+1,D]

        Recurrent backend:
            Preserves hidden-state memory.

        Transformer backend:
            Preserves historical attention context.

        Other backends:
            Uses standard autoregressive rollout.
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

        if self.backend == "recurrent":
            output = self.model.rollout(
                initial_state=initial_state,
                actions=actions,
                memory=memory,
                deterministic=deterministic,
                teacher_states=teacher_states,
                teacher_forcing_ratio=(
                    teacher_forcing_ratio
                ),
            )

            return TransitionRolloutOutput(
                states=output.states,
                means=output.means,
                logvars=output.logvars,
                uncertainties=output.uncertainties,
                actions=actions,
                final_memory=output.final_memory,
            )

        if self.backend == "transformer":
            output = self.model.rollout(
                initial_state=initial_state,
                actions=actions,
                history_states=history_states,
                history_actions=history_actions,
                deterministic=deterministic,
                teacher_states=teacher_states,
                teacher_forcing_ratio=(
                    teacher_forcing_ratio
                ),
            )

            return TransitionRolloutOutput(
                states=output.states,
                means=output.means,
                logvars=output.logvars,
                uncertainties=output.uncertainties,
                actions=actions,
            )

        current = initial_state

        states = [current]
        means = []
        logvars = []
        uncertainties = []

        horizon = actions.shape[1]

        for step in range(horizon):

            prediction = self.predict(
                state=current,
                action=actions[:, step, :],
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

        return TransitionRolloutOutput(
            states=torch.stack(states, dim=1),
            means=torch.stack(means, dim=1),
            logvars=torch.stack(logvars, dim=1),
            uncertainties=torch.stack(
                uncertainties,
                dim=1,
            ),
            actions=actions,
        )

    def compute_loss(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, Tensor]:
        """
        One-step supervised transition loss.

        Uses native mixture NLL when the selected
        backend is a Gaussian mixture model.
        """

        self._validate_inputs(
            state,
            action,
        )

        if next_state.shape != state.shape:
            raise ValueError(
                "next_state must be [B,D]"
            )

        if self.backend == "stochastic":
            return self.model.compute_loss(
                state,
                action,
                next_state,
            )

        mean, logvar = self.forward(
            state,
            action,
        )

        if (
            self.backend == "deterministic"
            or not self.config.stochastic
        ):
            prediction_loss = F.mse_loss(
                mean,
                next_state,
            )
        else:
            prediction_loss = (
                0.5
                * (
                    math_log_2pi()
                    + logvar
                    + (next_state - mean).pow(2)
                    * torch.exp(-logvar)
                )
            ).mean()

        return {
            "loss": prediction_loss,
            "prediction_loss": prediction_loss,
        }

    def compute_rollout_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Multi-step supervised transition loss.

        target_states:
            [B,H+1,D]

        Uses the selected backend's native
        rollout loss when available.
        """

        if hasattr(
            self.model,
            "compute_rollout_loss",
        ):
            result = self.model.compute_rollout_loss(
                initial_state=initial_state,
                actions=actions,
                target_states=target_states,
                teacher_forcing_ratio=(
                    teacher_forcing_ratio
                ),
            )

            return {
                key: value
                for key, value in result.items()
                if isinstance(value, Tensor)
                and value.ndim == 0
            }

        if hasattr(
            self.model,
            "compute_training_loss",
        ):
            return self.model.compute_training_loss(
                initial_state=initial_state,
                actions=actions,
                target_states=target_states,
                teacher_forcing_ratio=(
                    teacher_forcing_ratio
                ),
            )

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

        if (
            self.backend == "deterministic"
            or not self.config.stochastic
        ):
            errors = (
                output.means - targets
            ).pow(2).mean(dim=-1)
        else:
            errors = (
                0.5
                * (
                    math_log_2pi()
                    + output.logvars
                    + (targets - output.means).pow(2)
                    * torch.exp(-output.logvars)
                )
            ).mean(dim=-1)

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
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, float]:
        """
        One supervised optimizer step.
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
            if isinstance(value, Tensor)
            and value.ndim == 0
        }

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> TransitionRolloutOutput:
        """
        Inference-only future trajectory generation.
        """

        was_training = self.training
        self.eval()

        try:
            return self.rollout(
                initial_state=initial_state,
                actions=actions,
                deterministic=deterministic,
                **kwargs,
            )
        finally:
            self.train(was_training)

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """
        Save model configuration and weights.
        """

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "backend": self.backend,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "TransitionModel":
        """
        Load a trusted NeuroWorld checkpoint.

        Uses weights_only=True to avoid unsafe
        general-purpose pickle deserialization.
        """

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        config = TransitionModelConfig(
            **checkpoint["config"]
        )

        model = cls(config)

        model.load_state_dict(
            checkpoint["state_dict"]
        )

        model.eval()

        return model


def math_log_2pi() -> float:
    """Natural logarithm of 2*pi."""

    import math

    return math.log(2.0 * math.pi)


if __name__ == "__main__":

    torch.manual_seed(42)

    config = TransitionModelConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        backend="stochastic",
        distribution="gaussian",
        stochastic=True,
    )

    model = TransitionModel(config)

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

    prediction = model.predict(
        state,
        action,
    )

    print(
        "Predicted state:",
        prediction.next_state.shape,
    )

    print(
        "Uncertainty:",
        prediction.uncertainty.shape,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    metrics = model.training_step(
        optimizer=optimizer,
        state=state,
        action=action,
        next_state=next_state,
    )

    print("Training metrics:", metrics)

    actions = torch.randn(
        batch_size,
        horizon,
        config.action_dim,
    )

    rollout = model.generate(
        initial_state=state,
        actions=actions,
    )

    print(
        "Generated trajectory:",
        rollout.states.shape,
    )

    print(
        "Final states:",
        rollout.final_states.shape,
    )

    model.save_checkpoint(
        "checkpoints/transition_model.pt"
    )

    restored = TransitionModel.load_checkpoint(
        "checkpoints/transition_model.pt"
    )

    print(
        "Restored backend:",
        restored.backend,
    )
```
