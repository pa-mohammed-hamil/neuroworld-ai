
"""
NeuroWorld AI
File: neuroworld/dynamics/stochastic_dynamics.py

Stochastic Neural Dynamics Model

Features:
- Action-conditioned stochastic latent transitions
- Diagonal Gaussian and Gaussian mixture models
- Multimodal future-state distributions
- Reparameterized Gaussian sampling
- Monte Carlo future trajectory generation
- Aleatoric uncertainty estimation
- Predictive entropy approximation
- Residual latent-state transitions
- Gaussian / mixture negative log-likelihood
- Multi-step rollout with teacher forcing
- Gradient clipping and supervised training
- Stateless compatibility with imagination modules
- CPU / CUDA support

Dependencies:
    pip install torch

Standard interface:
    mean, logvar = model(state, action)

Extended interface:
    prediction = model.predict(state, action)

Shapes:
    state:  [B,D]
    action: [B,A]
    mean:   [B,D]
    logvar: [B,D]
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
class StochasticDynamicsConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    num_layers: int = 3
    dropout: float = 0.0

    distribution: str = "gaussian"
    num_mixtures: int = 5

    residual: bool = True
    use_layer_norm: bool = True
    use_action_gating: bool = True

    temperature: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    rollout_discount: float = 0.99
    gradient_clip_norm: float = 1.0

    # Optional regularization on predicted
    # latent-state changes.
    transition_regularization: float = 0.0


@dataclass
class StochasticPrediction:
    mean: Tensor
    logvar: Tensor

    next_state: Tensor

    # Mean predictive variance across latent
    # dimensions. Includes mixture spread.
    uncertainty: Tensor

    # Gaussian moment-matched entropy estimate.
    entropy: Tensor

    mixture_logits: Optional[Tensor] = None
    component_means: Optional[Tensor] = None
    component_logvars: Optional[Tensor] = None

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()


@dataclass
class StochasticRolloutOutput:
    """
    states:
        [B,H+1,D]

    means:
        [B,H,D]

    logvars:
        [B,H,D]

    uncertainties:
        [B,H]

    entropies:
        [B,H]
    """

    states: Tensor
    means: Tensor
    logvars: Tensor

    uncertainties: Tensor
    entropies: Tensor

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]

    @property
    def horizon(self) -> int:
        return self.means.shape[1]


@dataclass
class MonteCarloRolloutOutput:
    """
    trajectories:
        [B,N,H+1,D]

    final_mean:
        [B,D]

    final_variance:
        [B,D]

    trajectory_uncertainties:
        [B,N,H]
    """

    trajectories: Tensor
    final_mean: Tensor
    final_variance: Tensor
    trajectory_uncertainties: Tensor


class StochasticResidualBlock(nn.Module):
    """Residual feedforward neural block."""

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


class StochasticDynamics(nn.Module):
    """
    Action-conditioned stochastic dynamics.

    Supports:

    1. Gaussian transitions:
       p(z_next | z, a) = N(mean, variance)

    2. Gaussian mixture transitions:
       p(z_next | z, a)
           = sum_k weight_k * N(mean_k, variance_k)

    forward() returns moment-matched Gaussian
    parameters for compatibility with existing
    NeuroWorld imagination modules.

    For exact mixture sampling and mixture NLL,
    use predict() and compute_loss().
    """

    def __init__(
        self,
        config: Optional[
            StochasticDynamicsConfig
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or StochasticDynamicsConfig()
        )

        cfg = self.config

        if min(
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_layers,
        ) <= 0:
            raise ValueError(
                "Dimensions and layer count "
                "must be positive"
            )

        if cfg.distribution not in (
            "gaussian",
            "mixture",
        ):
            raise ValueError(
                "distribution must be "
                "'gaussian' or 'mixture'"
            )

        if cfg.num_mixtures < 1:
            raise ValueError(
                "num_mixtures must be positive"
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

        if cfg.transition_regularization < 0:
            raise ValueError(
                "transition_regularization "
                "must be non-negative"
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

        self.blocks = nn.Sequential(
            *[
                StochasticResidualBlock(
                    cfg.hidden_dim,
                    cfg.dropout,
                    cfg.use_layer_norm,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        self.output_norm = (
            nn.LayerNorm(cfg.hidden_dim)
            if cfg.use_layer_norm
            else nn.Identity()
        )

        num_components = (
            cfg.num_mixtures
            if cfg.distribution == "mixture"
            else 1
        )

        self.num_components = num_components

        self.mean_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim * num_components,
        )

        self.logvar_head = nn.Linear(
            cfg.hidden_dim,
            cfg.latent_dim * num_components,
        )

        self.mixture_head = (
            nn.Linear(
                cfg.hidden_dim,
                num_components,
            )
            if cfg.distribution == "mixture"
            else None
        )

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

    def encode_transition(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        """
        Encode state-action features.

        Returns:
            [B,hidden_dim]
        """

        self._validate_transition(
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

        hidden = torch.cat(
            [
                state_features,
                action_features,
            ],
            dim=-1,
        )

        hidden = self.fusion(hidden)
        hidden = self.blocks(hidden)

        return self.output_norm(hidden)

    def distribution_parameters(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Return Gaussian mixture parameters.

        component_means:
            [B,K,D]

        component_logvars:
            [B,K,D]

        mixture_logits:
            [B,K]

        For Gaussian mode, K=1.
        """

        cfg = self.config

        hidden = self.encode_transition(
            state,
            action,
        )

        batch_size = state.shape[0]

        means = self.mean_head(hidden).reshape(
            batch_size,
            self.num_components,
            cfg.latent_dim,
        )

        if cfg.residual:
            means = (
                means + state.unsqueeze(1)
            )

        logvars = (
            self.logvar_head(hidden)
            .reshape(
                batch_size,
                self.num_components,
                cfg.latent_dim,
            )
            .clamp(
                cfg.min_logvar,
                cfg.max_logvar,
            )
        )

        if self.mixture_head is None:
            logits = state.new_zeros(
                batch_size,
                1,
            )
        else:
            logits = self.mixture_head(hidden)

        return means, logvars, logits

    @staticmethod
    def mixture_moments(
        component_means: Tensor,
        component_logvars: Tensor,
        mixture_logits: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Moment-match a Gaussian mixture.

        Returns:
            mean: [B,D]
            logvar: [B,D]

        Note:
            Moment matching does not preserve
            multimodal distribution shape.
        """

        weights = torch.softmax(
            mixture_logits,
            dim=-1,
        ).unsqueeze(-1)

        mean = (
            weights * component_means
        ).sum(dim=1)

        second_moment = (
            weights
            * (
                component_logvars.exp()
                + component_means.pow(2)
            )
        ).sum(dim=1)

        variance = (
            second_moment - mean.pow(2)
        ).clamp_min(1e-8)

        return mean, variance.log()

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Standard NeuroWorld dynamics interface.

        Returns:
            mean: [B,D]
            logvar: [B,D]

        Mixture mode returns moment-matched
        Gaussian parameters.
        """

        (
            component_means,
            component_logvars,
            mixture_logits,
        ) = self.distribution_parameters(
            state,
            action,
        )

        return self.mixture_moments(
            component_means,
            component_logvars,
            mixture_logits,
        )

    def sample_distribution(
        self,
        component_means: Tensor,
        component_logvars: Tensor,
        mixture_logits: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> Tensor:
        """
        Sample a transition.

        Gaussian mode:
            Reparameterized normal sampling.

        Mixture mode:
            Categorical component selection
            followed by Gaussian sampling.

        Categorical selection is not differentiable
        with respect to mixture logits.
        """

        if temperature is None:
            temperature = self.config.temperature

        if temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if deterministic or temperature == 0:
            mean, _ = self.mixture_moments(
                component_means,
                component_logvars,
                mixture_logits,
            )

            return mean

        batch_size = component_means.shape[0]

        if self.num_components == 1:
            selected_mean = (
                component_means[:, 0, :]
            )

            selected_logvar = (
                component_logvars[:, 0, :]
            )
        else:
            distribution = (
                torch.distributions.Categorical(
                    logits=mixture_logits
                )
            )

            indices = distribution.sample()

            batch_indices = torch.arange(
                batch_size,
                device=indices.device,
            )

            selected_mean = component_means[
                batch_indices,
                indices,
            ]

            selected_logvar = component_logvars[
                batch_indices,
                indices,
            ]

        std = torch.exp(
            0.5 * selected_logvar
        )

        noise = torch.randn_like(
            selected_mean
        )

        return (
            selected_mean
            + temperature * std * noise
        )

    def predict(
        self,
        state: Tensor,
        action: Tensor,
        deterministic: bool = False,
        temperature: Optional[float] = None,
    ) -> StochasticPrediction:
        """
        Predict next state and distribution metrics.
        """

        (
            component_means,
            component_logvars,
            mixture_logits,
        ) = self.distribution_parameters(
            state,
            action,
        )

        mean, logvar = self.mixture_moments(
            component_means,
            component_logvars,
            mixture_logits,
        )

        next_state = self.sample_distribution(
            component_means,
            component_logvars,
            mixture_logits,
            deterministic=deterministic,
            temperature=temperature,
        )

        uncertainty = (
            logvar.exp().mean(dim=-1)
        )

        # Entropy of the moment-matched Gaussian,
        # not the exact entropy of the mixture.
        entropy = (
            0.5
            * (
                logvar
                + LOG_2PI
                + 1.0
            )
        ).sum(dim=-1)

        return StochasticPrediction(
            mean=mean,
            logvar=logvar,
            next_state=next_state,
            uncertainty=uncertainty,
            entropy=entropy,
            mixture_logits=mixture_logits,
            component_means=component_means,
            component_logvars=component_logvars,
        )

    @staticmethod
    def gaussian_nll(
        target: Tensor,
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        Gaussian negative log-likelihood.

        Returns:
            [B] per-sample NLL
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
        ).sum(dim=-1)

    def mixture_nll(
        self,
        target: Tensor,
        component_means: Tensor,
        component_logvars: Tensor,
        mixture_logits: Tensor,
    ) -> Tensor:
        """
        Exact Gaussian mixture NLL.

        Returns:
            [B] per-sample NLL
        """

        if target.ndim != 2:
            raise ValueError(
                "target must be [B,D]"
            )

        if (
            component_means.shape
            != component_logvars.shape
        ):
            raise ValueError(
                "Mixture parameter shape mismatch"
            )

        if (
            component_means.shape[0]
            != target.shape[0]
            or component_means.shape[-1]
            != target.shape[-1]
        ):
            raise ValueError(
                "Target dimension mismatch"
            )

        component_nll = self.gaussian_nll(
            target.unsqueeze(1).expand_as(
                component_means
            ),
            component_means,
            component_logvars,
        )

        log_weights = F.log_softmax(
            mixture_logits,
            dim=-1,
        )

        log_probability = torch.logsumexp(
            log_weights - component_nll,
            dim=-1,
        )

        return -log_probability

    def compute_loss(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Supervised one-step stochastic loss.
        """

        (
            component_means,
            component_logvars,
            mixture_logits,
        ) = self.distribution_parameters(
            state,
            action,
        )

        if next_state.shape != state.shape:
            raise ValueError(
                "next_state must be [B,D]"
            )

        if self.config.distribution == "mixture":
            nll = self.mixture_nll(
                next_state,
                component_means,
                component_logvars,
                mixture_logits,
            )
        else:
            nll = self.gaussian_nll(
                next_state,
                component_means[:, 0, :],
                component_logvars[:, 0, :],
            )

        prediction_loss = nll.mean()

        mean, _ = self.mixture_moments(
            component_means,
            component_logvars,
            mixture_logits,
        )

        transition_penalty = (
            mean - state
        ).pow(2).mean()

        total_loss = (
            prediction_loss
            + self.config.transition_regularization
            * transition_penalty
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "transition_penalty": transition_penalty,
        }

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        deterministic: bool = False,
        teacher_states: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> StochasticRolloutOutput:
        """
        Generate a multi-step stochastic trajectory.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        teacher_states:
            Optional [B,H+1,D]

        Repeated stochastic rollouts can produce
        different possible futures.
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
        entropies = []

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

            entropies.append(
                prediction.entropy
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

        return StochasticRolloutOutput(
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
            entropies=torch.stack(
                entropies,
                dim=1,
            ),
        )

    def compute_rollout_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Discounted multi-step supervised NLL.

        Evaluates the full transition distribution
        at each rollout step.
        """

        cfg = self.config

        if target_states.shape != (
            initial_state.shape[0],
            actions.shape[1] + 1,
            cfg.latent_dim,
        ):
            raise ValueError(
                "target_states must be [B,H+1,D]"
            )

        # Use deterministic feedback for stable
        # supervised rollout optimization.
        current = initial_state
        horizon = actions.shape[1]

        per_step_losses = []

        for step in range(horizon):

            action = actions[:, step, :]
            target = target_states[:, step + 1, :]

            (
                component_means,
                component_logvars,
                mixture_logits,
            ) = self.distribution_parameters(
                current,
                action,
            )

            if cfg.distribution == "mixture":
                nll = self.mixture_nll(
                    target,
                    component_means,
                    component_logvars,
                    mixture_logits,
                )
            else:
                nll = self.gaussian_nll(
                    target,
                    component_means[:, 0, :],
                    component_logvars[:, 0, :],
                )

            per_step_losses.append(nll)

            current, _ = self.mixture_moments(
                component_means,
                component_logvars,
                mixture_logits,
            )

            if (
                teacher_forcing_ratio > 0
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
                    target,
                    current,
                )

        errors = torch.stack(
            per_step_losses,
            dim=1,
        )

        discounts = (
            cfg.rollout_discount
            ** torch.arange(
                horizon,
                device=errors.device,
                dtype=errors.dtype,
            )
        )

        rollout_loss = (
            errors * discounts.unsqueeze(0)
        ).sum(dim=1).mean() / discounts.sum()

        return {
            "loss": rollout_loss,
            "rollout_loss": rollout_loss,
        }

    def training_step(
        self,
        optimizer: torch.optim.Optimizer,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, float]:
        """
        Perform one supervised optimizer step.
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
    ) -> StochasticRolloutOutput:
        """
        Inference-only trajectory generation.
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

    @torch.no_grad()
    def monte_carlo_rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        num_samples: int = 32,
    ) -> MonteCarloRolloutOutput:
        """
        Simulate multiple possible futures.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        Returns:
            trajectories: [B,N,H+1,D]

        Each sample uses independent stochastic
        transition draws.
        """

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be positive"
            )

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B,D]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B,H,A]"
            )

        batch_size = initial_state.shape[0]
        horizon = actions.shape[1]

        expanded_states = (
            initial_state.unsqueeze(1)
            .expand(-1, num_samples, -1)
            .reshape(
                batch_size * num_samples,
                self.config.latent_dim,
            )
        )

        expanded_actions = (
            actions.unsqueeze(1)
            .expand(-1, num_samples, -1, -1)
            .reshape(
                batch_size * num_samples,
                horizon,
                self.config.action_dim,
            )
        )

        output = self.generate(
            initial_state=expanded_states,
            actions=expanded_actions,
            deterministic=False,
        )

        trajectories = output.states.reshape(
            batch_size,
            num_samples,
            horizon + 1,
            self.config.latent_dim,
        )

        uncertainties = (
            output.uncertainties.reshape(
                batch_size,
                num_samples,
                horizon,
            )
        )

        final_states = trajectories[:, :, -1, :]

        final_mean = final_states.mean(dim=1)

        final_variance = final_states.var(
            dim=1,
            unbiased=False,
        )

        return MonteCarloRolloutOutput(
            trajectories=trajectories,
            final_mean=final_mean,
            final_variance=final_variance,
            trajectory_uncertainties=uncertainties,
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    config = StochasticDynamicsConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=3,
        distribution="mixture",
        num_mixtures=5,
        temperature=0.5,
    )

    model = StochasticDynamics(config)

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

    rollout_output = model.generate(
        initial_state=state,
        actions=actions,
    )

    print(
        "Trajectory:",
        rollout_output.states.shape,
    )

    monte_carlo = model.monte_carlo_rollout(
        initial_state=state,
        actions=actions,
        num_samples=16,
    )

    print(
        "Monte Carlo trajectories:",
        monte_carlo.trajectories.shape,
    )

    print(
        "Final-state variance:",
        monte_carlo.final_variance.shape,
    )
