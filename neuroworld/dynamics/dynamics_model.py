
"""
NeuroWorld AI
File: neuroworld/dynamics/dynamics_model.py

Neural World Dynamics Model

Features:
- Action-conditioned latent state transitions
- Deterministic and stochastic dynamics
- Residual state prediction
- Gaussian probabilistic transitions
- Aleatoric uncertainty estimation
- Multi-step latent rollouts
- Teacher forcing during training
- One-step and multi-step training losses
- Optional reward and termination heads
- Gradient-based optimization
- CPU / CUDA support

Dependencies:
    pip install torch

Main interface:
    mean, logvar = model(state, action)

Shapes:
    state:  [B, latent_dim]
    action: [B, action_dim]
    mean:   [B, latent_dim]
    logvar: [B, latent_dim]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DynamicsModelConfig:
    latent_dim: int = 128
    action_dim: int = 32

    hidden_dim: int = 256
    num_layers: int = 3

    dropout: float = 0.0
    residual: bool = True

    stochastic: bool = True
    temperature: float = 1.0

    use_layer_norm: bool = True
    use_action_gating: bool = True

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    predict_reward: bool = True
    predict_termination: bool = True

    prediction_loss_weight: float = 1.0
    reward_loss_weight: float = 0.1
    termination_loss_weight: float = 0.1

    rollout_discount: float = 0.99
    gradient_clip_norm: float = 1.0


@dataclass
class DynamicsPrediction:
    mean: Tensor
    logvar: Tensor
    next_state: Tensor
    uncertainty: Tensor

    reward: Optional[Tensor] = None
    termination_logit: Optional[Tensor] = None

    @property
    def variance(self) -> Tensor:
        return self.logvar.exp()

    @property
    def termination_probability(
        self,
    ) -> Optional[Tensor]:
        if self.termination_logit is None:
            return None

        return torch.sigmoid(
            self.termination_logit
        )


@dataclass
class DynamicsRolloutOutput:
    """
    states: [B,H+1,D]
    means: [B,H,D]
    logvars: [B,H,D]
    uncertainties: [B,H]
    rewards: [B,H], optional
    termination_probabilities: [B,H], optional
    """

    states: Tensor
    means: Tensor
    logvars: Tensor
    uncertainties: Tensor

    rewards: Optional[Tensor] = None
    termination_probabilities: Optional[Tensor] = None

    @property
    def final_states(self) -> Tensor:
        return self.states[:, -1, :]


class DynamicsResidualBlock(nn.Module):
    """
    Residual feedforward dynamics block.
    """

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


class DynamicsModel(nn.Module):
    """
    Neural action-conditioned world dynamics.

    Predicts a Gaussian distribution over the next
    latent state:

        p(z[t+1] | z[t], a[t])

    Returns:
        mean, logvar

    Compatible with:
        LatentRollout
        WorldRollout
        FutureGenerator
        BranchGenerator
        CounterfactualEngine
        ScenarioGenerator
        TrajectoryGenerator
        ImaginationEngine
    """

    def __init__(
        self,
        config: Optional[DynamicsModelConfig] = None,
    ):
        super().__init__()

        self.config = config or DynamicsModelConfig()
        cfg = self.config

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
                "Invalid log-variance limits"
            )

        if not 0 <= cfg.rollout_discount <= 1:
            raise ValueError(
                "rollout_discount must be in [0,1]"
            )

        for name in (
            "prediction_loss_weight",
            "reward_loss_weight",
            "termination_loss_weight",
        ):
            if getattr(cfg, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative"
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
                DynamicsResidualBlock(
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

        self.reward_head = (
            nn.Linear(cfg.hidden_dim, 1)
            if cfg.predict_reward
            else None
        )

        self.termination_head = (
            nn.Linear(cfg.hidden_dim, 1)
            if cfg.predict_termination
            else None
        )

        # Initialize residual mean near identity.
        nn.init.normal_(
            self.mean_head.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.mean_head.bias)

        # Initial Gaussian variance exp(-2).
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(
            self.logvar_head.bias,
            -2.0,
        )

    def _validate_inputs(
        self,
        state: Tensor,
        action: Tensor,
    ) -> None:
        cfg = self.config

        if state.ndim != 2 or action.ndim != 2:
            raise ValueError(
                "state and action must be 2D tensors"
            )

        if state.shape[0] == 0:
            raise ValueError("Empty batch")

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
                "State and action must share device/dtype"
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
            hidden: [B,hidden_dim]
        """

        self._validate_inputs(state, action)

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

    def _predict_heads(
        self,
        state: Tensor,
        action: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
    ]:
        hidden = self.encode_transition(
            state,
            action,
        )

        mean = self.mean_head(hidden)

        if self.config.residual:
            mean = state + mean

        logvar = self.logvar_head(hidden).clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )

        if not self.config.stochastic:
            # The interface remains probabilistic,
            # but forward sampling can be disabled.
            # This fixed variance is not calibrated.
            logvar = torch.full_like(
                mean,
                self.config.min_logvar,
            )

        reward = None

        if self.reward_head is not None:
            reward = self.reward_head(
                hidden
            ).squeeze(-1)

        termination_logit = None

        if self.termination_head is not None:
            termination_logit = (
                self.termination_head(hidden)
                .squeeze(-1)
            )

        return (
            mean,
            logvar,
            reward,
            termination_logit,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Standard interface for imagination modules.

        Returns:
            mean: [B,D]
            logvar: [B,D]
        """

        mean, logvar, _, _ = self._predict_heads(
            state,
            action,
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
        Sample next latent state from predicted
        Gaussian distribution.
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
    ) -> DynamicsPrediction:
        """
        Predict next state and auxiliary outputs.
        """

        (
            mean,
            logvar,
            reward,
            termination_logit,
        ) = self._predict_heads(state, action)

        next_state = self.sample_next_state(
            mean,
            logvar,
            deterministic=deterministic,
            temperature=temperature,
        )

        uncertainty = (
            logvar.exp().mean(dim=-1)
        )

        return DynamicsPrediction(
            mean=mean,
            logvar=logvar,
            next_state=next_state,
            uncertainty=uncertainty,
            reward=reward,
            termination_logit=termination_logit,
        )

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        deterministic: bool = False,
        teacher_states: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> DynamicsRolloutOutput:
        """
        Predict a multi-step future trajectory.

        initial_state:
            [B,D]

        actions:
            [B,H,A]

        teacher_states:
            Optional [B,H+1,D]

        teacher_forcing_ratio:
            Probability of feeding the ground-truth
            state into the next transition.

        Output states always contain predictions.
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

        for value in (
            actions,
            teacher_states,
        ):
            if value is None:
                continue

            if (
                value.device != initial_state.device
                or value.dtype != initial_state.dtype
            ):
                raise ValueError(
                    "All inputs must share device/dtype"
                )

            if not torch.isfinite(value).all().item():
                raise ValueError(
                    "Input contains NaN or Inf"
                )

        current = initial_state

        states = [current]
        means = []
        logvars = []
        uncertainties = []

        rewards = []
        termination_probabilities = []

        horizon = actions.shape[1]

        for step in range(horizon):

            prediction = self.predict(
                current,
                actions[:, step, :],
                deterministic=deterministic,
            )

            states.append(prediction.next_state)
            means.append(prediction.mean)
            logvars.append(prediction.logvar)
            uncertainties.append(
                prediction.uncertainty
            )

            if prediction.reward is not None:
                rewards.append(prediction.reward)

            if (
                prediction.termination_probability
                is not None
            ):
                termination_probabilities.append(
                    prediction.termination_probability
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

        return DynamicsRolloutOutput(
            states=torch.stack(states, dim=1),
            means=torch.stack(means, dim=1),
            logvars=torch.stack(logvars, dim=1),
            uncertainties=torch.stack(
                uncertainties, dim=1
            ),
            rewards=(
                torch.stack(rewards, dim=1)
                if rewards
                else None
            ),
            termination_probabilities=(
                torch.stack(
                    termination_probabilities,
                    dim=1,
                )
                if termination_probabilities
                else None
            ),
        )

    def gaussian_nll(
        self,
        target: Tensor,
        mean: Tensor,
        logvar: Tensor,
    ) -> Tensor:
        """
        Diagonal Gaussian negative log likelihood.

        Returns:
            scalar loss
        """

        if (
            target.shape != mean.shape
            or mean.shape != logvar.shape
        ):
            raise ValueError(
                "Gaussian NLL tensor shape mismatch"
            )

        loss = 0.5 * (
            logvar
            + (target - mean).pow(2)
            * torch.exp(-logvar)
            + 1.8378770664093453
        )

        return loss.mean()

    def compute_loss(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward_target: Optional[Tensor] = None,
        termination_target: Optional[
            Tensor
        ] = None,
    ) -> Dict[str, Tensor]:
        """
        One-step supervised training objective.

        Required:
            state: [B,D]
            action: [B,A]
            next_state: [B,D]

        Optional:
            reward_target: [B]
            termination_target: [B]
        """

        (
            mean,
            logvar,
            reward,
            termination_logit,
        ) = self._predict_heads(state, action)

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

        total = (
            self.config.prediction_loss_weight
            * prediction_loss
        )

        losses = {
            "prediction_loss": prediction_loss,
        }

        if reward_target is not None:
            if reward is None:
                raise ValueError(
                    "Reward prediction head is disabled"
                )

            if reward_target.shape != reward.shape:
                raise ValueError(
                    "reward_target must be [B]"
                )

            reward_loss = F.mse_loss(
                reward,
                reward_target,
            )

            losses["reward_loss"] = reward_loss

            total = (
                total
                + self.config.reward_loss_weight
                * reward_loss
            )

        if termination_target is not None:
            if termination_logit is None:
                raise ValueError(
                    "Termination head is disabled"
                )

            if (
                termination_target.shape
                != termination_logit.shape
            ):
                raise ValueError(
                    "termination_target must be [B]"
                )

            termination_loss = (
                F.binary_cross_entropy_with_logits(
                    termination_logit,
                    termination_target.float(),
                )
            )

            losses["termination_loss"] = (
                termination_loss
            )

            total = (
                total
                + self.config.termination_loss_weight
                * termination_loss
            )

        losses["loss"] = total

        return losses

    def compute_rollout_loss(
        self,
        initial_state: Tensor,
        actions: Tensor,
        target_states: Tensor,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Multi-step supervised trajectory loss.

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

        rollout_loss = (
            errors * discounts.unsqueeze(0)
        ).sum(dim=1).mean() / discounts.sum()

        total = (
            self.config.prediction_loss_weight
            * rollout_loss
        )

        return {
            "loss": total,
            "rollout_loss": rollout_loss,
        }

    def training_step(
        self,
        optimizer: torch.optim.Optimizer,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward_target: Optional[Tensor] = None,
        termination_target: Optional[
            Tensor
        ] = None,
    ) -> Dict[str, float]:
        """
        Perform one supervised optimizer step.
        """

        self.train()

        optimizer.zero_grad(set_to_none=True)

        losses = self.compute_loss(
            state=state,
            action=action,
            next_state=next_state,
            reward_target=reward_target,
            termination_target=termination_target,
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
    ) -> DynamicsRolloutOutput:
        """
        Inference-only future generation.
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

    config = DynamicsModelConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        num_layers=3,
        stochastic=True,
        predict_reward=True,
        predict_termination=True,
    )

    model = DynamicsModel(config)

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

    reward_target = torch.randn(batch_size)

    termination_target = torch.randint(
        0,
        2,
        (batch_size,),
    ).float()

    mean, logvar = model(state, action)

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    metrics = model.training_step(
        optimizer=optimizer,
        state=state,
        action=action,
        next_state=next_state,
        reward_target=reward_target,
        termination_target=termination_target,
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
        "Generated states:",
        output.states.shape,
    )

    print(
        "Uncertainties:",
        output.uncertainties.shape,
    )
