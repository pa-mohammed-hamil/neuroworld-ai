
"""
NeuroWorld AI
File: neuroworld/imagination/latent_rollout.py

Latent Rollout Engine

Features:
- Multi-step latent state rollouts
- Action-conditioned world-model dynamics
- Deterministic and stochastic transitions
- Batched parallel trajectories
- Reward prediction
- Uncertainty estimation
- Discounted trajectory scoring
- Optional early termination
- Gradient-enabled model-based learning
- CPU / CUDA compatibility

Dependencies:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn


RewardFunction = Callable[
    [Tensor, Tensor, Tensor], Tensor
]

TerminationFunction = Callable[
    [Tensor], Tensor
]


@dataclass
class LatentRolloutConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    horizon: int = 10

    stochastic: bool = True
    temperature: float = 0.5

    discount_factor: float = 0.99
    uncertainty_penalty: float = 0.1

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    enable_early_termination: bool = False


@dataclass
class LatentRolloutOutput:
    """
    states:
        [B, H+1, latent_dim]

    actions:
        [B, H, action_dim]

    rewards:
        [B, H]

    uncertainties:
        [B, H]

    valid_steps:
        [B, H] boolean

    scores:
        [B]
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    uncertainties: Tensor
    valid_steps: Tensor
    scores: Tensor

    @property
    def final_states(self) -> Tensor:
        lengths = self.valid_steps.long().sum(dim=1)

        batch_indices = torch.arange(
            self.states.shape[0],
            device=self.states.device,
        )

        return self.states[
            batch_indices,
            lengths,
        ]

    @property
    def trajectory_lengths(self) -> Tensor:
        return self.valid_steps.long().sum(dim=1)

    @property
    def total_rewards(self) -> Tensor:
        return self.rewards.sum(dim=1)


class ExampleLatentDynamics(nn.Module):
    """
    Trainable example world dynamics model.

    Replace with a trained transition model.

    Input:
        state: [B, latent_dim]
        action: [B, action_dim]

    Output:
        mean: [B, latent_dim]
        logvar: [B, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        self.logvar_head = nn.Linear(
            hidden_dim,
            latent_dim,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        if state.ndim != 2 or action.ndim != 2:
            raise ValueError(
                "State and action must be 2D tensors"
            )

        if state.shape != (
            action.shape[0],
            self.latent_dim,
        ):
            raise ValueError(
                "Invalid state shape"
            )

        if action.shape[-1] != self.action_dim:
            raise ValueError(
                "Invalid action dimension"
            )

        features = torch.cat(
            [state, action],
            dim=-1,
        )

        hidden = self.backbone(features)

        mean = state + self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


class LatentRollout(nn.Module):
    """
    Generate future latent-state trajectories.

    Compatible with world models exposing:

        mean, logvar = dynamics_model(state, action)

    Both outputs must have shape:
        [B, latent_dim]
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[
            LatentRolloutConfig
        ] = None,
        reward_fn: Optional[
            RewardFunction
        ] = None,
        termination_fn: Optional[
            TerminationFunction
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or LatentRolloutConfig()
        )

        self.dynamics_model = dynamics_model
        self.reward_fn = reward_fn
        self.termination_fn = termination_fn

        cfg = self.config

        if (
            cfg.latent_dim <= 0
            or cfg.action_dim <= 0
            or cfg.horizon <= 0
        ):
            raise ValueError(
                "Dimensions and horizon must be positive"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if not 0 <= cfg.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in [0,1]"
            )

        if cfg.uncertainty_penalty < 0:
            raise ValueError(
                "uncertainty_penalty must be non-negative"
            )

        if cfg.min_logvar > cfg.max_logvar:
            raise ValueError(
                "min_logvar cannot exceed max_logvar"
            )

    def _validate_inputs(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> None:

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

        if actions.shape[1:] != (
            cfg.horizon,
            cfg.action_dim,
        ):
            raise ValueError(
                "Invalid action dimensions"
            )

        if actions.shape[0] != initial_state.shape[0]:
            raise ValueError(
                "Batch dimensions must match"
            )

        if initial_state.shape[0] == 0:
            raise ValueError(
                "Batch size cannot be zero"
            )

        if (
            initial_state.device != actions.device
            or initial_state.dtype != actions.dtype
        ):
            raise ValueError(
                "State and actions must share device/dtype"
            )

        for name, value in (
            ("initial_state", initial_state),
            ("actions", actions),
        ):
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(value).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def predict_next(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict next state and model variance.

        Uncertainty:
            Mean predicted variance per state.
        """

        mean, logvar = self.dynamics_model(
            state,
            action,
        )

        if mean.shape != state.shape:
            raise ValueError(
                "Dynamics mean shape mismatch"
            )

        if logvar.shape != state.shape:
            raise ValueError(
                "Dynamics logvar shape mismatch"
            )

        logvar = logvar.clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )

        variance = logvar.exp()

        if self.config.stochastic:
            noise = torch.randn_like(mean)

            next_state = (
                mean
                + self.config.temperature
                * torch.sqrt(variance)
                * noise
            )
        else:
            next_state = mean

        uncertainty = variance.mean(dim=-1)

        return next_state, uncertainty

    def calculate_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Demonstration reward:
        penalize large latent transitions.

        Replace with a task-specific reward function.
        """

        if self.reward_fn is not None:
            reward = self.reward_fn(
                state,
                action,
                next_state,
            )
        else:
            reward = -0.01 * (
                next_state - state
            ).pow(2).mean(dim=-1)

        if reward.shape != state.shape[:-1]:
            raise ValueError(
                "Reward function must return [B]"
            )

        return reward

    def forward(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> LatentRolloutOutput:
        """
        Roll out a batch of action sequences.

        initial_state:
            [B, latent_dim]

        actions:
            [B, horizon, action_dim]
        """

        self._validate_inputs(
            initial_state,
            actions,
        )

        cfg = self.config

        batch_size = initial_state.shape[0]
        device = initial_state.device

        state = initial_state

        alive = torch.ones(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

        states = [state]
        rewards = []
        uncertainties = []
        valid_steps = []

        for step in range(cfg.horizon):

            action = actions[:, step, :]

            active = alive

            if not active.any().item():
                # Preserve fixed-length output shapes.
                states.append(state)

                rewards.append(
                    state.new_zeros(batch_size)
                )

                uncertainties.append(
                    state.new_zeros(batch_size)
                )

                valid_steps.append(active)

                continue

            next_state, uncertainty = (
                self.predict_next(
                    state,
                    action,
                )
            )

            reward = self.calculate_reward(
                state,
                action,
                next_state,
            )

            # Inactive trajectories remain unchanged.
            next_state = torch.where(
                active.unsqueeze(-1),
                next_state,
                state,
            )

            reward = torch.where(
                active,
                reward,
                torch.zeros_like(reward),
            )

            uncertainty = torch.where(
                active,
                uncertainty,
                torch.zeros_like(uncertainty),
            )

            states.append(next_state)
            rewards.append(reward)
            uncertainties.append(uncertainty)
            valid_steps.append(active)

            state = next_state

            if (
                cfg.enable_early_termination
                and self.termination_fn is not None
            ):
                terminated = self.termination_fn(
                    state
                )

                if terminated.shape != (batch_size,):
                    raise ValueError(
                        "Termination function must "
                        "return [B]"
                    )

                alive = (
                    alive & ~terminated.bool()
                )

        state_tensor = torch.stack(
            states,
            dim=1,
        )

        reward_tensor = torch.stack(
            rewards,
            dim=1,
        )

        uncertainty_tensor = torch.stack(
            uncertainties,
            dim=1,
        )

        valid_tensor = torch.stack(
            valid_steps,
            dim=1,
        )

        discounts = (
            cfg.discount_factor
            ** torch.arange(
                cfg.horizon,
                device=device,
                dtype=initial_state.dtype,
            )
        )

        step_scores = (
            reward_tensor
            - cfg.uncertainty_penalty
            * uncertainty_tensor
        )

        scores = (
            step_scores
            * discounts.unsqueeze(0)
        ).sum(dim=1)

        return LatentRolloutOutput(
            states=state_tensor,
            actions=actions,
            rewards=reward_tensor,
            uncertainties=uncertainty_tensor,
            valid_steps=valid_tensor,
            scores=scores,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> LatentRolloutOutput:
        """
        Inference-only latent rollout.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(
                initial_state,
                actions,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def rollout_statistics(
        self,
        output: LatentRolloutOutput,
    ) -> dict:
        """
        Return JSON-compatible rollout statistics.
        """

        valid_counts = (
            output.valid_steps.sum(dim=1)
            .clamp_min(1)
        )

        mean_uncertainty = (
            output.uncertainties.sum(dim=1)
            / valid_counts
        )

        return {
            "batch_size": output.states.shape[0],
            "horizon": output.actions.shape[1],
            "trajectory_lengths": (
                output.trajectory_lengths
                .cpu().tolist()
            ),
            "total_rewards": (
                output.total_rewards
                .cpu().tolist()
            ),
            "scores": (
                output.scores.cpu().tolist()
            ),
            "mean_uncertainty": (
                mean_uncertainty.cpu().tolist()
            ),
        }


if __name__ == "__main__":

    torch.manual_seed(42)

    config = LatentRolloutConfig(
        latent_dim=128,
        action_dim=32,
        horizon=10,
        stochastic=True,
        temperature=0.3,
        enable_early_termination=True,
    )

    dynamics = ExampleLatentDynamics(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    )

    def termination_fn(state: Tensor) -> Tensor:
        # Example-only stopping criterion.
        return (
            torch.linalg.vector_norm(
                state,
                dim=-1,
            ) > 30.0
        )

    rollout = LatentRollout(
        dynamics_model=dynamics,
        config=config,
        termination_fn=termination_fn,
    )

    initial_state = torch.randn(
        4,
        config.latent_dim,
    )

    actions = torch.randn(
        4,
        config.horizon,
        config.action_dim,
    ).clamp(-1.0, 1.0)

    output = rollout.generate(
        initial_state,
        actions,
    )

    print("States:", output.states.shape)
    print("Actions:", output.actions.shape)
    print("Rewards:", output.rewards.shape)
    print(
        "Uncertainties:",
        output.uncertainties.shape,
    )
    print("Scores:", output.scores.shape)

    print(
        "Final states:",
        output.final_states.shape,
    )

    print(
        "Statistics:",
        rollout.rollout_statistics(output),
    )
