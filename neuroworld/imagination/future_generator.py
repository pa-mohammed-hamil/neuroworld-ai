
"""
NeuroWorld AI
File: neuroworld/imagination/future_generator.py

Neural Future Generation Engine

Features:
- Multi-step future state generation
- Action-conditioned latent predictions
- Deterministic and stochastic rollouts
- Multiple possible future trajectories
- Autoregressive state transitions
- Uncertainty propagation
- Monte Carlo future sampling
- Future trajectory ranking
- Goal-conditioned scoring
- Discounted reward estimation
- CPU / GPU support

Dependencies:
    pip install torch
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class FutureGeneratorConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    horizon: int = 10
    num_futures: int = 8

    stochastic: bool = True
    temperature: float = 0.5

    discount_factor: float = 0.99
    uncertainty_penalty: float = 0.1
    goal_weight: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0


@dataclass
class FutureGenerationOutput:
    """
    Output dimensions:

    trajectories:
        [B, K, H+1, latent_dim]

    actions:
        [B, K, H, action_dim]

    uncertainties:
        [B, K, H]

    rewards:
        [B, K, H]

    scores:
        [B, K]

    best_indices:
        [B]
    """

    trajectories: Tensor
    actions: Tensor
    uncertainties: Tensor
    rewards: Tensor
    scores: Tensor
    best_indices: Tensor

    @property
    def final_states(self) -> Tensor:
        return self.trajectories[:, :, -1, :]

    @property
    def best_trajectories(self) -> Tensor:
        batch_indices = torch.arange(
            self.trajectories.shape[0],
            device=self.trajectories.device,
        )

        return self.trajectories[
            batch_indices,
            self.best_indices,
        ]


class ExampleFutureDynamics(nn.Module):
    """
    Trainable example dynamics network.

    Replace with a trained world-model dynamics module.

    Interface:
        mean, logvar = model(state, action)
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

        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "State and action batch shapes must match"
            )

        features = torch.cat(
            [state, action],
            dim=-1,
        )

        hidden = self.backbone(features)

        delta = self.mean_head(hidden)

        mean = state + delta

        logvar = self.logvar_head(hidden)

        return mean, logvar


class FutureGenerator(nn.Module):
    """
    Generate multiple possible future trajectories.

    Each future is generated using a learned transition
    model and an action sequence.

    Stochastic sampling introduces variation between
    imagined futures.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[FutureGeneratorConfig] = None,
        reward_fn: Optional[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or FutureGeneratorConfig()
        )

        self.dynamics_model = dynamics_model
        self.reward_fn = reward_fn

        cfg = self.config

        if (
            cfg.latent_dim <= 0
            or cfg.action_dim <= 0
            or cfg.horizon <= 0
            or cfg.num_futures <= 0
        ):
            raise ValueError(
                "Dimensions, horizon and num_futures "
                "must be positive"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if not 0.0 <= cfg.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0, 1]"
            )

        if cfg.uncertainty_penalty < 0:
            raise ValueError(
                "uncertainty_penalty must be non-negative"
            )

    def _validate_inputs(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> None:

        cfg = self.config

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B, latent_dim]"
            )

        if initial_state.shape[-1] != cfg.latent_dim:
            raise ValueError(
                "Invalid initial state dimension"
            )

        if actions.ndim not in (3, 4):
            raise ValueError(
                "Actions must be [B,H,A] or [B,K,H,A]"
            )

        if actions.shape[0] != initial_state.shape[0]:
            raise ValueError(
                "Batch size mismatch"
            )

        if actions.shape[-2:] != (
            cfg.horizon,
            cfg.action_dim,
        ):
            raise ValueError(
                "Action horizon or dimension mismatch"
            )

        if actions.ndim == 4 and actions.shape[1] != (
            cfg.num_futures
        ):
            raise ValueError(
                "Actions K must match num_futures"
            )

        if (
            actions.device != initial_state.device
            or actions.dtype != initial_state.dtype
        ):
            raise ValueError(
                "Actions and states must share device and dtype"
            )

        if not torch.isfinite(initial_state).all().item():
            raise ValueError(
                "Initial state contains non-finite values"
            )

        if not torch.isfinite(actions).all().item():
            raise ValueError(
                "Actions contain non-finite values"
            )

    def _prepare_actions(
        self,
        actions: Tensor,
    ) -> Tensor:
        """
        Expand shared action sequences across futures.

        [B,H,A] -> [B,K,H,A]
        """

        if actions.ndim == 3:
            return actions.unsqueeze(1).expand(
                -1,
                self.config.num_futures,
                -1,
                -1,
            )

        return actions

    def predict_next(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict one transition.

        states: [N,D]
        actions: [N,A]

        Returns:
            next_states: [N,D]
            uncertainty: [N]
        """

        mean, logvar = self.dynamics_model(
            states,
            actions,
        )

        if mean.shape != states.shape:
            raise ValueError(
                "Dynamics mean shape mismatch"
            )

        if logvar.shape != states.shape:
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

            next_states = (
                mean
                + self.config.temperature
                * torch.sqrt(variance)
                * noise
            )
        else:
            next_states = mean

        uncertainty = variance.mean(dim=-1)

        return next_states, uncertainty

    def calculate_reward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> Tensor:
        """
        Default reward is a demonstration heuristic.

        It penalizes large changes in latent state.
        Replace it with a task-specific reward function.
        """

        if self.reward_fn is not None:
            rewards = self.reward_fn(
                states,
                actions,
                next_states,
            )
        else:
            rewards = -0.01 * (
                next_states - states
            ).pow(2).mean(dim=-1)

        if rewards.shape != states.shape[:-1]:
            raise ValueError(
                "Reward function must return [N]"
            )

        return rewards

    def forward(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> FutureGenerationOutput:
        """
        Generate K possible futures for each initial state.

        initial_state:
            [B,D]

        actions:
            [B,H,A] or [B,K,H,A]

        goal_state:
            [B,D] (optional)
        """

        self._validate_inputs(
            initial_state,
            actions,
        )

        cfg = self.config

        if goal_state is not None:
            if goal_state.shape != initial_state.shape:
                raise ValueError(
                    "Goal state shape mismatch"
                )

            if (
                goal_state.device != initial_state.device
                or goal_state.dtype != initial_state.dtype
            ):
                raise ValueError(
                    "Goal state must share device and dtype"
                )

        batch_size = initial_state.shape[0]
        num_futures = cfg.num_futures

        actions = self._prepare_actions(actions)

        current = initial_state.unsqueeze(1).expand(
            -1,
            num_futures,
            -1,
        )

        trajectory_steps = [current]
        uncertainty_steps = []
        reward_steps = []

        for step in range(cfg.horizon):

            action = actions[:, :, step, :]

            flat_states = current.reshape(
                -1,
                cfg.latent_dim,
            )

            flat_actions = action.reshape(
                -1,
                cfg.action_dim,
            )

            next_states, uncertainty = (
                self.predict_next(
                    flat_states,
                    flat_actions,
                )
            )

            rewards = self.calculate_reward(
                flat_states,
                flat_actions,
                next_states,
            )

            current = next_states.reshape(
                batch_size,
                num_futures,
                cfg.latent_dim,
            )

            uncertainty = uncertainty.reshape(
                batch_size,
                num_futures,
            )

            rewards = rewards.reshape(
                batch_size,
                num_futures,
            )

            trajectory_steps.append(current)
            uncertainty_steps.append(uncertainty)
            reward_steps.append(rewards)

        trajectories = torch.stack(
            trajectory_steps,
            dim=2,
        )

        uncertainties = torch.stack(
            uncertainty_steps,
            dim=2,
        )

        rewards = torch.stack(
            reward_steps,
            dim=2,
        )

        discount = (
            cfg.discount_factor
            ** torch.arange(
                cfg.horizon,
                device=initial_state.device,
                dtype=initial_state.dtype,
            )
        )

        step_scores = (
            rewards
            - cfg.uncertainty_penalty
            * uncertainties
        )

        scores = (
            step_scores
            * discount.view(1, 1, -1)
        ).sum(dim=-1)

        if goal_state is not None:
            expanded_goal = goal_state.unsqueeze(1)

            goal_distance = torch.linalg.vector_norm(
                trajectories[:, :, -1, :]
                - expanded_goal,
                dim=-1,
            )

            scores = (
                scores
                - cfg.goal_weight * goal_distance
            )

        best_indices = scores.argmax(dim=1)

        return FutureGenerationOutput(
            trajectories=trajectories,
            actions=actions,
            uncertainties=uncertainties,
            rewards=rewards,
            scores=scores,
            best_indices=best_indices,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> FutureGenerationOutput:
        """
        Generate futures without gradient tracking.
        Suitable for inference and planning.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(
                initial_state,
                actions,
                goal_state,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def evaluate_futures(
        self,
        output: FutureGenerationOutput,
    ) -> dict:
        """
        Summarize imagined future trajectories.
        """

        final_states = output.final_states

        mean_final_state = final_states.mean(
            dim=1
        )

        diversity = torch.linalg.vector_norm(
            final_states
            - mean_final_state.unsqueeze(1),
            dim=-1,
        ).mean(dim=1)

        return {
            "mean_score": (
                output.scores.mean(dim=1).cpu().tolist()
            ),
            "best_score": (
                output.scores.max(dim=1).values
                .cpu().tolist()
            ),
            "mean_uncertainty": (
                output.uncertainties.mean(
                    dim=(1, 2)
                ).cpu().tolist()
            ),
            "future_diversity": (
                diversity.cpu().tolist()
            ),
            "best_indices": (
                output.best_indices.cpu().tolist()
            ),
        }


if __name__ == "__main__":

    torch.manual_seed(42)

    config = FutureGeneratorConfig(
        latent_dim=128,
        action_dim=32,
        horizon=10,
        num_futures=6,
        stochastic=True,
        temperature=0.4,
    )

    dynamics = ExampleFutureDynamics(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
    )

    generator = FutureGenerator(
        dynamics_model=dynamics,
        config=config,
    )

    batch_size = 3

    initial_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    actions = torch.randn(
        batch_size,
        config.horizon,
        config.action_dim,
    ).clamp(-1.0, 1.0)

    goal_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    output = generator.generate(
        initial_state=initial_state,
        actions=actions,
        goal_state=goal_state,
    )

    print(
        "Trajectories:",
        output.trajectories.shape,
    )

    print(
        "Actions:",
        output.actions.shape,
    )

    print(
        "Uncertainties:",
        output.uncertainties.shape,
    )

    print(
        "Scores:",
        output.scores.shape,
    )

    print(
        "Best trajectories:",
        output.best_trajectories.shape,
    )

    metrics = generator.evaluate_futures(output)

    print("Evaluation:", metrics)
