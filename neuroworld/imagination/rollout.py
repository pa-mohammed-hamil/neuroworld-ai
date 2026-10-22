
"""
NeuroWorld AI
File: neuroworld/imagination/rollout.py

World Model Rollout Engine

Features:
- Multi-step imagined world trajectories
- Action-conditioned latent dynamics
- Deterministic and stochastic transitions
- Parallel candidate rollouts
- Reward, risk and uncertainty scoring
- Goal-conditioned trajectory ranking
- Early termination and valid-step masks
- Differentiable forward pass
- Inference-only generation
- CPU and GPU support

Expected dynamics interface:
    mean, logvar = dynamics_model(state, action)

Both outputs:
    [N, latent_dim]

Install:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor, nn


TransitionMetric = Callable[
    [Tensor, Tensor, Tensor], Tensor
]

TerminationFunction = Callable[[Tensor], Tensor]


@dataclass
class RolloutConfig:
    latent_dim: int = 128
    action_dim: int = 32
    horizon: int = 10

    num_candidates: int = 8

    stochastic: bool = True
    temperature: float = 0.5

    discount_factor: float = 0.99

    uncertainty_penalty: float = 0.1
    risk_penalty: float = 0.5
    goal_weight: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    enable_early_termination: bool = False


@dataclass
class RolloutOutput:
    """
    Tensor shapes:

    states:
        [B, K, H+1, latent_dim]

    actions:
        [B, K, H, action_dim]

    rewards:
        [B, K, H]

    risks:
        [B, K, H]

    uncertainties:
        [B, K, H]

    valid_steps:
        [B, K, H]

    scores:
        [B, K]

    best_indices:
        [B]
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    risks: Tensor
    uncertainties: Tensor
    valid_steps: Tensor
    scores: Tensor
    best_indices: Tensor

    @property
    def trajectory_lengths(self) -> Tensor:
        return self.valid_steps.long().sum(dim=-1)

    @property
    def final_states(self) -> Tensor:
        batch_size, candidates = self.states.shape[:2]

        lengths = self.trajectory_lengths

        batch_indices = torch.arange(
            batch_size,
            device=self.states.device,
        )[:, None]

        candidate_indices = torch.arange(
            candidates,
            device=self.states.device,
        )[None, :]

        return self.states[
            batch_indices,
            candidate_indices,
            lengths,
        ]

    @property
    def best_trajectories(self) -> Tensor:
        batch_indices = torch.arange(
            self.states.shape[0],
            device=self.states.device,
        )

        return self.states[
            batch_indices,
            self.best_indices,
        ]

    @property
    def best_scores(self) -> Tensor:
        return self.scores.gather(
            1,
            self.best_indices.unsqueeze(1),
        ).squeeze(1)


class ExampleWorldDynamics(nn.Module):
    """
    Example trainable transition model.

    This network is not pretrained.
    Train it on observed or simulated transitions
    before relying on its predictions.
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

        self.encoder = nn.Sequential(
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

        if state.shape[0] != action.shape[0]:
            raise ValueError(
                "Batch dimensions must match"
            )

        if state.shape[-1] != self.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if action.shape[-1] != self.action_dim:
            raise ValueError(
                "Invalid action dimension"
            )

        features = torch.cat(
            [state, action],
            dim=-1,
        )

        hidden = self.encoder(features)

        mean = state + self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


class WorldRollout(nn.Module):
    """
    Generate and evaluate multiple imagined futures.

    Supported actions:

        [B, H, A]
            Shared actions across K stochastic rollouts.

        [B, K, H, A]
            Independent action sequence per candidate.

    Returns:
        RolloutOutput
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[RolloutConfig] = None,
        reward_fn: Optional[TransitionMetric] = None,
        risk_fn: Optional[TransitionMetric] = None,
        termination_fn: Optional[
            TerminationFunction
        ] = None,
    ):
        super().__init__()

        self.dynamics_model = dynamics_model
        self.config = config or RolloutConfig()

        self.reward_fn = reward_fn
        self.risk_fn = risk_fn
        self.termination_fn = termination_fn

        cfg = self.config

        if (
            cfg.latent_dim <= 0
            or cfg.action_dim <= 0
            or cfg.horizon <= 0
            or cfg.num_candidates <= 0
        ):
            raise ValueError(
                "Dimensions, horizon and candidates "
                "must be positive"
            )

        if not 0.0 <= cfg.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0,1]"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if cfg.min_logvar > cfg.max_logvar:
            raise ValueError(
                "min_logvar must not exceed max_logvar"
            )

        if (
            cfg.uncertainty_penalty < 0
            or cfg.risk_penalty < 0
            or cfg.goal_weight < 0
        ):
            raise ValueError(
                "Penalty and goal weights must "
                "be non-negative"
            )

    def _validate_inputs(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor],
    ) -> None:

        cfg = self.config

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B,D]"
            )

        if initial_state.shape[0] == 0:
            raise ValueError(
                "Batch size must be positive"
            )

        if initial_state.shape[-1] != cfg.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if actions.ndim not in (3, 4):
            raise ValueError(
                "actions must be [B,H,A] "
                "or [B,K,H,A]"
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
                "Action dimensions mismatch"
            )

        if (
            actions.ndim == 4
            and actions.shape[1] != cfg.num_candidates
        ):
            raise ValueError(
                "Candidate count mismatch"
            )

        if goal_state is not None:
            if goal_state.shape != initial_state.shape:
                raise ValueError(
                    "goal_state must be [B,D]"
                )

        for name, tensor in (
            ("initial_state", initial_state),
            ("actions", actions),
            ("goal_state", goal_state),
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
                    f"{name} must share device/dtype"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def _prepare_actions(
        self,
        actions: Tensor,
    ) -> Tensor:

        if actions.ndim == 3:
            return actions.unsqueeze(1).expand(
                -1,
                self.config.num_candidates,
                -1,
                -1,
            )

        return actions

    def predict_transition(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            next_state: [N,D]
            uncertainty: [N]

        Uncertainty is mean predicted variance.
        """

        mean, logvar = self.dynamics_model(
            state, action
        )

        if mean.shape != state.shape:
            raise ValueError(
                "Transition mean shape mismatch"
            )

        if logvar.shape != state.shape:
            raise ValueError(
                "Transition logvar shape mismatch"
            )

        logvar = logvar.clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )

        variance = logvar.exp()

        if self.config.stochastic:
            next_state = (
                mean
                + self.config.temperature
                * torch.sqrt(variance)
                * torch.randn_like(mean)
            )
        else:
            next_state = mean

        uncertainty = variance.mean(dim=-1)

        return next_state, uncertainty

    def evaluate_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:

        if self.reward_fn is not None:
            reward = self.reward_fn(
                state,
                action,
                next_state,
            )
        else:
            # Demonstration-only heuristic.
            reward = -0.01 * (
                next_state - state
            ).pow(2).mean(dim=-1)

        if reward.shape != state.shape[:-1]:
            raise ValueError(
                "Reward function must return [N]"
            )

        return reward

    def evaluate_risk(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:

        if self.risk_fn is not None:
            risk = self.risk_fn(
                state,
                action,
                next_state,
            )
        else:
            # Demonstration-only heuristic.
            risk = 0.001 * (
                next_state.pow(2).mean(dim=-1)
            )

        if risk.shape != state.shape[:-1]:
            raise ValueError(
                "Risk function must return [N]"
            )

        return risk

    def forward(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> RolloutOutput:
        """
        Execute parallel imagined rollouts.

        Returns fixed-length trajectories. Once a
        candidate terminates, its state is frozen
        and subsequent steps are marked invalid.
        """

        self._validate_inputs(
            initial_state,
            actions,
            goal_state,
        )

        cfg = self.config

        batch_size = initial_state.shape[0]
        candidates = cfg.num_candidates

        actions = self._prepare_actions(actions)

        current = initial_state.unsqueeze(1).expand(
            -1,
            candidates,
            -1,
        )

        alive = torch.ones(
            batch_size,
            candidates,
            device=initial_state.device,
            dtype=torch.bool,
        )

        state_steps = [current]
        reward_steps = []
        risk_steps = []
        uncertainty_steps = []
        valid_steps = []

        for step in range(cfg.horizon):

            action = actions[:, :, step, :]

            active = alive.clone()

            if not active.any().item():
                state_steps.append(current)

                zeros = current.new_zeros(
                    batch_size,
                    candidates,
                )

                reward_steps.append(zeros)
                risk_steps.append(zeros)
                uncertainty_steps.append(zeros)
                valid_steps.append(active)

                continue

            flat_state = current.reshape(
                -1,
                cfg.latent_dim,
            )

            flat_action = action.reshape(
                -1,
                cfg.action_dim,
            )

            next_state, uncertainty = (
                self.predict_transition(
                    flat_state,
                    flat_action,
                )
            )

            reward = self.evaluate_reward(
                flat_state,
                flat_action,
                next_state,
            )

            risk = self.evaluate_risk(
                flat_state,
                flat_action,
                next_state,
            )

            next_state = next_state.reshape(
                batch_size,
                candidates,
                cfg.latent_dim,
            )

            reward = reward.reshape(
                batch_size,
                candidates,
            )

            risk = risk.reshape(
                batch_size,
                candidates,
            )

            uncertainty = uncertainty.reshape(
                batch_size,
                candidates,
            )

            current = torch.where(
                active.unsqueeze(-1),
                next_state,
                current,
            )

            zeros = torch.zeros_like(reward)

            reward = torch.where(
                active, reward, zeros
            )

            risk = torch.where(
                active, risk, zeros
            )

            uncertainty = torch.where(
                active, uncertainty, zeros
            )

            state_steps.append(current)
            reward_steps.append(reward)
            risk_steps.append(risk)
            uncertainty_steps.append(uncertainty)
            valid_steps.append(active)

            if (
                cfg.enable_early_termination
                and self.termination_fn is not None
            ):
                terminated = self.termination_fn(
                    current.reshape(
                        -1,
                        cfg.latent_dim,
                    )
                )

                if terminated.shape != (
                    batch_size * candidates,
                ):
                    raise ValueError(
                        "Termination function must "
                        "return [B*K]"
                    )

                terminated = terminated.reshape(
                    batch_size,
                    candidates,
                ).bool()

                alive = alive & ~terminated

        states = torch.stack(
            state_steps, dim=2
        )

        rewards = torch.stack(
            reward_steps, dim=2
        )

        risks = torch.stack(
            risk_steps, dim=2
        )

        uncertainties = torch.stack(
            uncertainty_steps, dim=2
        )

        valid_steps_tensor = torch.stack(
            valid_steps, dim=2
        )

        discounts = (
            cfg.discount_factor
            ** torch.arange(
                cfg.horizon,
                device=initial_state.device,
                dtype=initial_state.dtype,
            )
        )

        step_scores = (
            rewards
            - cfg.risk_penalty * risks
            - cfg.uncertainty_penalty
            * uncertainties
        )

        scores = (
            step_scores
            * discounts.view(1, 1, -1)
        ).sum(dim=-1)

        if goal_state is not None:

            lengths = (
                valid_steps_tensor.long()
                .sum(dim=-1)
            )

            batch_indices = torch.arange(
                batch_size,
                device=initial_state.device,
            )[:, None]

            candidate_indices = torch.arange(
                candidates,
                device=initial_state.device,
            )[None, :]

            final_states = states[
                batch_indices,
                candidate_indices,
                lengths,
            ]

            goal_distance = torch.linalg.vector_norm(
                final_states
                - goal_state.unsqueeze(1),
                dim=-1,
            )

            scores = (
                scores
                - cfg.goal_weight * goal_distance
            )

        best_indices = scores.argmax(dim=1)

        return RolloutOutput(
            states=states,
            actions=actions,
            rewards=rewards,
            risks=risks,
            uncertainties=uncertainties,
            valid_steps=valid_steps_tensor,
            scores=scores,
            best_indices=best_indices,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> RolloutOutput:
        """
        Inference-only generation.
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
    def statistics(
        self,
        output: RolloutOutput,
    ) -> dict:
        """
        JSON-compatible rollout summary.
        """

        valid_counts = (
            output.valid_steps.sum(dim=-1)
            .clamp_min(1)
        )

        mean_uncertainty = (
            output.uncertainties.sum(dim=-1)
            / valid_counts
        )

        return {
            "batch_size": output.states.shape[0],
            "num_candidates": output.states.shape[1],
            "horizon": output.actions.shape[2],
            "best_indices": (
                output.best_indices.cpu().tolist()
            ),
            "best_scores": (
                output.best_scores.cpu().tolist()
            ),
            "trajectory_lengths": (
                output.trajectory_lengths
                .cpu().tolist()
            ),
            "mean_uncertainty": (
                mean_uncertainty.cpu().tolist()
            ),
            "mean_reward": (
                output.rewards.mean(dim=-1)
                .cpu().tolist()
            ),
            "mean_risk": (
                output.risks.mean(dim=-1)
                .cpu().tolist()
            ),
        }


if __name__ == "__main__":

    torch.manual_seed(42)

    config = RolloutConfig(
        latent_dim=128,
        action_dim=32,
        horizon=10,
        num_candidates=6,
        stochastic=True,
        temperature=0.3,
        enable_early_termination=True,
    )

    dynamics = ExampleWorldDynamics(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
    )

    def termination_fn(state: Tensor) -> Tensor:
        # Demonstration-only stopping rule.
        return (
            torch.linalg.vector_norm(
                state, dim=-1
            ) > 30.0
        )

    rollout = WorldRollout(
        dynamics_model=dynamics,
        config=config,
        termination_fn=termination_fn,
    )

    batch_size = 3

    initial_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    candidate_actions = torch.randn(
        batch_size,
        config.num_candidates,
        config.horizon,
        config.action_dim,
    ).clamp(-1.0, 1.0)

    goal_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    output = rollout.generate(
        initial_state=initial_state,
        actions=candidate_actions,
        goal_state=goal_state,
    )

    print("States:", output.states.shape)
    print("Actions:", output.actions.shape)
    print("Rewards:", output.rewards.shape)
    print("Risks:", output.risks.shape)
    print(
        "Uncertainties:",
        output.uncertainties.shape,
    )
    print("Scores:", output.scores.shape)

    print(
        "Best trajectories:",
        output.best_trajectories.shape,
    )

    print(
        "Statistics:",
        rollout.statistics(output),
    )
