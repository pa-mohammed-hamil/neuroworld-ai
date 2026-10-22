
"""
NeuroWorld AI
File: neuroworld/imagination/trajectory_generator.py

Neural Trajectory Generation Engine

Features:
- Action-conditioned trajectory generation
- Goal-conditioned latent planning
- Random-shooting trajectory search
- Cross-Entropy Method (CEM) optimization
- Deterministic and stochastic world rollouts
- Reward, risk and uncertainty scoring
- Action bounds and smoothness penalties
- Batched parallel trajectory generation
- Best trajectory selection
- Gradient-enabled forward pass
- Inference-only planning
- CPU / GPU support

Dependencies:
    pip install torch

Integration:
    neuroworld/imagination/rollout.py
    neuroworld/imagination/imagination_engine.py
    neuroworld/planning/planner.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from neuroworld.imagination.rollout import (
    RolloutConfig,
    RolloutOutput,
    WorldRollout,
)


TransitionMetric = Callable[
    [Tensor, Tensor, Tensor], Tensor
]

TerminationFunction = Callable[[Tensor], Tensor]


@dataclass
class TrajectoryGeneratorConfig:
    latent_dim: int = 128
    action_dim: int = 32

    horizon: int = 10
    num_candidates: int = 64

    method: str = "cem"

    cem_iterations: int = 5
    elite_fraction: float = 0.2
    cem_smoothing: float = 0.2

    initial_action_std: float = 0.5
    min_action_std: float = 0.05

    action_min: float = -1.0
    action_max: float = 1.0

    stochastic: bool = False
    temperature: float = 0.5

    discount_factor: float = 0.99

    uncertainty_penalty: float = 0.1
    risk_penalty: float = 0.5
    goal_weight: float = 1.0

    action_penalty: float = 0.01
    smoothness_penalty: float = 0.02

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    enable_early_termination: bool = False


@dataclass
class TrajectoryGenerationOutput:
    """
    Output shapes:

    trajectories:
        [B, K, H+1, D]

    actions:
        [B, K, H, A]

    rewards:
        [B, K, H]

    risks:
        [B, K, H]

    uncertainties:
        [B, K, H]

    scores:
        [B, K]

    best_indices:
        [B]

    best_actions:
        [B, H, A]

    best_trajectories:
        [B, H+1, D]
    """

    trajectories: Tensor
    actions: Tensor
    rewards: Tensor
    risks: Tensor
    uncertainties: Tensor
    valid_steps: Tensor

    scores: Tensor
    best_indices: Tensor

    best_actions: Tensor
    best_trajectories: Tensor
    best_scores: Tensor

    action_mean: Optional[Tensor] = None
    action_std: Optional[Tensor] = None

    @property
    def final_states(self) -> Tensor:
        lengths = self.valid_steps.long().sum(
            dim=-1
        )

        batch_size, candidates = (
            self.trajectories.shape[:2]
        )

        batch_idx = torch.arange(
            batch_size,
            device=self.trajectories.device,
        )[:, None]

        candidate_idx = torch.arange(
            candidates,
            device=self.trajectories.device,
        )[None, :]

        return self.trajectories[
            batch_idx,
            candidate_idx,
            lengths,
        ]


class TrajectoryGenerator(nn.Module):
    """
    Generate candidate action sequences and evaluate
    them through a learned world-model rollout.

    Supported methods:

        random:
            Random-shooting trajectory optimization.

        cem:
            Cross-Entropy Method trajectory optimization.

    Required dynamics model:

        mean, logvar = dynamics_model(state, action)

    Both outputs:
        [N, latent_dim]

    Important:
        This module optimizes predictions from a model.
        It does not guarantee safe or feasible actions.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[
            TrajectoryGeneratorConfig
        ] = None,
        reward_fn: Optional[
            TransitionMetric
        ] = None,
        risk_fn: Optional[
            TransitionMetric
        ] = None,
        termination_fn: Optional[
            TerminationFunction
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or TrajectoryGeneratorConfig()
        )

        cfg = self.config

        if (
            cfg.latent_dim <= 0
            or cfg.action_dim <= 0
            or cfg.horizon <= 0
            or cfg.num_candidates <= 0
        ):
            raise ValueError(
                "Dimensions, horizon and candidate "
                "count must be positive"
            )

        if cfg.method not in ("random", "cem"):
            raise ValueError(
                "method must be 'random' or 'cem'"
            )

        if cfg.cem_iterations <= 0:
            raise ValueError(
                "cem_iterations must be positive"
            )

        if not 0 < cfg.elite_fraction <= 1:
            raise ValueError(
                "elite_fraction must be in (0,1]"
            )

        if not 0 <= cfg.cem_smoothing < 1:
            raise ValueError(
                "cem_smoothing must be in [0,1)"
            )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be less than action_max"
            )

        if (
            cfg.initial_action_std <= 0
            or cfg.min_action_std <= 0
        ):
            raise ValueError(
                "Action standard deviations "
                "must be positive"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        if not 0 <= cfg.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in [0,1]"
            )

        if cfg.min_logvar > cfg.max_logvar:
            raise ValueError(
                "Invalid log-variance limits"
            )

        for name in (
            "uncertainty_penalty",
            "risk_penalty",
            "goal_weight",
            "action_penalty",
            "smoothness_penalty",
        ):
            if getattr(cfg, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative"
                )

        rollout_config = RolloutConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            num_candidates=cfg.num_candidates,
            stochastic=cfg.stochastic,
            temperature=cfg.temperature,
            discount_factor=cfg.discount_factor,
            uncertainty_penalty=cfg.uncertainty_penalty,
            risk_penalty=cfg.risk_penalty,
            goal_weight=cfg.goal_weight,
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
            enable_early_termination=(
                cfg.enable_early_termination
            ),
        )

        self.rollout_engine = WorldRollout(
            dynamics_model=dynamics_model,
            config=rollout_config,
            reward_fn=reward_fn,
            risk_fn=risk_fn,
            termination_fn=termination_fn,
        )

    def _validate_state(
        self,
        initial_state: Tensor,
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

        if goal_state is not None:
            if goal_state.shape != initial_state.shape:
                raise ValueError(
                    "goal_state must be [B,D]"
                )

        for name, tensor in (
            ("initial_state", initial_state),
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

    def _validate_action_sequence(
        self,
        actions: Tensor,
        batch_size: int,
    ) -> None:

        expected = (
            batch_size,
            self.config.horizon,
            self.config.action_dim,
        )

        if actions.shape != expected:
            raise ValueError(
                f"Action sequence must be {expected}"
            )

        if not torch.isfinite(actions).all().item():
            raise ValueError(
                "Actions contain NaN or Inf"
            )

    def _clip_actions(
        self,
        actions: Tensor,
    ) -> Tensor:

        return actions.clamp(
            self.config.action_min,
            self.config.action_max,
        )

    def _sample_actions(
        self,
        mean: Tensor,
        std: Tensor,
    ) -> Tensor:
        """
        Sample K action sequences.

        mean/std:
            [B,H,A]

        Returns:
            [B,K,H,A]
        """

        cfg = self.config

        noise = torch.randn(
            mean.shape[0],
            cfg.num_candidates,
            cfg.horizon,
            cfg.action_dim,
            device=mean.device,
            dtype=mean.dtype,
        )

        actions = (
            mean.unsqueeze(1)
            + std.unsqueeze(1) * noise
        )

        return self._clip_actions(actions)

    def _score_actions(
        self,
        rollout_output: RolloutOutput,
    ) -> Tensor:
        """
        Add action magnitude and smoothness penalties
        to world-model trajectory scores.
        """

        cfg = self.config

        actions = rollout_output.actions
        valid = rollout_output.valid_steps.to(
            actions.dtype
        )

        action_energy = (
            actions.pow(2).mean(dim=-1)
            * valid
        ).sum(dim=-1)

        action_count = valid.sum(
            dim=-1
        ).clamp_min(1.0)

        action_energy = (
            action_energy / action_count
        )

        if cfg.horizon > 1:
            action_difference = (
                actions[:, :, 1:, :]
                - actions[:, :, :-1, :]
            )

            pair_valid = (
                valid[:, :, 1:]
                * valid[:, :, :-1]
            )

            smoothness = (
                action_difference.pow(2)
                .mean(dim=-1)
                * pair_valid
            ).sum(dim=-1)

            smoothness = smoothness / (
                pair_valid.sum(dim=-1)
                .clamp_min(1.0)
            )
        else:
            smoothness = torch.zeros_like(
                action_energy
            )

        return (
            rollout_output.scores
            - cfg.action_penalty * action_energy
            - cfg.smoothness_penalty * smoothness
        )

    def evaluate_candidates(
        self,
        initial_state: Tensor,
        candidate_actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> tuple[RolloutOutput, Tensor]:
        """
        Evaluate candidate action sequences.

        candidate_actions:
            [B,K,H,A]
        """

        cfg = self.config

        expected = (
            initial_state.shape[0],
            cfg.num_candidates,
            cfg.horizon,
            cfg.action_dim,
        )

        if candidate_actions.shape != expected:
            raise ValueError(
                f"candidate_actions must be {expected}"
            )

        if (
            candidate_actions.device
            != initial_state.device
            or candidate_actions.dtype
            != initial_state.dtype
        ):
            raise ValueError(
                "Actions must share device/dtype "
                "with initial state"
            )

        if not torch.isfinite(
            candidate_actions
        ).all().item():
            raise ValueError(
                "Candidate actions contain NaN or Inf"
            )

        rollout_output = self.rollout_engine(
            initial_state=initial_state,
            actions=candidate_actions,
            goal_state=goal_state,
        )

        scores = self._score_actions(
            rollout_output
        )

        return rollout_output, scores

    def _build_output(
        self,
        rollout_output: RolloutOutput,
        scores: Tensor,
        action_mean: Optional[Tensor] = None,
        action_std: Optional[Tensor] = None,
    ) -> TrajectoryGenerationOutput:
        """
        Select best candidate per batch item.
        """

        batch_size = scores.shape[0]

        best_indices = scores.argmax(dim=1)

        batch_indices = torch.arange(
            batch_size,
            device=scores.device,
        )

        best_actions = rollout_output.actions[
            batch_indices,
            best_indices,
        ]

        best_trajectories = rollout_output.states[
            batch_indices,
            best_indices,
        ]

        best_scores = scores[
            batch_indices,
            best_indices,
        ]

        return TrajectoryGenerationOutput(
            trajectories=rollout_output.states,
            actions=rollout_output.actions,
            rewards=rollout_output.rewards,
            risks=rollout_output.risks,
            uncertainties=(
                rollout_output.uncertainties
            ),
            valid_steps=rollout_output.valid_steps,
            scores=scores,
            best_indices=best_indices,
            best_actions=best_actions,
            best_trajectories=best_trajectories,
            best_scores=best_scores,
            action_mean=action_mean,
            action_std=action_std,
        )

    def random_shooting(
        self,
        initial_state: Tensor,
        goal_state: Optional[Tensor] = None,
        initial_actions: Optional[Tensor] = None,
    ) -> TrajectoryGenerationOutput:
        """
        Generate random candidate action sequences
        and select the highest-scoring trajectory.
        """

        self._validate_state(
            initial_state,
            goal_state,
        )

        cfg = self.config

        batch_size = initial_state.shape[0]

        if initial_actions is None:
            mean = initial_state.new_zeros(
                batch_size,
                cfg.horizon,
                cfg.action_dim,
            )
        else:
            self._validate_action_sequence(
                initial_actions,
                batch_size,
            )

            mean = self._clip_actions(
                initial_actions
            )

        std = torch.full_like(
            mean,
            cfg.initial_action_std,
        )

        candidate_actions = self._sample_actions(
            mean,
            std,
        )

        rollout_output, scores = (
            self.evaluate_candidates(
                initial_state,
                candidate_actions,
                goal_state,
            )
        )

        return self._build_output(
            rollout_output,
            scores,
            action_mean=mean,
            action_std=std,
        )

    def cem(
        self,
        initial_state: Tensor,
        goal_state: Optional[Tensor] = None,
        initial_actions: Optional[Tensor] = None,
    ) -> TrajectoryGenerationOutput:
        """
        Cross-Entropy Method (CEM).

        Algorithm:
        1. Initialize action distribution.
        2. Sample K candidate trajectories.
        3. Evaluate candidate scores.
        4. Select highest-scoring elites.
        5. Update action distribution.
        6. Repeat for configured iterations.

        The final returned candidates are evaluated
        using the updated action distribution.
        """

        self._validate_state(
            initial_state,
            goal_state,
        )

        cfg = self.config

        batch_size = initial_state.shape[0]

        if initial_actions is None:
            mean = initial_state.new_zeros(
                batch_size,
                cfg.horizon,
                cfg.action_dim,
            )
        else:
            self._validate_action_sequence(
                initial_actions,
                batch_size,
            )

            mean = self._clip_actions(
                initial_actions
            )

        std = torch.full_like(
            mean,
            cfg.initial_action_std,
        )

        elite_count = max(
            1,
            int(
                cfg.num_candidates
                * cfg.elite_fraction
            ),
        )

        for _ in range(cfg.cem_iterations):

            candidate_actions = (
                self._sample_actions(
                    mean,
                    std,
                )
            )

            _, scores = self.evaluate_candidates(
                initial_state,
                candidate_actions,
                goal_state,
            )

            elite_indices = torch.topk(
                scores,
                k=elite_count,
                dim=1,
            ).indices

            elite_actions = torch.gather(
                candidate_actions,
                dim=1,
                index=elite_indices[
                    :, :, None, None
                ].expand(
                    -1,
                    -1,
                    cfg.horizon,
                    cfg.action_dim,
                ),
            )

            elite_mean = elite_actions.mean(
                dim=1
            )

            # unbiased=False avoids NaN when
            # elite_count == 1.
            elite_std = elite_actions.std(
                dim=1,
                unbiased=False,
            ).clamp_min(
                cfg.min_action_std
            )

            smoothing = cfg.cem_smoothing

            mean = (
                smoothing * mean
                + (1.0 - smoothing)
                * elite_mean
            )

            std = (
                smoothing * std
                + (1.0 - smoothing)
                * elite_std
            ).clamp_min(
                cfg.min_action_std
            )

            mean = self._clip_actions(mean)

        # Evaluate fresh candidates from the final
        # optimized action distribution.
        final_actions = self._sample_actions(
            mean,
            std,
        )

        final_rollout, final_scores = (
            self.evaluate_candidates(
                initial_state,
                final_actions,
                goal_state,
            )
        )

        return self._build_output(
            final_rollout,
            final_scores,
            action_mean=mean,
            action_std=std,
        )

    def forward(
        self,
        initial_state: Tensor,
        goal_state: Optional[Tensor] = None,
        initial_actions: Optional[Tensor] = None,
    ) -> TrajectoryGenerationOutput:
        """
        Execute the configured planning method.
        """

        if self.config.method == "random":
            return self.random_shooting(
                initial_state,
                goal_state,
                initial_actions,
            )

        return self.cem(
            initial_state,
            goal_state,
            initial_actions,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        goal_state: Optional[Tensor] = None,
        initial_actions: Optional[Tensor] = None,
    ) -> TrajectoryGenerationOutput:
        """
        Inference-only trajectory generation.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(
                initial_state=initial_state,
                goal_state=goal_state,
                initial_actions=initial_actions,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def summarize(
        self,
        output: TrajectoryGenerationOutput,
    ) -> dict:
        """
        Return JSON-compatible planning metrics.
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
            "method": self.config.method,
            "batch_size": output.actions.shape[0],
            "num_candidates": (
                output.actions.shape[1]
            ),
            "horizon": self.config.horizon,
            "best_indices": (
                output.best_indices.detach()
                .cpu().tolist()
            ),
            "best_scores": (
                output.best_scores.detach()
                .cpu().tolist()
            ),
            "mean_uncertainty": (
                mean_uncertainty.detach()
                .cpu().tolist()
            ),
            "best_action_sequences": (
                output.best_actions.detach()
                .cpu().tolist()
            ),
        }


if __name__ == "__main__":

    from neuroworld.imagination.rollout import (
        ExampleWorldDynamics,
    )

    torch.manual_seed(42)

    config = TrajectoryGeneratorConfig(
        latent_dim=128,
        action_dim=32,
        horizon=10,
        num_candidates=64,
        method="cem",
        cem_iterations=5,
        elite_fraction=0.2,
        stochastic=False,
    )

    dynamics = ExampleWorldDynamics(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
    )

    generator = TrajectoryGenerator(
        dynamics_model=dynamics,
        config=config,
    )

    batch_size = 2

    initial_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    goal_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    output = generator.generate(
        initial_state=initial_state,
        goal_state=goal_state,
    )

    print(
        "Trajectories:",
        output.trajectories.shape,
    )

    print(
        "Candidate actions:",
        output.actions.shape,
    )

    print(
        "Scores:",
        output.scores.shape,
    )

    print(
        "Best actions:",
        output.best_actions.shape,
    )

    print(
        "Best trajectories:",
        output.best_trajectories.shape,
    )

    print(
        "Best scores:",
        output.best_scores,
    )

    print(
        "Summary:",
        generator.summarize(output),
    )
