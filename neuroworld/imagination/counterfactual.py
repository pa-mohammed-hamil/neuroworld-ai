
"""
NeuroWorld AI
File: neuroworld/imagination/counterfactual.py

Counterfactual Imagination and Reasoning Engine

Features:
- What-if scenario generation
- Action-based counterfactual interventions
- Latent-state interventions
- Baseline vs alternative future comparison
- Multi-step world-model rollouts
- Reward and risk evaluation
- Predictive uncertainty tracking
- Counterfactual ranking
- Batched GPU / CPU inference
- Integration with trained world models

Dependencies:
    pip install torch
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class CounterfactualConfig:
    latent_dim: int = 128
    action_dim: int = 32

    horizon: int = 10
    discount_factor: float = 0.99

    stochastic: bool = False
    temperature: float = 0.5

    uncertainty_penalty: float = 0.1
    risk_penalty: float = 0.5

    min_logvar: float = -10.0
    max_logvar: float = 5.0


@dataclass
class CounterfactualIntervention:
    """
    A hypothetical intervention.

    action_overrides:
        Mapping from time step to replacement action.

    state_overrides:
        Mapping from time step to replacement latent state.

    A state override at step t replaces the state
    before the action at step t is applied.
    """

    name: str
    action_overrides: Optional[Dict[int, Tensor]] = None
    state_overrides: Optional[Dict[int, Tensor]] = None


@dataclass
class CounterfactualTrajectory:
    """
    Imagined trajectory and evaluation metrics.

    states:        [B, H+1, D]
    actions:       [B, H, A]
    rewards:       [B, H]
    risks:         [B, H]
    uncertainties: [B, H]
    scores:        [B]
    """

    name: str
    states: Tensor
    actions: Tensor
    rewards: Tensor
    risks: Tensor
    uncertainties: Tensor
    scores: Tensor

    @property
    def final_state(self) -> Tensor:
        return self.states[:, -1, :]


@dataclass
class CounterfactualComparison:
    baseline: CounterfactualTrajectory
    alternatives: List[CounterfactualTrajectory]
    score_differences: Tensor
    final_state_distances: Tensor
    best_indices: Tensor


class CounterfactualEngine(nn.Module):
    """
    Generate alternative futures using a world model.

    Required transition model interface:

        mean, logvar = transition_model(state, action)

    Both outputs must have shape [B, latent_dim].

    The engine does not itself learn causal structure.
    Counterfactual reliability depends on the quality
    of the trained transition model and interventions.
    """

    def __init__(
        self,
        transition_model: nn.Module,
        config: Optional[CounterfactualConfig] = None,
        reward_fn: Optional[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = None,
        risk_fn: Optional[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = None,
    ):
        super().__init__()

        self.config = config or CounterfactualConfig()
        self.transition_model = transition_model

        self.reward_fn = reward_fn
        self.risk_fn = risk_fn

        cfg = self.config

        if cfg.latent_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError("Dimensions must be positive")

        if cfg.horizon <= 0:
            raise ValueError("Horizon must be positive")

        if not 0 <= cfg.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in [0, 1]"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature cannot be negative"
            )

        if (
            cfg.risk_penalty < 0
            or cfg.uncertainty_penalty < 0
        ):
            raise ValueError(
                "Penalty weights cannot be negative"
            )

    def _validate_inputs(
        self,
        initial_state: Tensor,
        actions: Tensor,
    ) -> None:

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B, D]"
            )

        if actions.ndim != 3:
            raise ValueError(
                "actions must be [B, H, A]"
            )

        if initial_state.shape[0] != actions.shape[0]:
            raise ValueError(
                "Batch dimensions must match"
            )

        if initial_state.shape[-1] != (
            self.config.latent_dim
        ):
            raise ValueError(
                "Invalid latent state dimension"
            )

        if actions.shape[1:] != (
            self.config.horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                "Invalid action sequence dimensions"
            )

        if (
            initial_state.device != actions.device
            or initial_state.dtype != actions.dtype
        ):
            raise ValueError(
                "State and actions must share device and dtype"
            )

    def _resolve_override(
        self,
        override: Tensor,
        reference: Tensor,
    ) -> Tensor:
        """
        Accept one override per batch or one shared override.

        Valid shapes:
            [D]
            [B, D]
        """

        if override.ndim == 1:
            override = override.unsqueeze(0).expand(
                reference.shape[0], -1
            )

        if override.shape != reference.shape:
            raise ValueError(
                "Intervention shape mismatch"
            )

        return override.to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def predict_transition(
        self,
        state: Tensor,
        action: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict the next latent state and model variance.

        noise allows common random numbers across
        alternative scenarios for fairer comparison.
        """

        mean, logvar = self.transition_model(
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
            if noise is None:
                noise = torch.randn_like(mean)

            if noise.shape != mean.shape:
                raise ValueError(
                    "Transition noise shape mismatch"
                )

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

    def evaluate_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Default demonstration reward:
        penalize large state displacement.
        """

        if self.reward_fn is not None:
            result = self.reward_fn(
                state, action, next_state
            )
        else:
            result = -0.01 * (
                next_state - state
            ).pow(2).mean(dim=-1)

        if result.shape != state.shape[:-1]:
            raise ValueError(
                "Reward function must return [B]"
            )

        return result

    def evaluate_risk(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Default demonstration risk:
        penalize large latent-state norms.

        Replace with a domain-specific risk model.
        """

        if self.risk_fn is not None:
            result = self.risk_fn(
                state, action, next_state
            )
        else:
            result = (
                next_state.pow(2).mean(dim=-1)
                * 0.001
            )

        if result.shape != state.shape[:-1]:
            raise ValueError(
                "Risk function must return [B]"
            )

        return result

    def rollout(
        self,
        initial_state: Tensor,
        actions: Tensor,
        intervention: Optional[
            CounterfactualIntervention
        ] = None,
        shared_noise: Optional[Tensor] = None,
    ) -> CounterfactualTrajectory:
        """
        Roll out one baseline or alternative scenario.

        shared_noise:
            [B, H, latent_dim]
        """

        self._validate_inputs(
            initial_state, actions
        )

        cfg = self.config

        if shared_noise is not None:
            expected = (
                initial_state.shape[0],
                cfg.horizon,
                cfg.latent_dim,
            )
            if shared_noise.shape != expected:
                raise ValueError(
                    "shared_noise shape mismatch"
                )

        state = initial_state

        states = [state]
        executed_actions = []
        rewards = []
        risks = []
        uncertainties = []

        action_overrides = (
            intervention.action_overrides or {}
            if intervention is not None
            else {}
        )

        state_overrides = (
            intervention.state_overrides or {}
            if intervention is not None
            else {}
        )

        for step in range(cfg.horizon):

            if step in state_overrides:
                state = self._resolve_override(
                    state_overrides[step],
                    state,
                )
                # State at step t is replaced by intervention.
                states[-1] = state

            action = actions[:, step, :]

            if step in action_overrides:
                action = self._resolve_override(
                    action_overrides[step],
                    action,
                )

            noise = (
                shared_noise[:, step, :]
                if shared_noise is not None
                else None
            )

            next_state, uncertainty = (
                self.predict_transition(
                    state, action, noise
                )
            )

            reward = self.evaluate_reward(
                state, action, next_state
            )

            risk = self.evaluate_risk(
                state, action, next_state
            )

            executed_actions.append(action)
            rewards.append(reward)
            risks.append(risk)
            uncertainties.append(uncertainty)

            state = next_state
            states.append(state)

        state_tensor = torch.stack(
            states, dim=1
        )

        action_tensor = torch.stack(
            executed_actions, dim=1
        )

        reward_tensor = torch.stack(
            rewards, dim=1
        )

        risk_tensor = torch.stack(
            risks, dim=1
        )

        uncertainty_tensor = torch.stack(
            uncertainties, dim=1
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
            reward_tensor
            - cfg.risk_penalty * risk_tensor
            - cfg.uncertainty_penalty
            * uncertainty_tensor
        )

        scores = (
            step_scores * discount.unsqueeze(0)
        ).sum(dim=1)

        name = (
            intervention.name
            if intervention is not None
            else "baseline"
        )

        return CounterfactualTrajectory(
            name=name,
            states=state_tensor,
            actions=action_tensor,
            rewards=reward_tensor,
            risks=risk_tensor,
            uncertainties=uncertainty_tensor,
            scores=scores,
        )

    def compare(
        self,
        initial_state: Tensor,
        actions: Tensor,
        interventions: List[
            CounterfactualIntervention
        ],
    ) -> CounterfactualComparison:
        """
        Compare baseline and alternative scenarios.

        best_indices:
            0 means baseline is best.
            1..N refer to alternatives in input order.
        """

        if not interventions:
            raise ValueError(
                "At least one intervention is required"
            )

        shared_noise = None

        if self.config.stochastic:
            shared_noise = torch.randn(
                initial_state.shape[0],
                self.config.horizon,
                self.config.latent_dim,
                device=initial_state.device,
                dtype=initial_state.dtype,
            )

        baseline = self.rollout(
            initial_state,
            actions,
            shared_noise=shared_noise,
        )

        alternatives = [
            self.rollout(
                initial_state,
                actions,
                intervention=intervention,
                shared_noise=shared_noise,
            )
            for intervention in interventions
        ]

        score_differences = torch.stack(
            [
                alternative.scores - baseline.scores
                for alternative in alternatives
            ],
            dim=1,
        )

        final_state_distances = torch.stack(
            [
                torch.linalg.vector_norm(
                    alternative.final_state
                    - baseline.final_state,
                    dim=-1,
                )
                for alternative in alternatives
            ],
            dim=1,
        )

        all_scores = torch.stack(
            [
                baseline.scores,
                *[
                    alternative.scores
                    for alternative in alternatives
                ],
            ],
            dim=1,
        )

        best_indices = all_scores.argmax(dim=1)

        return CounterfactualComparison(
            baseline=baseline,
            alternatives=alternatives,
            score_differences=score_differences,
            final_state_distances=final_state_distances,
            best_indices=best_indices,
        )

    @torch.no_grad()
    def evaluate_scenarios(
        self,
        initial_state: Tensor,
        actions: Tensor,
        interventions: List[
            CounterfactualIntervention
        ],
    ) -> Dict[str, object]:
        """
        Return detached, serializable scenario metrics.
        """

        was_training = self.training
        self.eval()

        try:
            comparison = self.compare(
                initial_state,
                actions,
                interventions,
            )

            names = [
                "baseline",
                *[
                    alternative.name
                    for alternative in comparison.alternatives
                ],
            ]

            return {
                "scenario_names": names,
                "baseline_scores": (
                    comparison.baseline.scores
                    .cpu().tolist()
                ),
                "score_differences": (
                    comparison.score_differences
                    .cpu().tolist()
                ),
                "final_state_distances": (
                    comparison.final_state_distances
                    .cpu().tolist()
                ),
                "best_scenario_indices": (
                    comparison.best_indices
                    .cpu().tolist()
                ),
                "best_scenario_names": [
                    names[index]
                    for index in comparison.best_indices
                    .cpu().tolist()
                ],
            }
        finally:
            self.train(was_training)


class ExampleTransitionModel(nn.Module):
    """
    Demonstration-only trainable latent dynamics model.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim, latent_dim
        )

        self.logvar_head = nn.Linear(
            hidden_dim, latent_dim
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        hidden = self.network(
            torch.cat([state, action], dim=-1)
        )

        mean = state + self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        return mean, logvar


if __name__ == "__main__":

    torch.manual_seed(42)

    config = CounterfactualConfig(
        latent_dim=128,
        action_dim=32,
        horizon=8,
        stochastic=True,
        temperature=0.3,
    )

    transition_model = ExampleTransitionModel(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
    )

    engine = CounterfactualEngine(
        transition_model=transition_model,
        config=config,
    )

    batch_size = 2

    initial_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    baseline_actions = torch.randn(
        batch_size,
        config.horizon,
        config.action_dim,
    ).clamp(-1.0, 1.0)

    interventions = [
        CounterfactualIntervention(
            name="slow_down",
            action_overrides={
                2: torch.zeros(config.action_dim),
                3: torch.zeros(config.action_dim),
            },
        ),
        CounterfactualIntervention(
            name="alternative_action",
            action_overrides={
                2: torch.ones(
                    config.action_dim
                ) * 0.25,
            },
        ),
        CounterfactualIntervention(
            name="changed_world_state",
            state_overrides={
                3: initial_state * 0.8,
            },
        ),
    ]

    comparison = engine.compare(
        initial_state,
        baseline_actions,
        interventions,
    )

    print(
        "Baseline states:",
        comparison.baseline.states.shape,
    )

    print(
        "Score differences:",
        comparison.score_differences.shape,
    )

    print(
        "Best scenario indices:",
        comparison.best_indices.tolist(),
    )

    report = engine.evaluate_scenarios(
        initial_state,
        baseline_actions,
        interventions,
    )

    print("Scenario report:", report)
