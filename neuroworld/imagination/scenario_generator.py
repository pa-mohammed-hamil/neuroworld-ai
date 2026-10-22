
"""
NeuroWorld AI
File: neuroworld/imagination/scenario_generator.py

Neural Scenario Generation Engine

Features:
- Generate multiple hypothetical scenarios
- Baseline and alternative action sequences
- Action perturbation and exploration
- Environmental disturbance modeling
- Goal-conditioned trajectory scoring
- Risk-aware scenario evaluation
- Stochastic world-model predictions
- Scenario diversity measurement
- Best-scenario selection
- Batched PyTorch inference
- Integration with WorldRollout

Dependencies:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from neuroworld.imagination.rollout import (
    WorldRollout,
    RolloutConfig,
    RolloutOutput,
)


TransitionMetric = Callable[
    [Tensor, Tensor, Tensor], Tensor
]

TerminationFunction = Callable[[Tensor], Tensor]


@dataclass
class ScenarioGeneratorConfig:
    latent_dim: int = 128
    action_dim: int = 32
    horizon: int = 10
    num_scenarios: int = 8

    stochastic: bool = True
    temperature: float = 0.5

    action_noise_std: float = 0.2
    state_noise_std: float = 0.05

    action_min: float = -1.0
    action_max: float = 1.0

    discount_factor: float = 0.99

    uncertainty_penalty: float = 0.1
    risk_penalty: float = 0.5
    goal_weight: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    enable_early_termination: bool = False


@dataclass
class ScenarioSpecification:
    """
    Defines a hypothetical scenario.

    action_override:
        Optional action sequence [B,H,A] or [H,A].

    action_scale:
        Multiplicative action adjustment.

    action_bias:
        Additive action adjustment.

    initial_state_offset:
        Optional latent-state perturbation
        [B,D] or [D].

    metadata:
        Scenario description and context.
    """

    name: str

    action_override: Optional[Tensor] = None

    action_scale: float = 1.0
    action_bias: float = 0.0

    initial_state_offset: Optional[Tensor] = None

    metadata: Dict[str, object] = field(
        default_factory=dict
    )


@dataclass
class ScenarioGenerationOutput:
    """
    Scenario generation results.

    states:
        [B,K,H+1,D]

    actions:
        [B,K,H,A]

    rewards:
        [B,K,H]

    risks:
        [B,K,H]

    uncertainties:
        [B,K,H]

    scores:
        [B,K]

    best_indices:
        [B]

    diversity:
        [B]
    """

    names: List[str]
    states: Tensor
    actions: Tensor
    rewards: Tensor
    risks: Tensor
    uncertainties: Tensor
    valid_steps: Tensor
    scores: Tensor
    best_indices: Tensor
    diversity: Tensor
    metadata: List[Dict[str, object]]

    @property
    def final_states(self) -> Tensor:
        lengths = self.valid_steps.long().sum(
            dim=-1
        )

        batch_size, num_scenarios = (
            self.states.shape[:2]
        )

        batch_idx = torch.arange(
            batch_size,
            device=self.states.device,
        )[:, None]

        scenario_idx = torch.arange(
            num_scenarios,
            device=self.states.device,
        )[None, :]

        return self.states[
            batch_idx,
            scenario_idx,
            lengths,
        ]

    @property
    def best_trajectories(self) -> Tensor:
        batch_idx = torch.arange(
            self.states.shape[0],
            device=self.states.device,
        )

        return self.states[
            batch_idx,
            self.best_indices,
        ]

    @property
    def best_scores(self) -> Tensor:
        return self.scores.gather(
            1,
            self.best_indices.unsqueeze(1),
        ).squeeze(1)


class ScenarioGenerator(nn.Module):
    """
    Generate and evaluate imagined future scenarios.

    Required dynamics model interface:

        mean, logvar = dynamics_model(state, action)

    Both outputs:
        [N, latent_dim]

    Scenario generation is model-based simulation,
    not proof of real-world causal outcomes.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[
            ScenarioGeneratorConfig
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
            config or ScenarioGeneratorConfig()
        )

        cfg = self.config

        if (
            cfg.latent_dim <= 0
            or cfg.action_dim <= 0
            or cfg.horizon <= 0
            or cfg.num_scenarios <= 0
        ):
            raise ValueError(
                "Dimensions, horizon and scenario "
                "count must be positive"
            )

        if (
            cfg.action_noise_std < 0
            or cfg.state_noise_std < 0
            or cfg.temperature < 0
        ):
            raise ValueError(
                "Noise and temperature must be "
                "non-negative"
            )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be less than action_max"
            )

        rollout_config = RolloutConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            num_candidates=cfg.num_scenarios,
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

    def _validate_inputs(
        self,
        initial_state: Tensor,
        baseline_actions: Tensor,
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

        expected_actions = (
            initial_state.shape[0],
            cfg.horizon,
            cfg.action_dim,
        )

        if baseline_actions.shape != expected_actions:
            raise ValueError(
                "baseline_actions must be [B,H,A]"
            )

        if (
            goal_state is not None
            and goal_state.shape != initial_state.shape
        ):
            raise ValueError(
                "goal_state must be [B,D]"
            )

        for name, value in (
            ("initial_state", initial_state),
            ("baseline_actions", baseline_actions),
            ("goal_state", goal_state),
        ):
            if value is None:
                continue

            if not torch.is_floating_point(value):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if (
                value.device != initial_state.device
                or value.dtype != initial_state.dtype
            ):
                raise ValueError(
                    f"{name} device/dtype mismatch"
                )

            if not torch.isfinite(value).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def _expand_override(
        self,
        value: Tensor,
        reference: Tensor,
    ) -> Tensor:
        """
        Expand shared overrides across batch.

        Accepts:
            [H,A] or [B,H,A] for actions
            [D] or [B,D] for state offsets
        """

        if value.ndim == reference.ndim - 1:
            value = value.unsqueeze(0).expand(
                reference.shape[0],
                *value.shape,
            )

        if value.shape != reference.shape:
            raise ValueError(
                "Scenario override shape mismatch"
            )

        value = value.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if not torch.isfinite(value).all().item():
            raise ValueError(
                "Scenario override contains NaN or Inf"
            )

        return value

    def build_actions(
        self,
        baseline_actions: Tensor,
        specifications: Optional[
            List[ScenarioSpecification]
        ] = None,
    ) -> tuple[
        Tensor,
        List[str],
        List[Dict[str, object]],
    ]:
        """
        Generate candidate action sequences.

        Scenario 0 is the baseline.
        Additional scenarios are either explicitly
        specified or generated through perturbation.
        """

        cfg = self.config

        batch_size = baseline_actions.shape[0]

        candidates = [
            baseline_actions.clamp(
                cfg.action_min,
                cfg.action_max,
            )
        ]

        names = ["baseline"]
        metadata = [{"type": "baseline"}]

        specifications = specifications or []

        if len(specifications) > cfg.num_scenarios - 1:
            raise ValueError(
                "Too many scenario specifications"
            )

        for specification in specifications:

            if specification.action_override is not None:
                candidate = self._expand_override(
                    specification.action_override,
                    baseline_actions,
                )
            else:
                candidate = (
                    baseline_actions
                    * specification.action_scale
                    + specification.action_bias
                )

            candidate = candidate.clamp(
                cfg.action_min,
                cfg.action_max,
            )

            candidates.append(candidate)
            names.append(specification.name)

            metadata.append(
                dict(specification.metadata)
            )

        while len(candidates) < cfg.num_scenarios:

            noise = torch.randn_like(
                baseline_actions
            ) * cfg.action_noise_std

            candidate = (
                baseline_actions + noise
            ).clamp(
                cfg.action_min,
                cfg.action_max,
            )

            candidates.append(candidate)

            names.append(
                f"random_scenario_{len(candidates)-1}"
            )

            metadata.append({
                "type": "action_perturbation",
                "noise_std": cfg.action_noise_std,
            })

        return (
            torch.stack(candidates, dim=1),
            names,
            metadata,
        )

    def build_initial_states(
        self,
        initial_state: Tensor,
        specifications: Optional[
            List[ScenarioSpecification]
        ] = None,
    ) -> Tensor:
        """
        Build scenario-specific initial latent states.

        Returns:
            [B,K,D]

        Scenario 0 remains the original state.
        """

        cfg = self.config

        states = [initial_state]

        specifications = specifications or []

        for specification in specifications:

            if specification.initial_state_offset is None:
                state = initial_state
            else:
                offset = self._expand_override(
                    specification.initial_state_offset,
                    initial_state,
                )

                state = initial_state + offset

            states.append(state)

        while len(states) < cfg.num_scenarios:

            if cfg.state_noise_std > 0:
                state = (
                    initial_state
                    + torch.randn_like(initial_state)
                    * cfg.state_noise_std
                )
            else:
                state = initial_state

            states.append(state)

        return torch.stack(states, dim=1)

    def _rollout_scenario_states(
        self,
        scenario_states: Tensor,
        scenario_actions: Tensor,
        goal_state: Optional[Tensor],
    ) -> RolloutOutput:
        """
        WorldRollout accepts one initial state per
        batch item and K candidate action sequences.

        To support different initial states for each
        scenario, flatten B*K into independent batch
        items, using one candidate per item.

        This method temporarily uses a dedicated
        one-candidate WorldRollout configuration.
        """

        cfg = self.config

        batch_size = scenario_states.shape[0]
        num_scenarios = scenario_states.shape[1]

        flat_states = scenario_states.reshape(
            batch_size * num_scenarios,
            cfg.latent_dim,
        )

        flat_actions = scenario_actions.reshape(
            batch_size * num_scenarios,
            cfg.horizon,
            cfg.action_dim,
        )

        flat_goal = None

        if goal_state is not None:
            flat_goal = goal_state.unsqueeze(1).expand(
                -1,
                num_scenarios,
                -1,
            ).reshape(
                batch_size * num_scenarios,
                cfg.latent_dim,
            )

        # Reuse the trained dynamics model and metric
        # functions while changing candidate count.
        one_candidate_config = RolloutConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            num_candidates=1,
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

        engine = WorldRollout(
            dynamics_model=self.rollout_engine.dynamics_model,
            config=one_candidate_config,
            reward_fn=self.rollout_engine.reward_fn,
            risk_fn=self.rollout_engine.risk_fn,
            termination_fn=(
                self.rollout_engine.termination_fn
            ),
        )

        return engine(
            initial_state=flat_states,
            actions=flat_actions,
            goal_state=flat_goal,
        )

    def forward(
        self,
        initial_state: Tensor,
        baseline_actions: Tensor,
        goal_state: Optional[Tensor] = None,
        specifications: Optional[
            List[ScenarioSpecification]
        ] = None,
    ) -> ScenarioGenerationOutput:
        """
        Generate and evaluate hypothetical scenarios.

        initial_state:
            [B,D]

        baseline_actions:
            [B,H,A]

        goal_state:
            [B,D], optional
        """

        self._validate_inputs(
            initial_state,
            baseline_actions,
            goal_state,
        )

        cfg = self.config

        scenario_actions, names, metadata = (
            self.build_actions(
                baseline_actions,
                specifications,
            )
        )

        scenario_states = (
            self.build_initial_states(
                initial_state,
                specifications,
            )
        )

        raw_output = self._rollout_scenario_states(
            scenario_states,
            scenario_actions,
            goal_state,
        )

        batch_size = initial_state.shape[0]
        num_scenarios = cfg.num_scenarios

        def reshape_metric(
            tensor: Tensor,
        ) -> Tensor:
            # Remove the one-candidate dimension,
            # then restore scenario dimension.
            return tensor.squeeze(1).reshape(
                batch_size,
                num_scenarios,
                *tensor.shape[2:],
            )

        states = reshape_metric(raw_output.states)
        actions = reshape_metric(raw_output.actions)
        rewards = reshape_metric(raw_output.rewards)
        risks = reshape_metric(raw_output.risks)

        uncertainties = reshape_metric(
            raw_output.uncertainties
        )

        valid_steps = reshape_metric(
            raw_output.valid_steps
        )

        scores = raw_output.scores.reshape(
            batch_size,
            num_scenarios,
        )

        best_indices = scores.argmax(dim=1)

        lengths = valid_steps.long().sum(dim=-1)

        batch_idx = torch.arange(
            batch_size,
            device=states.device,
        )[:, None]

        scenario_idx = torch.arange(
            num_scenarios,
            device=states.device,
        )[None, :]

        final_states = states[
            batch_idx,
            scenario_idx,
            lengths,
        ]

        center = final_states.mean(
            dim=1,
            keepdim=True,
        )

        diversity = torch.linalg.vector_norm(
            final_states - center,
            dim=-1,
        ).mean(dim=1)

        return ScenarioGenerationOutput(
            names=names,
            states=states,
            actions=actions,
            rewards=rewards,
            risks=risks,
            uncertainties=uncertainties,
            valid_steps=valid_steps,
            scores=scores,
            best_indices=best_indices,
            diversity=diversity,
            metadata=metadata,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        baseline_actions: Tensor,
        goal_state: Optional[Tensor] = None,
        specifications: Optional[
            List[ScenarioSpecification]
        ] = None,
    ) -> ScenarioGenerationOutput:
        """
        Inference-only scenario generation.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(
                initial_state=initial_state,
                baseline_actions=baseline_actions,
                goal_state=goal_state,
                specifications=specifications,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def summarize(
        self,
        output: ScenarioGenerationOutput,
    ) -> Dict[str, object]:
        """
        Return JSON-compatible scenario statistics.
        """

        best_indices = (
            output.best_indices.detach()
            .cpu().tolist()
        )

        return {
            "scenario_names": output.names,
            "batch_size": output.states.shape[0],
            "num_scenarios": output.states.shape[1],
            "best_indices": best_indices,
            "best_scenario_names": [
                output.names[index]
                for index in best_indices
            ],
            "best_scores": (
                output.best_scores.detach()
                .cpu().tolist()
            ),
            "scenario_scores": (
                output.scores.detach()
                .cpu().tolist()
            ),
            "diversity": (
                output.diversity.detach()
                .cpu().tolist()
            ),
            "mean_uncertainty": (
                output.uncertainties.mean(
                    dim=-1
                ).detach().cpu().tolist()
            ),
        }


if __name__ == "__main__":

    from neuroworld.imagination.rollout import (
        ExampleWorldDynamics,
    )

    torch.manual_seed(42)

    config = ScenarioGeneratorConfig(
        latent_dim=128,
        action_dim=32,
        horizon=10,
        num_scenarios=6,
        stochastic=True,
        temperature=0.3,
        action_noise_std=0.2,
        state_noise_std=0.05,
    )

    dynamics = ExampleWorldDynamics(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
    )

    generator = ScenarioGenerator(
        dynamics_model=dynamics,
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

    goal_state = torch.randn(
        batch_size,
        config.latent_dim,
    )

    scenarios = [
        ScenarioSpecification(
            name="reduced_action",
            action_scale=0.5,
            metadata={
                "description": "Reduced control intensity"
            },
        ),
        ScenarioSpecification(
            name="emergency_stop",
            action_override=torch.zeros(
                config.horizon,
                config.action_dim,
            ),
            metadata={
                "description": "Zero-action scenario"
            },
        ),
        ScenarioSpecification(
            name="disturbed_environment",
            initial_state_offset=torch.randn(
                config.latent_dim,
            ) * 0.1,
            metadata={
                "description": "Perturbed initial state"
            },
        ),
    ]

    output = generator.generate(
        initial_state=initial_state,
        baseline_actions=baseline_actions,
        goal_state=goal_state,
        specifications=scenarios,
    )

    print("States:", output.states.shape)
    print("Actions:", output.actions.shape)
    print("Rewards:", output.rewards.shape)
    print("Risks:", output.risks.shape)
    print("Scores:", output.scores.shape)

    print(
        "Best trajectories:",
        output.best_trajectories.shape,
    )

    print(
        "Scenario summary:",
        generator.summarize(output),
    )
