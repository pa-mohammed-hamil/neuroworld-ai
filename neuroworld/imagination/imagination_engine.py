
"""
NeuroWorld AI
File: neuroworld/imagination/imagination_engine.py

Neural Imagination Engine

Features:
- Multi-step future imagination
- Action-conditioned world-model rollouts
- Stochastic and deterministic prediction
- Multi-branch imagination
- Counterfactual what-if reasoning
- Goal-conditioned future selection
- Reward and risk evaluation
- Uncertainty-aware scoring
- Imagined trajectory comparison
- CPU / GPU inference

Dependencies:
    pip install torch

Integrates with:
    future_generator.py
    branch_generator.py
    counterfactual.py
    state_representation.py
    transition_model.py
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from neuroworld.imagination.future_generator import (
    FutureGenerator,
    FutureGeneratorConfig,
    FutureGenerationOutput,
)

from neuroworld.imagination.branch_generator import (
    BranchGenerator,
    BranchGeneratorConfig,
    ImaginationBranch,
)

from neuroworld.imagination.counterfactual import (
    CounterfactualEngine,
    CounterfactualConfig,
    CounterfactualIntervention,
    CounterfactualComparison,
)


RewardFunction = Callable[
    [Tensor, Tensor, Tensor], Tensor
]

RiskFunction = Callable[
    [Tensor, Tensor, Tensor], Tensor
]

ActionSampler = Callable[
    [Tensor, int], Tensor
]


@dataclass
class ImaginationEngineConfig:
    latent_dim: int = 128
    action_dim: int = 32

    horizon: int = 10
    num_futures: int = 8

    branching_factor: int = 4
    beam_width: int = 8

    stochastic: bool = True
    temperature: float = 0.5

    discount_factor: float = 0.99

    uncertainty_penalty: float = 0.1
    risk_penalty: float = 0.5
    goal_weight: float = 1.0

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    enable_future_generation: bool = True
    enable_branching: bool = True
    enable_counterfactuals: bool = True


@dataclass
class ImaginationResult:
    """
    Unified output of the imagination engine.
    """

    future_output: Optional[
        FutureGenerationOutput
    ] = None

    branches: Optional[
        List[List[ImaginationBranch]]
    ] = None

    counterfactual_output: Optional[
        CounterfactualComparison
    ] = None

    best_trajectory: Optional[Tensor] = None
    best_score: Optional[Tensor] = None

    metadata: Dict[str, object] = field(
        default_factory=dict
    )


class ImaginationEngine(nn.Module):
    """
    Central imagination orchestration engine.

    The engine combines three reasoning mechanisms:

    1. FutureGenerator:
       Samples possible futures for action sequences.

    2. BranchGenerator:
       Explores alternative actions using beam search.

    3. CounterfactualEngine:
       Compares hypothetical interventions with a
       baseline action sequence.

    Required dynamics model interface:

        mean, logvar = dynamics_model(state, action)

    Both outputs:
        [batch, latent_dim]

    The dynamics model must be trained separately.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[
            ImaginationEngineConfig
        ] = None,
        reward_fn: Optional[
            RewardFunction
        ] = None,
        risk_fn: Optional[
            RiskFunction
        ] = None,
        action_sampler: Optional[
            ActionSampler
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or ImaginationEngineConfig()
        )

        cfg = self.config

        if cfg.latent_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError(
                "Latent and action dimensions "
                "must be positive"
            )

        if cfg.horizon <= 0 or cfg.num_futures <= 0:
            raise ValueError(
                "Horizon and num_futures "
                "must be positive"
            )

        if cfg.branching_factor <= 0:
            raise ValueError(
                "branching_factor must be positive"
            )

        if cfg.beam_width <= 0:
            raise ValueError(
                "beam_width must be positive"
            )

        if not 0.0 <= cfg.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in [0,1]"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "temperature cannot be negative"
            )

        if (
            cfg.uncertainty_penalty < 0
            or cfg.risk_penalty < 0
        ):
            raise ValueError(
                "Penalty weights must be non-negative"
            )

        self.dynamics_model = dynamics_model

        # Share the same trained dynamics model with
        # all three imagination components.
        #
        # Keep the canonical model registered once
        # in this module to avoid duplicate state keys.

        future_config = FutureGeneratorConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            num_futures=cfg.num_futures,
            stochastic=cfg.stochastic,
            temperature=cfg.temperature,
            discount_factor=cfg.discount_factor,
            uncertainty_penalty=cfg.uncertainty_penalty,
            goal_weight=cfg.goal_weight,
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )

        branch_config = BranchGeneratorConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            branching_factor=cfg.branching_factor,
            beam_width=cfg.beam_width,
            stochastic=cfg.stochastic,
            temperature=cfg.temperature,
            discount_factor=cfg.discount_factor,
            uncertainty_penalty=cfg.uncertainty_penalty,
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )

        counterfactual_config = CounterfactualConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            stochastic=cfg.stochastic,
            temperature=cfg.temperature,
            discount_factor=cfg.discount_factor,
            uncertainty_penalty=cfg.uncertainty_penalty,
            risk_penalty=cfg.risk_penalty,
            min_logvar=cfg.min_logvar,
            max_logvar=cfg.max_logvar,
        )

        # Construct optional components without
        # registering the same dynamics model repeatedly.
        #
        # PyTorch supports shared module references,
        # but state_dict() may expose multiple paths.
        # The engine's dynamics_model is the canonical
        # reference used for checkpointing.

        self.future_generator = FutureGenerator(
            dynamics_model=dynamics_model,
            config=future_config,
            reward_fn=reward_fn,
        )

        self.branch_generator = BranchGenerator(
            transition_model=dynamics_model,
            config=branch_config,
            reward_fn=reward_fn,
            action_sampler=action_sampler,
        )

        self.counterfactual_engine = (
            CounterfactualEngine(
                transition_model=dynamics_model,
                config=counterfactual_config,
                reward_fn=reward_fn,
                risk_fn=risk_fn,
            )
        )

    def _validate_state(
        self,
        initial_state: Tensor,
    ) -> None:

        if initial_state.ndim != 2:
            raise ValueError(
                "Initial state must be [B,D]"
            )

        if initial_state.shape[-1] != (
            self.config.latent_dim
        ):
            raise ValueError(
                "Invalid latent dimension"
            )

        if initial_state.shape[0] == 0:
            raise ValueError(
                "Batch cannot be empty"
            )

        if not torch.is_floating_point(initial_state):
            raise TypeError(
                "Initial state must be floating point"
            )

        if not torch.isfinite(
            initial_state
        ).all().item():
            raise ValueError(
                "Initial state contains NaN or Inf"
            )

    def imagine_futures(
        self,
        initial_state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> FutureGenerationOutput:
        """
        Generate multiple possible futures.

        initial_state: [B,D]
        actions: [B,H,A] or [B,K,H,A]
        goal_state: [B,D], optional
        """

        if not self.config.enable_future_generation:
            raise RuntimeError(
                "Future generation is disabled"
            )

        self._validate_state(initial_state)

        return self.future_generator(
            initial_state=initial_state,
            actions=actions,
            goal_state=goal_state,
        )

    def imagine_branches(
        self,
        initial_state: Tensor,
    ) -> List[List[ImaginationBranch]]:
        """
        Generate alternative action-conditioned
        future trajectories using beam search.
        """

        if not self.config.enable_branching:
            raise RuntimeError(
                "Branch generation is disabled"
            )

        self._validate_state(initial_state)

        return self.branch_generator.generate(
            initial_state
        )

    def imagine_counterfactuals(
        self,
        initial_state: Tensor,
        actions: Tensor,
        interventions: List[
            CounterfactualIntervention
        ],
    ) -> CounterfactualComparison:
        """
        Compare baseline and hypothetical futures.
        """

        if not self.config.enable_counterfactuals:
            raise RuntimeError(
                "Counterfactual reasoning is disabled"
            )

        self._validate_state(initial_state)

        return self.counterfactual_engine.compare(
            initial_state=initial_state,
            actions=actions,
            interventions=interventions,
        )

    def select_best_future(
        self,
        output: FutureGenerationOutput,
    ) -> tuple[Tensor, Tensor]:
        """
        Select highest-scoring future for each batch.

        Returns:
            trajectories: [B,H+1,D]
            scores: [B]
        """

        batch_size = output.scores.shape[0]

        batch_indices = torch.arange(
            batch_size,
            device=output.scores.device,
        )

        indices = output.best_indices

        trajectories = output.trajectories[
            batch_indices,
            indices,
        ]

        scores = output.scores[
            batch_indices,
            indices,
        ]

        return trajectories, scores

    def select_best_branch(
        self,
        branches: List[
            List[ImaginationBranch]
        ],
    ) -> tuple[Tensor, Tensor]:
        """
        Select best branch from each batch.

        Returns:
            trajectories: [B,H+1,D]
            scores: [B]
        """

        if not branches or any(
            len(item) == 0 for item in branches
        ):
            raise ValueError(
                "Every batch must contain branches"
            )

        best = [
            max(
                batch_branches,
                key=lambda branch: (
                    branch.score.detach().item()
                ),
            )
            for batch_branches in branches
        ]

        trajectories = torch.stack(
            [branch.states for branch in best],
            dim=0,
        )

        scores = torch.stack(
            [branch.score for branch in best],
            dim=0,
        )

        return trajectories, scores

    def forward(
        self,
        initial_state: Tensor,
        actions: Optional[Tensor] = None,
        goal_state: Optional[Tensor] = None,
        interventions: Optional[
            List[CounterfactualIntervention]
        ] = None,
        run_branches: bool = False,
    ) -> ImaginationResult:
        """
        Unified imagination execution.

        If actions are supplied:
            Generate possible futures.

        If run_branches=True:
            Explore alternative actions.

        If interventions are supplied:
            Compare counterfactual scenarios.

        Note:
            Branch and future scores use related
            but not identical objectives when
            goal conditioning is enabled.
        """

        self._validate_state(initial_state)

        result = ImaginationResult()

        result.metadata = {
            "batch_size": initial_state.shape[0],
            "latent_dim": self.config.latent_dim,
            "horizon": self.config.horizon,
            "stochastic": self.config.stochastic,
            "num_futures": self.config.num_futures,
        }

        if actions is not None:
            if self.config.enable_future_generation:
                result.future_output = (
                    self.imagine_futures(
                        initial_state,
                        actions,
                        goal_state,
                    )
                )

                (
                    result.best_trajectory,
                    result.best_score,
                ) = self.select_best_future(
                    result.future_output
                )

        if run_branches:
            result.branches = self.imagine_branches(
                initial_state
            )

            branch_trajectory, branch_score = (
                self.select_best_branch(
                    result.branches
                )
            )

            # Do not compare incompatible objectives:
            # branch scoring does not include the
            # future generator's goal-distance term.
            if result.best_trajectory is None:
                result.best_trajectory = (
                    branch_trajectory
                )
                result.best_score = branch_score

        if interventions is not None:
            if actions is None:
                raise ValueError(
                    "Counterfactuals require baseline actions"
                )

            result.counterfactual_output = (
                self.imagine_counterfactuals(
                    initial_state,
                    actions,
                    interventions,
                )
            )

        if (
            result.future_output is None
            and result.branches is None
            and result.counterfactual_output is None
        ):
            raise ValueError(
                "No imagination task was executed"
            )

        return result

    @torch.no_grad()
    def generate(
        self,
        initial_state: Tensor,
        actions: Optional[Tensor] = None,
        goal_state: Optional[Tensor] = None,
        interventions: Optional[
            List[CounterfactualIntervention]
        ] = None,
        run_branches: bool = False,
    ) -> ImaginationResult:
        """
        Inference-only imagination execution.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(
                initial_state=initial_state,
                actions=actions,
                goal_state=goal_state,
                interventions=interventions,
                run_branches=run_branches,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def summarize(
        self,
        result: ImaginationResult,
    ) -> Dict[str, object]:
        """
        Convert key imagination metrics into
        JSON-compatible Python values.
        """

        summary: Dict[str, object] = dict(
            result.metadata
        )

        if result.future_output is not None:
            output = result.future_output

            summary["future_scores"] = (
                output.scores.detach()
                .cpu().tolist()
            )

            summary["best_future_indices"] = (
                output.best_indices.detach()
                .cpu().tolist()
            )

            summary["mean_future_uncertainty"] = (
                output.uncertainties.mean(
                    dim=(1, 2)
                ).detach().cpu().tolist()
            )

        if result.branches is not None:
            summary["branch_counts"] = [
                len(batch_branches)
                for batch_branches in result.branches
            ]

            summary["best_branch_scores"] = [
                max(
                    branch.score.detach().item()
                    for branch in batch_branches
                )
                for batch_branches in result.branches
            ]

        if result.counterfactual_output is not None:
            comparison = result.counterfactual_output

            summary["counterfactual_best_indices"] = (
                comparison.best_indices.detach()
                .cpu().tolist()
            )

            summary["counterfactual_score_differences"] = (
                comparison.score_differences.detach()
                .cpu().tolist()
            )

        if result.best_score is not None:
            summary["selected_scores"] = (
                result.best_score.detach()
                .cpu().tolist()
            )

        return summary


if __name__ == "__main__":

    from neuroworld.imagination.future_generator import (
        ExampleFutureDynamics,
    )

    torch.manual_seed(42)

    config = ImaginationEngineConfig(
        latent_dim=128,
        action_dim=32,
        horizon=8,
        num_futures=6,
        branching_factor=4,
        beam_width=5,
        stochastic=True,
        temperature=0.3,
    )

    dynamics = ExampleFutureDynamics(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
    )

    engine = ImaginationEngine(
        dynamics_model=dynamics,
        config=config,
    )

    batch_size = 2

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

    interventions = [
        CounterfactualIntervention(
            name="stop_action",
            action_overrides={
                2: torch.zeros(
                    config.action_dim
                ),
            },
        ),
        CounterfactualIntervention(
            name="alternative_action",
            action_overrides={
                3: torch.ones(
                    config.action_dim
                ) * 0.25,
            },
        ),
    ]

    result = engine.generate(
        initial_state=initial_state,
        actions=actions,
        goal_state=goal_state,
        interventions=interventions,
        run_branches=True,
    )

    if result.best_trajectory is not None:
        print(
            "Best trajectory:",
            result.best_trajectory.shape,
        )

    print(
        "Imagination summary:",
        engine.summarize(result),
    )
