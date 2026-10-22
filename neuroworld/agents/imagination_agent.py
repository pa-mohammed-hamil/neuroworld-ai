
"""
NeuroWorld AI
File: neuroworld/agents/imagination_agent.py

Autonomous Imagination and Planning Agent

Features:
- Latent-space future imagination
- World-model trajectory prediction
- Random shooting and CEM action planning
- Goal-conditioned imagination
- Risk-aware trajectory selection
- Uncertainty-aware scoring
- Imagination memory integration
- Counterfactual action comparison
- Receding-horizon action selection
- Multi-candidate trajectory evaluation
- Planning statistics
- CPU/CUDA support

Dependencies:
    pip install torch

Compatible modules:
    neuroworld.imagination.imagination_engine
    neuroworld.imagination.trajectory_generator
    neuroworld.imagination.imagination_memory
    neuroworld.dynamics.transition_model

Important:
    Imagined trajectories are model predictions,
    not verified real-world future events.

    Physical robot commands must be checked by
    an independent safety controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from neuroworld.imagination.imagination_engine import (
    ImaginationEngine,
    ImaginationEngineConfig,
)

from neuroworld.imagination.trajectory_generator import (
    TrajectoryGenerator,
    TrajectoryGeneratorConfig,
    TrajectoryGenerationOutput,
)

from neuroworld.imagination.imagination_memory import (
    ImaginationMemory,
    ImaginationMemoryConfig,
)


RewardFunction = Callable[
    [Tensor, Tensor, Tensor],
    Tensor,
]

RiskFunction = Callable[
    [Tensor, Tensor, Tensor],
    Tensor,
]


@dataclass
class ImaginationAgentConfig:
    latent_dim: int = 128
    action_dim: int = 32

    planning_horizon: int = 10
    num_candidates: int = 64

    planning_method: str = "cem"
    cem_iterations: int = 5
    elite_fraction: float = 0.2

    stochastic: bool = False
    temperature: float = 0.5

    action_min: float = -1.0
    action_max: float = 1.0

    discount_factor: float = 0.99

    uncertainty_penalty: float = 0.1
    risk_penalty: float = 0.5
    goal_weight: float = 1.0

    memory_capacity: int = 10000
    store_imaginations: bool = True

    enable_counterfactuals: bool = True
    enable_branching: bool = True

    max_stored_candidates: int = 5


@dataclass
class ImaginedPlan:
    actions: Tensor
    trajectory: Tensor
    score: Tensor

    candidate_scores: Tensor
    candidate_actions: Tensor
    candidate_trajectories: Tensor

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ImaginationDecision:
    action: Tensor
    plan: ImaginedPlan

    predicted_next_state: Tensor
    predicted_final_state: Tensor

    score: Tensor


@dataclass
class ImaginationAgentStatistics:
    total_plans: int
    total_decisions: int
    stored_episodes: int

    mean_plan_score: float
    best_plan_score: float


class ImaginationAgent(nn.Module):
    """
    Model-based imagination agent.

    Main pipeline:

        Current latent state
                |
                v
        Candidate action generation
                |
                v
        World-model rollout
                |
                v
        Reward/risk evaluation
                |
                v
        Best imagined trajectory
                |
                v
        First action selection
                |
                v
        Replan after observation

    The agent uses receding-horizon planning.

    It does not directly execute actions
    in an external environment.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[
            ImaginationAgentConfig
        ] = None,
        reward_fn: Optional[
            RewardFunction
        ] = None,
        risk_fn: Optional[
            RiskFunction
        ] = None,
        imagination_memory: Optional[
            ImaginationMemory
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or ImaginationAgentConfig()
        )

        self._validate_config()

        cfg = self.config

        self.dynamics_model = dynamics_model

        self.reward_fn = reward_fn
        self.risk_fn = risk_fn

        trajectory_config = (
            TrajectoryGeneratorConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                horizon=cfg.planning_horizon,
                num_candidates=cfg.num_candidates,
                method=cfg.planning_method,
                cem_iterations=cfg.cem_iterations,
                elite_fraction=cfg.elite_fraction,
                stochastic=cfg.stochastic,
                temperature=cfg.temperature,
                action_min=cfg.action_min,
                action_max=cfg.action_max,
                discount_factor=cfg.discount_factor,
                uncertainty_penalty=(
                    cfg.uncertainty_penalty
                ),
                risk_penalty=cfg.risk_penalty,
                goal_weight=cfg.goal_weight,
            )
        )

        self.trajectory_generator = (
            TrajectoryGenerator(
                dynamics_model=dynamics_model,
                config=trajectory_config,
                reward_fn=reward_fn,
                risk_fn=risk_fn,
            )
        )

        engine_config = ImaginationEngineConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.planning_horizon,
            num_futures=cfg.num_candidates,
            stochastic=cfg.stochastic,
            temperature=cfg.temperature,
            discount_factor=cfg.discount_factor,
            risk_penalty=cfg.risk_penalty,
            uncertainty_penalty=(
                cfg.uncertainty_penalty
            ),
            goal_weight=cfg.goal_weight,
            enable_future_generation=True,
            enable_branching=cfg.enable_branching,
            enable_counterfactuals=(
                cfg.enable_counterfactuals
            ),
        )

        self.imagination_engine = (
            ImaginationEngine(
                dynamics_model=dynamics_model,
                config=engine_config,
                reward_fn=reward_fn,
                risk_fn=risk_fn,
            )
        )

        self.imagination_memory = (
            imagination_memory
            if imagination_memory is not None
            else ImaginationMemory(
                ImaginationMemoryConfig(
                    latent_dim=cfg.latent_dim,
                    action_dim=cfg.action_dim,
                    capacity=cfg.memory_capacity,
                )
            )
        )

        self.total_plans = 0
        self.total_decisions = 0

        self._plan_scores: List[float] = []

    def _validate_config(self) -> None:
        cfg = self.config

        for name in (
            "latent_dim",
            "action_dim",
            "planning_horizon",
            "num_candidates",
            "cem_iterations",
            "memory_capacity",
            "max_stored_candidates",
        ):
            if getattr(cfg, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if cfg.planning_method not in (
            "cem",
            "random",
        ):
            raise ValueError(
                "planning_method must be "
                "'cem' or 'random'"
            )

        if not 0 < cfg.elite_fraction <= 1:
            raise ValueError(
                "elite_fraction must be in (0,1]"
            )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be < action_max"
            )

        if not 0 <= cfg.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in [0,1]"
            )

        if cfg.temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        for name in (
            "uncertainty_penalty",
            "risk_penalty",
            "goal_weight",
        ):
            value = getattr(cfg, name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite "
                    "and nonnegative"
                )

    def _validate_state(
        self,
        state: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> None:
        if (
            state.ndim != 2
            or state.shape[-1]
            != self.config.latent_dim
        ):
            raise ValueError(
                "state must be [B,latent_dim]"
            )

        if not torch.isfinite(state).all().item():
            raise ValueError(
                "state contains NaN or Inf"
            )

        if goal_state is not None:
            if goal_state.shape != state.shape:
                raise ValueError(
                    "goal_state must match state"
                )

            if goal_state.device != state.device:
                raise ValueError(
                    "goal_state device mismatch"
                )

            if goal_state.dtype != state.dtype:
                raise ValueError(
                    "goal_state dtype mismatch"
                )

    def _store_plan(
        self,
        output: TrajectoryGenerationOutput,
    ) -> None:
        """
        Store selected imagined trajectories.

        Each trajectory is stored independently
        in imagination memory.
        """

        if not self.config.store_imaginations:
            return

        batch_size = output.actions.shape[0]

        candidate_count = output.actions.shape[1]

        top_k = min(
            self.config.max_stored_candidates,
            candidate_count,
        )

        for batch_index in range(batch_size):

            candidate_scores = (
                output.scores[batch_index]
            )

            top_indices = torch.topk(
                candidate_scores,
                k=top_k,
            ).indices

            for candidate_index in top_indices.tolist():

                states = (
                    output.trajectories[
                        batch_index,
                        candidate_index,
                    ]
                )

                actions = (
                    output.actions[
                        batch_index,
                        candidate_index,
                    ]
                )

                rewards = (
                    output.rewards[
                        batch_index,
                        candidate_index,
                    ]
                )

                uncertainties = (
                    output.uncertainties[
                        batch_index,
                        candidate_index,
                    ]
                )

                score = float(
                    candidate_scores[
                        candidate_index
                    ].detach()
                )

                self.imagination_memory.add(
                    states=states.detach(),
                    actions=actions.detach(),
                    rewards=rewards.detach(),
                    uncertainties=(
                        uncertainties.detach()
                    ),
                    score=score,
                    source="imagination_agent",
                    metadata={
                        "batch_index": batch_index,
                        "candidate_index": (
                            candidate_index
                        ),
                    },
                )

    @torch.no_grad()
    def imagine(
        self,
        state: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> ImaginedPlan:
        """
        Generate and evaluate imagined futures.

        Returns:
            Best action sequence
            Best predicted trajectory
            Candidate scores
            Candidate trajectories

        Shapes:
            actions: [B,H,A]
            trajectory: [B,H+1,D]
            score: [B]
        """

        self._validate_state(
            state,
            goal_state,
        )

        output = self.trajectory_generator.generate(
            initial_state=state,
            goal_state=goal_state,
        )

        self.total_plans += 1

        best_scores = (
            output.best_scores.detach()
        )

        self._plan_scores.extend(
            best_scores.cpu().tolist()
        )

        self._store_plan(output)

        return ImaginedPlan(
            actions=output.best_actions,
            trajectory=output.best_trajectories,
            score=output.best_scores,
            candidate_scores=output.scores,
            candidate_actions=output.actions,
            candidate_trajectories=(
                output.trajectories
            ),
            metadata={
                "planning_method": (
                    self.config.planning_method
                ),
                "horizon": (
                    self.config.planning_horizon
                ),
                "num_candidates": (
                    self.config.num_candidates
                ),
            },
        )

    @torch.no_grad()
    def select_action(
        self,
        state: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> ImaginationDecision:
        """
        Receding-horizon action selection.

        Only the first action of the best
        imagined plan is returned.

        Replan after receiving the next
        real observation.
        """

        plan = self.imagine(
            state=state,
            goal_state=goal_state,
        )

        first_action = (
            plan.actions[:, 0, :]
        )

        predicted_next_state = (
            plan.trajectory[:, 1, :]
        )

        predicted_final_state = (
            plan.trajectory[:, -1, :]
        )

        self.total_decisions += 1

        return ImaginationDecision(
            action=first_action,
            plan=plan,
            predicted_next_state=(
                predicted_next_state
            ),
            predicted_final_state=(
                predicted_final_state
            ),
            score=plan.score,
        )

    @torch.no_grad()
    def compare_action_sequences(
        self,
        state: Tensor,
        candidate_actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compare externally supplied action plans.

        candidate_actions:
            [B,K,H,A]

        Returns:
            scores [B,K]
            best_indices [B]
            best_actions [B,H,A]
            best_trajectories [B,H+1,D]
        """

        self._validate_state(
            state,
            goal_state,
        )

        cfg = self.config

        if (
            candidate_actions.ndim != 4
            or candidate_actions.shape[0]
            != state.shape[0]
            or candidate_actions.shape[2]
            != cfg.planning_horizon
            or candidate_actions.shape[3]
            != cfg.action_dim
        ):
            raise ValueError(
                "candidate_actions must be "
                "[B,K,H,A]"
            )

        rollout, scores = (
            self.trajectory_generator
            .evaluate_candidates(
                initial_state=state,
                candidate_actions=candidate_actions,
                goal_state=goal_state,
            )
        )

        best_indices = scores.argmax(
            dim=1
        )

        batch_indices = torch.arange(
            state.shape[0],
            device=state.device,
        )

        return {
            "scores": scores,
            "best_indices": best_indices,
            "best_actions": (
                candidate_actions[
                    batch_indices,
                    best_indices,
                ]
            ),
            "best_trajectories": (
                rollout.states[
                    batch_indices,
                    best_indices,
                ]
            ),
        }

    @torch.no_grad()
    def imagine_futures(
        self,
        state: Tensor,
        actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> Any:
        """
        Use the full imagination engine
        to generate multiple possible futures.

        actions:
            [B,H,A] or [B,K,H,A]
        """

        self._validate_state(
            state,
            goal_state,
        )

        return (
            self.imagination_engine
            .imagine_futures(
                initial_state=state,
                actions=actions,
                goal_state=goal_state,
            )
        )

    @torch.no_grad()
    def imagine_branches(
        self,
        state: Tensor,
    ) -> Any:
        """
        Generate alternative future branches.
        """

        if not self.config.enable_branching:
            raise RuntimeError(
                "Branching is disabled"
            )

        self._validate_state(state)

        return (
            self.imagination_engine
            .imagine_branches(
                initial_state=state,
            )
        )

    @torch.no_grad()
    def retrieve_similar_imaginations(
        self,
        state: Tensor,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Retrieve imagined experiences similar
        to a single latent state.

        state:
            [D] or [1,D]
        """

        if state.ndim == 2:
            if state.shape[0] != 1:
                raise ValueError(
                    "Memory search supports "
                    "one query state at a time"
                )

            state = state.squeeze(0)

        if (
            state.ndim != 1
            or state.shape[0]
            != self.config.latent_dim
        ):
            raise ValueError(
                "state must be [latent_dim]"
            )

        return self.imagination_memory.search(
            query_state=state,
            top_k=top_k,
        )

    def reset_statistics(self) -> None:
        """
        Reset runtime planning statistics.

        Does not clear imagination memory.
        """

        self.total_plans = 0
        self.total_decisions = 0
        self._plan_scores.clear()

    def statistics(
        self,
    ) -> ImaginationAgentStatistics:
        """
        Summarize imagination agent activity.
        """

        if self._plan_scores:
            mean_score = (
                sum(self._plan_scores)
                / len(self._plan_scores)
            )

            best_score = max(
                self._plan_scores
            )
        else:
            mean_score = 0.0
            best_score = 0.0

        memory_stats = (
            self.imagination_memory.statistics()
        )

        stored_episodes = memory_stats.get(
            "size",
            0,
        )

        return ImaginationAgentStatistics(
            total_plans=self.total_plans,
            total_decisions=self.total_decisions,
            stored_episodes=stored_episodes,
            mean_plan_score=mean_score,
            best_plan_score=best_score,
        )


if __name__ == "__main__":

    from neuroworld.dynamics.transition_model import (
        TransitionModel,
        TransitionModelConfig,
    )

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    latent_dim = 128
    action_dim = 32

    dynamics = TransitionModel(
        TransitionModelConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            backend="stochastic",
        )
    ).to(device)

    agent = ImaginationAgent(
        dynamics_model=dynamics,
        config=ImaginationAgentConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            planning_horizon=10,
            num_candidates=64,
            planning_method="cem",
            cem_iterations=5,
            store_imaginations=True,
        ),
    ).to(device)

    batch_size = 2

    current_state = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    goal_state = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    # Generate imagined future trajectories.
    plan = agent.imagine(
        state=current_state,
        goal_state=goal_state,
    )

    print(
        "Best actions:",
        plan.actions.shape,
    )

    print(
        "Best trajectory:",
        plan.trajectory.shape,
    )

    print(
        "Plan scores:",
        plan.score,
    )

    # Select the first action.
    decision = agent.select_action(
        state=current_state,
        goal_state=goal_state,
    )

    print(
        "Selected action:",
        decision.action.shape,
    )

    print(
        "Predicted next state:",
        decision.predicted_next_state.shape,
    )

    # Compare candidate action sequences.
    candidate_actions = torch.rand(
        batch_size,
        8,
        10,
        action_dim,
        device=device,
    ) * 2.0 - 1.0

    comparison = (
        agent.compare_action_sequences(
            state=current_state,
            candidate_actions=candidate_actions,
            goal_state=goal_state,
        )
    )

    print(
        "Candidate scores:",
        comparison["scores"].shape,
    )

    # Retrieve similar imagined experiences.
    memories = (
        agent.retrieve_similar_imaginations(
            state=current_state[0],
            top_k=3,
        )
    )

    print(
        "Retrieved memories:",
        len(memories),
    )

    print(
        "Agent statistics:",
        agent.statistics(),
    )
