
"""
NeuroWorld AI
File: neuroworld/agents/planning_agent.py

Goal-Conditioned Autonomous Planning Agent

Features:
- Model-based latent-space planning
- Cross-Entropy Method (CEM)
- Random-shooting optimization
- Goal-conditioned action planning
- Risk-aware trajectory selection
- Predictive uncertainty penalties
- Bounded continuous actions
- Action smoothness regularization
- Candidate action-sequence comparison
- Receding-horizon planning
- Warm-start action sequences
- Plan validation and rejection
- Planning history and statistics
- Checkpoint-compatible configuration
- CPU/CUDA support

Dependencies:
    pip install torch

Compatible with:
    neuroworld.dynamics.transition_model
    neuroworld.imagination.trajectory_generator

Important:
    This agent generates simulated plans.
    It does not execute physical actions.

    Independent safety validation is required
    before commanding real robots.
"""

from __future__ import annotations

import json
import math
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from neuroworld.imagination.trajectory_generator import (
    TrajectoryGenerator,
    TrajectoryGeneratorConfig,
)


RewardFunction = Callable[
    [Tensor, Tensor, Tensor],
    Tensor,
]

RiskFunction = Callable[
    [Tensor, Tensor, Tensor],
    Tensor,
]

ActionValidator = Callable[
    [Tensor],
    Tensor,
]


@dataclass
class PlanningAgentConfig:
    latent_dim: int = 128
    action_dim: int = 32

    planning_horizon: int = 15
    num_candidates: int = 128

    planning_method: str = "cem"

    cem_iterations: int = 6
    elite_fraction: float = 0.15
    cem_smoothing: float = 0.2

    initial_action_std: float = 0.5
    min_action_std: float = 0.05

    action_min: float = -1.0
    action_max: float = 1.0

    stochastic: bool = False
    temperature: float = 0.5

    discount_factor: float = 0.99

    goal_weight: float = 1.0
    risk_penalty: float = 1.0
    uncertainty_penalty: float = 0.2

    action_penalty: float = 0.01
    smoothness_penalty: float = 0.02

    enable_warm_start: bool = True

    minimum_acceptable_score: Optional[
        float
    ] = None

    history_capacity: int = 1000


@dataclass
class PlanningGoal:
    target_state: Tensor

    tolerance: float = 0.5

    description: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class PlanningResult:
    actions: Tensor
    trajectory: Tensor

    scores: Tensor
    best_scores: Tensor
    best_indices: Tensor

    candidate_actions: Tensor
    candidate_trajectories: Tensor

    goal_distances: Optional[Tensor]

    accepted: Tensor

    planning_time_seconds: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class PlanningDecision:
    action: Tensor

    predicted_next_state: Tensor
    predicted_final_state: Tensor

    expected_score: Tensor
    accepted: Tensor

    plan: PlanningResult


@dataclass
class PlanningHistoryEntry:
    timestamp: float
    mean_score: float
    best_score: float
    acceptance_rate: float
    planning_time_seconds: float


class PlanningAgent(nn.Module):
    """
    Goal-conditioned model-based planning agent.

    Planning pipeline:

        Current latent state
                |
                v
        Candidate action sequences
                |
                v
        World-model rollout
                |
                v
        Reward + goal + risk scoring
                |
                v
        CEM / random-shooting optimization
                |
                v
        Best trajectory
                |
                v
        Action validation
                |
                v
        First-action decision

    Receding-horizon control:
        Only the first action is selected.

        After receiving the next observation,
        the agent should plan again.
    """

    def __init__(
        self,
        dynamics_model: nn.Module,
        config: Optional[
            PlanningAgentConfig
        ] = None,
        reward_fn: Optional[
            RewardFunction
        ] = None,
        risk_fn: Optional[
            RiskFunction
        ] = None,
        action_validator: Optional[
            ActionValidator
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or PlanningAgentConfig()
        )

        self._validate_config()

        self.dynamics_model = dynamics_model

        self.reward_fn = reward_fn
        self.risk_fn = risk_fn

        self.action_validator = (
            action_validator
        )

        cfg = self.config

        trajectory_config = (
            TrajectoryGeneratorConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                horizon=cfg.planning_horizon,
                num_candidates=cfg.num_candidates,
                method=cfg.planning_method,
                cem_iterations=cfg.cem_iterations,
                elite_fraction=cfg.elite_fraction,
                cem_smoothing=cfg.cem_smoothing,
                initial_action_std=(
                    cfg.initial_action_std
                ),
                min_action_std=cfg.min_action_std,
                action_min=cfg.action_min,
                action_max=cfg.action_max,
                stochastic=cfg.stochastic,
                temperature=cfg.temperature,
                discount_factor=cfg.discount_factor,
                goal_weight=cfg.goal_weight,
                risk_penalty=cfg.risk_penalty,
                uncertainty_penalty=(
                    cfg.uncertainty_penalty
                ),
                action_penalty=cfg.action_penalty,
                smoothness_penalty=(
                    cfg.smoothness_penalty
                ),
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

        self._previous_actions: Optional[
            Tensor
        ] = None

        self.history: List[
            PlanningHistoryEntry
        ] = []

        self.total_plans = 0
        self.total_decisions = 0

    def _validate_config(self) -> None:
        cfg = self.config

        for name in (
            "latent_dim",
            "action_dim",
            "planning_horizon",
            "num_candidates",
            "cem_iterations",
            "history_capacity",
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

        if not 0 <= cfg.cem_smoothing <= 1:
            raise ValueError(
                "cem_smoothing must be in [0,1]"
            )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be < action_max"
            )

        if cfg.initial_action_std <= 0:
            raise ValueError(
                "initial_action_std must be positive"
            )

        if cfg.min_action_std <= 0:
            raise ValueError(
                "min_action_std must be positive"
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
            "goal_weight",
            "risk_penalty",
            "uncertainty_penalty",
            "action_penalty",
            "smoothness_penalty",
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
    ) -> None:
        if (
            state.ndim != 2
            or state.shape[-1]
            != self.config.latent_dim
        ):
            raise ValueError(
                "state must be [B,latent_dim]"
            )

        if not torch.is_floating_point(state):
            raise TypeError(
                "state must be floating point"
            )

        if not torch.isfinite(state).all().item():
            raise ValueError(
                "state contains NaN or Inf"
            )

    def _validate_goal(
        self,
        state: Tensor,
        goal: Optional[PlanningGoal],
    ) -> Optional[Tensor]:
        if goal is None:
            return None

        target = goal.target_state

        if target.shape != state.shape:
            raise ValueError(
                "goal target must match state shape"
            )

        if target.device != state.device:
            raise ValueError(
                "goal device mismatch"
            )

        if target.dtype != state.dtype:
            raise ValueError(
                "goal dtype mismatch"
            )

        if not torch.isfinite(target).all().item():
            raise ValueError(
                "goal contains NaN or Inf"
            )

        if goal.tolerance < 0:
            raise ValueError(
                "goal tolerance must be nonnegative"
            )

        return target

    def _validate_action_sequence(
        self,
        actions: Tensor,
        batch_size: int,
    ) -> None:
        cfg = self.config

        if actions.shape != (
            batch_size,
            cfg.planning_horizon,
            cfg.action_dim,
        ):
            raise ValueError(
                "actions must be [B,H,A]"
            )

        if not torch.isfinite(actions).all().item():
            raise ValueError(
                "actions contain NaN or Inf"
            )

    def _validate_candidate_actions(
        self,
        actions: Tensor,
        batch_size: int,
    ) -> None:
        cfg = self.config

        if (
            actions.ndim != 4
            or actions.shape[0] != batch_size
            or actions.shape[2]
            != cfg.planning_horizon
            or actions.shape[3]
            != cfg.action_dim
        ):
            raise ValueError(
                "candidate actions must be [B,K,H,A]"
            )

        if actions.shape[1] < 1:
            raise ValueError(
                "At least one candidate required"
            )

        if not torch.isfinite(actions).all().item():
            raise ValueError(
                "candidate actions contain NaN or Inf"
            )

    def _validate_selected_actions(
        self,
        actions: Tensor,
    ) -> Tensor:
        """
        Apply optional action validation.

        The validator must return a boolean
        tensor [B], where True means accepted.
        """

        batch_size = actions.shape[0]

        within_bounds = (
            (actions >= self.config.action_min)
            & (actions <= self.config.action_max)
        ).all(dim=(-1, -2))

        finite = torch.isfinite(
            actions
        ).all(dim=(-1, -2))

        accepted = within_bounds & finite

        if self.action_validator is not None:
            custom = self.action_validator(
                actions
            )

            if custom.shape != (
                batch_size,
            ):
                raise ValueError(
                    "action_validator must "
                    "return [B]"
                )

            if custom.dtype != torch.bool:
                raise TypeError(
                    "action_validator must "
                    "return boolean values"
                )

            accepted = accepted & custom

        return accepted

    def _warm_start_candidates(
        self,
        state: Tensor,
    ) -> Optional[Tensor]:
        """
        Shift the previous action plan
        one step forward.

        Returns:
            [B,H,A] or None

        Used as an additional candidate,
        not as a replacement for CEM.
        """

        if not self.config.enable_warm_start:
            return None

        previous = self._previous_actions

        if previous is None:
            return None

        expected_shape = (
            state.shape[0],
            self.config.planning_horizon,
            self.config.action_dim,
        )

        if previous.shape != expected_shape:
            return None

        shifted = torch.cat(
            [
                previous[:, 1:, :],
                previous[:, -1:, :],
            ],
            dim=1,
        )

        return shifted.to(
            device=state.device,
            dtype=state.dtype,
        ).clamp(
            self.config.action_min,
            self.config.action_max,
        )

    @torch.no_grad()
    def _generate_trajectories(
        self,
        state: Tensor,
        goal_state: Optional[Tensor],
    ) -> Any:
        """
        Run trajectory optimization.

        The underlying TrajectoryGenerator
        implements CEM or random shooting.
        """

        return self.trajectory_generator.generate(
            initial_state=state,
            goal_state=goal_state,
        )

    @torch.no_grad()
    def _evaluate_candidates(
        self,
        state: Tensor,
        candidate_actions: Tensor,
        goal_state: Optional[Tensor],
    ) -> tuple[Any, Tensor]:
        return (
            self.trajectory_generator
            .evaluate_candidates(
                initial_state=state,
                candidate_actions=candidate_actions,
                goal_state=goal_state,
            )
        )

    @torch.no_grad()
    def plan(
        self,
        state: Tensor,
        goal: Optional[PlanningGoal] = None,
    ) -> PlanningResult:
        """
        Generate an optimized action plan.

        Shapes:
            actions: [B,H,A]
            trajectory: [B,H+1,D]
            scores: [B,K]
            best_scores: [B]
            accepted: [B]

        Rejected plans retain their diagnostic
        trajectories but must not be executed.
        """

        self._validate_state(state)

        goal_state = self._validate_goal(
            state,
            goal,
        )

        start_time = time.perf_counter()

        output = self._generate_trajectories(
            state,
            goal_state,
        )

        candidate_actions = output.actions
        candidate_trajectories = (
            output.trajectories
        )

        scores = output.scores

        # Evaluate shifted previous plan as
        # an additional candidate.
        warm_start = (
            self._warm_start_candidates(
                state
            )
        )

        if warm_start is not None:
            warm_actions = (
                warm_start.unsqueeze(1)
            )

            warm_rollout, warm_scores = (
                self._evaluate_candidates(
                    state,
                    warm_actions,
                    goal_state,
                )
            )

            candidate_actions = torch.cat(
                [
                    candidate_actions,
                    warm_actions,
                ],
                dim=1,
            )

            candidate_trajectories = torch.cat(
                [
                    candidate_trajectories,
                    warm_rollout.states,
                ],
                dim=1,
            )

            scores = torch.cat(
                [
                    scores,
                    warm_scores,
                ],
                dim=1,
            )

        best_indices = scores.argmax(
            dim=1
        )

        batch_indices = torch.arange(
            state.shape[0],
            device=state.device,
        )

        best_actions = candidate_actions[
            batch_indices,
            best_indices,
        ]

        best_trajectories = (
            candidate_trajectories[
                batch_indices,
                best_indices,
            ]
        )

        best_scores = scores[
            batch_indices,
            best_indices,
        ]

        accepted = (
            self._validate_selected_actions(
                best_actions
            )
        )

        threshold = (
            self.config.minimum_acceptable_score
        )

        if threshold is not None:
            accepted = (
                accepted
                & (best_scores >= threshold)
            )

        goal_distances = None

        if goal_state is not None:
            goal_distances = (
                torch.linalg.vector_norm(
                    best_trajectories[:, -1, :]
                    - goal_state,
                    dim=-1,
                )
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        self.total_plans += 1

        mean_score = float(
            best_scores.mean()
        )

        best_score = float(
            best_scores.max()
        )

        acceptance_rate = float(
            accepted.float().mean()
        )

        self.history.append(
            PlanningHistoryEntry(
                timestamp=time.time(),
                mean_score=mean_score,
                best_score=best_score,
                acceptance_rate=(
                    acceptance_rate
                ),
                planning_time_seconds=elapsed,
            )
        )

        if (
            len(self.history)
            > self.config.history_capacity
        ):
            self.history = self.history[
                -self.config.history_capacity:
            ]

        # Keep previous plan for next
        # receding-horizon optimization.
        self._previous_actions = (
            best_actions.detach().clone()
        )

        return PlanningResult(
            actions=best_actions,
            trajectory=best_trajectories,
            scores=scores,
            best_scores=best_scores,
            best_indices=best_indices,
            candidate_actions=candidate_actions,
            candidate_trajectories=(
                candidate_trajectories
            ),
            goal_distances=goal_distances,
            accepted=accepted,
            planning_time_seconds=elapsed,
            metadata={
                "planning_method": (
                    self.config.planning_method
                ),
                "planning_horizon": (
                    self.config.planning_horizon
                ),
                "num_candidates": (
                    candidate_actions.shape[1]
                ),
                "warm_start_used": (
                    warm_start is not None
                ),
            },
        )

    @torch.no_grad()
    def select_action(
        self,
        state: Tensor,
        goal: Optional[PlanningGoal] = None,
    ) -> PlanningDecision:
        """
        Receding-horizon planning decision.

        Returns the first action of the
        best simulated trajectory.

        If the plan is rejected, the returned
        action is zeroed.

        A zero action is not universally safe;
        physical execution still requires
        independent safety validation.
        """

        result = self.plan(
            state,
            goal,
        )

        action = (
            result.actions[:, 0, :]
        )

        safe_action = torch.where(
            result.accepted.unsqueeze(-1),
            action,
            torch.zeros_like(action),
        )

        self.total_decisions += 1

        return PlanningDecision(
            action=safe_action,
            predicted_next_state=(
                result.trajectory[:, 1, :]
            ),
            predicted_final_state=(
                result.trajectory[:, -1, :]
            ),
            expected_score=(
                result.best_scores
            ),
            accepted=result.accepted,
            plan=result,
        )

    @torch.no_grad()
    def compare_plans(
        self,
        state: Tensor,
        candidate_actions: Tensor,
        goal: Optional[PlanningGoal] = None,
    ) -> Dict[str, Tensor]:
        """
        Compare externally supplied plans.

        candidate_actions:
            [B,K,H,A]

        Returns:
            scores
            best_indices
            best_actions
            best_trajectories
        """

        self._validate_state(state)

        self._validate_candidate_actions(
            candidate_actions,
            state.shape[0],
        )

        goal_state = self._validate_goal(
            state,
            goal,
        )

        rollout, scores = (
            self._evaluate_candidates(
                state,
                candidate_actions,
                goal_state,
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
    def evaluate_goal_reaching(
        self,
        plan: PlanningResult,
        goal: PlanningGoal,
    ) -> Dict[str, Tensor]:
        """
        Evaluate predicted goal proximity.

        Success is based on predicted
        latent-state distance.

        This does not prove the physical
        goal will be reached.
        """

        final_state = (
            plan.trajectory[:, -1, :]
        )

        target = goal.target_state

        distances = (
            torch.linalg.vector_norm(
                final_state - target,
                dim=-1,
            )
        )

        predicted_success = (
            distances <= goal.tolerance
        )

        return {
            "goal_distances": distances,
            "predicted_success": (
                predicted_success
            ),
            "predicted_success_rate": (
                predicted_success.float().mean()
            ),
        }

    def reset(self) -> None:
        """
        Reset planning history and warm start.
        """

        self._previous_actions = None
        self.history.clear()

        self.total_plans = 0
        self.total_decisions = 0

    def statistics(
        self,
    ) -> Dict[str, float]:
        """
        Return aggregate planning statistics.
        """

        if not self.history:
            return {
                "total_plans": float(
                    self.total_plans
                ),
                "total_decisions": float(
                    self.total_decisions
                ),
                "mean_plan_score": 0.0,
                "best_plan_score": 0.0,
                "mean_acceptance_rate": 0.0,
                "mean_planning_time": 0.0,
            }

        count = len(self.history)

        return {
            "total_plans": float(
                self.total_plans
            ),
            "total_decisions": float(
                self.total_decisions
            ),
            "mean_plan_score": sum(
                item.mean_score
                for item in self.history
            ) / count,
            "best_plan_score": max(
                item.best_score
                for item in self.history
            ),
            "mean_acceptance_rate": sum(
                item.acceptance_rate
                for item in self.history
            ) / count,
            "mean_planning_time": sum(
                item.planning_time_seconds
                for item in self.history
            ) / count,
        }

    def export_history(
        self,
        path: str | Path,
    ) -> None:
        """
        Export planning history to JSON.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "config": asdict(self.config),
            "statistics": self.statistics(),
            "history": [
                asdict(entry)
                for entry in self.history
            ],
        }

        path.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
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

    world_model = TransitionModel(
        TransitionModelConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            backend="stochastic",
        )
    ).to(device)

    agent = PlanningAgent(
        dynamics_model=world_model,
        config=PlanningAgentConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            planning_horizon=10,
            num_candidates=64,
            planning_method="cem",
            cem_iterations=5,
            goal_weight=1.0,
            risk_penalty=0.5,
            uncertainty_penalty=0.1,
        ),
    ).to(device)

    batch_size = 2

    current_state = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    target_state = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    goal = PlanningGoal(
        target_state=target_state,
        tolerance=1.0,
        description="Reach target latent state",
    )

    # -----------------------------------------
    # Generate a plan
    # -----------------------------------------

    plan = agent.plan(
        state=current_state,
        goal=goal,
    )

    print(
        "Best action sequence:",
        plan.actions.shape,
    )

    print(
        "Predicted trajectory:",
        plan.trajectory.shape,
    )

    print(
        "Best scores:",
        plan.best_scores,
    )

    print(
        "Accepted:",
        plan.accepted,
    )

    # -----------------------------------------
    # Receding-horizon action decision
    # -----------------------------------------

    decision = agent.select_action(
        state=current_state,
        goal=goal,
    )

    print(
        "Selected action:",
        decision.action.shape,
    )

    print(
        "Predicted next state:",
        decision.predicted_next_state.shape,
    )

    # -----------------------------------------
    # Evaluate goal reaching
    # -----------------------------------------

    evaluation = (
        agent.evaluate_goal_reaching(
            decision.plan,
            goal,
        )
    )

    print(
        "Predicted goal success:",
        evaluation["predicted_success"],
    )

    # -----------------------------------------
    # Compare candidate plans
    # -----------------------------------------

    candidate_actions = (
        torch.rand(
            batch_size,
            8,
            10,
            action_dim,
            device=device,
        ) * 2.0 - 1.0
    )

    comparison = agent.compare_plans(
        state=current_state,
        candidate_actions=candidate_actions,
        goal=goal,
    )

    print(
        "Candidate scores:",
        comparison["scores"].shape,
    )

    # -----------------------------------------
    # Statistics and export
    # -----------------------------------------

    print(
        "Planning statistics:",
        agent.statistics(),
    )

    agent.export_history(
        "artifacts/planning_history.json"
    )
