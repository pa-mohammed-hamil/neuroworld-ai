
```python
"""
NeuroWorld AI
File: neuroworld/agents/world_agent.py

Autonomous World Intelligence Agent

Features:
- Central coordination of NeuroWorld agents
- Goal-conditioned decision-making
- Exploration versus exploitation
- World-model future-state prediction
- Planning and imagination integration
- Curiosity-based exploration integration
- Evaluation-agent integration
- Decision confidence estimation
- Predictive uncertainty monitoring
- Action validation and rejection
- Human approval support
- Decision history and statistics
- World-state observation tracking
- JSON report export
- CPU/CUDA support

Dependencies:
    pip install torch

Integrated modules:
    neuroworld.agents.planning_agent
    neuroworld.agents.imagination_agent
    neuroworld.agents.exploration_agent
    neuroworld.agents.curiosity_agent
    neuroworld.agents.evaluation_agent
    neuroworld.dynamics.transition_model

Notes:
    This module coordinates decision proposals.
    It does not execute actions on physical hardware.

    Safety checks, permissions, emergency stops
    and human approval must be enforced by the
    actual execution system.
"""

from __future__ import annotations

import json
import math
import time

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from neuroworld.agents.planning_agent import (
    PlanningAgent,
    PlanningAgentConfig,
    PlanningGoal,
)

from neuroworld.agents.imagination_agent import (
    ImaginationAgent,
    ImaginationAgentConfig,
)

from neuroworld.agents.exploration_agent import (
    ExplorationAgent,
    ExplorationAgentConfig,
)

from neuroworld.agents.evaluation_agent import (
    EvaluationAgent,
    EvaluationAgentConfig,
    EvaluationBatch,
)


class WorldAgentMode(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    IMAGINING = "imagining"
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    EVALUATING = "evaluating"
    WAITING_APPROVAL = "waiting_approval"
    BLOCKED = "blocked"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class WorldAgentConfig:
    latent_dim: int = 128
    action_dim: int = 32

    planning_horizon: int = 10
    num_candidates: int = 64

    exploration_probability: float = 0.15

    action_min: float = -1.0
    action_max: float = 1.0

    goal_tolerance: float = 0.5

    max_predictive_uncertainty: Optional[
        float
    ] = None

    enable_planning: bool = True
    enable_imagination: bool = True
    enable_exploration: bool = True
    enable_evaluation: bool = True

    require_human_approval: bool = False

    history_capacity: int = 1000

    deterministic_prediction: bool = True


@dataclass
class WorldObservation:
    latent_state: Tensor

    timestamp: float = field(
        default_factory=time.time
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class WorldGoal:
    target_state: Tensor
    tolerance: float = 0.5
    description: str = ""


@dataclass
class WorldPrediction:
    next_state: Tensor

    uncertainty: Optional[Tensor]

    mean: Tensor

    logvar: Optional[Tensor]


@dataclass
class WorldDecision:
    action: Tensor

    mode: WorldAgentMode

    accepted: Tensor

    predicted_next_state: Optional[Tensor]

    predicted_uncertainty: Optional[Tensor]

    score: Optional[Tensor]

    requires_approval: bool

    reason: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class WorldDecisionRecord:
    timestamp: float

    mode: str

    accepted_count: int
    rejected_count: int

    mean_score: Optional[float]

    mean_uncertainty: Optional[float]

    reason: str


ActionValidator = Callable[
    [Tensor],
    Tensor,
]


class WorldAgent(nn.Module):
    """
    Central NeuroWorld autonomous decision agent.

    Architecture:

        Observation
             |
             v
        Latent World State
             |
             v
        World Model
             |
             v
        Decision Coordinator
             |
        +----+----+
        |         |
        v         v
    Planning   Exploration
        |         |
        v         v
    Imagination  Curiosity
        |         |
        +----+----+
             |
             v
        Candidate Action
             |
             v
        Validation / Approval
             |
             v
        Proposed Decision

    The agent uses receding-horizon planning
    when goal-directed planning is enabled.

    No hardware execution is performed.
    """

    def __init__(
        self,
        world_model: nn.Module,
        config: Optional[
            WorldAgentConfig
        ] = None,
        planning_agent: Optional[
            PlanningAgent
        ] = None,
        imagination_agent: Optional[
            ImaginationAgent
        ] = None,
        exploration_agent: Optional[
            ExplorationAgent
        ] = None,
        evaluation_agent: Optional[
            EvaluationAgent
        ] = None,
        action_validator: Optional[
            ActionValidator
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or WorldAgentConfig()
        )

        self._validate_config()

        cfg = self.config

        self.world_model = world_model

        self.action_validator = (
            action_validator
        )

        if cfg.enable_planning:
            self.planning_agent = (
                planning_agent
                if planning_agent is not None
                else PlanningAgent(
                    dynamics_model=world_model,
                    config=PlanningAgentConfig(
                        latent_dim=cfg.latent_dim,
                        action_dim=cfg.action_dim,
                        planning_horizon=(
                            cfg.planning_horizon
                        ),
                        num_candidates=(
                            cfg.num_candidates
                        ),
                        action_min=cfg.action_min,
                        action_max=cfg.action_max,
                    ),
                )
            )
        else:
            self.planning_agent = None

        if cfg.enable_imagination:
            self.imagination_agent = (
                imagination_agent
                if imagination_agent is not None
                else ImaginationAgent(
                    dynamics_model=world_model,
                    config=ImaginationAgentConfig(
                        latent_dim=cfg.latent_dim,
                        action_dim=cfg.action_dim,
                        planning_horizon=(
                            cfg.planning_horizon
                        ),
                        num_candidates=(
                            cfg.num_candidates
                        ),
                        action_min=cfg.action_min,
                        action_max=cfg.action_max,
                    ),
                )
            )
        else:
            self.imagination_agent = None

        if cfg.enable_exploration:
            self.exploration_agent = (
                exploration_agent
                if exploration_agent is not None
                else ExplorationAgent(
                    config=ExplorationAgentConfig(
                        latent_dim=cfg.latent_dim,
                        action_dim=cfg.action_dim,
                        action_space="continuous",
                        exploration_strategy="gaussian",
                        action_min=cfg.action_min,
                        action_max=cfg.action_max,
                    ),
                    world_model=world_model,
                )
            )
        else:
            self.exploration_agent = None

        if cfg.enable_evaluation:
            self.evaluation_agent = (
                evaluation_agent
                if evaluation_agent is not None
                else EvaluationAgent(
                    EvaluationAgentConfig(
                        latent_dim=cfg.latent_dim,
                        action_dim=cfg.action_dim,
                        goal_tolerance=(
                            cfg.goal_tolerance
                        ),
                    )
                )
            )
        else:
            self.evaluation_agent = None

        self.mode = WorldAgentMode.IDLE

        self.current_observation: Optional[
            WorldObservation
        ] = None

        self.current_goal: Optional[
            WorldGoal
        ] = None

        self.decision_history: List[
            WorldDecisionRecord
        ] = []

        self.total_decisions = 0
        self.total_accepted = 0
        self.total_rejected = 0

        self.emergency_stop_active = False

    def _validate_config(self) -> None:
        cfg = self.config

        for name in (
            "latent_dim",
            "action_dim",
            "planning_horizon",
            "num_candidates",
            "history_capacity",
        ):
            if getattr(cfg, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if not (
            0 <= cfg.exploration_probability <= 1
        ):
            raise ValueError(
                "exploration_probability "
                "must be in [0,1]"
            )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be < action_max"
            )

        if cfg.goal_tolerance < 0:
            raise ValueError(
                "goal_tolerance must be nonnegative"
            )

        if (
            cfg.max_predictive_uncertainty
            is not None
            and (
                not math.isfinite(
                    cfg.max_predictive_uncertainty
                )
                or cfg.max_predictive_uncertainty < 0
            )
        ):
            raise ValueError(
                "Invalid uncertainty threshold"
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

    def observe(
        self,
        latent_state: Tensor,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> WorldObservation:
        """
        Update the current observed world state.

        The state should come from the
        perception/latent encoder pipeline.
        """

        self._validate_state(
            latent_state
        )

        observation = WorldObservation(
            latent_state=(
                latent_state.detach().clone()
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        self.current_observation = (
            observation
        )

        return observation

    def set_goal(
        self,
        goal: WorldGoal,
    ) -> None:
        """
        Register a goal in latent space.
        """

        self._validate_state(
            goal.target_state
        )

        if goal.tolerance < 0:
            raise ValueError(
                "Goal tolerance must be nonnegative"
            )

        self.current_goal = goal

    def clear_goal(self) -> None:
        self.current_goal = None

    def emergency_stop(self) -> None:
        """
        Block future action proposals.

        Physical emergency-stop systems
        must be implemented independently.
        """

        self.emergency_stop_active = True

        self.mode = (
            WorldAgentMode.EMERGENCY_STOP
        )

    def clear_emergency_stop(
        self,
        authorized: bool = False,
    ) -> None:
        """
        Require explicit authorization
        before clearing emergency stop.
        """

        if not authorized:
            raise PermissionError(
                "Authorization required to "
                "clear emergency stop"
            )

        self.emergency_stop_active = False
        self.mode = WorldAgentMode.IDLE

    @torch.no_grad()
    def predict_next_state(
        self,
        state: Tensor,
        action: Tensor,
    ) -> WorldPrediction:
        """
        Predict the next latent state.

        Model interface:
            (mean, logvar) = model(state, action)

        Deterministic models returning
        only next_state are also supported.
        """

        self._validate_state(state)

        if action.shape != (
            state.shape[0],
            self.config.action_dim,
        ):
            raise ValueError(
                "action must be [B,action_dim]"
            )

        result = self.world_model(
            state,
            action,
        )

        if isinstance(result, tuple):
            mean, logvar = result[:2]

            uncertainty = (
                logvar.exp().mean(dim=-1)
            )

            if self.config.deterministic_prediction:
                next_state = mean
            else:
                next_state = (
                    mean
                    + torch.randn_like(mean)
                    * torch.exp(0.5 * logvar)
                )
        else:
            mean = result
            logvar = None

            next_state = mean
            uncertainty = None

        return WorldPrediction(
            next_state=next_state,
            uncertainty=uncertainty,
            mean=mean,
            logvar=logvar,
        )

    def _validate_action(
        self,
        action: Tensor,
    ) -> Tensor:
        """
        Validate action bounds and optional
        domain-specific action constraints.

        Returns a boolean mask [B].
        """

        if action.ndim != 2:
            raise ValueError(
                "action must be [B,A]"
            )

        if action.shape[-1] != (
            self.config.action_dim
        ):
            raise ValueError(
                "Invalid action dimension"
            )

        finite = torch.isfinite(
            action
        ).all(dim=-1)

        bounded = (
            (action >= self.config.action_min)
            & (action <= self.config.action_max)
        ).all(dim=-1)

        accepted = finite & bounded

        if self.action_validator is not None:
            custom = self.action_validator(
                action
            )

            if custom.shape != (
                action.shape[0],
            ):
                raise ValueError(
                    "Action validator must "
                    "return [B]"
                )

            if custom.dtype != torch.bool:
                raise TypeError(
                    "Action validator must "
                    "return boolean values"
                )

            accepted &= custom

        return accepted

    def _goal_target(
        self,
        state: Tensor,
    ) -> Optional[Tensor]:
        if self.current_goal is None:
            return None

        target = (
            self.current_goal.target_state
        )

        if target.shape != state.shape:
            raise ValueError(
                "Goal batch shape mismatch"
            )

        if target.device != state.device:
            raise ValueError(
                "Goal device mismatch"
            )

        return target

    @torch.no_grad()
    def _planning_action(
        self,
        state: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Produce goal-conditioned
        model-based planning action.
        """

        if self.planning_agent is None:
            raise RuntimeError(
                "Planning agent is disabled"
            )

        target = self._goal_target(state)

        goal = None

        if target is not None:
            goal = PlanningGoal(
                target_state=target,
                tolerance=(
                    self.current_goal.tolerance
                ),
            )

        decision = (
            self.planning_agent.select_action(
                state=state,
                goal=goal,
            )
        )

        return (
            decision.action,
            decision.expected_score,
            decision.accepted,
        )

    @torch.no_grad()
    def _imagination_action(
        self,
        state: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Produce action using imagined
        future trajectories.
        """

        if self.imagination_agent is None:
            raise RuntimeError(
                "Imagination agent is disabled"
            )

        decision = (
            self.imagination_agent.select_action(
                state=state,
                goal_state=(
                    self._goal_target(state)
                ),
            )
        )

        accepted = torch.ones(
            state.shape[0],
            dtype=torch.bool,
            device=state.device,
        )

        return (
            decision.action,
            decision.score,
            accepted,
        )

    @torch.no_grad()
    def _exploration_action(
        self,
        state: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Produce exploratory action.
        """

        if self.exploration_agent is None:
            raise RuntimeError(
                "Exploration agent is disabled"
            )

        selection = (
            self.exploration_agent.select_action(
                state=state,
                explore=True,
            )
        )

        accepted = torch.ones(
            state.shape[0],
            dtype=torch.bool,
            device=state.device,
        )

        return (
            selection.action,
            selection.exploration_score,
            accepted,
        )

    def _record_decision(
        self,
        decision: WorldDecision,
    ) -> None:
        """
        Update counters and decision history.
        """

        self.total_decisions += 1

        accepted_count = int(
            decision.accepted.sum().item()
        )

        rejected_count = (
            decision.accepted.numel()
            - accepted_count
        )

        self.total_accepted += (
            accepted_count
        )

        self.total_rejected += (
            rejected_count
        )

        mean_score = None

        if decision.score is not None:
            mean_score = float(
                decision.score.mean().item()
            )

        mean_uncertainty = None

        if (
            decision.predicted_uncertainty
            is not None
        ):
            mean_uncertainty = float(
                decision.predicted_uncertainty
                .mean().item()
            )

        self.decision_history.append(
            WorldDecisionRecord(
                timestamp=time.time(),
                mode=decision.mode.value,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                mean_score=mean_score,
                mean_uncertainty=(
                    mean_uncertainty
                ),
                reason=decision.reason,
            )
        )

        if (
            len(self.decision_history)
            > self.config.history_capacity
        ):
            self.decision_history = (
                self.decision_history[
                    -self.config.history_capacity:
                ]
            )

    @torch.no_grad()
    def decide(
        self,
        state: Optional[Tensor] = None,
        strategy: str = "auto",
        approved: bool = False,
    ) -> WorldDecision:
        """
        Generate the next action proposal.

        Strategies:
            auto
            planning
            imagination
            exploration

        Automatic strategy:
            - Random exploration with configured
              exploration probability
            - Otherwise goal-directed planning
            - Otherwise imagination
            - Otherwise exploration

        Human approval:
            When require_human_approval=True,
            approved=True is needed for the
            proposal to be marked accepted.

        The approval flag is a prototype
        interface, not an authentication system.
        """

        if state is None:
            if self.current_observation is None:
                raise RuntimeError(
                    "No world observation available"
                )

            state = (
                self.current_observation
                .latent_state
            )

        self._validate_state(state)

        batch_size = state.shape[0]

        zeros = torch.zeros(
            batch_size,
            self.config.action_dim,
            device=state.device,
            dtype=state.dtype,
        )

        rejected = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=state.device,
        )

        if self.emergency_stop_active:
            decision = WorldDecision(
                action=zeros,
                mode=(
                    WorldAgentMode.EMERGENCY_STOP
                ),
                accepted=rejected,
                predicted_next_state=None,
                predicted_uncertainty=None,
                score=None,
                requires_approval=False,
                reason="Emergency stop active",
            )

            self._record_decision(decision)

            return decision

        valid_strategies = {
            "auto",
            "planning",
            "imagination",
            "exploration",
        }

        if strategy not in valid_strategies:
            raise ValueError(
                "Unknown decision strategy"
            )

        if strategy == "auto":

            explore = (
                self.config.enable_exploration
                and torch.rand(1).item()
                < self.config.exploration_probability
            )

            if explore:
                strategy = "exploration"

            elif (
                self.current_goal is not None
                and self.planning_agent is not None
            ):
                strategy = "planning"

            elif self.imagination_agent is not None:
                strategy = "imagination"

            elif self.exploration_agent is not None:
                strategy = "exploration"

            elif self.planning_agent is not None:
                strategy = "planning"

            else:
                raise RuntimeError(
                    "No decision agent enabled"
                )

        if strategy == "planning":
            action, score, accepted = (
                self._planning_action(state)
            )

            mode = WorldAgentMode.PLANNING

        elif strategy == "imagination":
            action, score, accepted = (
                self._imagination_action(state)
            )

            mode = WorldAgentMode.IMAGINING

        else:
            action, score, accepted = (
                self._exploration_action(state)
            )

            mode = WorldAgentMode.EXPLORING

        accepted = (
            accepted
            & self._validate_action(action)
        )

        prediction = self.predict_next_state(
            state,
            action,
        )

        if (
            self.config.max_predictive_uncertainty
            is not None
            and prediction.uncertainty is not None
        ):
            accepted = (
                accepted
                & (
                    prediction.uncertainty
                    <= self.config
                    .max_predictive_uncertainty
                )
            )

        requires_approval = (
            self.config.require_human_approval
            and not approved
        )

        if requires_approval:
            accepted = torch.zeros_like(
                accepted
            )

            mode = (
                WorldAgentMode.WAITING_APPROVAL
            )

        safe_action = torch.where(
            accepted.unsqueeze(-1),
            action,
            zeros,
        )

        if accepted.all().item():
            reason = (
                "Action proposal accepted"
            )
        elif requires_approval:
            reason = (
                "Human approval required"
            )
        else:
            reason = (
                "One or more action proposals rejected"
            )

        decision = WorldDecision(
            action=safe_action,
            mode=mode,
            accepted=accepted,
            predicted_next_state=(
                prediction.next_state
            ),
            predicted_uncertainty=(
                prediction.uncertainty
            ),
            score=score,
            requires_approval=(
                requires_approval
            ),
            reason=reason,
            metadata={
                "strategy": strategy,
                "batch_size": batch_size,
            },
        )

        self.mode = mode

        self._record_decision(
            decision
        )

        return decision

    @torch.no_grad()
    def evaluate_world_model(
        self,
        batch: EvaluationBatch,
        model_name: str = "NeuroWorld",
    ) -> Any:
        """
        Run the world-model evaluation agent.
        """

        if self.evaluation_agent is None:
            raise RuntimeError(
                "Evaluation agent is disabled"
            )

        self.mode = (
            WorldAgentMode.EVALUATING
        )

        try:
            return self.evaluation_agent.evaluate(
                model=self.world_model,
                batch=batch,
                model_name=model_name,
            )
        finally:
            self.mode = WorldAgentMode.IDLE

    def reset_episode(self) -> None:
        """
        Reset observation and episodic memory.

        Emergency stop remains active until
        explicitly cleared.
        """

        self.current_observation = None
        self.current_goal = None

        if self.exploration_agent is not None:
            self.exploration_agent.reset_episode()

        if self.planning_agent is not None:
            self.planning_agent.reset()

        if not self.emergency_stop_active:
            self.mode = WorldAgentMode.IDLE

    def statistics(self) -> Dict[str, Any]:
        """
        Return aggregate world-agent metrics.
        """

        total_outcomes = (
            self.total_accepted
            + self.total_rejected
        )

        acceptance_rate = (
            self.total_accepted / total_outcomes
            if total_outcomes > 0
            else 0.0
        )

        return {
            "mode": self.mode.value,
            "total_decisions": self.total_decisions,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": acceptance_rate,
            "emergency_stop_active": (
                self.emergency_stop_active
            ),
            "has_observation": (
                self.current_observation is not None
            ),
            "has_goal": (
                self.current_goal is not None
            ),
        }

    def export_history(
        self,
        path: str | Path,
    ) -> None:
        """
        Export decision history and statistics.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "config": asdict(self.config),
            "statistics": self.statistics(),
            "decisions": [
                asdict(item)
                for item in self.decision_history
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

    agent = WorldAgent(
        world_model=world_model,
        config=WorldAgentConfig(
            latent_dim=latent_dim,
            action_dim=action_dim,
            planning_horizon=10,
            num_candidates=64,
            exploration_probability=0.2,
            enable_planning=True,
            enable_imagination=True,
            enable_exploration=True,
            enable_evaluation=True,
            require_human_approval=False,
        ),
    ).to(device)

    batch_size = 2

    state = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    target = torch.randn(
        batch_size,
        latent_dim,
        device=device,
    )

    # Observe current world state.
    agent.observe(
        latent_state=state,
        metadata={
            "source": "latent_encoder",
        },
    )

    # Set desired goal.
    agent.set_goal(
        WorldGoal(
            target_state=target,
            tolerance=0.5,
            description="Reach target state",
        )
    )

    # Generate action proposal.
    decision = agent.decide(
        strategy="auto",
    )

    print(
        "Decision mode:",
        decision.mode.value,
    )

    print(
        "Selected action shape:",
        decision.action.shape,
    )

    print(
        "Accepted:",
        decision.accepted,
    )

    print(
        "Predicted next state:",
        (
            decision.predicted_next_state.shape
            if decision.predicted_next_state
            is not None
            else None
        ),
    )

    # Run world-model evaluation.
    evaluation_batch = EvaluationBatch(
        states=torch.randn(
            batch_size,
            11,
            latent_dim,
            device=device,
        ),
        actions=torch.randn(
            batch_size,
            10,
            action_dim,
            device=device,
        ),
    )

    report = agent.evaluate_world_model(
        evaluation_batch,
        model_name="NeuroWorld-WorldModel",
    )

    print(
        "Evaluation score:",
        report.overall_score,
    )

    # Demonstrate emergency stop.
    agent.emergency_stop()

    stopped_decision = agent.decide()

    print(
        "Emergency stop mode:",
        stopped_decision.mode.value,
    )

    print(
        "Statistics:",
        agent.statistics(),
    )

    agent.export_history(
        "artifacts/world_agent_history.json"
    )
```
