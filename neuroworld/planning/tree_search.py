
"""
NeuroWorld AI
File: neuroworld/planning/tree_search.py

Goal-Conditioned Monte Carlo Tree Search.

Features:
    - Neural world-model integration
    - PUCT tree selection
    - Learned policy priors
    - Goal-conditioned search
    - Uncertainty-aware planning
    - Action constraint validation
    - Monte Carlo simulations
    - Discounted return backup
    - Maximum search depth
    - Search time budget
    - Root action statistics
    - Best trajectory extraction
    - UnifiedPlanner adapter
    - Synthetic navigation example

Install:
    pip install torch numpy

Run:
    python -m neuroworld.planning.tree_search

Safety:
    Search results are predictions.
    This module does not execute robot actions.
"""

from __future__ import annotations

import math
import time

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# EXCEPTIONS
# ============================================================

class TreeSearchError(Exception):
    pass


class TreeSearchValidationError(TreeSearchError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TreeSearchConfig:
    state_dim: int = 4
    action_dim: int = 2

    position_dim: int = 2

    max_depth: int = 15
    simulations: int = 256

    branching_factor: int = 9

    exploration_constant: float = 1.5
    discount_factor: float = 0.99

    goal_tolerance: float = 0.25
    goal_reward: float = 10.0

    progress_weight: float = 2.0
    distance_penalty: float = 0.05

    action_penalty: float = 0.01
    uncertainty_penalty: float = 0.2

    maximum_uncertainty: float = 0.5

    action_min: float = -1.0
    action_max: float = 1.0

    max_action_change: float = 0.5

    time_budget_seconds: float = 5.0

    seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(
            self.state_dim,
            self.action_dim,
            self.position_dim,
            self.max_depth,
            self.simulations,
            self.branching_factor,
        ) <= 0:
            raise ValueError(
                "Dimensions and search sizes "
                "must be positive"
            )

        if self.position_dim > self.state_dim:
            raise ValueError(
                "position_dim exceeds state_dim"
            )

        if not 0 < self.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in (0,1]"
            )

        if self.exploration_constant < 0:
            raise ValueError(
                "exploration_constant must be nonnegative"
            )

        if self.goal_tolerance <= 0:
            raise ValueError(
                "goal_tolerance must be positive"
            )

        if self.maximum_uncertainty < 0:
            raise ValueError(
                "maximum_uncertainty must be nonnegative"
            )

        if self.action_min >= self.action_max:
            raise ValueError(
                "Invalid action bounds"
            )

        if self.max_action_change <= 0:
            raise ValueError(
                "max_action_change must be positive"
            )

        if self.time_budget_seconds <= 0:
            raise ValueError(
                "time_budget_seconds must be positive"
            )


# ============================================================
# WORLD MODEL PROTOCOL
# ============================================================

class TreeWorldModel(Protocol):
    """
    Predict one state transition.

    Inputs:
        state: [state_dim]
        action: [action_dim]

    Outputs:
        next_state: [state_dim]
        reward: float
        uncertainty: float
        done: bool
    """

    def predict_step(
        self,
        state: Tensor,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        float,
        bool,
    ]:
        ...


# ============================================================
# WORLD MODEL ADAPTER
# ============================================================

class CallableTreeWorldModel:
    """
    Wrap arbitrary world-model functions.

    dynamics_fn:
        (state, action) -> next_state

    reward_fn:
        (state, action, next_state) -> reward

    uncertainty_fn:
        (state, action, next_state) -> uncertainty

    termination_fn:
        (next_state) -> bool
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        reward_fn: Callable,
        uncertainty_fn: Callable | None = None,
        termination_fn: Callable | None = None,
    ) -> None:
        self.dynamics_fn = dynamics_fn
        self.reward_fn = reward_fn

        self.uncertainty_fn = uncertainty_fn
        self.termination_fn = termination_fn

    def predict_step(
        self,
        state: Tensor,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        float,
        bool,
    ]:
        next_state = self.dynamics_fn(
            state,
            action,
        )

        reward = float(
            self.reward_fn(
                state,
                action,
                next_state,
            )
        )

        uncertainty = 0.0

        if self.uncertainty_fn is not None:
            uncertainty = float(
                self.uncertainty_fn(
                    state,
                    action,
                    next_state,
                )
            )

        done = False

        if self.termination_fn is not None:
            done = bool(
                self.termination_fn(
                    next_state
                )
            )

        return (
            next_state,
            reward,
            uncertainty,
            done,
        )


# ============================================================
# SEARCH NODE
# ============================================================

@dataclass
class SearchNode:
    state: Tensor

    parent: SearchNode | None = None

    action: Tensor | None = None

    prior: float = 1.0

    reward: float = 0.0
    uncertainty: float = 0.0

    terminal: bool = False

    depth: int = 0

    visits: int = 0
    value_sum: float = 0.0

    children: list[SearchNode] = field(
        default_factory=list
    )

    expanded: bool = False

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    def path_from_root(
        self,
    ) -> list[SearchNode]:
        path = []
        node: SearchNode | None = self

        while node is not None:
            path.append(node)
            node = node.parent

        return list(reversed(path))


# ============================================================
# ACTION SAMPLER
# ============================================================

class DiscreteActionSampler:
    """
    Creates a fixed set of candidate actions.

    For 2D controls:
        generates a grid of acceleration
        or movement commands.

    For higher dimensions:
        samples random action vectors.
    """

    def __init__(
        self,
        config: TreeSearchConfig,
    ) -> None:
        self.config = config

        self.generator = torch.Generator(
            device=config.device
        )

        self.generator.manual_seed(
            config.seed
        )

    def sample(
        self,
        previous_action: Tensor | None = None,
    ) -> Tensor:
        config = self.config

        if config.action_dim == 2:
            grid_size = math.ceil(
                math.sqrt(
                    config.branching_factor
                )
            )

            values = torch.linspace(
                config.action_min,
                config.action_max,
                grid_size,
                device=config.device,
            )

            mesh = torch.cartesian_prod(
                values,
                values,
            )

            actions = mesh[
                :config.branching_factor
            ]

        else:
            actions = (
                torch.rand(
                    config.branching_factor,
                    config.action_dim,
                    generator=self.generator,
                    device=config.device,
                )
                * (
                    config.action_max
                    - config.action_min
                )
                + config.action_min
            )

            actions[0] = 0.0

        if previous_action is not None:
            lower = (
                previous_action
                - config.max_action_change
            )

            upper = (
                previous_action
                + config.max_action_change
            )

            actions = torch.maximum(
                actions,
                lower,
            )

            actions = torch.minimum(
                actions,
                upper,
            )

        return actions.clamp(
            config.action_min,
            config.action_max,
        )


# ============================================================
# POLICY PRIOR
# ============================================================

class TreePolicyPrior:
    """
    Optional learned action-prior model.

    policy_fn:
        (state, actions, goal) -> logits [K]

    Without a learned policy, use uniform
    priors over candidate actions.
    """

    def __init__(
        self,
        policy_fn: Callable | None = None,
    ) -> None:
        self.policy_fn = policy_fn

    def probabilities(
        self,
        state: Tensor,
        actions: Tensor,
        goal: Tensor,
    ) -> Tensor:
        count = actions.shape[0]

        if self.policy_fn is None:
            return torch.full(
                (count,),
                1.0 / count,
                device=actions.device,
            )

        logits = torch.as_tensor(
            self.policy_fn(
                state,
                actions,
                goal,
            ),
            dtype=torch.float32,
            device=actions.device,
        )

        if logits.shape != (count,):
            raise TreeSearchValidationError(
                "Policy must return [K] logits"
            )

        if not torch.isfinite(logits).all():
            raise TreeSearchValidationError(
                "Policy logits must be finite"
            )

        return torch.softmax(
            logits,
            dim=0,
        )


# ============================================================
# GOAL EVALUATOR
# ============================================================

class TreeGoalEvaluator:
    """
    Computes goal progress and terminal reward.
    """

    def __init__(
        self,
        config: TreeSearchConfig,
    ) -> None:
        self.config = config

    def distance(
        self,
        state: Tensor,
        goal: Tensor,
    ) -> float:
        return float(
            torch.linalg.vector_norm(
                state[:self.config.position_dim]
                - goal,
            ).item()
        )

    def reached(
        self,
        state: Tensor,
        goal: Tensor,
    ) -> bool:
        return (
            self.distance(state, goal)
            <= self.config.goal_tolerance
        )

    def progress_reward(
        self,
        state: Tensor,
        next_state: Tensor,
        goal: Tensor,
    ) -> float:
        before = self.distance(
            state,
            goal,
        )

        after = self.distance(
            next_state,
            goal,
        )

        reward = (
            self.config.progress_weight
            * (before - after)
        )

        reward -= (
            self.config.distance_penalty
            * after
        )

        if (
            after <= self.config.goal_tolerance
            and before > self.config.goal_tolerance
        ):
            reward += (
                self.config.goal_reward
            )

        return reward


# ============================================================
# SEARCH RESULT
# ============================================================

@dataclass
class TreeSearchResult:
    success: bool

    first_action: Tensor | None
    action_sequence: Tensor | None

    predicted_states: Tensor | None

    best_score: float | None

    goal_reached: bool

    simulations_completed: int
    nodes_expanded: int

    root_action_visits: list[int]
    root_action_values: list[float]

    elapsed_seconds: float

    message: str

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    def to_numpy(self) -> dict[str, Any]:
        def convert(
            value: Tensor | None,
        ) -> np.ndarray | None:
            if value is None:
                return None

            return (
                value.detach()
                .cpu()
                .numpy()
            )

        return {
            "success": self.success,
            "first_action": convert(
                self.first_action
            ),
            "action_sequence": convert(
                self.action_sequence
            ),
            "predicted_states": convert(
                self.predicted_states
            ),
            "best_score": self.best_score,
            "goal_reached": self.goal_reached,
            "simulations_completed": (
                self.simulations_completed
            ),
            "nodes_expanded": (
                self.nodes_expanded
            ),
            "root_action_visits": (
                self.root_action_visits
            ),
            "root_action_values": (
                self.root_action_values
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "message": self.message,
            "diagnostics": self.diagnostics,
        }


# ============================================================
# MONTE CARLO TREE SEARCH
# ============================================================

class MonteCarloTreeSearch:
    """
    PUCT-based world-model tree search.

    Selection:
        Q(s,a) + c * P(s,a)
        * sqrt(parent_visits)/(1+child_visits)

    Backup:
        discounted cumulative reward
        plus terminal leaf heuristic.

    The tree is rebuilt for each plan call.
    """

    def __init__(
        self,
        world_model: TreeWorldModel,
        config: TreeSearchConfig | None = None,
        *,
        action_sampler: (
            DiscreteActionSampler | None
        ) = None,
        policy_prior: (
            TreePolicyPrior | None
        ) = None,
        state_constraint: (
            Callable[[Tensor], bool]
            | None
        ) = None,
    ) -> None:
        self.config = (
            config or TreeSearchConfig()
        )

        self.device = torch.device(
            self.config.device
        )

        self.world_model = world_model

        self.action_sampler = (
            action_sampler
            or DiscreteActionSampler(
                self.config
            )
        )

        self.policy_prior = (
            policy_prior
            or TreePolicyPrior()
        )

        self.state_constraint = (
            state_constraint
        )

        self.goal_evaluator = (
            TreeGoalEvaluator(
                self.config
            )
        )

        self.previous_action = torch.zeros(
            self.config.action_dim,
            device=self.device,
        )

        self._expanded_nodes = 0

    def reset(self) -> None:
        self.previous_action.zero_()

    def acknowledge_action(
        self,
        action: Tensor | np.ndarray,
    ) -> None:
        action = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self.device,
        )

        if action.shape != (
            self.config.action_dim,
        ):
            raise TreeSearchValidationError(
                "Invalid applied action"
            )

        if not torch.isfinite(action).all():
            raise TreeSearchValidationError(
                "Applied action must be finite"
            )

        if (
            (action < self.config.action_min).any()
            or (action > self.config.action_max).any()
        ):
            raise TreeSearchValidationError(
                "Applied action violates bounds"
            )

        self.previous_action = action.clone()

    def _validate_state(
        self,
        state: Tensor | np.ndarray,
    ) -> Tensor:
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        if state.shape != (
            self.config.state_dim,
        ):
            raise TreeSearchValidationError(
                "Invalid state shape"
            )

        if not torch.isfinite(state).all():
            raise TreeSearchValidationError(
                "State contains non-finite values"
            )

        return state

    def _validate_goal(
        self,
        goal: Tensor | np.ndarray,
    ) -> Tensor:
        goal = torch.as_tensor(
            goal,
            dtype=torch.float32,
            device=self.device,
        )

        if goal.shape != (
            self.config.position_dim,
        ):
            raise TreeSearchValidationError(
                "Goal must have position_dim values"
            )

        if not torch.isfinite(goal).all():
            raise TreeSearchValidationError(
                "Goal contains non-finite values"
            )

        return goal

    def _valid_state(
        self,
        state: Tensor,
    ) -> bool:
        if self.state_constraint is None:
            return True

        outcome = self.state_constraint(
            state
        )

        if isinstance(outcome, Tensor):
            if outcome.numel() != 1:
                raise TreeSearchValidationError(
                    "State constraint must be scalar"
                )

            return bool(outcome.item())

        return bool(outcome)

    @torch.no_grad()
    def _expand(
        self,
        node: SearchNode,
        goal: Tensor,
    ) -> None:
        if (
            node.expanded
            or node.terminal
            or node.depth >= self.config.max_depth
        ):
            return

        previous_action = (
            node.action
            if node.action is not None
            else self.previous_action
        )

        actions = self.action_sampler.sample(
            previous_action
        )

        priors = self.policy_prior.probabilities(
            node.state,
            actions,
            goal,
        )

        proposed = []

        for index in range(
            actions.shape[0]
        ):
            action = actions[index]

            (
                next_state,
                model_reward,
                uncertainty,
                done,
            ) = self.world_model.predict_step(
                node.state,
                action,
            )

            next_state = self._validate_state(
                next_state
            )

            model_reward = float(
                model_reward
            )

            uncertainty = float(
                uncertainty
            )

            if not math.isfinite(
                model_reward
            ):
                continue

            if (
                not math.isfinite(uncertainty)
                or uncertainty < 0
                or uncertainty
                > self.config.maximum_uncertainty
            ):
                continue

            if not self._valid_state(
                next_state
            ):
                continue

            reward = (
                model_reward
                + self.goal_evaluator.progress_reward(
                    node.state,
                    next_state,
                    goal,
                )
            )

            reward -= (
                self.config.action_penalty
                * float(
                    action.square()
                    .sum()
                    .item()
                )
            )

            reward -= (
                self.config.uncertainty_penalty
                * uncertainty
            )

            terminal = (
                bool(done)
                or self.goal_evaluator.reached(
                    next_state,
                    goal,
                )
            )

            proposed.append(
                (
                    next_state,
                    action.clone(),
                    reward,
                    uncertainty,
                    terminal,
                    float(
                        priors[index].item()
                    ),
                )
            )

        total_prior = sum(
            item[5]
            for item in proposed
        )

        for (
            next_state,
            action,
            reward,
            uncertainty,
            terminal,
            prior,
        ) in proposed:
            child = SearchNode(
                state=next_state,
                parent=node,
                action=action,
                prior=(
                    prior / total_prior
                    if total_prior > 0
                    else 1.0 / len(proposed)
                ),
                reward=reward,
                uncertainty=uncertainty,
                terminal=terminal,
                depth=node.depth + 1,
            )

            node.children.append(
                child
            )

        node.expanded = True
        self._expanded_nodes += 1

    def _select_child(
        self,
        node: SearchNode,
    ) -> SearchNode:
        if not node.children:
            raise TreeSearchError(
                "Cannot select from empty children"
            )

        parent_scale = math.sqrt(
            max(1, node.visits)
        )

        def score(
            child: SearchNode,
        ) -> float:
            exploitation = (
                child.reward
                + self.config.discount_factor
                * child.mean_value
                if child.visits > 0
                else 0.0
            )

            exploration = (
                self.config.exploration_constant
                * child.prior
                * parent_scale
                / (1 + child.visits)
            )

            return (
                exploitation
                + exploration
            )

        return max(
            node.children,
            key=score,
        )

    def _leaf_value(
        self,
        node: SearchNode,
        goal: Tensor,
    ) -> float:
        distance = (
            self.goal_evaluator.distance(
                node.state,
                goal,
            )
        )

        return -distance

    def _backup(
        self,
        path: list[SearchNode],
        leaf_value: float,
    ) -> None:
        """
        Each node stores its estimated future
        value from that node onward.

        Edge rewards are incorporated when
        backing up to the parent.
        """
        value = leaf_value

        for index in range(
            len(path) - 1,
            -1,
            -1,
        ):
            node = path[index]

            node.visits += 1
            node.value_sum += value

            if index > 0:
                value = (
                    node.reward
                    + self.config.discount_factor
                    * value
                )

    @torch.no_grad()
    def plan(
        self,
        initial_state: Tensor | np.ndarray,
        goal: Tensor | np.ndarray,
    ) -> TreeSearchResult:
        started = time.perf_counter()

        state = self._validate_state(
            initial_state
        )

        goal = self._validate_goal(
            goal
        )

        if not self._valid_state(state):
            raise TreeSearchValidationError(
                "Initial state violates constraints"
            )

        root = SearchNode(
            state=state.clone(),
            terminal=self.goal_evaluator.reached(
                state,
                goal,
            ),
        )

        self._expanded_nodes = 0

        if root.terminal:
            return TreeSearchResult(
                success=True,
                first_action=None,
                action_sequence=torch.empty(
                    (0, self.config.action_dim),
                    device=self.device,
                ),
                predicted_states=state.unsqueeze(
                    0
                ),
                best_score=0.0,
                goal_reached=True,
                simulations_completed=0,
                nodes_expanded=0,
                root_action_visits=[],
                root_action_values=[],
                elapsed_seconds=(
                    time.perf_counter() - started
                ),
                message="Goal already reached",
            )

        simulations_completed = 0

        for _ in range(
            self.config.simulations
        ):
            if (
                time.perf_counter() - started
                >= self.config.time_budget_seconds
            ):
                break

            node = root
            path = [root]

            # Selection
            while (
                node.expanded
                and node.children
                and not node.terminal
                and node.depth
                < self.config.max_depth
            ):
                node = self._select_child(
                    node
                )

                path.append(
                    node
                )

            # Expansion
            if (
                not node.terminal
                and node.depth
                < self.config.max_depth
            ):
                self._expand(
                    node,
                    goal,
                )

                if node.children:
                    node = self._select_child(
                        node
                    )

                    path.append(
                        node
                    )

            # Evaluation
            if node.terminal:
                leaf_value = 0.0
            else:
                leaf_value = self._leaf_value(
                    node,
                    goal,
                )

            # Backup
            self._backup(
                path,
                leaf_value,
            )

            simulations_completed += 1

        if not root.children:
            return TreeSearchResult(
                success=False,
                first_action=None,
                action_sequence=None,
                predicted_states=None,
                best_score=None,
                goal_reached=False,
                simulations_completed=(
                    simulations_completed
                ),
                nodes_expanded=(
                    self._expanded_nodes
                ),
                root_action_visits=[],
                root_action_values=[],
                elapsed_seconds=(
                    time.perf_counter() - started
                ),
                message=(
                    "No feasible search branches"
                ),
            )

        # Choose most visited root child.
        best_child = max(
            root.children,
            key=lambda child: (
                child.visits,
                child.reward
                + self.config.discount_factor
                * child.mean_value,
            ),
        )

        # Extract a representative path using
        # the most visited descendant.
        path = [root, best_child]
        node = best_child

        while (
            node.children
            and not node.terminal
            and node.depth
            < self.config.max_depth
        ):
            node = max(
                node.children,
                key=lambda child: (
                    child.visits,
                    child.reward
                    + self.config.discount_factor
                    * child.mean_value,
                ),
            )

            path.append(
                node
            )

        actions = torch.stack(
            [
                item.action
                for item in path[1:]
            ],
            dim=0,
        )

        states = torch.stack(
            [
                item.state
                for item in path
            ],
            dim=0,
        )

        goal_reached = (
            self.goal_evaluator.reached(
                states[-1],
                goal,
            )
        )

        root_values = [
            (
                child.reward
                + self.config.discount_factor
                * child.mean_value
            )
            for child in root.children
        ]

        return TreeSearchResult(
            success=True,
            first_action=actions[0],
            action_sequence=actions,
            predicted_states=states,
            best_score=(
                best_child.reward
                + self.config.discount_factor
                * best_child.mean_value
            ),
            goal_reached=goal_reached,
            simulations_completed=(
                simulations_completed
            ),
            nodes_expanded=(
                self._expanded_nodes
            ),
            root_action_visits=[
                child.visits
                for child in root.children
            ],
            root_action_values=root_values,
            elapsed_seconds=(
                time.perf_counter() - started
            ),
            message=(
                "Tree search completed"
            ),
            diagnostics={
                "tree_depth": len(path) - 1,
                "root_children": len(
                    root.children
                ),
                "terminal_path": (
                    path[-1].terminal
                ),
            },
        )


# ============================================================
# UNIFIED PLANNER ADAPTER
# ============================================================

class UnifiedTreeSearchAdapter:
    """
    Connects MCTS to UnifiedPlanner.

    Note:
        MCTS can return fewer than max_depth
        actions, depending on expansion.

        Its predicted states must match
        the unified planner's state_dim.
    """

    def __init__(
        self,
        planner: MonteCarloTreeSearch,
    ) -> None:
        self.planner = planner

    def plan(
        self,
        request: Any,
    ) -> Any:
        from neuroworld.planning.planner import (
            StrategyPlan,
        )

        if request.goal_state is None:
            raise TreeSearchValidationError(
                "Tree search requires a goal"
            )

        goal = torch.as_tensor(
            request.goal_state,
            dtype=torch.float32,
        )[
            :self.planner.config.position_dim
        ]

        result = self.planner.plan(
            request.current_state,
            goal,
        )

        if (
            not result.success
            or result.action_sequence is None
            or result.action_sequence.shape[0] == 0
        ):
            raise TreeSearchError(
                result.message
            )

        return StrategyPlan(
            actions=result.action_sequence,
            states=result.predicted_states,
            predicted_score=result.best_score,
            metadata={
                "source": "tree_search",
                "simulations": (
                    result.simulations_completed
                ),
                "nodes_expanded": (
                    result.nodes_expanded
                ),
                "goal_reached": (
                    result.goal_reached
                ),
            },
        )


# ============================================================
# SYNTHETIC WORLD MODEL
# ============================================================

class NavigationTreeWorldModel:
    """
    Simple 2D point-mass dynamics.

    State:
        [x,y,vx,vy]

    Action:
        [ax,ay]
    """

    def __init__(
        self,
        timestep: float = 0.2,
    ) -> None:
        self.timestep = timestep

    def predict_step(
        self,
        state: Tensor,
        action: Tensor,
    ) -> tuple[
        Tensor,
        float,
        float,
        bool,
    ]:
        position = state[:2]
        velocity = state[2:4]

        next_velocity = (
            velocity
            + action * self.timestep
        )

        next_position = (
            position
            + next_velocity
            * self.timestep
        )

        next_state = torch.cat(
            [
                next_position,
                next_velocity,
            ]
        )

        reward = -0.01 * float(
            action.square().sum().item()
        )

        uncertainty = 0.01

        return (
            next_state,
            reward,
            uncertainty,
            False,
        )


# ============================================================
# FACTORY
# ============================================================

def create_tree_search(
    world_model: TreeWorldModel,
    config: TreeSearchConfig | None = None,
    **kwargs: Any,
) -> MonteCarloTreeSearch:
    return MonteCarloTreeSearch(
        world_model,
        config,
        **kwargs,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - TREE SEARCH PLANNER"
    )
    print("=" * 65)

    config = TreeSearchConfig(
        state_dim=4,
        action_dim=2,
        position_dim=2,
        max_depth=12,
        simulations=200,
        branching_factor=9,
        time_budget_seconds=5.0,
    )

    world_model = NavigationTreeWorldModel(
        timestep=0.2
    )

    planner = MonteCarloTreeSearch(
        world_model,
        config,
        state_constraint=lambda state: bool(
            (
                (state[:2] >= -5.0)
                & (state[:2] <= 5.0)
            ).all().item()
        ),
    )

    state = np.array(
        [0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    goal = np.array(
        [2.0, 2.0],
        dtype=np.float32,
    )

    result = planner.plan(
        state,
        goal,
    )

    print(
        "Success:",
        result.success,
    )

    print(
        "Goal reached:",
        result.goal_reached,
    )

    print(
        "Simulations:",
        result.simulations_completed,
    )

    print(
        "Expanded nodes:",
        result.nodes_expanded,
    )

    print(
        "Best score:",
        result.best_score,
    )

    print(
        "Elapsed:",
        round(
            result.elapsed_seconds,
            4,
        ),
        "seconds",
    )

    if result.action_sequence is not None:
        print(
            "Action sequence:",
            result.action_sequence.shape,
        )

        print(
            "First action:",
            result.first_action,
        )

        print(
            "Predicted states:",
            result.predicted_states.shape,
        )

    print(
        "\nTree search demonstration completed"
    )


if __name__ == "__main__":
    main()
