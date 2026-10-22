
```python
"""
NeuroWorld AI
File: neuroworld/agents/exploration_agent.py

Intelligent Exploration & Action Selection Agent

Features:
- Discrete and continuous action exploration
- Epsilon-greedy discrete action selection
- Softmax/Boltzmann exploration
- Gaussian continuous-action exploration
- Curiosity-based intrinsic motivation
- RND and episodic novelty integration
- World-model candidate action evaluation
- Goal-directed exploration
- Uncertainty-aware exploration
- Action masking and bounds
- Experience replay buffer
- Optional Q-network training
- Reward-based exploration learning
- Checkpoint saving/loading
- CPU/CUDA compatibility

Dependencies:
    pip install torch

Integration:
    neuroworld.agents.curiosity_agent
    neuroworld.dynamics.transition_model

Safety:
    This agent generates candidate actions.
    It does not directly control hardware.
    Hardware actions require an independent
    safety controller and authorization.
"""

from __future__ import annotations

import math
import random

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from neuroworld.agents.curiosity_agent import (
    CuriosityAgent,
    CuriosityAgentConfig,
)


@dataclass
class ExplorationAgentConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    action_space: str = "continuous"
    num_discrete_actions: int = 8

    exploration_strategy: str = "epsilon_greedy"

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000

    softmax_temperature: float = 1.0

    action_noise_std: float = 0.2
    action_min: float = -1.0
    action_max: float = 1.0

    num_action_candidates: int = 32

    intrinsic_reward_weight: float = 0.1
    extrinsic_reward_weight: float = 1.0

    uncertainty_weight: float = 0.1
    novelty_weight: float = 0.1
    goal_weight: float = 1.0
    action_penalty: float = 0.01

    learning_rate: float = 1e-4
    discount_factor: float = 0.99
    target_update_interval: int = 100

    replay_capacity: int = 10000
    batch_size: int = 64

    gradient_clip_norm: float = 1.0


@dataclass
class ExplorationAction:
    action: Tensor
    action_index: Optional[Tensor]
    exploration_score: Optional[Tensor]
    epsilon: float
    strategy: str


@dataclass
class ExplorationTransition:
    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    done: Tensor


@dataclass
class ExplorationTrainingMetrics:
    loss: float
    mean_q_value: float
    mean_target_q: float
    epsilon: float


class ExplorationReplayBuffer:
    """
    Bounded replay buffer for discrete-action
    reinforcement learning.

    Stores detached CPU tensors.
    """

    def __init__(
        self,
        capacity: int,
    ):
        if capacity <= 0:
            raise ValueError(
                "Replay capacity must be positive"
            )

        self.capacity = capacity

        self.memory: Deque[
            ExplorationTransition
        ] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.memory)

    def add(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """
        Add a batch of transitions.

        state:      [B,D]
        action:     [B] discrete indices
        reward:     [B]
        next_state: [B,D]
        done:       [B]
        """

        batch_size = state.shape[0]

        if (
            state.ndim != 2
            or next_state.shape != state.shape
            or action.shape != (batch_size,)
            or reward.shape != (batch_size,)
            or done.shape != (batch_size,)
        ):
            raise ValueError(
                "Invalid replay transition shapes"
            )

        for index in range(batch_size):
            self.memory.append(
                ExplorationTransition(
                    state=state[index].detach().cpu(),
                    action=action[index].detach().cpu(),
                    reward=reward[index].detach().cpu(),
                    next_state=(
                        next_state[index].detach().cpu()
                    ),
                    done=done[index].detach().cpu(),
                )
            )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> ExplorationTransition:
        if len(self.memory) < batch_size:
            raise ValueError(
                "Not enough replay samples"
            )

        samples = random.sample(
            list(self.memory),
            batch_size,
        )

        return ExplorationTransition(
            state=torch.stack(
                [item.state for item in samples]
            ).to(device),
            action=torch.stack(
                [item.action for item in samples]
            ).to(device),
            reward=torch.stack(
                [item.reward for item in samples]
            ).to(device),
            next_state=torch.stack(
                [item.next_state for item in samples]
            ).to(device),
            done=torch.stack(
                [item.done for item in samples]
            ).to(device),
        )

    def clear(self) -> None:
        self.memory.clear()


class ExplorationQNetwork(nn.Module):
    """
    Discrete action-value network.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_actions: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(
        self,
        state: Tensor,
    ) -> Tensor:
        return self.network(state)


class ContinuousActionPolicy(nn.Module):
    """
    Continuous action policy producing
    bounded action means.

    Exploration noise is applied separately.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        state: Tensor,
    ) -> Tensor:
        return self.network(state)


class ExplorationAgent(nn.Module):
    """
    Intelligent exploration agent.

    Discrete mode:
        - Epsilon-greedy
        - Softmax exploration
        - Double DQN-style target selection

    Continuous mode:
        - Learned action proposal network
        - Gaussian exploration noise
        - Optional world-model candidate search

    Curiosity:
        - Forward prediction error
        - Random Network Distillation
        - Episodic novelty

    The continuous policy is initialized
    randomly and is not optimized by the
    discrete Q-learning training method.
    """

    def __init__(
        self,
        config: Optional[
            ExplorationAgentConfig
        ] = None,
        curiosity_agent: Optional[
            CuriosityAgent
        ] = None,
        world_model: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.config = (
            config or ExplorationAgentConfig()
        )

        self._validate_config()

        cfg = self.config

        self.curiosity_agent = (
            curiosity_agent
            if curiosity_agent is not None
            else CuriosityAgent(
                CuriosityAgentConfig(
                    latent_dim=cfg.latent_dim,
                    action_dim=cfg.action_dim,
                )
            )
        )

        self.world_model = world_model

        self.q_network = ExplorationQNetwork(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_actions=cfg.num_discrete_actions,
        )

        self.target_q_network = (
            ExplorationQNetwork(
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                num_actions=cfg.num_discrete_actions,
            )
        )

        self.target_q_network.load_state_dict(
            self.q_network.state_dict()
        )

        for parameter in (
            self.target_q_network.parameters()
        ):
            parameter.requires_grad_(False)

        self.continuous_policy = (
            ContinuousActionPolicy(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                hidden_dim=cfg.hidden_dim,
            )
        )

        self.q_optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=cfg.learning_rate,
        )

        self.replay_buffer = (
            ExplorationReplayBuffer(
                cfg.replay_capacity
            )
        )

        self.exploration_steps = 0
        self.training_steps = 0

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.action_space not in (
            "discrete",
            "continuous",
        ):
            raise ValueError(
                "action_space must be discrete "
                "or continuous"
            )

        if cfg.exploration_strategy not in (
            "epsilon_greedy",
            "softmax",
            "gaussian",
        ):
            raise ValueError(
                "Unsupported exploration strategy"
            )

        if (
            cfg.action_space == "discrete"
            and cfg.exploration_strategy == "gaussian"
        ):
            raise ValueError(
                "Gaussian strategy requires "
                "continuous actions"
            )

        if (
            cfg.action_space == "continuous"
            and cfg.exploration_strategy != "gaussian"
        ):
            raise ValueError(
                "Continuous actions require "
                "gaussian exploration"
            )

        for value in (
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.num_discrete_actions,
            cfg.num_action_candidates,
            cfg.epsilon_decay_steps,
            cfg.target_update_interval,
            cfg.replay_capacity,
            cfg.batch_size,
        ):
            if value <= 0:
                raise ValueError(
                    "Dimensions and counts "
                    "must be positive"
                )

        if not (
            0 <= cfg.epsilon_end
            <= cfg.epsilon_start
            <= 1
        ):
            raise ValueError(
                "Invalid epsilon range"
            )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be < action_max"
            )

        if cfg.action_noise_std < 0:
            raise ValueError(
                "action_noise_std must be nonnegative"
            )

        if cfg.softmax_temperature <= 0:
            raise ValueError(
                "softmax_temperature must be positive"
            )

        if not 0 <= cfg.discount_factor <= 1:
            raise ValueError(
                "discount_factor must be in [0,1]"
            )

        if cfg.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

    def epsilon(self) -> float:
        cfg = self.config

        progress = min(
            self.exploration_steps
            / cfg.epsilon_decay_steps,
            1.0,
        )

        return (
            cfg.epsilon_start
            + progress
            * (cfg.epsilon_end - cfg.epsilon_start)
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

        if not torch.isfinite(state).all().item():
            raise ValueError(
                "State contains NaN or Inf"
            )

    @torch.no_grad()
    def select_discrete_action(
        self,
        state: Tensor,
        valid_action_mask: Optional[
            Tensor
        ] = None,
        explore: bool = True,
    ) -> ExplorationAction:
        """
        Select discrete actions using
        epsilon-greedy or softmax exploration.
        """

        self._validate_state(state)

        q_values = self.q_network(state)

        if valid_action_mask is None:
            valid = torch.ones_like(
                q_values,
                dtype=torch.bool,
            )
        else:
            if (
                valid_action_mask.shape
                != q_values.shape
                or valid_action_mask.dtype
                != torch.bool
            ):
                raise ValueError(
                    "Invalid action mask"
                )

            valid = valid_action_mask

        if not valid.any(dim=-1).all().item():
            raise ValueError(
                "Each state requires at least "
                "one valid action"
            )

        masked_q = q_values.masked_fill(
            ~valid,
            float("-inf"),
        )

        greedy = masked_q.argmax(dim=-1)

        epsilon = self.epsilon()

        if not explore:
            selected = greedy

        elif (
            self.config.exploration_strategy
            == "epsilon_greedy"
        ):
            random_scores = torch.rand_like(
                q_values
            ).masked_fill(
                ~valid,
                -1.0,
            )

            random_actions = (
                random_scores.argmax(dim=-1)
            )

            explore_mask = (
                torch.rand(
                    state.shape[0],
                    device=state.device,
                )
                < epsilon
            )

            selected = torch.where(
                explore_mask,
                random_actions,
                greedy,
            )

        else:
            probabilities = F.softmax(
                masked_q
                / self.config.softmax_temperature,
                dim=-1,
            )

            selected = torch.multinomial(
                probabilities,
                num_samples=1,
            ).squeeze(-1)

        if explore:
            self.exploration_steps += 1

        selected_q = q_values.gather(
            1,
            selected.unsqueeze(-1),
        ).squeeze(-1)

        return ExplorationAction(
            action=selected,
            action_index=selected,
            exploration_score=selected_q,
            epsilon=epsilon,
            strategy=(
                self.config.exploration_strategy
            ),
        )

    @torch.no_grad()
    def select_continuous_action(
        self,
        state: Tensor,
        explore: bool = True,
    ) -> ExplorationAction:
        """
        Generate bounded continuous actions.
        """

        self._validate_state(state)

        cfg = self.config

        normalized_action = (
            self.continuous_policy(state)
        )

        center = (
            cfg.action_min + cfg.action_max
        ) / 2.0

        half_range = (
            cfg.action_max - cfg.action_min
        ) / 2.0

        action = (
            center
            + half_range * normalized_action
        )

        if explore and cfg.action_noise_std > 0:
            action = (
                action
                + torch.randn_like(action)
                * cfg.action_noise_std
            )

        action = action.clamp(
            cfg.action_min,
            cfg.action_max,
        )

        epsilon = self.epsilon()

        if explore:
            self.exploration_steps += 1

        return ExplorationAction(
            action=action,
            action_index=None,
            exploration_score=None,
            epsilon=epsilon,
            strategy="gaussian",
        )

    @torch.no_grad()
    def select_action(
        self,
        state: Tensor,
        valid_action_mask: Optional[
            Tensor
        ] = None,
        explore: bool = True,
    ) -> ExplorationAction:
        """
        Unified action selection interface.
        """

        if self.config.action_space == "discrete":
            return self.select_discrete_action(
                state,
                valid_action_mask,
                explore,
            )

        if valid_action_mask is not None:
            raise ValueError(
                "Discrete action mask is not "
                "supported for continuous actions"
            )

        return self.select_continuous_action(
            state,
            explore,
        )

    @torch.no_grad()
    def evaluate_action_candidates(
        self,
        state: Tensor,
        candidate_actions: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Score continuous actions using a
        one-step world-model prediction.

        state:
            [B,D]

        candidate_actions:
            [B,K,A]

        goal_state:
            Optional [B,D]

        Returns:
            scores:      [B,K]
            next_states: [B,K,D]

        Score:
            novelty_weight * uncertainty
            - action_penalty * action_magnitude
            - goal_weight * goal_distance

        When no goal is provided, exploration
        favors uncertain predictions.

        Model uncertainty is an exploration
        heuristic, not a safety guarantee.
        """

        if self.world_model is None:
            raise RuntimeError(
                "World model is required "
                "for candidate evaluation"
            )

        self._validate_state(state)

        cfg = self.config

        if (
            candidate_actions.ndim != 3
            or candidate_actions.shape[0]
            != state.shape[0]
            or candidate_actions.shape[-1]
            != cfg.action_dim
        ):
            raise ValueError(
                "candidate_actions must be [B,K,A]"
            )

        if (
            candidate_actions.device != state.device
            or candidate_actions.dtype != state.dtype
        ):
            raise ValueError(
                "Candidate action device/dtype mismatch"
            )

        batch_size, candidates, _ = (
            candidate_actions.shape
        )

        expanded_states = (
            state.unsqueeze(1)
            .expand(
                -1,
                candidates,
                -1,
            )
        )

        flat_states = expanded_states.reshape(
            batch_size * candidates,
            cfg.latent_dim,
        )

        flat_actions = candidate_actions.reshape(
            batch_size * candidates,
            cfg.action_dim,
        )

        result = self.world_model(
            flat_states,
            flat_actions,
        )

        if isinstance(result, tuple):
            mean, logvar = result[:2]

            uncertainty = (
                logvar.exp().mean(dim=-1)
            )
        else:
            mean = result

            uncertainty = torch.zeros(
                batch_size * candidates,
                device=state.device,
                dtype=state.dtype,
            )

        next_states = mean.reshape(
            batch_size,
            candidates,
            cfg.latent_dim,
        )

        uncertainty = uncertainty.reshape(
            batch_size,
            candidates,
        )

        action_cost = (
            candidate_actions.square()
            .mean(dim=-1)
        )

        scores = (
            cfg.uncertainty_weight
            * uncertainty
            - cfg.action_penalty
            * action_cost
        )

        if goal_state is not None:
            if goal_state.shape != state.shape:
                raise ValueError(
                    "goal_state must be [B,D]"
                )

            distances = (
                next_states
                - goal_state.unsqueeze(1)
            ).square().mean(dim=-1)

            scores = (
                scores
                - cfg.goal_weight * distances
            )

        return scores, next_states

    @torch.no_grad()
    def plan_exploration_action(
        self,
        state: Tensor,
        goal_state: Optional[Tensor] = None,
    ) -> ExplorationAction:
        """
        Sample and rank continuous actions
        using the world model.

        This is one-step candidate search,
        not multi-step trajectory planning.
        """

        if self.config.action_space != "continuous":
            raise ValueError(
                "World-model candidate search "
                "requires continuous actions"
            )

        self._validate_state(state)

        cfg = self.config

        batch_size = state.shape[0]
        candidates = cfg.num_action_candidates

        # Include the current policy proposal.
        base = self.select_continuous_action(
            state,
            explore=False,
        ).action

        random_actions = (
            torch.rand(
                batch_size,
                candidates,
                cfg.action_dim,
                device=state.device,
                dtype=state.dtype,
            )
            * (
                cfg.action_max
                - cfg.action_min
            )
            + cfg.action_min
        )

        random_actions[:, 0, :] = base

        scores, _ = (
            self.evaluate_action_candidates(
                state,
                random_actions,
                goal_state,
            )
        )

        best_indices = scores.argmax(dim=1)

        batch_indices = torch.arange(
            batch_size,
            device=state.device,
        )

        best_actions = random_actions[
            batch_indices,
            best_indices,
        ]

        best_scores = scores[
            batch_indices,
            best_indices,
        ]

        self.exploration_steps += 1

        return ExplorationAction(
            action=best_actions,
            action_index=None,
            exploration_score=best_scores,
            epsilon=self.epsilon(),
            strategy="world_model_search",
        )

    def compute_exploration_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        extrinsic_reward: Tensor,
        update_statistics: bool = True,
        update_memory: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Combine environment reward and
        curiosity-based intrinsic reward.

        For discrete environments, pass an
        encoded action vector [B,action_dim],
        not raw integer action indices.
        """

        curiosity = self.curiosity_agent(
            state=state,
            action=action,
            next_state=next_state,
            extrinsic_reward=None,
            update_statistics=update_statistics,
            update_memory=update_memory,
        )

        intrinsic = (
            curiosity.intrinsic_reward.detach()
        )

        total = (
            self.config.extrinsic_reward_weight
            * extrinsic_reward
            + self.config.intrinsic_reward_weight
            * intrinsic
        )

        return {
            "intrinsic_reward": intrinsic,
            "extrinsic_reward": extrinsic_reward,
            "total_reward": total,
            "prediction_error": (
                curiosity.prediction_error.detach()
            ),
            "rnd_error": (
                curiosity.rnd_error.detach()
            ),
            "episodic_novelty": (
                curiosity.episodic_novelty.detach()
            ),
        }

    def store_transition(
        self,
        state: Tensor,
        action_indices: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """
        Store discrete RL transitions.

        Continuous transitions require a
        separate continuous-control learner.
        """

        if self.config.action_space != "discrete":
            raise ValueError(
                "Replay training currently supports "
                "discrete actions only"
            )

        self.replay_buffer.add(
            state,
            action_indices.long(),
            reward,
            next_state,
            done,
        )

    def training_step(
        self,
    ) -> Optional[ExplorationTrainingMetrics]:
        """
        Train discrete Q-network using replay.

        Double DQN target:

            a* = argmax_a Q_online(s',a)

            target =
                r + gamma * (1-done)
                * Q_target(s',a*)

        Returns None when replay memory
        has insufficient transitions.
        """

        if self.config.action_space != "discrete":
            raise ValueError(
                "Q-learning training is "
                "discrete-action only"
            )

        cfg = self.config

        if len(self.replay_buffer) < cfg.batch_size:
            return None

        device = next(
            self.q_network.parameters()
        ).device

        batch = self.replay_buffer.sample(
            cfg.batch_size,
            device,
        )

        actions = batch.action.long()

        if (
            (actions < 0).any().item()
            or (
                actions
                >= cfg.num_discrete_actions
            ).any().item()
        ):
            raise ValueError(
                "Replay contains invalid action index"
            )

        q_values = self.q_network(
            batch.state
        )

        selected_q = q_values.gather(
            1,
            actions.unsqueeze(-1),
        ).squeeze(-1)

        with torch.no_grad():
            next_online_q = self.q_network(
                batch.next_state
            )

            next_actions = (
                next_online_q.argmax(dim=-1)
            )

            next_target_q = (
                self.target_q_network(
                    batch.next_state
                )
            )

            next_q = next_target_q.gather(
                1,
                next_actions.unsqueeze(-1),
            ).squeeze(-1)

            targets = (
                batch.reward
                + cfg.discount_factor
                * (
                    1.0
                    - batch.done.float()
                )
                * next_q
            )

        loss = F.smooth_l1_loss(
            selected_q,
            targets,
        )

        self.q_optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        if cfg.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                cfg.gradient_clip_norm,
            )

        self.q_optimizer.step()

        self.training_steps += 1

        if (
            self.training_steps
            % cfg.target_update_interval
            == 0
        ):
            self.target_q_network.load_state_dict(
                self.q_network.state_dict()
            )

        return ExplorationTrainingMetrics(
            loss=float(loss.detach()),
            mean_q_value=float(
                selected_q.detach().mean()
            ),
            mean_target_q=float(
                targets.detach().mean()
            ),
            epsilon=self.epsilon(),
        )

    def reset_episode(self) -> None:
        """
        Clear episodic curiosity memory.
        """

        self.curiosity_agent.reset_episode()

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """
        Save exploration networks, optimizer,
        configuration and training counters.

        Replay buffer is not included.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "config": asdict(self.config),
                "model_state": self.state_dict(),
                "optimizer_state": (
                    self.q_optimizer.state_dict()
                ),
                "exploration_steps": (
                    self.exploration_steps
                ),
                "training_steps": (
                    self.training_steps
                ),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        world_model: Optional[nn.Module] = None,
    ) -> "ExplorationAgent":
        """
        Restore exploration agent.

        Uses safe tensor checkpoint loading.
        """

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        config = ExplorationAgentConfig(
            **checkpoint["config"]
        )

        agent = cls(
            config=config,
            world_model=world_model,
        )

        agent.load_state_dict(
            checkpoint["model_state"]
        )

        agent.q_optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        agent.exploration_steps = int(
            checkpoint["exploration_steps"]
        )

        agent.training_steps = int(
            checkpoint["training_steps"]
        )

        agent.eval()

        return agent


if __name__ == "__main__":

    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -----------------------------------------
    # Discrete Exploration
    # -----------------------------------------

    discrete_config = ExplorationAgentConfig(
        latent_dim=128,
        action_dim=8,
        action_space="discrete",
        num_discrete_actions=8,
        exploration_strategy="epsilon_greedy",
        batch_size=16,
    )

    discrete_agent = ExplorationAgent(
        discrete_config
    ).to(device)

    batch_size = 8

    states = torch.randn(
        batch_size,
        128,
        device=device,
    )

    selected = discrete_agent.select_action(
        states
    )

    print(
        "Discrete action indices:",
        selected.action,
    )

    # Encode discrete actions for curiosity.
    encoded_actions = F.one_hot(
        selected.action.long(),
        num_classes=8,
    ).float()

    next_states = torch.randn(
        batch_size,
        128,
        device=device,
    )

    environment_rewards = torch.randn(
        batch_size,
        device=device,
    )

    reward_info = (
        discrete_agent.compute_exploration_reward(
            state=states,
            action=encoded_actions,
            next_state=next_states,
            extrinsic_reward=environment_rewards,
        )
    )

    print(
        "Intrinsic rewards:",
        reward_info["intrinsic_reward"],
    )

    for _ in range(3):
        discrete_agent.store_transition(
            state=states,
            action_indices=selected.action,
            reward=reward_info["total_reward"],
            next_state=next_states,
            done=torch.zeros(
                batch_size,
                device=device,
            ),
        )

    metrics = discrete_agent.training_step()

    print("Q-learning metrics:", metrics)

    # -----------------------------------------
    # Continuous Exploration
    # -----------------------------------------

    continuous_config = ExplorationAgentConfig(
        latent_dim=128,
        action_dim=32,
        action_space="continuous",
        exploration_strategy="gaussian",
    )

    continuous_agent = ExplorationAgent(
        continuous_config
    ).to(device)

    continuous_action = (
        continuous_agent.select_action(
            states
        )
    )

    print(
        "Continuous actions:",
        continuous_action.action.shape,
    )

    # -----------------------------------------
    # Save Checkpoint
    # -----------------------------------------

    discrete_agent.save_checkpoint(
        "checkpoints/exploration_agent.pt"
    )

    restored = ExplorationAgent.load_checkpoint(
        "checkpoints/exploration_agent.pt",
        map_location=device,
    ).to(device)

    print(
        "Restored exploration steps:",
        restored.exploration_steps,
    )
```
