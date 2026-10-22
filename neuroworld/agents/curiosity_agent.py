
"""
NeuroWorld AI
File: neuroworld/agents/curiosity_agent.py

Curiosity-Driven Exploration Agent

Features:
- Intrinsic curiosity rewards
- Forward dynamics prediction error
- Random Network Distillation (RND)
- Episodic state novelty
- Running reward normalization
- Intrinsic / extrinsic reward combination
- Epsilon-greedy action selection
- Optional action masking
- Batched PyTorch operations
- Gradient clipping
- Save/load checkpoints
- CPU and CUDA support

Dependencies:
    pip install torch

Notes:
    1. RND measures novelty relative to a
       fixed randomly initialized target.

    2. Forward prediction error can reflect
       aleatoric noise, not just learnable novelty.

    3. Episodic novelty uses state similarity
       within a bounded memory.

    4. The agent selects discrete actions.
       A continuous-action policy can be
       integrated separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class CuriosityAgentConfig:
    latent_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256

    rnd_feature_dim: int = 128

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    prediction_reward_weight: float = 0.4
    rnd_reward_weight: float = 0.4
    episodic_reward_weight: float = 0.2

    intrinsic_reward_scale: float = 0.1
    extrinsic_reward_scale: float = 1.0

    normalize_intrinsic_reward: bool = True
    normalization_epsilon: float = 1e-8
    reward_clip: float = 10.0

    episodic_memory_capacity: int = 1000
    novelty_neighbors: int = 5

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000

    gradient_clip_norm: float = 1.0


@dataclass
class CuriosityOutput:
    prediction_error: Tensor
    rnd_error: Tensor
    episodic_novelty: Tensor
    intrinsic_reward: Tensor
    total_reward: Tensor


class RunningMeanStd(nn.Module):
    """
    Running statistics using parallel variance
    updates.

    Stores mean, variance and count as buffers,
    allowing device transfers and checkpoints.
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
    ):
        super().__init__()

        self.register_buffer(
            "mean",
            torch.zeros((), dtype=torch.float64),
        )

        self.register_buffer(
            "var",
            torch.ones((), dtype=torch.float64),
        )

        self.register_buffer(
            "count",
            torch.tensor(
                epsilon,
                dtype=torch.float64,
            ),
        )

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        values = values.detach().double().reshape(-1)

        if values.numel() == 0:
            return

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)

        batch_count = torch.tensor(
            values.numel(),
            dtype=torch.float64,
            device=self.mean.device,
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        m_a = self.var * self.count
        m_b = batch_var * batch_count

        correction = (
            delta.square()
            * self.count
            * batch_count
            / total_count
        )

        new_var = (
            m_a + m_b + correction
        ) / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var.clamp_min(1e-12))
        self.count.copy_(total_count)

    def normalize(
        self,
        values: Tensor,
        epsilon: float = 1e-8,
    ) -> Tensor:
        """
        Scale rewards by running standard
        deviation without subtracting the mean.

        This preserves nonnegative novelty
        rewards while controlling their scale.
        """

        scale = torch.sqrt(
            self.var.to(
                device=values.device,
                dtype=values.dtype,
            )
            + epsilon
        )

        return values / scale


class ForwardCuriosityModel(nn.Module):
    """
    Predict next latent state from current
    latent state and action.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tensor:
        features = torch.cat(
            [state, action],
            dim=-1,
        )

        delta = self.network(features)

        return state + delta


class RNDNetwork(nn.Module):
    """
    Random Network Distillation network.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        feature_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        state: Tensor,
    ) -> Tensor:
        return self.network(state)


class EpisodicNoveltyMemory:
    """
    Bounded memory of previously visited states.

    Novelty is estimated using distances to
    nearest stored states.

    This is an approximate exploration heuristic,
    not a calibrated density estimator.
    """

    def __init__(
        self,
        latent_dim: int,
        capacity: int = 1000,
        neighbors: int = 5,
    ):
        if latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if capacity <= 0 or neighbors <= 0:
            raise ValueError(
                "capacity and neighbors must "
                "be positive"
            )

        self.latent_dim = latent_dim
        self.capacity = capacity
        self.neighbors = neighbors

        self.states: Optional[Tensor] = None

    def reset(self) -> None:
        self.states = None

    @torch.no_grad()
    def add(
        self,
        states: Tensor,
    ) -> None:
        if (
            states.ndim != 2
            or states.shape[-1] != self.latent_dim
        ):
            raise ValueError(
                "states must be [B,latent_dim]"
            )

        detached = states.detach().clone()

        if self.states is None:
            self.states = detached[
                -self.capacity:
            ]
            return

        stored = self.states.to(
            device=detached.device,
            dtype=detached.dtype,
        )

        self.states = torch.cat(
            [stored, detached],
            dim=0,
        )[-self.capacity:]

    @torch.no_grad()
    def novelty(
        self,
        states: Tensor,
    ) -> Tensor:
        if (
            states.ndim != 2
            or states.shape[-1] != self.latent_dim
        ):
            raise ValueError(
                "states must be [B,latent_dim]"
            )

        if (
            self.states is None
            or self.states.shape[0] == 0
        ):
            return torch.ones(
                states.shape[0],
                device=states.device,
                dtype=states.dtype,
            )

        stored = self.states.to(
            device=states.device,
            dtype=states.dtype,
        )

        distances = torch.cdist(
            states.unsqueeze(0),
            stored.unsqueeze(0),
        ).squeeze(0)

        k = min(
            self.neighbors,
            stored.shape[0],
        )

        nearest = distances.topk(
            k=k,
            largest=False,
            dim=-1,
        ).values

        average_distance = nearest.mean(dim=-1)

        # Bounded novelty in [0,1).
        return (
            average_distance
            / (1.0 + average_distance)
        )


class CuriosityAgent(nn.Module):
    """
    Curiosity-driven exploration agent.

    Combines:
        1. Forward-model prediction error
        2. Random Network Distillation
        3. Episodic novelty

    Intrinsic reward:

        r_int =
            w_forward * prediction_error
            + w_rnd * rnd_error
            + w_episode * episodic_novelty

    Combined reward:

        r_total =
            extrinsic_scale * r_ext
            + intrinsic_scale * r_int

    The agent includes an epsilon-greedy
    action-selection utility. Action values
    are supplied by an external policy or
    value model.
    """

    def __init__(
        self,
        config: Optional[
            CuriosityAgentConfig
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or CuriosityAgentConfig()
        )

        cfg = self.config

        self._validate_config()

        self.forward_model = ForwardCuriosityModel(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.hidden_dim,
        )

        self.rnd_target = RNDNetwork(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            feature_dim=cfg.rnd_feature_dim,
        )

        self.rnd_predictor = RNDNetwork(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            feature_dim=cfg.rnd_feature_dim,
        )

        # Target must remain fixed.
        for parameter in (
            self.rnd_target.parameters()
        ):
            parameter.requires_grad_(False)

        self.rnd_target.eval()

        self.reward_statistics = RunningMeanStd()

        self.episodic_memory = (
            EpisodicNoveltyMemory(
                latent_dim=cfg.latent_dim,
                capacity=(
                    cfg.episodic_memory_capacity
                ),
                neighbors=cfg.novelty_neighbors,
            )
        )

        self.optimizer = torch.optim.AdamW(
            list(self.forward_model.parameters())
            + list(self.rnd_predictor.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.exploration_steps = 0

    def _validate_config(self) -> None:
        cfg = self.config

        dimensions = (
            cfg.latent_dim,
            cfg.action_dim,
            cfg.hidden_dim,
            cfg.rnd_feature_dim,
            cfg.episodic_memory_capacity,
            cfg.novelty_neighbors,
            cfg.epsilon_decay_steps,
        )

        if any(value <= 0 for value in dimensions):
            raise ValueError(
                "Dimensions and capacities "
                "must be positive"
            )

        if cfg.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        weights = (
            cfg.prediction_reward_weight,
            cfg.rnd_reward_weight,
            cfg.episodic_reward_weight,
            cfg.intrinsic_reward_scale,
            cfg.extrinsic_reward_scale,
        )

        if any(value < 0 for value in weights):
            raise ValueError(
                "Reward weights must be nonnegative"
            )

        if not (
            0 <= cfg.epsilon_end
            <= cfg.epsilon_start
            <= 1
        ):
            raise ValueError(
                "Require 0 <= epsilon_end "
                "<= epsilon_start <= 1"
            )

        if cfg.reward_clip <= 0:
            raise ValueError(
                "reward_clip must be positive"
            )

    def train(
        self,
        mode: bool = True,
    ) -> "CuriosityAgent":
        super().train(mode)

        # Keep the fixed target in evaluation mode.
        self.rnd_target.eval()

        return self

    def _validate_transition(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> None:
        cfg = self.config

        if (
            state.ndim != 2
            or state.shape[-1] != cfg.latent_dim
        ):
            raise ValueError(
                "state must be [B,latent_dim]"
            )

        if next_state.shape != state.shape:
            raise ValueError(
                "next_state shape mismatch"
            )

        if action.shape != (
            state.shape[0],
            cfg.action_dim,
        ):
            raise ValueError(
                "action must be [B,action_dim]"
            )

        if not (
            state.device
            == action.device
            == next_state.device
        ):
            raise ValueError(
                "Transition tensors must share device"
            )

        if not (
            state.dtype
            == action.dtype
            == next_state.dtype
        ):
            raise ValueError(
                "Transition tensors must share dtype"
            )

        for name, tensor in (
            ("state", state),
            ("action", action),
            ("next_state", next_state),
        ):
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

    def prediction_error(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Forward dynamics prediction error.

        Returns:
            [B]
        """

        predicted = self.forward_model(
            state,
            action,
        )

        return (
            predicted - next_state
        ).pow(2).mean(dim=-1)

    def rnd_error(
        self,
        next_state: Tensor,
    ) -> Tensor:
        """
        Random Network Distillation error.

        Returns:
            [B]
        """

        with torch.no_grad():
            target_features = self.rnd_target(
                next_state
            )

        predicted_features = (
            self.rnd_predictor(next_state)
        )

        return (
            predicted_features - target_features
        ).pow(2).mean(dim=-1)

    def compute_intrinsic_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        update_statistics: bool = False,
        update_memory: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute curiosity components.

        Episodic novelty is evaluated before
        optionally adding next_state to memory.

        Returns:
            intrinsic_reward [B]
            prediction_error [B]
            rnd_error [B]
            episodic_novelty [B]
        """

        self._validate_transition(
            state,
            action,
            next_state,
        )

        cfg = self.config

        forward_error = self.prediction_error(
            state,
            action,
            next_state,
        )

        rnd_error = self.rnd_error(
            next_state
        )

        novelty = self.episodic_memory.novelty(
            next_state
        )

        raw_reward = (
            cfg.prediction_reward_weight
            * forward_error
            + cfg.rnd_reward_weight
            * rnd_error
            + cfg.episodic_reward_weight
            * novelty
        )

        if update_statistics:
            self.reward_statistics.update(
                raw_reward
            )

        if cfg.normalize_intrinsic_reward:
            intrinsic = (
                self.reward_statistics.normalize(
                    raw_reward,
                    cfg.normalization_epsilon,
                )
            )
        else:
            intrinsic = raw_reward

        intrinsic = intrinsic.clamp(
            min=0.0,
            max=cfg.reward_clip,
        )

        if update_memory:
            self.episodic_memory.add(
                next_state
            )

        return (
            intrinsic,
            forward_error,
            rnd_error,
            novelty,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        extrinsic_reward: Optional[
            Tensor
        ] = None,
        update_statistics: bool = False,
        update_memory: bool = False,
    ) -> CuriosityOutput:
        """
        Calculate combined exploration reward.
        """

        (
            intrinsic,
            prediction_error,
            rnd_error,
            novelty,
        ) = self.compute_intrinsic_reward(
            state,
            action,
            next_state,
            update_statistics=update_statistics,
            update_memory=update_memory,
        )

        if extrinsic_reward is None:
            extrinsic_reward = torch.zeros_like(
                intrinsic
            )
        else:
            if extrinsic_reward.shape != (
                state.shape[0],
            ):
                raise ValueError(
                    "extrinsic_reward must be [B]"
                )

            if (
                extrinsic_reward.device
                != state.device
            ):
                raise ValueError(
                    "extrinsic_reward device mismatch"
                )

            if not torch.isfinite(
                extrinsic_reward
            ).all().item():
                raise ValueError(
                    "Non-finite extrinsic reward"
                )

        total_reward = (
            self.config.extrinsic_reward_scale
            * extrinsic_reward
            + self.config.intrinsic_reward_scale
            * intrinsic
        )

        return CuriosityOutput(
            prediction_error=prediction_error,
            rnd_error=rnd_error,
            episodic_novelty=novelty,
            intrinsic_reward=intrinsic,
            total_reward=total_reward,
        )

    def training_step(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Dict[str, float]:
        """
        Train forward curiosity and RND predictor.

        The fixed RND target is never optimized.

        Intrinsic reward is computed before
        updating curiosity networks.
        """

        self._validate_transition(
            state,
            action,
            next_state,
        )

        self.train()

        self.optimizer.zero_grad(
            set_to_none=True
        )

        forward_loss = self.prediction_error(
            state,
            action,
            next_state,
        ).mean()

        rnd_loss = self.rnd_error(
            next_state
        ).mean()

        loss = forward_loss + rnd_loss

        loss.backward()

        if self.config.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                list(
                    self.forward_model.parameters()
                )
                + list(
                    self.rnd_predictor.parameters()
                ),
                self.config.gradient_clip_norm,
            )

        self.optimizer.step()

        return {
            "loss": float(loss.detach()),
            "forward_loss": float(
                forward_loss.detach()
            ),
            "rnd_loss": float(
                rnd_loss.detach()
            ),
        }

    def epsilon(
        self,
    ) -> float:
        """
        Linear epsilon decay.
        """

        cfg = self.config

        progress = min(
            self.exploration_steps
            / cfg.epsilon_decay_steps,
            1.0,
        )

        return (
            cfg.epsilon_start
            + progress
            * (
                cfg.epsilon_end
                - cfg.epsilon_start
            )
        )

    @torch.no_grad()
    def select_action(
        self,
        action_values: Tensor,
        valid_action_mask: Optional[
            Tensor
        ] = None,
        explore: bool = True,
    ) -> Tensor:
        """
        Epsilon-greedy discrete action selection.

        action_values:
            [B,num_actions]

        valid_action_mask:
            Optional boolean [B,num_actions].
            True means the action is allowed.

        Returns:
            action_indices [B]

        The external policy/value model provides
        action_values. This class does not learn
        an action-value function by itself.
        """

        if (
            action_values.ndim != 2
            or action_values.shape[1] < 1
        ):
            raise ValueError(
                "action_values must be [B,N]"
            )

        if not torch.isfinite(
            action_values
        ).all().item():
            raise ValueError(
                "action_values must be finite"
            )

        if valid_action_mask is None:
            valid = torch.ones_like(
                action_values,
                dtype=torch.bool,
            )
        else:
            if (
                valid_action_mask.shape
                != action_values.shape
            ):
                raise ValueError(
                    "Invalid action mask shape"
                )

            if valid_action_mask.dtype != torch.bool:
                raise TypeError(
                    "Action mask must be boolean"
                )

            valid = valid_action_mask

        if not valid.any(dim=-1).all().item():
            raise ValueError(
                "Every batch item requires "
                "at least one valid action"
            )

        masked_values = (
            action_values.masked_fill(
                ~valid,
                float("-inf"),
            )
        )

        greedy_actions = (
            masked_values.argmax(dim=-1)
        )

        if not explore:
            return greedy_actions

        epsilon = self.epsilon()

        random_scores = torch.rand(
            action_values.shape,
            device=action_values.device,
        )

        random_scores = (
            random_scores.masked_fill(
                ~valid,
                -1.0,
            )
        )

        random_actions = (
            random_scores.argmax(dim=-1)
        )

        explore_mask = (
            torch.rand(
                action_values.shape[0],
                device=action_values.device,
            )
            < epsilon
        )

        selected = torch.where(
            explore_mask,
            random_actions,
            greedy_actions,
        )

        self.exploration_steps += 1

        return selected

    def reset_episode(self) -> None:
        """
        Clear episodic novelty memory.

        Call at the beginning of each episode.
        """

        self.episodic_memory.reset()

    @torch.no_grad()
    def evaluate_curiosity(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> CuriosityOutput:
        """
        Evaluate curiosity without updating
        memory or running statistics.
        """

        was_training = self.training
        self.eval()

        try:
            return self.forward(
                state,
                action,
                next_state,
                update_statistics=False,
                update_memory=False,
            )
        finally:
            self.train(was_training)

    def save_checkpoint(
        self,
        path: str | Path,
    ) -> None:
        """
        Save model, optimizer, running statistics,
        exploration steps and episodic memory.
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
                    self.optimizer.state_dict()
                ),
                "exploration_steps": (
                    self.exploration_steps
                ),
                "episodic_states": (
                    None
                    if self.episodic_memory.states
                    is None
                    else self.episodic_memory.states
                    .detach()
                    .cpu()
                ),
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        load_optimizer: bool = True,
    ) -> "CuriosityAgent":
        """
        Load a trusted checkpoint.

        Requires a PyTorch version supporting
        weights_only=True.
        """

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )

        config = CuriosityAgentConfig(
            **checkpoint["config"]
        )

        agent = cls(config)

        agent.load_state_dict(
            checkpoint["model_state"]
        )

        if load_optimizer:
            agent.optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        agent.exploration_steps = int(
            checkpoint["exploration_steps"]
        )

        episodic_states = checkpoint.get(
            "episodic_states"
        )

        if episodic_states is not None:
            agent.episodic_memory.states = (
                episodic_states
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

    config = CuriosityAgentConfig(
        latent_dim=128,
        action_dim=32,
        hidden_dim=256,
        rnd_feature_dim=128,
    )

    agent = CuriosityAgent(
        config
    ).to(device)

    batch_size = 8

    states = torch.randn(
        batch_size,
        config.latent_dim,
        device=device,
    )

    actions = torch.randn(
        batch_size,
        config.action_dim,
        device=device,
    )

    next_states = torch.randn(
        batch_size,
        config.latent_dim,
        device=device,
    )

    extrinsic_rewards = torch.randn(
        batch_size,
        device=device,
    )

    # Curiosity reward calculation.
    output = agent(
        state=states,
        action=actions,
        next_state=next_states,
        extrinsic_reward=extrinsic_rewards,
        update_statistics=True,
        update_memory=True,
    )

    print(
        "Intrinsic reward:",
        output.intrinsic_reward.shape,
    )

    print(
        "Total reward:",
        output.total_reward.shape,
    )

    # Train curiosity networks.
    metrics = agent.training_step(
        states,
        actions,
        next_states,
    )

    print("Training metrics:", metrics)

    # Discrete action selection.
    action_values = torch.randn(
        batch_size,
        6,
        device=device,
    )

    action_indices = agent.select_action(
        action_values,
        explore=True,
    )

    print(
        "Selected actions:",
        action_indices,
    )

    print(
        "Exploration epsilon:",
        agent.epsilon(),
    )

    # Save checkpoint.
    agent.save_checkpoint(
        "checkpoints/curiosity_agent.pt"
    )

    # Load checkpoint.
    restored = CuriosityAgent.load_checkpoint(
        "checkpoints/curiosity_agent.pt",
        map_location=device,
    ).to(device)

    print(
        "Restored exploration steps:",
        restored.exploration_steps,
    )
