
"""
NeuroWorld AI
File: neuroworld/rl/intrinsic_reward.py

Intrinsic Motivation and Curiosity Reward Engine.

Features
--------
1. Random Network Distillation (RND)
2. Intrinsic Curiosity Module (ICM)
3. Forward-model prediction error
4. Episodic KNN novelty
5. State-count exploration
6. World-model uncertainty reward
7. Running reward normalization
8. Reward clipping
9. Combined intrinsic/extrinsic rewards
10. Gradient clipping
11. PyTorch training
12. Checkpoint save/load
13. Reproducible demonstration

Dependencies:
    pip install torch numpy

Conventions:
    states:       [B, state_dim]
    actions:      [B, action_dim]
    next_states:  [B, state_dim]
    rewards:      [B]

This module assumes continuous actions.

For discrete actions, encode actions as one-hot
vectors before passing them to ICM.

Intrinsic rewards are exploration signals,
not measures of safety or real-world success.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# EXCEPTIONS
# ============================================================

class IntrinsicRewardError(Exception):
    """Base intrinsic reward exception."""


class IntrinsicRewardValidationError(
    IntrinsicRewardError
):
    """Invalid input or configuration."""


# ============================================================
# REWARD TYPES
# ============================================================

class IntrinsicRewardType(str, Enum):
    RND = "rnd"
    ICM = "icm"
    PREDICTION_ERROR = "prediction_error"
    EPISODIC_NOVELTY = "episodic_novelty"
    STATE_COUNT = "state_count"
    UNCERTAINTY = "uncertainty"
    COMBINED = "combined"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class IntrinsicRewardConfig:
    state_dim: int = 128
    action_dim: int = 8

    hidden_dim: int = 256
    feature_dim: int = 128

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    rnd_weight: float = 1.0
    icm_weight: float = 1.0
    prediction_error_weight: float = 0.5
    episodic_novelty_weight: float = 0.5
    state_count_weight: float = 0.25
    uncertainty_weight: float = 0.25

    intrinsic_scale: float = 0.1
    extrinsic_scale: float = 1.0

    icm_forward_loss_weight: float = 0.2
    icm_inverse_loss_weight: float = 0.8

    episodic_memory_capacity: int = 10000
    novelty_k: int = 5
    novelty_distance_scale: float = 1.0

    state_hash_precision: int = 2

    normalize_rewards: bool = True
    reward_clip_min: float = 0.0
    reward_clip_max: float = 10.0

    normalization_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "state_dim",
            "action_dim",
            "hidden_dim",
            "feature_dim",
            "episodic_memory_capacity",
            "novelty_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive"
            )

        if self.novelty_distance_scale <= 0:
            raise ValueError(
                "novelty_distance_scale "
                "must be positive"
            )

        if self.normalization_epsilon <= 0:
            raise ValueError(
                "normalization_epsilon "
                "must be positive"
            )

        if (
            self.reward_clip_min
            >= self.reward_clip_max
        ):
            raise ValueError(
                "Invalid reward clipping range"
            )

        for name in (
            "rnd_weight",
            "icm_weight",
            "prediction_error_weight",
            "episodic_novelty_weight",
            "state_count_weight",
            "uncertainty_weight",
            "intrinsic_scale",
            "extrinsic_scale",
            "icm_forward_loss_weight",
            "icm_inverse_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} cannot be negative"
                )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_features(
    features: Tensor,
    dimension: int,
    name: str,
) -> Tensor:
    if not isinstance(features, Tensor):
        raise IntrinsicRewardValidationError(
            f"{name} must be a Tensor"
        )

    if features.ndim == 1:
        features = features.unsqueeze(0)

    if (
        features.ndim != 2
        or features.shape[-1] != dimension
    ):
        raise IntrinsicRewardValidationError(
            f"{name} must have shape "
            f"[B,{dimension}]"
        )

    if not torch.is_floating_point(features):
        raise IntrinsicRewardValidationError(
            f"{name} must be floating point"
        )

    if not torch.isfinite(features).all():
        raise IntrinsicRewardValidationError(
            f"{name} contains non-finite values"
        )

    return features


def validate_transition(
    states: Tensor,
    actions: Tensor,
    next_states: Tensor,
    config: IntrinsicRewardConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    states = validate_features(
        states,
        config.state_dim,
        "states",
    )

    actions = validate_features(
        actions,
        config.action_dim,
        "actions",
    )

    next_states = validate_features(
        next_states,
        config.state_dim,
        "next_states",
    )

    if not (
        states.shape[0]
        == actions.shape[0]
        == next_states.shape[0]
    ):
        raise IntrinsicRewardValidationError(
            "Transition batch sizes must match"
        )

    return states, actions, next_states


# ============================================================
# NETWORK FACTORY
# ============================================================

def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(
            input_dim,
            hidden_dim,
        ),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),

        nn.Linear(
            hidden_dim,
            hidden_dim,
        ),
        nn.SiLU(),

        nn.Linear(
            hidden_dim,
            output_dim,
        ),
    )


# ============================================================
# RUNNING REWARD NORMALIZER
# ============================================================

class RunningRewardNormalizer:
    """
    Tracks reward mean and variance using
    parallel Welford updates.

    Returns nonnegative normalized reward
    magnitudes based on running standard
    deviation.

    Does not subtract the mean, preserving
    nonnegative curiosity rewards.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
    ) -> None:
        self.epsilon = epsilon

        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(
        self,
        values: Tensor,
    ) -> None:
        array = (
            values.detach()
            .float()
            .reshape(-1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if array.size == 0:
            return

        batch_count = int(
            array.size
        )

        batch_mean = float(
            array.mean()
        )

        batch_m2 = float(
            np.sum(
                (array - batch_mean) ** 2
            )
        )

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total_count = (
            self.count + batch_count
        )

        delta = (
            batch_mean - self.mean
        )

        self.mean += (
            delta
            * batch_count
            / total_count
        )

        self.m2 += (
            batch_m2
            + delta**2
            * self.count
            * batch_count
            / total_count
        )

        self.count = total_count

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 1.0

        return max(
            self.m2 / self.count,
            self.epsilon,
        )

    @property
    def std(self) -> float:
        return math.sqrt(
            self.variance
        )

    def normalize(
        self,
        values: Tensor,
        *,
        update: bool = True,
    ) -> Tensor:
        if update:
            self.update(values)

        return values / max(
            self.std,
            self.epsilon,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        self.epsilon = float(
            state["epsilon"]
        )

        self.count = int(
            state["count"]
        )

        self.mean = float(
            state["mean"]
        )

        self.m2 = float(
            state["m2"]
        )


# ============================================================
# RANDOM NETWORK DISTILLATION
# ============================================================

class RandomNetworkDistillation(nn.Module):
    """
    RND curiosity:

        intrinsic_reward =
            || predictor(s') - target(s') ||^2

    The target network is randomly
    initialized and frozen.

    The predictor learns to approximate
    the fixed target.

    Unfamiliar states generally produce
    higher prediction errors.
    """

    def __init__(
        self,
        config: IntrinsicRewardConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.target = build_mlp(
            config.state_dim,
            config.hidden_dim,
            config.feature_dim,
        )

        self.predictor = build_mlp(
            config.state_dim,
            config.hidden_dim,
            config.feature_dim,
        )

        for parameter in (
            self.target.parameters()
        ):
            parameter.requires_grad_(
                False
            )

        self.target.eval()

    def train(
        self,
        mode: bool = True,
    ) -> "RandomNetworkDistillation":
        super().train(mode)

        # Keep the random target frozen.
        self.target.eval()

        return self

    def forward(
        self,
        next_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        next_states = validate_features(
            next_states,
            self.config.state_dim,
            "next_states",
        )

        with torch.no_grad():
            target_features = self.target(
                next_states
            )

        predicted_features = self.predictor(
            next_states
        )

        errors = (
            predicted_features
            - target_features
        ).square().mean(
            dim=-1
        )

        return (
            errors,
            errors.mean(),
        )

    @torch.no_grad()
    def intrinsic_reward(
        self,
        next_states: Tensor,
    ) -> Tensor:
        errors, _ = self.forward(
            next_states
        )

        return errors


# ============================================================
# INTRINSIC CURIOSITY MODULE
# ============================================================

class IntrinsicCuriosityModule(nn.Module):
    """
    ICM:

        phi(s) = feature_encoder(s)

        forward_model(
            phi(s), action
        ) -> predicted phi(s')

        inverse_model(
            phi(s), phi(s')
        ) -> predicted action

    Intrinsic reward is the forward-model
    feature prediction error.
    """

    def __init__(
        self,
        config: IntrinsicRewardConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.feature_encoder = build_mlp(
            config.state_dim,
            config.hidden_dim,
            config.feature_dim,
        )

        self.forward_model = build_mlp(
            (
                config.feature_dim
                + config.action_dim
            ),
            config.hidden_dim,
            config.feature_dim,
        )

        self.inverse_model = build_mlp(
            config.feature_dim * 2,
            config.hidden_dim,
            config.action_dim,
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> dict[str, Tensor]:
        states, actions, next_states = (
            validate_transition(
                states,
                actions,
                next_states,
                self.config,
            )
        )

        current_features = (
            self.feature_encoder(
                states
            )
        )

        next_features = (
            self.feature_encoder(
                next_states
            )
        )

        forward_input = torch.cat(
            [
                current_features,
                actions,
            ],
            dim=-1,
        )

        predicted_next_features = (
            self.forward_model(
                forward_input
            )
        )

        inverse_input = torch.cat(
            [
                current_features,
                next_features,
            ],
            dim=-1,
        )

        predicted_actions = (
            self.inverse_model(
                inverse_input
            )
        )

        # Stop gradients through the target
        # features for the forward objective.
        forward_errors = (
            predicted_next_features
            - next_features.detach()
        ).square().mean(
            dim=-1
        )

        inverse_errors = (
            predicted_actions - actions
        ).square().mean(
            dim=-1
        )

        forward_loss = (
            forward_errors.mean()
        )

        inverse_loss = (
            inverse_errors.mean()
        )

        total_loss = (
            self.config.icm_forward_loss_weight
            * forward_loss
            + self.config.icm_inverse_loss_weight
            * inverse_loss
        )

        return {
            "intrinsic_reward": (
                forward_errors
            ),
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
            "total_loss": total_loss,
            "predicted_actions": (
                predicted_actions
            ),
            "predicted_next_features": (
                predicted_next_features
            ),
        }


# ============================================================
# STATE PREDICTION ERROR MODEL
# ============================================================

class StatePredictionErrorModel(nn.Module):
    """
    Learns:

        f(state, action) -> next_state

    Curiosity reward is prediction error
    in state space.
    """

    def __init__(
        self,
        config: IntrinsicRewardConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.network = build_mlp(
            (
                config.state_dim
                + config.action_dim
            ),
            config.hidden_dim,
            config.state_dim,
        )

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        states, actions, next_states = (
            validate_transition(
                states,
                actions,
                next_states,
                self.config,
            )
        )

        features = torch.cat(
            [
                states,
                actions,
            ],
            dim=-1,
        )

        # Residual state prediction.
        predicted_delta = self.network(
            features
        )

        predicted_next_state = (
            states + predicted_delta
        )

        errors = (
            predicted_next_state
            - next_states
        ).square().mean(
            dim=-1
        )

        return (
            errors,
            errors.mean(),
            predicted_next_state,
        )


# ============================================================
# EPISODIC NOVELTY MEMORY
# ============================================================

class EpisodicNoveltyMemory:
    """
    K-nearest-neighbor novelty reward.

    Stores embeddings for one episode.

    Novelty increases with distance
    from previously visited states.

    Reset at episode boundaries.
    """

    def __init__(
        self,
        capacity: int = 10000,
        k: int = 5,
        distance_scale: float = 1.0,
    ) -> None:
        if capacity <= 0 or k <= 0:
            raise ValueError(
                "Invalid novelty memory configuration"
            )

        if distance_scale <= 0:
            raise ValueError(
                "distance_scale must be positive"
            )

        self.capacity = capacity
        self.k = k
        self.distance_scale = distance_scale

        self.memory: deque[np.ndarray] = (
            deque(maxlen=capacity)
        )

    def reset(self) -> None:
        self.memory.clear()

    def __len__(self) -> int:
        return len(
            self.memory
        )

    def novelty(
        self,
        embedding: Any,
        *,
        update: bool = True,
    ) -> float:
        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if not np.isfinite(embedding).all():
            raise IntrinsicRewardValidationError(
                "Embedding contains non-finite values"
            )

        if len(self.memory) == 0:
            novelty_reward = 1.0

        else:
            stored = np.stack(
                list(self.memory)
            )

            if stored.shape[1] != embedding.size:
                raise IntrinsicRewardValidationError(
                    "Embedding dimension mismatch"
                )

            distances = np.linalg.norm(
                stored - embedding[None, :],
                axis=1,
            )

            nearest_count = min(
                self.k,
                distances.size,
            )

            nearest = np.partition(
                distances,
                nearest_count - 1,
            )[:nearest_count]

            mean_distance = float(
                nearest.mean()
            )

            novelty_reward = (
                mean_distance
                / (
                    mean_distance
                    + self.distance_scale
                )
            )

        if update:
            self.memory.append(
                embedding.copy()
            )

        return float(
            novelty_reward
        )

    def batch_novelty(
        self,
        embeddings: Tensor,
        *,
        update: bool = True,
    ) -> Tensor:
        if embeddings.ndim != 2:
            raise IntrinsicRewardValidationError(
                "Expected embeddings [B,D]"
            )

        rewards = []

        for embedding in (
            embeddings.detach()
            .cpu()
            .numpy()
        ):
            rewards.append(
                self.novelty(
                    embedding,
                    update=update,
                )
            )

        return torch.tensor(
            rewards,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )


# ============================================================
# STATE COUNT EXPLORATION
# ============================================================

class StateCountExploration:
    """
    Hash-based pseudo-count exploration.

    State vectors are rounded to a
    configurable precision.

    Reward:
        1 / sqrt(visit_count)

    This is a simple discretized count,
    not a learned density estimator.
    """

    def __init__(
        self,
        precision: int = 2,
    ) -> None:
        if precision < 0:
            raise ValueError(
                "precision must be nonnegative"
            )

        self.precision = precision

        self.counts: dict[
            tuple[float, ...],
            int,
        ] = {}

    def reset(self) -> None:
        self.counts.clear()

    def _key(
        self,
        state: Any,
    ) -> tuple[float, ...]:
        array = np.asarray(
            state,
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(array).all():
            raise IntrinsicRewardValidationError(
                "State contains non-finite values"
            )

        return tuple(
            np.round(
                array,
                decimals=self.precision,
            ).tolist()
        )

    def reward(
        self,
        state: Any,
        *,
        update: bool = True,
    ) -> float:
        key = self._key(
            state
        )

        current_count = (
            self.counts.get(
                key,
                0,
            )
        )

        next_count = (
            current_count + 1
        )

        if update:
            self.counts[key] = next_count

        return float(
            1.0 / math.sqrt(next_count)
        )

    def batch_reward(
        self,
        states: Tensor,
        *,
        update: bool = True,
    ) -> Tensor:
        rewards = []

        for state in (
            states.detach()
            .cpu()
            .numpy()
        ):
            rewards.append(
                self.reward(
                    state,
                    update=update,
                )
            )

        return torch.tensor(
            rewards,
            dtype=states.dtype,
            device=states.device,
        )


# ============================================================
# REWARD OUTPUT
# ============================================================

@dataclass
class IntrinsicRewardOutput:
    intrinsic_reward: Tensor
    extrinsic_reward: Tensor
    combined_reward: Tensor

    rnd_reward: Tensor
    icm_reward: Tensor
    prediction_error_reward: Tensor
    episodic_novelty_reward: Tensor
    state_count_reward: Tensor
    uncertainty_reward: Tensor

    def metrics(self) -> dict[str, float]:
        tensors = {
            "intrinsic_reward": (
                self.intrinsic_reward
            ),
            "extrinsic_reward": (
                self.extrinsic_reward
            ),
            "combined_reward": (
                self.combined_reward
            ),
            "rnd_reward": (
                self.rnd_reward
            ),
            "icm_reward": (
                self.icm_reward
            ),
            "prediction_error_reward": (
                self.prediction_error_reward
            ),
            "episodic_novelty_reward": (
                self.episodic_novelty_reward
            ),
            "state_count_reward": (
                self.state_count_reward
            ),
            "uncertainty_reward": (
                self.uncertainty_reward
            ),
        }

        return {
            name: float(
                value.detach().mean().item()
            )
            for name, value in tensors.items()
        }


# ============================================================
# INTRINSIC REWARD ENGINE
# ============================================================

class IntrinsicRewardEngine(nn.Module):
    """
    Combines multiple exploration rewards.

    Neural modules:
        RND
        ICM
        State prediction model

    Non-neural modules:
        Episodic KNN novelty
        State count exploration
        Running reward normalization

    Reward components are computed without
    gradients during reward generation.

    Training is handled separately by
    IntrinsicRewardTrainer.
    """

    def __init__(
        self,
        config: IntrinsicRewardConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = (
            config or IntrinsicRewardConfig()
        )

        self.rnd = RandomNetworkDistillation(
            self.config
        )

        self.icm = IntrinsicCuriosityModule(
            self.config
        )

        self.prediction_model = (
            StatePredictionErrorModel(
                self.config
            )
        )

        self.episodic_memory = (
            EpisodicNoveltyMemory(
                capacity=(
                    self.config.episodic_memory_capacity
                ),
                k=self.config.novelty_k,
                distance_scale=(
                    self.config.novelty_distance_scale
                ),
            )
        )

        self.state_counter = (
            StateCountExploration(
                precision=(
                    self.config.state_hash_precision
                )
            )
        )

        self.normalizers = {
            name: RunningRewardNormalizer(
                self.config.normalization_epsilon
            )
            for name in (
                "rnd",
                "icm",
                "prediction_error",
                "episodic_novelty",
                "state_count",
                "uncertainty",
            )
        }

    def reset_episode(
        self,
        *,
        reset_state_counts: bool = False,
    ) -> None:
        self.episodic_memory.reset()

        if reset_state_counts:
            self.state_counter.reset()

    @torch.no_grad()
    def compute(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
        *,
        extrinsic_rewards: Tensor | None = None,
        uncertainties: Tensor | None = None,
        reward_type: IntrinsicRewardType = (
            IntrinsicRewardType.COMBINED
        ),
        update_memory: bool = True,
        update_normalizers: bool = True,
    ) -> IntrinsicRewardOutput:
        states, actions, next_states = (
            validate_transition(
                states,
                actions,
                next_states,
                self.config,
            )
        )

        reward_type = IntrinsicRewardType(
            reward_type
        )

        batch_size = states.shape[0]

        zeros = torch.zeros(
            batch_size,
            device=states.device,
            dtype=states.dtype,
        )

        if extrinsic_rewards is None:
            extrinsic_rewards = zeros.clone()

        else:
            extrinsic_rewards = (
                extrinsic_rewards.to(
                    device=states.device,
                    dtype=states.dtype,
                ).reshape(-1)
            )

            if (
                extrinsic_rewards.shape[0]
                != batch_size
            ):
                raise IntrinsicRewardValidationError(
                    "Extrinsic reward batch mismatch"
                )

            if not torch.isfinite(
                extrinsic_rewards
            ).all():
                raise IntrinsicRewardValidationError(
                    "Non-finite extrinsic rewards"
                )

        if uncertainties is None:
            uncertainties = zeros.clone()

        else:
            uncertainties = uncertainties.to(
                device=states.device,
                dtype=states.dtype,
            ).reshape(-1)

            if uncertainties.shape[0] != batch_size:
                raise IntrinsicRewardValidationError(
                    "Uncertainty batch mismatch"
                )

            if (
                not torch.isfinite(
                    uncertainties
                ).all()
                or torch.any(uncertainties < 0)
            ):
                raise IntrinsicRewardValidationError(
                    "Uncertainty must be finite "
                    "and nonnegative"
                )

        # ----------------------------------------------------
        # RND
        # ----------------------------------------------------

        rnd_reward = (
            self.rnd.intrinsic_reward(
                next_states
            )
        )

        # ----------------------------------------------------
        # ICM
        # ----------------------------------------------------

        icm_result = self.icm(
            states,
            actions,
            next_states,
        )

        icm_reward = (
            icm_result["intrinsic_reward"]
        )

        # ----------------------------------------------------
        # PREDICTION ERROR
        # ----------------------------------------------------

        (
            prediction_error_reward,
            _,
            _,
        ) = self.prediction_model(
            states,
            actions,
            next_states,
        )

        # ----------------------------------------------------
        # EPISODIC NOVELTY
        # ----------------------------------------------------

        novelty_embeddings = (
            self.icm.feature_encoder(
                next_states
            )
        )

        episodic_novelty_reward = (
            self.episodic_memory.batch_novelty(
                novelty_embeddings,
                update=update_memory,
            )
        )

        # ----------------------------------------------------
        # STATE COUNT
        # ----------------------------------------------------

        state_count_reward = (
            self.state_counter.batch_reward(
                next_states,
                update=update_memory,
            )
        )

        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        uncertainty_reward = (
            uncertainties.clone()
        )

        components = {
            "rnd": rnd_reward,
            "icm": icm_reward,
            "prediction_error": (
                prediction_error_reward
            ),
            "episodic_novelty": (
                episodic_novelty_reward
            ),
            "state_count": (
                state_count_reward
            ),
            "uncertainty": (
                uncertainty_reward
            ),
        }

        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        if self.config.normalize_rewards:
            components = {
                name: (
                    self.normalizers[name].normalize(
                        values,
                        update=update_normalizers,
                    )
                )
                for name, values in components.items()
            }

        weights = {
            "rnd": self.config.rnd_weight,
            "icm": self.config.icm_weight,
            "prediction_error": (
                self.config.prediction_error_weight
            ),
            "episodic_novelty": (
                self.config.episodic_novelty_weight
            ),
            "state_count": (
                self.config.state_count_weight
            ),
            "uncertainty": (
                self.config.uncertainty_weight
            ),
        }

        # ----------------------------------------------------
        # SELECT REWARD TYPE
        # ----------------------------------------------------

        if reward_type == IntrinsicRewardType.COMBINED:
            intrinsic_reward = sum(
                (
                    weights[name]
                    * values
                    for name, values
                    in components.items()
                ),
                zeros.clone(),
            )

        else:
            selected_name = reward_type.value

            intrinsic_reward = (
                weights[selected_name]
                * components[selected_name]
            )

        intrinsic_reward = torch.clamp(
            intrinsic_reward,
            min=self.config.reward_clip_min,
            max=self.config.reward_clip_max,
        )

        combined_reward = (
            self.config.extrinsic_scale
            * extrinsic_rewards
            + self.config.intrinsic_scale
            * intrinsic_reward
        )

        return IntrinsicRewardOutput(
            intrinsic_reward=intrinsic_reward,
            extrinsic_reward=extrinsic_rewards,
            combined_reward=combined_reward,
            rnd_reward=components["rnd"],
            icm_reward=components["icm"],
            prediction_error_reward=(
                components["prediction_error"]
            ),
            episodic_novelty_reward=(
                components["episodic_novelty"]
            ),
            state_count_reward=(
                components["state_count"]
            ),
            uncertainty_reward=(
                components["uncertainty"]
            ),
        )


# ============================================================
# INTRINSIC REWARD TRAINER
# ============================================================

class IntrinsicRewardTrainer:
    """
    Trains neural curiosity components.

    Loss:
        RND prediction loss
        ICM forward/inverse loss
        State prediction loss

    The optimizer updates:
        RND predictor
        ICM networks
        State prediction model

    RND target parameters remain frozen.
    """

    def __init__(
        self,
        engine: IntrinsicRewardEngine,
    ) -> None:
        self.engine = engine
        self.config = engine.config

        parameters = [
            parameter
            for parameter in engine.parameters()
            if parameter.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.training_steps = 0

    def train_step(
        self,
        states: Tensor,
        actions: Tensor,
        next_states: Tensor,
    ) -> dict[str, float]:
        self.engine.train()

        states, actions, next_states = (
            validate_transition(
                states,
                actions,
                next_states,
                self.config,
            )
        )

        # RND loss.
        _, rnd_loss = self.engine.rnd(
            next_states
        )

        # ICM loss.
        icm_output = self.engine.icm(
            states,
            actions,
            next_states,
        )

        icm_loss = (
            icm_output["total_loss"]
        )

        # State prediction loss.
        (
            _,
            prediction_loss,
            _,
        ) = self.engine.prediction_model(
            states,
            actions,
            next_states,
        )

        total_loss = (
            rnd_loss
            + icm_loss
            + prediction_loss
        )

        if not torch.isfinite(total_loss):
            raise IntrinsicRewardError(
                "Non-finite training loss"
            )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        total_loss.backward()

        grad_norm = (
            nn.utils.clip_grad_norm_(
                self.engine.parameters(),
                self.config.max_grad_norm,
            )
        )

        self.optimizer.step()

        self.training_steps += 1

        return {
            "total_loss": float(
                total_loss.detach().item()
            ),
            "rnd_loss": float(
                rnd_loss.detach().item()
            ),
            "icm_loss": float(
                icm_loss.detach().item()
            ),
            "icm_forward_loss": float(
                icm_output[
                    "forward_loss"
                ].detach().item()
            ),
            "icm_inverse_loss": float(
                icm_output[
                    "inverse_loss"
                ].detach().item()
            ),
            "prediction_loss": float(
                prediction_loss.detach().item()
            ),
            "grad_norm": float(
                grad_norm.detach().item()
            ),
            "training_steps": (
                self.training_steps
            ),
        }


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def save_intrinsic_reward(
    engine: IntrinsicRewardEngine,
    path: str | Path,
    *,
    trainer: IntrinsicRewardTrainer | None = None,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload: dict[str, Any] = {
        "config": asdict(
            engine.config
        ),
        "model_state": (
            engine.state_dict()
        ),
        "normalizers": {
            name: normalizer.state_dict()
            for name, normalizer
            in engine.normalizers.items()
        },
        "state_counts": (
            engine.state_counter.counts
        ),
        "episodic_memory": list(
            engine.episodic_memory.memory
        ),
    }

    if trainer is not None:
        payload["optimizer_state"] = (
            trainer.optimizer.state_dict()
        )

        payload["training_steps"] = (
            trainer.training_steps
        )

    torch.save(
        payload,
        path,
    )

    return path


def load_intrinsic_reward(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    restore_trainer: bool = False,
) -> (
    IntrinsicRewardEngine
    | tuple[
        IntrinsicRewardEngine,
        IntrinsicRewardTrainer,
    ]
):
    """
    Load a trusted local checkpoint.

    This checkpoint contains Python objects
    for novelty memory and state counts.
    Only load checkpoints you trust.
    """

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    config = IntrinsicRewardConfig(
        **checkpoint["config"]
    )

    engine = IntrinsicRewardEngine(
        config
    ).to(device)

    engine.load_state_dict(
        checkpoint["model_state"]
    )

    for name, state in (
        checkpoint["normalizers"].items()
    ):
        engine.normalizers[
            name
        ].load_state_dict(
            state
        )

    engine.state_counter.counts = dict(
        checkpoint["state_counts"]
    )

    engine.episodic_memory.reset()

    for embedding in checkpoint[
        "episodic_memory"
    ]:
        engine.episodic_memory.memory.append(
            np.asarray(
                embedding,
                dtype=np.float32,
            )
        )

    engine.eval()

    if not restore_trainer:
        return engine

    trainer = IntrinsicRewardTrainer(
        engine
    )

    if "optimizer_state" in checkpoint:
        trainer.optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

    trainer.training_steps = int(
        checkpoint.get(
            "training_steps",
            0,
        )
    )

    return (
        engine,
        trainer,
    )


# ============================================================
# FACTORY
# ============================================================

def create_intrinsic_reward_engine(
    config: IntrinsicRewardConfig | None = None,
) -> IntrinsicRewardEngine:
    return IntrinsicRewardEngine(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 65)
    print(
        "NEUROWORLD AI - INTRINSIC REWARD ENGINE"
    )
    print("=" * 65)

    config = IntrinsicRewardConfig(
        state_dim=32,
        action_dim=4,
        hidden_dim=128,
        feature_dim=64,
        intrinsic_scale=0.1,
        normalize_rewards=True,
    )

    engine = create_intrinsic_reward_engine(
        config
    )

    trainer = IntrinsicRewardTrainer(
        engine
    )

    batch_size = 16

    states = torch.randn(
        batch_size,
        config.state_dim,
    )

    actions = torch.randn(
        batch_size,
        config.action_dim,
    )

    next_states = (
        states
        + 0.05
        * torch.randn_like(states)
    )

    extrinsic_rewards = torch.randn(
        batch_size
    )

    uncertainties = (
        torch.rand(batch_size)
        * 0.2
    )

    # --------------------------------------------------------
    # BEFORE TRAINING
    # --------------------------------------------------------

    engine.eval()

    before = engine.compute(
        states,
        actions,
        next_states,
        extrinsic_rewards=extrinsic_rewards,
        uncertainties=uncertainties,
    )

    print(
        "Initial reward metrics:",
        before.metrics(),
    )

    # --------------------------------------------------------
    # TRAIN CURIOSITY MODELS
    # --------------------------------------------------------

    for step in range(10):
        metrics = trainer.train_step(
            states,
            actions,
            next_states,
        )

        if step in (0, 4, 9):
            print(
                f"Training step {step + 1}:",
                metrics,
            )

    # --------------------------------------------------------
    # COMPUTE REWARDS
    # --------------------------------------------------------

    engine.eval()

    output = engine.compute(
        states,
        actions,
        next_states,
        extrinsic_rewards=extrinsic_rewards,
        uncertainties=uncertainties,
    )

    print(
        "Final reward metrics:",
        output.metrics(),
    )

    print(
        "Combined reward shape:",
        output.combined_reward.shape,
    )

    # --------------------------------------------------------
    # EPISODIC MEMORY RESET
    # --------------------------------------------------------

    engine.reset_episode()

    print(
        "Episodic memory size:",
        len(engine.episodic_memory),
    )

    # --------------------------------------------------------
    # SAVE AND LOAD
    # --------------------------------------------------------

    checkpoint = save_intrinsic_reward(
        engine,
        "models/intrinsic_reward_demo.pt",
        trainer=trainer,
    )

    loaded_engine, loaded_trainer = (
        load_intrinsic_reward(
            checkpoint,
            restore_trainer=True,
        )
    )

    print(
        "Loaded engine:",
        type(loaded_engine).__name__,
    )

    print(
        "Restored training steps:",
        loaded_trainer.training_steps,
    )

    print(
        "Intrinsic reward demonstration completed"
    )


if __name__ == "__main__":
    main()
