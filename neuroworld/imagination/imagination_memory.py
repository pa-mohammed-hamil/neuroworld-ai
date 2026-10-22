
"""
NeuroWorld AI
File: neuroworld/imagination/imagination_memory.py

Imagination Memory Engine

Features:
- Store imagined latent trajectories
- Episodic imagination memory
- Reward and uncertainty tracking
- Cosine similarity retrieval
- Top-K memory search
- Reward-based memory ranking
- Uncertainty-aware filtering
- Memory deduplication
- Capacity-based eviction
- Save/load memory checkpoints
- CPU/GPU tensor support

Dependencies:
    pip install torch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import uuid

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class ImaginationMemoryConfig:
    latent_dim: int = 128
    action_dim: int = 32
    capacity: int = 10000

    similarity_threshold: float = 0.995
    max_uncertainty: float = float("inf")

    reward_weight: float = 1.0
    similarity_weight: float = 1.0
    uncertainty_weight: float = 0.1

    eviction_strategy: str = "oldest"
    storage_device: str = "cpu"
    eps: float = 1e-8


@dataclass
class ImaginationEpisode:
    """
    One stored imagined trajectory.

    states:
        [H+1, latent_dim]

    actions:
        [H, action_dim]

    rewards:
        [H]

    uncertainties:
        [H]
    """

    episode_id: str
    states: Tensor
    actions: Tensor
    rewards: Tensor
    uncertainties: Tensor

    score: float
    created_at: float

    source: str = "imagination"
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def initial_state(self) -> Tensor:
        return self.states[0]

    @property
    def final_state(self) -> Tensor:
        return self.states[-1]

    @property
    def horizon(self) -> int:
        return self.actions.shape[0]

    @property
    def mean_uncertainty(self) -> float:
        if self.uncertainties.numel() == 0:
            return 0.0

        return float(
            self.uncertainties.mean().item()
        )

    @property
    def total_reward(self) -> float:
        return float(
            self.rewards.sum().item()
        )


@dataclass
class MemorySearchResult:
    episode: ImaginationEpisode
    similarity: float
    ranking_score: float


class ImaginationMemory:
    """
    In-memory episodic storage for imagined futures.

    Designed for:
    - World-model imagination
    - Model-based reinforcement learning
    - Trajectory reuse
    - Planning memory
    - Counterfactual scenario retrieval

    This implementation is a single-process memory
    store. It is not a distributed vector database.
    """

    def __init__(
        self,
        config: Optional[
            ImaginationMemoryConfig
        ] = None,
    ):
        self.config = (
            config or ImaginationMemoryConfig()
        )

        cfg = self.config

        if cfg.latent_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError(
                "Dimensions must be positive"
            )

        if cfg.capacity <= 0:
            raise ValueError(
                "Capacity must be positive"
            )

        if not -1.0 <= cfg.similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be in [-1,1]"
            )

        if cfg.eviction_strategy not in (
            "oldest",
            "lowest_score",
        ):
            raise ValueError(
                "Unsupported eviction strategy"
            )

        self.device = torch.device(
            cfg.storage_device
        )

        self._episodes: Dict[
            str, ImaginationEpisode
        ] = {}

    def __len__(self) -> int:
        return len(self._episodes)

    def _store_tensor(
        self,
        tensor: Tensor,
    ) -> Tensor:
        """
        Detach from computation graph and copy to
        the configured memory-storage device.
        """

        return tensor.detach().to(
            device=self.device,
            dtype=torch.float32,
        ).contiguous().clone()

    def _validate_episode(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        uncertainties: Tensor,
    ) -> None:

        cfg = self.config

        if states.ndim != 2:
            raise ValueError(
                "states must be [H+1, latent_dim]"
            )

        if actions.ndim != 2:
            raise ValueError(
                "actions must be [H, action_dim]"
            )

        if states.shape[-1] != cfg.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if actions.shape[-1] != cfg.action_dim:
            raise ValueError(
                "Invalid action dimension"
            )

        horizon = actions.shape[0]

        if horizon <= 0:
            raise ValueError(
                "Trajectory horizon must be positive"
            )

        if states.shape[0] != horizon + 1:
            raise ValueError(
                "states length must equal H+1"
            )

        if rewards.shape != (horizon,):
            raise ValueError(
                "rewards must have shape [H]"
            )

        if uncertainties.shape != (horizon,):
            raise ValueError(
                "uncertainties must have shape [H]"
            )

        for name, tensor in (
            ("states", states),
            ("actions", actions),
            ("rewards", rewards),
            ("uncertainties", uncertainties),
        ):
            if not torch.is_floating_point(tensor):
                raise TypeError(
                    f"{name} must be floating point"
                )

            if not torch.isfinite(tensor).all().item():
                raise ValueError(
                    f"{name} contains NaN or Inf"
                )

        if (uncertainties < 0).any().item():
            raise ValueError(
                "Uncertainty values must be non-negative"
            )

    def _find_duplicate(
        self,
        initial_state: Tensor,
    ) -> Optional[str]:
        """
        Approximate duplicate detection using
        initial-state cosine similarity.

        Note:
            This compares only starting states,
            not complete trajectories.
        """

        if not self._episodes:
            return None

        query = F.normalize(
            initial_state.reshape(1, -1),
            dim=-1,
            eps=self.config.eps,
        )

        ids = list(self._episodes.keys())

        stored_states = torch.stack(
            [
                self._episodes[key].initial_state
                for key in ids
            ],
            dim=0,
        )

        stored_states = F.normalize(
            stored_states,
            dim=-1,
            eps=self.config.eps,
        )

        similarities = (
            query @ stored_states.T
        ).squeeze(0)

        best_index = int(
            similarities.argmax().item()
        )

        if (
            similarities[best_index].item()
            >= self.config.similarity_threshold
        ):
            return ids[best_index]

        return None

    def _evict_if_needed(self) -> None:
        """
        Remove episodes until memory is below capacity.
        """

        while len(self._episodes) >= (
            self.config.capacity
        ):
            if self.config.eviction_strategy == "oldest":
                victim = min(
                    self._episodes.values(),
                    key=lambda item: item.created_at,
                )
            else:
                victim = min(
                    self._episodes.values(),
                    key=lambda item: item.score,
                )

            del self._episodes[victim.episode_id]

    def add(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        uncertainties: Tensor,
        score: Optional[float] = None,
        source: str = "imagination",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        deduplicate: bool = False,
    ) -> Optional[str]:
        """
        Store one imagined episode.

        Returns:
            episode ID if stored or updated.
            None if rejected by uncertainty filter.
        """

        self._validate_episode(
            states,
            actions,
            rewards,
            uncertainties,
        )

        mean_uncertainty = float(
            uncertainties.mean().item()
        )

        if mean_uncertainty > (
            self.config.max_uncertainty
        ):
            return None

        if score is None:
            score = float(
                rewards.sum().item()
                - self.config.uncertainty_weight
                * uncertainties.sum().item()
            )

        if not torch.isfinite(
            torch.tensor(score)
        ).item():
            raise ValueError(
                "Score must be finite"
            )

        stored_states = self._store_tensor(states)
        stored_actions = self._store_tensor(actions)
        stored_rewards = self._store_tensor(rewards)
        stored_uncertainties = self._store_tensor(
            uncertainties
        )

        duplicate_id = None

        if deduplicate:
            duplicate_id = self._find_duplicate(
                stored_states[0]
            )

        if duplicate_id is not None:
            existing = self._episodes[duplicate_id]

            # Retain the higher-scoring trajectory
            # for a similar initial state.
            if score <= existing.score:
                return duplicate_id

            del self._episodes[duplicate_id]

        self._evict_if_needed()

        episode_id = uuid.uuid4().hex

        episode = ImaginationEpisode(
            episode_id=episode_id,
            states=stored_states,
            actions=stored_actions,
            rewards=stored_rewards,
            uncertainties=stored_uncertainties,
            score=float(score),
            created_at=time.time(),
            source=source,
            metadata=dict(metadata or {}),
        )

        self._episodes[episode_id] = episode

        return episode_id

    def add_future_output(
        self,
        output: Any,
        source: str = "future_generator",
        deduplicate: bool = False,
    ) -> List[str]:
        """
        Store trajectories from FutureGenerationOutput.

        Expected shapes:
            trajectories [B,K,H+1,D]
            actions [B,K,H,A]
            rewards [B,K,H]
            uncertainties [B,K,H]
            scores [B,K]
        """

        ids = []

        batch_size, num_futures = (
            output.trajectories.shape[:2]
        )

        for batch in range(batch_size):
            for future in range(num_futures):

                episode_id = self.add(
                    states=output.trajectories[
                        batch, future
                    ],
                    actions=output.actions[
                        batch, future
                    ],
                    rewards=output.rewards[
                        batch, future
                    ],
                    uncertainties=output.uncertainties[
                        batch, future
                    ],
                    score=float(
                        output.scores[
                            batch, future
                        ].detach().item()
                    ),
                    source=source,
                    metadata={
                        "batch_index": batch,
                        "future_index": future,
                    },
                    deduplicate=deduplicate,
                )

                if episode_id is not None:
                    ids.append(episode_id)

        return ids

    def add_branches(
        self,
        branches: List[List[Any]],
        source: str = "branch_generator",
        deduplicate: bool = False,
    ) -> List[str]:
        """
        Store output from BranchGenerator.generate().
        """

        ids = []

        for batch_index, batch_branches in enumerate(
            branches
        ):
            for branch_index, branch in enumerate(
                batch_branches
            ):

                episode_id = self.add(
                    states=branch.states,
                    actions=branch.actions,
                    rewards=branch.rewards,
                    uncertainties=branch.uncertainties,
                    score=float(
                        branch.score.detach().item()
                    ),
                    source=source,
                    metadata={
                        "batch_index": batch_index,
                        "branch_index": branch_index,
                    },
                    deduplicate=deduplicate,
                )

                if episode_id is not None:
                    ids.append(episode_id)

        return ids

    def get(
        self,
        episode_id: str,
    ) -> Optional[ImaginationEpisode]:
        """
        Retrieve an episode by ID.
        """

        return self._episodes.get(episode_id)

    def remove(self, episode_id: str) -> bool:
        """
        Delete an episode by ID.
        """

        return (
            self._episodes.pop(
                episode_id, None
            ) is not None
        )

    def clear(self) -> None:
        """
        Remove all stored imagined episodes.
        """

        self._episodes.clear()

    def search(
        self,
        query_state: Tensor,
        top_k: int = 5,
        min_similarity: float = -1.0,
        max_uncertainty: Optional[float] = None,
        source: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """
        Retrieve similar imagined episodes.

        Query:
            [latent_dim]

        Ranking combines:
        - Initial-state cosine similarity
        - Stored trajectory score
        - Mean predicted uncertainty
        """

        if query_state.shape != (
            self.config.latent_dim,
        ):
            raise ValueError(
                "query_state must be [latent_dim]"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError(
                "min_similarity must be in [-1,1]"
            )

        if not self._episodes:
            return []

        query = self._store_tensor(
            query_state
        )

        query = F.normalize(
            query.unsqueeze(0),
            dim=-1,
            eps=self.config.eps,
        )

        candidates = [
            episode
            for episode in self._episodes.values()
            if (
                source is None
                or episode.source == source
            )
            and (
                max_uncertainty is None
                or episode.mean_uncertainty
                <= max_uncertainty
            )
        ]

        if not candidates:
            return []

        states = torch.stack(
            [
                episode.initial_state
                for episode in candidates
            ],
            dim=0,
        )

        states = F.normalize(
            states,
            dim=-1,
            eps=self.config.eps,
        )

        similarities = (
            query @ states.T
        ).squeeze(0)

        results = []

        for index, episode in enumerate(candidates):

            similarity = float(
                similarities[index].item()
            )

            if similarity < min_similarity:
                continue

            ranking_score = (
                self.config.similarity_weight
                * similarity
                + self.config.reward_weight
                * episode.score
                - self.config.uncertainty_weight
                * episode.mean_uncertainty
            )

            results.append(
                MemorySearchResult(
                    episode=episode,
                    similarity=similarity,
                    ranking_score=ranking_score,
                )
            )

        results.sort(
            key=lambda item: item.ranking_score,
            reverse=True,
        )

        return results[:top_k]

    def best_episodes(
        self,
        top_k: int = 5,
    ) -> List[ImaginationEpisode]:
        """
        Return highest-scoring stored episodes.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        return sorted(
            self._episodes.values(),
            key=lambda item: item.score,
            reverse=True,
        )[:top_k]

    def statistics(self) -> Dict[str, Any]:
        """
        Summarize imagination-memory utilization.
        """

        if not self._episodes:
            return {
                "count": 0,
                "capacity": self.config.capacity,
                "utilization": 0.0,
                "mean_score": 0.0,
                "mean_uncertainty": 0.0,
                "mean_horizon": 0.0,
            }

        episodes = list(
            self._episodes.values()
        )

        count = len(episodes)

        return {
            "count": count,
            "capacity": self.config.capacity,
            "utilization": (
                count / self.config.capacity
            ),
            "mean_score": sum(
                episode.score for episode in episodes
            ) / count,
            "mean_uncertainty": sum(
                episode.mean_uncertainty
                for episode in episodes
            ) / count,
            "mean_horizon": sum(
                episode.horizon
                for episode in episodes
            ) / count,
        }

    def save(self, path: str | Path) -> None:
        """
        Persist memory to a PyTorch checkpoint.

        Only load checkpoints from trusted sources.
        """

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": 1,
            "config": vars(self.config),
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "states": episode.states.cpu(),
                    "actions": episode.actions.cpu(),
                    "rewards": episode.rewards.cpu(),
                    "uncertainties": (
                        episode.uncertainties.cpu()
                    ),
                    "score": episode.score,
                    "created_at": episode.created_at,
                    "source": episode.source,
                    "metadata": episode.metadata,
                }
                for episode in self._episodes.values()
            ],
        }

        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        storage_device: Optional[str] = None,
    ) -> "ImaginationMemory":
        """
        Restore memory from a trusted checkpoint.

        Uses weights_only=True to restrict
        unpickling to supported safe types.
        """

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if payload.get("version") != 1:
            raise ValueError(
                "Unsupported memory checkpoint version"
            )

        config_data = dict(payload["config"])

        if storage_device is not None:
            config_data["storage_device"] = (
                storage_device
            )

        memory = cls(
            ImaginationMemoryConfig(**config_data)
        )

        for item in payload["episodes"]:
            memory._validate_episode(
                item["states"],
                item["actions"],
                item["rewards"],
                item["uncertainties"],
            )

            episode = ImaginationEpisode(
                episode_id=item["episode_id"],
                states=memory._store_tensor(
                    item["states"]
                ),
                actions=memory._store_tensor(
                    item["actions"]
                ),
                rewards=memory._store_tensor(
                    item["rewards"]
                ),
                uncertainties=memory._store_tensor(
                    item["uncertainties"]
                ),
                score=float(item["score"]),
                created_at=float(item["created_at"]),
                source=item["source"],
                metadata=dict(item["metadata"]),
            )

            memory._episodes[
                episode.episode_id
            ] = episode

        if len(memory) > memory.config.capacity:
            raise ValueError(
                "Checkpoint exceeds configured capacity"
            )

        return memory


if __name__ == "__main__":

    torch.manual_seed(42)

    config = ImaginationMemoryConfig(
        latent_dim=128,
        action_dim=32,
        capacity=1000,
        similarity_threshold=0.995,
    )

    memory = ImaginationMemory(config)

    horizon = 10

    states = torch.randn(
        horizon + 1,
        config.latent_dim,
    )

    actions = torch.randn(
        horizon,
        config.action_dim,
    )

    rewards = torch.randn(horizon)

    uncertainties = torch.rand(horizon)

    episode_id = memory.add(
        states=states,
        actions=actions,
        rewards=rewards,
        uncertainties=uncertainties,
        source="demo",
        metadata={
            "scenario": "navigation",
            "environment": "simulation",
        },
    )

    print("Stored episode:", episode_id)

    results = memory.search(
        query_state=states[0],
        top_k=5,
    )

    for result in results:
        print(
            "Episode:", result.episode.episode_id,
            "Similarity:", result.similarity,
            "Ranking:", result.ranking_score,
        )

    print("Statistics:", memory.statistics())

    checkpoint_path = (
        "checkpoints/imagination_memory.pt"
    )

    memory.save(checkpoint_path)

    restored = ImaginationMemory.load(
        checkpoint_path
    )

    print("Restored episodes:", len(restored))
    print("Restored stats:", restored.statistics())
