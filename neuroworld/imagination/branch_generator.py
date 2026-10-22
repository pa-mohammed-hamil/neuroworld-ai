
"""
NeuroWorld AI
File: neuroworld/imagination/branch_generator.py

Imagination Branch Generation Engine

Features:
- Multi-branch future imagination
- Action-conditioned world-state transitions
- Stochastic future exploration
- Beam search and branch pruning
- Reward-based trajectory ranking
- Uncertainty-aware planning
- Batched PyTorch inference
- Differentiable transition rollouts
- CPU / GPU compatibility

Install:
    pip install torch
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BranchGeneratorConfig:
    latent_dim: int = 128
    action_dim: int = 32

    horizon: int = 10
    branching_factor: int = 4
    beam_width: int = 8

    stochastic: bool = True
    temperature: float = 0.5

    discount_factor: float = 0.99
    uncertainty_penalty: float = 0.1

    max_logvar: float = 4.0
    min_logvar: float = -10.0


@dataclass
class ImaginationBranch:
    """
    One imagined future trajectory.

    states:       [H+1, latent_dim]
    actions:      [H, action_dim]
    rewards:      [H]
    uncertainties:[H]

    score:
        Discounted reward minus uncertainty penalty.
    """

    states: Tensor
    actions: Tensor
    rewards: Tensor
    uncertainties: Tensor
    score: Tensor

    @property
    def final_state(self) -> Tensor:
        return self.states[-1]

    @property
    def length(self) -> int:
        return self.actions.shape[0]


class NeuralTransitionModel(nn.Module):
    """
    Example action-conditioned latent dynamics model.

    This is a trainable model, not pretrained physics.

    Input:
        state:  [..., latent_dim]
        action: [..., action_dim]

    Output:
        next_mean: [..., latent_dim]
        next_logvar: [..., latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(
                latent_dim + action_dim,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.mean_head = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        self.logvar_head = nn.Linear(
            hidden_dim,
            latent_dim,
        )

    def forward(
        self,
        state: Tensor,
        action: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError(
                "State/action batch shapes must match"
            )

        if state.shape[-1] != self.latent_dim:
            raise ValueError(
                "Invalid latent dimension"
            )

        if action.shape[-1] != self.action_dim:
            raise ValueError(
                "Invalid action dimension"
            )

        features = torch.cat(
            [state, action],
            dim=-1,
        )

        hidden = self.backbone(features)

        delta = self.mean_head(hidden)
        logvar = self.logvar_head(hidden)

        next_mean = state + delta

        return next_mean, logvar


class BranchGenerator(nn.Module):
    """
    Generate and rank imagined future branches.

    The transition model predicts possible next states.
    The action sampler proposes candidate actions.
    The reward function evaluates each transition.

    Beam search retains the best candidate branches.
    """

    def __init__(
        self,
        transition_model: nn.Module,
        config: Optional[BranchGeneratorConfig] = None,
        reward_fn: Optional[
            Callable[[Tensor, Tensor, Tensor], Tensor]
        ] = None,
        action_sampler: Optional[
            Callable[[Tensor, int], Tensor]
        ] = None,
    ):
        super().__init__()

        self.config = (
            config or BranchGeneratorConfig()
        )

        self.transition_model = transition_model

        self.reward_fn = reward_fn
        self.action_sampler = action_sampler

        cfg = self.config

        if (
            cfg.horizon <= 0
            or cfg.branching_factor <= 0
            or cfg.beam_width <= 0
        ):
            raise ValueError(
                "Horizon, branching factor and beam width "
                "must be positive"
            )

        if cfg.temperature < 0:
            raise ValueError(
                "Temperature cannot be negative"
            )

        if not 0.0 <= cfg.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be between 0 and 1"
            )

    def sample_actions(
        self,
        states: Tensor,
        num_actions: int,
    ) -> Tensor:
        """
        Return actions with shape:
            [batch, num_actions, action_dim]

        Default sampler uses Gaussian random actions
        in [-1, 1]. Replace with a trained policy or
        environment-specific valid-action sampler.
        """

        if self.action_sampler is not None:
            actions = self.action_sampler(
                states,
                num_actions,
            )
        else:
            actions = torch.randn(
                states.shape[0],
                num_actions,
                self.config.action_dim,
                device=states.device,
                dtype=states.dtype,
            ).clamp(-1.0, 1.0)

        expected = (
            states.shape[0],
            num_actions,
            self.config.action_dim,
        )

        if actions.shape != expected:
            raise ValueError(
                f"Action sampler must return {expected}"
            )

        return actions

    def calculate_reward(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
    ) -> Tensor:
        """
        Return one reward per candidate transition.

        The default is a demonstration-only heuristic
        that penalizes large state displacement.
        """

        if self.reward_fn is not None:
            rewards = self.reward_fn(
                state,
                action,
                next_state,
            )
        else:
            rewards = -0.01 * (
                next_state - state
            ).pow(2).mean(dim=-1)

        if rewards.shape != state.shape[:-1]:
            raise ValueError(
                "Reward function must return [batch]"
            )

        return rewards

    def predict_next(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict next states and estimated uncertainty.

        Returns:
            next_states: [N, latent_dim]
            uncertainty: [N]
        """

        mean, logvar = self.transition_model(
            states,
            actions,
        )

        if mean.shape != states.shape:
            raise ValueError(
                "Transition mean shape mismatch"
            )

        if logvar.shape != states.shape:
            raise ValueError(
                "Transition logvar shape mismatch"
            )

        logvar = logvar.clamp(
            self.config.min_logvar,
            self.config.max_logvar,
        )

        std = torch.exp(0.5 * logvar)

        if self.config.stochastic:
            noise = torch.randn_like(std)

            next_states = (
                mean
                + self.config.temperature
                * std
                * noise
            )
        else:
            next_states = mean

        uncertainty = (
            std.pow(2).mean(dim=-1)
        )

        return next_states, uncertainty

    def generate(
        self,
        initial_state: Tensor,
    ) -> List[List[ImaginationBranch]]:
        """
        Generate imagined futures for a batch.

        initial_state:
            [batch, latent_dim]

        Returns:
            One list of ranked branches per batch item.

        Each branch contains the initial state followed
        by its predicted future states.
        """

        cfg = self.config

        if initial_state.ndim != 2:
            raise ValueError(
                "initial_state must be [B, latent_dim]"
            )

        if initial_state.shape[-1] != cfg.latent_dim:
            raise ValueError(
                "Invalid initial latent dimension"
            )

        if initial_state.shape[0] == 0:
            raise ValueError(
                "Batch size must be positive"
            )

        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype

        # Beam tensors:
        # states: [B, K, T+1, D]
        # actions: [B, K, T, A]

        state_history = initial_state[:, None, None, :]

        action_history = torch.empty(
            batch_size,
            1,
            0,
            cfg.action_dim,
            device=device,
            dtype=dtype,
        )

        reward_history = torch.empty(
            batch_size,
            1,
            0,
            device=device,
            dtype=dtype,
        )

        uncertainty_history = torch.empty(
            batch_size,
            1,
            0,
            device=device,
            dtype=dtype,
        )

        beam_scores = torch.zeros(
            batch_size,
            1,
            device=device,
            dtype=dtype,
        )

        for step in range(cfg.horizon):

            beam_count = state_history.shape[1]

            current_states = state_history[:, :, -1, :]

            flat_states = current_states.reshape(
                batch_size * beam_count,
                cfg.latent_dim,
            )

            proposed_actions = self.sample_actions(
                flat_states,
                cfg.branching_factor,
            )

            proposed_actions = proposed_actions.reshape(
                batch_size,
                beam_count,
                cfg.branching_factor,
                cfg.action_dim,
            )

            expanded_states = current_states.unsqueeze(
                2
            ).expand(
                -1,
                -1,
                cfg.branching_factor,
                -1,
            )

            flat_expanded_states = expanded_states.reshape(
                -1,
                cfg.latent_dim,
            )

            flat_actions = proposed_actions.reshape(
                -1,
                cfg.action_dim,
            )

            next_states, uncertainties = (
                self.predict_next(
                    flat_expanded_states,
                    flat_actions,
                )
            )

            rewards = self.calculate_reward(
                flat_expanded_states,
                flat_actions,
                next_states,
            )

            next_states = next_states.reshape(
                batch_size,
                beam_count,
                cfg.branching_factor,
                cfg.latent_dim,
            )

            rewards = rewards.reshape(
                batch_size,
                beam_count,
                cfg.branching_factor,
            )

            uncertainties = uncertainties.reshape(
                batch_size,
                beam_count,
                cfg.branching_factor,
            )

            step_value = (
                rewards
                - cfg.uncertainty_penalty
                * uncertainties
            )

            candidate_scores = (
                beam_scores.unsqueeze(-1)
                + (cfg.discount_factor ** step)
                * step_value
            )

            candidate_count = (
                beam_count * cfg.branching_factor
            )

            flattened_scores = candidate_scores.reshape(
                batch_size,
                candidate_count,
            )

            keep_count = min(
                cfg.beam_width,
                candidate_count,
            )

            selected_scores, selected_indices = (
                torch.topk(
                    flattened_scores,
                    k=keep_count,
                    dim=1,
                )
            )

            parent_indices = (
                selected_indices // cfg.branching_factor
            )

            def gather_history(
                history: Tensor,
            ) -> Tensor:
                """
                Select parent histories for surviving beams.
                """

                trailing_shape = history.shape[2:]

                gather_index = parent_indices.reshape(
                    batch_size,
                    keep_count,
                    *([1] * len(trailing_shape)),
                ).expand(
                    batch_size,
                    keep_count,
                    *trailing_shape,
                )

                return torch.gather(
                    history,
                    dim=1,
                    index=gather_index,
                )

            selected_states = torch.gather(
                next_states.reshape(
                    batch_size,
                    candidate_count,
                    cfg.latent_dim,
                ),
                1,
                selected_indices.unsqueeze(-1).expand(
                    -1, -1, cfg.latent_dim
                ),
            )

            selected_actions = torch.gather(
                proposed_actions.reshape(
                    batch_size,
                    candidate_count,
                    cfg.action_dim,
                ),
                1,
                selected_indices.unsqueeze(-1).expand(
                    -1, -1, cfg.action_dim
                ),
            )

            selected_rewards = torch.gather(
                rewards.reshape(
                    batch_size,
                    candidate_count,
                ),
                1,
                selected_indices,
            )

            selected_uncertainties = torch.gather(
                uncertainties.reshape(
                    batch_size,
                    candidate_count,
                ),
                1,
                selected_indices,
            )

            state_history = torch.cat(
                [
                    gather_history(state_history),
                    selected_states.unsqueeze(2),
                ],
                dim=2,
            )

            action_history = torch.cat(
                [
                    gather_history(action_history),
                    selected_actions.unsqueeze(2),
                ],
                dim=2,
            )

            reward_history = torch.cat(
                [
                    gather_history(reward_history),
                    selected_rewards.unsqueeze(2),
                ],
                dim=2,
            )

            uncertainty_history = torch.cat(
                [
                    gather_history(uncertainty_history),
                    selected_uncertainties.unsqueeze(2),
                ],
                dim=2,
            )

            beam_scores = selected_scores

        results: List[List[ImaginationBranch]] = []

        for batch_index in range(batch_size):

            branches = []

            for beam_index in range(
                state_history.shape[1]
            ):

                branches.append(
                    ImaginationBranch(
                        states=state_history[
                            batch_index, beam_index
                        ],
                        actions=action_history[
                            batch_index, beam_index
                        ],
                        rewards=reward_history[
                            batch_index, beam_index
                        ],
                        uncertainties=uncertainty_history[
                            batch_index, beam_index
                        ],
                        score=beam_scores[
                            batch_index, beam_index
                        ],
                    )
                )

            results.append(branches)

        return results

    def best_branch(
        self,
        initial_state: Tensor,
    ) -> List[ImaginationBranch]:
        """
        Return the highest-scoring branch
        for each batch item.
        """

        branches = self.generate(initial_state)

        return [
            batch_branches[0]
            for batch_branches in branches
        ]


if __name__ == "__main__":

    torch.manual_seed(42)

    config = BranchGeneratorConfig(
        latent_dim=128,
        action_dim=32,
        horizon=6,
        branching_factor=4,
        beam_width=5,
        stochastic=True,
    )

    transition_model = NeuralTransitionModel(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
    )

    generator = BranchGenerator(
        transition_model=transition_model,
        config=config,
    )

    initial_state = torch.randn(
        2,
        config.latent_dim,
    )

    imagined_branches = generator.generate(
        initial_state
    )

    for batch_index, branches in enumerate(
        imagined_branches
    ):
        print(f"\nWorld {batch_index + 1}")

        for index, branch in enumerate(branches):
            print(
                f"Branch {index + 1}: "
                f"score={branch.score.item():.4f}, "
                f"steps={branch.length}, "
                f"states={tuple(branch.states.shape)}"
            )

    best = generator.best_branch(initial_state)

    print(
        "\nBest branch final state:",
        best[0].final_state.shape,
    )
