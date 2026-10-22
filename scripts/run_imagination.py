

#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/run_imagination.py

Latent Imagination and Future Simulation Runner

Features:
    - Load a trained world/dynamics model
    - Load an initial latent state
    - Random-shooting trajectory search
    - Cross-Entropy Method (CEM) planning
    - Simulate multiple possible futures
    - Goal-conditioned trajectory scoring
    - Action and uncertainty penalties
    - Counterfactual action interventions
    - Reproducible random seeds
    - CPU/CUDA support
    - NPZ, JSON and optional PNG export

Dependencies:
    pip install numpy torch matplotlib

Examples:
    python scripts/run_imagination.py \
        --model-factory \
        neuroworld.dynamics.transition_model:TransitionModel \
        --checkpoint models/checkpoints/world_model.pt \
        --latent-dim 128 \
        --action-dim 32 \
        --horizon 15 \
        --candidates 128 \
        --method cem

    python scripts/run_imagination.py \
        --model-factory \
        neuroworld.dynamics.transition_model:TransitionModel \
        --checkpoint models/checkpoints/world_model.pt \
        --initial-state data/states/initial.npy \
        --goal-state data/states/goal.npy \
        --method random \
        --plot

Important:
    A trained checkpoint is required for meaningful
    world-model predictions.

    Random initial states and random model weights
    are not evidence of physical realism.

    This script performs offline simulation only.
    It does not control physical robots.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import random
import sys
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("neuroworld.imagination")


@dataclass
class ImaginationRunConfig:
    model_factory: str = (
        "neuroworld.dynamics.transition_model:"
        "TransitionModel"
    )

    checkpoint_path: Optional[str] = None

    initial_state_path: Optional[str] = None
    goal_state_path: Optional[str] = None

    latent_dim: int = 128
    action_dim: int = 32

    horizon: int = 15
    num_candidates: int = 128

    method: str = "cem"

    cem_iterations: int = 6
    elite_fraction: float = 0.15
    cem_smoothing: float = 0.2

    initial_action_std: float = 0.5
    minimum_action_std: float = 0.05

    action_min: float = -1.0
    action_max: float = 1.0

    goal_weight: float = 1.0
    uncertainty_penalty: float = 0.1
    action_penalty: float = 0.01
    smoothness_penalty: float = 0.02

    stochastic: bool = False
    temperature: float = 1.0

    device: str = "auto"
    seed: int = 42

    output_dir: str = (
        "artifacts/imagination"
    )

    save_all_candidates: bool = True
    plot: bool = False

    allow_untrained: bool = False


@dataclass
class ImaginationRunResult:
    success: bool

    method: str
    horizon: int
    candidates: int

    best_score: float

    initial_goal_distance: Optional[float]
    final_goal_distance: Optional[float]

    mean_candidate_score: float
    mean_uncertainty: float

    execution_seconds: float

    output_files: dict[str, str] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class ImaginationRunner:
    """
    Generate imagined latent futures using
    an action-conditioned dynamics model.

    Supported dynamics outputs:
        Tensor next_state
        (mean, logvar)
        Object with .mean and .logvar
        Object with .next_state

    Model contract:
        state:  [B, latent_dim]
        action: [B, action_dim]
        output: [B, latent_dim]

    If the model exposes predict(...),
    that method is preferred.
    """

    def __init__(
        self,
        config: Optional[
            ImaginationRunConfig
        ] = None,
    ) -> None:
        self.config = (
            config or ImaginationRunConfig()
        )

        self._validate_config()
        self._seed_everything()

        self.device = self._select_device()

        self.model: Optional[
            torch.nn.Module
        ] = None

        self.initial_state: Optional[
            torch.Tensor
        ] = None

        self.goal_state: Optional[
            torch.Tensor
        ] = None

        self.generator = torch.Generator(
            device=self.device
        )

        self.generator.manual_seed(
            self.config.seed
        )

    def _validate_config(self) -> None:
        c = self.config

        if c.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive"
            )

        if c.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if c.horizon <= 0:
            raise ValueError(
                "horizon must be positive"
            )

        if c.num_candidates <= 0:
            raise ValueError(
                "num_candidates must be positive"
            )

        if c.method not in {
            "cem",
            "random",
        }:
            raise ValueError(
                "method must be cem or random"
            )

        if c.cem_iterations <= 0:
            raise ValueError(
                "cem_iterations must be positive"
            )

        if not 0 < c.elite_fraction <= 1:
            raise ValueError(
                "elite_fraction must be in (0, 1]"
            )

        if not 0 <= c.cem_smoothing < 1:
            raise ValueError(
                "cem_smoothing must be in [0, 1)"
            )

        if (
            c.initial_action_std <= 0
            or c.minimum_action_std <= 0
        ):
            raise ValueError(
                "Action standard deviations "
                "must be positive"
            )

        if c.action_min >= c.action_max:
            raise ValueError(
                "Invalid action limits"
            )

        if (
            not math.isfinite(c.temperature)
            or c.temperature <= 0
        ):
            raise ValueError(
                "temperature must be positive"
            )

        for name in (
            "goal_weight",
            "uncertainty_penalty",
            "action_penalty",
            "smoothness_penalty",
        ):
            value = getattr(c, name)

            if (
                not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be finite "
                    "and nonnegative"
                )

    @staticmethod
    def _resolve_path(
        value: str,
    ) -> Path:
        path = Path(value).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        return path.resolve()

    def _seed_everything(self) -> None:
        seed = self.config.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _select_device(
        self,
    ) -> torch.device:
        requested = self.config.device

        if requested == "auto":
            requested = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        if (
            requested.startswith("cuda")
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but unavailable"
            )

        return torch.device(requested)

    @staticmethod
    def _import_model_factory(
        import_path: str,
    ) -> Any:
        if ":" not in import_path:
            raise ValueError(
                "Expected module.path:ClassName"
            )

        module_name, class_name = (
            import_path.split(":", 1)
        )

        module = importlib.import_module(
            module_name
        )

        factory = getattr(
            module,
            class_name,
        )

        if not callable(factory):
            raise TypeError(
                "Model factory is not callable"
            )

        return factory

    def load_model(
        self,
    ) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        factory = self._import_model_factory(
            self.config.model_factory
        )

        model = factory()

        if not isinstance(
            model,
            torch.nn.Module,
        ):
            raise TypeError(
                "Model factory must return "
                "a PyTorch nn.Module"
            )

        if self.config.checkpoint_path:
            checkpoint_path = (
                self._resolve_path(
                    self.config.checkpoint_path
                )
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    checkpoint_path
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get(
                    "model_state_dict",
                    checkpoint.get(
                        "state_dict",
                        checkpoint,
                    ),
                )
            else:
                state_dict = checkpoint

            if not isinstance(
                state_dict,
                dict,
            ):
                raise TypeError(
                    "Checkpoint must contain "
                    "model state_dict"
                )

            model.load_state_dict(
                state_dict,
                strict=True,
            )

        elif not self.config.allow_untrained:
            raise ValueError(
                "A trained checkpoint is required. "
                "Use --allow-untrained only "
                "for pipeline testing."
            )

        model.to(self.device)
        model.eval()

        self.model = model

        return model

    def _load_state(
        self,
        path: Optional[str],
        random_if_missing: bool,
    ) -> Optional[torch.Tensor]:
        if path is None:
            if not random_if_missing:
                return None

            return torch.randn(
                1,
                self.config.latent_dim,
                device=self.device,
                generator=self.generator,
            )

        state_path = self._resolve_path(
            path
        )

        if not state_path.is_file():
            raise FileNotFoundError(
                state_path
            )

        suffix = state_path.suffix.lower()

        if suffix == ".npy":
            array = np.load(
                state_path,
                allow_pickle=False,
            )

        elif suffix == ".npz":
            with np.load(
                state_path,
                allow_pickle=False,
            ) as archive:
                if "state" in archive:
                    array = archive[
                        "state"
                    ].copy()
                elif "latent_state" in archive:
                    array = archive[
                        "latent_state"
                    ].copy()
                else:
                    raise KeyError(
                        "NPZ must contain state "
                        "or latent_state"
                    )

        else:
            raise ValueError(
                "State file must be .npy or .npz"
            )

        tensor = torch.as_tensor(
            array,
            dtype=torch.float32,
            device=self.device,
        )

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.shape != (
            1,
            self.config.latent_dim,
        ):
            raise ValueError(
                "Expected state shape "
                f"[1, {self.config.latent_dim}], "
                f"got {list(tensor.shape)}"
            )

        if not torch.isfinite(
            tensor
        ).all():
            raise ValueError(
                "State contains NaN or infinity"
            )

        return tensor

    def prepare(
        self,
    ) -> None:
        self.load_model()

        self.initial_state = self._load_state(
            self.config.initial_state_path,
            random_if_missing=True,
        )

        self.goal_state = self._load_state(
            self.config.goal_state_path,
            random_if_missing=False,
        )

    def _predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.model is None:
            raise RuntimeError(
                "Model not loaded"
            )

        if hasattr(
            self.model,
            "predict",
        ):
            try:
                output = self.model.predict(
                    state,
                    action,
                    deterministic=(
                        not self.config.stochastic
                    ),
                )
            except TypeError:
                output = self.model.predict(
                    state,
                    action,
                )
        else:
            output = self.model(
                state,
                action,
            )

        mean: Optional[
            torch.Tensor
        ] = None

        logvar: Optional[
            torch.Tensor
        ] = None

        predicted: Optional[
            torch.Tensor
        ] = None

        if isinstance(
            output,
            torch.Tensor,
        ):
            predicted = output

        elif isinstance(
            output,
            (tuple, list),
        ):
            if not output:
                raise ValueError(
                    "Empty dynamics output"
                )

            mean = output[0]

            if len(output) > 1:
                logvar = output[1]

        else:
            mean = getattr(
                output,
                "mean",
                None,
            )

            logvar = getattr(
                output,
                "logvar",
                None,
            )

            predicted = getattr(
                output,
                "next_state",
                None,
            )

        if mean is None:
            mean = predicted

        if not isinstance(
            mean,
            torch.Tensor,
        ):
            raise TypeError(
                "Dynamics model did not return "
                "a valid state prediction"
            )

        if mean.shape != state.shape:
            raise ValueError(
                "Predicted state shape mismatch"
            )

        if logvar is not None:
            if (
                not isinstance(
                    logvar,
                    torch.Tensor,
                )
                or logvar.shape != state.shape
            ):
                raise ValueError(
                    "Invalid predictive log-variance"
                )

            logvar = torch.clamp(
                logvar,
                min=-20.0,
                max=10.0,
            )

            variance = torch.exp(
                logvar
            )

            uncertainty = (
                variance.mean(
                    dim=-1
                )
            )

            if self.config.stochastic:
                noise = torch.randn(
                    mean.shape,
                    device=mean.device,
                    dtype=mean.dtype,
                    generator=self.generator,
                )

                next_state = (
                    mean
                    + noise
                    * torch.sqrt(
                        variance
                    )
                    * self.config.temperature
                )
            else:
                next_state = mean

        else:
            # Without a probabilistic output,
            # uncertainty is unavailable.
            uncertainty = torch.zeros(
                state.shape[0],
                device=state.device,
                dtype=state.dtype,
            )

            next_state = mean

        if not torch.isfinite(
            next_state
        ).all():
            raise RuntimeError(
                "Dynamics produced non-finite states"
            )

        return next_state, uncertainty

    def _sample_actions(
        self,
        candidates: int,
    ) -> torch.Tensor:
        c = self.config

        shape = (
            candidates,
            c.horizon,
            c.action_dim,
        )

        actions = torch.rand(
            shape,
            device=self.device,
            generator=self.generator,
        )

        return (
            c.action_min
            + actions
            * (
                c.action_max
                - c.action_min
            )
        )

    def rollout(
        self,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Actions:
            [K, H, A]

        Returns:
            states: [K, H+1, D]
            uncertainty: [K, H]
            scores: [K]
        """

        if self.initial_state is None:
            raise RuntimeError(
                "Initial state not prepared"
            )

        if (
            actions.ndim != 3
            or actions.shape[1]
            != self.config.horizon
            or actions.shape[2]
            != self.config.action_dim
        ):
            raise ValueError(
                "Invalid action sequence shape"
            )

        candidates = actions.shape[0]

        state = (
            self.initial_state.expand(
                candidates, -1
            ).clone()
        )

        states = [state]
        uncertainties = []

        with torch.inference_mode():
            for step in range(
                self.config.horizon
            ):
                action = actions[
                    :, step, :
                ]

                state, uncertainty = (
                    self._predict(
                        state,
                        action,
                    )
                )

                states.append(state)
                uncertainties.append(
                    uncertainty
                )

            trajectories = torch.stack(
                states,
                dim=1,
            )

            uncertainty_tensor = (
                torch.stack(
                    uncertainties,
                    dim=1,
                )
            )

            scores = self._score(
                trajectories,
                actions,
                uncertainty_tensor,
            )

        return {
            "states": trajectories,
            "actions": actions,
            "uncertainty": (
                uncertainty_tensor
            ),
            "scores": scores,
        }

    def _score(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        c = self.config

        candidates = (
            actions.shape[0]
        )

        scores = torch.zeros(
            candidates,
            device=self.device,
        )

        if self.goal_state is not None:
            goal = self.goal_state.expand(
                candidates, -1
            )

            final_distance = (
                torch.linalg.vector_norm(
                    states[:, -1, :] - goal,
                    dim=-1,
                )
            )

            scores -= (
                c.goal_weight
                * final_distance
            )

        scores -= (
            c.uncertainty_penalty
            * uncertainty.mean(
                dim=1
            )
        )

        scores -= (
            c.action_penalty
            * actions.square().mean(
                dim=(1, 2)
            )
        )

        if c.horizon > 1:
            differences = (
                actions[:, 1:, :]
                - actions[:, :-1, :]
            )

            scores -= (
                c.smoothness_penalty
                * differences.square().mean(
                    dim=(1, 2)
                )
            )

        return scores

    def random_shooting(
        self,
    ) -> dict[str, torch.Tensor]:
        actions = self._sample_actions(
            self.config.num_candidates
        )

        return self.rollout(
            actions
        )

    def cem(
        self,
    ) -> dict[str, torch.Tensor]:
        c = self.config

        shape = (
            c.horizon,
            c.action_dim,
        )

        midpoint = (
            c.action_min
            + c.action_max
        ) / 2.0

        mean = torch.full(
            shape,
            midpoint,
            device=self.device,
        )

        std = torch.full(
            shape,
            c.initial_action_std,
            device=self.device,
        )

        elite_count = max(
            1,
            math.ceil(
                c.num_candidates
                * c.elite_fraction
            ),
        )

        best_result = None
        best_score = -math.inf

        for iteration in range(
            c.cem_iterations
        ):
            noise = torch.randn(
                (
                    c.num_candidates,
                    c.horizon,
                    c.action_dim,
                ),
                device=self.device,
                generator=self.generator,
            )

            actions = (
                mean.unsqueeze(0)
                + noise
                * std.unsqueeze(0)
            )

            actions = actions.clamp(
                c.action_min,
                c.action_max,
            )

            result = self.rollout(
                actions
            )

            scores = result["scores"]

            values, indices = torch.topk(
                scores,
                k=elite_count,
            )

            elite_actions = actions[
                indices
            ]

            elite_mean = (
                elite_actions.mean(
                    dim=0
                )
            )

            elite_std = (
                elite_actions.std(
                    dim=0,
                    unbiased=False,
                )
            )

            alpha = c.cem_smoothing

            mean = (
                alpha * mean
                + (1 - alpha)
                * elite_mean
            )

            std = (
                alpha * std
                + (1 - alpha)
                * elite_std
            )

            std = std.clamp_min(
                c.minimum_action_std
            )

            current_best = float(
                values[0].item()
            )

            if current_best > best_score:
                best_score = current_best

                best_index = int(
                    indices[0].item()
                )

                best_result = {
                    "states": (
                        result["states"][
                            best_index
                        ].clone()
                    ),
                    "actions": (
                        result["actions"][
                            best_index
                        ].clone()
                    ),
                    "uncertainty": (
                        result["uncertainty"][
                            best_index
                        ].clone()
                    ),
                    "score": (
                        result["scores"][
                            best_index
                        ].clone()
                    ),
                }

            LOGGER.info(
                "CEM iteration %d/%d | "
                "best score: %.6f",
                iteration + 1,
                c.cem_iterations,
                current_best,
            )

        if best_result is None:
            raise RuntimeError(
                "CEM failed to find a candidate"
            )

        # Include the best trajectory found
        # across all CEM iterations in the
        # final candidate collection.
        final_result = self.rollout(
            mean.unsqueeze(0)
        )

        final_result["states"] = torch.cat(
            [
                final_result["states"],
                best_result["states"].unsqueeze(0),
            ],
            dim=0,
        )

        final_result["actions"] = torch.cat(
            [
                final_result["actions"],
                best_result["actions"].unsqueeze(0),
            ],
            dim=0,
        )

        final_result["uncertainty"] = torch.cat(
            [
                final_result["uncertainty"],
                best_result["uncertainty"].unsqueeze(0),
            ],
            dim=0,
        )

        final_result["scores"] = torch.cat(
            [
                final_result["scores"],
                best_result["score"].reshape(1),
            ],
            dim=0,
        )

        return final_result

    def counterfactual(
        self,
        baseline_actions: torch.Tensor,
        step: int,
        replacement_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compare a baseline plan with an
        intervention at a specified step.

        Baseline actions:
            [H, A]

        Replacement action:
            [A]
        """

        if not (
            0 <= step
            < self.config.horizon
        ):
            raise ValueError(
                "Intervention step out of range"
            )

        if baseline_actions.shape != (
            self.config.horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                "Invalid baseline action shape"
            )

        if replacement_action.shape != (
            self.config.action_dim,
        ):
            raise ValueError(
                "Invalid replacement action shape"
            )

        intervention_actions = (
            baseline_actions.clone()
        )

        intervention_actions[
            step
        ] = replacement_action.clamp(
            self.config.action_min,
            self.config.action_max,
        )

        return self.rollout(
            torch.stack(
                [
                    baseline_actions,
                    intervention_actions,
                ],
                dim=0,
            )
        )

    def run(
        self,
    ) -> ImaginationRunResult:
        started = time.perf_counter()

        self.prepare()

        if self.config.method == "cem":
            result = self.cem()
        else:
            result = self.random_shooting()

        scores = result["scores"]

        best_index = int(
            torch.argmax(
                scores
            ).item()
        )

        best_states = result["states"][
            best_index
        ]

        best_actions = result["actions"][
            best_index
        ]

        best_uncertainty = (
            result["uncertainty"][
                best_index
            ]
        )

        initial_distance = None
        final_distance = None

        if self.goal_state is not None:
            initial_distance = float(
                torch.linalg.vector_norm(
                    self.initial_state
                    - self.goal_state
                ).item()
            )

            final_distance = float(
                torch.linalg.vector_norm(
                    best_states[-1:]
                    - self.goal_state
                ).item()
            )

        output_files = self._export(
            result,
            best_index,
        )

        return ImaginationRunResult(
            success=True,
            method=self.config.method,
            horizon=self.config.horizon,
            candidates=int(
                result["actions"].shape[0]
            ),
            best_score=float(
                scores[best_index].item()
            ),
            initial_goal_distance=(
                initial_distance
            ),
            final_goal_distance=(
                final_distance
            ),
            mean_candidate_score=float(
                scores.mean().item()
            ),
            mean_uncertainty=float(
                best_uncertainty.mean().item()
            ),
            execution_seconds=(
                time.perf_counter() - started
            ),
            output_files=output_files,
            metadata={
                "device": str(self.device),
                "latent_dim": (
                    self.config.latent_dim
                ),
                "action_dim": (
                    self.config.action_dim
                ),
                "checkpoint": (
                    self.config.checkpoint_path
                ),
                "best_actions_shape": list(
                    best_actions.shape
                ),
                "best_states_shape": list(
                    best_states.shape
                ),
            },
        )

    def _export(
        self,
        result: dict[str, torch.Tensor],
        best_index: int,
    ) -> dict[str, str]:
        output_dir = self._resolve_path(
            self.config.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        files: dict[str, str] = {}

        best_path = (
            output_dir
            / "best_imagination.npz"
        )

        np.savez_compressed(
            best_path,
            states=(
                result["states"][
                    best_index
                ].detach()
                .cpu()
                .numpy()
            ),
            actions=(
                result["actions"][
                    best_index
                ].detach()
                .cpu()
                .numpy()
            ),
            uncertainty=(
                result["uncertainty"][
                    best_index
                ].detach()
                .cpu()
                .numpy()
            ),
            score=np.asarray(
                float(
                    result["scores"][
                        best_index
                    ].item()
                ),
                dtype=np.float32,
            ),
        )

        files["best_imagination"] = str(
            best_path
        )

        if self.config.save_all_candidates:
            candidates_path = (
                output_dir
                / "candidate_futures.npz"
            )

            np.savez_compressed(
                candidates_path,
                states=(
                    result["states"]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                actions=(
                    result["actions"]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                uncertainty=(
                    result["uncertainty"]
                    .detach()
                    .cpu()
                    .numpy()
                ),
                scores=(
                    result["scores"]
                    .detach()
                    .cpu()
                    .numpy()
                ),
            )

            files["candidates"] = str(
                candidates_path
            )

        if self.config.plot:
            plot_path = (
                output_dir
                / "imagination_plot.png"
            )

            self._plot(
                result,
                best_index,
                plot_path,
            )

            files["plot"] = str(
                plot_path
            )

        return files

    @staticmethod
    def _plot(
        result: dict[str, torch.Tensor],
        best_index: int,
        output_path: Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        states = (
            result["states"]
            .detach()
            .cpu()
            .numpy()
        )

        best = states[
            best_index
        ]

        if best.shape[-1] < 2:
            raise ValueError(
                "Plot requires latent_dim >= 2"
            )

        figure, axis = plt.subplots(
            figsize=(8, 6)
        )

        for trajectory in states:
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                alpha=0.15,
            )

        axis.plot(
            best[:, 0],
            best[:, 1],
            linewidth=2.5,
            label="Best imagined future",
        )

        axis.scatter(
            best[0, 0],
            best[0, 1],
            marker="o",
            label="Initial state",
        )

        axis.scatter(
            best[-1, 0],
            best[-1, 1],
            marker="x",
            label="Final state",
        )

        axis.set_title(
            "NeuroWorld AI - Imagined Latent Futures"
        )

        axis.set_xlabel(
            "Latent dimension 1"
        )

        axis.set_ylabel(
            "Latent dimension 2"
        )

        axis.grid(
            alpha=0.2
        )

        axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
        )

        plt.close(
            figure
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NeuroWorld AI imagination runner"
        )
    )

    parser.add_argument(
        "--model-factory",
        default=(
            "neuroworld.dynamics.transition_model:"
            "TransitionModel"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--initial-state",
        default=None,
    )

    parser.add_argument(
        "--goal-state",
        default=None,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--action-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--method",
        choices=[
            "cem",
            "random",
        ],
        default="cem",
    )

    parser.add_argument(
        "--cem-iterations",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default=(
            "artifacts/imagination"
        ),
    )

    parser.add_argument(
        "--plot",
        action="store_true",
    )

    parser.add_argument(
        "--allow-untrained",
        action="store_true",
    )

    parser.add_argument(
        "--no-save-candidates",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = ImaginationRunConfig(
        model_factory=args.model_factory,
        checkpoint_path=args.checkpoint,
        initial_state_path=args.initial_state,
        goal_state_path=args.goal_state,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        horizon=args.horizon,
        num_candidates=args.candidates,
        method=args.method,
        cem_iterations=args.cem_iterations,
        device=args.device,
        seed=args.seed,
        output_dir=args.output,
        plot=args.plot,
        allow_untrained=args.allow_untrained,
        save_all_candidates=(
            not args.no_save_candidates
        ),
    )

    try:
        runner = ImaginationRunner(
            config
        )

        result = runner.run()

    except Exception:
        LOGGER.exception(
            "Imagination run failed"
        )

        return 1

    output_dir = runner._resolve_path(
        config.output_dir
    )

    report_path = (
        output_dir
        / "imagination_report.json"
    )

    report_path.write_text(
        json.dumps(
            {
                "project": "NeuroWorld AI",
                "timestamp": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "configuration": asdict(
                    config
                ),
                "result": asdict(
                    result
                ),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("         NEUROWORLD AI IMAGINATION")
    print("=" * 60)

    print(
        "Method:",
        result.method,
    )

    print(
        "Horizon:",
        result.horizon,
    )

    print(
        "Candidates:",
        result.candidates,
    )

    print(
        "Best score:",
        f"{result.best_score:.6f}",
    )

    print(
        "Mean uncertainty:",
        f"{result.mean_uncertainty:.6f}",
    )

    print(
        "Execution time:",
        f"{result.execution_seconds:.3f}s",
    )

    print(
        "Report:",
        report_path,
    )

    for name, path in (
        result.output_files.items()
    ):
        print(
            f"{name}: {path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
