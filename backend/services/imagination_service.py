
"""
NeuroWorld AI
backend/app/services/imagination_service.py

Action-conditioned neural imagination service.

Features:
- Latent-state imagination
- Multi-step future rollouts
- Multiple future branches
- Action-conditioned predictions
- Counterfactual scenarios
- Uncertainty-aware termination
- Trajectory scoring
- Model adapter interface
- Thread-safe session storage
- Async-compatible execution
- FastAPI service integration
- Development demonstration model

Dependencies:
    pip install numpy pydantic

The demonstration model is NOT a trained neural network.
Replace it with a trained world-model adapter for real
neural imagination.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..config import get_settings
from ..core.exceptions import (
    ImaginationError,
    ImaginationLimitError,
    InvalidLatentStateError,
    ModelNotLoadedError,
)
from ..core.logging import (
    get_logger,
    log_imagination,
    PerformanceTimer,
)
from ..core.telemetry import record_imagination


logger = get_logger("services.imagination")


# ============================================================
# DATA MODELS
# ============================================================

class ImaginationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(min_length=1, max_length=128)

    initial_state: list[float] = Field(
        min_length=1
    )

    action_sequences: list[list[list[float]]] = Field(
        min_length=1
    )

    horizon: int = Field(ge=1, le=128)

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    stop_on_uncertainty: bool = True

    metadata: dict[str, str] = Field(
        default_factory=dict
    )


class ImaginationStep(BaseModel):
    step: int
    action: list[float]
    state: list[float]
    uncertainty: float = Field(ge=0, le=1)
    reward: float | None = None
    terminal: bool = False


class ImaginationBranch(BaseModel):
    branch_id: str
    steps: list[ImaginationStep]
    final_state: list[float]
    cumulative_reward: float
    mean_uncertainty: float
    terminated_early: bool


class ImaginationResult(BaseModel):
    imagination_id: str
    world_id: str
    model_name: str
    created_at: datetime
    horizon: int
    branch_count: int
    duration_ms: float
    branches: list[ImaginationBranch]
    metadata: dict[str, str] = Field(
        default_factory=dict
    )


class CounterfactualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(min_length=1, max_length=128)
    initial_state: list[float] = Field(min_length=1)

    baseline_actions: list[list[float]] = Field(
        min_length=1
    )

    alternative_actions: list[list[float]] = Field(
        min_length=1
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )


class CounterfactualResult(BaseModel):
    counterfactual_id: str
    world_id: str
    baseline: ImaginationBranch
    alternative: ImaginationBranch
    final_state_distance: float
    reward_difference: float
    created_at: datetime


@dataclass(frozen=True)
class Transition:
    """
    One world-model prediction.

    uncertainty must be a calibrated or explicitly
    defined value between 0 and 1.
    """

    next_state: np.ndarray
    uncertainty: float
    reward: float = 0.0
    terminal: bool = False


# ============================================================
# WORLD MODEL INTERFACE
# ============================================================

class WorldModelAdapter(Protocol):
    """
    Interface implemented by a neural world model.

    Model adapters must expose:
        name
        state_dim
        action_dim
        predict_transition(state, action)
    """

    name: str
    state_dim: int
    action_dim: int

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Transition:
        ...


# ============================================================
# DEMONSTRATION WORLD MODEL
# ============================================================

class DemoWorldModel:
    """
    Deterministic synthetic transition model.

    Only for integration testing and demonstrations.

    Transition:
        next_state = state + action * scale

    It does not learn dynamics or perform neural inference.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 8,
        scale: float = 0.1,
    ) -> None:

        if state_dim < 1 or action_dim < 1:
            raise ValueError(
                "State and action dimensions must be positive"
            )

        if state_dim != action_dim:
            raise ValueError(
                "Demo model requires matching dimensions"
            )

        self.name = "demo_synthetic_world_model"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scale = float(scale)

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Transition:

        next_state = state + action * self.scale

        reward = -float(
            np.linalg.norm(next_state)
        )

        return Transition(
            next_state=next_state,
            uncertainty=0.05,
            reward=reward,
            terminal=False,
        )


# ============================================================
# VALIDATION
# ============================================================

def _validated_vector(
    values: Any,
    *,
    name: str,
    dimension: int,
) -> np.ndarray:

    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (ValueError, TypeError) as exc:
        raise InvalidLatentStateError(
            f"{name} must contain numeric values"
        ) from exc

    if array.shape != (dimension,):
        raise InvalidLatentStateError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.all(np.isfinite(array)):
        raise InvalidLatentStateError(
            f"{name} contains non-finite values"
        )

    return array.copy()


def _validate_transition(
    transition: Transition,
    *,
    state_dim: int,
) -> Transition:

    if not isinstance(transition, Transition):
        raise ImaginationError(
            "World model returned an invalid transition"
        )

    next_state = _validated_vector(
        transition.next_state,
        name="next_state",
        dimension=state_dim,
    )

    uncertainty = float(
        transition.uncertainty
    )

    reward = float(
        transition.reward
    )

    if (
        not math.isfinite(uncertainty)
        or not 0 <= uncertainty <= 1
    ):
        raise ImaginationError(
            "World model returned invalid uncertainty"
        )

    if not math.isfinite(reward):
        raise ImaginationError(
            "World model returned invalid reward"
        )

    return Transition(
        next_state=next_state,
        uncertainty=uncertainty,
        reward=reward,
        terminal=bool(transition.terminal),
    )


# ============================================================
# IMAGINATION ENGINE
# ============================================================

class ImaginationService:
    """
    Generate imagined future trajectories using a
    registered world-model adapter.

    No external actions or physical hardware commands
    are executed by this service.
    """

    def __init__(
        self,
        model: WorldModelAdapter | None = None,
        *,
        max_branches: int = 32,
        max_horizon: int = 128,
        max_history: int = 100,
    ) -> None:

        if max_branches < 1:
            raise ValueError(
                "max_branches must be positive"
            )

        if max_horizon < 1:
            raise ValueError(
                "max_horizon must be positive"
            )

        if max_history < 1:
            raise ValueError(
                "max_history must be positive"
            )

        self._model = model
        self.max_branches = max_branches
        self.max_horizon = max_horizon
        self.max_history = max_history

        self._results: dict[
            str,
            ImaginationResult,
        ] = {}

        self._lock = threading.RLock()

    @property
    def model(self) -> WorldModelAdapter:
        with self._lock:
            model = self._model

        if model is None:
            raise ModelNotLoadedError(
                "No world model is registered"
            )

        return model

    def register_model(
        self,
        model: WorldModelAdapter,
    ) -> None:

        if (
            model.state_dim < 1
            or model.action_dim < 1
        ):
            raise ValueError(
                "Model dimensions must be positive"
            )

        if not callable(
            getattr(
                model,
                "predict_transition",
                None,
            )
        ):
            raise ValueError(
                "Model must implement predict_transition"
            )

        with self._lock:
            self._model = model

        logger.info(
            "World model registered: %s",
            model.name,
        )

    def unregister_model(self) -> None:
        with self._lock:
            self._model = None

    def _rollout(
        self,
        *,
        model: WorldModelAdapter,
        initial_state: np.ndarray,
        actions: Sequence[Sequence[float]],
        horizon: int,
        uncertainty_threshold: float,
        stop_on_uncertainty: bool,
    ) -> ImaginationBranch:

        state = initial_state.copy()

        steps: list[ImaginationStep] = []
        uncertainties: list[float] = []

        cumulative_reward = 0.0
        terminated_early = False

        for index in range(horizon):
            action = _validated_vector(
                actions[index],
                name=f"action[{index}]",
                dimension=model.action_dim,
            )

            raw_transition = model.predict_transition(
                state.copy(),
                action.copy(),
            )

            transition = _validate_transition(
                raw_transition,
                state_dim=model.state_dim,
            )

            uncertainties.append(
                transition.uncertainty
            )

            # Do not advance into an untrusted predicted
            # state when uncertainty exceeds the threshold.
            if (
                stop_on_uncertainty
                and transition.uncertainty
                > uncertainty_threshold
            ):
                terminated_early = True
                break

            state = transition.next_state.copy()

            cumulative_reward += transition.reward

            steps.append(
                ImaginationStep(
                    step=index + 1,
                    action=action.tolist(),
                    state=state.tolist(),
                    uncertainty=transition.uncertainty,
                    reward=transition.reward,
                    terminal=transition.terminal,
                )
            )

            if transition.terminal:
                terminated_early = (
                    index + 1 < horizon
                )
                break

        mean_uncertainty = (
            float(np.mean(uncertainties))
            if uncertainties
            else 0.0
        )

        return ImaginationBranch(
            branch_id=uuid.uuid4().hex,
            steps=steps,
            final_state=state.tolist(),
            cumulative_reward=float(
                cumulative_reward
            ),
            mean_uncertainty=mean_uncertainty,
            terminated_early=terminated_early,
        )

    def imagine(
        self,
        request: ImaginationRequest,
    ) -> ImaginationResult:

        started = time.perf_counter()
        model = self.model

        if request.horizon > self.max_horizon:
            raise ImaginationLimitError(
                "Requested horizon exceeds configured limit"
            )

        branch_count = len(
            request.action_sequences
        )

        if branch_count > self.max_branches:
            raise ImaginationLimitError(
                "Requested branches exceed configured limit"
            )

        initial_state = _validated_vector(
            request.initial_state,
            name="initial_state",
            dimension=model.state_dim,
        )

        for branch_index, actions in enumerate(
            request.action_sequences
        ):
            if len(actions) != request.horizon:
                raise InvalidLatentStateError(
                    "Every action sequence must contain "
                    f"exactly {request.horizon} actions; "
                    f"branch {branch_index} has {len(actions)}"
                )

        branches = []

        try:
            for actions in request.action_sequences:
                branch = self._rollout(
                    model=model,
                    initial_state=initial_state,
                    actions=actions,
                    horizon=request.horizon,
                    uncertainty_threshold=(
                        request.uncertainty_threshold
                    ),
                    stop_on_uncertainty=(
                        request.stop_on_uncertainty
                    ),
                )

                branches.append(branch)

        except (
            ImaginationError,
            InvalidLatentStateError,
        ):
            raise

        except Exception as exc:
            logger.exception(
                "Imagination execution failed"
            )

            raise ImaginationError(
                "World-model imagination failed"
            ) from exc

        duration_ms = (
            time.perf_counter() - started
        ) * 1000

        result = ImaginationResult(
            imagination_id=uuid.uuid4().hex,
            world_id=request.world_id,
            model_name=model.name,
            created_at=datetime.now(
                timezone.utc
            ),
            horizon=request.horizon,
            branch_count=len(branches),
            duration_ms=duration_ms,
            branches=branches,
            metadata=request.metadata,
        )

        with self._lock:
            self._results[
                result.imagination_id
            ] = result

            # Bound memory usage.
            while len(self._results) > self.max_history:
                oldest_id = next(
                    iter(self._results)
                )
                del self._results[oldest_id]

        log_imagination(
            world_id=request.world_id,
            rollout_count=len(branches),
            horizon=request.horizon,
            duration_ms=duration_ms,
            success=True,
        )

        record_imagination(
            duration_seconds=duration_ms / 1000,
            branches=len(branches),
            horizon=request.horizon,
            success=True,
        )

        return result

    async def imagine_async(
        self,
        request: ImaginationRequest,
    ) -> ImaginationResult:
        """
        Execute synchronous model inference in a worker
        thread so the FastAPI event loop is not blocked.

        For GPU-heavy production workloads, use a
        dedicated inference queue or model server.
        """

        return await asyncio.to_thread(
            self.imagine,
            request,
        )

    def counterfactual(
        self,
        request: CounterfactualRequest,
    ) -> CounterfactualResult:

        if (
            len(request.baseline_actions)
            != len(request.alternative_actions)
        ):
            raise InvalidLatentStateError(
                "Baseline and alternative action "
                "sequences must have equal length"
            )

        horizon = len(
            request.baseline_actions
        )

        imagination = self.imagine(
            ImaginationRequest(
                world_id=request.world_id,
                initial_state=request.initial_state,
                action_sequences=[
                    request.baseline_actions,
                    request.alternative_actions,
                ],
                horizon=horizon,
                uncertainty_threshold=(
                    request.uncertainty_threshold
                ),
            )
        )

        baseline, alternative = imagination.branches

        baseline_state = np.asarray(
            baseline.final_state,
            dtype=np.float64,
        )

        alternative_state = np.asarray(
            alternative.final_state,
            dtype=np.float64,
        )

        distance = float(
            np.linalg.norm(
                baseline_state - alternative_state
            )
        )

        return CounterfactualResult(
            counterfactual_id=uuid.uuid4().hex,
            world_id=request.world_id,
            baseline=baseline,
            alternative=alternative,
            final_state_distance=distance,
            reward_difference=(
                alternative.cumulative_reward
                - baseline.cumulative_reward
            ),
            created_at=datetime.now(
                timezone.utc
            ),
        )

    def get_result(
        self,
        imagination_id: str,
    ) -> ImaginationResult | None:

        with self._lock:
            return self._results.get(
                imagination_id
            )

    def list_results(
        self,
        *,
        world_id: str | None = None,
        limit: int = 50,
    ) -> list[ImaginationResult]:

        if not 1 <= limit <= 1000:
            raise ValueError(
                "limit must be between 1 and 1000"
            )

        with self._lock:
            results = list(
                self._results.values()
            )

        if world_id is not None:
            results = [
                result
                for result in results
                if result.world_id == world_id
            ]

        results.sort(
            key=lambda item: item.created_at,
            reverse=True,
        )

        return results[:limit]

    def evaluate_branches(
        self,
        result: ImaginationResult,
        *,
        uncertainty_penalty: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Rank imagined trajectories by cumulative reward
        minus an uncertainty penalty.

        This is a heuristic score, not a validated
        measure of real-world safety or correctness.
        """

        if (
            not math.isfinite(uncertainty_penalty)
            or uncertainty_penalty < 0
        ):
            raise ValueError(
                "uncertainty_penalty must be nonnegative"
            )

        rankings = []

        for branch in result.branches:
            score = (
                branch.cumulative_reward
                - uncertainty_penalty
                * branch.mean_uncertainty
            )

            rankings.append({
                "branch_id": branch.branch_id,
                "score": float(score),
                "cumulative_reward": (
                    branch.cumulative_reward
                ),
                "mean_uncertainty": (
                    branch.mean_uncertainty
                ),
                "steps": len(branch.steps),
                "terminated_early": (
                    branch.terminated_early
                ),
            })

        rankings.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return rankings

    def status(self) -> dict[str, Any]:

        with self._lock:
            model = self._model
            history_count = len(
                self._results
            )

        return {
            "service": "imagination",
            "model_loaded": model is not None,
            "model_name": (
                model.name
                if model is not None
                else None
            ),
            "state_dim": (
                model.state_dim
                if model is not None
                else None
            ),
            "action_dim": (
                model.action_dim
                if model is not None
                else None
            ),
            "max_branches": self.max_branches,
            "max_horizon": self.max_horizon,
            "stored_results": history_count,
            "execution_mode": "model_adapter",
        }


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_imagination_service(
    model: WorldModelAdapter | None = None,
) -> ImaginationService:
    """
    Create a service using project configuration.

    No demonstration model is loaded automatically.
    """

    settings = get_settings()

    return ImaginationService(
        model=model,
        max_branches=settings.imagination.max_branches,
        max_horizon=settings.imagination.rollout_horizon,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:

    model = DemoWorldModel(
        state_dim=4,
        action_dim=4,
    )

    service = ImaginationService(
        model=model,
        max_branches=8,
        max_horizon=16,
    )

    request = ImaginationRequest(
        world_id="demo-world-001",
        initial_state=[
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        action_sequences=[
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
        ],
        horizon=3,
    )

    result = service.imagine(request)

    print("=" * 65)
    print("NEUROWORLD AI — IMAGINATION SERVICE")
    print("=" * 65)

    print(
        result.model_dump_json(indent=2)
    )

    print()
    print("Branch rankings:")

    for ranking in service.evaluate_branches(
        result
    ):
        print(ranking)

    counterfactual = service.counterfactual(
        CounterfactualRequest(
            world_id="demo-world-001",
            initial_state=[0, 0, 0, 0],
            baseline_actions=[
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            alternative_actions=[
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        )
    )

    print()
    print("Counterfactual comparison:")

    print(
        counterfactual.model_dump_json(
            indent=2
        )
    )

    print()
    print("Service status:")
    print(service.status())

    print()
    print(
        "Demonstration complete. "
        "Results use synthetic deterministic dynamics."
    )


if __name__ == "__main__":
    main()
