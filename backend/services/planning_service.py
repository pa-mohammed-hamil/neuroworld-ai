
"""
NeuroWorld AI
backend/app/services/planning_service.py

Model-based planning for neural world models.

Features:
    - Model Predictive Control (MPC)
    - Cross-Entropy Method (CEM)
    - Random shooting
    - Goal-conditioned planning
    - Latent-state planning
    - Action constraints
    - Uncertainty-aware scoring
    - Candidate trajectory evaluation
    - Replanning
    - Planning history
    - Async-compatible execution
    - Telemetry integration

Dependencies:
    pip install numpy pydantic

The planner operates on predicted latent states.
It does not execute real-world hardware actions.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid

from datetime import datetime, timezone
from typing import Any, Literal, Sequence

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from ..core.exceptions import (
    ActionConstraintError,
    ModelNotLoadedError,
    NoValidPlanError,
    PlanningError,
)
from ..core.logging import get_logger
from ..core.telemetry import record_planning

from .imagination_service import (
    WorldModelAdapter,
    Transition,
)


logger = get_logger("services.planning")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class PlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(min_length=1, max_length=128)

    initial_state: list[float] = Field(min_length=1)
    goal_state: list[float] = Field(min_length=1)

    algorithm: Literal[
        "cem",
        "random_shooting",
    ] = "cem"

    horizon: int = Field(default=12, ge=1, le=128)
    candidate_count: int = Field(
        default=128,
        ge=2,
        le=4096,
    )

    iterations: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    elite_fraction: float = Field(
        default=0.1,
        gt=0,
        le=1,
    )

    action_min: float = -1.0
    action_max: float = 1.0

    goal_weight: float = Field(
        default=1.0,
        ge=0,
    )

    action_penalty: float = Field(
        default=0.01,
        ge=0,
    )

    uncertainty_penalty: float = Field(
        default=0.5,
        ge=0,
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )

    seed: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_constraints(self):
        if (
            not math.isfinite(self.action_min)
            or not math.isfinite(self.action_max)
        ):
            raise ValueError(
                "Action limits must be finite"
            )

        if self.action_min >= self.action_max:
            raise ValueError(
                "action_min must be less than action_max"
            )

        return self


class PlannedStep(BaseModel):
    step: int
    action: list[float]
    predicted_state: list[float]
    uncertainty: float
    reward: float
    goal_distance: float


class PlanningResult(BaseModel):
    plan_id: str
    world_id: str
    model_name: str
    algorithm: str
    created_at: datetime

    horizon: int
    candidate_count: int
    iterations: int

    best_score: float
    predicted_final_distance: float
    total_predicted_reward: float
    mean_uncertainty: float

    first_action: list[float]
    action_sequence: list[list[float]]
    predicted_trajectory: list[PlannedStep]

    duration_ms: float
    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class PlanningCandidate(BaseModel):
    score: float
    final_distance: float
    cumulative_reward: float
    mean_uncertainty: float
    valid: bool


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _vector(
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
    except (TypeError, ValueError) as exc:
        raise PlanningError(
            f"{name} must contain numeric values"
        ) from exc

    if array.shape != (dimension,):
        raise PlanningError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.all(np.isfinite(array)):
        raise PlanningError(
            f"{name} must contain finite values"
        )

    return array.copy()


def _validated_transition(
    transition: Transition,
    *,
    state_dim: int,
) -> Transition:

    if not isinstance(transition, Transition):
        raise PlanningError(
            "World model returned invalid transition"
        )

    state = _vector(
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
        raise PlanningError(
            "World model returned invalid uncertainty"
        )

    if not math.isfinite(reward):
        raise PlanningError(
            "World model returned invalid reward"
        )

    return Transition(
        next_state=state,
        uncertainty=uncertainty,
        reward=reward,
        terminal=bool(transition.terminal),
    )


# ============================================================
# PLANNING SERVICE
# ============================================================

class PlanningService:
    """
    Goal-conditioned latent-space planning service.

    Plans are predictions only. They must not be
    interpreted as authorization to control hardware.
    """

    def __init__(
        self,
        model: WorldModelAdapter | None = None,
        *,
        max_horizon: int = 128,
        max_candidates: int = 4096,
        max_iterations: int = 100,
        max_history: int = 100,
    ) -> None:

        if min(
            max_horizon,
            max_candidates,
            max_iterations,
            max_history,
        ) < 1:
            raise ValueError(
                "Planning limits must be positive"
            )

        self._model = model

        self.max_horizon = max_horizon
        self.max_candidates = max_candidates
        self.max_iterations = max_iterations
        self.max_history = max_history

        self._history: dict[
            str,
            PlanningResult,
        ] = {}

        self._lock = threading.RLock()

    @property
    def model(self) -> WorldModelAdapter:
        with self._lock:
            model = self._model

        if model is None:
            raise ModelNotLoadedError(
                "Planning world model is not loaded"
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
            "Planning model registered: %s",
            model.name,
        )

    def _validate_request(
        self,
        request: PlanningRequest,
        model: WorldModelAdapter,
    ) -> tuple[np.ndarray, np.ndarray]:

        if request.horizon > self.max_horizon:
            raise PlanningError(
                "Requested horizon exceeds service limit"
            )

        if request.candidate_count > self.max_candidates:
            raise PlanningError(
                "Requested candidate count exceeds limit"
            )

        if request.iterations > self.max_iterations:
            raise PlanningError(
                "Requested iterations exceed limit"
            )

        initial_state = _vector(
            request.initial_state,
            name="initial_state",
            dimension=model.state_dim,
        )

        goal_state = _vector(
            request.goal_state,
            name="goal_state",
            dimension=model.state_dim,
        )

        return initial_state, goal_state

    def _evaluate_candidate(
        self,
        *,
        model: WorldModelAdapter,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        actions: np.ndarray,
        request: PlanningRequest,
        return_steps: bool = False,
    ) -> tuple[PlanningCandidate, list[PlannedStep]]:

        state = initial_state.copy()

        steps: list[PlannedStep] = []
        uncertainties: list[float] = []

        total_reward = 0.0
        action_cost = 0.0

        valid = True

        for index, action in enumerate(actions):

            if (
                np.any(action < request.action_min)
                or np.any(action > request.action_max)
            ):
                valid = False
                break

            transition = _validated_transition(
                model.predict_transition(
                    state.copy(),
                    action.copy(),
                ),
                state_dim=model.state_dim,
            )

            if (
                transition.uncertainty
                > request.uncertainty_threshold
            ):
                valid = False
                break

            state = transition.next_state.copy()

            total_reward += transition.reward

            action_cost += float(
                np.sum(action ** 2)
            )

            uncertainties.append(
                transition.uncertainty
            )

            distance = float(
                np.linalg.norm(
                    state - goal_state
                )
            )

            if return_steps:
                steps.append(
                    PlannedStep(
                        step=index + 1,
                        action=action.tolist(),
                        predicted_state=state.tolist(),
                        uncertainty=transition.uncertainty,
                        reward=transition.reward,
                        goal_distance=distance,
                    )
                )

            if transition.terminal:
                break

        final_distance = float(
            np.linalg.norm(
                state - goal_state
            )
        )

        mean_uncertainty = (
            float(np.mean(uncertainties))
            if uncertainties
            else 0.0
        )

        if not valid:
            score = float("-inf")
        else:
            score = (
                total_reward
                - request.goal_weight * final_distance
                - request.action_penalty * action_cost
                - request.uncertainty_penalty
                * mean_uncertainty
            )

        candidate = PlanningCandidate(
            score=score if valid else -1e30,
            final_distance=final_distance,
            cumulative_reward=float(total_reward),
            mean_uncertainty=mean_uncertainty,
            valid=valid,
        )

        return candidate, steps

    def _random_shooting(
        self,
        *,
        model: WorldModelAdapter,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        request: PlanningRequest,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, PlanningCandidate]:

        best_actions = None
        best_candidate = None

        for _ in range(request.candidate_count):

            actions = rng.uniform(
                request.action_min,
                request.action_max,
                size=(
                    request.horizon,
                    model.action_dim,
                ),
            )

            candidate, _ = self._evaluate_candidate(
                model=model,
                initial_state=initial_state,
                goal_state=goal_state,
                actions=actions,
                request=request,
            )

            if not candidate.valid:
                continue

            if (
                best_candidate is None
                or candidate.score
                > best_candidate.score
            ):
                best_candidate = candidate
                best_actions = actions.copy()

        if best_actions is None:
            raise NoValidPlanError(
                "No valid action sequence found"
            )

        return best_actions, best_candidate

    def _cem(
        self,
        *,
        model: WorldModelAdapter,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        request: PlanningRequest,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, PlanningCandidate]:

        action_shape = (
            request.horizon,
            model.action_dim,
        )

        mean = np.full(
            action_shape,
            (request.action_min + request.action_max)
            / 2,
            dtype=np.float64,
        )

        std = np.full(
            action_shape,
            (request.action_max - request.action_min)
            / 2,
            dtype=np.float64,
        )

        elite_count = max(
            1,
            int(
                request.candidate_count
                * request.elite_fraction
            ),
        )

        best_actions = None
        best_candidate = None

        for iteration in range(
            request.iterations
        ):

            sampled_actions = rng.normal(
                loc=mean,
                scale=std,
                size=(
                    request.candidate_count,
                    request.horizon,
                    model.action_dim,
                ),
            )

            sampled_actions = np.clip(
                sampled_actions,
                request.action_min,
                request.action_max,
            )

            valid_candidates = []

            for actions in sampled_actions:

                candidate, _ = self._evaluate_candidate(
                    model=model,
                    initial_state=initial_state,
                    goal_state=goal_state,
                    actions=actions,
                    request=request,
                )

                if not candidate.valid:
                    continue

                valid_candidates.append(
                    (candidate.score, actions.copy())
                )

                if (
                    best_candidate is None
                    or candidate.score
                    > best_candidate.score
                ):
                    best_candidate = candidate
                    best_actions = actions.copy()

            if not valid_candidates:
                continue

            valid_candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            elite_actions = np.stack([
                item[1]
                for item in valid_candidates[
                    :elite_count
                ]
            ])

            new_mean = np.mean(
                elite_actions,
                axis=0,
            )

            new_std = np.std(
                elite_actions,
                axis=0,
            )

            # Smoothed updates prevent sudden collapse.
            smoothing = 0.2

            mean = (
                smoothing * mean
                + (1 - smoothing) * new_mean
            )

            std = (
                smoothing * std
                + (1 - smoothing)
                * np.maximum(new_std, 0.01)
            )

        if best_actions is None:
            raise NoValidPlanError(
                "CEM could not find a valid plan"
            )

        return best_actions, best_candidate

    def plan(
        self,
        request: PlanningRequest,
    ) -> PlanningResult:

        started = time.perf_counter()

        model = self.model

        initial_state, goal_state = (
            self._validate_request(
                request,
                model,
            )
        )

        rng = np.random.default_rng(
            request.seed
        )

        try:
            if request.algorithm == "cem":
                best_actions, best_candidate = (
                    self._cem(
                        model=model,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        request=request,
                        rng=rng,
                    )
                )

            elif request.algorithm == "random_shooting":
                best_actions, best_candidate = (
                    self._random_shooting(
                        model=model,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        request=request,
                        rng=rng,
                    )
                )

            else:
                raise PlanningError(
                    "Unsupported planning algorithm"
                )

            # Re-evaluate the selected candidate to
            # capture its predicted trajectory.
            final_candidate, steps = (
                self._evaluate_candidate(
                    model=model,
                    initial_state=initial_state,
                    goal_state=goal_state,
                    actions=best_actions,
                    request=request,
                    return_steps=True,
                )
            )

            if not final_candidate.valid:
                raise NoValidPlanError(
                    "Selected plan failed final validation"
                )

        except (PlanningError, NoValidPlanError):
            record_planning(
                algorithm=request.algorithm,
                duration_seconds=(
                    time.perf_counter() - started
                ),
                candidate_count=request.candidate_count,
                success=False,
            )
            raise

        except Exception as exc:
            logger.exception(
                "Planning execution failed"
            )

            record_planning(
                algorithm=request.algorithm,
                duration_seconds=(
                    time.perf_counter() - started
                ),
                candidate_count=request.candidate_count,
                success=False,
            )

            raise PlanningError(
                "Planning execution failed"
            ) from exc

        duration_ms = (
            time.perf_counter() - started
        ) * 1000

        result = PlanningResult(
            plan_id=uuid.uuid4().hex,
            world_id=request.world_id,
            model_name=model.name,
            algorithm=request.algorithm,
            created_at=datetime.now(
                timezone.utc
            ),
            horizon=request.horizon,
            candidate_count=request.candidate_count,
            iterations=(
                request.iterations
                if request.algorithm == "cem"
                else 1
            ),
            best_score=final_candidate.score,
            predicted_final_distance=(
                final_candidate.final_distance
            ),
            total_predicted_reward=(
                final_candidate.cumulative_reward
            ),
            mean_uncertainty=(
                final_candidate.mean_uncertainty
            ),
            first_action=best_actions[0].tolist(),
            action_sequence=best_actions.tolist(),
            predicted_trajectory=steps,
            duration_ms=duration_ms,
            metadata=request.metadata,
        )

        with self._lock:
            self._history[result.plan_id] = result

            while len(self._history) > self.max_history:
                oldest_id = next(
                    iter(self._history)
                )
                del self._history[oldest_id]

        record_planning(
            algorithm=request.algorithm,
            duration_seconds=duration_ms / 1000,
            candidate_count=request.candidate_count,
            success=True,
        )

        logger.info(
            "Plan generated: id=%s algorithm=%s "
            "score=%.4f duration_ms=%.2f",
            result.plan_id,
            result.algorithm,
            result.best_score,
            result.duration_ms,
        )

        return result

    async def plan_async(
        self,
        request: PlanningRequest,
    ) -> PlanningResult:
        """
        Run CPU-bound planning outside the FastAPI
        event loop.

        For heavy neural models, use dedicated GPU
        workers and a bounded job queue.
        """

        return await asyncio.to_thread(
            self.plan,
            request,
        )

    def replan(
        self,
        *,
        previous_plan_id: str,
        current_state: Sequence[float],
        goal_state: Sequence[float],
        seed: int | None = None,
    ) -> PlanningResult:
        """
        Replan using the observed current state.

        Previous plans are not automatically executed.
        """

        previous = self.get_plan(
            previous_plan_id
        )

        request = PlanningRequest(
            world_id=previous.world_id,
            initial_state=list(current_state),
            goal_state=list(goal_state),
            algorithm=previous.algorithm,
            horizon=previous.horizon,
            candidate_count=previous.candidate_count,
            iterations=previous.iterations,
            seed=seed,
        )

        return self.plan(request)

    def get_plan(
        self,
        plan_id: str,
    ) -> PlanningResult:

        with self._lock:
            result = self._history.get(
                plan_id
            )

        if result is None:
            raise PlanningError(
                "Plan not found"
            )

        return result

    def list_plans(
        self,
        *,
        world_id: str | None = None,
        limit: int = 50,
    ) -> list[PlanningResult]:

        if not 1 <= limit <= 1000:
            raise ValueError(
                "limit must be between 1 and 1000"
            )

        with self._lock:
            results = list(
                self._history.values()
            )

        if world_id is not None:
            results = [
                item
                for item in results
                if item.world_id == world_id
            ]

        results.sort(
            key=lambda item: item.created_at,
            reverse=True,
        )

        return results[:limit]

    def status(self) -> dict[str, Any]:

        with self._lock:
            model = self._model
            plan_count = len(
                self._history
            )

        return {
            "service": "planning",
            "model_loaded": model is not None,
            "model_name": (
                model.name
                if model is not None
                else None
            ),
            "max_horizon": self.max_horizon,
            "max_candidates": self.max_candidates,
            "max_iterations": self.max_iterations,
            "stored_plans": plan_count,
            "algorithms": [
                "cem",
                "random_shooting",
            ],
            "hardware_execution_enabled": False,
        }


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_planning_service(
    model: WorldModelAdapter | None = None,
) -> PlanningService:

    return PlanningService(
        model=model,
        max_horizon=128,
        max_candidates=4096,
        max_iterations=100,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:

    from .imagination_service import (
        DemoWorldModel,
    )

    model = DemoWorldModel(
        state_dim=4,
        action_dim=4,
        scale=0.1,
    )

    service = PlanningService(
        model=model,
    )

    request = PlanningRequest(
        world_id="demo-world-001",
        initial_state=[
            0.0, 0.0, 0.0, 0.0
        ],
        goal_state=[
            1.0, 1.0, 0.0, 0.0
        ],
        algorithm="cem",
        horizon=8,
        candidate_count=64,
        iterations=4,
        seed=42,
    )

    result = service.plan(
        request
    )

    print("=" * 65)
    print("NEUROWORLD AI — PLANNING SERVICE")
    print("=" * 65)

    print(
        result.model_dump_json(indent=2)
    )

    print()
    print("Recommended first action:")
    print(result.first_action)

    print()
    print("Service status:")
    print(service.status())

    print()
    print(
        "Demonstration complete. "
        "Planning used synthetic model dynamics."
    )


if __name__ == "__main__":
    main()
