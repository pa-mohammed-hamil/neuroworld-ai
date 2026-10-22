
"""
NeuroWorld AI
backend/app/services/prediction_service.py

World-model prediction service.

Features:
    - Next-state prediction
    - Multi-step future-state prediction
    - Action-conditioned trajectories
    - Alternative future scenarios
    - Batch prediction
    - Uncertainty tracking
    - Uncertainty-based early termination
    - Prediction history
    - Async-compatible execution
    - Telemetry and structured logging
    - Model adapter integration

Dependencies:
    pip install numpy pydantic psutil prometheus-client

The service requires a registered world-model adapter.
It does not generate neural predictions independently.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid

from collections import OrderedDict
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
    ModelNotLoadedError,
    PredictionError,
    PredictionHorizonExceededError,
    UncertaintyTooHighError,
)
from ..core.logging import get_logger
from ..core.telemetry import (
    record_world_prediction,
)

from .imagination_service import (
    Transition,
    WorldModelAdapter,
)


logger = get_logger("services.prediction")


# ============================================================
# REQUEST MODELS
# ============================================================

class NextStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    current_state: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    action: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class FuturePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    initial_state: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    actions: list[list[float]] = Field(
        min_length=1,
        max_length=128,
    )

    horizon: int = Field(
        ge=1,
        le=128,
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )

    stop_on_uncertainty: bool = True

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_action_count(self):
        if len(self.actions) != self.horizon:
            raise ValueError(
                "Action count must equal horizon"
            )

        return self


class ScenarioPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    initial_state: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    scenarios: dict[
        str,
        list[list[float]],
    ] = Field(
        min_length=1,
        max_length=32,
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class PredictedStep(BaseModel):
    step: int
    action: list[float]
    predicted_state: list[float]
    uncertainty: float = Field(
        ge=0,
        le=1,
    )
    predicted_reward: float
    terminal: bool


class PredictionResult(BaseModel):
    prediction_id: str
    world_id: str
    model_name: str
    prediction_type: Literal[
        "next_state",
        "future",
    ]
    created_at: datetime

    initial_state: list[float]
    final_state: list[float]

    requested_horizon: int
    completed_steps: int

    steps: list[PredictedStep]

    cumulative_reward: float
    mean_uncertainty: float
    max_uncertainty: float

    terminated_early: bool
    termination_reason: str | None = None

    duration_ms: float

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class ScenarioPredictionResult(BaseModel):
    scenario_prediction_id: str
    world_id: str
    created_at: datetime
    scenarios: dict[str, PredictionResult]
    final_state_distances: dict[str, float]


class PredictionServiceStatus(BaseModel):
    service: str
    model_loaded: bool
    model_name: str | None
    state_dim: int | None
    action_dim: int | None
    max_horizon: int
    max_batch_size: int
    stored_predictions: int


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
        raise PredictionError(
            f"{name} must contain numeric values"
        ) from exc

    if array.shape != (dimension,):
        raise PredictionError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.all(np.isfinite(array)):
        raise PredictionError(
            f"{name} contains non-finite values"
        )

    return array.copy()


def _transition(
    value: Transition,
    *,
    state_dim: int,
) -> Transition:
    if not isinstance(value, Transition):
        raise PredictionError(
            "World model returned an invalid transition"
        )

    state = _vector(
        value.next_state,
        name="next_state",
        dimension=state_dim,
    )

    try:
        uncertainty = float(value.uncertainty)
        reward = float(value.reward)
    except (TypeError, ValueError) as exc:
        raise PredictionError(
            "World model returned invalid metrics"
        ) from exc

    if (
        not math.isfinite(uncertainty)
        or not 0 <= uncertainty <= 1
    ):
        raise PredictionError(
            "Prediction uncertainty must be in [0, 1]"
        )

    if not math.isfinite(reward):
        raise PredictionError(
            "Predicted reward must be finite"
        )

    return Transition(
        next_state=state,
        uncertainty=uncertainty,
        reward=reward,
        terminal=bool(value.terminal),
    )


# ============================================================
# PREDICTION SERVICE
# ============================================================

class PredictionService:
    """
    World-model prediction orchestration.

    Predictions are simulations of possible future
    latent states, not verified real-world outcomes.

    Storage is bounded and process-local.
    """

    def __init__(
        self,
        model: WorldModelAdapter | None = None,
        *,
        max_horizon: int = 128,
        max_batch_size: int = 32,
        max_history: int = 200,
    ) -> None:
        if min(
            max_horizon,
            max_batch_size,
            max_history,
        ) < 1:
            raise ValueError(
                "Service limits must be positive"
            )

        self._model = model

        self.max_horizon = max_horizon
        self.max_batch_size = max_batch_size
        self.max_history = max_history

        self._history: OrderedDict[
            str,
            PredictionResult,
        ] = OrderedDict()

        self._lock = threading.RLock()

    @property
    def model(self) -> WorldModelAdapter:
        with self._lock:
            model = self._model

        if model is None:
            raise ModelNotLoadedError(
                "No prediction world model is loaded"
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
            "Prediction model registered: %s",
            model.name,
        )

    def unregister_model(self) -> None:
        with self._lock:
            self._model = None

    def _store(
        self,
        result: PredictionResult,
    ) -> None:
        with self._lock:
            self._history[
                result.prediction_id
            ] = result

            while len(self._history) > self.max_history:
                self._history.popitem(last=False)

    def _predict_sequence(
        self,
        *,
        model: WorldModelAdapter,
        world_id: str,
        initial_state: Sequence[float],
        actions: Sequence[Sequence[float]],
        uncertainty_threshold: float,
        stop_on_uncertainty: bool,
        prediction_type: Literal[
            "next_state",
            "future",
        ],
        metadata: dict[str, str] | None = None,
    ) -> PredictionResult:
        started = time.perf_counter()

        state = _vector(
            initial_state,
            name="initial_state",
            dimension=model.state_dim,
        )

        initial = state.tolist()
        requested_horizon = len(actions)

        if requested_horizon > self.max_horizon:
            raise PredictionHorizonExceededError(
                "Prediction horizon exceeds service limit"
            )

        steps: list[PredictedStep] = []
        uncertainties: list[float] = []

        cumulative_reward = 0.0
        termination_reason = None

        try:
            for index, raw_action in enumerate(actions):
                action = _vector(
                    raw_action,
                    name=f"action[{index}]",
                    dimension=model.action_dim,
                )

                raw_transition = (
                    model.predict_transition(
                        state.copy(),
                        action.copy(),
                    )
                )

                transition = _transition(
                    raw_transition,
                    state_dim=model.state_dim,
                )

                if (
                    transition.uncertainty
                    > uncertainty_threshold
                ):
                    if stop_on_uncertainty:
                        termination_reason = (
                            "uncertainty_threshold_exceeded"
                        )
                        break

                state = transition.next_state.copy()

                cumulative_reward += transition.reward

                uncertainties.append(
                    transition.uncertainty
                )

                steps.append(
                    PredictedStep(
                        step=index + 1,
                        action=action.tolist(),
                        predicted_state=state.tolist(),
                        uncertainty=transition.uncertainty,
                        predicted_reward=transition.reward,
                        terminal=transition.terminal,
                    )
                )

                if transition.terminal:
                    termination_reason = (
                        "model_terminal_state"
                    )
                    break

        except PredictionError:
            record_world_prediction(
                duration_seconds=(
                    time.perf_counter() - started
                ),
                success=False,
            )
            raise

        except Exception as exc:
            logger.exception(
                "World-model prediction failed"
            )

            record_world_prediction(
                duration_seconds=(
                    time.perf_counter() - started
                ),
                success=False,
            )

            raise PredictionError(
                "World-model prediction failed"
            ) from exc

        duration_ms = (
            time.perf_counter() - started
        ) * 1000

        result = PredictionResult(
            prediction_id=uuid.uuid4().hex,
            world_id=world_id,
            model_name=model.name,
            prediction_type=prediction_type,
            created_at=datetime.now(
                timezone.utc
            ),
            initial_state=initial,
            final_state=state.tolist(),
            requested_horizon=requested_horizon,
            completed_steps=len(steps),
            steps=steps,
            cumulative_reward=float(
                cumulative_reward
            ),
            mean_uncertainty=(
                float(np.mean(uncertainties))
                if uncertainties
                else 0.0
            ),
            max_uncertainty=(
                float(np.max(uncertainties))
                if uncertainties
                else 0.0
            ),
            terminated_early=(
                len(steps) < requested_horizon
            ),
            termination_reason=termination_reason,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self._store(result)

        record_world_prediction(
            duration_seconds=duration_ms / 1000,
            uncertainty=(
                result.mean_uncertainty
                if uncertainties
                else None
            ),
            success=True,
        )

        logger.info(
            "Prediction completed: id=%s "
            "world=%s steps=%s duration_ms=%.2f",
            result.prediction_id,
            result.world_id,
            result.completed_steps,
            result.duration_ms,
        )

        return result

    def predict_next_state(
        self,
        request: NextStateRequest,
    ) -> PredictionResult:
        """
        Predict one transition from the current state
        and action.

        Rejects a prediction when uncertainty exceeds
        the configured threshold.
        """

        result = self._predict_sequence(
            model=self.model,
            world_id=request.world_id,
            initial_state=request.current_state,
            actions=[request.action],
            uncertainty_threshold=(
                request.uncertainty_threshold
            ),
            stop_on_uncertainty=True,
            prediction_type="next_state",
            metadata=request.metadata,
        )

        if result.completed_steps == 0:
            raise UncertaintyTooHighError(
                "Next-state prediction exceeds "
                "the uncertainty threshold"
            )

        return result

    def predict_future(
        self,
        request: FuturePredictionRequest,
    ) -> PredictionResult:
        """
        Predict an action-conditioned future sequence.
        """

        return self._predict_sequence(
            model=self.model,
            world_id=request.world_id,
            initial_state=request.initial_state,
            actions=request.actions,
            uncertainty_threshold=(
                request.uncertainty_threshold
            ),
            stop_on_uncertainty=(
                request.stop_on_uncertainty
            ),
            prediction_type="future",
            metadata=request.metadata,
        )

    def predict_batch(
        self,
        requests: Sequence[
            FuturePredictionRequest
        ],
    ) -> list[PredictionResult]:
        """
        Process multiple prediction requests.

        This is sequential orchestration, not GPU
        vectorized batch inference.
        """

        if not requests:
            raise PredictionError(
                "Prediction batch cannot be empty"
            )

        if len(requests) > self.max_batch_size:
            raise PredictionError(
                "Prediction batch exceeds configured limit"
            )

        return [
            self.predict_future(request)
            for request in requests
        ]

    def predict_scenarios(
        self,
        request: ScenarioPredictionRequest,
    ) -> ScenarioPredictionResult:
        """
        Compare alternative action-conditioned futures
        from the same initial state.
        """

        scenarios = {}

        for name, actions in request.scenarios.items():
            if not name.strip():
                raise PredictionError(
                    "Scenario names cannot be empty"
                )

            scenarios[name] = self.predict_future(
                FuturePredictionRequest(
                    world_id=request.world_id,
                    initial_state=request.initial_state,
                    actions=actions,
                    horizon=len(actions),
                    uncertainty_threshold=(
                        request.uncertainty_threshold
                    ),
                    stop_on_uncertainty=True,
                    metadata={
                        "scenario": name,
                    },
                )
            )

        names = list(scenarios)

        distances = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                first = scenarios[names[i]]
                second = scenarios[names[j]]

                state_a = np.asarray(
                    first.final_state,
                    dtype=np.float64,
                )

                state_b = np.asarray(
                    second.final_state,
                    dtype=np.float64,
                )

                distance = float(
                    np.linalg.norm(
                        state_a - state_b
                    )
                )

                distances[
                    f"{names[i]}__vs__{names[j]}"
                ] = distance

        return ScenarioPredictionResult(
            scenario_prediction_id=uuid.uuid4().hex,
            world_id=request.world_id,
            created_at=datetime.now(
                timezone.utc
            ),
            scenarios=scenarios,
            final_state_distances=distances,
        )

    async def predict_next_state_async(
        self,
        request: NextStateRequest,
    ) -> PredictionResult:
        return await asyncio.to_thread(
            self.predict_next_state,
            request,
        )

    async def predict_future_async(
        self,
        request: FuturePredictionRequest,
    ) -> PredictionResult:
        return await asyncio.to_thread(
            self.predict_future,
            request,
        )

    async def predict_batch_async(
        self,
        requests: Sequence[
            FuturePredictionRequest
        ],
    ) -> list[PredictionResult]:
        return await asyncio.to_thread(
            self.predict_batch,
            requests,
        )

    def get_prediction(
        self,
        prediction_id: str,
    ) -> PredictionResult | None:
        with self._lock:
            return self._history.get(
                prediction_id
            )

    def list_predictions(
        self,
        *,
        world_id: str | None = None,
        limit: int = 50,
    ) -> list[PredictionResult]:
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
                result
                for result in results
                if result.world_id == world_id
            ]

        results.sort(
            key=lambda result: result.created_at,
            reverse=True,
        )

        return results[:limit]

    def clear_history(self) -> int:
        with self._lock:
            count = len(self._history)
            self._history.clear()

        return count

    def status(self) -> PredictionServiceStatus:
        with self._lock:
            model = self._model
            history_count = len(self._history)

        return PredictionServiceStatus(
            service="prediction",
            model_loaded=model is not None,
            model_name=(
                model.name
                if model is not None
                else None
            ),
            state_dim=(
                model.state_dim
                if model is not None
                else None
            ),
            action_dim=(
                model.action_dim
                if model is not None
                else None
            ),
            max_horizon=self.max_horizon,
            max_batch_size=self.max_batch_size,
            stored_predictions=history_count,
        )


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_prediction_service(
    model: WorldModelAdapter | None = None,
) -> PredictionService:
    """
    Create the service without automatically loading
    the synthetic demonstration model.
    """

    return PredictionService(
        model=model,
        max_horizon=128,
        max_batch_size=32,
        max_history=200,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from .imagination_service import DemoWorldModel

    model = DemoWorldModel(
        state_dim=4,
        action_dim=4,
        scale=0.1,
    )

    service = PredictionService(
        model=model
    )

    next_result = service.predict_next_state(
        NextStateRequest(
            world_id="demo-world-001",
            current_state=[
                0.0, 0.0, 0.0, 0.0
            ],
            action=[
                1.0, 0.0, 0.0, 0.0
            ],
        )
    )

    future_result = service.predict_future(
        FuturePredictionRequest(
            world_id="demo-world-001",
            initial_state=[
                0.0, 0.0, 0.0, 0.0
            ],
            actions=[
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            horizon=3,
        )
    )

    scenario_result = service.predict_scenarios(
        ScenarioPredictionRequest(
            world_id="demo-world-001",
            initial_state=[
                0.0, 0.0, 0.0, 0.0
            ],
            scenarios={
                "move_x": [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                "move_y": [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
            },
        )
    )

    print("=" * 65)
    print("NEUROWORLD AI — PREDICTION SERVICE")
    print("=" * 65)

    print("\nNEXT STATE:")
    print(
        next_result.model_dump_json(
            indent=2
        )
    )

    print("\nFUTURE PREDICTION:")
    print(
        future_result.model_dump_json(
            indent=2
        )
    )

    print("\nSCENARIO COMPARISON:")
    print(
        scenario_result.model_dump_json(
            indent=2
        )
    )

    print("\nSERVICE STATUS:")
    print(
        service.status().model_dump_json(
            indent=2
        )
    )

    print(
        "\nDemonstration complete. "
        "All predictions above use synthetic dynamics."
    )


if __name__ == "__main__":
    main()
