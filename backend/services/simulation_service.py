
"""
NeuroWorld AI
backend/app/services/simulation_service.py

Simulation orchestration service.

Features:
    - Simulation session creation
    - Environment registration
    - Simulation reset and stepping
    - Multi-step episode execution
    - Synthetic physics environment
    - Action and observation validation
    - Trajectory recording
    - Reward accumulation
    - Episode termination and truncation
    - World-model prediction comparison
    - Session history and status
    - Async-compatible execution
    - Prometheus telemetry

Dependencies:
    pip install numpy pydantic psutil prometheus-client

The included synthetic environment is a development
simulator, not a validated physical-world simulator.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ..core.logging import get_logger
from ..core.telemetry import (
    record_simulation_episode,
    track_simulation,
)

from .imagination_service import (
    Transition,
    WorldModelAdapter,
)


logger = get_logger("services.simulation")


# ============================================================
# EXCEPTIONS
# ============================================================

class SimulationServiceError(Exception):
    """Base simulation service error."""


class SimulationNotFound(SimulationServiceError):
    """Requested simulation session does not exist."""


class EnvironmentNotFound(SimulationServiceError):
    """Requested environment is not registered."""


class SimulationValidationError(SimulationServiceError):
    """Invalid simulation input."""


class SimulationStateError(SimulationServiceError):
    """Invalid simulation lifecycle operation."""


# ============================================================
# ENVIRONMENT INTERFACE
# ============================================================

class SimulationEnvironment(Protocol):
    """
    Environment adapter interface.

    reset(seed) returns the initial state.

    step(action) returns:
        next_state, reward, terminated, truncated, info

    The environment must expose:
        name
        state_dim
        action_dim
    """

    name: str
    state_dim: int
    action_dim: int

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:
        ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        ...


# ============================================================
# SYNTHETIC PHYSICS ENVIRONMENT
# ============================================================

class SyntheticPhysicsEnvironment:
    """
    Simple two-dimensional point-mass environment.

    State:
        [position_x, position_y, velocity_x, velocity_y]

    Action:
        [acceleration_x, acceleration_y]

    Euler integration:
        velocity += acceleration * dt
        position += velocity * dt

    Reward:
        negative Euclidean distance to target.

    This is a synthetic numerical simulation.
    """

    name = "synthetic_physics"

    state_dim = 4
    action_dim = 2

    def __init__(
        self,
        *,
        time_step: float = 0.1,
        max_steps: int = 200,
        target: tuple[float, float] = (5.0, 5.0),
        target_radius: float = 0.25,
        action_limit: float = 1.0,
    ) -> None:

        if (
            not math.isfinite(time_step)
            or time_step <= 0
        ):
            raise ValueError(
                "time_step must be positive"
            )

        if max_steps < 1:
            raise ValueError(
                "max_steps must be positive"
            )

        if (
            not math.isfinite(action_limit)
            or action_limit <= 0
        ):
            raise ValueError(
                "action_limit must be positive"
            )

        if (
            not math.isfinite(target_radius)
            or target_radius <= 0
        ):
            raise ValueError(
                "target_radius must be positive"
            )

        self.time_step = time_step
        self.max_steps = max_steps

        self.target = np.asarray(
            target,
            dtype=np.float64,
        )

        if (
            self.target.shape != (2,)
            or not np.all(np.isfinite(self.target))
        ):
            raise ValueError(
                "target must contain two finite coordinates"
            )

        self.target_radius = target_radius
        self.action_limit = action_limit

        self.state = np.zeros(
            self.state_dim,
            dtype=np.float64,
        )

        self.step_count = 0

        self._rng = np.random.default_rng()

    def reset(
        self,
        seed: int | None = None,
    ) -> np.ndarray:

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.state = np.zeros(
            self.state_dim,
            dtype=np.float64,
        )

        self.step_count = 0

        return self.state.copy()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        if action.shape != (self.action_dim,):
            raise SimulationValidationError(
                "Action must contain two values"
            )

        if not np.all(np.isfinite(action)):
            raise SimulationValidationError(
                "Action must contain finite values"
            )

        if np.any(
            np.abs(action) > self.action_limit
        ):
            raise SimulationValidationError(
                "Action exceeds configured limits"
            )

        dt = self.time_step

        position = self.state[:2].copy()
        velocity = self.state[2:].copy()

        velocity += action * dt
        position += velocity * dt

        self.state = np.concatenate([
            position,
            velocity,
        ])

        self.step_count += 1

        distance = float(
            np.linalg.norm(
                position - self.target
            )
        )

        terminated = (
            distance <= self.target_radius
        )

        truncated = (
            self.step_count >= self.max_steps
            and not terminated
        )

        reward = -distance

        info = {
            "distance_to_target": distance,
            "target_reached": terminated,
            "step_count": self.step_count,
        }

        return (
            self.state.copy(),
            float(reward),
            terminated,
            truncated,
            info,
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class CreateSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment_name: str = Field(
        default="synthetic_physics",
        min_length=1,
        max_length=128,
    )

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    seed: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class SimulationStepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(
        min_length=1,
        max_length=128,
    )

    action: list[float] = Field(
        min_length=1,
        max_length=16384,
    )


class SimulationEpisodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(
        min_length=1,
        max_length=128,
    )

    actions: list[list[float]] = Field(
        min_length=1,
        max_length=10000,
    )

    stop_on_terminal: bool = True


# ============================================================
# RESPONSE MODELS
# ============================================================

class SimulationStepResult(BaseModel):
    step: int
    action: list[float]
    state: list[float]
    reward: float
    cumulative_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(
        default_factory=dict,
    )


class SimulationSession(BaseModel):
    simulation_id: str
    world_id: str
    environment_name: str

    created_at: datetime
    updated_at: datetime

    status: str

    initial_state: list[float]
    current_state: list[float]

    step_count: int
    cumulative_reward: float

    terminated: bool
    truncated: bool

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class SimulationEpisodeResult(BaseModel):
    simulation_id: str
    world_id: str
    environment_name: str

    steps: list[SimulationStepResult]

    executed_steps: int
    total_reward: float

    terminated: bool
    truncated: bool

    duration_ms: float


class ModelComparisonResult(BaseModel):
    simulation_id: str
    model_name: str
    step_count: int
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    final_state_error: float


# ============================================================
# INTERNAL SESSION STATE
# ============================================================

class _RuntimeSession:
    def __init__(
        self,
        *,
        public: SimulationSession,
        environment: SimulationEnvironment,
    ) -> None:

        self.public = public
        self.environment = environment

        self.trajectory: list[
            SimulationStepResult
        ] = []

        self.lock = threading.RLock()


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _vector(
    values: Any,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:

    try:
        array = np.asarray(
            values,
            dtype=np.float64,
        )
    except (ValueError, TypeError) as exc:
        raise SimulationValidationError(
            f"{name} must be numeric"
        ) from exc

    if array.shape != (dimension,):
        raise SimulationValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.all(np.isfinite(array)):
        raise SimulationValidationError(
            f"{name} must contain finite values"
        )

    return array.copy()


def _finite_reward(value: Any) -> float:

    try:
        reward = float(value)
    except (ValueError, TypeError) as exc:
        raise SimulationValidationError(
            "Environment returned invalid reward"
        ) from exc

    if not math.isfinite(reward):
        raise SimulationValidationError(
            "Environment returned non-finite reward"
        )

    return reward


# ============================================================
# SIMULATION SERVICE
# ============================================================

class SimulationService:
    """
    Manage independent simulation environments.

    Each session owns its own environment instance.
    No physical actuators or external hardware are
    controlled by this service.
    """

    def __init__(
        self,
        *,
        max_sessions: int = 100,
        max_trajectory_steps: int = 10000,
    ) -> None:

        if max_sessions < 1:
            raise ValueError(
                "max_sessions must be positive"
            )

        if max_trajectory_steps < 1:
            raise ValueError(
                "max_trajectory_steps must be positive"
            )

        self.max_sessions = max_sessions

        self.max_trajectory_steps = (
            max_trajectory_steps
        )

        self._environments: dict[
            str,
            Callable[[], SimulationEnvironment],
        ] = {}

        self._sessions: OrderedDict[
            str,
            _RuntimeSession,
        ] = OrderedDict()

        self._lock = threading.RLock()

        self.register_environment(
            "synthetic_physics",
            SyntheticPhysicsEnvironment,
        )

    def register_environment(
        self,
        name: str,
        factory: Callable[
            [],
            SimulationEnvironment,
        ],
    ) -> None:

        if not name.strip():
            raise ValueError(
                "Environment name cannot be empty"
            )

        if not callable(factory):
            raise ValueError(
                "Environment factory must be callable"
            )

        with self._lock:
            self._environments[name] = factory

        logger.info(
            "Simulation environment registered: %s",
            name,
        )

    def list_environments(self) -> list[str]:
        with self._lock:
            return sorted(
                self._environments
            )

    def create_simulation(
        self,
        request: CreateSimulationRequest,
    ) -> SimulationSession:

        with self._lock:
            factory = self._environments.get(
                request.environment_name
            )

        if factory is None:
            raise EnvironmentNotFound(
                "Environment is not registered"
            )

        environment = factory()

        initial_state = _vector(
            environment.reset(
                seed=request.seed
            ),
            dimension=environment.state_dim,
            name="initial_state",
        )

        now = datetime.now(timezone.utc)

        session = SimulationSession(
            simulation_id=uuid.uuid4().hex,
            world_id=request.world_id,
            environment_name=(
                request.environment_name
            ),
            created_at=now,
            updated_at=now,
            status="ready",
            initial_state=initial_state.tolist(),
            current_state=initial_state.tolist(),
            step_count=0,
            cumulative_reward=0.0,
            terminated=False,
            truncated=False,
            metadata=request.metadata,
        )

        runtime = _RuntimeSession(
            public=session,
            environment=environment,
        )

        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise SimulationStateError(
                    "Maximum active session count reached"
                )

            self._sessions[
                session.simulation_id
            ] = runtime

        logger.info(
            "Simulation created: id=%s environment=%s",
            session.simulation_id,
            session.environment_name,
        )

        return session.model_copy(deep=True)

    def _get_runtime(
        self,
        simulation_id: str,
    ) -> _RuntimeSession:

        with self._lock:
            runtime = self._sessions.get(
                simulation_id
            )

        if runtime is None:
            raise SimulationNotFound(
                "Simulation session not found"
            )

        return runtime

    def get_simulation(
        self,
        simulation_id: str,
    ) -> SimulationSession:

        runtime = self._get_runtime(
            simulation_id
        )

        with runtime.lock:
            return runtime.public.model_copy(
                deep=True
            )

    def step(
        self,
        request: SimulationStepRequest,
    ) -> SimulationStepResult:

        runtime = self._get_runtime(
            request.simulation_id
        )

        with runtime.lock:
            session = runtime.public

            if session.terminated or session.truncated:
                raise SimulationStateError(
                    "Simulation has ended; reset before stepping"
                )

            if (
                len(runtime.trajectory)
                >= self.max_trajectory_steps
            ):
                raise SimulationStateError(
                    "Trajectory step limit reached"
                )

            action = _vector(
                request.action,
                dimension=runtime.environment.action_dim,
                name="action",
            )

            (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            ) = runtime.environment.step(action)

            state = _vector(
                next_state,
                dimension=runtime.environment.state_dim,
                name="next_state",
            )

            reward = _finite_reward(reward)

            session.step_count += 1
            session.cumulative_reward += reward

            session.current_state = state.tolist()

            session.terminated = bool(terminated)
            session.truncated = bool(truncated)

            session.status = (
                "completed"
                if session.terminated or session.truncated
                else "running"
            )

            session.updated_at = datetime.now(
                timezone.utc
            )

            step_result = SimulationStepResult(
                step=session.step_count,
                action=action.tolist(),
                state=state.tolist(),
                reward=reward,
                cumulative_reward=(
                    session.cumulative_reward
                ),
                terminated=session.terminated,
                truncated=session.truncated,
                info=info,
            )

            runtime.trajectory.append(
                step_result
            )

            return step_result.model_copy(
                deep=True
            )

    def run_episode(
        self,
        request: SimulationEpisodeRequest,
    ) -> SimulationEpisodeResult:

        runtime = self._get_runtime(
            request.simulation_id
        )

        started = time.perf_counter()

        executed: list[
            SimulationStepResult
        ] = []

        with track_simulation():
            try:
                for action in request.actions:
                    result = self.step(
                        SimulationStepRequest(
                            simulation_id=(
                                request.simulation_id
                            ),
                            action=action,
                        )
                    )

                    executed.append(result)

                    # Terminal states cannot be stepped again.
                    if result.terminated or result.truncated:
                        break

            except Exception:
                logger.exception(
                    "Simulation episode failed"
                )

                session = self.get_simulation(
                    request.simulation_id
                )

                record_simulation_episode(
                    environment=session.environment_name,
                    steps=len(executed),
                    duration_seconds=(
                        time.perf_counter() - started
                    ),
                    success=False,
                )

                raise

        session = self.get_simulation(
            request.simulation_id
        )

        duration_ms = (
            time.perf_counter() - started
        ) * 1000

        record_simulation_episode(
            environment=session.environment_name,
            steps=len(executed),
            duration_seconds=duration_ms / 1000,
            success=not session.truncated,
        )

        return SimulationEpisodeResult(
            simulation_id=session.simulation_id,
            world_id=session.world_id,
            environment_name=session.environment_name,
            steps=executed,
            executed_steps=len(executed),
            total_reward=float(
                sum(step.reward for step in executed)
            ),
            terminated=session.terminated,
            truncated=session.truncated,
            duration_ms=duration_ms,
        )

    def reset_simulation(
        self,
        simulation_id: str,
        *,
        seed: int | None = None,
    ) -> SimulationSession:

        runtime = self._get_runtime(
            simulation_id
        )

        with runtime.lock:
            initial_state = _vector(
                runtime.environment.reset(
                    seed=seed
                ),
                dimension=runtime.environment.state_dim,
                name="initial_state",
            )

            session = runtime.public

            session.initial_state = (
                initial_state.tolist()
            )

            session.current_state = (
                initial_state.tolist()
            )

            session.step_count = 0
            session.cumulative_reward = 0.0

            session.terminated = False
            session.truncated = False

            session.status = "ready"
            session.updated_at = datetime.now(
                timezone.utc
            )

            runtime.trajectory.clear()

            return session.model_copy(
                deep=True
            )

    def get_trajectory(
        self,
        simulation_id: str,
        *,
        limit: int = 1000,
    ) -> list[SimulationStepResult]:

        if not 1 <= limit <= self.max_trajectory_steps:
            raise SimulationValidationError(
                "Invalid trajectory limit"
            )

        runtime = self._get_runtime(
            simulation_id
        )

        with runtime.lock:
            return [
                step.model_copy(deep=True)
                for step in runtime.trajectory[-limit:]
            ]

    def compare_with_world_model(
        self,
        *,
        simulation_id: str,
        model: WorldModelAdapter,
    ) -> ModelComparisonResult:
        """
        Compare one-step world-model predictions with
        recorded environment transitions.

        Every prediction uses the actual previous state
        and recorded action, avoiding autoregressive
        error accumulation in this metric.
        """

        runtime = self._get_runtime(
            simulation_id
        )

        with runtime.lock:
            session = runtime.public.model_copy(
                deep=True
            )

            trajectory = [
                item.model_copy(deep=True)
                for item in runtime.trajectory
            ]

        if not trajectory:
            raise SimulationValidationError(
                "Simulation trajectory is empty"
            )

        if (
            model.state_dim
            != len(session.initial_state)
        ):
            raise SimulationValidationError(
                "World-model state dimension mismatch"
            )

        if (
            model.action_dim
            != len(trajectory[0].action)
        ):
            raise SimulationValidationError(
                "World-model action dimension mismatch"
            )

        previous_state = np.asarray(
            session.initial_state,
            dtype=np.float64,
        )

        errors = []

        for step in trajectory:
            action = np.asarray(
                step.action,
                dtype=np.float64,
            )

            transition = model.predict_transition(
                previous_state.copy(),
                action.copy(),
            )

            if not isinstance(
                transition,
                Transition,
            ):
                raise SimulationValidationError(
                    "World model returned invalid transition"
                )

            predicted_state = _vector(
                transition.next_state,
                dimension=model.state_dim,
                name="predicted_state",
            )

            actual_state = np.asarray(
                step.state,
                dtype=np.float64,
            )

            errors.append(
                predicted_state - actual_state
            )

            previous_state = actual_state

        error_array = np.stack(
            errors
        )

        mae = float(
            np.mean(np.abs(error_array))
        )

        mse = float(
            np.mean(error_array ** 2)
        )

        rmse = float(
            np.sqrt(mse)
        )

        final_error = float(
            np.linalg.norm(
                error_array[-1]
            )
        )

        return ModelComparisonResult(
            simulation_id=simulation_id,
            model_name=model.name,
            step_count=len(trajectory),
            mean_absolute_error=mae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            final_state_error=final_error,
        )

    async def run_episode_async(
        self,
        request: SimulationEpisodeRequest,
    ) -> SimulationEpisodeResult:

        return await asyncio.to_thread(
            self.run_episode,
            request,
        )

    async def step_async(
        self,
        request: SimulationStepRequest,
    ) -> SimulationStepResult:

        return await asyncio.to_thread(
            self.step,
            request,
        )

    def list_simulations(
        self,
        *,
        world_id: str | None = None,
    ) -> list[SimulationSession]:

        with self._lock:
            runtimes = list(
                self._sessions.values()
            )

        sessions = []

        for runtime in runtimes:
            with runtime.lock:
                session = runtime.public.model_copy(
                    deep=True
                )

            if (
                world_id is None
                or session.world_id == world_id
            ):
                sessions.append(session)

        return sessions

    def delete_simulation(
        self,
        simulation_id: str,
    ) -> bool:

        with self._lock:
            runtime = self._sessions.pop(
                simulation_id,
                None,
            )

        return runtime is not None

    def status(self) -> dict[str, Any]:

        with self._lock:
            session_count = len(
                self._sessions
            )

            environments = sorted(
                self._environments
            )

        return {
            "service": "simulation",
            "status": "ready",
            "registered_environments": environments,
            "session_count": session_count,
            "max_sessions": self.max_sessions,
            "max_trajectory_steps": (
                self.max_trajectory_steps
            ),
            "storage": "in_memory",
            "hardware_control_enabled": False,
        }


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_simulation_service() -> SimulationService:

    return SimulationService(
        max_sessions=100,
        max_trajectory_steps=10000,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:

    service = create_simulation_service()

    session = service.create_simulation(
        CreateSimulationRequest(
            world_id="demo-world-001",
            environment_name="synthetic_physics",
            seed=42,
        )
    )

    print("=" * 65)
    print("NEUROWORLD AI — SIMULATION SERVICE")
    print("=" * 65)

    print("\nCREATED SESSION:")
    print(
        session.model_dump_json(indent=2)
    )

    result = service.run_episode(
        SimulationEpisodeRequest(
            simulation_id=session.simulation_id,
            actions=[
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.0, 0.0],
            ],
        )
    )

    print("\nEPISODE RESULT:")
    print(
        result.model_dump_json(indent=2)
    )

    print("\nTRAJECTORY:")
    for step in service.get_trajectory(
        session.simulation_id
    ):
        print(
            step.model_dump_json()
        )

    print("\nSERVICE STATUS:")
    print(
        service.status()
    )

    print(
        "\nDemonstration complete. "
        "The environment uses synthetic physics."
    )


if __name__ == "__main__":
    main()
