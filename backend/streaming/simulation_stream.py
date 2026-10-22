
"""
NeuroWorld AI
backend/app/streaming/simulation_stream.py

Real-time simulation event streaming.

Features:
- Simulation session streaming
- Real-time state updates
- Action, reward and trajectory events
- Episode progress tracking
- Multiple WebSocket subscribers
- Independent bounded subscriber queues
- Event replay
- Pause / resume / cancellation
- Terminal-state detection
- Simulation status reporting
- Async-compatible execution
- Authorization-ready WebSocket router
- Graceful shutdown

Dependencies:
    pip install fastapi numpy "pydantic>=2"

Important:
- Requires SimulationService.
- Uses synthetic or registered virtual environments.
- Does not control physical robots or vehicles.
- Session and world authorization must be enforced
  before exposing HTTP or WebSocket endpoints.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from backend.app.services.simulation_service import (
    SimulationService,
    SimulationStepRequest,
    SimulationStepResult,
)


logger = logging.getLogger(
    "neuroworld.streaming.simulation"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class SimulationStreamError(Exception):
    pass


class SimulationStreamNotFound(SimulationStreamError):
    pass


class SimulationStreamCapacityError(SimulationStreamError):
    pass


class SimulationStreamStateError(SimulationStreamError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

class SimulationStreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    simulation_id: str = Field(
        min_length=1,
        max_length=128,
    )

    actions: list[list[float]] = Field(
        min_length=1,
        max_length=10000,
    )

    interval_ms: int = Field(
        default=100,
        ge=0,
        le=10000,
    )

    queue_size: int = Field(
        default=64,
        ge=1,
        le=512,
    )

    stop_on_terminal: bool = True

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )

    @model_validator(mode="after")
    def validate_actions(self):
        for index, action in enumerate(self.actions):
            if not action:
                raise ValueError(
                    f"Action {index} cannot be empty"
                )

            if not all(
                math.isfinite(value)
                for value in action
            ):
                raise ValueError(
                    f"Action {index} must be finite"
                )

        return self


# ============================================================
# ENUMS
# ============================================================

class SimulationStreamStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SimulationEventType(str, Enum):
    STARTED = "started"
    STEP = "step"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# ============================================================
# RESPONSE MODELS
# ============================================================

class SimulationStreamEvent(BaseModel):
    event_id: str
    stream_id: str
    simulation_id: str
    world_id: str

    event_type: SimulationEventType
    timestamp: datetime

    step: int = 0
    total_steps: int = 0

    action: list[float] | None = None
    state: list[float] | None = None

    reward: float | None = None
    cumulative_reward: float | None = None

    terminated: bool = False
    truncated: bool = False

    progress_percent: float = 0.0

    info: dict = Field(default_factory=dict)

    message: str | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class SimulationStreamInfo(BaseModel):
    stream_id: str
    simulation_id: str
    world_id: str

    status: SimulationStreamStatus

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    total_steps: int
    completed_steps: int

    progress_percent: float

    cumulative_reward: float
    terminated: bool
    truncated: bool

    subscribers: int
    dropped_events: int

    last_error: str | None = None


# ============================================================
# INTERNAL SESSION
# ============================================================

@dataclass
class _StreamSession:
    stream_id: str
    world_id: str
    config: SimulationStreamConfig

    status: SimulationStreamStatus = (
        SimulationStreamStatus.CREATED
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    completed_steps: int = 0
    cumulative_reward: float = 0.0

    terminated: bool = False
    truncated: bool = False

    dropped_events: int = 0
    last_error: str | None = None

    subscribers: dict[
        str,
        asyncio.Queue[SimulationStreamEvent],
    ] = field(default_factory=dict)

    history: deque[SimulationStreamEvent] = field(
        default_factory=lambda: deque(maxlen=256)
    )

    task: asyncio.Task | None = None

    cancel_requested: bool = False

    resume_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )

    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )


# ============================================================
# STREAMING SERVICE
# ============================================================

class SimulationStreamService:
    """
    Streams SimulationService execution events.

    Each stream executes a sequence of actions
    against an existing simulation session.

    Multiple subscribers receive independent copies
    of events through bounded asyncio queues.

    A simulation session can have only one active
    stream at a time.

    The implementation is process-local.
    """

    def __init__(
        self,
        simulation_service: SimulationService,
        *,
        max_streams: int = 50,
        max_subscribers_per_stream: int = 20,
        max_history: int = 256,
    ) -> None:

        if min(
            max_streams,
            max_subscribers_per_stream,
            max_history,
        ) < 1:
            raise ValueError(
                "Stream limits must be positive"
            )

        self.simulation_service = simulation_service

        self.max_streams = max_streams
        self.max_subscribers_per_stream = (
            max_subscribers_per_stream
        )
        self.max_history = max_history

        self._sessions: dict[
            str,
            _StreamSession,
        ] = {}

        self._active_simulations: set[str] = set()

        self._lock = asyncio.Lock()

    # --------------------------------------------------------
    # SESSION MANAGEMENT
    # --------------------------------------------------------

    async def create_stream(
        self,
        config: SimulationStreamConfig,
        *,
        autostart: bool = True,
    ) -> str:
        simulation = await asyncio.to_thread(
            self.simulation_service.get_simulation,
            config.simulation_id,
        )

        if simulation.terminated or simulation.truncated:
            raise SimulationStreamStateError(
                "Simulation has ended; reset it first"
            )

        async with self._lock:
            if len(self._sessions) >= self.max_streams:
                raise SimulationStreamCapacityError(
                    "Maximum stream count reached"
                )

            if (
                config.simulation_id
                in self._active_simulations
            ):
                raise SimulationStreamStateError(
                    "Simulation already has an active stream"
                )

            stream_id = uuid.uuid4().hex

            session = _StreamSession(
                stream_id=stream_id,
                world_id=simulation.world_id,
                config=config,
                history=deque(
                    maxlen=self.max_history
                ),
            )

            session.resume_event.set()

            self._sessions[stream_id] = session

            self._active_simulations.add(
                config.simulation_id
            )

        if autostart:
            await self.start_stream(stream_id)

        return stream_id

    async def _get_session(
        self,
        stream_id: str,
    ) -> _StreamSession:
        async with self._lock:
            session = self._sessions.get(
                stream_id
            )

        if session is None:
            raise SimulationStreamNotFound(
                "Simulation stream not found"
            )

        return session

    async def start_stream(
        self,
        stream_id: str,
    ) -> None:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            if session.status != SimulationStreamStatus.CREATED:
                raise SimulationStreamStateError(
                    "Stream has already started"
                )

            session.status = (
                SimulationStreamStatus.RUNNING
            )

            session.started_at = datetime.now(
                timezone.utc
            )

            session.task = asyncio.create_task(
                self._run_simulation(session)
            )

    # --------------------------------------------------------
    # EVENT CREATION
    # --------------------------------------------------------

    def _make_event(
        self,
        session: _StreamSession,
        *,
        event_type: SimulationEventType,
        step_result: SimulationStepResult | None = None,
        message: str | None = None,
    ) -> SimulationStreamEvent:
        total_steps = len(
            session.config.actions
        )

        progress = (
            session.completed_steps / total_steps * 100
            if total_steps
            else 0.0
        )

        return SimulationStreamEvent(
            event_id=uuid.uuid4().hex,
            stream_id=session.stream_id,
            simulation_id=(
                session.config.simulation_id
            ),
            world_id=session.world_id,
            event_type=event_type,
            timestamp=datetime.now(
                timezone.utc
            ),
            step=session.completed_steps,
            total_steps=total_steps,
            action=(
                step_result.action
                if step_result
                else None
            ),
            state=(
                step_result.state
                if step_result
                else None
            ),
            reward=(
                step_result.reward
                if step_result
                else None
            ),
            cumulative_reward=(
                session.cumulative_reward
            ),
            terminated=session.terminated,
            truncated=session.truncated,
            progress_percent=progress,
            info=(
                step_result.info
                if step_result
                else {}
            ),
            message=message,
            metadata=session.config.metadata,
        )

    # --------------------------------------------------------
    # EVENT BROADCASTING
    # --------------------------------------------------------

    async def _publish(
        self,
        session: _StreamSession,
        event: SimulationStreamEvent,
    ) -> None:
        async with session.lock:
            session.history.append(event)

            for queue in session.subscribers.values():
                if queue.full():
                    try:
                        queue.get_nowait()
                        queue.task_done()

                        session.dropped_events += 1

                    except asyncio.QueueEmpty:
                        pass

                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    session.dropped_events += 1

    # --------------------------------------------------------
    # SIMULATION EXECUTION
    # --------------------------------------------------------

    async def _run_simulation(
        self,
        session: _StreamSession,
    ) -> None:
        config = session.config

        try:
            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type=SimulationEventType.STARTED,
                    message="Simulation stream started",
                ),
            )

            for action in config.actions:
                await session.resume_event.wait()

                if session.cancel_requested:
                    break

                result = await asyncio.to_thread(
                    self.simulation_service.step,
                    SimulationStepRequest(
                        simulation_id=(
                            config.simulation_id
                        ),
                        action=action,
                    ),
                )

                session.completed_steps += 1

                session.cumulative_reward = (
                    result.cumulative_reward
                )

                session.terminated = result.terminated
                session.truncated = result.truncated

                await self._publish(
                    session,
                    self._make_event(
                        session,
                        event_type=SimulationEventType.STEP,
                        step_result=result,
                        message="Simulation step completed",
                    ),
                )

                # A terminal or truncated simulation
                # cannot accept additional steps.
                if result.terminated or result.truncated:
                    break

                if config.interval_ms > 0:
                    await asyncio.sleep(
                        config.interval_ms / 1000
                    )

            if session.cancel_requested:
                session.status = (
                    SimulationStreamStatus.CANCELLED
                )

                event_type = (
                    SimulationEventType.CANCELLED
                )

                message = "Simulation stream cancelled"

            else:
                session.status = (
                    SimulationStreamStatus.COMPLETED
                )

                event_type = (
                    SimulationEventType.COMPLETED
                )

                message = "Simulation stream completed"

            session.finished_at = datetime.now(
                timezone.utc
            )

            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type=event_type,
                    message=message,
                ),
            )

        except asyncio.CancelledError:
            session.status = (
                SimulationStreamStatus.CANCELLED
            )

            session.finished_at = datetime.now(
                timezone.utc
            )

            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type=SimulationEventType.CANCELLED,
                    message="Simulation task cancelled",
                ),
            )

        except Exception as exc:
            session.status = (
                SimulationStreamStatus.FAILED
            )

            session.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            session.finished_at = datetime.now(
                timezone.utc
            )

            logger.exception(
                "Simulation stream failed: %s",
                session.stream_id,
            )

            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type=SimulationEventType.FAILED,
                    message="Simulation stream failed",
                ),
            )

        finally:
            async with self._lock:
                self._active_simulations.discard(
                    config.simulation_id
                )

    # --------------------------------------------------------
    # SUBSCRIPTIONS
    # --------------------------------------------------------

    async def subscribe(
        self,
        stream_id: str,
        *,
        replay_history: bool = True,
    ) -> tuple[
        str,
        asyncio.Queue[SimulationStreamEvent],
    ]:
        session = await self._get_session(
            stream_id
        )

        subscriber_id = uuid.uuid4().hex

        queue: asyncio.Queue[
            SimulationStreamEvent
        ] = asyncio.Queue(
            maxsize=session.config.queue_size
        )

        async with session.lock:
            if (
                len(session.subscribers)
                >= self.max_subscribers_per_stream
            ):
                raise SimulationStreamCapacityError(
                    "Maximum subscribers reached"
                )

            if replay_history:
                for event in list(
                    session.history
                )[-session.config.queue_size:]:
                    queue.put_nowait(event)

            session.subscribers[
                subscriber_id
            ] = queue

        return subscriber_id, queue

    async def unsubscribe(
        self,
        stream_id: str,
        subscriber_id: str,
    ) -> None:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            session.subscribers.pop(
                subscriber_id,
                None,
            )

    # --------------------------------------------------------
    # PAUSE / RESUME
    # --------------------------------------------------------

    async def pause_stream(
        self,
        stream_id: str,
    ) -> None:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            if session.status != SimulationStreamStatus.RUNNING:
                raise SimulationStreamStateError(
                    "Only running streams can be paused"
                )

            session.status = (
                SimulationStreamStatus.PAUSED
            )

            session.resume_event.clear()

        await self._publish(
            session,
            self._make_event(
                session,
                event_type=SimulationEventType.PAUSED,
                message="Simulation stream paused",
            ),
        )

    async def resume_stream(
        self,
        stream_id: str,
    ) -> None:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            if session.status != SimulationStreamStatus.PAUSED:
                raise SimulationStreamStateError(
                    "Only paused streams can be resumed"
                )

            session.status = (
                SimulationStreamStatus.RUNNING
            )

            session.resume_event.set()

        await self._publish(
            session,
            self._make_event(
                session,
                event_type=SimulationEventType.RESUMED,
                message="Simulation stream resumed",
            ),
        )

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    async def cancel_stream(
        self,
        stream_id: str,
    ) -> bool:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            if session.status not in {
                SimulationStreamStatus.CREATED,
                SimulationStreamStatus.RUNNING,
                SimulationStreamStatus.PAUSED,
            }:
                return False

            session.cancel_requested = True

            session.resume_event.set()

            task = session.task

            if session.status == SimulationStreamStatus.CREATED:
                session.status = (
                    SimulationStreamStatus.CANCELLED
                )
                session.finished_at = datetime.now(
                    timezone.utc
                )

        if task is not None:
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            self._active_simulations.discard(
                session.config.simulation_id
            )

        return True

    async def delete_stream(
        self,
        stream_id: str,
    ) -> bool:
        await self.cancel_stream(stream_id)

        async with self._lock:
            return (
                self._sessions.pop(
                    stream_id,
                    None,
                )
                is not None
            )

    async def shutdown(self) -> None:
        async with self._lock:
            ids = list(
                self._sessions
            )

        for stream_id in ids:
            await self.delete_stream(stream_id)

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    async def get_status(
        self,
        stream_id: str,
    ) -> SimulationStreamInfo:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            total = len(
                session.config.actions
            )

            progress = (
                session.completed_steps / total * 100
                if total
                else 0.0
            )

            return SimulationStreamInfo(
                stream_id=session.stream_id,
                simulation_id=(
                    session.config.simulation_id
                ),
                world_id=session.world_id,
                status=session.status,
                created_at=session.created_at,
                started_at=session.started_at,
                finished_at=session.finished_at,
                total_steps=total,
                completed_steps=(
                    session.completed_steps
                ),
                progress_percent=progress,
                cumulative_reward=(
                    session.cumulative_reward
                ),
                terminated=session.terminated,
                truncated=session.truncated,
                subscribers=len(
                    session.subscribers
                ),
                dropped_events=(
                    session.dropped_events
                ),
                last_error=session.last_error,
            )

    async def get_history(
        self,
        stream_id: str,
        *,
        limit: int = 100,
    ) -> list[SimulationStreamEvent]:
        session = await self._get_session(
            stream_id
        )

        if not 1 <= limit <= self.max_history:
            raise SimulationStreamError(
                "Invalid history limit"
            )

        async with session.lock:
            return [
                event.model_copy(deep=True)
                for event in list(
                    session.history
                )[-limit:]
            ]

    async def list_streams(
        self,
    ) -> list[SimulationStreamInfo]:
        async with self._lock:
            ids = list(
                self._sessions
            )

        return [
            await self.get_status(stream_id)
            for stream_id in ids
        ]


# ============================================================
# WEBSOCKET ROUTER
# ============================================================

AuthorizationCallback = Callable[
    [WebSocket, str],
    bool | Awaitable[bool],
]


def create_simulation_stream_router(
    service: SimulationStreamService,
    *,
    authorize: AuthorizationCallback | None = None,
) -> APIRouter:
    """
    Deliver simulation events to WebSocket clients.

    Authorization callback:
        authorize(websocket, world_id) -> bool

    It must verify identity and world access.

    Without an authorization callback, connections
    are rejected.
    """

    router = APIRouter(
        tags=["Simulation Streaming"]
    )

    @router.websocket(
        "/api/v1/streaming/simulations/{stream_id}"
    )
    async def simulation_websocket(
        websocket: WebSocket,
        stream_id: str,
    ):
        try:
            session = await service._get_session(
                stream_id
            )
        except SimulationStreamNotFound:
            await websocket.close(code=4404)
            return

        if authorize is None:
            await websocket.close(code=4403)
            return

        try:
            permitted = authorize(
                websocket,
                session.world_id,
            )

            if inspect.isawaitable(permitted):
                permitted = await permitted

            if permitted is not True:
                await websocket.close(code=4403)
                return

        except Exception:
            await websocket.close(code=4403)
            return

        try:
            subscriber_id, queue = (
                await service.subscribe(
                    stream_id,
                    replay_history=True,
                )
            )
        except SimulationStreamCapacityError:
            await websocket.close(code=4429)
            return

        await websocket.accept()

        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=2.0,
                    )

                except asyncio.TimeoutError:
                    status = await service.get_status(
                        stream_id
                    )

                    if status.status in {
                        SimulationStreamStatus.COMPLETED,
                        SimulationStreamStatus.CANCELLED,
                        SimulationStreamStatus.FAILED,
                    }:
                        break

                    continue

                await websocket.send_json(
                    event.model_dump(
                        mode="json"
                    )
                )

                queue.task_done()

                if event.event_type in {
                    SimulationEventType.COMPLETED,
                    SimulationEventType.CANCELLED,
                    SimulationEventType.FAILED,
                }:
                    break

        except WebSocketDisconnect:
            pass

        finally:
            await service.unsubscribe(
                stream_id,
                subscriber_id,
            )

            try:
                await websocket.close()
            except RuntimeError:
                pass

    return router


# ============================================================
# FACTORY
# ============================================================

def create_simulation_stream_service(
    simulation_service: SimulationService,
) -> SimulationStreamService:
    return SimulationStreamService(
        simulation_service=simulation_service,
        max_streams=50,
        max_subscribers_per_stream=20,
        max_history=256,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:
    from backend.app.services.simulation_service import (
        CreateSimulationRequest,
        create_simulation_service,
    )

    simulation_service = (
        create_simulation_service()
    )

    simulation = (
        simulation_service.create_simulation(
            CreateSimulationRequest(
                world_id="demo-world-001",
                environment_name="synthetic_physics",
                seed=42,
            )
        )
    )

    stream_service = (
        create_simulation_stream_service(
            simulation_service
        )
    )

    stream_id = await stream_service.create_stream(
        SimulationStreamConfig(
            simulation_id=(
                simulation.simulation_id
            ),
            actions=[
                [1.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
                [0.0, 0.0],
                [-0.5, -0.5],
            ],
            interval_ms=100,
        ),
        autostart=False,
    )

    subscriber_id, queue = (
        await stream_service.subscribe(
            stream_id
        )
    )

    await stream_service.start_stream(
        stream_id
    )

    print("=" * 65)
    print("NEUROWORLD AI — SIMULATION STREAM")
    print("=" * 65)

    try:
        while True:
            event = await queue.get()

            print(
                event.model_dump_json(
                    indent=2
                )
            )

            queue.task_done()

            if event.event_type in {
                SimulationEventType.COMPLETED,
                SimulationEventType.CANCELLED,
                SimulationEventType.FAILED,
            }:
                break

        print("\nFINAL STATUS:")

        status = await stream_service.get_status(
            stream_id
        )

        print(
            status.model_dump_json(
                indent=2
            )
        )

    finally:
        await stream_service.unsubscribe(
            stream_id,
            subscriber_id,
        )

        await stream_service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
