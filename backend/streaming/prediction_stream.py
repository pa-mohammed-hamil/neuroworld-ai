
"""
NeuroWorld AI
backend/app/streaming/prediction_stream.py

Real-time world-model prediction streaming.

Features:
- Async prediction stream sessions
- WebSocket prediction delivery
- Next-state and future-state updates
- Action-conditioned latent trajectories
- Prediction uncertainty and rewards
- Multi-client subscriptions
- Bounded subscriber queues
- Slow-client backpressure handling
- Stream cancellation and cleanup
- Session progress and status
- World-model adapter integration

Dependencies:
    pip install fastapi numpy pydantic

Notes:
- Requires an existing PredictionService.
- A registered world model performs predictions.
- Streaming exposes model predictions, not verified
  future events.
- No real-world actions are executed.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.prediction_service import (
    PredictionService,
    FuturePredictionRequest,
)


logger = logging.getLogger(
    "neuroworld.streaming.prediction"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class PredictionStreamError(Exception):
    pass


class PredictionStreamNotFound(PredictionStreamError):
    pass


class PredictionStreamCapacityError(PredictionStreamError):
    pass


class PredictionStreamValidationError(PredictionStreamError):
    pass


# ============================================================
# STREAM CONFIGURATION
# ============================================================

class PredictionStreamConfig(BaseModel):
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

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )

    stop_on_uncertainty: bool = True

    interval_ms: int = Field(
        default=100,
        ge=0,
        le=10000,
    )

    queue_size: int = Field(
        default=32,
        ge=1,
        le=512,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class PredictionStreamStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class PredictionStreamEvent(BaseModel):
    event_id: str
    stream_id: str
    world_id: str

    event_type: str
    timestamp: datetime

    step: int = 0
    total_steps: int = 0

    state: list[float] | None = None
    action: list[float] | None = None

    uncertainty: float | None = None
    reward: float | None = None

    progress_percent: float = 0.0

    message: str | None = None
    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class PredictionStreamInfo(BaseModel):
    stream_id: str
    world_id: str
    status: PredictionStreamStatus

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    total_steps: int
    completed_steps: int

    progress_percent: float
    subscribers: int
    dropped_events: int

    last_error: str | None = None


# ============================================================
# INTERNAL SESSION
# ============================================================

@dataclass
class _PredictionSession:
    stream_id: str
    config: PredictionStreamConfig

    status: PredictionStreamStatus = (
        PredictionStreamStatus.CREATED
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    completed_steps: int = 0
    dropped_events: int = 0
    last_error: str | None = None

    subscribers: dict[
        str,
        asyncio.Queue[PredictionStreamEvent],
    ] = field(default_factory=dict)

    history: deque[PredictionStreamEvent] = field(
        default_factory=lambda: deque(maxlen=256)
    )

    task: asyncio.Task | None = None
    cancel_requested: bool = False

    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )


# ============================================================
# STREAMING SERVICE
# ============================================================

class PredictionStreamService:
    """
    Streams prediction results over async queues.

    A separate queue is maintained for each subscriber.
    Slow subscribers drop their oldest pending events.

    This implementation runs inside one Python process.
    For multi-worker deployments, use Redis Streams,
    Kafka, or another shared message broker.
    """

    def __init__(
        self,
        prediction_service: PredictionService,
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

        self.prediction_service = prediction_service

        self.max_streams = max_streams
        self.max_subscribers_per_stream = (
            max_subscribers_per_stream
        )
        self.max_history = max_history

        self._sessions: dict[
            str,
            _PredictionSession,
        ] = {}

        self._lock = asyncio.Lock()

    # --------------------------------------------------------
    # SESSION MANAGEMENT
    # --------------------------------------------------------

    async def create_stream(
        self,
        config: PredictionStreamConfig,
        *,
        autostart: bool = True,
    ) -> str:
        """
        Create a prediction stream.

        The underlying model must be registered with
        PredictionService before execution.
        """

        model = self.prediction_service.model

        if len(config.initial_state) != model.state_dim:
            raise PredictionStreamValidationError(
                "Initial state dimension mismatch"
            )

        if not all(
            math.isfinite(value)
            for value in config.initial_state
        ):
            raise PredictionStreamValidationError(
                "Initial state must be finite"
            )

        for index, action in enumerate(config.actions):
            if len(action) != model.action_dim:
                raise PredictionStreamValidationError(
                    f"Action {index} dimension mismatch"
                )

            if not all(
                math.isfinite(value)
                for value in action
            ):
                raise PredictionStreamValidationError(
                    f"Action {index} must be finite"
                )

        if (
            len(config.actions)
            > self.prediction_service.max_horizon
        ):
            raise PredictionStreamValidationError(
                "Prediction horizon exceeds model limit"
            )

        async with self._lock:
            if len(self._sessions) >= self.max_streams:
                raise PredictionStreamCapacityError(
                    "Maximum prediction streams reached"
                )

            stream_id = uuid.uuid4().hex

            session = _PredictionSession(
                stream_id=stream_id,
                config=config,
                history=deque(
                    maxlen=self.max_history
                ),
            )

            self._sessions[stream_id] = session

        if autostart:
            await self.start_stream(stream_id)

        return stream_id

    async def _get_session(
        self,
        stream_id: str,
    ) -> _PredictionSession:

        async with self._lock:
            session = self._sessions.get(
                stream_id
            )

        if session is None:
            raise PredictionStreamNotFound(
                "Prediction stream not found"
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
            if session.status != PredictionStreamStatus.CREATED:
                raise PredictionStreamError(
                    "Stream has already started"
                )

            session.status = PredictionStreamStatus.RUNNING
            session.started_at = datetime.now(
                timezone.utc
            )

            session.task = asyncio.create_task(
                self._run_prediction(session)
            )

    # --------------------------------------------------------
    # EVENT DELIVERY
    # --------------------------------------------------------

    async def _publish(
        self,
        session: _PredictionSession,
        event: PredictionStreamEvent,
    ) -> None:

        async with session.lock:
            session.history.append(event)

            queues = list(
                session.subscribers.values()
            )

            for queue in queues:
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

    def _make_event(
        self,
        session: _PredictionSession,
        *,
        event_type: str,
        step: int = 0,
        state: list[float] | None = None,
        action: list[float] | None = None,
        uncertainty: float | None = None,
        reward: float | None = None,
        message: str | None = None,
    ) -> PredictionStreamEvent:

        total = len(
            session.config.actions
        )

        progress = (
            100.0 * step / total
            if total
            else 0.0
        )

        return PredictionStreamEvent(
            event_id=uuid.uuid4().hex,
            stream_id=session.stream_id,
            world_id=session.config.world_id,
            event_type=event_type,
            timestamp=datetime.now(
                timezone.utc
            ),
            step=step,
            total_steps=total,
            state=state,
            action=action,
            uncertainty=uncertainty,
            reward=reward,
            progress_percent=min(
                100.0,
                progress,
            ),
            message=message,
            metadata=session.config.metadata,
        )

    # --------------------------------------------------------
    # PREDICTION EXECUTION
    # --------------------------------------------------------

    async def _run_prediction(
        self,
        session: _PredictionSession,
    ) -> None:
        """
        Run one-step predictions sequentially.

        Each step uses the previously predicted state.
        This is an autoregressive rollout.

        Neural inference is delegated to
        PredictionService using a worker thread.
        """

        config = session.config
        state = list(config.initial_state)

        try:
            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type="started",
                    state=state,
                    message="Prediction stream started",
                ),
            )

            for index, action in enumerate(
                config.actions,
                start=1,
            ):
                if session.cancel_requested:
                    break

                request = FuturePredictionRequest(
                    world_id=config.world_id,
                    initial_state=state,
                    actions=[action],
                    horizon=1,
                    uncertainty_threshold=(
                        config.uncertainty_threshold
                    ),
                    stop_on_uncertainty=(
                        config.stop_on_uncertainty
                    ),
                    metadata=config.metadata,
                )

                result = (
                    await self.prediction_service
                    .predict_future_async(request)
                )

                if session.cancel_requested:
                    break

                if result.completed_steps == 0:
                    await self._publish(
                        session,
                        self._make_event(
                            session,
                            event_type="uncertainty_limit",
                            step=index - 1,
                            state=state,
                            message=(
                                "Prediction stopped because "
                                "uncertainty exceeded threshold"
                            ),
                        ),
                    )
                    break

                predicted_step = result.steps[0]

                state = list(
                    predicted_step.predicted_state
                )

                session.completed_steps = index

                await self._publish(
                    session,
                    self._make_event(
                        session,
                        event_type="prediction",
                        step=index,
                        state=state,
                        action=action,
                        uncertainty=(
                            predicted_step.uncertainty
                        ),
                        reward=(
                            predicted_step.predicted_reward
                        ),
                        message="Predicted state available",
                    ),
                )

                if predicted_step.terminal:
                    break

                if config.interval_ms > 0:
                    await asyncio.sleep(
                        config.interval_ms / 1000
                    )

            if session.cancel_requested:
                session.status = (
                    PredictionStreamStatus.CANCELLED
                )

                event_type = "cancelled"
                message = "Prediction stream cancelled"

            else:
                session.status = (
                    PredictionStreamStatus.COMPLETED
                )

                event_type = "completed"
                message = "Prediction stream completed"

            session.finished_at = datetime.now(
                timezone.utc
            )

            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type=event_type,
                    step=session.completed_steps,
                    state=state,
                    message=message,
                ),
            )

        except asyncio.CancelledError:
            session.status = (
                PredictionStreamStatus.CANCELLED
            )

            session.finished_at = datetime.now(
                timezone.utc
            )

            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type="cancelled",
                    step=session.completed_steps,
                    state=state,
                    message="Prediction task cancelled",
                ),
            )

        except Exception as exc:
            session.status = (
                PredictionStreamStatus.FAILED
            )

            session.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            session.finished_at = datetime.now(
                timezone.utc
            )

            logger.exception(
                "Prediction stream failed: %s",
                session.stream_id,
            )

            await self._publish(
                session,
                self._make_event(
                    session,
                    event_type="error",
                    step=session.completed_steps,
                    state=state,
                    message="Prediction stream failed",
                ),
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
        asyncio.Queue[PredictionStreamEvent],
    ]:
        session = await self._get_session(
            stream_id
        )

        subscriber_id = uuid.uuid4().hex

        queue: asyncio.Queue[
            PredictionStreamEvent
        ] = asyncio.Queue(
            maxsize=session.config.queue_size
        )

        async with session.lock:
            if (
                len(session.subscribers)
                >= self.max_subscribers_per_stream
            ):
                raise PredictionStreamCapacityError(
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
                PredictionStreamStatus.CREATED,
                PredictionStreamStatus.RUNNING,
            }:
                return False

            session.cancel_requested = True
            task = session.task

            if session.status == PredictionStreamStatus.CREATED:
                session.status = (
                    PredictionStreamStatus.CANCELLED
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
            await self.delete_stream(
                stream_id
            )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    async def get_status(
        self,
        stream_id: str,
    ) -> PredictionStreamInfo:
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

            return PredictionStreamInfo(
                stream_id=session.stream_id,
                world_id=session.config.world_id,
                status=session.status,
                created_at=session.created_at,
                started_at=session.started_at,
                finished_at=session.finished_at,
                total_steps=total,
                completed_steps=(
                    session.completed_steps
                ),
                progress_percent=progress,
                subscribers=len(
                    session.subscribers
                ),
                dropped_events=(
                    session.dropped_events
                ),
                last_error=session.last_error,
            )

    async def list_streams(
        self,
    ) -> list[PredictionStreamInfo]:
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


def create_prediction_stream_router(
    service: PredictionStreamService,
    *,
    authorize: AuthorizationCallback | None = None,
) -> APIRouter:
    """
    WebSocket delivery for prediction events.

    The authorization callback receives:
        websocket
        world_id

    It must validate user identity and access to
    the requested world.

    Without a callback, connections are rejected.
    """

    router = APIRouter(
        tags=["Prediction Streaming"],
    )

    @router.websocket(
        "/api/v1/streaming/predictions/{stream_id}"
    )
    async def prediction_websocket(
        websocket: WebSocket,
        stream_id: str,
    ):
        try:
            session = await service._get_session(
                stream_id
            )
        except PredictionStreamNotFound:
            await websocket.close(code=4404)
            return

        if authorize is None:
            await websocket.close(code=4403)
            return

        try:
            permitted = authorize(
                websocket,
                session.config.world_id,
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
        except PredictionStreamCapacityError:
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
                        PredictionStreamStatus.COMPLETED,
                        PredictionStreamStatus.CANCELLED,
                        PredictionStreamStatus.FAILED,
                    }:
                        break

                    continue

                await websocket.send_json(
                    event.model_dump(mode="json")
                )

                queue.task_done()

                if event.event_type in {
                    "completed",
                    "cancelled",
                    "error",
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

def create_prediction_stream_service(
    prediction_service: PredictionService,
) -> PredictionStreamService:
    return PredictionStreamService(
        prediction_service=prediction_service,
        max_streams=50,
        max_subscribers_per_stream=20,
        max_history=256,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:
    from backend.app.services.imagination_service import (
        DemoWorldModel,
    )

    model = DemoWorldModel(
        state_dim=4,
        action_dim=4,
        scale=0.1,
    )

    prediction_service = PredictionService(
        model=model
    )

    service = create_prediction_stream_service(
        prediction_service
    )

    stream_id = await service.create_stream(
        PredictionStreamConfig(
            world_id="demo-world-001",
            initial_state=[
                0.0, 0.0, 0.0, 0.0
            ],
            actions=[
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            interval_ms=200,
        ),
        autostart=False,
    )

    subscriber_id, queue = await service.subscribe(
        stream_id
    )

    await service.start_stream(stream_id)

    print("=" * 65)
    print("NEUROWORLD AI — PREDICTION STREAM")
    print("=" * 65)

    try:
        while True:
            event = await queue.get()

            print(
                event.model_dump_json(indent=2)
            )

            queue.task_done()

            if event.event_type in {
                "completed",
                "cancelled",
                "error",
            }:
                break

        print("\nFINAL STATUS:")
        status = await service.get_status(
            stream_id
        )

        print(
            status.model_dump_json(indent=2)
        )

    finally:
        await service.unsubscribe(
            stream_id,
            subscriber_id,
        )

        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
