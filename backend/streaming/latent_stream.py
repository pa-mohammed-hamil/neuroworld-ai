
"""
NeuroWorld AI
backend/app/streaming/latent_stream.py

Real-time latent-state streaming.

Features:
- Latent vector streaming
- Observation-to-latent encoding adapters
- Manual latent-state publishing
- Action-conditioned latent transitions
- Async event broadcasting
- Multiple WebSocket subscribers
- Bounded queues and backpressure
- Recent-event replay
- Stream pause and resume
- Stream lifecycle management
- Sequence numbers and timestamps
- State-dimension validation
- Uncertainty tracking
- WebSocket authorization hooks

Dependencies:
    pip install numpy fastapi "pydantic>=2"

Notes:
- Real latent encoding requires a registered encoder.
- Real model transitions require a world-model adapter.
- Streams are process-local.
- No physical hardware is controlled.
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
from typing import Any, Awaitable, Callable, Protocol

import numpy as np

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


logger = logging.getLogger(
    "neuroworld.streaming.latent"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class LatentStreamError(Exception):
    """Base latent streaming error."""


class LatentStreamNotFound(LatentStreamError):
    """Requested stream does not exist."""


class LatentStreamCapacityError(LatentStreamError):
    """Stream capacity has been reached."""


class LatentStreamValidationError(LatentStreamError):
    """Invalid latent state or observation."""


class LatentEncoderNotLoaded(LatentStreamError):
    """No latent encoder has been registered."""


class LatentModelNotLoaded(LatentStreamError):
    """No world-model adapter has been registered."""


# ============================================================
# MODEL INTERFACES
# ============================================================

class LatentEncoder(Protocol):
    """
    Observation encoder interface.

    Input:
        Numerical observation vector.

    Output:
        Numerical latent-state vector.
    """

    name: str
    observation_dim: int
    latent_dim: int

    def encode(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        ...


class LatentTransitionModel(Protocol):
    """
    Action-conditioned latent transition interface.

    Compatible with a world-model adapter implementing
    predict_transition(state, action).
    """

    name: str
    state_dim: int
    action_dim: int

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        ...


# ============================================================
# ENUMS
# ============================================================

class LatentStreamStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class LatentEventType(str, Enum):
    INITIALIZED = "initialized"
    STATE_UPDATED = "state_updated"
    OBSERVATION_ENCODED = "observation_encoded"
    TRANSITION_PREDICTED = "transition_predicted"
    PAUSED = "paused"
    RESUMED = "resumed"
    CLOSED = "closed"


# ============================================================
# REQUEST MODELS
# ============================================================

class CreateLatentStreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str = Field(
        min_length=1,
        max_length=128,
    )

    latent_dimension: int = Field(
        ge=1,
        le=16384,
    )

    initial_state: list[float] | None = None

    queue_size: int = Field(
        default=32,
        ge=1,
        le=512,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class PublishLatentStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    source: str = Field(
        default="manual",
        min_length=1,
        max_length=128,
    )

    expected_sequence: int | None = Field(
        default=None,
        ge=0,
    )

    uncertainty: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class EncodeObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    observation_id: str | None = Field(
        default=None,
        max_length=128,
    )

    expected_sequence: int | None = Field(
        default=None,
        ge=0,
    )


class PredictLatentTransitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: list[float] = Field(
        min_length=1,
        max_length=16384,
    )

    expected_sequence: int | None = Field(
        default=None,
        ge=0,
    )

    uncertainty_threshold: float = Field(
        default=1.0,
        ge=0,
        le=1,
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class LatentStreamEvent(BaseModel):
    event_id: str
    stream_id: str
    world_id: str

    event_type: LatentEventType

    sequence: int
    timestamp: datetime

    state: list[float] | None = None

    uncertainty: float | None = None
    predicted_reward: float | None = None

    source: str | None = None
    observation_id: str | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class LatentStreamInfo(BaseModel):
    stream_id: str
    world_id: str

    status: LatentStreamStatus

    latent_dimension: int

    created_at: datetime
    updated_at: datetime

    sequence: int

    current_state: list[float]

    subscribers: int
    dropped_events: int

    encoder_name: str | None = None
    transition_model_name: str | None = None


# ============================================================
# VALIDATION
# ============================================================

def validate_vector(
    values: Any,
    *,
    dimension: int,
    name: str,
) -> np.ndarray:
    try:
        vector = np.asarray(
            values,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise LatentStreamValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if vector.shape != (dimension,):
        raise LatentStreamValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.all(np.isfinite(vector)):
        raise LatentStreamValidationError(
            f"{name} must contain finite values"
        )

    return vector.copy()


def validate_uncertainty(
    value: Any,
) -> float:
    try:
        uncertainty = float(value)
    except (TypeError, ValueError) as exc:
        raise LatentStreamValidationError(
            "Uncertainty must be numeric"
        ) from exc

    if (
        not math.isfinite(uncertainty)
        or not 0 <= uncertainty <= 1
    ):
        raise LatentStreamValidationError(
            "Uncertainty must be between 0 and 1"
        )

    return uncertainty


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# INTERNAL STREAM SESSION
# ============================================================

@dataclass
class _LatentSession:
    stream_id: str
    config: CreateLatentStreamRequest

    state: np.ndarray

    status: LatentStreamStatus = (
        LatentStreamStatus.ACTIVE
    )

    sequence: int = 0

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    subscribers: dict[
        str,
        asyncio.Queue[LatentStreamEvent],
    ] = field(default_factory=dict)

    history: deque[LatentStreamEvent] = field(
        default_factory=lambda: deque(maxlen=256)
    )

    dropped_events: int = 0

    lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )


# ============================================================
# LATENT STREAM SERVICE
# ============================================================

class LatentStreamService:
    """
    Manages real-time latent-state streams.

    Each stream maintains:
        - Current latent state
        - Monotonic sequence number
        - Recent event history
        - Independent subscriber queues

    Encoders and transition models are registered
    separately and shared across streams.

    One service instance should be owned by one
    application event loop.
    """

    def __init__(
        self,
        *,
        encoder: LatentEncoder | None = None,
        transition_model: (
            LatentTransitionModel | None
        ) = None,
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

        self.encoder = encoder
        self.transition_model = transition_model

        self.max_streams = max_streams

        self.max_subscribers_per_stream = (
            max_subscribers_per_stream
        )

        self.max_history = max_history

        self._sessions: dict[
            str,
            _LatentSession,
        ] = {}

        self._lock = asyncio.Lock()

    # --------------------------------------------------------
    # MODEL REGISTRATION
    # --------------------------------------------------------

    def register_encoder(
        self,
        encoder: LatentEncoder,
    ) -> None:
        if (
            encoder.observation_dim < 1
            or encoder.latent_dim < 1
        ):
            raise LatentStreamValidationError(
                "Encoder dimensions must be positive"
            )

        self.encoder = encoder

        logger.info(
            "Latent encoder registered: %s",
            encoder.name,
        )

    def register_transition_model(
        self,
        model: LatentTransitionModel,
    ) -> None:
        if (
            model.state_dim < 1
            or model.action_dim < 1
        ):
            raise LatentStreamValidationError(
                "Transition model dimensions must be positive"
            )

        self.transition_model = model

        logger.info(
            "Latent transition model registered: %s",
            model.name,
        )

    # --------------------------------------------------------
    # STREAM CREATION
    # --------------------------------------------------------

    async def create_stream(
        self,
        request: CreateLatentStreamRequest,
    ) -> str:
        if request.initial_state is None:
            state = np.zeros(
                request.latent_dimension,
                dtype=np.float64,
            )
        else:
            state = validate_vector(
                request.initial_state,
                dimension=request.latent_dimension,
                name="initial_state",
            )

        async with self._lock:
            if len(self._sessions) >= self.max_streams:
                raise LatentStreamCapacityError(
                    "Maximum latent stream count reached"
                )

            stream_id = uuid.uuid4().hex

            session = _LatentSession(
                stream_id=stream_id,
                config=request,
                state=state,
                history=deque(
                    maxlen=self.max_history
                ),
            )

            self._sessions[stream_id] = session

        await self._publish(
            session,
            LatentStreamEvent(
                event_id=uuid.uuid4().hex,
                stream_id=stream_id,
                world_id=request.world_id,
                event_type=LatentEventType.INITIALIZED,
                sequence=0,
                timestamp=utc_now(),
                state=state.tolist(),
                source="initialization",
                metadata=request.metadata,
            ),
        )

        return stream_id

    async def _get_session(
        self,
        stream_id: str,
    ) -> _LatentSession:
        async with self._lock:
            session = self._sessions.get(
                stream_id
            )

        if session is None:
            raise LatentStreamNotFound(
                "Latent stream not found"
            )

        return session

    # --------------------------------------------------------
    # EVENT PUBLISHING
    # --------------------------------------------------------

    async def _publish(
        self,
        session: _LatentSession,
        event: LatentStreamEvent,
    ) -> None:
        async with session.lock:
            self._publish_locked(
                session,
                event,
            )

    def _publish_locked(
        self,
        session: _LatentSession,
        event: LatentStreamEvent,
    ) -> None:
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

    def _commit_state_locked(
        self,
        session: _LatentSession,
        *,
        state: np.ndarray,
        event_type: LatentEventType,
        source: str,
        uncertainty: float | None = None,
        predicted_reward: float | None = None,
        observation_id: str | None = None,
    ) -> LatentStreamEvent:
        session.state = state.copy()
        session.sequence += 1
        session.updated_at = utc_now()

        event = LatentStreamEvent(
            event_id=uuid.uuid4().hex,
            stream_id=session.stream_id,
            world_id=session.config.world_id,
            event_type=event_type,
            sequence=session.sequence,
            timestamp=session.updated_at,
            state=state.tolist(),
            uncertainty=uncertainty,
            predicted_reward=predicted_reward,
            source=source,
            observation_id=observation_id,
            metadata=session.config.metadata,
        )

        self._publish_locked(
            session,
            event,
        )

        return event

    @staticmethod
    def _require_active(
        session: _LatentSession,
    ) -> None:
        if session.status != LatentStreamStatus.ACTIVE:
            raise LatentStreamError(
                "Latent stream is not active"
            )

    @staticmethod
    def _check_sequence(
        session: _LatentSession,
        expected_sequence: int | None,
    ) -> None:
        if (
            expected_sequence is not None
            and expected_sequence != session.sequence
        ):
            raise LatentStreamValidationError(
                "Latent stream sequence conflict"
            )

    # --------------------------------------------------------
    # MANUAL LATENT UPDATES
    # --------------------------------------------------------

    async def publish_state(
        self,
        stream_id: str,
        request: PublishLatentStateRequest,
    ) -> LatentStreamEvent:
        session = await self._get_session(
            stream_id
        )

        state = validate_vector(
            request.state,
            dimension=session.config.latent_dimension,
            name="state",
        )

        async with session.lock:
            self._require_active(session)

            self._check_sequence(
                session,
                request.expected_sequence,
            )

            return self._commit_state_locked(
                session,
                state=state,
                event_type=(
                    LatentEventType.STATE_UPDATED
                ),
                source=request.source,
                uncertainty=request.uncertainty,
            )

    # --------------------------------------------------------
    # OBSERVATION ENCODING
    # --------------------------------------------------------

    async def encode_observation(
        self,
        stream_id: str,
        request: EncodeObservationRequest,
    ) -> LatentStreamEvent:
        encoder = self.encoder

        if encoder is None:
            raise LatentEncoderNotLoaded(
                "No latent encoder registered"
            )

        session = await self._get_session(
            stream_id
        )

        if (
            encoder.latent_dim
            != session.config.latent_dimension
        ):
            raise LatentStreamValidationError(
                "Encoder latent dimension mismatch"
            )

        observation = validate_vector(
            request.observation,
            dimension=encoder.observation_dim,
            name="observation",
        )

        # Hold the session lock while encoding so that
        # state updates remain sequential.
        async with session.lock:
            self._require_active(session)

            self._check_sequence(
                session,
                request.expected_sequence,
            )

            encoded = await asyncio.to_thread(
                encoder.encode,
                observation.copy(),
            )

            latent = validate_vector(
                encoded,
                dimension=encoder.latent_dim,
                name="encoded_latent",
            )

            return self._commit_state_locked(
                session,
                state=latent,
                event_type=(
                    LatentEventType.OBSERVATION_ENCODED
                ),
                source=f"encoder:{encoder.name}",
                observation_id=(
                    request.observation_id
                ),
            )

    # --------------------------------------------------------
    # LATENT TRANSITIONS
    # --------------------------------------------------------

    async def predict_transition(
        self,
        stream_id: str,
        request: PredictLatentTransitionRequest,
    ) -> LatentStreamEvent:
        model = self.transition_model

        if model is None:
            raise LatentModelNotLoaded(
                "No transition model registered"
            )

        session = await self._get_session(
            stream_id
        )

        if (
            model.state_dim
            != session.config.latent_dimension
        ):
            raise LatentStreamValidationError(
                "Transition model state dimension mismatch"
            )

        action = validate_vector(
            request.action,
            dimension=model.action_dim,
            name="action",
        )

        async with session.lock:
            self._require_active(session)

            self._check_sequence(
                session,
                request.expected_sequence,
            )

            transition = await asyncio.to_thread(
                model.predict_transition,
                session.state.copy(),
                action.copy(),
            )

            next_state = validate_vector(
                transition.next_state,
                dimension=model.state_dim,
                name="next_state",
            )

            uncertainty = validate_uncertainty(
                transition.uncertainty
            )

            try:
                reward = float(
                    transition.reward
                )
            except (TypeError, ValueError) as exc:
                raise LatentStreamValidationError(
                    "Predicted reward must be numeric"
                ) from exc

            if not math.isfinite(reward):
                raise LatentStreamValidationError(
                    "Predicted reward must be finite"
                )

            if (
                uncertainty
                > request.uncertainty_threshold
            ):
                raise LatentStreamValidationError(
                    "Transition uncertainty exceeds threshold"
                )

            return self._commit_state_locked(
                session,
                state=next_state,
                event_type=(
                    LatentEventType.TRANSITION_PREDICTED
                ),
                source=f"model:{model.name}",
                uncertainty=uncertainty,
                predicted_reward=reward,
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
        asyncio.Queue[LatentStreamEvent],
    ]:
        session = await self._get_session(
            stream_id
        )

        subscriber_id = uuid.uuid4().hex

        queue: asyncio.Queue[
            LatentStreamEvent
        ] = asyncio.Queue(
            maxsize=session.config.queue_size
        )

        async with session.lock:
            if (
                len(session.subscribers)
                >= self.max_subscribers_per_stream
            ):
                raise LatentStreamCapacityError(
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
    # PAUSE / RESUME / CLOSE
    # --------------------------------------------------------

    async def pause_stream(
        self,
        stream_id: str,
    ) -> None:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            if session.status != LatentStreamStatus.ACTIVE:
                raise LatentStreamError(
                    "Only active streams can be paused"
                )

            session.status = LatentStreamStatus.PAUSED
            session.updated_at = utc_now()

            self._publish_locked(
                session,
                LatentStreamEvent(
                    event_id=uuid.uuid4().hex,
                    stream_id=stream_id,
                    world_id=session.config.world_id,
                    event_type=LatentEventType.PAUSED,
                    sequence=session.sequence,
                    timestamp=session.updated_at,
                    state=session.state.tolist(),
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
            if session.status != LatentStreamStatus.PAUSED:
                raise LatentStreamError(
                    "Only paused streams can be resumed"
                )

            session.status = LatentStreamStatus.ACTIVE
            session.updated_at = utc_now()

            self._publish_locked(
                session,
                LatentStreamEvent(
                    event_id=uuid.uuid4().hex,
                    stream_id=stream_id,
                    world_id=session.config.world_id,
                    event_type=LatentEventType.RESUMED,
                    sequence=session.sequence,
                    timestamp=session.updated_at,
                    state=session.state.tolist(),
                ),
            )

    async def close_stream(
        self,
        stream_id: str,
    ) -> None:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            if session.status == LatentStreamStatus.CLOSED:
                return

            session.status = LatentStreamStatus.CLOSED
            session.updated_at = utc_now()

            self._publish_locked(
                session,
                LatentStreamEvent(
                    event_id=uuid.uuid4().hex,
                    stream_id=stream_id,
                    world_id=session.config.world_id,
                    event_type=LatentEventType.CLOSED,
                    sequence=session.sequence,
                    timestamp=session.updated_at,
                    state=session.state.tolist(),
                ),
            )

    async def delete_stream(
        self,
        stream_id: str,
    ) -> bool:
        await self.close_stream(stream_id)

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
            ids = list(self._sessions)

        for stream_id in ids:
            await self.delete_stream(stream_id)

    # --------------------------------------------------------
    # STATUS / HISTORY
    # --------------------------------------------------------

    async def get_status(
        self,
        stream_id: str,
    ) -> LatentStreamInfo:
        session = await self._get_session(
            stream_id
        )

        async with session.lock:
            return LatentStreamInfo(
                stream_id=session.stream_id,
                world_id=session.config.world_id,
                status=session.status,
                latent_dimension=(
                    session.config.latent_dimension
                ),
                created_at=session.created_at,
                updated_at=session.updated_at,
                sequence=session.sequence,
                current_state=session.state.tolist(),
                subscribers=len(
                    session.subscribers
                ),
                dropped_events=(
                    session.dropped_events
                ),
                encoder_name=(
                    self.encoder.name
                    if self.encoder is not None
                    else None
                ),
                transition_model_name=(
                    self.transition_model.name
                    if self.transition_model is not None
                    else None
                ),
            )

    async def get_history(
        self,
        stream_id: str,
        *,
        limit: int = 100,
    ) -> list[LatentStreamEvent]:
        session = await self._get_session(
            stream_id
        )

        if not 1 <= limit <= self.max_history:
            raise LatentStreamValidationError(
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
    ) -> list[LatentStreamInfo]:
        async with self._lock:
            ids = list(self._sessions)

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


def create_latent_stream_router(
    service: LatentStreamService,
    *,
    authorize: AuthorizationCallback | None = None,
) -> APIRouter:
    """
    WebSocket latent-event delivery.

    Authorization must verify the user and world
    access before accepting the connection.

    Without an authorization callback, connections
    are rejected.
    """

    router = APIRouter(
        tags=["Latent Streaming"]
    )

    @router.websocket(
        "/api/v1/streaming/latents/{stream_id}"
    )
    async def latent_websocket(
        websocket: WebSocket,
        stream_id: str,
    ):
        try:
            session = await service._get_session(
                stream_id
            )
        except LatentStreamNotFound:
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
        except LatentStreamCapacityError:
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

                    if status.status == LatentStreamStatus.CLOSED:
                        break

                    continue

                await websocket.send_json(
                    event.model_dump(
                        mode="json"
                    )
                )

                queue.task_done()

                if event.event_type == LatentEventType.CLOSED:
                    break

        except (
            WebSocketDisconnect,
            LatentStreamNotFound,
        ):
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

def create_latent_stream_service(
    *,
    encoder: LatentEncoder | None = None,
    transition_model: (
        LatentTransitionModel | None
    ) = None,
) -> LatentStreamService:
    return LatentStreamService(
        encoder=encoder,
        transition_model=transition_model,
        max_streams=50,
        max_subscribers_per_stream=20,
        max_history=256,
    )


# ============================================================
# SYNTHETIC DEMONSTRATION
# ============================================================

class DemoLatentEncoder:
    """
    Example linear encoder for testing.

    This is not a trained neural encoder.
    """

    name = "demo_linear_encoder"
    observation_dim = 6
    latent_dim = 4

    def __init__(self) -> None:
        self.matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 1.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.1],
            ],
            dtype=np.float64,
        )

    def encode(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        return self.matrix @ observation


class DemoLatentTransition:
    """
    Synthetic latent dynamics for testing.

    next_state = state + action * 0.1
    """

    name = "demo_latent_dynamics"
    state_dim = 4
    action_dim = 4

    def predict_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Any:
        from backend.app.services.imagination_service import (
            Transition,
        )

        next_state = state + action * 0.1

        return Transition(
            next_state=next_state,
            uncertainty=0.05,
            reward=-float(
                np.linalg.norm(next_state)
            ),
            terminal=False,
        )


async def main() -> None:
    service = create_latent_stream_service(
        encoder=DemoLatentEncoder(),
        transition_model=DemoLatentTransition(),
    )

    stream_id = await service.create_stream(
        CreateLatentStreamRequest(
            world_id="demo-world-001",
            latent_dimension=4,
            initial_state=[
                0.0, 0.0, 0.0, 0.0
            ],
        )
    )

    subscriber_id, queue = await service.subscribe(
        stream_id
    )

    await service.encode_observation(
        stream_id,
        EncodeObservationRequest(
            observation=[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
            ],
            observation_id="observation-001",
        ),
    )

    await service.predict_transition(
        stream_id,
        PredictLatentTransitionRequest(
            action=[
                0.1, 0.2, 0.3, 0.4
            ],
        ),
    )

    await service.publish_state(
        stream_id,
        PublishLatentStateRequest(
            state=[
                0.5, 0.6, 0.7, 0.8
            ],
            source="manual_demo",
        ),
    )

    await service.close_stream(
        stream_id
    )

    print("=" * 60)
    print("NEUROWORLD AI — LATENT STREAM")
    print("=" * 60)

    while not queue.empty():
        event = queue.get_nowait()

        print(
            event.model_dump_json(indent=2)
        )

        queue.task_done()

    print("\nFINAL STATUS:")
    print(
        (
            await service.get_status(stream_id)
        ).model_dump_json(indent=2)
    )

    await service.unsubscribe(
        stream_id,
        subscriber_id,
    )

    await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
