
"""
NeuroWorld AI
backend/app/streaming/frame_stream.py

Real-time video frame streaming service.

Features:
- Live camera streaming
- Video-file streaming
- Synthetic frame generation
- OpenCV frame capture
- JPEG encoding
- Resolution and FPS control
- Async frame delivery
- WebSocket streaming
- Frame timestamps and metadata
- Frame processing hooks
- Bounded frame queues
- Dropped-frame tracking
- Stream lifecycle management

Dependencies:
    pip install opencv-python numpy fastapi

Security:
- Camera access is local and explicit.
- Video files must be inside an approved directory.
- WebSocket endpoints require authentication and
  world-level authorization before deployment.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, model_validator


logger = logging.getLogger("neuroworld.streaming.frames")


# ============================================================
# EXCEPTIONS
# ============================================================

class FrameStreamError(Exception):
    pass


class FrameSourceError(FrameStreamError):
    pass


class FrameEncodingError(FrameStreamError):
    pass


class FrameStreamNotFound(FrameStreamError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

class FrameStreamConfig(BaseModel):
    world_id: str = Field(min_length=1, max_length=128)

    source_type: Literal[
        "camera",
        "video",
        "synthetic",
    ] = "synthetic"

    camera_index: int = Field(default=0, ge=0, le=32)

    video_path: str | None = None

    width: int = Field(default=640, ge=64, le=1920)
    height: int = Field(default=360, ge=64, le=1080)

    fps: float = Field(default=15.0, gt=0, le=60)

    jpeg_quality: int = Field(
        default=75,
        ge=20,
        le=95,
    )

    queue_size: int = Field(default=4, ge=1, le=32)

    loop_video: bool = False

    @model_validator(mode="after")
    def validate_source(self):
        if (
            self.source_type == "video"
            and not self.video_path
        ):
            raise ValueError(
                "video_path is required for video sources"
            )

        return self


class FrameMetadata(BaseModel):
    frame_id: str
    stream_id: str
    world_id: str

    sequence: int
    timestamp: datetime

    width: int
    height: int

    encoding: str = "jpeg"
    size_bytes: int


class EncodedFrame(BaseModel):
    metadata: FrameMetadata
    image_base64: str


class FrameStreamStatus(BaseModel):
    stream_id: str
    world_id: str
    source_type: str

    running: bool

    frames_captured: int
    frames_delivered: int
    frames_dropped: int

    elapsed_seconds: float
    actual_fps: float

    queue_depth: int
    last_error: str | None = None


# ============================================================
# FRAME PROCESSING
# ============================================================

FrameProcessor = Callable[
    [np.ndarray],
    np.ndarray,
]


def validate_frame(
    frame: np.ndarray,
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise FrameStreamError(
            "Frame must be a NumPy array"
        )

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise FrameStreamError(
            "Frame must have shape (H, W, 3)"
        )

    if frame.dtype != np.uint8:
        raise FrameStreamError(
            "Frame must use uint8 pixels"
        )

    if frame.size == 0:
        raise FrameStreamError(
            "Frame cannot be empty"
        )

    return frame


def preprocess_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    frame = validate_frame(frame)

    resized = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    return resized


def encode_jpeg(
    frame: np.ndarray,
    *,
    quality: int = 75,
) -> bytes:
    frame = validate_frame(frame)

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality,
        ],
    )

    if not success:
        raise FrameEncodingError(
            "JPEG encoding failed"
        )

    return encoded.tobytes()


def frame_to_rgb(
    frame: np.ndarray,
) -> np.ndarray:
    return cv2.cvtColor(
        validate_frame(frame),
        cv2.COLOR_BGR2RGB,
    )


def frame_to_tensor(
    frame: np.ndarray,
) -> np.ndarray:
    """
    Convert BGR frame into normalized RGB CHW array.

    Output shape:
        (3, height, width)

    Value range:
        [0, 1]
    """
    rgb = frame_to_rgb(frame)

    return (
        rgb.astype(np.float32)
        .transpose(2, 0, 1)
        / 255.0
    )


# ============================================================
# SYNTHETIC FRAME SOURCE
# ============================================================

class SyntheticFrameSource:
    """
    Generates test frames without camera hardware.

    This is a visual test source, not a neural
    video-generation model.
    """

    def __init__(
        self,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height
        self.frame_index = 0

    def read(
        self,
    ) -> tuple[bool, np.ndarray]:
        frame = np.zeros(
            (
                self.height,
                self.width,
                3,
            ),
            dtype=np.uint8,
        )

        self.frame_index += 1

        x = (
            self.frame_index * 6
        ) % self.width

        y = self.height // 2

        cv2.circle(
            frame,
            (x, y),
            24,
            (0, 220, 255),
            -1,
        )

        cv2.putText(
            frame,
            "NeuroWorld AI - Synthetic Stream",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Frame: {self.frame_index}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (150, 255, 150),
            1,
            cv2.LINE_AA,
        )

        return True, frame

    def release(self) -> None:
        pass


# ============================================================
# STREAM RUNTIME
# ============================================================

@dataclass
class _StreamRuntime:
    stream_id: str
    config: FrameStreamConfig

    queue: asyncio.Queue[EncodedFrame]

    running: bool = False

    frames_captured: int = 0
    frames_delivered: int = 0
    frames_dropped: int = 0

    started_at: float = 0.0
    last_error: str | None = None

    task: asyncio.Task | None = None

    processors: list[FrameProcessor] = field(
        default_factory=list
    )


# ============================================================
# FRAME STREAM SERVICE
# ============================================================

class FrameStreamService:
    """
    Async frame-stream manager.

    Each stream has a bounded queue. If a consumer
    is slower than the producer, old frames are
    discarded to keep latency low.

    The default service supports one consuming
    WebSocket per stream. Use a subscriber/broadcast
    architecture for multiple simultaneous viewers.
    """

    def __init__(
        self,
        *,
        allowed_video_directory: str | Path = (
            "data/videos"
        ),
        max_streams: int = 8,
    ) -> None:
        if max_streams < 1:
            raise ValueError(
                "max_streams must be positive"
            )

        self.allowed_video_directory = Path(
            allowed_video_directory
        ).resolve()

        self.max_streams = max_streams

        self._streams: dict[
            str,
            _StreamRuntime,
        ] = {}

        self._lock = threading.RLock()

    def _resolve_video_path(
        self,
        video_path: str,
    ) -> Path:
        root = self.allowed_video_directory

        path = Path(video_path)

        if not path.is_absolute():
            path = root / path

        path = path.resolve()

        if not path.is_relative_to(root):
            raise FrameSourceError(
                "Video path is outside allowed directory"
            )

        if not path.is_file():
            raise FrameSourceError(
                "Video file does not exist"
            )

        if path.suffix.lower() not in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
        }:
            raise FrameSourceError(
                "Unsupported video extension"
            )

        return path

    def _open_source(
        self,
        config: FrameStreamConfig,
    ) -> Any:
        if config.source_type == "synthetic":
            return SyntheticFrameSource(
                config.width,
                config.height,
            )

        if config.source_type == "camera":
            capture = cv2.VideoCapture(
                config.camera_index
            )

        else:
            video_path = self._resolve_video_path(
                config.video_path or ""
            )

            capture = cv2.VideoCapture(
                str(video_path)
            )

        if not capture.isOpened():
            capture.release()

            raise FrameSourceError(
                "Unable to open frame source"
            )

        if config.source_type == "camera":
            capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                config.width,
            )

            capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                config.height,
            )

        return capture

    async def create_stream(
        self,
        config: FrameStreamConfig,
    ) -> str:
        """
        Create and start a frame stream.
        """
        with self._lock:
            if len(self._streams) >= self.max_streams:
                raise FrameStreamError(
                    "Maximum stream count reached"
                )

            stream_id = uuid.uuid4().hex

            runtime = _StreamRuntime(
                stream_id=stream_id,
                config=config,
                queue=asyncio.Queue(
                    maxsize=config.queue_size
                ),
            )

            self._streams[stream_id] = runtime

        runtime.running = True
        runtime.started_at = time.monotonic()

        runtime.task = asyncio.create_task(
            self._capture_loop(runtime)
        )

        logger.info(
            "Frame stream started: %s",
            stream_id,
        )

        return stream_id

    async def _capture_loop(
        self,
        runtime: _StreamRuntime,
    ) -> None:
        source = None

        try:
            # Opening and reading OpenCV captures may block.
            source = await asyncio.to_thread(
                self._open_source,
                runtime.config,
            )

            interval = 1.0 / runtime.config.fps

            while runtime.running:
                loop_started = time.monotonic()

                success, frame = await asyncio.to_thread(
                    source.read
                )

                if not success:
                    if (
                        runtime.config.source_type == "video"
                        and runtime.config.loop_video
                    ):
                        await asyncio.to_thread(
                            source.set,
                            cv2.CAP_PROP_POS_FRAMES,
                            0,
                        )
                        continue

                    break

                frame = preprocess_frame(
                    frame,
                    width=runtime.config.width,
                    height=runtime.config.height,
                )

                for processor in runtime.processors:
                    frame = processor(frame)
                    frame = validate_frame(frame)

                jpeg_bytes = encode_jpeg(
                    frame,
                    quality=runtime.config.jpeg_quality,
                )

                runtime.frames_captured += 1

                metadata = FrameMetadata(
                    frame_id=uuid.uuid4().hex,
                    stream_id=runtime.stream_id,
                    world_id=runtime.config.world_id,
                    sequence=runtime.frames_captured,
                    timestamp=datetime.now(
                        timezone.utc
                    ),
                    width=frame.shape[1],
                    height=frame.shape[0],
                    size_bytes=len(jpeg_bytes),
                )

                encoded_frame = EncodedFrame(
                    metadata=metadata,
                    image_base64=base64.b64encode(
                        jpeg_bytes
                    ).decode("ascii"),
                )

                if runtime.queue.full():
                    try:
                        runtime.queue.get_nowait()
                        runtime.queue.task_done()

                        runtime.frames_dropped += 1

                    except asyncio.QueueEmpty:
                        pass

                runtime.queue.put_nowait(
                    encoded_frame
                )

                elapsed = (
                    time.monotonic() - loop_started
                )

                await asyncio.sleep(
                    max(0.0, interval - elapsed)
                )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            runtime.last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "Frame capture failed: %s",
                runtime.stream_id,
            )

        finally:
            runtime.running = False

            if source is not None:
                await asyncio.to_thread(
                    source.release
                )

            logger.info(
                "Frame stream stopped: %s",
                runtime.stream_id,
            )

    def _get_stream(
        self,
        stream_id: str,
    ) -> _StreamRuntime:
        with self._lock:
            runtime = self._streams.get(
                stream_id
            )

        if runtime is None:
            raise FrameStreamNotFound(
                "Frame stream not found"
            )

        return runtime

    async def get_frame(
        self,
        stream_id: str,
        *,
        timeout: float = 5.0,
    ) -> EncodedFrame | None:
        runtime = self._get_stream(
            stream_id
        )

        try:
            frame = await asyncio.wait_for(
                runtime.queue.get(),
                timeout=timeout,
            )

            runtime.queue.task_done()
            runtime.frames_delivered += 1

            return frame

        except asyncio.TimeoutError:
            return None

    def add_processor(
        self,
        stream_id: str,
        processor: FrameProcessor,
    ) -> None:
        if not callable(processor):
            raise ValueError(
                "Frame processor must be callable"
            )

        runtime = self._get_stream(
            stream_id
        )

        runtime.processors.append(
            processor
        )

    async def stop_stream(
        self,
        stream_id: str,
    ) -> None:
        runtime = self._get_stream(
            stream_id
        )

        runtime.running = False

        if runtime.task is not None:
            runtime.task.cancel()

            try:
                await runtime.task
            except asyncio.CancelledError:
                pass

        with self._lock:
            self._streams.pop(
                stream_id,
                None,
            )

    async def stop_all(self) -> None:
        with self._lock:
            stream_ids = list(
                self._streams
            )

        for stream_id in stream_ids:
            await self.stop_stream(
                stream_id
            )

    def get_status(
        self,
        stream_id: str,
    ) -> FrameStreamStatus:
        runtime = self._get_stream(
            stream_id
        )

        elapsed = max(
            0.0,
            time.monotonic() - runtime.started_at,
        )

        actual_fps = (
            runtime.frames_captured / elapsed
            if elapsed > 0
            else 0.0
        )

        return FrameStreamStatus(
            stream_id=stream_id,
            world_id=runtime.config.world_id,
            source_type=runtime.config.source_type,
            running=runtime.running,
            frames_captured=(
                runtime.frames_captured
            ),
            frames_delivered=(
                runtime.frames_delivered
            ),
            frames_dropped=(
                runtime.frames_dropped
            ),
            elapsed_seconds=elapsed,
            actual_fps=actual_fps,
            queue_depth=runtime.queue.qsize(),
            last_error=runtime.last_error,
        )

    def list_streams(
        self,
    ) -> list[FrameStreamStatus]:
        with self._lock:
            stream_ids = list(
                self._streams
            )

        return [
            self.get_status(stream_id)
            for stream_id in stream_ids
        ]


# ============================================================
# FASTAPI WEBSOCKET ROUTER
# ============================================================

def create_frame_stream_router(
    service: FrameStreamService,
    *,
    authorize: Callable[
        [WebSocket, str],
        Any,
    ] | None = None,
) -> APIRouter:
    """
    WebSocket frame delivery.

    The authorization callback must validate the
    connected user and their access to the stream.

    If no authorization callback is supplied, the
    WebSocket rejects all connections (fail closed).
    """

    router = APIRouter(
        tags=["Frame Streaming"],
    )

    @router.websocket(
        "/api/v1/streaming/frames/{stream_id}"
    )
    async def stream_frames(
        websocket: WebSocket,
        stream_id: str,
    ):
        try:
            runtime = service._get_stream(
                stream_id
            )
        except FrameStreamNotFound:
            await websocket.close(code=4404)
            return

        if authorize is None:
            await websocket.close(code=4403)
            return

        try:
            authorized = authorize(
                websocket,
                runtime.config.world_id,
            )

            if asyncio.iscoroutine(authorized):
                authorized = await authorized

            if authorized is not True:
                await websocket.close(code=4403)
                return

        except Exception:
            await websocket.close(code=4403)
            return

        await websocket.accept()

        try:
            while True:
                frame = await service.get_frame(
                    stream_id,
                    timeout=2.0,
                )

                if frame is None:
                    status = service.get_status(
                        stream_id
                    )

                    if not status.running:
                        break

                    continue

                await websocket.send_json(
                    frame.model_dump(
                        mode="json"
                    )
                )

        except (
            WebSocketDisconnect,
            FrameStreamNotFound,
        ):
            pass

        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return router


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_frame_stream_service(
    *,
    allowed_video_directory: str | Path = (
        "data/videos"
    ),
) -> FrameStreamService:
    return FrameStreamService(
        allowed_video_directory=(
            allowed_video_directory
        ),
        max_streams=8,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:
    service = create_frame_stream_service()

    stream_id = await service.create_stream(
        FrameStreamConfig(
            world_id="demo-world-001",
            source_type="synthetic",
            width=640,
            height=360,
            fps=10,
        )
    )

    print("=" * 60)
    print("NEUROWORLD AI — FRAME STREAM")
    print("=" * 60)

    try:
        for _ in range(5):
            frame = await service.get_frame(
                stream_id
            )

            if frame is None:
                continue

            print(
                "Frame:",
                frame.metadata.sequence,
                "Size:",
                frame.metadata.size_bytes,
                "bytes",
            )

        print()
        print(
            service.get_status(
                stream_id
            ).model_dump_json(indent=2)
        )

    finally:
        await service.stop_all()


if __name__ == "__main__":
    asyncio.run(main())
