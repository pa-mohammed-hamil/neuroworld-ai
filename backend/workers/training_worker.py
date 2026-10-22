
"""
NeuroWorld AI
backend/app/workers/training_worker.py

Background training worker for neural world models.

Features:
- Background model training orchestration
- Priority-based job scheduling
- Multiple training jobs
- Configurable worker concurrency
- TrainingService integration
- Training status polling
- Epoch and progress monitoring
- Checkpoint result tracking
- Early stopping support
- Cooperative cancellation
- Bounded job queue
- Retry handling
- Training job history
- Graceful shutdown

Dependencies:
    pip install torch numpy "pydantic>=2"

Uses:
    backend.app.services.training_service

Notes:
- Models and datasets must be registered with
  TrainingService before submitting jobs.
- TrainingService performs actual optimization.
- TrainingWorker handles orchestration.
- Jobs are process-local and not durable.
"""

from __future__ import annotations

import copy
import heapq
import logging
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from backend.app.services.training_service import (
    TrainingService,
    TrainingConfig,
    TrainingStatus,
    create_training_service,
)


logger = logging.getLogger(
    "neuroworld.workers.training"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TrainingWorkerError(Exception):
    pass


class TrainingWorkerJobNotFound(TrainingWorkerError):
    pass


class TrainingWorkerQueueFull(TrainingWorkerError):
    pass


class TrainingWorkerCancelled(TrainingWorkerError):
    pass


class TrainingWorkerValidationError(TrainingWorkerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TrainingWorkerJobStatus(str, Enum):
    QUEUED = "queued"
    SUBMITTING = "submitting"
    RUNNING = "running"
    COMPLETED = "completed"
    EARLY_STOPPED = "early_stopped"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrainingWorkerConfig:
    max_workers: int = 2
    max_queue_size: int = 100
    max_job_history: int = 500

    poll_interval_seconds: float = 0.5

    max_retries: int = 0
    retry_delay_seconds: float = 1.0

    def __post_init__(self):
        if min(
            self.max_workers,
            self.max_queue_size,
            self.max_job_history,
        ) < 1:
            raise ValueError(
                "Worker limits must be positive"
            )

        if self.poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be positive"
            )

        if not 0 <= self.max_retries <= 5:
            raise ValueError(
                "max_retries must be between 0 and 5"
            )

        if self.retry_delay_seconds < 0:
            raise ValueError(
                "retry_delay_seconds cannot be negative"
            )


# ============================================================
# TRAINING REQUEST
# ============================================================

@dataclass(frozen=True)
class TrainingWorkerRequest:
    training_config: TrainingConfig

    priority: int = 5

    max_retries: int | None = None

    metadata: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not 0 <= self.priority <= 10:
            raise TrainingWorkerValidationError(
                "priority must be between 0 and 10"
            )

        if (
            self.max_retries is not None
            and not 0 <= self.max_retries <= 5
        ):
            raise TrainingWorkerValidationError(
                "max_retries must be between 0 and 5"
            )


# ============================================================
# INTERNAL JOB
# ============================================================

@dataclass
class _TrainingWorkerJob:
    job_id: str
    request: TrainingWorkerRequest

    status: TrainingWorkerJobStatus = (
        TrainingWorkerJobStatus.QUEUED
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    training_job_id: str | None = None

    attempts: int = 0
    progress_percent: float = 0.0

    current_epoch: int = 0
    total_epochs: int = 0

    latest_metrics: dict[str, Any] = field(
        default_factory=dict
    )

    checkpoints: list[str] = field(
        default_factory=list
    )

    result: dict[str, Any] | None = None
    error: str | None = None

    duration_seconds: float | None = None

    cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


# ============================================================
# SERIALIZATION UTILITIES
# ============================================================

def serialize_value(
    value: Any,
) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json"
        )

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): serialize_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize_value(item)
            for item in value
        ]

    return value


def get_field(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(
        value,
        name,
        default,
    )


def normalize_status(
    value: Any,
) -> str:
    if isinstance(value, Enum):
        return str(value.value).lower()

    return str(value).lower()


# ============================================================
# TRAINING WORKER
# ============================================================

class TrainingWorker:
    """
    Priority-based training job coordinator.

    TrainingService owns the actual PyTorch
    training executor and checkpoint lifecycle.

    TrainingWorker:
        1. Accepts training requests
        2. Orders requests by priority
        3. Submits jobs to TrainingService
        4. Polls training progress
        5. Propagates cancellation
        6. Stores completed job summaries

    Higher priority values execute first.

    The worker does not directly modify
    neural-network parameters.
    """

    TERMINAL_STATUSES = {
        "completed",
        "early_stopped",
        "cancelled",
        "failed",
    }

    def __init__(
        self,
        training_service: TrainingService,
        *,
        config: TrainingWorkerConfig | None = None,
    ) -> None:
        self.training_service = training_service

        self.config = (
            config or TrainingWorkerConfig()
        )

        self._jobs: dict[
            str,
            _TrainingWorkerJob,
        ] = {}

        self._queue: list[
            tuple[int, int, str]
        ] = []

        self._sequence = 0

        self._lock = threading.RLock()

        self._condition = threading.Condition(
            self._lock
        )

        self._closed = False

        self._threads: list[
            threading.Thread
        ] = []

        for index in range(
            self.config.max_workers
        ):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"neuroworld-training-{index}",
                daemon=True,
            )

            thread.start()

            self._threads.append(thread)

    # --------------------------------------------------------
    # SUBMIT TRAINING
    # --------------------------------------------------------

    def submit(
        self,
        request: TrainingWorkerRequest,
    ) -> str:
        with self._condition:
            if self._closed:
                raise TrainingWorkerError(
                    "Training worker is closed"
                )

            queued_count = sum(
                job.status
                == TrainingWorkerJobStatus.QUEUED
                for job in self._jobs.values()
            )

            if (
                queued_count
                >= self.config.max_queue_size
            ):
                raise TrainingWorkerQueueFull(
                    "Training queue is full"
                )

            if (
                len(self._jobs)
                >= self.config.max_job_history
            ):
                self._prune_completed_locked()

                if (
                    len(self._jobs)
                    >= self.config.max_job_history
                ):
                    raise TrainingWorkerQueueFull(
                        "Training job history is full"
                    )

            job_id = uuid.uuid4().hex

            job = _TrainingWorkerJob(
                job_id=job_id,
                request=request,
                total_epochs=(
                    request.training_config.epochs
                ),
            )

            self._jobs[job_id] = job

            self._sequence += 1

            heapq.heappush(
                self._queue,
                (
                    -request.priority,
                    self._sequence,
                    job_id,
                ),
            )

            self._condition.notify()

        logger.info(
            "Training worker job submitted: %s",
            job_id,
        )

        return job_id

    def _prune_completed_locked(self) -> None:
        removable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in {
                    TrainingWorkerJobStatus.COMPLETED,
                    TrainingWorkerJobStatus.EARLY_STOPPED,
                    TrainingWorkerJobStatus.FAILED,
                    TrainingWorkerJobStatus.CANCELLED,
                }
            ),
            key=lambda job: job.created_at,
        )

        for job in removable:
            if (
                len(self._jobs)
                < self.config.max_job_history
            ):
                break

            self._jobs.pop(
                job.job_id,
                None,
            )

    # --------------------------------------------------------
    # WORKER LOOP
    # --------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while (
                    not self._queue
                    and not self._closed
                ):
                    self._condition.wait()

                if (
                    self._closed
                    and not self._queue
                ):
                    return

                _, _, job_id = heapq.heappop(
                    self._queue
                )

                job = self._jobs.get(
                    job_id
                )

            if job is None:
                continue

            with job.lock:
                if job.cancel_event.is_set():
                    job.status = (
                        TrainingWorkerJobStatus.CANCELLED
                    )

                    job.finished_at = datetime.now(
                        timezone.utc
                    )

                    continue

            self._execute_job(job)

    # --------------------------------------------------------
    # TRAINING EXECUTION
    # --------------------------------------------------------

    def _execute_job(
        self,
        job: _TrainingWorkerJob,
    ) -> None:
        started = time.perf_counter()

        with job.lock:
            job.status = (
                TrainingWorkerJobStatus.SUBMITTING
            )

            job.started_at = datetime.now(
                timezone.utc
            )

        max_retries = (
            job.request.max_retries
            if job.request.max_retries is not None
            else self.config.max_retries
        )

        total_attempts = max_retries + 1

        try:
            for attempt in range(
                1,
                total_attempts + 1,
            ):
                if job.cancel_event.is_set():
                    raise TrainingWorkerCancelled(
                        "Training cancelled"
                    )

                with job.lock:
                    job.attempts = attempt

                try:
                    training_job_id = (
                        self.training_service.start_training(
                            job.request.training_config
                        )
                    )

                    with job.lock:
                        job.training_job_id = (
                            training_job_id
                        )

                        job.status = (
                            TrainingWorkerJobStatus.RUNNING
                        )

                    if job.cancel_event.is_set():
                        self.training_service.cancel_training(
                            training_job_id
                        )

                    self._monitor_training(
                        job,
                        training_job_id,
                    )

                    return

                except TrainingWorkerCancelled:
                    raise

                except Exception as exc:
                    if attempt >= total_attempts:
                        raise

                    logger.warning(
                        "Training job %s retry %s/%s: %s",
                        job.job_id,
                        attempt,
                        total_attempts,
                        exc,
                    )

                    delay = min(
                        self.config.retry_delay_seconds
                        * (2 ** (attempt - 1)),
                        30.0,
                    )

                    if job.cancel_event.wait(delay):
                        raise TrainingWorkerCancelled(
                            "Cancelled during retry"
                        )

        except TrainingWorkerCancelled:
            with job.lock:
                job.status = (
                    TrainingWorkerJobStatus.CANCELLED
                )

        except Exception as exc:
            with job.lock:
                job.status = (
                    TrainingWorkerJobStatus.FAILED
                )

                job.error = (
                    f"{type(exc).__name__}: {exc}"
                )

            logger.exception(
                "Training worker job failed: %s",
                job.job_id,
            )

        finally:
            with job.lock:
                job.finished_at = datetime.now(
                    timezone.utc
                )

                job.duration_seconds = (
                    time.perf_counter() - started
                )

    # --------------------------------------------------------
    # TRAINING MONITOR
    # --------------------------------------------------------

    def _monitor_training(
        self,
        job: _TrainingWorkerJob,
        training_job_id: str,
    ) -> None:
        while True:
            if job.cancel_event.is_set():
                self.training_service.cancel_training(
                    training_job_id
                )

            training_status = (
                self.training_service.get_training_status(
                    training_job_id
                )
            )

            status = normalize_status(
                get_field(
                    training_status,
                    "status",
                    "unknown",
                )
            )

            current_epoch = get_field(
                training_status,
                "current_epoch",
                0,
            )

            if current_epoch is None:
                current_epoch = 0

            current_epoch = int(
                current_epoch
            )

            total_epochs = (
                job.request.training_config.epochs
            )

            progress = (
                current_epoch / total_epochs * 100
                if total_epochs > 0
                else 0.0
            )

            metrics = get_field(
                training_status,
                "latest_metrics",
                {},
            )

            if metrics is None:
                metrics = {}

            checkpoints = get_field(
                training_status,
                "checkpoints",
                [],
            )

            if checkpoints is None:
                checkpoints = []

            with job.lock:
                job.current_epoch = current_epoch

                job.progress_percent = min(
                    max(progress, 0.0),
                    100.0,
                )

                job.latest_metrics = serialize_value(
                    metrics
                )

                job.checkpoints = [
                    str(path)
                    for path in checkpoints
                ]

            if status in self.TERMINAL_STATUSES:
                serialized = serialize_value(
                    training_status
                )

                with job.lock:
                    job.result = serialized

                    if status == "completed":
                        job.status = (
                            TrainingWorkerJobStatus.COMPLETED
                        )

                        job.progress_percent = 100.0

                    elif status == "early_stopped":
                        job.status = (
                            TrainingWorkerJobStatus.EARLY_STOPPED
                        )

                    elif status == "cancelled":
                        job.status = (
                            TrainingWorkerJobStatus.CANCELLED
                        )

                    elif status == "failed":
                        job.status = (
                            TrainingWorkerJobStatus.FAILED
                        )

                        job.error = str(
                            get_field(
                                training_status,
                                "error",
                                "Training failed",
                            )
                        )

                return

            time.sleep(
                self.config.poll_interval_seconds
            )

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    def cancel(
        self,
        job_id: str,
    ) -> bool:
        job = self._get_job(
            job_id
        )

        with job.lock:
            if job.status not in {
                TrainingWorkerJobStatus.QUEUED,
                TrainingWorkerJobStatus.SUBMITTING,
                TrainingWorkerJobStatus.RUNNING,
            }:
                return False

            job.cancel_event.set()

            training_job_id = (
                job.training_job_id
            )

            if (
                job.status
                == TrainingWorkerJobStatus.QUEUED
            ):
                job.status = (
                    TrainingWorkerJobStatus.CANCELLED
                )

                job.finished_at = datetime.now(
                    timezone.utc
                )

        if training_job_id is not None:
            self.training_service.cancel_training(
                training_job_id
            )

        return True

    # --------------------------------------------------------
    # JOB RETRIEVAL
    # --------------------------------------------------------

    def _get_job(
        self,
        job_id: str,
    ) -> _TrainingWorkerJob:
        with self._lock:
            job = self._jobs.get(
                job_id
            )

        if job is None:
            raise TrainingWorkerJobNotFound(
                "Training worker job not found"
            )

        return job

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        job = self._get_job(
            job_id
        )

        with job.lock:
            return {
                "job_id": job.job_id,
                "training_job_id": (
                    job.training_job_id
                ),
                "model_name": (
                    job.request.training_config.model_name
                ),
                "status": job.status.value,
                "priority": job.request.priority,
                "attempts": job.attempts,
                "current_epoch": (
                    job.current_epoch
                ),
                "total_epochs": (
                    job.total_epochs
                ),
                "progress_percent": (
                    job.progress_percent
                ),
                "latest_metrics": copy.deepcopy(
                    job.latest_metrics
                ),
                "checkpoints": list(
                    job.checkpoints
                ),
                "created_at": (
                    job.created_at.isoformat()
                ),
                "started_at": (
                    job.started_at.isoformat()
                    if job.started_at
                    else None
                ),
                "finished_at": (
                    job.finished_at.isoformat()
                    if job.finished_at
                    else None
                ),
                "duration_seconds": (
                    job.duration_seconds
                ),
                "result": copy.deepcopy(
                    job.result
                ),
                "error": job.error,
                "metadata": (
                    job.request.metadata
                ),
            }

    def list_jobs(
        self,
        *,
        status: TrainingWorkerJobStatus | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(
                self._jobs
            )

        jobs = [
            self.get_job(job_id)
            for job_id in ids
        ]

        if status is not None:
            jobs = [
                job
                for job in jobs
                if job["status"] == status.value
            ]

        if model_name is not None:
            jobs = [
                job
                for job in jobs
                if job["model_name"] == model_name
            ]

        return jobs

    # --------------------------------------------------------
    # WORKER STATUS
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(
                self._jobs.values()
            )

        counts = {
            status.value: 0
            for status in TrainingWorkerJobStatus
        }

        for job in jobs:
            counts[job.status.value] += 1

        return {
            "worker": "training",
            "running": not self._closed,
            "max_workers": (
                self.config.max_workers
            ),
            "max_queue_size": (
                self.config.max_queue_size
            ),
            "total_jobs": len(jobs),
            "jobs_by_status": counts,
            "threads_alive": sum(
                thread.is_alive()
                for thread in self._threads
            ),
            "training_service": serialize_value(
                self.training_service.status()
            ),
        }

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = True,
        shutdown_training_service: bool = False,
    ) -> None:
        with self._condition:
            self._closed = True

            if cancel_pending:
                for job in self._jobs.values():
                    if job.status in {
                        TrainingWorkerJobStatus.QUEUED,
                        TrainingWorkerJobStatus.SUBMITTING,
                        TrainingWorkerJobStatus.RUNNING,
                    }:
                        job.cancel_event.set()

            self._condition.notify_all()

        if wait:
            for thread in self._threads:
                thread.join()

        if shutdown_training_service:
            self.training_service.shutdown()


# ============================================================
# FACTORY
# ============================================================

def create_training_worker(
    training_service: (
        TrainingService | None
    ) = None,
    *,
    config: TrainingWorkerConfig | None = None,
) -> TrainingWorker:
    if training_service is None:
        training_service = (
            create_training_service()
        )

    return TrainingWorker(
        training_service=training_service,
        config=config,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    """
    Demonstration of worker initialization.

    Before submitting a training job, register
    a model factory and dataset provider with
    TrainingService.
    """

    training_service = (
        create_training_service()
    )

    worker = create_training_worker(
        training_service=training_service,
        config=TrainingWorkerConfig(
            max_workers=2,
            max_queue_size=100,
            poll_interval_seconds=0.5,
        ),
    )

    print("=" * 60)
    print("NEUROWORLD AI — TRAINING WORKER")
    print("=" * 60)

    print("\nAVAILABLE MODELS:")
    print(
        training_service.list_models()
    )

    print("\nWORKER STATUS:")
    print(
        worker.status()
    )

    print(
        "\nRegister a training model and dataset "
        "before submitting jobs."
    )

    worker.shutdown(
        wait=True,
        cancel_pending=True,
        shutdown_training_service=True,
    )


if __name__ == "__main__":
    main()
