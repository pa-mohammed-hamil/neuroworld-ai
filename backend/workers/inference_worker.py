
"""
NeuroWorld AI
backend/app/workers/imagination_worker.py

Background imagination and counterfactual worker.

Features:
- Asynchronous background job execution
- Imagination rollout jobs
- Counterfactual reasoning jobs
- Multiple imagination branches
- Job priority scheduling
- Bounded job queue
- Configurable worker concurrency
- Cooperative cancellation
- Retry with exponential backoff
- Job progress and lifecycle tracking
- Result retrieval and history
- Thread-safe job management
- Graceful worker shutdown

Requirements:
    pip install numpy "pydantic>=2"

Important:
- ImaginationService must have a registered world model.
- Predicted futures are model-generated hypotheses.
- Cancellation cannot interrupt a synchronous model
  call that is already executing.
- In-memory jobs are not durable across restarts.
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

from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.imagination_service import (
    ImaginationRequest,
    CounterfactualRequest,
    ImaginationService,
    create_imagination_service,
)


logger = logging.getLogger(
    "neuroworld.workers.imagination"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class ImaginationWorkerError(Exception):
    pass


class ImaginationJobNotFound(ImaginationWorkerError):
    pass


class ImaginationQueueFull(ImaginationWorkerError):
    pass


class ImaginationJobCancelled(ImaginationWorkerError):
    pass


class ImaginationJobValidationError(ImaginationWorkerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ImaginationJobType(str, Enum):
    ROLLOUT = "rollout"
    COUNTERFACTUAL = "counterfactual"


class ImaginationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# REQUEST MODELS
# ============================================================

class ImaginationJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_type: ImaginationJobType

    payload: dict[str, Any]

    priority: int = Field(
        default=5,
        ge=0,
        le=10,
    )

    max_retries: int = Field(
        default=0,
        ge=0,
        le=5,
    )

    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0,
        le=30,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class ImaginationJobInfo(BaseModel):
    job_id: str
    job_type: ImaginationJobType
    world_id: str

    status: ImaginationJobStatus

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    priority: int

    attempts: int
    max_retries: int

    progress_percent: float

    result: dict[str, Any] | None = None
    error: str | None = None

    duration_seconds: float | None = None

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


# ============================================================
# INTERNAL JOB
# ============================================================

@dataclass
class _ImaginationJob:
    job_id: str
    request: ImaginationJobRequest
    validated_payload: (
        ImaginationRequest | CounterfactualRequest
    )
    world_id: str

    status: ImaginationJobStatus = (
        ImaginationJobStatus.QUEUED
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    attempts: int = 0
    progress_percent: float = 0.0

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
# WORKER SERVICE
# ============================================================

class ImaginationWorker:
    """
    Priority-based background imagination worker.

    Higher priority numbers execute first.

    A condition variable coordinates worker threads.
    Jobs are stored in a bounded in-memory registry.

    Each job executes an existing ImaginationService
    operation.

    This class does not create or train a world model.
    """

    def __init__(
        self,
        imagination_service: ImaginationService,
        *,
        max_workers: int = 2,
        max_queue_size: int = 100,
        max_job_history: int = 500,
    ) -> None:
        if min(
            max_workers,
            max_queue_size,
            max_job_history,
        ) < 1:
            raise ValueError(
                "Worker limits must be positive"
            )

        self.imagination_service = (
            imagination_service
        )

        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.max_job_history = max_job_history

        self._jobs: dict[
            str,
            _ImaginationJob,
        ] = {}

        # Heap items:
        # (-priority, insertion_order, job_id)
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

        for index in range(max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=(
                    f"neuroworld-imagination-{index}"
                ),
                daemon=True,
            )

            thread.start()

            self._threads.append(thread)

    # --------------------------------------------------------
    # JOB SUBMISSION
    # --------------------------------------------------------

    def submit(
        self,
        request: ImaginationJobRequest,
    ) -> str:
        if request.job_type == ImaginationJobType.ROLLOUT:
            validated = ImaginationRequest.model_validate(
                request.payload
            )

        elif (
            request.job_type
            == ImaginationJobType.COUNTERFACTUAL
        ):
            validated = CounterfactualRequest.model_validate(
                request.payload
            )

        else:
            raise ImaginationJobValidationError(
                "Unsupported imagination job type"
            )

        with self._condition:
            if self._closed:
                raise ImaginationWorkerError(
                    "Imagination worker is closed"
                )

            queued = sum(
                job.status == ImaginationJobStatus.QUEUED
                for job in self._jobs.values()
            )

            if queued >= self.max_queue_size:
                raise ImaginationQueueFull(
                    "Imagination job queue is full"
                )

            if len(self._jobs) >= self.max_job_history:
                self._prune_completed_locked()

                if (
                    len(self._jobs)
                    >= self.max_job_history
                ):
                    raise ImaginationQueueFull(
                        "Job history capacity reached"
                    )

            job_id = uuid.uuid4().hex

            job = _ImaginationJob(
                job_id=job_id,
                request=request,
                validated_payload=validated,
                world_id=validated.world_id,
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
            "Imagination job submitted: %s",
            job_id,
        )

        return job_id

    def _prune_completed_locked(self) -> None:
        removable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in {
                    ImaginationJobStatus.COMPLETED,
                    ImaginationJobStatus.FAILED,
                    ImaginationJobStatus.CANCELLED,
                }
            ),
            key=lambda job: job.created_at,
        )

        for job in removable:
            if len(self._jobs) < self.max_job_history:
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

                job = self._jobs.get(job_id)

            if job is None:
                continue

            with job.lock:
                if job.cancel_event.is_set():
                    job.status = (
                        ImaginationJobStatus.CANCELLED
                    )
                    job.finished_at = datetime.now(
                        timezone.utc
                    )
                    continue

            self._execute_job(job)

    # --------------------------------------------------------
    # JOB EXECUTION
    # --------------------------------------------------------

    def _execute_job(
        self,
        job: _ImaginationJob,
    ) -> None:
        started = time.perf_counter()

        with job.lock:
            job.status = ImaginationJobStatus.RUNNING
            job.started_at = datetime.now(
                timezone.utc
            )
            job.progress_percent = 5.0

        total_attempts = (
            job.request.max_retries + 1
        )

        try:
            for attempt in range(
                1,
                total_attempts + 1,
            ):
                if job.cancel_event.is_set():
                    raise ImaginationJobCancelled(
                        "Job cancelled"
                    )

                with job.lock:
                    job.attempts = attempt
                    job.progress_percent = 10.0

                try:
                    result = self._dispatch(job)

                    if job.cancel_event.is_set():
                        raise ImaginationJobCancelled(
                            "Job cancelled after execution"
                        )

                    if hasattr(result, "model_dump"):
                        serialized = result.model_dump(
                            mode="json"
                        )

                    elif isinstance(result, dict):
                        serialized = copy.deepcopy(
                            result
                        )

                    else:
                        raise ImaginationWorkerError(
                            "Unsupported result format"
                        )

                    with job.lock:
                        job.result = serialized
                        job.status = (
                            ImaginationJobStatus.COMPLETED
                        )
                        job.progress_percent = 100.0

                    logger.info(
                        "Imagination job completed: %s",
                        job.job_id,
                    )

                    return

                except ImaginationJobCancelled:
                    raise

                except Exception as exc:
                    if attempt >= total_attempts:
                        raise

                    delay = min(
                        job.request.retry_delay_seconds
                        * (2 ** (attempt - 1)),
                        30.0,
                    )

                    logger.warning(
                        "Imagination job %s retry %s/%s: %s",
                        job.job_id,
                        attempt,
                        total_attempts,
                        exc,
                    )

                    if job.cancel_event.wait(delay):
                        raise ImaginationJobCancelled(
                            "Job cancelled during retry"
                        )

        except ImaginationJobCancelled:
            with job.lock:
                job.status = (
                    ImaginationJobStatus.CANCELLED
                )

        except Exception as exc:
            with job.lock:
                job.status = (
                    ImaginationJobStatus.FAILED
                )
                job.error = (
                    f"{type(exc).__name__}: {exc}"
                )

            logger.exception(
                "Imagination job failed: %s",
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
    # DISPATCH
    # --------------------------------------------------------

    def _dispatch(
        self,
        job: _ImaginationJob,
    ) -> Any:
        if (
            job.request.job_type
            == ImaginationJobType.ROLLOUT
        ):
            return self.imagination_service.imagine(
                job.validated_payload
            )

        if (
            job.request.job_type
            == ImaginationJobType.COUNTERFACTUAL
        ):
            return self.imagination_service.counterfactual(
                job.validated_payload
            )

        raise ImaginationJobValidationError(
            "Unknown imagination job type"
        )

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    def cancel(
        self,
        job_id: str,
    ) -> bool:
        job = self._get_job(job_id)

        with job.lock:
            if job.status not in {
                ImaginationJobStatus.QUEUED,
                ImaginationJobStatus.RUNNING,
            }:
                return False

            job.cancel_event.set()

            if job.status == ImaginationJobStatus.QUEUED:
                job.status = (
                    ImaginationJobStatus.CANCELLED
                )
                job.finished_at = datetime.now(
                    timezone.utc
                )

        return True

    # --------------------------------------------------------
    # JOB RETRIEVAL
    # --------------------------------------------------------

    def _get_job(
        self,
        job_id: str,
    ) -> _ImaginationJob:
        with self._lock:
            job = self._jobs.get(job_id)

        if job is None:
            raise ImaginationJobNotFound(
                "Imagination job not found"
            )

        return job

    def get_job(
        self,
        job_id: str,
    ) -> ImaginationJobInfo:
        job = self._get_job(job_id)

        with job.lock:
            return ImaginationJobInfo(
                job_id=job.job_id,
                job_type=job.request.job_type,
                world_id=job.world_id,
                status=job.status,
                created_at=job.created_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                priority=job.request.priority,
                attempts=job.attempts,
                max_retries=job.request.max_retries,
                progress_percent=(
                    job.progress_percent
                ),
                result=copy.deepcopy(
                    job.result
                ),
                error=job.error,
                duration_seconds=(
                    job.duration_seconds
                ),
                metadata=job.request.metadata,
            )

    def list_jobs(
        self,
        *,
        status: ImaginationJobStatus | None = None,
        world_id: str | None = None,
    ) -> list[ImaginationJobInfo]:
        with self._lock:
            ids = list(self._jobs)

        results = [
            self.get_job(job_id)
            for job_id in ids
        ]

        if status is not None:
            results = [
                item
                for item in results
                if item.status == status
            ]

        if world_id is not None:
            results = [
                item
                for item in results
                if item.world_id == world_id
            ]

        return results

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(
                self._jobs.values()
            )

            queued = sum(
                job.status == ImaginationJobStatus.QUEUED
                for job in jobs
            )

            running = sum(
                job.status == ImaginationJobStatus.RUNNING
                for job in jobs
            )

            completed = sum(
                job.status == ImaginationJobStatus.COMPLETED
                for job in jobs
            )

            failed = sum(
                job.status == ImaginationJobStatus.FAILED
                for job in jobs
            )

            cancelled = sum(
                job.status == ImaginationJobStatus.CANCELLED
                for job in jobs
            )

            return {
                "worker": "imagination",
                "running": not self._closed,
                "max_workers": self.max_workers,
                "max_queue_size": self.max_queue_size,
                "total_jobs": len(jobs),
                "queued_jobs": queued,
                "running_jobs": running,
                "completed_jobs": completed,
                "failed_jobs": failed,
                "cancelled_jobs": cancelled,
                "threads_alive": sum(
                    thread.is_alive()
                    for thread in self._threads
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
    ) -> None:
        with self._condition:
            self._closed = True

            if cancel_pending:
                for job in self._jobs.values():
                    if job.status in {
                        ImaginationJobStatus.QUEUED,
                        ImaginationJobStatus.RUNNING,
                    }:
                        job.cancel_event.set()

            self._condition.notify_all()

        if wait:
            for thread in self._threads:
                thread.join()


# ============================================================
# FACTORY
# ============================================================

def create_imagination_worker(
    imagination_service: (
        ImaginationService | None
    ) = None,
    *,
    max_workers: int = 2,
) -> ImaginationWorker:
    if imagination_service is None:
        imagination_service = (
            create_imagination_service()
        )

    return ImaginationWorker(
        imagination_service=imagination_service,
        max_workers=max_workers,
        max_queue_size=100,
        max_job_history=500,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.services.imagination_service import (
        DemoWorldModel,
    )

    model = DemoWorldModel(
        state_dim=4,
        action_dim=4,
    )

    imagination_service = (
        create_imagination_service(
            model=model
        )
    )

    worker = create_imagination_worker(
        imagination_service=imagination_service,
        max_workers=2,
    )

    request = ImaginationJobRequest(
        job_type=ImaginationJobType.ROLLOUT,
        priority=8,
        payload={
            "world_id": "demo-world-001",
            "initial_state": [
                0.0, 0.0, 0.0, 0.0
            ],
            "action_sequences": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
            ],
            "horizon": 3,
            "uncertainty_threshold": 1.0,
        },
        metadata={
            "experiment": "synthetic_imagination",
        },
    )

    job_id = worker.submit(request)

    print("=" * 60)
    print("NEUROWORLD AI — IMAGINATION WORKER")
    print("=" * 60)

    print("Job ID:", job_id)

    while True:
        job = worker.get_job(job_id)

        if job.status in {
            ImaginationJobStatus.COMPLETED,
            ImaginationJobStatus.FAILED,
            ImaginationJobStatus.CANCELLED,
        }:
            break

        time.sleep(0.1)

    print(
        "\nJOB RESULT:",
        job.model_dump_json(indent=2),
    )

    print("\nWORKER STATUS:")
    print(worker.status())

    worker.shutdown()


if __name__ == "__main__":
    main()
