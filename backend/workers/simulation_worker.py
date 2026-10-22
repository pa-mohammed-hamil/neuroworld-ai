
"""
NeuroWorld AI
backend/app/workers/simulation_worker.py

Background simulation execution worker.

Features:
- Background simulation jobs
- Action-sequence rollouts
- Multiple virtual environments
- Priority-based job scheduling
- Thread-safe worker pool
- Bounded job queue
- Pause and resume
- Cooperative cancellation
- Retry handling
- Simulation progress tracking
- Trajectory collection
- Reward tracking
- Episode statistics
- Worker health monitoring
- Graceful shutdown

Dependencies:
    pip install numpy "pydantic>=2"

Uses:
    backend.app.services.simulation_service

Important:
- Simulations operate in virtual environments.
- No physical robot or vehicle commands are issued.
- Jobs and results are stored in memory.
- Simulation sessions are locked against concurrent
  jobs submitted through this worker.
"""

from __future__ import annotations

import copy
import heapq
import logging
import math
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from backend.app.services.simulation_service import (
    SimulationService,
    CreateSimulationRequest,
    SimulationStepRequest,
    create_simulation_service,
)


logger = logging.getLogger(
    "neuroworld.workers.simulation"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class SimulationWorkerError(Exception):
    pass


class SimulationJobNotFound(SimulationWorkerError):
    pass


class SimulationQueueFull(SimulationWorkerError):
    pass


class SimulationJobCancelled(SimulationWorkerError):
    pass


class SimulationJobValidationError(SimulationWorkerError):
    pass


class SimulationSessionBusy(SimulationWorkerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SimulationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# REQUEST MODEL
# ============================================================

@dataclass(frozen=True)
class SimulationJobRequest:
    simulation_id: str
    actions: list[list[float]]

    priority: int = 5

    stop_on_terminal: bool = True

    max_retries: int = 0
    retry_delay_seconds: float = 0.5

    metadata: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.simulation_id.strip():
            raise SimulationJobValidationError(
                "simulation_id is required"
            )

        if not self.actions:
            raise SimulationJobValidationError(
                "At least one action is required"
            )

        if len(self.actions) > 10000:
            raise SimulationJobValidationError(
                "Too many simulation actions"
            )

        if not 0 <= self.priority <= 10:
            raise SimulationJobValidationError(
                "priority must be between 0 and 10"
            )

        if not 0 <= self.max_retries <= 5:
            raise SimulationJobValidationError(
                "max_retries must be between 0 and 5"
            )

        if not 0 <= self.retry_delay_seconds <= 30:
            raise SimulationJobValidationError(
                "Invalid retry delay"
            )

        for index, action in enumerate(
            self.actions
        ):
            if not action:
                raise SimulationJobValidationError(
                    f"Action {index} is empty"
                )

            try:
                finite = all(
                    math.isfinite(float(value))
                    for value in action
                )
            except (TypeError, ValueError) as exc:
                raise SimulationJobValidationError(
                    f"Action {index} must be numeric"
                ) from exc

            if not finite:
                raise SimulationJobValidationError(
                    f"Action {index} must be finite"
                )


# ============================================================
# JOB RESULTS
# ============================================================

@dataclass
class SimulationTrajectoryStep:
    step: int

    action: list[float]
    state: list[float]

    reward: float
    cumulative_reward: float

    terminated: bool
    truncated: bool

    info: dict[str, Any]

    latency_ms: float


@dataclass
class SimulationJobResult:
    job_id: str
    simulation_id: str
    world_id: str

    completed_steps: int
    requested_steps: int

    trajectory: list[
        SimulationTrajectoryStep
    ]

    cumulative_reward: float
    average_reward: float

    terminated: bool
    truncated: bool

    total_latency_ms: float

    metadata: dict[str, str]


# ============================================================
# INTERNAL JOB STATE
# ============================================================

@dataclass
class _SimulationJob:
    job_id: str
    request: SimulationJobRequest
    world_id: str

    status: SimulationJobStatus = (
        SimulationJobStatus.QUEUED
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

    result: SimulationJobResult | None = None

    error: str | None = None
    duration_seconds: float | None = None

    cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    resume_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


# ============================================================
# SIMULATION WORKER
# ============================================================

class SimulationWorker:
    """
    Priority-based background simulation worker.

    Each job executes actions sequentially against
    one existing SimulationService session.

    A simulation session may have only one active
    worker job at a time.

    Higher priority values execute first.

    Important:
        Do not concurrently manipulate the same
        simulation session through external API calls.
        Use a shared session coordinator in production.
    """

    def __init__(
        self,
        simulation_service: SimulationService,
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

        self.simulation_service = (
            simulation_service
        )

        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.max_job_history = max_job_history

        self._jobs: dict[
            str,
            _SimulationJob,
        ] = {}

        self._queue: list[
            tuple[int, int, str]
        ] = []

        self._reserved_sessions: set[str] = set()

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
                name=f"neuroworld-simulation-{index}",
                daemon=True,
            )

            thread.start()
            self._threads.append(thread)

    # --------------------------------------------------------
    # JOB SUBMISSION
    # --------------------------------------------------------

    def submit(
        self,
        request: SimulationJobRequest,
    ) -> str:
        simulation = (
            self.simulation_service.get_simulation(
                request.simulation_id
            )
        )

        if simulation.terminated or simulation.truncated:
            raise SimulationJobValidationError(
                "Simulation has ended; reset it first"
            )

        with self._condition:
            if self._closed:
                raise SimulationWorkerError(
                    "Simulation worker is closed"
                )

            if (
                request.simulation_id
                in self._reserved_sessions
            ):
                raise SimulationSessionBusy(
                    "Simulation session already has a job"
                )

            queued = sum(
                job.status == SimulationJobStatus.QUEUED
                for job in self._jobs.values()
            )

            if queued >= self.max_queue_size:
                raise SimulationQueueFull(
                    "Simulation queue is full"
                )

            if len(self._jobs) >= self.max_job_history:
                self._prune_completed_locked()

                if (
                    len(self._jobs)
                    >= self.max_job_history
                ):
                    raise SimulationQueueFull(
                        "Job history capacity reached"
                    )

            job_id = uuid.uuid4().hex

            job = _SimulationJob(
                job_id=job_id,
                request=request,
                world_id=simulation.world_id,
            )

            job.resume_event.set()

            self._jobs[job_id] = job

            self._reserved_sessions.add(
                request.simulation_id
            )

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
            "Simulation job submitted: %s",
            job_id,
        )

        return job_id

    def _prune_completed_locked(self) -> None:
        removable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status in {
                    SimulationJobStatus.COMPLETED,
                    SimulationJobStatus.FAILED,
                    SimulationJobStatus.CANCELLED,
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
                cancelled = (
                    job.cancel_event.is_set()
                )

            if cancelled:
                self._finish_cancelled(job)
                continue

            self._execute_job(job)

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def _execute_job(
        self,
        job: _SimulationJob,
    ) -> None:
        started = time.perf_counter()

        with job.lock:
            job.status = SimulationJobStatus.RUNNING
            job.started_at = datetime.now(
                timezone.utc
            )

        trajectory: list[
            SimulationTrajectoryStep
        ] = []

        total_steps = len(
            job.request.actions
        )

        try:
            for index, action in enumerate(
                job.request.actions,
                start=1,
            ):
                self._wait_if_paused(job)

                if job.cancel_event.is_set():
                    raise SimulationJobCancelled(
                        "Simulation job cancelled"
                    )

                step_result = self._execute_step_with_retry(
                    job,
                    action,
                )

                latency_ms = (
                    step_result["latency_ms"]
                )

                result = step_result["result"]

                trajectory.append(
                    SimulationTrajectoryStep(
                        step=index,
                        action=list(action),
                        state=list(result.state),
                        reward=float(result.reward),
                        cumulative_reward=float(
                            result.cumulative_reward
                        ),
                        terminated=bool(
                            result.terminated
                        ),
                        truncated=bool(
                            result.truncated
                        ),
                        info=copy.deepcopy(
                            result.info
                        ),
                        latency_ms=latency_ms,
                    )
                )

                with job.lock:
                    job.progress_percent = (
                        index / total_steps * 100
                    )

                if result.terminated or result.truncated:
                    break

            if job.cancel_event.is_set():
                raise SimulationJobCancelled(
                    "Simulation job cancelled"
                )

            cumulative_reward = (
                trajectory[-1].cumulative_reward
                if trajectory
                else 0.0
            )

            result = SimulationJobResult(
                job_id=job.job_id,
                simulation_id=(
                    job.request.simulation_id
                ),
                world_id=job.world_id,
                completed_steps=len(trajectory),
                requested_steps=total_steps,
                trajectory=trajectory,
                cumulative_reward=(
                    cumulative_reward
                ),
                average_reward=(
                    sum(
                        step.reward
                        for step in trajectory
                    ) / len(trajectory)
                    if trajectory
                    else 0.0
                ),
                terminated=(
                    trajectory[-1].terminated
                    if trajectory
                    else False
                ),
                truncated=(
                    trajectory[-1].truncated
                    if trajectory
                    else False
                ),
                total_latency_ms=sum(
                    step.latency_ms
                    for step in trajectory
                ),
                metadata=job.request.metadata,
            )

            with job.lock:
                job.result = result
                job.status = (
                    SimulationJobStatus.COMPLETED
                )
                job.progress_percent = 100.0

            logger.info(
                "Simulation job completed: %s",
                job.job_id,
            )

        except SimulationJobCancelled:
            with job.lock:
                job.status = (
                    SimulationJobStatus.CANCELLED
                )

        except Exception as exc:
            with job.lock:
                job.status = (
                    SimulationJobStatus.FAILED
                )

                job.error = (
                    f"{type(exc).__name__}: {exc}"
                )

            logger.exception(
                "Simulation job failed: %s",
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

            with self._condition:
                self._reserved_sessions.discard(
                    job.request.simulation_id
                )

    # --------------------------------------------------------
    # RETRY EXECUTION
    # --------------------------------------------------------

    def _execute_step_with_retry(
        self,
        job: _SimulationJob,
        action: list[float],
    ) -> dict[str, Any]:
        total_attempts = (
            job.request.max_retries + 1
        )

        for attempt in range(
            1,
            total_attempts + 1,
        ):
            if job.cancel_event.is_set():
                raise SimulationJobCancelled(
                    "Job cancelled"
                )

            with job.lock:
                job.attempts += 1

            started = time.perf_counter()

            try:
                result = self.simulation_service.step(
                    SimulationStepRequest(
                        simulation_id=(
                            job.request.simulation_id
                        ),
                        action=action,
                    )
                )

                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                return {
                    "result": result,
                    "latency_ms": latency_ms,
                }

            except Exception as exc:
                if attempt >= total_attempts:
                    raise

                delay = min(
                    job.request.retry_delay_seconds
                    * (2 ** (attempt - 1)),
                    10.0,
                )

                logger.warning(
                    "Simulation step retry %s/%s: %s",
                    attempt,
                    total_attempts,
                    exc,
                )

                if job.cancel_event.wait(delay):
                    raise SimulationJobCancelled(
                        "Cancelled during retry"
                    )

        raise SimulationWorkerError(
            "Simulation step retries exhausted"
        )

    # --------------------------------------------------------
    # PAUSE / RESUME
    # --------------------------------------------------------

    def _wait_if_paused(
        self,
        job: _SimulationJob,
    ) -> None:
        while not job.resume_event.wait(
            timeout=0.1
        ):
            if job.cancel_event.is_set():
                raise SimulationJobCancelled(
                    "Job cancelled while paused"
                )

    def pause(
        self,
        job_id: str,
    ) -> bool:
        job = self._get_job(
            job_id
        )

        with job.lock:
            if job.status != SimulationJobStatus.RUNNING:
                return False

            job.status = (
                SimulationJobStatus.PAUSED
            )

            job.resume_event.clear()

        return True

    def resume(
        self,
        job_id: str,
    ) -> bool:
        job = self._get_job(
            job_id
        )

        with job.lock:
            if job.status != SimulationJobStatus.PAUSED:
                return False

            job.status = (
                SimulationJobStatus.RUNNING
            )

            job.resume_event.set()

        return True

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
                SimulationJobStatus.QUEUED,
                SimulationJobStatus.RUNNING,
                SimulationJobStatus.PAUSED,
            }:
                return False

            job.cancel_event.set()
            job.resume_event.set()

            if job.status == SimulationJobStatus.QUEUED:
                job.status = (
                    SimulationJobStatus.CANCELLED
                )

                job.finished_at = datetime.now(
                    timezone.utc
                )

                with self._condition:
                    self._reserved_sessions.discard(
                        job.request.simulation_id
                    )

        return True

    def _finish_cancelled(
        self,
        job: _SimulationJob,
    ) -> None:
        with job.lock:
            job.status = (
                SimulationJobStatus.CANCELLED
            )

            if job.finished_at is None:
                job.finished_at = datetime.now(
                    timezone.utc
                )

        with self._condition:
            self._reserved_sessions.discard(
                job.request.simulation_id
            )

    # --------------------------------------------------------
    # JOB RETRIEVAL
    # --------------------------------------------------------

    def _get_job(
        self,
        job_id: str,
    ) -> _SimulationJob:
        with self._lock:
            job = self._jobs.get(
                job_id
            )

        if job is None:
            raise SimulationJobNotFound(
                "Simulation job not found"
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
            result = None

            if job.result is not None:
                result = {
                    "job_id": job.result.job_id,
                    "simulation_id": (
                        job.result.simulation_id
                    ),
                    "world_id": job.result.world_id,
                    "completed_steps": (
                        job.result.completed_steps
                    ),
                    "requested_steps": (
                        job.result.requested_steps
                    ),
                    "trajectory": [
                        {
                            "step": step.step,
                            "action": step.action,
                            "state": step.state,
                            "reward": step.reward,
                            "cumulative_reward": (
                                step.cumulative_reward
                            ),
                            "terminated": step.terminated,
                            "truncated": step.truncated,
                            "info": step.info,
                            "latency_ms": step.latency_ms,
                        }
                        for step in job.result.trajectory
                    ],
                    "cumulative_reward": (
                        job.result.cumulative_reward
                    ),
                    "average_reward": (
                        job.result.average_reward
                    ),
                    "terminated": (
                        job.result.terminated
                    ),
                    "truncated": (
                        job.result.truncated
                    ),
                    "total_latency_ms": (
                        job.result.total_latency_ms
                    ),
                    "metadata": (
                        job.result.metadata
                    ),
                }

            return {
                "job_id": job.job_id,
                "simulation_id": (
                    job.request.simulation_id
                ),
                "world_id": job.world_id,
                "status": job.status.value,
                "priority": job.request.priority,
                "progress_percent": (
                    job.progress_percent
                ),
                "attempts": job.attempts,
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
                "result": copy.deepcopy(result),
                "error": job.error,
            }

    def list_jobs(
        self,
        *,
        status: SimulationJobStatus | None = None,
        world_id: str | None = None,
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

        if world_id is not None:
            jobs = [
                job
                for job in jobs
                if job["world_id"] == world_id
            ]

        return jobs

    # --------------------------------------------------------
    # MONITORING
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(
                self._jobs.values()
            )

            reserved = len(
                self._reserved_sessions
            )

        counts = {
            status.value: 0
            for status in SimulationJobStatus
        }

        for job in jobs:
            counts[job.status.value] += 1

        return {
            "worker": "simulation",
            "running": not self._closed,
            "max_workers": self.max_workers,
            "max_queue_size": self.max_queue_size,
            "total_jobs": len(jobs),
            "jobs_by_status": counts,
            "reserved_simulations": reserved,
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
                        SimulationJobStatus.QUEUED,
                        SimulationJobStatus.RUNNING,
                        SimulationJobStatus.PAUSED,
                    }:
                        job.cancel_event.set()
                        job.resume_event.set()

            self._condition.notify_all()

        if wait:
            for thread in self._threads:
                thread.join()


# ============================================================
# FACTORY
# ============================================================

def create_simulation_worker(
    simulation_service: (
        SimulationService | None
    ) = None,
    *,
    max_workers: int = 2,
) -> SimulationWorker:
    if simulation_service is None:
        simulation_service = (
            create_simulation_service()
        )

    return SimulationWorker(
        simulation_service=simulation_service,
        max_workers=max_workers,
        max_queue_size=100,
        max_job_history=500,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
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

    worker = create_simulation_worker(
        simulation_service=simulation_service,
        max_workers=2,
    )

    request = SimulationJobRequest(
        simulation_id=simulation.simulation_id,
        actions=[
            [1.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.0, 0.0],
            [-0.5, -0.5],
        ],
        priority=8,
        metadata={
            "experiment": "synthetic_physics_demo",
        },
    )

    job_id = worker.submit(request)

    print("=" * 60)
    print("NEUROWORLD AI — SIMULATION WORKER")
    print("=" * 60)

    print("Job ID:", job_id)

    try:
        while True:
            job = worker.get_job(job_id)

            if job["status"] in {
                "completed",
                "failed",
                "cancelled",
            }:
                break

            time.sleep(0.1)

        print("\nJOB RESULT:")
        print(job)

        print("\nWORKER STATUS:")
        print(worker.status())

    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
