
"""
NeuroWorld AI
backend/app/workers/evaluation_worker.py

Background evaluation worker.

Features:
- Background evaluation jobs
- Prediction and trajectory evaluation
- Planning and simulation evaluation
- Uncertainty calibration
- Custom evaluation adapters
- Benchmark execution
- Job queue and concurrency limits
- Job cancellation
- Progress tracking
- Evaluation result storage
- Retry support
- Worker health and statistics
- Graceful shutdown

Dependencies:
    pip install numpy pydantic

This worker runs within one Python process.
For distributed production execution, replace
the executor with Celery, RQ, or a durable queue.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid

from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np

from backend.app.services.evaluation_service import (
    EvaluationService,
    create_evaluation_service,
)


logger = logging.getLogger(
    "neuroworld.workers.evaluation"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class EvaluationWorkerError(Exception):
    pass


class EvaluationJobNotFound(EvaluationWorkerError):
    pass


class EvaluationQueueFull(EvaluationWorkerError):
    pass


class EvaluationJobCancelled(EvaluationWorkerError):
    pass


class EvaluationJobValidationError(EvaluationWorkerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class EvaluationJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationJobType(str, Enum):
    PREDICTION = "prediction"
    TRAJECTORY = "trajectory"
    UNCERTAINTY = "uncertainty"
    PLANNING = "planning"
    SIMULATION = "simulation"
    CUSTOM = "custom"


# ============================================================
# JOB MODELS
# ============================================================

@dataclass(frozen=True)
class EvaluationJobRequest:
    evaluation_type: EvaluationJobType
    model_name: str
    payload: dict[str, Any]

    priority: int = 5
    max_retries: int = 0

    metadata: dict[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self):
        if not self.model_name.strip():
            raise EvaluationJobValidationError(
                "model_name is required"
            )

        if not 0 <= self.priority <= 10:
            raise EvaluationJobValidationError(
                "priority must be between 0 and 10"
            )

        if not 0 <= self.max_retries <= 5:
            raise EvaluationJobValidationError(
                "max_retries must be between 0 and 5"
            )


@dataclass
class EvaluationJob:
    job_id: str
    request: EvaluationJobRequest

    status: EvaluationJobStatus = (
        EvaluationJobStatus.QUEUED
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    progress_percent: float = 0.0

    attempts: int = 0

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
# CUSTOM EVALUATOR INTERFACE
# ============================================================

CustomEvaluator = Callable[
    [dict[str, Any]],
    dict[str, Any],
]


# ============================================================
# EVALUATION WORKER
# ============================================================

class EvaluationWorker:
    """
    Background evaluation execution service.

    Uses a bounded ThreadPoolExecutor.

    EvaluationService handles metric calculations.
    EvaluationWorker handles job orchestration.

    Job results are kept in memory.
    """

    def __init__(
        self,
        evaluation_service: EvaluationService,
        *,
        max_workers: int = 2,
        max_jobs: int = 100,
    ) -> None:
        if max_workers < 1 or max_jobs < 1:
            raise ValueError(
                "Worker limits must be positive"
            )

        self.evaluation_service = (
            evaluation_service
        )

        self.max_workers = max_workers
        self.max_jobs = max_jobs

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=(
                "neuroworld-evaluation"
            ),
        )

        self._jobs: dict[
            str,
            EvaluationJob,
        ] = {}

        self._futures: dict[
            str,
            Future,
        ] = {}

        self._custom_evaluators: dict[
            str,
            CustomEvaluator,
        ] = {}

        self._lock = threading.RLock()

        self._closed = False

    # --------------------------------------------------------
    # CUSTOM EVALUATORS
    # --------------------------------------------------------

    def register_evaluator(
        self,
        name: str,
        evaluator: CustomEvaluator,
    ) -> None:
        if not name.strip():
            raise EvaluationJobValidationError(
                "Evaluator name cannot be empty"
            )

        if not callable(evaluator):
            raise EvaluationJobValidationError(
                "Evaluator must be callable"
            )

        with self._lock:
            self._custom_evaluators[name] = evaluator

    def list_evaluators(
        self,
    ) -> list[str]:
        with self._lock:
            return sorted(
                self._custom_evaluators
            )

    # --------------------------------------------------------
    # JOB SUBMISSION
    # --------------------------------------------------------

    def submit(
        self,
        request: EvaluationJobRequest,
    ) -> str:
        with self._lock:
            if self._closed:
                raise EvaluationWorkerError(
                    "Evaluation worker is closed"
                )

            active_jobs = sum(
                job.status in {
                    EvaluationJobStatus.QUEUED,
                    EvaluationJobStatus.RUNNING,
                }
                for job in self._jobs.values()
            )

            if active_jobs >= self.max_jobs:
                raise EvaluationQueueFull(
                    "Evaluation queue is full"
                )

            job_id = uuid.uuid4().hex

            job = EvaluationJob(
                job_id=job_id,
                request=request,
            )

            self._jobs[job_id] = job

            future = self._executor.submit(
                self._execute_job,
                job,
            )

            self._futures[job_id] = future

        logger.info(
            "Evaluation job submitted: %s",
            job_id,
        )

        return job_id

    # --------------------------------------------------------
    # JOB EXECUTION
    # --------------------------------------------------------

    def _execute_job(
        self,
        job: EvaluationJob,
    ) -> None:
        started = time.perf_counter()

        with job.lock:
            if job.cancel_event.is_set():
                job.status = (
                    EvaluationJobStatus.CANCELLED
                )
                job.finished_at = datetime.now(
                    timezone.utc
                )
                return

            job.status = EvaluationJobStatus.RUNNING
            job.started_at = datetime.now(
                timezone.utc
            )

        total_attempts = (
            job.request.max_retries + 1
        )

        try:
            for attempt in range(
                1,
                total_attempts + 1,
            ):
                if job.cancel_event.is_set():
                    raise EvaluationJobCancelled(
                        "Evaluation cancelled"
                    )

                with job.lock:
                    job.attempts = attempt
                    job.progress_percent = 10.0

                try:
                    result = self._dispatch(
                        job.request
                    )

                    if job.cancel_event.is_set():
                        raise EvaluationJobCancelled(
                            "Evaluation cancelled"
                        )

                    with job.lock:
                        job.result = result
                        job.progress_percent = 100.0
                        job.status = (
                            EvaluationJobStatus.COMPLETED
                        )

                    logger.info(
                        "Evaluation job completed: %s",
                        job.job_id,
                    )

                    return

                except EvaluationJobCancelled:
                    raise

                except Exception as exc:
                    if attempt >= total_attempts:
                        raise

                    logger.warning(
                        "Evaluation retry %s/%s for %s: %s",
                        attempt,
                        total_attempts,
                        job.job_id,
                        exc,
                    )

                    # Short bounded retry delay.
                    if job.cancel_event.wait(
                        min(2 ** (attempt - 1), 5)
                    ):
                        raise EvaluationJobCancelled(
                            "Evaluation cancelled"
                        )

        except EvaluationJobCancelled:
            with job.lock:
                job.status = (
                    EvaluationJobStatus.CANCELLED
                )

        except Exception as exc:
            with job.lock:
                job.status = (
                    EvaluationJobStatus.FAILED
                )

                job.error = (
                    f"{type(exc).__name__}: {exc}"
                )

            logger.exception(
                "Evaluation job failed: %s",
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
    # EVALUATION DISPATCH
    # --------------------------------------------------------

    def _dispatch(
        self,
        request: EvaluationJobRequest,
    ) -> dict[str, Any]:
        evaluation_type = (
            request.evaluation_type
        )

        payload = request.payload

        if evaluation_type == EvaluationJobType.PREDICTION:
            result = (
                self.evaluation_service.evaluate_prediction(
                    model_name=request.model_name,
                    predictions=np.asarray(
                        payload["predictions"],
                        dtype=np.float64,
                    ),
                    targets=np.asarray(
                        payload["targets"],
                        dtype=np.float64,
                    ),
                    metadata=request.metadata,
                )
            )

        elif evaluation_type == EvaluationJobType.TRAJECTORY:
            result = (
                self.evaluation_service.evaluate_trajectory(
                    model_name=request.model_name,
                    predicted_trajectories=np.asarray(
                        payload["predicted_trajectories"],
                        dtype=np.float64,
                    ),
                    target_trajectories=np.asarray(
                        payload["target_trajectories"],
                        dtype=np.float64,
                    ),
                    metadata=request.metadata,
                )
            )

        elif evaluation_type == EvaluationJobType.UNCERTAINTY:
            result = (
                self.evaluation_service.evaluate_uncertainty(
                    model_name=request.model_name,
                    predicted_uncertainties=np.asarray(
                        payload["predicted_uncertainties"],
                        dtype=np.float64,
                    ),
                    observed_errors=np.asarray(
                        payload["observed_errors"],
                        dtype=np.float64,
                    ),
                    metadata=request.metadata,
                )
            )

        elif evaluation_type == EvaluationJobType.PLANNING:
            result = (
                self.evaluation_service.evaluate_planning(
                    model_name=request.model_name,
                    planning_results=payload[
                        "planning_results"
                    ],
                    metadata=request.metadata,
                )
            )

        elif evaluation_type == EvaluationJobType.SIMULATION:
            result = (
                self.evaluation_service.evaluate_simulation(
                    model_name=request.model_name,
                    episodes=payload["episodes"],
                    metadata=request.metadata,
                )
            )

        elif evaluation_type == EvaluationJobType.CUSTOM:
            evaluator_name = payload.get(
                "evaluator_name"
            )

            with self._lock:
                evaluator = (
                    self._custom_evaluators.get(
                        evaluator_name
                    )
                )

            if evaluator is None:
                raise EvaluationJobValidationError(
                    "Custom evaluator is not registered"
                )

            result = evaluator(
                copy.deepcopy(payload)
            )

        else:
            raise EvaluationJobValidationError(
                "Unsupported evaluation type"
            )

        if hasattr(result, "model_dump"):
            return result.model_dump(
                mode="json"
            )

        if isinstance(result, dict):
            return result

        raise EvaluationWorkerError(
            "Evaluator returned unsupported result"
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
                EvaluationJobStatus.QUEUED,
                EvaluationJobStatus.RUNNING,
            }:
                return False

            job.cancel_event.set()

        future = self._futures.get(
            job_id
        )

        if future is not None:
            future.cancel()

        return True

    # --------------------------------------------------------
    # JOB RETRIEVAL
    # --------------------------------------------------------

    def _get_job(
        self,
        job_id: str,
    ) -> EvaluationJob:
        with self._lock:
            job = self._jobs.get(
                job_id
            )

        if job is None:
            raise EvaluationJobNotFound(
                "Evaluation job not found"
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
                "evaluation_type": (
                    job.request.evaluation_type.value
                ),
                "model_name": (
                    job.request.model_name
                ),
                "status": job.status.value,
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
                "progress_percent": (
                    job.progress_percent
                ),
                "attempts": job.attempts,
                "duration_seconds": (
                    job.duration_seconds
                ),
                "result": copy.deepcopy(
                    job.result
                ),
                "error": job.error,
            }

    def list_jobs(
        self,
        *,
        status: EvaluationJobStatus | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(
                self._jobs
            )

        results = [
            self.get_job(job_id)
            for job_id in ids
        ]

        if status is not None:
            results = [
                item
                for item in results
                if item["status"] == status.value
            ]

        return results

    # --------------------------------------------------------
    # WORKER STATUS
    # --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(
                self._jobs.values()
            )

            evaluator_names = sorted(
                self._custom_evaluators
            )

        counts = {
            status.value: 0
            for status in EvaluationJobStatus
        }

        for job in jobs:
            counts[job.status.value] += 1

        return {
            "worker": "evaluation",
            "running": not self._closed,
            "max_workers": self.max_workers,
            "max_jobs": self.max_jobs,
            "total_jobs": len(jobs),
            "jobs_by_status": counts,
            "custom_evaluators": evaluator_names,
        }

    # --------------------------------------------------------
    # GRACEFUL SHUTDOWN
    # --------------------------------------------------------

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_running: bool = True,
    ) -> None:
        with self._lock:
            self._closed = True

            if cancel_running:
                for job in self._jobs.values():
                    if job.status in {
                        EvaluationJobStatus.QUEUED,
                        EvaluationJobStatus.RUNNING,
                    }:
                        job.cancel_event.set()

        self._executor.shutdown(
            wait=wait,
            cancel_futures=False,
        )


# ============================================================
# FACTORY
# ============================================================

def create_evaluation_worker(
    evaluation_service: (
        EvaluationService | None
    ) = None,
    *,
    max_workers: int = 2,
) -> EvaluationWorker:
    if evaluation_service is None:
        evaluation_service = (
            create_evaluation_service()
        )

    return EvaluationWorker(
        evaluation_service=evaluation_service,
        max_workers=max_workers,
        max_jobs=100,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    worker = create_evaluation_worker()

    request = EvaluationJobRequest(
        evaluation_type=(
            EvaluationJobType.PREDICTION
        ),
        model_name="demo_world_model",
        payload={
            "predictions": [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
            ],
            "targets": [
                [1.1, 1.9, 3.2],
                [2.1, 2.8, 4.1],
                [2.9, 4.2, 4.8],
            ],
        },
        metadata={
            "dataset": "synthetic_demo",
        },
    )

    job_id = worker.submit(request)

    print("=" * 60)
    print("NEUROWORLD AI — EVALUATION WORKER")
    print("=" * 60)

    print("Job ID:", job_id)

    future = worker._futures[job_id]
    future.result()

    result = worker.get_job(
        job_id
    )

    print("Status:", result["status"])
    print("Result:", result["result"])
    print("Duration:", result["duration_seconds"])

    print("\nWORKER STATUS:")
    print(worker.status())

    worker.shutdown()


if __name__ == "__main__":
    main()
