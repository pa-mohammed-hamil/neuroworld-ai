
"""
NeuroWorld AI
backend/app/core/telemetry.py

Telemetry and observability for the NeuroWorld AI platform.

Features:
    - CPU and RAM monitoring
    - NVIDIA GPU monitoring (when available)
    - API request counters and latency
    - Neural model inference metrics
    - Training metrics and progress
    - Imagination rollout metrics
    - World simulation metrics
    - Planning performance metrics
    - Error counters
    - Prometheus-compatible metric exposition
    - FastAPI telemetry middleware
    - Runtime health snapshots

Required:
    pip install fastapi psutil prometheus-client

Optional:
    pip install torch

Usage:
    from backend.app.core.telemetry import (
        setup_telemetry,
        TelemetryMiddleware,
        telemetry_router,
    )

    setup_telemetry()
    app.add_middleware(TelemetryMiddleware)
    app.include_router(telemetry_router)
"""

from __future__ import annotations

import logging
import math
import os
import platform
import threading
import time

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import psutil

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from ..config import get_settings
from .logging import get_logger


logger = get_logger("telemetry")

START_TIME = time.time()

# Dedicated registry avoids collisions with other modules
# that may use the default Prometheus registry.
REGISTRY = CollectorRegistry(auto_describe=True)


# ============================================================
# API METRICS
# ============================================================

HTTP_REQUESTS = Counter(
    "neuroworld_http_requests_total",
    "Total HTTP requests",
    ["method", "route", "status"],
    registry=REGISTRY,
)

HTTP_LATENCY = Histogram(
    "neuroworld_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "route"],
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2.5,
        5,
        10,
        30,
    ),
    registry=REGISTRY,
)

HTTP_IN_PROGRESS = Gauge(
    "neuroworld_http_requests_in_progress",
    "Current HTTP requests in progress",
    registry=REGISTRY,
)


# ============================================================
# INFERENCE METRICS
# ============================================================

MODEL_INFERENCE_TOTAL = Counter(
    "neuroworld_model_inference_total",
    "Total model inference calls",
    ["model", "status"],
    registry=REGISTRY,
)

MODEL_INFERENCE_LATENCY = Histogram(
    "neuroworld_model_inference_duration_seconds",
    "Model inference duration",
    ["model"],
    registry=REGISTRY,
)

MODEL_BATCH_SIZE = Histogram(
    "neuroworld_model_batch_size",
    "Inference batch sizes",
    ["model"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "neuroworld_model_loaded",
    "Whether a model is loaded",
    ["model"],
    registry=REGISTRY,
)


# ============================================================
# TRAINING METRICS
# ============================================================

TRAINING_EPOCH = Gauge(
    "neuroworld_training_epoch",
    "Current training epoch",
    ["model"],
    registry=REGISTRY,
)

TRAINING_LOSS = Gauge(
    "neuroworld_training_loss",
    "Current training loss",
    ["model", "split"],
    registry=REGISTRY,
)

TRAINING_LEARNING_RATE = Gauge(
    "neuroworld_training_learning_rate",
    "Current learning rate",
    ["model"],
    registry=REGISTRY,
)

TRAINING_DURATION = Histogram(
    "neuroworld_training_epoch_duration_seconds",
    "Training epoch duration",
    ["model"],
    registry=REGISTRY,
)


# ============================================================
# WORLD MODEL METRICS
# ============================================================

WORLD_PREDICTIONS = Counter(
    "neuroworld_world_predictions_total",
    "Total world-model predictions",
    ["status"],
    registry=REGISTRY,
)

WORLD_PREDICTION_LATENCY = Histogram(
    "neuroworld_world_prediction_duration_seconds",
    "World-model prediction duration",
    registry=REGISTRY,
)

WORLD_PREDICTION_UNCERTAINTY = Histogram(
    "neuroworld_world_prediction_uncertainty",
    "Reported prediction uncertainty",
    buckets=(
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ),
    registry=REGISTRY,
)


# ============================================================
# IMAGINATION METRICS
# ============================================================

IMAGINATION_ROLLOUTS = Counter(
    "neuroworld_imagination_rollouts_total",
    "Total imagination rollouts",
    ["status"],
    registry=REGISTRY,
)

IMAGINATION_DURATION = Histogram(
    "neuroworld_imagination_duration_seconds",
    "Imagination execution duration",
    registry=REGISTRY,
)

IMAGINATION_BRANCHES = Histogram(
    "neuroworld_imagination_branches",
    "Branches generated per imagination request",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
    registry=REGISTRY,
)

IMAGINATION_HORIZON = Histogram(
    "neuroworld_imagination_horizon",
    "Imagination rollout horizon",
    buckets=(1, 4, 8, 16, 32, 64, 128, 256),
    registry=REGISTRY,
)


# ============================================================
# PLANNING METRICS
# ============================================================

PLANNING_REQUESTS = Counter(
    "neuroworld_planning_requests_total",
    "Planning requests",
    ["algorithm", "status"],
    registry=REGISTRY,
)

PLANNING_DURATION = Histogram(
    "neuroworld_planning_duration_seconds",
    "Planning execution duration",
    ["algorithm"],
    registry=REGISTRY,
)

PLANNING_CANDIDATES = Histogram(
    "neuroworld_planning_candidates",
    "Planning candidates evaluated",
    ["algorithm"],
    registry=REGISTRY,
)


# ============================================================
# SIMULATION METRICS
# ============================================================

SIMULATION_EPISODES = Counter(
    "neuroworld_simulation_episodes_total",
    "Simulation episodes",
    ["environment", "status"],
    registry=REGISTRY,
)

SIMULATION_STEPS = Counter(
    "neuroworld_simulation_steps_total",
    "Simulation steps",
    ["environment"],
    registry=REGISTRY,
)

SIMULATION_DURATION = Histogram(
    "neuroworld_simulation_duration_seconds",
    "Simulation episode duration",
    ["environment"],
    registry=REGISTRY,
)

ACTIVE_SIMULATIONS = Gauge(
    "neuroworld_active_simulations",
    "Number of active simulations",
    registry=REGISTRY,
)


# ============================================================
# ERROR METRICS
# ============================================================

APPLICATION_ERRORS = Counter(
    "neuroworld_application_errors_total",
    "Application errors",
    ["component", "error_type"],
    registry=REGISTRY,
)


# ============================================================
# SYSTEM METRICS
# ============================================================

CPU_USAGE = Gauge(
    "neuroworld_cpu_usage_percent",
    "System CPU usage percentage",
    registry=REGISTRY,
)

MEMORY_USAGE = Gauge(
    "neuroworld_memory_usage_percent",
    "System RAM usage percentage",
    registry=REGISTRY,
)

MEMORY_USED_BYTES = Gauge(
    "neuroworld_memory_used_bytes",
    "System RAM used",
    registry=REGISTRY,
)

PROCESS_MEMORY_BYTES = Gauge(
    "neuroworld_process_memory_bytes",
    "Current application process RSS memory",
    registry=REGISTRY,
)

GPU_MEMORY_ALLOCATED = Gauge(
    "neuroworld_gpu_memory_allocated_bytes",
    "PyTorch CUDA memory allocated",
    registry=REGISTRY,
)

GPU_MEMORY_RESERVED = Gauge(
    "neuroworld_gpu_memory_reserved_bytes",
    "PyTorch CUDA memory reserved",
    registry=REGISTRY,
)

GPU_MEMORY_TOTAL = Gauge(
    "neuroworld_gpu_memory_total_bytes",
    "CUDA device total memory",
    registry=REGISTRY,
)

UPTIME_SECONDS = Gauge(
    "neuroworld_uptime_seconds",
    "Application process uptime",
    registry=REGISTRY,
)


# ============================================================
# METRIC VALIDATION
# ============================================================

def _safe_label(value: str, max_length: int = 80) -> str:
    """
    Restrict label length and characters.

    Never use user IDs, world IDs, request IDs, file names,
    or arbitrary paths as Prometheus labels.
    """

    text = str(value).strip().lower()

    text = "".join(
        char if char.isalnum() or char in "_-" else "_"
        for char in text
    )

    return text[:max_length] or "unknown"


def _finite_nonnegative(
    value: float,
    field_name: str,
) -> float:
    number = float(value)

    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"{field_name} must be finite and nonnegative"
        )

    return number


def _finite_number(
    value: float,
    field_name: str,
) -> float:
    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return number


# ============================================================
# INFERENCE RECORDING
# ============================================================

def record_inference(
    *,
    model_name: str,
    duration_seconds: float,
    batch_size: int = 1,
    success: bool = True,
) -> None:

    model = _safe_label(model_name)

    duration = _finite_nonnegative(
        duration_seconds,
        "duration_seconds",
    )

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    status = "success" if success else "failure"

    MODEL_INFERENCE_TOTAL.labels(
        model=model,
        status=status,
    ).inc()

    MODEL_INFERENCE_LATENCY.labels(
        model=model,
    ).observe(duration)

    MODEL_BATCH_SIZE.labels(
        model=model,
    ).observe(batch_size)


def set_model_loaded(
    model_name: str,
    loaded: bool,
) -> None:
    MODEL_LOADED.labels(
        model=_safe_label(model_name),
    ).set(1 if loaded else 0)


# ============================================================
# TRAINING RECORDING
# ============================================================

def record_training_epoch(
    *,
    model_name: str,
    epoch: int,
    train_loss: float,
    validation_loss: float | None = None,
    learning_rate: float | None = None,
    duration_seconds: float | None = None,
) -> None:

    model = _safe_label(model_name)

    if epoch < 0:
        raise ValueError("epoch cannot be negative")

    TRAINING_EPOCH.labels(
        model=model,
    ).set(epoch)

    TRAINING_LOSS.labels(
        model=model,
        split="train",
    ).set(_finite_number(train_loss, "train_loss"))

    if validation_loss is not None:
        TRAINING_LOSS.labels(
            model=model,
            split="validation",
        ).set(
            _finite_number(
                validation_loss,
                "validation_loss",
            )
        )

    if learning_rate is not None:
        TRAINING_LEARNING_RATE.labels(
            model=model,
        ).set(
            _finite_nonnegative(
                learning_rate,
                "learning_rate",
            )
        )

    if duration_seconds is not None:
        TRAINING_DURATION.labels(
            model=model,
        ).observe(
            _finite_nonnegative(
                duration_seconds,
                "duration_seconds",
            )
        )


# ============================================================
# WORLD MODEL RECORDING
# ============================================================

def record_world_prediction(
    *,
    duration_seconds: float,
    uncertainty: float | None = None,
    success: bool = True,
) -> None:

    WORLD_PREDICTIONS.labels(
        status="success" if success else "failure",
    ).inc()

    WORLD_PREDICTION_LATENCY.observe(
        _finite_nonnegative(
            duration_seconds,
            "duration_seconds",
        )
    )

    if uncertainty is not None:
        value = _finite_number(
            uncertainty,
            "uncertainty",
        )

        if not 0 <= value <= 1:
            raise ValueError(
                "uncertainty must be between 0 and 1"
            )

        WORLD_PREDICTION_UNCERTAINTY.observe(value)


# ============================================================
# IMAGINATION RECORDING
# ============================================================

def record_imagination(
    *,
    duration_seconds: float,
    branches: int,
    horizon: int,
    success: bool = True,
) -> None:

    if branches < 0 or horizon < 0:
        raise ValueError(
            "branches and horizon cannot be negative"
        )

    IMAGINATION_ROLLOUTS.labels(
        status="success" if success else "failure",
    ).inc()

    IMAGINATION_DURATION.observe(
        _finite_nonnegative(
            duration_seconds,
            "duration_seconds",
        )
    )

    IMAGINATION_BRANCHES.observe(branches)
    IMAGINATION_HORIZON.observe(horizon)


# ============================================================
# PLANNING RECORDING
# ============================================================

def record_planning(
    *,
    algorithm: str,
    duration_seconds: float,
    candidate_count: int,
    success: bool = True,
) -> None:

    if candidate_count < 0:
        raise ValueError(
            "candidate_count cannot be negative"
        )

    name = _safe_label(algorithm)

    PLANNING_REQUESTS.labels(
        algorithm=name,
        status="success" if success else "failure",
    ).inc()

    PLANNING_DURATION.labels(
        algorithm=name,
    ).observe(
        _finite_nonnegative(
            duration_seconds,
            "duration_seconds",
        )
    )

    PLANNING_CANDIDATES.labels(
        algorithm=name,
    ).observe(candidate_count)


# ============================================================
# SIMULATION RECORDING
# ============================================================

def record_simulation_episode(
    *,
    environment: str,
    steps: int,
    duration_seconds: float,
    success: bool = True,
) -> None:

    if steps < 0:
        raise ValueError("steps cannot be negative")

    name = _safe_label(environment)

    SIMULATION_EPISODES.labels(
        environment=name,
        status="success" if success else "failure",
    ).inc()

    SIMULATION_STEPS.labels(
        environment=name,
    ).inc(steps)

    SIMULATION_DURATION.labels(
        environment=name,
    ).observe(
        _finite_nonnegative(
            duration_seconds,
            "duration_seconds",
        )
    )


@contextmanager
def track_simulation() -> Iterator[None]:
    ACTIVE_SIMULATIONS.inc()

    try:
        yield
    finally:
        ACTIVE_SIMULATIONS.dec()


# ============================================================
# ERROR RECORDING
# ============================================================

def record_error(
    *,
    component: str,
    error_type: str,
) -> None:

    APPLICATION_ERRORS.labels(
        component=_safe_label(component),
        error_type=_safe_label(error_type),
    ).inc()


# ============================================================
# SYSTEM MONITORING
# ============================================================

def collect_system_metrics() -> dict[str, Any]:
    """
    Collect a system snapshot and update Prometheus gauges.

    CUDA memory values represent allocations managed by
    the current PyTorch process, not all GPU applications.
    """

    cpu_percent = psutil.cpu_percent(interval=None)

    memory = psutil.virtual_memory()

    process = psutil.Process(os.getpid())

    process_memory = process.memory_info().rss

    uptime = max(0.0, time.time() - START_TIME)

    CPU_USAGE.set(cpu_percent)
    MEMORY_USAGE.set(memory.percent)
    MEMORY_USED_BYTES.set(memory.used)
    PROCESS_MEMORY_BYTES.set(process_memory)
    UPTIME_SECONDS.set(uptime)

    result: dict[str, Any] = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "cpu": {
            "percent": cpu_percent,
            "logical_cores": psutil.cpu_count(
                logical=True
            ),
        },
        "memory": {
            "percent": memory.percent,
            "used_bytes": memory.used,
            "total_bytes": memory.total,
            "available_bytes": memory.available,
        },
        "process": {
            "pid": os.getpid(),
            "memory_rss_bytes": process_memory,
            "uptime_seconds": round(uptime, 2),
        },
        "gpu": {
            "available": False,
        },
    }

    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()

            properties = torch.cuda.get_device_properties(
                device_index
            )

            allocated = torch.cuda.memory_allocated(
                device_index
            )

            reserved = torch.cuda.memory_reserved(
                device_index
            )

            total = properties.total_memory

            GPU_MEMORY_ALLOCATED.set(allocated)
            GPU_MEMORY_RESERVED.set(reserved)
            GPU_MEMORY_TOTAL.set(total)

            result["gpu"] = {
                "available": True,
                "backend": "cuda",
                "device": properties.name,
                "memory_allocated_bytes": allocated,
                "memory_reserved_bytes": reserved,
                "memory_total_bytes": total,
            }

        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            result["gpu"] = {
                "available": True,
                "backend": "mps",
                "device": "Apple Metal",
            }

    except ImportError:
        pass

    except Exception as exc:
        logger.warning(
            "GPU metrics collection failed: %s",
            type(exc).__name__,
        )

        result["gpu"] = {
            "available": False,
            "error": "GPU telemetry unavailable",
        }

    return result


# ============================================================
# PERFORMANCE TIMER
# ============================================================

@contextmanager
def measure_duration(
    metric: Histogram,
    *,
    labels: dict[str, str] | None = None,
) -> Iterator[None]:
    """
    Time an operation using a Prometheus Histogram.

    Example:
        with measure_duration(
            MODEL_INFERENCE_LATENCY,
            labels={"model": "world_transformer"},
        ):
            run_inference()
    """

    start = time.perf_counter()

    try:
        yield

    finally:
        elapsed = time.perf_counter() - start

        if labels:
            metric.labels(**labels).observe(elapsed)
        else:
            metric.observe(elapsed)


# ============================================================
# FASTAPI TELEMETRY MIDDLEWARE
# ============================================================

class TelemetryMiddleware:
    """
    Pure ASGI middleware.

    Uses route templates where available to avoid
    high-cardinality URL labels.

    Request bodies and credentials are never collected.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(
        self,
        scope,
        receive,
        send,
    ) -> None:

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        method = scope.get("method", "UNKNOWN")

        status_code = 500

        HTTP_IN_PROGRESS.inc()

        async def send_wrapper(message):
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

            await send(message)

        try:
            await self.app(
                scope,
                receive,
                send_wrapper,
            )

        finally:
            elapsed = time.perf_counter() - start

            route = scope.get("route")

            if route is not None:
                route_path = getattr(
                    route,
                    "path",
                    "unknown",
                )
            else:
                route_path = "unmatched"

            route_path = str(route_path)[:160]

            HTTP_REQUESTS.labels(
                method=method,
                route=route_path,
                status=str(status_code),
            ).inc()

            HTTP_LATENCY.labels(
                method=method,
                route=route_path,
            ).observe(elapsed)

            HTTP_IN_PROGRESS.dec()


# ============================================================
# TELEMETRY API
# ============================================================

telemetry_router = APIRouter(
    prefix="/api/v1/telemetry",
    tags=["Telemetry"],
)


@telemetry_router.get("/system")
async def system_telemetry(request: Request):
    """
    System telemetry is restricted to authenticated
    administrators or an internal monitoring identity.

    This example uses an application-supplied guard.
    Register an appropriate guard in app.state.
    """

    guard = getattr(
        request.app.state,
        "telemetry_authorizer",
        None,
    )

    if guard is None:
        raise HTTPException(
            status_code=503,
            detail="Telemetry authorization is not configured",
        )

    result = guard(request)

    if hasattr(result, "__await__"):
        result = await result

    if result is not True:
        raise HTTPException(
            status_code=403,
            detail="Telemetry access denied",
        )

    return collect_system_metrics()


@telemetry_router.get("/metrics")
async def prometheus_metrics(request: Request):
    guard = getattr(
        request.app.state,
        "telemetry_authorizer",
        None,
    )

    if guard is None:
        raise HTTPException(
            status_code=503,
            detail="Telemetry authorization is not configured",
        )

    result = guard(request)

    if hasattr(result, "__await__"):
        result = await result

    if result is not True:
        raise HTTPException(
            status_code=403,
            detail="Telemetry access denied",
        )

    collect_system_metrics()

    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


# ============================================================
# SETUP
# ============================================================

def setup_telemetry() -> None:
    logger.info(
        "NeuroWorld AI telemetry initialized"
    )


# ============================================================
# SELF-TEST
# ============================================================

def main() -> None:
    import json

    setup_telemetry()

    print("=" * 65)
    print("NEUROWORLD AI — TELEMETRY")
    print("=" * 65)

    print(
        json.dumps(
            collect_system_metrics(),
            indent=2,
        )
    )

    # Demo measurements, not actual model execution.
    record_inference(
        model_name="demo_world_model",
        duration_seconds=0.024,
        batch_size=2,
    )

    record_training_epoch(
        model_name="demo_world_model",
        epoch=1,
        train_loss=0.42,
        validation_loss=0.48,
        learning_rate=0.0001,
        duration_seconds=2.4,
    )

    record_world_prediction(
        duration_seconds=0.018,
        uncertainty=0.12,
    )

    record_imagination(
        duration_seconds=0.13,
        branches=4,
        horizon=16,
    )

    record_planning(
        algorithm="cem",
        duration_seconds=0.21,
        candidate_count=128,
    )

    record_simulation_episode(
        environment="synthetic_physics",
        steps=100,
        duration_seconds=1.5,
    )

    print()
    print("Prometheus metrics preview:")

    output = generate_latest(
        REGISTRY
    ).decode("utf-8")

    print(output[:3000])

    print()
    print("Telemetry self-test completed.")


if __name__ == "__main__":
    main()
