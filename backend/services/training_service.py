
"""
NeuroWorld AI
backend/app/services/training_service.py

PyTorch training orchestration service.

Features:
- Model training and validation
- CPU / CUDA / MPS support
- Mixed precision on CUDA
- Gradient clipping
- Early stopping
- Learning-rate scheduling
- Checkpoint saving and restoration
- Training job lifecycle
- Cancellation between batches
- Epoch metrics and progress tracking
- Bounded concurrent training jobs
- Thread-safe job registry

Important:
This service trains registered PyTorch models using
trusted, explicitly provided datasets.

It does not automatically train a world model unless
an appropriate architecture and dataset are supplied.
"""

from __future__ import annotations

import copy
import math
import os
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
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# EXCEPTIONS
# ============================================================

class TrainingServiceError(Exception):
    pass


class TrainingValidationError(TrainingServiceError):
    pass


class TrainingJobNotFound(TrainingServiceError):
    pass


class TrainingCapacityError(TrainingServiceError):
    pass


class TrainingCancelled(TrainingServiceError):
    pass


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    device: str = "auto"
    mixed_precision: bool = True
    checkpoint_every: int = 5
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self):
        if not self.model_name.strip():
            raise TrainingValidationError(
                "model_name is required"
            )

        if not 1 <= self.epochs <= 10000:
            raise TrainingValidationError(
                "epochs must be between 1 and 10000"
            )

        if not 1 <= self.batch_size <= 65536:
            raise TrainingValidationError(
                "Invalid batch size"
            )

        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise TrainingValidationError(
                "learning_rate must be positive"
            )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise TrainingValidationError(
                "weight_decay must be nonnegative"
            )

        if (
            not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise TrainingValidationError(
                "gradient_clip_norm must be positive"
            )

        if self.early_stopping_patience < 0:
            raise TrainingValidationError(
                "early_stopping_patience cannot be negative"
            )

        if self.checkpoint_every < 0:
            raise TrainingValidationError(
                "checkpoint_every cannot be negative"
            )

        if self.num_workers < 0:
            raise TrainingValidationError(
                "num_workers cannot be negative"
            )


# ============================================================
# JOB MODELS
# ============================================================

class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    EARLY_STOPPED = "early_stopped"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float
    duration_seconds: float


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: TrainingStatus
    created_at: datetime

    started_at: datetime | None = None
    finished_at: datetime | None = None

    current_epoch: int = 0
    total_epochs: int = 0
    progress_percent: float = 0.0

    best_loss: float | None = None
    best_epoch: int | None = None

    history: list[EpochMetrics] = field(
        default_factory=list
    )

    checkpoint_path: str | None = None
    error: str | None = None

    cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


# ============================================================
# DEVICE MANAGEMENT
# ============================================================

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise TrainingValidationError(
                "CUDA is unavailable"
            )

    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise TrainingValidationError(
                "MPS is unavailable"
            )

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise TrainingValidationError(
            "Invalid training device"
        ) from exc

    if device.type not in {"cpu", "cuda", "mps"}:
        raise TrainingValidationError(
            "Unsupported training device"
        )

    return device


# ============================================================
# MODEL / DATA REGISTRATION
# ============================================================

class TrainingDatasetProvider(Protocol):
    def __call__(
        self,
    ) -> tuple[Dataset, Dataset | None]:
        ...


@dataclass(frozen=True)
class TrainingRegistration:
    model_factory: Callable[[], nn.Module]
    dataset_provider: TrainingDatasetProvider
    loss_factory: Callable[[], nn.Module]


# ============================================================
# TRAINING SERVICE
# ============================================================

class TrainingService:
    """
    Asynchronous job orchestration using a bounded
    thread pool.

    Registered factories are trusted application code.
    Do not allow arbitrary client-supplied Python
    classes, filesystem paths, or pickle payloads.

    Production multi-worker deployments should replace
    this process-local executor with a persistent queue.
    """

    def __init__(
        self,
        checkpoint_directory: str | Path = (
            "models/checkpoints"
        ),
        *,
        max_concurrent_jobs: int = 1,
        max_jobs: int = 100,
    ):
        if max_concurrent_jobs < 1 or max_jobs < 1:
            raise ValueError(
                "Job limits must be positive"
            )

        self.checkpoint_directory = Path(
            checkpoint_directory
        ).resolve()

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.max_concurrent_jobs = (
            max_concurrent_jobs
        )
        self.max_jobs = max_jobs

        self._registrations: dict[
            str,
            TrainingRegistration,
        ] = {}

        self._jobs: dict[str, TrainingJob] = {}
        self._futures: dict[str, Future] = {}

        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_jobs,
            thread_name_prefix="neuroworld-training",
        )

        self._lock = threading.RLock()
        self._closed = False

    def register_model(
        self,
        name: str,
        *,
        model_factory: Callable[[], nn.Module],
        dataset_provider: TrainingDatasetProvider,
        loss_factory: Callable[[], nn.Module] = (
            nn.MSELoss
        ),
    ) -> None:
        if not name.strip():
            raise TrainingValidationError(
                "Model name cannot be empty"
            )

        if not all(
            callable(item)
            for item in (
                model_factory,
                dataset_provider,
                loss_factory,
            )
        ):
            raise TrainingValidationError(
                "Training factories must be callable"
            )

        with self._lock:
            self._registrations[name] = (
                TrainingRegistration(
                    model_factory=model_factory,
                    dataset_provider=dataset_provider,
                    loss_factory=loss_factory,
                )
            )

    def list_models(self) -> list[str]:
        with self._lock:
            return sorted(self._registrations)

    def start_training(
        self,
        config: TrainingConfig,
    ) -> str:
        with self._lock:
            if self._closed:
                raise TrainingServiceError(
                    "Training service is closed"
                )

            registration = self._registrations.get(
                config.model_name
            )

            if registration is None:
                raise TrainingValidationError(
                    "Model is not registered"
                )

            active = sum(
                job.status in {
                    TrainingStatus.QUEUED,
                    TrainingStatus.RUNNING,
                }
                for job in self._jobs.values()
            )

            if active >= self.max_concurrent_jobs:
                raise TrainingCapacityError(
                    "Training capacity reached"
                )

            if len(self._jobs) >= self.max_jobs:
                # Remove only completed jobs.
                removable = [
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status not in {
                        TrainingStatus.QUEUED,
                        TrainingStatus.RUNNING,
                    }
                ]

                if not removable:
                    raise TrainingCapacityError(
                        "Training history is full"
                    )

                oldest_id = removable[0]
                self._jobs.pop(oldest_id)
                self._futures.pop(oldest_id, None)

            job_id = uuid.uuid4().hex

            job = TrainingJob(
                job_id=job_id,
                model_name=config.model_name,
                status=TrainingStatus.QUEUED,
                created_at=datetime.now(
                    timezone.utc
                ),
                total_epochs=config.epochs,
            )

            self._jobs[job_id] = job

            future = self._executor.submit(
                self._execute_training,
                job,
                config,
                registration,
            )

            self._futures[job_id] = future

        return job_id

    def _make_loader(
        self,
        dataset: Dataset,
        config: TrainingConfig,
        *,
        shuffle: bool,
    ) -> DataLoader:
        if len(dataset) == 0:
            raise TrainingValidationError(
                "Training dataset cannot be empty"
            )

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=False,
        )

    @staticmethod
    def _unpack_batch(
        batch: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            isinstance(batch, (tuple, list))
            and len(batch) == 2
        ):
            raise TrainingValidationError(
                "Each batch must contain (inputs, targets)"
            )

        inputs, targets = batch

        if not (
            isinstance(inputs, torch.Tensor)
            and isinstance(targets, torch.Tensor)
        ):
            raise TrainingValidationError(
                "Inputs and targets must be tensors"
            )

        return (
            inputs.to(device),
            targets.to(device),
        )

    def _run_epoch(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
        device: torch.device,
        config: TrainingConfig,
        job: TrainingJob,
    ) -> float:
        training = optimizer is not None

        model.train(training)

        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            if job.cancel_event.is_set():
                raise TrainingCancelled(
                    "Training cancellation requested"
                )

            inputs, targets = self._unpack_batch(
                batch,
                device,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=device.type,
                    enabled=(
                        config.mixed_precision
                        and device.type == "cuda"
                    ),
                ):
                    predictions = model(inputs)
                    loss = criterion(
                        predictions,
                        targets,
                    )

                if loss.ndim != 0:
                    raise TrainingValidationError(
                        "Loss must be a scalar"
                    )

                if not torch.isfinite(loss):
                    raise TrainingServiceError(
                        "Non-finite training loss"
                    )

                if training:
                    if scaler is not None:
                        scaler.scale(loss).backward()

                        scaler.unscale_(optimizer)

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.gradient_clip_norm,
                        )

                        scaler.step(optimizer)
                        scaler.update()

                    else:
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config.gradient_clip_norm,
                        )

                        optimizer.step()

            batch_size = inputs.shape[0]

            total_loss += (
                float(loss.detach().item())
                * batch_size
            )

            total_samples += batch_size

        if total_samples == 0:
            raise TrainingValidationError(
                "No samples processed"
            )

        return total_loss / total_samples

    def _save_checkpoint(
        self,
        *,
        job: TrainingJob,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        config: TrainingConfig,
        suffix: str,
    ) -> Path:
        # Filenames use server-generated UUIDs only.
        path = self.checkpoint_directory / (
            f"{job.job_id}_{suffix}.pt"
        )

        temporary = path.with_suffix(".tmp")

        checkpoint = {
            "format_version": 1,
            "job_id": job.job_id,
            "model_name": config.model_name,
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        }

        torch.save(
            checkpoint,
            temporary,
        )

        os.replace(
            temporary,
            path,
        )

        return path

    def _execute_training(
        self,
        job: TrainingJob,
        config: TrainingConfig,
        registration: TrainingRegistration,
    ) -> None:
        try:
            if job.cancel_event.is_set():
                raise TrainingCancelled(
                    "Cancelled before training"
                )

            with job.lock:
                job.status = TrainingStatus.RUNNING
                job.started_at = datetime.now(
                    timezone.utc
                )

            # Set seeds for repeatability.
            # This does not guarantee deterministic
            # execution across all devices/operators.
            torch.manual_seed(config.seed)

            device = resolve_device(
                config.device
            )

            model = registration.model_factory()

            if not isinstance(model, nn.Module):
                raise TrainingValidationError(
                    "Model factory must return nn.Module"
                )

            model = model.to(device)

            train_dataset, val_dataset = (
                registration.dataset_provider()
            )

            train_loader = self._make_loader(
                train_dataset,
                config,
                shuffle=True,
            )

            val_loader = (
                self._make_loader(
                    val_dataset,
                    config,
                    shuffle=False,
                )
                if val_dataset is not None
                else None
            )

            criterion = (
                registration.loss_factory()
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=2,
                )
            )

            scaler = (
                torch.amp.GradScaler("cuda")
                if (
                    config.mixed_precision
                    and device.type == "cuda"
                )
                else None
            )

            best_loss = float("inf")
            best_epoch = None
            patience_counter = 0

            for epoch in range(
                1,
                config.epochs + 1,
            ):
                if job.cancel_event.is_set():
                    raise TrainingCancelled(
                        "Training cancelled"
                    )

                started = time.perf_counter()

                train_loss = self._run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    config=config,
                    job=job,
                )

                validation_loss = None

                if val_loader is not None:
                    validation_loss = self._run_epoch(
                        model=model,
                        loader=val_loader,
                        criterion=criterion,
                        optimizer=None,
                        scaler=None,
                        device=device,
                        config=config,
                        job=job,
                    )

                monitored_loss = (
                    validation_loss
                    if validation_loss is not None
                    else train_loss
                )

                scheduler.step(
                    monitored_loss
                )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                metrics = EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    learning_rate=current_lr,
                    duration_seconds=(
                        time.perf_counter() - started
                    ),
                )

                with job.lock:
                    job.history.append(metrics)
                    job.current_epoch = epoch
                    job.progress_percent = (
                        epoch / config.epochs
                    ) * 100

                if (
                    monitored_loss
                    < best_loss - config.min_delta
                ):
                    best_loss = monitored_loss
                    best_epoch = epoch
                    patience_counter = 0

                    checkpoint_path = (
                        self._save_checkpoint(
                            job=job,
                            model=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            loss=monitored_loss,
                            config=config,
                            suffix="best",
                        )
                    )

                    with job.lock:
                        job.best_loss = best_loss
                        job.best_epoch = best_epoch
                        job.checkpoint_path = str(
                            checkpoint_path
                        )

                else:
                    patience_counter += 1

                if (
                    config.checkpoint_every > 0
                    and epoch
                    % config.checkpoint_every == 0
                ):
                    self._save_checkpoint(
                        job=job,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        loss=monitored_loss,
                        config=config,
                        suffix="latest",
                    )

                if (
                    config.early_stopping_patience > 0
                    and patience_counter
                    >= config.early_stopping_patience
                ):
                    with job.lock:
                        job.status = (
                            TrainingStatus.EARLY_STOPPED
                        )
                    break

            with job.lock:
                if job.status == TrainingStatus.RUNNING:
                    job.status = TrainingStatus.COMPLETED

        except TrainingCancelled:
            with job.lock:
                job.status = TrainingStatus.CANCELLED

        except Exception as exc:
            with job.lock:
                job.status = TrainingStatus.FAILED
                job.error = (
                    f"{type(exc).__name__}: {exc}"
                )

        finally:
            with job.lock:
                job.finished_at = datetime.now(
                    timezone.utc
                )

    def cancel_training(
        self,
        job_id: str,
    ) -> bool:
        job = self._get_job(job_id)

        with job.lock:
            if job.status not in {
                TrainingStatus.QUEUED,
                TrainingStatus.RUNNING,
            }:
                return False

            job.cancel_event.set()

        return True

    def _get_job(
        self,
        job_id: str,
    ) -> TrainingJob:
        with self._lock:
            job = self._jobs.get(job_id)

        if job is None:
            raise TrainingJobNotFound(
                "Training job not found"
            )

        return job

    def get_training_status(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        job = self._get_job(job_id)

        with job.lock:
            return {
                "job_id": job.job_id,
                "model_name": job.model_name,
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
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "progress_percent": (
                    job.progress_percent
                ),
                "best_loss": job.best_loss,
                "best_epoch": job.best_epoch,
                "checkpoint_path": (
                    job.checkpoint_path
                ),
                "error": job.error,
                "history": [
                    {
                        "epoch": item.epoch,
                        "train_loss": item.train_loss,
                        "validation_loss": (
                            item.validation_loss
                        ),
                        "learning_rate": (
                            item.learning_rate
                        ),
                        "duration_seconds": (
                            item.duration_seconds
                        ),
                    }
                    for item in job.history
                ],
            }

    def list_training_jobs(
        self,
    ) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._jobs)

        return [
            self.get_training_status(job_id)
            for job_id in ids
        ]

    def load_checkpoint(
        self,
        *,
        job_id: str,
        model: nn.Module,
        device: str = "cpu",
    ) -> nn.Module:
        """
        Restore model weights from a checkpoint
        created by this service.

        Checkpoint loading must be restricted to
        trusted server-generated artifacts.
        """
        job = self._get_job(job_id)

        with job.lock:
            checkpoint_path = job.checkpoint_path

        if checkpoint_path is None:
            raise TrainingServiceError(
                "No checkpoint available"
            )

        target_device = resolve_device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=target_device,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(target_device)
        model.eval()

        return model

    def status(self) -> dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
            models = sorted(
                self._registrations
            )

        return {
            "service": "training",
            "registered_models": models,
            "total_jobs": len(jobs),
            "active_jobs": sum(
                job.status in {
                    TrainingStatus.QUEUED,
                    TrainingStatus.RUNNING,
                }
                for job in jobs
            ),
            "max_concurrent_jobs": (
                self.max_concurrent_jobs
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "checkpoint_directory": str(
                self.checkpoint_directory
            ),
        }

    def shutdown(
        self,
        *,
        wait: bool = True,
    ) -> None:
        with self._lock:
            self._closed = True

            for job in self._jobs.values():
                if job.status in {
                    TrainingStatus.QUEUED,
                    TrainingStatus.RUNNING,
                }:
                    job.cancel_event.set()

        self._executor.shutdown(
            wait=wait,
            cancel_futures=False,
        )


# ============================================================
# SERVICE FACTORY
# ============================================================

def create_training_service(
    checkpoint_directory: str | Path = (
        "models/checkpoints"
    ),
) -> TrainingService:
    return TrainingService(
        checkpoint_directory=checkpoint_directory,
        max_concurrent_jobs=1,
        max_jobs=100,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from torch.utils.data import TensorDataset

    def model_factory() -> nn.Module:
        return nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def dataset_provider():
        generator = torch.Generator().manual_seed(42)

        inputs = torch.randn(
            512,
            4,
            generator=generator,
        )

        targets = inputs * 0.5 + 0.2

        train = TensorDataset(
            inputs[:400],
            targets[:400],
        )

        validation = TensorDataset(
            inputs[400:],
            targets[400:],
        )

        return train, validation

    service = create_training_service()

    service.register_model(
        "demo_dynamics_model",
        model_factory=model_factory,
        dataset_provider=dataset_provider,
        loss_factory=nn.MSELoss,
    )

    config = TrainingConfig(
        model_name="demo_dynamics_model",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        device="auto",
        seed=42,
    )

    job_id = service.start_training(config)

    print("NeuroWorld AI — Training Service")
    print("Training job:", job_id)

    future = service._futures[job_id]
    future.result()

    result = service.get_training_status(
        job_id
    )

    print("Status:", result["status"])
    print("Best loss:", result["best_loss"])
    print("Best epoch:", result["best_epoch"])
    print(
        "Checkpoint:",
        result["checkpoint_path"],
    )

    service.shutdown()


if __name__ == "__main__":
    main()
