
"""
NeuroWorld AI
File: training/utils/checkpoint_io.py

Production-oriented checkpoint management for
PyTorch world-model and reinforcement-learning training.

Features:
- Model checkpoint save/load
- Optimizer and scheduler state persistence
- AMP GradScaler state persistence
- Epoch and global-step tracking
- Best and latest checkpoint management
- Atomic checkpoint writes
- Checkpoint retention policies
- Resume training
- Multiple-model checkpoint support
- RNG state save/restore
- Checkpoint metadata and validation
- Model state-key normalization
- DDP-compatible checkpoint handling
- Optional SHA-256 integrity verification

Security:
    Only load checkpoints from trusted sources.
    weights_only=True is used when loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_VERSION = "1.0"


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"

    prefix: str = "neuroworld"

    save_every_n_epochs: int = 1
    keep_last_n: int = 5

    save_optimizer: bool = True
    save_scheduler: bool = True
    save_scaler: bool = True
    save_rng_state: bool = True

    save_best: bool = True
    save_latest: bool = True

    monitor: str = "val_loss"
    mode: str = "min"

    atomic_save: bool = True
    verify_checksum: bool = False

    strict_model_loading: bool = True
    restore_rng_state: bool = False

    def __post_init__(self) -> None:
        if self.save_every_n_epochs < 1:
            raise ValueError(
                "save_every_n_epochs must be >= 1"
            )

        if self.keep_last_n < 0:
            raise ValueError(
                "keep_last_n must be >= 0"
            )

        if self.mode not in {"min", "max"}:
            raise ValueError(
                "mode must be 'min' or 'max'"
            )

        if not self.prefix:
            raise ValueError(
                "prefix cannot be empty"
            )


@dataclass
class CheckpointMetadata:
    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    monitor: str = "val_loss"
    version: str = CHECKPOINT_VERSION
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unwrap_model(
    model: nn.Module,
) -> nn.Module:
    """
    Unwrap common model wrappers such as
    DistributedDataParallel and DataParallel.
    """
    while isinstance(
        model,
        (
            nn.DataParallel,
            nn.parallel.DistributedDataParallel,
        ),
    ):
        model = model.module

    return model


def normalize_state_dict_keys(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Remove a leading 'module.' prefix
    commonly introduced by DDP.
    """
    normalized = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]

        if key in normalized:
            raise ValueError(
                f"Duplicate normalized key: {key}"
            )

        normalized[key] = value

    return normalized


def calculate_file_sha256(
    path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _atomic_torch_save(
    data: Mapping[str, Any],
    path: Path,
) -> None:
    """
    Save to a temporary file and atomically
    replace the destination.

    The temporary file is created in the
    destination directory to support
    same-filesystem replacement.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )

    os.close(descriptor)

    try:
        torch.save(
            dict(data),
            temporary_path,
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _write_json_atomic(
    data: Mapping[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )

    os.close(descriptor)

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                dict(data),
                file,
                indent=2,
                allow_nan=False,
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def capture_rng_state() -> dict[str, Any]:
    """
    Capture random generator states.

    NumPy state is converted to a dictionary
    of safe primitives and tensors, compatible
    with torch.load(weights_only=True).
    """
    numpy_state = np.random.get_state()

    state: dict[str, Any] = {
        "python": list(
            random.getstate()[1]
        ),
        "python_version": int(
            random.getstate()[0]
        ),
        "python_gauss": (
            random.getstate()[2]
        ),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(
                numpy_state[1].copy()
            ),
            "position": int(
                numpy_state[2]
            ),
            "has_gauss": int(
                numpy_state[3]
            ),
            "cached_gaussian": float(
                numpy_state[4]
            ),
        },
        "torch_cpu": (
            torch.get_rng_state()
        ),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = [
            tensor.cpu()
            for tensor in (
                torch.cuda.get_rng_state_all()
            )
        ]

    return state


def restore_rng_state(
    state: Mapping[str, Any],
) -> None:
    if "python" in state:
        random.setstate(
            (
                int(
                    state.get(
                        "python_version",
                        3,
                    )
                ),
                tuple(
                    state["python"]
                ),
                state.get(
                    "python_gauss"
                ),
            )
        )

    if "numpy" in state:
        numpy_state = state["numpy"]

        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"]
                .cpu()
                .numpy(),
                int(
                    numpy_state["position"]
                ),
                int(
                    numpy_state["has_gauss"]
                ),
                float(
                    numpy_state["cached_gaussian"]
                ),
            )
        )

    if "torch_cpu" in state:
        torch.set_rng_state(
            state["torch_cpu"].cpu()
        )

    if (
        "torch_cuda" in state
        and torch.cuda.is_available()
    ):
        cuda_states = state["torch_cuda"]

        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(
                cuda_states
            )


def _normalize_models(
    models: (
        nn.Module
        | Mapping[str, nn.Module]
    ),
) -> dict[str, nn.Module]:
    if isinstance(models, nn.Module):
        return {
            "model": models
        }

    if not models:
        raise ValueError(
            "models cannot be empty"
        )

    for name, model in models.items():
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"{name} is not an nn.Module"
            )

    return dict(models)


def _normalize_optimizers(
    optimizers: (
        Optimizer
        | Mapping[str, Optimizer]
        | None
    ),
) -> dict[str, Optimizer]:
    if optimizers is None:
        return {}

    if isinstance(optimizers, Optimizer):
        return {
            "optimizer": optimizers
        }

    return dict(optimizers)


def _normalize_stateful_objects(
    objects: Any,
    default_name: str,
) -> dict[str, Any]:
    if objects is None:
        return {}

    if isinstance(objects, Mapping):
        return dict(objects)

    return {
        default_name: objects
    }


def build_checkpoint(
    *,
    models: (
        nn.Module
        | Mapping[str, nn.Module]
    ),
    optimizers: (
        Optimizer
        | Mapping[str, Optimizer]
        | None
    ) = None,
    schedulers: Any = None,
    scaler: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    monitor: str = "val_loss",
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    include_rng_state: bool = True,
) -> dict[str, Any]:
    """
    Build a serializable training checkpoint.
    """
    model_map = _normalize_models(
        models
    )

    optimizer_map = _normalize_optimizers(
        optimizers
    )

    scheduler_map = _normalize_stateful_objects(
        schedulers,
        "scheduler",
    )

    metadata = CheckpointMetadata(
        epoch=int(epoch),
        global_step=int(global_step),
        best_metric=(
            None
            if best_metric is None
            else float(best_metric)
        ),
        monitor=monitor,
        timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
    )

    checkpoint: dict[str, Any] = {
        "metadata": metadata.to_dict(),
        "models": {
            name: unwrap_model(
                model
            ).state_dict()
            for name, model in model_map.items()
        },
        "optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in optimizer_map.items()
        },
        "schedulers": {
            name: scheduler.state_dict()
            for name, scheduler in scheduler_map.items()
        },
        "metrics": dict(metrics or {}),
        "extra": dict(extra or {}),
    }

    if scaler is not None:
        checkpoint["scaler"] = (
            scaler.state_dict()
        )

    if include_rng_state:
        checkpoint["rng_state"] = (
            capture_rng_state()
        )

    return checkpoint


def save_checkpoint(
    path: str | Path,
    *,
    models: (
        nn.Module
        | Mapping[str, nn.Module]
    ),
    optimizers: (
        Optimizer
        | Mapping[str, Optimizer]
        | None
    ) = None,
    schedulers: Any = None,
    scaler: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    monitor: str = "val_loss",
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    include_rng_state: bool = True,
    atomic: bool = True,
    write_checksum: bool = False,
) -> Path:
    """
    Save a complete training checkpoint.
    """
    path = Path(path)

    checkpoint = build_checkpoint(
        models=models,
        optimizers=optimizers,
        schedulers=schedulers,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        monitor=monitor,
        metrics=metrics,
        extra=extra,
        include_rng_state=include_rng_state,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if atomic:
        _atomic_torch_save(
            checkpoint,
            path,
        )
    else:
        torch.save(
            checkpoint,
            path,
        )

    if write_checksum:
        checksum = calculate_file_sha256(
            path
        )

        _write_json_atomic(
            {
                "algorithm": "sha256",
                "sha256": checksum,
            },
            path.with_suffix(
                path.suffix + ".sha256.json"
            ),
        )

    return path


def verify_checkpoint_checksum(
    path: str | Path,
) -> bool:
    path = Path(path)

    checksum_path = path.with_suffix(
        path.suffix + ".sha256.json"
    )

    if not checksum_path.exists():
        raise FileNotFoundError(
            f"Checksum not found: {checksum_path}"
        )

    with open(
        checksum_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    expected = metadata["sha256"]

    actual = calculate_file_sha256(
        path
    )

    return expected == actual


def read_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    verify_checksum: bool = False,
) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    if verify_checksum:
        if not verify_checkpoint_checksum(
            path
        ):
            raise ValueError(
                "Checkpoint checksum mismatch"
            )

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be a dictionary"
        )

    if "models" not in checkpoint:
        raise KeyError(
            "Checkpoint has no model states"
        )

    return checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    models: (
        nn.Module
        | Mapping[str, nn.Module]
    ),
    optimizers: (
        Optimizer
        | Mapping[str, Optimizer]
        | None
    ) = None,
    schedulers: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_scaler: bool = True,
    load_rng_state: bool = False,
    verify_checksum: bool = False,
) -> dict[str, Any]:
    """
    Restore a training checkpoint.

    Models are restored before optimizers.

    For exact resume:
        Restore model, optimizer, scheduler,
        scaler and RNG states.
    """
    checkpoint = read_checkpoint(
        path,
        map_location=map_location,
        verify_checksum=verify_checksum,
    )

    model_map = _normalize_models(
        models
    )

    optimizer_map = _normalize_optimizers(
        optimizers
    )

    scheduler_map = _normalize_stateful_objects(
        schedulers,
        "scheduler",
    )

    model_results = {}

    for name, model in model_map.items():
        if name not in checkpoint["models"]:
            raise KeyError(
                f"Missing model state: {name}"
            )

        state = normalize_state_dict_keys(
            checkpoint["models"][name]
        )

        result = unwrap_model(
            model
        ).load_state_dict(
            state,
            strict=strict,
        )

        model_results[name] = {
            "missing_keys": list(
                result.missing_keys
            ),
            "unexpected_keys": list(
                result.unexpected_keys
            ),
        }

    if load_optimizer:
        saved_optimizers = checkpoint.get(
            "optimizers",
            {},
        )

        for name, optimizer in optimizer_map.items():
            if name not in saved_optimizers:
                raise KeyError(
                    f"Missing optimizer state: {name}"
                )

            optimizer.load_state_dict(
                saved_optimizers[name]
            )

    if load_scheduler:
        saved_schedulers = checkpoint.get(
            "schedulers",
            {},
        )

        for name, scheduler in scheduler_map.items():
            if name not in saved_schedulers:
                raise KeyError(
                    f"Missing scheduler state: {name}"
                )

            scheduler.load_state_dict(
                saved_schedulers[name]
            )

    if (
        load_scaler
        and scaler is not None
        and "scaler" in checkpoint
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    if (
        load_rng_state
        and "rng_state" in checkpoint
    ):
        restore_rng_state(
            checkpoint["rng_state"]
        )

    return {
        "metadata": checkpoint.get(
            "metadata",
            {},
        ),
        "metrics": checkpoint.get(
            "metrics",
            {},
        ),
        "extra": checkpoint.get(
            "extra",
            {},
        ),
        "model_results": model_results,
    }


def save_model_weights(
    model: nn.Module,
    path: str | Path,
    *,
    atomic: bool = True,
) -> Path:
    """
    Save model weights only.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = unwrap_model(
        model
    ).state_dict()

    if atomic:
        _atomic_torch_save(
            state,
            path,
        )
    else:
        torch.save(
            state,
            path,
        )

    return path


def load_model_weights(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, list[str]]:
    """
    Load a model-only state dictionary.
    """
    state = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    state = normalize_state_dict_keys(
        state
    )

    result = unwrap_model(
        model
    ).load_state_dict(
        state,
        strict=strict,
    )

    return {
        "missing_keys": list(
            result.missing_keys
        ),
        "unexpected_keys": list(
            result.unexpected_keys
        ),
    }


def get_checkpoint_metadata(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = read_checkpoint(
        path,
        map_location=map_location,
    )

    return dict(
        checkpoint.get(
            "metadata",
            {},
        )
    )


class CheckpointManager:
    """
    Training checkpoint manager.

    Supports:
        - Periodic epoch checkpoints
        - Best checkpoint
        - Latest checkpoint
        - Retention cleanup
        - Training resume
        - Multi-model training

    Example:
        manager = CheckpointManager(
            CheckpointConfig(
                directory="checkpoints/world_model",
                monitor="val_loss",
                mode="min",
            )
        )

        manager.save(
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
            epoch=epoch,
            global_step=step,
            metrics={"val_loss": val_loss},
        )
    """

    def __init__(
        self,
        config: (
            CheckpointConfig
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> None:
        if config is None:
            config = CheckpointConfig()

        elif isinstance(config, Mapping):
            config = CheckpointConfig(
                **dict(config)
            )

        self.config = config

        self.directory = Path(
            config.directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_metric: float | None = None
        self.best_epoch: int | None = None

        self.latest_path = (
            self.directory
            / f"{config.prefix}_latest.pt"
        )

        self.best_path = (
            self.directory
            / f"{config.prefix}_best.pt"
        )

    def _is_better(
        self,
        metric: float,
    ) -> bool:
        if not math.isfinite(metric):
            return False

        if self.best_metric is None:
            return True

        if self.config.mode == "min":
            return metric < self.best_metric

        return metric > self.best_metric

    def _epoch_path(
        self,
        epoch: int,
    ) -> Path:
        return (
            self.directory
            / (
                f"{self.config.prefix}"
                f"_epoch_{epoch:06d}.pt"
            )
        )

    def _save_to_path(
        self,
        path: Path,
        *,
        models: (
            nn.Module
            | Mapping[str, nn.Module]
        ),
        optimizers: (
            Optimizer
            | Mapping[str, Optimizer]
            | None
        ),
        schedulers: Any,
        scaler: Any,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, Any],
        extra: Mapping[str, Any] | None,
    ) -> Path:
        return save_checkpoint(
            path,
            models=models,
            optimizers=(
                optimizers
                if self.config.save_optimizer
                else None
            ),
            schedulers=(
                schedulers
                if self.config.save_scheduler
                else None
            ),
            scaler=(
                scaler
                if self.config.save_scaler
                else None
            ),
            epoch=epoch,
            global_step=global_step,
            best_metric=self.best_metric,
            monitor=self.config.monitor,
            metrics=metrics,
            extra=extra,
            include_rng_state=(
                self.config.save_rng_state
            ),
            atomic=self.config.atomic_save,
            write_checksum=(
                self.config.verify_checksum
            ),
        )

    def save(
        self,
        *,
        models: (
            nn.Module
            | Mapping[str, nn.Module]
        ),
        optimizers: (
            Optimizer
            | Mapping[str, Optimizer]
            | None
        ) = None,
        schedulers: Any = None,
        scaler: Any = None,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, str]:
        """
        Save periodic, latest and best checkpoints.

        Returns:
            Dictionary of saved checkpoint paths.
        """
        metrics = dict(
            metrics or {}
        )

        saved_paths = {}

        metric = metrics.get(
            self.config.monitor
        )

        is_best = False

        if metric is not None:
            if isinstance(metric, torch.Tensor):
                if metric.numel() != 1:
                    raise ValueError(
                        "Monitored metric must be scalar"
                    )

                metric = float(
                    metric.detach().item()
                )

            metric = float(metric)

            if self._is_better(metric):
                self.best_metric = metric
                self.best_epoch = epoch
                is_best = True

        should_save_epoch = (
            force
            or epoch
            % self.config.save_every_n_epochs
            == 0
        )

        if should_save_epoch:
            epoch_path = self._epoch_path(
                epoch
            )

            self._save_to_path(
                epoch_path,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                extra=extra,
            )

            saved_paths["epoch"] = str(
                epoch_path
            )

            self._cleanup_old_checkpoints()

        if self.config.save_latest:
            self._save_to_path(
                self.latest_path,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                extra=extra,
            )

            saved_paths["latest"] = str(
                self.latest_path
            )

        if (
            self.config.save_best
            and is_best
        ):
            self._save_to_path(
                self.best_path,
                models=models,
                optimizers=optimizers,
                schedulers=schedulers,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                extra=extra,
            )

            saved_paths["best"] = str(
                self.best_path
            )

        return saved_paths

    def _cleanup_old_checkpoints(
        self,
    ) -> None:
        """
        Retain the newest N periodic
        epoch checkpoints.

        Best and latest checkpoints are
        not included in retention cleanup.
        """
        keep = self.config.keep_last_n

        checkpoints = sorted(
            self.directory.glob(
                f"{self.config.prefix}"
                "_epoch_*.pt"
            ),
            key=lambda path: path.name,
        )

        if keep == 0:
            old_checkpoints = checkpoints
        else:
            old_checkpoints = checkpoints[:-keep]

        for path in old_checkpoints:
            path.unlink(
                missing_ok=True
            )

            checksum_path = path.with_suffix(
                path.suffix + ".sha256.json"
            )

            checksum_path.unlink(
                missing_ok=True
            )

    def list_checkpoints(
        self,
    ) -> list[Path]:
        return sorted(
            self.directory.glob(
                f"{self.config.prefix}*.pt"
            )
        )

    def get_latest_checkpoint(
        self,
    ) -> Path | None:
        if self.latest_path.exists():
            return self.latest_path

        epoch_checkpoints = sorted(
            self.directory.glob(
                f"{self.config.prefix}"
                "_epoch_*.pt"
            )
        )

        if epoch_checkpoints:
            return epoch_checkpoints[-1]

        return None

    def get_best_checkpoint(
        self,
    ) -> Path | None:
        if self.best_path.exists():
            return self.best_path

        return None

    def resume(
        self,
        *,
        models: (
            nn.Module
            | Mapping[str, nn.Module]
        ),
        optimizers: (
            Optimizer
            | Mapping[str, Optimizer]
            | None
        ) = None,
        schedulers: Any = None,
        scaler: Any = None,
        path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool | None = None,
        restore_rng: bool | None = None,
    ) -> dict[str, Any]:
        """
        Resume training from the latest
        or explicitly selected checkpoint.
        """
        if path is None:
            path = self.get_latest_checkpoint()

        if path is None:
            raise FileNotFoundError(
                "No checkpoint available for resume"
            )

        result = load_checkpoint(
            path,
            models=models,
            optimizers=optimizers,
            schedulers=schedulers,
            scaler=scaler,
            map_location=map_location,
            strict=(
                self.config.strict_model_loading
                if strict is None
                else strict
            ),
            load_optimizer=(
                self.config.save_optimizer
            ),
            load_scheduler=(
                self.config.save_scheduler
            ),
            load_scaler=(
                self.config.save_scaler
            ),
            load_rng_state=(
                self.config.restore_rng_state
                if restore_rng is None
                else restore_rng
            ),
            verify_checksum=(
                self.config.verify_checksum
            ),
        )

        metadata = result["metadata"]

        best_metric = metadata.get(
            "best_metric"
        )

        if best_metric is not None:
            self.best_metric = float(
                best_metric
            )

        return {
            **result,
            "checkpoint_path": str(path),
            "start_epoch": int(
                metadata.get("epoch", 0)
            ) + 1,
            "global_step": int(
                metadata.get(
                    "global_step",
                    0,
                )
            ),
        }

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "directory": str(
                self.directory
            ),
            "prefix": self.config.prefix,
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "checkpoint_count": len(
                self.list_checkpoints()
            ),
            "latest_exists": (
                self.latest_path.exists()
            ),
            "best_exists": (
                self.best_path.exists()
            ),
        }


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    class DemoWorldModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )

            self.dynamics = nn.Sequential(
                nn.Linear(256 + 32, 256),
                nn.GELU(),
                nn.Linear(256, 128),
            )

        def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor,
        ) -> torch.Tensor:
            latent = self.encoder(
                state
            )

            features = torch.cat(
                [latent, action],
                dim=-1,
            )

            return self.dynamics(
                features
            )

    model = DemoWorldModel().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6,
        )
    )

    manager = CheckpointManager(
        CheckpointConfig(
            directory="checkpoints/demo_world_model",
            prefix="world_model",
            save_every_n_epochs=1,
            keep_last_n=3,
            monitor="val_loss",
            mode="min",
        )
    )

    global_step = 0

    for epoch in range(1, 6):
        model.train()

        for _ in range(10):
            states = torch.randn(
                16,
                128,
                device=device,
            )

            actions = torch.randn(
                16,
                32,
                device=device,
            )

            targets = (
                states
                + 0.05
                * torch.randn_like(states)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                states,
                actions,
            )

            loss = torch.nn.functional.mse_loss(
                predictions,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            global_step += 1

        scheduler.step()

        validation_loss = float(
            loss.detach().item()
        )

        saved = manager.save(
            models=model,
            optimizers=optimizer,
            schedulers=scheduler,
            epoch=epoch,
            global_step=global_step,
            metrics={
                "val_loss": validation_loss
            },
        )

        print(
            f"Epoch {epoch} | "
            f"Validation Loss: "
            f"{validation_loss:.6f} | "
            f"Saved: {saved}"
        )

    print("\nCheckpoint statistics:")
    print(
        manager.get_statistics()
    )

    print("\nResume checkpoint:")

    resumed = manager.resume(
        models=model,
        optimizers=optimizer,
        schedulers=scheduler,
        map_location=device,
    )

    print(
        "Resume epoch:",
        resumed["start_epoch"],
    )

    print(
        "Global step:",
        resumed["global_step"],
    )
