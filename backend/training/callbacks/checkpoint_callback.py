
"""
NeuroWorld AI
File: training/callbacks/checkpoint_callback.py

Features:
- Save model and optimizer state
- Save learning-rate scheduler state
- Save AMP gradient scaler state
- Track best validation metric
- Save checkpoints every N epochs
- Keep only latest N periodic checkpoints
- Save best and last checkpoints
- Restore training from checkpoint
- Atomic checkpoint writes
- Store training metadata
- Optional distributed training support

Requires: PyTorch 2.x
"""

from __future__ import annotations

import logging
import math
import os
import random
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"

    save_every_n_epochs: int = 1
    save_best: bool = True
    save_last: bool = True

    max_checkpoints: int = 5
    filename_prefix: str = "neuroworld"

    save_optimizer: bool = True
    save_scheduler: bool = True
    save_scaler: bool = True
    save_rng_state: bool = True

    rank_zero_only: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")

        if self.save_every_n_epochs < 1:
            raise ValueError(
                "save_every_n_epochs must be >= 1"
            )

        if self.max_checkpoints < 1:
            raise ValueError(
                "max_checkpoints must be >= 1"
            )

        if not self.filename_prefix:
            raise ValueError(
                "filename_prefix cannot be empty"
            )


class CheckpointCallback:
    """
    Checkpoint manager for PyTorch training.

    Example:
        callback = CheckpointCallback(
            CheckpointConfig(
                directory="checkpoints/world_model",
                monitor="val_loss",
                mode="min",
            )
        )

        callback.on_epoch_end(
            epoch=0,
            model=model,
            optimizer=optimizer,
            metrics={"val_loss": 0.25},
        )
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        config: CheckpointConfig | None = None,
    ) -> None:
        self.config = config or CheckpointConfig()

        self.directory = Path(
            self.config.directory
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_score = (
            math.inf
            if self.config.mode == "min"
            else -math.inf
        )

        self.best_epoch: int | None = None
        self.last_epoch: int | None = None

        self._periodic_files: list[Path] = []

    def _is_main_process(self) -> bool:
        if not self.config.rank_zero_only:
            return True

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return (
                torch.distributed.get_rank() == 0
            )

        return True

    @staticmethod
    def _unwrap_model(
        model: nn.Module,
    ) -> nn.Module:
        """
        Handle DistributedDataParallel and
        DataParallel model wrappers.
        """
        if isinstance(
            model,
            (
                nn.DataParallel,
                nn.parallel.DistributedDataParallel,
            ),
        ):
            return model.module

        return model

    def _is_improvement(
        self,
        score: float,
    ) -> bool:
        if not math.isfinite(score):
            return False

        if self.config.mode == "min":
            return score < self.best_score

        return score > self.best_score

    @staticmethod
    def _rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            state["cuda"] = (
                torch.cuda.get_rng_state_all()
            )

        return state

    @staticmethod
    def _restore_rng_state(
        state: dict[str, Any],
    ) -> None:
        if "python" in state:
            random.setstate(state["python"])

        if "numpy" in state:
            np.random.set_state(state["numpy"])

        if "torch" in state:
            torch.set_rng_state(state["torch"])

        if (
            "cuda" in state
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(
                state["cuda"]
            )

    def _build_checkpoint(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer | None,
        scheduler: LRScheduler | None,
        scaler: Any | None,
        metrics: dict[str, float],
        global_step: int,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        model = self._unwrap_model(model)

        checkpoint: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "metrics": dict(metrics),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "metadata": metadata or {},
        }

        if (
            optimizer is not None
            and self.config.save_optimizer
        ):
            checkpoint["optimizer_state_dict"] = (
                optimizer.state_dict()
            )

        if (
            scheduler is not None
            and self.config.save_scheduler
        ):
            checkpoint["scheduler_state_dict"] = (
                scheduler.state_dict()
            )

        if (
            scaler is not None
            and self.config.save_scaler
        ):
            checkpoint["scaler_state_dict"] = (
                scaler.state_dict()
            )

        if self.config.save_rng_state:
            checkpoint["rng_state"] = (
                self._rng_state()
            )

        return checkpoint

    def _atomic_save(
        self,
        checkpoint: dict[str, Any],
        destination: Path,
    ) -> None:
        """
        Write to a temporary file and atomically
        replace the destination after success.
        """
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd, temporary_path = tempfile.mkstemp(
            prefix=".checkpoint_",
            suffix=".tmp",
            dir=str(destination.parent),
        )

        os.close(fd)

        try:
            torch.save(
                checkpoint,
                temporary_path,
            )

            os.replace(
                temporary_path,
                destination,
            )

        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _checkpoint_path(
        self,
        suffix: str,
    ) -> Path:
        return self.directory / (
            f"{self.config.filename_prefix}_"
            f"{suffix}.pt"
        )

    def _cleanup_periodic(self) -> None:
        while (
            len(self._periodic_files)
            > self.config.max_checkpoints
        ):
            oldest = self._periodic_files.pop(0)

            if oldest.exists():
                oldest.unlink()

    def on_epoch_end(
        self,
        *,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: Any | None = None,
        global_step: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Called after validation at the end
        of every epoch.

        Epoch numbering is zero-based.
        """
        if epoch < 0:
            raise ValueError(
                "epoch must be nonnegative"
            )

        if not self._is_main_process():
            return {
                "saved": False,
                "reason": "non_main_process",
            }

        self.last_epoch = epoch

        score = metrics.get(
            self.config.monitor
        )

        improved = False

        if score is not None:
            score = float(score)

            if self._is_improvement(score):
                self.best_score = score
                self.best_epoch = epoch
                improved = True

        checkpoint = self._build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=metrics,
            global_step=global_step,
            metadata=metadata,
        )

        saved_paths: list[str] = []

        if self.config.save_last:
            path = self._checkpoint_path("last")

            self._atomic_save(
                checkpoint,
                path,
            )

            saved_paths.append(str(path))

        if (
            self.config.save_best
            and improved
        ):
            path = self._checkpoint_path("best")

            self._atomic_save(
                checkpoint,
                path,
            )

            saved_paths.append(str(path))

        if (
            (epoch + 1)
            % self.config.save_every_n_epochs
            == 0
        ):
            path = self._checkpoint_path(
                f"epoch_{epoch + 1:06d}"
            )

            self._atomic_save(
                checkpoint,
                path,
            )

            self._periodic_files.append(path)

            self._cleanup_periodic()

            saved_paths.append(str(path))

        if self.config.verbose and saved_paths:
            logger.info(
                "Epoch %d checkpoints: %s",
                epoch + 1,
                saved_paths,
            )

        return {
            "saved": bool(saved_paths),
            "improved": improved,
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_score": (
                self.best_score
                if math.isfinite(self.best_score)
                else None
            ),
            "paths": saved_paths,
        }

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: Any | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
        restore_rng: bool = False,
        trusted_source: bool = False,
    ) -> dict[str, Any]:
        """
        Restore checkpoint state.

        Only load checkpoints from trusted sources.
        Full training checkpoints contain Python
        and NumPy RNG states and optimizer data.
        """
        if not trusted_source:
            raise PermissionError(
                "Checkpoint loading requires "
                "trusted_source=True. "
                "Never load untrusted checkpoint files."
            )

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                "Invalid checkpoint format"
            )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Missing model_state_dict"
            )

        model = self._unwrap_model(model)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if (
            scaler is not None
            and "scaler_state_dict" in checkpoint
        ):
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        if (
            restore_rng
            and "rng_state" in checkpoint
        ):
            self._restore_rng_state(
                checkpoint["rng_state"]
            )

        self.best_score = float(
            checkpoint.get(
                "best_score",
                self.best_score,
            )
        )

        self.best_epoch = checkpoint.get(
            "best_epoch"
        )

        self.last_epoch = checkpoint.get(
            "epoch"
        )

        return {
            "epoch": self.last_epoch,
            "next_epoch": (
                self.last_epoch + 1
                if self.last_epoch is not None
                else 0
            ),
            "global_step": checkpoint.get(
                "global_step",
                0,
            ),
            "metrics": checkpoint.get(
                "metrics",
                {},
            ),
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "metadata": checkpoint.get(
                "metadata",
                {},
            ),
        }

    def load_best(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.load_checkpoint(
            self._checkpoint_path("best"),
            **kwargs,
        )

    def load_last(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.load_checkpoint(
            self._checkpoint_path("last"),
            **kwargs,
        )
