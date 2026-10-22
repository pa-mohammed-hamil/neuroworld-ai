
#!/usr/bin/env python3
"""
NeuroWorld AI
File: scripts/train_world_model.py

Train an action-conditioned world/dynamics model.

Dataset format (from build_trajectories.py):
    train.npz
    val.npz
    test.npz

Required arrays:
    states       [N, state_dim]
    actions      [N, action_dim]
    next_states  [N, state_dim]

Optional:
    episode_ids  [N]
    robot_ids    [N]
    steps        [N]
    terminated   [N]
    truncated    [N]

Features:
    - Deterministic and stochastic dynamics
    - One-step and multi-step training
    - Gaussian negative log-likelihood
    - Teacher-forced or autoregressive rollout
    - Train/validation evaluation
    - Early stopping
    - Learning-rate scheduling
    - Mixed precision on CUDA
    - Gradient clipping
    - Checkpoint/resume
    - JSON history and summary
    - Compatible checkpoint for run_imagination.py
      when the same model architecture is used

Dependencies:
    pip install torch numpy

Examples:
    python scripts/train_world_model.py

    python scripts/train_world_model.py \
        --epochs 100 \
        --batch-size 128 \
        --hidden-dim 256

    python scripts/train_world_model.py \
        --rollout-horizon 5 \
        --rollout-weight 0.5

    python scripts/train_world_model.py \
        --resume artifacts/world_model/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOGGER = logging.getLogger("neuroworld.world_model")


@dataclass
class TrainConfig:
    data_dir: str = "data/trajectories"
    output_dir: str = "artifacts/world_model"

    model_type: str = "stochastic"
    hidden_dim: int = 256
    num_layers: int = 3

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5

    rollout_horizon: int = 1
    rollout_weight: float = 0.5

    teacher_forcing: bool = False

    min_logvar: float = -10.0
    max_logvar: float = 5.0

    gradient_clip: float = 1.0
    patience: int = 15

    num_workers: int = 0
    seed: int = 42

    device: str = "auto"
    mixed_precision: bool = True

    save_every: int = 10
    resume: Optional[str] = None


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    return device


class TrajectoryDataset(Dataset):
    """
    Builds contiguous windows of transitions.

    Windows stay within the same episode and robot.
    They do not cross terminal transitions.
    """

    def __init__(
        self,
        path: Path,
        horizon: int = 1,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as data:
            required = {
                "states",
                "actions",
                "next_states",
            }

            missing = required - set(data.files)

            if missing:
                raise ValueError(
                    f"Missing arrays: {sorted(missing)}"
                )

            self.states = data["states"].astype(np.float32)
            self.actions = data["actions"].astype(np.float32)
            self.next_states = data["next_states"].astype(
                np.float32
            )

            n = len(self.states)

            self.episode_ids = (
                data["episode_ids"].copy()
                if "episode_ids" in data
                else np.arange(n)
            )

            self.robot_ids = (
                data["robot_ids"].astype(str)
                if "robot_ids" in data
                else np.full(n, "robot")
            )

            self.steps = (
                data["steps"].copy()
                if "steps" in data
                else np.zeros(n, dtype=np.int64)
            )

            self.terminated = (
                data["terminated"].astype(bool)
                if "terminated" in data
                else np.zeros(n, dtype=bool)
            )

            self.truncated = (
                data["truncated"].astype(bool)
                if "truncated" in data
                else np.zeros(n, dtype=bool)
            )

        if (
            self.states.ndim != 2
            or self.actions.ndim != 2
            or self.next_states.ndim != 2
        ):
            raise ValueError("Expected 2D transition arrays")

        if not (
            len(self.states)
            == len(self.actions)
            == len(self.next_states)
        ):
            raise ValueError("Transition counts differ")

        if self.states.shape != self.next_states.shape:
            raise ValueError("State dimensions differ")

        if len(self.states) == 0:
            raise ValueError("Empty dataset")

        for array in (
            self.states,
            self.actions,
            self.next_states,
        ):
            if not np.isfinite(array).all():
                raise ValueError("Non-finite dataset values")

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        self.horizon = horizon

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if horizon > 1 and (
            len(np.unique(self.episode_ids)) == len(self.states)
        ):
            raise ValueError(
                "Multi-step training requires episode_ids, "
                "robot_ids and steps metadata"
            )

        self.windows = self._build_windows()

        if not self.windows:
            raise ValueError(
                f"No valid windows for horizon={horizon}"
            )

    def _build_windows(self) -> list[list[int]]:
        groups: dict[tuple[str, str], list[int]] = {}

        for index in range(len(self.states)):
            key = (
                str(self.episode_ids[index]),
                str(self.robot_ids[index]),
            )
            groups.setdefault(key, []).append(index)

        windows = []

        for indices in groups.values():
            indices.sort(key=lambda i: int(self.steps[i]))

            for start in range(
                len(indices) - self.horizon + 1
            ):
                window = indices[
                    start:start + self.horizon
                ]

                valid = True

                for position in range(len(window) - 1):
                    current = window[position]
                    following = window[position + 1]

                    if (
                        int(self.steps[following])
                        != int(self.steps[current]) + 1
                    ):
                        valid = False
                        break

                    if (
                        self.terminated[current]
                        or self.truncated[current]
                    ):
                        valid = False
                        break

                if valid:
                    windows.append(window)

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        window = self.windows[index]

        return (
            torch.from_numpy(
                self.states[window[0]].copy()
            ),
            torch.from_numpy(
                self.actions[window].copy()
            ),
            torch.from_numpy(
                self.next_states[window].copy()
            ),
        )


class ActionConditionedWorldModel(nn.Module):
    """
    Feed-forward action-conditioned dynamics.

    state:  [B, D]
    action: [B, A]

    Returns:
        mean:   [B, D]
        logvar: [B, D]

    For deterministic models, logvar is zero.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        stochastic: bool = True,
        min_logvar: float = -10.0,
        max_logvar: float = 5.0,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic

        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        layers: list[nn.Module] = []

        input_dim = state_dim + action_dim

        for index in range(num_layers):
            layers.extend(
                [
                    nn.Linear(
                        input_dim if index == 0 else hidden_dim,
                        hidden_dim,
                    ),
                    nn.SiLU(),
                ]
            )

        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(
            hidden_dim,
            state_dim,
        )

        self.logvar_head = (
            nn.Linear(hidden_dim, state_dim)
            if stochastic
            else None
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        features = torch.cat(
            [state, action],
            dim=-1,
        )

        hidden = self.backbone(features)

        # Residual prediction: next = current + delta
        mean = state + self.mean_head(hidden)

        if self.logvar_head is not None:
            logvar = self.logvar_head(hidden).clamp(
                self.min_logvar,
                self.max_logvar,
            )
        else:
            logvar = torch.zeros_like(mean)

        return mean, logvar

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        deterministic: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        mean, logvar = self.forward(state, action)

        if deterministic or not self.stochastic:
            return mean, logvar

        noise = torch.randn_like(mean)

        next_state = (
            mean
            + torch.exp(0.5 * logvar) * noise
        )

        return next_state, logvar


class WorldModelTrainer:
    def __init__(
        self,
        config: TrainConfig,
    ) -> None:
        self.config = config
        self._validate_config()

        set_seed(config.seed)

        self.device = get_device(config.device)

        self.output_dir = resolve_path(
            config.output_dir
        )

        self.checkpoint_dir = (
            self.output_dir / "checkpoints"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.train_dataset = TrajectoryDataset(
            resolve_path(config.data_dir) / "train.npz",
            horizon=config.rollout_horizon,
        )

        self.val_dataset = TrajectoryDataset(
            resolve_path(config.data_dir) / "val.npz",
            horizon=config.rollout_horizon,
        )

        if (
            self.train_dataset.state_dim
            != self.val_dataset.state_dim
            or self.train_dataset.action_dim
            != self.val_dataset.action_dim
        ):
            raise ValueError(
                "Training/validation dimensions differ"
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.model = ActionConditionedWorldModel(
            state_dim=self.train_dataset.state_dim,
            action_dim=self.train_dataset.action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            stochastic=config.model_type == "stochastic",
            min_logvar=config.min_logvar,
            max_logvar=config.max_logvar,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        )

        self.use_amp = (
            config.mixed_precision
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.start_epoch = 0
        self.best_val_loss = math.inf
        self.bad_epochs = 0

        self.history: list[
            dict[str, Any]
        ] = []

        if config.resume:
            self.resume(config.resume)

    def _validate_config(self) -> None:
        c = self.config

        if c.model_type not in {
            "deterministic",
            "stochastic",
        }:
            raise ValueError(
                "model_type must be deterministic "
                "or stochastic"
            )

        if (
            c.hidden_dim <= 0
            or c.num_layers <= 0
            or c.epochs <= 0
            or c.batch_size <= 0
            or c.rollout_horizon <= 0
        ):
            raise ValueError(
                "Model/training dimensions must be positive"
            )

        if c.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if not 0 <= c.rollout_weight <= 1:
            raise ValueError(
                "rollout_weight must be between 0 and 1"
            )

        if c.min_logvar >= c.max_logvar:
            raise ValueError(
                "Invalid log-variance bounds"
            )

        if c.patience <= 0 or c.save_every <= 0:
            raise ValueError(
                "Patience and save interval must be positive"
            )

    def _prediction_loss(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.model_type == "deterministic":
            return F.mse_loss(mean, target)

        # Compute Gaussian NLL in float32.
        mean = mean.float()
        logvar = logvar.float()
        target = target.float()

        inverse_variance = torch.exp(
            -logvar
        )

        nll = 0.5 * (
            math.log(2.0 * math.pi)
            + logvar
            + (target - mean).square()
            * inverse_variance
        )

        return nll.mean()

    def _batch_loss(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, float],
    ]:
        horizon = actions.shape[1]

        current = initial_state

        one_step_losses = []
        rollout_losses = []

        for step in range(horizon):
            mean, logvar = self.model(
                current,
                actions[:, step],
            )

            target = targets[:, step]

            prediction_loss = self._prediction_loss(
                mean,
                logvar,
                target,
            )

            if step == 0:
                one_step_losses.append(
                    prediction_loss
                )

            rollout_losses.append(
                prediction_loss
            )

            if self.config.teacher_forcing:
                current = target
            else:
                current = mean

        one_step_loss = torch.stack(
            one_step_losses
        ).mean()

        rollout_loss = torch.stack(
            rollout_losses
        ).mean()

        weight = (
            self.config.rollout_weight
            if horizon > 1
            else 0.0
        )

        total_loss = (
            (1.0 - weight) * one_step_loss
            + weight * rollout_loss
        )

        metrics = {
            "one_step_loss": float(
                one_step_loss.detach().item()
            ),
            "rollout_loss": float(
                rollout_loss.detach().item()
            ),
        }

        return total_loss, metrics

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> dict[str, float]:
        self.model.train(training)

        totals = {
            "loss": 0.0,
            "one_step_loss": 0.0,
            "rollout_loss": 0.0,
        }

        samples = 0

        for (
            initial_states,
            actions,
            targets,
        ) in loader:
            initial_states = initial_states.to(
                self.device,
                non_blocking=True,
            )

            actions = actions.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            if training:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    loss, metrics = self._batch_loss(
                        initial_states,
                        actions,
                        targets,
                    )

                if training:
                    self.scaler.scale(loss).backward()

                    self.scaler.unscale_(
                        self.optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                    self.scaler.step(
                        self.optimizer
                    )

                    self.scaler.update()

            batch_size = len(initial_states)
            samples += batch_size

            totals["loss"] += (
                float(loss.detach().item())
                * batch_size
            )

            for key in (
                "one_step_loss",
                "rollout_loss",
            ):
                totals[key] += (
                    metrics[key] * batch_size
                )

        if samples == 0:
            raise RuntimeError("Empty training loader")

        return {
            key: value / samples
            for key, value in totals.items()
        }

    def save_checkpoint(
        self,
        name: str,
        epoch: int,
    ) -> Path:
        path = self.checkpoint_dir / name
        temporary = path.with_suffix(".tmp")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
            ),
            "scaler_state_dict": (
                self.scaler.state_dict()
            ),
            "best_val_loss": self.best_val_loss,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": asdict(self.config),
            "state_dim": self.train_dataset.state_dim,
            "action_dim": self.train_dataset.action_dim,
        }

        torch.save(checkpoint, temporary)
        temporary.replace(path)

        return path

    def resume(self, value: str) -> None:
        path = resolve_path(value)

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        self.start_epoch = int(
            checkpoint["epoch"]
        ) + 1

        self.best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                math.inf,
            )
        )

        self.bad_epochs = int(
            checkpoint.get(
                "bad_epochs",
                0,
            )
        )

        self.history = list(
            checkpoint.get(
                "history",
                [],
            )
        )

        LOGGER.info(
            "Resumed training at epoch %d",
            self.start_epoch + 1,
        )

    def train(self) -> dict[str, Any]:
        started = time.perf_counter()

        for epoch in range(
            self.start_epoch,
            self.config.epochs,
        ):
            train_metrics = self._run_epoch(
                self.train_loader,
                training=True,
            )

            with torch.no_grad():
                val_metrics = self._run_epoch(
                    self.val_loader,
                    training=False,
                )

            self.scheduler.step(
                val_metrics["loss"]
            )

            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": val_metrics,
                "learning_rate": (
                    self.optimizer.param_groups[0]["lr"]
                ),
            }

            self.history.append(record)

            improved = (
                val_metrics["loss"]
                < self.best_val_loss
            )

            if improved:
                self.best_val_loss = (
                    val_metrics["loss"]
                )

                self.bad_epochs = 0

                self.save_checkpoint(
                    "best.pt",
                    epoch,
                )
            else:
                self.bad_epochs += 1

            self.save_checkpoint(
                "last.pt",
                epoch,
            )

            if (
                (epoch + 1)
                % self.config.save_every
                == 0
            ):
                self.save_checkpoint(
                    f"epoch_{epoch + 1:04d}.pt",
                    epoch,
                )

            LOGGER.info(
                "Epoch %d/%d | "
                "train=%.6f | "
                "val=%.6f | "
                "lr=%.8f",
                epoch + 1,
                self.config.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                self.optimizer.param_groups[0]["lr"],
            )

            self._save_history()

            if self.bad_epochs >= self.config.patience:
                LOGGER.info(
                    "Early stopping at epoch %d",
                    epoch + 1,
                )
                break

        summary = {
            "project": "NeuroWorld AI",
            "model": "ActionConditionedWorldModel",
            "model_type": self.config.model_type,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "epochs_completed": len(self.history),
            "best_validation_loss": (
                self.best_val_loss
            ),
            "state_dim": self.train_dataset.state_dim,
            "action_dim": self.train_dataset.action_dim,
            "train_windows": len(self.train_dataset),
            "val_windows": len(self.val_dataset),
            "training_seconds": (
                time.perf_counter() - started
            ),
            "device": str(self.device),
            "best_checkpoint": str(
                self.checkpoint_dir / "best.pt"
            ),
        }

        summary_path = (
            self.output_dir / "training_summary.json"
        )

        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        return summary

    def _save_history(self) -> None:
        path = (
            self.output_dir / "training_history.json"
        )

        path.write_text(
            json.dumps(
                self.history,
                indent=2,
            ),
            encoding="utf-8",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NeuroWorld world model"
    )

    parser.add_argument(
        "--data",
        default="data/trajectories",
    )

    parser.add_argument(
        "--output",
        default="artifacts/world_model",
    )

    parser.add_argument(
        "--model-type",
        choices=[
            "deterministic",
            "stochastic",
        ],
        default="stochastic",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--rollout-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--resume",
        default=None,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    config = TrainConfig(
        data_dir=args.data,
        output_dir=args.output,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        rollout_horizon=args.rollout_horizon,
        rollout_weight=args.rollout_weight,
        teacher_forcing=args.teacher_forcing,
        device=args.device,
        seed=args.seed,
        num_workers=args.workers,
        resume=args.resume,
        mixed_precision=not args.no_amp,
    )

    try:
        trainer = WorldModelTrainer(config)
        summary = trainer.train()

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        return 0

    except Exception:
        LOGGER.exception(
            "World-model training failed"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
