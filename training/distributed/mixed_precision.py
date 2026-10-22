# training/distributed/mixed_precision.py

"""
NeuroWorld AI - Mixed Precision Training

Features:
- FP32, FP16 and BF16
- Automatic precision selection
- CPU and CUDA support
- Automatic mixed precision (AMP)
- Gradient scaling for FP16
- Gradient clipping
- Non-finite gradient handling
- Optimizer step management
- Gradient accumulation compatibility
- AMP checkpoint state management
- DDP and FSDP compatibility

Requires PyTorch >= 2.1.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class MixedPrecisionConfig:
    enabled: bool = True
    precision: str = "auto"
    device: str = "auto"

    gradient_scaling: bool = True
    initial_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    gradient_clip_norm: float | None = 1.0
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        if self.precision not in {
            "auto", "fp32", "fp16", "bf16"
        }:
            raise ValueError(
                "precision must be auto, fp32, fp16 or bf16"
            )

        if self.initial_scale <= 0:
            raise ValueError("initial_scale must be positive")

        if self.growth_factor <= 1:
            raise ValueError("growth_factor must be > 1")

        if not 0 < self.backoff_factor < 1:
            raise ValueError(
                "backoff_factor must be between 0 and 1"
            )

        if self.growth_interval < 1:
            raise ValueError(
                "growth_interval must be >= 1"
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive"
            )


@dataclass
class OptimizerStepResult:
    stepped: bool
    skipped: bool
    gradients_finite: bool
    gradient_norm: float | None
    loss_scale: float


class MixedPrecisionManager:
    def __init__(
        self,
        config: MixedPrecisionConfig | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or MixedPrecisionConfig()

        self.device = self._resolve_device(
            device if device is not None
            else self.config.device
        )

        self.precision = self._resolve_precision()

        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]

        self.amp_enabled = (
            self.config.enabled
            and self.precision != "fp32"
        )

        self.scaler_enabled = (
            self.amp_enabled
            and self.precision == "fp16"
            and self.device.type == "cuda"
            and self.config.gradient_scaling
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.scaler_enabled,
            init_scale=self.config.initial_scale,
            growth_factor=self.config.growth_factor,
            backoff_factor=self.config.backoff_factor,
            growth_interval=self.config.growth_interval,
        )

        self.optimizer_steps = 0
        self.skipped_steps = 0

    @staticmethod
    def _resolve_device(
        requested: torch.device | str,
    ) -> torch.device:
        device = torch.device(requested) if str(
            requested
        ) != "auto" else None

        if device is None:
            if torch.cuda.is_available():
                local_rank = int(
                    __import__("os").environ.get(
                        "LOCAL_RANK", "0"
                    )
                )
                device = torch.device(
                    "cuda", local_rank
                )
            else:
                device = torch.device("cpu")

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")

            index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )

            if index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Invalid CUDA device index: {index}"
                )

            torch.cuda.set_device(index)

            return torch.device("cuda", index)

        if device.type not in {"cpu", "mps"}:
            raise ValueError(
                f"Unsupported device: {device}"
            )

        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable")

        return device

    def _resolve_precision(self) -> str:
        if not self.config.enabled:
            return "fp32"

        requested = self.config.precision

        if requested == "fp32":
            return "fp32"

        if requested == "auto":
            if self.device.type == "cuda":
                return (
                    "bf16"
                    if torch.cuda.is_bf16_supported()
                    else "fp16"
                )

            return "fp32"

        if requested == "fp16":
            if self.device.type != "cuda":
                raise RuntimeError(
                    "FP16 training requires CUDA "
                    "in this implementation"
                )

            if not self.config.gradient_scaling:
                raise ValueError(
                    "FP16 requires gradient_scaling=True"
                )

            return "fp16"

        if requested == "bf16":
            if self.device.type == "cuda":
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(
                        "BF16 unsupported on this GPU"
                    )

            elif self.device.type != "cpu":
                raise RuntimeError(
                    "BF16 is supported only on "
                    "CPU or CUDA in this manager"
                )

            return "bf16"

        raise ValueError(
            f"Unsupported precision: {requested}"
        )

    def autocast(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()

        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=True,
        )

    def forward(
        self,
        model: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self.autocast():
            return model(*args, **kwargs)

    def backward(
        self,
        loss: Tensor,
        *,
        accumulation_steps: int = 1,
    ) -> Tensor:
        if accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be >= 1"
            )

        if not isinstance(loss, Tensor):
            raise TypeError("loss must be a Tensor")

        if loss.ndim != 0:
            raise ValueError(
                "loss must be a scalar Tensor"
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "Non-finite loss detected"
            )

        scaled_loss = (
            loss / accumulation_steps
        )

        self.scaler.scale(
            scaled_loss
        ).backward()

        return scaled_loss.detach()

    @staticmethod
    def _gradients_finite(
        model: nn.Module,
    ) -> bool:
        for parameter in model.parameters():
            grad = parameter.grad

            if grad is None:
                continue

            if grad.is_sparse:
                grad = grad.coalesce().values()

            if not torch.isfinite(
                grad
            ).all().item():
                return False

        return True

    @staticmethod
    def _clip_gradients(
        model: nn.Module,
        max_norm: float,
    ) -> float:
        # FSDP exposes a distributed-aware
        # gradient clipping implementation.
        if hasattr(model, "clip_grad_norm_"):
            norm = model.clip_grad_norm_(
                max_norm
            )
        else:
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm,
            )

        if isinstance(norm, Tensor):
            return float(norm.detach().item())

        return float(norm)

    def optimizer_step(
        self,
        optimizer: Optimizer,
        *,
        model: nn.Module | None = None,
        scheduler: Any | None = None,
        zero_grad: bool = True,
    ) -> OptimizerStepResult:
        if model is None and (
            self.config.detect_nonfinite
            or self.config.gradient_clip_norm is not None
        ):
            raise ValueError(
                "model is required for gradient "
                "validation and clipping"
            )

        if self.scaler_enabled:
            self.scaler.unscale_(optimizer)

        gradients_finite = True

        if (
            model is not None
            and self.config.detect_nonfinite
        ):
            gradients_finite = (
                self._gradients_finite(model)
            )

        gradient_norm = None

        if (
            model is not None
            and self.config.gradient_clip_norm
            is not None
            and gradients_finite
        ):
            gradient_norm = self._clip_gradients(
                model,
                self.config.gradient_clip_norm,
            )

            if not torch.isfinite(
                torch.tensor(gradient_norm)
            ).item():
                gradients_finite = False

        stepped = False

        if self.scaler_enabled:
            old_scale = self.scaler.get_scale()

            # GradScaler skips the optimizer update
            # if it finds NaN/Inf gradients.
            self.scaler.step(optimizer)
            self.scaler.update()

            stepped = (
                self.scaler.get_scale()
                >= old_scale
            )

        elif gradients_finite:
            optimizer.step()
            stepped = True

        if stepped:
            self.optimizer_steps += 1

            if scheduler is not None:
                scheduler.step()

        else:
            self.skipped_steps += 1

        if zero_grad:
            optimizer.zero_grad(
                set_to_none=True
            )

        return OptimizerStepResult(
            stepped=stepped,
            skipped=not stepped,
            gradients_finite=gradients_finite,
            gradient_norm=gradient_norm,
            loss_scale=float(
                self.scaler.get_scale()
            ),
        )

    def train_step(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Any,
        inputs: Any,
        targets: Any,
        *,
        scheduler: Any | None = None,
    ) -> dict[str, Any]:
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        with self.autocast():
            if isinstance(inputs, dict):
                predictions = model(**inputs)

            elif isinstance(inputs, tuple):
                predictions = model(*inputs)

            else:
                predictions = model(inputs)

            loss = loss_fn(
                predictions,
                targets,
            )

        self.backward(loss)

        result = self.optimizer_step(
            optimizer,
            model=model,
            scheduler=scheduler,
        )

        return {
            "loss": float(
                loss.detach().item()
            ),
            "optimizer_stepped": result.stepped,
            "gradients_finite": (
                result.gradients_finite
            ),
            "gradient_norm": result.gradient_norm,
            "loss_scale": result.loss_scale,
        }

    @torch.no_grad()
    def validation_step(
        self,
        model: nn.Module,
        loss_fn: Any,
        inputs: Any,
        targets: Any,
    ) -> dict[str, Any]:
        model.eval()

        with self.autocast():
            if isinstance(inputs, dict):
                predictions = model(**inputs)

            elif isinstance(inputs, tuple):
                predictions = model(*inputs)

            else:
                predictions = model(inputs)

            loss = loss_fn(
                predictions,
                targets,
            )

        return {
            "loss": float(
                loss.detach().item()
            ),
            "predictions": predictions,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "optimizer_steps": self.optimizer_steps,
            "skipped_steps": self.skipped_steps,
            "scaler_state_dict": (
                self.scaler.state_dict()
            ),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        self.optimizer_steps = int(
            state.get("optimizer_steps", 0)
        )

        self.skipped_steps = int(
            state.get("skipped_steps", 0)
        )

        scaler_state = state.get(
            "scaler_state_dict"
        )

        if (
            scaler_state is not None
            and self.scaler_enabled
        ):
            self.scaler.load_state_dict(
                scaler_state
            )

    def info(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "precision": self.precision,
            "amp_enabled": self.amp_enabled,
            "gradient_scaling": self.scaler_enabled,
            "loss_scale": float(
                self.scaler.get_scale()
            ),
            "optimizer_steps": self.optimizer_steps,
            "skipped_steps": self.skipped_steps,
        }


def create_mixed_precision_manager(
    *,
    precision: str = "auto",
    device: str = "auto",
    gradient_clip_norm: float | None = 1.0,
) -> MixedPrecisionManager:
    config = MixedPrecisionConfig(
        precision=precision,
        device=device,
        gradient_clip_norm=gradient_clip_norm,
    )

    return MixedPrecisionManager(
        config
    )


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    manager = MixedPrecisionManager(
        MixedPrecisionConfig(
            enabled=True,
            precision="auto",
            device=device,
            gradient_clip_norm=1.0,
        )
    )

    inputs = torch.randn(
        16, 128, device=device
    )

    targets = torch.randn(
        16, 128, device=device
    )

    loss_fn = nn.MSELoss()

    for step in range(5):
        result = manager.train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            inputs=inputs,
            targets=targets,
        )

        print(
            f"Step {step + 1}: {result}"
        )

    print(manager.info())