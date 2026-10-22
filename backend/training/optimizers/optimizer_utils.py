
"""
NeuroWorld AI
File: training/optimizers/optimizer_utils.py

Optimizer utilities for world models, generative AI,
robotics, reinforcement learning and distributed training.

Features:
- Gradient norm calculation
- NaN/Inf gradient detection
- Gradient clipping
- Optimizer parameter statistics
- Learning-rate management
- Parameter freezing and unfreezing
- Optimizer state device migration
- Optimizer checkpoint save/load
- Gradient accumulation helper
- Mixed precision training helper
- Optimizer state memory estimation
- Training diagnostics

Requires:
    PyTorch >= 2.1
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass
class OptimizerUtilsConfig:
    max_gradient_norm: float | None = 1.0
    gradient_clip_value: float | None = None
    norm_type: float = 2.0

    accumulation_steps: int = 1

    mixed_precision: bool = False
    precision_dtype: str = "float16"

    detect_nonfinite: bool = True
    skip_nonfinite_steps: bool = True

    zero_grad_set_to_none: bool = True

    def __post_init__(self) -> None:
        if (
            self.max_gradient_norm is not None
            and self.max_gradient_norm <= 0
        ):
            raise ValueError(
                "max_gradient_norm must be positive"
            )

        if (
            self.gradient_clip_value is not None
            and self.gradient_clip_value <= 0
        ):
            raise ValueError(
                "gradient_clip_value must be positive"
            )

        if self.norm_type <= 0:
            raise ValueError(
                "norm_type must be positive"
            )

        if self.accumulation_steps < 1:
            raise ValueError(
                "accumulation_steps must be >= 1"
            )

        if self.precision_dtype not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError(
                "Unsupported precision dtype"
            )


def get_trainable_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def get_optimizer_parameters(
    optimizer: Optimizer,
) -> list[nn.Parameter]:
    parameters = []
    seen = set()

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue

            seen.add(id(parameter))
            parameters.append(parameter)

    return parameters


def count_model_parameters(
    model: nn.Module,
) -> dict[str, int]:
    total = 0
    trainable = 0
    frozen = 0

    for parameter in model.parameters():
        count = parameter.numel()

        total += count

        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def count_optimizer_parameters(
    optimizer: Optimizer,
) -> int:
    return sum(
        parameter.numel()
        for parameter in get_optimizer_parameters(
            optimizer
        )
    )


@torch.no_grad()
def calculate_gradient_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: float = 2.0,
) -> float:
    gradients = []

    for parameter in parameters:
        gradient = parameter.grad

        if gradient is None:
            continue

        if gradient.is_sparse:
            gradient = gradient.coalesce().values()

        gradients.append(
            gradient.detach().float()
        )

    if not gradients:
        return 0.0

    if norm_type == float("inf"):
        norms = [
            gradient.abs().max()
            for gradient in gradients
        ]

        return max(
            float(norm.item())
            for norm in norms
        )

    norms = [
        torch.linalg.vector_norm(
            gradient,
            ord=norm_type,
        )
        for gradient in gradients
    ]

    total_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                norm.to(norms[0].device)
                for norm in norms
            ]
        ),
        ord=norm_type,
    )

    return float(
        total_norm.item()
    )


@torch.no_grad()
def gradients_are_finite(
    parameters: Iterable[nn.Parameter],
) -> bool:
    for parameter in parameters:
        gradient = parameter.grad

        if gradient is None:
            continue

        if gradient.is_sparse:
            gradient = gradient.coalesce().values()

        if not torch.isfinite(
            gradient
        ).all().item():
            return False

    return True


@torch.no_grad()
def count_nonfinite_gradients(
    parameters: Iterable[nn.Parameter],
) -> dict[str, int]:
    nan_count = 0
    inf_count = 0

    for parameter in parameters:
        gradient = parameter.grad

        if gradient is None:
            continue

        if gradient.is_sparse:
            gradient = gradient.coalesce().values()

        nan_count += int(
            torch.isnan(
                gradient
            ).sum().item()
        )

        inf_count += int(
            torch.isinf(
                gradient
            ).sum().item()
        )

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float | None = 1.0,
    clip_value: float | None = None,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = True,
) -> float:
    parameters = list(parameters)

    if max_norm is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
        )

        gradient_norm = float(
            total_norm.item()
        )

    else:
        gradient_norm = calculate_gradient_norm(
            parameters,
            norm_type=norm_type,
        )

    if clip_value is not None:
        torch.nn.utils.clip_grad_value_(
            parameters,
            clip_value=clip_value,
        )

    return gradient_norm


def get_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def set_learning_rate(
    optimizer: Optimizer,
    learning_rate: float,
    *,
    group_index: int | None = None,
) -> None:
    if (
        not math.isfinite(learning_rate)
        or learning_rate < 0
    ):
        raise ValueError(
            "learning_rate must be finite "
            "and nonnegative"
        )

    if group_index is None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    else:
        optimizer.param_groups[
            group_index
        ]["lr"] = learning_rate


def scale_learning_rates(
    optimizer: Optimizer,
    factor: float,
) -> None:
    if (
        not math.isfinite(factor)
        or factor < 0
    ):
        raise ValueError(
            "factor must be finite "
            "and nonnegative"
        )

    for group in optimizer.param_groups:
        group["lr"] *= factor


def get_weight_decays(
    optimizer: Optimizer,
) -> list[float]:
    return [
        float(
            group.get(
                "weight_decay",
                0.0,
            )
        )
        for group in optimizer.param_groups
    ]


def set_weight_decay(
    optimizer: Optimizer,
    weight_decay: float,
    *,
    group_index: int | None = None,
) -> None:
    if (
        not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError(
            "weight_decay must be finite "
            "and nonnegative"
        )

    if group_index is None:
        for group in optimizer.param_groups:
            group["weight_decay"] = weight_decay
    else:
        optimizer.param_groups[
            group_index
        ]["weight_decay"] = weight_decay


def freeze_parameters(
    model: nn.Module,
    *,
    prefixes: Iterable[str] | None = None,
) -> int:
    """
    Disable gradients for selected parameters.

    If prefixes is None, freeze all parameters.

    Important:
        Freezing parameters does not remove them
        from an existing optimizer. Rebuild the
        optimizer if parameter groups must change.
    """
    prefixes = (
        None
        if prefixes is None
        else tuple(prefixes)
    )

    frozen = 0

    for name, parameter in model.named_parameters():
        selected = (
            prefixes is None
            or any(
                name == prefix
                or name.startswith(prefix + ".")
                for prefix in prefixes
            )
        )

        if selected:
            parameter.requires_grad_(False)
            parameter.grad = None
            frozen += 1

    return frozen


def unfreeze_parameters(
    model: nn.Module,
    *,
    prefixes: Iterable[str] | None = None,
) -> int:
    prefixes = (
        None
        if prefixes is None
        else tuple(prefixes)
    )

    unfrozen = 0

    for name, parameter in model.named_parameters():
        selected = (
            prefixes is None
            or any(
                name == prefix
                or name.startswith(prefix + ".")
                for prefix in prefixes
            )
        )

        if selected:
            parameter.requires_grad_(True)
            unfrozen += 1

    return unfrozen


def _move_state_value(
    value: Any,
    device: torch.device,
) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)

    if isinstance(value, dict):
        return {
            key: _move_state_value(
                item,
                device,
            )
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _move_state_value(
                item,
                device,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _move_state_value(
                item,
                device,
            )
            for item in value
        )

    return value


def optimizer_to_device(
    optimizer: Optimizer,
    device: str | torch.device,
) -> None:
    """
    Move optimizer state tensors to device.

    Normally optimizer states should follow
    the device of their corresponding parameters.
    """
    device = torch.device(device)

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_state_value(
                value,
                device,
            )


def estimate_optimizer_state_memory(
    optimizer: Optimizer,
) -> dict[str, float]:
    """
    Estimate allocated optimizer-state memory.

    Does not include parameter or gradient
    storage, allocator overhead, or future
    state allocations.
    """
    seen: set[int] = set()
    total_bytes = 0

    def count_value(value: Any) -> None:
        nonlocal total_bytes

        if isinstance(value, Tensor):
            storage = value.untyped_storage()
            pointer = storage.data_ptr()

            if pointer not in seen:
                seen.add(pointer)
                total_bytes += storage.nbytes()

        elif isinstance(value, Mapping):
            for item in value.values():
                count_value(item)

        elif isinstance(value, (list, tuple)):
            for item in value:
                count_value(item)

    for state in optimizer.state.values():
        count_value(state)

    return {
        "bytes": float(total_bytes),
        "kilobytes": total_bytes / 1024,
        "megabytes": total_bytes / (1024**2),
        "gigabytes": total_bytes / (1024**3),
    }


def get_optimizer_statistics(
    optimizer: Optimizer,
) -> dict[str, Any]:
    parameters = get_optimizer_parameters(
        optimizer
    )

    return {
        "optimizer": optimizer.__class__.__name__,
        "parameter_groups": len(
            optimizer.param_groups
        ),
        "parameter_tensors": len(parameters),
        "parameter_count": sum(
            parameter.numel()
            for parameter in parameters
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in parameters
            if parameter.requires_grad
        ),
        "parameters_with_gradients": sum(
            parameter.numel()
            for parameter in parameters
            if parameter.grad is not None
        ),
        "learning_rates": get_learning_rates(
            optimizer
        ),
        "weight_decays": get_weight_decays(
            optimizer
        ),
        "gradient_norm": calculate_gradient_norm(
            parameters
        ),
        "gradients_finite": gradients_are_finite(
            parameters
        ),
        "optimizer_state_memory": (
            estimate_optimizer_state_memory(
                optimizer
            )
        ),
    }


def save_optimizer_state(
    optimizer: Optimizer,
    path: str | Path,
    *,
    step: int = 0,
    epoch: int = 0,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "extra": dict(extra or {}),
        },
        path,
    )


def load_optimizer_state(
    optimizer: Optimizer,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a trusted optimizer checkpoint.

    The optimizer must have compatible
    parameter groups and parameter ordering.
    """
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    return {
        "step": int(
            checkpoint.get("step", 0)
        ),
        "epoch": int(
            checkpoint.get("epoch", 0)
        ),
        "extra": dict(
            checkpoint.get("extra", {})
        ),
    }


class OptimizerStepHelper:
    """
    Training-step helper with AMP and
    gradient accumulation.

    This helper is intended for single-process
    training. For DDP gradient synchronization,
    use the project's distributed trainer.

    Usage:
        helper = OptimizerStepHelper(optimizer)

        with helper.autocast():
            loss = criterion(model(x), y)

        helper.backward(loss)
        helper.step()
        helper.flush()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        config: OptimizerUtilsConfig | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = (
            config or OptimizerUtilsConfig()
        )

        self.parameters = get_optimizer_parameters(
            optimizer
        )

        if not self.parameters:
            raise ValueError(
                "Optimizer has no parameters"
            )

        if device is None:
            device = self.parameters[0].device

        self.device = torch.device(device)
        self.device_type = self.device.type

        self.precision_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.precision_dtype]

        self.use_amp = (
            self.config.mixed_precision
            and self.device_type in ("cuda", "cpu")
            and self.precision_dtype
            in (torch.float16, torch.bfloat16)
        )

        if (
            self.use_amp
            and self.device_type == "cpu"
            and self.precision_dtype == torch.float16
        ):
            raise ValueError(
                "Use bfloat16 for CPU autocast"
            )

        self.use_scaler = (
            self.use_amp
            and self.device_type == "cuda"
            and self.precision_dtype == torch.float16
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_scaler,
        )

        self.accumulation_counter = 0
        self.optimizer_steps = 0
        self.skipped_steps = 0

        self.last_loss = 0.0
        self.last_gradient_norm = 0.0

        self.zero_grad()

    def autocast(self):
        if not self.use_amp:
            return nullcontext()

        return torch.autocast(
            device_type=self.device_type,
            dtype=self.precision_dtype,
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(
            set_to_none=(
                self.config.zero_grad_set_to_none
            )
        )

        self.accumulation_counter = 0

    def backward(
        self,
        loss: Tensor,
    ) -> None:
        if not isinstance(loss, Tensor):
            raise TypeError(
                "loss must be a Tensor"
            )

        if loss.ndim != 0:
            raise ValueError(
                "loss must be scalar"
            )

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                "Non-finite training loss"
            )

        self.last_loss = float(
            loss.detach().item()
        )

        scaled_loss = (
            loss
            / self.config.accumulation_steps
        )

        if self.use_scaler:
            self.scaler.scale(
                scaled_loss
            ).backward()
        else:
            scaled_loss.backward()

        self.accumulation_counter += 1

    def _perform_step(
        self,
        *,
        partial: bool = False,
    ) -> bool:
        if self.accumulation_counter == 0:
            return False

        if self.use_scaler:
            self.scaler.unscale_(
                self.optimizer
            )

        if partial:
            correction = (
                self.config.accumulation_steps
                / self.accumulation_counter
            )

            with torch.no_grad():
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(
                            correction
                        )

        if (
            self.config.detect_nonfinite
            and not gradients_are_finite(
                self.parameters
            )
        ):
            self.skipped_steps += 1

            if self.use_scaler:
                # GradScaler detects non-finite
                # gradients and updates its scale.
                self.scaler.step(
                    self.optimizer
                )
                self.scaler.update()

            self.zero_grad()

            if self.config.skip_nonfinite_steps:
                return False

            raise FloatingPointError(
                "Non-finite gradients detected"
            )

        self.last_gradient_norm = clip_gradients(
            self.parameters,
            max_norm=self.config.max_gradient_norm,
            clip_value=self.config.gradient_clip_value,
            norm_type=self.config.norm_type,
            error_if_nonfinite=(
                self.config.detect_nonfinite
            ),
        )

        if self.use_scaler:
            previous_scale = (
                self.scaler.get_scale()
            )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            updated = (
                self.scaler.get_scale()
                >= previous_scale
            )
        else:
            self.optimizer.step()
            updated = True

        if updated:
            self.optimizer_steps += 1
        else:
            self.skipped_steps += 1

        self.zero_grad()
        return updated

    def step(self) -> bool:
        if (
            self.accumulation_counter
            < self.config.accumulation_steps
        ):
            return False

        return self._perform_step()

    def flush(self) -> bool:
        """
        Apply remaining accumulated gradients
        at the end of an epoch.
        """
        if self.accumulation_counter == 0:
            return False

        return self._perform_step(
            partial=True
        )

    def train_step(
        self,
        loss: Tensor,
    ) -> dict[str, Any]:
        self.backward(loss)

        updated = self.step()

        return {
            "loss": self.last_loss,
            "optimizer_updated": updated,
            "optimizer_steps": self.optimizer_steps,
            "skipped_steps": self.skipped_steps,
            "accumulation_counter": (
                self.accumulation_counter
            ),
            "gradient_norm": (
                self.last_gradient_norm
            ),
            "learning_rates": get_learning_rates(
                self.optimizer
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "skipped_steps": self.skipped_steps,
            "accumulation_counter": (
                self.accumulation_counter
            ),
            "last_loss": self.last_loss,
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
            "mixed_precision": self.use_amp,
            "gradient_scaler_enabled": (
                self.use_scaler
            ),
            "optimizer": get_optimizer_statistics(
                self.optimizer
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "scaler": (
                self.scaler.state_dict()
                if self.use_scaler
                else None
            ),
            "optimizer_steps": self.optimizer_steps,
            "skipped_steps": self.skipped_steps,
            "last_loss": self.last_loss,
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
    ) -> None:
        self.optimizer.load_state_dict(
            state["optimizer"]
        )

        if (
            self.use_scaler
            and state.get("scaler") is not None
        ):
            self.scaler.load_state_dict(
                state["scaler"]
            )

        self.optimizer_steps = int(
            state.get("optimizer_steps", 0)
        )

        self.skipped_steps = int(
            state.get("skipped_steps", 0)
        )

        self.last_loss = float(
            state.get("last_loss", 0.0)
        )

        self.last_gradient_norm = float(
            state.get("last_gradient_norm", 0.0)
        )

        # Accumulated gradients are not included
        # in the optimizer state dictionary.
        self.zero_grad()


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
    )

    helper = OptimizerStepHelper(
        optimizer,
        OptimizerUtilsConfig(
            max_gradient_norm=1.0,
            accumulation_steps=2,
            mixed_precision=(
                device.type == "cuda"
            ),
            precision_dtype=(
                "float16"
                if device.type == "cuda"
                else "bfloat16"
            ),
        ),
        device=device,
    )

    model.train()

    for step in range(10):
        inputs = torch.randn(
            16,
            128,
            device=device,
        )

        targets = torch.randn(
            16,
            128,
            device=device,
        )

        with helper.autocast():
            predictions = model(inputs)

            loss = torch.nn.functional.mse_loss(
                predictions.float(),
                targets.float(),
            )

        metrics = helper.train_step(loss)

        print(
            f"Step {step + 1} | "
            f"Loss: {metrics['loss']:.6f} | "
            f"Updated: {metrics['optimizer_updated']} | "
            f"Grad norm: {metrics['gradient_norm']:.4f}"
        )

    helper.flush()

    print("\nModel parameters:")
    print(count_model_parameters(model))

    print("\nOptimizer statistics:")
    print(helper.get_statistics())
