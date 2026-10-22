
"""
NeuroWorld AI
File: training/optimizers/gradient_clipping.py

Gradient clipping utilities for:
- World-model training
- Transformer architectures
- Video prediction
- Reinforcement learning
- Robotics and Physical AI
- Distributed PyTorch training

Features:
- Global gradient norm clipping
- Gradient value clipping
- Adaptive Gradient Clipping (AGC)
- Per-parameter gradient clipping
- Per-layer gradient clipping
- Gradient norm monitoring
- NaN/Inf gradient detection
- Optional non-finite gradient sanitization
- Gradient statistics and logging
- AMP GradScaler integration
- Configurable clipping strategies

Requirements:
    torch >= 2.1

Important:
    Call gradient clipping after backward().
    For FP16 AMP, call scaler.unscale_(optimizer)
    before clipping and scaler.step(optimizer)
    afterward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


ClippingMethod = Literal[
    "norm",
    "value",
    "adaptive",
    "per_parameter",
    "per_layer",
    "none",
]


@dataclass
class GradientClippingConfig:
    method: ClippingMethod = "norm"

    max_norm: float = 1.0
    norm_type: float = 2.0
    clip_value: float = 1.0

    adaptive_clip_factor: float = 0.01
    adaptive_epsilon: float = 1e-3

    per_parameter_max_norm: float = 1.0
    per_layer_max_norm: float = 1.0

    detect_nonfinite: bool = True
    sanitize_nonfinite: bool = False

    nan_replacement: float = 0.0
    posinf_replacement: float = 0.0
    neginf_replacement: float = 0.0

    error_if_nonfinite: bool = False
    track_statistics: bool = True

    def __post_init__(self) -> None:
        if self.method not in {
            "norm",
            "value",
            "adaptive",
            "per_parameter",
            "per_layer",
            "none",
        }:
            raise ValueError(
                f"Unsupported clipping method: {self.method}"
            )

        positive_fields = (
            "max_norm",
            "norm_type",
            "clip_value",
            "adaptive_clip_factor",
            "adaptive_epsilon",
            "per_parameter_max_norm",
            "per_layer_max_norm",
        )

        for name in positive_fields:
            value = getattr(self, name)

            if (
                math.isnan(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive"
                )

        for name in (
            "nan_replacement",
            "posinf_replacement",
            "neginf_replacement",
        ):
            if not math.isfinite(
                getattr(self, name)
            ):
                raise ValueError(
                    f"{name} must be finite"
                )


@dataclass
class GradientClippingResult:
    method: str
    applied: bool
    finite_before: bool
    finite_after: bool

    norm_before: float
    norm_after: float

    clipped_parameters: int
    total_parameters: int
    parameters_with_gradients: int

    nan_count: int = 0
    inf_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "applied": self.applied,
            "finite_before": self.finite_before,
            "finite_after": self.finite_after,
            "norm_before": self.norm_before,
            "norm_after": self.norm_after,
            "clipped_parameters": (
                self.clipped_parameters
            ),
            "total_parameters": (
                self.total_parameters
            ),
            "parameters_with_gradients": (
                self.parameters_with_gradients
            ),
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
        }


def _trainable_parameters(
    model_or_parameters: (
        nn.Module
        | Iterable[nn.Parameter]
    ),
) -> list[nn.Parameter]:
    if isinstance(
        model_or_parameters,
        nn.Module,
    ):
        parameters = (
            model_or_parameters.parameters()
        )
    else:
        parameters = model_or_parameters

    result = []
    seen = set()

    for parameter in parameters:
        if not isinstance(
            parameter,
            nn.Parameter,
        ):
            raise TypeError(
                "Expected nn.Parameter objects"
            )

        if not parameter.requires_grad:
            continue

        if id(parameter) in seen:
            continue

        seen.add(id(parameter))
        result.append(parameter)

    return result


def _gradient_parameters(
    parameters: Iterable[nn.Parameter],
) -> list[nn.Parameter]:
    return [
        parameter
        for parameter in parameters
        if parameter.grad is not None
    ]


def _gradient_values(
    parameter: nn.Parameter,
) -> Tensor:
    gradient = parameter.grad

    if gradient is None:
        raise ValueError(
            "Parameter has no gradient"
        )

    if gradient.is_sparse:
        return gradient.coalesce().values()

    return gradient


@torch.no_grad()
def calculate_global_gradient_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: float = 2.0,
) -> Tensor:
    """
    Calculate the global gradient norm.

    Returns a scalar Tensor on the same
    device as the first gradient.

    Supports dense and sparse gradients.
    """
    gradients = []

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradients.append(
            _gradient_values(parameter)
        )

    if not gradients:
        return torch.tensor(0.0)

    device = gradients[0].device

    if norm_type == float("inf"):
        norms = [
            gradient.detach().abs().max()
            if gradient.numel()
            else torch.zeros(
                (),
                device=gradient.device,
            )
            for gradient in gradients
        ]

        return torch.stack(
            [
                norm.to(
                    device=device,
                    dtype=torch.float32,
                )
                for norm in norms
            ]
        ).max()

    norms = [
        torch.linalg.vector_norm(
            gradient.detach().float(),
            ord=norm_type,
        ).to(device)
        for gradient in gradients
    ]

    return torch.linalg.vector_norm(
        torch.stack(norms),
        ord=norm_type,
    )


@torch.no_grad()
def gradients_are_finite(
    parameters: Iterable[nn.Parameter],
) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = _gradient_values(
            parameter
        )

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
        if parameter.grad is None:
            continue

        gradient = _gradient_values(
            parameter
        )

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
def sanitize_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> int:
    """
    Replace NaN/Inf gradient entries.

    Warning:
        Sanitization can conceal training
        instability. Prefer skipping the
        optimizer update when non-finite
        gradients are detected.
    """
    modified = 0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = _gradient_values(
            parameter
        )

        invalid = ~torch.isfinite(
            gradient
        )

        if invalid.any().item():
            gradient.nan_to_num_(
                nan=nan,
                posinf=posinf,
                neginf=neginf,
            )

            modified += 1

    return modified


@torch.no_grad()
def clip_gradient_norm(
    parameters: Iterable[nn.Parameter],
    max_norm: float = 1.0,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> Tensor:
    """
    Global norm clipping.

    Returns the total gradient norm
    before clipping.
    """
    parameters = list(parameters)

    if max_norm <= 0:
        raise ValueError(
            "max_norm must be positive"
        )

    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )


@torch.no_grad()
def clip_gradient_value(
    parameters: Iterable[nn.Parameter],
    clip_value: float = 1.0,
) -> None:
    if clip_value <= 0:
        raise ValueError(
            "clip_value must be positive"
        )

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = _gradient_values(
            parameter
        )

        gradient.clamp_(
            min=-clip_value,
            max=clip_value,
        )


def _unitwise_norm(
    tensor: Tensor,
) -> Tensor:
    """
    Unit-wise norm for Adaptive Gradient
    Clipping (AGC).

    Shapes:
        [D]          -> scalar
        [O,I]        -> [O,1]
        [O,I,H,W]    -> [O,1,1,1]
    """
    if tensor.ndim <= 1:
        return torch.linalg.vector_norm(
            tensor.float()
        )

    if tensor.ndim in (2, 3):
        return torch.linalg.vector_norm(
            tensor.float(),
            dim=1,
            keepdim=True,
        )

    dimensions = tuple(
        range(1, tensor.ndim)
    )

    return torch.linalg.vector_norm(
        tensor.float(),
        dim=dimensions,
        keepdim=True,
    )


@torch.no_grad()
def adaptive_gradient_clipping(
    parameters: Iterable[nn.Parameter],
    *,
    clip_factor: float = 0.01,
    epsilon: float = 1e-3,
) -> int:
    """
    Adaptive Gradient Clipping.

    Clips gradients relative to the
    corresponding parameter norms.

    Reference concept:
        AGC / unit-wise gradient clipping.

    Returns:
        Number of parameter tensors
        containing clipped gradient units.
    """
    if clip_factor <= 0:
        raise ValueError(
            "clip_factor must be positive"
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive"
        )

    clipped_count = 0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad

        if gradient.is_sparse:
            raise ValueError(
                "AGC does not support sparse gradients"
            )

        parameter_norm = _unitwise_norm(
            parameter.detach()
        )

        gradient_norm = _unitwise_norm(
            gradient.detach()
        )

        max_norm = (
            parameter_norm.clamp_min(epsilon)
            * clip_factor
        )

        clipping_needed = (
            gradient_norm > max_norm
        )

        if not clipping_needed.any().item():
            continue

        scale = (
            max_norm
            / gradient_norm.clamp_min(1e-12)
        ).clamp(max=1.0)

        gradient.mul_(
            scale.to(
                dtype=gradient.dtype,
                device=gradient.device,
            )
        )

        clipped_count += 1

    return clipped_count


@torch.no_grad()
def clip_per_parameter_norm(
    parameters: Iterable[nn.Parameter],
    *,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> int:
    """
    Independently clip each parameter tensor.
    """
    if max_norm <= 0:
        raise ValueError(
            "max_norm must be positive"
        )

    clipped_count = 0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = _gradient_values(
            parameter
        )

        norm = torch.linalg.vector_norm(
            gradient.float(),
            ord=norm_type,
        )

        if norm.item() > max_norm:
            scale = (
                max_norm
                / norm.clamp_min(1e-12)
            )

            gradient.mul_(
                scale.to(
                    gradient.dtype
                )
            )

            clipped_count += 1

    return clipped_count


@torch.no_grad()
def clip_per_layer_norm(
    model: nn.Module,
    *,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> int:
    """
    Clip each module's directly owned
    parameters as one gradient group.

    Shared parameters are clipped once.
    """
    if max_norm <= 0:
        raise ValueError(
            "max_norm must be positive"
        )

    clipped_count = 0
    seen: set[int] = set()

    for module in model.modules():
        parameters = []

        for parameter in module.parameters(
            recurse=False
        ):
            if (
                not parameter.requires_grad
                or parameter.grad is None
                or id(parameter) in seen
            ):
                continue

            seen.add(id(parameter))
            parameters.append(parameter)

        if not parameters:
            continue

        old_norm = calculate_global_gradient_norm(
            parameters,
            norm_type=norm_type,
        )

        if old_norm.item() > max_norm:
            clip_gradient_norm(
                parameters,
                max_norm=max_norm,
                norm_type=norm_type,
            )

            clipped_count += len(parameters)

    return clipped_count


class GradientClipper:
    """
    Configurable gradient clipping manager.

    Usage:
        clipper = GradientClipper(
            GradientClippingConfig(
                method="norm",
                max_norm=1.0,
            )
        )

        loss.backward()

        result = clipper.clip(model)

        optimizer.step()

    FP16 AMP usage:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clipper.clip(model)
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(
        self,
        config: GradientClippingConfig | None = None,
    ) -> None:
        self.config = (
            config or GradientClippingConfig()
        )

        self.step_count = 0
        self.total_clipped_parameters = 0
        self.nonfinite_steps = 0

        self.last_result: (
            GradientClippingResult | None
        ) = None

        self.history: list[
            dict[str, Any]
        ] = []

    def clip(
        self,
        model_or_parameters: (
            nn.Module
            | Iterable[nn.Parameter]
        ),
        *,
        optimizer: Optimizer | None = None,
        scaler: Any | None = None,
        already_unscaled: bool = False,
    ) -> GradientClippingResult:
        """
        Clip gradients according to config.

        If scaler is provided, the gradients
        are unscaled before clipping.

        Do not call scaler.unscale_(optimizer)
        twice during the same optimizer step.

        The caller remains responsible for
        scaler.step(), scaler.update(),
        and optimizer.zero_grad().
        """
        parameters = _trainable_parameters(
            model_or_parameters
        )

        gradient_parameters = (
            _gradient_parameters(parameters)
        )

        if (
            scaler is not None
            and not already_unscaled
        ):
            if optimizer is None:
                raise ValueError(
                    "optimizer is required when "
                    "using GradScaler"
                )

            scaler.unscale_(optimizer)

        counts = count_nonfinite_gradients(
            gradient_parameters
        )

        finite_before = (
            counts["nan_count"] == 0
            and counts["inf_count"] == 0
        )

        if not finite_before:
            self.nonfinite_steps += 1

            if self.config.sanitize_nonfinite:
                sanitize_gradients(
                    gradient_parameters,
                    nan=self.config.nan_replacement,
                    posinf=self.config.posinf_replacement,
                    neginf=self.config.neginf_replacement,
                )

            elif self.config.error_if_nonfinite:
                raise FloatingPointError(
                    "Non-finite gradients detected"
                )

        finite_after_sanitization = (
            gradients_are_finite(
                gradient_parameters
            )
        )

        norm_before = (
            calculate_global_gradient_norm(
                gradient_parameters,
                norm_type=self.config.norm_type,
            )
        )

        norm_before_value = float(
            norm_before.item()
        )

        clipped_count = 0
        applied = False

        # Avoid modifying non-finite gradients
        # unless sanitization was explicitly
        # requested.
        if (
            finite_after_sanitization
            and gradient_parameters
        ):
            method = self.config.method

            if method == "norm":
                old_norm = clip_gradient_norm(
                    gradient_parameters,
                    max_norm=self.config.max_norm,
                    norm_type=self.config.norm_type,
                    error_if_nonfinite=(
                        self.config.error_if_nonfinite
                    ),
                )

                if old_norm.item() > self.config.max_norm:
                    clipped_count = len(
                        gradient_parameters
                    )

                applied = True

            elif method == "value":
                clipped_count = sum(
                    int(
                        (
                            _gradient_values(parameter)
                            .abs()
                            > self.config.clip_value
                        ).any().item()
                    )
                    for parameter in gradient_parameters
                )

                clip_gradient_value(
                    gradient_parameters,
                    clip_value=self.config.clip_value,
                )

                applied = True

            elif method == "adaptive":
                clipped_count = (
                    adaptive_gradient_clipping(
                        gradient_parameters,
                        clip_factor=(
                            self.config.adaptive_clip_factor
                        ),
                        epsilon=(
                            self.config.adaptive_epsilon
                        ),
                    )
                )

                applied = True

            elif method == "per_parameter":
                clipped_count = (
                    clip_per_parameter_norm(
                        gradient_parameters,
                        max_norm=(
                            self.config.per_parameter_max_norm
                        ),
                        norm_type=self.config.norm_type,
                    )
                )

                applied = True

            elif method == "per_layer":
                if not isinstance(
                    model_or_parameters,
                    nn.Module,
                ):
                    raise TypeError(
                        "Per-layer clipping requires "
                        "an nn.Module"
                    )

                clipped_count = (
                    clip_per_layer_norm(
                        model_or_parameters,
                        max_norm=(
                            self.config.per_layer_max_norm
                        ),
                        norm_type=self.config.norm_type,
                    )
                )

                applied = True

        norm_after = (
            calculate_global_gradient_norm(
                gradient_parameters,
                norm_type=self.config.norm_type,
            )
        )

        finite_after = gradients_are_finite(
            gradient_parameters
        )

        self.step_count += 1

        self.total_clipped_parameters += (
            clipped_count
        )

        result = GradientClippingResult(
            method=self.config.method,
            applied=applied,
            finite_before=finite_before,
            finite_after=finite_after,
            norm_before=norm_before_value,
            norm_after=float(
                norm_after.item()
            ),
            clipped_parameters=clipped_count,
            total_parameters=len(parameters),
            parameters_with_gradients=len(
                gradient_parameters
            ),
            nan_count=counts["nan_count"],
            inf_count=counts["inf_count"],
        )

        self.last_result = result

        if self.config.track_statistics:
            self.history.append(
                result.to_dict()
            )

        return result

    def should_skip_optimizer_step(self) -> bool:
        """
        True when the most recent gradients
        are still non-finite.

        If using GradScaler, let GradScaler
        manage its own skip/update behavior.
        """
        if self.last_result is None:
            return False

        return not self.last_result.finite_after

    def get_last_metrics(
        self,
    ) -> dict[str, Any]:
        if self.last_result is None:
            return {}

        return self.last_result.to_dict()

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        if not self.history:
            return {
                "steps": self.step_count,
                "nonfinite_steps": (
                    self.nonfinite_steps
                ),
                "total_clipped_parameters": (
                    self.total_clipped_parameters
                ),
            }

        finite_norms = [
            entry["norm_before"]
            for entry in self.history
            if math.isfinite(
                entry["norm_before"]
            )
        ]

        mean_norm = (
            sum(finite_norms)
            / len(finite_norms)
            if finite_norms
            else 0.0
        )

        return {
            "steps": self.step_count,
            "nonfinite_steps": (
                self.nonfinite_steps
            ),
            "total_clipped_parameters": (
                self.total_clipped_parameters
            ),
            "mean_gradient_norm_before": (
                mean_norm
            ),
            "last_gradient_norm": (
                self.history[-1]["norm_after"]
            ),
            "clipped_steps": sum(
                entry["clipped_parameters"] > 0
                for entry in self.history
            ),
        }

    def reset_statistics(self) -> None:
        self.step_count = 0
        self.total_clipped_parameters = 0
        self.nonfinite_steps = 0
        self.last_result = None
        self.history.clear()

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "total_clipped_parameters": (
                self.total_clipped_parameters
            ),
            "nonfinite_steps": (
                self.nonfinite_steps
            ),
            "history": list(self.history),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        self.step_count = int(
            state.get("step_count", 0)
        )

        self.total_clipped_parameters = int(
            state.get(
                "total_clipped_parameters",
                0,
            )
        )

        self.nonfinite_steps = int(
            state.get("nonfinite_steps", 0)
        )

        self.history = list(
            state.get("history", [])
        )

        self.last_result = None


def create_gradient_clipper(
    *,
    method: ClippingMethod = "norm",
    max_norm: float = 1.0,
    clip_value: float = 1.0,
    adaptive_clip_factor: float = 0.01,
    detect_nonfinite: bool = True,
) -> GradientClipper:
    return GradientClipper(
        GradientClippingConfig(
            method=method,
            max_norm=max_norm,
            clip_value=clip_value,
            adaptive_clip_factor=(
                adaptive_clip_factor
            ),
            detect_nonfinite=detect_nonfinite,
        )
    )


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
    )

    clipper = GradientClipper(
        GradientClippingConfig(
            method="norm",
            max_norm=1.0,
            detect_nonfinite=True,
            sanitize_nonfinite=False,
        )
    )

    model.train()

    for step in range(5):
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

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(inputs)

        loss = F.mse_loss(
            predictions,
            targets,
        )

        loss.backward()

        result = clipper.clip(model)

        if not clipper.should_skip_optimizer_step():
            optimizer.step()

        print(
            f"Step {step + 1} | "
            f"Loss: {loss.item():.6f} | "
            f"Grad before: "
            f"{result.norm_before:.6f} | "
            f"Grad after: "
            f"{result.norm_after:.6f} | "
            f"Clipped: "
            f"{result.clipped_parameters}"
        )

    print(
        "\nGradient clipping statistics:"
    )

    print(
        clipper.get_statistics()
    )
