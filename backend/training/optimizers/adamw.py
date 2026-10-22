
"""
NeuroWorld AI
File: training/optimizers/adamw.py

AdamW optimizer utilities for world-model training,
robotics, reinforcement learning, and generative AI.

Features:
- PyTorch AdamW optimizer
- Automatic weight-decay parameter grouping
- No decay for biases, normalization and embeddings
- Separate learning rates for model components
- Optional layer-wise learning-rate decay
- Gradient clipping
- Gradient accumulation
- Mixed precision (FP16 / BF16)
- Optimizer checkpoint save/load
- Learning-rate updates
- Optimizer statistics
- NaN/Inf gradient detection
- Optimizer factory

Requires:
    torch >= 2.1
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer


@dataclass
class AdamWConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    amsgrad: bool = False
    foreach: bool | None = None
    fused: bool | None = None

    maximize: bool = False
    capturable: bool = False

    exclude_bias_from_weight_decay: bool = True
    exclude_norm_from_weight_decay: bool = True
    exclude_embedding_from_weight_decay: bool = True

    layerwise_lr_decay: float = 1.0

    gradient_clip_norm: float | None = 1.0
    gradient_clip_value: float | None = None

    gradient_accumulation_steps: int = 1

    mixed_precision: bool = False
    precision_dtype: str = "float16"

    detect_nonfinite_gradients: bool = True
    skip_nonfinite_steps: bool = True

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.learning_rate)
            and self.learning_rate > 0
        ):
            raise ValueError(
                "learning_rate must be positive"
            )

        if not (
            math.isfinite(self.weight_decay)
            and self.weight_decay >= 0
        ):
            raise ValueError(
                "weight_decay must be nonnegative"
            )

        if not 0 <= self.beta1 < 1:
            raise ValueError(
                "beta1 must be in [0,1)"
            )

        if not 0 <= self.beta2 < 1:
            raise ValueError(
                "beta2 must be in [0,1)"
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if not 0 < self.layerwise_lr_decay <= 1:
            raise ValueError(
                "layerwise_lr_decay must be in (0,1]"
            )

        if (
            self.gradient_clip_norm is not None
            and self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be positive"
            )

        if (
            self.gradient_clip_value is not None
            and self.gradient_clip_value <= 0
        ):
            raise ValueError(
                "gradient_clip_value must be positive"
            )

        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1"
            )

        if self.precision_dtype not in {
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError(
                "Unsupported precision_dtype"
            )


def _is_norm_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.SyncBatchNorm,
            nn.LocalResponseNorm,
        ),
    )


def _parameter_module_map(
    model: nn.Module,
) -> dict[str, nn.Module]:
    result: dict[str, nn.Module] = {}

    for module_name, module in model.named_modules():
        for parameter_name, _ in module.named_parameters(
            recurse=False
        ):
            full_name = (
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )

            result[full_name] = module

    return result


def _use_weight_decay(
    name: str,
    parameter: nn.Parameter,
    module: nn.Module | None,
    config: AdamWConfig,
) -> bool:
    if parameter.ndim <= 1:
        return False

    if (
        config.exclude_bias_from_weight_decay
        and name.endswith(".bias")
    ):
        return False

    if (
        config.exclude_norm_from_weight_decay
        and module is not None
        and _is_norm_module(module)
    ):
        return False

    if (
        config.exclude_embedding_from_weight_decay
        and isinstance(module, nn.Embedding)
    ):
        return False

    return True


def build_adamw_parameter_groups(
    model: nn.Module,
    config: AdamWConfig,
    *,
    component_learning_rates: Mapping[str, float]
    | None = None,
) -> list[dict[str, Any]]:
    """
    Create parameter groups with selective
    weight decay and optional component LRs.

    component_learning_rates example:
        {
            "encoder": 1e-4,
            "dynamics": 3e-4,
            "policy": 1e-4,
        }

    Prefix matching uses module boundaries.
    """
    module_map = _parameter_module_map(model)

    component_learning_rates = dict(
        component_learning_rates or {}
    )

    for prefix, lr in component_learning_rates.items():
        if not prefix:
            raise ValueError(
                "Component prefix cannot be empty"
            )

        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(
                f"Invalid learning rate for {prefix}"
            )

    groups: dict[
        tuple[float, float],
        dict[str, Any],
    ] = {}

    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if id(parameter) in seen:
            continue

        seen.add(id(parameter))

        lr = config.learning_rate

        matching_prefixes = [
            prefix
            for prefix in component_learning_rates
            if (
                name == prefix
                or name.startswith(prefix + ".")
            )
        ]

        if matching_prefixes:
            longest = max(
                matching_prefixes,
                key=len,
            )

            lr = component_learning_rates[longest]

        module = module_map.get(name)

        decay = (
            config.weight_decay
            if _use_weight_decay(
                name,
                parameter,
                module,
                config,
            )
            else 0.0
        )

        key = (lr, decay)

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": decay,
                "parameter_names": [],
            }

        groups[key]["params"].append(parameter)
        groups[key]["parameter_names"].append(name)

    if not groups:
        raise ValueError(
            "Model has no trainable parameters"
        )

    return list(groups.values())


def create_adamw_optimizer(
    model: nn.Module,
    config: AdamWConfig | None = None,
    *,
    component_learning_rates: Mapping[str, float]
    | None = None,
) -> AdamW:
    config = config or AdamWConfig()

    parameter_groups = build_adamw_parameter_groups(
        model,
        config,
        component_learning_rates=(
            component_learning_rates
        ),
    )

    return AdamW(
        parameter_groups,
        lr=config.learning_rate,
        betas=(
            config.beta1,
            config.beta2,
        ),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
        foreach=config.foreach,
        fused=config.fused,
        maximize=config.maximize,
        capturable=config.capturable,
    )


def calculate_gradient_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: float = 2.0,
) -> float:
    gradients = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not gradients:
        return 0.0

    norms = [
        torch.linalg.vector_norm(
            gradient.float(),
            ord=norm_type,
        )
        for gradient in gradients
    ]

    total_norm = torch.linalg.vector_norm(
        torch.stack(norms),
        ord=norm_type,
    )

    return float(
        total_norm.item()
    )


def gradients_are_finite(
    parameters: Iterable[nn.Parameter],
) -> bool:
    for parameter in parameters:
        gradient = parameter.grad

        if gradient is None:
            continue

        if not torch.isfinite(
            gradient
        ).all().item():
            return False

    return True


class AdamWManager:
    """
    Training manager wrapping torch.optim.AdamW.

    Supports:
        - Mixed precision
        - Gradient accumulation
        - Gradient clipping
        - Checkpointing
        - Learning-rate management
        - Optimizer statistics

    Usage:
        manager = AdamWManager(model)

        for batch in dataloader:
            with manager.autocast():
                loss = model_loss(batch)

            manager.backward(loss)
            manager.step()

        manager.flush()
    """

    def __init__(
        self,
        model: nn.Module,
        config: AdamWConfig | None = None,
        *,
        component_learning_rates: Mapping[str, float]
        | None = None,
    ) -> None:
        self.model = model

        self.config = (
            config or AdamWConfig()
        )

        self.optimizer = create_adamw_optimizer(
            model,
            self.config,
            component_learning_rates=(
                component_learning_rates
            ),
        )

        self.parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        try:
            self.device = next(
                model.parameters()
            ).device
        except StopIteration as exc:
            raise ValueError(
                "Model has no parameters"
            ) from exc

        self.device_type = self.device.type

        self.autocast_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.precision_dtype]

        self.use_amp = (
            self.config.mixed_precision
            and self.device_type in {
                "cuda",
                "cpu",
            }
            and self.autocast_dtype
            in {
                torch.float16,
                torch.bfloat16,
            }
        )

        if (
            self.use_amp
            and self.device_type == "cpu"
            and self.autocast_dtype == torch.float16
        ):
            raise ValueError(
                "Use bfloat16 for CPU mixed precision"
            )

        self.use_scaler = (
            self.use_amp
            and self.device_type == "cuda"
            and self.autocast_dtype == torch.float16
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_scaler,
        )

        self.accumulation_counter = 0
        self.optimizer_steps = 0
        self.skipped_steps = 0

        self.last_gradient_norm = 0.0
        self.last_loss = 0.0

        self.optimizer.zero_grad(
            set_to_none=True
        )

    def autocast(self):
        if not self.use_amp:
            return nullcontext()

        return torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=True,
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(
            set_to_none=True
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
            / self.config.gradient_accumulation_steps
        )

        if self.use_scaler:
            self.scaler.scale(
                scaled_loss
            ).backward()
        else:
            scaled_loss.backward()

        self.accumulation_counter += 1

    def _clip_gradients(self) -> float:
        if self.config.gradient_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(
                self.parameters,
                max_norm=(
                    self.config.gradient_clip_norm
                ),
                error_if_nonfinite=False,
            )

            gradient_norm = float(
                norm.item()
            )

        else:
            gradient_norm = calculate_gradient_norm(
                self.parameters
            )

        if self.config.gradient_clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                self.parameters,
                clip_value=(
                    self.config.gradient_clip_value
                ),
            )

        return gradient_norm

    def _optimizer_step(
        self,
        *,
        partial_accumulation: bool = False,
    ) -> bool:
        if self.accumulation_counter == 0:
            return False

        if self.use_scaler:
            self.scaler.unscale_(
                self.optimizer
            )

        # Correct the gradient scale when
        # flushing an incomplete accumulation.
        if partial_accumulation:
            correction = (
                self.config.gradient_accumulation_steps
                / self.accumulation_counter
            )

            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(
                        correction
                    )

        if self.config.detect_nonfinite_gradients:
            if not gradients_are_finite(
                self.parameters
            ):
                self.skipped_steps += 1

                if self.use_scaler:
                    # GradScaler can update its
                    # dynamic scale after an inf/NaN.
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

        self.last_gradient_norm = (
            self._clip_gradients()
        )

        if self.use_scaler:
            old_scale = self.scaler.get_scale()

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            new_scale = self.scaler.get_scale()

            # A reduced scale indicates the
            # optimizer step was skipped.
            updated = (
                new_scale >= old_scale
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
        """
        Perform an optimizer update when
        the accumulation target is reached.

        Returns True only when an update
        actually occurs.
        """
        if (
            self.accumulation_counter
            < self.config.gradient_accumulation_steps
        ):
            return False

        return self._optimizer_step()

    def flush(self) -> bool:
        """
        Apply pending accumulated gradients
        at the end of an epoch.

        In distributed training, ensure all
        workers flush consistently.
        """
        if self.accumulation_counter == 0:
            return False

        return self._optimizer_step(
            partial_accumulation=True
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
            "accumulation_counter": (
                self.accumulation_counter
            ),
            "gradient_norm": (
                self.last_gradient_norm
            ),
            "learning_rates": (
                self.get_learning_rates()
            ),
        }

    def get_learning_rates(
        self,
    ) -> list[float]:
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def set_learning_rate(
        self,
        learning_rate: float,
        *,
        group_index: int | None = None,
    ) -> None:
        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be positive"
            )

        if group_index is None:
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        else:
            self.optimizer.param_groups[
                group_index
            ]["lr"] = learning_rate

    def multiply_learning_rate(
        self,
        factor: float,
    ) -> None:
        if (
            not math.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError(
                "factor must be positive"
            )

        for group in self.optimizer.param_groups:
            group["lr"] *= factor

    def get_optimizer_statistics(
        self,
    ) -> dict[str, Any]:
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters
        )

        parameters_with_gradients = sum(
            parameter.numel()
            for parameter in self.parameters
            if parameter.grad is not None
        )

        return {
            "optimizer": "AdamW",
            "device": str(self.device),
            "trainable_parameters": (
                trainable_parameters
            ),
            "parameters_with_gradients": (
                parameters_with_gradients
            ),
            "parameter_groups": len(
                self.optimizer.param_groups
            ),
            "learning_rates": (
                self.get_learning_rates()
            ),
            "weight_decays": [
                float(group["weight_decay"])
                for group in self.optimizer.param_groups
            ],
            "optimizer_steps": (
                self.optimizer_steps
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "accumulation_counter": (
                self.accumulation_counter
            ),
            "gradient_norm": (
                self.last_gradient_norm
            ),
            "last_loss": self.last_loss,
            "mixed_precision": self.use_amp,
            "precision_dtype": (
                self.config.precision_dtype
            ),
            "gradient_scaler_enabled": (
                self.use_scaler
            ),
        }

    def state_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "optimizer": (
                self.optimizer.state_dict()
            ),
            "scaler": (
                self.scaler.state_dict()
                if self.use_scaler
                else None
            ),
            "optimizer_steps": (
                self.optimizer_steps
            ),
            "skipped_steps": (
                self.skipped_steps
            ),
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
            "last_loss": self.last_loss,
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
            state.get(
                "optimizer_steps",
                0,
            )
        )

        self.skipped_steps = int(
            state.get(
                "skipped_steps",
                0,
            )
        )

        self.last_gradient_norm = float(
            state.get(
                "last_gradient_norm",
                0.0,
            )
        )

        self.last_loss = float(
            state.get(
                "last_loss",
                0.0,
            )
        )

        # Pending gradients are not included
        # in optimizer.state_dict().
        self.zero_grad()

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        include_model: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "optimizer_manager": (
                self.state_dict()
            ),
            "extra": dict(extra or {}),
        }

        if include_model:
            checkpoint["model"] = (
                self.model.state_dict()
            )

        torch.save(
            checkpoint,
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_model: bool = True,
        strict: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        if load_model:
            if "model" not in checkpoint:
                raise KeyError(
                    "Checkpoint has no model state"
                )

            self.model.load_state_dict(
                checkpoint["model"],
                strict=strict,
            )

        self.load_state_dict(
            checkpoint["optimizer_manager"]
        )

        return dict(
            checkpoint.get(
                "extra",
                {},
            )
        )


def create_adamw_manager(
    model: nn.Module,
    *,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    gradient_clip_norm: float | None = 1.0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = False,
    precision_dtype: str = "float16",
    component_learning_rates: Mapping[str, float]
    | None = None,
) -> AdamWManager:
    config = AdamWConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_norm=gradient_clip_norm,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        mixed_precision=mixed_precision,
        precision_dtype=precision_dtype,
    )

    return AdamWManager(
        model,
        config,
        component_learning_rates=(
            component_learning_rates
        ),
    )


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
            states: Tensor,
            actions: Tensor,
        ) -> Tensor:
            latent = self.encoder(states)

            features = torch.cat(
                [latent, actions],
                dim=-1,
            )

            return self.dynamics(
                features
            )

    model = DemoWorldModel().to(
        device
    )

    manager = create_adamw_manager(
        model,
        learning_rate=3e-4,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=2,
        mixed_precision=(
            device.type == "cuda"
        ),
        precision_dtype=(
            "float16"
            if device.type == "cuda"
            else "bfloat16"
        ),
        component_learning_rates={
            "encoder": 1e-4,
            "dynamics": 3e-4,
        },
    )

    model.train()

    for step in range(10):
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
            + 0.05 * torch.randn_like(states)
        )

        with manager.autocast():
            predictions = model(
                states,
                actions,
            )

            loss = F.mse_loss(
                predictions.float(),
                targets.float(),
            )

        result = manager.train_step(
            loss
        )

        print(
            f"Step {step + 1} | "
            f"Loss: {result['loss']:.6f} | "
            f"Updated: "
            f"{result['optimizer_updated']} | "
            f"Grad norm: "
            f"{result['gradient_norm']:.4f}"
        )

    manager.flush()

    print(
        "\nOptimizer statistics:"
    )

    print(
        manager.get_optimizer_statistics()
    )
