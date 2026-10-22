
"""
NeuroWorld AI
File: training/optimizers/lion.py

Lion (EvoLved Sign Momentum) optimizer.

Features:
- Lion optimizer implementation
- Decoupled weight decay
- Momentum-based sign updates
- Parameter groups
- Automatic weight-decay exclusions
- Component-specific learning rates
- Gradient clipping
- Gradient accumulation
- FP16/BF16 mixed precision
- Optimizer state checkpointing
- Gradient and optimizer statistics
- NaN/Inf gradient detection

Reference:
    Symbolic Discovery of Optimization Algorithms
    Chen et al., 2023

Lion update:
    c_t = beta1 * m_(t-1) + (1-beta1) * g_t
    theta_t = theta_(t-1) - lr * sign(c_t)
    m_t = beta2 * m_(t-1) + (1-beta2) * g_t

Weight decay is applied separately.

Requirements:
    PyTorch >= 2.1
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer with decoupled weight decay.

    Args:
        params:
            Iterable of parameters or parameter groups.
        lr:
            Learning rate.
        betas:
            Momentum coefficients.
        weight_decay:
            Decoupled weight decay.
        maximize:
            Maximize the objective instead of minimizing.

    Notes:
        Lion uses sign-based parameter updates,
        so learning rates typically need to be
        smaller than AdamW learning rates.

        Sparse gradients are not supported.
    """

    def __init__(
        self,
        params: Iterable[
            nn.Parameter | dict[str, Any]
        ],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
        maximize: bool = False,
    ) -> None:
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(
                "Learning rate must be positive"
            )

        if (
            not math.isfinite(weight_decay)
            or weight_decay < 0
        ):
            raise ValueError(
                "Weight decay must be nonnegative"
            )

        if len(betas) != 2:
            raise ValueError(
                "betas must contain two values"
            )

        for beta in betas:
            if not 0 <= beta < 1:
                raise ValueError(
                    "Each beta must be in [0,1)"
                )

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "maximize": maximize,
        }

        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(
        self,
        closure=None,
    ) -> Tensor | None:
        """
        Perform one Lion optimization step.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            for parameter in group["params"]:
                gradient = parameter.grad

                if gradient is None:
                    continue

                if gradient.is_sparse:
                    raise RuntimeError(
                        "Lion does not support "
                        "sparse gradients"
                    )

                if maximize:
                    gradient = -gradient

                state = self.state[parameter]

                if len(state) == 0:
                    state["exp_avg"] = (
                        torch.zeros_like(
                            parameter,
                            memory_format=(
                                torch.preserve_format
                            ),
                        )
                    )

                momentum = state["exp_avg"]

                # Decoupled weight decay.
                if weight_decay != 0:
                    parameter.mul_(
                        1.0 - lr * weight_decay
                    )

                # c_t = beta1*m + (1-beta1)*g
                update = momentum.mul(
                    beta1
                ).add(
                    gradient,
                    alpha=1.0 - beta1,
                )

                # theta = theta - lr*sign(c_t)
                parameter.add_(
                    update.sign(),
                    alpha=-lr,
                )

                # m_t = beta2*m + (1-beta2)*g
                momentum.mul_(
                    beta2
                ).add_(
                    gradient,
                    alpha=1.0 - beta2,
                )

        return loss


@dataclass
class LionConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    beta1: float = 0.9
    beta2: float = 0.99

    maximize: bool = False

    exclude_bias_from_weight_decay: bool = True
    exclude_norm_from_weight_decay: bool = True
    exclude_embedding_from_weight_decay: bool = True

    gradient_clip_norm: float | None = 1.0
    gradient_clip_value: float | None = None

    gradient_accumulation_steps: int = 1

    mixed_precision: bool = False
    precision_dtype: str = "float16"

    detect_nonfinite_gradients: bool = True
    skip_nonfinite_steps: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "learning_rate must be positive"
            )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
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


def _module_map(
    model: nn.Module,
) -> dict[str, nn.Module]:
    mapping: dict[str, nn.Module] = {}

    for module_name, module in model.named_modules():
        for name, _ in module.named_parameters(
            recurse=False
        ):
            full_name = (
                f"{module_name}.{name}"
                if module_name
                else name
            )

            mapping[full_name] = module

    return mapping


def _is_normalization(
    module: nn.Module,
) -> bool:
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


def _should_decay(
    name: str,
    parameter: nn.Parameter,
    module: nn.Module | None,
    config: LionConfig,
) -> bool:
    if parameter.ndim <= 1:
        return False

    if (
        config.exclude_bias_from_weight_decay
        and (
            name == "bias"
            or name.endswith(".bias")
        )
    ):
        return False

    if (
        config.exclude_norm_from_weight_decay
        and module is not None
        and _is_normalization(module)
    ):
        return False

    if (
        config.exclude_embedding_from_weight_decay
        and isinstance(module, nn.Embedding)
    ):
        return False

    return True


def build_lion_parameter_groups(
    model: nn.Module,
    config: LionConfig,
    *,
    component_learning_rates: Mapping[
        str, float
    ] | None = None,
) -> list[dict[str, Any]]:
    """
    Build Lion parameter groups.

    Supports:
        - Selective weight decay
        - Encoder-specific learning rates
        - Dynamics-specific learning rates
        - Policy-specific learning rates
        - Reward-model-specific learning rates
    """
    component_learning_rates = dict(
        component_learning_rates or {}
    )

    for name, learning_rate in (
        component_learning_rates.items()
    ):
        if not name:
            raise ValueError(
                "Component prefix cannot be empty"
            )

        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError(
                f"Invalid learning rate for {name}"
            )

    modules = _module_map(model)

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

        learning_rate = config.learning_rate

        matching_prefixes = [
            prefix
            for prefix in component_learning_rates
            if (
                name == prefix
                or name.startswith(prefix + ".")
            )
        ]

        if matching_prefixes:
            prefix = max(
                matching_prefixes,
                key=len,
            )

            learning_rate = (
                component_learning_rates[prefix]
            )

        module = modules.get(name)

        weight_decay = (
            config.weight_decay
            if _should_decay(
                name,
                parameter,
                module,
                config,
            )
            else 0.0
        )

        key = (
            learning_rate,
            weight_decay,
        )

        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "parameter_names": [],
            }

        groups[key]["params"].append(
            parameter
        )

        groups[key]["parameter_names"].append(
            name
        )

    if not groups:
        raise ValueError(
            "No trainable parameters found"
        )

    return list(
        groups.values()
    )


def create_lion_optimizer(
    model: nn.Module,
    config: LionConfig | None = None,
    *,
    component_learning_rates: Mapping[
        str, float
    ] | None = None,
) -> Lion:
    config = config or LionConfig()

    parameter_groups = (
        build_lion_parameter_groups(
            model,
            config,
            component_learning_rates=(
                component_learning_rates
            ),
        )
    )

    return Lion(
        parameter_groups,
        lr=config.learning_rate,
        betas=(
            config.beta1,
            config.beta2,
        ),
        weight_decay=config.weight_decay,
        maximize=config.maximize,
    )


@torch.no_grad()
def calculate_gradient_norm(
    parameters: Iterable[nn.Parameter],
) -> float:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not gradients:
        return 0.0

    norms = [
        torch.linalg.vector_norm(
            gradient
        )
        for gradient in gradients
    ]

    total_norm = torch.linalg.vector_norm(
        torch.stack(norms)
    )

    return float(
        total_norm.item()
    )


@torch.no_grad()
def gradients_are_finite(
    parameters: Iterable[nn.Parameter],
) -> bool:
    for parameter in parameters:
        if parameter.grad is None:
            continue

        if not torch.isfinite(
            parameter.grad
        ).all().item():
            return False

    return True


class LionManager:
    """
    Training manager for the Lion optimizer.

    Handles:
        - Backpropagation
        - Gradient accumulation
        - AMP
        - Gradient clipping
        - Optimizer stepping
        - Learning-rate management
        - Checkpointing
        - Training statistics

    Example:
        manager = LionManager(model)

        with manager.autocast():
            loss = criterion(model(x), y)

        manager.backward(loss)
        manager.step()
    """

    def __init__(
        self,
        model: nn.Module,
        config: LionConfig | None = None,
        *,
        component_learning_rates: Mapping[
            str, float
        ] | None = None,
    ) -> None:
        self.model = model
        self.config = config or LionConfig()

        self.optimizer = create_lion_optimizer(
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

        self.device = (
            self.parameters[0].device
        )

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
            and self.autocast_dtype in {
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

        self.last_loss = 0.0
        self.last_gradient_norm = 0.0

        self.optimizer.zero_grad(
            set_to_none=True
        )

    def autocast(self):
        if not self.use_amp:
            return nullcontext()

        return torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
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
                "Non-finite loss detected"
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
            gradient_norm = (
                calculate_gradient_norm(
                    self.parameters
                )
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

            updated = (
                self.scaler.get_scale()
                >= old_scale
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
            < self.config.gradient_accumulation_steps
        ):
            return False

        return self._optimizer_step()

    def flush(self) -> bool:
        """
        Complete a partially accumulated
        optimizer update at epoch end.
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
            "optimizer_steps": (
                self.optimizer_steps
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

    def get_statistics(
        self,
    ) -> dict[str, Any]:
        return {
            "optimizer": "Lion",
            "device": str(self.device),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters
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
            "last_loss": self.last_loss,
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
            "mixed_precision": self.use_amp,
            "precision_dtype": (
                self.config.precision_dtype
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
            "last_loss": self.last_loss,
            "last_gradient_norm": (
                self.last_gradient_norm
            ),
            "config": asdict(self.config),
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

        self.last_loss = float(
            state.get(
                "last_loss",
                0.0,
            )
        )

        self.last_gradient_norm = float(
            state.get(
                "last_gradient_norm",
                0.0,
            )
        )

        # Pending gradients are not stored
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
            "lion_manager": (
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
            checkpoint["lion_manager"]
        )

        return dict(
            checkpoint.get(
                "extra",
                {},
            )
        )


def create_lion_manager(
    model: nn.Module,
    *,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.99,
    gradient_clip_norm: float | None = 1.0,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = False,
    precision_dtype: str = "float16",
    component_learning_rates: Mapping[
        str, float
    ] | None = None,
) -> LionManager:
    config = LionConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        gradient_clip_norm=gradient_clip_norm,
        gradient_accumulation_steps=(
            gradient_accumulation_steps
        ),
        mixed_precision=mixed_precision,
        precision_dtype=precision_dtype,
    )

    return LionManager(
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
            latent = self.encoder(
                states
            )

            features = torch.cat(
                [latent, actions],
                dim=-1,
            )

            return self.dynamics(
                features
            )

    model = DemoWorldModel().to(device)

    manager = create_lion_manager(
        model,
        learning_rate=1e-4,
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
            "encoder": 5e-5,
            "dynamics": 1e-4,
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

            loss = torch.nn.functional.mse_loss(
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

    print("\nLion optimizer statistics:")

    print(
        manager.get_statistics()
    )
