
"""
NeuroWorld AI
File: training/optimizers/optimizer_factory.py

Unified optimizer factory for:
- World models
- Video prediction
- Tokenizer / VQ-VAE
- Dynamics models
- Reward models
- Policy networks
- Robotics and Physical AI
- Joint and multi-model training

Supported optimizers:
    AdamW
    Adam
    Lion
    SGD
    RMSprop
    Adagrad
    Adadelta
    RAdam
    NAdam
    Adamax

Features:
- Optimizer creation from config
- Model-specific learning rates
- Parameter groups
- Automatic weight-decay exclusions
- Frozen-parameter filtering
- Duplicate-parameter protection
- Multiple-model optimization
- Learning-rate management
- Optimizer state checkpointing
- Optimizer statistics
- YAML-compatible configuration
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

try:
    from .lion import Lion
except ImportError:
    from lion import Lion


SUPPORTED_OPTIMIZERS = (
    "adamw",
    "adam",
    "lion",
    "sgd",
    "rmsprop",
    "adagrad",
    "adadelta",
    "radam",
    "nadam",
    "adamax",
)


@dataclass
class OptimizerConfig:
    name: str = "adamw"

    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False

    alpha: float = 0.99
    rho: float = 0.9

    amsgrad: bool = False
    maximize: bool = False

    foreach: bool | None = None
    fused: bool | None = None

    exclude_bias_from_weight_decay: bool = True
    exclude_norm_from_weight_decay: bool = True
    exclude_embedding_from_weight_decay: bool = True

    exclude_1d_parameters_from_weight_decay: bool = True

    layerwise_lr_decay: float = 1.0

    def __post_init__(self) -> None:
        self.name = self.name.lower().strip()

        if self.name not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {self.name}. "
                f"Available: {SUPPORTED_OPTIMIZERS}"
            )

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

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if not 0 <= self.momentum < 1:
            raise ValueError(
                "momentum must be in [0,1)"
            )

        if self.dampening < 0:
            raise ValueError(
                "dampening must be nonnegative"
            )

        if not 0 < self.alpha < 1:
            raise ValueError(
                "alpha must be in (0,1)"
            )

        if not 0 < self.rho < 1:
            raise ValueError(
                "rho must be in (0,1)"
            )

        if not 0 < self.layerwise_lr_decay <= 1:
            raise ValueError(
                "layerwise_lr_decay must be in (0,1]"
            )

        if self.nesterov:
            if self.momentum <= 0:
                raise ValueError(
                    "Nesterov requires momentum > 0"
                )

            if self.dampening != 0:
                raise ValueError(
                    "Nesterov requires dampening = 0"
                )


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


def _get_module_mapping(
    model: nn.Module,
) -> dict[str, nn.Module]:
    mapping = {}

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


def _should_apply_weight_decay(
    name: str,
    parameter: nn.Parameter,
    module: nn.Module | None,
    config: OptimizerConfig,
) -> bool:
    if (
        config.exclude_1d_parameters_from_weight_decay
        and parameter.ndim <= 1
    ):
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


def _validate_learning_rates(
    learning_rates: Mapping[str, float],
) -> None:
    for name, lr in learning_rates.items():
        if not name:
            raise ValueError(
                "Learning-rate prefix cannot be empty"
            )

        if not math.isfinite(lr) or lr <= 0:
            raise ValueError(
                f"Invalid learning rate for {name}"
            )


def _find_learning_rate(
    parameter_name: str,
    default_lr: float,
    learning_rates: Mapping[str, float],
) -> float:
    matches = [
        prefix
        for prefix in learning_rates
        if (
            parameter_name == prefix
            or parameter_name.startswith(
                prefix + "."
            )
        )
    ]

    if not matches:
        return default_lr

    prefix = max(
        matches,
        key=len,
    )

    return learning_rates[prefix]


def build_parameter_groups(
    model: nn.Module,
    config: OptimizerConfig,
    *,
    component_learning_rates: Mapping[
        str, float
    ] | None = None,
    frozen_prefixes: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Create parameter groups for one model.

    Args:
        model:
            PyTorch model.

        config:
            Optimizer configuration.

        component_learning_rates:
            Optional learning rates by module prefix.

        frozen_prefixes:
            Module prefixes excluded from optimization.

    Returns:
        List of optimizer parameter groups.
    """
    component_learning_rates = dict(
        component_learning_rates or {}
    )

    frozen_prefixes = tuple(
        frozen_prefixes or ()
    )

    _validate_learning_rates(
        component_learning_rates
    )

    module_mapping = _get_module_mapping(
        model
    )

    groups: dict[
        tuple[float, float],
        dict[str, Any],
    ] = {}

    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if any(
            name == prefix
            or name.startswith(prefix + ".")
            for prefix in frozen_prefixes
        ):
            continue

        if id(parameter) in seen:
            continue

        seen.add(id(parameter))

        learning_rate = _find_learning_rate(
            name,
            config.learning_rate,
            component_learning_rates,
        )

        module = module_mapping.get(
            name
        )

        weight_decay = (
            config.weight_decay
            if _should_apply_weight_decay(
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


def build_multi_model_parameter_groups(
    models: Mapping[str, nn.Module],
    config: OptimizerConfig,
    *,
    model_learning_rates: Mapping[
        str, float
    ] | None = None,
) -> list[dict[str, Any]]:
    """
    Build groups for multiple models.

    Example:
        models = {
            "tokenizer": tokenizer,
            "dynamics": dynamics,
            "reward": reward_model,
            "policy": policy,
        }

    Shared parameters are included once.
    """
    model_learning_rates = dict(
        model_learning_rates or {}
    )

    _validate_learning_rates(
        model_learning_rates
    )

    all_groups: dict[
        tuple[float, float],
        dict[str, Any],
    ] = {}

    seen: set[int] = set()

    for model_name, model in models.items():
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"{model_name} is not an nn.Module"
            )

        model_lr = model_learning_rates.get(
            model_name,
            config.learning_rate,
        )

        module_mapping = _get_module_mapping(
            model
        )

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            if id(parameter) in seen:
                continue

            seen.add(id(parameter))

            module = module_mapping.get(
                name
            )

            weight_decay = (
                config.weight_decay
                if _should_apply_weight_decay(
                    name,
                    parameter,
                    module,
                    config,
                )
                else 0.0
            )

            key = (
                model_lr,
                weight_decay,
            )

            if key not in all_groups:
                all_groups[key] = {
                    "params": [],
                    "lr": model_lr,
                    "weight_decay": weight_decay,
                    "parameter_names": [],
                }

            all_groups[key]["params"].append(
                parameter
            )

            all_groups[key]["parameter_names"].append(
                f"{model_name}.{name}"
            )

    if not all_groups:
        raise ValueError(
            "No trainable parameters found"
        )

    return list(
        all_groups.values()
    )


def _create_optimizer_from_groups(
    groups: list[dict[str, Any]],
    config: OptimizerConfig,
) -> Optimizer:
    name = config.name

    if name == "adamw":
        return torch.optim.AdamW(
            groups,
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
        )

    if name == "adam":
        return torch.optim.Adam(
            groups,
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
        )

    if name == "lion":
        return Lion(
            groups,
            lr=config.learning_rate,
            betas=(
                config.beta1,
                config.beta2,
            ),
            weight_decay=config.weight_decay,
            maximize=config.maximize,
        )

    if name == "sgd":
        return torch.optim.SGD(
            groups,
            lr=config.learning_rate,
            momentum=config.momentum,
            dampening=config.dampening,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
            maximize=config.maximize,
            foreach=config.foreach,
        )

    if name == "rmsprop":
        return torch.optim.RMSprop(
            groups,
            lr=config.learning_rate,
            alpha=config.alpha,
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            maximize=config.maximize,
            foreach=config.foreach,
        )

    if name == "adagrad":
        return torch.optim.Adagrad(
            groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.epsilon,
            maximize=config.maximize,
            foreach=config.foreach,
        )

    if name == "adadelta":
        return torch.optim.Adadelta(
            groups,
            lr=config.learning_rate,
            rho=config.rho,
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            maximize=config.maximize,
            foreach=config.foreach,
        )

    if name == "radam":
        return torch.optim.RAdam(
            groups,
            lr=config.learning_rate,
            betas=(
                config.beta1,
                config.beta2,
            ),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            maximize=config.maximize,
        )

    if name == "nadam":
        return torch.optim.NAdam(
            groups,
            lr=config.learning_rate,
            betas=(
                config.beta1,
                config.beta2,
            ),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            maximize=config.maximize,
        )

    if name == "adamax":
        return torch.optim.Adamax(
            groups,
            lr=config.learning_rate,
            betas=(
                config.beta1,
                config.beta2,
            ),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            maximize=config.maximize,
        )

    raise ValueError(
        f"Unsupported optimizer: {name}"
    )


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig | Mapping[str, Any] | None = None,
    *,
    component_learning_rates: Mapping[
        str, float
    ] | None = None,
    frozen_prefixes: Iterable[str] | None = None,
) -> Optimizer:
    """
    Main optimizer factory.

    Example:
        optimizer = create_optimizer(
            model,
            OptimizerConfig(
                name="adamw",
                learning_rate=3e-4,
            ),
        )
    """
    if config is None:
        config = OptimizerConfig()

    elif isinstance(config, Mapping):
        config = optimizer_config_from_dict(
            config
        )

    if not isinstance(
        config,
        OptimizerConfig,
    ):
        raise TypeError(
            "config must be OptimizerConfig or mapping"
        )

    groups = build_parameter_groups(
        model,
        config,
        component_learning_rates=(
            component_learning_rates
        ),
        frozen_prefixes=frozen_prefixes,
    )

    return _create_optimizer_from_groups(
        groups,
        config,
    )


def create_multi_model_optimizer(
    models: Mapping[str, nn.Module],
    config: OptimizerConfig | Mapping[str, Any] | None = None,
    *,
    model_learning_rates: Mapping[
        str, float
    ] | None = None,
) -> Optimizer:
    if config is None:
        config = OptimizerConfig()

    elif isinstance(config, Mapping):
        config = optimizer_config_from_dict(
            config
        )

    groups = build_multi_model_parameter_groups(
        models,
        config,
        model_learning_rates=(
            model_learning_rates
        ),
    )

    return _create_optimizer_from_groups(
        groups,
        config,
    )


def optimizer_config_from_dict(
    values: Mapping[str, Any],
) -> OptimizerConfig:
    """
    Create optimizer config from a dictionary.

    Supports:
        name
        type
        lr
        learning_rate
        betas
        weight_decay

    Example YAML:
        optimizer:
          name: adamw
          lr: 0.0003
          weight_decay: 0.01
          betas: [0.9, 0.999]
    """
    data = dict(values)

    if "optimizer" in data:
        nested = data["optimizer"]

        if isinstance(nested, Mapping):
            data = dict(nested)

    aliases = {
        "type": "name",
        "optimizer_type": "name",
        "lr": "learning_rate",
        "eps": "epsilon",
    }

    for old_name, new_name in aliases.items():
        if old_name in data:
            if new_name not in data:
                data[new_name] = data[old_name]

            del data[old_name]

    if "betas" in data:
        betas = data.pop("betas")

        if len(betas) != 2:
            raise ValueError(
                "betas must contain two values"
            )

        data.setdefault(
            "beta1",
            float(betas[0]),
        )

        data.setdefault(
            "beta2",
            float(betas[1]),
        )

    valid_fields = {
        field.name
        for field in fields(OptimizerConfig)
    }

    filtered = {
        key: value
        for key, value in data.items()
        if key in valid_fields
    }

    return OptimizerConfig(
        **filtered
    )


def get_supported_optimizers() -> list[str]:
    return list(
        SUPPORTED_OPTIMIZERS
    )


def get_optimizer_learning_rates(
    optimizer: Optimizer,
) -> list[float]:
    return [
        float(group["lr"])
        for group in optimizer.param_groups
    ]


def set_optimizer_learning_rate(
    optimizer: Optimizer,
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
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    else:
        optimizer.param_groups[
            group_index
        ]["lr"] = learning_rate


def multiply_optimizer_learning_rate(
    optimizer: Optimizer,
    factor: float,
) -> None:
    if (
        not math.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError(
            "factor must be positive"
        )

    for group in optimizer.param_groups:
        group["lr"] *= factor


def get_optimizer_statistics(
    optimizer: Optimizer,
) -> dict[str, Any]:
    total_parameters = 0
    trainable_parameters = 0
    parameter_tensors = 0

    seen: set[int] = set()

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                continue

            seen.add(id(parameter))

            parameter_tensors += 1
            total_parameters += parameter.numel()

            if parameter.requires_grad:
                trainable_parameters += (
                    parameter.numel()
                )

    return {
        "optimizer": (
            optimizer.__class__.__name__
        ),
        "parameter_groups": len(
            optimizer.param_groups
        ),
        "parameter_tensors": (
            parameter_tensors
        ),
        "total_parameters": (
            total_parameters
        ),
        "trainable_parameters": (
            trainable_parameters
        ),
        "learning_rates": (
            get_optimizer_learning_rates(
                optimizer
            )
        ),
        "weight_decays": [
            float(
                group.get(
                    "weight_decay",
                    0.0,
                )
            )
            for group in optimizer.param_groups
        ],
        "optimizer_state_entries": len(
            optimizer.state
        ),
    }


def save_optimizer_checkpoint(
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

    checkpoint = {
        "optimizer": (
            optimizer.state_dict()
        ),
        "step": int(step),
        "epoch": int(epoch),
        "extra": dict(extra or {}),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_optimizer_checkpoint(
    optimizer: Optimizer,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
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
            checkpoint.get(
                "step",
                0,
            )
        ),
        "epoch": int(
            checkpoint.get(
                "epoch",
                0,
            )
        ),
        "extra": dict(
            checkpoint.get(
                "extra",
                {},
            )
        ),
    }


class OptimizerFactory:
    """
    Object-oriented optimizer factory.

    Example:
        factory = OptimizerFactory(
            OptimizerConfig(
                name="adamw",
                learning_rate=3e-4,
            )
        )

        optimizer = factory.create(model)
    """

    def __init__(
        self,
        config: OptimizerConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = OptimizerConfig()

        elif isinstance(config, Mapping):
            config = optimizer_config_from_dict(
                config
            )

        self.config = config

    def create(
        self,
        model: nn.Module,
        *,
        component_learning_rates: Mapping[
            str, float
        ] | None = None,
        frozen_prefixes: Iterable[str] | None = None,
    ) -> Optimizer:
        return create_optimizer(
            model,
            self.config,
            component_learning_rates=(
                component_learning_rates
            ),
            frozen_prefixes=frozen_prefixes,
        )

    def create_multi_model(
        self,
        models: Mapping[str, nn.Module],
        *,
        model_learning_rates: Mapping[
            str, float
        ] | None = None,
    ) -> Optimizer:
        return create_multi_model_optimizer(
            models,
            self.config,
            model_learning_rates=(
                model_learning_rates
            ),
        )

    def update_config(
        self,
        **kwargs: Any,
    ) -> None:
        data = asdict(
            self.config
        )

        data.update(
            kwargs
        )

        self.config = OptimizerConfig(
            **data
        )

    def get_config(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self.config
        )

    @staticmethod
    def available() -> list[str]:
        return get_supported_optimizers()


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

            self.reward = nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(
            self,
            states: Tensor,
            actions: Tensor,
        ) -> tuple[Tensor, Tensor]:
            latent = self.encoder(
                states
            )

            features = torch.cat(
                [latent, actions],
                dim=-1,
            )

            predicted_states = (
                self.dynamics(features)
            )

            predicted_rewards = (
                self.reward(predicted_states)
            )

            return (
                predicted_states,
                predicted_rewards,
            )

    model = DemoWorldModel().to(
        device
    )

    factory = OptimizerFactory(
        OptimizerConfig(
            name="adamw",
            learning_rate=3e-4,
            weight_decay=0.01,
        )
    )

    optimizer = factory.create(
        model,
        component_learning_rates={
            "encoder": 1e-4,
            "dynamics": 3e-4,
            "reward": 2e-4,
        },
    )

    print(
        "Supported optimizers:",
        factory.available(),
    )

    print(
        "Optimizer statistics:",
        get_optimizer_statistics(
            optimizer
        ),
    )

    model.train()

    for step in range(5):
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

        target_states = (
            states
            + 0.05 * torch.randn_like(states)
        )

        target_rewards = torch.randn(
            16,
            1,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predicted_states, predicted_rewards = (
            model(
                states,
                actions,
            )
        )

        state_loss = (
            torch.nn.functional.mse_loss(
                predicted_states,
                target_states,
            )
        )

        reward_loss = (
            torch.nn.functional.mse_loss(
                predicted_rewards,
                target_rewards,
            )
        )

        total_loss = (
            state_loss
            + 0.5 * reward_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1} | "
            f"Loss: {total_loss.item():.6f}"
        )

    print(
        "\nFinal optimizer statistics:"
    )

    print(
        get_optimizer_statistics(
            optimizer
        )
    )
