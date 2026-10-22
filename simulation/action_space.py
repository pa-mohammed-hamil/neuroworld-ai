
```python
"""
NeuroWorld AI
File: neuroworld/environment/action_space.py

Action Space Management

Features:
- Discrete action spaces
- Continuous action spaces
- Multi-discrete action spaces
- Hybrid discrete + continuous actions
- Random action sampling
- Action normalization and denormalization
- Action clipping
- Action validation
- Valid-action masking
- Batch action sampling
- PyTorch tensor support
- Action encoding for world models
- Action-space serialization

Dependencies:
    pip install torch
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


class ActionSpaceType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"
    HYBRID = "hybrid"


@dataclass
class ActionSpaceConfig:
    action_type: str = "continuous"

    action_dim: int = 32

    num_discrete_actions: int = 8

    multidiscrete_nvec: Tuple[int, ...] = (3, 4, 5)

    action_min: float = -1.0
    action_max: float = 1.0

    normalized_min: float = -1.0
    normalized_max: float = 1.0

    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class HybridAction:
    discrete: Tensor
    continuous: Tensor


@dataclass
class ActionValidationResult:
    valid: Tensor
    reason: List[str]


class ActionSpace:
    """
    General-purpose action space.

    Discrete:
        Action index [B]

    Continuous:
        Action vector [B,A]

    Multi-discrete:
        Multiple categorical indices [B,N]

    Hybrid:
        Discrete index [B]
        Continuous parameters [B,A]

    All sampling operations support batches.
    """

    def __init__(
        self,
        config: Optional[ActionSpaceConfig] = None,
    ):
        self.config = config or ActionSpaceConfig()

        self._validate_config()

        self.action_type = ActionSpaceType(
            self.config.action_type
        )

        self.device = torch.device(
            self.config.device
        )

        self.dtype = getattr(
            torch,
            self.config.dtype,
        )

        if self.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
        ):
            raise ValueError(
                "dtype must be a floating-point torch dtype"
            )

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.action_type not in {
            item.value for item in ActionSpaceType
        }:
            raise ValueError(
                f"Unsupported action type: {cfg.action_type}"
            )

        if cfg.action_dim <= 0:
            raise ValueError(
                "action_dim must be positive"
            )

        if cfg.num_discrete_actions <= 0:
            raise ValueError(
                "num_discrete_actions must be positive"
            )

        if (
            not cfg.multidiscrete_nvec
            or any(
                n <= 0
                for n in cfg.multidiscrete_nvec
            )
        ):
            raise ValueError(
                "multidiscrete_nvec must contain "
                "positive integers"
            )

        for value in (
            cfg.action_min,
            cfg.action_max,
            cfg.normalized_min,
            cfg.normalized_max,
        ):
            if not math.isfinite(value):
                raise ValueError(
                    "Action bounds must be finite"
                )

        if cfg.action_min >= cfg.action_max:
            raise ValueError(
                "action_min must be less than action_max"
            )

        if cfg.normalized_min >= cfg.normalized_max:
            raise ValueError(
                "normalized_min must be less "
                "than normalized_max"
            )

    @property
    def encoded_dim(self) -> int:
        """
        Dimension of the vector representation
        supplied to a world model.
        """

        if self.action_type == ActionSpaceType.DISCRETE:
            return self.config.num_discrete_actions

        if self.action_type == ActionSpaceType.CONTINUOUS:
            return self.config.action_dim

        if self.action_type == ActionSpaceType.MULTI_DISCRETE:
            return sum(
                self.config.multidiscrete_nvec
            )

        return (
            self.config.num_discrete_actions
            + self.config.action_dim
        )

    def _sample_continuous(
        self,
        batch_size: int,
    ) -> Tensor:
        cfg = self.config

        return (
            torch.rand(
                batch_size,
                cfg.action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            * (cfg.action_max - cfg.action_min)
            + cfg.action_min
        )

    def sample(
        self,
        batch_size: int = 1,
        valid_action_mask: Optional[Tensor] = None,
    ) -> Tensor | HybridAction:
        """
        Sample random actions.

        valid_action_mask is supported for
        discrete and hybrid action types.

        Mask:
            [B,num_discrete_actions]

        True means an action is allowed.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        cfg = self.config

        if self.action_type == ActionSpaceType.CONTINUOUS:
            if valid_action_mask is not None:
                raise ValueError(
                    "Discrete mask cannot be used "
                    "with continuous actions"
                )

            return self._sample_continuous(
                batch_size
            )

        if self.action_type == ActionSpaceType.MULTI_DISCRETE:
            if valid_action_mask is not None:
                raise ValueError(
                    "Use per-category masks for "
                    "multi-discrete actions"
                )

            columns = [
                torch.randint(
                    low=0,
                    high=n,
                    size=(batch_size,),
                    device=self.device,
                )
                for n in cfg.multidiscrete_nvec
            ]

            return torch.stack(
                columns,
                dim=-1,
            )

        if valid_action_mask is None:
            indices = torch.randint(
                low=0,
                high=cfg.num_discrete_actions,
                size=(batch_size,),
                device=self.device,
            )
        else:
            self._validate_mask(
                valid_action_mask,
                batch_size,
            )

            probabilities = (
                valid_action_mask.float()
            )

            probabilities = (
                probabilities
                / probabilities.sum(
                    dim=-1,
                    keepdim=True,
                )
            )

            indices = torch.multinomial(
                probabilities,
                num_samples=1,
            ).squeeze(-1)

        if self.action_type == ActionSpaceType.DISCRETE:
            return indices

        return HybridAction(
            discrete=indices,
            continuous=self._sample_continuous(
                batch_size
            ),
        )

    def _validate_mask(
        self,
        mask: Tensor,
        batch_size: int,
    ) -> None:
        expected = (
            batch_size,
            self.config.num_discrete_actions,
        )

        if mask.shape != expected:
            raise ValueError(
                f"Mask must have shape {expected}"
            )

        if mask.dtype != torch.bool:
            raise TypeError(
                "Action mask must be boolean"
            )

        if mask.device != self.device:
            raise ValueError(
                "Action mask device mismatch"
            )

        if not mask.any(dim=-1).all().item():
            raise ValueError(
                "Every sample needs at least "
                "one valid action"
            )

    @staticmethod
    def _integer_values(
        actions: Tensor,
    ) -> Tensor:
        if actions.dtype == torch.bool:
            return torch.zeros_like(
                actions,
                dtype=torch.bool,
            )

        if actions.is_floating_point():
            return (
                torch.isfinite(actions)
                & (actions == actions.round())
            )

        return torch.ones_like(
            actions,
            dtype=torch.bool,
        )

    def validate(
        self,
        actions: Tensor | HybridAction,
        valid_action_mask: Optional[Tensor] = None,
    ) -> ActionValidationResult:
        """
        Validate actions without modifying them.

        Returns a boolean validity mask [B].
        """

        cfg = self.config

        if self.action_type == ActionSpaceType.HYBRID:
            if not isinstance(actions, HybridAction):
                raise TypeError(
                    "Hybrid actions require HybridAction"
                )

            discrete = actions.discrete
            continuous = actions.continuous

            if discrete.ndim != 1:
                raise ValueError(
                    "Hybrid discrete indices must be [B]"
                )

            batch_size = discrete.shape[0]

            if continuous.shape != (
                batch_size,
                cfg.action_dim,
            ):
                raise ValueError(
                    "Hybrid continuous actions must be [B,A]"
                )

            valid_discrete = (
                self._integer_values(discrete)
                & (discrete >= 0)
                & (
                    discrete
                    < cfg.num_discrete_actions
                )
            )

            valid_continuous = (
                torch.isfinite(continuous)
                & (continuous >= cfg.action_min)
                & (continuous <= cfg.action_max)
            ).all(dim=-1)

            valid = (
                valid_discrete
                & valid_continuous
            )

            indices = discrete

        else:
            if not isinstance(actions, Tensor):
                raise TypeError(
                    "Expected torch.Tensor"
                )

            if self.action_type == ActionSpaceType.DISCRETE:
                if actions.ndim != 1:
                    raise ValueError(
                        "Discrete actions must be [B]"
                    )

                batch_size = actions.shape[0]

                valid = (
                    self._integer_values(actions)
                    & (actions >= 0)
                    & (
                        actions
                        < cfg.num_discrete_actions
                    )
                )

                indices = actions

            elif self.action_type == ActionSpaceType.CONTINUOUS:
                if (
                    actions.ndim != 2
                    or actions.shape[1]
                    != cfg.action_dim
                ):
                    raise ValueError(
                        "Continuous actions must be [B,A]"
                    )

                valid = (
                    torch.isfinite(actions)
                    & (actions >= cfg.action_min)
                    & (actions <= cfg.action_max)
                ).all(dim=-1)

                indices = None

            else:
                dimensions = len(
                    cfg.multidiscrete_nvec
                )

                if (
                    actions.ndim != 2
                    or actions.shape[1] != dimensions
                ):
                    raise ValueError(
                        "Multi-discrete actions "
                        "must be [B,N]"
                    )

                valid = self._integer_values(
                    actions
                ).all(dim=-1)

                for index, count in enumerate(
                    cfg.multidiscrete_nvec
                ):
                    column = actions[:, index]

                    valid &= (
                        (column >= 0)
                        & (column < count)
                    )

                indices = None

        if valid_action_mask is not None:
            if indices is None:
                raise ValueError(
                    "Action masks are only supported "
                    "for discrete/hybrid actions"
                )

            self._validate_mask(
                valid_action_mask,
                indices.shape[0],
            )

            safe_indices = indices.long().clamp(
                0,
                cfg.num_discrete_actions - 1,
            )

            allowed = valid_action_mask.gather(
                1,
                safe_indices.unsqueeze(-1),
            ).squeeze(-1)

            valid &= allowed

        reasons = [
            "valid" if flag else "invalid_action"
            for flag in valid.detach().cpu().tolist()
        ]

        return ActionValidationResult(
            valid=valid,
            reason=reasons,
        )

    def clip(
        self,
        actions: Tensor | HybridAction,
    ) -> Tensor | HybridAction:
        """
        Clip continuous components.

        Discrete actions are not silently
        changed because that could select
        a different action.
        """

        cfg = self.config

        if self.action_type == ActionSpaceType.CONTINUOUS:
            if not isinstance(actions, Tensor):
                raise TypeError(
                    "Expected continuous tensor"
                )

            return actions.clamp(
                cfg.action_min,
                cfg.action_max,
            )

        if self.action_type == ActionSpaceType.HYBRID:
            if not isinstance(actions, HybridAction):
                raise TypeError(
                    "Expected HybridAction"
                )

            return HybridAction(
                discrete=actions.discrete,
                continuous=actions.continuous.clamp(
                    cfg.action_min,
                    cfg.action_max,
                ),
            )

        return actions

    def normalize(
        self,
        actions: Tensor,
    ) -> Tensor:
        """
        Normalize continuous actions from
        configured bounds to normalized bounds.
        """

        if self.action_type not in (
            ActionSpaceType.CONTINUOUS,
            ActionSpaceType.HYBRID,
        ):
            raise ValueError(
                "Normalization requires "
                "continuous action values"
            )

        cfg = self.config

        fraction = (
            (actions - cfg.action_min)
            / (cfg.action_max - cfg.action_min)
        )

        return (
            cfg.normalized_min
            + fraction
            * (
                cfg.normalized_max
                - cfg.normalized_min
            )
        )

    def denormalize(
        self,
        normalized_actions: Tensor,
    ) -> Tensor:
        """
        Map normalized continuous actions
        back to physical action bounds.
        """

        cfg = self.config

        fraction = (
            (
                normalized_actions
                - cfg.normalized_min
            )
            / (
                cfg.normalized_max
                - cfg.normalized_min
            )
        )

        return (
            cfg.action_min
            + fraction
            * (
                cfg.action_max
                - cfg.action_min
            )
        )

    def encode(
        self,
        actions: Tensor | HybridAction,
    ) -> Tensor:
        """
        Convert actions to vectors suitable
        for a world model.

        Discrete:
            one-hot [B,N]

        Continuous:
            normalized [B,A]

        Multi-discrete:
            concatenated one-hot [B,sum(nvec)]

        Hybrid:
            concatenated discrete one-hot
            and normalized continuous values.
        """

        result = self.validate(actions)

        if not result.valid.all().item():
            raise ValueError(
                "Cannot encode invalid actions"
            )

        cfg = self.config

        if self.action_type == ActionSpaceType.DISCRETE:
            assert isinstance(actions, Tensor)

            return F.one_hot(
                actions.long(),
                num_classes=cfg.num_discrete_actions,
            ).to(self.dtype)

        if self.action_type == ActionSpaceType.CONTINUOUS:
            assert isinstance(actions, Tensor)

            return self.normalize(
                actions
            ).to(self.dtype)

        if self.action_type == ActionSpaceType.MULTI_DISCRETE:
            assert isinstance(actions, Tensor)

            vectors = [
                F.one_hot(
                    actions[:, index].long(),
                    num_classes=count,
                )
                for index, count in enumerate(
                    cfg.multidiscrete_nvec
                )
            ]

            return torch.cat(
                vectors,
                dim=-1,
            ).to(self.dtype)

        assert isinstance(actions, HybridAction)

        discrete_vector = F.one_hot(
            actions.discrete.long(),
            num_classes=cfg.num_discrete_actions,
        ).to(self.dtype)

        continuous_vector = self.normalize(
            actions.continuous
        ).to(self.dtype)

        return torch.cat(
            [
                discrete_vector,
                continuous_vector,
            ],
            dim=-1,
        )

    def decode(
        self,
        encoded_actions: Tensor,
    ) -> Tensor | HybridAction:
        """
        Decode model action vectors.

        Categorical vectors are decoded using
        argmax, so the operation is not
        differentiable for discrete actions.
        """

        if (
            encoded_actions.ndim != 2
            or encoded_actions.shape[1]
            != self.encoded_dim
        ):
            raise ValueError(
                "Encoded action shape mismatch"
            )

        cfg = self.config

        if self.action_type == ActionSpaceType.DISCRETE:
            return encoded_actions.argmax(
                dim=-1
            )

        if self.action_type == ActionSpaceType.CONTINUOUS:
            return self.denormalize(
                encoded_actions
            )

        if self.action_type == ActionSpaceType.MULTI_DISCRETE:
            pieces = torch.split(
                encoded_actions,
                cfg.multidiscrete_nvec,
                dim=-1,
            )

            return torch.stack(
                [
                    piece.argmax(dim=-1)
                    for piece in pieces
                ],
                dim=-1,
            )

        discrete_vector = encoded_actions[
            :,
            :cfg.num_discrete_actions,
        ]

        continuous_vector = encoded_actions[
            :,
            cfg.num_discrete_actions:,
        ]

        return HybridAction(
            discrete=discrete_vector.argmax(
                dim=-1
            ),
            continuous=self.denormalize(
                continuous_vector
            ),
        )

    def contains(
        self,
        actions: Tensor | HybridAction,
    ) -> bool:
        """
        Return True only if every action
        in the batch is valid.
        """

        return bool(
            self.validate(actions)
            .valid.all().item()
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "action_dim": self.config.action_dim,
            "encoded_dim": self.encoded_dim,
            "num_discrete_actions": (
                self.config.num_discrete_actions
            ),
            "multidiscrete_nvec": list(
                self.config.multidiscrete_nvec
            ),
            "action_bounds": [
                self.config.action_min,
                self.config.action_max,
            ],
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    def save_config(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                asdict(self.config),
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_config(
        cls,
        path: str | Path,
    ) -> "ActionSpace":
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        if "multidiscrete_nvec" in data:
            data["multidiscrete_nvec"] = tuple(
                data["multidiscrete_nvec"]
            )

        return cls(
            ActionSpaceConfig(**data)
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    # ----------------------------------
    # Continuous Action Space
    # ----------------------------------

    continuous = ActionSpace(
        ActionSpaceConfig(
            action_type="continuous",
            action_dim=32,
            action_min=-1.0,
            action_max=1.0,
        )
    )

    actions = continuous.sample(
        batch_size=4
    )

    encoded = continuous.encode(actions)

    print("Continuous actions:", actions.shape)
    print("Encoded:", encoded.shape)
    print("Valid:", continuous.contains(actions))

    # ----------------------------------
    # Discrete Action Space
    # ----------------------------------

    discrete = ActionSpace(
        ActionSpaceConfig(
            action_type="discrete",
            num_discrete_actions=8,
        )
    )

    discrete_actions = discrete.sample(
        batch_size=4
    )

    discrete_encoded = discrete.encode(
        discrete_actions
    )

    print("Discrete:", discrete_actions)
    print("One-hot:", discrete_encoded.shape)

    # ----------------------------------
    # Multi-Discrete Action Space
    # ----------------------------------

    multidiscrete = ActionSpace(
        ActionSpaceConfig(
            action_type="multi_discrete",
            multidiscrete_nvec=(3, 4, 5),
        )
    )

    multi_actions = multidiscrete.sample(
        batch_size=4
    )

    print("Multi-discrete:", multi_actions)
    print(
        "Multi-discrete encoded:",
        multidiscrete.encode(
            multi_actions
        ).shape,
    )

    # ----------------------------------
    # Hybrid Action Space
    # ----------------------------------

    hybrid = ActionSpace(
        ActionSpaceConfig(
            action_type="hybrid",
            action_dim=6,
            num_discrete_actions=4,
        )
    )

    hybrid_actions = hybrid.sample(
        batch_size=4
    )

    hybrid_encoded = hybrid.encode(
        hybrid_actions
    )

    print(
        "Hybrid discrete:",
        hybrid_actions.discrete,
    )

    print(
        "Hybrid continuous:",
        hybrid_actions.continuous.shape,
    )

    print(
        "Hybrid encoded:",
        hybrid_encoded.shape,
    )

    # ----------------------------------
    # Configuration Export
    # ----------------------------------

    continuous.save_config(
        "configs/action_space.json"
    )

    restored = ActionSpace.load_config(
        "configs/action_space.json"
    )

    print("Restored:", restored.summary())
```
