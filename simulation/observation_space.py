
```python
"""
NeuroWorld AI
File: neuroworld/environment/observation_space.py

Multimodal Observation Space

Features:
- Continuous vector observations
- Discrete observation spaces
- Image observation spaces
- Multimodal dictionary observations
- Sensor observation validation
- Observation normalization
- Running mean and variance
- Observation clipping
- Missing sensor handling
- Image preprocessing
- Latent-state encoding
- Batch observations
- Device management
- Serialization
- CPU/CUDA support

Dependencies:
    pip install torch
"""

from __future__ import annotations

import json
import math

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ObservationSpaceType(str, Enum):
    VECTOR = "vector"
    DISCRETE = "discrete"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


@dataclass
class ObservationSpaceConfig:
    observation_type: str = "vector"

    observation_dim: int = 128
    latent_dim: int = 128

    num_discrete_states: int = 16

    image_channels: int = 3
    image_height: int = 64
    image_width: int = 64

    sensor_dimensions: Dict[str, int] = field(
        default_factory=lambda: {
            "position": 3,
            "velocity": 3,
            "imu": 6,
            "lidar": 32,
        }
    )

    observation_min: float = -1000.0
    observation_max: float = 1000.0

    normalize_observations: bool = True
    clip_normalized_value: float = 10.0

    allow_missing_sensors: bool = False

    device: str = "cpu"
    dtype: str = "float32"

    epsilon: float = 1e-6


@dataclass
class ObservationValidationResult:
    valid: Tensor
    reasons: list[str]


@dataclass
class EncodedObservation:
    latent_state: Tensor
    raw_features: Tensor
    observation_type: str
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class RunningObservationStatistics(nn.Module):
    """
    Numerically stable running statistics.

    Uses batch-wise mean/variance updates.

    Statistics are stored as float64 buffers.
    """

    def __init__(
        self,
        feature_dim: int,
        epsilon: float = 1e-6,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(
                "feature_dim must be positive"
            )

        self.epsilon = epsilon

        self.register_buffer(
            "mean",
            torch.zeros(
                feature_dim,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "variance",
            torch.ones(
                feature_dim,
                dtype=torch.float64,
            ),
        )

        self.register_buffer(
            "count",
            torch.tensor(
                0.0,
                dtype=torch.float64,
            ),
        )

    @torch.no_grad()
    def update(
        self,
        observations: Tensor,
    ) -> None:
        if observations.ndim != 2:
            raise ValueError(
                "Observations must be [B,D]"
            )

        if observations.shape[-1] != (
            self.mean.shape[0]
        ):
            raise ValueError(
                "Feature dimension mismatch"
            )

        if observations.shape[0] == 0:
            return

        values = observations.detach().to(
            device=self.mean.device,
            dtype=torch.float64,
        )

        if not torch.isfinite(values).all().item():
            raise ValueError(
                "Statistics update contains NaN/Inf"
            )

        batch_count = values.shape[0]

        batch_mean = values.mean(dim=0)

        batch_variance = values.var(
            dim=0,
            unbiased=False,
        )

        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance)
            self.count.fill_(batch_count)
            return

        total = self.count + batch_count

        delta = batch_mean - self.mean

        updated_mean = (
            self.mean
            + delta * batch_count / total
        )

        m_a = self.variance * self.count
        m_b = batch_variance * batch_count

        correction = (
            delta.square()
            * self.count
            * batch_count
            / total
        )

        updated_variance = (
            m_a + m_b + correction
        ) / total

        self.mean.copy_(updated_mean)

        self.variance.copy_(
            updated_variance.clamp_min(0.0)
        )

        self.count.copy_(total)

    def normalize(
        self,
        observations: Tensor,
    ) -> Tensor:
        if self.count.item() == 0:
            return observations

        mean = self.mean.to(
            device=observations.device,
            dtype=observations.dtype,
        )

        variance = self.variance.to(
            device=observations.device,
            dtype=observations.dtype,
        )

        return (
            (observations - mean)
            / torch.sqrt(
                variance + self.epsilon
            )
        )

    def reset(self) -> None:
        self.mean.zero_()
        self.variance.fill_(1.0)
        self.count.zero_()


class ObservationEncoder(nn.Module):
    """
    Trainable feature-to-latent encoder.

    The encoder is randomly initialized.
    It must be trained before its output
    represents meaningful world states.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        observations: Tensor,
    ) -> Tensor:
        return self.network(observations)


class ObservationSpace(nn.Module):
    """
    Observation-space management.

    Shapes:

    Vector:
        [B,D]

    Discrete:
        [B]

    Image:
        [B,C,H,W]

    Multimodal:
        {
            "position": [B,3],
            "velocity": [B,3],
            "imu": [B,6],
            "lidar": [B,32]
        }

    Encoded latent state:
        [B,latent_dim]
    """

    def __init__(
        self,
        config: Optional[
            ObservationSpaceConfig
        ] = None,
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.config = (
            config or ObservationSpaceConfig()
        )

        self._validate_config()

        cfg = self.config

        self.observation_type = (
            ObservationSpaceType(
                cfg.observation_type
            )
        )

        self.device = torch.device(
            cfg.device
        )

        self.dtype = getattr(
            torch,
            cfg.dtype,
        )

        self.input_dim = (
            self._calculate_input_dim()
        )

        self.statistics = (
            RunningObservationStatistics(
                feature_dim=self.input_dim,
                epsilon=cfg.epsilon,
            )
        )

        self.encoder = (
            encoder
            if encoder is not None
            else ObservationEncoder(
                input_dim=self.input_dim,
                latent_dim=cfg.latent_dim,
            )
        )

        self.to(
            device=self.device,
            dtype=self.dtype,
        )

    def _validate_config(self) -> None:
        cfg = self.config

        if cfg.observation_type not in {
            item.value
            for item in ObservationSpaceType
        }:
            raise ValueError(
                "Unsupported observation type"
            )

        for name in (
            "observation_dim",
            "latent_dim",
            "num_discrete_states",
            "image_channels",
            "image_height",
            "image_width",
        ):
            if getattr(cfg, name) <= 0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if (
            not cfg.sensor_dimensions
            or any(
                dimension <= 0
                for dimension
                in cfg.sensor_dimensions.values()
            )
        ):
            raise ValueError(
                "Sensor dimensions must be positive"
            )

        if not (
            math.isfinite(cfg.observation_min)
            and math.isfinite(cfg.observation_max)
            and cfg.observation_min
            < cfg.observation_max
        ):
            raise ValueError(
                "Invalid observation bounds"
            )

        if cfg.clip_normalized_value <= 0:
            raise ValueError(
                "clip_normalized_value "
                "must be positive"
            )

        if cfg.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive"
            )

        if cfg.dtype not in (
            "float32",
            "float64",
            "float16",
            "bfloat16",
        ):
            raise ValueError(
                "Unsupported observation dtype"
            )

    def _calculate_input_dim(self) -> int:
        cfg = self.config

        if self.observation_type == (
            ObservationSpaceType.VECTOR
        ):
            return cfg.observation_dim

        if self.observation_type == (
            ObservationSpaceType.DISCRETE
        ):
            return cfg.num_discrete_states

        if self.observation_type == (
            ObservationSpaceType.IMAGE
        ):
            return (
                cfg.image_channels
                * cfg.image_height
                * cfg.image_width
            )

        return sum(
            cfg.sensor_dimensions.values()
        )

    @property
    def encoded_dim(self) -> int:
        return self.config.latent_dim

    def _validate_vector(
        self,
        observation: Tensor,
    ) -> Tensor:
        if (
            observation.ndim != 2
            or observation.shape[-1]
            != self.config.observation_dim
        ):
            raise ValueError(
                "Vector observation must "
                "be [B,observation_dim]"
            )

        return torch.isfinite(
            observation
        ).all(dim=-1) & (
            (
                observation
                >= self.config.observation_min
            )
            & (
                observation
                <= self.config.observation_max
            )
        ).all(dim=-1)

    def _validate_discrete(
        self,
        observation: Tensor,
    ) -> Tensor:
        if observation.ndim != 1:
            raise ValueError(
                "Discrete observation must be [B]"
            )

        valid = (
            observation >= 0
        ) & (
            observation
            < self.config.num_discrete_states
        )

        if observation.is_floating_point():
            valid &= (
                torch.isfinite(observation)
                & (
                    observation
                    == observation.round()
                )
            )

        return valid

    def _validate_image(
        self,
        observation: Tensor,
    ) -> Tensor:
        cfg = self.config

        expected = (
            cfg.image_channels,
            cfg.image_height,
            cfg.image_width,
        )

        if (
            observation.ndim != 4
            or tuple(observation.shape[1:])
            != expected
        ):
            raise ValueError(
                "Image observation must "
                "be [B,C,H,W]"
            )

        if observation.dtype == torch.uint8:
            return torch.ones(
                observation.shape[0],
                dtype=torch.bool,
                device=observation.device,
            )

        valid = torch.isfinite(
            observation
        ).flatten(start_dim=1).all(dim=-1)

        valid &= (
            (observation >= 0)
            & (observation <= 1)
        ).flatten(start_dim=1).all(dim=-1)

        return valid

    def _validate_multimodal(
        self,
        observation: Mapping[
            str,
            Tensor,
        ],
    ) -> Tensor:
        cfg = self.config

        batch_size = None
        device = None

        valid = None

        for name, dimension in (
            cfg.sensor_dimensions.items()
        ):
            if name not in observation:
                if cfg.allow_missing_sensors:
                    continue

                raise ValueError(
                    f"Missing sensor: {name}"
                )

            sensor = observation[name]

            if (
                sensor.ndim != 2
                or sensor.shape[-1]
                != dimension
            ):
                raise ValueError(
                    f"Sensor {name} must "
                    f"have shape [B,{dimension}]"
                )

            if batch_size is None:
                batch_size = sensor.shape[0]
                device = sensor.device

                valid = torch.ones(
                    batch_size,
                    dtype=torch.bool,
                    device=device,
                )

            if (
                sensor.shape[0] != batch_size
                or sensor.device != device
            ):
                raise ValueError(
                    "Sensor batch/device mismatch"
                )

            sensor_valid = (
                torch.isfinite(sensor)
                & (sensor >= cfg.observation_min)
                & (sensor <= cfg.observation_max)
            ).all(dim=-1)

            valid &= sensor_valid

        if valid is None:
            raise ValueError(
                "At least one configured sensor "
                "must be provided"
            )

        unknown = (
            set(observation)
            - set(cfg.sensor_dimensions)
        )

        if unknown:
            raise ValueError(
                f"Unknown sensors: {sorted(unknown)}"
            )

        return valid

    def validate(
        self,
        observation: Tensor
        | Mapping[str, Tensor],
    ) -> ObservationValidationResult:
        """
        Validate observation shapes,
        numerical values and bounds.
        """

        if self.observation_type == (
            ObservationSpaceType.MULTIMODAL
        ):
            if not isinstance(
                observation,
                Mapping,
            ):
                raise TypeError(
                    "Multimodal observation "
                    "must be a mapping"
                )

            valid = self._validate_multimodal(
                observation
            )

        else:
            if not isinstance(
                observation,
                Tensor,
            ):
                raise TypeError(
                    "Observation must be a tensor"
                )

            if self.observation_type == (
                ObservationSpaceType.VECTOR
            ):
                valid = self._validate_vector(
                    observation
                )

            elif self.observation_type == (
                ObservationSpaceType.DISCRETE
            ):
                valid = self._validate_discrete(
                    observation
                )

            else:
                valid = self._validate_image(
                    observation
                )

        reasons = [
            "valid" if value else "invalid_observation"
            for value in (
                valid.detach().cpu().tolist()
            )
        ]

        return ObservationValidationResult(
            valid=valid,
            reasons=reasons,
        )

    def _flatten_observation(
        self,
        observation: Tensor
        | Mapping[str, Tensor],
    ) -> Tensor:
        """
        Convert supported observations
        into [B,input_dim] features.
        """

        cfg = self.config

        if self.observation_type == (
            ObservationSpaceType.VECTOR
        ):
            assert isinstance(observation, Tensor)

            return observation.to(
                device=self.device,
                dtype=self.dtype,
            )

        if self.observation_type == (
            ObservationSpaceType.DISCRETE
        ):
            assert isinstance(observation, Tensor)

            return F.one_hot(
                observation.long(),
                num_classes=cfg.num_discrete_states,
            ).to(
                device=self.device,
                dtype=self.dtype,
            )

        if self.observation_type == (
            ObservationSpaceType.IMAGE
        ):
            assert isinstance(observation, Tensor)

            image = observation.to(
                device=self.device,
                dtype=self.dtype,
            )

            if observation.dtype == torch.uint8:
                image = image / 255.0

            return image.flatten(
                start_dim=1
            )

        assert isinstance(observation, Mapping)

        provided = next(
            iter(observation.values())
        )

        batch_size = provided.shape[0]

        features = []

        for name, dimension in (
            cfg.sensor_dimensions.items()
        ):
            if name in observation:
                sensor = observation[name].to(
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                sensor = torch.zeros(
                    batch_size,
                    dimension,
                    device=self.device,
                    dtype=self.dtype,
                )

            features.append(sensor)

        return torch.cat(
            features,
            dim=-1,
        )

    def preprocess(
        self,
        observation: Tensor
        | Mapping[str, Tensor],
        update_statistics: bool = False,
    ) -> Tensor:
        """
        Validate and preprocess observations.

        Returns:
            [B,input_dim]
        """

        validation = self.validate(
            observation
        )

        if not validation.valid.all().item():
            raise ValueError(
                "Observation contains invalid values"
            )

        features = self._flatten_observation(
            observation
        )

        if update_statistics:
            self.statistics.update(
                features
            )

        if self.config.normalize_observations:
            features = self.statistics.normalize(
                features
            )

            features = features.clamp(
                -self.config.clip_normalized_value,
                self.config.clip_normalized_value,
            )

        return features

    def forward(
        self,
        observation: Tensor
        | Mapping[str, Tensor],
        update_statistics: bool = False,
    ) -> Tensor:
        """
        Encode observation into latent space.

        Returns:
            latent_state [B,latent_dim]
        """

        features = self.preprocess(
            observation,
            update_statistics=update_statistics,
        )

        return self.encoder(features)

    def encode(
        self,
        observation: Tensor
        | Mapping[str, Tensor],
        update_statistics: bool = False,
    ) -> EncodedObservation:
        """
        Return latent state and
        preprocessed input features.
        """

        features = self.preprocess(
            observation,
            update_statistics=update_statistics,
        )

        latent = self.encoder(features)

        return EncodedObservation(
            latent_state=latent,
            raw_features=features,
            observation_type=(
                self.observation_type.value
            ),
            metadata={
                "input_dim": self.input_dim,
                "latent_dim": (
                    self.config.latent_dim
                ),
                "batch_size": features.shape[0],
            },
        )

    def contains(
        self,
        observation: Tensor
        | Mapping[str, Tensor],
    ) -> bool:
        return bool(
            self.validate(
                observation
            ).valid.all().item()
        )

    def reset_statistics(self) -> None:
        self.statistics.reset()

    def summary(self) -> Dict[str, Any]:
        return {
            "observation_type": (
                self.observation_type.value
            ),
            "input_dim": self.input_dim,
            "latent_dim": (
                self.config.latent_dim
            ),
            "normalization_enabled": (
                self.config.normalize_observations
            ),
            "statistics_count": float(
                self.statistics.count.item()
            ),
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
    ) -> "ObservationSpace":
        data = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        return cls(
            ObservationSpaceConfig(**data)
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------
    # Vector Observation Space
    # ------------------------------------

    vector_space = ObservationSpace(
        ObservationSpaceConfig(
            observation_type="vector",
            observation_dim=128,
            latent_dim=128,
            device=device,
        )
    )

    vector_observation = torch.randn(
        4,
        128,
        device=device,
    )

    vector_result = vector_space.encode(
        vector_observation,
        update_statistics=True,
    )

    print(
        "Vector latent:",
        vector_result.latent_state.shape,
    )

    # ------------------------------------
    # Discrete Observation Space
    # ------------------------------------

    discrete_space = ObservationSpace(
        ObservationSpaceConfig(
            observation_type="discrete",
            num_discrete_states=16,
            latent_dim=128,
            device=device,
        )
    )

    discrete_observation = torch.tensor(
        [1, 3, 5, 7],
        device=device,
    )

    discrete_result = discrete_space.encode(
        discrete_observation
    )

    print(
        "Discrete latent:",
        discrete_result.latent_state.shape,
    )

    # ------------------------------------
    # Image Observation Space
    # ------------------------------------

    image_space = ObservationSpace(
        ObservationSpaceConfig(
            observation_type="image",
            image_channels=3,
            image_height=64,
            image_width=64,
            latent_dim=128,
            device=device,
        )
    )

    images = torch.randint(
        0,
        256,
        (2, 3, 64, 64),
        dtype=torch.uint8,
        device=device,
    )

    image_result = image_space.encode(
        images
    )

    print(
        "Image latent:",
        image_result.latent_state.shape,
    )

    # ------------------------------------
    # Multimodal Sensor Observations
    # ------------------------------------

    multimodal_space = ObservationSpace(
        ObservationSpaceConfig(
            observation_type="multimodal",
            latent_dim=128,
            device=device,
            sensor_dimensions={
                "position": 3,
                "velocity": 3,
                "imu": 6,
                "lidar": 32,
            },
        )
    )

    sensors = {
        "position": torch.randn(
            4, 3, device=device
        ),
        "velocity": torch.randn(
            4, 3, device=device
        ),
        "imu": torch.randn(
            4, 6, device=device
        ),
        "lidar": torch.rand(
            4, 32, device=device
        ),
    }

    multimodal_result = (
        multimodal_space.encode(
            sensors,
            update_statistics=True,
        )
    )

    print(
        "Multimodal latent:",
        multimodal_result.latent_state.shape,
    )

    print(
        "Observation summary:",
        multimodal_space.summary(),
    )

    # ------------------------------------
    # Save Configuration
    # ------------------------------------

    multimodal_space.save_config(
        "configs/observation_space.json"
    )
```
