
"""
NeuroWorld AI
backend/app/config.py

Centralized application configuration.

Features:
- Environment-based settings
- FastAPI and server configuration
- Neural world model settings
- Perception and latent-space settings
- Dynamics and imagination settings
- Planning and reinforcement learning settings
- Training and inference settings
- Database, Redis and storage configuration
- Security and CORS configuration
- GPU detection
- Configuration validation

Dependencies:
    pip install pydantic pydantic-settings

Usage:
    from backend.app.config import get_settings

    settings = get_settings()
    print(settings.app_name)
"""

from __future__ import annotations

import json
import logging
import os

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_DIR = PROJECT_ROOT / "training"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


# ============================================================
# ENUMERATIONS
# ============================================================

class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class PrecisionType(str, Enum):
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"


class WorldModelType(str, Enum):
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    AUTOREGRESSIVE = "autoregressive"
    HYBRID = "hybrid"


class PlannerType(str, Enum):
    MPC = "mpc"
    CEM = "cem"
    TREE_SEARCH = "tree_search"
    LATENT = "latent"
    GOAL_CONDITIONED = "goal_conditioned"


# ============================================================
# PERCEPTION SETTINGS
# ============================================================

class PerceptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_size: int = Field(
        default=224,
        ge=32,
        le=2048,
    )

    image_channels: int = Field(
        default=3,
        ge=1,
        le=4,
    )

    video_frames: int = Field(
        default=16,
        ge=1,
        le=256,
    )

    frame_rate: int = Field(
        default=24,
        ge=1,
        le=120,
    )

    patch_size: int = Field(
        default=16,
        ge=4,
        le=64,
    )

    vision_embedding_dim: int = Field(
        default=512,
        ge=32,
    )

    vision_transformer_layers: int = Field(
        default=8,
        ge=1,
    )

    attention_heads: int = Field(
        default=8,
        ge=1,
    )

    enable_depth_estimation: bool = True
    enable_optical_flow: bool = True
    enable_object_tracking: bool = True
    enable_scene_encoding: bool = True
    enable_multimodal_fusion: bool = True

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size"
            )

        if (
            self.vision_embedding_dim
            % self.attention_heads
            != 0
        ):
            raise ValueError(
                "vision_embedding_dim must be divisible "
                "by attention_heads"
            )

        return self


# ============================================================
# TOKENIZER SETTINGS
# ============================================================

class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocabulary_size: int = Field(
        default=8192,
        ge=128,
    )

    codebook_size: int = Field(
        default=1024,
        ge=32,
    )

    codebook_dim: int = Field(
        default=256,
        ge=16,
    )

    commitment_weight: float = Field(
        default=0.25,
        ge=0.0,
    )

    use_vector_quantization: bool = True
    use_spatial_tokens: bool = True
    use_temporal_tokens: bool = True
    use_action_tokens: bool = True


# ============================================================
# LATENT SPACE SETTINGS
# ============================================================

class LatentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latent_dim: int = Field(
        default=512,
        ge=32,
    )

    state_dim: int = Field(
        default=512,
        ge=32,
    )

    hidden_dim: int = Field(
        default=1024,
        ge=64,
    )

    latent_sequence_length: int = Field(
        default=32,
        ge=1,
    )

    stochastic_latent_dim: int = Field(
        default=128,
        ge=1,
    )

    deterministic_latent_dim: int = Field(
        default=384,
        ge=1,
    )

    use_variational_encoding: bool = True
    enable_latent_compression: bool = True

    kl_weight: float = Field(
        default=0.01,
        ge=0.0,
    )

    @model_validator(mode="after")
    def validate_latent_parts(self):
        total = (
            self.stochastic_latent_dim
            + self.deterministic_latent_dim
        )

        if total != self.latent_dim:
            raise ValueError(
                "stochastic_latent_dim + "
                "deterministic_latent_dim must equal latent_dim"
            )

        return self


# ============================================================
# WORLD MODEL SETTINGS
# ============================================================

class WorldModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: WorldModelType = (
        WorldModelType.TRANSFORMER
    )

    embedding_dim: int = Field(
        default=512,
        ge=32,
    )

    hidden_dim: int = Field(
        default=1024,
        ge=64,
    )

    transformer_layers: int = Field(
        default=12,
        ge=1,
    )

    attention_heads: int = Field(
        default=8,
        ge=1,
    )

    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
    )

    context_length: int = Field(
        default=32,
        ge=1,
    )

    prediction_horizon: int = Field(
        default=16,
        ge=1,
    )

    action_dim: int = Field(
        default=12,
        ge=1,
    )

    enable_action_conditioning: bool = True
    enable_uncertainty_estimation: bool = True
    enable_counterfactual_prediction: bool = True

    @model_validator(mode="after")
    def validate_attention(self):
        if self.embedding_dim % self.attention_heads:
            raise ValueError(
                "embedding_dim must be divisible "
                "by attention_heads"
            )

        return self


# ============================================================
# DYNAMICS SETTINGS
# ============================================================

class DynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_type: Literal[
        "transformer",
        "recurrent",
        "stochastic",
        "deterministic",
    ] = "transformer"

    hidden_dim: int = Field(
        default=512,
        ge=32,
    )

    num_layers: int = Field(
        default=6,
        ge=1,
    )

    transition_steps: int = Field(
        default=16,
        ge=1,
    )

    enable_physics_constraints: bool = True
    enable_stochastic_transitions: bool = True

    physics_loss_weight: float = Field(
        default=0.2,
        ge=0.0,
    )

    consistency_loss_weight: float = Field(
        default=0.1,
        ge=0.0,
    )


# ============================================================
# IMAGINATION SETTINGS
# ============================================================

class ImaginationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    rollout_horizon: int = Field(
        default=32,
        ge=1,
        le=1024,
    )

    max_branches: int = Field(
        default=8,
        ge=1,
        le=128,
    )

    max_parallel_rollouts: int = Field(
        default=4,
        ge=1,
        le=64,
    )

    imagination_temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=5.0,
    )

    enable_counterfactuals: bool = True
    enable_future_generation: bool = True
    enable_trajectory_generation: bool = True
    enable_imagination_memory: bool = True

    uncertainty_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# PLANNING SETTINGS
# ============================================================

class PlanningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planner_type: PlannerType = PlannerType.CEM

    planning_horizon: int = Field(
        default=20,
        ge=1,
        le=512,
    )

    optimization_iterations: int = Field(
        default=10,
        ge=1,
    )

    candidate_trajectories: int = Field(
        default=256,
        ge=1,
    )

    elite_fraction: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
    )

    discount_factor: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
    )

    enable_replanning: bool = True
    enable_constraint_checking: bool = True
    enable_safety_validation: bool = True

    @model_validator(mode="after")
    def validate_elite_count(self):
        if (
            self.candidate_trajectories
            * self.elite_fraction
            < 1
        ):
            raise ValueError(
                "Planning configuration must select "
                "at least one elite trajectory"
            )

        return self


# ============================================================
# REINFORCEMENT LEARNING SETTINGS
# ============================================================

class RLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    algorithm: Literal[
        "ppo",
        "sac",
        "dreamer",
        "model_based",
    ] = "dreamer"

    learning_rate: float = Field(
        default=0.0003,
        gt=0.0,
    )

    gamma: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
    )

    gae_lambda: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
    )

    entropy_coefficient: float = Field(
        default=0.01,
        ge=0.0,
    )

    replay_buffer_size: int = Field(
        default=100000,
        ge=1,
    )

    batch_size: int = Field(
        default=64,
        ge=1,
    )

    imagination_training_steps: int = Field(
        default=15,
        ge=1,
    )

    enable_intrinsic_reward: bool = True
    enable_imagined_replay: bool = True


# ============================================================
# TRAINING SETTINGS
# ============================================================

class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    epochs: int = Field(
        default=100,
        ge=1,
    )

    batch_size: int = Field(
        default=16,
        ge=1,
    )

    learning_rate: float = Field(
        default=0.0001,
        gt=0.0,
    )

    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
    )

    gradient_accumulation_steps: int = Field(
        default=4,
        ge=1,
    )

    max_gradient_norm: float = Field(
        default=1.0,
        gt=0.0,
    )

    warmup_steps: int = Field(
        default=1000,
        ge=0,
    )

    checkpoint_interval: int = Field(
        default=5,
        ge=1,
    )

    validation_interval: int = Field(
        default=1,
        ge=1,
    )

    early_stopping_patience: int = Field(
        default=10,
        ge=1,
    )

    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    enable_tensorboard: bool = True
    enable_mlflow: bool = False

    random_seed: int = 42

    train_split: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
    )

    validation_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
    )

    test_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
    )

    @model_validator(mode="after")
    def validate_splits(self):
        total = (
            self.train_split
            + self.validation_split
            + self.test_split
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Training splits must sum to 1.0"
            )

        return self


# ============================================================
# SIMULATION SETTINGS
# ============================================================

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    default_environment: str = "synthetic_physics"

    time_step: float = Field(
        default=0.05,
        gt=0.0,
    )

    max_episode_steps: int = Field(
        default=1000,
        ge=1,
    )

    max_parallel_worlds: int = Field(
        default=4,
        ge=1,
    )

    render_width: int = Field(
        default=1280,
        ge=64,
    )

    render_height: int = Field(
        default=720,
        ge=64,
    )

    enable_collision_detection: bool = True
    enable_gravity: bool = True
    enable_object_interactions: bool = True
    enable_scenario_recording: bool = True

    gravity: float = Field(
        default=9.81,
        ge=0.0,
    )


# ============================================================
# INFERENCE SETTINGS
# ============================================================

class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: DeviceType = DeviceType.AUTO
    precision: PrecisionType = PrecisionType.FP32

    batch_size: int = Field(
        default=4,
        ge=1,
    )

    max_concurrent_requests: int = Field(
        default=8,
        ge=1,
    )

    timeout_seconds: int = Field(
        default=120,
        ge=1,
    )

    enable_batching: bool = True
    enable_streaming: bool = True
    enable_cache: bool = True

    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
    )


# ============================================================
# SECURITY SETTINGS
# ============================================================

class SecurityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_authentication: bool = True
    enable_rate_limiting: bool = True

    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
    )

    max_upload_size_mb: int = Field(
        default=100,
        ge=1,
    )

    allowed_upload_extensions: list[str] = Field(
        default_factory=lambda: [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".mp4",
            ".webm",
            ".json",
            ".csv",
        ]
    )

    require_human_approval_for_external_actions: bool = True

    enable_audit_logging: bool = True


# ============================================================
# APPLICATION SETTINGS
# ============================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "NeuroWorld AI"
    app_version: str = "1.0.0"

    environment: Environment = Environment.DEVELOPMENT

    debug: bool = False

    api_prefix: str = "/api/v1"

    # Server
    host: str = "127.0.0.1"

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    workers: int = Field(
        default=1,
        ge=1,
    )

    # Logging
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    log_json: bool = False

    # Database
    database_url: str = (
        "sqlite:///./data/neuroworld.db"
    )

    # Redis
    redis_url: str = (
        "redis://127.0.0.1:6379/0"
    )

    enable_redis: bool = False

    # CORS
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
        ]
    )

    # Authentication
    secret_key: SecretStr | None = None

    jwt_algorithm: str = "HS256"

    # Runtime
    default_device: DeviceType = DeviceType.AUTO

    random_seed: int = 42

    # Feature flags
    enable_world_model: bool = True
    enable_imagination: bool = True
    enable_planning: bool = True
    enable_simulation: bool = True
    enable_reinforcement_learning: bool = True
    enable_training_api: bool = False

    # Nested settings
    perception: PerceptionConfig = Field(
        default_factory=PerceptionConfig
    )

    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig
    )

    latent: LatentConfig = Field(
        default_factory=LatentConfig
    )

    world_model: WorldModelConfig = Field(
        default_factory=WorldModelConfig
    )

    dynamics: DynamicsConfig = Field(
        default_factory=DynamicsConfig
    )

    imagination: ImaginationConfig = Field(
        default_factory=ImaginationConfig
    )

    planning: PlanningConfig = Field(
        default_factory=PlanningConfig
    )

    rl: RLConfig = Field(
        default_factory=RLConfig
    )

    training: TrainingConfig = Field(
        default_factory=TrainingConfig
    )

    simulation: SimulationConfig = Field(
        default_factory=SimulationConfig
    )

    inference: InferenceConfig = Field(
        default_factory=InferenceConfig
    )

    security: SecurityConfig = Field(
        default_factory=SecurityConfig
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            stripped = value.strip()

            if stripped.startswith("["):
                return json.loads(stripped)

            return [
                item.strip()
                for item in stripped.split(",")
                if item.strip()
            ]

        return value

    @model_validator(mode="after")
    def validate_environment(self):
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError(
                    "DEBUG must be disabled in production"
                )

            if not self.secret_key:
                raise ValueError(
                    "SECRET_KEY is required in production"
                )

            secret = self.secret_key.get_secret_value()

            if len(secret) < 32:
                raise ValueError(
                    "SECRET_KEY must contain at least "
                    "32 characters in production"
                )

            if self.enable_training_api:
                raise ValueError(
                    "Training API must be disabled "
                    "in production by default"
                )

            if "*" in self.cors_origins:
                raise ValueError(
                    "Wildcard CORS is not allowed "
                    "in production"
                )

        return self

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return DATA_DIR

    @property
    def models_dir(self) -> Path:
        return MODELS_DIR

    @property
    def checkpoints_dir(self) -> Path:
        return CHECKPOINTS_DIR

    def detect_device(self) -> str:
        """
        Detect an available computation device.

        Returns:
            cuda, mps, or cpu

        Does not download models or allocate GPU memory.
        """

        if self.default_device != DeviceType.AUTO:
            return self.default_device.value

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            mps = getattr(
                torch.backends,
                "mps",
                None,
            )

            if (
                mps is not None
                and mps.is_available()
            ):
                return "mps"

        except ImportError:
            pass

        return "cpu"

    def create_directories(self) -> None:
        """
        Create required local directories.

        Call during development startup or setup,
        not automatically during configuration import.
        """

        directories = [
            DATA_DIR,
            DATA_DIR / "raw",
            DATA_DIR / "processed",
            DATA_DIR / "demo",
            DATA_DIR / "simulations",
            DATA_DIR / "latent_states",
            MODELS_DIR,
            CHECKPOINTS_DIR,
            EXPERIMENTS_DIR,
            LOGS_DIR,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def public_info(self) -> dict[str, Any]:
        """
        Return non-sensitive configuration suitable
        for a health or metadata endpoint.
        """

        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "environment": self.environment.value,
            "api_prefix": self.api_prefix,
            "device": self.detect_device(),
            "features": {
                "world_model": self.enable_world_model,
                "imagination": self.enable_imagination,
                "planning": self.enable_planning,
                "simulation": self.enable_simulation,
                "reinforcement_learning": (
                    self.enable_reinforcement_learning
                ),
            },
        }


# ============================================================
# SETTINGS ACCESS
# ============================================================

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached, validated application settings.

    Example:
        settings = get_settings()
    """

    return Settings()


def reload_settings() -> Settings:
    """
    Clear cached settings and reload environment values.
    """

    get_settings.cache_clear()

    return get_settings()


# ============================================================
# LOGGING
# ============================================================

def configure_logging(
    settings: Settings | None = None,
) -> None:
    """
    Configure application logging.

    Avoids logging secret keys and credentials.
    """

    settings = settings or get_settings()

    logging.basicConfig(
        level=getattr(
            logging,
            settings.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


# ============================================================
# COMMAND-LINE DIAGNOSTICS
# ============================================================

def main() -> None:
    settings = get_settings()

    configure_logging(settings)

    print("=" * 65)
    print("NEUROWORLD AI — CONFIGURATION")
    print("=" * 65)

    info = settings.public_info()

    print(
        json.dumps(
            info,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Neural configuration:")

    print(
        f"  World model: "
        f"{settings.world_model.model_type.value}"
    )

    print(
        f"  Latent dimension: "
        f"{settings.latent.latent_dim}"
    )

    print(
        f"  Transformer layers: "
        f"{settings.world_model.transformer_layers}"
    )

    print(
        f"  Imagination horizon: "
        f"{settings.imagination.rollout_horizon}"
    )

    print(
        f"  Planning algorithm: "
        f"{settings.planning.planner_type.value}"
    )

    print(
        f"  RL algorithm: "
        f"{settings.rl.algorithm}"
    )

    print(
        f"  Training epochs: "
        f"{settings.training.epochs}"
    )

    print()
    print(
        "Configuration loaded successfully."
    )


if __name__ == "__main__":
    main()
