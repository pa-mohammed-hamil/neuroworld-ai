
"""
NeuroWorld AI
backend/app/core/config.py

Centralized core runtime configuration.

Compatible with:
    backend/app/config.py
    backend/app/dependencies.py
    backend/app/main.py

Features:
    - Shared application settings
    - CPU / CUDA / MPS detection
    - Neural model configuration
    - Imagination and planning limits
    - Simulation safety controls
    - Trusted model artifact paths
    - Runtime configuration validation
    - Environment-aware security
    - Health and configuration summaries

Install:
    pip install pydantic pydantic-settings

Usage:
    from backend.app.core.config import (
        get_core_config,
        get_settings,
    )

    config = get_core_config()
    print(config.runtime.device)
"""

from __future__ import annotations

import json
import logging
import os

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from ..config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    DeviceType,
    Environment,
    Settings,
    get_settings,
    reload_settings,
    configure_logging,
)


logger = logging.getLogger("neuroworld.core.config")


# ============================================================
# PATHS
# ============================================================

CORE_DIR = Path(__file__).resolve().parent

CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
EXPORTS_DIR = DATA_DIR / "exports"
SIMULATION_DIR = DATA_DIR / "simulations"
LATENT_STATES_DIR = DATA_DIR / "latent_states"
TRAINING_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


# ============================================================
# RUNTIME CONFIGURATION
# ============================================================

class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal[
        "cpu",
        "cuda",
        "mps",
    ] = "cpu"

    precision: Literal[
        "float32",
        "float16",
        "bfloat16",
    ] = "float32"

    seed: int = 42

    deterministic: bool = False

    max_concurrent_inference: int = Field(
        default=4,
        ge=1,
        le=128,
    )

    inference_timeout_seconds: int = Field(
        default=120,
        ge=1,
        le=3600,
    )

    max_batch_size: int = Field(
        default=16,
        ge=1,
        le=1024,
    )

    enable_inference_cache: bool = True

    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
    )

    enable_gpu_memory_monitoring: bool = True

    @model_validator(mode="after")
    def validate_precision(self):
        if (
            self.device == "cpu"
            and self.precision == "float16"
        ):
            raise ValueError(
                "float16 CPU execution is not "
                "supported by this runtime policy"
            )

        return self


# ============================================================
# WORLD MODEL RUNTIME
# ============================================================

class WorldRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    architecture: Literal[
        "transformer",
        "diffusion",
        "autoregressive",
        "hybrid",
    ] = "transformer"

    latent_dim: int = Field(
        default=512,
        ge=32,
    )

    state_dim: int = Field(
        default=512,
        ge=32,
    )

    action_dim: int = Field(
        default=12,
        ge=1,
    )

    context_length: int = Field(
        default=32,
        ge=1,
    )

    prediction_horizon: int = Field(
        default=16,
        ge=1,
    )

    transformer_layers: int = Field(
        default=12,
        ge=1,
    )

    attention_heads: int = Field(
        default=8,
        ge=1,
    )

    hidden_dim: int = Field(
        default=1024,
        ge=32,
    )

    dropout: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
    )

    enable_uncertainty: bool = True
    enable_action_conditioning: bool = True
    enable_counterfactuals: bool = True

    @model_validator(mode="after")
    def validate_attention(self):
        if self.latent_dim % self.attention_heads:
            raise ValueError(
                "latent_dim must be divisible "
                "by attention_heads"
            )

        return self


# ============================================================
# IMAGINATION RUNTIME
# ============================================================

class ImaginationRuntimeConfig(BaseModel):
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

    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=5.0,
    )

    uncertainty_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
    )

    enable_future_prediction: bool = True
    enable_counterfactuals: bool = True
    enable_trajectory_generation: bool = True

    require_trained_model: bool = True


# ============================================================
# PLANNING RUNTIME
# ============================================================

class PlanningRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    algorithm: Literal[
        "mpc",
        "cem",
        "tree_search",
        "latent",
        "goal_conditioned",
    ] = "cem"

    horizon: int = Field(
        default=20,
        ge=1,
        le=512,
    )

    candidates: int = Field(
        default=256,
        ge=1,
    )

    iterations: int = Field(
        default=10,
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
    enable_constraint_validation: bool = True

    require_human_approval_for_external_actions: bool = True

    @model_validator(mode="after")
    def validate_candidates(self):
        if self.candidates * self.elite_fraction < 1:
            raise ValueError(
                "At least one elite candidate is required"
            )

        return self


# ============================================================
# SIMULATION RUNTIME
# ============================================================

class SimulationRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True

    environment_name: str = "synthetic_physics"

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

    gravity: float = Field(
        default=9.81,
        ge=0.0,
    )

    enable_collision_detection: bool = True
    enable_physics_constraints: bool = True
    enable_recording: bool = True

    synthetic_only: bool = True

    allow_real_world_actuation: bool = False

    require_human_approval: bool = True

    @model_validator(mode="after")
    def validate_safety(self):
        if self.allow_real_world_actuation:
            raise ValueError(
                "Real-world actuation is not supported "
                "by this core configuration"
            )

        return self


# ============================================================
# MODEL ARTIFACT CONFIGURATION
# ============================================================

class ArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_root: Path = MODELS_DIR

    checkpoint_root: Path = CHECKPOINTS_DIR

    artifact_root: Path = ARTIFACTS_DIR

    cache_root: Path = CACHE_DIR

    allowed_extensions: tuple[str, ...] = (
        ".pt",
        ".pth",
        ".safetensors",
        ".json",
    )

    max_artifact_size_mb: int = Field(
        default=4096,
        ge=1,
    )

    require_manifest: bool = True

    require_checksum: bool = True

    allow_remote_model_downloads: bool = False

    def resolve_artifact(
        self,
        relative_path: str | Path,
    ) -> Path:
        """
        Resolve an artifact inside the configured root.

        Rejects path traversal and unsupported extensions.
        This is path validation, not file authenticity
        verification.
        """

        root = self.model_root.resolve()

        candidate = Path(relative_path)

        if candidate.is_absolute():
            raise ValueError(
                "Artifact path must be relative"
            )

        resolved = (root / candidate).resolve()

        if not resolved.is_relative_to(root):
            raise ValueError(
                "Artifact path escapes model directory"
            )

        if resolved.suffix.lower() not in self.allowed_extensions:
            raise ValueError(
                "Unsupported artifact extension"
            )

        return resolved


# ============================================================
# SAFETY CONFIGURATION
# ============================================================

class SafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_input_validation: bool = True

    enable_output_validation: bool = True

    enable_audit_logging: bool = True

    enable_model_uncertainty_checks: bool = True

    require_human_review: bool = True

    allow_external_actions: bool = False

    allow_autonomous_hardware_control: bool = False

    max_prediction_horizon: int = Field(
        default=128,
        ge=1,
    )

    max_imagination_branches: int = Field(
        default=32,
        ge=1,
    )

    max_planning_candidates: int = Field(
        default=1024,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_policy(self):
        if self.allow_autonomous_hardware_control:
            raise ValueError(
                "Autonomous hardware control "
                "is disabled in this project"
            )

        if self.allow_external_actions:
            raise ValueError(
                "External actions require a separately "
                "reviewed execution service"
            )

        return self


# ============================================================
# CORE CONFIGURATION
# ============================================================

class CoreConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    app_name: str = "NeuroWorld AI"

    app_version: str = "1.0.0"

    environment: Environment = Environment.DEVELOPMENT

    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig
    )

    world: WorldRuntimeConfig = Field(
        default_factory=WorldRuntimeConfig
    )

    imagination: ImaginationRuntimeConfig = Field(
        default_factory=ImaginationRuntimeConfig
    )

    planning: PlanningRuntimeConfig = Field(
        default_factory=PlanningRuntimeConfig
    )

    simulation: SimulationRuntimeConfig = Field(
        default_factory=SimulationRuntimeConfig
    )

    artifacts: ArtifactConfig = Field(
        default_factory=ArtifactConfig
    )

    safety: SafetyConfig = Field(
        default_factory=SafetyConfig
    )

    @model_validator(mode="after")
    def validate_core(self):
        if (
            self.imagination.rollout_horizon
            > self.safety.max_prediction_horizon
        ):
            raise ValueError(
                "Imagination rollout horizon exceeds "
                "the safety limit"
            )

        if (
            self.imagination.max_branches
            > self.safety.max_imagination_branches
        ):
            raise ValueError(
                "Imagination branches exceed safety limit"
            )

        if (
            self.planning.candidates
            > self.safety.max_planning_candidates
        ):
            raise ValueError(
                "Planning candidates exceed safety limit"
            )

        if self.world.enabled is False:
            if self.imagination.enabled:
                raise ValueError(
                    "Imagination requires world model"
                )

            if self.planning.enabled:
                raise ValueError(
                    "Planning requires world model"
                )

        return self

    def ensure_directories(self) -> None:
        """
        Create local development directories.

        Do not call automatically during module import.
        """

        directories = [
            DATA_DIR,
            CACHE_DIR,
            EXPORTS_DIR,
            SIMULATION_DIR,
            LATENT_STATES_DIR,
            TRAINING_DATA_DIR,
            MODELS_DIR,
            CHECKPOINTS_DIR,
            ARTIFACTS_DIR,
            LOGS_DIR,
            EVALUATION_DIR,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    def public_summary(self) -> dict[str, Any]:
        """
        Return a non-sensitive configuration summary.
        """

        return {
            "application": self.app_name,
            "version": self.app_version,
            "environment": self.environment.value,
            "runtime": {
                "device": self.runtime.device,
                "precision": self.runtime.precision,
                "max_concurrent_inference": (
                    self.runtime.max_concurrent_inference
                ),
            },
            "world_model": {
                "enabled": self.world.enabled,
                "architecture": self.world.architecture,
                "latent_dim": self.world.latent_dim,
                "prediction_horizon": (
                    self.world.prediction_horizon
                ),
            },
            "imagination": {
                "enabled": self.imagination.enabled,
                "rollout_horizon": (
                    self.imagination.rollout_horizon
                ),
                "max_branches": (
                    self.imagination.max_branches
                ),
            },
            "planning": {
                "enabled": self.planning.enabled,
                "algorithm": self.planning.algorithm,
                "horizon": self.planning.horizon,
            },
            "simulation": {
                "enabled": self.simulation.enabled,
                "synthetic_only": (
                    self.simulation.synthetic_only
                ),
            },
            "safety": {
                "human_review_required": (
                    self.safety.require_human_review
                ),
                "external_actions_allowed": (
                    self.safety.allow_external_actions
                ),
            },
        }


# ============================================================
# BUILD CONFIGURATION FROM MAIN SETTINGS
# ============================================================

def build_core_config(
    settings: Settings,
) -> CoreConfig:
    """
    Derive core runtime settings from the main Settings.

    The main configuration remains the source of truth.
    """

    selected_device = settings.detect_device()

    precision = settings.inference.precision.value

    # CPU execution defaults to float32.
    if (
        selected_device == "cpu"
        and precision == "float16"
    ):
        precision = "float32"

    return CoreConfig(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment,

        runtime=RuntimeConfig(
            device=selected_device,
            precision=precision,
            seed=settings.random_seed,
            max_concurrent_inference=(
                settings.inference.max_concurrent_requests
            ),
            inference_timeout_seconds=(
                settings.inference.timeout_seconds
            ),
            max_batch_size=settings.inference.batch_size,
            enable_inference_cache=(
                settings.inference.enable_cache
            ),
            cache_ttl_seconds=(
                settings.inference.cache_ttl_seconds
            ),
        ),

        world=WorldRuntimeConfig(
            enabled=settings.enable_world_model,
            architecture=(
                settings.world_model.model_type.value
            ),
            latent_dim=settings.latent.latent_dim,
            state_dim=settings.latent.state_dim,
            action_dim=settings.world_model.action_dim,
            context_length=(
                settings.world_model.context_length
            ),
            prediction_horizon=(
                settings.world_model.prediction_horizon
            ),
            transformer_layers=(
                settings.world_model.transformer_layers
            ),
            attention_heads=(
                settings.world_model.attention_heads
            ),
            hidden_dim=settings.world_model.hidden_dim,
            dropout=settings.world_model.dropout,
            enable_uncertainty=(
                settings.world_model
                .enable_uncertainty_estimation
            ),
            enable_action_conditioning=(
                settings.world_model
                .enable_action_conditioning
            ),
            enable_counterfactuals=(
                settings.world_model
                .enable_counterfactual_prediction
            ),
        ),

        imagination=ImaginationRuntimeConfig(
            enabled=(
                settings.enable_world_model
                and settings.enable_imagination
                and settings.imagination.enabled
            ),
            rollout_horizon=(
                settings.imagination.rollout_horizon
            ),
            max_branches=(
                settings.imagination.max_branches
            ),
            max_parallel_rollouts=(
                settings.imagination.max_parallel_rollouts
            ),
            temperature=(
                settings.imagination.imagination_temperature
            ),
            uncertainty_threshold=(
                settings.imagination.uncertainty_threshold
            ),
            enable_future_prediction=(
                settings.imagination.enable_future_generation
            ),
            enable_counterfactuals=(
                settings.imagination.enable_counterfactuals
            ),
            enable_trajectory_generation=(
                settings.imagination
                .enable_trajectory_generation
            ),
        ),

        planning=PlanningRuntimeConfig(
            enabled=(
                settings.enable_world_model
                and settings.enable_planning
            ),
            algorithm=(
                settings.planning.planner_type.value
            ),
            horizon=settings.planning.planning_horizon,
            candidates=(
                settings.planning.candidate_trajectories
            ),
            iterations=(
                settings.planning.optimization_iterations
            ),
            elite_fraction=(
                settings.planning.elite_fraction
            ),
            discount_factor=(
                settings.planning.discount_factor
            ),
            enable_replanning=(
                settings.planning.enable_replanning
            ),
            enable_constraint_validation=(
                settings.planning.enable_constraint_checking
            ),
            require_human_approval_for_external_actions=(
                settings.security
                .require_human_approval_for_external_actions
            ),
        ),

        simulation=SimulationRuntimeConfig(
            enabled=(
                settings.enable_simulation
                and settings.simulation.enabled
            ),
            environment_name=(
                settings.simulation.default_environment
            ),
            time_step=settings.simulation.time_step,
            max_episode_steps=(
                settings.simulation.max_episode_steps
            ),
            max_parallel_worlds=(
                settings.simulation.max_parallel_worlds
            ),
            gravity=settings.simulation.gravity,
            enable_collision_detection=(
                settings.simulation
                .enable_collision_detection
            ),
            enable_physics_constraints=(
                settings.simulation.enable_gravity
            ),
            enable_recording=(
                settings.simulation
                .enable_scenario_recording
            ),
        ),

        artifacts=ArtifactConfig(),

        safety=SafetyConfig(
            enable_audit_logging=(
                settings.security.enable_audit_logging
            ),
            require_human_review=(
                settings.security
                .require_human_approval_for_external_actions
            ),
        ),
    )


# ============================================================
# CACHED CONFIGURATION
# ============================================================

@lru_cache(maxsize=1)
def get_core_config() -> CoreConfig:
    return build_core_config(
        get_settings()
    )


def reload_core_config() -> CoreConfig:
    """
    Reload the main and core configurations.
    """

    reload_settings()
    get_core_config.cache_clear()

    return get_core_config()


# ============================================================
# FEATURE CHECKS
# ============================================================

def is_feature_enabled(
    feature: Literal[
        "world_model",
        "imagination",
        "planning",
        "simulation",
    ],
) -> bool:
    config = get_core_config()

    return {
        "world_model": config.world.enabled,
        "imagination": config.imagination.enabled,
        "planning": config.planning.enabled,
        "simulation": config.simulation.enabled,
    }[feature]


def require_core_feature(
    feature: Literal[
        "world_model",
        "imagination",
        "planning",
        "simulation",
    ],
) -> None:
    if not is_feature_enabled(feature):
        raise RuntimeError(
            f"Core feature '{feature}' is disabled"
        )


# ============================================================
# DIAGNOSTICS
# ============================================================

def main() -> None:
    config = get_core_config()

    print("=" * 65)
    print("NEUROWORLD AI — CORE CONFIGURATION")
    print("=" * 65)

    print(
        json.dumps(
            config.public_summary(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print("Paths:")
    print(f"  Project      : {PROJECT_ROOT}")
    print(f"  Models       : {MODELS_DIR}")
    print(f"  Checkpoints  : {CHECKPOINTS_DIR}")
    print(f"  Simulations  : {SIMULATION_DIR}")
    print(f"  Logs         : {LOGS_DIR}")

    print()
    print("Core configuration validated successfully.")


if __name__ == "__main__":
    main()
