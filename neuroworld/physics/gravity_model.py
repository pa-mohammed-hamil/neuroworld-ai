
"""
NeuroWorld AI
File: neuroworld/physics/gravity_model.py

Gravity Simulation & Physics Prediction Engine

Features:
    - Constant gravity in 2D and 3D
    - Earth, Moon and Mars gravity presets
    - Custom gravitational acceleration
    - Newtonian point-mass attraction
    - Multi-body gravitational acceleration
    - Free-fall and projectile prediction
    - Euler and velocity-Verlet integration
    - Batched PyTorch physics simulation
    - Gravity-related energy calculations
    - Trajectory generation
    - Physics consistency metrics
    - World-model integration adapter

Install:
    pip install numpy torch

Run:
    python -m neuroworld.physics.gravity_model

Limitations:
    - Classical Newtonian mechanics
    - No general relativity
    - No collision response
    - No fluid dynamics
    - No atmospheric drag by default
    - Point masses require softening near zero distance
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor


# ============================================================
# CONSTANTS
# ============================================================

GRAVITATIONAL_CONSTANT = 6.67430e-11

EARTH_GRAVITY = 9.80665
MOON_GRAVITY = 1.62
MARS_GRAVITY = 3.721

EPSILON = 1e-12


# ============================================================
# EXCEPTIONS
# ============================================================

class GravityModelError(Exception):
    """Base gravity-model exception."""


class GravityValidationError(GravityModelError):
    """Invalid physical input."""


# ============================================================
# ENUMERATIONS
# ============================================================

class GravityPreset(str, Enum):
    EARTH = "earth"
    MOON = "moon"
    MARS = "mars"
    ZERO = "zero"
    CUSTOM = "custom"


class IntegrationMethod(str, Enum):
    EULER = "euler"
    VELOCITY_VERLET = "velocity_verlet"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GravityConfig:
    spatial_dim: int = 3

    preset: GravityPreset = GravityPreset.EARTH

    gravity_magnitude: float = EARTH_GRAVITY

    timestep: float = 0.02

    integration_method: IntegrationMethod = (
        IntegrationMethod.VELOCITY_VERLET
    )

    gravitational_constant: float = (
        GRAVITATIONAL_CONSTANT
    )

    softening_length: float = 0.01

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if (
            not math.isfinite(self.gravity_magnitude)
            or self.gravity_magnitude < 0
        ):
            raise ValueError(
                "gravity_magnitude must be nonnegative"
            )

        if (
            not math.isfinite(self.timestep)
            or self.timestep <= 0
        ):
            raise ValueError(
                "timestep must be positive"
            )

        if (
            not math.isfinite(
                self.gravitational_constant
            )
            or self.gravitational_constant <= 0
        ):
            raise ValueError(
                "gravitational_constant must be positive"
            )

        if (
            not math.isfinite(self.softening_length)
            or self.softening_length < 0
        ):
            raise ValueError(
                "softening_length must be nonnegative"
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_vector(
    value: Any,
    dimension: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=np.float64,
    )

    if vector.shape != (dimension,):
        raise GravityValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(vector).all():
        raise GravityValidationError(
            f"{name} contains non-finite values"
        )

    return vector


def validate_mass(
    mass: float,
) -> float:
    mass = float(mass)

    if not math.isfinite(mass) or mass <= 0:
        raise GravityValidationError(
            "Mass must be positive and finite"
        )

    return mass


# ============================================================
# GRAVITY FIELD PROTOCOL
# ============================================================

class GravityField(Protocol):
    def acceleration(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        positions: [N,D]
        returns:   [N,D]
        """
        ...


# ============================================================
# UNIFORM GRAVITY FIELD
# ============================================================

class UniformGravityField:
    """
    Constant gravitational acceleration.

    Coordinates:
        2D: [x, y], y is vertical
        3D: [x, y, z], z is vertical

    Gravity points along the negative
    vertical axis.
    """

    def __init__(
        self,
        gravity_vector: np.ndarray | list[float],
    ) -> None:
        self.gravity_vector = np.asarray(
            gravity_vector,
            dtype=np.float64,
        )

        if self.gravity_vector.shape not in (
            (2,),
            (3,),
        ):
            raise GravityValidationError(
                "Gravity vector must be 2D or 3D"
            )

        if not np.isfinite(
            self.gravity_vector
        ).all():
            raise GravityValidationError(
                "Gravity vector must be finite"
            )

    def acceleration(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(
            positions,
            dtype=np.float64,
        )

        if (
            positions.ndim != 2
            or positions.shape[1]
            != len(self.gravity_vector)
        ):
            raise GravityValidationError(
                "Positions must be [N,D]"
            )

        return np.broadcast_to(
            self.gravity_vector,
            positions.shape,
        ).copy()


# ============================================================
# POINT-MASS GRAVITY
# ============================================================

@dataclass
class GravitationalBody:
    mass: float
    position: np.ndarray | list[float]

    body_id: str = "body"

    def __post_init__(self) -> None:
        self.mass = validate_mass(
            self.mass
        )

        self.position = np.asarray(
            self.position,
            dtype=np.float64,
        )

        if self.position.shape not in (
            (2,),
            (3,),
        ):
            raise GravityValidationError(
                "Body position must be 2D or 3D"
            )

        if not np.isfinite(
            self.position
        ).all():
            raise GravityValidationError(
                "Body position must be finite"
            )


class PointMassGravityField:
    """
    Newtonian gravitational field from
    one or more fixed point masses.

    Uses Plummer-style softening:

        a = G*M*r / (|r|^2 + eps^2)^(3/2)

    Source masses are fixed during each
    simulation rollout.
    """

    def __init__(
        self,
        bodies: list[GravitationalBody],
        *,
        gravitational_constant: float = (
            GRAVITATIONAL_CONSTANT
        ),
        softening_length: float = 0.01,
    ) -> None:
        if not bodies:
            raise GravityValidationError(
                "At least one body is required"
            )

        if (
            gravitational_constant <= 0
            or not math.isfinite(
                gravitational_constant
            )
        ):
            raise GravityValidationError(
                "Invalid gravitational constant"
            )

        if (
            softening_length < 0
            or not math.isfinite(
                softening_length
            )
        ):
            raise GravityValidationError(
                "Invalid softening length"
            )

        dimension = len(
            bodies[0].position
        )

        if any(
            len(body.position) != dimension
            for body in bodies
        ):
            raise GravityValidationError(
                "All bodies must share dimensions"
            )

        self.bodies = bodies
        self.dimension = dimension

        self.G = float(
            gravitational_constant
        )

        self.softening_length = float(
            softening_length
        )

    def acceleration(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(
            positions,
            dtype=np.float64,
        )

        if (
            positions.ndim != 2
            or positions.shape[1]
            != self.dimension
        ):
            raise GravityValidationError(
                "Positions must be [N,D]"
            )

        result = np.zeros_like(
            positions
        )

        for body in self.bodies:
            displacement = (
                body.position[None, :]
                - positions
            )

            distance_squared = np.sum(
                displacement ** 2,
                axis=-1,
                keepdims=True,
            )

            softened = (
                distance_squared
                + self.softening_length ** 2
            )

            denominator = np.maximum(
                softened ** 1.5,
                EPSILON,
            )

            result += (
                self.G
                * body.mass
                * displacement
                / denominator
            )

        return result


# ============================================================
# PHYSICS STATE
# ============================================================

@dataclass
class GravityState:
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.asarray(
            self.positions,
            dtype=np.float64,
        )

        self.velocities = np.asarray(
            self.velocities,
            dtype=np.float64,
        )

        self.masses = np.asarray(
            self.masses,
            dtype=np.float64,
        )

        if (
            self.positions.ndim != 2
            or self.positions.shape[1]
            not in (2, 3)
        ):
            raise GravityValidationError(
                "Positions must be [N,2] or [N,3]"
            )

        if (
            self.velocities.shape
            != self.positions.shape
        ):
            raise GravityValidationError(
                "Velocity dimensions mismatch"
            )

        if self.masses.shape != (
            self.positions.shape[0],
        ):
            raise GravityValidationError(
                "Masses must be [N]"
            )

        if (
            not np.isfinite(
                self.positions
            ).all()
            or not np.isfinite(
                self.velocities
            ).all()
            or not np.isfinite(
                self.masses
            ).all()
        ):
            raise GravityValidationError(
                "State contains non-finite values"
            )

        if np.any(self.masses <= 0):
            raise GravityValidationError(
                "Masses must be positive"
            )

    def copy(self) -> GravityState:
        return GravityState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy(),
        )


# ============================================================
# GRAVITY PRESET FACTORY
# ============================================================

def create_uniform_gravity(
    config: GravityConfig,
) -> UniformGravityField:
    magnitudes = {
        GravityPreset.EARTH: EARTH_GRAVITY,
        GravityPreset.MOON: MOON_GRAVITY,
        GravityPreset.MARS: MARS_GRAVITY,
        GravityPreset.ZERO: 0.0,
        GravityPreset.CUSTOM: (
            config.gravity_magnitude
        ),
    }

    magnitude = magnitudes[
        config.preset
    ]

    vector = np.zeros(
        config.spatial_dim,
        dtype=np.float64,
    )

    vertical_axis = (
        1
        if config.spatial_dim == 2
        else 2
    )

    vector[vertical_axis] = (
        -magnitude
    )

    return UniformGravityField(
        vector
    )


# ============================================================
# NUMERICAL GRAVITY INTEGRATOR
# ============================================================

class GravityIntegrator:
    """
    Integrates positions and velocities.

    Euler:
        x_next = x + v*dt
        v_next = v + a*dt

    Velocity Verlet:
        x_next = x + v*dt + 0.5*a*dt^2
        v_next = v + 0.5*(a+a_next)*dt
    """

    def __init__(
        self,
        field: GravityField,
        config: GravityConfig,
    ) -> None:
        self.field = field
        self.config = config

    def step(
        self,
        state: GravityState,
        timestep: float | None = None,
    ) -> GravityState:
        dt = (
            self.config.timestep
            if timestep is None
            else float(timestep)
        )

        if not math.isfinite(dt) or dt <= 0:
            raise GravityValidationError(
                "Timestep must be positive"
            )

        acceleration = (
            self.field.acceleration(
                state.positions
            )
        )

        if (
            acceleration.shape
            != state.positions.shape
        ):
            raise GravityValidationError(
                "Gravity field returned wrong shape"
            )

        if (
            self.config.integration_method
            == IntegrationMethod.EULER
        ):
            next_positions = (
                state.positions
                + state.velocities * dt
            )

            next_velocities = (
                state.velocities
                + acceleration * dt
            )

        else:
            next_positions = (
                state.positions
                + state.velocities * dt
                + 0.5 * acceleration * dt ** 2
            )

            next_acceleration = (
                self.field.acceleration(
                    next_positions
                )
            )

            next_velocities = (
                state.velocities
                + 0.5
                * (
                    acceleration
                    + next_acceleration
                )
                * dt
            )

        return GravityState(
            positions=next_positions,
            velocities=next_velocities,
            masses=state.masses.copy(),
        )


# ============================================================
# TRAJECTORY RESULT
# ============================================================

@dataclass
class GravityTrajectory:
    times: np.ndarray

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    kinetic_energy: np.ndarray

    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(),
            "positions": (
                self.positions.tolist()
            ),
            "velocities": (
                self.velocities.tolist()
            ),
            "accelerations": (
                self.accelerations.tolist()
            ),
            "kinetic_energy": (
                self.kinetic_energy.tolist()
            ),
            "metadata": self.metadata,
        }


# ============================================================
# GRAVITY SIMULATOR
# ============================================================

class GravitySimulator:
    """
    Predicts future positions and velocities.

    Output:
        positions [T+1,N,D]
        velocities [T+1,N,D]
        accelerations [T+1,N,D]
    """

    def __init__(
        self,
        field: GravityField,
        config: GravityConfig,
    ) -> None:
        self.field = field
        self.config = config

        self.integrator = (
            GravityIntegrator(
                field,
                config,
            )
        )

    def simulate(
        self,
        initial_state: GravityState,
        steps: int,
    ) -> GravityTrajectory:
        if steps < 0:
            raise GravityValidationError(
                "steps must be nonnegative"
            )

        if (
            initial_state.positions.shape[1]
            != self.config.spatial_dim
        ):
            raise GravityValidationError(
                "Spatial dimension mismatch"
            )

        state = initial_state.copy()

        positions = [
            state.positions.copy()
        ]

        velocities = [
            state.velocities.copy()
        ]

        accelerations = [
            self.field.acceleration(
                state.positions
            )
        ]

        energies = [
            self.kinetic_energy(
                state
            )
        ]

        for _ in range(steps):
            state = self.integrator.step(
                state
            )

            positions.append(
                state.positions.copy()
            )

            velocities.append(
                state.velocities.copy()
            )

            accelerations.append(
                self.field.acceleration(
                    state.positions
                )
            )

            energies.append(
                self.kinetic_energy(
                    state
                )
            )

        return GravityTrajectory(
            times=np.arange(
                steps + 1,
                dtype=np.float64,
            ) * self.config.timestep,
            positions=np.stack(
                positions
            ),
            velocities=np.stack(
                velocities
            ),
            accelerations=np.stack(
                accelerations
            ),
            kinetic_energy=np.stack(
                energies
            ),
            metadata={
                "steps": steps,
                "timestep": (
                    self.config.timestep
                ),
                "integration_method": (
                    self.config.integration_method.value
                ),
            },
        )

    @staticmethod
    def kinetic_energy(
        state: GravityState,
    ) -> np.ndarray:
        speed_squared = np.sum(
            state.velocities ** 2,
            axis=-1,
        )

        return (
            0.5
            * state.masses
            * speed_squared
        )


# ============================================================
# UNIFORM GRAVITY ENERGY
# ============================================================

class GravityEnergy:
    """
    Energy calculations for uniform gravity.

    The reference potential is zero at
    position coordinate 0 along the
    vertical axis.
    """

    @staticmethod
    def potential_energy(
        masses: np.ndarray,
        positions: np.ndarray,
        gravity_magnitude: float,
    ) -> np.ndarray:
        masses = np.asarray(
            masses,
            dtype=np.float64,
        )

        positions = np.asarray(
            positions,
            dtype=np.float64,
        )

        if (
            positions.ndim != 2
            or positions.shape[1]
            not in (2, 3)
        ):
            raise GravityValidationError(
                "Positions must be [N,2] or [N,3]"
            )

        if masses.shape != (
            positions.shape[0],
        ):
            raise GravityValidationError(
                "Mass shape mismatch"
            )

        if gravity_magnitude < 0:
            raise GravityValidationError(
                "Gravity magnitude must be nonnegative"
            )

        axis = (
            1
            if positions.shape[1] == 2
            else 2
        )

        return (
            masses
            * gravity_magnitude
            * positions[:, axis]
        )

    @staticmethod
    def total_energy(
        state: GravityState,
        gravity_magnitude: float,
    ) -> np.ndarray:
        kinetic = (
            GravitySimulator.kinetic_energy(
                state
            )
        )

        potential = (
            GravityEnergy.potential_energy(
                state.masses,
                state.positions,
                gravity_magnitude,
            )
        )

        return (
            kinetic + potential
        )


# ============================================================
# ANALYTICAL PROJECTILE MODEL
# ============================================================

class ProjectileMotion:
    """
    Exact constant-acceleration trajectory.

    Does not include drag or collisions.
    """

    def __init__(
        self,
        gravity_vector: np.ndarray,
    ) -> None:
        self.gravity_vector = np.asarray(
            gravity_vector,
            dtype=np.float64,
        )

        if self.gravity_vector.shape not in (
            (2,),
            (3,),
        ):
            raise GravityValidationError(
                "Gravity vector must be 2D or 3D"
            )

    def predict(
        self,
        initial_position: np.ndarray,
        initial_velocity: np.ndarray,
        times: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        dimension = len(
            self.gravity_vector
        )

        position = validate_vector(
            initial_position,
            dimension,
            "initial_position",
        )

        velocity = validate_vector(
            initial_velocity,
            dimension,
            "initial_velocity",
        )

        times = np.asarray(
            times,
            dtype=np.float64,
        )

        if (
            times.ndim != 1
            or not np.isfinite(times).all()
            or np.any(times < 0)
        ):
            raise GravityValidationError(
                "times must be a finite, "
                "nonnegative 1D array"
            )

        t = times[:, None]

        positions = (
            position[None, :]
            + velocity[None, :] * t
            + 0.5
            * self.gravity_vector[None, :]
            * t ** 2
        )

        velocities = (
            velocity[None, :]
            + self.gravity_vector[None, :]
            * t
        )

        return (
            positions,
            velocities,
        )


# ============================================================
# PYTORCH GRAVITY MODEL
# ============================================================

class TorchGravityModel(torch.nn.Module):
    """
    Differentiable uniform-gravity model.

    Supports batched tensors:
        positions [B,D]
        velocities [B,D]

    Optional external acceleration:
        [B,D]

    Useful for physics-informed training.
    """

    def __init__(
        self,
        config: GravityConfig,
    ) -> None:
        super().__init__()

        self.config = config

        field = create_uniform_gravity(
            config
        )

        self.register_buffer(
            "gravity_vector",
            torch.tensor(
                field.gravity_vector,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        *,
        timestep: float | None = None,
        external_acceleration: (
            Tensor | None
        ) = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        if (
            positions.ndim != 2
            or positions.shape[1]
            != self.config.spatial_dim
        ):
            raise GravityValidationError(
                "positions must be [B,D]"
            )

        if velocities.shape != positions.shape:
            raise GravityValidationError(
                "Velocity shape mismatch"
            )

        dt = (
            self.config.timestep
            if timestep is None
            else float(timestep)
        )

        if not math.isfinite(dt) or dt <= 0:
            raise GravityValidationError(
                "Invalid timestep"
            )

        gravity = self.gravity_vector.to(
            device=positions.device,
            dtype=positions.dtype,
        )

        acceleration = gravity.expand_as(
            positions
        )

        if external_acceleration is not None:
            if (
                external_acceleration.shape
                != positions.shape
            ):
                raise GravityValidationError(
                    "External acceleration shape mismatch"
                )

            acceleration = (
                acceleration
                + external_acceleration
            )

        next_positions = (
            positions
            + velocities * dt
            + 0.5
            * acceleration
            * dt ** 2
        )

        next_velocities = (
            velocities
            + acceleration * dt
        )

        return (
            next_positions,
            next_velocities,
        )


# ============================================================
# WORLD-MODEL ADAPTER
# ============================================================

class GravityDynamicsAdapter:
    """
    Adapter for neural world-model planning.

    State format:
        [position_D, velocity_D]

    Action:
        external acceleration [D]

    state_dim = 2*spatial_dim
    action_dim = spatial_dim
    """

    def __init__(
        self,
        config: GravityConfig,
    ) -> None:
        self.config = config

        self.model = TorchGravityModel(
            config
        )

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        dimension = (
            self.config.spatial_dim
        )

        if (
            states.ndim != 2
            or states.shape[1]
            != 2 * dimension
        ):
            raise GravityValidationError(
                "States must be [B,2*D]"
            )

        if actions.shape != (
            states.shape[0],
            dimension,
        ):
            raise GravityValidationError(
                "Actions must be [B,D]"
            )

        positions = states[
            :, :dimension
        ]

        velocities = states[
            :, dimension:
        ]

        next_positions, next_velocities = (
            self.model(
                positions,
                velocities,
                external_acceleration=actions,
            )
        )

        return torch.cat(
            [
                next_positions,
                next_velocities,
            ],
            dim=-1,
        )


# ============================================================
# PHYSICS CONSISTENCY EVALUATOR
# ============================================================

@dataclass
class GravityConsistencyResult:
    position_mae: float
    velocity_mae: float
    position_rmse: float
    velocity_rmse: float

    def to_dict(self) -> dict[str, float]:
        return {
            "position_mae": self.position_mae,
            "velocity_mae": self.velocity_mae,
            "position_rmse": self.position_rmse,
            "velocity_rmse": self.velocity_rmse,
        }


class GravityConsistencyEvaluator:
    """
    Compares predicted states with
    gravity-based reference states.
    """

    @staticmethod
    def evaluate(
        predicted_positions: np.ndarray,
        predicted_velocities: np.ndarray,
        reference_positions: np.ndarray,
        reference_velocities: np.ndarray,
    ) -> GravityConsistencyResult:
        predicted_positions = np.asarray(
            predicted_positions,
            dtype=np.float64,
        )

        predicted_velocities = np.asarray(
            predicted_velocities,
            dtype=np.float64,
        )

        reference_positions = np.asarray(
            reference_positions,
            dtype=np.float64,
        )

        reference_velocities = np.asarray(
            reference_velocities,
            dtype=np.float64,
        )

        if (
            predicted_positions.shape
            != reference_positions.shape
            or predicted_velocities.shape
            != reference_velocities.shape
        ):
            raise GravityValidationError(
                "Prediction and reference shapes mismatch"
            )

        position_error = (
            predicted_positions
            - reference_positions
        )

        velocity_error = (
            predicted_velocities
            - reference_velocities
        )

        return GravityConsistencyResult(
            position_mae=float(
                np.mean(
                    np.abs(position_error)
                )
            ),
            velocity_mae=float(
                np.mean(
                    np.abs(velocity_error)
                )
            ),
            position_rmse=float(
                np.sqrt(
                    np.mean(
                        position_error ** 2
                    )
                )
            ),
            velocity_rmse=float(
                np.sqrt(
                    np.mean(
                        velocity_error ** 2
                    )
                )
            ),
        )


# ============================================================
# FACTORY
# ============================================================

def create_gravity_simulator(
    config: GravityConfig | None = None,
) -> GravitySimulator:
    config = (
        config or GravityConfig()
    )

    field = create_uniform_gravity(
        config
    )

    return GravitySimulator(
        field,
        config,
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - GRAVITY MODEL"
    )
    print("=" * 65)

    config = GravityConfig(
        spatial_dim=3,
        preset=GravityPreset.EARTH,
        timestep=0.02,
        integration_method=(
            IntegrationMethod.VELOCITY_VERLET
        ),
    )

    simulator = create_gravity_simulator(
        config
    )

    initial_state = GravityState(
        positions=np.array(
            [
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 20.0],
            ]
        ),
        velocities=np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ),
        masses=np.array(
            [1.0, 2.0]
        ),
    )

    trajectory = simulator.simulate(
        initial_state,
        steps=50,
    )

    print(
        "Trajectory shape:",
        trajectory.positions.shape,
    )

    print(
        "Final positions:\n",
        trajectory.positions[-1],
    )

    print(
        "Final velocities:\n",
        trajectory.velocities[-1],
    )

    print(
        "Final kinetic energy:",
        trajectory.kinetic_energy[-1],
    )

    print("\nPyTorch gravity prediction")

    torch_model = TorchGravityModel(
        config
    )

    positions = torch.tensor(
        [
            [0.0, 0.0, 10.0]
        ]
    )

    velocities = torch.tensor(
        [
            [1.0, 0.0, 0.0]
        ]
    )

    next_positions, next_velocities = (
        torch_model(
            positions,
            velocities,
        )
    )

    print(
        "Next positions:",
        next_positions,
    )

    print(
        "Next velocities:",
        next_velocities,
    )

    print(
        "\nGravity model demonstration completed"
    )


if __name__ == "__main__":
    main()
