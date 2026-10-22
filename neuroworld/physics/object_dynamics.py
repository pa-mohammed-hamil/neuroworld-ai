
"""
NeuroWorld AI
File: neuroworld/physics/object_dynamics.py

Object Dynamics & Physics Prediction Engine

Features:
    - 2D and 3D object dynamics
    - Newton's second law (F = ma)
    - Gravity and external forces
    - Linear drag
    - Static and dynamic objects
    - Impulse-based velocity changes
    - Object mass and physical properties
    - Semi-implicit Euler integration
    - Velocity-Verlet integration
    - Batched multi-object simulation
    - Object trajectory prediction
    - Momentum and kinetic energy
    - PyTorch differentiable dynamics
    - World-model planning adapter
    - Physics consistency evaluation
    - Serialization and demonstration

Dependencies:
    pip install numpy torch

Run:
    python -m neuroworld.physics.object_dynamics

Limitations:
    - Translational point-mass dynamics
    - No rotational inertia or torque
    - No collision detection or resolution
    - No articulated-body constraints
    - No deformable-body simulation
    - Not a safety-certified physics engine
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import Tensor


EPSILON = 1e-12


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectDynamicsError(Exception):
    """Base object dynamics exception."""


class ObjectDynamicsValidationError(
    ObjectDynamicsError
):
    """Invalid dynamics configuration or state."""


# ============================================================
# ENUMERATIONS
# ============================================================

class ObjectBodyType(str, Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"


class DynamicsIntegrator(str, Enum):
    SEMI_IMPLICIT_EULER = "semi_implicit_euler"
    VELOCITY_VERLET = "velocity_verlet"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObjectDynamicsConfig:
    spatial_dim: int = 3

    timestep: float = 0.02

    enable_gravity: bool = True
    gravity_magnitude: float = 9.80665

    linear_drag: float = 0.05

    max_speed: float = 100.0
    max_acceleration: float = 100.0

    integrator: DynamicsIntegrator = (
        DynamicsIntegrator.VELOCITY_VERLET
    )

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        positive_fields = (
            "timestep",
            "max_speed",
            "max_acceleration",
        )

        for name in positive_fields:
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be positive"
                )

        nonnegative_fields = (
            "gravity_magnitude",
            "linear_drag",
        )

        for name in nonnegative_fields:
            value = getattr(self, name)

            if (
                not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be nonnegative"
                )


# ============================================================
# VALIDATION HELPERS
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
        raise ObjectDynamicsValidationError(
            f"{name} must have shape ({dimension},)"
        )

    if not np.isfinite(vector).all():
        raise ObjectDynamicsValidationError(
            f"{name} contains non-finite values"
        )

    return vector


def validate_mass(
    mass: float,
) -> float:
    mass = float(mass)

    if (
        not math.isfinite(mass)
        or mass <= 0
    ):
        raise ObjectDynamicsValidationError(
            "Mass must be positive and finite"
        )

    return mass


def clip_norm(
    vector: np.ndarray,
    maximum: float,
) -> np.ndarray:
    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude <= maximum:
        return vector.copy()

    if magnitude <= EPSILON:
        return vector.copy()

    return (
        vector
        * maximum
        / magnitude
    )


# ============================================================
# OBJECT PHYSICAL PROPERTIES
# ============================================================

@dataclass(frozen=True)
class ObjectPhysicalProperties:
    mass: float = 1.0

    drag_coefficient: float = 0.05

    restitution: float = 0.2
    friction: float = 0.5

    def __post_init__(self) -> None:
        validate_mass(
            self.mass
        )

        if (
            not math.isfinite(
                self.drag_coefficient
            )
            or self.drag_coefficient < 0
        ):
            raise ValueError(
                "drag_coefficient must be nonnegative"
            )

        if not (
            0 <= self.restitution <= 1
        ):
            raise ValueError(
                "restitution must be in [0,1]"
            )

        if (
            not math.isfinite(self.friction)
            or self.friction < 0
        ):
            raise ValueError(
                "friction must be nonnegative"
            )


# ============================================================
# OBJECT STATE
# ============================================================

@dataclass
class ObjectState:
    object_id: str

    position: np.ndarray | list[float]
    velocity: np.ndarray | list[float]

    properties: ObjectPhysicalProperties = field(
        default_factory=ObjectPhysicalProperties
    )

    body_type: ObjectBodyType = (
        ObjectBodyType.DYNAMIC
    )

    accumulated_force: np.ndarray = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.position = np.asarray(
            self.position,
            dtype=np.float64,
        )

        if self.position.shape not in (
            (2,),
            (3,),
        ):
            raise ObjectDynamicsValidationError(
                "Position must be 2D or 3D"
            )

        self.velocity = validate_vector(
            self.velocity,
            len(self.position),
            "velocity",
        )

        if not np.isfinite(
            self.position
        ).all():
            raise ObjectDynamicsValidationError(
                "Position must be finite"
            )

        self.accumulated_force = np.zeros_like(
            self.position
        )

        if self.body_type == ObjectBodyType.STATIC:
            self.velocity[:] = 0.0

    @property
    def dimension(self) -> int:
        return len(
            self.position
        )

    @property
    def mass(self) -> float:
        return self.properties.mass

    @property
    def inverse_mass(self) -> float:
        if self.body_type == ObjectBodyType.STATIC:
            return 0.0

        return 1.0 / self.mass

    @property
    def speed(self) -> float:
        return float(
            np.linalg.norm(
                self.velocity
            )
        )

    @property
    def kinetic_energy(self) -> float:
        if self.body_type == ObjectBodyType.STATIC:
            return 0.0

        return (
            0.5
            * self.mass
            * float(
                np.dot(
                    self.velocity,
                    self.velocity,
                )
            )
        )

    @property
    def momentum(self) -> np.ndarray:
        if self.body_type == ObjectBodyType.STATIC:
            return np.zeros_like(
                self.velocity
            )

        return (
            self.mass
            * self.velocity
        )

    def apply_force(
        self,
        force: np.ndarray | list[float],
    ) -> None:
        if self.body_type == ObjectBodyType.STATIC:
            return

        self.accumulated_force += (
            validate_vector(
                force,
                self.dimension,
                "force",
            )
        )

    def apply_impulse(
        self,
        impulse: np.ndarray | list[float],
    ) -> None:
        if self.body_type == ObjectBodyType.STATIC:
            return

        impulse = validate_vector(
            impulse,
            self.dimension,
            "impulse",
        )

        self.velocity += (
            impulse
            * self.inverse_mass
        )

    def clear_forces(self) -> None:
        self.accumulated_force.fill(
            0.0
        )

    def copy(self) -> ObjectState:
        result = ObjectState(
            object_id=self.object_id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            properties=self.properties,
            body_type=self.body_type,
        )

        result.accumulated_force = (
            self.accumulated_force.copy()
        )

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "position": (
                self.position.tolist()
            ),
            "velocity": (
                self.velocity.tolist()
            ),
            "mass": self.mass,
            "body_type": (
                self.body_type.value
            ),
            "speed": self.speed,
            "kinetic_energy": (
                self.kinetic_energy
            ),
        }


# ============================================================
# FORCE MODEL
# ============================================================

class ObjectForceModel:
    """
    Calculates total force on an object.

    F_total = F_external + F_gravity + F_drag

    Gravity:
        F_g = m*g

    Linear drag:
        F_drag = -c*v
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig,
    ) -> None:
        self.config = config

    def gravity_vector(self) -> np.ndarray:
        gravity = np.zeros(
            self.config.spatial_dim,
            dtype=np.float64,
        )

        if self.config.enable_gravity:
            axis = (
                1
                if self.config.spatial_dim == 2
                else 2
            )

            gravity[axis] = (
                -self.config.gravity_magnitude
            )

        return gravity

    def total_force(
        self,
        state: ObjectState,
    ) -> np.ndarray:
        if state.body_type == ObjectBodyType.STATIC:
            return np.zeros(
                state.dimension,
                dtype=np.float64,
            )

        if (
            state.dimension
            != self.config.spatial_dim
        ):
            raise ObjectDynamicsValidationError(
                "Spatial dimension mismatch"
            )

        gravity_force = (
            state.mass
            * self.gravity_vector()
        )

        drag_force = (
            -state.properties.drag_coefficient
            * state.velocity
        )

        return (
            state.accumulated_force
            + gravity_force
            + drag_force
        )

    def acceleration(
        self,
        state: ObjectState,
    ) -> np.ndarray:
        if state.body_type == ObjectBodyType.STATIC:
            return np.zeros(
                state.dimension,
                dtype=np.float64,
            )

        acceleration = (
            self.total_force(state)
            * state.inverse_mass
        )

        return clip_norm(
            acceleration,
            self.config.max_acceleration,
        )


# ============================================================
# OBJECT DYNAMICS INTEGRATOR
# ============================================================

class ObjectDynamicsIntegrator:
    """
    Numerical object-state integration.

    Semi-implicit Euler:
        v_next = v + a*dt
        x_next = x + v_next*dt

    Velocity-Verlet:
        x_next = x + v*dt + 0.5*a*dt^2
        v_next = v + 0.5*(a+a_next)*dt

    With velocity-dependent drag,
    acceleration is recomputed at
    the predicted next velocity.

    Velocity clipping is approximate
    and can break energy conservation.
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig,
    ) -> None:
        self.config = config

        self.force_model = (
            ObjectForceModel(
                config
            )
        )

    def step(
        self,
        state: ObjectState,
        timestep: float | None = None,
    ) -> ObjectState:
        dt = (
            self.config.timestep
            if timestep is None
            else float(timestep)
        )

        if (
            not math.isfinite(dt)
            or dt <= 0
        ):
            raise ObjectDynamicsValidationError(
                "Invalid timestep"
            )

        result = state.copy()

        if state.body_type == ObjectBodyType.STATIC:
            result.clear_forces()
            return result

        acceleration = (
            self.force_model.acceleration(
                state
            )
        )

        initial_velocity = (
            state.velocity.copy()
        )

        if (
            self.config.integrator
            == DynamicsIntegrator.SEMI_IMPLICIT_EULER
        ):
            next_velocity = (
                initial_velocity
                + acceleration * dt
            )

            next_velocity = clip_norm(
                next_velocity,
                self.config.max_speed,
            )

            next_position = (
                state.position
                + next_velocity * dt
            )

        else:
            next_position = (
                state.position
                + initial_velocity * dt
                + 0.5
                * acceleration
                * dt ** 2
            )

            provisional_velocity = (
                initial_velocity
                + acceleration * dt
            )

            provisional = state.copy()
            provisional.position = (
                next_position.copy()
            )
            provisional.velocity = (
                provisional_velocity
            )

            next_acceleration = (
                self.force_model.acceleration(
                    provisional
                )
            )

            next_velocity = (
                initial_velocity
                + 0.5
                * (
                    acceleration
                    + next_acceleration
                )
                * dt
            )

            next_velocity = clip_norm(
                next_velocity,
                self.config.max_speed,
            )

        result.position = (
            next_position
        )

        result.velocity = (
            next_velocity
        )

        result.clear_forces()

        return result


# ============================================================
# OBJECT TRAJECTORY
# ============================================================

@dataclass
class ObjectTrajectory:
    object_id: str

    times: np.ndarray

    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray

    kinetic_energies: np.ndarray

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
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
            "kinetic_energies": (
                self.kinetic_energies.tolist()
            ),
            "metadata": self.metadata,
        }


# ============================================================
# OBJECT TRAJECTORY PREDICTOR
# ============================================================

class ObjectTrajectoryPredictor:
    """
    Predicts future object states.

    Applied forces persist for one
    integration step unless reapplied.
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig,
    ) -> None:
        self.config = config

        self.integrator = (
            ObjectDynamicsIntegrator(
                config
            )
        )

        self.force_model = (
            ObjectForceModel(
                config
            )
        )

    def predict(
        self,
        initial_state: ObjectState,
        steps: int,
    ) -> ObjectTrajectory:
        if steps < 0:
            raise ObjectDynamicsValidationError(
                "steps must be nonnegative"
            )

        state = initial_state.copy()

        positions = [
            state.position.copy()
        ]

        velocities = [
            state.velocity.copy()
        ]

        accelerations = [
            self.force_model.acceleration(
                state
            )
        ]

        energies = [
            state.kinetic_energy
        ]

        for _ in range(steps):
            state = self.integrator.step(
                state
            )

            positions.append(
                state.position.copy()
            )

            velocities.append(
                state.velocity.copy()
            )

            accelerations.append(
                self.force_model.acceleration(
                    state
                )
            )

            energies.append(
                state.kinetic_energy
            )

        times = np.arange(
            steps + 1,
            dtype=np.float64,
        ) * self.config.timestep

        return ObjectTrajectory(
            object_id=initial_state.object_id,
            times=times,
            positions=np.stack(
                positions
            ),
            velocities=np.stack(
                velocities
            ),
            accelerations=np.stack(
                accelerations
            ),
            kinetic_energies=np.asarray(
                energies,
                dtype=np.float64,
            ),
            metadata={
                "steps": steps,
                "timestep": (
                    self.config.timestep
                ),
                "integrator": (
                    self.config.integrator.value
                ),
            },
        )


# ============================================================
# MULTI-OBJECT WORLD
# ============================================================

@dataclass
class MultiObjectStepResult:
    time: float

    positions: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]

    total_kinetic_energy: float
    total_momentum: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "positions": {
                key: value.tolist()
                for key, value
                in self.positions.items()
            },
            "velocities": {
                key: value.tolist()
                for key, value
                in self.velocities.items()
            },
            "total_kinetic_energy": (
                self.total_kinetic_energy
            ),
            "total_momentum": (
                self.total_momentum.tolist()
            ),
        }


class MultiObjectDynamicsWorld:
    """
    Simulates independent objects.

    Does not resolve collisions.
    For object-object contact use
    interaction_model.py.
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig,
    ) -> None:
        self.config = config

        self.integrator = (
            ObjectDynamicsIntegrator(
                config
            )
        )

        self.objects: dict[
            str,
            ObjectState,
        ] = {}

        self.time = 0.0

    def add_object(
        self,
        state: ObjectState,
    ) -> None:
        if (
            state.dimension
            != self.config.spatial_dim
        ):
            raise ObjectDynamicsValidationError(
                "Spatial dimension mismatch"
            )

        if state.object_id in self.objects:
            raise ObjectDynamicsValidationError(
                "Duplicate object ID"
            )

        self.objects[
            state.object_id
        ] = state.copy()

    def get_object(
        self,
        object_id: str,
    ) -> ObjectState:
        if object_id not in self.objects:
            raise KeyError(
                f"Unknown object: {object_id}"
            )

        return self.objects[
            object_id
        ]

    def apply_force(
        self,
        object_id: str,
        force: np.ndarray | list[float],
    ) -> None:
        self.get_object(
            object_id
        ).apply_force(
            force
        )

    def apply_impulse(
        self,
        object_id: str,
        impulse: np.ndarray | list[float],
    ) -> None:
        self.get_object(
            object_id
        ).apply_impulse(
            impulse
        )

    def step(
        self,
    ) -> MultiObjectStepResult:
        next_objects = {}

        for object_id, state in self.objects.items():
            next_objects[
                object_id
            ] = self.integrator.step(
                state
            )

        self.objects = next_objects

        self.time += (
            self.config.timestep
        )

        total_momentum = np.zeros(
            self.config.spatial_dim,
            dtype=np.float64,
        )

        total_energy = 0.0

        for state in self.objects.values():
            total_momentum += (
                state.momentum
            )

            total_energy += (
                state.kinetic_energy
            )

        return MultiObjectStepResult(
            time=self.time,
            positions={
                key: state.position.copy()
                for key, state
                in self.objects.items()
            },
            velocities={
                key: state.velocity.copy()
                for key, state
                in self.objects.items()
            },
            total_kinetic_energy=(
                total_energy
            ),
            total_momentum=(
                total_momentum
            ),
        )

    def simulate(
        self,
        steps: int,
    ) -> list[MultiObjectStepResult]:
        if steps < 0:
            raise ObjectDynamicsValidationError(
                "steps must be nonnegative"
            )

        return [
            self.step()
            for _ in range(steps)
        ]


# ============================================================
# TORCH OBJECT DYNAMICS
# ============================================================

class TorchObjectDynamics(torch.nn.Module):
    """
    Differentiable batched object dynamics.

    states:
        [B,N,2*D]

    Layout per object:
        [position_D,velocity_D]

    forces:
        [B,N,D]

    masses:
        [N] or [B,N]

    dynamic_mask:
        [N] or [B,N]
        True = dynamic, False = static

    Output:
        next_states [B,N,2*D]

    Supports gravity, drag, and
    speed/acceleration clipping.

    No collision handling.
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig,
    ) -> None:
        super().__init__()

        self.config = config

        gravity = torch.zeros(
            config.spatial_dim,
            dtype=torch.float32,
        )

        if config.enable_gravity:
            axis = (
                1
                if config.spatial_dim == 2
                else 2
            )

            gravity[axis] = (
                -config.gravity_magnitude
            )

        self.register_buffer(
            "gravity_vector",
            gravity,
        )

    @staticmethod
    def _clip_norm(
        vectors: Tensor,
        maximum: float,
    ) -> Tensor:
        norm = torch.linalg.vector_norm(
            vectors,
            dim=-1,
            keepdim=True,
        )

        scale = (
            maximum
            / norm.clamp_min(
                EPSILON
            )
        ).clamp(max=1.0)

        return vectors * scale

    def forward(
        self,
        states: Tensor,
        forces: Tensor,
        masses: Tensor,
        dynamic_mask: Tensor | None = None,
    ) -> Tensor:
        dimension = (
            self.config.spatial_dim
        )

        if (
            states.ndim != 3
            or states.shape[-1]
            != 2 * dimension
        ):
            raise ObjectDynamicsValidationError(
                "States must be [B,N,2*D]"
            )

        batch_size, object_count, _ = (
            states.shape
        )

        if forces.shape != (
            batch_size,
            object_count,
            dimension,
        ):
            raise ObjectDynamicsValidationError(
                "Forces must be [B,N,D]"
            )

        if masses.ndim == 1:
            if masses.shape != (
                object_count,
            ):
                raise ObjectDynamicsValidationError(
                    "Masses must be [N]"
                )

            masses = masses.unsqueeze(
                0
            ).expand(
                batch_size,
                -1,
            )

        elif masses.shape != (
            batch_size,
            object_count,
        ):
            raise ObjectDynamicsValidationError(
                "Masses must be [N] or [B,N]"
            )

        masses = masses.to(
            device=states.device,
            dtype=states.dtype,
        )

        if (
            not torch.isfinite(states).all()
            or not torch.isfinite(forces).all()
            or not torch.isfinite(masses).all()
            or (masses <= 0).any()
        ):
            raise ObjectDynamicsValidationError(
                "Inputs must be finite; masses positive"
            )

        if dynamic_mask is None:
            dynamic_mask = torch.ones(
                (
                    batch_size,
                    object_count,
                ),
                dtype=torch.bool,
                device=states.device,
            )

        elif dynamic_mask.ndim == 1:
            if dynamic_mask.shape != (
                object_count,
            ):
                raise ObjectDynamicsValidationError(
                    "Invalid dynamic mask"
                )

            dynamic_mask = (
                dynamic_mask.to(
                    device=states.device,
                    dtype=torch.bool,
                )
                .unsqueeze(0)
                .expand(
                    batch_size,
                    -1,
                )
            )

        elif dynamic_mask.shape == (
            batch_size,
            object_count,
        ):
            dynamic_mask = dynamic_mask.to(
                device=states.device,
                dtype=torch.bool,
            )

        else:
            raise ObjectDynamicsValidationError(
                "Invalid dynamic mask shape"
            )

        positions = states[
            :, :, :dimension
        ]

        velocities = states[
            :, :, dimension:
        ]

        gravity = self.gravity_vector.to(
            device=states.device,
            dtype=states.dtype,
        )

        gravity = gravity.view(
            1,
            1,
            dimension,
        )

        acceleration = (
            forces
            / masses.unsqueeze(-1)
            + gravity
            - self.config.linear_drag
            * velocities
            / masses.unsqueeze(-1)
        )

        acceleration = self._clip_norm(
            acceleration,
            self.config.max_acceleration,
        )

        dt = self.config.timestep

        next_velocities = (
            velocities
            + acceleration * dt
        )

        next_velocities = self._clip_norm(
            next_velocities,
            self.config.max_speed,
        )

        if (
            self.config.integrator
            == DynamicsIntegrator.VELOCITY_VERLET
        ):
            # Approximate Verlet update.
            next_positions = (
                positions
                + velocities * dt
                + 0.5
                * acceleration
                * dt ** 2
            )

        else:
            next_positions = (
                positions
                + next_velocities * dt
            )

        mask = dynamic_mask.unsqueeze(
            -1
        )

        next_positions = torch.where(
            mask,
            next_positions,
            positions,
        )

        next_velocities = torch.where(
            mask,
            next_velocities,
            torch.zeros_like(
                velocities
            ),
        )

        return torch.cat(
            [
                next_positions,
                next_velocities,
            ],
            dim=-1,
        )


# ============================================================
# WORLD-MODEL ADAPTER
# ============================================================

class ObjectDynamicsAdapter:
    """
    Flattened world-model interface.

    State:
        [B,N*2*D]

    Action:
        [B,N*D]

    Returns:
        [B,N*2*D]

    Actions represent external forces,
    not acceleration commands.
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig,
        masses: Tensor | np.ndarray,
        dynamic_mask: (
            Tensor | np.ndarray | None
        ) = None,
    ) -> None:
        self.config = config

        self.masses = torch.as_tensor(
            masses,
            dtype=torch.float32,
        )

        if self.masses.ndim != 1:
            raise ObjectDynamicsValidationError(
                "masses must be [N]"
            )

        if (
            not torch.isfinite(self.masses).all()
            or (self.masses <= 0).any()
        ):
            raise ObjectDynamicsValidationError(
                "Invalid masses"
            )

        self.object_count = (
            self.masses.shape[0]
        )

        self.dynamic_mask = (
            None
            if dynamic_mask is None
            else torch.as_tensor(
                dynamic_mask,
                dtype=torch.bool,
            )
        )

        if (
            self.dynamic_mask is not None
            and self.dynamic_mask.shape
            != (self.object_count,)
        ):
            raise ObjectDynamicsValidationError(
                "dynamic_mask must be [N]"
            )

        self.model = TorchObjectDynamics(
            config
        )

        self.state_dim = (
            self.object_count
            * 2
            * config.spatial_dim
        )

        self.action_dim = (
            self.object_count
            * config.spatial_dim
        )

    def predict_next_state(
        self,
        states: Tensor,
        actions: Tensor,
    ) -> Tensor:
        if (
            states.ndim != 2
            or states.shape[1]
            != self.state_dim
        ):
            raise ObjectDynamicsValidationError(
                "Invalid flattened state shape"
            )

        if actions.shape != (
            states.shape[0],
            self.action_dim,
        ):
            raise ObjectDynamicsValidationError(
                "Invalid flattened action shape"
            )

        batch_size = states.shape[0]

        dimension = (
            self.config.spatial_dim
        )

        object_states = states.reshape(
            batch_size,
            self.object_count,
            2 * dimension,
        )

        object_forces = actions.reshape(
            batch_size,
            self.object_count,
            dimension,
        )

        next_states = self.model(
            object_states,
            object_forces,
            self.masses.to(
                device=states.device,
                dtype=states.dtype,
            ),
            (
                None
                if self.dynamic_mask is None
                else self.dynamic_mask.to(
                    device=states.device
                )
            ),
        )

        return next_states.reshape(
            batch_size,
            self.state_dim,
        )


# ============================================================
# PHYSICS CONSISTENCY EVALUATOR
# ============================================================

@dataclass
class ObjectDynamicsMetrics:
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


class ObjectDynamicsEvaluator:
    """
    Compares predicted object states
    with reference trajectories.
    """

    @staticmethod
    def evaluate(
        predicted: np.ndarray,
        reference: np.ndarray,
        spatial_dim: int,
    ) -> ObjectDynamicsMetrics:
        predicted = np.asarray(
            predicted,
            dtype=np.float64,
        )

        reference = np.asarray(
            reference,
            dtype=np.float64,
        )

        if predicted.shape != reference.shape:
            raise ObjectDynamicsValidationError(
                "Prediction/reference shape mismatch"
            )

        if (
            predicted.ndim < 2
            or predicted.shape[-1]
            != 2 * spatial_dim
        ):
            raise ObjectDynamicsValidationError(
                "Last dimension must be 2*D"
            )

        position_error = (
            predicted[..., :spatial_dim]
            - reference[..., :spatial_dim]
        )

        velocity_error = (
            predicted[..., spatial_dim:]
            - reference[..., spatial_dim:]
        )

        return ObjectDynamicsMetrics(
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
# MAIN OBJECT DYNAMICS MODEL
# ============================================================

class ObjectDynamicsModel:
    """
    High-level object dynamics API.
    """

    def __init__(
        self,
        config: ObjectDynamicsConfig | None = None,
    ) -> None:
        self.config = (
            config or ObjectDynamicsConfig()
        )

        self.force_model = (
            ObjectForceModel(
                self.config
            )
        )

        self.integrator = (
            ObjectDynamicsIntegrator(
                self.config
            )
        )

        self.predictor = (
            ObjectTrajectoryPredictor(
                self.config
            )
        )

    def step(
        self,
        state: ObjectState,
    ) -> ObjectState:
        return self.integrator.step(
            state
        )

    def predict(
        self,
        state: ObjectState,
        steps: int,
    ) -> ObjectTrajectory:
        return self.predictor.predict(
            state,
            steps,
        )

    def create_world(
        self,
    ) -> MultiObjectDynamicsWorld:
        return MultiObjectDynamicsWorld(
            self.config
        )


# ============================================================
# FACTORY
# ============================================================

def create_object_dynamics_model(
    config: ObjectDynamicsConfig | None = None,
) -> ObjectDynamicsModel:
    return ObjectDynamicsModel(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - OBJECT DYNAMICS"
    )
    print("=" * 65)

    config = ObjectDynamicsConfig(
        spatial_dim=2,
        timestep=0.02,
        enable_gravity=True,
        gravity_magnitude=9.80665,
        linear_drag=0.02,
    )

    model = ObjectDynamicsModel(
        config
    )

    print("\n1. Single-object prediction")

    ball = ObjectState(
        object_id="ball_01",
        position=[0.0, 10.0],
        velocity=[2.0, 0.0],
        properties=ObjectPhysicalProperties(
            mass=1.0,
            drag_coefficient=0.02,
        ),
    )

    trajectory = model.predict(
        ball,
        steps=50,
    )

    print(
        "Trajectory shape:",
        trajectory.positions.shape,
    )

    print(
        "Final position:",
        trajectory.positions[-1],
    )

    print(
        "Final velocity:",
        trajectory.velocities[-1],
    )

    print("\n2. Multi-object simulation")

    world = model.create_world()

    world.add_object(
        ObjectState(
            object_id="object_A",
            position=[0.0, 5.0],
            velocity=[1.0, 0.0],
            properties=ObjectPhysicalProperties(
                mass=2.0
            ),
        )
    )

    world.add_object(
        ObjectState(
            object_id="object_B",
            position=[2.0, 8.0],
            velocity=[-0.5, 0.0],
            properties=ObjectPhysicalProperties(
                mass=3.0
            ),
        )
    )

    world.apply_force(
        "object_A",
        [10.0, 0.0],
    )

    results = world.simulate(
        steps=30
    )

    print(
        "Final world state:",
        results[-1].to_dict(),
    )

    print("\n3. PyTorch dynamics")

    adapter = ObjectDynamicsAdapter(
        config,
        masses=np.array(
            [2.0, 3.0]
        ),
    )

    states = torch.tensor(
        [
            [
                0.0, 5.0, 1.0, 0.0,
                2.0, 8.0, -0.5, 0.0,
            ]
        ],
        dtype=torch.float32,
    )

    forces = torch.tensor(
        [
            [
                10.0, 0.0,
                0.0, 0.0,
            ]
        ],
        dtype=torch.float32,
    )

    next_states = (
        adapter.predict_next_state(
            states,
            forces,
        )
    )

    print(
        "Next states:",
        next_states,
    )

    print(
        "\nObject dynamics demonstration completed"
    )


if __name__ == "__main__":
    main()
