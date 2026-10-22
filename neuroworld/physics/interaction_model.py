
"""
NeuroWorld AI
File: neuroworld/physics/interaction_model.py

Object Interaction & Physical Dynamics Engine

Features:
    - 2D and 3D rigid-body interaction
    - Sphere/sphere contact detection
    - Contact forces and normal impulses
    - Coefficient of restitution
    - Coulomb friction approximation
    - External forces and impulses
    - Gravity integration
    - Pairwise collision resolution
    - Static and dynamic bodies
    - Momentum and kinetic energy
    - Object pushing and manipulation
    - Multi-object simulation
    - Contact event logging
    - Batched PyTorch interaction features
    - World-model state prediction adapter

Dependencies:
    pip install numpy torch

Run:
    python -m neuroworld.physics.interaction_model

Scope:
    Simplified translational rigid-body physics.
    No rotation, torque, joints, deformable
    bodies, or production-grade contact solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor


EPS = 1e-9


# ============================================================
# EXCEPTIONS
# ============================================================

class InteractionError(Exception):
    """Base physics interaction exception."""


class InteractionValidationError(InteractionError):
    """Invalid physics state or input."""


# ============================================================
# ENUMERATIONS
# ============================================================

class BodyType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class InteractionType(str, Enum):
    CONTACT = "contact"
    COLLISION = "collision"
    PUSH = "push"
    IMPULSE = "impulse"
    FORCE = "force"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class InteractionConfig:
    spatial_dim: int = 2

    timestep: float = 0.02

    gravity_magnitude: float = 9.80665
    enable_gravity: bool = True

    default_restitution: float = 0.2
    default_friction: float = 0.5

    solver_iterations: int = 8

    penetration_slop: float = 0.001
    position_correction: float = 0.8

    max_speed: float = 100.0

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if (
            not math.isfinite(self.timestep)
            or self.timestep <= 0
        ):
            raise ValueError(
                "timestep must be positive"
            )

        if (
            not math.isfinite(self.gravity_magnitude)
            or self.gravity_magnitude < 0
        ):
            raise ValueError(
                "Invalid gravity magnitude"
            )

        for name in (
            "default_restitution",
            "default_friction",
            "position_correction",
        ):
            value = getattr(self, name)

            if not math.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if not (
            0 <= self.default_restitution <= 1
        ):
            raise ValueError(
                "Restitution must be in [0,1]"
            )

        if self.default_friction < 0:
            raise ValueError(
                "Friction must be nonnegative"
            )

        if not (
            0 <= self.position_correction <= 1
        ):
            raise ValueError(
                "Position correction must be in [0,1]"
            )

        if self.solver_iterations <= 0:
            raise ValueError(
                "solver_iterations must be positive"
            )

        if self.penetration_slop < 0:
            raise ValueError(
                "penetration_slop must be nonnegative"
            )

        if self.max_speed <= 0:
            raise ValueError(
                "max_speed must be positive"
            )


# ============================================================
# HELPERS
# ============================================================

def as_vector(
    value: Any,
    dimension: int,
    name: str,
) -> np.ndarray:
    vector = np.asarray(
        value,
        dtype=np.float64,
    )

    if vector.shape != (dimension,):
        raise InteractionValidationError(
            f"{name} must be ({dimension},)"
        )

    if not np.isfinite(vector).all():
        raise InteractionValidationError(
            f"{name} must be finite"
        )

    return vector


def normalize(
    vector: np.ndarray,
) -> np.ndarray:
    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude <= EPS:
        result = np.zeros_like(vector)
        result[0] = 1.0
        return result

    return vector / magnitude


# ============================================================
# PHYSICAL MATERIAL
# ============================================================

@dataclass(frozen=True)
class PhysicalMaterial:
    restitution: float = 0.2
    friction: float = 0.5

    def __post_init__(self) -> None:
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
# RIGID BODY
# ============================================================

@dataclass
class RigidBody:
    body_id: str

    position: np.ndarray | list[float]
    velocity: np.ndarray | list[float]

    radius: float
    mass: float = 1.0

    body_type: BodyType = BodyType.DYNAMIC

    material: PhysicalMaterial = field(
        default_factory=PhysicalMaterial
    )

    force: np.ndarray = field(
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
            raise InteractionValidationError(
                "Position must be 2D or 3D"
            )

        self.velocity = as_vector(
            self.velocity,
            len(self.position),
            "velocity",
        )

        if not np.isfinite(
            self.position
        ).all():
            raise InteractionValidationError(
                "Position must be finite"
            )

        if (
            not math.isfinite(self.radius)
            or self.radius <= 0
        ):
            raise InteractionValidationError(
                "radius must be positive"
            )

        if (
            not math.isfinite(self.mass)
            or self.mass <= 0
        ):
            raise InteractionValidationError(
                "mass must be positive"
            )

        self.force = np.zeros_like(
            self.position
        )

        if self.body_type == BodyType.STATIC:
            self.velocity[:] = 0.0

    @property
    def dimension(self) -> int:
        return len(self.position)

    @property
    def inverse_mass(self) -> float:
        if self.body_type == BodyType.STATIC:
            return 0.0

        return 1.0 / self.mass

    @property
    def momentum(self) -> np.ndarray:
        if self.body_type == BodyType.STATIC:
            return np.zeros_like(
                self.velocity
            )

        return self.mass * self.velocity

    @property
    def kinetic_energy(self) -> float:
        if self.body_type == BodyType.STATIC:
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

    def apply_force(
        self,
        force: np.ndarray | list[float],
    ) -> None:
        if self.body_type == BodyType.STATIC:
            return

        self.force += as_vector(
            force,
            self.dimension,
            "force",
        )

    def apply_impulse(
        self,
        impulse: np.ndarray | list[float],
    ) -> None:
        if self.body_type == BodyType.STATIC:
            return

        impulse = as_vector(
            impulse,
            self.dimension,
            "impulse",
        )

        self.velocity += (
            impulse
            * self.inverse_mass
        )

    def clear_forces(self) -> None:
        self.force.fill(0.0)

    def copy(self) -> RigidBody:
        return RigidBody(
            body_id=self.body_id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            radius=self.radius,
            mass=self.mass,
            body_type=self.body_type,
            material=self.material,
        )


# ============================================================
# CONTACT INFORMATION
# ============================================================

@dataclass
class Contact:
    body_a: str
    body_b: str

    normal: np.ndarray

    penetration: float
    contact_point: np.ndarray

    relative_normal_velocity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_a": self.body_a,
            "body_b": self.body_b,
            "normal": self.normal.tolist(),
            "penetration": self.penetration,
            "contact_point": (
                self.contact_point.tolist()
            ),
            "relative_normal_velocity": (
                self.relative_normal_velocity
            ),
        }


# ============================================================
# CONTACT DETECTOR
# ============================================================

class ContactDetector:
    """
    Sphere/sphere contact detection.

    Uses center distances and combined
    collision radii.

    Normal points from A toward B.
    """

    def detect(
        self,
        a: RigidBody,
        b: RigidBody,
    ) -> Contact | None:
        if a.dimension != b.dimension:
            raise InteractionValidationError(
                "Body dimensions mismatch"
            )

        displacement = (
            b.position - a.position
        )

        distance = float(
            np.linalg.norm(
                displacement
            )
        )

        combined_radius = (
            a.radius + b.radius
        )

        if distance > combined_radius:
            return None

        if distance <= EPS:
            relative_velocity = (
                b.velocity - a.velocity
            )

            normal = normalize(
                relative_velocity
            )
        else:
            normal = (
                displacement / distance
            )

        penetration = max(
            0.0,
            combined_radius - distance,
        )

        contact_point = (
            a.position
            + normal
            * (
                a.radius
                - penetration * 0.5
            )
        )

        relative_velocity = (
            b.velocity - a.velocity
        )

        relative_normal_velocity = float(
            np.dot(
                relative_velocity,
                normal,
            )
        )

        return Contact(
            body_a=a.body_id,
            body_b=b.body_id,
            normal=normal,
            penetration=penetration,
            contact_point=contact_point,
            relative_normal_velocity=(
                relative_normal_velocity
            ),
        )


# ============================================================
# CONTACT SOLVER
# ============================================================

@dataclass
class ContactResolution:
    normal_impulse: float
    friction_impulse: float

    position_corrected: bool


class ContactSolver:
    """
    Simplified sequential impulse solver.

    Normal impulse:
        j = -(1+e)*(v_rel dot n)
            / (inv_mass_a+inv_mass_b)

    Friction:
        Tangential impulse limited by
        Coulomb friction coefficient.

    Position correction:
        Separates overlapping bodies.

    Does not model angular momentum.
    """

    def __init__(
        self,
        config: InteractionConfig,
    ) -> None:
        self.config = config

    def resolve(
        self,
        a: RigidBody,
        b: RigidBody,
        contact: Contact,
    ) -> ContactResolution:
        inverse_mass_sum = (
            a.inverse_mass
            + b.inverse_mass
        )

        if inverse_mass_sum <= EPS:
            return ContactResolution(
                normal_impulse=0.0,
                friction_impulse=0.0,
                position_corrected=False,
            )

        normal = contact.normal

        relative_velocity = (
            b.velocity - a.velocity
        )

        normal_speed = float(
            np.dot(
                relative_velocity,
                normal,
            )
        )

        normal_impulse_magnitude = 0.0
        friction_impulse_magnitude = 0.0

        if normal_speed < 0:
            restitution = min(
                a.material.restitution,
                b.material.restitution,
            )

            normal_impulse_magnitude = (
                -(1.0 + restitution)
                * normal_speed
                / inverse_mass_sum
            )

            impulse = (
                normal_impulse_magnitude
                * normal
            )

            a.velocity -= (
                impulse
                * a.inverse_mass
            )

            b.velocity += (
                impulse
                * b.inverse_mass
            )

            # Friction uses the updated
            # relative velocity.
            relative_velocity = (
                b.velocity - a.velocity
            )

            tangent_velocity = (
                relative_velocity
                - float(
                    np.dot(
                        relative_velocity,
                        normal,
                    )
                )
                * normal
            )

            tangent_length = float(
                np.linalg.norm(
                    tangent_velocity
                )
            )

            if tangent_length > EPS:
                tangent = (
                    tangent_velocity
                    / tangent_length
                )

                desired_friction = (
                    -float(
                        np.dot(
                            relative_velocity,
                            tangent,
                        )
                    )
                    / inverse_mass_sum
                )

                friction_coefficient = math.sqrt(
                    a.material.friction
                    * b.material.friction
                )

                friction_limit = (
                    friction_coefficient
                    * normal_impulse_magnitude
                )

                friction_impulse_magnitude = float(
                    np.clip(
                        desired_friction,
                        -friction_limit,
                        friction_limit,
                    )
                )

                friction_impulse = (
                    friction_impulse_magnitude
                    * tangent
                )

                a.velocity -= (
                    friction_impulse
                    * a.inverse_mass
                )

                b.velocity += (
                    friction_impulse
                    * b.inverse_mass
                )

        penetration_error = max(
            0.0,
            contact.penetration
            - self.config.penetration_slop,
        )

        position_corrected = (
            penetration_error > 0
        )

        if position_corrected:
            correction = (
                self.config.position_correction
                * penetration_error
                / inverse_mass_sum
                * normal
            )

            a.position -= (
                correction
                * a.inverse_mass
            )

            b.position += (
                correction
                * b.inverse_mass
            )

        return ContactResolution(
            normal_impulse=(
                normal_impulse_magnitude
            ),
            friction_impulse=abs(
                friction_impulse_magnitude
            ),
            position_corrected=(
                position_corrected
            ),
        )


# ============================================================
# INTERACTION EVENT
# ============================================================

@dataclass
class InteractionEvent:
    event_type: InteractionType

    body_a: str
    body_b: str | None

    time: float

    normal_impulse: float = 0.0
    friction_impulse: float = 0.0

    penetration: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "body_a": self.body_a,
            "body_b": self.body_b,
            "time": self.time,
            "normal_impulse": (
                self.normal_impulse
            ),
            "friction_impulse": (
                self.friction_impulse
            ),
            "penetration": self.penetration,
            "metadata": self.metadata,
        }


# ============================================================
# SIMULATION STEP RESULT
# ============================================================

@dataclass
class InteractionStepResult:
    time: float

    positions: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]

    contacts: list[Contact]
    events: list[InteractionEvent]

    total_kinetic_energy: float

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
            "contacts": [
                contact.to_dict()
                for contact in self.contacts
            ],
            "events": [
                event.to_dict()
                for event in self.events
            ],
            "total_kinetic_energy": (
                self.total_kinetic_energy
            ),
        }


# ============================================================
# INTERACTION WORLD
# ============================================================

class InteractionWorld:
    """
    Multi-object interaction simulator.

    Per step:
        1. Integrate external forces
        2. Apply gravity
        3. Update positions
        4. Detect contacts
        5. Resolve contact impulses
        6. Correct penetrations
        7. Log contact events

    This uses discrete collision detection.
    Fast objects may tunnel through others.
    """

    def __init__(
        self,
        config: InteractionConfig | None = None,
    ) -> None:
        self.config = (
            config or InteractionConfig()
        )

        self.bodies: dict[
            str,
            RigidBody,
        ] = {}

        self.detector = (
            ContactDetector()
        )

        self.solver = (
            ContactSolver(
                self.config
            )
        )

        self.time = 0.0

        self.event_history: list[
            InteractionEvent
        ] = []

    def add_body(
        self,
        body: RigidBody,
    ) -> None:
        if body.dimension != (
            self.config.spatial_dim
        ):
            raise InteractionValidationError(
                "Body dimensions mismatch"
            )

        if body.body_id in self.bodies:
            raise InteractionValidationError(
                "Duplicate body ID"
            )

        self.bodies[
            body.body_id
        ] = body

    def remove_body(
        self,
        body_id: str,
    ) -> None:
        self.bodies.pop(
            body_id,
            None,
        )

    def get_body(
        self,
        body_id: str,
    ) -> RigidBody:
        if body_id not in self.bodies:
            raise KeyError(
                f"Unknown body: {body_id}"
            )

        return self.bodies[
            body_id
        ]

    def apply_force(
        self,
        body_id: str,
        force: np.ndarray | list[float],
    ) -> None:
        self.get_body(
            body_id
        ).apply_force(
            force
        )

    def apply_impulse(
        self,
        body_id: str,
        impulse: np.ndarray | list[float],
    ) -> None:
        self.get_body(
            body_id
        ).apply_impulse(
            impulse
        )

        self.event_history.append(
            InteractionEvent(
                event_type=InteractionType.IMPULSE,
                body_a=body_id,
                body_b=None,
                time=self.time,
            )
        )

    def _gravity_vector(
        self,
    ) -> np.ndarray:
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

    def _integrate(
        self,
        dt: float,
    ) -> None:
        gravity = (
            self._gravity_vector()
        )

        for body in self.bodies.values():
            if body.body_type == BodyType.STATIC:
                body.clear_forces()
                continue

            acceleration = (
                body.force
                * body.inverse_mass
                + gravity
            )

            body.velocity += (
                acceleration * dt
            )

            speed = float(
                np.linalg.norm(
                    body.velocity
                )
            )

            if speed > self.config.max_speed:
                body.velocity *= (
                    self.config.max_speed
                    / speed
                )

            body.position += (
                body.velocity * dt
            )

            body.clear_forces()

    def _detect_contacts(
        self,
    ) -> list[
        tuple[
            RigidBody,
            RigidBody,
            Contact,
        ]
    ]:
        bodies = list(
            self.bodies.values()
        )

        contacts = []

        for i in range(
            len(bodies)
        ):
            for j in range(
                i + 1,
                len(bodies),
            ):
                a = bodies[i]
                b = bodies[j]

                if (
                    a.body_type == BodyType.STATIC
                    and b.body_type == BodyType.STATIC
                ):
                    continue

                contact = self.detector.detect(
                    a,
                    b,
                )

                if contact is not None:
                    contacts.append(
                        (a, b, contact)
                    )

        return contacts

    def step(
        self,
        timestep: float | None = None,
    ) -> InteractionStepResult:
        dt = (
            self.config.timestep
            if timestep is None
            else float(timestep)
        )

        if (
            not math.isfinite(dt)
            or dt <= 0
        ):
            raise InteractionValidationError(
                "Invalid timestep"
            )

        self._integrate(
            dt
        )

        events = []

        # Sequential contact resolution.
        for _ in range(
            self.config.solver_iterations
        ):
            contacts = (
                self._detect_contacts()
            )

            if not contacts:
                break

            for a, b, contact in contacts:
                resolution = (
                    self.solver.resolve(
                        a,
                        b,
                        contact,
                    )
                )

                if (
                    resolution.normal_impulse > EPS
                    or resolution.position_corrected
                ):
                    events.append(
                        InteractionEvent(
                            event_type=(
                                InteractionType.COLLISION
                                if resolution.normal_impulse > EPS
                                else InteractionType.CONTACT
                            ),
                            body_a=a.body_id,
                            body_b=b.body_id,
                            time=self.time + dt,
                            normal_impulse=(
                                resolution.normal_impulse
                            ),
                            friction_impulse=(
                                resolution.friction_impulse
                            ),
                            penetration=(
                                contact.penetration
                            ),
                        )
                    )

        self.time += dt

        self.event_history.extend(
            events
        )

        final_contacts = [
            contact
            for _, _, contact
            in self._detect_contacts()
        ]

        return InteractionStepResult(
            time=self.time,
            positions={
                body_id: body.position.copy()
                for body_id, body
                in self.bodies.items()
            },
            velocities={
                body_id: body.velocity.copy()
                for body_id, body
                in self.bodies.items()
            },
            contacts=final_contacts,
            events=events,
            total_kinetic_energy=sum(
                body.kinetic_energy
                for body in self.bodies.values()
            ),
        )

    def simulate(
        self,
        steps: int,
    ) -> list[InteractionStepResult]:
        if steps < 0:
            raise InteractionValidationError(
                "steps must be nonnegative"
            )

        return [
            self.step()
            for _ in range(steps)
        ]

    def reset(
        self,
        *,
        clear_bodies: bool = False,
    ) -> None:
        self.time = 0.0
        self.event_history.clear()

        if clear_bodies:
            self.bodies.clear()


# ============================================================
# OBJECT PUSHING MODEL
# ============================================================

class ObjectPushingModel:
    """
    Applies a simplified pushing force
    to an object.

    Direction is normalized.
    Force magnitude is in newtons.

    Contact geometry is not verified here;
    caller must confirm physical contact.
    """

    @staticmethod
    def push(
        body: RigidBody,
        direction: np.ndarray | list[float],
        force_magnitude: float,
    ) -> None:
        if (
            not math.isfinite(force_magnitude)
            or force_magnitude < 0
        ):
            raise InteractionValidationError(
                "Force magnitude must be nonnegative"
            )

        direction = as_vector(
            direction,
            body.dimension,
            "direction",
        )

        if np.linalg.norm(direction) <= EPS:
            raise InteractionValidationError(
                "Push direction cannot be zero"
            )

        body.apply_force(
            normalize(direction)
            * force_magnitude
        )


# ============================================================
# INTERACTION FEATURES
# ============================================================

class InteractionFeatureEncoder:
    """
    Encodes pairwise interaction features
    for ML/world-model training.

    Output per object pair:
        relative position [D]
        relative velocity [D]
        center distance [1]
        signed clearance [1]
        combined radius [1]
        inverse mass sum [1]

    Total feature dimension:
        2*D + 4
    """

    def __init__(
        self,
        spatial_dim: int,
    ) -> None:
        if spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        self.spatial_dim = spatial_dim

    def encode_pair(
        self,
        a: RigidBody,
        b: RigidBody,
    ) -> np.ndarray:
        if (
            a.dimension != self.spatial_dim
            or b.dimension != self.spatial_dim
        ):
            raise InteractionValidationError(
                "Body dimension mismatch"
            )

        relative_position = (
            b.position - a.position
        )

        relative_velocity = (
            b.velocity - a.velocity
        )

        distance = float(
            np.linalg.norm(
                relative_position
            )
        )

        combined_radius = (
            a.radius + b.radius
        )

        clearance = (
            distance - combined_radius
        )

        return np.concatenate(
            [
                relative_position,
                relative_velocity,
                np.array(
                    [
                        distance,
                        clearance,
                        combined_radius,
                        (
                            a.inverse_mass
                            + b.inverse_mass
                        ),
                    ],
                    dtype=np.float64,
                ),
            ]
        )


# ============================================================
# TORCH INTERACTION MODEL
# ============================================================

class TorchInteractionModel(torch.nn.Module):
    """
    Batched differentiable interaction
    feature model for spheres.

    positions:
        [B,N,D]

    velocities:
        [B,N,D]

    radii:
        [N] or [B,N]

    Returns:
        pairwise relative position [B,N,N,D]
        pairwise relative velocity [B,N,N,D]
        clearance [B,N,N]
        contact proximity [B,N,N]

    This model computes features only.
    It does not resolve impulses.
    """

    def __init__(
        self,
        spatial_dim: int = 2,
        proximity_scale: float = 0.1,
    ) -> None:
        super().__init__()

        if spatial_dim not in (2, 3):
            raise ValueError(
                "spatial_dim must be 2 or 3"
            )

        if proximity_scale <= 0:
            raise ValueError(
                "proximity_scale must be positive"
            )

        self.spatial_dim = spatial_dim
        self.proximity_scale = proximity_scale

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        radii: Tensor,
    ) -> dict[str, Tensor]:
        if (
            positions.ndim != 3
            or positions.shape[-1]
            != self.spatial_dim
        ):
            raise InteractionValidationError(
                "Positions must be [B,N,D]"
            )

        if velocities.shape != positions.shape:
            raise InteractionValidationError(
                "Velocity shape mismatch"
            )

        batch_size, object_count, _ = (
            positions.shape
        )

        if radii.ndim == 1:
            if radii.shape != (
                object_count,
            ):
                raise InteractionValidationError(
                    "Invalid radii shape"
                )

            radii = radii.unsqueeze(
                0
            ).expand(
                batch_size,
                -1,
            )

        elif radii.shape != (
            batch_size,
            object_count,
        ):
            raise InteractionValidationError(
                "Radii must be [N] or [B,N]"
            )

        if (radii <= 0).any():
            raise InteractionValidationError(
                "Radii must be positive"
            )

        relative_position = (
            positions[:, None, :, :]
            - positions[:, :, None, :]
        )

        relative_velocity = (
            velocities[:, None, :, :]
            - velocities[:, :, None, :]
        )

        distance = torch.linalg.vector_norm(
            relative_position,
            dim=-1,
        )

        combined_radius = (
            radii[:, :, None]
            + radii[:, None, :]
        )

        clearance = (
            distance - combined_radius
        )

        proximity = torch.sigmoid(
            -clearance
            / self.proximity_scale
        )

        # Exclude self-interactions.
        identity = torch.eye(
            object_count,
            dtype=torch.bool,
            device=positions.device,
        ).unsqueeze(0)

        proximity = proximity.masked_fill(
            identity,
            0.0,
        )

        return {
            "relative_position": (
                relative_position
            ),
            "relative_velocity": (
                relative_velocity
            ),
            "distance": distance,
            "clearance": clearance,
            "contact_proximity": proximity,
        }


# ============================================================
# WORLD-MODEL ADAPTER
# ============================================================

class InteractionDynamicsAdapter:
    """
    Exposes InteractionWorld as a
    predict_next_state interface.

    Flattened state layout per body:
        [position_D, velocity_D]

    Actions:
        External force per dynamic body,
        flattened in body_id order.

    IMPORTANT:
        Prediction uses a fresh cloned
        simulation world each call.

    This adapter is NumPy-based and
    non-differentiable.
    """

    def __init__(
        self,
        template_world: InteractionWorld,
        body_ids: list[str],
    ) -> None:
        self.template_world = template_world
        self.body_ids = body_ids

        self.spatial_dim = (
            template_world.config.spatial_dim
        )

        for body_id in body_ids:
            template_world.get_body(
                body_id
            )

        self.state_dim = (
            len(body_ids)
            * 2
            * self.spatial_dim
        )

        self.action_dim = (
            len(body_ids)
            * self.spatial_dim
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
            raise InteractionValidationError(
                "Invalid state dimensions"
            )

        if actions.shape != (
            states.shape[0],
            self.action_dim,
        ):
            raise InteractionValidationError(
                "Invalid action dimensions"
            )

        outputs = []

        for batch_index in range(
            states.shape[0]
        ):
            state_np = (
                states[batch_index]
                .detach()
                .cpu()
                .numpy()
            )

            action_np = (
                actions[batch_index]
                .detach()
                .cpu()
                .numpy()
            )

            world = InteractionWorld(
                self.template_world.config
            )

            for index, body_id in enumerate(
                self.body_ids
            ):
                template_body = (
                    self.template_world.get_body(
                        body_id
                    )
                )

                body = template_body.copy()

                start = (
                    index
                    * 2
                    * self.spatial_dim
                )

                position_end = (
                    start
                    + self.spatial_dim
                )

                velocity_end = (
                    position_end
                    + self.spatial_dim
                )

                body.position = state_np[
                    start:position_end
                ].copy()

                body.velocity = state_np[
                    position_end:velocity_end
                ].copy()

                world.add_body(
                    body
                )

                action_start = (
                    index
                    * self.spatial_dim
                )

                action_end = (
                    action_start
                    + self.spatial_dim
                )

                body.apply_force(
                    action_np[
                        action_start:action_end
                    ]
                )

            world.step()

            next_state = []

            for body_id in self.body_ids:
                body = world.get_body(
                    body_id
                )

                next_state.extend(
                    body.position.tolist()
                )

                next_state.extend(
                    body.velocity.tolist()
                )

            outputs.append(
                next_state
            )

        return torch.tensor(
            outputs,
            dtype=states.dtype,
            device=states.device,
        )


# ============================================================
# MAIN INTERACTION MODEL
# ============================================================

class InteractionModel:
    """
    High-level interaction physics API.
    """

    def __init__(
        self,
        config: InteractionConfig | None = None,
    ) -> None:
        self.config = (
            config or InteractionConfig()
        )

        self.world = InteractionWorld(
            self.config
        )

    def add_object(
        self,
        body: RigidBody,
    ) -> None:
        self.world.add_body(
            body
        )

    def apply_force(
        self,
        body_id: str,
        force: np.ndarray | list[float],
    ) -> None:
        self.world.apply_force(
            body_id,
            force,
        )

    def apply_impulse(
        self,
        body_id: str,
        impulse: np.ndarray | list[float],
    ) -> None:
        self.world.apply_impulse(
            body_id,
            impulse,
        )

    def step(
        self,
    ) -> InteractionStepResult:
        return self.world.step()

    def simulate(
        self,
        steps: int,
    ) -> list[InteractionStepResult]:
        return self.world.simulate(
            steps
        )


# ============================================================
# FACTORY
# ============================================================

def create_interaction_model(
    config: InteractionConfig | None = None,
) -> InteractionModel:
    return InteractionModel(
        config
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("=" * 65)
    print(
        "NEUROWORLD AI - INTERACTION MODEL"
    )
    print("=" * 65)

    config = InteractionConfig(
        spatial_dim=2,
        timestep=0.02,
        enable_gravity=False,
        solver_iterations=8,
    )

    model = InteractionModel(
        config
    )

    robot = RigidBody(
        body_id="robot",
        position=[0.0, 0.0],
        velocity=[1.5, 0.0],
        radius=0.25,
        mass=5.0,
        material=PhysicalMaterial(
            restitution=0.1,
            friction=0.6,
        ),
    )

    box_object = RigidBody(
        body_id="movable_object",
        position=[1.2, 0.0],
        velocity=[0.0, 0.0],
        radius=0.3,
        mass=2.0,
        material=PhysicalMaterial(
            restitution=0.2,
            friction=0.5,
        ),
    )

    obstacle = RigidBody(
        body_id="static_obstacle",
        position=[3.0, 0.0],
        velocity=[0.0, 0.0],
        radius=0.4,
        body_type=BodyType.STATIC,
    )

    model.add_object(
        robot
    )

    model.add_object(
        box_object
    )

    model.add_object(
        obstacle
    )

    print("\nInitial state")

    for body in model.world.bodies.values():
        print(
            body.body_id,
            body.position,
            body.velocity,
        )

    print("\nRunning simulation")

    for step in range(100):
        if step < 30:
            model.apply_force(
                "robot",
                [5.0, 0.0],
            )

        result = model.step()

        if result.events:
            print(
                f"Step {step}:",
                [
                    event.to_dict()
                    for event in result.events
                ],
            )

    print("\nFinal state")

    for body in model.world.bodies.values():
        print(
            body.body_id,
            "position:",
            body.position,
            "velocity:",
            body.velocity,
        )

    print(
        "\nTotal events:",
        len(model.world.event_history),
    )

    print("\nPyTorch interaction features")

    torch_model = TorchInteractionModel(
        spatial_dim=2
    )

    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [2.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    velocities = torch.zeros_like(
        positions
    )

    radii = torch.tensor(
        [0.3, 0.3, 0.4],
        dtype=torch.float32,
    )

    features = torch_model(
        positions,
        velocities,
        radii,
    )

    print(
        "Clearance shape:",
        features["clearance"].shape,
    )

    print(
        "Contact proximity:",
        features["contact_proximity"],
    )

    print(
        "\nInteraction model demonstration completed"
    )


if __name__ == "__main__":
    main()
