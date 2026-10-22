
"""
NeuroWorld AI
File: training/losses/physics.py

Differentiable physics losses for world-model,
robotics, and physical dynamics training.

Features:
- Position / velocity / acceleration consistency
- Newton's second law (F = ma)
- Momentum conservation
- Kinetic and potential energy conservation
- Gravity-aware motion
- Collision penetration penalty
- Nonnegative contact-force penalty
- Friction-cone constraint
- Joint position and velocity limits
- Quaternion normalization
- Smooth trajectory regularization
- Variable time-step support
- Sequence masking
- Weighted multi-objective losses
- Custom physics loss registration
- Component metrics

Units:
    Position: meters
    Velocity: meters/second
    Acceleration: meters/second²
    Force: newtons
    Mass: kilograms
    Energy: joules
    Time: seconds

Requires:
    torch >= 2.1

Important:
    These are differentiable physics constraints,
    not a complete rigid-body simulator.
    Collision distances, forces, mass, inertia,
    and coordinate frames must be supplied
    consistently by the calling model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PhysicsLossFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Tensor | None,
]


@dataclass
class PhysicsLossConfig:
    position_weight: float = 1.0
    velocity_weight: float = 0.5
    acceleration_weight: float = 0.25
    newton_weight: float = 0.5
    momentum_weight: float = 0.1
    energy_weight: float = 0.1
    gravity_weight: float = 0.1
    collision_weight: float = 1.0
    contact_weight: float = 0.25
    friction_weight: float = 0.25
    joint_limit_weight: float = 0.5
    quaternion_weight: float = 0.1
    smoothness_weight: float = 0.05

    loss_type: str = "mse"
    huber_delta: float = 1.0

    gravity: float = 9.81
    default_dt: float = 0.02

    collision_margin: float = 0.0
    friction_coefficient: float = 0.5

    normalize_weights: bool = False
    detect_nonfinite: bool = True

    def __post_init__(self) -> None:
        weights = (
            self.position_weight,
            self.velocity_weight,
            self.acceleration_weight,
            self.newton_weight,
            self.momentum_weight,
            self.energy_weight,
            self.gravity_weight,
            self.collision_weight,
            self.contact_weight,
            self.friction_weight,
            self.joint_limit_weight,
            self.quaternion_weight,
            self.smoothness_weight,
        )

        if any(
            not math.isfinite(w) or w < 0
            for w in weights
        ):
            raise ValueError(
                "Physics weights must be finite "
                "and nonnegative"
            )

        if self.loss_type not in {
            "mse", "mae", "huber"
        }:
            raise ValueError(
                "Invalid physics loss type"
            )

        if self.huber_delta <= 0:
            raise ValueError(
                "huber_delta must be positive"
            )

        if self.gravity < 0:
            raise ValueError(
                "gravity must be nonnegative"
            )

        if self.default_dt <= 0:
            raise ValueError(
                "default_dt must be positive"
            )

        if self.collision_margin < 0:
            raise ValueError(
                "collision_margin must be nonnegative"
            )

        if self.friction_coefficient < 0:
            raise ValueError(
                "friction_coefficient must be nonnegative"
            )


class PhysicsLoss(nn.Module):
    """
    Physics-informed multi-objective loss.

    Output keys:
        predicted_positions
        predicted_velocities
        predicted_accelerations
        predicted_forces
        predicted_momentum
        predicted_kinetic_energy
        predicted_potential_energy
        predicted_total_energy
        predicted_gravity_acceleration
        signed_distances
        normal_forces
        tangential_forces
        predicted_joint_positions
        predicted_joint_velocities
        predicted_quaternions

    Target keys:
        positions
        velocities
        accelerations
        forces
        masses
        momentum
        energy
        gravity_acceleration
        collision_margin
        friction_coefficient
        joint_lower_limits
        joint_upper_limits
        joint_velocity_limits
        dt
        mask

    Sequence conventions:
        [B,T,D] for positions/velocities
        [B,T] or [B,T,1] for mass/time

    Components activate only when the
    required inputs are available.
    """

    def __init__(
        self,
        config: PhysicsLossConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or PhysicsLossConfig()

        self.weights = {
            "position": self.config.position_weight,
            "velocity": self.config.velocity_weight,
            "acceleration": (
                self.config.acceleration_weight
            ),
            "newton": self.config.newton_weight,
            "momentum": self.config.momentum_weight,
            "energy": self.config.energy_weight,
            "gravity": self.config.gravity_weight,
            "collision": self.config.collision_weight,
            "contact": self.config.contact_weight,
            "friction": self.config.friction_weight,
            "joint_limit": (
                self.config.joint_limit_weight
            ),
            "quaternion": (
                self.config.quaternion_weight
            ),
            "smoothness": (
                self.config.smoothness_weight
            ),
        }

        self.custom_losses: dict[
            str, PhysicsLossFunction
        ] = {}

        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _get(
        values: Mapping[str, Any],
        *names: str,
    ) -> Tensor | None:
        for name in names:
            value = values.get(name)

            if isinstance(value, Tensor):
                return value

        return None

    @staticmethod
    def _broadcast(
        value: Tensor,
        reference: Tensor,
    ) -> Tensor:
        value = value.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        # [B,T] -> [B,T,1] for vector fields.
        if (
            reference.ndim >= 3
            and value.ndim == reference.ndim - 1
            and value.shape == reference.shape[:-1]
        ):
            value = value.unsqueeze(-1)

        # [B] -> [B,1,1] for temporal vector fields.
        if (
            reference.ndim >= 3
            and value.ndim == 1
            and value.shape[0] == reference.shape[0]
        ):
            value = value.reshape(
                reference.shape[0],
                *([1] * (reference.ndim - 1)),
            )

        try:
            return torch.broadcast_to(
                value,
                reference.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Tensor cannot broadcast to "
                f"{tuple(reference.shape)}"
            ) from exc

    @staticmethod
    def _mask(
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor | None:
        mask = PhysicsLoss._get(
            targets,
            "mask",
            "sequence_mask",
        )

        if mask is None:
            return None

        mask = mask.to(
            device=reference.device,
            dtype=torch.bool,
        )

        if mask.shape[0] != reference.shape[0]:
            raise ValueError(
                "Mask batch dimension mismatch"
            )

        if (
            reference.ndim >= 3
            and mask.ndim >= 2
            and mask.shape[1] != reference.shape[1]
        ):
            if mask.shape[1] > reference.shape[1]:
                mask = mask[
                    :,
                    -reference.shape[1]:,
                ]
            else:
                raise ValueError(
                    "Mask sequence length mismatch"
                )

        while mask.ndim < reference.ndim:
            mask = mask.unsqueeze(-1)

        try:
            return torch.broadcast_to(
                mask,
                reference.shape,
            )

        except RuntimeError as exc:
            raise ValueError(
                "Invalid physics mask shape"
            ) from exc

    @staticmethod
    def _masked_mean(
        values: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        if mask is None:
            return values.mean()

        weights = mask.to(
            device=values.device,
            dtype=values.dtype,
        )

        return (
            (values * weights).sum()
            / weights.sum().clamp_min(1.0)
        )

    def _elementwise(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Physics prediction and target "
                "shapes must match"
            )

        prediction = prediction.float()
        target = target.float()

        if self.config.loss_type == "mse":
            return (
                prediction - target
            ).square()

        if self.config.loss_type == "mae":
            return (
                prediction - target
            ).abs()

        return F.huber_loss(
            prediction,
            target,
            delta=self.config.huber_delta,
            reduction="none",
        )

    def _regression(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self._masked_mean(
            self._elementwise(
                prediction,
                target,
            ),
            mask,
        )

    def _dt(
        self,
        targets: Mapping[str, Any],
        reference: Tensor,
    ) -> Tensor:
        """
        Return a time-step tensor broadcastable
        to a temporal vector.

        dt may be:
            scalar
            [B]
            [B,T-1]
            [B,T-1,1]
        """
        value = targets.get("dt")

        if value is None:
            return torch.full(
                (
                    reference.shape[0],
                    reference.shape[1],
                    1,
                ),
                self.config.default_dt,
                device=reference.device,
                dtype=reference.dtype,
            )

        if not isinstance(value, Tensor):
            value = torch.as_tensor(
                value,
                device=reference.device,
                dtype=reference.dtype,
            )

        value = value.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if value.ndim == 0:
            value = value.reshape(1, 1, 1)

        elif value.ndim == 1:
            if value.shape[0] == reference.shape[0]:
                value = value[:, None, None]

            elif value.shape[0] == reference.shape[1]:
                value = value[None, :, None]

            else:
                raise ValueError(
                    "Invalid dt vector length"
                )

        elif value.ndim == 2:
            value = value.unsqueeze(-1)

        try:
            value = torch.broadcast_to(
                value,
                (
                    reference.shape[0],
                    reference.shape[1],
                    1,
                ),
            )

        except RuntimeError as exc:
            raise ValueError(
                "dt cannot broadcast to [B,T,1]"
            ) from exc

        if not torch.all(value > 0).item():
            raise ValueError(
                "Time steps must be positive"
            )

        return value

    def position_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_positions",
        )

        target = self._get(
            targets,
            "positions",
        )

        if prediction is None or target is None:
            return None

        return self._regression(
            prediction,
            target,
            self._mask(targets, prediction),
        )

    def velocity_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_velocities",
        )

        target = self._get(
            targets,
            "velocities",
        )

        if prediction is not None and target is not None:
            return self._regression(
                prediction,
                target,
                self._mask(targets, prediction),
            )

        # Alternative: velocity consistency
        # from predicted positions.
        positions = self._get(
            outputs,
            "predicted_positions",
        )

        if positions is None or target is None:
            return None

        if positions.ndim != 3:
            return None

        if positions.shape[1] < 2:
            return None

        if target.shape == positions.shape:
            target = target[:, 1:]

        dt = self._dt(
            targets,
            positions[:, 1:],
        )

        derived_velocity = (
            positions[:, 1:]
            - positions[:, :-1]
        ) / dt

        return self._regression(
            derived_velocity,
            target,
            self._mask(
                targets,
                derived_velocity,
            ),
        )

    def acceleration_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        prediction = self._get(
            outputs,
            "predicted_accelerations",
        )

        target = self._get(
            targets,
            "accelerations",
        )

        if prediction is not None and target is not None:
            return self._regression(
                prediction,
                target,
                self._mask(targets, prediction),
            )

        velocities = self._get(
            outputs,
            "predicted_velocities",
        )

        if velocities is None or target is None:
            return None

        if velocities.ndim != 3:
            return None

        if velocities.shape[1] < 2:
            return None

        if target.shape == velocities.shape:
            target = target[:, 1:]

        dt = self._dt(
            targets,
            velocities[:, 1:],
        )

        derived_acceleration = (
            velocities[:, 1:]
            - velocities[:, :-1]
        ) / dt

        return self._regression(
            derived_acceleration,
            target,
            self._mask(
                targets,
                derived_acceleration,
            ),
        )

    def newton_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Newton's second law:
            F_net = m * a

        Forces must represent the total
        net force in the same coordinate
        frame as acceleration.
        """
        acceleration = self._get(
            outputs,
            "predicted_accelerations",
        )

        forces = self._get(
            outputs,
            "predicted_forces",
            "net_forces",
        )

        if forces is None:
            forces = self._get(
                targets,
                "forces",
            )

        masses = self._get(
            targets,
            "masses",
        )

        if (
            acceleration is None
            or forces is None
            or masses is None
        ):
            return None

        acceleration = acceleration.float()

        masses = self._broadcast(
            masses,
            acceleration,
        )

        forces = self._broadcast(
            forces,
            acceleration,
        )

        if not torch.all(masses > 0).item():
            raise ValueError(
                "Masses must be positive"
            )

        residual = (
            forces - masses * acceleration
        )

        return self._regression(
            residual,
            torch.zeros_like(residual),
            self._mask(targets, residual),
        )

    def momentum_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Momentum conservation across time.

        This loss assumes the system is
        isolated with negligible external
        impulse. Disable it otherwise.
        """
        momentum = self._get(
            outputs,
            "predicted_momentum",
        )

        if momentum is None:
            velocity = self._get(
                outputs,
                "predicted_velocities",
            )

            masses = self._get(
                targets,
                "masses",
            )

            if velocity is None or masses is None:
                return None

            momentum = (
                velocity.float()
                * self._broadcast(
                    masses,
                    velocity,
                )
            )

        if (
            momentum.ndim < 3
            or momentum.shape[1] < 2
        ):
            return None

        difference = (
            momentum[:, 1:]
            - momentum[:, :-1]
        )

        return self._regression(
            difference,
            torch.zeros_like(difference),
            self._mask(targets, difference),
        )

    def energy_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Mechanical energy conservation.

        Assumes a conservative system.
        Disable for actuated or dissipative
        dynamics unless energy accounting
        includes external work and losses.
        """
        energy = self._get(
            outputs,
            "predicted_total_energy",
        )

        if energy is None:
            kinetic = self._get(
                outputs,
                "predicted_kinetic_energy",
            )

            potential = self._get(
                outputs,
                "predicted_potential_energy",
            )

            if kinetic is None or potential is None:
                return None

            energy = kinetic + potential

        target_energy = self._get(
            targets,
            "energy",
            "total_energy",
        )

        if target_energy is not None:
            return self._regression(
                energy,
                target_energy,
                self._mask(targets, energy),
            )

        if (
            energy.ndim < 2
            or energy.shape[1] < 2
        ):
            return None

        difference = (
            energy[:, 1:]
            - energy[:, :-1]
        )

        return self._regression(
            difference,
            torch.zeros_like(difference),
            self._mask(targets, difference),
        )

    def gravity_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Gravity acceleration consistency.

        A gravity vector is expected to
        use the same coordinate frame
        as the model's acceleration.

        By default, -Z is downward.
        """
        predicted = self._get(
            outputs,
            "predicted_gravity_acceleration",
        )

        if predicted is None:
            return None

        target = self._get(
            targets,
            "gravity_acceleration",
        )

        if target is None:
            if predicted.shape[-1] != 3:
                raise ValueError(
                    "Default gravity requires "
                    "3D acceleration vectors"
                )

            target = torch.zeros_like(
                predicted
            )

            target[..., 2] = (
                -self.config.gravity
            )

        else:
            target = self._broadcast(
                target,
                predicted,
            )

        return self._regression(
            predicted,
            target,
            self._mask(targets, predicted),
        )

    def collision_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Signed-distance collision penalty.

        Positive distance: separated
        Zero distance: contact
        Negative distance: penetration
        """
        distances = self._get(
            outputs,
            "signed_distances",
            "collision_signed_distances",
        )

        if distances is None:
            return None

        margin = targets.get(
            "collision_margin",
            self.config.collision_margin,
        )

        if isinstance(margin, Tensor):
            margin = self._broadcast(
                margin,
                distances,
            )

        penetration = F.relu(
            margin - distances.float()
        )

        return self._masked_mean(
            penetration.square(),
            self._mask(targets, distances),
        )

    def contact_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Penalize negative normal forces.

        Contact complementarity can be
        supplied as an additional custom loss.
        """
        normal_forces = self._get(
            outputs,
            "normal_forces",
        )

        if normal_forces is None:
            return None

        violation = F.relu(
            -normal_forces.float()
        )

        return self._masked_mean(
            violation.square(),
            self._mask(targets, normal_forces),
        )

    def friction_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Coulomb friction cone constraint:

            ||F_t|| <= mu * F_n

        Tangential forces:
            [B,T,2] or [B,T,3]

        Normal force:
            [B,T] or [B,T,1]
        """
        tangential = self._get(
            outputs,
            "tangential_forces",
        )

        normal = self._get(
            outputs,
            "normal_forces",
        )

        if tangential is None or normal is None:
            return None

        tangential_magnitude = (
            tangential.float()
            .norm(dim=-1)
        )

        if (
            normal.ndim
            == tangential_magnitude.ndim + 1
            and normal.shape[-1] == 1
        ):
            normal = normal.squeeze(-1)

        normal = self._broadcast(
            normal,
            tangential_magnitude,
        )

        coefficient = targets.get(
            "friction_coefficient",
            self.config.friction_coefficient,
        )

        if isinstance(coefficient, Tensor):
            coefficient = self._broadcast(
                coefficient,
                tangential_magnitude,
            )

        violation = F.relu(
            tangential_magnitude
            - coefficient * normal.clamp_min(0)
        )

        return self._masked_mean(
            violation.square(),
            self._mask(
                targets,
                violation,
            ),
        )

    def joint_limit_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        positions = self._get(
            outputs,
            "predicted_joint_positions",
        )

        if positions is None:
            return None

        lower = self._get(
            targets,
            "joint_lower_limits",
        )

        upper = self._get(
            targets,
            "joint_upper_limits",
        )

        if lower is None or upper is None:
            return None

        positions = positions.float()

        lower = self._broadcast(
            lower,
            positions,
        )

        upper = self._broadcast(
            upper,
            positions,
        )

        if not torch.all(lower <= upper).item():
            raise ValueError(
                "Joint lower limits exceed upper limits"
            )

        lower_violation = F.relu(
            lower - positions
        )

        upper_violation = F.relu(
            positions - upper
        )

        penalty = (
            lower_violation.square()
            + upper_violation.square()
        )

        velocities = self._get(
            outputs,
            "predicted_joint_velocities",
        )

        velocity_limits = self._get(
            targets,
            "joint_velocity_limits",
        )

        if (
            velocities is not None
            and velocity_limits is not None
        ):
            if velocities.shape != positions.shape:
                raise ValueError(
                    "Joint velocity shape mismatch"
                )

            velocity_limits = self._broadcast(
                velocity_limits,
                velocities,
            )

            velocity_violation = F.relu(
                velocities.abs()
                - velocity_limits
            )

            penalty = (
                penalty
                + velocity_violation.square()
            )

        return self._masked_mean(
            penalty,
            self._mask(targets, positions),
        )

    def quaternion_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Enforce unit quaternion norm.

        Quaternion layout:
            [..., 4]
        """
        quaternions = self._get(
            outputs,
            "predicted_quaternions",
        )

        if quaternions is None:
            return None

        if quaternions.shape[-1] != 4:
            raise ValueError(
                "Quaternions require last dimension 4"
            )

        norm = quaternions.float().norm(
            dim=-1
        )

        penalty = (
            norm - 1.0
        ).square()

        return self._masked_mean(
            penalty,
            self._mask(targets, penalty),
        )

    def smoothness_loss(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> Tensor | None:
        """
        Penalize excessive acceleration
        changes (jerk) in trajectories.
        """
        acceleration = self._get(
            outputs,
            "predicted_accelerations",
        )

        if acceleration is None:
            return None

        if (
            acceleration.ndim < 3
            or acceleration.shape[1] < 2
        ):
            return None

        difference = (
            acceleration[:, 1:]
            - acceleration[:, :-1]
        )

        dt = self._dt(
            targets,
            difference,
        )

        jerk = difference / dt

        return self._masked_mean(
            jerk.float().square(),
            self._mask(targets, jerk),
        )

    def register_loss(
        self,
        name: str,
        function: PhysicsLossFunction,
        *,
        weight: float = 1.0,
    ) -> None:
        if not name:
            raise ValueError(
                "Loss name cannot be empty"
            )

        if name in self.weights:
            raise ValueError(
                f"Loss '{name}' already exists"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Custom weight must be finite "
                "and nonnegative"
            )

        self.custom_losses[name] = function
        self.weights[name] = float(weight)

    def set_weight(
        self,
        name: str,
        weight: float,
    ) -> None:
        if name not in self.weights:
            raise KeyError(
                f"Unknown physics loss: {name}"
            )

        if (
            not math.isfinite(weight)
            or weight < 0
        ):
            raise ValueError(
                "Weight must be finite "
                "and nonnegative"
            )

        self.weights[name] = float(weight)

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        return_components: bool = True,
    ) -> (
        Tensor
        | tuple[Tensor, dict[str, Tensor]]
    ):
        functions = {
            "position": self.position_loss,
            "velocity": self.velocity_loss,
            "acceleration": self.acceleration_loss,
            "newton": self.newton_loss,
            "momentum": self.momentum_loss,
            "energy": self.energy_loss,
            "gravity": self.gravity_loss,
            "collision": self.collision_loss,
            "contact": self.contact_loss,
            "friction": self.friction_loss,
            "joint_limit": self.joint_limit_loss,
            "quaternion": self.quaternion_loss,
            "smoothness": self.smoothness_loss,
        }

        components: dict[str, Tensor] = {}

        for name, function in functions.items():
            if self.weights[name] <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is not None:
                components[name] = loss

        for name, function in self.custom_losses.items():
            if self.weights[name] <= 0:
                continue

            loss = function(
                outputs,
                targets,
            )

            if loss is not None:
                components[name] = loss

        if not components:
            raise ValueError(
                "No active physics loss components. "
                "Check model outputs and targets."
            )

        for name, loss in components.items():
            if not isinstance(loss, Tensor):
                raise TypeError(
                    f"{name} must return a Tensor"
                )

            if loss.ndim != 0:
                raise ValueError(
                    f"{name} loss must be scalar"
                )

            if (
                self.config.detect_nonfinite
                and not torch.isfinite(loss).item()
            ):
                raise FloatingPointError(
                    f"Non-finite physics loss: {name}"
                )

        normalizer = 1.0

        if self.config.normalize_weights:
            normalizer = sum(
                self.weights[name]
                for name in components
            )

        weighted = {
            name: (
                loss * self.weights[name]
                / normalizer
            )
            for name, loss in components.items()
        }

        total = torch.stack(
            list(weighted.values())
        ).sum()

        self.last_metrics = {
            "physics_total": float(
                total.detach().item()
            ),
            **{
                f"physics/{name}": float(
                    loss.detach().item()
                )
                for name, loss in components.items()
            },
            **{
                f"physics_weighted/{name}": float(
                    loss.detach().item()
                )
                for name, loss in weighted.items()
            },
        }

        if return_components:
            return total, {
                "total": total,
                **components,
            }

        return total

    def get_last_metrics(
        self,
    ) -> dict[str, float]:
        return dict(
            self.last_metrics
        )

    def reset_metrics(self) -> None:
        self.last_metrics.clear()


def create_physics_loss(
    *,
    position_weight: float = 1.0,
    velocity_weight: float = 0.5,
    acceleration_weight: float = 0.25,
    newton_weight: float = 0.5,
    momentum_weight: float = 0.1,
    energy_weight: float = 0.1,
    collision_weight: float = 1.0,
    joint_limit_weight: float = 0.5,
) -> PhysicsLoss:
    return PhysicsLoss(
        PhysicsLossConfig(
            position_weight=position_weight,
            velocity_weight=velocity_weight,
            acceleration_weight=acceleration_weight,
            newton_weight=newton_weight,
            momentum_weight=momentum_weight,
            energy_weight=energy_weight,
            collision_weight=collision_weight,
            joint_limit_weight=joint_limit_weight,
        )
    )


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 8
    sequence_length = 12
    state_dim = 3

    class DemoPhysicsModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(6, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )

            self.position_head = nn.Linear(
                128,
                state_dim,
            )

            self.velocity_head = nn.Linear(
                128,
                state_dim,
            )

            self.acceleration_head = nn.Linear(
                128,
                state_dim,
            )

        def forward(
            self,
            positions: Tensor,
            velocities: Tensor,
        ) -> dict[str, Tensor]:
            features = torch.cat(
                [positions, velocities],
                dim=-1,
            )

            hidden = self.network(
                features
            )

            return {
                "predicted_positions": (
                    self.position_head(hidden)
                ),
                "predicted_velocities": (
                    self.velocity_head(hidden)
                ),
                "predicted_accelerations": (
                    self.acceleration_head(hidden)
                ),
            }

    model = DemoPhysicsModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    criterion = PhysicsLoss(
        PhysicsLossConfig(
            position_weight=1.0,
            velocity_weight=0.5,
            acceleration_weight=0.25,
            newton_weight=0.5,
            momentum_weight=0.0,
            energy_weight=0.0,
            gravity_weight=0.0,
            collision_weight=0.0,
            contact_weight=0.0,
            friction_weight=0.0,
            joint_limit_weight=0.0,
            quaternion_weight=0.0,
            smoothness_weight=0.05,
        )
    )

    for step in range(5):
        positions = torch.randn(
            batch_size,
            sequence_length,
            state_dim,
        )

        velocities = torch.randn(
            batch_size,
            sequence_length,
            state_dim,
        )

        accelerations = torch.randn(
            batch_size,
            sequence_length,
            state_dim,
        )

        masses = torch.ones(
            batch_size,
            sequence_length,
            1,
        )

        forces = (
            masses * accelerations
        )

        outputs = model(
            positions,
            velocities,
        )

        targets = {
            "positions": positions,
            "velocities": velocities,
            "accelerations": accelerations,
            "masses": masses,
            "forces": forces,
            "dt": torch.tensor(0.02),
            "mask": torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
            ),
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss, components = criterion(
            outputs,
            targets,
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        print(
            f"Step {step + 1}: "
            f"{criterion.get_last_metrics()}"
        )
