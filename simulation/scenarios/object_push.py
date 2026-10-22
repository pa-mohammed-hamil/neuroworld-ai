
```python
"""
NeuroWorld AI
File: neuroworld/physics/object_push.py

2D Object Pushing and Contact Simulation

Features:
- Robot-to-object push interactions
- Contact detection
- Push force and direction
- Object mass and friction
- Static and kinetic friction
- Acceleration and velocity prediction
- Maximum force and speed constraints
- Obstacle collision prevention
- World boundary checking
- Multi-step push simulation
- Predicted object trajectories
- Push success and failure reporting
- Integration with ObjectInteractionEngine

Assumptions:
- Circular objects in a 2D environment
- Pure translational movement
- Fixed ground contact
- Simplified Coulomb friction
- No object rotation or tipping
- No deformable-body physics
- No real hardware control

Dependencies:
    Python 3.10+
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point2D = Tuple[float, float]

EPSILON = 1e-9


class PushStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NO_CONTACT = "no_contact"
    INSUFFICIENT_FORCE = "insufficient_force"
    FORCE_LIMIT = "force_limit"
    COLLISION = "collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID_INPUT = "invalid_input"


@dataclass
class ObjectPushConfig:
    timestep: float = 0.02
    max_steps: int = 500

    gravity: float = 9.81

    static_friction: float = 0.4
    kinetic_friction: float = 0.3

    max_push_force: float = 100.0
    max_object_speed: float = 2.0

    contact_tolerance: float = 0.05
    safety_margin: float = 0.02

    world_width: float = 10.0
    world_height: float = 10.0

    stop_on_collision: bool = True

    velocity_epsilon: float = 1e-5
    position_tolerance: float = 1e-3


@dataclass
class PushBody:
    body_id: str

    x: float
    y: float

    radius: float
    mass: float

    vx: float = 0.0
    vy: float = 0.0

    movable: bool = True

    static_friction: Optional[float] = None
    kinetic_friction: Optional[float] = None

    @property
    def position(self) -> Point2D:
        return self.x, self.y

    @property
    def velocity(self) -> Point2D:
        return self.vx, self.vy

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class Pusher:
    pusher_id: str

    x: float
    y: float

    radius: float = 0.2
    max_force: float = 100.0

    @property
    def position(self) -> Point2D:
        return self.x, self.y


@dataclass
class PushCommand:
    direction_x: float
    direction_y: float

    force: float
    duration: float

    stop_at_target: Optional[Point2D] = None


@dataclass
class PushTrajectoryPoint:
    time: float

    x: float
    y: float

    vx: float
    vy: float

    ax: float
    ay: float

    applied_force: float


@dataclass
class PushResult:
    success: bool
    status: PushStatus

    object_id: str
    pusher_id: str

    initial_position: Point2D
    final_position: Point2D

    displacement: Point2D
    distance_traveled: float

    final_velocity: Point2D

    steps: int
    elapsed_time: float

    trajectory: List[
        PushTrajectoryPoint
    ] = field(default_factory=list)

    message: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class ObjectPushEngine:
    """
    Simplified 2D contact-pushing simulator.

    Push commands apply force along a
    normalized direction.

    The object moves under:
        F_net = F_push - F_friction

    Acceleration:
        a = F_net / mass

    Motion:
        v_next = v + a * dt
        x_next = x + v_next * dt

    This is a simplified model suitable
    for prototyping, not high-fidelity
    manipulation physics.
    """

    def __init__(
        self,
        config: Optional[
            ObjectPushConfig
        ] = None,
    ):
        self.config = (
            config or ObjectPushConfig()
        )

        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config

        positive = (
            cfg.timestep,
            cfg.gravity,
            cfg.max_push_force,
            cfg.max_object_speed,
            cfg.world_width,
            cfg.world_height,
            cfg.velocity_epsilon,
            cfg.position_tolerance,
        )

        if any(
            not math.isfinite(value)
            or value <= 0
            for value in positive
        ):
            raise ValueError(
                "Positive configuration values "
                "must be finite and greater than zero"
            )

        nonnegative = (
            cfg.static_friction,
            cfg.kinetic_friction,
            cfg.contact_tolerance,
            cfg.safety_margin,
        )

        if any(
            not math.isfinite(value)
            or value < 0
            for value in nonnegative
        ):
            raise ValueError(
                "Friction, tolerance and margin "
                "must be finite and nonnegative"
            )

        if cfg.max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

    @staticmethod
    def _distance(
        a: Point2D,
        b: Point2D,
    ) -> float:
        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    @staticmethod
    def _normalize(
        x: float,
        y: float,
    ) -> Point2D:
        length = math.hypot(x, y)

        if (
            not math.isfinite(length)
            or length <= EPSILON
        ):
            raise ValueError(
                "Push direction must be nonzero "
                "and finite"
            )

        return x / length, y / length

    def _validate_body(
        self,
        body: PushBody,
    ) -> None:
        values = (
            body.x,
            body.y,
            body.radius,
            body.mass,
            body.vx,
            body.vy,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Body properties must be finite"
            )

        if body.radius <= 0:
            raise ValueError(
                "Object radius must be positive"
            )

        if body.mass <= 0:
            raise ValueError(
                "Object mass must be positive"
            )

        for friction in (
            body.static_friction,
            body.kinetic_friction,
        ):
            if (
                friction is not None
                and (
                    not math.isfinite(friction)
                    or friction < 0
                )
            ):
                raise ValueError(
                    "Invalid object friction"
                )

    def _validate_command(
        self,
        command: PushCommand,
    ) -> None:
        if (
            not math.isfinite(command.force)
            or command.force < 0
        ):
            raise ValueError(
                "Push force must be finite "
                "and nonnegative"
            )

        if (
            not math.isfinite(command.duration)
            or command.duration <= 0
        ):
            raise ValueError(
                "Push duration must be positive "
                "and finite"
            )

        self._normalize(
            command.direction_x,
            command.direction_y,
        )

        if command.stop_at_target is not None:
            if not all(
                math.isfinite(value)
                for value
                in command.stop_at_target
            ):
                raise ValueError(
                    "Target must contain "
                    "finite coordinates"
                )

    def _inside_world(
        self,
        position: Point2D,
        radius: float,
    ) -> bool:
        margin = self.config.safety_margin

        return (
            radius + margin
            <= position[0]
            <= self.config.world_width
            - radius - margin
            and radius + margin
            <= position[1]
            <= self.config.world_height
            - radius - margin
        )

    def _contact_valid(
        self,
        pusher: Pusher,
        body: PushBody,
        direction: Point2D,
    ) -> bool:
        """
        Verify that the pusher is touching
        the object from behind the intended
        push direction.

        This simplified check does not
        model the pusher's contact surface.
        """

        center_distance = self._distance(
            pusher.position,
            body.position,
        )

        expected = (
            pusher.radius + body.radius
        )

        if abs(
            center_distance - expected
        ) > self.config.contact_tolerance:
            return False

        if center_distance <= EPSILON:
            return False

        contact_normal = (
            (body.x - pusher.x)
            / center_distance,
            (body.y - pusher.y)
            / center_distance,
        )

        alignment = (
            contact_normal[0] * direction[0]
            + contact_normal[1] * direction[1]
        )

        return alignment >= 0.7

    def _path_collision(
        self,
        body: PushBody,
        start: Point2D,
        end: Point2D,
        obstacles: Sequence[PushBody],
    ) -> bool:
        """
        Swept-circle collision check.

        Prevents the object from passing
        through static circular obstacles.
        """

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length_squared = (
            dx * dx + dy * dy
        )

        for obstacle in obstacles:
            if obstacle.body_id == body.body_id:
                continue

            if length_squared <= EPSILON:
                closest = start
            else:
                t = (
                    (
                        (obstacle.x - start[0]) * dx
                        + (obstacle.y - start[1]) * dy
                    ) / length_squared
                )

                t = max(
                    0.0,
                    min(1.0, t),
                )

                closest = (
                    start[0] + t * dx,
                    start[1] + t * dy,
                )

            clearance = self._distance(
                closest,
                obstacle.position,
            )

            minimum_distance = (
                body.radius
                + obstacle.radius
                + self.config.safety_margin
            )

            if clearance < minimum_distance:
                return True

        return False

    def _friction_coefficients(
        self,
        body: PushBody,
    ) -> Tuple[float, float]:
        static = (
            body.static_friction
            if body.static_friction is not None
            else self.config.static_friction
        )

        kinetic = (
            body.kinetic_friction
            if body.kinetic_friction is not None
            else self.config.kinetic_friction
        )

        return static, kinetic

    def _calculate_acceleration(
        self,
        body: PushBody,
        applied_force: Point2D,
    ) -> Point2D:
        """
        Compute translational acceleration
        under a simplified friction model.

        Static friction prevents movement
        until the applied force exceeds
        the static friction threshold.

        Kinetic friction opposes the
        direction of motion.
        """

        mu_static, mu_kinetic = (
            self._friction_coefficients(body)
        )

        normal_force = (
            body.mass * self.config.gravity
        )

        static_limit = (
            mu_static * normal_force
        )

        kinetic_force = (
            mu_kinetic * normal_force
        )

        force_x, force_y = applied_force

        force_magnitude = math.hypot(
            force_x,
            force_y,
        )

        speed = body.speed

        if (
            speed <= self.config.velocity_epsilon
            and force_magnitude <= static_limit
        ):
            return 0.0, 0.0

        if speed > self.config.velocity_epsilon:
            friction_x = (
                -kinetic_force
                * body.vx / speed
            )

            friction_y = (
                -kinetic_force
                * body.vy / speed
            )
        elif force_magnitude > EPSILON:
            friction_x = (
                -kinetic_force
                * force_x / force_magnitude
            )

            friction_y = (
                -kinetic_force
                * force_y / force_magnitude
            )
        else:
            friction_x = 0.0
            friction_y = 0.0

        net_x = force_x + friction_x
        net_y = force_y + friction_y

        return (
            net_x / body.mass,
            net_y / body.mass,
        )

    def _result(
        self,
        status: PushStatus,
        body: PushBody,
        pusher: Pusher,
        initial_position: Point2D,
        distance_traveled: float,
        elapsed_time: float,
        trajectory: List[
            PushTrajectoryPoint
        ],
        message: str,
    ) -> PushResult:
        displacement = (
            body.x - initial_position[0],
            body.y - initial_position[1],
        )

        return PushResult(
            success=status == PushStatus.SUCCESS,
            status=status,
            object_id=body.body_id,
            pusher_id=pusher.pusher_id,
            initial_position=initial_position,
            final_position=body.position,
            displacement=displacement,
            distance_traveled=distance_traveled,
            final_velocity=body.velocity,
            steps=max(0, len(trajectory) - 1),
            elapsed_time=elapsed_time,
            trajectory=trajectory,
            message=message,
        )

    def push(
        self,
        body: PushBody,
        pusher: Pusher,
        command: PushCommand,
        obstacles: Sequence[PushBody] = (),
    ) -> PushResult:
        """
        Simulate an object being pushed.

        Mutates body position and velocity.

        The pusher's contact pose is treated
        as a fixed initial condition.
        Continuous pusher-object contact
        is not dynamically simulated.
        """

        self._validate_body(body)
        self._validate_command(command)

        if (
            pusher.radius <= 0
            or pusher.max_force <= 0
            or not all(
                math.isfinite(value)
                for value in (
                    pusher.x,
                    pusher.y,
                    pusher.radius,
                    pusher.max_force,
                )
            )
        ):
            raise ValueError(
                "Invalid pusher properties"
            )

        for obstacle in obstacles:
            self._validate_body(obstacle)

        initial_position = body.position

        trajectory = [
            PushTrajectoryPoint(
                time=0.0,
                x=body.x,
                y=body.y,
                vx=body.vx,
                vy=body.vy,
                ax=0.0,
                ay=0.0,
                applied_force=0.0,
            )
        ]

        if not body.movable:
            return self._result(
                PushStatus.INVALID_INPUT,
                body,
                pusher,
                initial_position,
                0.0,
                0.0,
                trajectory,
                "Object is not movable",
            )

        direction = self._normalize(
            command.direction_x,
            command.direction_y,
        )

        if not self._inside_world(
            body.position,
            body.radius,
        ):
            return self._result(
                PushStatus.OUT_OF_BOUNDS,
                body,
                pusher,
                initial_position,
                0.0,
                0.0,
                trajectory,
                "Initial object position "
                "is outside the world",
            )

        if not self._contact_valid(
            pusher,
            body,
            direction,
        ):
            return self._result(
                PushStatus.NO_CONTACT,
                body,
                pusher,
                initial_position,
                0.0,
                0.0,
                trajectory,
                "Pusher is not in valid contact",
            )

        allowed_force = min(
            self.config.max_push_force,
            pusher.max_force,
        )

        if command.force > allowed_force:
            return self._result(
                PushStatus.FORCE_LIMIT,
                body,
                pusher,
                initial_position,
                0.0,
                0.0,
                trajectory,
                "Requested force exceeds "
                "the configured limit",
            )

        mu_static, _ = (
            self._friction_coefficients(body)
        )

        static_threshold = (
            mu_static
            * body.mass
            * self.config.gravity
        )

        if (
            body.speed
            <= self.config.velocity_epsilon
            and command.force <= static_threshold
        ):
            return self._result(
                PushStatus.INSUFFICIENT_FORCE,
                body,
                pusher,
                initial_position,
                0.0,
                0.0,
                trajectory,
                "Force is insufficient "
                "to overcome static friction",
            )

        distance_traveled = 0.0
        elapsed = 0.0

        target_reached = False

        for _ in range(
            self.config.max_steps
        ):
            remaining = (
                command.duration - elapsed
            )

            if remaining <= EPSILON:
                break

            dt = min(
                self.config.timestep,
                remaining,
            )

            applied_force = (
                command.force * direction[0],
                command.force * direction[1],
            )

            ax, ay = (
                self._calculate_acceleration(
                    body,
                    applied_force,
                )
            )

            next_vx = (
                body.vx + ax * dt
            )

            next_vy = (
                body.vy + ay * dt
            )

            # Prevent numerical friction
            # from reversing the velocity
            # when the applied force is zero.
            if command.force <= EPSILON:
                if (
                    body.vx * next_vx
                    + body.vy * next_vy
                ) < 0:
                    next_vx = 0.0
                    next_vy = 0.0

            speed = math.hypot(
                next_vx,
                next_vy,
            )

            if speed > self.config.max_object_speed:
                scale = (
                    self.config.max_object_speed
                    / speed
                )

                next_vx *= scale
                next_vy *= scale

            next_position = (
                body.x + next_vx * dt,
                body.y + next_vy * dt,
            )

            if not self._inside_world(
                next_position,
                body.radius,
            ):
                body.vx = 0.0
                body.vy = 0.0

                return self._result(
                    PushStatus.OUT_OF_BOUNDS,
                    body,
                    pusher,
                    initial_position,
                    distance_traveled,
                    elapsed,
                    trajectory,
                    "Push stopped at world boundary",
                )

            if self._path_collision(
                body,
                body.position,
                next_position,
                obstacles,
            ):
                body.vx = 0.0
                body.vy = 0.0

                return self._result(
                    PushStatus.COLLISION,
                    body,
                    pusher,
                    initial_position,
                    distance_traveled,
                    elapsed,
                    trajectory,
                    "Push stopped before obstacle",
                )

            movement = self._distance(
                body.position,
                next_position,
            )

            body.x, body.y = next_position
            body.vx = next_vx
            body.vy = next_vy

            elapsed += dt
            distance_traveled += movement

            trajectory.append(
                PushTrajectoryPoint(
                    time=elapsed,
                    x=body.x,
                    y=body.y,
                    vx=body.vx,
                    vy=body.vy,
                    ax=ax,
                    ay=ay,
                    applied_force=command.force,
                )
            )

            if command.stop_at_target is not None:
                target_distance = self._distance(
                    body.position,
                    command.stop_at_target,
                )

                if (
                    target_distance
                    <= self.config.position_tolerance
                ):
                    target_reached = True
                    body.vx = 0.0
                    body.vy = 0.0
                    break

        if command.stop_at_target is not None:
            if target_reached:
                status = PushStatus.SUCCESS
                message = "Target reached"
            else:
                status = PushStatus.PARTIAL
                message = (
                    "Push ended before reaching target"
                )
        elif elapsed >= command.duration - EPSILON:
            status = PushStatus.SUCCESS
            message = "Push duration completed"
        else:
            status = PushStatus.PARTIAL
            message = (
                "Push ended at maximum simulation steps"
            )

        return self._result(
            status,
            body,
            pusher,
            initial_position,
            distance_traveled,
            elapsed,
            trajectory,
            message,
        )

    def predict_push(
        self,
        body: PushBody,
        pusher: Pusher,
        command: PushCommand,
        obstacles: Sequence[PushBody] = (),
    ) -> PushResult:
        """
        Predict push outcome without
        modifying the original object.
        """

        predicted_body = PushBody(
            body_id=body.body_id,
            x=body.x,
            y=body.y,
            radius=body.radius,
            mass=body.mass,
            vx=body.vx,
            vy=body.vy,
            movable=body.movable,
            static_friction=body.static_friction,
            kinetic_friction=body.kinetic_friction,
        )

        return self.push(
            predicted_body,
            pusher,
            command,
            obstacles,
        )

    def estimate_required_force(
        self,
        body: PushBody,
        desired_acceleration: float = 0.0,
    ) -> float:
        """
        Estimate minimum push force for
        initiating motion and acceleration.
        """

        self._validate_body(body)

        if (
            not math.isfinite(desired_acceleration)
            or desired_acceleration < 0
        ):
            raise ValueError(
                "Desired acceleration must "
                "be finite and nonnegative"
            )

        static, kinetic = (
            self._friction_coefficients(body)
        )

        normal_force = (
            body.mass * self.config.gravity
        )

        breakaway_force = (
            static * normal_force
        )

        moving_force = (
            body.mass * desired_acceleration
            + kinetic * normal_force
        )

        return max(
            breakaway_force,
            moving_force,
        )

    def calculate_push_energy(
        self,
        result: PushResult,
        applied_force: float,
    ) -> float:
        """
        Approximate positive mechanical work.

        Assumes the object moves in the
        applied force direction.
        """

        if applied_force < 0:
            raise ValueError(
                "Force cannot be negative"
            )

        return (
            applied_force
            * result.distance_traveled
        )

    def summarize(
        self,
        result: PushResult,
    ) -> Dict[str, Any]:
        return {
            "status": result.status.value,
            "success": result.success,
            "object_id": result.object_id,
            "pusher_id": result.pusher_id,
            "initial_position": (
                result.initial_position
            ),
            "final_position": (
                result.final_position
            ),
            "displacement": result.displacement,
            "distance_traveled": (
                result.distance_traveled
            ),
            "final_velocity": (
                result.final_velocity
            ),
            "steps": result.steps,
            "elapsed_time": (
                result.elapsed_time
            ),
            "trajectory_points": len(
                result.trajectory
            ),
            "message": result.message,
        }


if __name__ == "__main__":

    engine = ObjectPushEngine(
        ObjectPushConfig(
            timestep=0.02,
            max_steps=500,
            world_width=10.0,
            world_height=10.0,
            max_push_force=100.0,
        )
    )

    box = PushBody(
        body_id="warehouse_box",
        x=2.0,
        y=2.0,
        radius=0.25,
        mass=5.0,
    )

    robot = Pusher(
        pusher_id="robot_1",
        x=1.55,
        y=2.0,
        radius=0.2,
        max_force=80.0,
    )

    obstacles = [
        PushBody(
            body_id="wall_obstacle",
            x=5.0,
            y=2.0,
            radius=0.4,
            mass=1000.0,
            movable=False,
        )
    ]

    command = PushCommand(
        direction_x=1.0,
        direction_y=0.0,
        force=35.0,
        duration=2.0,
    )

    print(
        "Estimated required force:",
        engine.estimate_required_force(
            box,
            desired_acceleration=0.5,
        ),
    )

    # Predict without modifying the box.
    prediction = engine.predict_push(
        body=box,
        pusher=robot,
        command=command,
        obstacles=obstacles,
    )

    print(
        "Predicted result:",
        engine.summarize(prediction),
    )

    # Execute in the simulated environment.
    result = engine.push(
        body=box,
        pusher=robot,
        command=command,
        obstacles=obstacles,
    )

    print(
        "Actual simulation result:",
        engine.summarize(result),
    )

    print(
        "Final object position:",
        box.position,
    )
```
