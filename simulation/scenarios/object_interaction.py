
```python
"""
NeuroWorld AI
File: neuroworld/environment/object_interaction.py

Object Interaction and Manipulation Simulation

Features:
- Object registration and management
- Robot/end-effector registration
- Object proximity and reachability
- Grasp and release simulation
- Pick-and-place operations
- Push and pull interactions
- Carrying attached objects
- Object collision detection
- Workspace boundary validation
- Object ownership and locking
- Interaction state transitions
- Event history
- JSON save/load
- Interaction statistics

Physics assumptions:
- 2D circular object footprints
- Quasi-static object movement
- No force/torque simulation
- No articulated-hand dynamics
- No contact/friction solver

Dependencies:
    Python 3.10+
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Point2D = Tuple[float, float]


class ObjectState(str, Enum):
    FREE = "free"
    GRASPED = "grasped"
    CARRIED = "carried"
    PLACED = "placed"
    LOCKED = "locked"


class InteractionType(str, Enum):
    INSPECT = "inspect"
    GRASP = "grasp"
    RELEASE = "release"
    PICK = "pick"
    PLACE = "place"
    PUSH = "push"
    PULL = "pull"
    MOVE = "move"


class InteractionStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    NOT_FOUND = "not_found"
    OUT_OF_REACH = "out_of_reach"
    COLLISION = "collision"
    INVALID_STATE = "invalid_state"
    CAPACITY_EXCEEDED = "capacity_exceeded"


@dataclass
class ObjectInteractionConfig:
    world_width: float = 10.0
    world_height: float = 10.0

    default_reach: float = 1.0
    default_grip_capacity: float = 5.0

    safety_margin: float = 0.05

    max_push_distance: float = 1.0
    max_pull_distance: float = 1.0

    enable_collision_detection: bool = True
    enforce_workspace_bounds: bool = True

    history_capacity: int = 10000


@dataclass
class SimulatedObject:
    object_id: str
    x: float
    y: float

    radius: float = 0.15
    mass: float = 1.0

    movable: bool = True
    graspable: bool = True

    state: ObjectState = ObjectState.FREE

    held_by: Optional[str] = None

    label: str = ""
    category: str = "generic"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def position(self) -> Point2D:
        return self.x, self.y


@dataclass
class Manipulator:
    manipulator_id: str
    x: float
    y: float

    reach: float = 1.0
    grip_capacity: float = 5.0

    held_object_id: Optional[str] = None

    enabled: bool = True

    @property
    def position(self) -> Point2D:
        return self.x, self.y


@dataclass
class InteractionResult:
    success: bool
    status: InteractionStatus

    interaction_type: InteractionType

    manipulator_id: str
    object_id: str

    previous_position: Optional[Point2D] = None
    new_position: Optional[Point2D] = None

    message: str = ""

    timestamp: float = field(
        default_factory=time.time
    )


@dataclass
class InteractionEvent:
    timestamp: float
    interaction_type: str

    manipulator_id: str
    object_id: str

    success: bool
    status: str
    message: str


@dataclass
class InteractionStatistics:
    total_interactions: int = 0
    successful_interactions: int = 0
    rejected_interactions: int = 0

    grasps: int = 0
    releases: int = 0
    pushes: int = 0
    pulls: int = 0
    placements: int = 0


class ObjectInteractionEngine:
    """
    High-level simulated object interaction.

    This engine manages discrete object
    manipulation operations in a 2D world.

    It does not command real actuators.
    """

    def __init__(
        self,
        config: Optional[
            ObjectInteractionConfig
        ] = None,
    ):
        self.config = (
            config or ObjectInteractionConfig()
        )

        self._validate_config()

        self.objects: Dict[
            str, SimulatedObject
        ] = {}

        self.manipulators: Dict[
            str, Manipulator
        ] = {}

        self.events: List[
            InteractionEvent
        ] = []

        self.statistics = (
            InteractionStatistics()
        )

        self._lock = threading.RLock()

    def _validate_config(self) -> None:
        cfg = self.config

        positive = (
            cfg.world_width,
            cfg.world_height,
            cfg.default_reach,
            cfg.default_grip_capacity,
            cfg.max_push_distance,
            cfg.max_pull_distance,
        )

        if any(
            not math.isfinite(value) or value <= 0
            for value in positive
        ):
            raise ValueError(
                "World dimensions, reach, "
                "capacity and movement limits "
                "must be positive and finite"
            )

        if (
            not math.isfinite(cfg.safety_margin)
            or cfg.safety_margin < 0
        ):
            raise ValueError(
                "Invalid safety margin"
            )

        if cfg.history_capacity <= 0:
            raise ValueError(
                "history_capacity must be positive"
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
    def _validate_point(
        point: Point2D,
    ) -> None:
        if (
            len(point) != 2
            or not all(
                math.isfinite(value)
                for value in point
            )
        ):
            raise ValueError(
                "Position must contain "
                "two finite coordinates"
            )

    def _inside_world(
        self,
        x: float,
        y: float,
        radius: float,
    ) -> bool:
        if not self.config.enforce_workspace_bounds:
            return True

        return (
            radius <= x
            <= self.config.world_width - radius
            and radius <= y
            <= self.config.world_height - radius
        )

    def _collision_free(
        self,
        object_id: str,
        position: Point2D,
        radius: float,
    ) -> bool:
        if not self.config.enable_collision_detection:
            return True

        for other in self.objects.values():
            if other.object_id == object_id:
                continue

            distance = self._distance(
                position,
                other.position,
            )

            required_clearance = (
                radius
                + other.radius
                + self.config.safety_margin
            )

            if distance < required_clearance:
                return False

        return True

    def _valid_position(
        self,
        object_id: str,
        position: Point2D,
        radius: float,
    ) -> bool:
        self._validate_point(position)

        return (
            self._inside_world(
                position[0],
                position[1],
                radius,
            )
            and self._collision_free(
                object_id,
                position,
                radius,
            )
        )

    def _path_clear(
        self,
        object_id: str,
        start: Point2D,
        end: Point2D,
        radius: float,
    ) -> bool:
        """
        Check a straight swept-circle path.

        Used for simplified manipulation
        motion validation.
        """

        if not self._inside_world(
            start[0], start[1], radius
        ):
            return False

        if not self._inside_world(
            end[0], end[1], radius
        ):
            return False

        if not self.config.enable_collision_detection:
            return True

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length_squared = dx * dx + dy * dy

        for other in self.objects.values():
            if other.object_id == object_id:
                continue

            if length_squared <= 1e-12:
                closest = start
            else:
                projection = (
                    (
                        (other.x - start[0]) * dx
                        + (other.y - start[1]) * dy
                    ) / length_squared
                )

                projection = max(
                    0.0,
                    min(1.0, projection),
                )

                closest = (
                    start[0] + projection * dx,
                    start[1] + projection * dy,
                )

            separation = self._distance(
                closest,
                other.position,
            )

            if separation < (
                radius
                + other.radius
                + self.config.safety_margin
            ):
                return False

        return True

    def register_object(
        self,
        obj: SimulatedObject,
    ) -> None:
        with self._lock:
            if obj.object_id in self.objects:
                raise ValueError(
                    f"Duplicate object: {obj.object_id}"
                )

            if (
                obj.radius <= 0
                or obj.mass < 0
                or not math.isfinite(obj.radius)
                or not math.isfinite(obj.mass)
            ):
                raise ValueError(
                    "Invalid object geometry or mass"
                )

            if (
                obj.held_by is not None
                or obj.state in (
                    ObjectState.GRASPED,
                    ObjectState.CARRIED,
                )
            ):
                raise ValueError(
                    "Register objects unheld; "
                    "use grasp() to attach them"
                )

            if not self._valid_position(
                obj.object_id,
                obj.position,
                obj.radius,
            ):
                raise ValueError(
                    "Object placement invalid"
                )

            self.objects[obj.object_id] = (
                copy.deepcopy(obj)
            )

    def register_manipulator(
        self,
        manipulator: Manipulator,
    ) -> None:
        with self._lock:
            if (
                manipulator.manipulator_id
                in self.manipulators
            ):
                raise ValueError(
                    "Duplicate manipulator ID"
                )

            if (
                manipulator.reach <= 0
                or manipulator.grip_capacity <= 0
            ):
                raise ValueError(
                    "Invalid reach or grip capacity"
                )

            self._validate_point(
                manipulator.position
            )

            if (
                manipulator.held_object_id
                is not None
            ):
                raise ValueError(
                    "New manipulators must start empty"
                )

            self.manipulators[
                manipulator.manipulator_id
            ] = copy.deepcopy(manipulator)

    def _get_entities(
        self,
        manipulator_id: str,
        object_id: str,
    ) -> Tuple[
        Optional[Manipulator],
        Optional[SimulatedObject],
    ]:
        return (
            self.manipulators.get(manipulator_id),
            self.objects.get(object_id),
        )

    def _within_reach(
        self,
        manipulator: Manipulator,
        position: Point2D,
    ) -> bool:
        return (
            self._distance(
                manipulator.position,
                position,
            )
            <= manipulator.reach
        )

    def _record(
        self,
        result: InteractionResult,
    ) -> InteractionResult:
        self.statistics.total_interactions += 1

        if result.success:
            self.statistics.successful_interactions += 1
        else:
            self.statistics.rejected_interactions += 1

        if result.success:
            counters = {
                InteractionType.GRASP: "grasps",
                InteractionType.RELEASE: "releases",
                InteractionType.PUSH: "pushes",
                InteractionType.PULL: "pulls",
                InteractionType.PLACE: "placements",
            }

            counter_name = counters.get(
                result.interaction_type
            )

            if counter_name is not None:
                setattr(
                    self.statistics,
                    counter_name,
                    getattr(
                        self.statistics,
                        counter_name,
                    ) + 1,
                )

        self.events.append(
            InteractionEvent(
                timestamp=result.timestamp,
                interaction_type=(
                    result.interaction_type.value
                ),
                manipulator_id=(
                    result.manipulator_id
                ),
                object_id=result.object_id,
                success=result.success,
                status=result.status.value,
                message=result.message,
            )
        )

        if (
            len(self.events)
            > self.config.history_capacity
        ):
            self.events.pop(0)

        return result

    def _result(
        self,
        success: bool,
        status: InteractionStatus,
        operation: InteractionType,
        manipulator_id: str,
        object_id: str,
        message: str,
        previous: Optional[Point2D] = None,
        current: Optional[Point2D] = None,
    ) -> InteractionResult:
        return self._record(
            InteractionResult(
                success=success,
                status=status,
                interaction_type=operation,
                manipulator_id=manipulator_id,
                object_id=object_id,
                previous_position=previous,
                new_position=current,
                message=message,
            )
        )

    def inspect(
        self,
        object_id: str,
        manipulator_id: str = "observer",
    ) -> Dict[str, Any]:
        with self._lock:
            obj = self.objects.get(object_id)

            if obj is None:
                raise KeyError(
                    f"Unknown object: {object_id}"
                )

            return asdict(obj)

    def grasp(
        self,
        manipulator_id: str,
        object_id: str,
    ) -> InteractionResult:
        with self._lock:
            operation = InteractionType.GRASP

            manipulator, obj = self._get_entities(
                manipulator_id,
                object_id,
            )

            if manipulator is None or obj is None:
                return self._result(
                    False,
                    InteractionStatus.NOT_FOUND,
                    operation,
                    manipulator_id,
                    object_id,
                    "Manipulator or object not found",
                )

            if not manipulator.enabled:
                return self._result(
                    False,
                    InteractionStatus.REJECTED,
                    operation,
                    manipulator_id,
                    object_id,
                    "Manipulator disabled",
                )

            if (
                manipulator.held_object_id
                is not None
                or obj.held_by is not None
                or obj.state == ObjectState.LOCKED
                or not obj.graspable
                or not obj.movable
            ):
                return self._result(
                    False,
                    InteractionStatus.INVALID_STATE,
                    operation,
                    manipulator_id,
                    object_id,
                    "Object unavailable for grasping",
                )

            if obj.mass > manipulator.grip_capacity:
                return self._result(
                    False,
                    InteractionStatus.CAPACITY_EXCEEDED,
                    operation,
                    manipulator_id,
                    object_id,
                    "Object exceeds grip capacity",
                )

            if not self._within_reach(
                manipulator,
                obj.position,
            ):
                return self._result(
                    False,
                    InteractionStatus.OUT_OF_REACH,
                    operation,
                    manipulator_id,
                    object_id,
                    "Object outside manipulator reach",
                )

            obj.held_by = manipulator_id
            obj.state = ObjectState.GRASPED

            manipulator.held_object_id = object_id

            return self._result(
                True,
                InteractionStatus.SUCCESS,
                operation,
                manipulator_id,
                object_id,
                "Object grasped",
                previous=obj.position,
                current=obj.position,
            )

    def release(
        self,
        manipulator_id: str,
    ) -> InteractionResult:
        with self._lock:
            manipulator = self.manipulators.get(
                manipulator_id
            )

            object_id = (
                manipulator.held_object_id
                if manipulator is not None
                else ""
            )

            obj = self.objects.get(object_id)

            if manipulator is None or obj is None:
                return self._result(
                    False,
                    InteractionStatus.NOT_FOUND,
                    InteractionType.RELEASE,
                    manipulator_id,
                    object_id,
                    "No held object found",
                )

            if not self._valid_position(
                object_id,
                obj.position,
                obj.radius,
            ):
                return self._result(
                    False,
                    InteractionStatus.COLLISION,
                    InteractionType.RELEASE,
                    manipulator_id,
                    object_id,
                    "Current position is unsafe",
                )

            obj.held_by = None
            obj.state = ObjectState.FREE

            manipulator.held_object_id = None

            return self._result(
                True,
                InteractionStatus.SUCCESS,
                InteractionType.RELEASE,
                manipulator_id,
                object_id,
                "Object released",
                previous=obj.position,
                current=obj.position,
            )

    def pick(
        self,
        manipulator_id: str,
        object_id: str,
    ) -> InteractionResult:
        """
        Alias for grasp, reported as PICK.
        """

        with self._lock:
            result = self.grasp(
                manipulator_id,
                object_id,
            )

            # Preserve a single event by updating
            # the grasp event's operation.
            result.interaction_type = (
                InteractionType.PICK
            )

            if self.events:
                self.events[-1].interaction_type = (
                    InteractionType.PICK.value
                )

            if result.success:
                self.statistics.grasps -= 1

            return result

    def place(
        self,
        manipulator_id: str,
        position: Point2D,
    ) -> InteractionResult:
        with self._lock:
            self._validate_point(position)

            manipulator = self.manipulators.get(
                manipulator_id
            )

            object_id = (
                manipulator.held_object_id
                if manipulator is not None
                else ""
            )

            obj = self.objects.get(object_id)

            if manipulator is None or obj is None:
                return self._result(
                    False,
                    InteractionStatus.NOT_FOUND,
                    InteractionType.PLACE,
                    manipulator_id,
                    object_id,
                    "No held object to place",
                )

            if not manipulator.enabled:
                return self._result(
                    False,
                    InteractionStatus.REJECTED,
                    InteractionType.PLACE,
                    manipulator_id,
                    object_id,
                    "Manipulator disabled",
                )

            if not self._within_reach(
                manipulator,
                position,
            ):
                return self._result(
                    False,
                    InteractionStatus.OUT_OF_REACH,
                    InteractionType.PLACE,
                    manipulator_id,
                    object_id,
                    "Placement target out of reach",
                )

            if not self._path_clear(
                object_id,
                obj.position,
                position,
                obj.radius,
            ):
                return self._result(
                    False,
                    InteractionStatus.COLLISION,
                    InteractionType.PLACE,
                    manipulator_id,
                    object_id,
                    "Placement path obstructed",
                )

            previous = obj.position

            obj.x, obj.y = position
            obj.state = ObjectState.PLACED
            obj.held_by = None

            manipulator.held_object_id = None

            return self._result(
                True,
                InteractionStatus.SUCCESS,
                InteractionType.PLACE,
                manipulator_id,
                object_id,
                "Object placed successfully",
                previous=previous,
                current=position,
            )

    def _translate(
        self,
        manipulator_id: str,
        object_id: str,
        displacement: Point2D,
        operation: InteractionType,
    ) -> InteractionResult:
        self._validate_point(displacement)

        manipulator, obj = self._get_entities(
            manipulator_id,
            object_id,
        )

        if manipulator is None or obj is None:
            return self._result(
                False,
                InteractionStatus.NOT_FOUND,
                operation,
                manipulator_id,
                object_id,
                "Manipulator or object not found",
            )

        if (
            not manipulator.enabled
            or not obj.movable
            or obj.held_by is not None
            or obj.state == ObjectState.LOCKED
        ):
            return self._result(
                False,
                InteractionStatus.INVALID_STATE,
                operation,
                manipulator_id,
                object_id,
                "Object cannot be moved",
            )

        if not self._within_reach(
            manipulator,
            obj.position,
        ):
            return self._result(
                False,
                InteractionStatus.OUT_OF_REACH,
                operation,
                manipulator_id,
                object_id,
                "Object is outside reach",
            )

        dx, dy = displacement
        movement_distance = math.hypot(dx, dy)

        maximum = (
            self.config.max_push_distance
            if operation == InteractionType.PUSH
            else self.config.max_pull_distance
        )

        if movement_distance > maximum:
            return self._result(
                False,
                InteractionStatus.REJECTED,
                operation,
                manipulator_id,
                object_id,
                "Requested movement exceeds limit",
            )

        target = (
            obj.x + dx,
            obj.y + dy,
        )

        if not self._within_reach(
            manipulator,
            target,
        ):
            return self._result(
                False,
                InteractionStatus.OUT_OF_REACH,
                operation,
                manipulator_id,
                object_id,
                "Target outside manipulator reach",
            )

        if not self._path_clear(
            object_id,
            obj.position,
            target,
            obj.radius,
        ):
            return self._result(
                False,
                InteractionStatus.COLLISION,
                operation,
                manipulator_id,
                object_id,
                "Movement path obstructed",
            )

        previous = obj.position

        obj.x, obj.y = target
        obj.state = ObjectState.FREE

        return self._result(
            True,
            InteractionStatus.SUCCESS,
            operation,
            manipulator_id,
            object_id,
            "Object moved successfully",
            previous=previous,
            current=target,
        )

    def push(
        self,
        manipulator_id: str,
        object_id: str,
        displacement: Point2D,
    ) -> InteractionResult:
        with self._lock:
            return self._translate(
                manipulator_id,
                object_id,
                displacement,
                InteractionType.PUSH,
            )

    def pull(
        self,
        manipulator_id: str,
        object_id: str,
        displacement: Point2D,
    ) -> InteractionResult:
        with self._lock:
            return self._translate(
                manipulator_id,
                object_id,
                displacement,
                InteractionType.PULL,
            )

    def move_manipulator(
        self,
        manipulator_id: str,
        position: Point2D,
    ) -> bool:
        """
        Move the manipulator position.

        If an object is held, translate
        the object by the same displacement.

        Returns False if the move is unsafe.
        """

        with self._lock:
            self._validate_point(position)

            manipulator = self.manipulators.get(
                manipulator_id
            )

            if (
                manipulator is None
                or not manipulator.enabled
            ):
                return False

            if not self._inside_world(
                position[0],
                position[1],
                0.0,
            ):
                return False

            dx = position[0] - manipulator.x
            dy = position[1] - manipulator.y

            held_id = manipulator.held_object_id

            if held_id is not None:
                obj = self.objects[held_id]

                target = (
                    obj.x + dx,
                    obj.y + dy,
                )

                if not self._path_clear(
                    held_id,
                    obj.position,
                    target,
                    obj.radius,
                ):
                    return False

                obj.x, obj.y = target
                obj.state = ObjectState.CARRIED

            manipulator.x, manipulator.y = position

            return True

    def lock_object(
        self,
        object_id: str,
    ) -> None:
        with self._lock:
            obj = self.objects[object_id]

            if obj.held_by is not None:
                raise RuntimeError(
                    "Cannot lock a held object"
                )

            obj.state = ObjectState.LOCKED

    def unlock_object(
        self,
        object_id: str,
    ) -> None:
        with self._lock:
            obj = self.objects[object_id]

            if obj.state == ObjectState.LOCKED:
                obj.state = ObjectState.FREE

    def get_object(
        self,
        object_id: str,
    ) -> SimulatedObject:
        with self._lock:
            return copy.deepcopy(
                self.objects[object_id]
            )

    def get_manipulator(
        self,
        manipulator_id: str,
    ) -> Manipulator:
        with self._lock:
            return copy.deepcopy(
                self.manipulators[manipulator_id]
            )

    def reset(self) -> None:
        with self._lock:
            self.objects.clear()
            self.manipulators.clear()
            self.events.clear()

            self.statistics = (
                InteractionStatistics()
            )

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_objects": len(self.objects),
                "total_manipulators": len(
                    self.manipulators
                ),
                "held_objects": sum(
                    obj.held_by is not None
                    for obj in self.objects.values()
                ),
                "locked_objects": sum(
                    obj.state == ObjectState.LOCKED
                    for obj in self.objects.values()
                ),
                "statistics": asdict(
                    self.statistics
                ),
            }

    def save(
        self,
        path: str | Path,
    ) -> None:
        with self._lock:
            path = Path(path)

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            payload = {
                "config": asdict(self.config),
                "objects": [
                    asdict(obj)
                    for obj in self.objects.values()
                ],
                "manipulators": [
                    asdict(manipulator)
                    for manipulator
                    in self.manipulators.values()
                ],
                "events": [
                    asdict(event)
                    for event in self.events
                ],
                "statistics": asdict(
                    self.statistics
                ),
            }

            path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ObjectInteractionEngine":
        payload = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        engine = cls(
            ObjectInteractionConfig(
                **payload["config"]
            )
        )

        for item in payload.get(
            "objects", []
        ):
            item["state"] = ObjectState(
                item["state"]
            )

            obj = SimulatedObject(**item)

            if obj.object_id in engine.objects:
                raise ValueError(
                    "Duplicate object in saved data"
                )

            engine.objects[obj.object_id] = obj

        for item in payload.get(
            "manipulators", []
        ):
            manipulator = Manipulator(**item)

            if (
                manipulator.manipulator_id
                in engine.manipulators
            ):
                raise ValueError(
                    "Duplicate manipulator in saved data"
                )

            engine.manipulators[
                manipulator.manipulator_id
            ] = manipulator

        engine.events = [
            InteractionEvent(**item)
            for item in payload.get(
                "events", []
            )
        ]

        engine.statistics = (
            InteractionStatistics(
                **payload.get(
                    "statistics", {}
                )
            )
        )

        # Validate ownership consistency.
        for obj in engine.objects.values():
            if obj.held_by is None:
                if obj.state in (
                    ObjectState.GRASPED,
                    ObjectState.CARRIED,
                ):
                    raise ValueError(
                        "Unowned object has held state"
                    )
                continue

            manipulator = engine.manipulators.get(
                obj.held_by
            )

            if (
                manipulator is None
                or manipulator.held_object_id
                != obj.object_id
            ):
                raise ValueError(
                    "Inconsistent object ownership"
                )

        for manipulator in engine.manipulators.values():
            held_id = manipulator.held_object_id

            if held_id is None:
                continue

            obj = engine.objects.get(held_id)

            if (
                obj is None
                or obj.held_by
                != manipulator.manipulator_id
            ):
                raise ValueError(
                    "Inconsistent manipulator ownership"
                )

        return engine


if __name__ == "__main__":

    engine = ObjectInteractionEngine(
        ObjectInteractionConfig(
            world_width=10.0,
            world_height=10.0,
            default_reach=1.5,
            default_grip_capacity=5.0,
        )
    )

    engine.register_manipulator(
        Manipulator(
            manipulator_id="robot_arm_1",
            x=2.0,
            y=2.0,
            reach=1.5,
            grip_capacity=5.0,
        )
    )

    engine.register_object(
        SimulatedObject(
            object_id="box_001",
            x=2.5,
            y=2.0,
            radius=0.2,
            mass=2.0,
            label="Warehouse Box",
            category="package",
        )
    )

    engine.register_object(
        SimulatedObject(
            object_id="box_002",
            x=5.0,
            y=5.0,
            radius=0.3,
            mass=3.0,
            label="Storage Box",
        )
    )

    print(
        "Initial state:",
        engine.summary(),
    )

    grasp_result = engine.grasp(
        "robot_arm_1",
        "box_001",
    )

    print(
        "Grasp:",
        grasp_result.status.value,
    )

    moved = engine.move_manipulator(
        "robot_arm_1",
        (2.5, 2.5),
    )

    print(
        "Manipulator moved:",
        moved,
    )

    place_result = engine.place(
        "robot_arm_1",
        (3.0, 2.5),
    )

    print(
        "Place:",
        place_result.status.value,
    )

    push_result = engine.push(
        "robot_arm_1",
        "box_001",
        (0.3, 0.0),
    )

    print(
        "Push:",
        push_result.status.value,
    )

    print(
        "Final object:",
        engine.get_object("box_001"),
    )

    engine.save(
        "artifacts/object_interactions.json"
    )

    restored = ObjectInteractionEngine.load(
        "artifacts/object_interactions.json"
    )

    print(
        "Restored summary:",
        restored.summary(),
    )
```
