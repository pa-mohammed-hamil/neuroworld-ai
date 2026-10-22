
"""
NeuroWorld AI
File: backend/app/database/repositories/trajectory_repository.py

Trajectory persistence.

Expected Trajectory model fields:
    id
    world_id
    experiment_id
    robot_id
    episode_id
    states
    actions
    rewards
    metadata_json
    total_reward
    steps
    created_at

states/actions/rewards are assumed to use
JSON-compatible database columns.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.database.models import Trajectory

from .base import BaseRepository


class TrajectoryRepository(
    BaseRepository[Trajectory]
):
    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(
            session,
            Trajectory,
        )

    @staticmethod
    def _validate_sequence(
        values: Sequence[Any],
        name: str,
    ) -> None:
        if not isinstance(
            values,
            (list, tuple),
        ):
            raise TypeError(
                f"{name} must be a list or tuple"
            )

        if not values:
            raise ValueError(
                f"{name} cannot be empty"
            )

    @staticmethod
    def _validate_numeric_vector(
        values: Sequence[Any],
        name: str,
    ) -> None:
        for value in values:
            if not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    f"{name} must be numeric"
                )

            if not math.isfinite(
                float(value)
            ):
                raise ValueError(
                    f"{name} contains non-finite values"
                )

    def create_trajectory(
        self,
        *,
        world_id: Any,
        robot_id: str,
        episode_id: str,
        states: list[list[float]],
        actions: list[list[float]],
        rewards: list[float],
        experiment_id: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Trajectory:
        self._validate_sequence(
            states,
            "states",
        )

        self._validate_sequence(
            actions,
            "actions",
        )

        self._validate_sequence(
            rewards,
            "rewards",
        )

        if not (
            len(states)
            == len(actions) + 1
            == len(rewards) + 1
        ):
            raise ValueError(
                "Expected len(states) = "
                "len(actions) + 1 = "
                "len(rewards) + 1"
            )

        state_dim = len(states[0])
        action_dim = len(actions[0])

        if state_dim < 1 or action_dim < 1:
            raise ValueError(
                "State/action dimensions must be positive"
            )

        for state in states:
            if len(state) != state_dim:
                raise ValueError(
                    "Inconsistent state dimensions"
                )

            self._validate_numeric_vector(
                state,
                "state",
            )

        for action in actions:
            if len(action) != action_dim:
                raise ValueError(
                    "Inconsistent action dimensions"
                )

            self._validate_numeric_vector(
                action,
                "action",
            )

        self._validate_numeric_vector(
            rewards,
            "rewards",
        )

        return self.create(
            world_id=world_id,
            experiment_id=experiment_id,
            robot_id=robot_id,
            episode_id=episode_id,
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=float(
                sum(rewards)
            ),
            steps=len(actions),
            metadata_json=metadata or {},
        )

    def list_by_world(
        self,
        world_id: Any,
        *,
        limit: int = 100,
    ) -> list[Trajectory]:
        return self.list(
            world_id=world_id,
            limit=limit,
            order_by="created_at",
        )

    def list_by_experiment(
        self,
        experiment_id: Any,
        *,
        limit: int = 100,
    ) -> list[Trajectory]:
        return self.list(
            experiment_id=experiment_id,
            limit=limit,
            order_by="created_at",
        )

    def list_by_robot(
        self,
        robot_id: str,
        *,
        limit: int = 100,
    ) -> list[Trajectory]:
        return self.list(
            robot_id=robot_id,
            limit=limit,
            order_by="created_at",
        )

    def get_episode(
        self,
        *,
        world_id: Any,
        robot_id: str,
        episode_id: str,
    ) -> Trajectory | None:
        return self.get_by(
            world_id=world_id,
            robot_id=robot_id,
            episode_id=episode_id,
        )

    def get_best_trajectories(
        self,
        world_id: Any,
        *,
        limit: int = 10,
    ) -> list[Trajectory]:
        if not 1 <= limit <= 1000:
            raise ValueError(
                "Invalid limit"
            )

        statement = (
            select(Trajectory)
            .where(
                Trajectory.world_id == world_id
            )
            .order_by(
                Trajectory.total_reward.desc()
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(
                statement
            ).all()
        )

    def get_world_statistics(
        self,
        world_id: Any,
    ) -> dict[str, Any]:
        statement = (
            select(
                func.count(
                    Trajectory.id
                ),
                func.avg(
                    Trajectory.total_reward
                ),
                func.max(
                    Trajectory.total_reward
                ),
                func.min(
                    Trajectory.total_reward
                ),
                func.avg(
                    Trajectory.steps
                ),
            )
            .where(
                Trajectory.world_id == world_id
            )
        )

        result = self.session.execute(
            statement
        ).one()

        return {
            "world_id": world_id,
            "trajectory_count": int(
                result[0] or 0
            ),
            "average_reward": float(
                result[1] or 0.0
            ),
            "maximum_reward": (
                float(result[2])
                if result[2] is not None
                else None
            ),
            "minimum_reward": (
                float(result[3])
                if result[3] is not None
                else None
            ),
            "average_steps": float(
                result[4] or 0.0
            ),
        }

    def delete_world_trajectories(
        self,
        world_id: Any,
    ) -> int:
        return self.delete_by(
            world_id=world_id,
        )
