
"""
NeuroWorld AI
File: backend/app/database/repositories/world_repository.py

World persistence and lifecycle management.

Expected World model fields:
    id
    name
    description
    scenario_type
    configuration
    status
    owner_id
    created_at
    updated_at
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import World

from .base import BaseRepository


class WorldRepository(BaseRepository[World]):
    VALID_STATUSES = {
        "created",
        "ready",
        "running",
        "paused",
        "completed",
        "failed",
        "archived",
    }

    ALLOWED_TRANSITIONS = {
        "created": {"ready", "archived"},
        "ready": {"running", "archived"},
        "running": {
            "paused",
            "completed",
            "failed",
        },
        "paused": {
            "running",
            "completed",
            "failed",
        },
        "completed": {"archived"},
        "failed": {"ready", "archived"},
        "archived": set(),
    }

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(
            session,
            World,
        )

    def create_world(
        self,
        *,
        name: str,
        owner_id: Any,
        scenario_type: str = "empty",
        description: str = "",
        configuration: dict[str, Any] | None = None,
    ) -> World:
        if not name.strip():
            raise ValueError(
                "World name is required"
            )

        return self.create(
            name=name.strip(),
            owner_id=owner_id,
            scenario_type=scenario_type,
            description=description,
            configuration=configuration or {},
            status="created",
        )

    def get_owner_world(
        self,
        world_id: Any,
        owner_id: Any,
    ) -> World | None:
        return self.get_by(
            id=world_id,
            owner_id=owner_id,
        )

    def list_by_owner(
        self,
        owner_id: Any,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[World]:
        return self.list(
            owner_id=owner_id,
            skip=skip,
            limit=limit,
            order_by="created_at",
        )

    def list_by_status(
        self,
        status: str,
        *,
        limit: int = 100,
    ) -> list[World]:
        if status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid world status"
            )

        return self.list(
            status=status,
            limit=limit,
        )

    def update_configuration(
        self,
        world_id: Any,
        configuration: dict[str, Any],
        *,
        owner_id: Any,
    ) -> World | None:
        world = self.get_owner_world(
            world_id,
            owner_id,
        )

        if world is None:
            return None

        if world.status == "running":
            raise ValueError(
                "Cannot modify a running world"
            )

        return self.update(
            world.id,
            configuration=configuration,
        )

    def change_status(
        self,
        world_id: Any,
        new_status: str,
        *,
        owner_id: Any,
    ) -> World | None:
        if new_status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid world status"
            )

        # Serialize transitions on databases
        # that support SELECT FOR UPDATE.
        with self.session.begin_nested():
            statement = (
                select(World)
                .where(
                    World.id == world_id,
                    World.owner_id == owner_id,
                )
                .with_for_update()
            )

            world = self.session.scalar(
                statement
            )

            if world is None:
                return None

            if world.status == new_status:
                return world

            allowed = self.ALLOWED_TRANSITIONS.get(
                world.status,
                set(),
            )

            if new_status not in allowed:
                raise ValueError(
                    f"Invalid transition: "
                    f"{world.status} -> {new_status}"
                )

            world.status = new_status

            self.session.flush()

        self._commit()
        self.session.refresh(world)

        return world

    def search_worlds(
        self,
        query: str,
        *,
        owner_id: Any,
        limit: int = 50,
    ) -> list[World]:
        if not 1 <= limit <= 1000:
            raise ValueError(
                "Invalid limit"
            )

        statement = (
            select(World)
            .where(
                World.owner_id == owner_id,
                World.name.ilike(
                    f"%{query}%"
                ),
            )
            .order_by(
                World.created_at.desc()
            )
            .limit(limit)
        )

        return list(
            self.session.scalars(
                statement
            ).all()
        )

    def archive_world(
        self,
        world_id: Any,
        *,
        owner_id: Any,
    ) -> World | None:
        return self.change_status(
            world_id,
            "archived",
            owner_id=owner_id,
        )
