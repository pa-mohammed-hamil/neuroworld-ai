
"""
NeuroWorld AI
File: backend/app/database/repositories/experiment_repository.py

Experiment tracking repository.

Expected Experiment model fields:
    id
    name
    experiment_type
    model_name
    configuration
    metrics
    status
    owner_id
    started_at
    completed_at
    created_at
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Experiment

from .base import BaseRepository


class ExperimentRepository(
    BaseRepository[Experiment]
):
    VALID_STATUSES = {
        "created",
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
    }

    TRANSITIONS = {
        "created": {"queued", "cancelled"},
        "queued": {
            "running",
            "cancelled",
            "failed",
        },
        "running": {
            "completed",
            "failed",
            "cancelled",
        },
        "completed": set(),
        "failed": set(),
        "cancelled": set(),
    }

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(
            session,
            Experiment,
        )

    def create_experiment(
        self,
        *,
        name: str,
        experiment_type: str,
        model_name: str,
        owner_id: Any,
        configuration: dict[str, Any] | None = None,
    ) -> Experiment:
        if not name.strip():
            raise ValueError(
                "Experiment name is required"
            )

        return self.create(
            name=name.strip(),
            experiment_type=experiment_type,
            model_name=model_name,
            owner_id=owner_id,
            configuration=configuration or {},
            metrics={},
            status="created",
        )

    def change_status(
        self,
        experiment_id: Any,
        new_status: str,
        *,
        owner_id: Any,
    ) -> Experiment | None:
        if new_status not in self.VALID_STATUSES:
            raise ValueError(
                "Invalid experiment status"
            )

        statement = (
            select(Experiment)
            .where(
                Experiment.id == experiment_id,
                Experiment.owner_id == owner_id,
            )
            .with_for_update()
        )

        experiment = self.session.scalar(
            statement
        )

        if experiment is None:
            return None

        if experiment.status == new_status:
            return experiment

        allowed = self.TRANSITIONS.get(
            experiment.status,
            set(),
        )

        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition: "
                f"{experiment.status} -> {new_status}"
            )

        now = datetime.now(
            timezone.utc
        )

        experiment.status = new_status

        if new_status == "running":
            experiment.started_at = now

        if new_status in {
            "completed",
            "failed",
            "cancelled",
        }:
            experiment.completed_at = now

        self._commit()
        self.session.refresh(experiment)

        return experiment

    def update_metrics(
        self,
        experiment_id: Any,
        metrics: dict[str, Any],
        *,
        owner_id: Any,
    ) -> Experiment | None:
        statement = (
            select(Experiment)
            .where(
                Experiment.id == experiment_id,
                Experiment.owner_id == owner_id,
            )
            .with_for_update()
        )

        experiment = self.session.scalar(
            statement
        )

        if experiment is None:
            return None

        if experiment.status not in {
            "running",
            "completed",
        }:
            raise ValueError(
                "Metrics can only be updated "
                "for running or completed experiments"
            )

        current = dict(
            experiment.metrics or {}
        )

        current.update(metrics)

        # Assign a new dictionary so SQLAlchemy
        # detects changes to a JSON column.
        experiment.metrics = current

        self._commit()
        self.session.refresh(experiment)

        return experiment

    def list_by_owner(
        self,
        owner_id: Any,
        *,
        limit: int = 100,
    ) -> list[Experiment]:
        return self.list(
            owner_id=owner_id,
            limit=limit,
            order_by="created_at",
        )

    def list_by_model(
        self,
        model_name: str,
        *,
        owner_id: Any,
        limit: int = 100,
    ) -> list[Experiment]:
        return self.list(
            model_name=model_name,
            owner_id=owner_id,
            limit=limit,
            order_by="created_at",
        )

    def list_running(
        self,
        *,
        limit: int = 100,
    ) -> list[Experiment]:
        return self.list(
            status="running",
            limit=limit,
            order_by="created_at",
        )

    def best_experiment(
        self,
        *,
        owner_id: Any,
        model_name: str,
        metric_name: str = "validation_loss",
        minimize: bool = True,
    ) -> Experiment | None:
        """
        Compare completed experiments.

        Metrics are read in Python for portability
        across SQLite and PostgreSQL JSON columns.
        """
        statement = (
            select(Experiment)
            .where(
                Experiment.owner_id == owner_id,
                Experiment.model_name == model_name,
                Experiment.status == "completed",
            )
        )

        experiments = list(
            self.session.scalars(
                statement
            ).all()
        )

        candidates = []

        for experiment in experiments:
            metrics = (
                experiment.metrics or {}
            )

            value = metrics.get(
                metric_name
            )

            if isinstance(
                value,
                (int, float),
            ):
                candidates.append(
                    (
                        float(value),
                        experiment,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=not minimize,
        )

        return candidates[0][1]
