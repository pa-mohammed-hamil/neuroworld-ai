
"""
NeuroWorld AI
File: backend/app/database/repositories/checkpoint_repository.py

Model checkpoint metadata repository.

Expected Checkpoint model fields:
    id
    experiment_id
    model_name
    version
    epoch
    global_step
    storage_uri
    metrics
    is_best
    is_active
    created_at

This repository stores checkpoint metadata.
Actual checkpoint files must be saved and
verified by a separate storage service.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Checkpoint

from .base import BaseRepository


class CheckpointRepository(
    BaseRepository[Checkpoint]
):
    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(
            session,
            Checkpoint,
        )

    def register_checkpoint(
        self,
        *,
        experiment_id: Any,
        model_name: str,
        version: str,
        epoch: int,
        global_step: int,
        storage_uri: str,
        metrics: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Checkpoint:
        if epoch < 0 or global_step < 0:
            raise ValueError(
                "Epoch and step must be nonnegative"
            )

        if not storage_uri.strip():
            raise ValueError(
                "Checkpoint storage URI is required"
            )

        if self.get_by(
            model_name=model_name,
            version=version,
        ) is not None:
            raise ValueError(
                "Checkpoint version already exists"
            )

        checkpoint = self.create(
            experiment_id=experiment_id,
            model_name=model_name,
            version=version,
            epoch=epoch,
            global_step=global_step,
            storage_uri=storage_uri,
            metrics=metrics or {},
            is_best=False,
            is_active=False,
        )

        if is_best:
            self.mark_best(
                checkpoint.id
            )

        return checkpoint

    def list_by_experiment(
        self,
        experiment_id: Any,
        *,
        limit: int = 100,
    ) -> list[Checkpoint]:
        return self.list(
            experiment_id=experiment_id,
            limit=limit,
            order_by="created_at",
        )

    def list_by_model(
        self,
        model_name: str,
        *,
        limit: int = 100,
    ) -> list[Checkpoint]:
        return self.list(
            model_name=model_name,
            limit=limit,
            order_by="created_at",
        )

    def latest_checkpoint(
        self,
        model_name: str,
    ) -> Checkpoint | None:
        statement = (
            select(Checkpoint)
            .where(
                Checkpoint.model_name == model_name
            )
            .order_by(
                Checkpoint.created_at.desc(),
                Checkpoint.id.desc(),
            )
            .limit(1)
        )

        return self.session.scalar(
            statement
        )

    def best_checkpoint(
        self,
        model_name: str,
    ) -> Checkpoint | None:
        return self.get_by(
            model_name=model_name,
            is_best=True,
        )

    def active_checkpoint(
        self,
        model_name: str,
    ) -> Checkpoint | None:
        return self.get_by(
            model_name=model_name,
            is_active=True,
        )

    def mark_best(
        self,
        checkpoint_id: Any,
    ) -> Checkpoint:
        checkpoint = self.get_or_raise(
            checkpoint_id
        )

        statement = (
            select(Checkpoint)
            .where(
                Checkpoint.model_name
                == checkpoint.model_name
            )
            .with_for_update()
        )

        checkpoints = list(
            self.session.scalars(
                statement
            ).all()
        )

        for item in checkpoints:
            item.is_best = (
                item.id == checkpoint_id
            )

        self._commit()
        self.session.refresh(checkpoint)

        return checkpoint

    def activate_checkpoint(
        self,
        checkpoint_id: Any,
    ) -> Checkpoint:
        checkpoint = self.get_or_raise(
            checkpoint_id
        )

        statement = (
            select(Checkpoint)
            .where(
                Checkpoint.model_name
                == checkpoint.model_name
            )
            .with_for_update()
        )

        checkpoints = list(
            self.session.scalars(
                statement
            ).all()
        )

        for item in checkpoints:
            item.is_active = (
                item.id == checkpoint_id
            )

        self._commit()
        self.session.refresh(checkpoint)

        return checkpoint

    def rollback_to_version(
        self,
        model_name: str,
        version: str,
    ) -> Checkpoint:
        checkpoint = self.get_by(
            model_name=model_name,
            version=version,
        )

        if checkpoint is None:
            raise LookupError(
                "Checkpoint version not found"
            )

        # Only changes active metadata.
        # The serving system must reload the model.
        return self.activate_checkpoint(
            checkpoint.id
        )

    def update_checkpoint_metrics(
        self,
        checkpoint_id: Any,
        metrics: dict[str, Any],
    ) -> Checkpoint | None:
        checkpoint = self.get(
            checkpoint_id
        )

        if checkpoint is None:
            return None

        merged = dict(
            checkpoint.metrics or {}
        )

        merged.update(metrics)

        return self.update(
            checkpoint_id,
            metrics=merged,
        )
