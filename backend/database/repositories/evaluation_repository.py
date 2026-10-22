
"""
NeuroWorld AI
File: backend/app/database/repositories/evaluation_repository.py

Evaluation and benchmark persistence.

Expected Evaluation model fields:
    id
    experiment_id
    checkpoint_id
    model_name
    evaluation_type
    dataset_name
    metrics
    configuration
    passed
    created_at

Supported evaluation categories include:
    prediction
    reconstruction
    temporal
    dynamics
    physics
    contrastive
    consistency
    robustness
    planning
"""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Evaluation

from .base import BaseRepository


class EvaluationRepository(
    BaseRepository[Evaluation]
):
    EVALUATION_TYPES = {
        "prediction",
        "reconstruction",
        "temporal",
        "dynamics",
        "physics",
        "contrastive",
        "consistency",
        "robustness",
        "planning",
        "calibration",
        "generalization",
        "trajectory",
    }

    def __init__(
        self,
        session: Session,
    ) -> None:
        super().__init__(
            session,
            Evaluation,
        )

    @staticmethod
    def _validate_json(
        value: Any,
        path: str = "metrics",
    ) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                EvaluationRepository._validate_json(
                    item,
                    f"{path}.{key}",
                )

        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                EvaluationRepository._validate_json(
                    item,
                    f"{path}[{index}]",
                )

        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite value at {path}"
                )

    def create_evaluation(
        self,
        *,
        model_name: str,
        evaluation_type: str,
        dataset_name: str,
        metrics: dict[str, Any],
        passed: bool,
        experiment_id: Any | None = None,
        checkpoint_id: Any | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> Evaluation:
        if evaluation_type not in self.EVALUATION_TYPES:
            raise ValueError(
                "Unsupported evaluation type"
            )

        if not model_name.strip():
            raise ValueError(
                "Model name is required"
            )

        self._validate_json(
            metrics
        )

        self._validate_json(
            configuration or {},
            "configuration",
        )

        return self.create(
            experiment_id=experiment_id,
            checkpoint_id=checkpoint_id,
            model_name=model_name,
            evaluation_type=evaluation_type,
            dataset_name=dataset_name,
            metrics=metrics,
            configuration=configuration or {},
            passed=bool(passed),
        )

    def list_by_model(
        self,
        model_name: str,
        *,
        limit: int = 100,
    ) -> list[Evaluation]:
        return self.list(
            model_name=model_name,
            limit=limit,
            order_by="created_at",
        )

    def list_by_experiment(
        self,
        experiment_id: Any,
        *,
        limit: int = 100,
    ) -> list[Evaluation]:
        return self.list(
            experiment_id=experiment_id,
            limit=limit,
            order_by="created_at",
        )

    def list_by_checkpoint(
        self,
        checkpoint_id: Any,
        *,
        limit: int = 100,
    ) -> list[Evaluation]:
        return self.list(
            checkpoint_id=checkpoint_id,
            limit=limit,
            order_by="created_at",
        )

    def list_by_type(
        self,
        evaluation_type: str,
        *,
        limit: int = 100,
    ) -> list[Evaluation]:
        if evaluation_type not in self.EVALUATION_TYPES:
            raise ValueError(
                "Unsupported evaluation type"
            )

        return self.list(
            evaluation_type=evaluation_type,
            limit=limit,
            order_by="created_at",
        )

    def latest_evaluation(
        self,
        *,
        model_name: str,
        evaluation_type: str,
    ) -> Evaluation | None:
        statement = (
            select(Evaluation)
            .where(
                Evaluation.model_name == model_name,
                Evaluation.evaluation_type
                == evaluation_type,
            )
            .order_by(
                Evaluation.created_at.desc(),
                Evaluation.id.desc(),
            )
            .limit(1)
        )

        return self.session.scalar(
            statement
        )

    def get_pass_rate(
        self,
        model_name: str,
        evaluation_type: str | None = None,
    ) -> dict[str, Any]:
        statement = (
            select(Evaluation)
            .where(
                Evaluation.model_name == model_name
            )
        )

        if evaluation_type is not None:
            statement = statement.where(
                Evaluation.evaluation_type
                == evaluation_type
            )

        records = list(
            self.session.scalars(
                statement
            ).all()
        )

        total = len(records)

        passed = sum(
            bool(record.passed)
            for record in records
        )

        return {
            "model_name": model_name,
            "evaluation_type": evaluation_type,
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (
                passed / total
                if total
                else 0.0
            ),
        }

    def compare_models(
        self,
        model_names: list[str],
        *,
        evaluation_type: str,
        metric_name: str = "rmse",
        minimize: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Compare latest evaluations for models
        on the same evaluation category.

        For rigorous comparisons, also require
        the same dataset, preprocessing, and
        evaluation configuration.
        """
        if evaluation_type not in self.EVALUATION_TYPES:
            raise ValueError(
                "Unsupported evaluation type"
            )

        results = []

        for model_name in model_names:
            evaluation = self.latest_evaluation(
                model_name=model_name,
                evaluation_type=evaluation_type,
            )

            if evaluation is None:
                continue

            value = (
                evaluation.metrics or {}
            ).get(metric_name)

            if not isinstance(
                value,
                (int, float),
            ):
                continue

            if not math.isfinite(float(value)):
                continue

            results.append(
                {
                    "model_name": model_name,
                    "evaluation_id": evaluation.id,
                    "dataset_name": evaluation.dataset_name,
                    "metric": metric_name,
                    "value": float(value),
                    "passed": bool(
                        evaluation.passed
                    ),
                }
            )

        results.sort(
            key=lambda item: item["value"],
            reverse=not minimize,
        )

        return results

    def delete_experiment_evaluations(
        self,
        experiment_id: Any,
    ) -> int:
        return self.delete_by(
            experiment_id=experiment_id,
        )
