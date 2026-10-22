
"""
NeuroWorld AI
backend/app/database/evaluation.py

Persistent Evaluation Results and Benchmark Storage.

Features:
- Save evaluation results
- Store model benchmark metrics
- Retrieve evaluation history
- Filter by model, type and status
- Compare model evaluations
- Aggregate model performance
- Store experiment metadata
- JSON-compatible NumPy serialization
- Pagination
- JSON and CSV export
- Database transaction management
- PostgreSQL and SQLite support

Dependencies:
    pip install sqlalchemy numpy

Run:
    python -m backend.app.database.evaluation

Uses:
    backend.app.database.session.Base
    backend.app.database.session.session_scope

Notes:
    Evaluation scores are not necessarily
    comparable across different datasets,
    metric configurations, or horizons.

    This module stores evaluation results.
    It does not execute neural model inference.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    delete,
    func,
    select,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    mapped_column,
)

from backend.app.database.session import (
    Base,
    session_scope,
)


logger = logging.getLogger(
    "neuroworld.database.evaluation"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class EvaluationDatabaseError(Exception):
    """Base evaluation database exception."""


class EvaluationValidationError(EvaluationDatabaseError):
    """Invalid evaluation data."""


class EvaluationNotFoundError(EvaluationDatabaseError):
    """Evaluation record not found."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_datetime(
    value: datetime,
) -> datetime:
    if value.tzinfo is None:
        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(timezone.utc)


def validate_identifier(
    value: str,
    name: str,
    max_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise EvaluationValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > max_length:
        raise EvaluationValidationError(
            f"Invalid {name}"
        )

    return value


def json_safe(
    value: Any,
) -> Any:
    """
    Convert dataclasses, NumPy arrays and
    scalar values to JSON-compatible objects.

    Reject non-finite values and unsupported
    Python objects.
    """

    if is_dataclass(value) and not isinstance(
        value,
        type,
    ):
        return json_safe(
            asdict(value)
        )

    if isinstance(value, np.ndarray):
        return json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return json_safe(
            value.item()
        )

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, datetime):
        return normalize_datetime(
            value
        ).isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationValidationError(
                "Non-finite numbers are not allowed"
            )

        return value

    if value is None or isinstance(
        value,
        (str, int, bool),
    ):
        return value

    raise EvaluationValidationError(
        f"Unsupported value: {type(value).__name__}"
    )


def validate_json_object(
    value: Any,
    name: str,
) -> dict[str, Any]:
    normalized = json_safe(value)

    if not isinstance(normalized, dict):
        raise EvaluationValidationError(
            f"{name} must be a dictionary"
        )

    return normalized


def validate_score(
    value: float,
    name: str = "overall_score",
) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(score) or not 0 <= score <= 1:
        raise EvaluationValidationError(
            f"{name} must be between 0 and 1"
        )

    return score


# ============================================================
# ORM MODEL
# ============================================================

class EvaluationRecordORM(Base):
    """
    Persistent evaluation record.

    The metrics field supports arbitrary
    JSON-compatible evaluation metrics.
    """

    __tablename__ = "evaluation_records"

    __table_args__ = (
        Index(
            "ix_evaluation_model_type_created",
            "model_name",
            "evaluation_type",
            "created_at",
        ),
    )

    evaluation_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    model_version: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )

    evaluation_type: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    experiment_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="unspecified",
    )

    sample_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    horizon: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    overall_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    passed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    metrics_data: Mapped[dict[str, Any]] = mapped_column(
        "metrics",
        JSON,
        nullable=False,
    )

    raw_metrics_data: Mapped[dict[str, Any]] = mapped_column(
        "raw_metrics",
        JSON,
        nullable=False,
        default=dict,
    )

    details_data: Mapped[dict[str, Any]] = mapped_column(
        "details",
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )


# ============================================================
# PUBLIC DATA MODELS
# ============================================================

@dataclass(frozen=True)
class EvaluationRecord:
    evaluation_id: str
    model_name: str
    model_version: str
    evaluation_type: str

    experiment_id: str | None
    dataset_name: str

    sample_count: int
    horizon: int | None

    overall_score: float
    passed: bool

    metrics: dict[str, Any]
    raw_metrics: dict[str, Any]
    details: dict[str, Any]

    created_at: str


@dataclass(frozen=True)
class EvaluationSummary:
    model_name: str
    evaluation_count: int
    mean_score: float
    best_score: float
    worst_score: float
    pass_rate: float


@dataclass(frozen=True)
class ModelEvaluationComparison:
    evaluation_type: str
    dataset_name: str
    model_results: list[EvaluationRecord]
    best_model: str | None


# ============================================================
# REPOSITORY
# ============================================================

class EvaluationRepository:
    """
    Database-backed evaluation repository.

    Accepts an optional caller-owned Session.

    When a Session is supplied:
        - The caller controls commit/rollback.
        - The repository does not close it.

    Without a Session:
        - A managed transaction is used.
    """

    def _session(
        self,
        db: Session | None,
    ):
        if db is not None:
            return nullcontext(db)

        return session_scope()

    @staticmethod
    def _to_record(
        row: EvaluationRecordORM,
    ) -> EvaluationRecord:
        return EvaluationRecord(
            evaluation_id=row.evaluation_id,
            model_name=row.model_name,
            model_version=row.model_version,
            evaluation_type=row.evaluation_type,
            experiment_id=row.experiment_id,
            dataset_name=row.dataset_name,
            sample_count=row.sample_count,
            horizon=row.horizon,
            overall_score=row.overall_score,
            passed=row.passed,
            metrics=row.metrics_data,
            raw_metrics=row.raw_metrics_data,
            details=row.details_data,
            created_at=normalize_datetime(
                row.created_at
            ).isoformat(),
        )

    # --------------------------------------------------------
    # SAVE EVALUATION
    # --------------------------------------------------------

    def save_evaluation(
        self,
        *,
        model_name: str,
        evaluation_type: str,
        metrics: dict[str, Any],
        overall_score: float,
        passed: bool,
        sample_count: int,
        model_version: str = "unknown",
        dataset_name: str = "unspecified",
        horizon: int | None = None,
        experiment_id: str | None = None,
        raw_metrics: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        evaluation_id: str | None = None,
        db: Session | None = None,
    ) -> EvaluationRecord:
        model_name = validate_identifier(
            model_name,
            "model_name",
        )

        evaluation_type = validate_identifier(
            evaluation_type,
            "evaluation_type",
        )

        model_version = validate_identifier(
            model_version,
            "model_version",
        )

        dataset_name = validate_identifier(
            dataset_name,
            "dataset_name",
            255,
        )

        if experiment_id is not None:
            experiment_id = validate_identifier(
                experiment_id,
                "experiment_id",
            )

        if not isinstance(sample_count, int) or sample_count < 1:
            raise EvaluationValidationError(
                "sample_count must be a positive integer"
            )

        if horizon is not None and (
            not isinstance(horizon, int)
            or horizon < 1
        ):
            raise EvaluationValidationError(
                "horizon must be positive"
            )

        if not isinstance(passed, bool):
            raise EvaluationValidationError(
                "passed must be boolean"
            )

        overall_score = validate_score(
            overall_score
        )

        metrics = validate_json_object(
            metrics,
            "metrics",
        )

        raw_metrics = validate_json_object(
            raw_metrics or {},
            "raw_metrics",
        )

        details = validate_json_object(
            details or {},
            "details",
        )

        identifier = (
            validate_identifier(
                evaluation_id,
                "evaluation_id",
                64,
            )
            if evaluation_id is not None
            else uuid.uuid4().hex
        )

        row = EvaluationRecordORM(
            evaluation_id=identifier,
            model_name=model_name,
            model_version=model_version,
            evaluation_type=evaluation_type,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            sample_count=sample_count,
            horizon=horizon,
            overall_score=overall_score,
            passed=passed,
            metrics_data=metrics,
            raw_metrics_data=raw_metrics,
            details_data=details,
            created_at=utc_now(),
        )

        with self._session(db) as session:
            session.add(row)
            session.flush()

            record = self._to_record(row)

        logger.info(
            "Evaluation saved: model=%s type=%s id=%s",
            model_name,
            evaluation_type,
            identifier,
        )

        return record

    # --------------------------------------------------------
    # SAVE RESULT DATACLASS
    # --------------------------------------------------------

    def save_result(
        self,
        result: Any,
        *,
        evaluation_type: str,
        model_version: str = "unknown",
        dataset_name: str = "unspecified",
        experiment_id: str | None = None,
        db: Session | None = None,
    ) -> EvaluationRecord:
        """
        Save results from evaluation modules.

        Expected fields:
            evaluation_id
            model_name
            sample_count
            metrics.overall_score
            passed

        Optional fields:
            horizon
            raw_metrics
            details
        """

        data = json_safe(result)

        if not isinstance(data, dict):
            raise EvaluationValidationError(
                "Evaluation result must be an object"
            )

        metrics = data.get("metrics")

        if not isinstance(metrics, dict):
            raise EvaluationValidationError(
                "Result must contain metrics"
            )

        if "overall_score" not in metrics:
            raise EvaluationValidationError(
                "Result metrics require overall_score"
            )

        details = dict(
            data.get("details") or {}
        )

        # Preserve horizon-level diagnostics
        # when provided by the evaluator.
        if "horizon_metrics" in data:
            details["horizon_metrics"] = (
                data["horizon_metrics"]
            )

        if "object_metrics" in data:
            details["object_metrics"] = (
                data["object_metrics"]
            )

        return self.save_evaluation(
            evaluation_id=data.get(
                "evaluation_id"
            ),
            model_name=data["model_name"],
            evaluation_type=evaluation_type,
            model_version=model_version,
            dataset_name=dataset_name,
            experiment_id=experiment_id,
            sample_count=data["sample_count"],
            horizon=data.get("horizon"),
            metrics=metrics,
            overall_score=metrics["overall_score"],
            passed=data["passed"],
            raw_metrics=data.get(
                "raw_metrics",
                {},
            ),
            details=details,
            db=db,
        )

    # --------------------------------------------------------
    # GET EVALUATION
    # --------------------------------------------------------

    def get_evaluation(
        self,
        evaluation_id: str,
        *,
        db: Session | None = None,
    ) -> EvaluationRecord:
        evaluation_id = validate_identifier(
            evaluation_id,
            "evaluation_id",
            64,
        )

        with self._session(db) as session:
            row = session.get(
                EvaluationRecordORM,
                evaluation_id,
            )

            if row is None:
                raise EvaluationNotFoundError(
                    "Evaluation not found"
                )

            return self._to_record(
                row
            )

    # --------------------------------------------------------
    # LIST EVALUATIONS
    # --------------------------------------------------------

    def list_evaluations(
        self,
        *,
        model_name: str | None = None,
        evaluation_type: str | None = None,
        dataset_name: str | None = None,
        experiment_id: str | None = None,
        passed: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Session | None = None,
    ) -> list[EvaluationRecord]:
        if not 1 <= limit <= 1000:
            raise EvaluationValidationError(
                "limit must be between 1 and 1000"
            )

        if offset < 0:
            raise EvaluationValidationError(
                "offset cannot be negative"
            )

        statement = select(
            EvaluationRecordORM
        )

        if model_name is not None:
            statement = statement.where(
                EvaluationRecordORM.model_name
                == model_name
            )

        if evaluation_type is not None:
            statement = statement.where(
                EvaluationRecordORM.evaluation_type
                == evaluation_type
            )

        if dataset_name is not None:
            statement = statement.where(
                EvaluationRecordORM.dataset_name
                == dataset_name
            )

        if experiment_id is not None:
            statement = statement.where(
                EvaluationRecordORM.experiment_id
                == experiment_id
            )

        if passed is not None:
            statement = statement.where(
                EvaluationRecordORM.passed
                == passed
            )

        statement = (
            statement
            .order_by(
                EvaluationRecordORM.created_at.desc(),
                EvaluationRecordORM.evaluation_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            return [
                self._to_record(row)
                for row in rows
            ]

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    def get_model_summary(
        self,
        model_name: str,
        *,
        evaluation_type: str | None = None,
        dataset_name: str | None = None,
        db: Session | None = None,
    ) -> EvaluationSummary:
        model_name = validate_identifier(
            model_name,
            "model_name",
        )

        statement = select(
            func.count(
                EvaluationRecordORM.evaluation_id
            ),
            func.avg(
                EvaluationRecordORM.overall_score
            ),
            func.max(
                EvaluationRecordORM.overall_score
            ),
            func.min(
                EvaluationRecordORM.overall_score
            ),
            func.avg(
                func.cast(
                    EvaluationRecordORM.passed,
                    Integer,
                )
            ),
        ).where(
            EvaluationRecordORM.model_name
            == model_name
        )

        if evaluation_type is not None:
            statement = statement.where(
                EvaluationRecordORM.evaluation_type
                == evaluation_type
            )

        if dataset_name is not None:
            statement = statement.where(
                EvaluationRecordORM.dataset_name
                == dataset_name
            )

        with self._session(db) as session:
            row = session.execute(
                statement
            ).one()

        count, mean, best, worst, pass_rate = row

        return EvaluationSummary(
            model_name=model_name,
            evaluation_count=int(count or 0),
            mean_score=float(mean or 0),
            best_score=float(best or 0),
            worst_score=float(worst or 0),
            pass_rate=float(pass_rate or 0),
        )

    # --------------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------------

    def compare_models(
        self,
        *,
        model_names: list[str],
        evaluation_type: str,
        dataset_name: str,
        db: Session | None = None,
    ) -> ModelEvaluationComparison:
        """
        Compare the latest evaluation for each
        model on the same named dataset.

        For rigorous comparison, ensure the
        evaluation configurations, test samples
        and horizons are identical.
        """

        if not model_names:
            raise EvaluationValidationError(
                "At least one model is required"
            )

        evaluation_type = validate_identifier(
            evaluation_type,
            "evaluation_type",
        )

        dataset_name = validate_identifier(
            dataset_name,
            "dataset_name",
            255,
        )

        names = [
            validate_identifier(
                name,
                "model_name",
            )
            for name in model_names
        ]

        statement = (
            select(EvaluationRecordORM)
            .where(
                EvaluationRecordORM.model_name.in_(
                    names
                ),
                EvaluationRecordORM.evaluation_type
                == evaluation_type,
                EvaluationRecordORM.dataset_name
                == dataset_name,
            )
            .order_by(
                EvaluationRecordORM.created_at.desc(),
                EvaluationRecordORM.evaluation_id.desc(),
            )
        )

        with self._session(db) as session:
            rows = session.scalars(
                statement
            ).all()

            latest: dict[
                str,
                EvaluationRecord,
            ] = {}

            for row in rows:
                if row.model_name not in latest:
                    latest[row.model_name] = (
                        self._to_record(row)
                    )

        results = sorted(
            latest.values(),
            key=lambda record: record.overall_score,
            reverse=True,
        )

        return ModelEvaluationComparison(
            evaluation_type=evaluation_type,
            dataset_name=dataset_name,
            model_results=results,
            best_model=(
                results[0].model_name
                if results
                else None
            ),
        )

    # --------------------------------------------------------
    # DELETE EVALUATION
    # --------------------------------------------------------

    def delete_evaluation(
        self,
        evaluation_id: str,
        *,
        db: Session | None = None,
    ) -> bool:
        evaluation_id = validate_identifier(
            evaluation_id,
            "evaluation_id",
            64,
        )

        with self._session(db) as session:
            result = session.execute(
                delete(
                    EvaluationRecordORM
                ).where(
                    EvaluationRecordORM.evaluation_id
                    == evaluation_id
                )
            )

            return bool(
                result.rowcount
            )

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        records: list[EvaluationRecord],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                [
                    asdict(record)
                    for record in records
                ],
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return path

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        records: list[EvaluationRecord],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "evaluation_id",
            "model_name",
            "model_version",
            "evaluation_type",
            "experiment_id",
            "dataset_name",
            "sample_count",
            "horizon",
            "overall_score",
            "passed",
            "metrics",
            "raw_metrics",
            "details",
            "created_at",
        ]

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for record in records:
                row = asdict(record)

                for key in (
                    "metrics",
                    "raw_metrics",
                    "details",
                ):
                    row[key] = json.dumps(
                        row[key],
                        ensure_ascii=False,
                        allow_nan=False,
                    )

                writer.writerow(row)

        return path


# ============================================================
# FACTORY
# ============================================================

def create_evaluation_repository() -> EvaluationRepository:
    return EvaluationRepository()


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from backend.app.database.session import (
        create_database_tables,
        close_database,
    )

    logging.basicConfig(
        level=logging.INFO
    )

    create_database_tables()

    repository = create_evaluation_repository()

    record = repository.save_evaluation(
        model_name="neuroworld_demo_model",
        model_version="v1.0",
        evaluation_type="trajectory_accuracy",
        dataset_name="synthetic_navigation",
        sample_count=128,
        horizon=12,
        overall_score=0.92,
        passed=True,
        metrics={
            "trajectory_accuracy": 0.94,
            "position_accuracy": 0.93,
            "velocity_accuracy": 0.90,
            "final_position_accuracy": 0.91,
            "drift_score": 0.92,
            "overall_score": 0.92,
        },
        raw_metrics={
            "trajectory_rmse": 0.08,
            "average_displacement_error": 0.12,
            "final_displacement_error": 0.16,
        },
        details={
            "source": "synthetic_demo",
            "note": "Illustrative values only",
        },
    )

    print("=" * 60)
    print("NEUROWORLD AI - EVALUATION DATABASE")
    print("=" * 60)

    print(
        "Evaluation saved:",
        record.evaluation_id,
    )

    retrieved = repository.get_evaluation(
        record.evaluation_id
    )

    print(
        "Retrieved score:",
        retrieved.overall_score,
    )

    summary = repository.get_model_summary(
        "neuroworld_demo_model",
        evaluation_type="trajectory_accuracy",
        dataset_name="synthetic_navigation",
    )

    print(
        "Model summary:",
        asdict(summary),
    )

    records = repository.list_evaluations(
        model_name="neuroworld_demo_model",
        limit=100,
    )

    json_path = repository.export_json(
        records,
        "experiments/reports/evaluation_history.json",
    )

    csv_path = repository.export_csv(
        records,
        "experiments/reports/evaluation_history.csv",
    )

    print(
        "JSON report:",
        json_path,
    )

    print(
        "CSV report:",
        csv_path,
    )

    close_database()


if __name__ == "__main__":
    main()
