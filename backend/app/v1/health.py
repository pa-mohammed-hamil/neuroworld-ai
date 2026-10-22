


"""
NeuroWorld AI
backend/app/database/health.py

Database Health Monitoring and Diagnostics.

Features:
- Database connectivity checks
- SELECT 1 readiness probe
- Database response latency
- Connection pool statistics
- SQLite / PostgreSQL support
- Health-check history persistence
- Health status aggregation
- Database diagnostics
- JSON report export
- FastAPI-compatible health response

Dependencies:
    pip install sqlalchemy

Requires:
    backend.app.database.session.Base
    backend.app.database.session.get_engine
    backend.app.database.session.session_scope

Security:
    Never expose credentials, raw connection URLs,
    internal exception details, or host topology
    through public health endpoints.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    Mapped,
    Session,
    mapped_column,
)

from backend.app.database.session import (
    Base,
    get_engine,
    session_scope,
)


logger = logging.getLogger(
    "neuroworld.database.health"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class DatabaseHealthError(Exception):
    """Base health monitoring exception."""


class HealthValidationError(DatabaseHealthError):
    """Invalid health monitoring configuration."""


# ============================================================
# ENUMS
# ============================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_datetime(
    value: datetime | None,
) -> str | None:
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def validate_positive_number(
    value: float,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise HealthValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HealthValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result) or result <= 0:
        raise HealthValidationError(
            f"{name} must be positive and finite"
        )

    return result


# ============================================================
# DATABASE MODEL
# ============================================================

class DatabaseHealthCheckORM(Base):
    __tablename__ = "database_health_checks"

    __table_args__ = (
        Index(
            "ix_db_health_checked_status",
            "checked_at",
            "status",
        ),
    )

    check_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    database_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    connected: Mapped[bool] = mapped_column(
        nullable=False,
    )

    latency_ms: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    threshold_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    pool_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "pool",
        JSON,
        nullable=False,
        default=dict,
    )

    details_data: Mapped[
        dict[str, Any]
    ] = mapped_column(
        "details",
        JSON,
        nullable=False,
        default=dict,
    )

    error_category: Mapped[
        str | None
    ] = mapped_column(
        String(128),
        nullable=True,
    )

    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


# ============================================================
# PUBLIC DATA MODELS
# ============================================================

@dataclass(frozen=True)
class PoolStatistics:
    pool_type: str
    size: int | None
    checked_in: int | None
    checked_out: int | None
    overflow: int | None
    invalidated: int | None


@dataclass(frozen=True)
class DatabaseHealthResult:
    check_id: str
    status: str
    database_type: str
    connected: bool
    latency_ms: float | None
    threshold_ms: float
    pool: dict[str, Any]
    details: dict[str, Any]
    error_category: str | None
    checked_at: str


@dataclass(frozen=True)
class DatabaseHealthSummary:
    total_checks: int
    healthy_checks: int
    degraded_checks: int
    unhealthy_checks: int
    average_latency_ms: float | None
    maximum_latency_ms: float | None
    availability_rate: float


# ============================================================
# HEALTH REPOSITORY
# ============================================================

class DatabaseHealthRepository:
    """
    Store and retrieve health check history.

    Persistence failures must not be interpreted
    as successful database readiness.
    """

    @staticmethod
    def _session(
        db: Session | None,
    ):
        return (
            nullcontext(db)
            if db is not None
            else session_scope()
        )

    @staticmethod
    def _to_record(
        row: DatabaseHealthCheckORM,
    ) -> DatabaseHealthResult:
        return DatabaseHealthResult(
            check_id=row.check_id,
            status=row.status,
            database_type=row.database_type,
            connected=row.connected,
            latency_ms=row.latency_ms,
            threshold_ms=row.threshold_ms,
            pool=dict(row.pool_data),
            details=dict(row.details_data),
            error_category=row.error_category,
            checked_at=iso_datetime(
                row.checked_at
            ),
        )

    def save_check(
        self,
        result: DatabaseHealthResult,
        *,
        db: Session | None = None,
    ) -> DatabaseHealthResult:
        with self._session(db) as session:
            row = DatabaseHealthCheckORM(
                check_id=result.check_id,
                status=result.status,
                database_type=result.database_type,
                connected=result.connected,
                latency_ms=result.latency_ms,
                threshold_ms=result.threshold_ms,
                pool_data=result.pool,
                details_data=result.details,
                error_category=result.error_category,
                checked_at=datetime.fromisoformat(
                    result.checked_at
                ),
            )

            session.add(row)
            session.flush()

            return self._to_record(row)

    def get_latest_check(
        self,
        *,
        db: Session | None = None,
    ) -> DatabaseHealthResult | None:
        statement = (
            select(DatabaseHealthCheckORM)
            .order_by(
                DatabaseHealthCheckORM.checked_at.desc(),
                DatabaseHealthCheckORM.check_id.desc(),
            )
            .limit(1)
        )

        with self._session(db) as session:
            row = session.scalar(statement)

            return (
                self._to_record(row)
                if row is not None
                else None
            )

    def list_checks(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        status: HealthStatus | None = None,
        db: Session | None = None,
    ) -> list[DatabaseHealthResult]:
        if not 1 <= limit <= 1000 or offset < 0:
            raise HealthValidationError(
                "Invalid pagination"
            )

        statement = select(
            DatabaseHealthCheckORM
        )

        if status is not None:
            statement = statement.where(
                DatabaseHealthCheckORM.status
                == status.value
            )

        statement = (
            statement
            .order_by(
                DatabaseHealthCheckORM.checked_at.desc(),
                DatabaseHealthCheckORM.check_id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )

        with self._session(db) as session:
            return [
                self._to_record(row)
                for row in session.scalars(
                    statement
                ).all()
            ]

    def get_summary(
        self,
        *,
        hours: int = 24,
        db: Session | None = None,
    ) -> DatabaseHealthSummary:
        if not 1 <= hours <= 8760:
            raise HealthValidationError(
                "hours must be between 1 and 8760"
            )

        cutoff = utc_now() - timedelta(
            hours=hours
        )

        with self._session(db) as session:
            rows = session.execute(
                select(
                    DatabaseHealthCheckORM.status,
                    DatabaseHealthCheckORM.connected,
                    DatabaseHealthCheckORM.latency_ms,
                ).where(
                    DatabaseHealthCheckORM.checked_at
                    >= cutoff
                )
            ).all()

        total = len(rows)

        healthy = sum(
            row.status == HealthStatus.HEALTHY.value
            for row in rows
        )

        degraded = sum(
            row.status == HealthStatus.DEGRADED.value
            for row in rows
        )

        unhealthy = sum(
            row.status == HealthStatus.UNHEALTHY.value
            for row in rows
        )

        latencies = [
            row.latency_ms
            for row in rows
            if row.latency_ms is not None
        ]

        connected = sum(
            bool(row.connected)
            for row in rows
        )

        return DatabaseHealthSummary(
            total_checks=total,
            healthy_checks=healthy,
            degraded_checks=degraded,
            unhealthy_checks=unhealthy,
            average_latency_ms=(
                sum(latencies) / len(latencies)
                if latencies
                else None
            ),
            maximum_latency_ms=(
                max(latencies)
                if latencies
                else None
            ),
            availability_rate=(
                connected / total
                if total
                else 0.0
            ),
        )

    def cleanup_old_checks(
        self,
        *,
        retention_days: int = 30,
        db: Session | None = None,
    ) -> int:
        from sqlalchemy import delete

        if not 1 <= retention_days <= 3650:
            raise HealthValidationError(
                "Invalid retention period"
            )

        cutoff = utc_now() - timedelta(
            days=retention_days
        )

        with self._session(db) as session:
            result = session.execute(
                delete(DatabaseHealthCheckORM)
                .where(
                    DatabaseHealthCheckORM.checked_at
                    < cutoff
                )
            )

            return int(result.rowcount or 0)


# ============================================================
# DATABASE HEALTH MONITOR
# ============================================================

class DatabaseHealthMonitor:
    """
    Run lightweight database readiness probes.

    Important:
        Health check persistence is optional.

        A database can respond to SELECT 1
        while the application schema is missing
        or migration state is invalid.

        Use separate schema checks when needed.
    """

    def __init__(
        self,
        *,
        latency_threshold_ms: float = 250.0,
        persist_results: bool = False,
        repository: (
            DatabaseHealthRepository | None
        ) = None,
    ) -> None:
        self.latency_threshold_ms = (
            validate_positive_number(
                latency_threshold_ms,
                "latency_threshold_ms",
            )
        )

        self.persist_results = persist_results

        self.repository = (
            repository
            or DatabaseHealthRepository()
        )

    @staticmethod
    def get_pool_statistics(
        engine: Engine,
    ) -> PoolStatistics:
        pool = engine.pool

        def safe_pool_method(
            method_name: str,
        ) -> int | None:
            method = getattr(
                pool,
                method_name,
                None,
            )

            if not callable(method):
                return None

            try:
                value = method()

                return (
                    int(value)
                    if value is not None
                    else None
                )
            except Exception:
                return None

        return PoolStatistics(
            pool_type=type(pool).__name__,
            size=safe_pool_method("size"),
            checked_in=safe_pool_method(
                "checkedin"
            ),
            checked_out=safe_pool_method(
                "checkedout"
            ),
            overflow=safe_pool_method(
                "overflow"
            ),
            invalidated=None,
        )

    def check_database(
        self,
        *,
        persist: bool | None = None,
    ) -> DatabaseHealthResult:
        """
        Execute a lightweight SELECT 1 probe.

        Measures connection acquisition plus
        query execution time.

        Persistence is attempted separately,
        and failure is logged without hiding
        the primary probe result.
        """

        should_persist = (
            self.persist_results
            if persist is None
            else persist
        )

        check_id = uuid.uuid4().hex
        checked_at = utc_now()

        connected = False
        latency_ms = None
        database_type = "unknown"
        error_category = None
        pool_data: dict[str, Any] = {}

        details: dict[str, Any] = {
            "probe": "SELECT 1",
            "persistence_requested": should_persist,
        }

        start = time.perf_counter()

        try:
            engine = get_engine()

            database_type = (
                engine.dialect.name
            )

            with engine.connect() as connection:
                value = connection.execute(
                    text("SELECT 1")
                ).scalar_one()

                if value != 1:
                    raise DatabaseHealthError(
                        "Unexpected readiness response"
                    )

            connected = True

            latency_ms = round(
                (
                    time.perf_counter() - start
                ) * 1000,
                3,
            )

            pool_data = asdict(
                self.get_pool_statistics(
                    engine
                )
            )

            status = (
                HealthStatus.HEALTHY
                if latency_ms
                <= self.latency_threshold_ms
                else HealthStatus.DEGRADED
            )

        except Exception as exc:
            latency_ms = round(
                (
                    time.perf_counter() - start
                ) * 1000,
                3,
            )

            status = HealthStatus.UNHEALTHY

            error_category = (
                type(exc).__name__
            )

            logger.warning(
                "Database readiness probe failed: %s",
                error_category,
            )

        result = DatabaseHealthResult(
            check_id=check_id,
            status=status.value,
            database_type=database_type,
            connected=connected,
            latency_ms=latency_ms,
            threshold_ms=self.latency_threshold_ms,
            pool=pool_data,
            details=details,
            error_category=error_category,
            checked_at=iso_datetime(
                checked_at
            ),
        )

        if should_persist:
            try:
                self.repository.save_check(
                    result
                )
            except Exception as exc:
                logger.warning(
                    "Could not persist health result: %s",
                    type(exc).__name__,
                )

        return result

    # --------------------------------------------------------
    # PUBLIC READINESS RESPONSE
    # --------------------------------------------------------

    def public_health_response(
        self,
    ) -> dict[str, Any]:
        """
        Safe response for a public health endpoint.

        Does not expose connection strings,
        database hostnames or exception details.
        """

        result = self.check_database(
            persist=False
        )

        return {
            "component": "database",
            "status": result.status,
            "ready": result.connected,
        }

    # --------------------------------------------------------
    # INTERNAL DIAGNOSTICS
    # --------------------------------------------------------

    def internal_diagnostics(
        self,
    ) -> dict[str, Any]:
        """
        Intended for authenticated operators.
        """

        result = self.check_database(
            persist=False
        )

        return asdict(result)

    # --------------------------------------------------------
    # HEALTH HISTORY
    # --------------------------------------------------------

    def health_history(
        self,
        *,
        limit: int = 100,
    ) -> list[DatabaseHealthResult]:
        return self.repository.list_checks(
            limit=limit
        )

    def health_summary(
        self,
        *,
        hours: int = 24,
    ) -> DatabaseHealthSummary:
        return self.repository.get_summary(
            hours=hours
        )

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_report(
        self,
        output_path: str | Path,
        *,
        hours: int = 24,
        limit: int = 1000,
    ) -> Path:
        payload = {
            "generated_at": iso_datetime(
                utc_now()
            ),
            "current": asdict(
                self.check_database(
                    persist=False
                )
            ),
            "summary": asdict(
                self.health_summary(
                    hours=hours
                )
            ),
            "history": [
                asdict(item)
                for item in self.health_history(
                    limit=limit
                )
            ],
        }

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
                payload,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        return path


# ============================================================
# FACTORIES
# ============================================================

def create_database_health_repository() -> (
    DatabaseHealthRepository
):
    return DatabaseHealthRepository()


def create_database_health_monitor(
    *,
    latency_threshold_ms: float = 250.0,
    persist_results: bool = False,
) -> DatabaseHealthMonitor:
    return DatabaseHealthMonitor(
        latency_threshold_ms=latency_threshold_ms,
        persist_results=persist_results,
    )


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

    # All ORM models required for create_all()
    # must already be imported.
    create_database_tables()

    monitor = create_database_health_monitor(
        latency_threshold_ms=250.0,
        persist_results=True,
    )

    result = monitor.check_database()

    print("=" * 60)
    print("NEUROWORLD AI - DATABASE HEALTH")
    print("=" * 60)

    print(
        "Status:",
        result.status,
    )

    print(
        "Connected:",
        result.connected,
    )

    print(
        "Database:",
        result.database_type,
    )

    print(
        "Latency (ms):",
        result.latency_ms,
    )

    print(
        "Pool:",
        result.pool,
    )

    if result.connected:
        try:
            summary = monitor.health_summary(
                hours=24
            )

            print(
                "24h summary:",
                asdict(summary),
            )

            report = monitor.export_report(
                "experiments/reports/database_health.json"
            )

            print(
                "Report:",
                report,
            )

        except Exception as exc:
            print(
                "History unavailable:",
                type(exc).__name__,
            )

    close_database()


if __name__ == "__main__":
    main()
    