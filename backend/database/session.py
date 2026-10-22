
"""
NeuroWorld AI
backend/app/database/session.py

Database Connection and Session Management

Features:
- PostgreSQL and SQLite support
- SQLAlchemy 2.x ORM
- Database engine initialization
- Connection pooling
- FastAPI dependency injection
- Transaction management
- Automatic rollback on errors
- Database connectivity health checks
- Thread-safe session factory
- Explicit database shutdown
- Environment-based configuration

Install:
    pip install sqlalchemy psycopg[binary]

SQLite works without an additional database driver.

Environment:
    DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/neuroworld

For local SQLite:
    DATABASE_URL=sqlite:///./neuroworld.db

Usage:
    from backend.app.database.session import (
        Base,
        get_db,
        get_engine,
        get_session_factory,
        session_scope,
        check_database_health,
        close_database,
    )
"""

from __future__ import annotations

import logging
import os
import threading

from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Generator, Iterator

from sqlalchemy import (
    MetaData,
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Session,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool


logger = logging.getLogger(
    "neuroworld.database.session"
)


# ============================================================
# EXCEPTIONS
# ============================================================

class DatabaseError(Exception):
    """Base database exception."""


class DatabaseConfigurationError(DatabaseError):
    """Invalid database configuration."""


class DatabaseConnectionError(DatabaseError):
    """Database connection failure."""


class DatabaseTransactionError(DatabaseError):
    """Database transaction failure."""


# ============================================================
# NAMING CONVENTIONS
# ============================================================

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": (
        "fk_%(table_name)s_"
        "%(column_0_name)s_"
        "%(referred_table_name)s"
    ),
    "pk": "pk_%(table_name)s",
}


metadata = MetaData(
    naming_convention=NAMING_CONVENTION
)


# ============================================================
# ORM BASE
# ============================================================

class Base(DeclarativeBase):
    """
    Base class for all NeuroWorld ORM models.

    Example:
        class World(Base):
            __tablename__ = "worlds"
    """

    metadata = metadata


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

class DatabaseSettings:
    """
    Read database configuration from environment.

    Settings are intentionally independent from
    the application's Pydantic configuration
    to avoid circular imports.
    """

    def __init__(self) -> None:
        self.url = os.getenv(
            "DATABASE_URL",
            "sqlite:///./neuroworld.db",
        )

        self.echo = (
            os.getenv(
                "DATABASE_ECHO",
                "false",
            ).lower()
            in ("1", "true", "yes", "on")
        )

        self.pool_size = int(
            os.getenv(
                "DATABASE_POOL_SIZE",
                "10",
            )
        )

        self.max_overflow = int(
            os.getenv(
                "DATABASE_MAX_OVERFLOW",
                "20",
            )
        )

        self.pool_timeout = int(
            os.getenv(
                "DATABASE_POOL_TIMEOUT",
                "30",
            )
        )

        self.pool_recycle = int(
            os.getenv(
                "DATABASE_POOL_RECYCLE",
                "1800",
            )
        )

        self.connect_timeout = int(
            os.getenv(
                "DATABASE_CONNECT_TIMEOUT",
                "10",
            )
        )

        if self.pool_size < 1:
            raise DatabaseConfigurationError(
                "DATABASE_POOL_SIZE must be positive"
            )

        if self.max_overflow < 0:
            raise DatabaseConfigurationError(
                "DATABASE_MAX_OVERFLOW cannot be negative"
            )

        if self.pool_timeout < 1:
            raise DatabaseConfigurationError(
                "DATABASE_POOL_TIMEOUT must be positive"
            )

        if self.connect_timeout < 1:
            raise DatabaseConfigurationError(
                "DATABASE_CONNECT_TIMEOUT must be positive"
            )

        try:
            self.parsed_url = make_url(
                self.url
            )

        except Exception as exc:
            raise DatabaseConfigurationError(
                "Invalid DATABASE_URL"
            ) from exc

    @property
    def is_sqlite(self) -> bool:
        return (
            self.parsed_url.get_backend_name()
            == "sqlite"
        )

    @property
    def is_postgresql(self) -> bool:
        return (
            self.parsed_url.get_backend_name()
            == "postgresql"
        )

    def public_info(self) -> dict[str, Any]:
        """
        Never expose database passwords.
        """
        return {
            "driver": (
                self.parsed_url.drivername
            ),
            "database": (
                self.parsed_url.database
            ),
            "host": (
                self.parsed_url.host
            ),
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "echo": self.echo,
        }


# ============================================================
# GLOBAL STATE
# ============================================================

_engine: Engine | None = None

_session_factory: (
    sessionmaker[Session] | None
) = None

_engine_lock = threading.RLock()


# ============================================================
# ENGINE CREATION
# ============================================================

def create_database_engine(
    settings: DatabaseSettings | None = None,
) -> Engine:
    """
    Create a SQLAlchemy database engine.

    Engine creation does not necessarily open
    a connection immediately.
    """

    config = (
        settings or DatabaseSettings()
    )

    common_options: dict[str, Any] = {
        "echo": config.echo,
        "pool_pre_ping": True,
        "future": True,
    }

    if config.is_sqlite:
        common_options[
            "connect_args"
        ] = {
            "check_same_thread": False,
        }

        # Share a single in-memory database
        # across sessions and threads.
        if config.parsed_url.database in (
            None,
            "",
            ":memory:",
        ):
            common_options[
                "poolclass"
            ] = StaticPool

    else:
        common_options.update(
            {
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
                "pool_timeout": config.pool_timeout,
                "pool_recycle": config.pool_recycle,
            }
        )

        if config.is_postgresql:
            common_options[
                "connect_args"
            ] = {
                "connect_timeout": (
                    config.connect_timeout
                )
            }

    try:
        engine = create_engine(
            config.url,
            **common_options,
        )

    except Exception as exc:
        raise DatabaseConfigurationError(
            "Unable to create database engine"
        ) from exc

    if config.is_sqlite:
        @event.listens_for(
            engine,
            "connect",
        )
        def enable_sqlite_foreign_keys(
            dbapi_connection,
            connection_record,
        ) -> None:
            cursor = dbapi_connection.cursor()

            try:
                cursor.execute(
                    "PRAGMA foreign_keys=ON"
                )

            finally:
                cursor.close()

    logger.info(
        "Database engine configured: %s",
        config.public_info(),
    )

    return engine


# ============================================================
# ENGINE SINGLETON
# ============================================================

def get_engine() -> Engine:
    """
    Return the shared SQLAlchemy engine.

    The engine is initialized lazily.
    """

    global _engine

    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = create_database_engine()

    return _engine


# ============================================================
# SESSION FACTORY
# ============================================================

def get_session_factory() -> sessionmaker[Session]:
    """
    Return the shared session factory.

    A new Session must be created for each
    request, task, or independent unit of work.

    SQLAlchemy Session instances are not
    thread-safe and must not be shared.
    """

    global _session_factory

    if _session_factory is None:
        with _engine_lock:
            if _session_factory is None:
                _session_factory = sessionmaker(
                    bind=get_engine(),
                    class_=Session,
                    autoflush=False,
                    expire_on_commit=False,
                )

    return _session_factory


# ============================================================
# CREATE SESSION
# ============================================================

def create_db_session() -> Session:
    """
    Create an independent database session.

    Caller must close it after use.
    """

    factory = get_session_factory()

    return factory()


# ============================================================
# FASTAPI DEPENDENCY
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Example:

        @router.get("/worlds")
        def list_worlds(
            db: Session = Depends(get_db)
        ):
            ...

    Write operations should explicitly commit.

    On exceptions, the transaction is rolled
    back and the session is always closed.
    """

    db = create_db_session()

    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================
# TRANSACTION CONTEXT MANAGER
# ============================================================

@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Transactional session context manager.

    Automatically commits successful work.

    Automatically rolls back on errors.

    Example:

        with session_scope() as db:
            db.add(world)
    """

    db = create_db_session()

    try:
        yield db

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================
# READ-ONLY SESSION
# ============================================================

@contextmanager
def readonly_session() -> Iterator[Session]:
    """
    Session intended for read operations.

    Rolls back the transaction at exit.

    This is an application-level convention,
    not a database-enforced read-only mode.
    """

    db = create_db_session()

    try:
        yield db

    finally:
        db.rollback()
        db.close()


# ============================================================
# EXPLICIT TRANSACTION
# ============================================================

def execute_in_transaction(
    operation,
    *args,
    **kwargs,
):
    """
    Execute a callable in a managed transaction.

    The callable receives the database Session
    as its first argument.

    Example:

        def create_world(db, name):
            world = World(name=name)
            db.add(world)
            db.flush()
            return world.id

        world_id = execute_in_transaction(
            create_world,
            "Demo World",
        )
    """

    with session_scope() as db:
        return operation(
            db,
            *args,
            **kwargs,
        )


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_database_health() -> dict[str, Any]:
    """
    Check database connectivity.

    Safe to use from a FastAPI health endpoint.
    """

    try:
        engine = get_engine()

        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return {
            "status": "healthy",
            "connected": True,
            "database": (
                engine.url.get_backend_name()
            ),
        }

    except SQLAlchemyError as exc:
        logger.error(
            "Database health check failed: %s",
            type(exc).__name__,
        )

        return {
            "status": "unhealthy",
            "connected": False,
            "error": "Database connection failed",
        }

    except Exception:
        logger.exception(
            "Unexpected database health check error"
        )

        return {
            "status": "unhealthy",
            "connected": False,
            "error": "Unexpected database error",
        }


# ============================================================
# TABLE INITIALIZATION
# ============================================================

def create_database_tables() -> None:
    """
    Create tables registered on Base.metadata.

    Intended for local development and tests.

    Production applications should use Alembic
    migrations instead of create_all().
    """

    engine = get_engine()

    try:
        Base.metadata.create_all(
            bind=engine
        )

        logger.info(
            "Database tables initialized"
        )

    except SQLAlchemyError as exc:
        logger.exception(
            "Database initialization failed"
        )

        raise DatabaseConnectionError(
            "Failed to initialize database tables"
        ) from exc


# ============================================================
# DROP TABLES - DEVELOPMENT ONLY
# ============================================================

def drop_database_tables(
    *,
    confirm: bool = False,
) -> None:
    """
    Destructive operation.

    Disabled unless explicitly confirmed.

    Never call automatically during startup.
    """

    if not confirm:
        raise DatabaseError(
            "Table deletion requires confirm=True"
        )

    environment = os.getenv(
        "APP_ENV",
        "development",
    ).lower()

    if environment not in (
        "development",
        "test",
        "testing",
    ):
        raise DatabaseError(
            "Table deletion is disabled outside "
            "development/test environments"
        )

    Base.metadata.drop_all(
        bind=get_engine()
    )

    logger.warning(
        "All registered database tables dropped"
    )


# ============================================================
# DATABASE STATUS
# ============================================================

def get_database_status() -> dict[str, Any]:
    settings = DatabaseSettings()

    health = check_database_health()

    return {
        **settings.public_info(),
        **health,
    }


# ============================================================
# DATABASE SHUTDOWN
# ============================================================

def close_database() -> None:
    """
    Dispose of pooled connections.

    Call during FastAPI application shutdown.

    Do not call while active requests are
    still using database sessions.
    """

    global _engine
    global _session_factory

    with _engine_lock:
        engine = _engine

        _session_factory = None
        _engine = None

        if engine is not None:
            engine.dispose()

            logger.info(
                "Database engine disposed"
            )


# ============================================================
# RESET FOR TESTING
# ============================================================

def reset_database_for_testing(
    database_url: str,
) -> None:
    """
    Replace database configuration for tests.

    Only call before starting concurrent work.
    """

    environment = os.getenv(
        "APP_ENV",
        "development",
    ).lower()

    if environment not in (
        "development",
        "test",
        "testing",
    ):
        raise DatabaseError(
            "Database reset is disabled"
        )

    close_database()

    os.environ[
        "DATABASE_URL"
    ] = database_url


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
    )

    print("=" * 60)
    print("NEUROWORLD AI - DATABASE SESSION")
    print("=" * 60)

    settings = DatabaseSettings()

    print(
        "Configuration:",
        settings.public_info(),
    )

    health = check_database_health()

    print(
        "Health:",
        health,
    )

    close_database()


if __name__ == "__main__":
    main()
