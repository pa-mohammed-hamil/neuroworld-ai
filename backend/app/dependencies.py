
"""
NeuroWorld AI
backend/app/dependencies.py

FastAPI dependency injection and resource management.

Features
--------
- Cached application settings
- SQLAlchemy database session lifecycle
- Optional Redis connection
- Bearer-token authentication
- Role-based access control
- Request context and correlation IDs
- CPU / CUDA / MPS device selection
- Trusted local model loading
- Service dependency injection
- WebSocket token validation
- Readiness and dependency health checks

Dependencies
------------
pip install fastapi sqlalchemy pydantic pydantic-settings
pip install PyJWT redis torch

Notes
-----
- Redis, PyTorch, and JWT are optional until their features
  are used.
- This module does not issue JWTs or create user accounts.
- Authentication requires a configured JWT secret.
- Database models and application services can be registered
  as they are implemented.
- Never load untrusted PyTorch checkpoint files.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
import uuid

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Generic,
    Literal,
    TypeVar,
)

from fastapi import (
    Depends,
    Header,
    HTTPException,
    Request,
    WebSocket,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import (
    PROJECT_ROOT,
    DeviceType,
    Environment,
    Settings,
    get_settings,
)


logger = logging.getLogger("neuroworld.dependencies")

T = TypeVar("T")

bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description="JWT access token",
)


# ============================================================
# EXCEPTIONS
# ============================================================

class DependencyUnavailableError(RuntimeError):
    """Raised when an optional dependency is unavailable."""


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model is not available."""


class ServiceUnavailableError(RuntimeError):
    """Raised when an application service is not registered."""


# ============================================================
# REQUEST CONTEXT
# ============================================================

@dataclass(frozen=True)
class RequestContext:
    request_id: str
    started_at: float
    method: str
    path: str
    client_ip: str | None = None

    @property
    def elapsed_ms(self) -> float:
        return round(
            (time.perf_counter() - self.started_at) * 1000,
            2,
        )


def get_request_context(
    request: Request,
    x_request_id: Annotated[
        str | None,
        Header(alias="X-Request-ID"),
    ] = None,
) -> RequestContext:
    """
    Generate a safe request correlation ID.

    Client-supplied IDs are accepted only when short
    and restricted to a safe character set.
    """

    safe_id = None

    if x_request_id:
        candidate = x_request_id.strip()

        if (
            1 <= len(candidate) <= 80
            and all(
                char.isalnum() or char in "-_"
                for char in candidate
            )
        ):
            safe_id = candidate

    return RequestContext(
        request_id=safe_id or uuid.uuid4().hex,
        started_at=time.perf_counter(),
        method=request.method,
        path=request.url.path,
        client_ip=(
            request.client.host
            if request.client
            else None
        ),
    )


# ============================================================
# DATABASE ENGINE
# ============================================================

@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
    """
    Create a shared SQLAlchemy engine.

    Engine is thread-safe; individual sessions are not.
    """

    settings = get_settings()

    url = settings.database_url

    options: dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "echo": settings.debug,
    }

    if url.startswith("sqlite"):
        options["connect_args"] = {
            "check_same_thread": False,
            "timeout": 30,
        }

        if ":memory:" not in url:
            # Resolve SQLite relative paths against PROJECT_ROOT.
            prefix = "sqlite:///"

            if url.startswith(prefix) and not url.startswith(
                "sqlite:////"
            ):
                relative = url[len(prefix):]
                absolute = (PROJECT_ROOT / relative).resolve()

                if not absolute.is_relative_to(PROJECT_ROOT):
                    raise ValueError(
                        "SQLite database path escapes project root"
                    )

                absolute.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                url = "sqlite:///" + absolute.as_posix()

    engine = create_engine(
        url,
        **options,
    )

    return engine


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker:
    return sessionmaker(
        bind=get_database_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI database-session dependency.

    Commits are explicit in service/repository methods.
    A request without an explicit commit is rolled back.
    """

    factory = get_session_factory()
    session = factory()

    try:
        yield session

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]


# ============================================================
# REDIS
# ============================================================

@lru_cache(maxsize=1)
def get_redis_client():
    """
    Return an optional Redis client.

    The connection is not established until used.
    """

    settings = get_settings()

    if not settings.enable_redis:
        raise DependencyUnavailableError(
            "Redis is disabled in configuration"
        )

    try:
        import redis
    except ImportError as exc:
        raise DependencyUnavailableError(
            "Redis requires: pip install redis"
        ) from exc

    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
        health_check_interval=30,
    )


def require_redis():
    try:
        client = get_redis_client()
        client.ping()
        return client

    except Exception as exc:
        logger.warning(
            "Redis dependency unavailable: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis service is unavailable",
        ) from exc


# ============================================================
# AUTHENTICATION
# ============================================================

@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    roles: frozenset[str] = field(
        default_factory=frozenset
    )
    email: str | None = None
    permissions: frozenset[str] = field(
        default_factory=frozenset
    )
    token_id: str | None = None

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions


def _authentication_secret(settings: Settings) -> str:
    if settings.secret_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT authentication is not configured",
        )

    secret = settings.secret_key.get_secret_value()

    if len(secret) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT authentication is not securely configured",
        )

    return secret


def decode_access_token(
    token: str,
    settings: Settings | None = None,
) -> CurrentUser:
    """
    Validate an externally issued JWT access token.

    Required claims:
        sub: user ID
        exp: expiry
        iat: issued-at time
        iss: neuroworld-ai
        aud: neuroworld-api
        token_type: access

    Optional:
        roles: list[str]
        permissions: list[str]
        email: str
        jti: str

    Production applications should additionally check
    user status, revocation and current role assignments
    against the identity store.
    """

    settings = settings or get_settings()

    secret = _authentication_secret(settings)

    try:
        import jwt
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="JWT package is not installed",
        ) from exc

    # Restrict the accepted algorithm independently of
    # token headers. Never accept the 'none' algorithm.
    allowed_algorithms = {"HS256", "HS384", "HS512"}

    if settings.jwt_algorithm not in allowed_algorithms:
        raise HTTPException(
            status_code=503,
            detail="Unsupported JWT algorithm configuration",
        )

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.jwt_algorithm],
            issuer="neuroworld-ai",
            audience="neuroworld-api",
            options={
                "require": [
                    "sub",
                    "exp",
                    "iat",
                    "iss",
                    "aud",
                    "token_type",
                ],
            },
            leeway=10,
        )

    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.get("token_type") != "access":
        raise HTTPException(
            status_code=401,
            detail="Invalid token type",
        )

    user_id = payload.get("sub")

    if not isinstance(user_id, str) or not user_id.strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid user identifier",
        )

    roles = payload.get("roles", [])
    permissions = payload.get("permissions", [])

    if (
        not isinstance(roles, list)
        or not all(isinstance(v, str) for v in roles)
        or not isinstance(permissions, list)
        or not all(
            isinstance(v, str)
            for v in permissions
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization claims",
        )

    return CurrentUser(
        user_id=user_id,
        roles=frozenset(roles),
        permissions=frozenset(permissions),
        email=payload.get("email"),
        token_id=payload.get("jti"),
    )


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    settings: Annotated[
        Settings,
        Depends(get_settings),
    ],
) -> CurrentUser:
    """
    Require a valid bearer access token.

    Authentication cannot be bypassed by toggling
    development mode.
    """

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return decode_access_token(
        credentials.credentials,
        settings,
    )


AuthenticatedUser = Annotated[
    CurrentUser,
    Depends(get_current_user),
]


# ============================================================
# ROLE AND PERMISSION CHECKS
# ============================================================

def require_roles(*allowed_roles: str):
    """
    Dependency factory for role-based access control.

    Example:
        @router.get("/admin")
        def admin_only(
            user: CurrentUser = Depends(
                require_roles("admin")
            )
        ):
            return {"user": user.user_id}
    """

    if not allowed_roles:
        raise ValueError(
            "At least one role is required"
        )

    async def dependency(
        user: AuthenticatedUser,
    ) -> CurrentUser:

        if not any(
            role in user.roles
            for role in allowed_roles
        ):
            raise HTTPException(
                status_code=403,
                detail="Insufficient role privileges",
            )

        return user

    return dependency


def require_permissions(*required_permissions: str):
    """
    Require all specified permissions.
    """

    if not required_permissions:
        raise ValueError(
            "At least one permission is required"
        )

    async def dependency(
        user: AuthenticatedUser,
    ) -> CurrentUser:

        missing = [
            permission
            for permission in required_permissions
            if permission not in user.permissions
        ]

        if missing:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions",
            )

        return user

    return dependency


AdminUser = Annotated[
    CurrentUser,
    Depends(require_roles("admin")),
]

ResearchUser = Annotated[
    CurrentUser,
    Depends(
        require_roles(
            "admin",
            "researcher",
            "ml_engineer",
        )
    ),
]


# ============================================================
# GPU / DEVICE MANAGEMENT
# ============================================================

@dataclass(frozen=True)
class ComputeDevice:
    name: str
    available: bool
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    precision: str = "float32"


@lru_cache(maxsize=1)
def get_compute_device() -> ComputeDevice:
    settings = get_settings()

    try:
        import torch

    except ImportError:
        if settings.default_device not in (
            DeviceType.AUTO,
            DeviceType.CPU,
        ):
            raise DependencyUnavailableError(
                "PyTorch is required for the configured device"
            )

        return ComputeDevice(
            name="cpu",
            available=True,
        )

    requested = settings.default_device.value

    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            requested = "mps"
        else:
            requested = "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise DependencyUnavailableError(
                "CUDA was requested but is unavailable"
            )

        properties = torch.cuda.get_device_properties(0)

        return ComputeDevice(
            name="cuda",
            available=True,
            gpu_name=properties.name,
            gpu_memory_gb=round(
                properties.total_memory / (1024 ** 3),
                2,
            ),
        )

    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise DependencyUnavailableError(
                "MPS was requested but is unavailable"
            )

        return ComputeDevice(
            name="mps",
            available=True,
            gpu_name="Apple Metal",
        )

    return ComputeDevice(
        name="cpu",
        available=True,
    )


# ============================================================
# TRUSTED MODEL REGISTRY
# ============================================================

@dataclass
class ModelEntry:
    name: str
    model: Any
    device: str
    loaded_at: float
    source_path: str


class ModelRegistry:
    """
    Thread-safe in-process model registry.

    Models are loaded only from trusted local files
    under the project's models directory.

    For concurrent inference, individual model methods
    must also be safe for concurrent access.
    """

    def __init__(self):
        self._models: dict[str, ModelEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        model: Any,
        device: str = "cpu",
        source_path: str = "memory",
    ) -> None:
        with self._lock:
            self._models[name] = ModelEntry(
                name=name,
                model=model,
                device=device,
                loaded_at=time.time(),
                source_path=source_path,
            )

    def get(self, name: str) -> Any:
        with self._lock:
            entry = self._models.get(name)

            if entry is None:
                raise ModelUnavailableError(
                    f"Model '{name}' is not loaded"
                )

            return entry.model

    def contains(self, name: str) -> bool:
        with self._lock:
            return name in self._models

    def unregister(self, name: str) -> None:
        with self._lock:
            self._models.pop(name, None)

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": entry.name,
                    "device": entry.device,
                    "loaded_at": entry.loaded_at,
                    "source_path": entry.source_path,
                }
                for entry in self._models.values()
            ]

    def load_state_dict(
        self,
        name: str,
        model_factory: Callable[[], Any],
        checkpoint_relative_path: str,
        strict: bool = True,
    ) -> Any:
        """
        Load weights into a model created by model_factory.

        This uses torch.load(weights_only=True).

        Only locally trusted, reviewed model artifacts
        should be used. The factory must construct the
        matching model architecture.
        """

        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PyTorch is required to load models"
            ) from exc

        settings = get_settings()

        model_root = settings.models_dir.resolve()

        path = (
            model_root / checkpoint_relative_path
        ).resolve()

        if not path.is_relative_to(model_root):
            raise ValueError(
                "Checkpoint path must remain within models/"
            )

        if not path.is_file():
            raise ModelUnavailableError(
                f"Checkpoint does not exist: {path.name}"
            )

        if path.suffix not in {".pt", ".pth"}:
            raise ValueError(
                "Only .pt and .pth checkpoints are supported"
            )

        device = get_compute_device()

        # Instantiate trusted code rather than deserializing
        # a Python model object from the checkpoint.
        model = model_factory()

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(
            state_dict,
            strict=strict,
        )

        model.to(device.name)
        model.eval()

        self.register(
            name=name,
            model=model,
            device=device.name,
            source_path=str(path),
        )

        return model


@lru_cache(maxsize=1)
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


def require_model(model_name: str):
    """
    FastAPI dependency factory for a loaded model.
    """

    def dependency() -> Any:
        registry = get_model_registry()

        try:
            return registry.get(model_name)

        except ModelUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

    return dependency


# ============================================================
# SERVICE REGISTRY
# ============================================================

class ServiceRegistry:
    """
    Explicit service registry.

    Application startup registers implemented services.
    Missing services are not silently replaced by mocks.
    """

    def __init__(self):
        self._services: dict[str, Any] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        service: Any,
    ) -> None:
        with self._lock:
            self._services[name] = service

    def get(self, name: str) -> Any:
        with self._lock:
            if name not in self._services:
                raise ServiceUnavailableError(
                    f"Service '{name}' is not registered"
                )

            return self._services[name]

    def unregister(self, name: str) -> None:
        with self._lock:
            self._services.pop(name, None)

    def list_services(self) -> list[str]:
        with self._lock:
            return sorted(self._services)


@lru_cache(maxsize=1)
def get_service_registry() -> ServiceRegistry:
    return ServiceRegistry()


def require_service(service_name: str):
    """
    Inject a previously registered application service.
    """

    def dependency() -> Any:
        try:
            return get_service_registry().get(
                service_name
            )

        except ServiceUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            ) from exc

    return dependency


# Named service dependencies.
# These resolve only after services are registered.

get_world_service = require_service("world")
get_observation_service = require_service("observation")
get_imagination_service = require_service("imagination")
get_prediction_service = require_service("prediction")
get_simulation_service = require_service("simulation")
get_planning_service = require_service("planning")
get_training_service = require_service("training")
get_evaluation_service = require_service("evaluation")


# ============================================================
# FEATURE FLAGS
# ============================================================

def require_feature(feature_name: str):
    """
    Reject API requests when a feature is disabled.

    Example:
        dependencies=[
            Depends(require_feature("enable_imagination"))
        ]
    """

    allowed = {
        "enable_world_model",
        "enable_imagination",
        "enable_planning",
        "enable_simulation",
        "enable_reinforcement_learning",
        "enable_training_api",
    }

    if feature_name not in allowed:
        raise ValueError(
            f"Unknown feature flag: {feature_name}"
        )

    def dependency(
        settings: Annotated[
            Settings,
            Depends(get_settings),
        ],
    ) -> bool:

        if not getattr(settings, feature_name):
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Feature '{feature_name}' "
                    "is currently disabled"
                ),
            )

        return True

    return dependency


# ============================================================
# WEBSOCKET AUTHENTICATION
# ============================================================

async def authenticate_websocket(
    websocket: WebSocket,
) -> CurrentUser:
    """
    Authenticate a WebSocket connection.

    Requires:
        Authorization: Bearer <JWT>

    Query-string tokens are intentionally unsupported
    because URLs can be logged or exposed.

    For browser clients, use a secure same-site session
    cookie or a short-lived WebSocket ticket validated
    by an additional authentication implementation.
    """

    header = websocket.headers.get(
        "authorization",
        "",
    )

    scheme, _, token = header.partition(" ")

    if scheme.lower() != "bearer" or not token:
        await websocket.close(code=1008)
        raise HTTPException(
            status_code=401,
            detail="WebSocket authentication required",
        )

    try:
        return decode_access_token(token)

    except HTTPException:
        await websocket.close(code=1008)
        raise


# ============================================================
# DEPENDENCY HEALTH
# ============================================================

def check_database_health() -> dict[str, Any]:
    try:
        engine = get_database_engine()

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return {
            "status": "healthy",
        }

    except Exception as exc:
        logger.warning(
            "Database health check failed: %s",
            type(exc).__name__,
        )

        return {
            "status": "unhealthy",
            "error": "Database connection failed",
        }


def check_redis_health() -> dict[str, Any]:
    settings = get_settings()

    if not settings.enable_redis:
        return {
            "status": "disabled",
        }

    try:
        client = get_redis_client()
        client.ping()

        return {
            "status": "healthy",
        }

    except Exception:
        return {
            "status": "unhealthy",
            "error": "Redis connection failed",
        }


def check_model_health() -> dict[str, Any]:
    registry = get_model_registry()

    models = registry.list_models()

    return {
        "status": (
            "ready"
            if models
            else "not_loaded"
        ),
        "loaded_models": len(models),
        "models": [
            entry["name"]
            for entry in models
        ],
    }


def get_dependency_health() -> dict[str, Any]:
    """
    Return health information without exposing
    credentials, connection URLs or stack traces.
    """

    settings = get_settings()

    database = check_database_health()
    redis = check_redis_health()
    models = check_model_health()

    redis_ok = redis["status"] in {
        "healthy",
        "disabled",
    }

    ready = (
        database["status"] == "healthy"
        and redis_ok
    )

    return {
        "status": "ready" if ready else "degraded",
        "application": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment.value,
        "dependencies": {
            "database": database,
            "redis": redis,
            "models": models,
            "compute": {
                "device": settings.detect_device(),
            },
        },
    }


# ============================================================
# APPLICATION RESOURCE LIFECYCLE
# ============================================================

@asynccontextmanager
async def dependency_lifespan(app):
    """
    FastAPI lifespan manager.

    Example in main.py:

        from fastapi import FastAPI
        from .dependencies import dependency_lifespan

        app = FastAPI(
            title="NeuroWorld AI",
            lifespan=dependency_lifespan,
        )

    Register application services separately during
    startup after their implementations are available.
    """

    settings = get_settings()

    logger.info(
        "Starting %s version %s",
        settings.app_name,
        settings.app_version,
    )

    if settings.is_development:
        settings.create_directories()

    try:
        yield

    finally:
        logger.info("Shutting down NeuroWorld AI")

        try:
            get_database_engine().dispose()
        except Exception:
            logger.exception(
                "Failed to dispose database engine"
            )

        try:
            get_redis_client().close()
        except DependencyUnavailableError:
            pass
        except Exception:
            logger.exception(
                "Failed to close Redis client"
            )


# ============================================================
# FASTAPI DEPENDENCY ALIASES
# ============================================================

AppSettings = Annotated[
    Settings,
    Depends(get_settings),
]

RequestInfo = Annotated[
    RequestContext,
    Depends(get_request_context),
]

ModelRegistryDependency = Annotated[
    ModelRegistry,
    Depends(get_model_registry),
]

ServiceRegistryDependency = Annotated[
    ServiceRegistry,
    Depends(get_service_registry),
]

ComputeDeviceDependency = Annotated[
    ComputeDevice,
    Depends(get_compute_device),
]


# ============================================================
# DIAGNOSTIC ENTRY POINT
# ============================================================

def main() -> None:
    settings = get_settings()

    print("=" * 60)
    print("NEUROWORLD AI — DEPENDENCY DIAGNOSTICS")
    print("=" * 60)

    print(f"Application : {settings.app_name}")
    print(f"Environment : {settings.environment.value}")
    print(f"API prefix  : {settings.api_prefix}")

    try:
        device = get_compute_device()

        print(f"Compute     : {device.name}")

        if device.gpu_name:
            print(f"GPU         : {device.gpu_name}")

        if device.gpu_memory_gb is not None:
            print(
                f"GPU Memory  : "
                f"{device.gpu_memory_gb} GB"
            )

    except DependencyUnavailableError as exc:
        print(f"Compute     : unavailable ({exc})")

    print()
    print("Dependency health:")

    health = get_dependency_health()

    for name, result in health["dependencies"].items():
        print(f"  {name}: {result}")

    print()
    print("Registered services:")

    services = get_service_registry().list_services()

    if services:
        for service in services:
            print(f"  - {service}")
    else:
        print("  None registered")

    print()
    print("Loaded models:")

    models = get_model_registry().list_models()

    if models:
        for model in models:
            print(f"  - {model['name']}")
    else:
        print("  No models loaded")

    print("=" * 60)


if __name__ == "__main__":
    main()
