
"""
NeuroWorld AI
backend/app/core/exceptions.py

Centralized exception handling for the NeuroWorld AI platform.

Features:
- Structured application exceptions
- Standardized API error responses
- Model and inference errors
- Training and checkpoint errors
- World simulation and imagination errors
- Planning and reinforcement learning errors
- Authentication and authorization errors
- Database and external service errors
- FastAPI exception handlers
- Request correlation IDs
- Safe production error messages
- Structured logging

Dependencies:
    pip install fastapi pydantic

Usage:
    from backend.app.core.exceptions import (
        WorldNotFoundError,
        ModelNotLoadedError,
        register_exception_handlers,
    )

    raise WorldNotFoundError("world-001")

    register_exception_handlers(app)
"""

from __future__ import annotations

import logging
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import get_settings


logger = logging.getLogger("neuroworld.exceptions")


# ============================================================
# ERROR CODES
# ============================================================

class ErrorCode(str, Enum):

    # General
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    OPERATION_TIMEOUT = "OPERATION_TIMEOUT"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"

    # Authentication
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    ACCESS_DENIED = "ACCESS_DENIED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"

    # Models
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    MODEL_VERSION_MISMATCH = "MODEL_VERSION_MISMATCH"

    # Neural processing
    PERCEPTION_FAILED = "PERCEPTION_FAILED"
    TOKENIZATION_FAILED = "TOKENIZATION_FAILED"
    LATENT_ENCODING_FAILED = "LATENT_ENCODING_FAILED"
    LATENT_DECODING_FAILED = "LATENT_DECODING_FAILED"
    INVALID_LATENT_STATE = "INVALID_LATENT_STATE"

    # World models
    WORLD_NOT_FOUND = "WORLD_NOT_FOUND"
    WORLD_CREATION_FAILED = "WORLD_CREATION_FAILED"
    WORLD_STATE_INVALID = "WORLD_STATE_INVALID"
    WORLD_MODEL_FAILED = "WORLD_MODEL_FAILED"
    WORLD_TRANSITION_FAILED = "WORLD_TRANSITION_FAILED"

    # Dynamics / physics
    DYNAMICS_FAILED = "DYNAMICS_FAILED"
    PHYSICS_CONSTRAINT_VIOLATION = (
        "PHYSICS_CONSTRAINT_VIOLATION"
    )
    COLLISION_DETECTION_FAILED = (
        "COLLISION_DETECTION_FAILED"
    )

    # Prediction
    PREDICTION_FAILED = "PREDICTION_FAILED"
    PREDICTION_HORIZON_EXCEEDED = (
        "PREDICTION_HORIZON_EXCEEDED"
    )
    UNCERTAINTY_TOO_HIGH = "UNCERTAINTY_TOO_HIGH"

    # Imagination
    IMAGINATION_FAILED = "IMAGINATION_FAILED"
    IMAGINATION_LIMIT_EXCEEDED = (
        "IMAGINATION_LIMIT_EXCEEDED"
    )
    COUNTERFACTUAL_FAILED = "COUNTERFACTUAL_FAILED"
    ROLLOUT_FAILED = "ROLLOUT_FAILED"

    # Planning
    PLANNING_FAILED = "PLANNING_FAILED"
    NO_VALID_PLAN = "NO_VALID_PLAN"
    PLANNING_TIMEOUT = "PLANNING_TIMEOUT"
    ACTION_CONSTRAINT_VIOLATION = (
        "ACTION_CONSTRAINT_VIOLATION"
    )

    # Reinforcement learning
    RL_TRAINING_FAILED = "RL_TRAINING_FAILED"
    POLICY_NOT_FOUND = "POLICY_NOT_FOUND"
    REWARD_MODEL_FAILED = "REWARD_MODEL_FAILED"

    # Simulation
    SIMULATION_NOT_FOUND = "SIMULATION_NOT_FOUND"
    SIMULATION_FAILED = "SIMULATION_FAILED"
    SIMULATION_LIMIT_EXCEEDED = (
        "SIMULATION_LIMIT_EXCEEDED"
    )
    ENVIRONMENT_NOT_FOUND = "ENVIRONMENT_NOT_FOUND"

    # Training
    TRAINING_FAILED = "TRAINING_FAILED"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    DATASET_INVALID = "DATASET_INVALID"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_LOAD_FAILED = "CHECKPOINT_LOAD_FAILED"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"

    # Storage
    DATABASE_ERROR = "DATABASE_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"
    CACHE_ERROR = "CACHE_ERROR"

    # Security
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    UNSAFE_ACTION = "UNSAFE_ACTION"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    FILE_NOT_ALLOWED = "FILE_NOT_ALLOWED"


# ============================================================
# API ERROR RESPONSE
# ============================================================

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(
        default_factory=dict
    )


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
    request_id: str
    timestamp: str


def utc_timestamp() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_request_id(request: Request) -> str:
    """
    Prefer a server-assigned correlation ID.

    A middleware can populate request.state.request_id.
    """

    value = getattr(
        request.state,
        "request_id",
        None,
    )

    if isinstance(value, str) and value:
        return value

    return uuid.uuid4().hex


def build_error_response(
    *,
    code: ErrorCode | str,
    message: str,
    request_id: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:

    error_code = (
        code.value
        if isinstance(code, ErrorCode)
        else str(code)
    )

    response = ErrorResponse(
        error=ErrorDetail(
            code=error_code,
            message=message,
            details=dict(details or {}),
        ),
        request_id=request_id,
        timestamp=utc_timestamp(),
    )

    return response.model_dump(mode="json")


# ============================================================
# BASE APPLICATION EXCEPTION
# ============================================================

class NeuroWorldError(Exception):
    """
    Base exception for expected application failures.

    Public messages and details must not contain secrets,
    private model inputs or sensitive internal paths.
    """

    default_code = ErrorCode.INTERNAL_ERROR
    default_status_code = 500
    default_message = "An application error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:

        self.message = message or self.default_message

        self.code = code or self.default_code

        self.status_code = (
            status_code
            if status_code is not None
            else self.default_status_code
        )

        self.details = dict(details or {})

        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


# ============================================================
# GENERAL EXCEPTIONS
# ============================================================

class ValidationError(NeuroWorldError):
    default_code = ErrorCode.VALIDATION_ERROR
    default_status_code = 422
    default_message = "Input validation failed"


class InvalidRequestError(NeuroWorldError):
    default_code = ErrorCode.INVALID_REQUEST
    default_status_code = 400
    default_message = "Invalid request"


class ResourceNotFoundError(NeuroWorldError):
    default_code = ErrorCode.RESOURCE_NOT_FOUND
    default_status_code = 404
    default_message = "Requested resource was not found"


class ResourceConflictError(NeuroWorldError):
    default_code = ErrorCode.RESOURCE_CONFLICT
    default_status_code = 409
    default_message = "Resource conflict"


class ServiceUnavailableError(NeuroWorldError):
    default_code = ErrorCode.SERVICE_UNAVAILABLE
    default_status_code = 503
    default_message = "Service is unavailable"


class OperationTimeoutError(NeuroWorldError):
    default_code = ErrorCode.OPERATION_TIMEOUT
    default_status_code = 504
    default_message = "Operation timed out"


class RateLimitExceededError(NeuroWorldError):
    default_code = ErrorCode.RATE_LIMIT_EXCEEDED
    default_status_code = 429
    default_message = "Rate limit exceeded"


# ============================================================
# AUTHENTICATION EXCEPTIONS
# ============================================================

class AuthenticationError(NeuroWorldError):
    default_code = ErrorCode.AUTHENTICATION_REQUIRED
    default_status_code = 401
    default_message = "Authentication required"


class InvalidTokenError(AuthenticationError):
    default_code = ErrorCode.INVALID_TOKEN
    default_message = "Invalid authentication token"


class TokenExpiredError(AuthenticationError):
    default_code = ErrorCode.TOKEN_EXPIRED
    default_message = "Authentication token has expired"


class AccessDeniedError(NeuroWorldError):
    default_code = ErrorCode.ACCESS_DENIED
    default_status_code = 403
    default_message = "Access denied"


class PermissionDeniedError(AccessDeniedError):
    default_code = ErrorCode.INSUFFICIENT_PERMISSIONS
    default_message = "Insufficient permissions"


# ============================================================
# MODEL EXCEPTIONS
# ============================================================

class ModelError(NeuroWorldError):
    default_code = ErrorCode.MODEL_INFERENCE_FAILED
    default_status_code = 500
    default_message = "Neural model operation failed"


class ModelNotFoundError(ModelError):
    default_code = ErrorCode.MODEL_NOT_FOUND
    default_status_code = 404
    default_message = "Model was not found"


class ModelNotLoadedError(ModelError):
    default_code = ErrorCode.MODEL_NOT_LOADED
    default_status_code = 503
    default_message = "Model is not loaded"


class ModelLoadError(ModelError):
    default_code = ErrorCode.MODEL_LOAD_FAILED
    default_status_code = 503
    default_message = "Model could not be loaded"


class ModelInferenceError(ModelError):
    default_code = ErrorCode.MODEL_INFERENCE_FAILED
    default_message = "Model inference failed"


class ModelCompatibilityError(ModelError):
    default_code = ErrorCode.MODEL_INCOMPATIBLE
    default_status_code = 409
    default_message = "Model is incompatible"


class ModelVersionMismatchError(ModelError):
    default_code = ErrorCode.MODEL_VERSION_MISMATCH
    default_status_code = 409
    default_message = "Model version mismatch"


# ============================================================
# PERCEPTION AND LATENT SPACE
# ============================================================

class PerceptionError(NeuroWorldError):
    default_code = ErrorCode.PERCEPTION_FAILED
    default_status_code = 500
    default_message = "Perception processing failed"


class TokenizationError(NeuroWorldError):
    default_code = ErrorCode.TOKENIZATION_FAILED
    default_status_code = 500
    default_message = "Tokenization failed"


class LatentEncodingError(NeuroWorldError):
    default_code = ErrorCode.LATENT_ENCODING_FAILED
    default_status_code = 500
    default_message = "Latent encoding failed"


class LatentDecodingError(NeuroWorldError):
    default_code = ErrorCode.LATENT_DECODING_FAILED
    default_status_code = 500
    default_message = "Latent decoding failed"


class InvalidLatentStateError(NeuroWorldError):
    default_code = ErrorCode.INVALID_LATENT_STATE
    default_status_code = 422
    default_message = "Invalid latent state"


# ============================================================
# WORLD MODEL EXCEPTIONS
# ============================================================

class WorldError(NeuroWorldError):
    default_code = ErrorCode.WORLD_MODEL_FAILED
    default_status_code = 500
    default_message = "World model operation failed"


class WorldNotFoundError(WorldError):
    default_code = ErrorCode.WORLD_NOT_FOUND
    default_status_code = 404
    default_message = "World was not found"

    def __init__(
        self,
        world_id: str,
    ) -> None:
        super().__init__(
            "World was not found",
            details={"world_id": world_id},
        )


class WorldCreationError(WorldError):
    default_code = ErrorCode.WORLD_CREATION_FAILED
    default_message = "World creation failed"


class InvalidWorldStateError(WorldError):
    default_code = ErrorCode.WORLD_STATE_INVALID
    default_status_code = 422
    default_message = "Invalid world state"


class WorldTransitionError(WorldError):
    default_code = ErrorCode.WORLD_TRANSITION_FAILED
    default_message = "World transition failed"


# ============================================================
# DYNAMICS AND PHYSICS
# ============================================================

class DynamicsError(NeuroWorldError):
    default_code = ErrorCode.DYNAMICS_FAILED
    default_status_code = 500
    default_message = "Dynamics prediction failed"


class PhysicsConstraintError(NeuroWorldError):
    default_code = ErrorCode.PHYSICS_CONSTRAINT_VIOLATION
    default_status_code = 422
    default_message = "Physics constraint violated"


class CollisionDetectionError(NeuroWorldError):
    default_code = ErrorCode.COLLISION_DETECTION_FAILED
    default_status_code = 500
    default_message = "Collision detection failed"


# ============================================================
# PREDICTION EXCEPTIONS
# ============================================================

class PredictionError(NeuroWorldError):
    default_code = ErrorCode.PREDICTION_FAILED
    default_status_code = 500
    default_message = "Prediction failed"


class PredictionHorizonExceededError(PredictionError):
    default_code = ErrorCode.PREDICTION_HORIZON_EXCEEDED
    default_status_code = 422
    default_message = "Prediction horizon exceeds allowed limit"


class UncertaintyTooHighError(PredictionError):
    default_code = ErrorCode.UNCERTAINTY_TOO_HIGH
    default_status_code = 422
    default_message = "Prediction uncertainty exceeds allowed limit"


# ============================================================
# IMAGINATION EXCEPTIONS
# ============================================================

class ImaginationError(NeuroWorldError):
    default_code = ErrorCode.IMAGINATION_FAILED
    default_status_code = 500
    default_message = "Imagination generation failed"


class ImaginationLimitError(ImaginationError):
    default_code = ErrorCode.IMAGINATION_LIMIT_EXCEEDED
    default_status_code = 422
    default_message = "Imagination limit exceeded"


class CounterfactualError(ImaginationError):
    default_code = ErrorCode.COUNTERFACTUAL_FAILED
    default_message = "Counterfactual generation failed"


class RolloutError(ImaginationError):
    default_code = ErrorCode.ROLLOUT_FAILED
    default_message = "Latent rollout failed"


# ============================================================
# PLANNING EXCEPTIONS
# ============================================================

class PlanningError(NeuroWorldError):
    default_code = ErrorCode.PLANNING_FAILED
    default_status_code = 500
    default_message = "Planning failed"


class NoValidPlanError(PlanningError):
    default_code = ErrorCode.NO_VALID_PLAN
    default_status_code = 422
    default_message = "No valid plan could be generated"


class PlanningTimeoutError(PlanningError):
    default_code = ErrorCode.PLANNING_TIMEOUT
    default_status_code = 504
    default_message = "Planning timed out"


class ActionConstraintError(PlanningError):
    default_code = ErrorCode.ACTION_CONSTRAINT_VIOLATION
    default_status_code = 422
    default_message = "Action violates planning constraints"


# ============================================================
# REINFORCEMENT LEARNING
# ============================================================

class RLTrainingError(NeuroWorldError):
    default_code = ErrorCode.RL_TRAINING_FAILED
    default_status_code = 500
    default_message = "Reinforcement learning training failed"


class PolicyNotFoundError(NeuroWorldError):
    default_code = ErrorCode.POLICY_NOT_FOUND
    default_status_code = 404
    default_message = "Policy was not found"


class RewardModelError(NeuroWorldError):
    default_code = ErrorCode.REWARD_MODEL_FAILED
    default_status_code = 500
    default_message = "Reward model operation failed"


# ============================================================
# SIMULATION EXCEPTIONS
# ============================================================

class SimulationError(NeuroWorldError):
    default_code = ErrorCode.SIMULATION_FAILED
    default_status_code = 500
    default_message = "Simulation failed"


class SimulationNotFoundError(SimulationError):
    default_code = ErrorCode.SIMULATION_NOT_FOUND
    default_status_code = 404
    default_message = "Simulation was not found"


class SimulationLimitError(SimulationError):
    default_code = ErrorCode.SIMULATION_LIMIT_EXCEEDED
    default_status_code = 422
    default_message = "Simulation limit exceeded"


class EnvironmentNotFoundError(SimulationError):
    default_code = ErrorCode.ENVIRONMENT_NOT_FOUND
    default_status_code = 404
    default_message = "Simulation environment was not found"


# ============================================================
# TRAINING EXCEPTIONS
# ============================================================

class TrainingError(NeuroWorldError):
    default_code = ErrorCode.TRAINING_FAILED
    default_status_code = 500
    default_message = "Model training failed"


class DatasetNotFoundError(TrainingError):
    default_code = ErrorCode.DATASET_NOT_FOUND
    default_status_code = 404
    default_message = "Dataset was not found"


class InvalidDatasetError(TrainingError):
    default_code = ErrorCode.DATASET_INVALID
    default_status_code = 422
    default_message = "Dataset is invalid"


class CheckpointNotFoundError(TrainingError):
    default_code = ErrorCode.CHECKPOINT_NOT_FOUND
    default_status_code = 404
    default_message = "Checkpoint was not found"


class CheckpointLoadError(TrainingError):
    default_code = ErrorCode.CHECKPOINT_LOAD_FAILED
    default_status_code = 500
    default_message = "Checkpoint could not be loaded"


class GPUOutOfMemoryError(TrainingError):
    default_code = ErrorCode.GPU_OUT_OF_MEMORY
    default_status_code = 503
    default_message = "Insufficient GPU memory"


# ============================================================
# STORAGE EXCEPTIONS
# ============================================================

class DatabaseError(NeuroWorldError):
    default_code = ErrorCode.DATABASE_ERROR
    default_status_code = 503
    default_message = "Database operation failed"


class StorageError(NeuroWorldError):
    default_code = ErrorCode.STORAGE_ERROR
    default_status_code = 503
    default_message = "Storage operation failed"


class CacheError(NeuroWorldError):
    default_code = ErrorCode.CACHE_ERROR
    default_status_code = 503
    default_message = "Cache operation failed"


# ============================================================
# SECURITY EXCEPTIONS
# ============================================================

class SecurityViolationError(NeuroWorldError):
    default_code = ErrorCode.SECURITY_VIOLATION
    default_status_code = 403
    default_message = "Security policy violation"


class UnsafeActionError(NeuroWorldError):
    default_code = ErrorCode.UNSAFE_ACTION
    default_status_code = 403
    default_message = "Action blocked by safety policy"


class HumanApprovalRequiredError(NeuroWorldError):
    default_code = ErrorCode.HUMAN_APPROVAL_REQUIRED
    default_status_code = 409
    default_message = "Human approval is required"


class FileNotAllowedError(NeuroWorldError):
    default_code = ErrorCode.FILE_NOT_ALLOWED
    default_status_code = 415
    default_message = "File type is not allowed"


# ============================================================
# FASTAPI EXCEPTION HANDLERS
# ============================================================

async def neuroworld_exception_handler(
    request: Request,
    exc: NeuroWorldError,
) -> JSONResponse:
    """
    Handle expected NeuroWorld AI exceptions.
    """

    request_id = safe_request_id(request)

    if exc.status_code >= 500:
        logger.error(
            "Application error | request_id=%s | code=%s",
            request_id,
            exc.code.value,
        )
    else:
        logger.warning(
            "Request rejected | request_id=%s | code=%s",
            request_id,
            exc.code.value,
        )

    # Do not expose internal details for server failures.
    if exc.status_code >= 500:
        public_message = exc.default_message
        public_details = {}
    else:
        public_message = exc.message
        public_details = exc.details

    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_response(
            code=exc.code,
            message=public_message,
            request_id=request_id,
            details=public_details,
        ),
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:

    request_id = safe_request_id(request)

    if isinstance(exc.detail, str):
        message = exc.detail
    else:
        message = "Request could not be completed"

    if exc.status_code >= 500:
        message = "Service request failed"

    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_response(
            code=(
                ErrorCode.RESOURCE_NOT_FOUND
                if exc.status_code == 404
                else ErrorCode.INVALID_REQUEST
            ),
            message=message,
            request_id=request_id,
        ),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:

    request_id = safe_request_id(request)

    errors = []

    for error in exc.errors():
        errors.append(
            {
                "field": ".".join(
                    str(part)
                    for part in error.get("loc", [])
                ),
                "type": error.get("type", "invalid"),
                "message": error.get(
                    "msg",
                    "Invalid input",
                ),
            }
        )

    return JSONResponse(
        status_code=422,
        content=build_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            message="Request validation failed",
            request_id=request_id,
            details={
                "errors": errors[:25],
            },
        ),
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:

    request_id = safe_request_id(request)

    logger.exception(
        "Unhandled server exception | request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content=build_error_response(
            code=ErrorCode.INTERNAL_ERROR,
            message="An unexpected server error occurred",
            request_id=request_id,
        ),
    )


# ============================================================
# REGISTER HANDLERS
# ============================================================

def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Register centralized exception handlers.

    Call once during application setup.
    """

    app.add_exception_handler(
        NeuroWorldError,
        neuroworld_exception_handler,
    )

    app.add_exception_handler(
        HTTPException,
        http_exception_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )

    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )


# ============================================================
# EXAMPLE / SELF-TEST
# ============================================================

def main() -> None:

    examples = [
        WorldNotFoundError("world-001"),
        ModelNotLoadedError(),
        ImaginationLimitError(),
        PlanningTimeoutError(),
        SimulationNotFoundError(),
        HumanApprovalRequiredError(),
    ]

    print("=" * 65)
    print("NEUROWORLD AI — EXCEPTION SYSTEM")
    print("=" * 65)

    for exc in examples:
        print(
            f"{exc.__class__.__name__:<35} "
            f"HTTP {exc.status_code} | "
            f"{exc.code.value}"
        )

    print()
    print("Exception definitions loaded successfully.")


if __name__ == "__main__":
    main()
