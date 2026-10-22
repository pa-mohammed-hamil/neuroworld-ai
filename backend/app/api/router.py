"""Root API router for NeuroWorld AI.

Imports existing v1 endpoint modules if present. Missing optional endpoint
modules are skipped, allowing incremental development; import errors within
an existing module are not suppressed.
"""
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

from fastapi import APIRouter, Depends

from .dependencies import get_settings, verify_api_key

ENDPOINT_MODULES = (
    "worlds", "observations", "actions", "imagination", "predictions",
    "simulations", "planning", "agents", "experiments", "training",
    "models", "evaluation", "health",
)


def build_api_router() -> APIRouter:
    settings = get_settings()
    router = APIRouter(prefix=settings.api_prefix)

    for name in ENDPOINT_MODULES:
        qualified_name = f"{__package__}.v1.{name}"
        if find_spec(qualified_name) is None:
            continue
        module = import_module(qualified_name)
        endpoint_router = getattr(module, "router", None)
        if not isinstance(endpoint_router, APIRouter):
            raise TypeError(f"{qualified_name} must expose FastAPI APIRouter named 'router'")
        dependencies = [] if name == "health" else [Depends(verify_api_key)]
        router.include_router(endpoint_router, dependencies=dependencies)

    return router


api_router = build_api_router()
