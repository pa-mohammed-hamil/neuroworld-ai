"""Shared FastAPI dependencies for NeuroWorld AI."""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


class APISettings(BaseModel):
    api_prefix: str = "/api/v1"
    api_key: str | None = Field(default=None, repr=False)
    require_api_key: bool = True


@lru_cache
def get_settings() -> APISettings:
    """Read environment configuration once per process."""
    required = os.getenv("NEUROWORLD_REQUIRE_API_KEY", "true").lower() not in {"0", "false", "no"}
    return APISettings(
        api_prefix=os.getenv("NEUROWORLD_API_PREFIX", "/api/v1"),
        api_key=os.getenv("NEUROWORLD_API_KEY") or None,
        require_api_key=required,
    )


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    provided_key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[APISettings, Depends(get_settings)],
) -> None:
    """Fail closed unless explicitly configured for unauthenticated development."""
    if not settings.require_api_key:
        return
    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured; set NEUROWORLD_API_KEY",
        )
    if not provided_key or not secrets.compare_digest(provided_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )


SettingsDep = Annotated[APISettings, Depends(get_settings)]
AuthDep = Annotated[None, Depends(verify_api_key)]
