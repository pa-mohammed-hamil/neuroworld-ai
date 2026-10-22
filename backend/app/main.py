
"""
NeuroWorld AI — FastAPI Main Application

File: backend/app/main.py

Run from project root:
    python -m uvicorn backend.app.main:app --reload

Frontend:
    http://127.0.0.1:8000/

API documentation:
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings, PROJECT_ROOT
from .dependencies import (
    get_dependency_health,
    dependency_lifespan,
)

settings = get_settings()

FRONTEND_DIR = (PROJECT_ROOT / "frontend").resolve()
INDEX_FILE = FRONTEND_DIR / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with dependency_lifespan(app):
        yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "NeuroWorld AI — Neural World Model, "
        "Imagination, Simulation and Planning Platform"
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# ------------------------------------------------------------
# API
# ------------------------------------------------------------

@app.get("/api/v1/health", tags=["System"])
def health():
    return get_dependency_health()


@app.get("/api/v1/config", tags=["System"])
def public_config():
    return settings.public_info()


@app.get("/api/v1/status", tags=["System"])
def system_status():
    return {
        "application": settings.app_name,
        "version": settings.app_version,
        "frontend": INDEX_FILE.is_file(),
        "api": "running",
        "world_model_enabled": settings.enable_world_model,
        "imagination_enabled": settings.enable_imagination,
        "planning_enabled": settings.enable_planning,
        "simulation_enabled": settings.enable_simulation,
        "note": (
            "Enabled configuration flags do not mean "
            "models or services are loaded."
        ),
    }


# ------------------------------------------------------------
# FRONTEND
# ------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def home():
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found",
        )

    return FileResponse(INDEX_FILE)


# Mount static assets, if their folders exist.
for folder_name in (
    "assets",
    "css",
    "js",
    "images",
    "locales",
):
    folder = FRONTEND_DIR / folder_name

    if folder.is_dir():
        app.mount(
            f"/{folder_name}",
            StaticFiles(directory=str(folder)),
            name=f"frontend-{folder_name}",
        )


# Serve additional frontend pages.
# Only allow existing files directly inside frontend/.
@app.get("/{page_name}.html", include_in_schema=False)
async def frontend_page(page_name: str):
    page = (FRONTEND_DIR / f"{page_name}.html").resolve()

    if (
        page.parent != FRONTEND_DIR
        or not page.is_file()
    ):
        raise HTTPException(
            status_code=404,
            detail="Frontend page not found",
        )

    return FileResponse(page)


# ------------------------------------------------------------
# LOCAL DEVELOPMENT
# ------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
