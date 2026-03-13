from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.route_v1 import router as analyzer_router

API_V1_PREFIX = "/api/v1"


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Intentionally minimal.
    # Auth0 JWKS, Redis, and rate-limiter state are initialized lazily by their own modules.
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Analyzer API v1",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_csv_env("CORS_ALLOW_ORIGINS", "*"),
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").strip().lower()
        not in {"0", "false", "no"},
        allow_methods=_csv_env("CORS_ALLOW_METHODS", "*"),
        allow_headers=_csv_env("CORS_ALLOW_HEADERS", "*"),
    )

    v1_router = APIRouter(prefix=API_V1_PREFIX)
    v1_router.include_router(analyzer_router)
    app.include_router(v1_router)

    @app.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "service": "analyzer-api",
            "status": "ok",
        }

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return app


app = create_app()

__all__ = ["app", "create_app"]