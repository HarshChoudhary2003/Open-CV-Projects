"""
VisionAI Platform - FastAPI Application Entry Point
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.api import cameras, analytics, alerts, faces, reports, websocket

logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
logger = logging.getLogger("visionai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()

    # Ensure required directories exist
    for d in [settings.SNAPSHOT_DIR, settings.LOG_DIR, settings.REPORT_DIR, "weights"]:
        Path(d).mkdir(exist_ok=True)

    yield

    # Cleanup
    from app.services.pipeline import stop_all
    stop_all()
    logger.info("VisionAI Platform shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade real-time AI computer vision platform.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
PREFIX = settings.API_PREFIX

app.include_router(cameras.router,    prefix=PREFIX, tags=["Cameras"])
app.include_router(analytics.router,  prefix=PREFIX, tags=["Analytics"])
app.include_router(alerts.router,     prefix=PREFIX, tags=["Alerts"])
app.include_router(faces.router,      prefix=PREFIX, tags=["Face Registry"])
app.include_router(reports.router,    prefix=PREFIX, tags=["Reports"])
app.include_router(websocket.router,  tags=["WebSocket"])

# ── Static files (frontend build) ─────────────────────────────────────────────
frontend_dist = Path("../frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": settings.APP_VERSION}
