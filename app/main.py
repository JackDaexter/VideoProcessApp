"""
app/main.py — FastAPI application entry point for VideoProcessApp.

PhantomPilot Video Processing Backend — built on the OpenShorts stack
(mutonby/openshorts: faster-whisper + Gemini + MediaPipe + YOLOv8 + FFmpeg)

Endpoints:
  POST   /api/clip-generator     → Generate viral clips from a video
  POST   /api/ai-shorts          → Generate vertical 9:16 AI short
  POST   /api/youtube-studio     → Generate YouTube title/description/tags/thumbnail
  GET    /api/jobs               → List jobs (with filters)
  GET    /api/jobs/{job_id}      → Get job status + result
  DELETE /api/jobs/{job_id}      → Cancel a job
  GET    /health                 → Health check
  GET    /docs                   → Swagger UI
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.responses import HealthResponse
from app.routers import ai_shorts, clip_generator, jobs, youtube_studio

# ── Structured Logging ────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if os.getenv("APP_ENV") == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)


# ── Lifespan: Warm up all models at startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("server_starting", env=settings.app_env)

    os.makedirs(settings.temp_dir, exist_ok=True)

    # Pre-load all models to avoid cold-start latency on first request
    from app.services.openshots_service import (
        get_face_detection,
        get_whisper_model,
        get_yolo_model,
    )

    loop = asyncio.get_event_loop()
    log.info("preloading_models")
    await loop.run_in_executor(None, get_whisper_model)
    await loop.run_in_executor(None, get_yolo_model)
    await loop.run_in_executor(None, get_face_detection)
    log.info("models_ready", whisper=settings.whisper_model, crop_mode=settings.crop_mode)

    yield
    log.info("server_shutdown")


# ── FastAPI App ────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="PhantomPilot — VideoProcessApp",
    description=(
        "Video processing backend built on the **OpenShorts** stack "
        "(faster-whisper · Google Gemini · MediaPipe · YOLOv8 · FFmpeg). "
        "Generates viral clips, AI Shorts, and YouTube metadata from GCS videos."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten to your Toklo frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(clip_generator.router)
app.include_router(ai_shorts.router)
app.include_router(youtube_studio.router)
app.include_router(jobs.router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Used by Cloud Run health probes. Returns 200 when server is ready."""
    return HealthResponse(environment=settings.app_env)
