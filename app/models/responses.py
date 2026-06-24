"""
app/models/responses.py — Pydantic response models for all API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


# ── Job Status Enum ───────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    CLIP_GENERATOR = "clip_generator"
    AI_SHORTS = "ai_shorts"
    YOUTUBE_STUDIO = "youtube_studio"


# ── Job Response ──────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    job_id: str
    user_id: str
    type: JobType
    status: JobStatus
    current_step: Optional[str] = None
    input_url: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime
    message: str = "Job created successfully. Poll /api/jobs/{job_id} for status."


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int


# ── Shorts Responses ─────────────────────────────────────────────────────────

class ShortResponse(BaseModel):
    job_id: str
    user_id: str
    status: JobStatus
    input_url: str
    prompt: str
    result: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ShortListResponse(BaseModel):
    shorts: List[ShortResponse]
    total: int


# ── Clips Responses ───────────────────────────────────────────────────────────

class ClipItemResponse(BaseModel):
    """One individual clip inside a clip_generator job result."""
    clip_number: int
    gcs_uri: str
    download_url: str
    start: float
    end: float
    duration: float
    viral_hook_text: Optional[str] = None
    tiktok_description: Optional[str] = None
    instagram_description: Optional[str] = None
    youtube_title: Optional[str] = None


class ClipJobResponse(BaseModel):
    """A clip_generator job with its resolved list of clips."""
    job_id: str
    user_id: str
    status: JobStatus
    input_url: str
    prompt: str
    total_clips: int
    transcript: Optional[str] = None
    clips: List[ClipItemResponse]
    created_at: datetime
    updated_at: datetime


class ClipJobListResponse(BaseModel):
    jobs: List[ClipJobResponse]
    total: int


# ── Processing Result Payloads ────────────────────────────────────────────────

class ClipResult(BaseModel):
    """Stored in the `result` JSON column of the jobs table."""
    clips: List[Dict[str, Any]]  # [{url, start_time, end_time, duration, title}]
    transcript: Optional[str] = None
    total_clips: int


class AIShortsResult(BaseModel):
    """Stored in the `result` JSON column of the jobs table."""
    short_url: str           # GCS URI of the final 9:16 video
    duration: float          # Actual duration in seconds
    transcript: Optional[str] = None
    captions_burned: bool


class YouTubeStudioResult(BaseModel):
    """Stored in the `result` JSON column of the jobs table."""
    title: str
    description: str
    tags: List[str]
    thumbnail_url: str       # GCS URI of the extracted thumbnail
    transcript: Optional[str] = None


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    environment: str
