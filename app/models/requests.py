"""
app/models/requests.py — Pydantic request models for all API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Clip Generator ────────────────────────────────────────────────────────────

class ClipGeneratorOptions(BaseModel):
    max_clips: int = Field(default=5, ge=1, le=20, description="Max number of clips to extract")
    max_duration_per_clip: int = Field(default=60, ge=5, le=300, description="Max clip duration in seconds")
    min_duration_per_clip: int = Field(default=10, ge=3, le=60, description="Min clip duration in seconds")


class ClipGeneratorRequest(BaseModel):
    video_url: str = Field(..., description="GCS URI of the source video (gs://bucket/path)")
    prompt: str = Field(..., min_length=5, max_length=2000, description="Natural language prompt for clip selection")
    options: ClipGeneratorOptions = Field(default_factory=ClipGeneratorOptions)

    @field_validator("video_url")
    @classmethod
    def validate_gcs_url(cls, v: str) -> str:
        if not v.startswith("gs://"):
            raise ValueError("video_url must be a GCS URI starting with gs://")
        return v


# ── AI Shorts ─────────────────────────────────────────────────────────────────

class AIShortsOptions(BaseModel):
    target_duration: int = Field(default=60, ge=15, le=180, description="Target short duration in seconds")
    add_captions: bool = Field(default=True, description="Burn captions into the video")
    aspect_ratio: str = Field(default="9:16", description="Output aspect ratio (9:16 for vertical)")


class AIShortsRequest(BaseModel):
    video_url: str = Field(..., description="GCS URI of the source video (gs://bucket/path)")
    prompt: str = Field(..., min_length=5, max_length=2000, description="Prompt guiding the short content selection")
    options: AIShortsOptions = Field(default_factory=AIShortsOptions)

    @field_validator("video_url")
    @classmethod
    def validate_gcs_url(cls, v: str) -> str:
        if not v.startswith("gs://"):
            raise ValueError("video_url must be a GCS URI starting with gs://")
        return v


# ── YouTube Studio ────────────────────────────────────────────────────────────

class YouTubeStudioRequest(BaseModel):
    video_url: str = Field(..., description="GCS URI of the source video (gs://bucket/path)")
    prompt: str = Field(..., min_length=5, max_length=2000, description="Guidance for metadata generation style/focus")
    channel_context: Optional[str] = Field(None, max_length=500, description="Optional context about the channel/brand")

    @field_validator("video_url")
    @classmethod
    def validate_gcs_url(cls, v: str) -> str:
        if not v.startswith("gs://"):
            raise ValueError("video_url must be a GCS URI starting with gs://")
        return v
