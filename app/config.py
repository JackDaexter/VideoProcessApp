"""
app/config.py — Application settings loaded from environment variables.
Aligned with the actual OpenShorts stack (Gemini + faster-whisper + MediaPipe + YOLOv8).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API Auth ──────────────────────────────────────────────────────────────
    api_key: str

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str
    supabase_service_key: str

    # ── GCP Cloud Storage ─────────────────────────────────────────────────────
    gcp_project_id: str
    gcp_bucket_name: str
    google_application_credentials: str = "/app/gcp-key.json"
    gcs_upload_prefix: str = "uploaded"
    gcs_output_prefix: str = "output"

    # ── Google Gemini ─────────────────────────────────────────────────────────
    # Used for LLM-based clip selection (like OpenShorts does)
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"

    # ── OpenShorts Pipeline ───────────────────────────────────────────────────
    # faster-whisper model: tiny | base | small | medium | large-v3
    whisper_model: str = "base"
    # beam_size: 1=fastest, 5=most accurate (default OpenShorts value)
    whisper_beam_size: int = 3
    # whisper device: "cpu" or "cuda" (set to cuda on GPU instances)
    whisper_device: str = "cpu"
    # compute type: "int8" for CPU, "float16" for GPU
    whisper_compute_type: str = "int8"
    # Crop mode: TRACK (MediaPipe face tracking) | GENERAL (blurred background)
    # TODO : check after if allow option
    crop_mode: str = "GENERAL"
    # YOLO quality in TRACK mode: LOW | MEDIUM | HIGH | PREMIUM
    # LOW=never, MEDIUM=every 50 frames, HIGH=every 10 frames, PREMIUM=every frame
    reframe_quality: str = "MEDIUM"
    # Max clips processed in parallel (semaphore limit)
    max_concurrent_clips: int = 5
    # Max clips Gemini can select per video
    max_clips: int = 10
    # Temp directory for intermediate processing files
    temp_dir: str = "/tmp/video_process"

    # ── App Settings ──────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "info"
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the app settings."""
    return Settings()
