"""
app/services/studio_service.py — YouTube Studio pipeline.

Uses the real OpenShorts stack:
  1. Download video from GCS
  2. Transcribe with faster-whisper
  3. Generate title, description, tags, thumbnail_text via Gemini LLM
  4. Extract thumbnail frame with FFmpeg
  5. Upload thumbnail to GCS
  6. Update Supabase job
"""

import shutil
from pathlib import Path

import structlog

from app.config import get_settings
from app.db.supabase import update_job_status
from app.models.requests import YouTubeStudioRequest
from app.models.responses import JobStatus
from app.services import openshots_service as os_svc
from app.storage.gcs import download_from_gcs, generate_signed_url, upload_image_to_gcs

log = structlog.get_logger(__name__)


async def run_youtube_studio(job_id: str, request: YouTubeStudioRequest) -> None:
    """
    Background task: run the YouTube Studio metadata pipeline.
    """
    settings = get_settings()
    tmp_dir = Path(settings.temp_dir) / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Mark processing ────────────────────────────────────────────────
        await update_job_status(job_id, JobStatus.PROCESSING)
        log.info("studio_pipeline_start", job_id=job_id)

        # ── 2. Download source video ──────────────────────────────────────────
        local_video = str(tmp_dir / "source.mp4")
        await download_from_gcs(request.video_url, local_video)

        # ── 3. Transcribe with faster-whisper ─────────────────────────────────
        transcript_data = await os_svc.transcribe_video(local_video)

        # ── 4. Gemini generates YouTube metadata ──────────────────────────────
        metadata = await os_svc.generate_youtube_metadata_with_gemini(
            transcript_text=transcript_data["text"],
            user_prompt=request.prompt,
            channel_context=request.channel_context,
        )

        # ── 5. Extract thumbnail at 10% of video duration ─────────────────────
        duration = await os_svc.get_video_duration(local_video)
        thumb_ts = max(1.0, duration * 0.1)
        local_thumb = str(tmp_dir / "thumbnail.jpg")
        await os_svc.extract_thumbnail(local_video, local_thumb, timestamp=thumb_ts)

        # ── 6. Upload thumbnail to GCS ────────────────────────────────────────
        gcs_thumb_path = f"{settings.gcs_output_prefix}/{job_id}/thumbnail.jpg"
        gcs_thumb_uri = await upload_image_to_gcs(local_thumb, gcs_thumb_path)
        thumb_signed_url = await generate_signed_url(gcs_thumb_uri, expiration_minutes=720)

        # ── 7. Mark completed ─────────────────────────────────────────────────
        result = {
            "title": metadata.get("title", ""),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "thumbnail_text": metadata.get("thumbnail_text", ""),
            "thumbnail_url": gcs_thumb_uri,
            "thumbnail_download_url": thumb_signed_url,
            "transcript": transcript_data["text"],
        }
        await update_job_status(job_id, JobStatus.COMPLETED, result=result)
        log.info("studio_pipeline_done", job_id=job_id, title=result["title"][:60])

    except Exception as exc:
        log.exception("studio_pipeline_error", job_id=job_id, error=str(exc))
        await update_job_status(job_id, JobStatus.FAILED, error=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
