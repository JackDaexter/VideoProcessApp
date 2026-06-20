"""
app/services/clip_service.py — Clip Generator pipeline.

Uses the real OpenShorts stack:
  1. Download video from GCS
  2. Transcribe with faster-whisper (word-level timestamps)
  3. Send transcript + words to Gemini → get viral clip timestamps
  4. Cut each clip with FFmpeg (stream copy)
  5. Reframe to 9:16 with MediaPipe TRACK mode
  6. Upload clips to GCS
  7. Update Supabase job
"""

import shutil
from pathlib import Path

import structlog

from app.config import get_settings
from app.db.supabase import update_job_status, update_job_step
from app.models.requests import ClipGeneratorRequest
from app.models.responses import JobStatus
from app.services import openshots_service as os_svc
from app.storage.gcs import download_from_gcs, generate_signed_url, upload_to_gcs

log = structlog.get_logger(__name__)


async def run_clip_generator(job_id: str, request: ClipGeneratorRequest) -> None:
    """
    Background task: run the full Clip Generator pipeline.
    Client polls GET /api/jobs/{job_id} for updates.
    """
    settings = get_settings()
    tmp_dir = Path(settings.temp_dir) / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Mark processing ────────────────────────────────────────────────
        await update_job_status(job_id, JobStatus.PROCESSING)
        log.info("clip_pipeline_start", job_id=job_id)

        # ── 2. Download source video from GCS ─────────────────────────────────
        await update_job_step(job_id, "downloading_video")
        local_video = str(tmp_dir / "source.mp4")
        await download_from_gcs(request.video_url, local_video)

        # ── 3. Get video duration ─────────────────────────────────────────────
        await update_job_step(job_id, "analyzing_video")
        video_duration = await os_svc.get_video_duration(local_video)

        # ── 4. Transcribe with faster-whisper (word-level) ────────────────────
        await update_job_step(job_id, "transcribing")
        transcript_data = await os_svc.transcribe_video(local_video)

        # ── 5. Gemini LLM clip selection ──────────────────────────────────────
        await update_job_step(job_id, "selecting_clips")
        selected_clips = await os_svc.select_clips_with_gemini(
            transcript_text=transcript_data["text"],
            words=transcript_data["words"],
            video_duration=video_duration,
            user_prompt=request.prompt,
            max_clips=request.options.max_clips,
            min_duration=request.options.min_duration_per_clip,
            max_duration=request.options.max_duration_per_clip,
        )

        if not selected_clips:
            raise ValueError("Gemini found no suitable viral clips for this video and prompt.")

        # ── 6. Cut + reframe each clip ────────────────────────────────────────
        total_clips = len(selected_clips)
        clips_result = []
        for idx, clip in enumerate(selected_clips):
            clip_num = idx + 1
            log.info("processing_clip", job_id=job_id, clip=clip_num, start=clip["start"], end=clip["end"])

            # Cut the raw segment
            await update_job_step(job_id, f"cutting_clip_{clip_num}_of_{total_clips}")
            local_cut = str(tmp_dir / f"clip_{clip_num:02}_cut.mp4")
            await os_svc.cut_clip(local_video, local_cut, clip["start"], clip["end"])

            # Reframe to 9:16 (TRACK or GENERAL mode)
            await update_job_step(job_id, f"reframing_clip_{clip_num}_of_{total_clips}")
            local_vertical = str(tmp_dir / f"clip_{clip_num:02}_vertical.mp4")
            await os_svc.reframe_to_vertical(local_cut, local_vertical)

            # Upload to GCS
            await update_job_step(job_id, f"uploading_clip_{clip_num}_of_{total_clips}")
            gcs_path = f"{settings.gcs_output_prefix}/{job_id}/clips/clip_{clip_num:02}.mp4"
            gcs_uri = await upload_to_gcs(local_vertical, gcs_path)
            signed_url = await generate_signed_url(gcs_uri, expiration_minutes=720)

            clips_result.append({
                "clip_number": clip_num,
                "gcs_uri": gcs_uri,
                "download_url": signed_url,
                "start": clip["start"],
                "end": clip["end"],
                "duration": round(clip["end"] - clip["start"], 2),
                "viral_hook_text": clip.get("viral_hook_text", ""),
                "tiktok_description": clip.get("video_description_for_tiktok", ""),
                "instagram_description": clip.get("video_description_for_instagram", ""),
                "youtube_title": clip.get("video_title_for_youtube_short", ""),
            })

        # ── 7. Mark completed ─────────────────────────────────────────────────
        result = {
            "clips": clips_result,
            "total_clips": len(clips_result),
            "transcript": transcript_data["text"],
        }
        await update_job_status(job_id, JobStatus.COMPLETED, result=result)
        log.info("clip_pipeline_done", job_id=job_id, clips=len(clips_result))

    except Exception as exc:
        log.exception("clip_pipeline_error", job_id=job_id, error=str(exc))
        await update_job_status(job_id, JobStatus.FAILED, error=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
