"""
app/services/shorts_service.py — AI Shorts pipeline.

Uses the real OpenShorts stack:
  1. Download video from GCS
  2. Transcribe with faster-whisper (word-level)
  3. Gemini selects the single best viral moment
  4. Cut segment with FFmpeg
  5. Reframe to 9:16 (TRACK mode — face following)
  6. Generate SRT captions from faster-whisper segments
  7. Burn captions with FFmpeg
  8. Upload to GCS
  9. Update Supabase job
"""

import shutil
from pathlib import Path

import structlog

from app.config import get_settings
from app.db.supabase import update_job_status
from app.models.requests import AIShortsRequest
from app.models.responses import JobStatus
from app.services import openshots_service as os_svc
from app.storage.gcs import download_from_gcs, generate_signed_url, upload_to_gcs

log = structlog.get_logger(__name__)


async def run_ai_shorts(job_id: str, request: AIShortsRequest) -> None:
    """
    Background task: run the AI Shorts pipeline (single best short).
    """
    settings = get_settings()
    tmp_dir = Path(settings.temp_dir) / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── 1. Mark processing ────────────────────────────────────────────────
        await update_job_status(job_id, JobStatus.PROCESSING)
        log.info("shorts_pipeline_start", job_id=job_id)

        # ── 2. Download source video ──────────────────────────────────────────
        local_video = str(tmp_dir / "source.mp4")
        await download_from_gcs(request.video_url, local_video)

        # ── 3. Get duration ───────────────────────────────────────────────────
        video_duration = await os_svc.get_video_duration(local_video)

        # ── 4. Transcribe ─────────────────────────────────────────────────────
        transcript_data = await os_svc.transcribe_video(local_video)

        # ── 5. Gemini picks the single best moment ────────────────────────────
        clips = await os_svc.select_clips_with_gemini(
            transcript_text=transcript_data["text"],
            words=transcript_data["words"],
            video_duration=video_duration,
            user_prompt=request.prompt,
            max_clips=1,
        )

        if clips:
            best = clips[0]
        else:
            # Fallback: use first target_duration seconds
            target = float(request.options.target_duration)
            best = {
                "start": 0.0,
                "end": min(target, video_duration),
                "viral_hook_text": "",
                "video_description_for_tiktok": "",
                "video_description_for_instagram": "",
                "video_title_for_youtube_short": "",
            }

        log.info("shorts_best_clip", start=best["start"], end=best["end"])

        # ── 6. Cut the segment ────────────────────────────────────────────────
        local_cut = str(tmp_dir / "cut.mp4")
        await os_svc.cut_clip(local_video, local_cut, best["start"], best["end"])

        # ── 7. Reframe to 9:16 (face tracking) ───────────────────────────────
        local_vertical = str(tmp_dir / "vertical.mp4")
        await os_svc.reframe_to_vertical(local_cut, local_vertical)

        # ── 8. Burn captions (if requested) ───────────────────────────────────
        output_video = local_vertical
        captions_burned = False

        if request.options.add_captions:
            clip_segments = [
                s for s in transcript_data["segments"]
                if s["start"] >= best["start"] and s["end"] <= best["end"]
            ]
            if clip_segments:
                srt_content = os_svc.segments_to_srt(clip_segments, offset=best["start"])
                srt_path = str(tmp_dir / "captions.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                local_captioned = str(tmp_dir / "short_final.mp4")
                await os_svc.burn_captions(local_vertical, srt_path, local_captioned)
                output_video = local_captioned
                captions_burned = True

        # ── 9. Upload to GCS ──────────────────────────────────────────────────
        gcs_path = f"{settings.gcs_output_prefix}/{job_id}/short.mp4"
        gcs_uri = await upload_to_gcs(output_video, gcs_path)
        signed_url = await generate_signed_url(gcs_uri, expiration_minutes=720)

        actual_duration = await os_svc.get_video_duration(output_video)

        # ── 10. Mark completed ────────────────────────────────────────────────
        result = {
            "short_url": gcs_uri,
            "download_url": signed_url,
            "duration": round(actual_duration, 2),
            "captions_burned": captions_burned,
            "viral_hook_text": best.get("viral_hook_text", ""),
            "tiktok_description": best.get("video_description_for_tiktok", ""),
            "instagram_description": best.get("video_description_for_instagram", ""),
            "youtube_title": best.get("video_title_for_youtube_short", ""),
            "transcript": transcript_data["text"],
            "segment": {"start": best["start"], "end": best["end"]},
        }
        await update_job_status(job_id, JobStatus.COMPLETED, result=result)
        log.info("shorts_pipeline_done", job_id=job_id, duration=actual_duration)

    except Exception as exc:
        log.exception("shorts_pipeline_error", job_id=job_id, error=str(exc))
        await update_job_status(job_id, JobStatus.FAILED, error=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
