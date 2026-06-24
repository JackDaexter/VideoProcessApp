"""
app/routers/youtube_studio.py — POST /api/youtube-studio endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.auth import get_current_user
from app.db.supabase import create_job
from app.models.requests import YouTubeStudioRequest
from app.models.responses import JobCreatedResponse, JobStatus, JobType
from app.services.studio_service import run_youtube_studio
from app.storage.gcs import gcs_uri_belongs_to_user_upload

router = APIRouter(prefix="/api", tags=["YouTube Studio"])


@router.post(
    "/youtube-studio",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Generate YouTube metadata for a video",
    description=(
        "Submit a video to automatically generate a YouTube-ready title, description, "
        "tags, and thumbnail. Powered by Whisper transcription + NLP extraction. "
        "Poll GET /api/jobs/{job_id} for status and results."
    ),
)
async def youtube_studio(
    request: YouTubeStudioRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> JobCreatedResponse:
    """
    Create a YouTube Studio metadata generation job.

    - **video_url**: GCS URI of the source video (gs://bucket/path)
    - **prompt**: Style/focus guidance for the metadata generation
    - **channel_context**: Optional branding/channel info to include
    """
    if not gcs_uri_belongs_to_user_upload(request.video_url, user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="video_url must point to the current user's uploaded folder.",
        )

    job = await create_job(
        user_id=user_id,
        job_type=JobType.YOUTUBE_STUDIO,
        input_url=request.video_url,
        prompt=request.prompt,
        options={"channel_context": request.channel_context} if request.channel_context else {},
    )

    background_tasks.add_task(run_youtube_studio, job["id"], user_id, request)

    return JobCreatedResponse(
        job_id=job["id"],
        status=JobStatus.PENDING,
        created_at=datetime.now(timezone.utc),
    )
